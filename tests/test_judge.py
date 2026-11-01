from __future__ import annotations

import json
from typing import Any

import httpx
import pytest
import respx

from memjudge.judge import client as client_mod
from memjudge.judge.client import Completion, JudgeClient, JudgeError, JudgeModel
from memjudge.judge.judge import judge, load_prompt, parse_verdict, render_user
from memjudge.schema import Question, QuestionKind
from tests.conftest import FakeCompleter


def _q(kind: QuestionKind = QuestionKind.STANDARD, distractor: str | None = None) -> Question:
    return Question(
        id="q",
        conversation_id="c",
        question="What is the dog's name?",
        gold_answer="Biscuit",
        category="x",
        kind=kind,
        distractor_answer=distractor,
    )


def test_prompt_hash_is_stable_and_versioned() -> None:
    p = load_prompt()
    assert p.version == "v3" and p.hash.startswith("sha256:")
    assert "{context}" in p.user_template and "Reply with exactly one JSON object" in p.system


def test_context_placeholders_are_not_substituted() -> None:
    user = render_user(load_prompt(), _q(), "Ana: {gold_answer} {question}")
    assert "Ana: {gold_answer} {question}" in user
    assert "Gold answer: Biscuit" in user
    assert "(empty)" in render_user(load_prompt(), _q(), "  ")


def test_distractor_is_shown_for_abstention() -> None:
    user = render_user(load_prompt(), _q(QuestionKind.ABSTENTION, "Rex"), "")
    assert "Question type: ABSTENTION" in user and "must NOT be supported): Rex" in user


@pytest.mark.parametrize(
    "text",
    [
        '{"verdict": "correct", "reason": "ok", "stale": null, "gold_suspect": false}',
        '```json\n{"verdict": "incorrect", "reason": "no", "stale": true, '
        '"gold_suspect": false}\n```',
    ],
)
def test_parse_verdict_accepts_wrapped_json(text: str) -> None:
    assert parse_verdict(text)["verdict"] in ("correct", "incorrect")


@pytest.mark.parametrize(
    "text", ["no json here", '{"verdict": "maybe"}', '{"verdict": "correct", "stale": 3}']
)
def test_parse_verdict_rejects_bad_output(text: str) -> None:
    with pytest.raises(ValueError):
        parse_verdict(text)


def test_parse_verdict_takes_the_last_verdict_when_the_judge_corrects_itself() -> None:
    text = (
        '{"verdict": "correct", "reason": "first {thought}", "stale": null}\n\n'
        "Wait, that contradicts my reasoning. Corrected:\n\n"
        '{"verdict": "incorrect", "reason": "final", "stale": null, "gold_suspect": false}'
    )
    assert parse_verdict(text) == {
        "verdict": "incorrect",
        "reason": "final",
        "stale": None,
        "gold_suspect": False,
    }
    # An invalid object after a valid one does not replace it.
    assert parse_verdict('{"verdict": "correct"} then {"verdict": "maybe"}')["verdict"] == (
        "correct"
    )


def test_judge_retries_unparseable_then_reports_error() -> None:
    fake = FakeCompleter(reply="I think it's fine")
    v = judge(fake, load_prompt(), _q(), "Biscuit")
    assert v.correct is None and "unparseable" in (v.error or "")
    assert fake.calls == 2 and v.input_tokens == 200


def test_stale_only_kept_for_knowledge_update() -> None:
    reply = '{"verdict": "correct", "reason": "r", "stale": true, "gold_suspect": true}'
    assert judge(FakeCompleter(reply), load_prompt(), _q(), "x").stale is None
    v = judge(FakeCompleter(reply), load_prompt(), _q(QuestionKind.KNOWLEDGE_UPDATE), "x")
    assert v.stale is True and v.gold_suspect is True


def test_judge_error_becomes_error_verdict() -> None:
    class Failing:
        def complete(self, system: str, user: str) -> Completion:
            raise JudgeError("judge HTTP 401: bad key")

    v = judge(Failing(), load_prompt(), _q(), "x")
    assert v.correct is None and v.error == "judge HTTP 401: bad key"


def test_judge_error_after_unparseable_reply_keeps_its_tokens() -> None:
    class FailsSecond(FakeCompleter):
        def complete(self, system: str, user: str) -> Completion:
            if self.calls:
                raise JudgeError("judge HTTP 529: overloaded")
            return super().complete(system, user)

    v = judge(FailsSecond(reply="not json", cost=0.01), load_prompt(), _q(), "x")
    assert v.error == "judge HTTP 529: overloaded" and v.attempts == 2
    assert (v.input_tokens, v.output_tokens, v.cost_usd) == (100, 10, 0.01)


def test_judge_model_parse() -> None:
    j = JudgeModel.parse("openrouter:anthropic/claude-sonnet-5.5@anthropic")
    assert j.route == "openrouter" and j.upstream_provider == "anthropic"
    assert j.spec == "openrouter:anthropic/claude-sonnet-5.5@anthropic"
    with pytest.raises(ValueError, match="pin a provider"):
        JudgeModel.parse("openrouter:anthropic/claude-sonnet-5.5")
    with pytest.raises(ValueError):
        JudgeModel.parse("claude-sonnet-5-5")


@respx.mock
def test_anthropic_route_request_and_usage() -> None:
    route = respx.post("https://api.anthropic.com/v1/messages").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "claude-sonnet-5-5",
                "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "{}"}],
                "usage": {"input_tokens": 12, "output_tokens": 3},
            },
        )
    )
    c = JudgeClient(JudgeModel.parse("anthropic:claude-sonnet-5-5", effort="low"), "sk-test")
    out = c.complete("sys", "user")
    sent = json.loads(route.calls[0].request.content)
    assert route.calls[0].request.headers["x-api-key"] == "sk-test"
    assert sent["output_config"] == {"effort": "low"} and "temperature" not in sent
    assert sent["system"] == "sys" and out.text == "{}" and out.input_tokens == 12


@respx.mock
def test_openrouter_route_pins_provider_and_reads_cost() -> None:
    route = respx.post("https://openrouter.ai/api/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "model": "anthropic/claude-sonnet-5.5",
                "provider": "Anthropic",
                "choices": [{"message": {"content": "{}"}}],
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "cost": 0.0012},
            },
        )
    )
    c = JudgeClient(JudgeModel.parse("openrouter:anthropic/claude-sonnet-5.5@anthropic"), "or-key")
    out = c.complete("sys", "user")
    sent = json.loads(route.calls[0].request.content)
    assert sent["provider"] == {"order": ["anthropic"], "allow_fallbacks": False}
    assert out.cost_usd == 0.0012 and out.served_by == "anthropic/claude-sonnet-5.5@Anthropic"


@respx.mock
def test_google_route_skips_thought_parts() -> None:
    respx.post(url__regex=r"https://generativelanguage\.googleapis\.com/.*:generateContent").mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": "hmm", "thought": True}, {"text": "{}"}]}}
                ],
                "usageMetadata": {
                    "promptTokenCount": 7,
                    "candidatesTokenCount": 2,
                    "thoughtsTokenCount": 5,
                },
            },
        )
    )
    out = JudgeClient(JudgeModel.parse("google:gemini-x"), "g").complete("s", "u")
    assert out.text == "{}" and out.output_tokens == 7


@respx.mock
def test_retries_on_429_then_succeeds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_mod.time, "sleep", lambda s: None)
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        side_effect=[
            httpx.Response(429, headers={"retry-after": "1"}),
            httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}], "usage": {}}),
        ]
    )
    out = JudgeClient(JudgeModel.parse("openai:gpt-x"), "k").complete("s", "u")
    assert out.attempts == 2


@respx.mock
@pytest.mark.parametrize(
    ("spec", "url", "response"),
    [
        ("openai:gpt-x", "https://api.openai.com/v1/chat/completions", {"text": "<html>"}),
        (
            "anthropic:claude-sonnet-5-5",
            "https://api.anthropic.com/v1/messages",
            {"json": {"content": [{"text": "x"}]}},  # no "type": not the documented shape
        ),
    ],
)
def test_malformed_200_body_is_a_judge_error(spec: str, url: str, response: dict[str, Any]) -> None:
    respx.post(url).mock(return_value=httpx.Response(200, **response))
    with pytest.raises(JudgeError, match="malformed judge response"):
        JudgeClient(JudgeModel.parse(spec), "k").complete("s", "u")


@respx.mock
def test_non_retryable_error_does_not_leak_key() -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, text="invalid key")
    )
    with pytest.raises(JudgeError) as exc:
        JudgeClient(JudgeModel.parse("openai:gpt-x"), "secret-key").complete("s", "u")
    assert "secret-key" not in str(exc.value) and "401" in str(exc.value)
