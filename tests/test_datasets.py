from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from memjudge.datasets.adapters import locomo, longmemeval
from memjudge.datasets.errata import Change, Errata, apply_errata
from memjudge.datasets.loader import apply_selection, load_dataset
from memjudge.datasets.selection import Selection, conversation_sizes, select
from memjudge.schema import Dataset, QuestionKind

LOCOMO_SAMPLE = [
    {
        "sample_id": "conv-1",
        "conversation": {
            "speaker_a": "Caroline",
            "speaker_b": "Melanie",
            "session_1_date_time": "1:56 pm on 8 May, 2023",
            "session_1": [
                {
                    "speaker": "Caroline",
                    "dia_id": "D1:1",
                    "text": "I went to a support group yesterday.",
                },
                {
                    "speaker": "Melanie",
                    "dia_id": "D1:2",
                    "text": "I painted a sunrise.",
                    "img_url": ["x"],
                    "blip_caption": "a painting of a sunrise",
                },
            ],
            "session_2_date_time": "7:00 pm on 25 May, 2023",
            "session_2": [
                {"speaker": "Melanie", "dia_id": "D2:1", "text": "Self-care is important."}
            ],
            "session_3_date_time": "",
            "session_3": [],
        },
        "qa": [
            {
                "question": "When did Caroline go to the support group?",
                "answer": "7 May 2023",
                "evidence": ["D1:1"],
                "category": 2,
            },
            {
                "question": "When did Melanie paint?",
                "answer": 2023,
                "evidence": ["D1:2", "D"],
                "category": 4,
            },
            {
                "question": "What did Caroline realize?",
                "evidence": ["D2:1"],
                "category": 5,
                "adversarial_answer": "self-care is important",
            },
        ],
    }
]

LME_SAMPLE = [
    {
        "question_id": "q_abs",
        "question_type": "temporal-reasoning",
        "question": "Which came first?",
        "answer": "The information provided is not enough.",
        "question_date": "2023/04/10 (Mon) 23:07",
        "haystack_dates": ["2023/04/10 (Mon) 17:50", "2023/04/09 (Sun) 14:47"],
        "haystack_session_ids": ["s_late", "s_early"],
        "haystack_sessions": [
            [{"role": "user", "content": "late", "has_answer": True}],
            [{"role": "user", "content": "early"}, {"role": "assistant", "content": "ok"}],
        ],
        "answer_session_ids": ["s_late"],
    },
    {
        "question_id": "q_ku",
        "question_type": "knowledge-update",
        "question": "Personal best?",
        "answer": 25,
        "question_date": "2023/05/01 (Mon) 10:00",
        "haystack_dates": ["2023/04/01 (Sat) 10:00"],
        "haystack_session_ids": ["s1"],
        "haystack_sessions": [[{"role": "user", "content": "25 minutes", "has_answer": True}]],
        "answer_session_ids": ["s1"],
    },
]


def _dataset(conversations: list, questions: list) -> Dataset:
    return Dataset(
        name="t",
        version="v1",
        source="x",
        license="x",
        conversations=tuple(conversations),
        questions=tuple(questions),
    )


def test_locomo_adapter_maps_categories_and_abstention() -> None:
    convs, qs = locomo.convert(copy.deepcopy(LOCOMO_SAMPLE))
    ds = _dataset(convs, qs)
    assert [s.id for s in ds.conversations[0].sessions] == ["session_1", "session_2"]
    assert ds.conversations[0].sessions[0].started_at.isoformat() == "2023-05-08T13:56:00"
    assert (
        "[shares an image: a painting of a sunrise]"
        in ds.conversations[0].sessions[0].turns[1].text
    )
    temporal, single, adversarial = ds.questions
    assert temporal.category == "temporal" and temporal.kind is QuestionKind.STANDARD
    assert single.gold_answer == "2023" and single.evidence_turn_ids == (
        "D1:2",
    )  # junk "D" dropped
    assert adversarial.kind is QuestionKind.ABSTENTION
    assert adversarial.distractor_answer == "self-care is important"
    assert adversarial.gold_answer == locomo.ABSTENTION_GOLD
    assert temporal.asked_at == ds.conversations[0].sessions[-1].started_at


def test_longmemeval_adapter_sorts_sessions_and_maps_kinds() -> None:
    convs, qs = longmemeval.convert(copy.deepcopy(LME_SAMPLE))
    ds = _dataset(convs, qs)
    assert [s.id for s in ds.conversations[0].sessions] == ["s_early", "s_late"]
    abs_q, ku_q = ds.questions
    assert abs_q.kind is QuestionKind.ABSTENTION and abs_q.category == "temporal-reasoning"
    assert abs_q.evidence_turn_ids == ("s_late:0",)
    assert ku_q.kind is QuestionKind.KNOWLEDGE_UPDATE and ku_q.gold_answer == "25"
    assert ku_q.asked_at.isoformat() == "2023-05-01T10:00:00"


def test_longmemeval_adapter_ingests_a_repeated_session_once() -> None:
    raw = copy.deepcopy(LME_SAMPLE)
    item = raw[0]
    for key in ("haystack_dates", "haystack_session_ids", "haystack_sessions"):
        item[key].append(copy.deepcopy(item[key][0]))
    convs, _ = longmemeval.convert(raw)
    ids = [s.id for s in convs[0].sessions]
    assert len(ids) == len(set(ids)) == len(LME_SAMPLE[0]["haystack_session_ids"])
    item["haystack_sessions"][-1] = [{"role": "user", "content": "different"}]
    with pytest.raises(ValueError, match="different content"):
        longmemeval.convert(raw)


def test_load_tiny_and_hash_is_stable(tiny_path: Path) -> None:
    a, b = load_dataset(str(tiny_path)), load_dataset(str(tiny_path))
    assert a.hash == b.hash and a.hash.startswith("sha256:")
    assert not a.curated
    assert len(a.dataset.questions) == 5


def test_hash_changes_with_content(tiny_path: Path, tmp_path: Path) -> None:
    data = json.loads(tiny_path.read_text(encoding="utf-8"))
    data["questions"][0]["gold_answer"] = "Cookie"
    other = tmp_path / "other.json"
    other.write_text(json.dumps(data), encoding="utf-8")
    assert load_dataset(str(other)).hash != load_dataset(str(tiny_path)).hash


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d["questions"].append(dict(d["questions"][0])),  # duplicate question id
        lambda d: d["questions"][0].update(evidence_turn_ids=["nope"]),
        lambda d: d["questions"][0].update(conversation_id="missing"),
        lambda d: d["questions"][0].update(unexpected="field"),
    ],
)
def test_invalid_datasets_are_rejected(tiny_path: Path, mutate) -> None:  # type: ignore[no-untyped-def]
    data = json.loads(tiny_path.read_text(encoding="utf-8"))
    mutate(data)
    with pytest.raises(ValidationError):
        Dataset.model_validate(data)


def test_selection_by_conversation_category_and_size(tiny_path: Path) -> None:
    ds = load_dataset(str(tiny_path)).dataset  # c1: 4 turns, 4 questions; c2: 2 turns, 1 question

    only_c1 = select(ds, Selection(conversation_ids=("c1",)))
    assert [c.id for c in only_c1.conversations] == ["c1"] and len(only_c1.questions) == 4

    # Category filter drops conversations left without questions (nothing ingested for nothing).
    updates = select(ds, Selection(categories=("knowledge-update",)))
    assert [c.id for c in updates.conversations] == ["c1"]
    assert {q.category for q in updates.questions} == {"knowledge-update"}

    smallest = select(ds, Selection(smallest=1))
    assert [c.id for c in smallest.conversations] == ["c2"] and len(smallest.questions) == 1

    combined = select(ds, Selection(categories=("single-hop",), smallest=1))
    assert [q.id for q in combined.questions] == ["q5"]

    assert select(ds, Selection()) is ds
    assert [c.id for c, _, _ in conversation_sizes(ds)] == ["c2", "c1"]


def test_selection_errors_and_sampling(tiny_path: Path) -> None:
    ds = load_dataset(str(tiny_path)).dataset
    with pytest.raises(ValueError, match="unknown conversation"):
        select(ds, Selection(conversation_ids=("nope",)))
    with pytest.raises(ValueError, match="available"):
        select(ds, Selection(categories=("nope",)))
    with pytest.raises(ValidationError):
        Selection(smallest=1, sample=1)
    with pytest.raises(ValidationError):
        Selection(smallest=0)
    a, b = select(ds, Selection(sample=1, seed=3)), select(ds, Selection(sample=1, seed=3))
    assert a.conversations == b.conversations and len(a.conversations) == 1


def test_partial_load_keeps_full_hash(tiny_path: Path) -> None:
    full = load_dataset(str(tiny_path))
    part = apply_selection(full, Selection(smallest=1))
    assert part.hash == full.hash and part.selection == Selection(smallest=1)
    assert len(part.dataset.questions) == 1
    assert apply_selection(full, Selection()) is full
    assert Selection(conversation_ids=("c1",), sample=2).describe() == (
        "conversations c1; random 2 conversations (seed 0)"
    )


def test_errata_apply_and_guard(tiny_path: Path) -> None:
    qs = list(load_dataset(str(tiny_path)).dataset.questions)
    fixed = apply_errata(
        qs,
        Errata(
            dataset="tiny",
            version="v1",
            changes=[
                Change(
                    question_id="q1",
                    field="gold_answer",
                    old="Biscuit",
                    new="Biscuit the beagle",
                    reason="r",
                )
            ],
        ),
    )
    assert fixed[0].gold_answer == "Biscuit the beagle"
    assert qs[0].gold_answer == "Biscuit"  # input untouched
    with pytest.raises(ValueError, match="expected"):
        apply_errata(
            qs,
            Errata(
                dataset="t",
                version="v1",
                changes=[
                    Change(question_id="q1", field="gold_answer", old="Wrong", new="X", reason="r")
                ],
            ),
        )
    with pytest.raises(ValueError, match="cannot be patched"):
        apply_errata(
            qs,
            Errata(
                dataset="t",
                version="v1",
                changes=[Change(question_id="q1", field="question", old="x", new="y", reason="r")],
            ),
        )
