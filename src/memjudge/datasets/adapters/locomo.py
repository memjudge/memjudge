"""Adapter for LoCoMo (Maharana et al., 2024), ``data/locomo10.json``.

Category numbering follows the upstream code (not the paper's prose order):
1 multi-hop, 2 temporal, 3 open-domain, 4 single-hop, 5 adversarial.
Category 5 questions ask about something that was never said about the named
person; they are mapped to abstention questions with the upstream
``adversarial_answer`` kept as the distractor (see ADR 0005).
"""

from __future__ import annotations

import logging
import re
from datetime import datetime
from typing import Any

from memjudge.schema import Conversation, Question, QuestionKind, Session, Turn

log = logging.getLogger(__name__)

CATEGORIES = {
    1: "multi-hop",
    2: "temporal",
    3: "open-domain",
    4: "single-hop",
    5: "adversarial",
}
ABSTENTION_GOLD = "Not answerable: the conversation does not contain this information."

_SESSION_KEY = re.compile(r"^session_(\d+)$")


def parse_datetime(value: str) -> datetime:
    """Parse LoCoMo timestamps such as ``"1:56 pm on 8 May, 2023"``."""
    return datetime.strptime(value.strip(), "%I:%M %p on %d %B, %Y")


def convert(raw: list[dict[str, Any]]) -> tuple[list[Conversation], list[Question]]:
    conversations: list[Conversation] = []
    questions: list[Question] = []
    for sample in raw:
        conv_id = str(sample["sample_id"])
        conv = sample["conversation"]
        numbers = sorted(
            int(m.group(1)) for key in conv if (m := _SESSION_KEY.match(key)) and conv[key]
        )
        sessions: list[Session] = []
        for n in numbers:
            started = parse_datetime(conv[f"session_{n}_date_time"])
            turns = tuple(
                Turn(id=t["dia_id"], speaker=t["speaker"], text=_turn_text(t), timestamp=started)
                for t in conv[f"session_{n}"]
            )
            sessions.append(Session(id=f"session_{n}", started_at=started, turns=turns))
        conversation = Conversation(id=conv_id, sessions=tuple(sessions))
        conversations.append(conversation)

        known_turns = {t.id for s in sessions for t in s.turns}
        asked_at = sessions[-1].started_at if sessions else None
        for i, qa in enumerate(sample["qa"]):
            evidence = tuple(e for e in qa.get("evidence", []) if e in known_turns)
            dropped = [e for e in qa.get("evidence", []) if e not in known_turns]
            if dropped:
                log.debug("locomo %s q%d: ignoring unknown evidence ids %s", conv_id, i, dropped)
            category = int(qa["category"])
            if category == 5:
                questions.append(
                    Question(
                        id=f"{conv_id}:q{i}",
                        conversation_id=conv_id,
                        question=qa["question"],
                        gold_answer=ABSTENTION_GOLD,
                        category=CATEGORIES[5],
                        kind=QuestionKind.ABSTENTION,
                        asked_at=asked_at,
                        evidence_turn_ids=evidence,
                        distractor_answer=str(qa.get("adversarial_answer") or "") or None,
                    )
                )
            else:
                questions.append(
                    Question(
                        id=f"{conv_id}:q{i}",
                        conversation_id=conv_id,
                        question=qa["question"],
                        gold_answer=str(qa["answer"]),
                        category=CATEGORIES[category],
                        asked_at=asked_at,
                        evidence_turn_ids=evidence,
                    )
                )
    return conversations, questions


def _turn_text(turn: dict[str, Any]) -> str:
    text = str(turn["text"])
    # Image turns carry a generated caption; keep it so the content is not lost.
    caption = turn.get("blip_caption")
    if caption:
        text = f"{text} [shares an image: {caption}]"
    return text
