"""Adapter for LongMemEval (Wu et al., ICLR 2025), cleaned release.

Every LongMemEval question comes with its own haystack of sessions, so each
question becomes its own conversation. Turns flagged ``has_answer`` are the
evidence. Question ids ending in ``_abs`` are abstention questions;
``knowledge-update`` questions are judged for stale facts as well.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from memjudge.schema import Conversation, Question, QuestionKind, Session, Turn


def parse_datetime(value: str) -> datetime:
    """Parse LongMemEval timestamps such as ``"2023/04/10 (Mon) 23:07"``."""
    date_part, _, time_part = value.partition(" (")
    time_part = time_part.split(") ", 1)[1]
    return datetime.strptime(f"{date_part} {time_part}", "%Y/%m/%d %H:%M")


def convert(raw: list[dict[str, Any]]) -> tuple[list[Conversation], list[Question]]:
    conversations: list[Conversation] = []
    questions: list[Question] = []
    for item in raw:
        qid = str(item["question_id"])
        dated = [
            (parse_datetime(date), str(sid), turns)
            for date, sid, turns in zip(
                item["haystack_dates"],
                item["haystack_session_ids"],
                item["haystack_sessions"],
                strict=True,
            )
        ]
        dated.sort(key=lambda x: x[0])  # ingest in chronological order (stable sort)

        sessions: list[Session] = []
        evidence: list[str] = []
        seen: dict[str, Any] = {}
        for started, sid, raw_turns in dated:
            # A few haystacks list the same session twice; ingest it once.
            if sid in seen:
                if seen[sid] != raw_turns:
                    raise ValueError(f"{qid}: session {sid} repeated with different content")
                continue
            seen[sid] = raw_turns
            turns: list[Turn] = []
            for j, t in enumerate(raw_turns):
                turn_id = f"{sid}:{j}"
                turns.append(
                    Turn(
                        id=turn_id,
                        speaker=str(t["role"]),
                        text=str(t["content"]),
                        timestamp=started,
                    )
                )
                if t.get("has_answer"):
                    evidence.append(turn_id)
            sessions.append(Session(id=sid, started_at=started, turns=tuple(turns)))
        conversations.append(Conversation(id=qid, sessions=tuple(sessions)))

        qtype = str(item["question_type"])
        if qid.endswith("_abs"):
            kind = QuestionKind.ABSTENTION
        elif qtype == "knowledge-update":
            kind = QuestionKind.KNOWLEDGE_UPDATE
        else:
            kind = QuestionKind.STANDARD
        questions.append(
            Question(
                id=qid,
                conversation_id=qid,
                question=str(item["question"]),
                gold_answer=str(item["answer"]),
                category=qtype,  # _abs questions keep their type, as LongMemEval reports
                kind=kind,
                asked_at=parse_datetime(str(item["question_date"])),
                evidence_turn_ids=tuple(evidence),
            )
        )
    return conversations, questions
