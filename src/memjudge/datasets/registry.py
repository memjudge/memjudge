"""Curated datasets, pinned to an exact upstream revision and file hash.

Neither LoCoMo (CC BY-NC 4.0) nor LongMemEval (MIT, with third-party chat
haystacks) is redistributed: the raw file is downloaded from the pinned
upstream URL at run time and its SHA-256 is checked before use (ADR 0003).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from memjudge.datasets.adapters import locomo, longmemeval
from memjudge.schema import Conversation, Question

Converter = Callable[[Any], tuple[list[Conversation], list[Question]]]

_LME_REV = "98d7416c24c778c2fee6e6f3006e7a073259d48f"
_LME_BASE = f"https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/{_LME_REV}"


@dataclass(frozen=True)
class RegisteredDataset:
    name: str
    version: str
    title: str
    url: str
    sha256: str
    license: str
    source: str
    convert: Converter


DATASETS: dict[str, RegisteredDataset] = {
    d.name: d
    for d in [
        RegisteredDataset(
            name="locomo",
            version="v1",
            title="LoCoMo (10 conversations, 1,986 questions)",
            url="https://raw.githubusercontent.com/snap-research/locomo/"
            "3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/data/locomo10.json",
            sha256="79fa87e90f04081343b8c8debecb80a9a6842b76a7aa537dc9fdf651ea698ff4",
            license="CC-BY-NC-4.0",
            source="https://github.com/snap-research/locomo",
            convert=locomo.convert,
        ),
        RegisteredDataset(
            name="longmemeval-s",
            version="v1",
            title="LongMemEval-S cleaned (500 questions, ~115k tokens each)",
            url=f"{_LME_BASE}/longmemeval_s_cleaned.json",
            sha256="d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442",
            license="MIT",
            source="https://github.com/xiaowu0162/LongMemEval",
            convert=longmemeval.convert,
        ),
        RegisteredDataset(
            name="longmemeval-oracle",
            version="v1",
            title="LongMemEval oracle (evidence sessions only; quick checks)",
            url=f"{_LME_BASE}/longmemeval_oracle.json",
            sha256="821a2034d219ab45846873dd14c14f12cfe7776e73527a483f9dac095d38620c",
            license="MIT",
            source="https://github.com/xiaowu0162/LongMemEval",
            convert=longmemeval.convert,
        ),
    ]
}
