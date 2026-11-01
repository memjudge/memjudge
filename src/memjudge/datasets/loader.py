"""Load a curated dataset (download + verify + convert) or a local canonical JSON file."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path

import httpx

from memjudge.canonical import canonical_hash, file_hash
from memjudge.datasets.errata import apply_errata, load_errata
from memjudge.datasets.registry import DATASETS, RegisteredDataset
from memjudge.datasets.selection import Selection, select
from memjudge.schema import Dataset


@dataclass(frozen=True)
class LoadedDataset:
    dataset: Dataset
    hash: str
    """SHA-256 of the canonical converted dataset (after errata). Results compare only within it."""
    curated: bool
    upstream_sha256: str | None = None
    errata_applied: int = 0
    selection: Selection | None = None
    """Set when only part of the dataset is loaded; ``hash`` is still the full dataset's."""

    @property
    def label(self) -> str:
        return f"{self.dataset.name} {self.dataset.version} ({self.hash[7:15]})"


def cache_dir() -> Path:
    root = os.environ.get("MEMJUDGE_CACHE") or Path.home() / ".cache" / "memjudge"
    path = Path(root) / "datasets"
    path.mkdir(parents=True, exist_ok=True)
    return path


def cached_name(entry: RegisteredDataset) -> str:
    """File name of the pinned upstream file in the cache."""
    return f"{entry.name}-{entry.version}-{entry.sha256[:12]}.json"


def download(entry: RegisteredDataset, dest_dir: Path | None = None) -> Path:
    """Download the pinned upstream file once and verify its SHA-256."""
    dest_dir = dest_dir or cache_dir()
    dest = dest_dir / cached_name(entry)
    expected = "sha256:" + entry.sha256
    if dest.exists() and file_hash(dest) == expected:
        return dest
    tmp = dest.with_suffix(".part")
    with httpx.stream("GET", entry.url, follow_redirects=True, timeout=120) as resp:
        resp.raise_for_status()
        with tmp.open("wb") as f:
            for chunk in resp.iter_bytes(1 << 20):
                f.write(chunk)
    actual = file_hash(tmp)
    if actual != expected:
        tmp.unlink()
        raise ValueError(f"{entry.name}: upstream hash mismatch ({actual} != {expected})")
    tmp.replace(dest)
    return dest


def load_curated(name: str) -> LoadedDataset:
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}")
    entry = DATASETS[name]
    raw = json.loads(download(entry).read_text(encoding="utf-8"))
    conversations, questions = entry.convert(raw)
    errata = load_errata(entry.name, entry.version)
    if errata:
        questions = apply_errata(questions, errata)
    dataset = Dataset(
        name=entry.name,
        version=entry.version,
        source=entry.source,
        license=entry.license,
        conversations=tuple(conversations),
        questions=tuple(questions),
    )
    return LoadedDataset(
        dataset=dataset,
        hash=canonical_hash(dataset),
        curated=True,
        upstream_sha256=entry.sha256,
        errata_applied=len(errata.changes) if errata else 0,
    )


def load_file(path: Path) -> LoadedDataset:
    dataset = Dataset.model_validate_json(path.read_text(encoding="utf-8"))
    return LoadedDataset(dataset=dataset, hash=canonical_hash(dataset), curated=False)


def load_dataset(spec: str) -> LoadedDataset:
    """``spec`` is a curated dataset name (``locomo``) or a path to a canonical JSON file."""
    path = Path(spec)
    if path.suffix == ".json" and path.exists():
        return load_file(path)
    return load_curated(spec)


def apply_selection(loaded: LoadedDataset, selection: Selection) -> LoadedDataset:
    """Narrow a loaded dataset. The hash stays the full dataset's hash (see Selection)."""
    if selection.is_full:
        return loaded
    return replace(loaded, dataset=select(loaded.dataset, selection), selection=selection)
