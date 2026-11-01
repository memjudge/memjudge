"""A private GCS mirror of the hosted datasets' pinned upstream files.

Jobs copy the file into the dataset cache instead of downloading it from the dataset's
host, so their egress rules need no dataset hosts. The core loader still verifies the
file's SHA-256, so a bad mirror fails the run rather than changing it. The mirror is not
public, so it does not redistribute datasets whose licence forbids that.
"""

from __future__ import annotations

from memjudge.datasets.loader import (
    LoadedDataset,
    cache_dir,
    cached_name,
    download,
    load_dataset,
)
from memjudge.datasets.registry import DATASETS

from memjudge_runner.bundle import Blobs

PREFIX = "datasets/"


def fetch(blobs: Blobs, bucket: str, name: str) -> None:
    """Copy a curated dataset's upstream file from the mirror into the cache."""
    dest = cache_dir() / cached_name(DATASETS[name])
    if not dest.exists():
        blobs.download(bucket, PREFIX + dest.name, dest)


def load_hosted(blobs: Blobs, bucket: str, name: str, expected_hash: str) -> LoadedDataset:
    if name in DATASETS:
        fetch(blobs, bucket, name)
    loaded = load_dataset(name)
    if loaded.hash != expected_hash:
        raise RuntimeError(f"dataset {name} hash {loaded.hash} != expected {expected_hash}")
    return loaded


def push(blobs: Blobs, bucket: str, name: str) -> str:
    """Download a curated dataset from upstream (verified) and copy it into the mirror."""
    path = download(DATASETS[name])
    blobs.put(bucket, PREFIX + path.name, path.read_bytes())
    return PREFIX + path.name
