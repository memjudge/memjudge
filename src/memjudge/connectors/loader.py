"""Discover and load connectors from ``connectors/<system>/<name>/``."""

from __future__ import annotations

import importlib.util
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from memjudge.canonical import directory_hash
from memjudge.connectors.base import BaseConnector, ConnectorSpec

_PACKAGE_DIR = Path(__file__).resolve().parents[1]


def connectors_root() -> Path:
    """``$MEMJUDGE_CONNECTORS``, else the repo checkout, else the copy bundled in the wheel."""
    if env := os.environ.get("MEMJUDGE_CONNECTORS"):
        return Path(env)
    checkout = _PACKAGE_DIR.parents[1] / "connectors"
    if checkout.is_dir():
        return checkout
    return _PACKAGE_DIR / "_connectors"


@dataclass(frozen=True)
class ConnectorRef:
    """A connector's spec and code hash, read without importing its code."""

    spec: ConnectorSpec
    path: Path
    code_hash: str


@dataclass(frozen=True)
class LoadedConnector(ConnectorRef):
    cls: type[BaseConnector]

    def instantiate(
        self, env: Mapping[str, str], overrides: Mapping[str, Any] | None = None
    ) -> BaseConnector:
        missing = [k for k in self.spec.required_keys if not env.get(k)]
        if missing:
            raise KeyError(f"connector {self.spec.id} needs keys: {', '.join(missing)}")
        keys = {k: env[k] for k in self.spec.required_keys}
        return self.cls(keys, self.settings(overrides))

    def settings(self, overrides: Mapping[str, Any] | None = None) -> dict[str, Any]:
        return {**self.spec.settings, **(overrides or {})}


def list_connectors(root: Path | None = None) -> list[str]:
    root = root or connectors_root()
    return sorted(p.parent.relative_to(root).as_posix() for p in root.glob("*/*/connector.yaml"))


def load_spec(connector_id: str, root: Path | None = None) -> ConnectorSpec:
    """Read ``connector.yaml`` without importing the connector (or its vendor SDK)."""
    root = root or connectors_root()
    path = root / connector_id / "connector.yaml"
    if not path.is_file():
        raise KeyError(f"unknown connector {connector_id!r}; available: {list_connectors(root)}")
    spec = ConnectorSpec.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))
    if spec.id != connector_id:
        raise ValueError(f"connector.yaml id {spec.id!r} does not match folder {connector_id!r}")
    return spec


def connector_ref(connector_id: str, root: Path | None = None) -> ConnectorRef:
    """Spec and code hash only: for code that must not run the connector (or its SDK)."""
    root = root or connectors_root()
    spec = load_spec(connector_id, root)
    return ConnectorRef(
        spec=spec, path=root / connector_id, code_hash=directory_hash(root / connector_id)
    )


def load_connector(connector_id: str, root: Path | None = None) -> LoadedConnector:
    root = root or connectors_root()
    path = root / connector_id
    spec = load_spec(connector_id, root)

    module_name = "memjudge_connector_" + connector_id.replace("/", "_").replace("-", "_")
    module_spec = importlib.util.spec_from_file_location(module_name, path / "connector.py")
    if module_spec is None or module_spec.loader is None:
        raise ImportError(f"cannot import {path / 'connector.py'}")
    module = importlib.util.module_from_spec(module_spec)
    sys.modules[module_name] = module  # dataclasses and pickling look the module up by name
    try:
        module_spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        del sys.modules[module_name]
        hint = f" Install it with: pip install '{spec.sdk}'" if spec.sdk else ""
        raise ImportError(f"connector {connector_id} needs {exc.name!r}.{hint}") from exc
    except BaseException:
        del sys.modules[module_name]
        raise
    cls = getattr(module, "Connector", None)
    if not (isinstance(cls, type) and issubclass(cls, BaseConnector)):
        raise TypeError(f"{connector_id}: connector.py must define Connector(BaseConnector)")
    return LoadedConnector(spec=spec, path=path, code_hash=directory_hash(path), cls=cls)
