"""Output paths and small serialization helpers for reproducible runs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ExperimentPaths:
    """Filesystem layout for one named experiment profile."""

    root: Path
    profile: str

    @property
    def results(self) -> Path:
        return self.root / "results" / self.profile

    @property
    def figures(self) -> Path:
        return self.root / "figures" / self.profile

    def create(self) -> None:
        self.results.mkdir(parents=True, exist_ok=True)
        self.figures.mkdir(parents=True, exist_ok=True)


def canonical_json(data: dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, indent=2, allow_nan=True)


def configuration_hash(data: dict[str, Any]) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(data) + "\n", encoding="utf-8")


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def read_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)
