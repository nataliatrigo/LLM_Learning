"""Read saved comparison artifacts inside the active repository."""

from pathlib import Path
import json

import numpy as np


def reference_source(path, project_root):
    """Record a repository-relative source; external paths are not references."""
    return str(Path(path).resolve().relative_to(Path(project_root).resolve()))


def read_saved_bundle(base):
    """Decode a JSON/NPZ bundle without importing an experiment runner."""
    base = Path(base)
    metadata = json.loads(base.with_suffix(".json").read_text())
    with np.load(base.with_suffix(".npz"), allow_pickle=False) as stored:
        def unpack(value):
            if isinstance(value, dict):
                if set(value) == {"__array__"}:
                    return stored[value["__array__"]].copy()
                return {key: unpack(item) for key, item in value.items()}
            if isinstance(value, list):
                return [unpack(item) for item in value]
            return value
        return unpack(metadata)
