"""Import compatible completed scalar evaluations without constructing a game.

The source campaign stays read-only. A checkpoint identity is independent of
its campaign name, but binds its grid, economics, tolerances and numerical
sources. Only immutable generations referenced by its validated index move.
"""

from __future__ import annotations

from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
import tempfile
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def expected_checkpoint_identity(model, rho, config):
    """Use the original class's configuration method, without allocating Game.

    These are precisely the attributes read by ScalarExperiment.configuration
    and QualityExperiment.configuration. The source fingerprint also binds
    those implementations, so an incompatible future change fails closed.
    """
    from oblivious.oe_p0.code.core import ScalarExperiment
    from oblivious.oe_p0.code.primitives import Parameters
    from oblivious.oe_p0.code.refinement import _source_fingerprint

    if model == "p0":
        experiment_class = ScalarExperiment
    elif model == "monopoly":
        from oblivious.oe_following_monopoly.code.model import QualityExperiment
        experiment_class = QualityExperiment
    else:
        raise ValueError("Scalar checkpoints exist only for p0 and monopoly")

    experiment = experiment_class.__new__(experiment_class)
    parameters = dict(config["parameters"])
    parameters["rho"] = float(rho)
    experiment.params = Parameters(**parameters)
    experiment.n = 201
    experiment.game = SimpleNamespace(order=config.get("order", 128))
    experiment.dp_tol = config.get("dp_tol", 1e-10)
    experiment.stationary_tol = config.get("stationary_tol", 1e-11)
    experiment.stability_tol = config.get("stability_tol", 1e-7)
    experiment.near_tie_tol = config.get("near_tie_tol", 1e-7)
    if model == "monopoly":
        experiment.weighting = "purchase"
        experiment.p0_bounds = (experiment.params.p1, experiment.params.p2)
    return dict(
        configuration=experiment.configuration(),
        consistency_tolerance=config.get("tol", 1e-7),
        experiment_class=experiment_class.__module__ + "." + experiment_class.__qualname__,
        numerical_sources=_source_fingerprint(experiment_class),
    )


def _checkpoint_directory(campaign, model, rho):
    return (Path(campaign) / model / ("rho_" + str(float(rho)).replace(".", "p"))
            / "n201" / "search_checkpoints")


def _validated_index(directory, expected):
    index_path = directory / "evaluations.json"
    if index_path.is_symlink() or not index_path.is_file():
        raise ValueError(f"Checkpoint index is missing or is a symlink: {index_path}")
    index_bytes = index_path.read_bytes()
    index = json.loads(index_bytes)
    if index.get("version") != 1 or not isinstance(index.get("entries"), dict):
        raise ValueError(f"Unsupported checkpoint index: {index_path}")
    if index.get("identity") != expected:
        raise ValueError(f"Checkpoint configuration or numerical sources differ: {index_path}")
    files = {}
    for entry in index["entries"].values():
        stem, hashes = entry.get("stem"), entry.get("sha256")
        if (not isinstance(stem, str) or not stem or Path(stem).name != stem
                or stem in (".", "..") or not isinstance(hashes, dict)
                or set(hashes) != {".json", ".npz"}):
            raise ValueError(f"Malformed checkpoint generation: {index_path}")
        for suffix, expected_digest in hashes.items():
            path = directory / (stem + suffix)
            if (not isinstance(expected_digest, str) or len(expected_digest) != 64
                    or path.is_symlink() or not path.is_file()
                    or sha256(path.read_bytes()).hexdigest() != expected_digest):
                raise ValueError(f"Checkpoint entry is missing or corrupted: {path}")
            files[path.name] = expected_digest
    return index, index_bytes, files


def import_checkpoints(source_directory, target_directory, model, rho, config):
    """Import n201 scalar checkpoints between campaign roots, or return None.

    An existing target is reusable only when its identity, complete index and
    referenced file hashes agree exactly. Existing data are never overwritten.
    No numerical evaluation or stationary solve runs here.
    """
    if model not in ("p0", "monopoly"):
        return None
    source = _checkpoint_directory(source_directory, model, rho)
    target = _checkpoint_directory(target_directory, model, rho)
    if not (source / "evaluations.json").exists():
        return None
    if source.resolve() == target.resolve():
        raise ValueError("Source and target checkpoint directories must differ")

    expected = expected_checkpoint_identity(model, rho, config)
    index, index_bytes, files = _validated_index(source, expected)
    reused = False
    if target.exists() and any(target.iterdir()):
        existing, _, existing_files = _validated_index(target, expected)
        if existing != index or existing_files != files:
            raise ValueError(f"Existing target contains different checkpoint evaluations: {target}")
        reused = True
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".import_201_", dir=target.parent) as temporary:
            pending = Path(temporary) / "search_checkpoints"
            pending.mkdir()
            for name, expected_digest in files.items():
                destination = pending / name
                shutil.copyfile(source / name, destination)
                if sha256(destination.read_bytes()).hexdigest() != expected_digest:
                    raise ValueError(f"Checkpoint changed while copying: {source / name}")
            (pending / "evaluations.json").write_bytes(index_bytes)
            _validated_index(pending, expected)
            if target.exists():
                target.rmdir()  # Fails safely if another writer has populated it.
            pending.rename(target)

    return dict(
        kind="imported_completed_n201_evaluations",
        model=model, rho=float(rho), n=201,
        source_directory=str(source.resolve()),
        target_directory=str(target.resolve()),
        source_index_sha256=sha256(index_bytes).hexdigest(),
        identity_sha256=sha256(json.dumps(expected, sort_keys=True).encode()).hexdigest(),
        entries=len(index["entries"]), files=len(files),
        evaluation_kinds=dict(Counter(token.split(":", 1)[0] for token in index["entries"])),
        existing_identical_target=reused,
        scope="Completed evaluations only; an interrupted numerical solve restarts",
    )
