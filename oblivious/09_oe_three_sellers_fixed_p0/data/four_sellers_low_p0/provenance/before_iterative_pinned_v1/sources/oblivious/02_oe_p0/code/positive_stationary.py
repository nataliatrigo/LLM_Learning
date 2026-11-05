"""Positive GTH with sparse storage and an independently usable dense reference.

The compiled kernel performs the same nonnegative elimination as the original
dense implementation. It stores only the fill actually generated; it never
clips a positive probability or uses signed matrix inversion. The shared
library is compiled locally into a source-hashed temporary cache, with no
third-party build dependency and no fast-math. Parallel workers may build it
concurrently: only a completed, atomically renamed library is loaded.
"""

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile
import warnings

import numpy as np
from scipy.sparse import csr_matrix


def _permutation(n, permutation):
    perm = np.arange(n) if permutation is None else np.asarray(permutation)
    if (perm.shape != (n,) or not np.issubdtype(perm.dtype, np.integer)
            or not np.array_equal(np.sort(perm), np.arange(n))):
        raise ValueError("GTH ordering must be a permutation of the states.")
    return perm


def _calendar(law, rates, perm):
    law /= law.sum()
    calendar = np.zeros(len(rates))
    calendar[perm] = law / rates[perm]
    if np.any(~np.isfinite(calendar)) or calendar.min() < 0 or calendar.sum() <= 0:
        raise ArithmeticError("Positive elimination produced invalid calendar mass.")
    return calendar / calendar.sum()


def dense_positive_invariant(jump, rates, permutation=None):
    """Original subtraction-free dense algorithm, retained as a reference."""
    n = len(rates)
    perm = _permutation(n, permutation)
    reduced = jump[perm, :][:, perm].toarray()
    for k in range(n - 1, 0, -1):
        denominator = reduced[k, :k].sum()
        if not np.isfinite(denominator) or denominator <= 0:
            raise ArithmeticError("Positive elimination failed: reducibility or numerical underflow.")
        reduced[:k, k] /= denominator
        reduced[:k, :k] += np.outer(reduced[:k, k], reduced[k, :k])
    law = np.zeros(n)
    law[0] = 1
    for k in range(1, n):
        law[k] = law[:k] @ reduced[:k, k]
        if law[k] > 1e100:
            law[:k + 1] /= law[k]
    return _calendar(law, rates, perm)


@lru_cache(maxsize=1)
def _kernel():
    source = Path(__file__).with_suffix(".cpp")
    compiler = shutil.which("g++") or shutil.which("c++")
    if compiler is None:
        raise RuntimeError("Sparse positive GTH requires a local C++ compiler (g++ or c++).")
    identity = source.read_bytes() + platform.machine().encode() + compiler.encode()
    digest = hashlib.sha256(identity).hexdigest()[:24]
    cache = Path(tempfile.gettempdir()) / "llm_learning_positive_gth"
    cache.mkdir(mode=0o700, exist_ok=True)
    library = cache / f"positive_gth_{digest}.so"
    if not library.exists():
        descriptor, temporary = tempfile.mkstemp(prefix=f"{digest}_", suffix=".so", dir=cache)
        os.close(descriptor)
        try:
            build = subprocess.run(
                [compiler, "-std=c++17", "-O3", "-fPIC", "-shared", "-fno-fast-math",
                 "-ffp-contract=off", str(source), "-o", temporary],
                capture_output=True, text=True, timeout=120, check=False)
            if build.returncode:
                raise RuntimeError(f"Sparse positive GTH compilation failed: {build.stderr[-2000:]}")
            os.replace(temporary, library)
        finally:
            Path(temporary).unlink(missing_ok=True)
    loaded = ctypes.CDLL(str(library))
    function = loaded.positive_gth
    int_array = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    double_array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    function.argtypes = [ctypes.c_int64, int_array, int_array, double_array, double_array]
    function.restype = ctypes.c_int
    # Retain the library handle alongside its function pointer.
    return loaded, function


def positive_invariant(jump, rates, permutation=None):
    """Sparse compiled GTH, preserving every representable positive entry.

    ``permutation`` has exactly the same meaning as for the dense reference:
    elimination proceeds backwards through the supplied state numbering.
    Arithmetic errors are never retried with a signed or truncated algorithm.
    """
    rates = np.asarray(rates, dtype=float)
    n = len(rates)
    if rates.shape != (n,) or np.any(~np.isfinite(rates)) or np.any(rates <= 0):
        raise ValueError("GTH exit rates must be finite and strictly positive.")
    perm = _permutation(n, permutation)
    reduced = csr_matrix(jump)[perm, :][:, perm].tocsr()
    reduced.sum_duplicates()
    if (reduced.shape != (n, n) or np.any(~np.isfinite(reduced.data))
            or np.any(reduced.data < 0)):
        raise ValueError("GTH jump kernel must be finite, nonnegative, and square.")
    try:
        _, function = _kernel()
    except (OSError, RuntimeError, subprocess.TimeoutExpired) as exc:
        if n > 4096:
            raise RuntimeError(f"Cannot safely fall back to dense GTH for {n} states: {exc}") from exc
        warnings.warn(f"Sparse positive GTH unavailable; using dense reference: {exc}", RuntimeWarning)
        return dense_positive_invariant(reduced, rates[perm])[np.argsort(perm)]
    law = np.empty(n)
    status = function(n, np.ascontiguousarray(reduced.indptr, dtype=np.int64),
                      np.ascontiguousarray(reduced.indices, dtype=np.int64),
                      np.ascontiguousarray(reduced.data, dtype=np.float64), law)
    if status:
        reasons = {1: "reducibility or numerical underflow", 2: "nonfinite positive elimination",
                   3: "insufficient memory", 4: "invalid sparse elimination structure"}
        raise ArithmeticError(f"Positive elimination failed: {reasons.get(status, 'unknown failure')}.")
    return _calendar(law, rates, perm)
