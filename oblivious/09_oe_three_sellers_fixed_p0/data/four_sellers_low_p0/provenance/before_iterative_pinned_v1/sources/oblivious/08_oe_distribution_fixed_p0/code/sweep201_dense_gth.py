"""Optional dense-storage GTH with identical original floating-point order.

The kernel keeps the original backward elimination, per-row update sequence and
forward substitution. Dense zero entries replace absent sparse entries without
discarding any positive number. The original Python permutation, calendar and
certificate machinery is retained. Every allocation is guarded by the shared
campaign memory gate, using the actual number of stationary states.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
from hashlib import sha256
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tempfile

import numpy as np

from oblivious.oe_p0.code import positive_stationary as original
from oblivious.oe_distribution_fixed_p0.code import sweep201_memory_gate as memory_gate


def helper_reference():
    from oblivious.oe_distribution_fixed_p0.code import multi_p0_run as runner
    return {str(path.relative_to(runner.ROOT)): sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), Path(__file__).with_suffix('.cpp').resolve(),
                         Path(memory_gate.__file__).resolve())}


def _memory_lease(n):
    return memory_gate.gated(int(n))


@lru_cache(maxsize=1)
def _kernel():
    source = Path(__file__).with_suffix('.cpp')
    compiler = shutil.which('g++') or shutil.which('c++')
    if compiler is None:
        raise RuntimeError('Dense positive GTH requires a local C++ compiler')
    flags = ['-std=c++17', '-O3', '-fPIC', '-shared', '-fno-fast-math',
             '-ffp-contract=off', '-fopenmp']
    identity = source.read_bytes()+platform.machine().encode()+compiler.encode()+repr(flags).encode()
    digest = sha256(identity).hexdigest()[:24]
    cache = Path(tempfile.gettempdir())/'llm_learning_dense_positive_gth'
    cache.mkdir(mode=0o700, exist_ok=True)
    library = cache/f'positive_gth_{digest}.so'
    if not library.exists():
        descriptor, temporary = tempfile.mkstemp(prefix=f'{digest}_', suffix='.so', dir=cache)
        os.close(descriptor)
        try:
            build = subprocess.run([compiler, *flags, str(source), '-o', temporary],
                                   capture_output=True, text=True, timeout=120, check=False)
            if build.returncode:
                raise RuntimeError('Dense positive GTH compilation failed: '+build.stderr[-2000:])
            os.replace(temporary, library)
        finally:
            Path(temporary).unlink(missing_ok=True)
    loaded = ctypes.CDLL(str(library))
    raw = loaded.positive_gth
    int_array = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS')
    double_array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    raw.argtypes = [ctypes.c_int64, int_array, int_array, double_array, double_array]
    raw.restype = ctypes.c_int

    def leased_function(n, indptr, indices, data, law):
        with _memory_lease(n):
            return raw(n, indptr, indices, data, law)

    return loaded, leased_function


def install(threads=2):
    """Opt in for this process; original checks call the guarded new kernel."""
    if threads != 2:
        raise ValueError('The certified dense positive GTH backend uses exactly two threads')
    previous = original._kernel
    original._kernel = _kernel
    return previous
