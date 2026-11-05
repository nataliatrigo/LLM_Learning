"""Optional bitwise-equivalent row-parallel GTH, with original Python checks.

Only ``positive_stationary._kernel`` is replaced. The original permutation,
calendar conversion, positivity checks, full-kernel residual checks and reverse
ordering certificate remain in their original modules. No source is rewritten.
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


def helper_reference():
    from oblivious.oe_distribution_fixed_p0.code import multi_p0_run as runner
    return {str(path.relative_to(runner.ROOT)): sha256(path.read_bytes()).hexdigest()
            for path in (Path(__file__).resolve(), Path(__file__).with_suffix('.cpp').resolve())}


@lru_cache(maxsize=1)
def _kernel():
    source = Path(__file__).with_suffix('.cpp')
    compiler = shutil.which('g++') or shutil.which('c++')
    if compiler is None:
        raise RuntimeError('Parallel positive GTH requires a local C++ compiler')
    flags = ['-std=c++17', '-O3', '-fPIC', '-shared', '-fno-fast-math',
             '-ffp-contract=off', '-fopenmp']
    identity = source.read_bytes()+platform.machine().encode()+compiler.encode()+repr(flags).encode()
    digest = sha256(identity).hexdigest()[:24]
    cache = Path(tempfile.gettempdir())/'llm_learning_parallel_positive_gth'
    cache.mkdir(mode=0o700, exist_ok=True)
    library = cache/f'positive_gth_{digest}.so'
    if not library.exists():
        descriptor, temporary = tempfile.mkstemp(prefix=f'{digest}_', suffix='.so', dir=cache)
        os.close(descriptor)
        try:
            build = subprocess.run([compiler, *flags, str(source), '-o', temporary],
                                   capture_output=True, text=True, timeout=120, check=False)
            if build.returncode:
                raise RuntimeError('Parallel positive GTH compilation failed: '+build.stderr[-2000:])
            os.replace(temporary, library)
        finally:
            Path(temporary).unlink(missing_ok=True)
    loaded = ctypes.CDLL(str(library))
    function = loaded.positive_gth
    int_array = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags='C_CONTIGUOUS')
    double_array = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
    function.argtypes = [ctypes.c_int64, int_array, int_array, double_array, double_array]
    function.restype = ctypes.c_int
    loaded.positive_gth_set_threads.argtypes = [ctypes.c_int]
    loaded.positive_gth_set_threads.restype = ctypes.c_int
    return loaded, function


def set_threads(threads=2):
    if not isinstance(threads, int) or not 1 <= threads <= 64:
        raise ValueError('Parallel positive GTH threads must be an integer in [1,64]')
    loaded, _ = _kernel()
    if loaded.positive_gth_set_threads(threads):
        raise RuntimeError('Parallel positive GTH rejected thread count')


def install(threads=2):
    """Explicit per-process opt-in; return the unchanged original kernel loader."""
    set_threads(threads)
    previous = original._kernel
    original._kernel = _kernel
    return previous
