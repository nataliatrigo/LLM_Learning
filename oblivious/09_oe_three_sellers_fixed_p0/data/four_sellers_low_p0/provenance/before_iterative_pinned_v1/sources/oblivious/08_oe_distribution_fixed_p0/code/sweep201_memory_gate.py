"""Cross-process weighted memory leases for dense positive GTH allocations.

Six 4-GiB tokens cap reserved dense storage at 24 GiB. Each call reserves its
8*n*n array plus 64 MiB of overhead, rounded up to whole tokens. A coordinator
lock makes acquisition of multiple tokens atomic: insufficient partial leases
are immediately released before retrying. POSIX releases token locks when the
owning process exits, including abrupt termination.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import stat
import tempfile
import time

DEFAULT_CAPACITY = 6
TOKEN_BYTES = 4*1024**3
OVERHEAD_BYTES = 64*1024**2
POLL_SECONDS = .1
DEFAULT_DIRECTORY = Path(tempfile.gettempdir())/f'llm_learning_sweep201_dense_memory_{os.getuid()}'


@dataclass(frozen=True)
class MemoryLease:
    states: int
    reserved_bytes: int
    units: int
    token_ids: tuple[int,...]
    capacity: int
    token_bytes: int


def helper_reference():
    from oblivious.oe_distribution_fixed_p0.code.run_experiment import ROOT
    path = Path(__file__).resolve()
    return {str(path.relative_to(ROOT)):sha256(path.read_bytes()).hexdigest()}


def _close_tokens(tokens):
    for _,descriptor in tokens:
        os.close(descriptor)


def _private_directory(directory):
    directory.mkdir(mode=0o700,parents=True,exist_ok=True)
    information = directory.lstat()
    if (not stat.S_ISDIR(information.st_mode) or information.st_uid != os.getuid()
            or stat.S_IMODE(information.st_mode)&0o077):
        raise PermissionError('Dense GTH gate requires a private directory owned by this user')


@contextmanager
def gated(n,*,directory=None,capacity=DEFAULT_CAPACITY,token_bytes=TOKEN_BYTES,
          overhead_bytes=OVERHEAD_BYTES,poll_seconds=POLL_SECONDS):
    """Lease enough memory units for n actual states; no memory is allocated.

    Optional settings support small synthetic multiprocess tests. All processes
    sharing a directory must use the same capacity and token size. Oversized
    requests fail immediately, rather than waiting for an impossible lease.
    """
    if (isinstance(n,bool) or not isinstance(n,int) or n < 1
            or isinstance(capacity,bool) or not isinstance(capacity,int) or capacity < 1
            or isinstance(token_bytes,bool) or not isinstance(token_bytes,int) or token_bytes < 1
            or not isinstance(overhead_bytes,int) or overhead_bytes < 0
            or not math.isfinite(poll_seconds) or poll_seconds <= 0):
        raise ValueError('Dense GTH gate needs positive integer states/capacity/bytes and polling interval')
    reserved_bytes = 8*n*n+overhead_bytes
    units = (reserved_bytes+token_bytes-1)//token_bytes
    if units > capacity:
        raise ValueError(f'Dense GTH request needs {units} memory tokens; budget is {capacity}')
    directory = DEFAULT_DIRECTORY if directory is None else Path(directory)
    _private_directory(directory)
    flags = os.O_RDWR|os.O_CREAT|getattr(os,'O_CLOEXEC',0)|getattr(os,'O_NOFOLLOW',0)
    coordinator = os.open(directory/'coordinator.lock',flags,0o600)
    tokens = []
    settings = dict(capacity=capacity,token_bytes=token_bytes)
    try:
        while True:
            fcntl.flock(coordinator,fcntl.LOCK_EX)
            try:
                configuration = directory/'configuration.json'
                if configuration.exists():
                    if json.loads(configuration.read_text()) != settings:
                        raise ValueError('Dense GTH gate directory has a different capacity/token calibration')
                else:
                    # Only the coordinator owner may initialize this tiny file.
                    descriptor,pending = tempfile.mkstemp(prefix='configuration.pending.',dir=directory)
                    try:
                        with os.fdopen(descriptor,'w') as stream:
                            stream.write(json.dumps(settings,sort_keys=True)+'\n')
                        os.replace(pending,configuration)
                    finally:
                        Path(pending).unlink(missing_ok=True)
                for token in range(capacity):
                    descriptor = os.open(directory/f'token_{token:02d}.lock',flags,0o600)
                    try:
                        fcntl.flock(descriptor,fcntl.LOCK_EX|fcntl.LOCK_NB)
                    except BlockingIOError:
                        os.close(descriptor)
                        continue
                    tokens.append((token,descriptor))
                    if len(tokens) == units:
                        break
                if len(tokens) != units:
                    _close_tokens(tokens)
                    tokens = []
            except BaseException:
                _close_tokens(tokens)
                tokens = []
                raise
            finally:
                fcntl.flock(coordinator,fcntl.LOCK_UN)
            if len(tokens) == units:
                break
            time.sleep(poll_seconds)
        yield MemoryLease(states=n,reserved_bytes=reserved_bytes,units=units,
                          token_ids=tuple(token for token,_ in tokens),
                          capacity=capacity,token_bytes=token_bytes)
    finally:
        _close_tokens(tokens)
        os.close(coordinator)
