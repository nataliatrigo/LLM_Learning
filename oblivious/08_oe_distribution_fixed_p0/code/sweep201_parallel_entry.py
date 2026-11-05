"""Optional adaptive-face phase with two-thread, exact-order positive GTH.

The frozen worker, scheduler, certificates and numerical sources are untouched.
Each worker records this entry point and both backend source hashes as proposal
reference inputs. The backend changes only independent row execution order.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from oblivious.oe_distribution_fixed_p0.code import sweep201_face_entry as face_entry
from oblivious.oe_distribution_fixed_p0.code import sweep201_parallel_gth as backend

PARALLEL_MODULE = 'oblivious.oe_distribution_fixed_p0.code.sweep201_parallel_entry'
GTH_THREADS = 2


def main():
    original_reference = face_entry.entry.helper_reference
    original_install = face_entry.entry.worker.install

    def phase_reference():
        path = Path(__file__).resolve()
        return dict(original_reference(), **backend.helper_reference(), **{
            str(path.relative_to(face_entry.entry.worker.runner.ROOT)):
                sha256(path.read_bytes()).hexdigest()})

    def install_parallel_after_worker(calibration):
        original_install(calibration)
        backend.install(threads=GTH_THREADS)

    face_entry.entry.helper_reference = phase_reference
    face_entry.entry.worker.install = install_parallel_after_worker
    # The face wrapper already redirects every batch child to FACE_MODULE.
    face_entry.FACE_MODULE = PARALLEL_MODULE
    face_entry.main()


if __name__ == '__main__':
    main()
