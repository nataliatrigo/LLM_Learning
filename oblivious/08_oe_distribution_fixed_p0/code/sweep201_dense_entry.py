"""Adaptive-face execution with exact-order dense positive GTH and a RAM gate.

The mathematical kernel, floating-point operation order and original Python
certificates remain unchanged. Independent cases use six worker processes;
each dense GTH uses two threads and reserves its matrix in a shared 24 GiB
memory budget before allocation. Completed historical bundles are preserved.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from oblivious.oe_distribution_fixed_p0.code import sweep201_face_entry as face_entry
from oblivious.oe_distribution_fixed_p0.code import sweep201_dense_gth as backend

DENSE_MODULE = 'oblivious.oe_distribution_fixed_p0.code.sweep201_dense_entry'
GTH_THREADS = 2


def main():
    original_reference = face_entry.entry.helper_reference
    original_install = face_entry.entry.worker.install

    def phase_reference():
        path = Path(__file__).resolve()
        return dict(original_reference(), **backend.helper_reference(), **{
            str(path.relative_to(face_entry.entry.worker.runner.ROOT)):
                sha256(path.read_bytes()).hexdigest()})

    def install_dense_after_worker(calibration):
        original_install(calibration)
        backend.install(threads=GTH_THREADS)

    face_entry.entry.helper_reference = phase_reference
    face_entry.entry.worker.install = install_dense_after_worker
    face_entry.FACE_MODULE = DENSE_MODULE
    face_entry.main()


if __name__ == '__main__':
    main()
