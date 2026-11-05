"""Dense execution phase with pure-policy and complementarity proposals.

This repairs candidate search when an interpolated mixed node should become a
strict pure action. Every acceptance still uses the original full certificate.
The immutable dense-storage backend and its shared memory gate are reused.
"""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from oblivious.oe_distribution_fixed_p0.code import sweep201_dense_entry as phase
from oblivious.oe_distribution_fixed_p0.code import sweep201_complementarity_proposal as proposal

COMPLEMENTARITY_MODULE = 'oblivious.oe_distribution_fixed_p0.code.sweep201_complementarity_entry'


def main():
    entry = phase.face_entry.entry
    original_reference = entry.helper_reference
    original_install = entry.worker.install

    def phase_reference():
        path = Path(__file__).resolve()
        return dict(original_reference(), **proposal.helper_reference(), **{
            str(path.relative_to(entry.worker.runner.ROOT)):
                sha256(path.read_bytes()).hexdigest()})

    def install_proposal_after_worker(calibration):
        original_install(calibration)
        proposal.install()

    entry.helper_reference = phase_reference
    entry.worker.install = install_proposal_after_worker
    phase.DENSE_MODULE = COMPLEMENTARITY_MODULE
    phase.main()


if __name__ == '__main__':
    main()
