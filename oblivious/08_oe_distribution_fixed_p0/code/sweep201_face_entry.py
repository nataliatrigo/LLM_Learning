"""Explicit optional adaptive-face execution phase; frozen sources unchanged."""

from __future__ import annotations

from hashlib import sha256
from pathlib import Path

from oblivious.oe_distribution_fixed_p0.code import sweep201_quick_entry as entry
from oblivious.oe_distribution_fixed_p0.code import sweep201_face_search as face

FACE_MODULE = 'oblivious.oe_distribution_fixed_p0.code.sweep201_face_entry'


def main():
    original_reference = entry.helper_reference
    original_command = entry.child_command
    original_install = entry.quick.install

    def phase_reference():
        path = Path(__file__).resolve()
        return dict(original_reference(),**face.helper_reference(),**{
            str(path.relative_to(entry.worker.runner.ROOT)):sha256(path.read_bytes()).hexdigest()})

    def phase_command(command,enabled):
        command = original_command(command,enabled)
        return [FACE_MODULE if value == entry.ENTRY_MODULE else value for value in command]

    def install_face_after_quick():
        original_install()
        face.install()

    entry.helper_reference = phase_reference
    entry.ENTRY_MODULE = FACE_MODULE
    entry.child_command = phase_command
    entry.quick.install = install_face_after_quick
    entry.main()


if __name__ == '__main__':
    main()
