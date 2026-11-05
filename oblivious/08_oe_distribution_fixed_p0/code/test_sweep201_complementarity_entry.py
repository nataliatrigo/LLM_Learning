"""Verify complete entry wiring without running a worker or solver."""

from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import Mock, patch

from oblivious.oe_distribution_fixed_p0.code import sweep201_complementarity_entry as combined


class ComplementarityEntryChecks(TestCase):
    def test_dispatch_installation_and_all_execution_sources(self):
        phase = combined.phase
        face = phase.face_entry
        entry = face.entry
        calibration = object()
        original_install, dense_install, proposal_install = Mock(), Mock(), Mock()
        observed = {}

        def inspect_without_worker():
            command = ['python', '-m', entry.WORKER_MODULE, '--p0', '.3', '--rho', '.9']
            observed['child'] = entry.child_command(command, True)
            observed['references'] = entry.helper_reference()
            entry.worker.install(calibration)

        with ExitStack() as stack:
            for target, name in ((entry, 'helper_reference'), (entry, 'child_command'),
                                 (entry, 'ENTRY_MODULE'), (entry.quick, 'install'),
                                 (face, 'FACE_MODULE'), (phase, 'DENSE_MODULE')):
                stack.enter_context(patch.object(target, name, getattr(target, name)))
            stack.enter_context(patch.object(entry.worker, 'install', original_install))
            stack.enter_context(patch.object(phase.backend, 'install', dense_install))
            stack.enter_context(patch.object(combined.proposal, 'install', proposal_install))
            stack.enter_context(patch.object(entry, 'main', inspect_without_worker))
            combined.main()

        self.assertEqual(observed['child'][2], combined.COMPLEMENTARITY_MODULE)
        original_install.assert_called_once_with(calibration)
        dense_install.assert_called_once_with(threads=2)
        proposal_install.assert_called_once_with()
        root = entry.worker.runner.ROOT
        sources = [Path(combined.__file__), Path(combined.proposal.__file__),
                   Path(phase.__file__), Path(phase.backend.__file__),
                   Path(phase.backend.__file__).with_suffix('.cpp'),
                   Path(phase.backend.memory_gate.__file__), Path(face.__file__),
                   Path(face.face.__file__)]
        for path in sources:
            relative = str(path.resolve().relative_to(root))
            self.assertEqual(observed['references'][relative], sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
