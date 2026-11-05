"""Entrypoint wiring checks, with no solver or worker execution."""

from contextlib import ExitStack
from hashlib import sha256
from pathlib import Path
from unittest import TestCase, main
from unittest.mock import Mock, patch

from oblivious.oe_distribution_fixed_p0.code import sweep201_dense_entry as dense


class DenseEntryChecks(TestCase):
    def test_combined_child_dispatch_and_source_references(self):
        face = dense.face_entry
        entry = face.entry
        calibration = object()
        original_install = Mock()
        installed = Mock()
        observed = {}

        def inspect_without_worker():
            command = ['python', '-m', entry.WORKER_MODULE, '--p0', '.1', '--rho', '.9']
            observed['child'] = entry.child_command(command, True)
            observed['without_quick'] = entry.child_command(command, False)
            observed['references'] = entry.helper_reference()
            entry.worker.install(calibration)

        with ExitStack() as stack:
            # The wrappers intentionally set process-global hooks in production.
            for target, name in ((entry, 'helper_reference'), (entry, 'child_command'),
                                 (entry, 'ENTRY_MODULE'), (entry.quick, 'install'),
                                 (face, 'FACE_MODULE')):
                stack.enter_context(patch.object(target, name, getattr(target, name)))
            stack.enter_context(patch.object(entry.worker, 'install', original_install))
            stack.enter_context(patch.object(dense.backend, 'install', installed))
            stack.enter_context(patch.object(entry, 'main', inspect_without_worker))
            dense.main()

        self.assertEqual(observed['child'][2], dense.DENSE_MODULE)
        self.assertEqual(observed['without_quick'][2], dense.DENSE_MODULE)
        self.assertIn('--no-quick-prescreen', observed['without_quick'])
        original_install.assert_called_once_with(calibration)
        installed.assert_called_once_with(threads=2)
        root = entry.worker.runner.ROOT
        sources = [Path(dense.__file__), Path(dense.backend.memory_gate.__file__), Path(dense.backend.__file__),
                   Path(dense.backend.__file__).with_suffix('.cpp'),
                   Path(face.__file__), Path(face.face.__file__)]
        for path in sources:
            relative = str(path.resolve().relative_to(root))
            self.assertEqual(observed['references'][relative], sha256(path.read_bytes()).hexdigest())


if __name__ == '__main__':
    main()
