"""Expose stable Python module names for the numbered experiment folders.

For example: python -m oblivious.oe_distribution.code.run_experiment --help
The numbers control the on-disk order without appearing in Python imports.
"""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys

from .layout import EXPERIMENT_FOLDERS


def _register_experiments():
    parent = sys.modules[__name__]
    for name, folder in EXPERIMENT_FOLDERS.items():
        qualified_name = f"{__name__}.{name}"
        spec = spec_from_file_location(
            qualified_name, Path(__file__).parent / folder / "__init__.py",
        )
        module = module_from_spec(spec)
        sys.modules[qualified_name] = module
        setattr(parent, name, module)
        spec.loader.exec_module(module)


_register_experiments()
