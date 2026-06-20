"""Basic integration tests to verify the repository works."""

import importlib
import os

import abifusion


def test_public_api_imports():
    """The package's advertised public API is importable from the root."""
    from abifusion import (
        ABIFusion,
        choose_candidate,
        fuse_abi,
        parse_signature,
        split_args,
    )

    assert all([ABIFusion, choose_candidate, fuse_abi, parse_signature, split_args])


def test_submodules_importable():
    """Core submodules import without side effects."""
    assert importlib.import_module("abifusion.adapters")
    assert importlib.import_module("abifusion.core")
    assert importlib.import_module("abifusion.fusion")
    assert importlib.import_module("abifusion.ports")
    assert importlib.import_module("abifusion.utils")


def test_package_structure():
    """The package directory matches the declared structure."""
    package_dir = os.path.dirname(abifusion.__file__)

    for sub in ("", "core", "utils"):
        assert os.path.exists(os.path.join(package_dir, sub, "__init__.py"))
