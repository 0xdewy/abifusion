"""Basic integration tests to verify the repository works."""

import importlib
import os

import abifusion


def test_public_api_imports():
    """The package's advertised public API is importable from the root."""
    from abifusion import (
        OfflineABI,
        FourByteDatabase,
        ABIFusion,
        SelectorExtractor,
    )

    assert all([OfflineABI, FourByteDatabase, ABIFusion, SelectorExtractor])


def test_submodules_importable():
    """Core submodules import without side effects."""
    assert importlib.import_module("abifusion.data")
    assert importlib.import_module("abifusion.features")
    assert importlib.import_module("abifusion.utils")
    assert importlib.import_module("abifusion.fusion")


def test_package_structure():
    """The package directory matches the declared structure."""
    package_dir = os.path.dirname(abifusion.__file__)

    for sub in ("", "data", "features", "utils"):
        assert os.path.exists(os.path.join(package_dir, sub, "__init__.py"))
