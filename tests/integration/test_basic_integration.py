"""Basic integration tests to verify the repository works."""

import importlib
import os

import abi_reconstructor


def test_public_api_imports():
    """The package's advertised public API is importable from the root."""
    from abi_reconstructor import (
        ABIReconstructor,
        FourByteDatabase,
        FusionReconstructor,
        SelectorExtractor,
    )

    assert all([ABIReconstructor, FourByteDatabase, FusionReconstructor, SelectorExtractor])


def test_submodules_importable():
    """Core submodules import without side effects."""
    assert importlib.import_module("abi_reconstructor.data")
    assert importlib.import_module("abi_reconstructor.features")
    assert importlib.import_module("abi_reconstructor.utils")
    assert importlib.import_module("abi_reconstructor.fusion")


def test_package_structure():
    """The package directory matches the declared structure."""
    package_dir = os.path.dirname(abi_reconstructor.__file__)

    for sub in ("", "data", "features", "utils"):
        assert os.path.exists(os.path.join(package_dir, sub, "__init__.py"))
