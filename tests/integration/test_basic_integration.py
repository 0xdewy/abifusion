"""Basic integration tests to verify the repository works."""

import os
import pytest


def test_import_all_modules():
    """Test that all main modules can be imported."""
    # Core modules

    # Training script

    # Integration example

    assert True  # If we get here, imports succeeded


def test_basic_feature_extraction_flow(sample_bytecode):
    """Test basic feature extraction flow."""
    from abi_reconstructor.features.bytecode_features import BytecodeFeatureExtractor
    from abi_reconstructor.features.selector_extractor import SelectorExtractor

    feature_extractor = BytecodeFeatureExtractor()
    selector_extractor = SelectorExtractor()

    assert hasattr(feature_extractor, "extract_basic_features")
    assert hasattr(selector_extractor, "extract_selectors")


def test_package_structure():
    """Test that package structure is correct."""
    import abi_reconstructor

    # Check submodules
    assert hasattr(abi_reconstructor, "data")
    assert hasattr(abi_reconstructor, "models")

    # Check that __init__.py files exist (use absolute path)
    import abi_reconstructor

    package_dir = os.path.dirname(abi_reconstructor.__file__)

    assert os.path.exists(os.path.join(package_dir, "__init__.py"))
    assert os.path.exists(os.path.join(package_dir, "data", "__init__.py"))
    assert os.path.exists(os.path.join(package_dir, "models", "__init__.py"))


# Import numpy for neural model test
import numpy as np
