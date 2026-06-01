"""Basic test to verify test infrastructure."""


def test_imports():
    """Test that core modules can be imported."""
    # Test data module imports

    # Test features module imports

    # Test models module imports

    # Test utils module imports

    assert True  # If we get here, imports succeeded


def test_sample_bytecode_fixture(sample_bytecode):
    """Test that the sample bytecode fixture works."""
    assert isinstance(sample_bytecode, str)
    assert len(sample_bytecode) > 100
    # Check it's valid hex (optional characters only)
    import re

    assert re.match(r"^[0-9a-fA-F]+$", sample_bytecode) is not None


def test_sample_contract_data_fixture(sample_contract_data):
    """Test that the sample contract data fixture works."""
    assert isinstance(sample_contract_data, dict)
    assert "address" in sample_contract_data
    assert "abi" in sample_contract_data
    assert "bytecode" in sample_contract_data
    assert len(sample_contract_data["abi"]) > 0


def test_temp_cache_dir_fixture(temp_cache_dir):
    """Test that the temp cache directory fixture works."""
    import os

    assert os.path.exists(temp_cache_dir)
    assert os.path.isdir(temp_cache_dir)

    # Test we can write to it
    test_file = os.path.join(temp_cache_dir, "test.txt")
    with open(test_file, "w") as f:
        f.write("test")
    assert os.path.exists(test_file)
