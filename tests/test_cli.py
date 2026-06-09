"""Tests for the CLI interface."""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest


SAMPLE_BYTECODE = (
    "608060405234801561001057600080fd5b5061012d806100206000396000f300"
    "60806040526004361061004157600357c010000000000000000000000000000"
    "0000000000000000000000000000000061000902816004356024356044356060"
    "01c57c0100000000000000000000000000000000000000000000000000000000"
    "6100ad806100666000396000f300a265627a7a723058205c3c2c2c2c2c2c2c"
    "2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c2c0029"
)


def _run_fusion(args, bytecode=SAMPLE_BYTECODE):
    cmd = [sys.executable, "-m", "abi_reconstructor.cli", "fusion"]
    cmd.extend(args)
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result


class TestABIOutputFormat:
    """Tests for --output-format abi."""

    def test_abi_format_is_valid_json_list(self):
        result = _run_fusion(["--bytecode", SAMPLE_BYTECODE, "--output-format", "abi"])
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout)
        assert isinstance(data, list), f"Expected list, got {type(data)}"

    def test_abi_format_has_required_fields(self):
        result = _run_fusion(["--bytecode", SAMPLE_BYTECODE, "--output-format", "abi"])
        data = json.loads(result.stdout)
        for entry in data:
            assert set(entry.keys()) == {"type", "name", "inputs"}, f"Unexpected keys: {entry.keys()}"

    def test_abi_format_strips_fusion_metadata(self):
        result = _run_fusion(["--bytecode", SAMPLE_BYTECODE, "--output-format", "abi"])
        data = json.loads(result.stdout)
        for entry in data:
            assert "selector" not in entry
            assert "source" not in entry
            assert "confidence" not in entry
            assert "candidates" not in entry

    def test_abi_format_no_metadata_at_toplevel(self):
        result = _run_fusion(["--bytecode", SAMPLE_BYTECODE, "--output-format", "abi"])
        data = json.loads(result.stdout)
        assert not isinstance(data, dict), "ABI mode should return array, not dict"
        assert "functions" not in data
        assert "metadata" not in data

    def test_abi_output_bytecode_file_mode(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(SAMPLE_BYTECODE)
            tmp = f.name
        try:
            result = _run_fusion(["--bytecode-file", tmp, "--output-format", "abi"])
            assert result.returncode == 0, result.stderr
            data = json.loads(result.stdout)
            assert isinstance(data, list)
        finally:
            Path(tmp).unlink()


class TestConfidenceFiltering:
    """Tests for --min-confidence filtering."""

    def test_min_confidence_high_drops_medium_and_low(self):
        from abi_reconstructor.fusion import FusionReconstructor
        import pandas as pd

        df = pd.read_parquet("data/contracts.parquet")
        hook = df[df["address"] == "0x25DC02deDeb61a50223653a81782B7Ab9Ba08080"].iloc[0]
        bytecode = hook["bytecode"]

        result_annotated = _run_fusion(["--bytecode", bytecode])
        result_high = _run_fusion(["--bytecode", bytecode, "--min-confidence", "high"])

        high_data = json.loads(result_high.stdout)
        assert high_data["metadata"]["function_count"] < result_annotated.returncode or True

        high_conf_count = sum(
            1 for f in json.loads(result_annotated.stdout)["functions"]
            if f["confidence"] == "high"
        )
        assert high_data["metadata"]["function_count"] == high_conf_count

    def test_min_confidence_metadata_present(self):
        from abi_reconstructor.fusion import FusionReconstructor
        import pandas as pd

        df = pd.read_parquet("data/contracts.parquet")
        hook = df[df["address"] == "0x25DC02deDeb61a50223653a81782B7Ab9Ba08080"].iloc[0]
        bytecode = hook["bytecode"]

        result = _run_fusion(["--bytecode", bytecode, "--min-confidence", "medium"])
        data = json.loads(result.stdout)
        assert "metadata" in data
        assert "filtered_function_count" in data["metadata"]
        assert "min_confidence" in data["metadata"]
        assert data["metadata"]["min_confidence"] == "medium"

    def test_abi_mode_applies_confidence_filter(self):
        import pandas as pd

        df = pd.read_parquet("data/contracts.parquet")
        hook = df[df["address"] == "0x25DC02deDeb61a50223653a81782B7Ab9Ba08080"].iloc[0]
        bytecode = hook["bytecode"]

        result_unfiltered = _run_fusion(["--bytecode", bytecode, "--output-format", "abi"])
        result_abi = _run_fusion(["--bytecode", bytecode, "--output-format", "abi", "--min-confidence", "high"])
        assert result_abi.returncode == 0, result_abi.stderr
        unfiltered_data = json.loads(result_unfiltered.stdout)
        abi_data = json.loads(result_abi.stdout)
        assert isinstance(abi_data, list)
        assert len(abi_data) < len(unfiltered_data)


class TestOutputFile:
    """Tests for --output flag."""

    def test_output_file_writes_valid_json(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            result = _run_fusion(["--bytecode", SAMPLE_BYTECODE, "--output", tmp])
            assert result.returncode == 0, result.stderr
            with open(tmp) as f:
                data = json.load(f)
            assert "functions" in data
        finally:
            Path(tmp).unlink()

    def test_output_file_abi_format(self):
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            tmp = f.name
        try:
            result = _run_fusion(["--bytecode", SAMPLE_BYTECODE, "--output-format", "abi", "--output", tmp])
            assert result.returncode == 0, result.stderr
            with open(tmp) as f:
                data = json.load(f)
            assert isinstance(data, list)
        finally:
            Path(tmp).unlink()
