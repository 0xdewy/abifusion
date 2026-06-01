"""Tests for the ABI evaluation benchmark.

Tests metric computation, aggregation, and the benchmark script end-to-end.
"""

import json
import os
import sys
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

# Add scripts/eval to path for testing
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "eval"))

from benchmark import (
    compute_contract_metrics,
    aggregate_results,
    parse_ground_truth_functions,
    tokenize_abi_types,
    PerContractMetrics,
    EvalResults,
    _f1,
)


class TestTokenizeAbiTypes:
    def test_uint_normalizes_to_uint(self):
        assert tokenize_abi_types("uint256") == "uint"
        assert tokenize_abi_types("uint8") == "uint"
        assert tokenize_abi_types("uint128") == "uint"

    def test_int_normalizes_to_int(self):
        assert tokenize_abi_types("int256") == "int"
        assert tokenize_abi_types("int8") == "int"

    def test_bytes_normalizes_to_bytesN(self):
        assert tokenize_abi_types("bytes32") == "bytesN"
        assert tokenize_abi_types("bytes1") == "bytesN"

    def test_address_unchanged(self):
        assert tokenize_abi_types("address") == "address"

    def test_string_unchanged(self):
        assert tokenize_abi_types("string") == "string"

    def test_bool_unchanged(self):
        assert tokenize_abi_types("bool") == "bool"

    def test_array_types(self):
        assert tokenize_abi_types("address[]") == "address[]"
        assert tokenize_abi_types("uint256[]") == "uint[]"


class TestParseGroundTruthFunctions:
    def test_extracts_selectors_from_simple_abi(self):
        abi = [
            {"type": "function", "name": "transfer", "inputs": [
                {"name": "to", "type": "address"},
                {"name": "amount", "type": "uint256"},
            ]},
            {"type": "function", "name": "name", "inputs": [], "outputs": [
                {"name": "", "type": "string"},
            ]},
        ]
        funcs = parse_ground_truth_functions(abi)
        assert len(funcs) == 2

        transfer = next(f for f in funcs if f["name"] == "transfer")
        assert transfer["selector"] == "a9059cbb"  # keccak("transfer(address,uint256)")[:8]
        assert transfer["arity"] == 2
        assert transfer["input_types"] == ["address", "uint256"]

        name_func = next(f for f in funcs if f["name"] == "name")
        assert name_func["selector"] == "06fdde03"  # keccak("name()")[:8]
        assert name_func["arity"] == 0
        assert name_func["input_types"] == []

    def test_skips_non_function_entries(self):
        abi = [
            {"type": "event", "name": "Transfer", "inputs": []},
            {"type": "constructor", "inputs": [{"type": "address"}]},
            {"type": "function", "name": "foo", "inputs": []},
        ]
        funcs = parse_ground_truth_functions(abi)
        assert len(funcs) == 1
        assert funcs[0]["name"] == "foo"

    def test_empty_abi(self):
        assert parse_ground_truth_functions([]) == []

    def test_complex_types(self):
        abi = [
            {"type": "function", "name": "process", "inputs": [
                {"type": "address"},
                {"type": "uint256"},
                {"type": "bool"},
                {"type": "bytes32"},
            ]},
        ]
        funcs = parse_ground_truth_functions(abi)
        assert len(funcs) == 1
        assert funcs[0]["input_types"] == ["address", "uint256", "bool", "bytes32"]
        assert funcs[0]["arity"] == 4


class TestF1:
    def test_f1_perfect(self):
        assert _f1(1.0, 1.0) == 1.0

    def test_f1_zero(self):
        assert _f1(0.0, 1.0) == 0.0
        assert _f1(1.0, 0.0) == 0.0
        assert _f1(0.0, 0.0) == 0.0

    def test_f1_balanced(self):
        assert abs(_f1(0.75, 0.75) - 0.75) < 1e-6

    def test_f1_imbalanced(self):
        f1 = _f1(0.5, 1.0)
        expected = 2 * 0.5 * 1.0 / (0.5 + 1.0)
        assert abs(f1 - expected) < 1e-6


class TestComputeContractMetrics:
    def _make_gt(self, selectors_types: list) -> list:
        """Make ground-truth function list from [("selector", ["type",...]), ...]."""
        return [
            {"selector": s, "name": f"f_{s[:4]}", "input_types": types, "arity": len(types)}
            for s, types in selectors_types
        ]

    def _make_rec(self, selectors_types: list) -> list:
        """Make reconstructed function list."""
        return [
            {"selector": s, "name": f"f_{s[:4]}", "input_types": types, "arity": len(types)}
            for s, types in selectors_types
        ]

    def test_perfect_match(self):
        gt = self._make_gt([
            ("a9059cbb", ["address", "uint256"]),
            ("70a08231", ["address"]),
        ])
        rec = self._make_rec([
            ("a9059cbb", ["address", "uint256"]),
            ("70a08231", ["address"]),
        ])

        metrics = compute_contract_metrics(gt, rec)

        assert metrics.selector_f1 == 1.0
        assert metrics.prototype_f1 == 1.0
        assert metrics.type_accuracy == 1.0
        assert metrics.arity_accuracy == 1.0
        assert metrics.full_signature_accuracy == 1.0

    def test_selectors_match_types_wrong(self):
        gt = self._make_gt([
            ("a9059cbb", ["address", "uint256"]),
        ])
        rec = self._make_rec([
            ("a9059cbb", ["uint256", "address"]),  # types swapped
        ])

        metrics = compute_contract_metrics(gt, rec)

        assert metrics.selector_f1 == 1.0  # selector found
        assert metrics.prototype_f1 < 1.0  # types wrong
        assert metrics.arity_accuracy == 1.0  # arity correct
        assert metrics.type_accuracy == 0.0  # both types wrong
        assert metrics.full_signature_accuracy == 0.0

    def test_missing_selector(self):
        gt = self._make_gt([
            ("a9059cbb", ["address", "uint256"]),
            ("70a08231", ["address"]),
        ])
        rec = self._make_rec([
            ("a9059cbb", ["address", "uint256"]),
            # missing 70a08231
        ])

        metrics = compute_contract_metrics(gt, rec)

        assert metrics.selector_recall == 0.5  # 1 of 2 found
        assert metrics.selector_precision == 1.0  # all found are correct
        assert metrics.full_signature_accuracy == 0.5
        assert metrics.prototype_f1 < 1.0  # function not found → prototype counts as wrong

    def test_extra_selector(self):
        gt = self._make_gt([
            ("a9059cbb", ["address", "uint256"]),
        ])
        rec = self._make_rec([
            ("a9059cbb", ["address", "uint256"]),
            ("deadbeef", ["uint256"]),  # extra selector not in ground truth
        ])

        metrics = compute_contract_metrics(gt, rec)

        assert metrics.selector_recall == 1.0
        assert metrics.selector_precision == 0.5  # 1 of 2 correct
        # extra selector with no ground truth doesn't count toward type metrics

    def test_no_overlap(self):
        gt = self._make_gt([("a9059cbb", ["address", "uint256"])])
        rec = self._make_rec([("deadbeef", ["uint256"])])

        metrics = compute_contract_metrics(gt, rec)

        assert metrics.selector_f1 == 0.0
        assert metrics.prototype_f1 == 0.0
        assert metrics.full_signature_accuracy == 0.0

    def test_empty_ground_truth(self):
        metrics = compute_contract_metrics([], [])
        assert metrics.selector_f1 == 0.0
        assert metrics.prototype_f1 == 0.0
        assert metrics.full_signature_accuracy == 0.0

    def test_type_normalization_matches(self):
        """uint256 in ground truth should match uint in reconstructed (tokenized)."""
        gt = self._make_gt([
            ("a9059cbb", ["address", "uint256"]),
        ])
        rec = [
            {"selector": "a9059cbb", "name": "transfer", "input_types": ["address", "uint"], "arity": 2},
        ]

        metrics = compute_contract_metrics(gt, rec)
        assert metrics.type_accuracy == 1.0  # uint256 → uint normalization works
        assert metrics.full_signature_accuracy == 1.0


class TestAggregateResults:
    def test_aggregate_single_contract(self):
        metrics = PerContractMetrics(
            address="0x1", contract_name="C", compiler_version="0.8.0",
            gt_function_count=5, rec_function_count=5,
            selector_recall=1.0, selector_precision=1.0, selector_f1=1.0,
            type_accuracy=0.8, arity_accuracy=1.0, prototype_f1=0.8,
            full_signature_accuracy=0.8,
        )
        results = aggregate_results([metrics])
        assert results.total_contracts == 1
        assert results.total_functions == 5
        assert results.selector_f1 == 1.0
        assert results.prototype_f1 == 0.8

    def test_aggregate_multiple_contracts_weighted(self):
        m1 = PerContractMetrics(
            address="0x1", contract_name="C1", compiler_version="0.8.0",
            gt_function_count=10, rec_function_count=10,
            selector_recall=1.0, selector_precision=1.0, selector_f1=1.0,
            type_accuracy=1.0, arity_accuracy=1.0, prototype_f1=1.0,
            full_signature_accuracy=1.0,
        )
        m2 = PerContractMetrics(
            address="0x2", contract_name="C2", compiler_version="0.7.0",
            gt_function_count=2, rec_function_count=0,
            selector_recall=0.0, selector_precision=0.0, selector_f1=0.0,
            type_accuracy=0.0, arity_accuracy=0.0, prototype_f1=0.0,
            full_signature_accuracy=0.0,
        )

        results = aggregate_results([m1, m2])

        # m1 has 10 functions (perfect), m2 has 2 (completely wrong)
        # Micro-averaged prototype F1 should be weighted by function count
        expected_proto = (1.0 * 10 + 0.0 * 2) / 12
        assert abs(results.prototype_f1 - expected_proto) < 0.01

    def test_compiler_breakdown(self):
        m1 = PerContractMetrics(
            address="0x1", contract_name="C1", compiler_version="0.8.19",
            gt_function_count=10, rec_function_count=10,
            selector_recall=1.0, selector_precision=1.0, selector_f1=1.0,
            type_accuracy=1.0, arity_accuracy=1.0, prototype_f1=1.0,
            full_signature_accuracy=1.0,
        )
        m2 = PerContractMetrics(
            address="0x2", contract_name="C2", compiler_version="0.7.6",
            gt_function_count=5, rec_function_count=2,
            selector_recall=0.4, selector_precision=1.0, selector_f1=0.57,
            type_accuracy=0.0, arity_accuracy=0.0, prototype_f1=0.0,
            full_signature_accuracy=0.0,
        )

        results = aggregate_results([m1, m2])
        assert "0.8" in results.by_compiler or "0.7" in results.by_compiler

    def test_empty_aggregate(self):
        results = aggregate_results([])
        assert results.total_contracts == 0
        assert results.total_functions == 0
        assert results.selector_f1 == 0.0


class TestBenchmarkEndToEnd:
    """Test the benchmark script runs end-to-end with test contracts."""

    @pytest.fixture
    def test_contracts_path(self):
        return PROJECT_ROOT / "scripts" / "eval" / "test_contracts.json"

    def test_benchmark_runs_with_local_source(self, test_contracts_path, tmp_path):
        """benchmark.py --source local runs and produces output files."""
        import subprocess

        output_dir = tmp_path / "eval_output"
        db_path = PROJECT_ROOT / "cache" / "4byte.db"

        result = subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "eval" / "benchmark.py"),
                "--source", "local",
                "--input", str(test_contracts_path),
                "--sample", "3",
                "--db", str(db_path),
                "--output-dir", str(output_dir),
                "--stratify",
            ],
            capture_output=True, text=True, timeout=60,
        )

        assert result.returncode == 0, f"Benchmark failed:\n{result.stderr}"

        # Verify output files
        results_json = output_dir / "eval_results.json"
        summary_md = output_dir / "eval_summary.md"

        assert results_json.exists(), f"Missing {results_json}"
        assert summary_md.exists(), f"Missing {summary_md}"

        # Validate JSON structure
        with open(results_json) as f:
            data = json.load(f)
        assert "selector_f1" in data
        assert "prototype_f1" in data
        assert "per_contract" in data
        assert "by_compiler" in data
        assert data["total_contracts"] > 0

    def test_summary_md_contains_all_sections(self, test_contracts_path, tmp_path):
        """eval_summary.md contains required metric rows."""
        import subprocess

        output_dir = tmp_path / "eval_output"
        db_path = PROJECT_ROOT / "cache" / "4byte.db"

        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "eval" / "benchmark.py"),
                "--source", "local",
                "--input", str(test_contracts_path),
                "--sample", "3",
                "--db", str(db_path),
                "--output-dir", str(output_dir),
            ],
            capture_output=True, timeout=60,
        )

        md = (output_dir / "eval_summary.md").read_text()

        # Required sections
        assert "Aggregate Metrics" in md
        assert "Selector F1" in md
        assert "Prototype F1" in md
        assert "Type Accuracy" in md
        assert "Arity Accuracy" in md
        assert "Full-Signature Exact Match" in md
        # SOTA references for context
        assert "Heimdall-rs" in md
        assert "SCDBench" in md
