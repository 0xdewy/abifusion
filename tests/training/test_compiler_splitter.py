"""Tests for compiler-version-aware data splitting."""

import pytest
from abi_reconstructor.training.compiler_splitter import (
    CompilerSplit,
    annotate_samples_with_compiler_version,
    compare_splits,
    parse_compiler_version,
    random_baseline_split,
    split_by_compiler_version,
    version_in_range,
)


class TestParseCompilerVersion:
    def test_standard_format(self):
        assert parse_compiler_version("v0.8.19") == (0, 8, 19)
        assert parse_compiler_version("0.8.19") == (0, 8, 19)

    def test_with_commit_hash(self):
        assert parse_compiler_version("v0.8.19+commit.7dd6d404") == (0, 8, 19)
        assert parse_compiler_version("0.7.6+commit.7338295f") == (0, 7, 6)

    def test_early_version(self):
        assert parse_compiler_version("v0.4.26") == (0, 4, 26)
        assert parse_compiler_version("0.4.11") == (0, 4, 11)

    def test_invalid(self):
        assert parse_compiler_version("") is None
        assert parse_compiler_version("not a version") is None
        assert parse_compiler_version("vyper-0.3.0") == (0, 3, 0)

    def test_only_major_minor(self):
        assert parse_compiler_version("v0.8") is None  # needs patch version


class TestVersionInRange:
    def test_within_range(self):
        assert version_in_range((0, 7, 6), (0, 4, 0), (0, 7, 99)) is True
        assert version_in_range((0, 4, 0), None, (0, 7, 99)) is True

    def test_below_range(self):
        assert version_in_range((0, 3, 5), (0, 4, 0), (0, 7, 99)) is False

    def test_above_range(self):
        assert version_in_range((0, 8, 20), (0, 4, 0), (0, 7, 99)) is False

    def test_no_bounds(self):
        assert version_in_range((0, 5, 0)) is True


class TestAnnotateSamples:
    def test_annotates_with_metadata(self):
        samples = [
            {"contract_address": "0xaaa", "selector": "a9059cbb"},
            {"contract_address": "0xbbb", "selector": "70a08231"},
            {"address": "0xccc", "selector": "06fdde03"},
        ]
        metadata = {
            "0xaaa": {"compiler_version": "v0.8.19"},
            "0xbbb": {"compiler_version": "v0.4.26"},
            "0xccc": {"compiler_version": "v0.7.6"},
        }
        result = annotate_samples_with_compiler_version(samples, metadata)
        assert result[0]["compiler_version"] == "v0.8.19"
        assert result[1]["compiler_version"] == "v0.4.26"
        assert result[2]["compiler_version"] == "v0.7.6"

    def test_missing_metadata_skips(self):
        samples = [{"contract_address": "0xddd", "selector": "18160ddd"}]
        metadata = {}
        result = annotate_samples_with_compiler_version(samples, metadata)
        assert "compiler_version" not in result[0]


class TestSplitByCompilerVersion:
    def _make_samples(self, versions: list) -> list:
        samples = []
        for i, v in enumerate(versions):
            samples.append({
                "contract_address": f"0x{i:040x}",
                "selector": f"{i:08x}",
                "compiler_version": v,
                "function_name": f"func_{i}",
            })
        return samples

    def test_split_separates_versions(self):
        samples = self._make_samples([
            "v0.4.26", "v0.5.16", "v0.6.12",  # train
            "v0.7.6",                            # train
            "v0.8.19", "v0.8.20", "v0.8.7",    # test
        ])
        split = split_by_compiler_version(
            samples,
            train_version_min="0.4.0", train_version_max="0.7.99",
            test_version_min="0.8.0", test_version_max="0.8.99",
            val_split=0.0, shuffle=False, seed=42,
        )

        train_versions = {s["compiler_version"] for s in split.train}
        test_versions = {s["compiler_version"] for s in split.test}

        assert train_versions == {"v0.4.26", "v0.5.16", "v0.6.12", "v0.7.6"}
        assert test_versions == {"v0.8.19", "v0.8.20", "v0.8.7"}

    def test_no_overlap(self):
        samples = self._make_samples(
            ["v0.4.26"] * 20 + ["v0.8.19"] * 20
        )
        split = split_by_compiler_version(
            samples,
            train_version_min="0.4.0", train_version_max="0.7.99",
            test_version_min="0.8.0", test_version_max="0.9.99",
            val_split=0.1, shuffle=False,
        )

        train_addrs = {s["contract_address"] for s in split.train}
        test_addrs = {s["contract_address"] for s in split.test}
        val_addrs = {s["contract_address"] for s in split.val}

        assert not (train_addrs & test_addrs)
        assert not (val_addrs & test_addrs)

    def test_validation_split(self):
        samples = self._make_samples(["v0.4.26"] * 100)
        split = split_by_compiler_version(
            samples,
            train_version_min="0.4.0", train_version_max="0.7.99",
            test_version_min="0.8.0", test_version_max="0.9.99",
            val_split=0.1,
        )

        assert len(split.train) + len(split.val) == 100
        assert len(split.val) >= 1

    def test_no_matching_test_samples(self):
        samples = self._make_samples(["v0.4.26"] * 10)
        split = split_by_compiler_version(
            samples,
            train_version_min="0.4.0", train_version_max="0.7.99",
            test_version_min="0.8.0", test_version_max="0.9.99",
        )

        assert len(split.test) == 0
        assert len(split.train) > 0

    def test_samples_without_version_skipped(self):
        samples = [
            {"contract_address": "0xabc", "selector": "deadbeef", "function_name": "f"},
            self._make_samples(["v0.4.26"])[0],
        ]
        split = split_by_compiler_version(
            samples,
            train_version_min="0.4.0", train_version_max="0.7.99",
            test_version_min="0.8.0", test_version_max="0.9.99",
        )
        # The sample without compiler_version should be skipped
        assert len(split.train) + len(split.val) == 1


class TestRandomBaselineSplit:
    def test_splits_correct_sizes(self):
        samples = [{"id": i} for i in range(100)]
        split = random_baseline_split(samples, test_split=0.2, val_split=0.1)

        assert len(split.test) == 20
        assert len(split.val) == 8
        assert len(split.train) == 72
        assert len(split.train) + len(split.val) + len(split.test) == 100

    def test_no_overlap(self):
        samples = [{"id": i} for i in range(50)]
        split = random_baseline_split(samples, test_split=0.2, val_split=0.1)

        train_ids = {s["id"] for s in split.train}
        val_ids = {s["id"] for s in split.val}
        test_ids = {s["id"] for s in split.test}

        assert not (train_ids & test_ids)
        assert not (val_ids & test_ids)
        assert not (train_ids & val_ids)


class TestCompareSplits:
    def test_returns_structured_comparison(self):
        samples_c = [
            {"compiler_version": "v0.4.26", "id": 1},
            {"compiler_version": "v0.8.19", "id": 2},
        ]
        samples_b = [
            {"compiler_version": "v0.5.0", "id": 3},
        ]

        cs = CompilerSplit(
            name="test", train=samples_c[:1], val=[], test=samples_c[1:],
            train_versions=("v0.4", "v0.7"), test_versions=("v0.8",),
        )
        bs = CompilerSplit(
            name="random", train=samples_b, val=[], test=[],
        )

        comparison = compare_splits(cs, bs)
        assert "compiler_holdout" in comparison
        assert "random_baseline" in comparison
        assert comparison["compiler_holdout"]["train"]["n_samples"] == 1
