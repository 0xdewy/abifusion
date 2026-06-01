"""Compiler-version-aware data splitting for ABI reconstruction training.

Implements the hold-out experiment from the research: train on contracts
compiled with one compiler version range and evaluate on contracts compiled
with a different version. This tests whether the ML models learn genuine EVM
semantic invariants or merely fingerprint known Solidity compiler output.

Reference:
  - Chen et al. "SigRec." IEEE TSE, 2021. (SigRec's 31 rules are compiler-coded)
  - The novel insight from research/evm-bytecode-to-abi/RESEARCH.md:
    "Every ABI reconstruction method is ultimately a compiler model by another name."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class CompilerSplit:
    """A split of the dataset defined by compiler version constraints."""
    name: str
    train: List[Dict[str, Any]] = field(default_factory=list)
    val: List[Dict[str, Any]] = field(default_factory=list)
    test: List[Dict[str, Any]] = field(default_factory=list)
    # Version constraints used to create this split
    train_versions: Tuple[str, ...] = ()
    test_versions: Tuple[str, ...] = ()


def parse_compiler_version(version_str: str) -> Optional[Tuple[int, int, int]]:
    """Parse a Solidity compiler version string into (major, minor, patch).

    Handles formats: 'v0.8.19', '0.8.19+commit.abc123', 'v0.4.26'

    Returns None if parsing fails.
    """
    if not version_str:
        return None
    m = re.search(r'v?(\d+)\.(\d+)\.(\d+)', version_str)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return None


def version_in_range(
    version: Tuple[int, int, int],
    version_min: Optional[Tuple[int, int, int]] = None,
    version_max: Optional[Tuple[int, int, int]] = None,
) -> bool:
    """Check if a parsed compiler version falls within [min, max] inclusive."""
    if version_min is not None and version < version_min:
        return False
    if version_max is not None and version > version_max:
        return False
    return True


def _has_compiler_version(sample: Dict[str, Any]) -> bool:
    """Check if a sample carries compiler version metadata."""
    return bool(sample.get("compiler_version", ""))


def _get_parsed_version(sample: Dict[str, Any]) -> Optional[Tuple[int, int, int]]:
    return parse_compiler_version(sample.get("compiler_version", ""))


def annotate_samples_with_compiler_version(
    samples: List[Dict[str, Any]],
    contract_metadata: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Annotate training samples with compiler version from contract metadata.

    Args:
        samples: List of training samples (each has 'contract_address' or 'address').
        contract_metadata: Dict mapping contract address → metadata dict
                           (must contain 'compiler_version').

    Returns:
        The same samples list, mutated in-place with 'compiler_version' added.
    """
    for sample in samples:
        addr = sample.get("contract_address") or sample.get("address", "")
        if addr and addr in contract_metadata:
            sample["compiler_version"] = contract_metadata[addr].get("compiler_version", "")
    return samples


def split_by_compiler_version(
    samples: List[Dict[str, Any]],
    train_version_min: Optional[str] = None,
    train_version_max: Optional[str] = None,
    test_version_min: Optional[str] = None,
    test_version_max: Optional[str] = None,
    val_split: float = 0.1,
    shuffle: bool = True,
    seed: int = 42,
) -> CompilerSplit:
    """Split dataset by compiler version for hold-out evaluation.

    All samples with compiler version in [train_version_min, train_version_max]
    go into train/val. All samples with version in [test_version_min, test_version_max]
    go into test.

    Default: train on v0.4 through v0.7, test on v0.8.

    Args:
        samples: List of annotated training samples.
        train_version_min/max: Version range for training data (inclusive).
        test_version_min/max: Version range for test data (inclusive).
        val_split: Fraction of training data to use for validation.
        shuffle: Whether to shuffle before splitting.
        seed: Random seed for reproducibility.

    Returns:
        CompilerSplit with train, val, test lists.
    """
    train_min = parse_compiler_version(train_version_min or "0.4.0")
    train_max = parse_compiler_version(train_version_max or "0.7.99")
    test_min = parse_compiler_version(test_version_min or "0.8.0")
    test_max = parse_compiler_version(test_version_max or "0.9.99")

    train_pool = []
    test_pool = []

    for sample in samples:
        ver = _get_parsed_version(sample)
        if ver is None:
            continue

        if version_in_range(ver, train_min, train_max):
            train_pool.append(sample)
        elif version_in_range(ver, test_min, test_max):
            test_pool.append(sample)

    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(train_pool)
        rng.shuffle(test_pool)

    if val_split > 0:
        n_val = max(1, int(len(train_pool) * val_split))
        val_samples = train_pool[:n_val]
        train_samples = train_pool[n_val:]
    else:
        val_samples = []
        train_samples = list(train_pool)

    return CompilerSplit(
        name=f"holdout_{train_version_min or 'v0.4'}-{train_version_max or 'v0.7'}_vs_{test_version_min or 'v0.8'}",
        train=train_samples,
        val=val_samples,
        test=test_pool,
        train_versions=(train_version_min or "v0.4", train_version_max or "v0.7"),
        test_versions=(test_version_min or "v0.8", test_version_max or "v0.9"),
    )


def random_baseline_split(
    samples: List[Dict[str, Any]],
    test_split: float = 0.2,
    val_split: float = 0.1,
    seed: int = 42,
) -> CompilerSplit:
    """Create a standard random train/val/test split as baseline.

    Args:
        samples: List of training samples.
        test_split: Fraction for test.
        val_split: Fraction of training for validation.
        seed: Random seed.

    Returns:
        CompilerSplit with randomly assigned train/val/test.
    """
    rng = np.random.RandomState(seed)
    indices = list(range(len(samples)))
    rng.shuffle(indices)

    n_test = max(1, int(len(samples) * test_split))
    n_val = max(1, int((len(samples) - n_test) * val_split))
    n_train = len(samples) - n_test - n_val

    train_pool = [samples[i] for i in indices[:n_train]]
    val_pool = [samples[i] for i in indices[n_train:n_train + n_val]]
    test_pool = [samples[i] for i in indices[n_train + n_val:]]

    return CompilerSplit(
        name="random_baseline",
        train=train_pool,
        val=val_pool,
        test=test_pool,
    )


def compare_splits(
    compiler_split: CompilerSplit,
    baseline_split: CompilerSplit,
) -> Dict[str, Any]:
    """Compare compiler-holdout split against random baseline.

    Returns statistics about both splits for analysis.
    """
    def _stats(samples: List[Dict]) -> Dict[str, Any]:
        versions = {}
        for s in samples:
            v = s.get("compiler_version", "unknown")
            versions[v] = versions.get(v, 0) + 1
        return {
            "n_samples": len(samples),
            "compiler_versions": versions,
            "n_unique_versions": len(versions),
        }

    return {
        "compiler_holdout": {
            "train": _stats(compiler_split.train),
            "val": _stats(compiler_split.val),
            "test": _stats(compiler_split.test),
            "train_versions": compiler_split.train_versions,
            "test_versions": compiler_split.test_versions,
        },
        "random_baseline": {
            "train": _stats(baseline_split.train),
            "val": _stats(baseline_split.val),
            "test": _stats(baseline_split.test),
        },
    }
