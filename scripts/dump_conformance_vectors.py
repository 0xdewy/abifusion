#!/usr/bin/env python3
"""Generate tests/conformance/vectors.json from vectors.input.json.

Reads the hand-authored input bundles (one per fusion tier), runs each through
the pure :func:`abifusion.core.fuse.fuse_abi`, and freezes the result as the
``expected`` field of the committed conformance vectors. No network or evmole is
involved — the inputs are already-resolved adapter outputs, so this is fully
deterministic. Run it after intentionally changing core behavior.

A separate path (not wired here) can extend the vectors with real-contract
bundles: run the Python adapters (EvmoleBytecodeAnalyzer + SignatureLookup) on a
contract, serialize their output as an input bundle, and append it.
"""

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "tests" / "conformance"))

from _loader import to_core_kwargs  # noqa: E402

from abifusion.core.fuse import fuse_abi  # noqa: E402

INPUT_PATH = REPO_ROOT / "tests" / "conformance" / "vectors.input.json"
OUTPUT_PATH = REPO_ROOT / "tests" / "conformance" / "vectors.json"


def main() -> int:
    cases = json.loads(INPUT_PATH.read_text())
    out = []
    for case in cases:
        expected = fuse_abi(**to_core_kwargs(case["input"]))
        out.append({
            "name": case["name"],
            "comment": case.get("comment", ""),
            "input": case["input"],
            "expected": expected,
        })
    OUTPUT_PATH.write_text(json.dumps(out, indent=2) + "\n")
    print(f"Wrote {len(out)} conformance vectors to {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
