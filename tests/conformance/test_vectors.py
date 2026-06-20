"""Cross-language conformance vectors for the pure fusion core.

Each vector is a fully self-contained ``(input, expected)`` pair: feed the input
straight into :func:`abifusion.core.fuse.fuse_abi` (no evmole, no network, no
torch) and the output must equal ``expected`` byte-for-byte. A port to another
language reruns the identical ``vectors.json`` against its own ``fuse_abi`` and
must produce the same output — that is the contract these vectors lock in.

Regenerate ``vectors.json`` after intentionally changing core behavior:

    python scripts/dump_conformance_vectors.py
"""

import json
import sys
from pathlib import Path

import pytest

from abifusion.core.fuse import fuse_abi

sys.path.insert(0, str(Path(__file__).parent))
from _loader import to_core_kwargs  # noqa: E402

VECTORS_PATH = Path(__file__).parent / "vectors.json"


def _load_vectors():
    if not VECTORS_PATH.exists():
        pytest.skip("vectors.json not generated; run scripts/dump_conformance_vectors.py")
    return json.loads(VECTORS_PATH.read_text())


@pytest.mark.parametrize("case", _load_vectors(), ids=lambda c: c["name"])
def test_vector(case):
    result = fuse_abi(**to_core_kwargs(case["input"]))
    assert result == case["expected"]
