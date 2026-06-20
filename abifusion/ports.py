"""Adapter interfaces (ports) for the portable fusion core.

These Protocols capture every language-specific edge that the pure core
(``abifusion/core/fuse.py``) deliberately does *not* depend on: bytecode static
analysis, signature-database lookups, and the optional ML family predictor. The
Python implementations live in ``abifusion/adapters.py`` (bytecode analysis) and
the existing ``SignatureLookup`` / ``MLReconstructor`` classes.

A port to another language (TypeScript, Rust) reimplements these three small
adapters against its own evmole binding (evmole ships an npm package and a Rust
crate) and HTTP client, then feeds their output into a reimplementation of
``fuse_abi``. The conformance vectors in ``tests/conformance/`` validate that the
core produces identical output across languages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Set, Tuple, runtime_checkable


@dataclass
class BytecodeAnalysis:
    """Everything the core needs from static bytecode analysis.

    Attributes:
        evmole_types: selector -> recovered argument types (``None`` if the
            selector was found without recoverable argument structure).
        extra_selectors: selectors found by the fallback PUSH4+EQ scan (a
            superset may overlap with ``evmole_types``).
        v4_before_swap_present: whether the Uniswap V4 ``beforeSwap`` trigger
            selector appears in the raw bytecode.
        v4_pool_manager_present: whether the V4 ``poolManager`` trigger selector
            appears in the raw bytecode.
    """

    evmole_types: Dict[str, Optional[Tuple[str, ...]]]
    extra_selectors: Set[str]
    v4_before_swap_present: bool
    v4_pool_manager_present: bool


@runtime_checkable
class BytecodeAnalyzer(Protocol):
    """Bytecode -> selectors + argument structure + V4 trigger flags."""

    def analyze(self, bytecode: str) -> BytecodeAnalysis: ...


@runtime_checkable
class SignatureProvider(Protocol):
    """Selector -> candidate (name, types) signatures from a signature DB."""

    def candidates(self, selector: str) -> List[Tuple[str, Tuple[str, ...]]]: ...


@runtime_checkable
class FamilyPredictor(Protocol):
    """Optional ML tier: bytecode + selector -> family/type prediction.

    The prediction dict may contain ``family`` (str), ``family_top3`` (list of
    str), and ``types`` (list of str). An empty dict means "no prediction".
    """

    def predict(self, bytecode: str, selector: str) -> Dict[str, Any]: ...
