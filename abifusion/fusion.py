"""ABI Fusion orchestrator: wire the adapters into the pure fusion core.

Neither source is sufficient alone:

  * **4byte** maps a selector to candidate text signatures with *exact* types
    (it knows ``bytes32`` vs ``uint256``), but the database is full of spam
    collisions — e.g. ``a9059cbb`` returns ``workMyDirefulOwner(uint256,uint256)``
    ahead of the real ``transfer(address,uint256)``.
  * **evmole** recovers each function's argument *structure* by static analysis
    (no spam), but can't always tell ``bytes32`` from ``uint256`` or ``address``
    from ``uint160``, and misses some selectors.

The deterministic fusion logic lives in :mod:`abifusion.core.fuse` (I/O-free and
portable). This module is the thin Python orchestrator: it calls the adapters
(see :mod:`abifusion.ports` / :mod:`abifusion.adapters`) to resolve evmole
structure and signature-DB candidates, then hands them to
:func:`abifusion.core.fuse.fuse_abi`.

The pure helpers ``split_args``, ``parse_signature`` and ``choose_candidate`` are
re-exported here for backward compatibility with existing imports.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

try:
    from importlib.resources import files
except ImportError:  # Python <3.9
    from importlib_resources import files  # type: ignore[assignment,unused-ignore]

from abifusion.adapters import EvmoleBytecodeAnalyzer
from abifusion.core.fuse import (
    choose_candidate,
    fuse_abi,
    parse_signature,
    split_args,
)
from abifusion.ports import BytecodeAnalyzer
from abifusion.utils.signature_lookup import SignatureLookup

logger = logging.getLogger(__name__)

__all__ = [
    "ABIFusion",
    "choose_candidate",
    "parse_signature",
    "split_args",
    "fuse_abi",
]


class ABIFusion:
    """Fuse 4byte signatures with evmole analysis to build a Solidity ABI."""

    _known_selectors: Optional[Dict[str, Dict[str, Any]]] = None
    _known_interfaces: Optional[List[Dict[str, Any]]] = None

    def __init__(
        self,
        signature_lookup: Optional[SignatureLookup] = None,
        analyzer: Optional[BytecodeAnalyzer] = None,
    ):
        self.sig = signature_lookup or SignatureLookup()
        self.analyzer: BytecodeAnalyzer = analyzer or EvmoleBytecodeAnalyzer()
        self._ensure_known_selectors()
        self._ensure_known_interfaces()

    def _ensure_known_selectors(self) -> None:
        if ABIFusion._known_selectors is not None:
            return
        table_path = files("abifusion.data") / "known_selector_signatures.json"
        try:
            with open(table_path) as f:
                ABIFusion._known_selectors = json.load(f)
            logger.info("Loaded %d known selector signatures", len(ABIFusion._known_selectors))
        except FileNotFoundError:
            ABIFusion._known_selectors = {}
            logger.warning("Known selector table not found")

    def _ensure_known_interfaces(self) -> None:
        if ABIFusion._known_interfaces is not None:
            return
        table_path = files("abifusion.data") / "known_interface_sets.json"
        try:
            with open(table_path) as f:
                ABIFusion._known_interfaces = json.load(f)
            logger.info("Loaded %d known interface sets", len(ABIFusion._known_interfaces))
        except FileNotFoundError:
            ABIFusion._known_interfaces = []
            logger.debug("Known interface sets file not found")

    def _candidates(self, selector: str):
        rows = self.sig.lookup_openchain(selector) or self.sig.lookup_4byte(selector)
        return [
            parsed
            for s in rows
            if (parsed := parse_signature(s.get("text_signature", ""))) is not None
        ]

    def reconstruct(self, bytecode: str) -> Dict[str, Any]:
        analysis = self.analyzer.analyze(bytecode)
        evmole_types = analysis.evmole_types
        extra = analysis.extra_selectors

        all_selectors = sorted(set(evmole_types) | extra)
        candidates_by_selector = {sel: self._candidates(sel) for sel in all_selectors}

        # The pure core keeps an ``ml_predictions`` data-in seam for a future
        # model, but no ML subsystem is wired here: signature DBs + evmole +
        # curated tables cover the addressable space (the prior PyTorch family
        # classifier never meaningfully fired). Pass empty predictions.
        return fuse_abi(
            evmole_types,
            extra,
            candidates_by_selector,
            self._known_selectors or {},
            self._known_interfaces or [],
            v4_before_swap_present=analysis.v4_before_swap_present,
            v4_pool_manager_present=analysis.v4_pool_manager_present,
            ml_predictions={},
        )
