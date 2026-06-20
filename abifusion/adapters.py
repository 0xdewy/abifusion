"""Python implementations of the bytecode-analysis port.

This is the Python-specific edge that a port would reimplement against evmole's
npm package or Rust crate. The logic here is lifted verbatim from the inline
analysis that used to live in ``ABIFusion.reconstruct`` so behavior is identical.
"""

from __future__ import annotations

import logging
from typing import Dict, Optional, Set, Tuple

from abifusion.core.fuse import _TRIGGER_BEFORE_SWAP, _TRIGGER_POOL_MANAGER, split_args
from abifusion.ports import BytecodeAnalysis
from abifusion.utils.bytecode import extract_push4_selectors

logger = logging.getLogger(__name__)


class EvmoleBytecodeAnalyzer:
    """Analyze bytecode with evmole, plus a PUSH4+EQ fallback scan.

    Satisfies the ``BytecodeAnalyzer`` protocol. evmole supplies selectors and
    argument structure; the local PUSH4 scanner catches any extra selectors
    evmole missed. Both passes are wrapped so a failure in either degrades
    gracefully (matching the original inline behavior).
    """

    def analyze(self, bytecode: str) -> BytecodeAnalysis:
        code = bytecode[2:] if bytecode.startswith("0x") else bytecode

        evmole_types: Dict[str, Optional[Tuple[str, ...]]] = {}
        try:
            import evmole

            info = evmole.contract_info(code, selectors=True, arguments=True)
            if info is not None and info.functions is not None:
                for f in info.functions:
                    evmole_types[f.selector.lower()] = split_args(f.arguments or "")
        except Exception as e:
            logger.warning("evmole failed: %s", e)

        extra: Set[str] = set()
        try:
            extra.update(extract_push4_selectors(bytecode))
        except Exception as e:
            logger.warning("selector extraction failed: %s", e)

        bc_lower = bytecode.lower()
        return BytecodeAnalysis(
            evmole_types=evmole_types,
            extra_selectors=extra,
            v4_before_swap_present=_TRIGGER_BEFORE_SWAP in bc_lower,
            v4_pool_manager_present=_TRIGGER_POOL_MANAGER in bc_lower,
        )
