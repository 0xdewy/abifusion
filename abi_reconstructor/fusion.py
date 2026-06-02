"""Fusion ABI reconstructor: 4byte signatures + evmole, disambiguated together.

Neither source is sufficient alone:

  * **4byte** maps a selector to candidate text signatures with *exact* types
    (it knows ``bytes32`` vs ``uint256``), but the database is full of spam
    collisions — e.g. ``a9059cbb`` returns ``workMyDirefulOwner(uint256,uint256)``
    ahead of the real ``transfer(address,uint256)``.
  * **evmole** recovers each function's argument *structure* by static analysis
    (no spam), but can't always tell ``bytes32`` from ``uint256`` or ``address``
    from ``uint160``, and misses some selectors.

Fusing them — use evmole's structure to pick the right 4byte candidate, and the
4byte signature to supply the exact types — beats evmole's exact-type accuracy
(measured 94.3% vs 90.0% on 6,744 functions, 0 regressions; see
``eval_output/fusion_ceiling.md``).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from abi_reconstructor.utils.signature_lookup import SignatureLookup

logger = logging.getLogger(__name__)


def split_args(arg_str: str) -> Tuple[str, ...]:
    """Split a Solidity argument list, respecting nested tuple parens."""
    arg_str = arg_str.strip()
    if not arg_str:
        return ()
    out: List[str] = []
    depth = 0
    cur = ""
    for ch in arg_str:
        if ch == "(":
            depth += 1
            cur += ch
        elif ch == ")":
            depth -= 1
            cur += ch
        elif ch == "," and depth == 0:
            out.append(cur.strip())
            cur = ""
        else:
            cur += ch
    if cur.strip():
        out.append(cur.strip())
    return tuple(out)


def parse_signature(text_sig: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """``'transfer(address,uint256)'`` -> ``('transfer', ('address','uint256'))``."""
    i = text_sig.find("(")
    if i == -1:
        return None
    name = text_sig[:i]
    types = split_args(text_sig[i + 1 : text_sig.rfind(")")])
    return name, types


def choose_candidate(
    candidates: List[Tuple[str, Tuple[str, ...]]],
    evmole_types: Optional[Tuple[str, ...]],
) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """Pick the 4byte candidate that best matches evmole's type structure.

    Order: exact match to evmole → same arity with maximal per-position
    agreement → first candidate (4byte's default) when evmole gives no help.
    """
    if not candidates:
        return None
    if evmole_types is not None:
        for name, types in candidates:
            if types == evmole_types:
                return name, types
        same_arity = [c for c in candidates if len(c[1]) == len(evmole_types)]
        if same_arity:
            return max(
                same_arity,
                key=lambda c: sum(a == b for a, b in zip(c[1], evmole_types)),
            )
    return candidates[0]


class FusionReconstructor:
    """Reconstruct an ABI by fusing 4byte signatures with evmole analysis."""

    def __init__(self, signature_lookup: Optional[SignatureLookup] = None):
        self.sig = signature_lookup or SignatureLookup()

    def _candidates(self, selector: str) -> List[Tuple[str, Tuple[str, ...]]]:
        """4byte candidates for a selector, preferring openchain.

        openchain has higher coverage and far less collision spam, so it is the
        primary source; 4byte.directory is used only when openchain returns
        nothing (mixing the two re-introduces 4byte's spam and hurts accuracy).
        """
        rows = self.sig.lookup_openchain(selector) or self.sig.lookup_4byte(selector)
        out: List[Tuple[str, Tuple[str, ...]]] = []
        for s in rows:
            parsed = parse_signature(s.get("text_signature", ""))
            if parsed is not None:
                out.append(parsed)
        return out

    def reconstruct(self, bytecode: str) -> Dict[str, Any]:
        """Reconstruct the ABI from runtime bytecode.

        Returns ``{"functions": [...], "metadata": {...}}`` where each function
        has ``name``, ``inputs`` (list of ``{"type": ...}``), ``selector`` and a
        ``source`` tag (``4byte+evmole``, ``4byte``, ``evmole`` or ``selector``).
        """
        code = bytecode[2:] if bytecode.startswith("0x") else bytecode

        # evmole: per-function selector + argument structure.
        evmole_types: Dict[str, Optional[Tuple[str, ...]]] = {}
        try:
            import evmole

            info = evmole.contract_info(code, selectors=True, arguments=True)
            for f in info.functions:
                evmole_types[f.selector.lower()] = split_args(f.arguments or "")
        except Exception as e:  # noqa: BLE001
            logger.warning("evmole failed: %s", e)

        # Repo selector extractor recovers selectors evmole missed.
        extra: set = set()
        try:
            from abi_reconstructor.reconstructor import BytecodeParser

            for s in BytecodeParser().extract_selectors(bytecode):
                extra.add(s.selector.lower())
        except Exception as e:  # noqa: BLE001
            logger.warning("selector extraction failed: %s", e)

        functions = []
        for selector in sorted(set(evmole_types) | extra):
            ev = evmole_types.get(selector)
            candidates = self._candidates(selector)
            chosen = choose_candidate(candidates, ev)

            if chosen is not None:
                name, types = chosen
                source = "4byte+evmole" if ev is not None else "4byte"
            elif ev is not None:
                name, types, source = f"function_{selector}", ev, "evmole"
            else:
                name, types, source = f"function_{selector}", (), "selector"

            functions.append(
                {
                    "type": "function",
                    "name": name,
                    "selector": selector,
                    "inputs": [{"type": t, "name": ""} for t in types],
                    "source": source,
                }
            )

        return {
            "functions": functions,
            "metadata": {
                "function_count": len(functions),
                "evmole_selectors": len(evmole_types),
                "extra_selectors": len(extra - set(evmole_types)),
            },
        }
