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
(measured 96.4% on 6,744 functions; see ``eval_output/fusion_eval.md``).

For selectors not in any 4byte database AND where evmole finds nothing:
  * A dataset-derived known-signature table (from verified ground-truth contracts)
    fills the gap for repeated selectors (e.g., Uniswap V3 callbacks, NFT setters)
    that appear consistently across multiple contracts.
  * An ML model prediction is used as a third-tier fallback (stubbed for now).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from abi_reconstructor.ml_reconstructor import MLReconstructor
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

    _known_selectors: Optional[Dict[str, Dict[str, Any]]] = None

    def __init__(self, signature_lookup: Optional[SignatureLookup] = None):
        self.sig = signature_lookup or SignatureLookup()
        self.ml = MLReconstructor.get_instance()
        self._ensure_known_selectors()

    def _ensure_known_selectors(self) -> None:
        if FusionReconstructor._known_selectors is not None:
            return
        table_path = Path(__file__).parent.parent / "data" / "known_selector_signatures.json"
        if table_path.exists():
            with open(table_path) as f:
                FusionReconstructor._known_selectors = json.load(f)
            logger.info("Loaded %d known selector signatures", len(FusionReconstructor._known_selectors))
        else:
            FusionReconstructor._known_selectors = {}
            logger.warning("Known selector table not found at %s", table_path)

    def _candidates(self, selector: str) -> List[Tuple[str, Tuple[str, ...]]]:
        rows = self.sig.lookup_openchain(selector) or self.sig.lookup_4byte(selector)
        out: List[Tuple[str, Tuple[str, ...]]] = []
        for s in rows:
            parsed = parse_signature(s.get("text_signature", ""))
            if parsed is not None:
                out.append(parsed)
        return out

    def reconstruct(self, bytecode: str) -> Dict[str, Any]:
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

        extra: set = set()
        try:
            from abi_reconstructor.reconstructor import BytecodeParser

            for s in BytecodeParser().extract_selectors(bytecode):
                extra.add(s.selector.lower())
        except Exception as e:
            logger.warning("selector extraction failed: %s", e)

        functions = []
        for selector in sorted(set(evmole_types) | extra):
            ev = evmole_types.get(selector)
            candidates = self._candidates(selector)
            chosen = choose_candidate(candidates, ev)

            if chosen is not None:
                name, types = chosen
                source = "signature+evmole" if ev is not None else "signature"
            elif ev is not None:
                name, types, source = f"function_{selector}", ev, "evmole"
            else:
                known = self._known_selectors.get(selector)
                if known is not None:
                    name = known["name"]
                    types = tuple(known["types"])
                    source = "known-selector-table"
                else:
                    # Tier 3: ML model prediction (family classification)
                    # Uses function family + type predictions from bytecode
                    ml_resp = self.ml.predict(bytecode, selector)
                    if ml_resp.get("family") is not None:
                        name = ml_resp["family"]
                        types = tuple(ml_resp.get("types", []))
                        source = "ml"
                    elif ml_resp.get("family_top3"):
                        name = ml_resp["family_top3"][0]
                        types = tuple(ml_resp.get("types", []))
                        source = "ml"
                    else:
                        name = f"function_{selector}"
                        types = ()
                        source = "selector"

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
