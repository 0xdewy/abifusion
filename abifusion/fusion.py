"""ABI Fusion: 4byte signatures + evmole, disambiguated together.

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
(measured 96.1% on held-out contracts; see
``eval_output/canonical_evaluation.md``).

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

from abifusion.utils.signature_lookup import SignatureLookup

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


Confidence = str  # Literal["high", "medium", "low"] but plain str for readability


def choose_candidate(
    candidates: List[Tuple[str, Tuple[str, ...]]],
    evmole_types: Optional[Tuple[str, ...]],
) -> Optional[Tuple[str, Tuple[str, ...], Confidence]]:
    """Pick the 4byte candidate that best matches evmole's type structure.

    Returns (name, types, confidence) where confidence is:
      - high:   exact match to evmole's argument count and types
      - medium: same argument count with partial type overlap
      - low:    no evmole signal; picked first candidate from 4byte

    Returns None if no candidates are available.
    """
    if not candidates:
        return None
    if evmole_types is not None:
        for name, types in candidates:
            if types == evmole_types:
                return name, types, "high"
        same_arity = [c for c in candidates if len(c[1]) == len(evmole_types)]
        if same_arity:
            best = max(
                same_arity,
                key=lambda c: sum(a == b for a, b in zip(c[1], evmole_types)),
            )
            return best[0], best[1], "medium"
    return candidates[0][0], candidates[0][1], "low"


class ABIFusion:
    """Fuse 4byte signatures with evmole analysis to build a Solidity ABI."""

    _known_selectors: Optional[Dict[str, Dict[str, Any]]] = None
    _known_interfaces: Optional[List[Dict[str, Any]]] = None

    def __init__(self, signature_lookup: Optional[SignatureLookup] = None):
        self.sig = signature_lookup or SignatureLookup()
        self._ml: Optional[Any] = None
        self._ml_load_attempted = False
        self._ensure_known_selectors()
        self._ensure_known_interfaces()

    @property
    def ml(self) -> Optional[Any]:
        """Load the optional ML fallback only when it is actually needed."""
        if self._ml_load_attempted:
            return self._ml
        self._ml_load_attempted = True
        try:
            from abifusion.ml_reconstructor import MLReconstructor

            self._ml = MLReconstructor.get_instance()
        except Exception as e:
            logger.warning("ML fallback unavailable: %s", e)
            self._ml = None
        return self._ml

    def _ensure_known_selectors(self) -> None:
        if ABIFusion._known_selectors is not None:
            return
        table_path = Path(__file__).parent.parent / "data" / "known_selector_signatures.json"
        if table_path.exists():
            with open(table_path) as f:
                ABIFusion._known_selectors = json.load(f)
            logger.info("Loaded %d known selector signatures", len(ABIFusion._known_selectors))
        else:
            ABIFusion._known_selectors = {}
            logger.warning("Known selector table not found at %s", table_path)

    def _ensure_known_interfaces(self) -> None:
        if ABIFusion._known_interfaces is not None:
            return
        table_path = Path(__file__).parent.parent / "data" / "known_interface_sets.json"
        if table_path.exists():
            with open(table_path) as f:
                ABIFusion._known_interfaces = json.load(f)
            logger.info("Loaded %d known interface sets", len(ABIFusion._known_interfaces))
        else:
            ABIFusion._known_interfaces = []
            logger.debug("Known interface sets file not found at %s", table_path)

    def _complete_interfaces(
        self,
        bytecode_selectors: set,
        results_by_selector: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        completed: List[Dict[str, Any]] = []
        for iface in self._known_interfaces:
            triggers = set(iface["triggers"]["selector_set"])
            min_matches = iface["triggers"].get("min_matches", 2)

            matching = triggers & bytecode_selectors
            if len(matching) < min_matches:
                continue

            for func in iface["functions"]:
                sel = func["selector"]
                if sel in results_by_selector:
                    continue

                completed.append({
                    "type": "function",
                    "name": func["name"],
                    "selector": sel,
                    "inputs": [{"type": t, "name": ""} for t in func["types"]],
                    "source": "known-interface-completion",
                    "confidence": "medium",
                })
                results_by_selector[sel] = True

        return completed

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
            from abifusion.reconstructor import BytecodeParser

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
                name, types, confidence = chosen
                source = "signature+evmole" if ev is not None else "signature"
                func_candidates = [
                    {"name": c[0], "types": list(c[1]), "source": "4byte"}
                    for c in candidates
                ]
            elif ev is not None:
                name, types, source = f"function_{selector}", ev, "evmole"
                confidence = "high"
                func_candidates = None
            else:
                known = self._known_selectors.get(selector)
                if known is not None:
                    name = known["name"]
                    types = tuple(known["types"])
                    source = "known-selector-table"
                    confidence = "high"
                    func_candidates = None
                else:
                    # Tier 3: ML model prediction (family classification)
                    # Uses function family + type predictions from bytecode
                    ml = self.ml
                    ml_resp = ml.predict(bytecode, selector) if ml is not None else {}
                    if ml_resp.get("family") is not None:
                        name = ml_resp["family"]
                        types = tuple(ml_resp.get("types", []))
                        source = "ml"
                        confidence = "low"
                        func_candidates = None
                    elif ml_resp.get("family_top3"):
                        name = ml_resp["family_top3"][0]
                        types = tuple(ml_resp.get("types", []))
                        source = "ml"
                        confidence = "low"
                        func_candidates = None
                    else:
                        name = f"function_{selector}"
                        types = ()
                        source = "selector"
                        confidence = "low"
                        func_candidates = None

            functions.append(
                {
                    "type": "function",
                    "name": name,
                    "selector": selector,
                    "inputs": [{"type": t, "name": ""} for t in types],
                    "source": source,
                    "confidence": confidence,
                    "candidates": func_candidates,
                }
            )

        try:
            all_bytecode_selectors = set(evmole_types) | extra
            completed = self._complete_interfaces(
                all_bytecode_selectors,
                {f["selector"]: f for f in functions},
            )
            functions.extend(completed)
        except Exception as e:
            logger.warning("interface completion failed: %s", e)

        return {
            "functions": functions,
            "metadata": {
                "function_count": len(functions),
                "evmole_selectors": len(evmole_types),
                "extra_selectors": len(extra - set(evmole_types)),
            },
        }
