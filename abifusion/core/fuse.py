"""Pure fusion core: 4byte signatures + evmole structure, disambiguated together.

This module is the **portable core**. It is deterministic and I/O-free — it
imports nothing beyond the standard library and operates only on
already-resolved inputs supplied by the adapters (see ``abifusion/ports.py``):

  * ``evmole_types`` — per-selector argument structure recovered by static
    analysis (the ``BytecodeAnalyzer`` port).
  * ``candidates_by_selector`` — candidate text signatures per selector from a
    signature database (the ``SignatureProvider`` port).
  * ``known_selectors`` / ``known_interfaces`` — the curated tables under
    ``abifusion/core/data/``.
  * ``ml_predictions`` — optional family/type predictions (the
    ``FamilyPredictor`` port); the core never imports torch.

Fusing these — use evmole's structure to pick the right 4byte candidate, and the
4byte signature to supply the exact types — beats evmole's exact-type accuracy
(see ``eval_output/champion_benchmark.md``). Any language with these four inputs
can reproduce ``fuse_abi`` byte-for-byte; the conformance vectors in
``tests/conformance/`` exist to prove it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple


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


# Uniswap V4 hook callback selectors and the triggers that gate the post-pass.
_V4_HOOK_SELECTORS = [
    "bc29bafc", "468ead2c", "5cb32d10", "606e8192", "b6d4944a",
    "b772b8cc", "c13d1c69", "1f29cf9d", "d950bd74", "ebe1cdaf",
]

_TRIGGER_BEFORE_SWAP = "575e24b4"
_TRIGGER_POOL_MANAGER = "dc4c90d3"


def _complete_interfaces(
    bytecode_selectors: Set[str],
    results_by_selector: Dict[str, Any],
    known_interfaces: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Fill in remaining functions of a known interface when enough of its
    trigger selectors are present in the bytecode."""
    completed: List[Dict[str, Any]] = []
    for iface in known_interfaces:
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


def _emit_undiscovered_v4_hooks(
    existing: Dict[str, Dict[str, Any]],
    known_selectors: Dict[str, Dict[str, Any]],
    *,
    before_swap_present: bool,
    pool_manager_present: bool,
) -> List[Dict[str, Any]]:
    """Post-processing pass: emit V4 hook callbacks when the beforeSwap trigger
    is present in bytecode but the poolManager trigger is absent.

    Standard ``_complete_interfaces`` requires both ``575e24b4`` and ``dc4c90d3``
    (min_matches=2). Contracts with only ``575e24b4`` (beforeSwap) in bytecode
    would miss all 10 V4 hook callbacks — this pass fills that gap.

    Safety: ``dc4c90d3``-only contracts (non-V4 hooks that use poolManager) are
    excluded because this pass only fires on ``575e24b4`` presence.

    The two presence booleans are precomputed by the ``BytecodeAnalyzer`` adapter
    so the pure core never scans raw bytecode.
    """
    if not (before_swap_present and not pool_manager_present):
        return []

    completed = []
    for sel in _V4_HOOK_SELECTORS:
        if sel in existing:
            continue

        known = known_selectors.get(sel)
        if known is None:
            continue

        completed.append({
            "type": "function",
            "name": known["name"],
            "selector": sel,
            "inputs": [{"type": t, "name": ""} for t in known["types"]],
            "source": "known-interface-completion-post",
            "confidence": "medium",
            "candidates": None,
        })

    return completed


def fuse_abi(
    evmole_types: Dict[str, Optional[Tuple[str, ...]]],
    extra_selectors: Set[str],
    candidates_by_selector: Dict[str, List[Tuple[str, Tuple[str, ...]]]],
    known_selectors: Dict[str, Dict[str, Any]],
    known_interfaces: List[Dict[str, Any]],
    *,
    v4_before_swap_present: bool,
    v4_pool_manager_present: bool,
    ml_predictions: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Fuse evmole structure + signature-DB candidates into a Solidity ABI.

    All inputs are already resolved by the adapters; this function performs no
    I/O. See the module docstring for the meaning of each parameter.
    """
    ml_predictions = ml_predictions or {}

    functions = []
    for selector in sorted(set(evmole_types) | extra_selectors):
        ev = evmole_types.get(selector)
        candidates = candidates_by_selector.get(selector, [])
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
            known = known_selectors.get(selector)
            if known is not None:
                name = known["name"]
                types = tuple(known["types"])
                source = "known-selector-table"
                confidence = "high"
                func_candidates = None
            else:
                # Tier 3: ML model prediction (family classification).
                ml_resp = ml_predictions.get(selector, {})
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

    all_bytecode_selectors = set(evmole_types) | extra_selectors
    completed = _complete_interfaces(
        all_bytecode_selectors,
        {f["selector"]: f for f in functions},
        known_interfaces,
    )
    functions.extend(completed)

    existing = {f["selector"]: f for f in functions}
    post_completed = _emit_undiscovered_v4_hooks(
        existing,
        known_selectors or {},
        before_swap_present=v4_before_swap_present,
        pool_manager_present=v4_pool_manager_present,
    )
    functions.extend(post_completed)

    return {
        "functions": functions,
        "metadata": {
            "function_count": len(functions),
            "evmole_selectors": len(evmole_types),
            "extra_selectors": len(extra_selectors - set(evmole_types)),
        },
    }
