"""Shared, side-effect-free helpers for conformance vectors.

Kept separate from ``test_vectors.py`` so the generator script can import the
JSON->kwargs conversion without triggering pytest collection.
"""

from typing import Any, Dict


def to_core_kwargs(inp: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a language-neutral JSON input bundle into fuse_abi kwargs.

    JSON has no tuples or sets, so a port performs the analogous adaptation to
    its own native types. Here: arg-type lists -> tuples, ``null`` -> ``None``,
    the selector list -> a set.
    """
    evmole_types = {
        sel: (None if types is None else tuple(types))
        for sel, types in inp["evmole_types"].items()
    }
    candidates_by_selector = {
        sel: [(name, tuple(types)) for name, types in cands]
        for sel, cands in inp["candidates_by_selector"].items()
    }
    return {
        "evmole_types": evmole_types,
        "extra_selectors": set(inp["extra_selectors"]),
        "candidates_by_selector": candidates_by_selector,
        "known_selectors": inp["known_selectors"],
        "known_interfaces": inp["known_interfaces"],
        "v4_before_swap_present": inp["v4_before_swap_present"],
        "v4_pool_manager_present": inp["v4_pool_manager_present"],
        "ml_predictions": inp["ml_predictions"],
    }
