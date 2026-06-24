"""Bytecode analysis utilities shared across the project."""

from __future__ import annotations

import re
from typing import List

PUSH1 = 0x60
PUSH32 = 0x7F
CALLDATALOAD = 0x35
EQ = 0x14
PUSH4 = 0x63
MIN_SELECTOR_VALUE = 0x00000100


def clean_bytecode_hex(bytecode_hex: str) -> str:
    code = bytecode_hex[2:] if bytecode_hex.startswith("0x") else bytecode_hex
    code = re.sub(r"[^0-9a-f]", "", code.lower())
    return code[:-1] if len(code) % 2 else code


def extract_push4_selectors(bytecode_hex: str) -> List[str]:
    """Extract likely function selectors from PUSH4 ... EQ dispatch patterns."""
    code = clean_bytecode_hex(bytecode_hex)
    selectors = []
    seen = set()

    for i in range(0, len(code) - 10, 2):
        if code[i : i + 2] != f"{PUSH4:02x}":
            continue

        selector = code[i + 2 : i + 10]
        if selector in seen or selector == "00000000":
            continue
        if len(selector) != 8 or all(c == selector[0] for c in selector):
            continue

        try:
            if int(selector, 16) < MIN_SELECTOR_VALUE:
                continue
        except ValueError:
            continue

        search_end = min(len(code), i + 24)
        if any(code[j : j + 2] == f"{EQ:02x}" for j in range(i + 10, search_end, 2)):
            selectors.append(selector)
            seen.add(selector)

    return selectors


def extract_calldata_offsets(
    bytecode: bytes,
    start_offset: int,
    max_bytes: int = 500,
) -> List[int]:
    """Find PUSHN+CALLDATALOAD offset pairs in bytecode.

    Scans bytecode starting at start_offset for sequences of ``PUSHn <N>``
    followed by ``CALLDATALOAD``. Returns the list of unique pushed offset
    values (each should be a multiple of 32, starting at 4 for the first
    calldata parameter).

    Returns:
        Sorted list of unique calldata offsets found (e.g. [4, 36, 68]).
    """
    offsets = []
    i = start_offset
    end = min(start_offset + max_bytes, len(bytecode))
    last_push_data = None

    while i < end:
        op = bytecode[i]

        if PUSH1 <= op <= PUSH32:
            push_size = op - PUSH1 + 1
            if i + 1 + push_size <= end:
                last_push_data = bytecode[i + 1 : i + 1 + push_size]
            else:
                last_push_data = None
            i += 1 + push_size
            continue

        if op == CALLDATALOAD and last_push_data is not None:
            offset_val = int.from_bytes(last_push_data, "big")
            if offset_val % 32 == 0 and offset_val >= 4:
                offsets.append(offset_val)

        last_push_data = None
        i += 1

    return sorted(set(offsets))
