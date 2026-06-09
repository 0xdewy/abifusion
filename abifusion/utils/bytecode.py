"""Bytecode analysis utilities shared across the project."""

from __future__ import annotations

from typing import List, Optional

PUSH1 = 0x60
PUSH32 = 0x7F
CALLDATALOAD = 0x35


def parse_bytecode_hex(bytecode_hex: str) -> Optional[bytes]:
    code = bytecode_hex[2:] if bytecode_hex.startswith("0x") else bytecode_hex
    try:
        return bytes.fromhex(code)
    except ValueError:
        return None


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
