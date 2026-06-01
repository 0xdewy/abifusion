"""Utilities for efficient bytecode processing."""

from typing import List

import torch


def hex_to_tokens_vectorized(hex_str: str, max_len: int = 512) -> List[int]:
    """Convert hex string to token list using vectorized operations.

    Args:
        hex_str: Hex string (with or without 0x prefix)
        max_len: Maximum sequence length (truncate or pad)

    Returns:
        List of integer tokens (0-255)
    """
    # Clean hex
    hex_str = hex_str.lower().replace("0x", "")

    # Handle empty string
    if not hex_str:
        return [0] * max_len

    # Convert hex string to bytes using built-in function
    try:
        # This is the fastest way to convert hex to bytes
        byte_array = bytes.fromhex(hex_str)
    except ValueError:
        # Handle invalid hex by trying to clean it
        # Remove non-hex characters
        import re

        hex_str = re.sub(r"[^0-9a-f]", "", hex_str)
        if not hex_str:
            return [0] * max_len
        # Ensure even length
        if len(hex_str) % 2 != 0:
            hex_str = hex_str[:-1]  # Remove last character
        try:
            byte_array = bytes.fromhex(hex_str)
        except ValueError:
            # If still invalid, return zeros
            return [0] * max_len

    # Convert bytes to list of integers
    tokens = list(byte_array)

    # Pad or truncate
    if len(tokens) > max_len:
        tokens = tokens[:max_len]
    else:
        tokens = tokens + [0] * (max_len - len(tokens))

    return tokens


def hex_to_tokens_batch_vectorized(
    hex_strings: List[str], max_len: int = 512
) -> List[List[int]]:
    """Convert batch of hex strings to token lists using vectorized operations.

    Args:
        hex_strings: List of hex strings
        max_len: Maximum sequence length

    Returns:
        List of token lists
    """
    tokens_batch = []
    for hex_str in hex_strings:
        tokens = hex_to_tokens_vectorized(hex_str, max_len)
        tokens_batch.append(tokens)

    return tokens_batch


def hex_to_tensor_vectorized(hex_str: str, max_len: int = 512) -> torch.Tensor:
    """Convert hex string to tensor using vectorized operations.

    Args:
        hex_str: Hex string
        max_len: Maximum sequence length

    Returns:
        torch.LongTensor of shape [max_len]
    """
    tokens = hex_to_tokens_vectorized(hex_str, max_len)
    return torch.LongTensor(tokens)


def hex_to_tensor_batch_vectorized(
    hex_strings: List[str], max_len: int = 512
) -> torch.Tensor:
    """Convert batch of hex strings to tensor using vectorized operations.

    Args:
        hex_strings: List of hex strings
        max_len: Maximum sequence length

    Returns:
        torch.LongTensor of shape [batch_size, max_len]
    """
    tokens_batch = hex_to_tokens_batch_vectorized(hex_strings, max_len)
    return torch.LongTensor(tokens_batch)


def hex_to_tokens(hex_str: str, max_len: int = 512) -> List[int]:
    """Convert hex string to token list.

    Args:
        hex_str: Hex string
        max_len: Maximum sequence length

    Returns:
        List of integer tokens
    """
    return hex_to_tokens_vectorized(hex_str, max_len)


def clean_bytecode(bytecode: str) -> str:
    """Clean and normalize bytecode string.

    Args:
        bytecode: Raw bytecode (hex string or file path)

    Returns:
        Cleaned hex bytecode without 0x prefix or whitespace
    """
    if isinstance(bytecode, str):
        # Remove 0x prefix if present
        if bytecode.startswith("0x"):
            bytecode = bytecode[2:]

        # Remove all whitespace
        bytecode = bytecode.replace(" ", "").replace("\n", "").replace("\t", "")

        # Convert to lowercase
        bytecode = bytecode.lower()

    return bytecode


def extract_selector_context(
    bytecode: str, position: int, context_size: int = 500
) -> str:
    """Extract context around a selector position in bytecode.

    Extracts an asymmetric window matching training: context_size hex chars
    before the selector, the selector itself, and context_size hex chars after.

    Args:
        bytecode: Clean hex bytecode (no 0x prefix)
        position: Position of selector in hex characters (not bytes)
        context_size: Hex characters of context before and after selector

    Returns:
        Context string around selector
    """
    selector_len = 8

    start = max(0, position - context_size)
    end = min(len(bytecode), position + selector_len + context_size)
    context = bytecode[start:end]

    return context
