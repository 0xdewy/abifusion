"""abifusion - Fuse signature databases with evmole to build Solidity ABIs."""

__version__ = "0.1.0"
__author__ = "RL Attacker Team"
__description__ = "Fuse signature databases with evmole static analysis to build Solidity ABIs from EVM bytecode"

from .fusion import ABIFusion, choose_candidate, fuse_abi, parse_signature, split_args

__all__ = [
    "ABIFusion",
    "choose_candidate",
    "fuse_abi",
    "parse_signature",
    "split_args",
]
