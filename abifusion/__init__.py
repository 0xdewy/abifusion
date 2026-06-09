"""abifusion - Fuse signature databases with evmole to build Solidity ABIs.

The recommended path is :class:`ABIFusion`, which fuses signature databases
(openchain/4byte) with evmole's static analysis to reach ~98% exact parameter-type
accuracy. :class:`OfflineABI` is a fully offline rule-based fallback.
"""

__version__ = "0.1.0"
__author__ = "RL Attacker Team"
__description__ = "Fuse signature databases with evmole static analysis to build Solidity ABIs from EVM bytecode"

from .database import FourByteDatabase
from .fusion import ABIFusion
from .reconstructor import OfflineABI
from .selector_extractor import SelectorExtractor

__all__ = [
    "ABIFusion",
    "OfflineABI",
    "FourByteDatabase",
    "SelectorExtractor",
]
