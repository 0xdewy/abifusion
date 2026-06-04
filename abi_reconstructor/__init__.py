"""ABI Reconstructor - Reconstruct Ethereum contract ABIs from EVM bytecode.

The recommended path is :class:`FusionReconstructor`, which fuses signature
databases (openchain/4byte) with evmole's static analysis to reach ~96% exact
parameter-type accuracy. :class:`ABIReconstructor` is an offline rule-based
fallback (4byte + bytecode heuristics, no network).
"""

__version__ = "0.1.0"
__author__ = "RL Attacker Team"
__description__ = "Reconstruct contract ABIs from EVM bytecode by fusing signature databases with evmole"

from .database import FourByteDatabase
from .fusion import FusionReconstructor
from .reconstructor import ABIReconstructor
from .selector_extractor import SelectorExtractor

__all__ = [
    "FusionReconstructor",
    "ABIReconstructor",
    "FourByteDatabase",
    "SelectorExtractor",
]
