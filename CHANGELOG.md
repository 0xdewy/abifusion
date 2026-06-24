# Changelog

## v0.1.0 (2026-06-24)

First release.

- **Core:** `fuse_abi()` — deterministic, I/O-free fusion of evmole static analysis
  with openchain.xyz / 4byte.directory signature databases.
- **CLI:** `abifusion fusion --bytecode 0x...` with `--address`, `--bytecode-file`,
  `--output-format abi|annotated`, `--min-confidence`, and `--output` flags.
- **Python API:** `from abifusion import ABIFusion` → `.reconstruct(bytecode)`.
- **Accuracy:** 98.2% held-out exact parameter type accuracy on 1000 verified
  contracts (1925 functions) across Ethereum, Arbitrum, Optimism, and Base.
  evmole baseline achieves 91.7% on the same split.
- **Rust port:** `abifusion-core` crate (pure fusion engine) + `abifusion` crate
  (evmole adapter + HTTP signatures). Conformance-tested against the Python core.
- **Libs:** Python 3.8+, `requests`, `evmole >= 0.8.4`.
