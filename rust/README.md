# abifusion (Rust)

A native Rust port of abifusion's fusion engine — build Solidity ABIs from EVM
bytecode by fusing [evmole](https://crates.io/crates/evmole) static analysis with
signature databases (openchain → 4byte). See the repo-root `PORTING.md` for the
architecture and the cross-language contract.

Two crates:

- **`abifusion-core`** — the pure, I/O-free fusion engine (`fuse_abi`). Depends
  only on `serde`/`serde_json`. A direct port of `abifusion/core/fuse.py`; the
  dependency graph structurally guarantees it never touches evmole or the network.
- **`abifusion`** — adapters (evmole `BytecodeAnalyzer`, HTTP `SignatureProvider`)
  + the `reconstruct()` orchestrator. The curated tables in repo-root `data/` are
  compiled in via `include_str!` (single source of truth with Python).

The ML tier is intentionally omitted (the model is PyTorch-only with no ONNX
export); `fuse_abi` receives empty predictions. A CLI binary is not yet included.

## Use

```rust
let abi = abifusion::reconstruct("0x6080...");   // -> abifusion_core::Abi (Serialize)
println!("{}", serde_json::to_string_pretty(&abi)?);
```

## Test / build

```bash
cargo test -p abifusion-core   # runs the shared conformance vectors
cargo build --workspace
cargo run -p abifusion --example reconstruct -- <bytecode-hex>
# ABIFUSION_NO_SIGS=1 skips network lookups (deterministic, offline)
```

## Parity

`abifusion-core/tests/conformance.rs` runs the **same**
`tests/conformance/vectors.json` the Python suite uses and asserts identical
output. Parity has also been verified end-to-end against the Python
implementation on real contract bytecode (evmole arguments + fusion tiers).
