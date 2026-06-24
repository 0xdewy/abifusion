# abifusion

A small tool that rebuilds Solidity function ABIs from EVM bytecode by fusing
two imperfect signals.

```
evmole (static analysis)  +  4byte.directory (signature lookup)
     ↓                              ↓
  selector + arg structure     exact type names
     ↓                              ↓
          ┌──────────────────────┐
          │   fuse_abi() core    │
          │  choose_candidate()  │
          └──────────────────────┘
                    ↓
         function ABI with name + types
```

## The tool (`abifusion/`)

This is what `pip install abifusion` gives you. About 500 lines of Python.

| File | What it does |
| --- | --- |
| `core/fuse.py` | The pure, I/O-free fusion engine. Takes evmole output + 4byte candidates, picks the best match, outputs a function ABI. Deterministic — same inputs always produce the same output. |
| `fusion.py` | Orchestrator that wires the adapters into `fuse_abi`. The class you import as `from abifusion import ABIFusion`. |
| `adapters.py` | Calls evmole to get selectors + argument structure from bytecode. |
| `ports.py` | Protocol interfaces. Defines what an adapter must look like — makes the core portable to Rust/TypeScript. |
| `utils/signature_lookup.py` | Queries openchain.xyz and 4byte.directory for candidate text signatures per selector. |
| `utils/bytecode.py` | Bytecode helpers: hex cleaning, PUSH4+EQ selector scanning. |
| `cli.py` | `abifusion fusion --bytecode 0x...` entry point. |

## The rest of the repo (not installed)

| Directory | What it is |
| --- | --- |
| `tooling/abifusion_legacy/` | Research-only offline reconstructor with its own SQLite DB, ML features, and Etherscan fetcher. Predecessor to the fusion approach. Not imported by the release package. |
| `scripts/eval/` | Evaluation scripts: benchmark the tool against 500 verified mainnet contracts, mine failures, do holdout analysis. Run with `python scripts/eval/run_evaluation.py`. |
| `schema/` | Language-neutral JSON schemas for the output format and data tables. |
| `rust/` | Rust port of `fuse_abi` — same algorithm, validated against shared conformance vectors. |
| `tests/` | Pytest suite (141 tests). Includes conformance vectors that validate the Python and Rust cores produce identical output. |
| `data/` | Evaluation datasets (parquet files from Sourcify). |
| `eval_output/` | Generated evaluation reports (see `canonical_evaluation.md` for the latest). |
| `docs/` | Planning documents and research notes. |

## At A Glance

```bash
# The only command you need
abifusion fusion --bytecode "0x6080..."
abifusion fusion --bytecode-file contract.hex
abifusion fusion --address 0x...          # needs ETH_RPC_URL
```

```json
{
  "functions": [
    {
      "type": "function",
      "name": "transfer",
      "selector": "a9059cbb",
      "inputs": [{"type": "address"}, {"type": "uint256"}],
      "source": "signature+evmole",
      "confidence": "high"
    }
  ]
}
```

```python
from abifusion import ABIFusion
result = ABIFusion().reconstruct("0x6080...")
```

## Accuracy

Evaluated on a deterministic 80/20 split of **1000 verified contracts** across
Ethereum mainnet, Arbitrum, Optimism, and Base.

| Metric | Result |
| --- | ---: |
| **Fusion held-out accuracy** | **98.2%** (1890/1925 functions) |
| evmole baseline (same split) | 91.7% |
| Training accuracy | 98.6% |
| Generalization gap | +0.4pp |

Held-out category breakdown:

| Category | Accuracy |
| --- | ---: |
| Simple tokens | 99.6% |
| DeFi AMM | 97.9% |
| DeFi routers | 95.1% |
| Other | 94.8% |

See `eval_output/canonical_evaluation.md` for the full report.

## How It Works

Signature databases know exact types (`bytes32` vs `uint256`) but contain spam
and collisions. Static analysis recovers argument structure cleanly but can't
tell `address` from `uint160`.

`fuse_abi` uses each signal where it's strongest:

1. Run evmole → discover selectors and argument structure.
2. Look up each selector in openchain.xyz and 4byte.directory → get candidate text signatures.
3. Match signatures against evmole's structure → pick the one that agrees.
4. Fall back to evmole or a known-selector table when no signature is available.

Example: for selector `a9059cbb`, 4byte returns both `workMyDirefulOwner(uint256,uint256)` (spam) and `transfer(address,uint256)` (real). Evmole sees `address,uint256` in the bytecode, so fusion picks the real one.

## Installation

```bash
pip install abifusion       # just the tool
pip install abifusion[dev]  # tool + pytest, ruff, mypy, data libs
```

## Limitations

- No output (return type) recovery.
- Needs network access for best accuracy (signature lookups are cached).
- Proxy patterns and delegatecall routers need separate handling.
- Does not yet use Sourcify/Etherscan as a verified-ABI fallback.

## License

MIT. See [LICENSE](LICENSE).
