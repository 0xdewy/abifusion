# ABI Reconstructor

Reconstruct contract ABIs (Application Binary Interfaces) from EVM bytecode by
**fusing multiple imperfect signal sources** — signature databases (openchain,
4byte) and evmole's static analysis — so the result beats any single tool.

## Why fusion

No single tool recovers parameter types perfectly:

- **Signature databases** (openchain.xyz, 4byte.directory) know the *exact* types
  for a selector's real signature (e.g. `bytes32` vs `uint256`), but 4byte is
  full of spam collisions and neither covers every selector.
- **evmole** recovers each function's argument *structure* by static analysis
  (no spam), but can't always distinguish `bytes32`/`uint256` or `address`/
  `uint160`, and miscounts some arguments.

The **`FusionReconstructor`** uses evmole's structure to pick the correct
signature-database candidate, and the signature to supply the exact types evmole
can't infer.

## Results

Measured on 6,744 functions from 500 verified mainnet contracts with
ground-truth ABIs (`eval_output/fusion_eval.md`):

| Reconstructor | Exact parameter-type accuracy |
| --- | --- |
| **Fusion** (recommended) | **96.4%** |
| evmole baseline | 90.0% |

The fusion fixes 291 functions evmole gets wrong (mostly `bytes32`/`uint256` and
`address`/`uint160`) and breaks none. The ceiling is currently ~96.5%, capped by
signature-database coverage: the residual is selectors absent from openchain/4byte
where evmole is the only available signal.

## Installation

Requires Python 3.8+. [uv](https://github.com/astral-sh/uv) is recommended.

```bash
git clone <repository-url>
cd abi_reconstructor

uv pip install -e .          # or: pip install -e .
uv pip install -e ".[dev]"   # development tools (pytest, ruff, black, mypy)
```

The fusion path needs network access for signature lookups (results are cached).
Data-fetching and evaluation tooling read optional API keys from a `.env` file:

```bash
cp .env-example .env         # then add ETH_RPC_URL / ETHERSCAN_API_KEY as needed
```

## Usage

### Command line

```bash
# Recommended: fusion reconstruction (~96% exact parameter types; needs network)
abi-reconstruct fusion --bytecode "0x6080..."
abi-reconstruct fusion --bytecode-file contract.hex

# Offline rule-based reconstruction (4byte + bytecode heuristics, no network)
abi-reconstruct reconstruct --bytecode "0x6080..."

# Extract function selectors only
abi-reconstruct extract --bytecode "0x6080..."

# Local 4byte signature database
abi-reconstruct populate            # fill from 4byte.directory
abi-reconstruct stats               # show DB stats
abi-reconstruct export --output sigs.json
```

### Python

```python
from abi_reconstructor import FusionReconstructor

result = FusionReconstructor().reconstruct("0x6080...")
for fn in result["functions"]:
    types = ",".join(i["type"] for i in fn["inputs"])
    print(f"{fn['name']}({types})  selector={fn['selector']}  source={fn['source']}")
# result["functions"]: [{type, name, selector, inputs:[{type,name}], source}, ...]
```

For a fully offline fallback (no network, lower accuracy) use `ABIReconstructor`.

## Project structure

```
abi_reconstructor/
├── fusion.py              # FusionReconstructor — the recommended path
├── reconstructor.py       # ABIReconstructor — offline rule-based fallback
├── selector_extractor.py  # selector extraction from bytecode
├── database.py            # local 4byte SQLite database
├── cli.py                 # `abi-reconstruct` entry point
├── data/                  # contract fetching (Etherscan, Sourcify) + datasets
├── features/              # discriminating-instruction feature extraction
└── utils/                 # signature lookup, evmole comparison

scripts/eval/              # benchmarks and gap analysis against ground truth
eval_output/               # evaluation reports (fusion_eval.md, ...)
tests/                     # test suite
docs/research/             # background research notes
```

## Development

```bash
pytest tests/                 # run the test suite
ruff check abi_reconstructor/ # lint
black abi_reconstructor/      # format
mypy abi_reconstructor/       # type check
```

## Limitations

1. **Coverage-bound, not model-bound.** Accuracy is capped by selector coverage
   in openchain/4byte; selectors absent from both fall back to evmole alone.
2. **Network for fusion.** The fusion path queries signature databases (cached);
   the rule-based path is offline but less accurate.
3. **Dynamic dispatch.** Proxy patterns and dynamic dispatch are not resolved.

## License

MIT — see [LICENSE](LICENSE).

## Acknowledgments

- [evmole](https://github.com/cdump/evmole) for bytecode static analysis
- openchain.xyz and 4byte.directory for signature data
- Etherscan and Sourcify for verified contract data
