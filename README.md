# ABI Reconstructor

Reconstruct contract ABIs (Application Binary Interfaces) from EVM bytecode by
**fusing multiple imperfect signal sources** — signature databases (openchain,
4byte) and evmole's static analysis — so the result beats any single tool.

## Overview

No single tool recovers parameter types perfectly:

- **Signature databases** (openchain.xyz, 4byte.directory) know the *exact* types
  for a selector's real signature (e.g. `bytes32` vs `uint256`), but 4byte is
  full of spam collisions and neither covers every selector.
- **evmole** recovers each function's argument *structure* by static analysis
  (no spam), but can't always distinguish `bytes32`/`uint256` or `address`/
  `uint160`, and miscounts some arguments.

The **`FusionReconstructor`** uses evmole's structure to pick the correct
signature-database candidate, and the signature to supply the exact types evmole
can't infer. On 6,744 functions with ground-truth ABIs it reaches **96.4% exact
parameter-type accuracy vs evmole's 90.0%** — strictly better, zero regressions
(see `eval_output/fusion_eval.md`).

## Features

- **Fusion reconstruction**: openchain/4byte signatures disambiguated by evmole
  (~96% exact parameter types). This is the recommended path.
- **Rule-based reconstruction**: offline 4byte + bytecode heuristics, no network.
- **Selector extraction** from bytecode.
- **Signature database** (local 4byte SQLite + openchain/4byte API, cached).
- **Data + evaluation tooling**: fetch verified contracts (Etherscan/Sourcify),
  benchmark against ground truth (`scripts/eval/`).

> Note: an experimental pure-ML path (function-name classifier + parameter
> model) also exists, but it does **not** beat the fusion approach for types and
> is not the recommended reconstructor — see `eval_output/` for the analysis.

## Installation

### Prerequisites
- Python 3.8+
- UV package manager (recommended) or pip
- Git

### Quick Start

```bash
# Clone the repository
git clone <repository-url>
cd abi_reconstructor

# Install with UV (recommended)
uv pip install -e .

# Or install with pip
pip install -e .

# Install development dependencies
uv pip install -e ".[dev]"
```

### Environment Setup

Create a `.env` file based on `.env-example`:

```bash
cp .env-example .env
# Edit .env to add your API keys
```

Required environment variables:
- `ETH_RPC_URL`: Ethereum RPC endpoint
- `ETHERSCAN_API_KEY`: Etherscan API key (optional, for data fetching)
- `SOURCIFY_API_KEY`: Sourcify API key (optional)

## Usage

### Command Line Interface

```bash
# Recommended: fusion reconstruction (~96% exact parameter types; needs network)
abi-reconstruct fusion --bytecode "0x6080..."
abi-reconstruct fusion --bytecode-file contract.hex

# Offline rule-based reconstruction (4byte + heuristics)
abi-reconstruct reconstruct --bytecode "0x6080..."

# Extract function selectors only
abi-reconstruct extract --bytecode "0x6080..."

# Local 4byte signature database
abi-reconstruct populate            # fill from 4byte.directory
abi-reconstruct stats               # show DB stats
abi-reconstruct export --output sigs.json
```

### Python API

```python
from abi_reconstructor.fusion import FusionReconstructor

result = FusionReconstructor().reconstruct("0x6080...")
for fn in result["functions"]:
    types = ",".join(i["type"] for i in fn["inputs"])
    print(f"{fn['name']}  selector={fn['selector']}  source={fn['source']}")
# result["functions"]: [{type, name, selector, inputs:[{type,name}], source}, ...]
```

### Training Models

```bash
# Train both models
python scripts/training/train.py both --max-samples 10000

# Train function classifier only
python scripts/training/train.py function --max-samples 5000

# Train parameter model only
python scripts/training/train.py parameter --max-samples 5000

# Quick test with small dataset
python scripts/training/train.py test
```

## Project Structure

```
abi_reconstructor/          # Main package
├── data/                   # Data fetching (etherscan, sourcify)
├── features/                # Feature extraction
│   ├── bytecode_features.py
│   └── selector_extractor.py
├── models/                 # ML models
│   ├── bytecode_transformer.py
│   ├── function_name_classifier.py
│   ├── parameter_prediction_model.py
│   └── abi_reconstructor_pipeline.py
├── training/               # Training datasets
├── utils/                  # Utilities
│   ├── bytecode_utils.py
│   └── signature_lookup.py
├── cli.py                  # CLI entry point
├── database.py             # 4byte SQLite database
├── reconstructor.py         # ABI reconstruction logic
└── __init__.py

checkpoints/               # Trained model weights (.pth files)
scripts/                   # Build and utility scripts
tests/                    # Test suite
```
abi_reconstructor/
├── data/              # Data fetching and management
│   ├── etherscan_fetcher.py
│   ├── sourcify_client.py
│   └── dataset_builder.py
├── features/          # Feature extraction
│   ├── bytecode_features.py
│   └── selector_extractor.py
├── models/           # ML models
│   ├── function_name_classifier.py
│   ├── parameter_prediction_model.py
│   ├── bytecode_transformer.py
│   └── abi_reconstructor_pipeline.py
├── training/         # Training scripts and datasets
│   ├── train_ml_models.py
│   ├── parameter_dataset.py
│   └── bytecode_dataset.py
├── utils/            # Utilities
│   ├── bytecode_utils.py
│   └── signature_lookup.py
├── cli.py            # Command line interface
└── __init__.py

scripts/
├── data/            # Data processing scripts
├── training/        # Training scripts
├── debug/           # Debug and exploration scripts
└── shell/           # Shell scripts

tests/
├── data/           # Tests for data modules
├── features/       # Tests for feature extraction
├── models/         # Tests for ML models
├── integration/    # Integration tests
├── utils/          # Tests for utilities
└── validation/     # Validation tests
```

## Data Pipeline

1. **Data Collection**: Fetch verified contracts from Etherscan and Sourcify
2. **Feature Extraction**: Extract bytecode features and function selectors
3. **Dataset Creation**: Create training datasets with (bytecode, selector) → signature mappings
4. **Model Training**: Train the two ML models
5. **Inference**: Use trained models to reconstruct ABIs from new bytecode

### Data Sources

- **Etherscan**: Verified contract source code and ABIs
- **Sourcify**: Fully verified contracts with metadata
- **4byte.directory**: Function signature database

## Development

### Setting Up Development Environment

```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=abi_reconstructor --cov-report=html

# Linting
ruff check abi_reconstructor/

# Format code
black abi_reconstructor/

# Type checking
mypy abi_reconstructor/
```

### Code Style

- Follow PEP 8
- Use type hints for all function signatures
- Add docstrings to public functions/classes
- Write descriptive commit messages

### Testing

```bash
# Run all tests
pytest tests/

# Run specific test module
pytest tests/models/test_function_name_classifier.py -v

# Run with coverage report
pytest tests/ --cov=abi_reconstructor --cov-report=html
```

## Model Architecture

### Function Name Classifier
- Input: Bytecode features + selector context
- Architecture: Transformer or CNN encoder + classification head
- Output: Function name probabilities

### Parameter Prediction Model
- Input: Bytecode features + (optional) function name
- Architecture: Transformer or CNN encoder + multi-task head
- Outputs: Parameter count, parameter types, parameter mask

## Performance

Measured on 6,744 functions from 500 verified mainnet contracts
(`eval_output/fusion_eval.md`):

- **Fusion reconstructor**: **96.4%** exact parameter-type accuracy
- **evmole baseline**: 90.0%
- The fusion fixes 291 functions evmole gets wrong (mostly `bytes32`/`uint256`
  and `address`/`uint160`) and breaks none.
- Ceiling is currently capped near 96.5% by signature-DB coverage; the residual
  is selectors absent from openchain/4byte where evmole is the only signal.

The experimental pure-ML parameter model reaches only ~32% exact type accuracy
and is **not** recommended (`eval_output/plan05_fullscale.md`).

## Limitations

1. **Bytecode Length**: Models trained on contracts up to 24KB
2. **Function Complexity**: Best results on standard ERC20/ERC721 functions
3. **Novel Patterns**: May struggle with highly novel or obfuscated code
4. **External Calls**: Cannot resolve dynamic dispatch or proxy patterns

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

### Contribution Guidelines
- Write tests for new features
- Update documentation
- Follow existing code style
- Add type hints and docstrings

## License

MIT License - see LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{abi_reconstructor,
  title = {ABI Reconstructor: ML-based ABI Reconstruction from EVM Bytecode},
  author = {RL Attacker Team},
  year = {2024},
  url = {https://github.com/yourusername/abi_reconstructor}
}
```

## Support

- **Issues**: Report bugs or feature requests on GitHub Issues
- **Discussions**: Join discussions on GitHub Discussions
- **Contributing**: See CONTRIBUTING.md for development guidelines

## Acknowledgments

- Ethereum Foundation for the EVM
- Etherscan and Sourcify for contract data
- 4byte.directory for signature database
- PyTorch team for the deep learning framework