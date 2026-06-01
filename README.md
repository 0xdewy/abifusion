# ABI Reconstructor

A machine learning system for reconstructing contract ABIs (Application Binary Interfaces) from EVM bytecode using pure ML approaches.

## Overview

The ABI Reconstructor uses a two-model system:
1. **Function Name Classifier** - Predicts function names from bytecode and selector context
2. **Parameter Prediction Model** - Predicts function parameters (types and counts)

The system can reconstruct complete ABIs from raw bytecode without relying on traditional decompilation or symbolic execution.

## Features

- **Pure ML Approach**: No traditional decompilation or symbolic execution
- **Two-Model System**: Separate models for function names and parameters
- **Signature Disambiguation**: Resolves ambiguous function signatures using ML
- **Batch Processing**: Process multiple contracts efficiently
- **API Integration**: Fetches real contract data from Etherscan and Sourcify
- **Training Pipeline**: Complete pipeline for training models on contract data

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
# Reconstruct ABI from bytecode
abi-reconstruct reconstruct --bytecode "0x6080..."

# Reconstruct with specific selector
abi-reconstruct reconstruct --bytecode "0x6080..." --selector "a9059cbb"

# Batch reconstruction from file
abi-reconstruct batch --input contracts.json --output results.json

# Train models
abi-reconstruct train --run --max-samples 10000

# Run tests
abi-reconstruct test
```

### Python API

```python
from abi_reconstructor.models.abi_reconstructor_pipeline import ABIReconstructorPipeline

# Initialize pipeline
pipeline = ABIReconstructorPipeline()

# Reconstruct ABI from bytecode
result = pipeline.reconstruct_abi(
    bytecode="0x6080...",
    selector="a9059cbb"  # Optional
)

# Reconstruct complete ABI (extract all selectors)
result = pipeline.reconstruct_complete_abi(
    bytecode="0x6080...",
    max_selectors=20
)
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

- **Accuracy**: >80% on function name classification
- **Parameter Prediction**: >70% exact match on parameter types
- **Inference Time**: <1 second per contract (CPU)
- **Training Time**: ~2 hours on GPU for 10k samples

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