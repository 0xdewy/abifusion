---
name: ABI Reconstructor
description: >
  REQUIRED for ANY work on the ABI Reconstructor project.
  Use when writing tests, fetching contracts, training models, or modifying
  any files in the abi_reconstructor codebase. Triggers: testing, contract
  fetching, model training, feature extraction, signature disambiguation.
---

# ABI Reconstructor Skill

Manage the ABI Reconstructor project - a machine learning system for reconstructing
contract ABIs from EVM bytecode.

## Security Rules - CRITICAL

**NEVER READ .env FILES** - They contain sensitive API keys and RPC URLs.
- Ask users explicitly for environment variables if needed
- Validate all external inputs before processing
- Clean up temporary files containing sensitive data

## Testing Requirements

1. **Always run tests after changes**: `pytest tests/`
2. **Maintain >80% code coverage**
3. **Mock all external API calls** in tests (Etherscan, Sourcify, 4byte.directory)
4. **Test edge cases**: empty bytecode, malformed inputs, API failures

## Development Standards

1. **Follow PEP 8** style guide
2. **Use type hints** for all function signatures
3. **Add docstrings** to public functions/classes
4. **Write descriptive commit messages**

## Model Training Protocol

1. **Always split data** into train/val/test sets (70/15/15)
2. **Log training metrics** (loss, accuracy, precision, recall, F1)
3. **Save model checkpoints** regularly
4. **Document hyperparameters** and results

## Data Handling

1. **Cache API responses** to respect rate limits
2. **Validate contract data** before processing
3. **Implement retry logic** for failed operations
4. **Clean cache periodically** to manage disk space

## Project Structure Reference
```
abi_reconstructor/
├── data/           # Data fetching and management
├── models/         # ML models
├── features/       # Feature extraction
├── utils/         # Utilities
├── tests/         # Test files (to be created)
├── training/      # Training scripts
├── inference/     # Inference scripts
└── cache/         # Cache directory
```

## Common Workflows

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-mock pytest-cov

# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=abi_reconstructor --cov-report=html

# Run specific test module
pytest tests/models/test_neural_disambiguator.py -v
```

### Training a Model
```bash
# Full training
python train_disambiguator.py

# Quick test
python train_disambiguator.py --quick

# Test existing model
python train_disambiguator.py --test-only
```

### Using the Reconstructor
```bash
# Run integration example
python integration_example.py
```

## API Key Management

**NEVER commit API keys to version control**
- Store in `.env` file (gitignored)
- Load via environment variables
- Use `python-dotenv` in production

Example `.env` file:
```
ETH_RPC_URL=https://mainnet.infura.io/v3/YOUR_KEY
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_KEY
```

## Error Handling Guidelines

1. **API calls**: Implement retries with exponential backoff
2. **Data validation**: Check bytecode format, ABI structure
3. **Model predictions**: Handle edge cases gracefully
4. **Cache failures**: Fall back to live API calls

## Performance Optimization

1. **Cache aggressively**: API responses, feature extractions
2. **Batch operations**: Fetch multiple contracts at once
3. **Parallel processing**: Use threading for I/O-bound tasks
4. **Memory management**: Clear large objects when done

## Troubleshooting

### Common Issues

1. **API rate limiting**: Implement caching, respect limits
2. **Model not converging**: Check learning rate, data quality
3. **Memory errors**: Process contracts in smaller batches
4. **Cache corruption**: Clear cache directory and restart

### Debug Commands
```bash
# Check environment
python -c "import abi_reconstructor; print('Import OK')"

# Test data fetching
python -c "from abi_reconstructor.data.etherscan_fetcher import EtherscanClient; print('Etherscan client OK')"

# Test feature extraction
python -c "from abi_reconstructor.features.bytecode_features import BytecodeFeatureExtractor; print('Feature extractor OK')"
```

## Skill Triggers

Use this skill when:
- Editing any file in the `abi_reconstructor/` directory
- Writing tests for the project
- Fetching contract data
- Training or evaluating models
- Modifying feature extraction logic
- Working with cache or data pipelines
- Setting up development environment