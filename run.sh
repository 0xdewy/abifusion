#!/bin/bash
# ABI Reconstructor - Run Script
# Simplified interface for running ABI reconstruction with uv

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "=========================================="
echo "ABI Reconstructor (uv)"
echo "=========================================="

# Default values
CHECKPOINT_DIR="${CHECKPOINT_DIR:-cache/checkpoints}"
USE_CUDA="${USE_CUDA:-true}"
VERBOSE="${VERBOSE:-false}"

# Function to show usage
show_usage() {
    cat << EOF
Usage: ./run.sh COMMAND [OPTIONS]

Commands:
  reconstruct    Reconstruct ABI from bytecode
  batch          Batch reconstruction from file
  train          Train models (alias for ./train.sh)
  test           Run tests
  compare        Compare with evmole (benchmark)
  help           Show this help message

General Options:
  --checkpoint-dir DIR    Checkpoint directory (default: cache/checkpoints)
  --no-cuda               Disable CUDA/GPU
  --verbose, -v           Verbose output
  --help, -h              Show help for command

Environment variables:
  CHECKPOINT_DIR          Override checkpoint directory
  USE_CUDA                Set to 'false' to disable CUDA
  VERBOSE                 Set to 'true' for verbose output

Examples:
  ./run.sh reconstruct --bytecode "6080..."
  ./run.sh reconstruct contract.bin
  ./run.sh batch --input contracts.json --output results.json
  ./run.sh train --quick
  ./run.sh test
  ./run.sh compare --input contracts.json --output comparison.json
EOF
}

# Function to show reconstruct command help
show_reconstruct_help() {
    cat << EOF
Usage: ./run.sh reconstruct [OPTIONS]

Reconstruct ABI from bytecode.

Required:
  --bytecode HEX|FILE     Contract bytecode (hex string or file path)

Options:
  --selector HEX          Function selector (optional)
  --checkpoint-dir DIR    Checkpoint directory (default: cache/checkpoints)
  --output FILE           Output JSON file (default: stdout)
  --no-cuda               Disable CUDA/GPU
  --verbose, -v           Verbose output

Examples:
  ./run.sh reconstruct --bytecode "6080604052348015600f57600080fd5b..."
  ./run.sh reconstruct --bytecode contract.bin
  ./run.sh reconstruct --bytecode contract.bin --selector "a9059cbb" --output result.json
EOF
}

# Function to show batch command help
show_batch_help() {
    cat << EOF
Usage: ./run.sh batch [OPTIONS]

Batch reconstruction from file.

Required:
  --input FILE            Input JSON file with contracts

Options:
  --output FILE           Output JSON file (default: batch_results.json)
  --checkpoint-dir DIR    Checkpoint directory (default: cache/checkpoints)
  --no-cuda               Disable CUDA/GPU

Input file format (JSON):
  [
    {
      "address": "0x...",
      "bytecode": "6080...",
      "name": "ContractName"  # optional
    },
    ...
  ]

Examples:
  ./run.sh batch --input contracts.json
  ./run.sh batch --input contracts.json --output results.json
EOF
}

# Function to show compare command help
show_compare_help() {
    cat << EOF
Usage: ./run.sh compare [OPTIONS]

Compare ABI reconstruction with evmole (benchmark).

Required:
  --input FILE            Input JSON file with contracts (must have ABI for ground truth)

Options:
  --output FILE           Output JSON file (default: comparison_results.json)
  --checkpoint-dir DIR    Checkpoint directory (default: cache/checkpoints)
  --sample N              Sample size (default: all)
  --no-cuda               Disable CUDA/GPU

Input file format (JSON):
  [
    {
      "address": "0x...",
      "bytecode": "6080...",
      "abi": [...],        # Ground truth ABI
      "name": "ContractName"  # optional
    },
    ...
  ]

Examples:
  ./run.sh compare --input contracts_with_abi.json
  ./run.sh compare --input contracts_with_abi.json --output comparison.json --sample 100
EOF
}

# Parse global options first
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --no-cuda)
            USE_CUDA="false"
            shift
            ;;
        --verbose|-v)
            VERBOSE="true"
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            break
            ;;
    esac
done

# Get command
COMMAND="${1:-help}"
shift

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "❌ Error: 'uv' is not installed or not in PATH"
    echo "   Install with: curl -LsSf https://astral.sh/uv/install.sh | sh"
    echo "   Or visit: https://docs.astral.sh/uv/"
    exit 1
fi

# Check if project is installed with uv
if [[ ! -f ".venv/pyvenv.cfg" ]]; then
    echo "⚠️  UV virtual environment not found"
    echo "   Creating with: uv venv"
    uv venv
    echo "   Installing dependencies..."
    uv pip install -e ".[dev]"
fi

# Build common flags
COMMON_FLAGS=""
if [[ "$USE_CUDA" == "false" ]]; then
    COMMON_FLAGS="$COMMON_FLAGS --no-cuda"
fi
if [[ "$VERBOSE" == "true" ]]; then
    COMMON_FLAGS="$COMMON_FLAGS --verbose"
fi

# Handle commands
case "$COMMAND" in
    reconstruct)
        # Parse reconstruct options
        BYTECODE=""
        SELECTOR=""
        OUTPUT=""
        
        # Check if first argument is a file (not an option)
        if [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; then
            # First argument is not an option, assume it's a file path
            BYTECODE="$1"
            shift
        fi
        
        while [[ $# -gt 0 ]]; do
            case $1 in
                --bytecode)
                    BYTECODE="$2"
                    shift 2
                    ;;
                --selector)
                    SELECTOR="$2"
                    shift 2
                    ;;
                --output)
                    OUTPUT="$2"
                    shift 2
                    ;;
                --help|-h)
                    show_reconstruct_help
                    exit 0
                    ;;
                --verbose|-v)
                    VERBOSE="true"
                    shift
                    ;;
                *)
                    echo "Unknown option: $1"
                    show_reconstruct_help
                    exit 1
                    ;;
            esac
        done
        
        if [[ -z "$BYTECODE" ]]; then
            echo "Error: bytecode is required (hex string or file path)"
            show_reconstruct_help
            exit 1
        fi
        
        echo "Reconstructing ABI from bytecode..."
        echo "Bytecode length: ${#BYTECODE} characters"
        if [[ -n "$SELECTOR" ]]; then
            echo "Selector: $SELECTOR"
        fi
        echo ""
        
        cd "$PROJECT_ROOT"
        uv run -m abi_reconstructor.cli reconstruct \
            --bytecode "$BYTECODE" \
            ${SELECTOR:+--selector "$SELECTOR"} \
            --pipeline "$CHECKPOINT_DIR" \
            ${OUTPUT:+--output "$OUTPUT"} \
            $([[ "$VERBOSE" == "true" ]] && echo "--verbose") \
            $COMMON_FLAGS
        
        # Show evmole comparison if verbose
        if [[ "$VERBOSE" == "true" ]]; then
            echo ""
            echo "=== EVMole Comparison ==="
            uv run python3 -c "
from abi_reconstructor.utils.evmole_comparison import extract_selectors_with_evmole

bytecode = '''$BYTECODE'''
if bytecode.startswith('0x'):
    bytecode = bytecode[2:]

# Extract selectors with evmole
evmole_selectors = extract_selectors_with_evmole(bytecode)
print(f'EVMole found {len(evmole_selectors)} selectors: {evmole_selectors}')

if evmole_selectors:
    print('\\nNote: Compare with our results above. For detailed comparison, use:')
    print('  ./run.sh compare --bytecode \"YOUR_BYTECODE\"')
else:
    print('No selectors found by evmole')
"
        fi
        ;;
    
    batch)
        # Parse batch options
        INPUT=""
        OUTPUT=""
        
        while [[ $# -gt 0 ]]; do
            case $1 in
                --input)
                    INPUT="$2"
                    shift 2
                    ;;
                --output)
                    OUTPUT="$2"
                    shift 2
                    ;;
                --help|-h)
                    show_batch_help
                    exit 0
                    ;;
                *)
                    echo "Unknown option: $1"
                    show_batch_help
                    exit 1
                    ;;
            esac
        done
        
        if [[ -z "$INPUT" ]]; then
            echo "Error: --input is required"
            show_batch_help
            exit 1
        fi
        
        if [[ ! -f "$INPUT" ]]; then
            echo "Error: Input file not found: $INPUT"
            exit 1
        fi
        
        OUTPUT="${OUTPUT:-batch_results.json}"
        
        echo "Running batch reconstruction..."
        echo "Input file: $INPUT"
        echo "Output file: $OUTPUT"
        echo "Contract count: $(jq '. | length' "$INPUT" 2>/dev/null || echo "Unknown")"
        echo ""
        
        cd "$PROJECT_ROOT"
        uv run -m abi_reconstructor.cli batch \
            --input "$INPUT" \
            --output "$OUTPUT" \
            --pipeline "$CHECKPOINT_DIR" \
            $COMMON_FLAGS
        ;;
    
    train)
        # Delegate to train.sh
        echo "Running training..."
        echo ""
        cd "$PROJECT_ROOT"
        ./train.sh "$@"
        ;;
    
    test)
        echo "Running tests..."
        echo ""
        cd "$PROJECT_ROOT"
        make test
        ;;
    
    compare)
        # Parse compare options
        INPUT=""
        OUTPUT=""
        SAMPLE=""
        
        while [[ $# -gt 0 ]]; do
            case $1 in
                --input)
                    INPUT="$2"
                    shift 2
                    ;;
                --output)
                    OUTPUT="$2"
                    shift 2
                    ;;
                --sample)
                    SAMPLE="$2"
                    shift 2
                    ;;
                --help|-h)
                    show_compare_help
                    exit 0
                    ;;
                *)
                    echo "Unknown option: $1"
                    show_compare_help
                    exit 1
                    ;;
            esac
        done
        
        if [[ -z "$INPUT" ]]; then
            echo "Error: --input is required"
            show_compare_help
            exit 1
        fi
        
        if [[ ! -f "$INPUT" ]]; then
            echo "Error: Input file not found: $INPUT"
            exit 1
        fi
        
        OUTPUT="${OUTPUT:-comparison_results.json}"
        
        echo "Running comparison with evmole..."
        echo "Input file: $INPUT"
        echo "Output file: $OUTPUT"
        echo "Contract count: $(jq '. | length' "$INPUT" 2>/dev/null || echo "Unknown")"
        if [[ -n "$SAMPLE" ]]; then
            echo "Sample size: $SAMPLE"
        fi
        echo ""
        
        cd "$PROJECT_ROOT"
        uv run -m abi_reconstructor.cli compare \
            --input "$INPUT" \
            --output "$OUTPUT" \
            --pipeline "$CHECKPOINT_DIR" \
            ${SAMPLE:+--sample "$SAMPLE"} \
            $COMMON_FLAGS
        ;;
    
    help)
        show_usage
        ;;
    
    *)
        echo "Unknown command: $COMMAND"
        echo ""
        show_usage
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "Done!"
echo "=========================================="