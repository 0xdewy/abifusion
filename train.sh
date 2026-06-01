#!/bin/bash
# ABI Reconstructor - Training Script
# Simplified interface for training ML models with uv

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

echo "=========================================="
echo "ABI Reconstructor - Training (uv)"
echo "=========================================="

# Default values
DATA_DIR="${DATA_DIR:-data}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-cache/checkpoints}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"  # Default to 0 (all samples) for full training
BATCH_SIZE="${BATCH_SIZE:-8}"  # Reduced from 16 to save memory
LEARNING_RATE="${LEARNING_RATE:-0.001}"
EPOCHS="${EPOCHS:-10}"
USE_CUDA="${USE_CUDA:-true}"
DISABLE_CACHE="${DISABLE_CACHE:-true}"  # Disable caching by default to save memory
SMALL_MODEL="${SMALL_MODEL:-false}"
GRADIENT_CHECKPOINTING="${GRADIENT_CHECKPOINTING:-false}"
AUTO_BATCH_SIZE="${AUTO_BATCH_SIZE:-true}"  # Enable auto batch size by default

# Track if max-samples was specified via command line
MAX_SAMPLES_ARG_SPECIFIED=""
MAX_SAMPLES_ENV_SPECIFIED=""

# Check if MAX_SAMPLES was set via environment variable
if [[ -n "${MAX_SAMPLES+x}" ]] && [[ "$MAX_SAMPLES" != "0" ]]; then
    MAX_SAMPLES_ENV_SPECIFIED="true"
fi

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --data-dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --checkpoint-dir)
            CHECKPOINT_DIR="$2"
            shift 2
            ;;
        --max-samples)
            MAX_SAMPLES="$2"
            MAX_SAMPLES_ARG_SPECIFIED="true"
            shift 2
            ;;
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --learning-rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --no-cuda)
            USE_CUDA="false"
            shift
            ;;
        --disable-cache)
            DISABLE_CACHE="true"
            shift
            ;;
        --enable-cache)
            DISABLE_CACHE="false"
            shift
            ;;
        --small-model)
            SMALL_MODEL="true"
            shift
            ;;
        --gradient-checkpointing)
            GRADIENT_CHECKPOINTING="true"
            shift
            ;;
        --no-auto-batch-size)
            AUTO_BATCH_SIZE="false"
            shift
            ;;
        --quick)
            MAX_SAMPLES=100
            MAX_SAMPLES_ARG_SPECIFIED="true"
            BATCH_SIZE=2  # Further reduced for quick mode
            EPOCHS=2
            DISABLE_CACHE="true"  # Disable cache in quick mode to save memory
            SMALL_MODEL="true"  # Use small model in quick mode
            shift
            ;;
        --help|-h)
            cat << EOF
Usage: ./train.sh [OPTIONS]

Options:
  --data-dir DIR          Data directory (default: data)
  --checkpoint-dir DIR    Checkpoint directory (default: cache/checkpoints)
  --max-samples N         Maximum training samples (default: 0 = all samples)
  --batch-size N          Batch size (default: 8)
  --learning-rate RATE    Learning rate (default: 0.001)
  --epochs N              Number of epochs (default: 10)
  --no-cuda               Disable CUDA/GPU training
  --disable-cache         Disable caching of tokenized bytecode (saves memory, default)
  --enable-cache          Enable caching of tokenized bytecode (faster but uses more memory)
  --small-model           Use smaller model architecture (half size) for memory saving
  --gradient-checkpointing Enable gradient checkpointing (trades compute for memory)
  --no-auto-batch-size    Disable automatic batch size adjustment based on GPU memory
  --quick                 Quick test with minimal settings (small model, batch size 2)
  --help, -h              Show this help message

Environment variables:
  DATA_DIR                Override data directory
  CHECKPOINT_DIR          Override checkpoint directory
  MAX_SAMPLES             Override max samples (0 = all samples)
  BATCH_SIZE              Override batch size
  LEARNING_RATE           Override learning rate
  EPOCHS                  Override epochs
  USE_CUDA                Set to 'false' to disable CUDA
  DISABLE_CACHE           Set to 'true' to disable caching (saves memory, default)
  SMALL_MODEL             Set to 'true' to use smaller model architecture
  GRADIENT_CHECKPOINTING  Set to 'true' to enable gradient checkpointing
  AUTO_BATCH_SIZE         Set to 'false' to disable automatic batch size adjustment

Examples:
  ./train.sh                          # Interactive training (asks for sample count)
  ./train.sh --quick                  # Quick test
  ./train.sh --max-samples 5000 --batch-size 16
  MAX_SAMPLES=2000 ./train.sh        # Using env var (non-interactive)
EOF
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create directories if they don't exist
mkdir -p "$DATA_DIR"
mkdir -p "$CHECKPOINT_DIR"

# Interactive sample selection if not specified via command line or env var
if [[ -z "$MAX_SAMPLES_ARG_SPECIFIED" ]] && [[ -z "$MAX_SAMPLES_ENV_SPECIFIED" ]]; then
    echo ""
    echo "=========================================="
    echo "Sample Selection"
    echo "=========================================="
    echo "Available training samples: ~112,000"
    echo ""
    echo "How many samples would you like to train on?"
    echo "  0) All available samples (~112,000)"
    echo "  1) 50,000 samples (balanced dataset)"
    echo "  2) 10,000 samples (medium)"
    echo "  3) 5,000 samples (default)"
    echo "  4) 1,000 samples (quick)"
    echo "  5) Custom number"
    echo ""
    read -p "Enter choice (0-5): " choice
    
    case $choice in
        0)
            MAX_SAMPLES=0
            echo "Using ALL available samples (~112,000)"
            ;;
        1)
            MAX_SAMPLES=50000
            echo "Using 50,000 samples"
            ;;
        2)
            MAX_SAMPLES=10000
            echo "Using 10,000 samples"
            ;;
        3)
            MAX_SAMPLES=5000
            echo "Using 5,000 samples"
            ;;
        4)
            MAX_SAMPLES=1000
            echo "Using 1,000 samples"
            ;;
        5)
            read -p "Enter custom number of samples: " custom_samples
            if [[ "$custom_samples" =~ ^[0-9]+$ ]]; then
                MAX_SAMPLES=$custom_samples
                echo "Using $custom_samples samples"
            else
                echo "Invalid input. Using default (all samples)"
                MAX_SAMPLES=0
            fi
            ;;
        *)
            echo "Invalid choice. Using all samples"
            MAX_SAMPLES=0
            ;;
    esac
    echo ""
fi

echo "Configuration:"
echo "  Data directory:    $DATA_DIR"
echo "  Checkpoint dir:    $CHECKPOINT_DIR"
echo "  Max samples:       $MAX_SAMPLES"
echo "  Batch size:        $BATCH_SIZE"
echo "  Learning rate:     $LEARNING_RATE"
echo "  Epochs:            $EPOCHS"
echo "  Use CUDA:          $USE_CUDA"
echo "  Disable cache:     $DISABLE_CACHE"
echo "  Small model:       $SMALL_MODEL"
echo "  Gradient checkpointing: $GRADIENT_CHECKPOINTING"
echo "  Auto batch size:   $AUTO_BATCH_SIZE"
echo ""

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
fi

# Build CUDA flag
CUDA_FLAG=""
if [[ "$USE_CUDA" == "false" ]]; then
    CUDA_FLAG="--no-cuda"
fi

# Build cache flag
CACHE_FLAG=""
if [[ "$DISABLE_CACHE" == "true" ]]; then
    CACHE_FLAG="--disable-cache"
fi

# Build small model flag
SMALL_MODEL_FLAG=""
if [[ "$SMALL_MODEL" == "true" ]]; then
    SMALL_MODEL_FLAG="--small-model"
fi

# Build gradient checkpointing flag
GRADIENT_CHECKPOINTING_FLAG=""
if [[ "$GRADIENT_CHECKPOINTING" == "true" ]]; then
    GRADIENT_CHECKPOINTING_FLAG="--gradient-checkpointing"
fi

# Build auto batch size flag
AUTO_BATCH_SIZE_FLAG=""
if [[ "$AUTO_BATCH_SIZE" == "true" ]]; then
    AUTO_BATCH_SIZE_FLAG="--auto-batch-size"
fi

echo "Starting training with uv..."
echo ""

# Run the training with uv
cd "$PROJECT_ROOT"
uv run scripts/training/train.py both \
    --data-dir "$DATA_DIR" \
    --checkpoint-dir "$CHECKPOINT_DIR" \
    --max-samples "$MAX_SAMPLES" \
    --batch-size "$BATCH_SIZE" \
    --learning-rate "$LEARNING_RATE" \
    --epochs "$EPOCHS" \
    $CUDA_FLAG \
    $CACHE_FLAG \
    $SMALL_MODEL_FLAG \
    $GRADIENT_CHECKPOINTING_FLAG \
    $AUTO_BATCH_SIZE_FLAG

echo ""
echo "=========================================="
echo "Training completed!"
echo "Checkpoints saved to: $CHECKPOINT_DIR"
echo "=========================================="