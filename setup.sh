#!/bin/bash
# ABI Reconstructor - Setup Script
# One-command setup with uv

set -e  # Exit on error

echo "=========================================="
echo "ABI Reconstructor - Setup"
echo "=========================================="

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add uv to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    if ! command -v uv &> /dev/null; then
        echo "❌ Failed to install uv"
        echo "   Please install manually: https://docs.astral.sh/uv/"
        exit 1
    fi
    echo "✅ uv installed successfully"
else
    echo "✅ uv is already installed"
fi

echo ""
echo "Setting up project..."

# Create uv virtual environment
if [[ ! -f ".venv/pyvenv.cfg" ]]; then
    echo "Creating virtual environment..."
    uv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Install dependencies
echo "Installing dependencies..."
uv pip install -e ".[dev]"
echo "✅ Dependencies installed"

# Create necessary directories
echo "Creating directories..."
mkdir -p data cache/checkpoints cache/logs
echo "✅ Directories created"

# Copy environment example
if [[ ! -f ".env" ]]; then
    echo "Creating .env file from example..."
    cp .env-example .env
    echo "✅ .env file created"
    echo "   Edit .env to add your API keys if needed"
else
    echo "✅ .env file already exists"
fi

echo ""
echo "=========================================="
echo "Setup complete! 🎉"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Test the installation:"
echo "   ./run.sh test"
echo ""
echo "2. Try a quick training test:"
echo "   ./train.sh --quick"
echo ""
echo "3. Run the quick start guide:"
echo "   ./examples/quick_start.sh"
echo ""
echo "4. Check available commands:"
echo "   ./run.sh --help"
echo "   ./train.sh --help"
echo ""
echo "For more information, see README.md"
echo "=========================================="