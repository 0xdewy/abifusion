#!/bin/bash
# ABI Reconstructor - Clean Script
# Clean up build artifacts and old virtual environments

set -e  # Exit on error

echo "=========================================="
echo "ABI Reconstructor - Cleanup"
echo "=========================================="

echo "Cleaning build artifacts..."
make clean
echo "✅ Build artifacts cleaned"

echo ""
echo "Cleaning Python cache files..."
find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete
find . -type f -name ".coverage" -delete
echo "✅ Python cache cleaned"

echo ""
echo "Cleaning test artifacts..."
rm -rf .pytest_cache htmlcov .coverage test_output 2>/dev/null || true
echo "✅ Test artifacts cleaned"

echo ""
echo "Checking for old virtual environments..."
if [[ -d "venv" ]]; then
    echo "⚠️  Found old 'venv' directory (from pip)"
    read -p "   Remove it? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        echo "✅ Old venv removed"
    else
        echo "⚠️  Keeping old venv"
    fi
fi

if [[ -d ".venv" ]]; then
    echo "✅ UV virtual environment (.venv) is present"
    echo "   To remove: rm -rf .venv"
    echo "   To recreate: uv venv"
fi

echo ""
echo "Checking cache directories..."
if [[ -d "cache" ]]; then
    echo "Cache directory size: $(du -sh cache | cut -f1)"
    read -p "   Clear cache? [y/N] " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf cache/*
        mkdir -p cache/checkpoints cache/logs
        echo "✅ Cache cleared"
    fi
fi

echo ""
echo "=========================================="
echo "Cleanup complete! 🧹"
echo "=========================================="
echo ""
echo "To start fresh:"
echo "1. Remove UV venv: rm -rf .venv"
echo "2. Run setup: ./setup.sh"
echo ""
echo "Current disk usage:"
du -sh . --exclude=.venv 2>/dev/null || du -sh .
echo "=========================================="