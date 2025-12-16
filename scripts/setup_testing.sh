#!/bin/bash

# Setup testing environment for performance optimization testing
# This script prepares all testing tools and creates necessary directories

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}Setting up testing environment...${NC}"
echo ""

# Make scripts executable
echo "Making scripts executable..."
chmod +x scripts/*.sh
chmod +x scripts/*.py

# Install jq if not present (needed for JSON parsing)
if ! command -v jq &> /dev/null; then
    echo "Installing jq for JSON processing..."
    if [[ "$OSTYPE" == "darwin"* ]]; then
        # macOS
        if command -v brew &> /dev/null; then
            brew install jq
        else
            echo "Please install jq manually: https://stedolan.github.io/jq/download/"
        fi
    elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Linux
        if command -v apt-get &> /dev/null; then
            sudo apt-get install -y jq
        elif command -v yum &> /dev/null; then
            sudo yum install -y jq
        else
            echo "Please install jq manually: https://stedolan.github.io/jq/download/"
        fi
    fi
fi

# Check for required tools
echo ""
echo "Checking required tools..."

check_tool() {
    if command -v $1 &> /dev/null; then
        echo -e "  ${GREEN}✓${NC} $1"
        return 0
    else
        echo -e "  ✗ $1 (optional)"
        return 1
    fi
}

check_tool "nvcc"
check_tool "ncu"
check_tool "nvidia-smi"
check_tool "python3"
check_tool "jq"
check_tool "bc"

# Create directories
echo ""
echo "Creating directories..."
mkdir -p benchmark_results
mkdir -p profile_results
mkdir -p docs

echo -e "  ${GREEN}✓${NC} benchmark_results/"
echo -e "  ${GREEN}✓${NC} profile_results/"
echo -e "  ${GREEN}✓${NC} docs/"

# Build baseline
echo ""
echo "Building baseline binary..."
make clean > /dev/null 2>&1 || true
if make -j$(nproc) > /dev/null 2>&1; then
    echo -e "  ${GREEN}✓${NC} Baseline build successful"
else
    echo "  ✗ Build failed - please check compiler errors"
    exit 1
fi

echo ""
echo -e "${GREEN}Setup complete!${NC}"
echo ""
echo "Testing tools ready:"
echo "  • benchmark.sh           - Single optimization benchmark"
echo "  • profile.sh             - CUDA profiling with ncu"
echo "  • compare_results.py     - Compare benchmark results"
echo "  • test_all_optimizations.sh - Comprehensive test suite"
echo ""
echo "Quick start:"
echo "  1. Baseline:  ./scripts/benchmark.sh baseline 60"
echo "  2. Test all:  ./scripts/test_all_optimizations.sh"
echo "  3. Compare:   python3 scripts/compare_results.py benchmark_results/*.json"
echo ""
echo "See docs/testing-guide.md for detailed instructions"
