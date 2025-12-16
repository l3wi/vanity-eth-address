#!/bin/bash

# Test All Optimizations Separately
# This script builds and benchmarks each optimization independently

set -e

# Configuration
DURATION=20
DEVICE=0
WORK_SCALE=15
RESULTS_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY_FILE="${RESULTS_DIR}/summary_${TIMESTAMP}.txt"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

# Optimization list
declare -a OPTIMIZATIONS=(
    "baseline::"
    "warp_atomics:OPT_WARP_ATOMICS:"
    "bank_conflict_padding:OPT_BANK_CONFLICT_PADDING:"
    "vectorized_scoring:OPT_VECTORIZED_SCORING:"
    "improved_occupancy:OPT_IMPROVED_OCCUPANCY:"
    "coalesced_output:OPT_COALESCED_OUTPUT:"
)

# Create results directory
mkdir -p "$RESULTS_DIR"

# Print header
echo -e "${CYAN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Vanity-ETH-Address: Comprehensive Optimization Testing       ║${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Configuration:${NC}"
echo "  Duration per test: ${DURATION}s"
echo "  Device: $DEVICE"
echo "  Work Scale: $WORK_SCALE"
echo "  Results: $RESULTS_DIR"
echo "  Summary: $SUMMARY_FILE"
echo ""

# Initialize summary file
cat > "$SUMMARY_FILE" << EOF
=================================================================
Vanity-ETH-Address Optimization Test Summary
Generated: $(date)
=================================================================

Configuration:
  Duration: ${DURATION}s per test
  Device: $DEVICE
  Work Scale: $WORK_SCALE
  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i $DEVICE)

=================================================================
RESULTS:
=================================================================

EOF

# Function to build with specific optimization
build_optimization() {
    local name=$1
    local flags=$2

    echo -e "${YELLOW}Building: $name${NC}"

    if [ -z "$flags" ]; then
        # Baseline build
        make clean > /dev/null 2>&1
        make -j$(nproc) > /dev/null 2>&1
    else
        # Build with optimization flags
        make clean > /dev/null 2>&1
        make -j$(nproc) OPTS="-D${flags}" > /dev/null 2>&1
    fi

    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ Build successful${NC}"
        return 0
    else
        echo -e "  ${RED}✗ Build failed${NC}"
        return 1
    fi
}

# Function to run benchmark
run_benchmark() {
    local name=$1

    echo -e "${YELLOW}Running benchmark: $name${NC}"
    ./scripts/benchmark.sh "$name" "$DURATION"

    if [ $? -eq 0 ]; then
        echo -e "  ${GREEN}✓ Benchmark completed${NC}"
        return 0
    else
        echo -e "  ${RED}✗ Benchmark failed${NC}"
        return 1
    fi
}

# Track results for final comparison
declare -a RESULT_FILES=()

# Test each optimization
total=${#OPTIMIZATIONS[@]}
current=0

for opt_config in "${OPTIMIZATIONS[@]}"; do
    current=$((current + 1))
    IFS=':' read -r name flags desc <<< "$opt_config"

    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}Test [$current/$total]: $name${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    # Build
    if ! build_optimization "$name" "$flags"; then
        echo -e "${RED}Skipping benchmark due to build failure${NC}"
        continue
    fi

    # Small delay to ensure GPU is ready
    sleep 2

    # Benchmark
    if run_benchmark "$name"; then
        # Add result file to comparison list
        latest_result="${RESULTS_DIR}/${name}_latest.json"
        if [ -f "$latest_result" ]; then
            RESULT_FILES+=("$latest_result")

            # Extract speed for summary
            avg_speed=$(jq -r '.metrics.avg_speed_mkeys_per_sec' "$latest_result")
            echo "$name: $avg_speed MKeys/sec" >> "$SUMMARY_FILE"
        fi
    fi

    echo ""
done

# Generate comparison report
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}Generating comparison report...${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""

if [ ${#RESULT_FILES[@]} -gt 0 ]; then
    python3 scripts/compare_results.py "${RESULT_FILES[@]}" | tee -a "$SUMMARY_FILE"
else
    echo -e "${RED}No results to compare${NC}"
fi

echo ""
echo -e "${GREEN}╔═══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  All tests complete!                                          ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${BLUE}Summary saved to: $SUMMARY_FILE${NC}"
echo ""

# Generate visualization data
echo -e "${YELLOW}Generating visualization data...${NC}"
python3 - <<EOF
import json
import sys
from pathlib import Path

results = []
for filepath in ${RESULT_FILES[@]}:
    filepath_str = filepath.strip("'\"")
    with open(filepath_str, 'r') as f:
        data = json.load(f)
        results.append({
            'name': data['optimization'],
            'speed': data['metrics']['avg_speed_mkeys_per_sec']
        })

# Sort by speed
results.sort(key=lambda x: x['speed'], reverse=True)

# Create CSV for visualization
csv_file = "${RESULTS_DIR}/comparison_${TIMESTAMP}.csv"
with open(csv_file, 'w') as f:
    f.write("optimization,speed_mkeys_per_sec\n")
    for r in results:
        f.write(f"{r['name']},{r['speed']}\n")

print(f"Visualization data saved to: {csv_file}")
EOF

echo ""
echo -e "${GREEN}Testing complete!${NC}"
echo ""
echo "Next steps:"
echo "  1. Review summary: cat $SUMMARY_FILE"
echo "  2. Compare specific optimizations: python3 scripts/compare_results.py result1.json result2.json"
echo "  3. Profile best performer: ./scripts/profile.sh <optimization_name> memory"
echo ""
