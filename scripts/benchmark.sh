#!/bin/bash

# Vanity-ETH-Address Performance Benchmark Script
# Usage: ./scripts/benchmark.sh [optimization_name]
# Example: ./scripts/benchmark.sh baseline
#          ./scripts/benchmark.sh warp_atomics

set -e

# Configuration
DURATION=20              # Total test duration in seconds
WARMUP=5                # Warmup period to ignore
DEVICE=0                # GPU device ID
WORK_SCALE=15           # Default work scale
OUTPUT_DIR="benchmark_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Parse arguments
OPTIMIZATION=${1:-baseline}
CUSTOM_DURATION=${2:-$DURATION}

# Create output directory
mkdir -p "$OUTPUT_DIR"
RESULT_FILE="$OUTPUT_DIR/${OPTIMIZATION}_${TIMESTAMP}.json"

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  Vanity-ETH-Address Performance Benchmark                  ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Optimization: $OPTIMIZATION"
echo "  Duration: ${CUSTOM_DURATION}s (${WARMUP}s warmup)"
echo "  Device: $DEVICE"
echo "  Work Scale: $WORK_SCALE"
echo "  Output: $RESULT_FILE"
echo ""

# Check if binary exists
BINARY="./vanity-eth-address"
if [ ! -f "$BINARY" ]; then
    echo -e "${RED}Error: Binary not found at $BINARY${NC}"
    echo "Please compile first with: make"
    exit 1
fi

# Get GPU info
echo -e "${BLUE}GPU Information:${NC}"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader -i $DEVICE
echo ""

# Prepare test command
TEST_CMD="$BINARY --leading-zeros --device $DEVICE --work-scale $WORK_SCALE"

echo -e "${GREEN}Starting benchmark...${NC}"
echo ""

# Run the benchmark and capture output
TEMP_OUTPUT=$(mktemp)
echo "Running for ${CUSTOM_DURATION} seconds..."
echo ""

# Use timeout to kill after duration, ignore exit code 124 (timeout)
timeout --signal=SIGINT ${CUSTOM_DURATION}s $TEST_CMD 2>&1 | tee "$TEMP_OUTPUT" || exit_code=$?
if [ "${exit_code:-0}" -ne 0 ] && [ "${exit_code:-0}" -ne 124 ]; then
    echo -e "${RED}Error: Unexpected exit code ${exit_code}${NC}"
fi

echo ""
echo -e "${BLUE}Processing results...${NC}"

# Extract metrics from output
# Parse speed measurements (skip first WARMUP seconds worth of data)
SPEEDS=$(grep -oP '\d+\.\d+\s+MKeys/sec' "$TEMP_OUTPUT" | awk '{print $1}' | tail -n +$((WARMUP/2)))

if [ -z "$SPEEDS" ]; then
    echo -e "${RED}Error: No speed data found in output${NC}"
    rm "$TEMP_OUTPUT"
    exit 1
fi

# Calculate statistics
AVG_SPEED=$(echo "$SPEEDS" | awk '{sum+=$1; n++} END {if (n>0) print sum/n; else print 0}')
MIN_SPEED=$(echo "$SPEEDS" | sort -n | head -1)
MAX_SPEED=$(echo "$SPEEDS" | sort -n | tail -1)
MEDIAN_SPEED=$(echo "$SPEEDS" | sort -n | awk '{arr[NR]=$1} END {if (NR%2==1) print arr[(NR+1)/2]; else print (arr[NR/2]+arr[NR/2+1])/2}')

# Calculate standard deviation
STDDEV=$(echo "$SPEEDS" | awk -v avg="$AVG_SPEED" '{sum+=($1-avg)^2; n++} END {if (n>0) print sqrt(sum/n); else print 0}')

# Count samples
SAMPLE_COUNT=$(echo "$SPEEDS" | wc -l)

# Create JSON result
cat > "$RESULT_FILE" << EOF
{
  "optimization": "$OPTIMIZATION",
  "timestamp": "$TIMESTAMP",
  "duration": $CUSTOM_DURATION,
  "warmup": $WARMUP,
  "device": $DEVICE,
  "work_scale": $WORK_SCALE,
  "metrics": {
    "avg_speed_mkeys_per_sec": $AVG_SPEED,
    "min_speed_mkeys_per_sec": $MIN_SPEED,
    "max_speed_mkeys_per_sec": $MAX_SPEED,
    "median_speed_mkeys_per_sec": $MEDIAN_SPEED,
    "stddev_mkeys_per_sec": $STDDEV,
    "sample_count": $SAMPLE_COUNT
  },
  "gpu_info": "$(nvidia-smi --query-gpu=name --format=csv,noheader -i $DEVICE | tr -d '\n')"
}
EOF

# Display results
echo ""
echo -e "${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║  Benchmark Results: $OPTIMIZATION${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
printf "  %-25s %10.2f MKeys/sec\n" "Average Speed:" $AVG_SPEED
printf "  %-25s %10.2f MKeys/sec\n" "Median Speed:" $MEDIAN_SPEED
printf "  %-25s %10.2f MKeys/sec\n" "Min Speed:" $MIN_SPEED
printf "  %-25s %10.2f MKeys/sec\n" "Max Speed:" $MAX_SPEED
printf "  %-25s %10.2f MKeys/sec\n" "Std Deviation:" $STDDEV
printf "  %-25s %10d samples\n" "Sample Count:" $SAMPLE_COUNT
echo ""
echo -e "${BLUE}Results saved to: $RESULT_FILE${NC}"

# Cleanup
rm "$TEMP_OUTPUT"

# Compare with baseline if not baseline
if [ "$OPTIMIZATION" != "baseline" ] && [ -f "$OUTPUT_DIR/baseline_latest.json" ]; then
    echo ""
    echo -e "${YELLOW}Comparing with baseline...${NC}"

    BASELINE_SPEED=$(jq -r '.metrics.avg_speed_mkeys_per_sec' "$OUTPUT_DIR/baseline_latest.json")
    IMPROVEMENT=$(echo "scale=2; ($AVG_SPEED - $BASELINE_SPEED) / $BASELINE_SPEED * 100" | bc)

    if (( $(echo "$IMPROVEMENT > 0" | bc -l) )); then
        echo -e "  ${GREEN}+${IMPROVEMENT}% improvement over baseline${NC}"
    elif (( $(echo "$IMPROVEMENT < 0" | bc -l) )); then
        echo -e "  ${RED}${IMPROVEMENT}% regression from baseline${NC}"
    else
        echo -e "  ${YELLOW}No significant change from baseline${NC}"
    fi
fi

# Save as latest for this optimization
cp "$RESULT_FILE" "$OUTPUT_DIR/${OPTIMIZATION}_latest.json"

echo ""
echo -e "${GREEN}Benchmark complete!${NC}"
