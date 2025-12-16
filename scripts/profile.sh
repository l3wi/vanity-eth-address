#!/bin/bash

# CUDA Profiling Script using NVIDIA Nsight Compute
# Usage: ./scripts/profile.sh [optimization_name] [metric_set]
# Example: ./scripts/profile.sh baseline memory
#          ./scripts/profile.sh warp_atomics atomics

set -e

# Configuration
DEVICE=0
WORK_SCALE=15
OUTPUT_DIR="profile_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Colors
BLUE='\033[0;34m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Parse arguments
OPTIMIZATION=${1:-baseline}
METRIC_SET=${2:-summary}

# Create output directory
mkdir -p "$OUTPUT_DIR"
OUTPUT_FILE="$OUTPUT_DIR/${OPTIMIZATION}_${METRIC_SET}_${TIMESTAMP}"

# Check for ncu
if ! command -v ncu &> /dev/null; then
    echo "Error: NVIDIA Nsight Compute (ncu) not found"
    echo "Install from: https://developer.nvidia.com/nsight-compute"
    exit 1
fi

# Check if binary exists
BINARY="./vanity-eth-address"
if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found at $BINARY"
    exit 1
fi

echo -e "${BLUE}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  CUDA Profiling with Nsight Compute                        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo "  Optimization: $OPTIMIZATION"
echo "  Metric Set: $METRIC_SET"
echo "  Device: $DEVICE"
echo "  Output: $OUTPUT_FILE.*"
echo ""

# Define metric sets
case $METRIC_SET in
    "memory")
        METRICS="dram__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.per_second,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum.per_second,\
lts__t_sectors_op_read.sum,\
lts__t_sectors_op_write.sum"
        echo -e "${GREEN}Profiling memory bandwidth and throughput...${NC}"
        ;;

    "bank_conflicts")
        METRICS="l1tex__data_bank_reads_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_writes_pipe_lsu_mem_shared_op_st.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum"
        echo -e "${GREEN}Profiling shared memory bank conflicts...${NC}"
        ;;

    "occupancy")
        METRICS="sm__warps_active.avg.pct_of_peak,\
sm__maximum_warps_per_active_cycle_pct,\
sm__threads_launched.sum,\
achieved_occupancy"
        echo -e "${GREEN}Profiling occupancy and SM utilization...${NC}"
        ;;

    "atomics")
        METRICS="smsp__inst_executed_op_generic_atom_dot_add.sum,\
smsp__inst_executed_op_generic_atom_dot_max.sum,\
atomic_throughput"
        echo -e "${GREEN}Profiling atomic operations...${NC}"
        ;;

    "compute")
        METRICS="smsp__sass_thread_inst_executed_op_dadd_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dmul_pred_on.sum,\
smsp__sass_thread_inst_executed_op_dfma_pred_on.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"
        echo -e "${GREEN}Profiling compute throughput...${NC}"
        ;;

    "summary")
        METRICS=""
        echo -e "${GREEN}Running full summary profile...${NC}"
        ;;

    *)
        echo "Unknown metric set: $METRIC_SET"
        echo "Available sets: memory, bank_conflicts, occupancy, atomics, compute, summary"
        exit 1
        ;;
esac

# Build ncu command
NCU_CMD="ncu --target-processes all"

if [ -n "$METRICS" ]; then
    NCU_CMD="$NCU_CMD --metrics $METRICS"
fi

NCU_CMD="$NCU_CMD --csv --log-file ${OUTPUT_FILE}.csv"
NCU_CMD="$NCU_CMD $BINARY --leading-zeros --device $DEVICE --work-scale $WORK_SCALE"

echo ""
echo -e "${YELLOW}Running profiler (this may take a while)...${NC}"
echo ""

# Run profiling (limit to first few kernel launches to avoid excessive runtime)
echo "Running profiler (will timeout after 20s)..."
timeout --signal=SIGINT 20s $NCU_CMD 2>&1 | tee "${OUTPUT_FILE}.log" || exit_code=$?
if [ "${exit_code:-0}" -ne 0 ] && [ "${exit_code:-0}" -ne 124 ]; then
    echo -e "${YELLOW}Warning: Profiler exited with code ${exit_code}${NC}"
fi

echo ""
if [ -f "${OUTPUT_FILE}.csv" ]; then
    echo -e "${GREEN}Profiling complete!${NC}"
    echo ""
    echo "Results:"
    echo "  CSV: ${OUTPUT_FILE}.csv"
    echo "  Log: ${OUTPUT_FILE}.log"
    echo ""

    # Display summary
    echo -e "${BLUE}Key Metrics Summary:${NC}"

    # Parse and display key metrics from CSV
    if [ "$METRIC_SET" == "memory" ]; then
        echo "  Memory Throughput:"
        grep -E "dram__throughput|l1tex__t_bytes" "${OUTPUT_FILE}.csv" | head -5 || true
    elif [ "$METRIC_SET" == "bank_conflicts" ]; then
        echo "  Bank Conflicts:"
        grep -E "bank_conflicts" "${OUTPUT_FILE}.csv" | head -5 || true
    elif [ "$METRIC_SET" == "occupancy" ]; then
        echo "  Occupancy:"
        grep -E "warps_active|occupancy" "${OUTPUT_FILE}.csv" | head -5 || true
    elif [ "$METRIC_SET" == "atomics" ]; then
        echo "  Atomic Operations:"
        grep -E "atom" "${OUTPUT_FILE}.csv" | head -5 || true
    fi
else
    echo -e "${YELLOW}Warning: Profiling may have failed or was interrupted${NC}"
fi

echo ""
echo -e "${GREEN}Profile analysis complete!${NC}"
