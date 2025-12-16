# Performance Optimization Testing

This document provides a quick reference for testing optimizations independently.

## Setup (One-Time)

```bash
# Make scripts executable and setup environment
chmod +x scripts/setup_testing.sh
./scripts/setup_testing.sh
```

## Quick Testing Workflows

### 1. Baseline Benchmark

```bash
# Run 20-second baseline test
./scripts/benchmark.sh baseline 20
```

Output example:
```
╔════════════════════════════════════════════════════════════╗
║  Benchmark Results: baseline
╚════════════════════════════════════════════════════════════╝

  Average Speed:              3800.45 MKeys/sec
  Median Speed:               3798.20 MKeys/sec
  Min Speed:                  3775.12 MKeys/sec
  Max Speed:                  3825.67 MKeys/sec
  Std Deviation:                 12.34 MKeys/sec
```

### 2. Test Single Optimization

```bash
# Build with warp atomics optimization
make clean
make OPTS=-DOPT_WARP_ATOMICS

# Benchmark it
./scripts/benchmark.sh warp_atomics 60

# Automatic comparison with baseline
# Output shows: +12.34% improvement over baseline
```

### 3. Test All Optimizations (Automated)

```bash
# Runs all optimizations and generates comparison report
./scripts/test_all_optimizations.sh
```

This will:
- Build each optimization separately
- Benchmark each for 20 seconds (~3-4 minutes total)
- Generate comparison table
- Save results to `benchmark_results/summary_*.txt`

### 4. Compare Results

```bash
# Compare all results
python3 scripts/compare_results.py benchmark_results/*_latest.json

# Compare two specific results
python3 scripts/compare_results.py \
    benchmark_results/baseline_latest.json \
    benchmark_results/warp_atomics_latest.json
```

Output example:
```
====================================================================================================
BENCHMARK COMPARISON RESULTS
====================================================================================================

Baseline: baseline @ 3,800.45 MKeys/sec
GPU: NVIDIA GeForce RTX 4090

----------------------------------------------------------------------------------------------------
Optimization              Avg Speed        vs Baseline        Abs Diff      Stability
----------------------------------------------------------------------------------------------------
warp_atomics                4,268.91 ✓       +12.34%           +468.46            ±13.21
bank_conflict_padding       4,012.33 +        +5.58%           +211.88            ±11.45
baseline                    3,800.45 =        +0.00%              +0.00            ±12.34
====================================================================================================
```

### 5. Profile Optimization

```bash
# Profile memory bandwidth
./scripts/profile.sh warp_atomics memory

# Profile bank conflicts
./scripts/profile.sh warp_atomics bank_conflicts

# Profile occupancy
./scripts/profile.sh warp_atomics occupancy

# Profile atomic operations
./scripts/profile.sh warp_atomics atomics
```

---

## Available Optimizations

| Flag | Expected Gain | Description |
|------|---------------|-------------|
| `-DOPT_WARP_ATOMICS` | 10-15% | Warp-level reduction before global atomics |
| `-DOPT_BANK_CONFLICT_PADDING` | 5-8% | Add padding to avoid bank conflicts |
| `-DOPT_VECTORIZED_SCORING` | 5-10% | Vectorized memory loads for scoring |
| `-DOPT_IMPROVED_OCCUPANCY` | 3-7% | Adjust launch bounds for better SM usage |
| `-DOPT_COALESCED_OUTPUT` | 2-4% | Reorganize output buffer for coalescing |
| `-DOPT_ALL` | 25-40% | Enable all optimizations |

---

## Build Commands Reference

```bash
# Baseline (no optimizations)
make clean && make

# Single optimization
make clean && make OPTS=-DOPT_WARP_ATOMICS

# Multiple optimizations
make clean && make OPTS="-DOPT_WARP_ATOMICS -DOPT_BANK_CONFLICT_PADDING"

# All optimizations
make clean && make OPTS=-DOPT_ALL
```

---

## Testing Best Practices

1. **Always run baseline first**
   ```bash
   make clean && make
   ./scripts/benchmark.sh baseline 120
   ```

2. **Use consistent test duration**
   - Quick test: 60 seconds
   - Accurate test: 120 seconds
   - Stability test: 300+ seconds

3. **Run multiple times for accuracy**
   ```bash
   for i in {1..3}; do
       ./scripts/benchmark.sh baseline_run${i} 60
   done
   ```

4. **Check standard deviation**
   - Good: StdDev < 20 MKeys/sec
   - Needs investigation: StdDev > 50 MKeys/sec

5. **Ensure no other GPU workloads**
   ```bash
   # Check GPU usage before testing
   nvidia-smi
   ```

---

## Troubleshooting

### Scripts not executable
```bash
chmod +x scripts/*.sh scripts/*.py
```

### Build fails with optimization flags
```bash
# Check optimizations.h is included
grep "optimizations.h" src/main.cu

# Try clean build
make clean
rm -rf *.o
make OPTS=-DOPT_WARP_ATOMICS
```

### No improvement seen
1. Profile to identify bottlenecks
2. Check if optimization addresses actual bottleneck
3. Try different work scales
4. Verify GPU is not thermal throttling

### Profiling requires sudo
```bash
# On some systems, profiling requires elevated privileges
sudo ./scripts/profile.sh baseline memory
```

---

## Output Files

- `benchmark_results/` - JSON benchmark results
- `profile_results/` - CSV and log files from profiling
- `benchmark_results/summary_*.txt` - Human-readable summaries
- `benchmark_results/*_latest.json` - Most recent result per optimization

---

## Example: Complete Testing Session

```bash
# 1. Setup (first time only)
./scripts/setup_testing.sh

# 2. Baseline
make clean && make
./scripts/benchmark.sh baseline 120

# 3. Test top optimization
make clean && make OPTS=-DOPT_WARP_ATOMICS
./scripts/benchmark.sh warp_atomics 120

# 4. Compare results
python3 scripts/compare_results.py \
    benchmark_results/baseline_latest.json \
    benchmark_results/warp_atomics_latest.json

# 5. If good, profile it
./scripts/profile.sh warp_atomics atomics

# 6. If improvement > 10%, test all
./scripts/test_all_optimizations.sh
```

---

## Advanced: Custom Optimization Testing

To implement and test your own optimization:

1. **Add flag to `src/optimizations.h`:**
   ```cpp
   #ifndef OPT_MY_OPTIMIZATION
       #define OPT_MY_OPTIMIZATION 0
   #else
       #undef OPT_MY_OPTIMIZATION
       #define OPT_MY_OPTIMIZATION 1
   #endif
   ```

2. **Add conditional code:**
   ```cpp
   #if OPT_MY_OPTIMIZATION
       // Optimized version
   #else
       // Original version
   #endif
   ```

3. **Build and test:**
   ```bash
   make clean
   make OPTS=-DOPT_MY_OPTIMIZATION
   ./scripts/benchmark.sh my_optimization 120
   ```

4. **Compare:**
   ```bash
   python3 scripts/compare_results.py \
       benchmark_results/baseline_latest.json \
       benchmark_results/my_optimization_latest.json
   ```

---

For detailed information, see `docs/testing-guide.md`
