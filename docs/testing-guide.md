# Performance Testing Guide

This guide explains how to test each optimization separately and measure their impact.

---

## Quick Start

### 1. Test All Optimizations Automatically

Run the comprehensive test suite:

```bash
chmod +x scripts/*.sh scripts/*.py
./scripts/test_all_optimizations.sh
```

This will:
- Build each optimization separately
- Benchmark each for 60 seconds
- Generate comparison reports
- Create summary files

### 2. Test Individual Optimization

```bash
# Build with specific optimization
make clean
make OPTS=-DOPT_WARP_ATOMICS

# Benchmark it
./scripts/benchmark.sh warp_atomics 60

# Compare with baseline
python3 scripts/compare_results.py \
    benchmark_results/baseline_latest.json \
    benchmark_results/warp_atomics_latest.json
```

### 3. Profile Specific Optimization

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

Each optimization can be enabled via compile-time flags:

| Optimization | Flag | Expected Gain |
|-------------|------|---------------|
| Warp-level atomics | `-DOPT_WARP_ATOMICS` | 10-15% |
| Bank conflict padding | `-DOPT_BANK_CONFLICT_PADDING` | 5-8% |
| Vectorized scoring | `-DOPT_VECTORIZED_SCORING` | 5-10% |
| Improved occupancy | `-DOPT_IMPROVED_OCCUPANCY` | 3-7% |
| Coalesced output | `-DOPT_COALESCED_OUTPUT` | 2-4% |
| **All optimizations** | `-DOPT_ALL` | 25-40% |

---

## Manual Testing Workflow

### Step 1: Establish Baseline

```bash
# Build baseline (no optimizations)
make clean
make -j$(nproc)

# Run baseline benchmark
./scripts/benchmark.sh baseline 120

# Profile baseline
./scripts/profile.sh baseline memory
./scripts/profile.sh baseline occupancy
./scripts/profile.sh baseline atomics
```

### Step 2: Test Each Optimization

#### Test Warp Atomics

```bash
# Build with warp atomics
make clean
make OPTS=-DOPT_WARP_ATOMICS

# Benchmark
./scripts/benchmark.sh warp_atomics 120

# Profile atomic operations
./scripts/profile.sh warp_atomics atomics
```

#### Test Bank Conflict Padding

```bash
# Build with padding
make clean
make OPTS=-DOPT_BANK_CONFLICT_PADDING

# Benchmark
./scripts/benchmark.sh bank_conflict_padding 120

# Profile bank conflicts
./scripts/profile.sh bank_conflict_padding bank_conflicts
```

#### Test Vectorized Scoring

```bash
# Build with vectorization
make clean
make OPTS=-DOPT_VECTORIZED_SCORING

# Benchmark
./scripts/benchmark.sh vectorized_scoring 120

# Profile compute throughput
./scripts/profile.sh vectorized_scoring compute
```

### Step 3: Combine Optimizations

Test combinations to find synergies:

```bash
# Warp atomics + Bank padding
make clean
make OPTS="-DOPT_WARP_ATOMICS -DOPT_BANK_CONFLICT_PADDING"
./scripts/benchmark.sh warp_plus_bank 120

# All optimizations
make clean
make OPTS=-DOPT_ALL
./scripts/benchmark.sh all_optimizations 120
```

### Step 4: Compare Results

```bash
# Compare all results
python3 scripts/compare_results.py benchmark_results/*_latest.json

# Detailed comparison between two
python3 scripts/compare_results.py \
    benchmark_results/baseline_latest.json \
    benchmark_results/all_optimizations_latest.json
```

---

## Interpreting Results

### Benchmark Output

```
╔════════════════════════════════════════════════════════════╗
║  Benchmark Results: warp_atomics
╚════════════════════════════════════════════════════════════╝

  Average Speed:              3950.23 MKeys/sec
  Median Speed:               3948.50 MKeys/sec
  Min Speed:                  3920.11 MKeys/sec
  Max Speed:                  3975.67 MKeys/sec
  Std Deviation:                 15.23 MKeys/sec
  Sample Count:                     45 samples

Results saved to: benchmark_results/warp_atomics_20251216_143022.json

Comparing with baseline...
  +12.34% improvement over baseline
```

**Key Metrics:**
- **Average Speed**: Primary metric for comparison
- **Std Deviation**: Lower is better (more stable)
- **Min/Max**: Check for outliers
- **vs Baseline**: Percentage improvement

### Profile Output

#### Memory Bandwidth

```
Metric                                          Value
dram__throughput.avg.pct_of_peak_sustained     87.5%
l1tex__t_bytes_pipe_lsu_mem_global_op_ld       245 GB/s
```

**Good:** >80% DRAM utilization
**Needs work:** <60% utilization

#### Bank Conflicts

```
Metric                                          Value
l1tex__data_bank_conflicts_pipe_lsu_mem_shared  1,234
```

**Good:** <1000 conflicts per kernel
**Needs work:** >5000 conflicts

#### Occupancy

```
Metric                                          Value
achieved_occupancy                              0.85
sm__warps_active.avg.pct_of_peak               89.2%
```

**Good:** >75% occupancy
**Needs work:** <50% occupancy

#### Atomic Operations

```
Metric                                          Value
smsp__inst_executed_op_generic_atom_dot_add    8.4M
smsp__inst_executed_op_generic_atom_dot_max    8.4M
```

**Baseline:** 8.4M operations (all threads)
**Warp-optimized:** ~260K operations (warps only)

---

## Troubleshooting

### Benchmark Script Issues

**Problem:** Script not executable
```bash
chmod +x scripts/*.sh scripts/*.py
```

**Problem:** Binary not found
```bash
make clean && make
```

**Problem:** No speed data captured
```bash
# Increase duration for slower GPUs
./scripts/benchmark.sh baseline 180
```

### Profiling Issues

**Problem:** `ncu` not found
```bash
# Install NVIDIA Nsight Compute
# Download from: https://developer.nvidia.com/nsight-compute
export PATH=$PATH:/usr/local/cuda/bin
```

**Problem:** Profiling takes too long
```bash
# Reduce work scale for profiling
# Edit profile.sh and change WORK_SCALE=12
```

**Problem:** Permission denied
```bash
# May need sudo for profiling on some systems
sudo ./scripts/profile.sh baseline memory
```

### Build Issues

**Problem:** Optimization flag not recognized
```bash
# Check optimizations.h is included in main.cu
# Verify flag syntax: -DOPT_NAME (not -D OPT_NAME)
```

**Problem:** Linker errors with optimizations
```bash
# Clean build directory
make clean
rm -rf *.o
make OPTS=-DOPT_WARP_ATOMICS
```

---

## Advanced Testing

### Testing on Multiple GPUs

```bash
# Test on all GPUs
for device in 0 1 2 3; do
    DEVICE=$device ./scripts/benchmark.sh baseline_gpu${device} 60
done
```

### Long-Duration Stability Test

```bash
# 30-minute test
./scripts/benchmark.sh baseline_stability 1800

# Check standard deviation in results
```

### Memory Scaling Test

```bash
# Test different work scales
for scale in 14 15 16 17; do
    make clean
    make WORK_SCALE=$scale
    ./scripts/benchmark.sh baseline_scale${scale} 60
done
```

### Compiler Optimization Levels

```bash
# Test different optimization levels
make clean
make CXXFLAGS="-O2"
./scripts/benchmark.sh opt_level_O2 60

make clean
make CXXFLAGS="-O3"
./scripts/benchmark.sh opt_level_O3 60

make clean
make CXXFLAGS="-Ofast"
./scripts/benchmark.sh opt_level_Ofast 60
```

---

## Best Practices

1. **Always establish baseline first**
   - Run baseline test before any changes
   - Save baseline results for comparison

2. **Multiple runs for accuracy**
   - Run each test 3+ times
   - Use median or average of runs
   - Check for consistency (low std dev)

3. **Control variables**
   - Same GPU, same work scale
   - No other GPU workloads running
   - Consistent ambient temperature

4. **Profile before optimizing**
   - Identify actual bottlenecks
   - Don't assume - measure!

5. **Test combinations**
   - Some optimizations have synergies
   - Test best individual + best pairs

6. **Document everything**
   - Save all benchmark results
   - Note hardware specs
   - Record ambient conditions

---

## Example Testing Session

```bash
# 1. Clean start
make clean
git status  # Ensure clean working directory

# 2. Baseline
make -j$(nproc)
./scripts/benchmark.sh baseline 120
./scripts/profile.sh baseline memory
./scripts/profile.sh baseline atomics

# 3. Test top optimization
make clean
make OPTS=-DOPT_WARP_ATOMICS
./scripts/benchmark.sh warp_atomics 120
./scripts/profile.sh warp_atomics atomics

# 4. Compare
python3 scripts/compare_results.py \
    benchmark_results/baseline_latest.json \
    benchmark_results/warp_atomics_latest.json

# 5. If improvement > 10%, commit
# If improvement < 5%, investigate profiling data
```

---

## Automated CI/CD Testing

```yaml
# .github/workflows/performance-test.yml
name: Performance Regression Test

on: [push, pull_request]

jobs:
  benchmark:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v2
      - name: Build and Test
        run: |
          make clean
          make OPTS=-DOPT_ALL
          ./scripts/benchmark.sh ci_test 60
      - name: Compare with main
        run: |
          python3 scripts/compare_results.py \
            benchmark_results/main_baseline.json \
            benchmark_results/ci_test_latest.json
```

---

## Next Steps

After testing:

1. **Analyze results** - Which optimizations provide best gains?
2. **Combine winners** - Test synergies between best optimizations
3. **Profile bottlenecks** - Deep dive into remaining issues
4. **Iterate** - Refine and retest
5. **Document** - Update README with performance numbers

For questions or issues, open a GitHub issue with:
- Benchmark results JSON
- Profile output
- GPU model and driver version
- CUDA version
