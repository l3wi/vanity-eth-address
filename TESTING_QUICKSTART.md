# Testing Quick Start Guide

## 🚀 One-Command Testing

### Run Everything (Recommended First)

```bash
# From project root directory:
./scripts/setup_testing.sh && ./scripts/test_all_optimizations.sh
```

**What this does:**
1. Sets up testing environment (one-time setup)
2. Builds and tests all optimizations (20 seconds each)
3. Generates comparison report
4. Shows you which optimizations work best

**Total time:** ~3-4 minutes for all tests

---

## 📋 Step-by-Step Entry Point

If you prefer to run step-by-step:

### Step 1: Setup (First Time Only)
```bash
cd /Users/lewi/Documents/ethereum/vanity-eth-address
chmod +x scripts/setup_testing.sh
./scripts/setup_testing.sh
```

**What it does:**
- Makes scripts executable
- Checks for required tools (nvcc, ncu, nvidia-smi)
- Creates output directories
- Builds baseline binary
- Takes ~30 seconds

### Step 2: Run Baseline Test
```bash
./scripts/benchmark.sh baseline 20
```

**Output example:**
```
╔════════════════════════════════════════════════════════════╗
║  Vanity-ETH-Address Performance Benchmark                  ║
╚════════════════════════════════════════════════════════════╝

Configuration:
  Optimization: baseline
  Duration: 20s (5s warmup)
  Device: 0
  Work Scale: 15

GPU Information:
NVIDIA GeForce RTX 4090

Starting benchmark...
Running for 20 seconds...

[Program output showing MKeys/sec rates...]

Processing results...

╔════════════════════════════════════════════════════════════╗
║  Benchmark Results: baseline
╚════════════════════════════════════════════════════════════╝

  Average Speed:              3800.45 MKeys/sec
  Median Speed:               3798.20 MKeys/sec
  Min Speed:                  3775.12 MKeys/sec
  Max Speed:                  3825.67 MKeys/sec
  Std Deviation:                 12.34 MKeys/sec
  Sample Count:                     15 samples

Results saved to: benchmark_results/baseline_20251216_143022.json

Benchmark complete!
```

### Step 3: Test Single Optimization
```bash
# Build with optimization flag
make clean
make OPTS=-DOPT_WARP_ATOMICS

# Benchmark it (20 seconds)
./scripts/benchmark.sh warp_atomics 20
```

**Output includes automatic comparison:**
```
Comparing with baseline...
  +12.34% improvement over baseline
```

### Step 4: View Results
```bash
# Compare all results
python3 scripts/compare_results.py benchmark_results/*_latest.json
```

**Output:**
```
====================================================================
BENCHMARK COMPARISON RESULTS
====================================================================

Optimization              Avg Speed        vs Baseline
--------------------------------------------------------------------
warp_atomics                4,268.91 ✓       +12.34%
baseline                    3,800.45 =        +0.00%
====================================================================

TOP IMPROVEMENTS:
  1. warp_atomics: +12.34% (+468.46 MKeys/sec)
```

---

## 🎯 Testing Individual Optimizations

Each optimization can be tested separately:

### Warp-Level Atomics
```bash
make clean && make OPTS=-DOPT_WARP_ATOMICS
./scripts/benchmark.sh warp_atomics 20
```

### Bank Conflict Padding
```bash
make clean && make OPTS=-DOPT_BANK_CONFLICT_PADDING
./scripts/benchmark.sh bank_conflict_padding 20
```

### Vectorized Scoring
```bash
make clean && make OPTS=-DOPT_VECTORIZED_SCORING
./scripts/benchmark.sh vectorized_scoring 20
```

### Improved Occupancy
```bash
make clean && make OPTS=-DOPT_IMPROVED_OCCUPANCY
./scripts/benchmark.sh improved_occupancy 20
```

### All Optimizations Combined
```bash
make clean && make OPTS=-DOPT_ALL
./scripts/benchmark.sh all_optimizations 20
```

---

## 🔍 Profiling (Optional)

If you want to see detailed GPU metrics:

```bash
# Profile memory bandwidth
./scripts/profile.sh baseline memory

# Profile atomic operations
./scripts/profile.sh warp_atomics atomics

# Profile occupancy
./scripts/profile.sh baseline occupancy

# Profile bank conflicts
./scripts/profile.sh bank_conflict_padding bank_conflicts
```

**Note:** Profiling requires `ncu` (NVIDIA Nsight Compute) to be installed.

---

## 📊 Understanding Results

### Good Performance Indicators
- ✅ Improvement > 5% → Significant gain
- ✅ Std Deviation < 20 MKeys/sec → Stable performance
- ✅ Sample Count > 10 → Reliable data

### Red Flags
- ⚠️ Improvement < 1% → Optimization not helping
- ⚠️ Std Deviation > 50 MKeys/sec → Unstable performance
- ⚠️ Negative improvement → Performance regression

---

## 🛠️ Customizing Test Duration

If you want longer/shorter tests:

```bash
# Quick test (10 seconds)
./scripts/benchmark.sh baseline 10

# Medium test (default: 20 seconds)
./scripts/benchmark.sh baseline 20

# Longer test (60 seconds - more accurate)
./scripts/benchmark.sh baseline 60

# Stability test (5 minutes)
./scripts/benchmark.sh baseline 300
```

---

## 🔄 Re-running Tests

To re-run tests after code changes:

```bash
# Clean build
make clean

# Test baseline again
make
./scripts/benchmark.sh baseline_v2 20

# Compare with previous baseline
python3 scripts/compare_results.py \
    benchmark_results/baseline_latest.json \
    benchmark_results/baseline_v2_latest.json
```

---

## 📁 Output Locations

After running tests, find results here:

```
benchmark_results/
├── baseline_latest.json              # Latest baseline result
├── warp_atomics_latest.json          # Latest optimization result
├── summary_20251216_143022.txt       # Human-readable summary
└── comparison_20251216_143022.csv    # Data for visualization

profile_results/
├── baseline_memory_20251216.csv      # Profiling metrics
└── baseline_memory_20251216.log      # Profiling log
```

---

## 🐛 Troubleshooting

### "Binary not found"
```bash
make clean && make
```

### "timeout: command not found"
```bash
# macOS: install coreutils
brew install coreutils

# Linux: timeout is usually built-in
```

### "No speed data found"
- Program may not be outputting expected format
- Try running manually first:
  ```bash
  ./vanity-eth-address --leading-zeros --device 0
  # Press Ctrl+C after a few seconds
  ```

### "ncu: command not found"
- Profiling requires NVIDIA Nsight Compute
- Download from: https://developer.nvidia.com/nsight-compute
- Or skip profiling and just run benchmarks

---

## 📞 Next Steps After Testing

1. **Review summary:**
   ```bash
   cat benchmark_results/summary_*.txt
   ```

2. **Identify best optimization:**
   - Look for highest % improvement
   - Check for stable performance (low std dev)

3. **Profile the winner:**
   ```bash
   ./scripts/profile.sh <best_optimization> memory
   ```

4. **Test combinations:**
   ```bash
   make clean
   make OPTS="-DOPT_WARP_ATOMICS -DOPT_BANK_CONFLICT_PADDING"
   ./scripts/benchmark.sh combined_opts 20
   ```

5. **Commit the improvements:**
   - If improvement > 10%, consider merging optimization
   - Update README with new performance numbers

---

## 🎓 More Information

- **Full testing guide:** `docs/testing-guide.md`
- **Quick reference:** `README_TESTING.md`
- **Performance analysis:** `docs/performance-optimization-analysis.md`

---

## ⚡ TL;DR - Just Run This

```bash
# One command to rule them all:
./scripts/setup_testing.sh && ./scripts/test_all_optimizations.sh

# Wait 3-4 minutes, then view results:
cat benchmark_results/summary_*.txt
```

That's it! The scripts handle everything automatically.
