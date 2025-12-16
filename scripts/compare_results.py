#!/usr/bin/env python3

"""
Compare benchmark results across different optimizations
Usage: python3 scripts/compare_results.py [result1.json] [result2.json] ...
       python3 scripts/compare_results.py benchmark_results/*.json
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict
import statistics

def load_result(filepath: str) -> Dict:
    """Load a benchmark result JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def format_speed(speed: float) -> str:
    """Format speed in MKeys/sec"""
    return f"{speed:,.2f}"

def calculate_improvement(baseline: float, current: float) -> tuple:
    """Calculate improvement percentage and absolute difference"""
    if baseline == 0:
        return 0.0, 0.0

    diff = current - baseline
    pct = (diff / baseline) * 100
    return pct, diff

def print_comparison_table(results: List[Dict]):
    """Print a formatted comparison table"""
    if not results:
        print("No results to compare")
        return

    # Sort by average speed
    results.sort(key=lambda x: x['metrics']['avg_speed_mkeys_per_sec'], reverse=True)

    # Find baseline
    baseline = next((r for r in results if r['optimization'] == 'baseline'), results[-1])
    baseline_speed = baseline['metrics']['avg_speed_mkeys_per_sec']

    print("\n" + "="*100)
    print("BENCHMARK COMPARISON RESULTS")
    print("="*100)
    print(f"\nBaseline: {baseline['optimization']} @ {format_speed(baseline_speed)} MKeys/sec")
    print(f"GPU: {baseline['gpu_info']}")
    print("\n" + "-"*100)

    # Header
    print(f"{'Optimization':<25} {'Avg Speed':>15} {'vs Baseline':>15} {'Abs Diff':>15} {'Stability':>15}")
    print("-"*100)

    # Rows
    for result in results:
        opt_name = result['optimization']
        metrics = result['metrics']

        avg_speed = metrics['avg_speed_mkeys_per_sec']
        stddev = metrics['stddev_mkeys_per_sec']

        pct_improvement, abs_diff = calculate_improvement(baseline_speed, avg_speed)

        # Color coding for terminal
        if pct_improvement > 5:
            symbol = "✓"
            color = "\033[92m"  # Green
        elif pct_improvement > 0:
            symbol = "+"
            color = "\033[93m"  # Yellow
        elif pct_improvement < -5:
            symbol = "✗"
            color = "\033[91m"  # Red
        else:
            symbol = "="
            color = "\033[0m"   # Default

        reset = "\033[0m"

        stability = f"±{stddev:.2f}"

        print(f"{color}{opt_name:<25}{reset} "
              f"{format_speed(avg_speed):>15} "
              f"{symbol} {pct_improvement:>7.2f}% {reset:>5}"
              f"{abs_diff:>14.2f} "
              f"{stability:>15}")

    print("-"*100)

    # Statistical summary
    print("\nSTATISTICAL SUMMARY:")
    print(f"  Best performer: {results[0]['optimization']} ({format_speed(results[0]['metrics']['avg_speed_mkeys_per_sec'])} MKeys/sec)")
    print(f"  Worst performer: {results[-1]['optimization']} ({format_speed(results[-1]['metrics']['avg_speed_mkeys_per_sec'])} MKeys/sec)")

    speeds = [r['metrics']['avg_speed_mkeys_per_sec'] for r in results]
    print(f"  Mean: {format_speed(statistics.mean(speeds))} MKeys/sec")
    print(f"  Median: {format_speed(statistics.median(speeds))} MKeys/sec")
    if len(speeds) > 1:
        print(f"  Std Dev: {statistics.stdev(speeds):.2f} MKeys/sec")

    # Top improvements
    improvements = [(r['optimization'], *calculate_improvement(baseline_speed, r['metrics']['avg_speed_mkeys_per_sec']))
                   for r in results if r['optimization'] != baseline['optimization']]
    improvements.sort(key=lambda x: x[1], reverse=True)

    if improvements:
        print("\nTOP IMPROVEMENTS:")
        for i, (opt, pct, diff) in enumerate(improvements[:5], 1):
            print(f"  {i}. {opt}: +{pct:.2f}% (+{diff:.2f} MKeys/sec)")

    print("\n" + "="*100)

def print_detailed_comparison(result1: Dict, result2: Dict):
    """Print detailed comparison between two specific results"""
    print("\n" + "="*80)
    print(f"DETAILED COMPARISON: {result1['optimization']} vs {result2['optimization']}")
    print("="*80)

    m1 = result1['metrics']
    m2 = result2['metrics']

    metrics_to_compare = [
        ('Average Speed', 'avg_speed_mkeys_per_sec', 'MKeys/sec'),
        ('Median Speed', 'median_speed_mkeys_per_sec', 'MKeys/sec'),
        ('Min Speed', 'min_speed_mkeys_per_sec', 'MKeys/sec'),
        ('Max Speed', 'max_speed_mkeys_per_sec', 'MKeys/sec'),
        ('Std Deviation', 'stddev_mkeys_per_sec', 'MKeys/sec'),
    ]

    print(f"\n{'Metric':<20} {result1['optimization']:>20} {result2['optimization']:>20} {'Difference':>20}")
    print("-"*80)

    for label, key, unit in metrics_to_compare:
        v1 = m1[key]
        v2 = m2[key]
        diff_pct, diff_abs = calculate_improvement(v1, v2)

        print(f"{label:<20} {v1:>18.2f} {unit:>2} {v2:>18.2f} {unit:>2} {diff_pct:>+9.2f}%")

    print("="*80)

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/compare_results.py [result1.json] [result2.json] ...")
        print("       python3 scripts/compare_results.py benchmark_results/*.json")
        sys.exit(1)

    # Load all results
    results = []
    for filepath in sys.argv[1:]:
        if not os.path.exists(filepath):
            print(f"Warning: File not found: {filepath}")
            continue

        try:
            result = load_result(filepath)
            results.append(result)
        except Exception as e:
            print(f"Warning: Failed to load {filepath}: {e}")

    if not results:
        print("Error: No valid result files loaded")
        sys.exit(1)

    # Print comparison table
    print_comparison_table(results)

    # If exactly 2 results, also print detailed comparison
    if len(results) == 2:
        print("\n")
        print_detailed_comparison(results[0], results[1])

if __name__ == '__main__':
    main()
