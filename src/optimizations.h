/*
    Optimization Feature Flags
    Enable/disable specific optimizations via compile-time flags

    Usage:
        make OPTS=-DOPT_WARP_ATOMICS          # Enable warp-level atomics
        make OPTS="-DOPT_WARP_ATOMICS -DOPT_VECTORIZED_SCORING"  # Multiple opts
        make OPTS=-DOPT_ALL                   # Enable all optimizations
*/

#pragma once

// Master switch to enable all optimizations at once
#ifdef OPT_ALL
    #define OPT_WARP_ATOMICS
    #define OPT_BANK_CONFLICT_PADDING
    #define OPT_VECTORIZED_SCORING
    #define OPT_PERSISTENT_KERNELS
    #define OPT_IMPROVED_OCCUPANCY
    #define OPT_COALESCED_OUTPUT
#endif

// Individual optimization flags (can be set independently)

// OPT_WARP_ATOMICS: Use warp-level reduction before global atomics
// Expected gain: 10-15%
#ifndef OPT_WARP_ATOMICS
    #define OPT_WARP_ATOMICS 0
#else
    #undef OPT_WARP_ATOMICS
    #define OPT_WARP_ATOMICS 1
#endif

// OPT_BANK_CONFLICT_PADDING: Add padding to shared memory arrays
// Expected gain: 5-8%
#ifndef OPT_BANK_CONFLICT_PADDING
    #define OPT_BANK_CONFLICT_PADDING 0
#else
    #undef OPT_BANK_CONFLICT_PADDING
    #define OPT_BANK_CONFLICT_PADDING 1
#endif

// OPT_VECTORIZED_SCORING: Use vectorized loads for scoring functions
// Expected gain: 5-10%
#ifndef OPT_VECTORIZED_SCORING
    #define OPT_VECTORIZED_SCORING 0
#else
    #undef OPT_VECTORIZED_SCORING
    #define OPT_VECTORIZED_SCORING 1
#endif

// OPT_PERSISTENT_KERNELS: Use persistent thread pattern
// Expected gain: 8-12%
#ifndef OPT_PERSISTENT_KERNELS
    #define OPT_PERSISTENT_KERNELS 0
#else
    #undef OPT_PERSISTENT_KERNELS
    #define OPT_PERSISTENT_KERNELS 1
#endif

// OPT_IMPROVED_OCCUPANCY: Adjust launch bounds for better occupancy
// Expected gain: 3-7%
#ifndef OPT_IMPROVED_OCCUPANCY
    #define OPT_IMPROVED_OCCUPANCY 0
#else
    #undef OPT_IMPROVED_OCCUPANCY
    #define OPT_IMPROVED_OCCUPANCY 1
#endif

// OPT_COALESCED_OUTPUT: Reorganize output buffer for coalesced writes
// Expected gain: 2-4%
#ifndef OPT_COALESCED_OUTPUT
    #define OPT_COALESCED_OUTPUT 0
#else
    #undef OPT_COALESCED_OUTPUT
    #define OPT_COALESCED_OUTPUT 1
#endif

// Configuration-dependent constants
#if OPT_BANK_CONFLICT_PADDING
    #define PADDING_SIZE 8
#else
    #define PADDING_SIZE 0
#endif

#if OPT_IMPROVED_OCCUPANCY
    #define LAUNCH_BOUNDS_BLOCKS 3
#else
    #define LAUNCH_BOUNDS_BLOCKS 2
#endif

// Helper macros for conditional compilation
#define IF_OPT(flag, code) do { if (flag) { code } } while(0)
