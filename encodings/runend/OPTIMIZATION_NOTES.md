# Run-End Decode Optimization Notes

## Overview

This document describes the optimization work done on the run-end decode functions
in `vortex-runend`. Run-end encoding is effective when average run length > 8.

## Data Distributions Tested

### Bool Decode Distributions

| Distribution | Description | Use Case |
|--------------|-------------|----------|
| **Alternating** | 50/50 true/false runs | Worst case - maximum work |
| **MostlyTrue** | 90% true runs, 10% false | Bitmap with sparse false flags |
| **MostlyFalse** | 10% true runs, 90% false | Sparse validity masks |
| **AllTrue** | Single true run | Best case for true-heavy |
| **AllFalse** | Single false run | Best case for false-heavy |

### Run Length Parameters

| Length | Avg Run | Description | Runs in 1M elements |
|--------|---------|-------------|---------------------|
| 1M | 2 | Very short | 500,000 runs |
| 1M | 10 | Short | 100,000 runs |
| 1M | 100 | Medium | 10,000 runs |
| 1M | 1000 | Long | 1,000 runs |
| 1M | 10000 | Very long | 100 runs |
| 1M | 100000 | Extremely long | 10 runs |

## Implementations Tested

### 1. Original Bool Decode (`runend_decode_typed_bool_original`)

**Approach**: Use `BitBufferMut::append_n()` for each run.

```rust
let mut decoded = BitBufferMut::with_capacity(length);
for (end, value) in run_ends.zip_eq(values.iter()) {
    if end > decoded.len() {
        decoded.append_n(value, end - decoded.len());
    }
}
```

**Characteristics**:
- Simple and readable
- Per-run capacity checks
- Per-run length tracking overhead
- No optimization based on data distribution

### 2. Optimized Bool Decode (Pre-fill zeros, fill true runs)

**Approach**: Pre-allocate buffer with zeros, only fill true runs.

```rust
let mut decoded = BitBufferMut::new_unset(length);  // All zeros
let decoded_bytes = decoded.as_mut_slice();
for (end, value) in run_ends.zip_eq(values.iter()) {
    if end > current_pos && value {
        fill_bits_true(decoded_bytes, current_pos, end);
    }
    current_pos = end;
}
```

**Characteristics**:
- Pre-allocation eliminates per-run capacity checks
- Direct byte manipulation (no iterator overhead)
- Skips false runs entirely (they're already 0)
- Good for false-heavy data

### 3. Adaptive Bool Decode (Current Implementation)

**Approach**: Count true vs false values, choose optimal strategy.

```rust
let true_count = values.true_count();
let false_count = values.len() - true_count;

if true_count > false_count {
    // Pre-fill with 1s, clear false runs
    let mut decoded = BitBufferMut::new_set(length);
    for (end, value) in run_ends.zip_eq(values.iter()) {
        if end > current_pos && !value {
            fill_bits_false(decoded_bytes, current_pos, end);
        }
        current_pos = end;
    }
} else {
    // Pre-fill with 0s, fill true runs
    let mut decoded = BitBufferMut::new_unset(length);
    for (end, value) in run_ends.zip_eq(values.iter()) {
        if end > current_pos && value {
            fill_bits_true(decoded_bytes, current_pos, end);
        }
        current_pos = end;
    }
}
```

**Characteristics**:
- O(1) true_count check on values BitBuffer
- Chooses optimal fill direction based on distribution
- Minimizes work for skewed distributions
- Works well for all distributions

### 4. Primitive Decode (`runend_decode_typed_primitive`)

**Approach**: Pre-allocate buffer, use `slice::fill` for each run.

```rust
let mut decoded: BufferMut<T> = BufferMut::with_capacity(length);
unsafe { decoded.set_len(length) };
let decoded_slice = decoded.as_mut_slice();

for (end, value) in run_ends.zip_eq(values) {
    if end > current_pos {
        decoded_slice[current_pos..end].fill(*value);
        current_pos = end;
    }
}
```

**Characteristics**:
- LLVM auto-vectorizes `slice::fill` effectively
- Pre-allocation with `set_len` avoids double-write
- Direct slice access (no iterator overhead)

## Bit Manipulation Functions

### `fill_bits_true(slice, start, end)`

Fills bits [start, end) to 1 using byte-level operations:

1. Handle same-byte case with bit mask
2. First partial byte: OR with high bits mask
3. Middle bytes: `slice.fill(0xFF)` (vectorized by LLVM)
4. Last partial byte: OR with low bits mask

### `fill_bits_false(slice, start, end)`

Clears bits [start, end) to 0 using byte-level operations:

1. Handle same-byte case with inverted mask
2. First partial byte: AND with low bits mask
3. Middle bytes: `slice.fill(0x00)` (vectorized by LLVM)
4. Last partial byte: AND with high bits mask

## Performance Results

### Bool Decode: Original vs Optimized (1M elements)

| Distribution | Run Len | Original | Optimized | Speedup |
|--------------|---------|----------|-----------|---------|
| All False | 2 | 2.81 ms | 884 µs | **3.2x** |
| All False | 10 | 1.03 ms | 177 µs | **5.8x** |
| All False | 100 | 98.7 µs | 20.0 µs | **4.9x** |
| All False | 1000 | 10.9 µs | 4.44 µs | **2.5x** |
| All True | 2 | 2.83 ms | 889 µs | **3.2x** |
| All True | 10 | 1.03 ms | 177 µs | **5.8x** |
| All True | 100 | 98.8 µs | 19.9 µs | **5.0x** |
| All True | 1000 | 11.0 µs | 4.34 µs | **2.5x** |
| Alternating | 1000 | 11.0 µs | 5.94 µs | **1.9x** |

### Key Insights

1. **Short runs benefit most**: 5-6x speedup for avg run length 10
2. **Skewed distributions benefit from adaptive**: All-true and all-false perform equally well
3. **LLVM auto-vectorization works well**: `slice::fill` gets optimized to SIMD
4. **Pre-allocation matters**: Eliminating per-run checks provides significant gains

## SIMD Research Summary

Four research agents analyzed potential SIMD optimizations:

### Conclusion

LLVM's auto-vectorization already handles `slice::fill` well. Explicit SIMD would provide:
- Marginal gains (5-20%) for very long runs
- Added complexity and maintenance burden
- Architecture-specific code paths

### Recommendations (if explicit SIMD needed)

| Architecture | Approach | Threshold |
|--------------|----------|-----------|
| x86 AVX2 | `_mm256_storeu_si256` + broadcast | 32+ bytes |
| ARM NEON | `vdupq_n_u8` + `vst1q_u8` | 32+ bytes |
| Portable | `std::simd` (nightly) | N/A |

## Files Modified

- `encodings/runend/src/compress.rs` - Main decode functions
- `encodings/runend/src/iter.rs` - Fixed offset handling with `saturating_sub`
- `encodings/runend/benches/run_end_decode.rs` - Comprehensive benchmarks
- `encodings/runend/Cargo.toml` - Added benchmark entry

## Benchmark Results: Primitive Distributions (1M elements)

### Constant Value Distribution
| Type | Avg Run 8 | Avg Run 64 | Avg Run 1024 | Avg Run 10000 |
|------|-----------|------------|--------------|---------------|
| u8   | 310 µs    | 58 µs      | 27 µs        | 22 µs         |
| u32  | 259 µs    | 157 µs     | 155 µs       | 156 µs        |
| u64  | 401 µs    | 331 µs     | 326 µs       | 327 µs        |

### Sparse Non-Zero (90% zeros)
| Type | Avg Run 8 | Avg Run 64 | Avg Run 1024 | Avg Run 10000 |
|------|-----------|------------|--------------|---------------|
| u8   | 310 µs    | 59 µs      | 22 µs        | 22 µs         |
| u32  | 261 µs    | 157 µs     | 157 µs       | 156 µs        |
| u64  | 398 µs    | 328 µs     | 322 µs       | 327 µs        |

**Insight**: Performance is consistent across distributions - the current implementation doesn't benefit from skipping zero values. This is an optimization opportunity.

## Future Optimization Recommendations

Based on research from specialized agents targeting different scenarios:

### 1. Sparse Zero Optimization (High Priority)

**Current State**: Fills all runs including zeros.
**Optimization**: Use `BufferMut::zeroed()` + skip zero fills.

```rust
// If >50% of runs are zeros, switch strategy
let zero_count = values.iter().filter(|&&v| v == T::default()).count();
if zero_count * 2 > values.len() {
    let mut decoded = BufferMut::zeroed(length);  // OS lazy allocation
    for (end, value) in run_ends.zip_eq(values) {
        if *value != T::default() && end > current_pos {
            decoded_slice[current_pos..end].fill(*value);
        }
        current_pos = end;
    }
}
```

**Expected Gain**: 20-50% for sparse data (90% zeros).

### 2. Constant Value Detection (Medium Priority)

**Optimization**: If all run values are identical, use single memset.

```rust
let first = values[0];
if values.iter().all(|&v| v == first) {
    decoded_slice.fill(first);  // Single memset, no loop
    return PrimitiveArray::new(decoded, ...);
}
```

**Expected Gain**: 10-30% for constant arrays.

### 3. Run Length Classification (Medium Priority)

For short runs (8-64), reduce per-run overhead:

```rust
match run_len {
    0..=8 => fill_tiny(slice, val),     // Manual unroll
    9..=64 => fill_small(slice, val),   // Inline SIMD
    _ => slice.fill(val),               // Standard path
}
```

**Expected Gain**: 15-40% for short runs.

### 4. Non-Temporal Stores (Low Priority - Specialized)

For very long runs (>1MB), avoid cache pollution:

```rust
#[cfg(target_arch = "x86_64")]
unsafe fn fill_nontemporal(slice: &mut [u64], value: u64) {
    let broadcast = _mm256_set1_epi64x(value as i64);
    for chunk in slice.chunks_exact_mut(4) {
        _mm256_stream_si256(chunk.as_mut_ptr() as *mut __m256i, broadcast);
    }
    _mm_sfence();
}
```

**Expected Gain**: 20-40% for >1MB fills. Only beneficial for extremely long runs.

### 5. Small Type SIMD (u8) (Low Priority - Specialized)

For u8 with medium runs (64-512 bytes):

```rust
#[cfg(target_arch = "x86_64")]
unsafe fn fill_u8_avx2(slice: &mut [u8], value: u8) {
    let broadcast = _mm256_set1_epi8(value as i8);
    for chunk in slice.chunks_exact_mut(32) {
        _mm256_storeu_si256(chunk.as_mut_ptr() as *mut __m256i, broadcast);
    }
    // Handle remainder...
}
```

**Expected Gain**: 10-30% for u8 with 64-512 byte runs. LLVM auto-vectorization already handles this well.

## Priority Ranking

| Priority | Optimization | Expected Gain | Complexity | When to Use |
|----------|--------------|---------------|------------|-------------|
| 1 | Sparse Zero | 20-50% | Low | >50% zero runs |
| 2 | Constant Detection | 10-30% | Low | Constant arrays |
| 3 | Run Length Classification | 15-40% | Medium | Many short runs |
| 4 | Non-Temporal Stores | 20-40% | Medium | >1MB fills |
| 5 | Small Type SIMD | 10-30% | High | u8 with 64-512B runs |

## Key Insight

**LLVM auto-vectorization already does an excellent job** with `slice::fill`. The main optimization opportunities are:

1. **Avoiding unnecessary work** (sparse zeros, constant values)
2. **Reducing per-run overhead** (run length classification)
3. **Specialized paths for extreme cases** (non-temporal for huge fills)

Focus on correctness and maintainability first. Only add complex optimizations when benchmarks prove significant gains.

## Type-Specific Optimization Research

### u64 Optimizations (8-byte integers)

**Key findings**:
- LLVM auto-vectorizes `slice::fill` effectively with AVX2 (4 u64s per 256-bit register)
- Memory bandwidth is the bottleneck for long runs (>1024 elements)
- Per-run overhead is the bottleneck for short runs (8-64 elements)

**Recommendations**:
| Priority | Optimization | Expected Gain | Notes |
|----------|--------------|---------------|-------|
| 1 | Sparse zero detection | 20-50% | Use `BufferMut::zeroed()` + skip zero fills |
| 2 | Non-temporal stores | 20-40% | Only for runs >1024 elements (8KB) |
| 3 | Aligned allocation | 0-5% | Use `Alignment::new(64)` for cache lines |
| Low | Manual SIMD | ~5% | **Not recommended** - LLVM already handles this |

### f64 Optimizations (8-byte floats)

**Key findings**:
- Prefer integer SIMD (treat f64 as u64 bits) over FP SIMD
- No special handling needed for NaN, Inf, or denormal values
- Codebase pattern: `/home/user/vortex/vortex-compute/src/take/slice/avx2.rs` treats f64 as u64

**Recommendations**:
| Priority | Optimization | Expected Gain | Notes |
|----------|--------------|---------------|-------|
| 1 | Sparse zero detection | 20-40% | Same as u64, use integer comparison |
| 2 | Integer SIMD (if explicit) | 5-15% | `_mm256_set1_epi64x` + `to_bits()` |
| Low | FP SIMD | ~5% | No benefit over integer approach |
| None | Special value detection | 0% | Not needed - no performance impact |

### i128/Decimal Optimizations (16-byte integers)

**Key findings**:
- Only 2 i128s fit in AVX2 (256-bit), 4 in AVX-512
- Cache line = 64 bytes = 4 i128s
- LLVM auto-vectorization is already good

**Recommendations**:
| Priority | Optimization | Expected Gain | Notes |
|----------|--------------|---------------|-------|
| 1 | Sparse zero detection | 20-50% | Highest ROI for decimal columns |
| 2 | Constant detection | 10-30% | Common in decimal columns |
| 3 | Cache-line processing | 5-10% | Process 4 i128s (64 bytes) at a time |
| Low | Explicit AVX-512 | 10-20% | Only if benchmarks justify |

### AVX-512 Masked Operations

**Key findings**:
- Masked stores can eliminate scalar remainder loops
- Benefit proportional to run boundary frequency
- Best for short runs (avg 8-64 elements)
- Overlapping stores are simpler alternative to masking

**When AVX-512 masking is worth it**:
- Target is Intel server CPUs (Skylake-X+)
- Workloads have many short runs (avg <64 elements)
- Run-end decode is a profiled bottleneck

**When to skip AVX-512**:
- Portability important (ARM, WASM)
- LLVM auto-vectorization sufficient
- Typical workloads have long runs (avg >1000)

### ARM SVE (Scalable Vector Extension)

**Key findings**:
- Vector length agnostic (128-2048 bits, hardware determines)
- Predicate registers eliminate remainder loops
- `svwhilelt_b64(start, end)` creates partial vector mask
- Good for Graviton3/4 deployment

```rust
// SVE pseudocode (nightly only)
let pred = svptrue_b64();           // All lanes active
let broadcast = svdup_n_u64(value); // Broadcast to all lanes
svst1_u64(pred, ptr, broadcast);    // Store with predicate
```

## Summary: Optimization Priorities

| Rank | Optimization | Types | Gain | Complexity | Recommended |
|------|--------------|-------|------|------------|-------------|
| 1 | **Sparse zero detection** | All | 20-50% | Low | **Yes** |
| 2 | **Constant value detection** | All | 10-30% | Low | **Yes** |
| 3 | Run length classification | All | 15-40% | Medium | For short runs |
| 4 | Non-temporal stores | u64+ | 20-40% | Medium | For >1MB fills |
| 5 | AVX-512 masking | All | 5-15% | High | Only if justified |
| 6 | Explicit SIMD | u8 | 10-30% | High | Rarely |

**Bottom line**: LLVM's auto-vectorization is excellent. Focus on **avoiding unnecessary work** (sparse zeros, constant values) rather than micro-optimizing the fill operation itself.
