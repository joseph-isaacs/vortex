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

## Future Work

1. **Primitive distribution benchmarks**: Test with sparse values, constant runs, etc.
2. **Value-based optimizations**: Detect constant values for primitive decode
3. **Memory prefetching**: For very large arrays (>1MB)
4. **Non-temporal stores**: For large fills that exceed cache
