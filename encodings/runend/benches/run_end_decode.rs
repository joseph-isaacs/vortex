// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::unwrap_used, clippy::cast_possible_truncation)]

use divan::Bencher;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::compute::warm_up_vtables;
use vortex_array::validity::Validity;
use vortex_buffer::BitBuffer;
use vortex_buffer::BufferMut;
use vortex_dtype::NativePType;
use vortex_runend::compress::runend_decode_bools;
use vortex_runend::compress::runend_decode_bools_original;
use vortex_runend::compress::runend_decode_primitive;
use vortex_runend::compress::runend_decode_primitive_original;

fn main() {
    warm_up_vtables();
    divan::main();
}

// Benchmark parameters: (total_length, avg_run_length)
// Short runs = more iterations, long runs = fewer iterations but bigger memsets
const DECODE_ARGS: &[(usize, usize)] = &[
    // Small arrays
    (1_000, 2),   // Very short runs (500 runs)
    (1_000, 10),  // Short runs (100 runs)
    (1_000, 100), // Medium runs (10 runs)
    (1_000, 500), // Long runs (2 runs)
    // Medium arrays
    (100_000, 2),     // Very short runs
    (100_000, 10),    // Short runs
    (100_000, 100),   // Medium runs
    (100_000, 1000),  // Long runs
    (100_000, 10000), // Very long runs
    // Large arrays
    (1_000_000, 2),      // Very short runs
    (1_000_000, 10),     // Short runs
    (1_000_000, 100),    // Medium runs
    (1_000_000, 1000),   // Long runs
    (1_000_000, 10000),  // Very long runs
    (1_000_000, 100000), // Extremely long runs
];

/// Creates run ends and values for primitive benchmarks
fn create_primitive_test_data<T: Clone + Default + From<u8> + NativePType>(
    total_length: usize,
    avg_run_length: usize,
) -> (PrimitiveArray, PrimitiveArray) {
    let mut ends = BufferMut::<u32>::with_capacity(total_length / avg_run_length + 1);
    let mut values = BufferMut::<T>::with_capacity(total_length / avg_run_length + 1);

    let mut pos = 0usize;
    let mut val: u8 = 0;
    while pos < total_length {
        let run_len = avg_run_length.min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);
        values.push(<T as From<u8>>::from(val));
        val = val.wrapping_add(1);
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        PrimitiveArray::new(values.freeze(), Validity::NonNullable),
    )
}

/// Creates run ends and values for bool benchmarks
fn create_bool_test_data(
    total_length: usize,
    avg_run_length: usize,
) -> (PrimitiveArray, BoolArray) {
    let mut ends = BufferMut::<u32>::with_capacity(total_length / avg_run_length + 1);
    let mut values = Vec::with_capacity(total_length / avg_run_length + 1);

    let mut pos = 0usize;
    let mut val = false;
    while pos < total_length {
        let run_len = avg_run_length.min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);
        values.push(val);
        val = !val;
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        BoolArray::from(BitBuffer::from(values)),
    )
}

/// Creates run ends and values with validity mask for primitive benchmarks
fn create_primitive_test_data_with_validity<T: Clone + Default + From<u8> + NativePType>(
    total_length: usize,
    avg_run_length: usize,
    null_density: f64,
) -> (PrimitiveArray, PrimitiveArray) {
    let mut ends = BufferMut::<u32>::with_capacity(total_length / avg_run_length + 1);
    let mut values = BufferMut::<T>::with_capacity(total_length / avg_run_length + 1);
    let mut validity_bits = Vec::with_capacity(total_length / avg_run_length + 1);

    let mut pos = 0usize;
    let mut val: u8 = 0;
    while pos < total_length {
        let run_len = avg_run_length.min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);
        values.push(<T as From<u8>>::from(val));
        // Deterministic "random" based on position
        let is_valid = ((pos as f64 / total_length as f64) * 100.0) as u64 % 100
            >= (null_density * 100.0) as u64;
        validity_bits.push(is_valid);
        val = val.wrapping_add(1);
    }

    let validity = Validity::from(BitBuffer::from(validity_bits));

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        PrimitiveArray::new(values.freeze(), validity),
    )
}

// ============================================================================
// Primitive decode benchmarks (non-nullable)
// ============================================================================

#[divan::bench(types = [u8, u16, u32, u64, i32, i64, f32, f64], args = DECODE_ARGS)]
fn decode_primitive<T: Clone + Default + From<u8> + NativePType>(
    bencher: Bencher,
    (total_length, avg_run_length): (usize, usize),
) {
    let (ends, values) = create_primitive_test_data::<T>(total_length, avg_run_length);

    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

// ============================================================================
// Primitive decode benchmarks (with validity)
// ============================================================================

#[divan::bench(types = [u32, u64], args = DECODE_ARGS)]
fn decode_primitive_with_validity<T: Clone + Default + From<u8> + NativePType>(
    bencher: Bencher,
    (total_length, avg_run_length): (usize, usize),
) {
    let (ends, values) =
        create_primitive_test_data_with_validity::<T>(total_length, avg_run_length, 0.1);

    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

// ============================================================================
// Bool decode benchmarks
// ============================================================================

#[divan::bench(args = DECODE_ARGS)]
fn decode_bool(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data(total_length, avg_run_length);

    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

// ============================================================================
// Offset decode benchmarks (testing sliced arrays)
// ============================================================================

#[divan::bench(args = [(1_000_000, 100), (1_000_000, 1000)])]
fn decode_primitive_with_offset(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_primitive_test_data::<u64>(total_length, avg_run_length);
    let offset = total_length / 4; // Start at 25%
    let length = total_length / 2; // Read 50%

    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), offset, length));
}

#[divan::bench(args = [(1_000_000, 100), (1_000_000, 1000)])]
fn decode_bool_with_offset(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data(total_length, avg_run_length);
    let offset = total_length / 4;
    let length = total_length / 2;

    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), offset, length));
}

// ============================================================================
// Throughput measurement benchmarks
// ============================================================================

/// Measure raw throughput in GB/s for primitive decoding
#[divan::bench(args = [(10_000_000, 100), (10_000_000, 1000), (10_000_000, 10000)])]
fn decode_primitive_throughput(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_primitive_test_data::<u64>(total_length, avg_run_length);

    bencher
        .counter(divan::counter::BytesCount::new(total_length * 8))
        .bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

// ============================================================================
// Bool decode benchmarks with different distributions
// ============================================================================

/// Distribution types for bool benchmarks
#[derive(Clone, Copy)]
enum BoolDistribution {
    /// Alternating true/false (50/50)
    Alternating,
    /// Mostly true (90% true runs)
    MostlyTrue,
    /// Mostly false (90% false runs)
    MostlyFalse,
    /// All true
    AllTrue,
    /// All false
    AllFalse,
}

/// Creates bool test data with configurable distribution
fn create_bool_test_data_with_distribution(
    total_length: usize,
    avg_run_length: usize,
    distribution: BoolDistribution,
) -> (PrimitiveArray, BoolArray) {
    let mut ends = BufferMut::<u32>::with_capacity(total_length / avg_run_length + 1);
    let mut values = Vec::with_capacity(total_length / avg_run_length + 1);

    let mut pos = 0usize;
    let mut run_index = 0usize;

    while pos < total_length {
        let run_len = avg_run_length.min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);

        let val = match distribution {
            BoolDistribution::Alternating => run_index % 2 == 0,
            BoolDistribution::MostlyTrue => run_index % 10 != 0, // 90% true
            BoolDistribution::MostlyFalse => run_index % 10 == 0, // 10% true (90% false)
            BoolDistribution::AllTrue => true,
            BoolDistribution::AllFalse => false,
        };
        values.push(val);
        run_index += 1;
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        BoolArray::from(BitBuffer::from(values)),
    )
}

// Bool distribution benchmark args: (total_length, avg_run_length)
const BOOL_DIST_ARGS: &[(usize, usize)] = &[
    (1_000_000, 2),     // Very short runs
    (1_000_000, 10),    // Short runs
    (1_000_000, 100),   // Medium runs
    (1_000_000, 1000),  // Long runs
    (1_000_000, 10000), // Very long runs
];

#[divan::bench(args = BOOL_DIST_ARGS)]
fn decode_bool_alternating(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::Alternating,
    );
    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = BOOL_DIST_ARGS)]
fn decode_bool_mostly_true(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::MostlyTrue,
    );
    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = BOOL_DIST_ARGS)]
fn decode_bool_mostly_false(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::MostlyFalse,
    );
    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = BOOL_DIST_ARGS)]
fn decode_bool_all_true(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::AllTrue,
    );
    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = BOOL_DIST_ARGS)]
fn decode_bool_all_false(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::AllFalse,
    );
    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

// ============================================================================
// Original vs Optimized comparison benchmarks
// ============================================================================

/// Comparison args for original vs optimized (1M elements with various run lengths)
const COMPARISON_ARGS: &[(usize, usize)] = &[
    (1_000_000, 2),    // Very short runs (500K runs)
    (1_000_000, 10),   // Short runs (100K runs)
    (1_000_000, 100),  // Medium runs (10K runs)
    (1_000_000, 1000), // Long runs (1K runs)
];

// --- Original implementation benchmarks ---

#[divan::bench(args = COMPARISON_ARGS)]
fn original_bool_alternating(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::Alternating,
    );
    bencher.bench(|| runend_decode_bools_original(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = COMPARISON_ARGS)]
fn original_bool_mostly_true(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::MostlyTrue,
    );
    bencher.bench(|| runend_decode_bools_original(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = COMPARISON_ARGS)]
fn original_bool_mostly_false(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::MostlyFalse,
    );
    bencher.bench(|| runend_decode_bools_original(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = COMPARISON_ARGS)]
fn original_bool_all_true(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::AllTrue,
    );
    bencher.bench(|| runend_decode_bools_original(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = COMPARISON_ARGS)]
fn original_bool_all_false(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::AllFalse,
    );
    bencher.bench(|| runend_decode_bools_original(ends.clone(), values.clone(), 0, total_length));
}

// --- Optimized implementation benchmarks (for side-by-side comparison) ---

#[divan::bench(args = COMPARISON_ARGS)]
fn optimized_bool_alternating(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::Alternating,
    );
    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = COMPARISON_ARGS)]
fn optimized_bool_mostly_true(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::MostlyTrue,
    );
    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = COMPARISON_ARGS)]
fn optimized_bool_mostly_false(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::MostlyFalse,
    );
    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = COMPARISON_ARGS)]
fn optimized_bool_all_true(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::AllTrue,
    );
    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = COMPARISON_ARGS)]
fn optimized_bool_all_false(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::AllFalse,
    );
    bencher.bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

// ============================================================================
// Primitive decode benchmarks with different value distributions
// ============================================================================
// Note: Run-end encoding is typically chosen when avg run length > 8
// These benchmarks focus on realistic scenarios

/// Value distribution types for primitive benchmarks
#[derive(Clone, Copy)]
enum PrimitiveDistribution {
    /// All runs have the same constant value (e.g., all zeros)
    Constant,
    /// Values increment sequentially (0, 1, 2, 3, ...)
    Sequential,
    /// Values alternate between 0 and a large value
    Binary,
    /// Mostly zeros with occasional non-zero (10% non-zero)
    SparseNonZero,
    /// Values follow a pattern that repeats
    Repeating,
    /// Every run has a unique value (modulo 256)
    AllDifferent,
    /// Pseudo-random values using simple LCG
    Random,
    /// Groups of N consecutive runs with the same value
    Clustered(usize),
}

/// Creates primitive test data with configurable value distribution
fn create_primitive_test_data_with_distribution<T>(
    total_length: usize,
    avg_run_length: usize,
    distribution: PrimitiveDistribution,
) -> (PrimitiveArray, PrimitiveArray)
where
    T: Clone + Default + NativePType,
    T: From<u8>,
{
    let mut ends = BufferMut::<u32>::with_capacity(total_length / avg_run_length + 1);
    let mut values = BufferMut::<T>::with_capacity(total_length / avg_run_length + 1);

    let mut pos = 0usize;
    let mut run_index = 0usize;

    while pos < total_length {
        let run_len = avg_run_length.min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);

        let val: T = match distribution {
            PrimitiveDistribution::Constant => <T as From<u8>>::from(42u8),
            PrimitiveDistribution::Sequential => <T as From<u8>>::from((run_index % 256) as u8),
            PrimitiveDistribution::Binary => {
                if run_index % 2 == 0 {
                    <T as From<u8>>::from(0u8)
                } else {
                    <T as From<u8>>::from(255u8)
                }
            }
            PrimitiveDistribution::SparseNonZero => {
                if run_index % 10 == 0 {
                    <T as From<u8>>::from(((run_index / 10) % 256) as u8)
                } else {
                    <T as From<u8>>::from(0u8)
                }
            }
            PrimitiveDistribution::Repeating => <T as From<u8>>::from((run_index % 4) as u8),
            PrimitiveDistribution::AllDifferent => <T as From<u8>>::from((run_index % 256) as u8),
            PrimitiveDistribution::Random => {
                let rand = (run_index.wrapping_mul(1103515245).wrapping_add(12345)) % 256;
                <T as From<u8>>::from(rand as u8)
            }
            PrimitiveDistribution::Clustered(cluster_size) => {
                let cluster_index = run_index / cluster_size;
                <T as From<u8>>::from((cluster_index % 256) as u8)
            }
        };
        values.push(val);
        run_index += 1;
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        PrimitiveArray::new(values.freeze(), Validity::NonNullable),
    )
}

// Benchmark args for realistic run-end scenarios (avg run len >= 8)
// Format: (total_length, avg_run_length)
const PRIMITIVE_DIST_ARGS: &[(usize, usize)] = &[
    (1_000_000, 8),     // Minimum practical for run-end
    (1_000_000, 16),    // Short runs
    (1_000_000, 64),    // Medium runs
    (1_000_000, 256),   // Long runs
    (1_000_000, 1024),  // Very long runs
    (1_000_000, 10000), // Extremely long runs
];

// --- Constant value distribution ---

#[divan::bench(types = [u8, u32, u64], args = PRIMITIVE_DIST_ARGS)]
fn decode_primitive_constant<T>(bencher: Bencher, (total_length, avg_run_length): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Constant,
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

// --- Sequential value distribution ---

#[divan::bench(types = [u8, u32, u64], args = PRIMITIVE_DIST_ARGS)]
fn decode_primitive_sequential<T>(bencher: Bencher, (total_length, avg_run_length): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Sequential,
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

// --- Binary value distribution (0 and max alternating) ---

#[divan::bench(types = [u8, u32, u64], args = PRIMITIVE_DIST_ARGS)]
fn decode_primitive_binary<T>(bencher: Bencher, (total_length, avg_run_length): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Binary,
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

// --- Sparse non-zero distribution (90% zeros) ---

#[divan::bench(types = [u8, u32, u64], args = PRIMITIVE_DIST_ARGS)]
fn decode_primitive_sparse<T>(bencher: Bencher, (total_length, avg_run_length): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::SparseNonZero,
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

// --- Repeating pattern distribution ---

#[divan::bench(types = [u8, u32, u64], args = PRIMITIVE_DIST_ARGS)]
fn decode_primitive_repeating<T>(bencher: Bencher, (total_length, avg_run_length): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Repeating,
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

// ============================================================================
// Large array benchmarks (testing memory bandwidth limits)
// ============================================================================

const LARGE_ARRAY_ARGS: &[(usize, usize)] = &[
    (10_000_000, 100),   // 10M elements, medium runs
    (10_000_000, 1000),  // 10M elements, long runs
    (10_000_000, 10000), // 10M elements, very long runs
];

#[divan::bench(types = [u32, u64], args = LARGE_ARRAY_ARGS)]
fn decode_primitive_large<T>(bencher: Bencher, (total_length, avg_run_length): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Sequential,
    );
    bencher
        .counter(divan::counter::BytesCount::new(
            total_length * size_of::<T>(),
        ))
        .bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

#[divan::bench(args = LARGE_ARRAY_ARGS)]
fn decode_bool_large(bencher: Bencher, (total_length, avg_run_length): (usize, usize)) {
    let (ends, values) = create_bool_test_data_with_distribution(
        total_length,
        avg_run_length,
        BoolDistribution::Alternating,
    );
    bencher
        .counter(divan::counter::BytesCount::new(total_length / 8))
        .bench(|| runend_decode_bools(ends.clone(), values.clone(), 0, total_length));
}

// ============================================================================
// Constant value optimization benchmarks
// ============================================================================
// These benchmarks specifically test the constant value fast path optimization
// which uses a single fill() call instead of per-run fills when all values are identical.

/// Benchmark args for constant value optimization comparison
/// Format: (total_length, num_runs)
const CONSTANT_OPT_ARGS: &[(usize, usize)] = &[
    (1_000_000, 10),      // 10 runs of 100K each
    (1_000_000, 100),     // 100 runs of 10K each
    (1_000_000, 1000),    // 1K runs of 1K each
    (1_000_000, 10000),   // 10K runs of 100 each
    (1_000_000, 100000),  // 100K runs of 10 each
    (1_000_000, 1000000), // 1M runs of 1 each
];

/// Creates constant value test data (all values are 42)
fn create_constant_value_data<T>(
    total_length: usize,
    num_runs: usize,
) -> (PrimitiveArray, PrimitiveArray)
where
    T: Clone + Default + NativePType + From<u8>,
{
    let mut ends = BufferMut::<u32>::with_capacity(num_runs);
    let mut values = BufferMut::<T>::with_capacity(num_runs);

    let run_length = total_length / num_runs;
    let mut pos = 0usize;

    for i in 0..num_runs {
        pos += if i == num_runs - 1 {
            total_length - pos
        } else {
            run_length
        };
        ends.push(pos as u32);
        values.push(<T as From<u8>>::from(42u8)); // All same value
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        PrimitiveArray::new(values.freeze(), Validity::NonNullable),
    )
}

/// Creates non-constant value test data (values alternate between 0 and 255)
fn create_non_constant_value_data<T>(
    total_length: usize,
    num_runs: usize,
) -> (PrimitiveArray, PrimitiveArray)
where
    T: Clone + Default + NativePType + From<u8>,
{
    let mut ends = BufferMut::<u32>::with_capacity(num_runs);
    let mut values = BufferMut::<T>::with_capacity(num_runs);

    let run_length = total_length / num_runs;
    let mut pos = 0usize;

    for i in 0..num_runs {
        pos += if i == num_runs - 1 {
            total_length - pos
        } else {
            run_length
        };
        ends.push(pos as u32);
        // Alternating values to ensure non-constant
        values.push(<T as From<u8>>::from(if i % 2 == 0 { 0u8 } else { 255u8 }));
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        PrimitiveArray::new(values.freeze(), Validity::NonNullable),
    )
}

/// Benchmark constant value decoding (uses fast path with single fill)
#[divan::bench(types = [u32, u64], args = CONSTANT_OPT_ARGS)]
fn decode_constant_values<T>(bencher: Bencher, (total_length, num_runs): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_constant_value_data::<T>(total_length, num_runs);
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

/// Benchmark non-constant value decoding (uses normal per-run fill path)
#[divan::bench(types = [u32, u64], args = CONSTANT_OPT_ARGS)]
fn decode_non_constant_values<T>(bencher: Bencher, (total_length, num_runs): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_non_constant_value_data::<T>(total_length, num_runs);
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

// ============================================================================
// Variable run length benchmarks (simulating real-world irregular patterns)
// ============================================================================

/// Creates test data with variable run lengths (not uniform)
fn create_variable_run_data<T>(
    total_length: usize,
    avg_run_length: usize,
) -> (PrimitiveArray, PrimitiveArray)
where
    T: Clone + Default + NativePType + From<u8>,
{
    let mut ends = BufferMut::<u32>::with_capacity(total_length / avg_run_length + 1);
    let mut values = BufferMut::<T>::with_capacity(total_length / avg_run_length + 1);

    let mut pos = 0usize;
    let mut run_index = 0usize;

    // Use a simple pseudo-random pattern for variable run lengths
    // Pattern: short, medium, long, very long, repeat
    let run_multipliers = [0.25, 0.5, 1.0, 2.0, 4.0];

    while pos < total_length {
        let multiplier = run_multipliers[run_index % run_multipliers.len()];
        let run_len = ((avg_run_length as f64 * multiplier) as usize)
            .max(1)
            .min(total_length - pos);
        pos += run_len;
        ends.push(pos as u32);
        values.push(<T as From<u8>>::from((run_index % 256) as u8));
        run_index += 1;
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        PrimitiveArray::new(values.freeze(), Validity::NonNullable),
    )
}

const VARIABLE_RUN_ARGS: &[(usize, usize)] = &[
    (1_000_000, 16),   // Avg 16, actual varies 4-64
    (1_000_000, 64),   // Avg 64, actual varies 16-256
    (1_000_000, 256),  // Avg 256, actual varies 64-1024
    (1_000_000, 1024), // Avg 1024, actual varies 256-4096
];

#[divan::bench(types = [u32, u64], args = VARIABLE_RUN_ARGS)]
fn decode_primitive_variable_runs<T>(
    bencher: Bencher,
    (total_length, avg_run_length): (usize, usize),
) where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_variable_run_data::<T>(total_length, avg_run_length);
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

// ============================================================================
// Sparse zero optimization benchmarks
// ============================================================================
// These benchmarks compare the performance of decoding data with different
// proportions of zero values. When >50% of run values are zero, we use a
// zeroed buffer and skip zero fills during decoding.

/// Benchmark args for sparse zero optimization: (total_length, num_runs, zero_percentage)
const SPARSE_ZERO_ARGS: &[(usize, usize)] = &[
    (1_000_000, 100),   // 100 runs of 10K each
    (1_000_000, 1000),  // 1K runs of 1K each
    (1_000_000, 10000), // 10K runs of 100 each
];

/// Creates sparse data with a given percentage of zeros
fn create_sparse_data<T>(
    total_length: usize,
    num_runs: usize,
    zero_percentage: usize, // 0-100
) -> (PrimitiveArray, PrimitiveArray)
where
    T: Clone + Default + NativePType + From<u8>,
{
    let mut ends = BufferMut::<u32>::with_capacity(num_runs);
    let mut values = BufferMut::<T>::with_capacity(num_runs);

    let run_length = total_length / num_runs;
    let mut pos = 0usize;

    for i in 0..num_runs {
        pos += if i == num_runs - 1 {
            total_length - pos
        } else {
            run_length
        };
        ends.push(pos as u32);
        // Deterministic pattern: first zero_percentage% runs are zeros
        let is_zero = (i * 100 / num_runs) < zero_percentage;
        values.push(if is_zero {
            T::default()
        } else {
            <T as From<u8>>::from(((i % 255) + 1) as u8)
        });
    }

    (
        PrimitiveArray::new(ends.freeze(), Validity::NonNullable),
        PrimitiveArray::new(values.freeze(), Validity::NonNullable),
    )
}

/// Benchmark sparse data with 90% zeros (triggers sparse zero optimization)
#[divan::bench(types = [u32, u64], args = SPARSE_ZERO_ARGS)]
fn decode_sparse_90_percent_zeros<T>(bencher: Bencher, (total_length, num_runs): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_sparse_data::<T>(total_length, num_runs, 90);
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

/// Benchmark sparse data with 60% zeros (triggers sparse zero optimization)
#[divan::bench(types = [u32, u64], args = SPARSE_ZERO_ARGS)]
fn decode_sparse_60_percent_zeros<T>(bencher: Bencher, (total_length, num_runs): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_sparse_data::<T>(total_length, num_runs, 60);
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

/// Benchmark data with 50% zeros (boundary case - does NOT trigger optimization)
#[divan::bench(types = [u32, u64], args = SPARSE_ZERO_ARGS)]
fn decode_50_percent_zeros<T>(bencher: Bencher, (total_length, num_runs): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_sparse_data::<T>(total_length, num_runs, 50);
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

/// Benchmark dense data with 10% zeros (does NOT trigger sparse optimization)
#[divan::bench(types = [u32, u64], args = SPARSE_ZERO_ARGS)]
fn decode_dense_10_percent_zeros<T>(bencher: Bencher, (total_length, num_runs): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_sparse_data::<T>(total_length, num_runs, 10);
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

// ============================================================================
// All-different and varied value distribution benchmarks
// ============================================================================
// These benchmarks test decoding performance when run values are highly varied,
// not constant or sparse. This represents real-world scenarios where the data
// has many unique values.

// Short runs with varied values (high cardinality scenarios)
// Format: (total_length, avg_run_length)
const SHORT_RUN_ARGS: &[(usize, usize)] = &[
    (1_000_000, 16), // ~62,500 runs
    (1_000_000, 24), // ~41,666 runs
    (1_000_000, 32), // ~31,250 runs
];

// Medium runs with varied values
// Format: (total_length, avg_run_length)
const MEDIUM_RUN_ARGS: &[(usize, usize)] = &[
    (1_000_000, 64),  // ~15,625 runs
    (1_000_000, 100), // ~10,000 runs
    (1_000_000, 256), // ~3,906 runs
];

// --- All different value distribution (unique value per run) ---

/// Benchmark with all different values - short runs
#[divan::bench(types = [u8, u32, u64], args = SHORT_RUN_ARGS)]
fn decode_primitive_all_different_short<T>(
    bencher: Bencher,
    (total_length, avg_run_length): (usize, usize),
) where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::AllDifferent,
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

/// Benchmark with all different values - medium runs
#[divan::bench(types = [u8, u32, u64], args = MEDIUM_RUN_ARGS)]
fn decode_primitive_all_different_medium<T>(
    bencher: Bencher,
    (total_length, avg_run_length): (usize, usize),
) where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::AllDifferent,
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

// --- Clustered value distribution (groups of N runs with same value) ---

// Clustered benchmark args combining short and medium runs
const CLUSTERED_ARGS: &[(usize, usize)] = &[
    // Short runs
    (1_000_000, 16),
    (1_000_000, 24),
    (1_000_000, 32),
    // Medium runs
    (1_000_000, 64),
    (1_000_000, 100),
    (1_000_000, 256),
];

/// Benchmark with clustered values - cluster size 4
/// Each value repeats for 4 consecutive runs before changing
#[divan::bench(types = [u8, u32, u64], args = CLUSTERED_ARGS)]
fn decode_primitive_clustered_4<T>(bencher: Bencher, (total_length, avg_run_length): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Clustered(4),
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

/// Benchmark with clustered values - cluster size 8
/// Each value repeats for 8 consecutive runs before changing
#[divan::bench(types = [u8, u32, u64], args = CLUSTERED_ARGS)]
fn decode_primitive_clustered_8<T>(bencher: Bencher, (total_length, avg_run_length): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Clustered(8),
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

/// Benchmark with clustered values - cluster size 16
/// Each value repeats for 16 consecutive runs before changing
#[divan::bench(types = [u8, u32, u64], args = CLUSTERED_ARGS)]
fn decode_primitive_clustered_16<T>(
    bencher: Bencher,
    (total_length, avg_run_length): (usize, usize),
) where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Clustered(16),
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

// --- Clustered vs Unclustered comparison benchmarks ---
// These benchmarks compare clustered data (consecutive runs with same value)
// against unclustered data (every run has a different value) to demonstrate
// the benefit of the clustered values optimization.

/// Benchmark args for clustered vs unclustered comparison
const CLUSTERED_COMPARISON_ARGS: &[(usize, usize)] = &[
    (1_000_000, 16),  // avg run 16, ~62.5K runs
    (1_000_000, 32),  // avg run 32, ~31.25K runs
    (1_000_000, 64),  // avg run 64, ~15.6K runs
    (1_000_000, 128), // avg run 128, ~7.8K runs
];

/// Baseline: unclustered data - every run has a different value
#[divan::bench(types = [u32, u64], args = CLUSTERED_COMPARISON_ARGS)]
fn decode_unclustered_baseline<T>(bencher: Bencher, (total_length, avg_run_length): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::AllDifferent,
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

/// Clustered: 4 consecutive runs share the same value
#[divan::bench(types = [u32, u64], args = CLUSTERED_COMPARISON_ARGS)]
fn decode_clustered_4_optimized<T>(bencher: Bencher, (total_length, avg_run_length): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Clustered(4),
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

/// Clustered: 8 consecutive runs share the same value
#[divan::bench(types = [u32, u64], args = CLUSTERED_COMPARISON_ARGS)]
fn decode_clustered_8_optimized<T>(bencher: Bencher, (total_length, avg_run_length): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Clustered(8),
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

/// Clustered: 16 consecutive runs share the same value
#[divan::bench(types = [u32, u64], args = CLUSTERED_COMPARISON_ARGS)]
fn decode_clustered_16_optimized<T>(
    bencher: Bencher,
    (total_length, avg_run_length): (usize, usize),
) where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Clustered(16),
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

// --- Random value distribution (pseudo-random LCG values) ---

/// Benchmark with pseudo-random values - short runs
#[divan::bench(types = [u8, u32, u64], args = SHORT_RUN_ARGS)]
fn decode_primitive_random_short<T>(
    bencher: Bencher,
    (total_length, avg_run_length): (usize, usize),
) where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Random,
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

/// Benchmark with pseudo-random values - medium runs
#[divan::bench(types = [u8, u32, u64], args = MEDIUM_RUN_ARGS)]
fn decode_primitive_random_medium<T>(
    bencher: Bencher,
    (total_length, avg_run_length): (usize, usize),
) where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Random,
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

// ============================================================================
// Original vs Optimized primitive decode comparison benchmarks
// ============================================================================
// These benchmarks compare the original (simple) implementation against the
// optimized implementation which uses:
// - Constant value detection
// - Sparse zero optimization
// - Clustered values optimization
// - Short run fill optimization (fill_short)

/// Benchmark args for short-run comparison: (total_length, avg_run_length)
/// Focus on the short run case (16-32 elements average) where overhead matters most
const SHORT_RUN_COMPARISON_ARGS: &[(usize, usize)] = &[
    (1_000_000, 16), // ~62,500 runs - short runs
    (1_000_000, 24), // ~41,666 runs
    (1_000_000, 32), // ~31,250 runs
    (1_000_000, 64), // ~15,625 runs - medium runs for comparison
];

// --- Original implementation benchmarks for short runs ---

/// Original implementation: short runs with all different values
#[divan::bench(types = [u32, u64], args = SHORT_RUN_COMPARISON_ARGS)]
fn original_primitive_all_different<T>(
    bencher: Bencher,
    (total_length, avg_run_length): (usize, usize),
) where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::AllDifferent,
    );
    bencher
        .bench(|| runend_decode_primitive_original(ends.clone(), values.clone(), 0, total_length));
}

/// Optimized implementation: short runs with all different values
#[divan::bench(types = [u32, u64], args = SHORT_RUN_COMPARISON_ARGS)]
fn optimized_primitive_all_different<T>(
    bencher: Bencher,
    (total_length, avg_run_length): (usize, usize),
) where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::AllDifferent,
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

/// Original implementation: sequential values (0, 1, 2, 3, ...)
#[divan::bench(types = [u32, u64], args = SHORT_RUN_COMPARISON_ARGS)]
fn original_primitive_sequential<T>(
    bencher: Bencher,
    (total_length, avg_run_length): (usize, usize),
) where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Sequential,
    );
    bencher
        .bench(|| runend_decode_primitive_original(ends.clone(), values.clone(), 0, total_length));
}

/// Optimized implementation: sequential values (0, 1, 2, 3, ...)
#[divan::bench(types = [u32, u64], args = SHORT_RUN_COMPARISON_ARGS)]
fn optimized_primitive_sequential<T>(
    bencher: Bencher,
    (total_length, avg_run_length): (usize, usize),
) where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Sequential,
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

/// Original implementation: constant values (all 42)
#[divan::bench(types = [u32, u64], args = SHORT_RUN_COMPARISON_ARGS)]
fn original_primitive_constant<T>(bencher: Bencher, (total_length, avg_run_length): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Constant,
    );
    bencher
        .bench(|| runend_decode_primitive_original(ends.clone(), values.clone(), 0, total_length));
}

/// Optimized implementation: constant values (all 42) - should use fast path
#[divan::bench(types = [u32, u64], args = SHORT_RUN_COMPARISON_ARGS)]
fn optimized_primitive_constant<T>(bencher: Bencher, (total_length, avg_run_length): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::Constant,
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}

/// Original implementation: sparse data (90% zeros)
#[divan::bench(types = [u32, u64], args = SHORT_RUN_COMPARISON_ARGS)]
fn original_primitive_sparse<T>(bencher: Bencher, (total_length, avg_run_length): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::SparseNonZero,
    );
    bencher
        .bench(|| runend_decode_primitive_original(ends.clone(), values.clone(), 0, total_length));
}

/// Optimized implementation: sparse data (90% zeros) - should use zeroed buffer optimization
#[divan::bench(types = [u32, u64], args = SHORT_RUN_COMPARISON_ARGS)]
fn optimized_primitive_sparse<T>(bencher: Bencher, (total_length, avg_run_length): (usize, usize))
where
    T: Clone + Default + NativePType + From<u8>,
{
    let (ends, values) = create_primitive_test_data_with_distribution::<T>(
        total_length,
        avg_run_length,
        PrimitiveDistribution::SparseNonZero,
    );
    bencher.bench(|| runend_decode_primitive(ends.clone(), values.clone(), 0, total_length));
}
