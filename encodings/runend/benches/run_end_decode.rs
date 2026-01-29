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
use vortex_runend::compress::runend_decode_primitive;

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
