// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmarks for dictionary take operation with primitive and varbinview values.
//!
//! Tests different dictionary cardinalities (4, 16, 255) and various data distributions
//! (uniform, zipfian) to understand performance characteristics.

#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::unwrap_used)]
#![allow(unused_imports)]

use divan::Bencher;
use rand::distr::Uniform;
use rand::prelude::*;
use rand_distr::Zipf;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::DictArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::compute::warm_up_vtables;
use vortex_dtype::DType;
use vortex_dtype::Nullability;

fn main() {
    warm_up_vtables();
    divan::main();
}

/// Number of codes (indices to take).
const NUM_CODES: &[usize] = &[1_000, 10_000, 100_000];

/// Dictionary cardinalities to test (as requested: 4, 16, 255).
const NUM_VALUES: &[usize] = &[4, 16, 255];

// =============================================================================
// Primitive Dictionary Take Benchmarks
// =============================================================================

mod primitive {
    use super::*;

    /// Generate a dictionary array with u32 primitive values.
    fn gen_primitive_dict_uniform(num_values: usize, num_codes: usize) -> DictArray {
        let values = PrimitiveArray::from_iter(0..num_values as u32);

        let rng = StdRng::seed_from_u64(42);
        let range = Uniform::new(0u32, num_values as u32).unwrap();
        let codes = PrimitiveArray::from_iter(rng.sample_iter(range).take(num_codes));

        DictArray::try_new(codes.into_array(), values.into_array()).unwrap()
    }

    fn gen_primitive_dict_zipfian(num_values: usize, num_codes: usize) -> DictArray {
        let values = PrimitiveArray::from_iter(0..num_values as u32);

        let rng = StdRng::seed_from_u64(42);
        let zipf = Zipf::new(num_values as f64, 1.0).unwrap();
        let codes = PrimitiveArray::from_iter(
            rng.sample_iter(&zipf)
                .take(num_codes)
                .map(|i: f64| (i as u32 - 1).min(num_values as u32 - 1)),
        );

        DictArray::try_new(codes.into_array(), values.into_array()).unwrap()
    }

    fn gen_primitive_dict_sequential(num_values: usize, num_codes: usize) -> DictArray {
        let values = PrimitiveArray::from_iter(0..num_values as u32);

        // Sequential codes (good cache locality)
        let codes = PrimitiveArray::from_iter((0..num_codes).map(|i| (i % num_values) as u32));

        DictArray::try_new(codes.into_array(), values.into_array()).unwrap()
    }

    #[divan::bench(args = NUM_CODES, consts = NUM_VALUES)]
    fn canonicalize_uniform<const NUM_VALUES: usize>(bencher: Bencher, num_codes: usize) {
        let dict = gen_primitive_dict_uniform(NUM_VALUES, num_codes);
        bencher
            .with_inputs(|| &dict)
            .bench_refs(|dict| dict.to_canonical());
    }

    #[divan::bench(args = NUM_CODES, consts = NUM_VALUES)]
    fn canonicalize_zipfian<const NUM_VALUES: usize>(bencher: Bencher, num_codes: usize) {
        let dict = gen_primitive_dict_zipfian(NUM_VALUES, num_codes);
        bencher
            .with_inputs(|| &dict)
            .bench_refs(|dict| dict.to_canonical());
    }

    #[divan::bench(args = NUM_CODES, consts = NUM_VALUES)]
    fn canonicalize_sequential<const NUM_VALUES: usize>(bencher: Bencher, num_codes: usize) {
        let dict = gen_primitive_dict_sequential(NUM_VALUES, num_codes);
        bencher
            .with_inputs(|| &dict)
            .bench_refs(|dict| dict.to_canonical());
    }

    /// Benchmark with i64 values (8 bytes) to test larger value types.
    #[divan::bench(args = NUM_CODES, consts = NUM_VALUES)]
    fn canonicalize_i64_uniform<const NUM_VALUES: usize>(bencher: Bencher, num_codes: usize) {
        let values = PrimitiveArray::from_iter((0..NUM_VALUES).map(|i| i as i64 * 1000));

        let rng = StdRng::seed_from_u64(42);
        let range = Uniform::new(0u32, NUM_VALUES as u32).unwrap();
        let codes = PrimitiveArray::from_iter(rng.sample_iter(range).take(num_codes));

        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();

        bencher
            .with_inputs(|| &dict)
            .bench_refs(|dict| dict.to_canonical());
    }
}

// =============================================================================
// VarBinView Dictionary Take Benchmarks
// =============================================================================

mod varbinview {
    use super::*;

    /// Generate short strings (inlined in BinaryView, <= 12 bytes).
    fn gen_short_strings(num_values: usize) -> Vec<String> {
        (0..num_values).map(|i| format!("val_{i:04}")).collect()
    }

    /// Generate medium strings (16 bytes, may or may not be inlined).
    fn gen_medium_strings(num_values: usize) -> Vec<String> {
        (0..num_values)
            .map(|i| format!("medium_value_{i:06}"))
            .collect()
    }

    /// Generate long strings (> 16 bytes, always out-of-line).
    fn gen_long_strings(num_values: usize) -> Vec<String> {
        (0..num_values)
            .map(|i| format!("this_is_a_longer_string_value_{i:08}"))
            .collect()
    }

    fn gen_varbinview_dict(strings: &[String], num_codes: usize, distribution: &str) -> DictArray {
        let values = VarBinViewArray::from_iter(
            strings.iter().map(|s| Some(s.as_str())),
            DType::Utf8(Nullability::NonNullable),
        );

        let num_values = strings.len();
        let codes: Vec<u32> = match distribution {
            "uniform" => {
                let rng = StdRng::seed_from_u64(42);
                let range = Uniform::new(0u32, num_values as u32).unwrap();
                rng.sample_iter(range).take(num_codes).collect()
            }
            "zipfian" => {
                let rng = StdRng::seed_from_u64(42);
                let zipf = Zipf::new(num_values as f64, 1.0).unwrap();
                rng.sample_iter(&zipf)
                    .take(num_codes)
                    .map(|i: f64| (i as u32 - 1).min(num_values as u32 - 1))
                    .collect()
            }
            "sequential" => (0..num_codes).map(|i| (i % num_values) as u32).collect(),
            _ => unreachable!(),
        };

        let codes_arr = PrimitiveArray::from_iter(codes);
        DictArray::try_new(codes_arr.into_array(), values.into_array()).unwrap()
    }

    // --- Short strings (inlined) ---

    #[divan::bench(args = NUM_CODES, consts = NUM_VALUES)]
    fn short_uniform<const NUM_VALUES: usize>(bencher: Bencher, num_codes: usize) {
        let strings = gen_short_strings(NUM_VALUES);
        let dict = gen_varbinview_dict(&strings, num_codes, "uniform");
        bencher
            .with_inputs(|| &dict)
            .bench_refs(|dict| dict.to_canonical());
    }

    #[divan::bench(args = NUM_CODES, consts = NUM_VALUES)]
    fn short_zipfian<const NUM_VALUES: usize>(bencher: Bencher, num_codes: usize) {
        let strings = gen_short_strings(NUM_VALUES);
        let dict = gen_varbinview_dict(&strings, num_codes, "zipfian");
        bencher
            .with_inputs(|| &dict)
            .bench_refs(|dict| dict.to_canonical());
    }

    #[divan::bench(args = NUM_CODES, consts = NUM_VALUES)]
    fn short_sequential<const NUM_VALUES: usize>(bencher: Bencher, num_codes: usize) {
        let strings = gen_short_strings(NUM_VALUES);
        let dict = gen_varbinview_dict(&strings, num_codes, "sequential");
        bencher
            .with_inputs(|| &dict)
            .bench_refs(|dict| dict.to_canonical());
    }

    // --- Medium strings (~16 bytes) ---

    #[divan::bench(args = NUM_CODES, consts = NUM_VALUES)]
    fn medium_uniform<const NUM_VALUES: usize>(bencher: Bencher, num_codes: usize) {
        let strings = gen_medium_strings(NUM_VALUES);
        let dict = gen_varbinview_dict(&strings, num_codes, "uniform");
        bencher
            .with_inputs(|| &dict)
            .bench_refs(|dict| dict.to_canonical());
    }

    #[divan::bench(args = NUM_CODES, consts = NUM_VALUES)]
    fn medium_zipfian<const NUM_VALUES: usize>(bencher: Bencher, num_codes: usize) {
        let strings = gen_medium_strings(NUM_VALUES);
        let dict = gen_varbinview_dict(&strings, num_codes, "zipfian");
        bencher
            .with_inputs(|| &dict)
            .bench_refs(|dict| dict.to_canonical());
    }

    // --- Long strings (out-of-line) ---

    #[divan::bench(args = NUM_CODES, consts = NUM_VALUES)]
    fn long_uniform<const NUM_VALUES: usize>(bencher: Bencher, num_codes: usize) {
        let strings = gen_long_strings(NUM_VALUES);
        let dict = gen_varbinview_dict(&strings, num_codes, "uniform");
        bencher
            .with_inputs(|| &dict)
            .bench_refs(|dict| dict.to_canonical());
    }

    #[divan::bench(args = NUM_CODES, consts = NUM_VALUES)]
    fn long_zipfian<const NUM_VALUES: usize>(bencher: Bencher, num_codes: usize) {
        let strings = gen_long_strings(NUM_VALUES);
        let dict = gen_varbinview_dict(&strings, num_codes, "zipfian");
        bencher
            .with_inputs(|| &dict)
            .bench_refs(|dict| dict.to_canonical());
    }
}

// =============================================================================
// Direct take_canonical benchmarks (bypassing dict canonicalization path)
// =============================================================================

mod execute {
    use vortex_array::Canonical;
    use vortex_array::arrays::take_canonical;

    use super::*;

    /// Benchmark the raw take_canonical function for primitive values.
    #[divan::bench(args = NUM_CODES, consts = NUM_VALUES)]
    fn primitive_take_canonical<const NUM_VALUES: usize>(bencher: Bencher, num_codes: usize) {
        let values = PrimitiveArray::from_iter(0..NUM_VALUES as u32);

        let rng = StdRng::seed_from_u64(42);
        let range = Uniform::new(0u32, NUM_VALUES as u32).unwrap();
        let codes = PrimitiveArray::from_iter(rng.sample_iter(range).take(num_codes));

        let canonical_values = Canonical::Primitive(values);

        bencher
            .with_inputs(|| (canonical_values.clone(), &codes))
            .bench_refs(|(values, codes)| take_canonical(values.clone(), codes));
    }

    /// Benchmark the raw take_canonical function for varbinview values.
    #[divan::bench(args = NUM_CODES, consts = NUM_VALUES)]
    fn varbinview_take_canonical<const NUM_VALUES: usize>(bencher: Bencher, num_codes: usize) {
        let strings: Vec<String> = (0..NUM_VALUES).map(|i| format!("val_{i:04}")).collect();
        let values = VarBinViewArray::from_iter(
            strings.iter().map(|s| Some(s.as_str())),
            DType::Utf8(Nullability::NonNullable),
        );

        let rng = StdRng::seed_from_u64(42);
        let range = Uniform::new(0u32, NUM_VALUES as u32).unwrap();
        let codes = PrimitiveArray::from_iter(rng.sample_iter(range).take(num_codes));

        let canonical_values = Canonical::VarBinView(values);

        bencher
            .with_inputs(|| (canonical_values.clone(), &codes))
            .bench_refs(|(values, codes)| take_canonical(values.clone(), codes));
    }
}
