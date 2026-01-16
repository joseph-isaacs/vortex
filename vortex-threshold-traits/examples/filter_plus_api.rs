//! FILTER/PLUS OPTIMIZATION EXAMPLE
//!
//! This benchmark finds the optimal strategy for combining filter and plus operations,
//! parameterized by both mask density AND element type.
//!
//! Problem:
//!   Given arrays A, B and a boolean mask M, compute: (A + B) filtered by M
//!
//! Strategies:
//!   1. filter_then_add: filter(A, M) + filter(B, M)
//!   2. add_then_filter: filter(A + B, M)
//!
//! Trade-offs depend on:
//!   - Mask density (true count / total count)
//!   - Element type (affects SIMD lane width, memory bandwidth)
//!   - Array length
//!
//! The benchmark finds the optimal strategy empirically across these dimensions.

#![allow(
    dead_code,
    unused_imports,
    unused_variables,
    clippy::type_complexity,
    clippy::needless_doctest_main
)]

use rand::Rng;
use rand::SeedableRng;
use std::fmt::Debug;

// ============================================================================
// TARGET API - What we want to implement
// ============================================================================

/*
use vortex_threshold::{threshold_bench, Benchmark, ParamGrid, StatsBench};

// Parameterize by element type using separate benchmarks
#[threshold_bench]
fn filter_plus_u8() -> impl Benchmark {
    filter_plus_bench::<u8>("filter_plus_u8")
}

#[threshold_bench]
fn filter_plus_u16() -> impl Benchmark {
    filter_plus_bench::<u16>("filter_plus_u16")
}

#[threshold_bench]
fn filter_plus_u32() -> impl Benchmark {
    filter_plus_bench::<u32>("filter_plus_u32")
}

#[threshold_bench]
fn filter_plus_u64() -> impl Benchmark {
    filter_plus_bench::<u64>("filter_plus_u64")
}

fn filter_plus_bench<T: Element>(name: &str) -> impl Benchmark {
    StatsBench::new(name)
        // 1. Distributions - varying mask densities
        .distribution("very_sparse", |seed| gen_data::<T>(seed, 0.01))
        .distribution("sparse", |seed| gen_data::<T>(seed, 0.1))
        .distribution("medium", |seed| gen_data::<T>(seed, 0.5))
        .distribution("dense", |seed| gen_data::<T>(seed, 0.9))
        .distribution("very_dense", |seed| gen_data::<T>(seed, 0.99))

        .weight("medium", 2.0)
        .weight("sparse", 1.5)

        // 2. Stats
        .stats(FilterPlusStats::compute)

        // 3. Variants
        .baseline("add_then_filter", |data, stats| add_then_filter(data))
        .variant("filter_then_add", |data, stats| filter_then_add(data))
        .variant("adaptive", |data, stats| {
            if stats.mask_density < stats.density_threshold {
                filter_then_add(data)
            } else {
                add_then_filter(data)
            }
        })

        .build()
}

fn main() {
    threshold_runner::main();
}
*/

// ============================================================================
// Element Trait - Abstraction over primitive types
// ============================================================================

/// Trait for element types that can be used in filter/plus benchmarks
trait Element: Copy + Clone + Debug + Default + 'static {
    fn random(rng: &mut impl Rng) -> Self;
    fn type_name() -> &'static str;
    /// Wrapping add to avoid overflow panics in debug mode
    fn wrapping_add(self, other: Self) -> Self;
}

impl Element for u8 {
    fn random(rng: &mut impl Rng) -> Self {
        rng.random()
    }
    fn type_name() -> &'static str {
        "u8"
    }
    fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

impl Element for u16 {
    fn random(rng: &mut impl Rng) -> Self {
        rng.random()
    }
    fn type_name() -> &'static str {
        "u16"
    }
    fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

impl Element for u32 {
    fn random(rng: &mut impl Rng) -> Self {
        rng.random()
    }
    fn type_name() -> &'static str {
        "u32"
    }
    fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

impl Element for u64 {
    fn random(rng: &mut impl Rng) -> Self {
        rng.random()
    }
    fn type_name() -> &'static str {
        "u64"
    }
    fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

impl Element for i8 {
    fn random(rng: &mut impl Rng) -> Self {
        rng.random()
    }
    fn type_name() -> &'static str {
        "i8"
    }
    fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

impl Element for i16 {
    fn random(rng: &mut impl Rng) -> Self {
        rng.random()
    }
    fn type_name() -> &'static str {
        "i16"
    }
    fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

impl Element for i32 {
    fn random(rng: &mut impl Rng) -> Self {
        rng.random()
    }
    fn type_name() -> &'static str {
        "i32"
    }
    fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

impl Element for i64 {
    fn random(rng: &mut impl Rng) -> Self {
        rng.random()
    }
    fn type_name() -> &'static str {
        "i64"
    }
    fn wrapping_add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

// ============================================================================
// Data Types
// ============================================================================

#[derive(Clone)]
struct FilterPlusData<T> {
    /// First array of values
    a: Vec<T>,
    /// Second array of values
    b: Vec<T>,
    /// Boolean mask
    mask: Vec<bool>,
}

#[derive(Clone, Debug)]
struct FilterPlusStats {
    /// Number of elements
    len: usize,
    /// Fraction of true values in mask (0.0 to 1.0)
    mask_density: f64,
    /// Number of true values in mask
    true_count: usize,
}

impl FilterPlusStats {
    fn compute<T>(data: &FilterPlusData<T>) -> Self {
        let true_count = data.mask.iter().filter(|&&b| b).count();
        Self {
            len: data.a.len(),
            mask_density: if data.a.is_empty() {
                0.0
            } else {
                true_count as f64 / data.a.len() as f64
            },
            true_count,
        }
    }
}

// ============================================================================
// Distributions
// ============================================================================

/// Generate filter/plus test data with a specific mask density
fn gen_data<T: Element>(seed: u64, target_density: f64) -> FilterPlusData<T> {
    gen_data_with_len(seed, 100_000, target_density) // 100k elements by default
}

/// Generate filter/plus test data with specific length and mask density
fn gen_data_with_len<T: Element>(seed: u64, len: usize, target_density: f64) -> FilterPlusData<T> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    let a: Vec<T> = (0..len).map(|_| T::random(&mut rng)).collect();
    let b: Vec<T> = (0..len).map(|_| T::random(&mut rng)).collect();
    let mask: Vec<bool> = (0..len)
        .map(|_| rng.random::<f64>() < target_density)
        .collect();

    FilterPlusData { a, b, mask }
}

// ============================================================================
// Algorithm Variants
// ============================================================================

/// Strategy 1: Add first, then filter the result
///
/// Compute: filter(A + B, M)
///
/// Work: N additions + M copies (where M = true count)
fn add_then_filter<T: Element>(data: &FilterPlusData<T>) -> Vec<T> {
    // Add all elements first (using wrapping to avoid overflow in debug)
    let sum: Vec<T> = data
        .a
        .iter()
        .zip(&data.b)
        .map(|(&a, &b)| a.wrapping_add(b))
        .collect();

    // Filter the result
    sum.into_iter()
        .zip(&data.mask)
        .filter_map(|(v, &keep)| keep.then_some(v))
        .collect()
}

/// Strategy 2: Filter both arrays first, then add
///
/// Compute: filter(A, M) + filter(B, M)
///
/// Work: 2M copies + M additions (where M = true count)
fn filter_then_add<T: Element>(data: &FilterPlusData<T>) -> Vec<T> {
    // Filter both arrays
    let a_filtered: Vec<T> = data
        .a
        .iter()
        .zip(&data.mask)
        .filter_map(|(&v, &keep)| keep.then_some(v))
        .collect();

    let b_filtered: Vec<T> = data
        .b
        .iter()
        .zip(&data.mask)
        .filter_map(|(&v, &keep)| keep.then_some(v))
        .collect();

    // Add filtered arrays (using wrapping to avoid overflow in debug)
    a_filtered
        .into_iter()
        .zip(b_filtered)
        .map(|(a, b)| a.wrapping_add(b))
        .collect()
}

/// Adaptive strategy: pick based on mask density
fn adaptive<T: Element>(data: &FilterPlusData<T>, stats: &FilterPlusStats) -> Vec<T> {
    // This threshold would be found by the benchmark runner
    // Note: actual threshold may vary by element type!
    const DENSITY_THRESHOLD: f64 = 0.3;

    if stats.mask_density < DENSITY_THRESHOLD {
        filter_then_add(data)
    } else {
        add_then_filter(data)
    }
}

// ============================================================================
// ParamGrid trait (for variants with tunable parameters)
// ============================================================================

trait ParamGrid: Sized + Clone + Debug {
    fn iter_all() -> impl Iterator<Item = Self>;
    fn to_suffix(&self) -> String;
}

// ============================================================================
// Main - demos the example
// ============================================================================

fn run_single_benchmark<T: Element + PartialEq>(density: f64, iterations: usize) {
    let data: FilterPlusData<T> = gen_data(42, density);
    let stats = FilterPlusStats::compute(&data);

    // Verify correctness
    let result1 = add_then_filter(&data);
    let result2 = filter_then_add(&data);
    assert_eq!(result1, result2, "results differ!");

    // Warm up
    for _ in 0..5 {
        drop(add_then_filter(&data));
        drop(filter_then_add(&data));
    }

    // Time add_then_filter
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        drop(std::hint::black_box(add_then_filter(std::hint::black_box(
            &data,
        ))));
    }
    let t1 = start.elapsed();

    // Time filter_then_add
    let start = std::time::Instant::now();
    for _ in 0..iterations {
        drop(std::hint::black_box(filter_then_add(std::hint::black_box(
            &data,
        ))));
    }
    let t2 = start.elapsed();

    let winner = if t1 < t2 {
        "add_then_filter"
    } else {
        "filter_then_add"
    };

    let t1_per_iter = t1.as_nanos() as f64 / iterations as f64 / 1000.0;
    let t2_per_iter = t2.as_nanos() as f64 / iterations as f64 / 1000.0;

    println!(
        "{:>4} | {:>8} | {:>6.1} | {:>8} | {:>10.1}µs | {:>13.1}µs | {:>15}",
        T::type_name(),
        stats.len,
        stats.mask_density,
        stats.true_count,
        t1_per_iter,
        t2_per_iter,
        winner
    );
}

fn run_benchmark<T: Element + PartialEq>() {
    println!("Element type: {}", T::type_name());
    println!("{:-<60}", "");

    let densities = [
        ("very_sparse (1%)", 0.01),
        ("sparse (10%)", 0.10),
        ("medium (50%)", 0.50),
        ("dense (90%)", 0.90),
        ("very_dense (99%)", 0.99),
    ];

    println!(
        "{:20} | {:>8} | {:>8} | {:>10} | {:>10} | {:>15}",
        "Distribution", "Len", "Density", "add_then", "filter_then", "Winner"
    );
    println!(
        "{:-<20}-+-{:-<8}-+-{:-<8}-+-{:-<10}-+-{:-<10}-+-{:-<15}",
        "", "", "", "", "", ""
    );

    for (name, density) in densities {
        let data: FilterPlusData<T> = gen_data(42, density);
        let stats = FilterPlusStats::compute(&data);

        // Verify correctness
        let result1 = add_then_filter(&data);
        let result2 = filter_then_add(&data);
        assert_eq!(result1, result2, "{}: results differ!", name);

        // Warm up
        drop(add_then_filter(&data));
        drop(filter_then_add(&data));

        // Time add_then_filter
        let start = std::time::Instant::now();
        for _ in 0..10 {
            drop(std::hint::black_box(add_then_filter(std::hint::black_box(
                &data,
            ))));
        }
        let t1 = start.elapsed();

        // Time filter_then_add
        let start = std::time::Instant::now();
        for _ in 0..10 {
            drop(std::hint::black_box(filter_then_add(std::hint::black_box(
                &data,
            ))));
        }
        let t2 = start.elapsed();

        let winner = if t1 < t2 {
            "add_then_filter"
        } else {
            "filter_then_add"
        };

        println!(
            "{:20} | {:>8} | {:>8.2} | {:>8}µs | {:>10}µs | {:>15}",
            name,
            stats.len,
            stats.mask_density,
            t1.as_micros(),
            t2.as_micros(),
            winner
        );
    }
    println!();
}

fn main() {
    println!("Filter/Plus Optimization Benchmark");
    println!("===================================");
    println!();
    println!("Problem: Given A, B, M compute (A + B) filtered by M");
    println!();
    println!("Strategies:");
    println!("  1. add_then_filter: filter(A + B, M)");
    println!("  2. filter_then_add: filter(A, M) + filter(B, M)");
    println!();

    // Focused benchmark: 100k elements, densities 0.1 and 0.9
    const ITERATIONS: usize = 50;
    println!(
        "Benchmark: 100,000 elements, {} iterations per measurement",
        ITERATIONS
    );
    println!();
    println!(
        "{:>4} | {:>8} | {:>6} | {:>8} | {:>12} | {:>14} | {:>15}",
        "Type", "Len", "Dens.", "True#", "add_then", "filter_then", "Winner"
    );
    println!(
        "{:-<4}-+-{:-<8}-+-{:-<6}-+-{:-<8}-+-{:-<12}-+-{:-<14}-+-{:-<15}",
        "", "", "", "", "", "", ""
    );

    // Density 0.1 (sparse mask)
    run_single_benchmark::<u8>(0.1, ITERATIONS);
    run_single_benchmark::<u16>(0.1, ITERATIONS);
    run_single_benchmark::<u32>(0.1, ITERATIONS);
    run_single_benchmark::<u64>(0.1, ITERATIONS);
    println!();

    // Density 0.9 (dense mask)
    run_single_benchmark::<u8>(0.9, ITERATIONS);
    run_single_benchmark::<u16>(0.9, ITERATIONS);
    run_single_benchmark::<u32>(0.9, ITERATIONS);
    run_single_benchmark::<u64>(0.9, ITERATIONS);

    println!();
    println!("Key insight:");
    println!("  With this simple Vec-based implementation, add_then_filter wins because:");
    println!("  - filter_then_add creates two intermediate Vecs (extra allocation)");
    println!("  - filter_then_add iterates over mask twice vs once");
    println!();
    println!("  In a real SIMD/columnar implementation, trade-offs would differ based on:");
    println!("  - SIMD lane width (more u8s fit in a register than u64s)");
    println!("  - Memory bandwidth (u8 arrays are 8x smaller than u64)");
    println!("  - Cache effects (smaller elements = more fit in cache)");
    println!();
    println!("  This is why we need to benchmark empirically!");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_strategies_equivalent_for<T: Element + PartialEq>() {
        for density in [0.01, 0.1, 0.5, 0.9, 0.99] {
            for seed in 0..5 {
                let data: FilterPlusData<T> = gen_data(seed, density);
                let result1 = add_then_filter(&data);
                let result2 = filter_then_add(&data);
                assert_eq!(
                    result1,
                    result2,
                    "type={}, density={}, seed={}",
                    T::type_name(),
                    density,
                    seed
                );
            }
        }
    }

    #[test]
    fn test_strategies_equivalent_u8() {
        test_strategies_equivalent_for::<u8>();
    }

    #[test]
    fn test_strategies_equivalent_u16() {
        test_strategies_equivalent_for::<u16>();
    }

    #[test]
    fn test_strategies_equivalent_u32() {
        test_strategies_equivalent_for::<u32>();
    }

    #[test]
    fn test_strategies_equivalent_u64() {
        test_strategies_equivalent_for::<u64>();
    }

    #[test]
    fn test_stats_computation() {
        let data: FilterPlusData<u32> = gen_data(42, 0.5);
        let stats = FilterPlusStats::compute(&data);

        assert_eq!(stats.len, data.a.len());
        assert!(stats.mask_density >= 0.0 && stats.mask_density <= 1.0);
        assert_eq!(stats.true_count, data.mask.iter().filter(|&&b| b).count());
    }

    #[test]
    fn test_empty_mask() {
        let data = FilterPlusData {
            a: vec![1u32, 2, 3],
            b: vec![4u32, 5, 6],
            mask: vec![false, false, false],
        };
        let stats = FilterPlusStats::compute(&data);
        assert_eq!(stats.mask_density, 0.0);
        assert_eq!(stats.true_count, 0);

        let result1 = add_then_filter(&data);
        let result2 = filter_then_add(&data);
        assert!(result1.is_empty());
        assert!(result2.is_empty());
    }

    #[test]
    fn test_full_mask() {
        let data = FilterPlusData {
            a: vec![1u32, 2, 3],
            b: vec![4u32, 5, 6],
            mask: vec![true, true, true],
        };
        let stats = FilterPlusStats::compute(&data);
        assert_eq!(stats.mask_density, 1.0);
        assert_eq!(stats.true_count, 3);

        let result1 = add_then_filter(&data);
        let result2 = filter_then_add(&data);
        assert_eq!(result1, vec![5, 7, 9]);
        assert_eq!(result2, vec![5, 7, 9]);
    }

    #[test]
    fn test_adaptive_matches() {
        for density in [0.01, 0.1, 0.5, 0.9, 0.99] {
            let data: FilterPlusData<u32> = gen_data(42, density);
            let stats = FilterPlusStats::compute(&data);
            let expected = add_then_filter(&data);
            let adaptive_result = adaptive(&data, &stats);
            assert_eq!(expected, adaptive_result);
        }
    }
}
