//! TARGET API EXAMPLE
//!
//! This shows the complete target API for the threshold finder system.
//!
//! Data flow:
//!   seed → Data → Stats
//!                   ↓
//!            variant(data, stats) → Output
//!                   ↓
//!            aggregate across distributions
//!                   ↓
//!            find optimal (variant, params)

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

// ----------------------------------------------------------------------------
// Benchmark 1: Rank
// ----------------------------------------------------------------------------

#[derive(Clone, Debug, ParamGrid)]
struct ChunkedParams {
    #[param(values = [4, 8, 16, 32])]
    chunk_size: usize,
}

#[threshold_bench]
fn rank_bench() -> impl Benchmark {
    StatsBench::new("rank")
        // 1. Distributions
        .distribution("sparse", gen_sparse)
        .distribution("dense", gen_dense)
        .distribution("zipfian", gen_zipfian)
        .weight("zipfian", 2.0)

        // 2. Stats (computed from data, passed to variants)
        .stats(RankStats::compute)

        // 3. Baseline & Variants
        .baseline("naive", |data, stats| rank_naive(data))
        .variant("simd", |data, stats| rank_simd(data))
        .variant("adaptive", |data, stats| {
            if stats.density < 0.2 {
                rank_sparse_opt(data)
            } else {
                rank_dense_opt(data)
            }
        })
        .variant_params::<ChunkedParams>("chunked", |data, stats, p| {
            rank_chunked(data, p.chunk_size)
        })

        .build()
}

// ----------------------------------------------------------------------------
// Benchmark 2: Select
// ----------------------------------------------------------------------------

#[threshold_bench]
fn select_bench() -> impl Benchmark {
    StatsBench::new("select")
        .distribution("sparse", gen_sparse)
        .distribution("dense", gen_dense)

        .stats(SelectStats::compute)

        .baseline("naive", |data, stats| select_naive(data))
        .variant("binary_search", |data, stats| select_binary(data))

        .build()
}

// ----------------------------------------------------------------------------
// Benchmark 3: Popcount (no stats needed)
// ----------------------------------------------------------------------------

#[threshold_bench]
fn popcount_bench() -> impl Benchmark {
    StatsBench::new("popcount")
        .distribution("random", gen_random_words)
        .distribution("sparse", gen_sparse_words)
        .distribution("dense", gen_dense_words)

        // No .stats() call - variants just ignore stats parameter
        .stats(|_| ())  // unit stats

        .baseline("naive", |data, _| popcount_naive(data))
        .variant("builtin", |data, _| popcount_builtin(data))
        .variant("lookup", |data, _| popcount_lookup(data))

        .build()
}

// ----------------------------------------------------------------------------
// Main - runs all registered benchmarks
// ----------------------------------------------------------------------------

fn main() {
    // threshold_runner::main() does:
    // 1. Parse CLI args (filter, --list, --samples, --output)
    // 2. Collect all #[threshold_bench] functions via linkme
    // 3. Run matching benchmarks
    // 4. Print results / save to JSON
    threshold_runner::main();
}

// Run with:
//   cargo run --release                    # all benchmarks
//   cargo run --release -- rank            # just rank
//   cargo run --release -- --list          # list benchmarks
//   cargo run --release -- --samples 100   # 100 samples per dist
//   cargo run --release -- --output r.json # save results
*/

// ============================================================================
// Data Types
// ============================================================================

#[derive(Clone)]
struct RankData {
    bitmap: Vec<u64>,
    position: usize,
}

#[derive(Clone, Debug)]
struct RankStats {
    len: usize,
    density: f64,
}

impl RankStats {
    fn compute(data: &RankData) -> Self {
        let total_bits = data.bitmap.len() * 64;
        let set_bits: usize = data.bitmap.iter().map(|w| w.count_ones() as usize).sum();
        Self {
            len: data.bitmap.len(),
            density: if total_bits > 0 {
                set_bits as f64 / total_bits as f64
            } else {
                0.0
            },
        }
    }
}

// ============================================================================
// Distributions
// ============================================================================

fn gen_sparse(seed: u64) -> RankData {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let len = rng.random_range(100..10000);
    let bitmap = (0..len)
        .map(|_| rng.random::<u64>() & 0x1111_1111_1111_1111)
        .collect();
    let position = rng.random_range(0..len * 64);
    RankData { bitmap, position }
}

fn gen_dense(seed: u64) -> RankData {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let len = rng.random_range(100..10000);
    let bitmap = (0..len)
        .map(|_| rng.random::<u64>() | 0xEEEE_EEEE_EEEE_EEEE)
        .collect();
    let position = rng.random_range(0..len * 64);
    RankData { bitmap, position }
}

fn gen_zipfian(seed: u64) -> RankData {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let len = rng.random_range(100..10000);
    let bitmap = (0..len)
        .map(|i| {
            let density = 1.0 / (1.0 + i as f64).ln_1p();
            let mut word = 0u64;
            for bit in 0..64 {
                if rng.random::<f64>() < density {
                    word |= 1 << bit;
                }
            }
            word
        })
        .collect();
    let position = rng.random_range(0..len * 64);
    RankData { bitmap, position }
}

// ============================================================================
// Algorithm Variants
// ============================================================================

fn rank_naive(data: &RankData) -> usize {
    let target_word = data.position / 64;
    let bit_in_word = data.position % 64;
    let mut count = 0;
    for &word in &data.bitmap[..target_word] {
        count += word.count_ones() as usize;
    }
    if target_word < data.bitmap.len() {
        let mask = (1u64 << (bit_in_word + 1)) - 1;
        count += (data.bitmap[target_word] & mask).count_ones() as usize;
    }
    count
}

fn rank_simd(data: &RankData) -> usize {
    rank_naive(data) // placeholder
}

fn rank_sparse_opt(data: &RankData) -> usize {
    rank_naive(data) // placeholder
}

fn rank_dense_opt(data: &RankData) -> usize {
    rank_naive(data) // placeholder
}

fn rank_chunked(data: &RankData, chunk_size: usize) -> usize {
    let target_word = data.position / 64;
    let bit_in_word = data.position % 64;
    let mut count = 0;

    let chunks = data.bitmap[..target_word].chunks_exact(chunk_size);
    let remainder = chunks.remainder();
    for chunk in chunks {
        for &word in chunk {
            count += word.count_ones() as usize;
        }
    }
    for &word in remainder {
        count += word.count_ones() as usize;
    }

    if target_word < data.bitmap.len() {
        let mask = (1u64 << (bit_in_word + 1)) - 1;
        count += (data.bitmap[target_word] & mask).count_ones() as usize;
    }
    count
}

// ============================================================================
// ParamGrid trait (derive macro will generate this)
// ============================================================================

trait ParamGrid: Sized + Clone + Debug {
    fn iter_all() -> impl Iterator<Item = Self>;
    fn to_suffix(&self) -> String;
}

#[derive(Clone, Debug)]
struct ChunkedParams {
    chunk_size: usize,
}

// Generated by #[derive(ParamGrid)]
impl ParamGrid for ChunkedParams {
    fn iter_all() -> impl Iterator<Item = Self> {
        [4, 8, 16, 32]
            .into_iter()
            .map(|chunk_size| Self { chunk_size })
    }

    fn to_suffix(&self) -> String {
        format!("[{}]", self.chunk_size)
    }
}

#[derive(Clone, Debug)]
struct SimdParams {
    unroll: usize,
    prefetch: usize,
}

// Generated by #[derive(ParamGrid)]
impl ParamGrid for SimdParams {
    fn iter_all() -> impl Iterator<Item = Self> {
        [2, 4, 8].into_iter().flat_map(|unroll| {
            [0, 64, 128]
                .into_iter()
                .map(move |prefetch| Self { unroll, prefetch })
        })
    }

    fn to_suffix(&self) -> String {
        format!("[u={},p={}]", self.unroll, self.prefetch)
    }
}

// ============================================================================
// Main - demos the current implementation
// ============================================================================

fn main() {
    println!("Target API Example");
    println!("==================");
    println!();

    // Test distributions
    for (name, generator) in [
        ("sparse", gen_sparse as fn(u64) -> RankData),
        ("dense", gen_dense),
        ("zipfian", gen_zipfian),
    ] {
        let data = generator(42);
        let stats = RankStats::compute(&data);

        // Test all variants produce same result
        let naive = rank_naive(&data);
        let simd = rank_simd(&data);

        for p in ChunkedParams::iter_all() {
            let chunked = rank_chunked(&data, p.chunk_size);
            assert_eq!(naive, chunked, "chunked{} mismatch", p.to_suffix());
        }

        println!(
            "{:8} | len={:5} | density={:.2} | result={}",
            name, stats.len, stats.density, naive
        );
    }

    println!();
    println!("All variants correct!");
    println!();

    // Show param grid iteration
    println!("ChunkedParams grid:");
    for p in ChunkedParams::iter_all() {
        println!("  chunked{}", p.to_suffix());
    }

    println!();
    println!("SimdParams grid:");
    for p in SimdParams::iter_all() {
        println!("  simd{}", p.to_suffix());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_variants_correct() {
        for generator in [gen_sparse, gen_dense, gen_zipfian] {
            for seed in 0..10 {
                let data = generator(seed);
                let expected = rank_naive(&data);

                assert_eq!(rank_simd(&data), expected);

                for p in ChunkedParams::iter_all() {
                    assert_eq!(rank_chunked(&data, p.chunk_size), expected);
                }
            }
        }
    }

    #[test]
    fn test_param_grid_iteration() {
        let chunked: Vec<_> = ChunkedParams::iter_all().collect();
        assert_eq!(chunked.len(), 4);
        assert_eq!(chunked[0].chunk_size, 4);
        assert_eq!(chunked[3].chunk_size, 32);

        let simd: Vec<_> = SimdParams::iter_all().collect();
        assert_eq!(simd.len(), 9); // 3 * 3
    }

    #[test]
    fn test_param_suffix() {
        let p = ChunkedParams { chunk_size: 16 };
        assert_eq!(p.to_suffix(), "[16]");

        let p = SimdParams {
            unroll: 4,
            prefetch: 64,
        };
        assert_eq!(p.to_suffix(), "[u=4,p=64]");
    }
}
