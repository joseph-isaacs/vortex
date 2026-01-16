//! TARGET API EXAMPLE - Distribution-based benchmark API.
//!
//! This file demonstrates the target API for the threshold finder.
//! The key insight: generate realistic data first, then compute stats for reporting.
//!
//! Data flow:
//!   seed → Data → Stats (optional, for reporting)
//!            ↓
//!       benchmark variants
//!            ↓
//!       aggregate across distributions
//!            ↓
//!       find optimal variant

#![allow(dead_code, unused_imports, unused_variables, clippy::type_complexity)]

use rand::Rng;
use rand::SeedableRng;

// ============================================================================
// STEP 1: Define your Data type
// ============================================================================

/// The input to our algorithm: a bitmap and a position to query rank at.
#[derive(Clone)]
struct RankData {
    bitmap: Vec<u64>,
    position: usize,
}

// ============================================================================
// STEP 2: Define Stats (optional, for reporting/analysis)
// ============================================================================

/// Stats computed FROM data, used for analysis and reporting.
/// NOT used to generate data - that's backwards.
#[derive(Clone, Debug)]
struct RankStats {
    /// Length of the bitmap in u64 words
    len: usize,
    /// Fraction of bits set (0.0 to 1.0)
    density: f64,
}

impl RankStats {
    /// Compute stats from data (called after generation, for reporting)
    fn compute(data: &RankData) -> Self {
        let total_bits = data.bitmap.len() * 64;
        let set_bits: usize = data.bitmap.iter().map(|w| w.count_ones() as usize).sum();
        let density = if total_bits > 0 {
            set_bits as f64 / total_bits as f64
        } else {
            0.0
        };
        Self {
            len: data.bitmap.len(),
            density,
        }
    }
}

// ============================================================================
// STEP 3: Define data generators for each distribution
// ============================================================================

/// Generate a sparse bitmap (low density, ~10% bits set)
fn gen_sparse_bitmap(seed: u64) -> RankData {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let len = rng.random_range(100..10000);
    let density = 0.1;

    let bitmap: Vec<u64> = (0..len)
        .map(|_| {
            let mut word = 0u64;
            for bit in 0..64 {
                if rng.random::<f64>() < density {
                    word |= 1 << bit;
                }
            }
            word
        })
        .collect();

    let max_pos = len * 64;
    let position = rng.random_range(0..max_pos);

    RankData { bitmap, position }
}

/// Generate a dense bitmap (high density, ~90% bits set)
fn gen_dense_bitmap(seed: u64) -> RankData {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let len = rng.random_range(100..10000);
    let density = 0.9;

    let bitmap: Vec<u64> = (0..len)
        .map(|_| {
            let mut word = 0u64;
            for bit in 0..64 {
                if rng.random::<f64>() < density {
                    word |= 1 << bit;
                }
            }
            word
        })
        .collect();

    let max_pos = len * 64;
    let position = rng.random_range(0..max_pos);

    RankData { bitmap, position }
}

/// Generate a zipfian distribution (few words with many bits, many with few)
fn gen_zipfian_bitmap(seed: u64) -> RankData {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let len = rng.random_range(100..10000);

    let bitmap: Vec<u64> = (0..len)
        .map(|i| {
            // Zipfian: density decreases with index
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

    let max_pos = len * 64;
    let position = rng.random_range(0..max_pos);

    RankData { bitmap, position }
}

/// Generate small bitmaps (edge case: very small inputs)
fn gen_small_bitmap(seed: u64) -> RankData {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let len = rng.random_range(1..20); // Very small
    let density = 0.5;

    let bitmap: Vec<u64> = (0..len)
        .map(|_| {
            let mut word = 0u64;
            for bit in 0..64 {
                if rng.random::<f64>() < density {
                    word |= 1 << bit;
                }
            }
            word
        })
        .collect();

    let max_pos = len * 64;
    let position = rng.random_range(0..max_pos);

    RankData { bitmap, position }
}

// ============================================================================
// STEP 4: Implement algorithm variants
// ============================================================================

/// Naive bit-by-bit rank implementation
fn rank_naive(data: &RankData) -> usize {
    let mut count = 0;
    for (word_idx, &word) in data.bitmap.iter().enumerate() {
        let word_start = word_idx * 64;
        let word_end = word_start + 64;

        if data.position < word_start {
            break;
        }

        if data.position >= word_end {
            count += word.count_ones() as usize;
        } else {
            let bits_to_count = data.position - word_start + 1;
            let mask = (1u64 << bits_to_count) - 1;
            count += (word & mask).count_ones() as usize;
            break;
        }
    }
    count
}

/// SIMD-friendly implementation using popcount
fn rank_simd(data: &RankData) -> usize {
    let target_word = data.position / 64;
    let bit_in_word = data.position % 64;

    let mut count = 0;

    // Sum all complete words using popcount
    for &word in &data.bitmap[..target_word] {
        count += word.count_ones() as usize;
    }

    // Handle partial final word
    if target_word < data.bitmap.len() {
        let word = data.bitmap[target_word];
        let mask = (1u64 << (bit_in_word + 1)) - 1;
        count += (word & mask).count_ones() as usize;
    }

    count
}

/// Chunked implementation with loop unrolling
fn rank_chunked(data: &RankData) -> usize {
    let target_word = data.position / 64;
    let bit_in_word = data.position % 64;
    let chunk_size = 4;

    let mut count = 0;

    // Process in chunks of 4
    let full_chunks = target_word / chunk_size;
    for chunk_idx in 0..full_chunks {
        let start = chunk_idx * chunk_size;
        let chunk = &data.bitmap[start..start + chunk_size];
        count += chunk[0].count_ones() as usize;
        count += chunk[1].count_ones() as usize;
        count += chunk[2].count_ones() as usize;
        count += chunk[3].count_ones() as usize;
    }

    // Remaining full words
    let remaining_start = full_chunks * chunk_size;
    for &word in &data.bitmap[remaining_start..target_word] {
        count += word.count_ones() as usize;
    }

    // Partial final word
    if target_word < data.bitmap.len() {
        let word = data.bitmap[target_word];
        let mask = (1u64 << (bit_in_word + 1)) - 1;
        count += (word & mask).count_ones() as usize;
    }

    count
}

// ============================================================================
// STEP 5: Define the benchmark using distribution-based API
// ============================================================================

/*
// TARGET API - This is what we want to implement:

fn create_rank_benchmark() -> BuiltStatsBench<RankData, RankStats, usize> {
    StatsBench::new("rank")
        // Named distributions - each is a realistic workload
        .distribution("sparse", gen_sparse_bitmap)
        .distribution("dense", gen_dense_bitmap)
        .distribution("zipfian", gen_zipfian_bitmap)
        .distribution("small", gen_small_bitmap)

        // Optional: weight distributions by importance
        // (default weight is 1.0)
        .weight("zipfian", 2.0)  // real-world data is often zipfian

        // Optional: compute stats for reporting (Data → Stats)
        .stats(RankStats::compute)

        // Algorithm variants
        .baseline("naive", rank_naive)
        .variant("simd", rank_simd)
        .variant("chunked", rank_chunked)

        .build()
}

// Running the benchmark:

fn main() {
    let bench = create_rank_benchmark();

    // Run and get results
    let results = bench.run();

    // Print results table:
    //
    // Distribution   | naive    | simd     | chunked  | winner
    // ---------------|----------|----------|----------|--------
    // sparse         | 120µs    | 45µs     | 80µs     | simd
    // dense          | 450µs    | 50µs     | 200µs    | simd
    // zipfian (2.0x) | 200µs    | 60µs     | 90µs     | simd
    // small          | 5µs      | 4µs      | 6µs      | simd
    //
    // Aggregate winner: simd
    // Weighted scores: naive=970µs, simd=219µs, chunked=460µs

    results.print();

    // Save to JSON for CI
    results.save("rank_results.json").unwrap();
}
*/

// ============================================================================
// MAIN - Demonstrates the algorithms work correctly
// ============================================================================

fn main() {
    println!("Distribution-Based Benchmark API Example");
    println!("=========================================");
    println!();

    // Test each distribution
    let distributions: Vec<(&str, fn(u64) -> RankData)> = vec![
        ("sparse", gen_sparse_bitmap),
        ("dense", gen_dense_bitmap),
        ("zipfian", gen_zipfian_bitmap),
        ("small", gen_small_bitmap),
    ];

    for (name, generator) in &distributions {
        println!("Distribution: {}", name);

        // Generate a sample
        let data = generator(42);
        let stats = RankStats::compute(&data);
        println!(
            "  Sample: {} words, density={:.2}, position={}",
            stats.len, stats.density, data.position
        );

        // Verify all variants produce same result
        let naive_result = rank_naive(&data);
        let simd_result = rank_simd(&data);
        let chunked_result = rank_chunked(&data);

        assert_eq!(naive_result, simd_result, "simd mismatch for {}", name);
        assert_eq!(
            naive_result, chunked_result,
            "chunked mismatch for {}",
            name
        );

        println!("  Result: {} (all variants match)", naive_result);
        println!();
    }

    println!("All variants produce correct results!");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_distributions_correct() {
        let distributions: Vec<fn(u64) -> RankData> = vec![
            gen_sparse_bitmap,
            gen_dense_bitmap,
            gen_zipfian_bitmap,
            gen_small_bitmap,
        ];

        for generator in distributions {
            for seed in 0..10 {
                let data = generator(seed);

                let naive = rank_naive(&data);
                let simd = rank_simd(&data);
                let chunked = rank_chunked(&data);

                assert_eq!(naive, simd, "simd mismatch at seed {}", seed);
                assert_eq!(naive, chunked, "chunked mismatch at seed {}", seed);
            }
        }
    }

    #[test]
    fn test_stats_computation() {
        let data = RankData {
            bitmap: vec![0xFFFF_FFFF_FFFF_FFFF; 10], // All ones
            position: 500,
        };

        let stats = RankStats::compute(&data);
        assert_eq!(stats.len, 10);
        assert!((stats.density - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_sparse_density() {
        // Sparse should have low density
        let data = gen_sparse_bitmap(42);
        let stats = RankStats::compute(&data);
        assert!(
            stats.density < 0.3,
            "Sparse density too high: {}",
            stats.density
        );
    }

    #[test]
    fn test_dense_density() {
        // Dense should have high density
        let data = gen_dense_bitmap(42);
        let stats = RankStats::compute(&data);
        assert!(
            stats.density > 0.7,
            "Dense density too low: {}",
            stats.density
        );
    }
}
