//! FULL API DEMO - Complete example with distributions, stats, variants, types, and params
//!
//! This demonstrates:
//! 1. Named distributions (sparse, medium, dense)
//! 2. Stats computed from data (len, density)
//! 3. Multiple algorithm variants
//! 4. Generic over element types (u8, u16, u32)
//! 5. Per-variant parameters (chunk_size for chunked variant)
//! 6. Grid search across all dimensions
//!
//! Run with: cargo run --example full_api_demo -p vortex-threshold-traits

#![allow(
    clippy::disallowed_types,
    clippy::type_complexity,
    clippy::expect_used,
    clippy::use_debug,
    dead_code
)]

use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;

// ============================================================================
// ELEMENT TRAIT - Abstraction over primitive types
// ============================================================================

trait Element: Copy + Default + Send + Sync + Debug + 'static {
    const NAME: &'static str;
    fn random(rng: &mut impl Rng) -> Self;
    fn add(self, other: Self) -> Self;
}

impl Element for u8 {
    const NAME: &'static str = "u8";
    fn random(rng: &mut impl Rng) -> Self {
        rng.random_range(0..128)
    }
    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

impl Element for u16 {
    const NAME: &'static str = "u16";
    fn random(rng: &mut impl Rng) -> Self {
        rng.random_range(0..32768)
    }
    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

impl Element for u32 {
    const NAME: &'static str = "u32";
    fn random(rng: &mut impl Rng) -> Self {
        rng.random_range(0..u32::MAX / 2)
    }
    fn add(self, other: Self) -> Self {
        self.wrapping_add(other)
    }
}

// ============================================================================
// DATA TYPE - Generic over element type T
// ============================================================================

#[derive(Clone)]
struct FilterAddData<T> {
    a: Vec<T>,
    b: Vec<T>,
    mask: Vec<bool>,
}

// ============================================================================
// STATS - Computed FROM data, NOT generic (describes data without knowing T)
// ============================================================================

#[derive(Clone, Debug)]
struct FilterAddStats {
    len: usize,
    density: f64,
    distribution: String, // which named distribution generated this
}

impl FilterAddStats {
    /// Compute stats from data (Data -> Stats)
    fn compute<T>(data: &FilterAddData<T>, dist_name: &str) -> Self {
        let true_count = data.mask.iter().filter(|&&m| m).count();
        Self {
            len: data.a.len(),
            density: if data.a.is_empty() {
                0.0
            } else {
                true_count as f64 / data.a.len() as f64
            },
            distribution: dist_name.to_string(),
        }
    }
}

// ============================================================================
// NAMED DISTRIBUTIONS - Generate data with specific characteristics
// ============================================================================

/// A named distribution that generates data with specific density
#[derive(Clone, Debug)]
struct Distribution {
    name: &'static str,
    target_density: f64,
    weight: f64, // for weighted sampling in search
}

impl Distribution {
    fn new(name: &'static str, density: f64) -> Self {
        Self {
            name,
            target_density: density,
            weight: 1.0,
        }
    }

    fn with_weight(mut self, weight: f64) -> Self {
        self.weight = weight;
        self
    }

    /// Generate data for this distribution
    fn generate<T: Element>(&self, len: usize, seed: u64) -> FilterAddData<T> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        FilterAddData {
            a: (0..len).map(|_| T::random(&mut rng)).collect(),
            b: (0..len).map(|_| T::random(&mut rng)).collect(),
            mask: (0..len)
                .map(|_| rng.random::<f64>() < self.target_density)
                .collect(),
        }
    }
}

// ============================================================================
// PER-VARIANT PARAMETERS - Tunable params for specific variants
// ============================================================================

/// Parameters for the chunked variant
#[derive(Clone, Debug)]
struct ChunkedParams {
    chunk_size: usize,
}

impl ChunkedParams {
    /// Grid of parameter values to search
    fn grid() -> Vec<Self> {
        vec![
            Self { chunk_size: 4 },
            Self { chunk_size: 8 },
            Self { chunk_size: 16 },
            Self { chunk_size: 32 },
        ]
    }
}

/// Parameters for the SIMD variant
#[derive(Clone, Debug)]
struct SimdParams {
    unroll_factor: usize,
    prefetch_distance: usize,
}

impl SimdParams {
    fn grid() -> Vec<Self> {
        vec![
            Self { unroll_factor: 1, prefetch_distance: 0 },
            Self { unroll_factor: 2, prefetch_distance: 4 },
            Self { unroll_factor: 4, prefetch_distance: 8 },
        ]
    }
}

// ============================================================================
// ALGORITHM VARIANTS
// ============================================================================

/// Strategy 1: Add first, then filter
fn add_then_filter<T: Element>(data: &FilterAddData<T>) -> Vec<T> {
    let sum: Vec<T> = data
        .a
        .iter()
        .zip(&data.b)
        .map(|(&a, &b)| a.add(b))
        .collect();
    sum.into_iter()
        .zip(&data.mask)
        .filter(|&(_, &m)| m)
        .map(|(v, _)| v)
        .collect()
}

/// Strategy 2: Filter first, then add
fn filter_then_add<T: Element>(data: &FilterAddData<T>) -> Vec<T> {
    let a_filt: Vec<T> = data
        .a
        .iter()
        .zip(&data.mask)
        .filter(|&(_, &m)| m)
        .map(|(&v, _)| v)
        .collect();
    let b_filt: Vec<T> = data
        .b
        .iter()
        .zip(&data.mask)
        .filter(|&(_, &m)| m)
        .map(|(&v, _)| v)
        .collect();
    a_filt
        .iter()
        .zip(&b_filt)
        .map(|(&a, &b)| a.add(b))
        .collect()
}

/// Strategy 3: Chunked processing with tunable chunk_size
fn chunked_add_filter<T: Element>(data: &FilterAddData<T>, params: &ChunkedParams) -> Vec<T> {
    let mut result = Vec::new();

    for chunk_start in (0..data.a.len()).step_by(params.chunk_size) {
        let chunk_end = (chunk_start + params.chunk_size).min(data.a.len());

        for i in chunk_start..chunk_end {
            if data.mask[i] {
                result.push(data.a[i].add(data.b[i]));
            }
        }
    }

    result
}

/// Strategy 4: Simulated SIMD with tunable params
fn simd_add_filter<T: Element>(data: &FilterAddData<T>, params: &SimdParams) -> Vec<T> {
    // Simulate SIMD by processing in batches based on unroll factor
    let mut result = Vec::new();
    let unroll = params.unroll_factor;

    let mut i = 0;
    while i + unroll <= data.a.len() {
        // Process `unroll` elements at once
        for j in 0..unroll {
            if data.mask[i + j] {
                result.push(data.a[i + j].add(data.b[i + j]));
            }
        }
        i += unroll;
    }

    // Handle remainder
    while i < data.a.len() {
        if data.mask[i] {
            result.push(data.a[i].add(data.b[i]));
        }
        i += 1;
    }

    // Use prefetch_distance to avoid unused warning
    let _ = params.prefetch_distance;

    result
}

// ============================================================================
// TYPE-ERASED TRAITS (for internal search machinery)
// ============================================================================

trait ErasedData: Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

impl<T: Send + Sync + 'static> ErasedData for FilterAddData<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Type-erased variant configuration
trait ErasedVariant: Send + Sync {
    fn name(&self) -> String;
    fn run(&self, data: &dyn ErasedData) -> usize;
}

/// Type-erased config for a specific element type
trait ErasedTypeConfig: Send + Sync {
    fn type_name(&self) -> &'static str;
    fn generate(&self, dist: &Distribution, len: usize, seed: u64) -> Box<dyn ErasedData>;
    fn variants(&self) -> Vec<Box<dyn ErasedVariant>>;
}

// ============================================================================
// CONCRETE IMPLEMENTATIONS
// ============================================================================

/// A variant with no params (baseline)
struct SimpleVariant<T: Element> {
    name: &'static str,
    func: fn(&FilterAddData<T>) -> Vec<T>,
    _phantom: PhantomData<T>,
}

impl<T: Element> ErasedVariant for SimpleVariant<T> {
    fn name(&self) -> String {
        self.name.to_string()
    }

    fn run(&self, data: &dyn ErasedData) -> usize {
        let concrete: &FilterAddData<T> = data.as_any().downcast_ref().expect("type mismatch");
        (self.func)(concrete).len()
    }
}

/// A variant with params (expanded into multiple variants)
struct ParamVariant<T: Element, P: Clone + Debug> {
    base_name: &'static str,
    params: P,
    func: fn(&FilterAddData<T>, &P) -> Vec<T>,
    _phantom: PhantomData<T>,
}

impl<T: Element, P: Clone + Debug + Send + Sync + 'static> ErasedVariant for ParamVariant<T, P> {
    fn name(&self) -> String {
        format!("{}({:?})", self.base_name, self.params)
    }

    fn run(&self, data: &dyn ErasedData) -> usize {
        let concrete: &FilterAddData<T> = data.as_any().downcast_ref().expect("type mismatch");
        (self.func)(concrete, &self.params).len()
    }
}

/// Concrete config for element type T
struct TypeConfig<T: Element> {
    _phantom: PhantomData<T>,
}

impl<T: Element> ErasedTypeConfig for TypeConfig<T> {
    fn type_name(&self) -> &'static str {
        T::NAME
    }

    fn generate(&self, dist: &Distribution, len: usize, seed: u64) -> Box<dyn ErasedData> {
        Box::new(dist.generate::<T>(len, seed))
    }

    fn variants(&self) -> Vec<Box<dyn ErasedVariant>> {
        let mut variants: Vec<Box<dyn ErasedVariant>> = vec![
            // Simple variants (no params)
            Box::new(SimpleVariant::<T> {
                name: "add_then_filter",
                func: add_then_filter::<T>,
                _phantom: PhantomData,
            }),
            Box::new(SimpleVariant::<T> {
                name: "filter_then_add",
                func: filter_then_add::<T>,
                _phantom: PhantomData,
            }),
        ];

        // Parameterized variants - expanded for each param combination
        for params in ChunkedParams::grid() {
            variants.push(Box::new(ParamVariant::<T, ChunkedParams> {
                base_name: "chunked",
                params,
                func: chunked_add_filter::<T>,
                _phantom: PhantomData,
            }));
        }

        for params in SimdParams::grid() {
            variants.push(Box::new(ParamVariant::<T, SimdParams> {
                base_name: "simd",
                params,
                func: simd_add_filter::<T>,
                _phantom: PhantomData,
            }));
        }

        variants
    }
}

// ============================================================================
// BEVY-STYLE ForType TRAIT
// ============================================================================

trait ForType<T: Element> {
    fn config() -> Box<dyn ErasedTypeConfig>;
}

/// Marker struct for the FilterAdd benchmark
struct FilterAddBench;

impl<T: Element> ForType<T> for FilterAddBench {
    fn config() -> Box<dyn ErasedTypeConfig> {
        Box::new(TypeConfig::<T> {
            _phantom: PhantomData,
        })
    }
}

// ============================================================================
// TYPE LIST TRAIT (tuple impls)
// ============================================================================

trait TypeList<Marker> {
    fn register(configs: &mut Vec<Box<dyn ErasedTypeConfig>>);
}

impl<Marker, A> TypeList<Marker> for (A,)
where
    A: Element,
    Marker: ForType<A>,
{
    fn register(configs: &mut Vec<Box<dyn ErasedTypeConfig>>) {
        configs.push(<Marker as ForType<A>>::config());
    }
}

impl<Marker, A, B> TypeList<Marker> for (A, B)
where
    A: Element,
    B: Element,
    Marker: ForType<A> + ForType<B>,
{
    fn register(configs: &mut Vec<Box<dyn ErasedTypeConfig>>) {
        configs.push(<Marker as ForType<A>>::config());
        configs.push(<Marker as ForType<B>>::config());
    }
}

impl<Marker, A, B, C> TypeList<Marker> for (A, B, C)
where
    A: Element,
    B: Element,
    C: Element,
    Marker: ForType<A> + ForType<B> + ForType<C>,
{
    fn register(configs: &mut Vec<Box<dyn ErasedTypeConfig>>) {
        configs.push(<Marker as ForType<A>>::config());
        configs.push(<Marker as ForType<B>>::config());
        configs.push(<Marker as ForType<C>>::config());
    }
}

// ============================================================================
// FULL BENCHMARK BUILDER
// ============================================================================

struct FullStatsBench {
    name: String,
    distributions: Vec<Distribution>,
    len_values: Vec<usize>,
    type_configs: Vec<Box<dyn ErasedTypeConfig>>,
}

impl FullStatsBench {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            distributions: Vec::new(),
            len_values: vec![1024, 4096, 16384],
            type_configs: Vec::new(),
        }
    }

    /// Add a named distribution
    fn distribution(mut self, name: &'static str, density: f64) -> Self {
        self.distributions.push(Distribution::new(name, density));
        self
    }

    /// Add a distribution with custom weight
    fn distribution_weighted(mut self, name: &'static str, density: f64, weight: f64) -> Self {
        self.distributions
            .push(Distribution::new(name, density).with_weight(weight));
        self
    }

    /// Set the lengths to test
    fn len_values(mut self, values: Vec<usize>) -> Self {
        self.len_values = values;
        self
    }

    /// Register types using Bevy-style for_types
    fn for_types<Marker, Types>(mut self) -> Self
    where
        Types: TypeList<Marker>,
    {
        Types::register(&mut self.type_configs);
        self
    }

    fn build(self) -> BuiltFullBench {
        BuiltFullBench {
            name: self.name,
            distributions: self.distributions,
            len_values: self.len_values,
            type_configs: self.type_configs,
        }
    }
}

struct BuiltFullBench {
    name: String,
    distributions: Vec<Distribution>,
    len_values: Vec<usize>,
    type_configs: Vec<Box<dyn ErasedTypeConfig>>,
}

impl BuiltFullBench {
    /// Run full grid search
    fn search(&self) {
        println!("=== {} Grid Search ===", self.name);
        println!();

        let total_points = self.distributions.len()
            * self.len_values.len()
            * self.type_configs.len();

        let total_variants: usize = self.type_configs.iter().map(|c| c.variants().len()).sum();

        println!(
            "Grid: {} distributions × {} lengths × {} types = {} points",
            self.distributions.len(),
            self.len_values.len(),
            self.type_configs.len(),
            total_points
        );
        println!("Variants per type: ~{}", total_variants / self.type_configs.len().max(1));
        println!();

        // Track winners
        let mut winners: HashMap<String, usize> = HashMap::new();

        for type_config in &self.type_configs {
            let type_name = type_config.type_name();
            let variants = type_config.variants();

            println!("--- Type: {} ({} variants) ---", type_name, variants.len());

            for dist in &self.distributions {
                for &len in &self.len_values {
                    // Generate data
                    let data = type_config.generate(dist, len, 42);

                    // Time each variant
                    let mut best_variant = String::new();
                    let mut best_time = f64::MAX;

                    for variant in &variants {
                        let start = Instant::now();
                        let iters = 50;
                        for _ in 0..iters {
                            std::hint::black_box(variant.run(data.as_ref()));
                        }
                        let elapsed = start.elapsed().as_nanos() as f64 / iters as f64;

                        if elapsed < best_time {
                            best_time = elapsed;
                            best_variant = variant.name();
                        }
                    }

                    *winners.entry(best_variant.clone()).or_insert(0) += 1;

                    // Truncate variant name for display
                    let display_name = if best_variant.len() > 30 {
                        format!("{}...", &best_variant[..27])
                    } else {
                        best_variant.clone()
                    };

                    println!(
                        "  ({:>8}, {:>5}, {:>3}) -> {:>30} ({:>8.0} ns)",
                        dist.name, len, type_name, display_name, best_time
                    );
                }
            }
            println!();
        }

        // Print winner summary
        println!("=== Winner Summary ===");
        let mut winner_list: Vec<_> = winners.into_iter().collect();
        winner_list.sort_by(|a, b| b.1.cmp(&a.1));
        for (variant, count) in winner_list {
            println!("  {:>40}: {} wins", variant, count);
        }
    }

    /// Quick benchmark at a specific point
    fn bench_at(&self, dist_name: &str, len: usize, type_name: &str) {
        let dist = self
            .distributions
            .iter()
            .find(|d| d.name == dist_name)
            .expect("distribution not found");

        let type_config = self
            .type_configs
            .iter()
            .find(|c| c.type_name() == type_name)
            .expect("type not found");

        println!(
            "=== Quick Bench: {} @ ({}, {}, {}) ===",
            self.name, dist_name, len, type_name
        );
        println!();

        let data = type_config.generate(dist, len, 42);
        let variants = type_config.variants();

        for variant in &variants {
            let start = Instant::now();
            let iters = 100;
            for _ in 0..iters {
                std::hint::black_box(variant.run(data.as_ref()));
            }
            let elapsed = start.elapsed().as_nanos() as f64 / iters as f64;

            println!("  {:>40}: {:>10.0} ns", variant.name(), elapsed);
        }
    }
}

// ============================================================================
// MAIN - Demonstrates the full API
// ============================================================================

fn main() {
    println!("Full API Demo - Distributions, Stats, Types, Params");
    println!("====================================================");
    println!();

    // Build the benchmark with all dimensions
    let bench = FullStatsBench::new("filter_add")
        // Named distributions with target densities
        .distribution("sparse", 0.1)
        .distribution("medium", 0.5)
        .distribution("dense", 0.9)
        .distribution_weighted("very_sparse", 0.01, 2.0) // Higher weight
        // Lengths to search
        .len_values(vec![1024, 4096])
        // Types to benchmark (Bevy-style)
        .for_types::<FilterAddBench, (u8, u16, u32)>()
        .build();

    // Run full grid search
    bench.search();

    println!();
    println!("====================================================");
    println!();

    // Quick benchmark at a specific point
    bench.bench_at("medium", 4096, "u32");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_distribution_generation() {
        let dist = Distribution::new("test", 0.5);
        let data: FilterAddData<u32> = dist.generate(100, 42);

        assert_eq!(data.a.len(), 100);
        assert_eq!(data.b.len(), 100);
        assert_eq!(data.mask.len(), 100);
    }

    #[test]
    fn test_stats_computation() {
        let dist = Distribution::new("test", 0.5);
        let data: FilterAddData<u32> = dist.generate(1000, 42);
        let stats = FilterAddStats::compute(&data, "test");

        assert_eq!(stats.len, 1000);
        assert!(stats.density >= 0.3 && stats.density <= 0.7);
        assert_eq!(stats.distribution, "test");
    }

    #[test]
    fn test_all_variants_produce_same_output() {
        let dist = Distribution::new("test", 0.5);
        let data: FilterAddData<u32> = dist.generate(1000, 42);

        let baseline = add_then_filter(&data);
        let filter_first = filter_then_add(&data);
        let chunked = chunked_add_filter(&data, &ChunkedParams { chunk_size: 8 });
        let simd = simd_add_filter(
            &data,
            &SimdParams {
                unroll_factor: 2,
                prefetch_distance: 4,
            },
        );

        assert_eq!(baseline.len(), filter_first.len());
        assert_eq!(baseline.len(), chunked.len());
        assert_eq!(baseline.len(), simd.len());

        // Also check values match
        assert_eq!(baseline, filter_first);
        assert_eq!(baseline, chunked);
        assert_eq!(baseline, simd);
    }

    #[test]
    fn test_type_config_variants() {
        let config = TypeConfig::<u32> {
            _phantom: PhantomData,
        };
        let variants = config.variants();

        // Should have: 2 simple + 4 chunked params + 3 simd params = 9
        assert_eq!(variants.len(), 9);
    }

    #[test]
    fn test_full_bench_build() {
        let bench = FullStatsBench::new("test")
            .distribution("sparse", 0.1)
            .distribution("dense", 0.9)
            .len_values(vec![100])
            .for_types::<FilterAddBench, (u8, u32)>()
            .build();

        assert_eq!(bench.name, "test");
        assert_eq!(bench.distributions.len(), 2);
        assert_eq!(bench.len_values.len(), 1);
        assert_eq!(bench.type_configs.len(), 2);
    }
}
