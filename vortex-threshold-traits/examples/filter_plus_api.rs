//! FILTER/PLUS OPTIMIZATION EXAMPLE (Vortex Arrays)
//!
//! This benchmark finds the optimal strategy for combining filter and plus operations,
//! parameterized by both mask density AND element type, using Vortex arrays.
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
    clippy::needless_doctest_main,
    unexpected_cfgs
)]

use rand::Rng;
use rand::SeedableRng;
use std::fmt::Debug;

// These imports are for the design doc - not actual dependencies of this crate
// use vortex_array::arrays::PrimitiveArray;
// use vortex_array::compute::{add, filter};
// use vortex_array::{Array, ArrayRef, IntoArray};
// use vortex_dtype::{NativePType, PType};
// use vortex_error::VortexResult;
// use vortex_mask::Mask;

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
// The code below requires vortex_array as a dependency.
// Since this is a design document file, the code is in a cfg-gated module.
// For a working example, see generic_api_demo.rs
// ============================================================================

#[cfg(feature = "vortex-array")]
mod vortex_impl {
    use super::*;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::compute::{add, filter};
    use vortex_array::{Array, ArrayRef, IntoArray};
    use vortex_dtype::{NativePType, PType};
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    // ============================================================================
    // Element Trait - Abstraction over primitive types
    // ============================================================================

    /// Trait for element types that can be used in filter/plus benchmarks
    trait Element: NativePType + Default + 'static {
        /// Generate a random value that won't overflow when added to another value of the same type.
        /// Values are limited to half the max value to avoid overflow in addition.
        fn random_no_overflow(rng: &mut impl Rng) -> Self;
        fn type_name() -> &'static str;
    }

    impl Element for u8 {
        fn random_no_overflow(rng: &mut impl Rng) -> Self {
            rng.random_range(0..128) // Max sum = 254, no overflow
        }
        fn type_name() -> &'static str {
            "u8"
        }
    }

    impl Element for u16 {
        fn random_no_overflow(rng: &mut impl Rng) -> Self {
            rng.random_range(0..32768) // Max sum = 65534, no overflow
        }
        fn type_name() -> &'static str {
            "u16"
        }
    }

    impl Element for u32 {
        fn random_no_overflow(rng: &mut impl Rng) -> Self {
            rng.random_range(0..u32::MAX / 2)
        }
        fn type_name() -> &'static str {
            "u32"
        }
    }

    impl Element for u64 {
        fn random_no_overflow(rng: &mut impl Rng) -> Self {
            rng.random_range(0..u64::MAX / 2)
        }
        fn type_name() -> &'static str {
            "u64"
        }
    }

    impl Element for i8 {
        fn random_no_overflow(rng: &mut impl Rng) -> Self {
            rng.random_range(-64..64) // Max sum = 126, min sum = -128, no overflow
        }
        fn type_name() -> &'static str {
            "i8"
        }
    }

    impl Element for i16 {
        fn random_no_overflow(rng: &mut impl Rng) -> Self {
            rng.random_range(-16384..16384)
        }
        fn type_name() -> &'static str {
            "i16"
        }
    }

    impl Element for i32 {
        fn random_no_overflow(rng: &mut impl Rng) -> Self {
            rng.random_range(i32::MIN / 2..i32::MAX / 2)
        }
        fn type_name() -> &'static str {
            "i32"
        }
    }

    impl Element for i64 {
        fn random_no_overflow(rng: &mut impl Rng) -> Self {
            rng.random_range(i64::MIN / 2..i64::MAX / 2)
        }
        fn type_name() -> &'static str {
            "i64"
        }
    }

    // ============================================================================
    // Data Types
    // ============================================================================

    /// Benchmark data using Vortex arrays
    struct FilterPlusData {
        /// First array of values
        a: ArrayRef,
        /// Second array of values
        b: ArrayRef,
        /// Boolean mask
        mask: Mask,
        /// Length of arrays
        len: usize,
    }

    impl Clone for FilterPlusData {
        fn clone(&self) -> Self {
            Self {
                a: self.a.clone(),
                b: self.b.clone(),
                mask: self.mask.clone(),
                len: self.len,
            }
        }
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
        fn compute(data: &FilterPlusData) -> Self {
            let true_count = data.mask.true_count();
            Self {
                len: data.len,
                mask_density: if data.len == 0 {
                    0.0
                } else {
                    true_count as f64 / data.len as f64
                },
                true_count,
            }
        }
    }

    // ============================================================================
    // Distributions
    // ============================================================================

    /// Generate filter/plus test data with a specific mask density using Vortex arrays
    fn gen_data<T: Element>(seed: u64, target_density: f64) -> FilterPlusData {
        gen_data_with_len::<T>(seed, 100_000, target_density) // 100k elements by default
    }

    /// Generate filter/plus test data with specific length and mask density
    fn gen_data_with_len<T: Element>(seed: u64, len: usize, target_density: f64) -> FilterPlusData {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        // Generate random values that won't overflow when added
        let a_vals: Vec<T> = (0..len).map(|_| T::random_no_overflow(&mut rng)).collect();
        let b_vals: Vec<T> = (0..len).map(|_| T::random_no_overflow(&mut rng)).collect();
        let mask_vals: Vec<bool> = (0..len)
            .map(|_| rng.random::<f64>() < target_density)
            .collect();

        // Convert to Vortex arrays
        let a: PrimitiveArray = a_vals.into_iter().collect();
        let b: PrimitiveArray = b_vals.into_iter().collect();
        let mask = Mask::from_iter(mask_vals);

        FilterPlusData {
            a: a.into_array(),
            b: b.into_array(),
            mask,
            len,
        }
    }

    // ============================================================================
    // Algorithm Variants
    // ============================================================================

    /// Strategy 1: Add first, then filter the result
    ///
    /// Compute: filter(A + B, M)
    ///
    /// Work: N additions + M copies (where M = true count)
    fn add_then_filter(data: &FilterPlusData) -> VortexResult<ArrayRef> {
        // Add all elements first
        let sum = add(&data.a, &data.b)?;

        // Filter the result
        filter(&sum, &data.mask)
    }

    /// Strategy 2: Filter both arrays first, then add
    ///
    /// Compute: filter(A, M) + filter(B, M)
    ///
    /// Work: 2M copies + M additions (where M = true count)
    fn filter_then_add(data: &FilterPlusData) -> VortexResult<ArrayRef> {
        // Filter both arrays
        let a_filtered = filter(&data.a, &data.mask)?;
        let b_filtered = filter(&data.b, &data.mask)?;

        // Add filtered arrays
        add(&a_filtered, &b_filtered)
    }

    /// Adaptive strategy: pick based on mask density
    fn adaptive(data: &FilterPlusData, stats: &FilterPlusStats) -> VortexResult<ArrayRef> {
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

    fn run_single_benchmark<T: Element>(density: f64, iterations: usize) -> VortexResult<()> {
        let data: FilterPlusData = gen_data::<T>(42, density);
        let stats = FilterPlusStats::compute(&data);

        // Verify correctness - both strategies should produce arrays with same length
        let result1 = add_then_filter(&data)?;
        let result2 = filter_then_add(&data)?;
        assert_eq!(
            result1.len(),
            result2.len(),
            "results have different lengths!"
        );

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
            "{:>4} | {:>8} | {:>6.2} | {:>8} | {:>10.1}µs | {:>13.1}µs | {:>15}",
            T::type_name(),
            stats.len,
            stats.mask_density,
            stats.true_count,
            t1_per_iter,
            t2_per_iter,
            winner
        );
        Ok(())
    }

    // ============================================================================
    // TYPE-ERASED APPROACH: Vortex arrays are already type-erased (ArrayRef)
    // ============================================================================
    //
    // Since Vortex's ArrayRef is already type-erased, our algorithms (add_then_filter,
    // filter_then_add) work on ArrayRef directly. We only need type-specific DATA
    // GENERATION, not type-specific algorithms.
    //
    // This means:
    // - Data: FilterPlusData { a: ArrayRef, b: ArrayRef, mask: Mask } (type-erased)
    // - Stats: FilterPlusStats { len, density, ptype: PType } (includes type as dimension)
    // - Algorithms: work on ArrayRef (no generics needed)
    // - Generation: for_types registers a generator per PType
    //
    // ============================================================================

    /*
    // =============================================================================
    // PROPOSED API: Type-erased StatsBench with for_types
    // =============================================================================

    use vortex_dtype::PType;
    use vortex_threshold_traits::{StatsBench, StatsGrid, Scale, DataGenerator};

    // Stats now include PType as a dimension
    #[derive(Clone, Debug)]
    struct FilterPlusStatsV2 {
        len: usize,
        density: f64,
        ptype: PType,  // U8, U16, U32, U64, I8, I16, I32, I64, F32, F64
    }

    // Data is type-erased (already the case with ArrayRef)
    #[derive(Clone)]
    struct FilterPlusDataV2 {
        a: ArrayRef,
        b: ArrayRef,
        mask: Mask,
    }

    // Algorithms work on type-erased ArrayRef (no generics!)
    fn add_then_filter_v2(data: &FilterPlusDataV2) -> VortexResult<ArrayRef> {
        let sum = add(&data.a, &data.b)?;
        filter(&sum, &data.mask)
    }

    fn filter_then_add_v2(data: &FilterPlusDataV2) -> VortexResult<ArrayRef> {
        let a_filtered = filter(&data.a, &data.mask)?;
        let b_filtered = filter(&data.b, &data.mask)?;
        add(&a_filtered, &b_filtered)
    }

    // =============================================================================
    // for_types: Register a generator for each PType
    // =============================================================================

    /// A generator that creates data for a specific PType
    trait TypedGenerator: Send + Sync {
        fn ptype(&self) -> PType;
        fn generate(&self, len: usize, density: f64, seed: u64) -> FilterPlusDataV2;
    }

    /// Implement generator for each element type
    struct Generator<T>(std::marker::PhantomData<T>);

    impl<T: Element> TypedGenerator for Generator<T> {
        fn ptype(&self) -> PType {
            T::PTYPE  // e.g., PType::U8, PType::U32, etc.
        }

        fn generate(&self, len: usize, density: f64, seed: u64) -> FilterPlusDataV2 {
            gen_data_with_len::<T>(seed, len, density)  // existing function, returns type-erased
        }
    }

    // =============================================================================
    // Building the benchmark
    // =============================================================================

    fn create_filter_plus_benchmark() -> BuiltStatsBench<FilterPlusDataV2, FilterPlusStatsV2, ArrayRef> {
        StatsBench::<FilterPlusDataV2, FilterPlusStatsV2, ArrayRef>::new("filter_plus")
            .stats_grid(
                StatsGrid::new()
                    .dimension("len", Scale::log2(10, 17))           // 1K to 128K
                    .dimension("density", Scale::steps(0.0, 1.0, 5)) // 0%, 25%, 50%, 75%, 100%
                    // ptype dimension added automatically by for_types
            )
            // Register generators for each type - adds "ptype" dimension
            .for_types(|types| {
                types
                    .add::<u8>()
                    .add::<u16>()
                    .add::<u32>()
                    .add::<u64>()
            })
            // Generate dispatches to the right generator based on stats.ptype
            .generate(|stats: &FilterPlusStatsV2, seed, generators| {
                generators.get(stats.ptype).generate(stats.len, stats.density, seed)
            })
            // Algorithms are type-erased - no generics!
            .baseline("add_then_filter", |data| add_then_filter_v2(data))
            .variant("filter_then_add", |data| filter_then_add_v2(data))
            .build()
    }

    // =============================================================================
    // Alternative: Simpler API with closure
    // =============================================================================

    fn create_filter_plus_benchmark_v2() -> BuiltStatsBench<FilterPlusDataV2, FilterPlusStatsV2, ArrayRef> {
        StatsBench::<FilterPlusDataV2, FilterPlusStatsV2, ArrayRef>::new("filter_plus")
            .stats_grid(
                StatsGrid::new()
                    .dimension("len", Scale::log2(10, 17))
                    .dimension("density", Scale::steps(0.0, 1.0, 5))
                    .dimension("ptype", Scale::ptypes(&[PType::U8, PType::U16, PType::U32, PType::U64]))
            )
            // Generate uses stats.ptype to dispatch
            .generate(|stats: &FilterPlusStatsV2, seed| {
                match stats.ptype {
                    PType::U8  => gen_data_with_len::<u8>(seed, stats.len, stats.density),
                    PType::U16 => gen_data_with_len::<u16>(seed, stats.len, stats.density),
                    PType::U32 => gen_data_with_len::<u32>(seed, stats.len, stats.density),
                    PType::U64 => gen_data_with_len::<u64>(seed, stats.len, stats.density),
                    _ => panic!("unsupported ptype"),
                }
            })
            .stats(|data: &FilterPlusDataV2| {
                FilterPlusStatsV2 {
                    len: data.a.len(),
                    density: data.mask.true_count() as f64 / data.a.len() as f64,
                    ptype: data.a.dtype().try_into().unwrap(),  // extract PType from ArrayRef
                }
            })
            .baseline("add_then_filter", |data| add_then_filter_v2(data))
            .variant("filter_then_add", |data| filter_then_add_v2(data))
            .build()
    }

    // =============================================================================
    // Usage
    // =============================================================================

    fn run_type_erased_benchmark() {
        let bench = create_filter_plus_benchmark_v2();

        // Search across all (len, density, ptype) combinations
        let results = bench.search().run(|point| {
            FilterPlusStatsV2 {
                len: point.get_usize("len").unwrap_or(1024),
                density: point.get("density").unwrap_or(0.5),
                ptype: point.get_ptype("ptype").unwrap_or(PType::U32),
            }
        });

        results.print();
        // Output:
        // filter_plus search results:
        //   (len=1024, density=0.00, ptype=U8)  -> filter_then_add (12.3 ns)
        //   (len=1024, density=0.00, ptype=U16) -> filter_then_add (14.1 ns)
        //   (len=1024, density=0.00, ptype=U32) -> filter_then_add (18.2 ns)
        //   (len=1024, density=0.00, ptype=U64) -> filter_then_add (24.5 ns)
        //   (len=1024, density=0.50, ptype=U8)  -> add_then_filter (45.2 ns)
        //   ...
        //
        // Winners grouped by ptype show different crossover densities per type!
    }

    */

    // ============================================================================
    // APPROACH 3: NON-TYPE-ERASED GENERIC API (Compile-time generics)
    // ============================================================================
    //
    // For cases where Data/Output ARE generic (not type-erased like ArrayRef),
    // we need a different approach. The algorithms are generic over T.
    //
    // Key insight: the search must still run over type-erased stuff internally,
    // but the USER-FACING API can be fully generic.
    //
    // ============================================================================

    /*
    // =============================================================================
    // DATA TYPES (generic over T)
    // =============================================================================

    #[derive(Clone)]
    struct FilterAddData<T> {
        a: Vec<T>,
        b: Vec<T>,
        mask: Vec<bool>,
    }

    // Stats are NOT generic - they describe the data without knowing T
    #[derive(Clone, Debug)]
    struct FilterAddStats {
        len: usize,
        density: f64,
    }

    impl FilterAddStats {
        /// Compute stats FROM data (Data -> Stats)
        fn compute<T>(data: &FilterAddData<T>) -> Self {
            let true_count = data.mask.iter().filter(|&&m| m).count();
            Self {
                len: data.a.len(),
                density: true_count as f64 / data.a.len() as f64,
            }
        }
    }

    impl<T: Element> FilterAddData<T> {
        /// Generate data matching stats (Stats -> Data)
        fn generate(stats: &FilterAddStats, seed: u64) -> Self {
            let mut rng = StdRng::seed_from_u64(seed);
            Self {
                a: (0..stats.len).map(|_| T::random(&mut rng)).collect(),
                b: (0..stats.len).map(|_| T::random(&mut rng)).collect(),
                mask: (0..stats.len).map(|_| rng.gen::<f64>() < stats.density).collect(),
            }
        }
    }

    // =============================================================================
    // ALGORITHMS (generic over T)
    // =============================================================================

    fn add_then_filter_generic<T: Element>(data: &FilterAddData<T>) -> Vec<T> {
        let sum: Vec<T> = data.a.iter().zip(&data.b)
            .map(|(a, b)| a.wrapping_add(*b))
            .collect();
        sum.into_iter().zip(&data.mask)
            .filter(|(_, &m)| m)
            .map(|(v, _)| v)
            .collect()
    }

    fn filter_then_add_generic<T: Element>(data: &FilterAddData<T>) -> Vec<T> {
        let a_filt: Vec<T> = data.a.iter().zip(&data.mask)
            .filter(|(_, &m)| m).map(|(&v, _)| v).collect();
        let b_filt: Vec<T> = data.b.iter().zip(&data.mask)
            .filter(|(_, &m)| m).map(|(&v, _)| v).collect();
        a_filt.iter().zip(&b_filt).map(|(a, b)| a.wrapping_add(*b)).collect()
    }

    // =============================================================================
    // TYPE-ERASED INTERNALS (what for_types produces for search)
    // =============================================================================

    use std::any::Any;

    /// Type-erased config - this is what we store internally
    trait ErasedConfig: Send + Sync {
        fn type_name(&self) -> &'static str;
        fn generate_erased(&self, stats: &FilterAddStats, seed: u64) -> Box<dyn ErasedData>;
        fn run_variant_erased(&self, variant: &str, data: &dyn ErasedData) -> Box<dyn ErasedOutput>;
        fn compute_stats_erased(&self, data: &dyn ErasedData) -> FilterAddStats;
        fn variant_names(&self) -> Vec<&'static str>;
    }

    /// Type-erased data wrapper
    trait ErasedData: Send + Sync {
        fn as_any(&self) -> &dyn Any;
    }

    impl<T: Send + Sync + 'static> ErasedData for FilterAddData<T> {
        fn as_any(&self) -> &dyn Any { self }
    }

    /// Type-erased output (for correctness checking)
    trait ErasedOutput: Send + Sync {
        fn len(&self) -> usize;
    }

    impl<T: Send + Sync + 'static> ErasedOutput for Vec<T> {
        fn len(&self) -> usize { self.len() }
    }

    /// Concrete config for a specific type T
    struct ConcreteConfig<T: Element> {
        _phantom: std::marker::PhantomData<T>,
        baseline: (&'static str, fn(&FilterAddData<T>) -> Vec<T>),
        variants: Vec<(&'static str, fn(&FilterAddData<T>) -> Vec<T>)>,
    }

    impl<T: Element + Send + Sync + 'static> ErasedConfig for ConcreteConfig<T> {
        fn type_name(&self) -> &'static str {
            T::type_name()
        }

        fn generate_erased(&self, stats: &FilterAddStats, seed: u64) -> Box<dyn ErasedData> {
            Box::new(FilterAddData::<T>::generate(stats, seed))
        }

        fn run_variant_erased(&self, variant: &str, data: &dyn ErasedData) -> Box<dyn ErasedOutput> {
            let concrete: &FilterAddData<T> = data.as_any().downcast_ref().unwrap();

            if variant == self.baseline.0 {
                Box::new((self.baseline.1)(concrete))
            } else {
                for (name, f) in &self.variants {
                    if *name == variant {
                        return Box::new(f(concrete));
                    }
                }
                Box::new((self.baseline.1)(concrete))
            }
        }

        fn compute_stats_erased(&self, data: &dyn ErasedData) -> FilterAddStats {
            let concrete: &FilterAddData<T> = data.as_any().downcast_ref().unwrap();
            FilterAddStats::compute(concrete)
        }

        fn variant_names(&self) -> Vec<&'static str> {
            let mut names = vec![self.baseline.0];
            names.extend(self.variants.iter().map(|(n, _)| *n));
            names
        }
    }

    // =============================================================================
    // BEVY-STYLE API: Marker struct + ForType trait
    // =============================================================================

    /// Trait that knows how to configure a benchmark for type T
    trait ForType<T: Element> {
        fn config() -> ConcreteConfig<T>;
    }

    /// Marker struct for filter_add benchmark
    struct FilterAddBench;

    impl<T: Element + Send + Sync + 'static> ForType<T> for FilterAddBench {
        fn config() -> ConcreteConfig<T> {
            ConcreteConfig {
                _phantom: std::marker::PhantomData,
                baseline: ("add_then_filter", add_then_filter_generic::<T>),
                variants: vec![("filter_then_add", filter_then_add_generic::<T>)],
            }
        }
    }

    // =============================================================================
    // TYPE LIST TRAIT (tuple impls like Bevy)
    // =============================================================================

    trait TypeList<Marker> {
        fn register(configs: &mut Vec<Box<dyn ErasedConfig>>, type_names: &mut Vec<&'static str>);
    }

    impl<Marker, A> TypeList<Marker> for (A,)
    where
        A: Element + Send + Sync + 'static,
        Marker: ForType<A>,
    {
        fn register(configs: &mut Vec<Box<dyn ErasedConfig>>, type_names: &mut Vec<&'static str>) {
            configs.push(Box::new(<Marker as ForType<A>>::config()));
            type_names.push(A::type_name());
        }
    }

    impl<Marker, A, B> TypeList<Marker> for (A, B)
    where
        A: Element + Send + Sync + 'static,
        B: Element + Send + Sync + 'static,
        Marker: ForType<A> + ForType<B>,
    {
        fn register(configs: &mut Vec<Box<dyn ErasedConfig>>, type_names: &mut Vec<&'static str>) {
            configs.push(Box::new(<Marker as ForType<A>>::config()));
            configs.push(Box::new(<Marker as ForType<B>>::config()));
            type_names.push(A::type_name());
            type_names.push(B::type_name());
        }
    }

    impl<Marker, A, B, C> TypeList<Marker> for (A, B, C)
    where
        A: Element + Send + Sync + 'static,
        B: Element + Send + Sync + 'static,
        C: Element + Send + Sync + 'static,
        Marker: ForType<A> + ForType<B> + ForType<C>,
    {
        fn register(configs: &mut Vec<Box<dyn ErasedConfig>>, type_names: &mut Vec<&'static str>) {
            configs.push(Box::new(<Marker as ForType<A>>::config()));
            configs.push(Box::new(<Marker as ForType<B>>::config()));
            configs.push(Box::new(<Marker as ForType<C>>::config()));
            type_names.push(A::type_name());
            type_names.push(B::type_name());
            type_names.push(C::type_name());
        }
    }

    impl<Marker, A, B, C, D> TypeList<Marker> for (A, B, C, D)
    where
        A: Element + Send + Sync + 'static,
        B: Element + Send + Sync + 'static,
        C: Element + Send + Sync + 'static,
        D: Element + Send + Sync + 'static,
        Marker: ForType<A> + ForType<B> + ForType<C> + ForType<D>,
    {
        fn register(configs: &mut Vec<Box<dyn ErasedConfig>>, type_names: &mut Vec<&'static str>) {
            configs.push(Box::new(<Marker as ForType<A>>::config()));
            configs.push(Box::new(<Marker as ForType<B>>::config()));
            configs.push(Box::new(<Marker as ForType<C>>::config()));
            configs.push(Box::new(<Marker as ForType<D>>::config()));
            type_names.push(A::type_name());
            type_names.push(B::type_name());
            type_names.push(C::type_name());
            type_names.push(D::type_name());
        }
    }

    // =============================================================================
    // UNIFIED BUILDER
    // =============================================================================

    use std::collections::HashMap;

    struct UnifiedStatsBench {
        name: String,
        len_range: (usize, usize),      // (min, max) as log2
        density_steps: usize,
        configs: HashMap<String, Box<dyn ErasedConfig>>,
        type_names: Vec<&'static str>,
    }

    impl UnifiedStatsBench {
        fn new(name: &str) -> Self {
            Self {
                name: name.to_string(),
                len_range: (10, 17),
                density_steps: 5,
                configs: HashMap::new(),
                type_names: Vec::new(),
            }
        }

        fn len_range(mut self, min_log2: usize, max_log2: usize) -> Self {
            self.len_range = (min_log2, max_log2);
            self
        }

        fn density_steps(mut self, steps: usize) -> Self {
            self.density_steps = steps;
            self
        }

        /// Register benchmark configs for multiple types (Bevy-style)
        fn for_types<Marker, Types>(mut self) -> Self
        where
            Types: TypeList<Marker>,
        {
            let mut configs = Vec::new();
            let mut type_names = Vec::new();
            Types::register(&mut configs, &mut type_names);

            for (config, name) in configs.into_iter().zip(type_names.iter()) {
                self.configs.insert(name.to_string(), config);
            }
            self.type_names = type_names;
            self
        }

        fn build(self) -> BuiltUnifiedBench {
            BuiltUnifiedBench {
                name: self.name,
                len_range: self.len_range,
                density_steps: self.density_steps,
                configs: self.configs,
                type_names: self.type_names,
            }
        }
    }

    struct BuiltUnifiedBench {
        name: String,
        len_range: (usize, usize),
        density_steps: usize,
        configs: HashMap<String, Box<dyn ErasedConfig>>,
        type_names: Vec<&'static str>,
    }

    impl BuiltUnifiedBench {
        /// Run search over all (len, density, elem_type) combinations
        fn search(&self) {
            println!("Search: {}", self.name);
            println!("=========");

            // Generate grid points
            let lens: Vec<usize> = (self.len_range.0..=self.len_range.1)
                .map(|exp| 1 << exp)
                .collect();

            let densities: Vec<f64> = (0..=self.density_steps)
                .map(|i| i as f64 / self.density_steps as f64)
                .collect();

            println!("Grid: {} lens × {} densities × {} types = {} points",
                lens.len(), densities.len(), self.type_names.len(),
                lens.len() * densities.len() * self.type_names.len());
            println!();

            // For each grid point
            for &type_name in &self.type_names {
                let config = self.configs.get(type_name).unwrap();

                for &len in &lens {
                    for &density in &densities {
                        let stats = FilterAddStats { len, density };

                        // Generate type-erased data
                        let data = config.generate_erased(&stats, 42);

                        // Run each variant
                        let mut best_variant = "";
                        let mut best_time = f64::MAX;

                        for variant in config.variant_names() {
                            // Measure (simplified - real impl would use proper measurement)
                            let start = std::time::Instant::now();
                            for _ in 0..10 {
                                let _ = config.run_variant_erased(variant, data.as_ref());
                            }
                            let elapsed = start.elapsed().as_nanos() as f64 / 10.0;

                            if elapsed < best_time {
                                best_time = elapsed;
                                best_variant = variant;
                            }
                        }

                        println!("  (len={:>6}, density={:.2}, type={:>3}) -> {} ({:.0} ns)",
                            len, density, type_name, best_variant, best_time);
                    }
                }
            }
        }
    }

    // =============================================================================
    // USAGE EXAMPLE
    // =============================================================================

    fn run_unified_benchmark() {
        let bench = UnifiedStatsBench::new("filter_add")
            .len_range(10, 12)   // 1K to 4K (small for demo)
            .density_steps(2)    // 0%, 50%, 100%
            .for_types::<FilterAddBench, (u8, u16, u32)>()
            .build();

        bench.search();
    }

    // =============================================================================
    // SUMMARY: Three approaches for type genericity
    // =============================================================================
    //
    // 1. TYPE-ERASED (ArrayRef):
    //    - Data is already type-erased (Vortex ArrayRef)
    //    - Algorithms work on ArrayRef, no generics needed
    //    - Add "ptype" as a grid dimension
    //    - Match on ptype in generate()
    //    - BEST FOR: Vortex-specific benchmarks
    //
    // 2. GENERIC with for_types (Bevy-style):
    //    - Data/Output are generic over T
    //    - User writes marker struct + ForType<T> impl
    //    - for_types::<Marker, (T1, T2, T3)>() registers all types
    //    - Internally type-erased for search
    //    - BEST FOR: Non-Vortex benchmarks, reusable logic
    //
    // 3. SINGLE TYPE (no generics):
    //    - StatsBench<D, S, O> with fixed types
    //    - Simplest case
    //    - BEST FOR: One-off benchmarks
    //
    // The unified builder supports all three via:
    //    .typed::<D, S, O>()                      // single type
    //    .for_types::<Marker, (T1, T2, T3)>()     // multi-type generic
    //    .ptype_dimension(...)                     // type-erased
    //
    // ============================================================================

    */

    fn main() -> VortexResult<()> {
        println!("Filter/Plus Optimization Benchmark (Vortex Arrays)");
        println!("===================================================");
        println!();
        println!("Problem: Given A, B, M compute (A + B) filtered by M");
        println!();
        println!("Strategies:");
        println!("  1. add_then_filter: filter(A + B, M)");
        println!("  2. filter_then_add: filter(A, M) + filter(B, M)");
        println!();

        // Focused benchmark: 100k elements, densities 0.01, 0.1, and 0.9
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

        // Density 0.01 (very sparse mask)
        run_single_benchmark::<u8>(0.01, ITERATIONS)?;
        run_single_benchmark::<u16>(0.01, ITERATIONS)?;
        run_single_benchmark::<u32>(0.01, ITERATIONS)?;
        run_single_benchmark::<u64>(0.01, ITERATIONS)?;
        println!();

        // Density 0.1 (sparse mask)
        run_single_benchmark::<u8>(0.1, ITERATIONS)?;
        run_single_benchmark::<u16>(0.1, ITERATIONS)?;
        run_single_benchmark::<u32>(0.1, ITERATIONS)?;
        run_single_benchmark::<u64>(0.1, ITERATIONS)?;
        println!();

        // Density 0.9 (dense mask)
        run_single_benchmark::<u8>(0.9, ITERATIONS)?;
        run_single_benchmark::<u16>(0.9, ITERATIONS)?;
        run_single_benchmark::<u32>(0.9, ITERATIONS)?;
        run_single_benchmark::<u64>(0.9, ITERATIONS)?;

        println!();
        println!("Key insight:");
        println!("  Trade-offs depend on Vortex array implementation:");
        println!("  - SIMD vectorization in filter/add kernels");
        println!("  - Memory allocation patterns");
        println!("  - Cache effects with different element sizes");
        println!();
        println!("  This is why we need to benchmark empirically!");

        Ok(())
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        fn test_strategies_equivalent_for<T: Element>() -> VortexResult<()> {
            for density in [0.01, 0.1, 0.5, 0.9, 0.99] {
                for seed in 0..3 {
                    let data: FilterPlusData = gen_data::<T>(seed, density);
                    let result1 = add_then_filter(&data)?;
                    let result2 = filter_then_add(&data)?;
                    assert_eq!(
                        result1.len(),
                        result2.len(),
                        "type={}, density={}, seed={}: lengths differ",
                        T::type_name(),
                        density,
                        seed
                    );
                }
            }
            Ok(())
        }

        #[test]
        fn test_strategies_equivalent_u8() -> VortexResult<()> {
            test_strategies_equivalent_for::<u8>()
        }

        #[test]
        fn test_strategies_equivalent_u16() -> VortexResult<()> {
            test_strategies_equivalent_for::<u16>()
        }

        #[test]
        fn test_strategies_equivalent_u32() -> VortexResult<()> {
            test_strategies_equivalent_for::<u32>()
        }

        #[test]
        fn test_strategies_equivalent_u64() -> VortexResult<()> {
            test_strategies_equivalent_for::<u64>()
        }

        #[test]
        fn test_stats_computation() {
            let data: FilterPlusData = gen_data::<u32>(42, 0.5);
            let stats = FilterPlusStats::compute(&data);

            assert_eq!(stats.len, data.len);
            assert!(stats.mask_density >= 0.0 && stats.mask_density <= 1.0);
            assert_eq!(stats.true_count, data.mask.true_count());
        }

        #[test]
        fn test_empty_mask() -> VortexResult<()> {
            let data: FilterPlusData = gen_data_with_len::<u32>(42, 100, 0.0);
            let stats = FilterPlusStats::compute(&data);

            // With 0.0 density, expect very few (possibly 0) true values
            assert!(stats.mask_density < 0.1);

            let result1 = add_then_filter(&data)?;
            let result2 = filter_then_add(&data)?;
            assert_eq!(result1.len(), result2.len());
            Ok(())
        }

        #[test]
        fn test_full_mask() -> VortexResult<()> {
            let data: FilterPlusData = gen_data_with_len::<u32>(42, 100, 1.0);
            let stats = FilterPlusStats::compute(&data);

            // With 1.0 density, expect all or nearly all true values
            assert!(stats.mask_density > 0.9);

            let result1 = add_then_filter(&data)?;
            let result2 = filter_then_add(&data)?;
            assert_eq!(result1.len(), result2.len());
            assert_eq!(result1.len(), stats.true_count);
            Ok(())
        }

        #[test]
        fn test_adaptive_returns_valid() -> VortexResult<()> {
            for density in [0.01, 0.1, 0.5, 0.9, 0.99] {
                let data: FilterPlusData = gen_data::<u32>(42, density);
                let stats = FilterPlusStats::compute(&data);
                let result = adaptive(&data, &stats)?;
                assert_eq!(result.len(), stats.true_count);
            }
            Ok(())
        }
    }
} // End of #[cfg(feature = "vortex")] mod vortex_impl

// ============================================================================
// MAIN - This file is a design document. See generic_api_demo.rs for working code.
// ============================================================================

fn main() {
    println!("Filter/Plus API Design Document");
    println!("================================");
    println!();
    println!("This file contains API designs for the StatsBench system.");
    println!("The Vortex-dependent code is in a cfg-gated module.");
    println!();
    println!("For a working example, run:");
    println!("  cargo run --example generic_api_demo -p vortex-threshold-traits");
}
