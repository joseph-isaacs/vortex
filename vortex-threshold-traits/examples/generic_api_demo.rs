//! GENERIC API DEMO - Compilable example of type-generic StatsBench patterns
//!
//! This file demonstrates both:
//! 1. Type-erased approach (match on type in generate)
//! 2. Bevy-style for_types approach (marker struct + ForType trait)
//!
//! Run with: cargo run --example generic_api_demo -p vortex-threshold-traits

#![allow(
    clippy::disallowed_types,
    clippy::type_complexity,
    clippy::expect_used,
    clippy::use_debug,
    dead_code
)]

use std::any::Any;
use std::collections::HashMap;
use std::marker::PhantomData;
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;

// ============================================================================
// ELEMENT TRAIT - Abstraction over primitive types
// ============================================================================

trait Element: Copy + Default + Send + Sync + 'static {
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
// DATA AND STATS TYPES
// ============================================================================

#[derive(Clone)]
struct FilterAddData<T> {
    a: Vec<T>,
    b: Vec<T>,
    mask: Vec<bool>,
}

#[derive(Clone, Debug)]
struct FilterAddStats {
    len: usize,
    density: f64,
}

impl FilterAddStats {
    fn compute<T>(data: &FilterAddData<T>) -> Self {
        let true_count = data.mask.iter().filter(|&&m| m).count();
        Self {
            len: data.a.len(),
            density: if data.a.is_empty() {
                0.0
            } else {
                true_count as f64 / data.a.len() as f64
            },
        }
    }
}

impl<T: Element> FilterAddData<T> {
    fn generate(stats: &FilterAddStats, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self {
            a: (0..stats.len).map(|_| T::random(&mut rng)).collect(),
            b: (0..stats.len).map(|_| T::random(&mut rng)).collect(),
            mask: (0..stats.len)
                .map(|_| rng.random::<f64>() < stats.density)
                .collect(),
        }
    }
}

// ============================================================================
// ALGORITHMS
// ============================================================================

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

// ============================================================================
// TYPE-ERASED TRAITS
// ============================================================================

trait ErasedData: Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

impl<T: Send + Sync + 'static> ErasedData for FilterAddData<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

trait ErasedConfig: Send + Sync {
    fn type_name(&self) -> &'static str;
    fn generate_erased(&self, stats: &FilterAddStats, seed: u64) -> Box<dyn ErasedData>;
    fn run_variant_erased(&self, variant: &str, data: &dyn ErasedData) -> usize;
    fn variant_names(&self) -> Vec<&'static str>;
}

// ============================================================================
// CONCRETE CONFIG (implements ErasedConfig for a specific T)
// ============================================================================

struct ConcreteConfig<T: Element> {
    _phantom: PhantomData<T>,
    baseline: (&'static str, fn(&FilterAddData<T>) -> Vec<T>),
    variants: Vec<(&'static str, fn(&FilterAddData<T>) -> Vec<T>)>,
}

impl<T: Element> ErasedConfig for ConcreteConfig<T> {
    fn type_name(&self) -> &'static str {
        T::NAME
    }

    fn generate_erased(&self, stats: &FilterAddStats, seed: u64) -> Box<dyn ErasedData> {
        Box::new(FilterAddData::<T>::generate(stats, seed))
    }

    fn run_variant_erased(&self, variant: &str, data: &dyn ErasedData) -> usize {
        let concrete: &FilterAddData<T> = data.as_any().downcast_ref().expect("type mismatch");

        let result = if variant == self.baseline.0 {
            (self.baseline.1)(concrete)
        } else {
            self.variants
                .iter()
                .find(|(name, _)| *name == variant)
                .map(|(_, f)| f(concrete))
                .unwrap_or_else(|| (self.baseline.1)(concrete))
        };
        result.len()
    }

    fn variant_names(&self) -> Vec<&'static str> {
        let mut names = vec![self.baseline.0];
        names.extend(self.variants.iter().map(|(n, _)| *n));
        names
    }
}

// ============================================================================
// BEVY-STYLE: ForType trait + TypeList
// ============================================================================

trait ForType<T: Element> {
    fn config() -> ConcreteConfig<T>;
}

struct FilterAddBench;

impl<T: Element> ForType<T> for FilterAddBench {
    fn config() -> ConcreteConfig<T> {
        ConcreteConfig {
            _phantom: PhantomData,
            baseline: ("add_then_filter", add_then_filter::<T>),
            variants: vec![("filter_then_add", filter_then_add::<T>)],
        }
    }
}

trait TypeList<Marker> {
    fn register(configs: &mut Vec<Box<dyn ErasedConfig>>, type_names: &mut Vec<&'static str>);
}

impl<Marker, A> TypeList<Marker> for (A,)
where
    A: Element,
    Marker: ForType<A>,
{
    fn register(configs: &mut Vec<Box<dyn ErasedConfig>>, type_names: &mut Vec<&'static str>) {
        configs.push(Box::new(<Marker as ForType<A>>::config()));
        type_names.push(A::NAME);
    }
}

impl<Marker, A, B> TypeList<Marker> for (A, B)
where
    A: Element,
    B: Element,
    Marker: ForType<A> + ForType<B>,
{
    fn register(configs: &mut Vec<Box<dyn ErasedConfig>>, type_names: &mut Vec<&'static str>) {
        configs.push(Box::new(<Marker as ForType<A>>::config()));
        configs.push(Box::new(<Marker as ForType<B>>::config()));
        type_names.push(A::NAME);
        type_names.push(B::NAME);
    }
}

impl<Marker, A, B, C> TypeList<Marker> for (A, B, C)
where
    A: Element,
    B: Element,
    C: Element,
    Marker: ForType<A> + ForType<B> + ForType<C>,
{
    fn register(configs: &mut Vec<Box<dyn ErasedConfig>>, type_names: &mut Vec<&'static str>) {
        configs.push(Box::new(<Marker as ForType<A>>::config()));
        configs.push(Box::new(<Marker as ForType<B>>::config()));
        configs.push(Box::new(<Marker as ForType<C>>::config()));
        type_names.push(A::NAME);
        type_names.push(B::NAME);
        type_names.push(C::NAME);
    }
}

// ============================================================================
// UNIFIED BUILDER
// ============================================================================

struct UnifiedStatsBench {
    name: String,
    len_values: Vec<usize>,
    density_values: Vec<f64>,
    configs: HashMap<String, Box<dyn ErasedConfig>>,
    type_names: Vec<&'static str>,
}

impl UnifiedStatsBench {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            len_values: vec![1024, 4096],
            density_values: vec![0.1, 0.5, 0.9],
            configs: HashMap::new(),
            type_names: Vec::new(),
        }
    }

    fn len_values(mut self, values: Vec<usize>) -> Self {
        self.len_values = values;
        self
    }

    fn density_values(mut self, values: Vec<f64>) -> Self {
        self.density_values = values;
        self
    }

    /// Bevy-style: register configs for multiple types
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
            len_values: self.len_values,
            density_values: self.density_values,
            configs: self.configs,
            type_names: self.type_names,
        }
    }
}

struct BuiltUnifiedBench {
    name: String,
    len_values: Vec<usize>,
    density_values: Vec<f64>,
    configs: HashMap<String, Box<dyn ErasedConfig>>,
    type_names: Vec<&'static str>,
}

impl BuiltUnifiedBench {
    fn search(&self) {
        println!("Search: {}", self.name);
        println!("========{}", "=".repeat(self.name.len()));
        println!(
            "Grid: {} lens × {} densities × {} types",
            self.len_values.len(),
            self.density_values.len(),
            self.type_names.len()
        );
        println!();

        for &type_name in &self.type_names {
            let config = self.configs.get(type_name).expect("config not found");

            for &len in &self.len_values {
                for &density in &self.density_values {
                    let stats = FilterAddStats { len, density };
                    let data = config.generate_erased(&stats, 42);

                    let mut best_variant = "";
                    let mut best_time = f64::MAX;

                    for variant in config.variant_names() {
                        let start = Instant::now();
                        for _ in 0..100 {
                            std::hint::black_box(config.run_variant_erased(variant, data.as_ref()));
                        }
                        let elapsed = start.elapsed().as_nanos() as f64 / 100.0;

                        if elapsed < best_time {
                            best_time = elapsed;
                            best_variant = variant;
                        }
                    }

                    println!(
                        "  (len={:>5}, density={:.1}, type={:>3}) -> {:>16} ({:>8.0} ns)",
                        len, density, type_name, best_variant, best_time
                    );
                }
            }
        }
    }
}

// ============================================================================
// DEMO 1: Type-erased approach (manual match)
// ============================================================================

fn demo_type_erased() {
    println!("=== DEMO 1: Type-Erased Approach ===\n");

    #[derive(Clone, Copy, Debug)]
    enum ElemType {
        U8,
        U16,
        U32,
    }

    let types = [ElemType::U8, ElemType::U16, ElemType::U32];
    let lens = [1024usize, 4096];
    let densities = [0.1, 0.5, 0.9];

    for elem_type in types {
        for &len in &lens {
            for &density in &densities {
                let stats = FilterAddStats { len, density };

                // Match on type to generate data
                let (result_add_then, result_filter_then) = match elem_type {
                    ElemType::U8 => {
                        let data = FilterAddData::<u8>::generate(&stats, 42);
                        (add_then_filter(&data).len(), filter_then_add(&data).len())
                    }
                    ElemType::U16 => {
                        let data = FilterAddData::<u16>::generate(&stats, 42);
                        (add_then_filter(&data).len(), filter_then_add(&data).len())
                    }
                    ElemType::U32 => {
                        let data = FilterAddData::<u32>::generate(&stats, 42);
                        (add_then_filter(&data).len(), filter_then_add(&data).len())
                    }
                };

                assert_eq!(result_add_then, result_filter_then, "Results should match");
                println!(
                    "  (len={:>5}, density={:.1}, type={:?}) -> output_len={}",
                    len, density, elem_type, result_add_then
                );
            }
        }
    }
    println!();
}

// ============================================================================
// DEMO 2: Bevy-style for_types approach
// ============================================================================

fn demo_bevy_style() {
    println!("=== DEMO 2: Bevy-Style for_types ===\n");

    let bench = UnifiedStatsBench::new("filter_add")
        .len_values(vec![1024, 4096])
        .density_values(vec![0.1, 0.5, 0.9])
        .for_types::<FilterAddBench, (u8, u16, u32)>()
        .build();

    bench.search();
    println!();
}

// ============================================================================
// DEMO 3: Verification - check type state and dispatch are correct
// ============================================================================

fn demo_verification() {
    println!("=== DEMO 3: Verification of Type State & Dispatch ===\n");

    // Build the benchmark
    let bench = UnifiedStatsBench::new("filter_add")
        .len_values(vec![100]) // small for verification
        .density_values(vec![0.5])
        .for_types::<FilterAddBench, (u8, u16, u32)>()
        .build();

    println!("Registered types: {:?}", bench.type_names);
    println!("Configs count: {}", bench.configs.len());
    println!();

    // Verify each type config
    for &type_name in &bench.type_names {
        let config = bench.configs.get(type_name).expect("config missing");

        println!("--- Type: {} ---", type_name);
        println!("  config.type_name() = {}", config.type_name());
        println!("  variants: {:?}", config.variant_names());

        // Generate data and verify it's the right type
        let stats = FilterAddStats {
            len: 10,
            density: 0.5,
        };
        let data = config.generate_erased(&stats, 42);

        // Try to downcast to each type to verify which one it actually is
        let is_u8 = data.as_any().downcast_ref::<FilterAddData<u8>>().is_some();
        let is_u16 = data.as_any().downcast_ref::<FilterAddData<u16>>().is_some();
        let is_u32 = data.as_any().downcast_ref::<FilterAddData<u32>>().is_some();

        println!(
            "  Generated data type: u8={}, u16={}, u32={}",
            is_u8, is_u16, is_u32
        );

        // Verify the expected type matches
        let expected_type = match type_name {
            "u8" => is_u8,
            "u16" => is_u16,
            "u32" => is_u32,
            _ => false,
        };
        assert!(
            expected_type,
            "Type mismatch! Expected {} but got wrong type",
            type_name
        );
        println!("  ✓ Type verification PASSED");

        // Run variants and check they produce same output length
        let result_add = config.run_variant_erased("add_then_filter", data.as_ref());
        let result_filter = config.run_variant_erased("filter_then_add", data.as_ref());

        println!("  add_then_filter output len: {}", result_add);
        println!("  filter_then_add output len: {}", result_filter);
        assert_eq!(
            result_add, result_filter,
            "Variant outputs should have same length"
        );
        println!("  ✓ Variant output lengths match");

        println!();
    }

    // Verify ForType trait dispatch
    println!("--- ForType<T> Trait Dispatch Verification ---");

    let config_u8 = <FilterAddBench as ForType<u8>>::config();
    let config_u16 = <FilterAddBench as ForType<u16>>::config();
    let config_u32 = <FilterAddBench as ForType<u32>>::config();

    println!(
        "  ForType<u8>::config().type_name() = {}",
        config_u8.type_name()
    );
    println!(
        "  ForType<u16>::config().type_name() = {}",
        config_u16.type_name()
    );
    println!(
        "  ForType<u32>::config().type_name() = {}",
        config_u32.type_name()
    );

    assert_eq!(config_u8.type_name(), "u8");
    assert_eq!(config_u16.type_name(), "u16");
    assert_eq!(config_u32.type_name(), "u32");
    println!("  ✓ ForType trait dispatch PASSED");
    println!();

    // Verify TypeList registration order
    println!("--- TypeList Registration Order ---");
    let mut configs: Vec<Box<dyn ErasedConfig>> = Vec::new();
    let mut type_names: Vec<&'static str> = Vec::new();
    <(u8, u16, u32) as TypeList<FilterAddBench>>::register(&mut configs, &mut type_names);

    println!("  Registration order: {:?}", type_names);
    assert_eq!(type_names, vec!["u8", "u16", "u32"]);
    println!("  ✓ TypeList registration order PASSED");
    println!();

    // Verify algorithm correctness across types
    println!("--- Algorithm Correctness Across Types ---");
    for &type_name in &["u8", "u16", "u32"] {
        let config = &bench.configs[type_name];
        let stats = FilterAddStats {
            len: 1000,
            density: 0.3,
        };

        // Run 5 seeds and verify outputs match
        for seed in 0..5 {
            let data = config.generate_erased(&stats, seed);
            let r1 = config.run_variant_erased("add_then_filter", data.as_ref());
            let r2 = config.run_variant_erased("filter_then_add", data.as_ref());
            assert_eq!(
                r1, r2,
                "type={}, seed={}: output lengths differ",
                type_name, seed
            );
        }
        println!("  ✓ {} - 5 seeds verified", type_name);
    }
    println!();

    println!("All verifications PASSED!");
    println!();
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!("Generic API Demo");
    println!("================\n");

    demo_type_erased();
    demo_bevy_style();
    demo_verification();

    println!("Done! All demos and verifications passed.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_element_impls() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let a = u8::random(&mut rng);
        let b = u8::random(&mut rng);
        assert!(a < 128 && b < 128);
        let _ = a.add(b);
    }

    #[test]
    fn test_data_generation() {
        let stats = FilterAddStats {
            len: 100,
            density: 0.5,
        };
        let data = FilterAddData::<u32>::generate(&stats, 42);
        assert_eq!(data.a.len(), 100);
        assert_eq!(data.b.len(), 100);
        assert_eq!(data.mask.len(), 100);
    }

    #[test]
    fn test_stats_compute() {
        let stats = FilterAddStats {
            len: 100,
            density: 0.5,
        };
        let data = FilterAddData::<u32>::generate(&stats, 42);
        let computed = FilterAddStats::compute(&data);
        assert_eq!(computed.len, 100);
        assert!(computed.density >= 0.0 && computed.density <= 1.0);
    }

    #[test]
    fn test_algorithms_match() {
        let stats = FilterAddStats {
            len: 1000,
            density: 0.3,
        };
        let data = FilterAddData::<u32>::generate(&stats, 42);
        let r1 = add_then_filter(&data);
        let r2 = filter_then_add(&data);
        assert_eq!(r1.len(), r2.len());
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_erased_config() {
        let config = <FilterAddBench as ForType<u32>>::config();
        assert_eq!(config.type_name(), "u32");
        assert_eq!(
            config.variant_names(),
            vec!["add_then_filter", "filter_then_add"]
        );

        let stats = FilterAddStats {
            len: 100,
            density: 0.5,
        };
        let data = config.generate_erased(&stats, 42);
        let result = config.run_variant_erased("add_then_filter", data.as_ref());
        assert!(result > 0 && result <= 100);
    }

    #[test]
    fn test_type_list_registration() {
        let mut configs: Vec<Box<dyn ErasedConfig>> = Vec::new();
        let mut type_names: Vec<&'static str> = Vec::new();

        <(u8, u16, u32) as TypeList<FilterAddBench>>::register(&mut configs, &mut type_names);

        assert_eq!(configs.len(), 3);
        assert_eq!(type_names, vec!["u8", "u16", "u32"]);
    }

    #[test]
    fn test_unified_builder() {
        let bench = UnifiedStatsBench::new("test")
            .len_values(vec![100])
            .density_values(vec![0.5])
            .for_types::<FilterAddBench, (u8, u32)>()
            .build();

        assert_eq!(bench.name, "test");
        assert_eq!(bench.type_names, vec!["u8", "u32"]);
        assert_eq!(bench.configs.len(), 2);
    }
}
