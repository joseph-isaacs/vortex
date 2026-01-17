//! SIMPLE GENERIC API - User-friendly version with internal type erasure
//!
//! The user just writes generic functions and the framework handles type erasure.
//! No marker structs, no ForType trait, no TypeList - just simple generics!
//!
//! Run with: cargo run --example simple_generic_api -p vortex-threshold-traits

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
// ELEMENT TRAIT - User defines this for their types
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
// DATA TYPE - Generic over T
// ============================================================================

#[derive(Clone)]
struct FilterAddData<T> {
    a: Vec<T>,
    b: Vec<T>,
    mask: Vec<bool>,
}

impl<T: Element> FilterAddData<T> {
    fn generate(len: usize, density: f64, seed: u64) -> Self {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        Self {
            a: (0..len).map(|_| T::random(&mut rng)).collect(),
            b: (0..len).map(|_| T::random(&mut rng)).collect(),
            mask: (0..len)
                .map(|_| rng.random::<f64>() < density)
                .collect(),
        }
    }
}

// ============================================================================
// USER'S GENERIC ALGORITHMS - Just write normal generic functions!
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
// INTERNAL: Type erasure machinery (user never sees this)
// ============================================================================

trait ErasedData: Send + Sync {
    fn as_any(&self) -> &dyn Any;
}

impl<T: Send + Sync + 'static> ErasedData for FilterAddData<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }
}

/// Type-erased generator for a specific element type
trait TypedGenerator: Send + Sync {
    fn type_name(&self) -> &'static str;
    fn generate(&self, len: usize, density: f64, seed: u64) -> Box<dyn ErasedData>;
}

struct ConcreteGenerator<T: Element> {
    _phantom: PhantomData<T>,
}

impl<T: Element> TypedGenerator for ConcreteGenerator<T> {
    fn type_name(&self) -> &'static str {
        T::NAME
    }

    fn generate(&self, len: usize, density: f64, seed: u64) -> Box<dyn ErasedData> {
        Box::new(FilterAddData::<T>::generate(len, density, seed))
    }
}

/// Type-erased variant runner
trait ErasedVariant: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, data: &dyn ErasedData) -> usize;
}

struct ConcreteVariant<T: Element> {
    name: String,
    func: fn(&FilterAddData<T>) -> Vec<T>,
    _phantom: PhantomData<T>,
}

impl<T: Element> ErasedVariant for ConcreteVariant<T> {
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, data: &dyn ErasedData) -> usize {
        let concrete: &FilterAddData<T> = data.as_any().downcast_ref().expect("type mismatch");
        (self.func)(concrete).len()
    }
}

// ============================================================================
// INTERNAL: Type registration trait (tuple impls)
// ============================================================================

trait RegisterTypes {
    fn register_generators(generators: &mut Vec<Box<dyn TypedGenerator>>);
    fn register_variant(
        name: &str,
        variants: &mut HashMap<String, Vec<Box<dyn ErasedVariant>>>,
        func_u8: fn(&FilterAddData<u8>) -> Vec<u8>,
        func_u16: fn(&FilterAddData<u16>) -> Vec<u16>,
        func_u32: fn(&FilterAddData<u32>) -> Vec<u32>,
    );
}

// For simplicity, hardcode (u8, u16, u32) - could be generalized with macros
impl RegisterTypes for (u8, u16, u32) {
    fn register_generators(generators: &mut Vec<Box<dyn TypedGenerator>>) {
        generators.push(Box::new(ConcreteGenerator::<u8> {
            _phantom: PhantomData,
        }));
        generators.push(Box::new(ConcreteGenerator::<u16> {
            _phantom: PhantomData,
        }));
        generators.push(Box::new(ConcreteGenerator::<u32> {
            _phantom: PhantomData,
        }));
    }

    fn register_variant(
        name: &str,
        variants: &mut HashMap<String, Vec<Box<dyn ErasedVariant>>>,
        func_u8: fn(&FilterAddData<u8>) -> Vec<u8>,
        func_u16: fn(&FilterAddData<u16>) -> Vec<u16>,
        func_u32: fn(&FilterAddData<u32>) -> Vec<u32>,
    ) {
        variants.entry("u8".to_string()).or_default().push(Box::new(
            ConcreteVariant::<u8> {
                name: name.to_string(),
                func: func_u8,
                _phantom: PhantomData,
            },
        ));
        variants
            .entry("u16".to_string())
            .or_default()
            .push(Box::new(ConcreteVariant::<u16> {
                name: name.to_string(),
                func: func_u16,
                _phantom: PhantomData,
            }));
        variants
            .entry("u32".to_string())
            .or_default()
            .push(Box::new(ConcreteVariant::<u32> {
                name: name.to_string(),
                func: func_u32,
                _phantom: PhantomData,
            }));
    }
}

// ============================================================================
// SIMPLE BUILDER - What the user actually uses
// ============================================================================

struct SimpleStatsBench {
    name: String,
    distributions: Vec<(&'static str, f64)>, // (name, density)
    len_values: Vec<usize>,
    generators: Vec<Box<dyn TypedGenerator>>,
    variants: HashMap<String, Vec<Box<dyn ErasedVariant>>>, // type_name -> variants
}

impl SimpleStatsBench {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            distributions: Vec::new(),
            len_values: vec![1024, 4096],
            generators: Vec::new(),
            variants: HashMap::new(),
        }
    }

    /// Add a named distribution with target density
    fn distribution(mut self, name: &'static str, density: f64) -> Self {
        self.distributions.push((name, density));
        self
    }

    /// Set lengths to benchmark
    fn len_values(mut self, values: Vec<usize>) -> Self {
        self.len_values = values;
        self
    }

    /// Register types - framework handles monomorphization
    fn for_each_type<Types: RegisterTypes>(mut self) -> Self {
        Types::register_generators(&mut self.generators);
        self
    }

    /// Add baseline variant - pass your generic function!
    fn baseline(
        mut self,
        name: &str,
        // User passes: add_then_filter (the generic function)
        // We need all 3 monomorphized versions
        func_u8: fn(&FilterAddData<u8>) -> Vec<u8>,
        func_u16: fn(&FilterAddData<u16>) -> Vec<u16>,
        func_u32: fn(&FilterAddData<u32>) -> Vec<u32>,
    ) -> Self {
        <(u8, u16, u32) as RegisterTypes>::register_variant(
            name,
            &mut self.variants,
            func_u8,
            func_u16,
            func_u32,
        );
        self
    }

    /// Add a variant - same signature as baseline
    fn variant(
        self,
        name: &str,
        func_u8: fn(&FilterAddData<u8>) -> Vec<u8>,
        func_u16: fn(&FilterAddData<u16>) -> Vec<u16>,
        func_u32: fn(&FilterAddData<u32>) -> Vec<u32>,
    ) -> Self {
        self.baseline(name, func_u8, func_u16, func_u32)
    }

    fn build(self) -> BuiltSimpleBench {
        BuiltSimpleBench {
            name: self.name,
            distributions: self.distributions,
            len_values: self.len_values,
            generators: self.generators,
            variants: self.variants,
        }
    }
}

struct BuiltSimpleBench {
    name: String,
    distributions: Vec<(&'static str, f64)>,
    len_values: Vec<usize>,
    generators: Vec<Box<dyn TypedGenerator>>,
    variants: HashMap<String, Vec<Box<dyn ErasedVariant>>>,
}

impl BuiltSimpleBench {
    fn search(&self) {
        println!("=== {} Search ===", self.name);
        println!();

        let total = self.distributions.len() * self.len_values.len() * self.generators.len();
        println!(
            "Grid: {} distributions × {} lengths × {} types = {} points",
            self.distributions.len(),
            self.len_values.len(),
            self.generators.len(),
            total
        );
        println!();

        let mut winners: HashMap<String, usize> = HashMap::new();

        for generator in &self.generators {
            let type_name = generator.type_name();
            let type_variants = self.variants.get(type_name).expect("variants not found");

            println!("--- Type: {} ---", type_name);

            for &(dist_name, density) in &self.distributions {
                for &len in &self.len_values {
                    let data = generator.generate(len, density, 42);

                    let mut best_name = String::new();
                    let mut best_time = f64::MAX;

                    for variant in type_variants {
                        let start = Instant::now();
                        let iters = 50;
                        for _ in 0..iters {
                            std::hint::black_box(variant.run(data.as_ref()));
                        }
                        let elapsed = start.elapsed().as_nanos() as f64 / iters as f64;

                        if elapsed < best_time {
                            best_time = elapsed;
                            best_name = variant.name().to_string();
                        }
                    }

                    *winners.entry(best_name.clone()).or_insert(0) += 1;

                    println!(
                        "  ({:>8}, {:>5}, {:>3}) -> {:>16} ({:>8.0} ns)",
                        dist_name, len, type_name, best_name, best_time
                    );
                }
            }
            println!();
        }

        println!("=== Winners ===");
        let mut list: Vec<_> = winners.into_iter().collect();
        list.sort_by(|a, b| b.1.cmp(&a.1));
        for (name, count) in list {
            println!("  {:>20}: {} wins", name, count);
        }
    }
}

// ============================================================================
// MACROS: Make the API clean and simple
// ============================================================================

/// The cleanest API: declarative benchmark definition
///
/// Note: This macro is specialized for (u8, u16, u32) types.
/// A proc-macro could generalize this, but declarative macros can't easily
/// iterate over types and variants with different repetition counts.
macro_rules! define_bench {
    (
        name: $name:expr,
        distributions: [$(($dist_name:expr, $density:expr)),+ $(,)?],
        lengths: [$($len:expr),+ $(,)?],
        baseline: $baseline:ident,
        variants: [$($variant:ident),* $(,)?],
    ) => {{
        let bench = SimpleStatsBench::new($name)
            $(.distribution($dist_name, $density))+
            .len_values(vec![$($len),+])
            .for_each_type::<(u8, u16, u32)>()
            .baseline(
                stringify!($baseline),
                $baseline::<u8>,
                $baseline::<u16>,
                $baseline::<u32>,
            );

        // Add each variant
        $(
            let bench = bench.variant(
                stringify!($variant),
                $variant::<u8>,
                $variant::<u16>,
                $variant::<u32>,
            );
        )*

        bench.build()
    }};
}

/// Even simpler: just baseline + variant with automatic type expansion
macro_rules! add_variant {
    ($bench:expr, $name:expr, $func:ident) => {
        $bench.variant($name, $func::<u8>, $func::<u16>, $func::<u32>)
    };
}

macro_rules! add_baseline {
    ($bench:expr, $name:expr, $func:ident) => {
        $bench.baseline($name, $func::<u8>, $func::<u16>, $func::<u32>)
    };
}

// ============================================================================
// MAIN - Shows both API styles
// ============================================================================

fn main() {
    println!("Simple Generic API Demo");
    println!("=======================");
    println!();

    // =========================================================================
    // STYLE 1: Explicit (verbose - for understanding)
    // =========================================================================
    println!("Style 1: Explicit type instantiation (verbose)");
    println!("----------------------------------------------");

    let bench = SimpleStatsBench::new("filter_add")
        .distribution("sparse", 0.1)
        .distribution("dense", 0.9)
        .len_values(vec![1024])
        .for_each_type::<(u8, u16, u32)>()
        // Explicit: pass each monomorphized function
        .baseline(
            "add_then_filter",
            add_then_filter::<u8>,
            add_then_filter::<u16>,
            add_then_filter::<u32>,
        )
        .variant(
            "filter_then_add",
            filter_then_add::<u8>,
            filter_then_add::<u16>,
            filter_then_add::<u32>,
        )
        .build();

    bench.search();

    println!();
    println!("========================================");
    println!();

    // =========================================================================
    // STYLE 2: Using add_baseline!/add_variant! macros (medium)
    // =========================================================================
    println!("Style 2: Using helper macros (cleaner)");
    println!("-------------------------------------");

    let bench2 = SimpleStatsBench::new("filter_add_v2")
        .distribution("sparse", 0.1)
        .distribution("dense", 0.9)
        .len_values(vec![1024])
        .for_each_type::<(u8, u16, u32)>();

    let bench2 = add_baseline!(bench2, "add_then_filter", add_then_filter);
    let bench2 = add_variant!(bench2, "filter_then_add", filter_then_add);
    let bench2 = bench2.build();

    bench2.search();

    println!();
    println!("========================================");
    println!();

    // =========================================================================
    // STYLE 3: Declarative macro (cleanest!)
    // =========================================================================
    println!("Style 3: Declarative define_bench! macro (cleanest)");
    println!("--------------------------------------------------");

    // Note: types are (u8, u16, u32) by default in this simplified macro
    let bench3 = define_bench! {
        name: "filter_add_v3",
        distributions: [
            ("sparse", 0.1),
            ("medium", 0.5),
            ("dense", 0.9),
        ],
        lengths: [1024, 4096],
        baseline: add_then_filter,
        variants: [filter_then_add],
    };

    bench3.search();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_build() {
        let bench = SimpleStatsBench::new("test")
            .distribution("sparse", 0.1)
            .len_values(vec![100])
            .for_each_type::<(u8, u16, u32)>()
            .baseline(
                "add_then_filter",
                add_then_filter::<u8>,
                add_then_filter::<u16>,
                add_then_filter::<u32>,
            )
            .build();

        assert_eq!(bench.name, "test");
        assert_eq!(bench.generators.len(), 3);
        assert_eq!(bench.variants.len(), 3);
    }

    #[test]
    fn test_algorithms_match() {
        let data = FilterAddData::<u32>::generate(1000, 0.5, 42);
        let r1 = add_then_filter(&data);
        let r2 = filter_then_add(&data);
        assert_eq!(r1, r2);
    }
}
