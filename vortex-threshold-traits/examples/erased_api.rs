//! ERASED API - Clean type-erased interface, no explicit monomorphization
//!
//! The user writes generic functions once, registers them with a macro,
//! and the framework handles everything internally.
//!
//! Run with: cargo run --example erased_api -p vortex-threshold-traits

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
use std::time::Instant;

use rand::Rng;
use rand::SeedableRng;

// ============================================================================
// ELEMENT TRAIT
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
// TYPE-ERASED DATA
// ============================================================================

/// Type-erased data trait
trait ErasedData: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn type_name(&self) -> &'static str;
}

/// Generic data container
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

impl<T: Element> ErasedData for FilterAddData<T> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn type_name(&self) -> &'static str {
        T::NAME
    }
}

// ============================================================================
// USER'S ALGORITHMS - Written once, generic
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
// TYPE-ERASED VARIANT
// ============================================================================

/// A variant that can run on type-erased data
trait ErasedVariant: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, data: &dyn ErasedData) -> usize;
}

/// Macro to create a type-erased variant from a generic function
macro_rules! erased_variant {
    ($name:expr, $func:ident) => {{
        struct Variant;
        impl ErasedVariant for Variant {
            fn name(&self) -> &str {
                $name
            }
            fn run(&self, data: &dyn ErasedData) -> usize {
                // Try each type - one will succeed
                if let Some(d) = data.as_any().downcast_ref::<FilterAddData<u8>>() {
                    return $func(d).len();
                }
                if let Some(d) = data.as_any().downcast_ref::<FilterAddData<u16>>() {
                    return $func(d).len();
                }
                if let Some(d) = data.as_any().downcast_ref::<FilterAddData<u32>>() {
                    return $func(d).len();
                }
                panic!("Unknown data type");
            }
        }
        Box::new(Variant) as Box<dyn ErasedVariant>
    }};
}

// ============================================================================
// TYPE-ERASED GENERATOR
// ============================================================================

trait ErasedGenerator: Send + Sync {
    fn type_name(&self) -> &'static str;
    fn generate(&self, len: usize, density: f64, seed: u64) -> Box<dyn ErasedData>;
}

macro_rules! erased_generator {
    ($ty:ty) => {{
        struct Generator;
        impl ErasedGenerator for Generator {
            fn type_name(&self) -> &'static str {
                <$ty as Element>::NAME
            }
            fn generate(&self, len: usize, density: f64, seed: u64) -> Box<dyn ErasedData> {
                Box::new(FilterAddData::<$ty>::generate(len, density, seed))
            }
        }
        Box::new(Generator) as Box<dyn ErasedGenerator>
    }};
}

// ============================================================================
// SIMPLE BENCHMARK BUILDER
// ============================================================================

struct ErasedBench {
    name: String,
    distributions: Vec<(&'static str, f64)>,
    len_values: Vec<usize>,
    generators: Vec<Box<dyn ErasedGenerator>>,
    variants: Vec<Box<dyn ErasedVariant>>,
}

impl ErasedBench {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_string(),
            distributions: Vec::new(),
            len_values: vec![1024, 4096],
            generators: Vec::new(),
            variants: Vec::new(),
        }
    }

    fn distribution(mut self, name: &'static str, density: f64) -> Self {
        self.distributions.push((name, density));
        self
    }

    fn len_values(mut self, values: Vec<usize>) -> Self {
        self.len_values = values;
        self
    }

    fn types(mut self, generators: Vec<Box<dyn ErasedGenerator>>) -> Self {
        self.generators = generators;
        self
    }

    fn variant(mut self, v: Box<dyn ErasedVariant>) -> Self {
        self.variants.push(v);
        self
    }

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
        println!("Variants: {}", self.variants.len());
        println!();

        let mut winners: HashMap<String, usize> = HashMap::new();

        for generator in &self.generators {
            let type_name = generator.type_name();
            println!("--- Type: {} ---", type_name);

            for &(dist_name, density) in &self.distributions {
                for &len in &self.len_values {
                    let data = generator.generate(len, density, 42);

                    let mut best_name = String::new();
                    let mut best_time = f64::MAX;

                    for variant in &self.variants {
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
                        "  ({:>8}, {:>5}) -> {:>16} ({:>8.0} ns)",
                        dist_name, len, best_name, best_time
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
// CLEAN MACRO API
// ============================================================================

/// Define a complete benchmark with one macro call
macro_rules! bench {
    (
        name: $name:expr,
        types: [$($ty:ty),+ $(,)?],
        distributions: { $($dist_name:literal => $density:expr),+ $(,)? },
        lengths: [$($len:expr),+ $(,)?],
        baseline: $baseline:ident,
        variants: [$($variant:ident),* $(,)?],
    ) => {{
        ErasedBench::new($name)
            $(.distribution($dist_name, $density))+
            .len_values(vec![$($len),+])
            .types(vec![$(erased_generator!($ty)),+])
            .variant(erased_variant!(stringify!($baseline), $baseline))
            $(.variant(erased_variant!(stringify!($variant), $variant)))*
    }};
}

// ============================================================================
// MAIN
// ============================================================================

fn main() {
    println!("Erased API Demo - No explicit type instantiation!");
    println!("=================================================");
    println!();

    // THE CLEANEST API - user just names their functions
    let benchmark = bench! {
        name: "filter_add",
        types: [u8, u16, u32],
        distributions: {
            "sparse" => 0.1,
            "medium" => 0.5,
            "dense" => 0.9,
        },
        lengths: [1024, 4096],
        baseline: add_then_filter,
        variants: [filter_then_add],
    };

    benchmark.search();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_erased_variant() {
        let variant = erased_variant!("test", add_then_filter);
        let data = FilterAddData::<u32>::generate(100, 0.5, 42);
        let result = variant.run(&data);
        assert!(result > 0 && result <= 100);
    }

    #[test]
    fn test_erased_generator() {
        let generator = erased_generator!(u32);
        assert_eq!(generator.type_name(), "u32");
        let data = generator.generate(100, 0.5, 42);
        assert_eq!(data.type_name(), "u32");
    }

    #[test]
    fn test_bench_macro() {
        let benchmark = bench! {
            name: "test",
            types: [u8, u32],
            distributions: { "sparse" => 0.1 },
            lengths: [100],
            baseline: add_then_filter,
            variants: [filter_then_add],
        };

        assert_eq!(benchmark.name, "test");
        assert_eq!(benchmark.generators.len(), 2);
        assert_eq!(benchmark.variants.len(), 2);
    }

    #[test]
    fn test_algorithms_match() {
        let data = FilterAddData::<u32>::generate(1000, 0.5, 42);
        let r1 = add_then_filter(&data);
        let r2 = filter_then_add(&data);
        assert_eq!(r1, r2);
    }
}
