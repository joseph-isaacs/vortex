//! PARAM GRID DESIGN - Comprehensive derive macro for per-variant parameters
//!
//! This shows the target API for a ParamGrid derive macro with many options.
//! The derive would generate `fn grid() -> Vec<Self>` and related methods.
//!
//! Run with: cargo run --example param_grid_design -p vortex-threshold-traits

#![allow(
    clippy::disallowed_types,
    clippy::type_complexity,
    clippy::expect_used,
    clippy::use_debug,
    dead_code
)]

use std::fmt::Debug;

// ============================================================================
// PARAM GRID TRAIT - What the derive macro generates
// ============================================================================

/// Trait for parameter structs that can generate a grid of values
trait ParamGrid: Clone + Debug + Send + Sync + 'static {
    /// Generate all parameter combinations
    fn grid() -> Vec<Self>;

    /// Get a short string representation for display
    fn to_suffix(&self) -> String;

    /// Get the default/recommended value
    fn default_value() -> Self;

    /// Get a minimal grid for quick testing
    fn quick_grid() -> Vec<Self>;

    /// Get an extended grid for thorough testing
    fn thorough_grid() -> Vec<Self>;

    /// Generate values using a custom generator function
    /// Useful for dynamic parameter generation based on runtime conditions
    fn generate<F>(f: F) -> Vec<Self>
    where
        F: FnOnce() -> Vec<Self>,
    {
        f()
    }

    /// Generate values from an iterator
    fn from_iter<I, F>(iter: I, f: F) -> Vec<Self>
    where
        I: IntoIterator,
        F: Fn(I::Item) -> Self,
    {
        iter.into_iter().map(f).collect()
    }
}

// ============================================================================
// EXAMPLE 1: Simple values list
// ============================================================================

/// Simple chunked processing parameters
///
/// ```ignore
/// #[derive(ParamGrid)]
/// struct ChunkedParams {
///     #[param(values = [4, 8, 16, 32, 64])]
///     #[param(default = 16)]
///     chunk_size: usize,
/// }
/// ```
#[derive(Clone, Debug)]
struct ChunkedParams {
    chunk_size: usize,
}

impl ParamGrid for ChunkedParams {
    fn grid() -> Vec<Self> {
        vec![4, 8, 16, 32, 64]
            .into_iter()
            .map(|chunk_size| Self { chunk_size })
            .collect()
    }

    fn to_suffix(&self) -> String {
        format!("chunk={}", self.chunk_size)
    }

    fn default_value() -> Self {
        Self { chunk_size: 16 }
    }

    fn quick_grid() -> Vec<Self> {
        vec![Self { chunk_size: 16 }]
    }

    fn thorough_grid() -> Vec<Self> {
        vec![2, 4, 8, 16, 32, 64, 128]
            .into_iter()
            .map(|chunk_size| Self { chunk_size })
            .collect()
    }
}

// ============================================================================
// EXAMPLE 2: Log2 scale
// ============================================================================

/// Parameters with logarithmic scaling
///
/// ```ignore
/// #[derive(ParamGrid)]
/// struct BufferParams {
///     #[param(log2 = 10..=16)]  // 1024, 2048, 4096, 8192, 16384, 32768, 65536
///     #[param(default_log2 = 12)]  // 4096
///     buffer_size: usize,
/// }
/// ```
#[derive(Clone, Debug)]
struct BufferParams {
    buffer_size: usize,
}

impl ParamGrid for BufferParams {
    fn grid() -> Vec<Self> {
        (10..=16)
            .map(|exp| Self {
                buffer_size: 1 << exp,
            })
            .collect()
    }

    fn to_suffix(&self) -> String {
        format!("buf={}K", self.buffer_size / 1024)
    }

    fn default_value() -> Self {
        Self { buffer_size: 4096 }
    }

    fn quick_grid() -> Vec<Self> {
        vec![Self { buffer_size: 4096 }]
    }

    fn thorough_grid() -> Vec<Self> {
        (8..=18)
            .map(|exp| Self {
                buffer_size: 1 << exp,
            })
            .collect()
    }
}

// ============================================================================
// EXAMPLE 3: Linear scale with steps
// ============================================================================

/// Parameters with linear density scaling
///
/// ```ignore
/// #[derive(ParamGrid)]
/// struct ThresholdParams {
///     #[param(linear = 0.0..=1.0, steps = 5)]  // 0.0, 0.25, 0.5, 0.75, 1.0
///     #[param(default = 0.5)]
///     threshold: f64,
/// }
/// ```
#[derive(Clone, Debug)]
struct ThresholdParams {
    threshold: f64,
}

impl ParamGrid for ThresholdParams {
    fn grid() -> Vec<Self> {
        (0..=4)
            .map(|i| Self {
                threshold: i as f64 * 0.25,
            })
            .collect()
    }

    fn to_suffix(&self) -> String {
        format!("thresh={:.2}", self.threshold)
    }

    fn default_value() -> Self {
        Self { threshold: 0.5 }
    }

    fn quick_grid() -> Vec<Self> {
        vec![
            Self { threshold: 0.25 },
            Self { threshold: 0.5 },
            Self { threshold: 0.75 },
        ]
    }

    fn thorough_grid() -> Vec<Self> {
        (0..=10)
            .map(|i| Self {
                threshold: i as f64 * 0.1,
            })
            .collect()
    }
}

// ============================================================================
// EXAMPLE 4: Multiple fields (Cartesian product)
// ============================================================================

/// SIMD parameters with multiple tunable values
///
/// ```ignore
/// #[derive(ParamGrid)]
/// struct SimdParams {
///     #[param(values = [1, 2, 4, 8])]
///     #[param(default = 4)]
///     unroll_factor: usize,
///
///     #[param(values = [0, 4, 8, 16])]
///     #[param(default = 8)]
///     prefetch_distance: usize,
///
///     #[param(values = [true, false])]
///     #[param(default = true)]
///     use_fma: bool,
/// }
/// ```
#[derive(Clone, Debug)]
struct SimdParams {
    unroll_factor: usize,
    prefetch_distance: usize,
    use_fma: bool,
}

impl ParamGrid for SimdParams {
    fn grid() -> Vec<Self> {
        let unroll = [1, 2, 4, 8];
        let prefetch = [0, 4, 8, 16];
        let fma = [true, false];

        let mut result = Vec::new();
        for &unroll_factor in &unroll {
            for &prefetch_distance in &prefetch {
                for &use_fma in &fma {
                    result.push(Self {
                        unroll_factor,
                        prefetch_distance,
                        use_fma,
                    });
                }
            }
        }
        result
    }

    fn to_suffix(&self) -> String {
        format!(
            "unroll={},prefetch={},fma={}",
            self.unroll_factor, self.prefetch_distance, self.use_fma
        )
    }

    fn default_value() -> Self {
        Self {
            unroll_factor: 4,
            prefetch_distance: 8,
            use_fma: true,
        }
    }

    fn quick_grid() -> Vec<Self> {
        vec![Self::default_value()]
    }

    fn thorough_grid() -> Vec<Self> {
        Self::grid() // Full Cartesian product
    }
}

// ============================================================================
// EXAMPLE 5: Conditional parameters
// ============================================================================

/// Parameters that depend on CPU features
///
/// ```ignore
/// #[derive(ParamGrid)]
/// struct CpuAwareParams {
///     #[param(values = [128, 256, 512])]
///     #[param(if cfg!(target_feature = "avx512f"), values = [128, 256, 512, 1024])]
///     vector_width: usize,
///
///     #[param(values = [1, 2, 4])]
///     #[param(if cfg!(target_feature = "avx2"), values = [1, 2, 4, 8])]
///     lanes: usize,
/// }
/// ```
#[derive(Clone, Debug)]
struct CpuAwareParams {
    vector_width: usize,
    lanes: usize,
}

impl ParamGrid for CpuAwareParams {
    fn grid() -> Vec<Self> {
        // Simulated CPU feature detection
        let has_avx512 = false; // Would be: cfg!(target_feature = "avx512f")
        let has_avx2 = true; // Would be: cfg!(target_feature = "avx2")

        let widths = if has_avx512 {
            vec![128, 256, 512, 1024]
        } else {
            vec![128, 256, 512]
        };

        let lanes_options = if has_avx2 {
            vec![1, 2, 4, 8]
        } else {
            vec![1, 2, 4]
        };

        let mut result = Vec::new();
        for &vector_width in &widths {
            for &lanes in &lanes_options {
                result.push(Self {
                    vector_width,
                    lanes,
                });
            }
        }
        result
    }

    fn to_suffix(&self) -> String {
        format!("width={},lanes={}", self.vector_width, self.lanes)
    }

    fn default_value() -> Self {
        Self {
            vector_width: 256,
            lanes: 4,
        }
    }

    fn quick_grid() -> Vec<Self> {
        vec![Self::default_value()]
    }

    fn thorough_grid() -> Vec<Self> {
        Self::grid()
    }
}

// ============================================================================
// EXAMPLE 6: Named presets
// ============================================================================

/// Parameters with named presets
///
/// ```ignore
/// #[derive(ParamGrid)]
/// #[param(presets = ["fast", "balanced", "thorough", "memory_efficient"])]
/// struct AlgorithmParams {
///     #[param(preset_values = {
///         "fast" => 4,
///         "balanced" => 8,
///         "thorough" => 16,
///         "memory_efficient" => 2,
///     })]
///     chunk_size: usize,
///
///     #[param(preset_values = {
///         "fast" => 0,
///         "balanced" => 4,
///         "thorough" => 8,
///         "memory_efficient" => 0,
///     })]
///     lookahead: usize,
///
///     #[param(preset_values = {
///         "fast" => false,
///         "balanced" => true,
///         "thorough" => true,
///         "memory_efficient" => false,
///     })]
///     use_cache: bool,
/// }
/// ```
#[derive(Clone, Debug)]
struct AlgorithmParams {
    chunk_size: usize,
    lookahead: usize,
    use_cache: bool,
    preset_name: &'static str,
}

impl AlgorithmParams {
    fn fast() -> Self {
        Self {
            chunk_size: 4,
            lookahead: 0,
            use_cache: false,
            preset_name: "fast",
        }
    }

    fn balanced() -> Self {
        Self {
            chunk_size: 8,
            lookahead: 4,
            use_cache: true,
            preset_name: "balanced",
        }
    }

    fn thorough() -> Self {
        Self {
            chunk_size: 16,
            lookahead: 8,
            use_cache: true,
            preset_name: "thorough",
        }
    }

    fn memory_efficient() -> Self {
        Self {
            chunk_size: 2,
            lookahead: 0,
            use_cache: false,
            preset_name: "memory_efficient",
        }
    }
}

impl ParamGrid for AlgorithmParams {
    fn grid() -> Vec<Self> {
        vec![
            Self::fast(),
            Self::balanced(),
            Self::thorough(),
            Self::memory_efficient(),
        ]
    }

    fn to_suffix(&self) -> String {
        self.preset_name.to_string()
    }

    fn default_value() -> Self {
        Self::balanced()
    }

    fn quick_grid() -> Vec<Self> {
        vec![Self::fast(), Self::balanced()]
    }

    fn thorough_grid() -> Vec<Self> {
        Self::grid()
    }
}

// ============================================================================
// EXAMPLE 7: Dependent parameters
// ============================================================================

/// Parameters where one depends on another
///
/// ```ignore
/// #[derive(ParamGrid)]
/// struct DependentParams {
///     #[param(values = [16, 32, 64, 128])]
///     block_size: usize,
///
///     // num_blocks must divide evenly into total_size
///     #[param(depends_on = block_size)]
///     #[param(values = |block_size| {
///         let total = 1024;
///         (1..=8).filter(|&n| (total / block_size) % n == 0).collect()
///     })]
///     num_blocks: usize,
/// }
/// ```
#[derive(Clone, Debug)]
struct DependentParams {
    block_size: usize,
    num_blocks: usize,
}

impl ParamGrid for DependentParams {
    fn grid() -> Vec<Self> {
        let block_sizes = [16, 32, 64, 128];
        let total = 1024;

        let mut result = Vec::new();
        for &block_size in &block_sizes {
            let max_blocks = total / block_size;
            for num_blocks in 1..=8 {
                if max_blocks % num_blocks == 0 {
                    result.push(Self {
                        block_size,
                        num_blocks,
                    });
                }
            }
        }
        result
    }

    fn to_suffix(&self) -> String {
        format!("block={},n={}", self.block_size, self.num_blocks)
    }

    fn default_value() -> Self {
        Self {
            block_size: 64,
            num_blocks: 4,
        }
    }

    fn quick_grid() -> Vec<Self> {
        vec![Self::default_value()]
    }

    fn thorough_grid() -> Vec<Self> {
        Self::grid()
    }
}

// ============================================================================
// EXAMPLE 8: Weighted parameters for importance
// ============================================================================

/// Parameters with weights for search importance
///
/// ```ignore
/// #[derive(ParamGrid)]
/// struct WeightedParams {
///     #[param(values = [8, 16, 32], weight = 2.0)]  // More important
///     primary_size: usize,
///
///     #[param(values = [1, 2, 4], weight = 1.0)]    // Normal importance
///     secondary_factor: usize,
///
///     #[param(values = [true, false], weight = 0.5)] // Less important
///     optional_feature: bool,
/// }
/// ```
#[derive(Clone, Debug)]
struct WeightedParams {
    primary_size: usize,
    secondary_factor: usize,
    optional_feature: bool,
}

impl WeightedParams {
    /// Get the weight for this parameter combination (for weighted sampling)
    fn weight(&self) -> f64 {
        let mut w = 1.0;
        // Primary size weight
        w *= match self.primary_size {
            8 | 32 => 2.0,
            16 => 3.0, // Default gets extra weight
            _ => 1.0,
        };
        // Secondary factor weight
        w *= match self.secondary_factor {
            2 => 1.5,
            _ => 1.0,
        };
        // Optional feature weight
        if self.optional_feature {
            w *= 0.8;
        }
        w
    }
}

impl ParamGrid for WeightedParams {
    fn grid() -> Vec<Self> {
        let mut result = Vec::new();
        for &primary_size in &[8, 16, 32] {
            for &secondary_factor in &[1, 2, 4] {
                for &optional_feature in &[true, false] {
                    result.push(Self {
                        primary_size,
                        secondary_factor,
                        optional_feature,
                    });
                }
            }
        }
        result
    }

    fn to_suffix(&self) -> String {
        format!(
            "p={},s={},opt={}",
            self.primary_size, self.secondary_factor, self.optional_feature
        )
    }

    fn default_value() -> Self {
        Self {
            primary_size: 16,
            secondary_factor: 2,
            optional_feature: false,
        }
    }

    fn quick_grid() -> Vec<Self> {
        // Return high-weight combinations only
        Self::grid()
            .into_iter()
            .filter(|p| p.weight() >= 2.0)
            .collect()
    }

    fn thorough_grid() -> Vec<Self> {
        Self::grid()
    }
}

// ============================================================================
// EXAMPLE 9: Enum-based parameters
// ============================================================================

/// Algorithm selection via enum
///
/// ```ignore
/// #[derive(ParamGrid)]
/// struct StrategyParams {
///     #[param(all_variants)]
///     strategy: Strategy,
///
///     #[param(values = [1, 2, 4])]
///     parallelism: usize,
/// }
///
/// #[derive(Clone, Debug, ParamGridEnum)]
/// enum Strategy {
///     Naive,
///     Optimized,
///     Simd,
///     #[param(if cfg!(feature = "gpu"))]
///     Gpu,
/// }
/// ```
#[derive(Clone, Copy, Debug, PartialEq)]
enum Strategy {
    Naive,
    Optimized,
    Simd,
    // Gpu, // Would be conditionally included
}

impl Strategy {
    fn all() -> Vec<Self> {
        vec![Self::Naive, Self::Optimized, Self::Simd]
    }
}

#[derive(Clone, Debug)]
struct StrategyParams {
    strategy: Strategy,
    parallelism: usize,
}

impl ParamGrid for StrategyParams {
    fn grid() -> Vec<Self> {
        let mut result = Vec::new();
        for strategy in Strategy::all() {
            for &parallelism in &[1, 2, 4] {
                result.push(Self {
                    strategy,
                    parallelism,
                });
            }
        }
        result
    }

    fn to_suffix(&self) -> String {
        format!("{:?},par={}", self.strategy, self.parallelism)
    }

    fn default_value() -> Self {
        Self {
            strategy: Strategy::Optimized,
            parallelism: 2,
        }
    }

    fn quick_grid() -> Vec<Self> {
        vec![
            Self {
                strategy: Strategy::Naive,
                parallelism: 1,
            },
            Self {
                strategy: Strategy::Optimized,
                parallelism: 2,
            },
        ]
    }

    fn thorough_grid() -> Vec<Self> {
        Self::grid()
    }
}

// ============================================================================
// EXAMPLE 10: Range constraints
// ============================================================================

/// Parameters with explicit constraints
///
/// ```ignore
/// #[derive(ParamGrid)]
/// struct ConstrainedParams {
///     #[param(range = 1..=16)]
///     #[param(constraint = "must be power of 2")]
///     #[param(filter = |x| x.is_power_of_two())]
///     tile_size: usize,
///
///     #[param(range = 0.0..=1.0)]
///     #[param(constraint = "probability")]
///     dropout_rate: f64,
/// }
/// ```
#[derive(Clone, Debug)]
struct ConstrainedParams {
    tile_size: usize,   // Must be power of 2, 1..=16
    dropout_rate: f64,  // 0.0..=1.0
}

impl ParamGrid for ConstrainedParams {
    fn grid() -> Vec<Self> {
        let tile_sizes: Vec<usize> = (1..=16).filter(|x| usize::is_power_of_two(*x)).collect();
        let dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5];

        let mut result = Vec::new();
        for &tile_size in &tile_sizes {
            for &dropout_rate in &dropout_rates {
                result.push(Self {
                    tile_size,
                    dropout_rate,
                });
            }
        }
        result
    }

    fn to_suffix(&self) -> String {
        format!("tile={},drop={:.1}", self.tile_size, self.dropout_rate)
    }

    fn default_value() -> Self {
        Self {
            tile_size: 4,
            dropout_rate: 0.0,
        }
    }

    fn quick_grid() -> Vec<Self> {
        vec![
            Self {
                tile_size: 4,
                dropout_rate: 0.0,
            },
            Self {
                tile_size: 8,
                dropout_rate: 0.1,
            },
        ]
    }

    fn thorough_grid() -> Vec<Self> {
        Self::grid()
    }
}

// ============================================================================
// EXAMPLE 11: Generator-based parameters
// ============================================================================

/// Parameters generated dynamically at runtime
///
/// ```ignore
/// #[derive(ParamGrid)]
/// struct DynamicParams {
///     #[param(generator = || {
///         let cpu_cores = num_cpus::get();
///         (1..=cpu_cores).collect()
///     })]
///     num_threads: usize,
///
///     #[param(generator = |num_threads| {
///         // Cache size depends on thread count
///         vec![1024 * num_threads, 2048 * num_threads, 4096 * num_threads]
///     })]
///     cache_size: usize,
/// }
/// ```
#[derive(Clone, Debug)]
struct DynamicParams {
    num_threads: usize,
    cache_size: usize,
    batch_size: usize,
}

impl DynamicParams {
    /// Generator that creates params based on system resources
    fn system_aware_generator() -> Vec<Self> {
        // Simulate detecting CPU cores (would use num_cpus crate in real code)
        let cpu_cores = 4; // num_cpus::get()

        // Simulate detecting available memory
        let available_mb = 8192; // Would detect actual RAM

        let thread_options: Vec<usize> = (1..=cpu_cores).collect();
        let cache_options = vec![1024, 2048, 4096, 8192]
            .into_iter()
            .filter(|&c| c <= available_mb / 4)
            .collect::<Vec<_>>();

        let mut result = Vec::new();
        for &num_threads in &thread_options {
            for &cache_size in &cache_options {
                // Batch size scales with threads
                for batch_mult in [1, 2, 4] {
                    result.push(Self {
                        num_threads,
                        cache_size,
                        batch_size: 64 * num_threads * batch_mult,
                    });
                }
            }
        }
        result
    }

    /// Generator for memory-constrained environments
    fn low_memory_generator() -> Vec<Self> {
        vec![
            Self { num_threads: 1, cache_size: 512, batch_size: 32 },
            Self { num_threads: 2, cache_size: 512, batch_size: 64 },
            Self { num_threads: 1, cache_size: 1024, batch_size: 64 },
        ]
    }

    /// Generator for high-performance environments
    fn high_perf_generator() -> Vec<Self> {
        let cores = 8; // Assume 8 cores
        (4..=cores)
            .flat_map(|threads| {
                [4096, 8192, 16384].into_iter().map(move |cache| Self {
                    num_threads: threads,
                    cache_size: cache,
                    batch_size: 256 * threads,
                })
            })
            .collect()
    }
}

impl ParamGrid for DynamicParams {
    fn grid() -> Vec<Self> {
        Self::system_aware_generator()
    }

    fn to_suffix(&self) -> String {
        format!(
            "threads={},cache={}K,batch={}",
            self.num_threads,
            self.cache_size / 1024,
            self.batch_size
        )
    }

    fn default_value() -> Self {
        Self {
            num_threads: 2,
            cache_size: 2048,
            batch_size: 128,
        }
    }

    fn quick_grid() -> Vec<Self> {
        vec![Self::default_value()]
    }

    fn thorough_grid() -> Vec<Self> {
        Self::system_aware_generator()
    }
}

// ============================================================================
// EXAMPLE 12: Composable generators
// ============================================================================

/// Generators that can be combined
#[derive(Clone, Debug)]
struct ComposableParams {
    value: usize,
}

impl ComposableParams {
    /// Linear range generator
    fn linear(start: usize, end: usize, steps: usize) -> Vec<Self> {
        if steps <= 1 {
            return vec![Self { value: start }];
        }
        (0..steps)
            .map(|i| Self {
                value: start + (end - start) * i / (steps - 1),
            })
            .collect()
    }

    /// Logarithmic range generator
    fn log2_range(min_exp: u32, max_exp: u32) -> Vec<Self> {
        (min_exp..=max_exp)
            .map(|exp| Self { value: 1 << exp })
            .collect()
    }

    /// Fibonacci sequence generator
    fn fibonacci(count: usize) -> Vec<Self> {
        let mut fibs = vec![1, 1];
        while fibs.len() < count {
            let next = fibs[fibs.len() - 1] + fibs[fibs.len() - 2];
            fibs.push(next);
        }
        fibs.into_iter().take(count).map(|v| Self { value: v }).collect()
    }

    /// Prime numbers generator
    fn primes(max: usize) -> Vec<Self> {
        let mut is_prime = vec![true; max + 1];
        is_prime[0] = false;
        if max > 0 {
            is_prime[1] = false;
        }

        for i in 2..=((max as f64).sqrt() as usize) {
            if is_prime[i] {
                for j in (i * i..=max).step_by(i) {
                    is_prime[j] = false;
                }
            }
        }

        is_prime
            .into_iter()
            .enumerate()
            .filter(|(_, p)| *p)
            .map(|(v, _)| Self { value: v })
            .collect()
    }

    /// Combine multiple generators, removing duplicates
    fn combine(generators: Vec<Vec<Self>>) -> Vec<Self> {
        let mut seen = std::collections::HashSet::new();
        generators
            .into_iter()
            .flatten()
            .filter(|p| seen.insert(p.value))
            .collect()
    }
}

impl ParamGrid for ComposableParams {
    fn grid() -> Vec<Self> {
        // Combine different generation strategies
        Self::combine(vec![
            Self::log2_range(0, 4),      // 1, 2, 4, 8, 16
            Self::linear(1, 20, 5),       // 1, 5, 10, 15, 20
            Self::fibonacci(6),           // 1, 1, 2, 3, 5, 8
        ])
    }

    fn to_suffix(&self) -> String {
        format!("v={}", self.value)
    }

    fn default_value() -> Self {
        Self { value: 8 }
    }

    fn quick_grid() -> Vec<Self> {
        Self::log2_range(2, 4) // 4, 8, 16
    }

    fn thorough_grid() -> Vec<Self> {
        Self::linear(1, 32, 32) // Every value 1-32
    }
}

// ============================================================================
// DEMONSTRATION
// ============================================================================

fn print_grid<P: ParamGrid>(name: &str) {
    println!("=== {} ===", name);
    let grid = P::grid();
    println!("  Grid size: {}", grid.len());
    println!("  Default: {:?}", P::default_value());
    println!("  Quick grid: {}", P::quick_grid().len());
    println!("  Thorough grid: {}", P::thorough_grid().len());
    println!("  First 5 values:");
    for (i, p) in grid.iter().take(5).enumerate() {
        println!("    {}: {} -> {:?}", i, p.to_suffix(), p);
    }
    if grid.len() > 5 {
        println!("    ... ({} more)", grid.len() - 5);
    }
    println!();
}

fn main() {
    println!("ParamGrid Design - Comprehensive Parameter Options");
    println!("===================================================");
    println!();

    print_grid::<ChunkedParams>("1. Simple values list");
    print_grid::<BufferParams>("2. Log2 scale");
    print_grid::<ThresholdParams>("3. Linear scale");
    print_grid::<SimdParams>("4. Multiple fields (Cartesian)");
    print_grid::<CpuAwareParams>("5. Conditional params");
    print_grid::<AlgorithmParams>("6. Named presets");
    print_grid::<DependentParams>("7. Dependent params");
    print_grid::<WeightedParams>("8. Weighted params");
    print_grid::<StrategyParams>("9. Enum-based");
    print_grid::<ConstrainedParams>("10. Range constraints");
    print_grid::<DynamicParams>("11. Generator-based (system-aware)");
    print_grid::<ComposableParams>("12. Composable generators");

    println!("===================================================");
    println!("Summary of ParamGrid options:");
    println!();
    println!("  #[param(values = [...])]           - Explicit value list");
    println!("  #[param(log2 = min..=max)]         - Logarithmic scale");
    println!("  #[param(linear = min..max, steps)] - Linear scale");
    println!("  #[param(range = min..=max)]        - Integer range");
    println!("  #[param(default = value)]          - Default value");
    println!("  #[param(filter = |x| pred)]        - Filter values");
    println!("  #[param(depends_on = field)]       - Dependent params");
    println!("  #[param(weight = w)]               - Search importance");
    println!("  #[param(if condition)]             - Conditional inclusion");
    println!("  #[param(preset_values = {{...}})]   - Named presets");
    println!("  #[param(all_variants)]             - All enum variants");
    println!("  #[param(generator = || ...)]       - Dynamic generation");
    println!();
    println!("Generator methods:");
    println!("  ParamGrid::generate(|| ...)        - Custom generator closure");
    println!("  ParamGrid::from_iter(iter, f)      - From iterator + mapping");
    println!("  Type::linear(start, end, steps)    - Linear range");
    println!("  Type::log2_range(min, max)         - Log2 scale");
    println!("  Type::combine(generators)          - Merge multiple generators");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_chunked_params() {
        let grid = ChunkedParams::grid();
        assert_eq!(grid.len(), 5);
        assert_eq!(ChunkedParams::default_value().chunk_size, 16);
    }

    #[test]
    fn test_simd_params_cartesian() {
        let grid = SimdParams::grid();
        // 4 unroll × 4 prefetch × 2 fma = 32
        assert_eq!(grid.len(), 32);
    }

    #[test]
    fn test_dependent_params() {
        let grid = DependentParams::grid();
        // All combinations should satisfy the constraint
        for p in &grid {
            let total = 1024;
            let max_blocks = total / p.block_size;
            assert!(max_blocks % p.num_blocks == 0);
        }
    }

    #[test]
    fn test_constrained_params() {
        let grid = ConstrainedParams::grid();
        for p in &grid {
            assert!(p.tile_size.is_power_of_two());
            assert!((0.0..=1.0).contains(&p.dropout_rate));
        }
    }

    #[test]
    fn test_dynamic_params_generators() {
        let system = DynamicParams::system_aware_generator();
        let low_mem = DynamicParams::low_memory_generator();
        let high_perf = DynamicParams::high_perf_generator();

        assert!(!system.is_empty());
        assert!(!low_mem.is_empty());
        assert!(!high_perf.is_empty());

        // Low memory should have smaller values
        for p in &low_mem {
            assert!(p.cache_size <= 1024);
        }
    }

    #[test]
    fn test_composable_generators() {
        let linear = ComposableParams::linear(0, 100, 11);
        assert_eq!(linear.len(), 11);
        assert_eq!(linear[0].value, 0);
        assert_eq!(linear[10].value, 100);

        let log2 = ComposableParams::log2_range(0, 4);
        assert_eq!(log2.len(), 5);
        assert_eq!(log2.iter().map(|p| p.value).collect::<Vec<_>>(), vec![1, 2, 4, 8, 16]);

        let fibs = ComposableParams::fibonacci(6);
        assert_eq!(fibs.iter().map(|p| p.value).collect::<Vec<_>>(), vec![1, 1, 2, 3, 5, 8]);

        let primes = ComposableParams::primes(20);
        assert_eq!(
            primes.iter().map(|p| p.value).collect::<Vec<_>>(),
            vec![2, 3, 5, 7, 11, 13, 17, 19]
        );
    }

    #[test]
    fn test_combine_generators() {
        let combined = ComposableParams::combine(vec![
            ComposableParams::linear(1, 5, 5),  // 1, 2, 3, 4, 5
            ComposableParams::linear(3, 7, 5),  // 3, 4, 5, 6, 7
        ]);
        // Should deduplicate: 1, 2, 3, 4, 5, 6, 7
        assert_eq!(combined.len(), 7);
    }
}
