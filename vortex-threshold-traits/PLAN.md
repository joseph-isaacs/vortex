# ISA Threshold Finder - Planning Document

## Goal

Find the optimal algorithm variant (and its parameters) for each CPU class by benchmarking against realistic, named data distributions. The optimizer minimizes aggregate runtime across all distributions.

## Data Flow

```
seed → Data → Stats
               ↓
         variant(data, stats) → Output
         variant(data, stats, params) → Output  (for parameterized variants)
               ↓
         aggregate across distributions
               ↓
         find optimal (variant, params)
```

## API Overview

### 1. Define a Benchmark

```rust
use vortex_threshold::{threshold_bench, StatsBench, ParamGrid};

#[derive(Clone, Debug, ParamGrid)]
struct ChunkedParams {
    #[param(values = [4, 8, 16, 32])]
    chunk_size: usize,
}

#[threshold_bench]
fn rank_bench() -> impl Benchmark {
    StatsBench::new("rank")
        // 1. Distributions (data generators)
        .distribution("sparse", gen_sparse)
        .distribution("dense", gen_dense)
        .distribution("zipfian", gen_zipfian)
        .weight("zipfian", 2.0)  // real-world weight

        // 2. Stats (computed from data, passed to variants)
        .stats(RankStats::compute)

        // 3. Baseline & Variants
        .baseline("naive", |data, stats| rank_naive(data))
        .variant("simd", |data, stats| rank_simd(data))
        .variant("adaptive", |data, stats| {
            if stats.density < 0.2 { rank_sparse(data) }
            else { rank_dense(data) }
        })
        .variant_params::<ChunkedParams>("chunked", |data, stats, p| {
            rank_chunked(data, p.chunk_size)
        })

        .build()
}
```

### 2. Define Multiple Benchmarks

```rust
#[threshold_bench]
fn rank_bench() -> impl Benchmark { ... }

#[threshold_bench]
fn select_bench() -> impl Benchmark { ... }

#[threshold_bench]
fn popcount_bench() -> impl Benchmark { ... }

fn main() {
    threshold_runner::main();
}
```

### 3. Run Benchmarks

```bash
# Run all benchmarks
cargo run --release

# Run specific benchmark
cargo run --release -- rank

# Run matching pattern
cargo run --release -- 'pop*'

# List available benchmarks
cargo run --release -- --list

# Configure samples
cargo run --release -- --samples 100

# Output to JSON
cargo run --release -- --output results.json
```

### 4. Output

```
Benchmark: rank
Samples: 100 per distribution

Distribution    | avg len | avg density | naive  | simd   | chunked[8] | chunked[16] | winner
----------------|---------|-------------|--------|--------|------------|-------------|--------
sparse          | 5012    | 0.12        | 120µs  | 95µs   | 85µs       | 90µs        | chunked[8]
dense           | 4998    | 0.88        | 450µs  | 180µs  | 170µs      | 165µs       | chunked[16]
zipfian (2.0x)  | 5105    | 0.45        | 200µs  | 140µs  | 130µs      | 135µs       | chunked[8]

Best params: chunked { chunk_size: 8 }
Aggregate winner: chunked[8] (weighted score: 515µs)
```

## Components

### 1. Core Traits (`vortex-threshold-traits`)

```rust
/// Trait for benchmarks - object-safe for registry
pub trait Benchmark: Send + Sync {
    fn name(&self) -> &str;
    fn run(&self, samples: usize, seed: u64) -> BenchResults;
}

/// Trait for parameter grids - derive macro generates this
pub trait ParamGrid: Sized + Clone + Debug {
    fn iter_all() -> impl Iterator<Item = Self>;
    fn to_suffix(&self) -> String;  // e.g., "[8]" or "[u=4,p=64]"
}
```

### 2. Builder (`vortex-threshold-traits`)

```rust
pub struct StatsBench<D, S, O> { ... }

impl StatsBench<D, S, O> {
    pub fn new(name: &str) -> Self;

    // Distributions
    pub fn distribution(self, name: &str, gen: impl Fn(u64) -> D) -> Self;
    pub fn weight(self, name: &str, weight: f64) -> Self;

    // Stats
    pub fn stats<S2>(self, f: impl Fn(&D) -> S2) -> StatsBench<D, S2, O>;

    // Variants
    pub fn baseline<O2>(self, name: &str, f: impl Fn(&D, &S) -> O2) -> StatsBench<D, S, O2>;
    pub fn variant(self, name: &str, f: impl Fn(&D, &S) -> O) -> Self;
    pub fn variant_params<P: ParamGrid>(self, name: &str, f: impl Fn(&D, &S, &P) -> O) -> Self;

    pub fn build(self) -> impl Benchmark;
}
```

### 3. Macro (`vortex-threshold-macros`)

```rust
/// Registers a benchmark function for automatic collection
#[proc_macro_attribute]
pub fn threshold_bench(_attr: TokenStream, item: TokenStream) -> TokenStream {
    // Uses `linkme` crate to collect at link time
    // Generates:
    //   #[linkme::distributed_slice(BENCHMARKS)]
    //   static _BENCH_rank: fn() -> Box<dyn Benchmark> = || Box::new(rank_bench());
}

/// Derives ParamGrid trait for parameter structs
#[proc_macro_derive(ParamGrid, attributes(param))]
pub fn derive_param_grid(input: TokenStream) -> TokenStream {
    // Generates iter_all() and to_suffix()
}
```

### 4. Runner (`vortex-threshold-runner`)

```rust
pub fn main() {
    let args = Args::parse();

    // Collect all registered benchmarks
    let benchmarks: Vec<&dyn Benchmark> = BENCHMARKS
        .iter()
        .filter(|b| args.matches(b.name()))
        .collect();

    if args.list {
        for b in &benchmarks {
            println!("{}", b.name());
        }
        return;
    }

    for bench in benchmarks {
        let results = bench.run(args.samples, args.seed);
        results.print();

        if let Some(path) = &args.output {
            results.save(path).unwrap();
        }
    }
}
```

## Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1. Measurement | Measurer, warmup, CI | ✅ Done |
| 2. Distribution API | Builder with distributions | 🚧 In Progress |
| 3. ParamGrid | Derive macro for params | 🔲 Not Started |
| 4. Runner Macro | `#[threshold_bench]` | 🔲 Not Started |
| 5. CLI Runner | Filter, list, output | 🔲 Not Started |
| 6. Output | Terminal table, JSON | 🔲 Not Started |
| 7. CI Integration | Storage, codegen | 🔲 Not Started |

## Crate Structure

```
vortex-threshold-traits/     Core traits, builder, results
vortex-threshold-macros/     #[threshold_bench], #[derive(ParamGrid)]
vortex-threshold-runner/     CLI runner, main()
vortex-threshold/            Re-exports everything (user-facing)
```

## Open Questions

- Should stats be optional? (Currently required for variant signature)
- How to handle variants that don't need stats? (Use `_` or separate method?)
- Should we support async generators for IO-bound data loading?
