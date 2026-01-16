# Benchmark Runner Design

## Overview

The benchmark runner measures algorithm variants across named data distributions and finds the optimal (variant, params) combination that minimizes aggregate runtime.

## Key Concepts

1. **Distribution**: Named data generator (`seed → Data`)
2. **Data**: Raw input to the algorithm
3. **Stats**: Computed from data, passed to variants (for adaptive logic)
4. **Variants**: Algorithm implementations that receive `(data, stats)` or `(data, stats, params)`
5. **ParamGrid**: Per-variant tunable parameters (derive macro generates iteration)
6. **Aggregate Winner**: Variant that minimizes weighted runtime across all distributions

## Data Flow

```
seed → Data → Stats
               ↓
         variant(data, stats) → Output
         variant(data, stats, params) → Output
               ↓
         aggregate across distributions
               ↓
         find optimal (variant, params)
```

## API

### Defining a Benchmark

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
        // 1. Distributions
        .distribution("sparse", gen_sparse)
        .distribution("dense", gen_dense)
        .distribution("zipfian", gen_zipfian)
        .weight("zipfian", 2.0)

        // 2. Stats (passed to variants)
        .stats(RankStats::compute)

        // 3. Variants
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

### Running Benchmarks

```rust
fn main() {
    threshold_runner::main();
}
```

```bash
cargo run --release                    # all benchmarks
cargo run --release -- rank            # specific benchmark
cargo run --release -- --list          # list available
cargo run --release -- --samples 100   # configure
cargo run --release -- --output r.json # save results
```

## ParamGrid Derive Macro

```rust
#[derive(Clone, Debug, ParamGrid)]
struct ChunkedParams {
    #[param(values = [4, 8, 16, 32])]
    chunk_size: usize,
}

// Generates:
impl ParamGrid for ChunkedParams {
    fn iter_all() -> impl Iterator<Item = Self> {
        [4, 8, 16, 32].into_iter().map(|chunk_size| Self { chunk_size })
    }

    fn to_suffix(&self) -> String {
        format!("[{}]", self.chunk_size)
    }
}
```

### Multi-parameter grids

```rust
#[derive(Clone, Debug, ParamGrid)]
struct SimdParams {
    #[param(values = [2, 4, 8])]
    unroll: usize,
    #[param(values = [0, 64, 128])]
    prefetch: usize,
}

// Generates cartesian product: 3 × 3 = 9 combinations
// to_suffix() returns "[u=2,p=0]", "[u=2,p=64]", etc.
```

## Benchmark Run Loop

```rust
pub fn execute(self) -> BenchResults {
    let measurer = Measurer::new()
        .warmup_time(self.warmup)
        .measurement_time(self.measurement_time);

    let mut results = BenchResults::new(&self.bench.name);

    // For each distribution
    for dist in &self.bench.distributions {
        // Generate samples and compute stats
        let samples: Vec<(D, S)> = (0..self.num_samples)
            .map(|i| {
                let data = (dist.generator)(self.base_seed + i as u64);
                let stats = (self.bench.stats_fn)(&data);
                (data, stats)
            })
            .collect();

        // For each variant × param combination
        for (variant_name, run_fn) in self.bench.all_variant_combinations() {
            let measurements: Vec<MeasurementResult> = samples
                .iter()
                .map(|(data, stats)| {
                    measurer.measure(
                        || data.clone(),
                        |d| run_fn(d, stats),
                    )
                })
                .collect();

            results.add(&dist.name, &variant_name, aggregate(measurements));
        }
    }

    results.compute_winners();
    results
}
```

## Output

```
Benchmark: rank
Samples: 100 per distribution

Distribution    | avg len | avg density | naive  | simd   | chunked[8] | winner
----------------|---------|-------------|--------|--------|------------|--------
sparse          | 5012    | 0.12        | 120µs  | 95µs   | 85µs       | chunked[8]
dense           | 4998    | 0.88        | 450µs  | 180µs  | 170µs      | chunked[8]
zipfian (2.0x)  | 5105    | 0.45        | 200µs  | 140µs  | 130µs      | chunked[8]

Best params: chunked { chunk_size: 8 }
Aggregate winner: chunked[8] (weighted score: 515µs)
```

## Files

```
vortex-threshold-traits/
├── src/
│   ├── lib.rs           # Core traits, ParamGrid
│   ├── bench.rs         # StatsBench builder
│   ├── measure.rs       # Measurer with deferred drop
│   └── results.rs       # BenchResults, output formatting
│
vortex-threshold-macros/
├── src/
│   └── lib.rs           # #[threshold_bench], #[derive(ParamGrid)]
│
vortex-threshold-runner/
├── src/
│   └── main.rs          # CLI, threshold_runner::main()
```

## See Also

- [measurement.md](measurement.md) - Inner loop and timing details
- [../PLAN.md](../PLAN.md) - Full implementation plan
- [../STATE.md](../STATE.md) - Current status
