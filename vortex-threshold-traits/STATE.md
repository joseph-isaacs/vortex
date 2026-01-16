# ISA Threshold Finder - State Document

## Overview

The ISA Threshold Finder is a system for automatically detecting which algorithm implementation (and parameters) performs best across different data distributions and CPU architectures. It enables Vortex to select optimal implementations by benchmarking against realistic workloads.

## Core Concept: Distribution-Based Benchmarking

```
seed → Data → Stats (passed to variants)
               ↓
         variant(data, stats) → Output
         variant(data, stats, params) → Output  (parameterized)
               ↓
         aggregate across distributions
               ↓
         find optimal (variant, params)
```

Key insight: Generate realistic data first, compute stats, pass stats to variants (for adaptive logic), report stats per distribution.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Crate Structure                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  vortex-threshold-traits/      Core traits, builder, measurer            │
│  ├── StatsBench                Distribution-based benchmark builder      │
│  ├── Benchmark                 Trait for runnable benchmarks             │
│  ├── ParamGrid                 Trait for parameter iteration             │
│  ├── Measurer                  Deferred-drop measurement (Divan-style)   │
│  └── CpuClass                  Runtime CPU detection                     │
│                                                                          │
│  vortex-threshold-macros/      Proc macros                               │
│  ├── #[threshold_bench]        Auto-registration (like Divan)            │
│  └── #[derive(ParamGrid)]      Parameter grid iteration                  │
│                                                                          │
│  vortex-threshold-runner/      CLI runner                                │
│  ├── threshold_runner::main()  Collects and runs benchmarks              │
│  └── storage/sqlite            Optional persistence                      │
│                                                                          │
│  vortex-threshold/             Re-exports (user-facing crate)            │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## API

### Define a Benchmark

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
        // 1. Distributions (seed → Data)
        .distribution("sparse", gen_sparse)
        .distribution("dense", gen_dense)
        .distribution("zipfian", gen_zipfian)
        .weight("zipfian", 2.0)

        // 2. Stats (Data → Stats, passed to variants)
        .stats(RankStats::compute)

        // 3. Variants (receive data and stats)
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

### Define Multiple Benchmarks

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

### Run

```bash
cargo run --release                           # all benchmarks
cargo run --release -- rank                   # specific benchmark
cargo run --release -- 'pop*'                 # pattern match
cargo run --release -- --list                 # list available
cargo run --release -- --samples 100          # configure
cargo run --release -- --output results.json  # save results
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

## Measurement Inner Loop

Based on [Divan](https://docs.rs/divan) and [Criterion](https://docs.rs/criterion).

### Timing Guarantees

| Phase | Timed? |
|-------|--------|
| Input generation (`setup()`) | NO |
| Routine execution | YES |
| Output drop | NO (deferred) |
| Input drop | NO |

### Implementation

```rust
fn collect_samples(&self, setup: &impl Fn() -> I, routine: &impl Fn(&I) -> O) -> Vec<f64> {
    while measurement_start.elapsed() < self.measurement_time {
        // 1. Generate inputs OUTSIDE timing
        let inputs: Vec<I> = (0..iters_per_batch).map(|_| setup()).collect();

        // 2. Pre-allocate outputs
        let mut outputs: Vec<O> = Vec::with_capacity(iters_per_batch);

        // 3. Time ONLY routine execution
        let batch_start = Instant::now();
        for input in &inputs {
            outputs.push(black_box(routine(black_box(input))));
        }
        let batch_elapsed = batch_start.elapsed();

        samples.push(batch_elapsed.as_nanos() as f64 / iters_per_batch as f64);

        // 4. Drops OUTSIDE timing (deferred)
        drop(outputs);
        drop(inputs);
    }
}
```

### Key Techniques

- **`black_box` on both input AND output** - Prevents compiler optimization
- **Deferred output drop** - Outputs collected, dropped after timing
- **Batching** - Amortizes `Instant::now()` overhead
- **Warmup** - Stabilizes CPU frequency, fills caches

## Configuration

Priority: **CLI > runtime > builder > defaults**

```rust
// Builder defaults
StatsBench::new("rank")
    .warmup(Duration::from_millis(100))
    .measurement_time(Duration::from_millis(500))
    ...

// Runtime override
bench.run()
    .warmup(Duration::from_millis(50))
    .samples(100)
    .execute();
```

```bash
# CLI override (highest priority)
cargo run --release -- --warmup 50 --measurement 200 --samples 100
```

## Current Status

### Completed ✅
- [x] Measurement infrastructure with deferred drop
- [x] Statistical analysis (IQR outlier removal, bootstrap CI)
- [x] CPU feature detection and variant availability
- [x] CPU class detection (Intel/AMD/ARM families)
- [x] Scale types (log2, linear, steps, explicit)
- [x] SQLite storage backend schema

### In Progress 🚧
- [ ] Distribution-based builder API
- [ ] ParamGrid trait and derive macro
- [ ] `#[threshold_bench]` registration macro

### Not Started 🔲
- [ ] CLI runner with filter/list
- [ ] Terminal table output
- [ ] JSON export
- [ ] CI workflow integration
- [ ] Code generation for dispatch tables

## File Locations

```
vortex/
├── vortex-threshold-traits/
│   ├── src/
│   │   ├── lib.rs           # Re-exports, CpuClass, Variant
│   │   ├── bench.rs         # StatsBench builder (needs update)
│   │   ├── measure.rs       # Measurer with deferred drop ✅
│   │   ├── scale.rs         # Scale types
│   │   └── storage.rs       # BenchmarkStorage trait
│   ├── examples/
│   │   └── target_api.rs    # Target API example
│   └── PLAN.md              # Detailed implementation plan
│
├── vortex-threshold-macros/  # NEW - proc macros
│   └── (to be created)
│
├── vortex-threshold-runner/
│   ├── src/
│   │   ├── main.rs          # CLI runner
│   │   └── storage/sqlite   # SQLite backend
│   └── Cargo.toml
│
└── vortex-threshold/         # NEW - re-exports
    └── (to be created)
```

## Design Decisions

1. **Distribution-based benchmarking** - Named distributions, not parameter grids
2. **Stats passed to variants** - Enables adaptive dispatch logic
3. **ParamGrid derive macro** - Type-safe parameter iteration
4. **Macro-based registration** - `#[threshold_bench]` like Divan
5. **Deferred drop** - Output drops excluded from timing
6. **Weighted distributions** - Real data can matter more
7. **Config priority** - CLI > runtime > builder > defaults
8. **Static code generation** - Zero runtime cost dispatch tables
