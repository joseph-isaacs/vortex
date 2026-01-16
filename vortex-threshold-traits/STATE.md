# ISA Threshold Finder - State Document

## Overview

The ISA Threshold Finder is a system for automatically detecting which algorithm implementation performs best across different data distributions and CPU architectures. It enables Vortex to select optimal implementations by benchmarking against realistic workloads.

## Core Concept: Distribution-Based Benchmarking

Instead of searching over artificial parameter grids, we benchmark against **named data distributions** that represent real workloads:

```
seed → Data → Stats (computed for reporting)
         ↓
    run variants → measure
         ↓
    aggregate across distributions → find optimal variant
```

Key insight: Generate realistic data first, then compute its stats for analysis—don't try to generate data targeting specific stats.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           Crate Structure                                │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  vortex-threshold-traits     Core trait definitions & builder API        │
│  ├── StatsBench              Distribution-based benchmark builder        │
│  ├── Distribution            Named data generator (seed → Data)          │
│  ├── Variant                 Algorithm variant with CPU feature reqs     │
│  ├── CpuClass                Runtime CPU detection (Intel/AMD/ARM)       │
│  ├── Measurer                Statistical measurement infrastructure      │
│  └── BenchmarkStorage        Trait for result persistence                │
│                                                                          │
│  vortex-threshold-runner     CLI tool for running benchmarks             │
│  ├── Optimizer               Finds variant minimizing aggregate runtime  │
│  ├── examples/               Popcount, Sum, Rank demos                   │
│  └── storage/sqlite          SQLite backend for result persistence       │
│                                                                          │
│  vortex-threshold-aggregator Merges results, generates Rust code         │
│  └── Generates LazyLock      Static dispatch tables per CpuClass         │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

## API: Distribution-Based Builder

```rust
use vortex_threshold_traits::StatsBench;

StatsBench::new("rank")
    // Named distributions - each generates realistic data
    .distribution("uniform_sparse", |seed| gen_bitmap(seed, density: 0.1, len: 1..10000))
    .distribution("uniform_dense", |seed| gen_bitmap(seed, density: 0.9, len: 1..10000))
    .distribution("zipfian", |seed| gen_zipfian_bitmap(seed))
    .distribution("real_parquet", |seed| sample_from_parquet_corpus(seed))

    // Optional: compute stats from generated data (for analysis/reporting)
    .stats(|data| RankStats { len: data.len(), density: compute_density(data) })

    // Algorithm variants to compare
    .baseline("naive", rank_naive)
    .variant("simd", rank_simd)
    .variant("chunked", rank_chunked)
    .variant_with_features("avx2", &["avx2"], rank_avx2)

    // Optional: weight distributions by importance
    .weight("real_parquet", 2.0)  // real data matters more

    .build();
```

## Optimization Goal

The optimizer finds the variant (or variant parameters) that **minimizes aggregate runtime across all distributions**:

```
minimize: Σ (weight[dist] × mean_runtime[variant, dist])
```

Results are reported per-distribution:

```
Distribution       | naive    | simd     | chunked  | winner
-------------------|----------|----------|----------|--------
uniform_sparse     | 120µs    | 45µs     | 80µs     | simd
uniform_dense      | 450µs    | 50µs     | 200µs    | simd
zipfian            | 200µs    | 60µs     | 90µs     | simd
real_parquet       | 180µs    | 55µs     | 85µs     | simd

Aggregate winner: simd (weighted total: 385µs)
```

## Data Flow

1. **Define distributions**: Name your workloads, provide generator functions
2. **Generate samples**: For each distribution, generate N samples with different seeds
3. **Compute stats** (optional): Extract stats from each sample for grouping/analysis
4. **Benchmark**: Run each variant on each sample, measure with proper warmup/iteration
5. **Aggregate**: Compute per-distribution means, find weighted-optimal variant
6. **Report**: Show results grouped by distribution, highlight winners

## Storage Layer

### SQLite Backend (Optional Feature)

```rust
use vortex_threshold_runner::storage::SqliteStorage;

let storage = SqliteStorage::open("benchmarks.db")?;

// Store measurements grouped by distribution
storage.store_measurements(&measurements)?;

// Query by algorithm, distribution, variant, CPU class
let query = MeasurementQuery::new()
    .algorithm("rank")
    .distribution("real_parquet")
    .cpu_class(CpuClass::IntelSapphire)
    .since(yesterday);
let results = storage.query_measurements(&query)?;

// Compare performance between commits
let diffs = storage.compare_commits("abc123", "def456")?;
```

## CI Integration

GitHub Actions workflow (`.github/workflows/isa-thresholds.yml`) runs benchmarks on:

| Architecture | Runner | CPU Features |
|-------------|--------|--------------|
| Intel Sapphire Rapids | `runs-on: intel-sapphire` | AVX-512 |
| Intel Ice Lake | `runs-on: intel-icelake` | AVX-512 |
| AMD Genoa (Zen 4) | `runs-on: amd-genoa` | AVX-512 |
| AMD Milan (Zen 3) | `runs-on: amd-milan` | AVX2 |
| AWS Graviton 3 | `runs-on: graviton3` | NEON, SVE |
| AWS Graviton 2 | `runs-on: graviton2` | NEON |

## Current Status

### Completed
- [x] Measurement infrastructure (`Measurer`, warmup, batching)
- [x] Statistical analysis (IQR outlier removal, bootstrap confidence intervals)
- [x] CPU feature detection and variant availability
- [x] CPU class detection (Intel/AMD/ARM families)
- [x] Scale types (log2, linear, steps, explicit)
- [x] SQLite storage backend (schema exists)
- [x] GitHub Actions workflow template
- [x] Example benchmarks (popcount, sum)

### In Progress
- [ ] **Distribution-based API** - replace StatsGrid with named distributions
- [ ] Optimizer for aggregate runtime minimization
- [ ] Distribution weighting

### Not Yet Implemented
- [ ] JSON export (`.save()`, `.to_json()`)
- [ ] Per-variant parameter tuning (ParamGrid derive macro)
- [ ] Automatic PR comments with performance changes
- [ ] Integration with actual Vortex algorithms (rank, select, etc.)
- [ ] Dashboard/visualization for trends

## File Locations

```
vortex/
├── vortex-threshold-traits/
│   ├── src/
│   │   ├── lib.rs           # Re-exports, CpuClass, Variant
│   │   ├── bench.rs         # StatsBench builder API
│   │   ├── measure.rs       # Measurer, statistical analysis
│   │   ├── scale.rs         # Scale types (log2, linear, etc.)
│   │   ├── stats.rs         # StatsPoint, StatsGrid (being replaced)
│   │   └── storage.rs       # BenchmarkStorage trait
│   ├── examples/
│   │   └── target_api.rs    # Example usage (needs update)
│   └── Cargo.toml
│
├── vortex-threshold-runner/
│   ├── src/
│   │   ├── main.rs          # CLI, benchmark execution
│   │   ├── examples/
│   │   │   ├── mod.rs
│   │   │   ├── popcount.rs  # Popcount benchmark
│   │   │   └── sum.rs       # Sum benchmark
│   │   └── storage/
│   │       ├── mod.rs
│   │       └── sqlite.rs    # SQLite implementation
│   └── Cargo.toml
│
├── vortex-threshold-aggregator/
│   ├── src/
│   │   └── main.rs          # Merges JSON, generates Rust code
│   └── Cargo.toml
│
└── .github/workflows/
    └── isa-thresholds.yml   # Multi-architecture CI workflow
```

## Usage

### Running Benchmarks Locally

```bash
# Build the runner
cargo build -p vortex-threshold-runner --release

# Run benchmarks
./target/release/threshold-runner --output results.json

# Run specific algorithm
./target/release/threshold-runner --algorithm rank --output results.json
```

### Generated Code Example

```rust
use std::sync::LazyLock;
use vortex_threshold_traits::CpuClass;

/// Best variant per CPU class, determined by aggregate performance across distributions
static RANK_DISPATCH: LazyLock<fn(&RankData) -> usize> = LazyLock::new(|| {
    match CpuClass::detect() {
        CpuClass::IntelSapphire => rank_avx2,
        CpuClass::AmdGenoa => rank_avx2,
        CpuClass::Graviton3 => rank_neon,
        _ => rank_simd,  // default
    }
});
```

## Design Decisions

1. **Distribution-based benchmarking**: Test against named, realistic workloads rather than artificial parameter grids
2. **Aggregate optimization**: Find variant that minimizes weighted total across all distributions
3. **seed → Data → Stats**: Generate data first, compute stats for reporting (not the reverse)
4. **Static code generation**: Zero runtime cost, best variant baked into binary per CpuClass
5. **SQLite for persistence**: Rich queries, regression detection, no external dependencies
6. **Weighted distributions**: Real-world data distributions can matter more than synthetic ones
