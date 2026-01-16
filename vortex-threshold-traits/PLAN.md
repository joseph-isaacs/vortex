# ISA Threshold Finder - Planning Document

## Goal

Find the optimal algorithm variant for each CPU class by benchmarking against realistic, named data distributions. The optimizer minimizes aggregate runtime across all distributions.

## Core Design: Distribution-Based Benchmarking

### The Problem with Parameter Grids

Old approach: Define a grid of stats (len × density × ...) and try to generate data matching each point.

Issues:
- Hard to generate data matching specific stats
- Artificial - real workloads don't come from grids
- Combinatorial explosion with multiple dimensions

### New Approach: Named Distributions

Define named distributions that represent real workloads:

```rust
StatsBench::new("rank")
    .distribution("uniform_sparse", |seed| gen_bitmap(seed, density: 0.1))
    .distribution("uniform_dense", |seed| gen_bitmap(seed, density: 0.9))
    .distribution("zipfian", |seed| gen_zipfian(seed))
    .distribution("real_parquet", |seed| sample_from_corpus(seed))
    .weight("real_parquet", 2.0)  // real data matters more

    .stats(|data| RankStats::compute(data))  // optional, for reporting

    .baseline("naive", rank_naive)
    .variant("simd", rank_simd)
    .build();
```

Data flow:
```
seed → Data → Stats (optional, for reporting)
         ↓
    benchmark variants
         ↓
    aggregate across distributions
         ↓
    find optimal variant
```

## Components

### 1. Benchmark Core (`vortex-threshold-traits`)

**Distribution struct:**
```rust
struct Distribution<D> {
    name: String,
    generator: Box<dyn Fn(u64) -> D>,
    weight: f64,  // default 1.0
}
```

**StatsBench builder:**
```rust
StatsBench::new("name")
    .distribution(name, generator)  // required, at least one
    .weight(name, weight)           // optional
    .stats(fn)                      // optional, for reporting
    .baseline(name, fn)             // required
    .variant(name, fn)              // optional, zero or more
    .build()
```

**Measurer** (existing, keep as-is):
- Warmup, batching, black_box
- IQR outlier removal
- Bootstrap confidence intervals

### 2. Optimizer (`vortex-threshold-runner`)

**Aggregate scoring:**
```
score(variant) = Σ weight[dist] × mean_runtime[variant, dist]
```

**Output:**
```
Distribution       | naive    | simd     | chunked  | winner
-------------------|----------|----------|----------|--------
uniform_sparse     | 120µs    | 45µs     | 80µs     | simd
uniform_dense      | 450µs    | 50µs     | 200µs    | simd
zipfian            | 200µs    | 60µs     | 90µs     | simd
real_parquet (2x)  | 180µs    | 55µs     | 85µs     | simd

Aggregate winner: simd
Weighted scores: naive=1130µs, simd=260µs, chunked=535µs
```

### 3. Code Generation (`vortex-threshold-aggregator`)

Generate dispatch tables per CPU class:

```rust
static RANK_IMPL: LazyLock<fn(&Data) -> Output> = LazyLock::new(|| {
    match CpuClass::detect() {
        CpuClass::IntelSapphire => rank_avx2,
        CpuClass::Graviton3 => rank_neon,
        _ => rank_simd,
    }
});
```

## Implementation Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1. Measurement | Measurer, stats, CI | ✅ Done |
| 2. Distribution API | Replace StatsGrid | 🚧 In Progress |
| 3. Optimizer | Aggregate scoring | 🔲 Not Started |
| 4. ParamGrid | Per-variant params | 🔲 Not Started |
| 5. Output | JSON, terminal | 🔲 Not Started |
| 6. CI | Storage, codegen | 🔲 Not Started |

## Open Questions

- How many samples per distribution? (Fixed N, or adaptive?)
- How to handle distributions with different data sizes?
- Should we support distribution "families" (e.g., sparse with varying sizes)?

## Sub-Documents

- [plan/benchmark/plan.md](./plan/benchmark/plan.md) - Measurement details
- [plan/benchmark/measurement.md](./plan/benchmark/measurement.md) - Statistical methods
- [plan/storage/plan.md](./plan/storage/plan.md) - Persistence layer
- [plan/runners/plan.md](./plan/runners/plan.md) - CI infrastructure
