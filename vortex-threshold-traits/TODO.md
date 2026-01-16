# ISA Threshold Finder - Implementation TODO

## Overview

This tracks implementation progress. The goal is a system that:
1. Benchmarks algorithm variants against **named data distributions**
2. Computes stats from generated data (for analysis/reporting)
3. Finds the variant that **minimizes aggregate runtime** across distributions
4. Supports weighted distributions (real-world data can matter more)

---

## Phase 1: Measurement Quality ✅ COMPLETE

**Priority: HIGH** - Foundation for accurate benchmarks

- [x] **1.1 Implement `Measurer` struct**
  - Location: `vortex-threshold-traits/src/measure.rs`
  - [x] `black_box` on BOTH inputs AND outputs
  - [x] Warmup phase (~100ms default)
  - [x] Batched iterations (~100µs per batch)
  - [x] Input generation outside timed section
  - [x] Drops outside timed section

- [x] **1.2 Implement `BatchSize` enum**
  - [x] `Small`, `Large`, `PerIteration`, `NumIterations(usize)`

- [x] **1.3 Statistical analysis**
  - [x] IQR outlier removal
  - [x] Bootstrap confidence intervals (95%)
  - [x] `MeasurementResult` struct with median, mean, stddev, CI bounds, sample count

---

## Phase 2: Distribution-Based API 🚧 IN PROGRESS

**Priority: HIGH** - Core API redesign

### 2.1 Distribution Concept (NEW)

Replace `StatsGrid` with named distributions:

```rust
StatsBench::new("rank")
    .distribution("uniform_sparse", |seed| gen_bitmap(seed, 0.1))
    .distribution("uniform_dense", |seed| gen_bitmap(seed, 0.9))
    .distribution("zipfian", |seed| gen_zipfian(seed))
    .distribution("real_parquet", |seed| sample_parquet(seed))
```

- [ ] **2.1.1 Define `Distribution` struct**
  - Name (string identifier)
  - Generator function: `Fn(u64) -> Data`
  - Optional weight (default 1.0)

- [ ] **2.1.2 Update `StatsBench` builder**
  - [x] `.baseline()`, `.variant()` - existing, keep as-is
  - [ ] `.distribution(name, generator)` - NEW, replaces `.stats_grid()`
  - [ ] `.weight(name, weight)` - NEW, for distribution importance
  - [ ] Remove `.stats_grid()`, `.generate()` from required API

- [ ] **2.1.3 Optional stats computation**
  - [ ] `.stats(|data| Stats)` - optional, for reporting only
  - Stats are computed FROM data, not used to generate data

### 2.2 Keep from Old API

- [x] `Scale` enum (log2, linear, steps, explicit) - useful for param ranges
- [x] `Measurer` - measurement infrastructure
- [x] `Variant` with CPU feature requirements

---

## Phase 3: Optimizer 🔲 NOT STARTED

**Priority: HIGH** - Core value proposition

- [ ] **3.1 Aggregate runtime calculation**
  ```
  score(variant) = Σ weight[dist] × mean_runtime[variant, dist]
  ```

- [ ] **3.2 Winner detection**
  - Per-distribution winners
  - Aggregate winner (minimizes weighted sum)
  - Confidence intervals on winner selection

- [ ] **3.3 Results struct**
  ```rust
  struct BenchResults {
      per_distribution: HashMap<String, DistributionResults>,
      aggregate_winner: String,
      aggregate_scores: HashMap<String, f64>,
  }
  ```

---

## Phase 4: Per-Variant Parameters 🔲 NOT STARTED

**Priority: MEDIUM** - Enables parameter tuning

- [ ] **4.1 `ParamGrid` trait**
- [ ] **4.2 `ParamGrid` derive macro** (new crate)
- [ ] **4.3 `.variant_with_params::<P>(name, fn)`**

---

## Phase 5: Output & Export 🔲 NOT STARTED

**Priority: MEDIUM** - CI integration

- [ ] **5.1 Terminal output**
  - Per-distribution table
  - Aggregate summary
  - Winner highlighting

- [ ] **5.2 JSON export**
  - `.save(path)`
  - `.to_json()`

- [ ] **5.3 Comparison mode**
  - Compare against baseline commit

---

## Phase 6: Storage & CI 🔲 NOT STARTED

**Priority: LOW** - Scale to CI

- [ ] SQLite schema update for distributions
- [ ] Code generation for dispatch tables
- [ ] CI workflow testing

---

## Current Progress

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Measurement | **DONE** | `Measurer`, `BatchSize`, IQR, bootstrap CI |
| 2. Distribution API | **In Progress** | Replacing StatsGrid with named distributions |
| 3. Optimizer | Not started | Aggregate minimization |
| 4. ParamGrid | Not started | Per-variant param tuning |
| 5. Output | Not started | JSON, terminal display |
| 6. Storage/CI | Not started | SQLite, code gen |

---

## Next Steps

1. **Implement Distribution struct** - simple wrapper around name + generator
2. **Update StatsBench builder** - add `.distribution()`, make `.stats()` optional
3. **Implement Optimizer** - aggregate scoring, winner detection
4. **Update example** - show new distribution-based API
