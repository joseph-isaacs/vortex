# ISA Threshold Finder - Implementation TODO

## Overview

This tracks implementation progress against the design in `plan/`. The goal is a system that:
1. Benchmarks algorithm variants with tunable parameters
2. Computes stats from input data (len, density, etc.)
3. Finds crossover points where one variant becomes faster
4. Supports two modes: `bench()` (quick) and `search()` (full grid)

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

## Phase 2: Stats-Based API ✅ COMPLETE

**Priority: HIGH** - Core API redesign

- [x] **2.1 Define `Stats` concept**
  - Location: `vortex-threshold-traits/src/stats.rs`
  - [x] `StatsPoint` - dynamic stats container with named dimensions
  - [x] Stats computed from Data via user-provided closure

- [x] **2.2 Implement `StatsGrid`**
  - [x] Builder for multi-dimensional stats grids
  - [x] `.dimension("len", Scale::log2(6, 20))`
  - [x] `.dimension("density", Scale::steps(0.0, 1.0, 0.1))`
  - [x] Iterator over all stats combinations (`StatsGridIter`)

- [x] **2.3 Implement `Scale` enum**
  - Location: `vortex-threshold-traits/src/scale.rs`
  - [x] `Scale::log2`, `Scale::log`, `Scale::linear`, `Scale::steps`, `Scale::explicit`

- [x] **2.4 Implement `StatsBench` builder**
  - Location: `vortex-threshold-traits/src/bench.rs`
  - [x] `.stats()`, `.generate()`, `.stats_grid()`, `.baseline()`, `.variant()`

---

## Phase 3: Generic Type Support ✅ COMPLETE

**Priority: HIGH** - Enables benchmarks generic over element types

The problem: When benchmarking algorithms like filter+add that work with different element types (u8, u16, u32, u64), we want ONE StatsBench instance per piece of logic, not one per type. The type enumeration should be INTERNAL to the StatsBench.

### Solution: Bevy-style `for_types` API

- [x] **3.1 Type-erased traits**
  - `ErasedData` - wrapper for type-erased data with `as_any()` for downcast
  - `ErasedConfig` - type-erased config with `generate_erased()`, `run_variant_erased()`

- [x] **3.2 ForType<T> trait pattern**
  - Marker struct pattern (like Bevy's SystemParam)
  - `impl<T: Element> ForType<T> for FilterAddBench { fn config() -> ConcreteConfig<T> }`
  - No macros needed - trait impl provides type-specific config

- [x] **3.3 TypeList trait with tuple impls**
  - `impl<Marker, A, B, C> TypeList<Marker> for (A, B, C)`
  - Registers configs for each type in the tuple
  - Supports 1-4 types via tuple impls

- [x] **3.4 Working example**
  - See: `examples/generic_api_demo.rs`
  - Demonstrates: type registration, type-erased search, verification
  - Run: `cargo run --example generic_api_demo -p vortex-threshold-traits`

### User-facing API

```rust
// 1. Define marker struct
struct FilterAddBench;

// 2. Implement ForType<T> for each element type
impl<T: Element> ForType<T> for FilterAddBench {
    fn config() -> ConcreteConfig<T> {
        ConcreteConfig {
            _phantom: PhantomData,
            baseline: ("add_then_filter", add_then_filter::<T>),
            variants: vec![("filter_then_add", filter_then_add::<T>)],
        }
    }
}

// 3. Build benchmark with for_types
let bench = UnifiedStatsBench::new("filter_add")
    .len_values(vec![1024, 4096])
    .density_values(vec![0.1, 0.5, 0.9])
    .for_types::<FilterAddBench, (u8, u16, u32)>()  // Bevy-style!
    .build();

// 4. Search runs over all (len, density, type) combinations
bench.search();
```

---

## Phase 4: Per-Variant Parameters

**Priority: MEDIUM** - Enables parameter tuning per variant

- [ ] **4.1 Design `ParamGrid` trait**
- [ ] **4.2 Implement `ParamGrid` derive macro** (new crate)
- [ ] **4.3 Update `StatsBench` with `.variant_with_params::<P>()`**

---

## Phase 5: Two Modes (Bench vs Search) ⚠️ PARTIAL

- [x] **5.1 `BenchRunner`** - `.at()`, `.run()`, `.print()`
- [x] **5.2 `SearchRunner`** - grid search, winners detection
- [x] **5.3 Entry points** - `.bench()`, `.search()`
- [ ] `.refine()` - binary search refinement
- [ ] Crossover detection

---

## Phase 6: Output & Display ⚠️ PARTIAL

- [x] Basic terminal output with print()
- [ ] Divan-style aligned columns
- [ ] JSON export (`.save()`, `.to_json()`)
- [ ] Comparison mode

---

## Phase 7: Refinement, Storage & CI

- [ ] Binary search refinement
- [ ] SQLite storage
- [ ] Code generation
- [ ] CI workflow

---

## Current Progress

| Phase | Status | Notes |
|-------|--------|-------|
| 1. Measurement | **DONE** | `Measurer`, `BatchSize`, IQR, bootstrap CI |
| 2. Stats API | **DONE** | `Scale`, `StatsGrid`, `StatsBench` builder |
| 3. Generic Types | **DONE** | Bevy-style `for_types`, `ForType<T>`, `TypeList` |
| 4. ParamGrid | Not started | No derive macro yet |
| 5. Two Modes | **Partial** | Basic BenchRunner/SearchRunner work |
| 6. Output | **Partial** | Basic print(), no JSON/save |
| 7. Refinement/CI | Not started | Trait exists only |

---

## Next Steps

1. ~~**Phase 1** (Measurement) - DONE~~
2. ~~**Phase 2** (Stats API) - DONE~~
3. ~~**Phase 3** (Generic Types) - DONE~~
4. **Phase 4** (ParamGrid derive) - enables tunable per-variant params
5. **Phase 6** (Output) - JSON export, better terminal display

---

## Examples

| Example | Description | Status |
|---------|-------------|--------|
| `target_api.rs` | Rank benchmark with StatsBench | Working |
| `generic_api_demo.rs` | Generic type support with ForType trait | Working |
| `filter_plus_api.rs` | API design document (requires vortex_array) | Design doc only |
