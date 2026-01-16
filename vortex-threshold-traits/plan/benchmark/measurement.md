# Measurement Inner Loop

Based on best practices from [Divan](https://docs.rs/divan) and [Criterion](https://docs.rs/criterion).

## Key Principles

1. **`black_box` on BOTH inputs AND outputs** - Prevents compiler optimization
2. **Input generation OUTSIDE timing** - Don't measure allocation
3. **Output drop OUTSIDE timing** - Deferred drop (Divan-style)
4. **Batch iterations** - Amortize `Instant::now()` overhead
5. **Warmup phase** - Stabilize CPU frequency, fill caches

## Timing Guarantees

| Phase | Timed? | Notes |
|-------|--------|-------|
| Input generation (`setup()`) | NO | Allocation excluded |
| Pre-allocate output vec | NO | `Vec::with_capacity` |
| Routine execution | **YES** | Only this is measured |
| Output collection | YES | But not drop |
| Output drop | NO | Deferred until after timing |
| Input drop | NO | After batch complete |

## Implementation

```rust
fn collect_samples<I, O, S, R>(
    &self,
    setup: &S,
    routine: &R,
    iters_per_batch: usize,
) -> Vec<f64>
where
    S: Fn() -> I,
    R: Fn(&I) -> O,
{
    let mut samples = Vec::new();
    let measurement_start = Instant::now();

    while measurement_start.elapsed() < self.measurement_time {
        // 1. Generate batch of inputs OUTSIDE timing
        let inputs: Vec<I> = (0..iters_per_batch).map(|_| setup()).collect();

        // 2. Pre-allocate output storage to avoid allocation during timing
        let mut outputs: Vec<O> = Vec::with_capacity(iters_per_batch);

        // 3. Time ONLY the routine execution
        //    - black_box(input) prevents input caching across iterations
        //    - black_box(output) prevents dead code elimination
        //    - outputs collected, NOT dropped, during timing
        let batch_start = Instant::now();
        for input in &inputs {
            outputs.push(black_box(routine(black_box(input))));
        }
        let batch_elapsed = batch_start.elapsed();

        // 4. Record per-iteration time
        let per_iter_ns = batch_elapsed.as_nanos() as f64 / iters_per_batch as f64;
        samples.push(per_iter_ns);

        // 5. Drops happen HERE, OUTSIDE timing (deferred drop)
        drop(outputs);
        drop(inputs);
    }

    samples
}
```

## Why Deferred Drop?

Without deferred drop:
```rust
for input in &inputs {
    black_box(routine(black_box(input)));
    // Output dropped HERE, inside timing!
}
```

With deferred drop:
```rust
let mut outputs = Vec::with_capacity(iters_per_batch);
for input in &inputs {
    outputs.push(black_box(routine(black_box(input))));
    // Output NOT dropped - just stored
}
// batch timing ends
drop(outputs);  // Outputs dropped HERE, outside timing
```

This matches Divan's approach where "when a benchmarked function returns a value, it will not be dropped until after the current sample loop is finished."

## Why `black_box` on Both?

```rust
// BAD: Compiler might cache `routine(input)` result
for input in &inputs {
    black_box(routine(input));
}

// GOOD: Compiler can't optimize across iterations
for input in &inputs {
    outputs.push(black_box(routine(black_box(input))));
}
```

- `black_box(input)` - Prevents compiler from caching input across iterations
- `black_box(output)` - Prevents dead code elimination

## Batch Size

The batch size controls memory vs. overhead tradeoff:

| BatchSize | Overhead | Memory | Use Case |
|-----------|----------|--------|----------|
| `Small` | ~500ps/iter | High | Small inputs |
| `Large` | ~750ps/iter | Medium | Larger structs |
| `PerIteration` | ~350ns/iter | Low | Huge data, file handles |

Adaptive sizing targets ~100µs per batch to balance timer overhead vs. memory.

## Statistical Analysis

After collecting samples:

1. **IQR Outlier Removal** - Remove samples outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]
2. **Bootstrap Confidence Interval** - 95% CI for the median
3. **Report**: median, mean, stddev, min, max, CI bounds, sample count

```rust
pub struct MeasurementResult {
    pub median_ns: f64,
    pub mean_ns: f64,
    pub stddev_ns: f64,
    pub min_ns: f64,
    pub max_ns: f64,
    pub ci_lower_ns: f64,
    pub ci_upper_ns: f64,
    pub samples: usize,
    pub outliers_removed: usize,
}
```

## Warmup

Before measurement:
- Run for `warmup_time` (default 100ms)
- Stabilizes CPU frequency (turbo boost)
- Warms instruction and data caches
- Warms branch predictor

## Configuration

```rust
let measurer = Measurer::new()
    .warmup_time(Duration::from_millis(100))    // default
    .measurement_time(Duration::from_millis(500)) // default
    .batch_size(BatchSize::Small);              // default
```

Override at runtime or CLI:
```bash
cargo run --release -- --warmup 50 --measurement 200
```

## Consuming Routines

For routines that take ownership:

```rust
fn collect_samples_consuming<I, O, S, R>(...) -> Vec<f64> {
    // ...
    for input in inputs {  // consumes inputs
        outputs.push(black_box(routine(black_box(input))));
    }
    // Outputs still deferred-drop
}
```

Note: Input drop is unavoidable since routine consumes it, but output drop is still deferred.

## Checklist

When implementing measurement:

- [x] `black_box` on inputs AND outputs
- [x] Input generation runs outside timed section
- [x] Batch iterations to amortize timer overhead (~100µs per batch)
- [x] Drops happen outside timed section (deferred drop)
- [x] Warmup phase before measurement (~100ms)
- [x] Remove outliers using IQR method
- [x] Report confidence intervals (95% bootstrap CI)
- [x] Record sample count and outliers removed

## References

- [Divan drop handling](https://docs.rs/divan/latest/divan/attr.bench.html)
- [Criterion timing loops](https://bheisler.github.io/criterion.rs/book/user_guide/timing_loops.html)
- [std::hint::black_box RFC](https://rust-lang.github.io/rfcs/2360-bench-black-box.html)
- [Why my Rust benchmarks were wrong](https://gendignoux.com/blog/2022/01/31/rust-benchmarks.html)
