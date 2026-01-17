# Sampling Combinators for Columnar Data

A trait-based system for generating test data with composable distributions.

## Core Design

The design separates three orthogonal concerns:

| Layer | Trait | Responsibility |
|-------|-------|----------------|
| Generation | `Distribution<T>` | Produce values of type `T` |
| Composition | Combinators | Transform/wrap distributions |
| Materialization | `FromDistribution<T, D>` | Pack values into storage format |

This separation means:
- Distributions know nothing about storage
- Storage formats know nothing about statistics
- Combinators work with any distribution AND any storage format

## Core Traits

### `Distribution<T>`

```rust
pub trait Distribution<T> {
    fn sample(&self, rng: &mut impl Rng) -> T;
}
```

### `DistributionExt<T>`

```rust
pub trait DistributionExt<T>: Distribution<T> + Sized {
    fn with_nulls(self, null_prob: f64) -> WithNulls<Self>;
    fn with_runs(self, switch_prob: f64) -> Runs<Self>;
    fn map<U, F: Fn(T) -> U>(self, f: F) -> Map<Self, F, T>;
    fn sample_vec(&self, seed: u64, len: usize) -> Vec<T>;
}

impl<T, D: Distribution<T>> DistributionExt<T> for D {}
```

### `FromDistribution<T, D>`

```rust
pub trait FromDistribution<T, D: Distribution<T>>: Sized {
    fn from_distribution(dist: &D, seed: u64, len: usize) -> Self;
}

pub trait FromNullableDistribution<T, D: Distribution<Option<T>>>: Sized {
    fn from_nullable_distribution(dist: &D, seed: u64, len: usize) -> Self;
}
```

## Base Distributions

| Distribution | Parameters | Description |
|--------------|------------|-------------|
| `Uniform` | none | Full-range uniform random |
| `Bernoulli` | `true_prob: f64` | Biased boolean |
| `Range<T>` | `min, max` | Bounded uniform `[min, max)` |
| `OneOf<T>` | `values: Vec<T>` | Uniform from fixed set |
| `Constant<T>` | `value: T` | Always same value |
| `Normal` | `mean, std_dev` | Gaussian distribution |
| `Exponential` | `lambda` | Exponential distribution |
| `Weighted<T>` | `values, weights` | Non-uniform selection |

## Combinators

| Combinator | Transform | Description |
|------------|-----------|-------------|
| `WithNulls<D>` | `T` → `Option<T>` | Adds null probability |
| `Runs<D>` | `T` → `T` | Run-length friendly data |
| `Map<D, F>` | `T` → `U` | Transform output |
| `Zip<D1, D2>` | → `(T1, T2)` | Pair values |
| `Either<D1, D2, B>` | → `T` | Conditional selection |
| `Tuple2..5` | → `(T0, T1, ...)` | Struct-like composition |
| `ListDist<D, L>` | → `Vec<T>` | Variable-length lists |
| `StringDist<L>` | → `String` | Random strings |
| `Monotonic<D>` | → `i64` | Increasing values |
| `RandomWalk<D>` | → `f64` | Cumulative deltas |

## Encoding-Aware Generation

| Combinator | Encoding |
|------------|----------|
| `Runs<D>` | Run-end encoding |
| `OneOf<T>` | Dictionary encoding |
| `Monotonic<D>` | Delta encoding |
| `WithNulls<D>` | Validity bitmaps |
| `Constant<T>` | Constant encoding |

Generate encoded representation directly:

```rust
let dist = Runs::new(OneOf::new(vec![1, 2, 3]), 0.1);
let (values, run_ends) = dist.sample_run_ends(seed, 1000);
```

## Composition Examples

```rust
// Simple
Range::new(0i32, 100).sample_vec(seed, 1000)

// Runs
Runs::new(OneOf::new(vec![1, 2, 3]), 0.1).sample_vec_runs(seed, 1000)

// Full composition
Range::new(1, 100)
    .with_runs(0.1)
    .with_nulls(0.2)

// Struct-like
Tuple2(Range::new(0, 100), Range::new(0.0, 1.0)).sample_vec(seed, 100)

// Lists
ListDist::new(Range::new(0i32, 100), Range::new(1usize, 10))

// Strings
StringDist::alphanumeric(Range::new(5usize, 15))
```

## Custom Type Integration

```rust
struct MyColumn<T> { data: Vec<T> }

impl<D: Distribution<u8>> FromDistribution<u8, D> for MyColumn<u8> {
    fn from_distribution(dist: &D, seed: u64, len: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        MyColumn { data: (0..len).map(|_| dist.sample(&mut rng)).collect() }
    }
}

// Works with any distribution
MyColumn::from_distribution(&Range::new(0u8, 100), 42, 1000)
```

## Key Properties

- **Deterministic**: Same seed produces same output
- **Zero coupling**: Only depends on `rand`, not vortex types
- **Composable**: Combinators stack arbitrarily deep
- **Extensible**: New distributions and storage formats added independently

## Prior Art

Similar to property testing combinators (proptest, QuickCheck, Hypothesis) but focused on:
- Columnar data layouts (run-length, dictionary encoding patterns)
- Encoding-aware data generation
- Deterministic, reproducible output from seeds
