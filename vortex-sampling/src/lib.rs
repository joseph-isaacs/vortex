//! # Sampling Combinators for Columnar Data
//!
//! A trait-based system for generating test data with composable distributions.
//!
//! ## Design Philosophy
//!
//! This library is **completely decoupled from any specific data representation**.
//! The core traits work with any Rust types. Integration with specific systems
//! (Arrow, Vortex, Polars, etc.) is done via trait implementations, not library
//! coupling.
//!
//! The key separation:
//!
//! 1. **[`Distribution<T>`]** - Statistical properties of data. Knows nothing about
//!    storage. Just produces values of type `T`.
//!
//! 2. **Combinators** - Compose distributions. [`WithNulls`], [`Runs`], [`Map`], etc.
//!    Also storage-agnostic.
//!
//! 3. **Materialization** (user-provided) - Converts distributions into specific
//!    formats. You implement [`FromDistribution`] for your types.
//!
//! ## Example: Custom Data Structure
//!
//! ```rust
//! use vortex_sampling::{Distribution, FromDistribution, Range, uniform};
//! use rand::{Rng, SeedableRng};
//! use rand::rngs::StdRng;
//!
//! // Your custom columnar type
//! struct MyColumnStore {
//!     data: Vec<u8>,
//!     len: usize,
//! }
//!
//! // Implement materialization for your type
//! impl<D: Distribution<u8>> FromDistribution<u8, D> for MyColumnStore {
//!     fn from_distribution(dist: &D, seed: u64, len: usize) -> Self {
//!         let mut rng = StdRng::seed_from_u64(seed);
//!         let data = (0..len).map(|_| dist.sample(&mut rng)).collect();
//!         MyColumnStore { data, len }
//!     }
//! }
//!
//! // Now you can use any distribution with your type
//! let col = MyColumnStore::from_distribution(&Range::new(0u8, 100), 42, 1000);
//! assert_eq!(col.len, 1000);
//! ```
//!
//! ## Prior Art
//!
//! Similar to property testing combinators (proptest, QuickCheck, Hypothesis)
//! but focused on:
//! - Columnar data layouts (run-length, dictionary encoding patterns)
//! - Encoding-aware data generation
//! - Deterministic, reproducible output from seeds

use std::marker::PhantomData;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

// =============================================================================
// Core Traits
// =============================================================================

/// A distribution that can generate values of type `T`.
///
/// This is the fundamental building block. Distributions are composable and
/// describe *what* values look like, not *how* they're stored.
///
/// # Implementing for New Types
///
/// For simple types, implement directly:
///
/// ```rust
/// use vortex_sampling::Distribution;
/// use rand::Rng;
///
/// struct MyEnum { variant: u8 }
///
/// struct MyEnumDist;
///
/// impl Distribution<MyEnum> for MyEnumDist {
///     fn sample(&self, rng: &mut impl Rng) -> MyEnum {
///         MyEnum { variant: rng.random_range(0..4) }
///     }
/// }
/// ```
///
/// For complex types, compose from existing distributions:
///
/// ```rust
/// use vortex_sampling::{Distribution, Range, OneOf};
/// use rand::Rng;
///
/// struct Point { x: i32, y: i32 }
///
/// struct PointDist {
///     x: Range<i32>,
///     y: Range<i32>,
/// }
///
/// impl Distribution<Point> for PointDist {
///     fn sample(&self, rng: &mut impl Rng) -> Point {
///         Point {
///             x: self.x.sample(rng),
///             y: self.y.sample(rng),
///         }
///     }
/// }
/// ```
pub trait Distribution<T> {
    /// Generate the next value from this distribution.
    fn sample(&self, rng: &mut impl Rng) -> T;
}

/// Extension trait for distributions - enables combinator methods.
///
/// All `Distribution<T>` types automatically get these methods.
pub trait DistributionExt<T>: Distribution<T> + Sized {
    /// Wrap this distribution to produce `Option<T>` with nulls.
    ///
    /// ```rust
    /// use vortex_sampling::{Distribution, DistributionExt, Range};
    /// use rand::{Rng, SeedableRng};
    /// use rand::rngs::StdRng;
    ///
    /// let dist = Range::new(0i32, 100).with_nulls(0.2);
    /// let mut rng = StdRng::seed_from_u64(42);
    ///
    /// // ~20% of values will be None
    /// let samples: Vec<Option<i32>> = (0..100).map(|_| dist.sample(&mut rng)).collect();
    /// let null_count = samples.iter().filter(|v| v.is_none()).count();
    /// assert!(null_count > 10 && null_count < 40);
    /// ```
    fn with_nulls(self, null_prob: f64) -> WithNulls<Self> {
        WithNulls {
            inner: self,
            null_prob,
        }
    }

    /// Convert this distribution into run-length friendly data.
    ///
    /// Values repeat with probability `1 - switch_prob` on each sample.
    /// Good for testing run-end encoding.
    ///
    /// ```rust
    /// use vortex_sampling::{DistributionExt, OneOf, Runs};
    ///
    /// // 10% chance to switch values -> average run length ~10
    /// let dist = Runs::new(OneOf::new(vec![1, 2, 3]), 0.1);
    /// let samples = dist.sample_vec_runs(42, 100);
    /// ```
    fn with_runs(self, switch_prob: f64) -> Runs<Self> {
        Runs {
            values: self,
            switch_prob,
        }
    }

    /// Map the output of this distribution through a function.
    ///
    /// ```rust
    /// use vortex_sampling::{DistributionExt, Range};
    ///
    /// let dist = Range::new(0i32, 50).map(|x| x * 2);
    /// let samples = dist.sample_vec(42, 100);
    /// assert!(samples.iter().all(|&v| v % 2 == 0 && v < 100));
    /// ```
    fn map<U, F: Fn(T) -> U>(self, f: F) -> Map<Self, F, T> {
        Map {
            inner: self,
            f,
            _phantom: PhantomData,
        }
    }

    /// Filter and map in one step (like `filter_map` on iterators).
    fn filter_map<U, F: Fn(T) -> Option<U>>(self, f: F) -> FilterMap<Self, F, T> {
        FilterMap {
            inner: self,
            f,
            _phantom: PhantomData,
        }
    }

    /// Generate a fixed number of samples as a Vec.
    ///
    /// This is the simplest materialization - just collect into a Vec.
    fn sample_vec(&self, seed: u64, len: usize) -> Vec<T> {
        let mut rng = StdRng::seed_from_u64(seed);
        (0..len).map(|_| self.sample(&mut rng)).collect()
    }

    /// Generate samples using a provided RNG.
    fn sample_n(&self, rng: &mut impl Rng, len: usize) -> Vec<T> {
        (0..len).map(|_| self.sample(rng)).collect()
    }
}

impl<T, D: Distribution<T>> DistributionExt<T> for D {}

/// Trait for types that can be constructed from a distribution.
///
/// Implement this for your columnar types to enable:
/// ```ignore
/// MyArray::from_distribution(&Range::new(0, 100), seed, len)
/// ```
///
/// The separation of `FromDistribution` from `Distribution` is key:
/// - `Distribution<T>` knows how to generate values
/// - `FromDistribution` knows how to pack values into a specific format
pub trait FromDistribution<T, D: Distribution<T>>: Sized {
    /// Materialize values from the distribution into this type.
    fn from_distribution(dist: &D, seed: u64, len: usize) -> Self;
}

/// Trait for types that can be constructed from a nullable distribution.
///
/// Separated from [`FromDistribution`] because nullable data often requires
/// different storage (validity masks, option encoding, etc.)
pub trait FromNullableDistribution<T, D: Distribution<Option<T>>>: Sized {
    /// Materialize nullable values from the distribution into this type.
    fn from_nullable_distribution(dist: &D, seed: u64, len: usize) -> Self;
}

// =============================================================================
// Base Distributions
// =============================================================================

/// Uniform random distribution over the full range of `T`.
///
/// Works with any type that implements `rand::random()`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Uniform;

// Implement for common types
macro_rules! impl_uniform {
    ($($ty:ty),*) => {
        $(
            impl Distribution<$ty> for Uniform {
                fn sample(&self, rng: &mut impl Rng) -> $ty {
                    rng.random()
                }
            }
        )*
    };
}

impl_uniform!(
    bool, u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64
);

/// Biased boolean distribution with configurable true probability.
///
/// ```rust
/// use vortex_sampling::{Bernoulli, DistributionExt};
///
/// let biased = Bernoulli::new(0.9); // 90% true
/// let samples = biased.sample_vec(42, 1000);
/// let true_count = samples.iter().filter(|&&b| b).count();
/// assert!(true_count > 850);
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Bernoulli {
    pub true_prob: f64,
}

impl Bernoulli {
    pub fn new(true_prob: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&true_prob),
            "probability must be in [0, 1]"
        );
        Self { true_prob }
    }
}

impl Distribution<bool> for Bernoulli {
    fn sample(&self, rng: &mut impl Rng) -> bool {
        rng.random::<f64>() < self.true_prob
    }
}

/// Bounded uniform distribution over `[min, max)`.
///
/// ```rust
/// use vortex_sampling::{Range, DistributionExt};
///
/// let dist = Range::new(10i32, 20);
/// let samples = dist.sample_vec(42, 100);
/// assert!(samples.iter().all(|&v| v >= 10 && v < 20));
/// ```
#[derive(Debug, Clone, Copy)]
pub struct Range<T> {
    pub min: T,
    pub max: T,
}

impl<T> Range<T> {
    pub fn new(min: T, max: T) -> Self {
        Self { min, max }
    }
}

macro_rules! impl_range {
    ($($ty:ty),*) => {
        $(
            impl Distribution<$ty> for Range<$ty> {
                fn sample(&self, rng: &mut impl Rng) -> $ty {
                    rng.random_range(self.min..self.max)
                }
            }
        )*
    };
}

impl_range!(u8, u16, u32, u64, usize, i8, i16, i32, i64, f32, f64);

/// Sample uniformly from a fixed set of values (low cardinality).
///
/// Great for generating dictionary-encoding-friendly data.
///
/// ```rust
/// use vortex_sampling::{OneOf, DistributionExt};
///
/// let colors = OneOf::new(vec!["red", "green", "blue"]);
/// let samples = colors.sample_vec(42, 100);
/// assert!(samples.iter().all(|s| ["red", "green", "blue"].contains(s)));
/// ```
#[derive(Debug, Clone)]
pub struct OneOf<T> {
    pub values: Vec<T>,
}

impl<T> OneOf<T> {
    pub fn new(values: Vec<T>) -> Self {
        assert!(!values.is_empty(), "OneOf requires at least one value");
        Self { values }
    }

    pub fn from_slice(values: &[T]) -> Self
    where
        T: Clone,
    {
        Self::new(values.to_vec())
    }
}

impl<T: Clone> Distribution<T> for OneOf<T> {
    fn sample(&self, rng: &mut impl Rng) -> T {
        let idx = rng.random_range(0..self.values.len());
        self.values[idx].clone()
    }
}

/// Always produces the same constant value.
///
/// Useful for testing constant-encoding or as a base case.
#[derive(Debug, Clone, Copy)]
pub struct Constant<T> {
    pub value: T,
}

impl<T> Constant<T> {
    pub fn new(value: T) -> Self {
        Self { value }
    }
}

impl<T: Clone> Distribution<T> for Constant<T> {
    fn sample(&self, _rng: &mut impl Rng) -> T {
        self.value.clone()
    }
}

/// Normal (Gaussian) distribution.
///
/// Requires `rand_distr` feature for full functionality, but here's a
/// simple Box-Muller implementation.
#[derive(Debug, Clone, Copy)]
pub struct Normal {
    pub mean: f64,
    pub std_dev: f64,
}

impl Normal {
    pub fn new(mean: f64, std_dev: f64) -> Self {
        assert!(std_dev > 0.0, "std_dev must be positive");
        Self { mean, std_dev }
    }
}

impl Distribution<f64> for Normal {
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        // Box-Muller transform
        let u1: f64 = rng.random();
        let u2: f64 = rng.random();
        let z0 = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        self.mean + self.std_dev * z0
    }
}

/// Exponential distribution.
#[derive(Debug, Clone, Copy)]
pub struct Exponential {
    pub lambda: f64,
}

impl Exponential {
    pub fn new(lambda: f64) -> Self {
        assert!(lambda > 0.0, "lambda must be positive");
        Self { lambda }
    }
}

impl Distribution<f64> for Exponential {
    fn sample(&self, rng: &mut impl Rng) -> f64 {
        let u: f64 = rng.random();
        -u.ln() / self.lambda
    }
}

/// Weighted sampling from a set of values.
///
/// Unlike [`OneOf`], this allows non-uniform selection.
#[derive(Debug, Clone)]
pub struct Weighted<T> {
    pub values: Vec<T>,
    pub cumulative_weights: Vec<f64>,
}

impl<T> Weighted<T> {
    pub fn new(values: Vec<T>, weights: Vec<f64>) -> Self {
        assert_eq!(values.len(), weights.len());
        assert!(!values.is_empty());
        assert!(weights.iter().all(|&w| w >= 0.0));

        let total: f64 = weights.iter().sum();
        assert!(total > 0.0, "weights must sum to positive value");

        let cumulative_weights: Vec<f64> = weights
            .iter()
            .scan(0.0, |acc, &w| {
                *acc += w / total;
                Some(*acc)
            })
            .collect();

        Self {
            values,
            cumulative_weights,
        }
    }
}

impl<T: Clone> Distribution<T> for Weighted<T> {
    fn sample(&self, rng: &mut impl Rng) -> T {
        let u: f64 = rng.random();
        let idx = self
            .cumulative_weights
            .iter()
            .position(|&cw| u < cw)
            .unwrap_or(self.values.len() - 1);
        self.values[idx].clone()
    }
}

// =============================================================================
// Combinators
// =============================================================================

/// Generates run-length friendly data by holding values for multiple samples.
///
/// This produces data that compresses well with run-end encoding.
///
/// Note: The standard `Distribution::sample` method is stateless and doesn't
/// produce true runs. Use [`sample_vec_runs`](Runs::sample_vec_runs) for
/// proper run-length behavior.
#[derive(Debug, Clone)]
pub struct Runs<D> {
    pub values: D,
    pub switch_prob: f64,
}

impl<D> Runs<D> {
    pub fn new(values: D, switch_prob: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&switch_prob),
            "probability must be in [0, 1]"
        );
        Self {
            values,
            switch_prob,
        }
    }

    /// Generate a vector with proper run-length behavior.
    ///
    /// Unlike the stateless `sample()`, this maintains state across calls
    /// to produce actual runs.
    pub fn sample_vec_runs<T: Clone>(&self, seed: u64, len: usize) -> Vec<T>
    where
        D: Distribution<T>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        self.sample_n_runs(&mut rng, len)
    }

    /// Generate runs using a provided RNG.
    pub fn sample_n_runs<T: Clone>(&self, rng: &mut impl Rng, len: usize) -> Vec<T>
    where
        D: Distribution<T>,
    {
        let mut result = Vec::with_capacity(len);
        if len == 0 {
            return result;
        }

        let mut current = self.values.sample(rng);
        for _ in 0..len {
            if rng.random::<f64>() < self.switch_prob {
                current = self.values.sample(rng);
            }
            result.push(current.clone());
        }
        result
    }

    /// Generate run-ends representation directly: (values, run_ends).
    ///
    /// More efficient than generating the full array and then RLE-encoding.
    pub fn sample_run_ends<T: Clone + PartialEq>(&self, seed: u64, len: usize) -> (Vec<T>, Vec<u64>)
    where
        D: Distribution<T>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut values = Vec::new();
        let mut run_ends = Vec::new();

        if len == 0 {
            return (values, run_ends);
        }

        let mut current = self.values.sample(&mut rng);

        for i in 1..len {
            if rng.random::<f64>() < self.switch_prob {
                values.push(current.clone());
                run_ends.push(i as u64);
                current = self.values.sample(&mut rng);
            }
        }

        // Push final run
        values.push(current);
        run_ends.push(len as u64);

        (values, run_ends)
    }
}

impl<T: Clone, D: Distribution<T>> Distribution<T> for Runs<D> {
    fn sample(&self, rng: &mut impl Rng) -> T {
        // Stateless fallback - just delegates to inner
        // For true run behavior, use sample_vec_runs
        self.values.sample(rng)
    }
}

/// Wraps a distribution to add nulls with given probability.
#[derive(Debug, Clone)]
pub struct WithNulls<D> {
    pub inner: D,
    pub null_prob: f64,
}

impl<D> WithNulls<D> {
    pub fn new(inner: D, null_prob: f64) -> Self {
        assert!(
            (0.0..=1.0).contains(&null_prob),
            "probability must be in [0, 1]"
        );
        Self { inner, null_prob }
    }
}

impl<T, D: Distribution<T>> Distribution<Option<T>> for WithNulls<D> {
    fn sample(&self, rng: &mut impl Rng) -> Option<T> {
        if rng.random::<f64>() < self.null_prob {
            None
        } else {
            Some(self.inner.sample(rng))
        }
    }
}

/// Maps the output of a distribution through a function.
#[derive(Debug, Clone)]
pub struct Map<D, F, T> {
    inner: D,
    f: F,
    _phantom: PhantomData<T>,
}

impl<T, U, D: Distribution<T>, F: Fn(T) -> U> Distribution<U> for Map<D, F, T> {
    fn sample(&self, rng: &mut impl Rng) -> U {
        (self.f)(self.inner.sample(rng))
    }
}

/// Filter-maps the output of a distribution.
///
/// Keeps sampling until the predicate returns Some.
#[derive(Debug, Clone)]
pub struct FilterMap<D, F, T> {
    inner: D,
    f: F,
    _phantom: PhantomData<T>,
}

impl<T, U, D: Distribution<T>, F: Fn(T) -> Option<U>> Distribution<U> for FilterMap<D, F, T> {
    fn sample(&self, rng: &mut impl Rng) -> U {
        loop {
            if let Some(v) = (self.f)(self.inner.sample(rng)) {
                return v;
            }
        }
    }
}

/// Zips two distributions together into tuples.
#[derive(Debug, Clone)]
pub struct Zip<D1, D2> {
    pub first: D1,
    pub second: D2,
}

impl<D1, D2> Zip<D1, D2> {
    pub fn new(first: D1, second: D2) -> Self {
        Self { first, second }
    }
}

impl<T1, T2, D1: Distribution<T1>, D2: Distribution<T2>> Distribution<(T1, T2)> for Zip<D1, D2> {
    fn sample(&self, rng: &mut impl Rng) -> (T1, T2) {
        (self.first.sample(rng), self.second.sample(rng))
    }
}

/// Alternates between two distributions based on a boolean distribution.
#[derive(Debug, Clone)]
pub struct Either<D1, D2, B> {
    pub first: D1,
    pub second: D2,
    pub selector: B,
}

impl<D1, D2, B> Either<D1, D2, B> {
    pub fn new(first: D1, second: D2, selector: B) -> Self {
        Self {
            first,
            second,
            selector,
        }
    }
}

impl<T, D1: Distribution<T>, D2: Distribution<T>, B: Distribution<bool>> Distribution<T>
    for Either<D1, D2, B>
{
    fn sample(&self, rng: &mut impl Rng) -> T {
        if self.selector.sample(rng) {
            self.first.sample(rng)
        } else {
            self.second.sample(rng)
        }
    }
}

/// Chains distributions: first produces `n` from first, then from second.
#[derive(Debug, Clone)]
pub struct Chain<D1, D2> {
    pub first: D1,
    pub first_count: usize,
    pub second: D2,
}

impl<D1, D2> Chain<D1, D2> {
    pub fn new(first: D1, first_count: usize, second: D2) -> Self {
        Self {
            first,
            first_count,
            second,
        }
    }
}

// =============================================================================
// Struct/Tuple Distributions
// =============================================================================

/// Distribution over 2-tuples, composing one distribution per field.
#[derive(Debug, Clone)]
pub struct Tuple2<D0, D1>(pub D0, pub D1);

impl<T0, T1, D0: Distribution<T0>, D1: Distribution<T1>> Distribution<(T0, T1)> for Tuple2<D0, D1> {
    fn sample(&self, rng: &mut impl Rng) -> (T0, T1) {
        (self.0.sample(rng), self.1.sample(rng))
    }
}

/// Distribution over 3-tuples, composing one distribution per field.
#[derive(Debug, Clone)]
pub struct Tuple3<D0, D1, D2>(pub D0, pub D1, pub D2);

impl<T0, T1, T2, D0: Distribution<T0>, D1: Distribution<T1>, D2: Distribution<T2>>
    Distribution<(T0, T1, T2)> for Tuple3<D0, D1, D2>
{
    fn sample(&self, rng: &mut impl Rng) -> (T0, T1, T2) {
        (self.0.sample(rng), self.1.sample(rng), self.2.sample(rng))
    }
}

/// Distribution over 4-tuples, composing one distribution per field.
#[derive(Debug, Clone)]
pub struct Tuple4<D0, D1, D2, D3>(pub D0, pub D1, pub D2, pub D3);

impl<
    T0,
    T1,
    T2,
    T3,
    D0: Distribution<T0>,
    D1: Distribution<T1>,
    D2: Distribution<T2>,
    D3: Distribution<T3>,
> Distribution<(T0, T1, T2, T3)> for Tuple4<D0, D1, D2, D3>
{
    fn sample(&self, rng: &mut impl Rng) -> (T0, T1, T2, T3) {
        (
            self.0.sample(rng),
            self.1.sample(rng),
            self.2.sample(rng),
            self.3.sample(rng),
        )
    }
}

/// Distribution over 5-tuples, composing one distribution per field.
#[derive(Debug, Clone)]
pub struct Tuple5<D0, D1, D2, D3, D4>(pub D0, pub D1, pub D2, pub D3, pub D4);

impl<
    T0,
    T1,
    T2,
    T3,
    T4,
    D0: Distribution<T0>,
    D1: Distribution<T1>,
    D2: Distribution<T2>,
    D3: Distribution<T3>,
    D4: Distribution<T4>,
> Distribution<(T0, T1, T2, T3, T4)> for Tuple5<D0, D1, D2, D3, D4>
{
    fn sample(&self, rng: &mut impl Rng) -> (T0, T1, T2, T3, T4) {
        (
            self.0.sample(rng),
            self.1.sample(rng),
            self.2.sample(rng),
            self.3.sample(rng),
            self.4.sample(rng),
        )
    }
}

// =============================================================================
// Collection Distributions
// =============================================================================

/// Distribution over variable-length collections.
///
/// ```rust
/// use vortex_sampling::{ListDist, Range, DistributionExt};
///
/// let dist = ListDist::new(
///     Range::new(0i32, 100),    // element distribution
///     Range::new(1usize, 10),   // length distribution
/// );
/// let lists: Vec<Vec<i32>> = dist.sample_vec(42, 5);
/// ```
#[derive(Debug, Clone)]
pub struct ListDist<D, L> {
    /// Distribution for list elements.
    pub elements: D,
    /// Distribution for list lengths.
    pub lengths: L,
}

impl<D, L> ListDist<D, L> {
    pub fn new(elements: D, lengths: L) -> Self {
        Self { elements, lengths }
    }

    /// Fixed-length list distribution.
    pub fn fixed(elements: D, len: usize) -> ListDist<D, Constant<usize>> {
        ListDist {
            elements,
            lengths: Constant::new(len),
        }
    }
}

impl<T, D: Distribution<T>, L: Distribution<usize>> Distribution<Vec<T>> for ListDist<D, L> {
    fn sample(&self, rng: &mut impl Rng) -> Vec<T> {
        let len = self.lengths.sample(rng);
        (0..len).map(|_| self.elements.sample(rng)).collect()
    }
}

/// Distribution over strings with configurable alphabet and length.
#[derive(Debug, Clone)]
pub struct StringDist<L> {
    /// Characters to sample from.
    pub alphabet: Vec<char>,
    /// Distribution for string lengths.
    pub lengths: L,
}

impl<L> StringDist<L> {
    pub fn new(alphabet: Vec<char>, lengths: L) -> Self {
        assert!(!alphabet.is_empty(), "alphabet must not be empty");
        Self { alphabet, lengths }
    }

    pub fn alphanumeric(lengths: L) -> Self {
        let alphabet: Vec<char> = ('a'..='z').chain('A'..='Z').chain('0'..='9').collect();
        Self::new(alphabet, lengths)
    }

    pub fn lowercase(lengths: L) -> Self {
        Self::new(('a'..='z').collect(), lengths)
    }

    pub fn digits(lengths: L) -> Self {
        Self::new(('0'..='9').collect(), lengths)
    }
}

impl<L: Distribution<usize>> Distribution<String> for StringDist<L> {
    fn sample(&self, rng: &mut impl Rng) -> String {
        let len = self.lengths.sample(rng);
        (0..len)
            .map(|_| {
                let idx = rng.random_range(0..self.alphabet.len());
                self.alphabet[idx]
            })
            .collect()
    }
}

// =============================================================================
// Temporal/Sequential Distributions
// =============================================================================

/// Monotonically increasing values (good for sorted columns, timestamps).
#[derive(Debug, Clone)]
pub struct Monotonic<D> {
    /// Distribution for increments.
    pub increments: D,
    /// Starting value.
    pub start: i64,
}

impl<D> Monotonic<D> {
    pub fn new(start: i64, increments: D) -> Self {
        Self { increments, start }
    }

    /// Generate monotonically increasing values.
    pub fn sample_vec_monotonic(&self, seed: u64, len: usize) -> Vec<i64>
    where
        D: Distribution<u64>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut result = Vec::with_capacity(len);
        let mut current = self.start;

        for _ in 0..len {
            result.push(current);
            current = current.saturating_add(self.increments.sample(&mut rng) as i64);
        }
        result
    }
}

/// Random walk distribution (each value depends on previous).
#[derive(Debug, Clone)]
pub struct RandomWalk<D> {
    /// Distribution for step deltas.
    pub deltas: D,
    /// Starting value.
    pub start: f64,
}

impl<D> RandomWalk<D> {
    pub fn new(start: f64, deltas: D) -> Self {
        Self { deltas, start }
    }

    pub fn sample_vec_walk(&self, seed: u64, len: usize) -> Vec<f64>
    where
        D: Distribution<f64>,
    {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut result = Vec::with_capacity(len);
        let mut current = self.start;

        for _ in 0..len {
            result.push(current);
            current += self.deltas.sample(&mut rng);
        }
        result
    }
}

// =============================================================================
// Convenience Functions
// =============================================================================

/// Create a uniform distribution.
pub fn uniform() -> Uniform {
    Uniform
}

/// Create a Bernoulli distribution with given true probability.
pub fn bernoulli(true_prob: f64) -> Bernoulli {
    Bernoulli::new(true_prob)
}

/// Create a range distribution.
pub fn range<T>(min: T, max: T) -> Range<T> {
    Range::new(min, max)
}

/// Create a one-of distribution from a slice.
pub fn one_of<T: Clone>(values: &[T]) -> OneOf<T> {
    OneOf::from_slice(values)
}

/// Create a constant distribution.
pub fn constant<T>(value: T) -> Constant<T> {
    Constant::new(value)
}

/// Create a normal distribution.
pub fn normal(mean: f64, std_dev: f64) -> Normal {
    Normal::new(mean, std_dev)
}

/// Zip two distributions into tuples.
pub fn zip<D1, D2>(first: D1, second: D2) -> Zip<D1, D2> {
    Zip::new(first, second)
}

// =============================================================================
// Default Implementations for Vec<T>
// =============================================================================

impl<T, D: Distribution<T>> FromDistribution<T, D> for Vec<T> {
    fn from_distribution(dist: &D, seed: u64, len: usize) -> Self {
        dist.sample_vec(seed, len)
    }
}

// =============================================================================
// Example: Simple Nullable Array Type
// =============================================================================

/// A simple nullable array representation for demonstration.
///
/// Shows how any columnar format can implement [`FromNullableDistribution`].
#[derive(Debug, Clone)]
pub struct NullableVec<T> {
    pub values: Vec<T>,
    pub validity: Vec<bool>,
}

impl<T> NullableVec<T> {
    pub fn len(&self) -> usize {
        self.values.len()
    }

    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    pub fn null_count(&self) -> usize {
        self.validity.iter().filter(|&&v| !v).count()
    }

    pub fn get(&self, idx: usize) -> Option<&T> {
        self.validity[idx].then(|| &self.values[idx])
    }
}

impl<T: Default, D: Distribution<Option<T>>> FromNullableDistribution<T, D> for NullableVec<T> {
    fn from_nullable_distribution(dist: &D, seed: u64, len: usize) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut values = Vec::with_capacity(len);
        let mut validity = Vec::with_capacity(len);

        for _ in 0..len {
            match dist.sample(&mut rng) {
                Some(v) => {
                    values.push(v);
                    validity.push(true);
                }
                None => {
                    values.push(T::default());
                    validity.push(false);
                }
            }
        }

        NullableVec { values, validity }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used)]
mod tests {
    use super::*;

    #[test]
    fn test_uniform_bool() {
        let dist = Uniform;
        let samples: Vec<bool> = dist.sample_vec(42, 1000);

        let true_count = samples.iter().filter(|&&b| b).count();
        assert!(true_count > 400 && true_count < 600);
    }

    #[test]
    fn test_bernoulli() {
        let dist = Bernoulli::new(0.8);
        let samples: Vec<bool> = dist.sample_vec(42, 1000);

        let true_count = samples.iter().filter(|&&b| b).count();
        assert!(true_count > 700 && true_count < 900);
    }

    #[test]
    fn test_range() {
        let dist = Range::new(10i32, 20i32);
        let samples: Vec<i32> = dist.sample_vec(42, 1000);

        assert!(samples.iter().all(|&v| (10..20).contains(&v)));
    }

    #[test]
    fn test_one_of() {
        let dist = OneOf::new(vec![1, 2, 3]);
        let samples: Vec<i32> = dist.sample_vec(42, 1000);

        assert!(samples.iter().all(|v| [1, 2, 3].contains(v)));
    }

    #[test]
    fn test_runs() {
        let dist = Runs::new(OneOf::new(vec![1, 2, 3]), 0.1);
        let samples = dist.sample_vec_runs(42, 1000);

        // Count run lengths
        let mut run_lengths = vec![];
        let mut current_len = 1;
        for i in 1..samples.len() {
            if samples[i] == samples[i - 1] {
                current_len += 1;
            } else {
                run_lengths.push(current_len);
                current_len = 1;
            }
        }
        run_lengths.push(current_len);

        // With 10% switch probability, average run length should be ~10
        let avg_run_len: f64 = run_lengths.iter().sum::<usize>() as f64 / run_lengths.len() as f64;
        assert!(
            avg_run_len > 5.0 && avg_run_len < 20.0,
            "avg_run_len was {}",
            avg_run_len
        );
    }

    #[test]
    fn test_run_ends_generation() {
        let dist = Runs::new(OneOf::new(vec![1i32, 2, 3]), 0.2);
        let (values, run_ends) = dist.sample_run_ends(42, 100);

        // Verify run_ends format
        assert_eq!(*run_ends.last().unwrap(), 100);
        assert_eq!(values.len(), run_ends.len());

        // Verify monotonically increasing
        for i in 1..run_ends.len() {
            assert!(run_ends[i] > run_ends[i - 1]);
        }
    }

    #[test]
    fn test_with_nulls() {
        let dist = WithNulls::new(Range::new(0i32, 100), 0.3);
        let mut rng = StdRng::seed_from_u64(42);

        let samples: Vec<Option<i32>> = (0..1000).map(|_| dist.sample(&mut rng)).collect();

        let null_count = samples.iter().filter(|v| v.is_none()).count();
        assert!(
            null_count > 200 && null_count < 400,
            "null_count was {}",
            null_count
        );
    }

    #[test]
    fn test_full_composition() {
        // Runs of low-cardinality values with nulls
        let dist = WithNulls::new(Runs::new(OneOf::new(vec![1i32, 2, 3, 4, 5]), 0.15), 0.2);

        let result: NullableVec<i32> = NullableVec::from_nullable_distribution(&dist, 42, 1000);

        // Check validity
        let valid_count = result.validity.iter().filter(|&&v| v).count();
        assert!(
            valid_count > 700 && valid_count < 900,
            "valid_count was {}",
            valid_count
        );

        // Check values are from the set
        for i in 0..result.len() {
            if result.validity[i] {
                assert!([1, 2, 3, 4, 5].contains(&result.values[i]));
            }
        }
    }

    #[test]
    fn test_tuple_distribution() {
        let dist = Tuple2(Range::new(0i64, 1000), Range::new(0.0f64, 1.0));

        let samples: Vec<(i64, f64)> = dist.sample_vec(42, 100);

        for (id, value) in &samples {
            assert!((0..1000).contains(id));
            assert!(*value >= 0.0 && *value < 1.0);
        }
    }

    #[test]
    fn test_list_distribution() {
        let dist = ListDist::new(Range::new(0i32, 100), Range::new(1usize, 10));

        let samples: Vec<Vec<i32>> = dist.sample_vec(42, 100);

        for list in &samples {
            assert!(!list.is_empty() && list.len() < 10);
            assert!(list.iter().all(|&v| (0..100).contains(&v)));
        }
    }

    #[test]
    fn test_string_distribution() {
        let dist = StringDist::alphanumeric(Range::new(5usize, 15));

        let samples: Vec<String> = dist.sample_vec(42, 100);

        for s in &samples {
            assert!((5..15).contains(&s.len()));
            assert!(s.chars().all(|c| c.is_ascii_alphanumeric()));
        }
    }

    #[test]
    fn test_monotonic() {
        let dist = Monotonic::new(0, Range::new(1u64, 10));
        let samples = dist.sample_vec_monotonic(42, 100);

        // Should be strictly increasing
        for i in 1..samples.len() {
            assert!(samples[i] > samples[i - 1]);
        }
    }

    #[test]
    fn test_random_walk() {
        let dist = RandomWalk::new(0.0, Normal::new(0.0, 1.0));
        let samples = dist.sample_vec_walk(42, 100);

        assert_eq!(samples.len(), 100);
        // First value should be start value
        assert!((samples[0] - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn test_map_combinator() {
        let dist = Range::new(0i32, 50).map(|x| x * 2);
        let samples: Vec<i32> = dist.sample_vec(42, 100);

        assert!(samples.iter().all(|&v| v % 2 == 0 && (0..100).contains(&v)));
    }

    #[test]
    fn test_fluent_api() {
        let dist = Range::new(1i32, 100).with_runs(0.1).with_nulls(0.2);

        let mut rng = StdRng::seed_from_u64(42);
        let samples: Vec<Option<i32>> = (0..100).map(|_| dist.sample(&mut rng)).collect();

        let null_count = samples.iter().filter(|v| v.is_none()).count();
        assert!(null_count > 10 && null_count < 40);
    }

    #[test]
    fn test_weighted() {
        // 80% chance of 1, 20% chance of 2
        let dist = Weighted::new(vec![1, 2], vec![80.0, 20.0]);
        let samples: Vec<i32> = dist.sample_vec(42, 1000);

        let count_1 = samples.iter().filter(|&&v| v == 1).count();
        assert!(count_1 > 700 && count_1 < 900, "count_1 was {}", count_1);
    }

    #[test]
    fn test_either() {
        // 50% from [0,10), 50% from [90,100)
        let dist = Either::new(Range::new(0i32, 10), Range::new(90i32, 100), bernoulli(0.5));

        let samples: Vec<i32> = dist.sample_vec(42, 1000);

        let low_count = samples.iter().filter(|&&v| v < 50).count();
        let high_count = samples.iter().filter(|&&v| v >= 50).count();

        assert!(low_count > 400 && low_count < 600);
        assert!(high_count > 400 && high_count < 600);
    }

    #[test]
    fn test_normal_distribution() {
        let dist = Normal::new(100.0, 15.0);
        let samples: Vec<f64> = dist.sample_vec(42, 10000);

        let mean: f64 = samples.iter().sum::<f64>() / samples.len() as f64;
        assert!(
            (mean - 100.0).abs() < 1.0,
            "mean was {} (expected ~100)",
            mean
        );
    }

    #[test]
    fn test_deterministic_reproducibility() {
        let dist = Range::new(0i32, 1000);

        let samples1 = dist.sample_vec(42, 100);
        let samples2 = dist.sample_vec(42, 100);

        assert_eq!(samples1, samples2, "same seed should produce same output");
    }

    // Test that the design works with external types
    mod external_type_integration {
        use super::*;

        // Simulated external columnar type
        #[allow(dead_code)]
        struct ExternalColumn<T> {
            data: Vec<T>,
            metadata: String,
        }

        impl<T, D: Distribution<T>> FromDistribution<T, D> for ExternalColumn<T> {
            fn from_distribution(dist: &D, seed: u64, len: usize) -> Self {
                let mut rng = StdRng::seed_from_u64(seed);
                ExternalColumn {
                    data: (0..len).map(|_| dist.sample(&mut rng)).collect(),
                    metadata: format!("Generated with seed {}", seed),
                }
            }
        }

        #[test]
        fn test_external_type() {
            let col: ExternalColumn<i32> =
                ExternalColumn::from_distribution(&Range::new(0, 100), 42, 50);

            assert_eq!(col.data.len(), 50);
            assert!(col.data.iter().all(|&v| (0..100).contains(&v)));
        }
    }
}
