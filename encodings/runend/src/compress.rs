// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::ToCanonical;
use vortex_array::arrays::BoolArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityHelper;
use vortex_buffer::BitBuffer;
use vortex_buffer::BitBufferMut;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_buffer::buffer;
use vortex_dtype::NativePType;
use vortex_dtype::Nullability;
use vortex_dtype::match_each_native_ptype;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexExpect;
use vortex_mask::Mask;
use vortex_scalar::Scalar;

use crate::iter::trimmed_ends_iter;

// ============================================================================
// Helper functions
// ============================================================================

/// Checks if all values in the slice are identical.
/// Returns true for empty slices.
#[inline(always)]
fn is_constant_value<T: PartialEq>(values: &[T]) -> bool {
    match values.first() {
        Some(first) => values.iter().all(|v| v == first),
        None => true,
    }
}

/// Returns true if more than half of the values are zero.
///
/// This is used to determine whether to use a zeroed buffer optimization
/// where we pre-fill with zeros and skip zero-value fills during decoding.
#[inline(always)]
fn is_mostly_zeros<T: NativePType>(values: &[T]) -> bool {
    let zero = T::default();
    let zero_count = values.iter().filter(|&&v| v == zero).count();
    zero_count * 2 > values.len()
}

// ============================================================================
// Original implementation (for benchmarking comparison)
// ============================================================================

/// Original bool decode implementation using append_n for each run.
/// Kept for benchmarking comparison with the optimized version.
pub fn runend_decode_typed_bool_original(
    run_ends: impl Iterator<Item = usize>,
    values: &BitBuffer,
    length: usize,
) -> BoolArray {
    let mut decoded = BitBufferMut::with_capacity(length);
    for (end, value) in run_ends.zip_eq(values.iter()) {
        if end > decoded.len() {
            decoded.append_n(value, end - decoded.len());
        }
    }
    BoolArray::from_bit_buffer(decoded.freeze(), Validity::NonNullable)
}

/// Public wrapper for benchmarking the original bool decode
pub fn runend_decode_bools_original(
    ends: PrimitiveArray,
    values: BoolArray,
    offset: usize,
    length: usize,
) -> BoolArray {
    match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
        runend_decode_typed_bool_original(
            trimmed_ends_iter(ends.as_slice::<E>(), offset, length),
            values.bit_buffer(),
            length,
        )
    })
}

/// Run-end encode a `PrimitiveArray`, returning a tuple of `(ends, values)`.
pub fn runend_encode(array: &PrimitiveArray) -> (PrimitiveArray, ArrayRef) {
    let validity = match array.validity() {
        Validity::NonNullable => None,
        Validity::AllValid => None,
        Validity::AllInvalid => {
            // We can trivially return an all-null REE array
            return (
                PrimitiveArray::new(buffer![array.len() as u64], Validity::NonNullable),
                ConstantArray::new(Scalar::null(array.dtype().clone()), 1).into_array(),
            );
        }
        Validity::Array(a) => Some(a.to_bool().bit_buffer().clone()),
    };

    let (ends, values) = match validity {
        None => {
            match_each_native_ptype!(array.ptype(), |P| {
                let (ends, values) = runend_encode_primitive(array.as_slice::<P>());
                (
                    PrimitiveArray::new(ends, Validity::NonNullable),
                    PrimitiveArray::new(values, array.dtype().nullability().into()).into_array(),
                )
            })
        }
        Some(validity) => {
            match_each_native_ptype!(array.ptype(), |P| {
                let (ends, values) =
                    runend_encode_nullable_primitive(array.as_slice::<P>(), validity);
                (
                    PrimitiveArray::new(ends, Validity::NonNullable),
                    values.into_array(),
                )
            })
        }
    };

    let ends = ends
        .narrow()
        .vortex_expect("Ends must succeed downcasting")
        .to_primitive();

    (ends, values)
}

fn runend_encode_primitive<T: NativePType>(elements: &[T]) -> (Buffer<u64>, Buffer<T>) {
    let mut ends = BufferMut::empty();
    let mut values = BufferMut::empty();

    if elements.is_empty() {
        return (ends.freeze(), values.freeze());
    }

    // Run-end encode the values
    let mut prev = elements[0];
    let mut end = 1;
    for &e in elements.iter().skip(1) {
        if e != prev {
            ends.push(end);
            values.push(prev);
        }
        prev = e;
        end += 1;
    }
    ends.push(end);
    values.push(prev);

    (ends.freeze(), values.freeze())
}

fn runend_encode_nullable_primitive<T: NativePType>(
    elements: &[T],
    element_validity: BitBuffer,
) -> (Buffer<u64>, PrimitiveArray) {
    let mut ends = BufferMut::empty();
    let mut values = BufferMut::empty();
    let mut validity = BitBufferMut::with_capacity(values.capacity());

    if elements.is_empty() {
        return (
            ends.freeze(),
            PrimitiveArray::new(
                values,
                Validity::Array(BoolArray::from(validity.freeze()).into_array()),
            ),
        );
    }

    // Run-end encode the values
    let mut prev = element_validity.value(0).then(|| elements[0]);
    let mut end = 1;
    for e in elements
        .iter()
        .zip(element_validity.iter())
        .map(|(&e, is_valid)| is_valid.then_some(e))
        .skip(1)
    {
        if e != prev {
            ends.push(end);
            match prev {
                None => {
                    validity.append(false);
                    values.push(T::default());
                }
                Some(p) => {
                    validity.append(true);
                    values.push(p);
                }
            }
        }
        prev = e;
        end += 1;
    }
    ends.push(end);

    match prev {
        None => {
            validity.append(false);
            values.push(T::default());
        }
        Some(p) => {
            validity.append(true);
            values.push(p);
        }
    }

    (
        ends.freeze(),
        PrimitiveArray::new(values, Validity::from(validity.freeze())),
    )
}

pub fn runend_decode_primitive(
    ends: PrimitiveArray,
    values: PrimitiveArray,
    offset: usize,
    length: usize,
) -> PrimitiveArray {
    match_each_native_ptype!(values.ptype(), |P| {
        match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
            runend_decode_typed_primitive(
                trimmed_ends_iter(ends.as_slice::<E>(), offset, length),
                values.as_slice::<P>(),
                values.validity_mask(),
                values.dtype().nullability(),
                length,
            )
        })
    })
}

pub fn runend_decode_bools(
    ends: PrimitiveArray,
    values: BoolArray,
    offset: usize,
    length: usize,
) -> BoolArray {
    match_each_unsigned_integer_ptype!(ends.ptype(), |E| {
        runend_decode_typed_bool(
            trimmed_ends_iter(ends.as_slice::<E>(), offset, length),
            values.bit_buffer(),
            values.validity_mask(),
            values.dtype().nullability(),
            length,
        )
    })
}

pub fn runend_decode_typed_primitive<T: NativePType>(
    run_ends: impl Iterator<Item = usize>,
    values: &[T],
    values_validity: Mask,
    values_nullability: Nullability,
    length: usize,
) -> PrimitiveArray {
    match values_validity {
        Mask::AllTrue(_) => {
            // Fast path: all values are identical - use a single fill
            if let Some(&first_value) = values.first()
                && is_constant_value(values)
            {
                let mut decoded: BufferMut<T> = BufferMut::with_capacity(length);
                // SAFETY: We will initialize all elements via fill before returning
                unsafe { decoded.set_len(length) };
                decoded.as_mut_slice().fill(first_value);
                return PrimitiveArray::new(decoded.freeze(), values_nullability.into());
            }

            // Sparse zero optimization: if >50% of run values are zero,
            // use a zeroed buffer and skip zero fills
            if is_mostly_zeros(values) {
                let mut decoded: BufferMut<T> = BufferMut::zeroed(length);
                let decoded_slice = decoded.as_mut_slice();
                let mut current_pos = 0usize;
                let zero = T::default();

                for (end, value) in run_ends.zip_eq(values) {
                    debug_assert!(
                        end <= length,
                        "Runend end must be less than or equal to overall length"
                    );
                    // Skip runs that are entirely before the current position
                    // (can happen with offset when runs end before the view starts)
                    if end > current_pos {
                        // Only fill non-zero values (zeros are already set)
                        if *value != zero {
                            decoded_slice[current_pos..end].fill(*value);
                        }
                        current_pos = end;
                    }
                }
                return PrimitiveArray::new(decoded, values_nullability.into());
            }

            // Pre-allocate buffer capacity, then fill using slice::fill
            // which the compiler can vectorize more efficiently than per-element writes
            let mut decoded: BufferMut<T> = BufferMut::with_capacity(length);
            // SAFETY: We will initialize all elements before returning
            unsafe { decoded.set_len(length) };
            let decoded_slice = decoded.as_mut_slice();
            let mut current_pos = 0usize;

            for (end, value) in run_ends.zip_eq(values) {
                debug_assert!(
                    end <= length,
                    "Runend end must be less than or equal to overall length"
                );
                // Skip runs that are entirely before the current position
                // (can happen with offset when runs end before the view starts)
                if end > current_pos {
                    // Use slice::fill which gets vectorized by LLVM
                    decoded_slice[current_pos..end].fill(*value);
                    current_pos = end;
                }
            }
            PrimitiveArray::new(decoded, values_nullability.into())
        }
        Mask::AllFalse(_) => PrimitiveArray::new(Buffer::<T>::zeroed(length), Validity::AllInvalid),
        Mask::Values(mask) => {
            // For nullable values, we need zeroed buffer for null positions
            let mut decoded: BufferMut<T> = BufferMut::zeroed(length);
            let decoded_slice = decoded.as_mut_slice();
            let mut decoded_validity = BitBufferMut::with_capacity(length);
            let mut current_pos = 0usize;

            for (end, value) in run_ends.zip_eq(
                values
                    .iter()
                    .zip(mask.bit_buffer().iter())
                    .map(|(&v, is_valid)| is_valid.then_some(v)),
            ) {
                debug_assert!(
                    end <= length,
                    "Runend end must be less than or equal to overall length"
                );
                // Skip runs that are entirely before the current position
                if end > current_pos {
                    let run_len = end - current_pos;
                    match value {
                        None => {
                            decoded_validity.append_n(false, run_len);
                            // Leave zeroed for null values (already zeroed from BufferMut::zeroed)
                        }
                        Some(value) => {
                            decoded_validity.append_n(true, run_len);
                            decoded_slice[current_pos..end].fill(value);
                        }
                    }
                    current_pos = end;
                }
            }
            PrimitiveArray::new(decoded, Validity::from(decoded_validity.freeze()))
        }
    }
}

/// Fills bits in range [start, end) to true using byte-level operations.
/// Assumes the buffer is pre-initialized to all zeros.
#[inline(always)]
fn fill_bits_true(slice: &mut [u8], start: usize, end: usize) {
    if start >= end {
        return;
    }

    let start_byte = start / 8;
    let start_bit = start % 8;
    let end_byte = end / 8;
    let end_bit = end % 8;

    if start_byte == end_byte {
        // All bits in same byte
        // Use u16 to avoid overflow, then truncate (guaranteed to fit in u8 since max is 0xFF)
        #[allow(clippy::cast_possible_truncation)]
        let mask = ((1u16 << (end_bit - start_bit)) - 1) as u8;
        slice[start_byte] |= mask << start_bit;
    } else {
        // First partial byte
        if start_bit != 0 {
            slice[start_byte] |= !((1u8 << start_bit) - 1);
        }

        // Middle bytes (bulk memset to 0xFF)
        let fill_start = if start_bit != 0 {
            start_byte + 1
        } else {
            start_byte
        };
        if fill_start < end_byte {
            slice[fill_start..end_byte].fill(0xFF);
        }

        // Last partial byte
        if end_bit != 0 {
            slice[end_byte] |= (1u8 << end_bit) - 1;
        }
    }
}

/// Clears bits in range [start, end) to false using byte-level operations.
/// Assumes the buffer is pre-initialized to all ones.
#[inline(always)]
fn fill_bits_false(slice: &mut [u8], start: usize, end: usize) {
    if start >= end {
        return;
    }

    let start_byte = start / 8;
    let start_bit = start % 8;
    let end_byte = end / 8;
    let end_bit = end % 8;

    if start_byte == end_byte {
        // All bits in same byte - create mask with 0s in the range we want to clear
        #[allow(clippy::cast_possible_truncation)]
        let mask = ((1u16 << (end_bit - start_bit)) - 1) as u8;
        slice[start_byte] &= !(mask << start_bit);
    } else {
        // First partial byte - clear high bits from start_bit
        if start_bit != 0 {
            slice[start_byte] &= (1u8 << start_bit) - 1;
        }

        // Middle bytes (bulk memset to 0x00)
        let fill_start = if start_bit != 0 {
            start_byte + 1
        } else {
            start_byte
        };
        if fill_start < end_byte {
            slice[fill_start..end_byte].fill(0x00);
        }

        // Last partial byte - clear low bits up to end_bit
        if end_bit != 0 {
            slice[end_byte] &= !((1u8 << end_bit) - 1);
        }
    }
}

pub fn runend_decode_typed_bool(
    run_ends: impl Iterator<Item = usize>,
    values: &BitBuffer,
    values_validity: Mask,
    values_nullability: Nullability,
    length: usize,
) -> BoolArray {
    match values_validity {
        Mask::AllTrue(_) => {
            // Adaptive strategy: choose based on which value is more common
            // If more runs have true values, pre-fill with 1s and clear false runs
            // If more runs have false values, pre-fill with 0s and fill true runs
            let true_count = values.true_count();
            let false_count = values.len() - true_count;

            if true_count > false_count {
                // More true runs - pre-fill with 1s and clear false runs
                let mut decoded = BitBufferMut::new_set(length);
                let decoded_bytes = decoded.as_mut_slice();
                let mut current_pos = 0usize;

                for (end, value) in run_ends.zip_eq(values.iter()) {
                    // Only clear when value is false (true is already 1)
                    if end > current_pos && !value {
                        fill_bits_false(decoded_bytes, current_pos, end);
                    }
                    current_pos = end;
                }
                BoolArray::from_bit_buffer(decoded.freeze(), values_nullability.into())
            } else {
                // More or equal false runs - pre-fill with 0s and fill true runs
                let mut decoded = BitBufferMut::new_unset(length);
                let decoded_bytes = decoded.as_mut_slice();
                let mut current_pos = 0usize;

                for (end, value) in run_ends.zip_eq(values.iter()) {
                    // Only fill when value is true (false is already 0)
                    if end > current_pos && value {
                        fill_bits_true(decoded_bytes, current_pos, end);
                    }
                    current_pos = end;
                }
                BoolArray::from_bit_buffer(decoded.freeze(), values_nullability.into())
            }
        }
        Mask::AllFalse(_) => {
            BoolArray::from_bit_buffer(BitBuffer::new_unset(length), Validity::AllInvalid)
        }
        Mask::Values(mask) => {
            // For nullable values, adaptive strategy based on true count
            // (counting only valid values as true)
            let valid_true_count = values
                .iter()
                .zip(mask.bit_buffer().iter())
                .filter(|&(v, is_valid)| is_valid && v)
                .count();
            let valid_false_count = values
                .iter()
                .zip(mask.bit_buffer().iter())
                .filter(|&(v, is_valid)| is_valid && !v)
                .count();

            if valid_true_count > valid_false_count {
                // More true runs - pre-fill with 1s and clear false/null runs
                let mut decoded = BitBufferMut::new_set(length);
                let mut decoded_validity = BitBufferMut::new_unset(length);
                let decoded_bytes = decoded.as_mut_slice();
                let validity_bytes = decoded_validity.as_mut_slice();
                let mut current_pos = 0usize;

                for (end, value) in run_ends.zip_eq(
                    values
                        .iter()
                        .zip(mask.bit_buffer().iter())
                        .map(|(v, is_valid)| is_valid.then_some(v)),
                ) {
                    if end > current_pos {
                        match value {
                            None => {
                                // Null: clear decoded bits, validity stays false
                                fill_bits_false(decoded_bytes, current_pos, end);
                            }
                            Some(v) => {
                                // Valid: set validity bits to true
                                fill_bits_true(validity_bytes, current_pos, end);
                                // Clear decoded bits if value is false
                                if !v {
                                    fill_bits_false(decoded_bytes, current_pos, end);
                                }
                            }
                        }
                        current_pos = end;
                    }
                }
                BoolArray::from_bit_buffer(
                    decoded.freeze(),
                    Validity::from(decoded_validity.freeze()),
                )
            } else {
                // More or equal false runs - pre-fill with 0s and fill true runs
                let mut decoded = BitBufferMut::new_unset(length);
                let mut decoded_validity = BitBufferMut::new_unset(length);
                let decoded_bytes = decoded.as_mut_slice();
                let validity_bytes = decoded_validity.as_mut_slice();
                let mut current_pos = 0usize;

                for (end, value) in run_ends.zip_eq(
                    values
                        .iter()
                        .zip(mask.bit_buffer().iter())
                        .map(|(v, is_valid)| is_valid.then_some(v)),
                ) {
                    if end > current_pos {
                        match value {
                            None => {
                                // Validity stays false (already 0), decoded stays false
                            }
                            Some(v) => {
                                // Set validity bits to true
                                fill_bits_true(validity_bytes, current_pos, end);
                                // Set decoded bits if value is true
                                if v {
                                    fill_bits_true(decoded_bytes, current_pos, end);
                                }
                            }
                        }
                        current_pos = end;
                    }
                }
                BoolArray::from_bit_buffer(
                    decoded.freeze(),
                    Validity::from(decoded_validity.freeze()),
                )
            }
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_array::ToCanonical;
    use vortex_array::arrays::BoolArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;

    use crate::compress::runend_decode_bools;
    use crate::compress::runend_decode_primitive;
    use crate::compress::runend_encode;

    #[test]
    fn encode() {
        let arr = PrimitiveArray::from_iter([1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3]);
        let (ends, values) = runend_encode(&arr);
        let values = values.to_primitive();

        let expected_ends = PrimitiveArray::from_iter(vec![2u8, 5, 10]);
        assert_arrays_eq!(ends, expected_ends);
        let expected_values = PrimitiveArray::from_iter(vec![1i32, 2, 3]);
        assert_arrays_eq!(values, expected_values);
    }

    #[test]
    fn encode_nullable() {
        let arr = PrimitiveArray::new(
            buffer![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3],
            Validity::from(BitBuffer::from(vec![
                true, true, false, false, true, true, true, true, false, false,
            ])),
        );
        let (ends, values) = runend_encode(&arr);
        let values = values.to_primitive();

        let expected_ends = PrimitiveArray::from_iter(vec![2u8, 4, 5, 8, 10]);
        assert_arrays_eq!(ends, expected_ends);
        let expected_values =
            PrimitiveArray::from_option_iter(vec![Some(1i32), None, Some(2), Some(3), None]);
        assert_arrays_eq!(values, expected_values);
    }

    #[test]
    fn encode_all_null() {
        let arr = PrimitiveArray::new(
            buffer![0, 0, 0, 0, 0],
            Validity::from(BitBuffer::new_unset(5)),
        );
        let (ends, values) = runend_encode(&arr);
        let values = values.to_primitive();

        let expected_ends = PrimitiveArray::from_iter(vec![5u64]);
        assert_arrays_eq!(ends, expected_ends);
        let expected_values = PrimitiveArray::from_option_iter(vec![Option::<i32>::None]);
        assert_arrays_eq!(values, expected_values);
    }

    #[test]
    fn decode() {
        let ends = PrimitiveArray::from_iter([2u32, 5, 10]);
        let values = PrimitiveArray::from_iter([1i32, 2, 3]);
        let decoded = runend_decode_primitive(ends, values, 0, 10);

        let expected = PrimitiveArray::from_iter(vec![1i32, 1, 2, 2, 2, 3, 3, 3, 3, 3]);
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_bools_alternating() {
        // Alternating true/false: [T, T, F, F, F, T, T, T, T, T]
        let ends = PrimitiveArray::from_iter([2u32, 5, 10]);
        let values = BoolArray::from(BitBuffer::from(vec![true, false, true]));
        let decoded = runend_decode_bools(ends, values, 0, 10);

        let expected = BoolArray::from(BitBuffer::from(vec![
            true, true, false, false, false, true, true, true, true, true,
        ]));
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_bools_mostly_true() {
        // Mostly true: [T, T, T, T, T, F, T, T, T, T] - triggers true-heavy path
        let ends = PrimitiveArray::from_iter([5u32, 6, 10]);
        let values = BoolArray::from(BitBuffer::from(vec![true, false, true]));
        let decoded = runend_decode_bools(ends, values, 0, 10);

        let expected = BoolArray::from(BitBuffer::from(vec![
            true, true, true, true, true, false, true, true, true, true,
        ]));
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_bools_mostly_false() {
        // Mostly false: [F, F, F, F, F, T, F, F, F, F] - triggers false-heavy path
        let ends = PrimitiveArray::from_iter([5u32, 6, 10]);
        let values = BoolArray::from(BitBuffer::from(vec![false, true, false]));
        let decoded = runend_decode_bools(ends, values, 0, 10);

        let expected = BoolArray::from(BitBuffer::from(vec![
            false, false, false, false, false, true, false, false, false, false,
        ]));
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_bools_all_true() {
        // All true: single run
        let ends = PrimitiveArray::from_iter([10u32]);
        let values = BoolArray::from(BitBuffer::from(vec![true]));
        let decoded = runend_decode_bools(ends, values, 0, 10);

        let expected = BoolArray::from(BitBuffer::from(vec![
            true, true, true, true, true, true, true, true, true, true,
        ]));
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_bools_all_false() {
        // All false: single run
        let ends = PrimitiveArray::from_iter([10u32]);
        let values = BoolArray::from(BitBuffer::from(vec![false]));
        let decoded = runend_decode_bools(ends, values, 0, 10);

        let expected = BoolArray::from(BitBuffer::from(vec![
            false, false, false, false, false, false, false, false, false, false,
        ]));
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_bools_with_offset() {
        // Test with offset: [T, T, F, F, F, T, T, T, T, T] -> slice [2..8] = [F, F, F, T, T, T]
        let ends = PrimitiveArray::from_iter([2u32, 5, 10]);
        let values = BoolArray::from(BitBuffer::from(vec![true, false, true]));
        let decoded = runend_decode_bools(ends, values, 2, 6);

        let expected =
            BoolArray::from(BitBuffer::from(vec![false, false, false, true, true, true]));
        assert_arrays_eq!(decoded, expected);
    }

    // ============================================================================
    // Constant value detection tests for primitive decode
    // ============================================================================

    #[test]
    fn decode_constant_value_all_42s() {
        // All runs have the same value (42) - should trigger the constant value fast path
        let ends = PrimitiveArray::from_iter([3u32, 7, 10]);
        let values = PrimitiveArray::from_iter([42i32, 42, 42]);
        let decoded = runend_decode_primitive(ends, values, 0, 10);

        let expected = PrimitiveArray::from_iter(vec![42i32; 10]);
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_constant_value_single_run() {
        // Single run with constant value
        let ends = PrimitiveArray::from_iter([10u32]);
        let values = PrimitiveArray::from_iter([42i32]);
        let decoded = runend_decode_primitive(ends, values, 0, 10);

        let expected = PrimitiveArray::from_iter(vec![42i32; 10]);
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_non_constant_values() {
        // Non-constant values - should use normal path
        let ends = PrimitiveArray::from_iter([3u32, 7, 10]);
        let values = PrimitiveArray::from_iter([1i32, 2, 3]);
        let decoded = runend_decode_primitive(ends, values, 0, 10);

        let expected = PrimitiveArray::from_iter(vec![1i32, 1, 1, 2, 2, 2, 2, 3, 3, 3]);
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_empty_values_array() {
        // Edge case: empty values array (length 0)
        let ends = PrimitiveArray::from_iter(Vec::<u32>::new());
        let values = PrimitiveArray::from_iter(Vec::<i32>::new());
        let decoded = runend_decode_primitive(ends, values, 0, 0);

        let expected = PrimitiveArray::from_iter(Vec::<i32>::new());
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_constant_value_with_offset() {
        // Constant values with offset - should still work correctly
        // Full array: [42, 42, 42, 42, 42, 42, 42, 42, 42, 42]
        // With offset 2, length 6: [42, 42, 42, 42, 42, 42]
        let ends = PrimitiveArray::from_iter([3u32, 7, 10]);
        let values = PrimitiveArray::from_iter([42i32, 42, 42]);
        let decoded = runend_decode_primitive(ends, values, 2, 6);

        let expected = PrimitiveArray::from_iter(vec![42i32; 6]);
        assert_arrays_eq!(decoded, expected);
    }

    // ============================================================================
    // Sparse zero optimization tests for primitive decode
    // ============================================================================

    #[test]
    fn decode_sparse_data_mostly_zeros() {
        // Sparse data with 90% zeros (9 out of 10 runs are zeros)
        // This should trigger the sparse zero optimization path
        // Pattern: [0, 0, 0, 0, 0, 0, 0, 0, 0, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        let ends = PrimitiveArray::from_iter([
            1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20,
        ]);
        let values = PrimitiveArray::from_iter([
            0i32, 0, 0, 0, 0, 0, 0, 0, 0, 42, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        ]);
        let decoded = runend_decode_primitive(ends, values, 0, 20);

        let mut expected_data = vec![0i32; 20];
        expected_data[9] = 42;
        let expected = PrimitiveArray::from_iter(expected_data);
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_dense_data_mostly_nonzero() {
        // Dense data with only 10% zeros (1 out of 10 runs are zeros)
        // This should NOT trigger the sparse zero optimization path
        // Pattern: [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        let ends = PrimitiveArray::from_iter([1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let values = PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8, 9, 0]);
        let decoded = runend_decode_primitive(ends, values, 0, 10);

        let expected = PrimitiveArray::from_iter(vec![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 0]);
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_sparse_with_long_runs() {
        // Sparse data with long runs - 90% zeros (9 runs of 10 each = 90 zeros, 1 run = 10 non-zeros)
        // Total: 100 elements, 9 zero runs of length 10 each, 1 non-zero run of length 10
        let ends = PrimitiveArray::from_iter([10u32, 20, 30, 40, 50, 60, 70, 80, 90, 100]);
        let values = PrimitiveArray::from_iter([0i32, 0, 0, 0, 42, 0, 0, 0, 0, 0]);
        let decoded = runend_decode_primitive(ends, values, 0, 100);

        let mut expected_data = vec![0i32; 100];
        for i in 40..50 {
            expected_data[i] = 42;
        }
        let expected = PrimitiveArray::from_iter(expected_data);
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_exactly_50_percent_zeros() {
        // Exactly 50% zeros (5 zeros, 5 non-zeros) - should NOT trigger sparse optimization
        // because we require >50%, not >=50%
        let ends = PrimitiveArray::from_iter([1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let values = PrimitiveArray::from_iter([0i32, 1, 0, 2, 0, 3, 0, 4, 0, 5]);
        let decoded = runend_decode_primitive(ends, values, 0, 10);

        let expected = PrimitiveArray::from_iter(vec![0i32, 1, 0, 2, 0, 3, 0, 4, 0, 5]);
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_sparse_with_offset() {
        // Sparse data with offset
        // Full array (10 runs): [0, 0, 0, 0, 0, 42, 0, 0, 0, 0] - 90% zeros
        // With offset 3, length 4: [0, 0, 42, 0]
        let ends = PrimitiveArray::from_iter([1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let values = PrimitiveArray::from_iter([0i32, 0, 0, 0, 0, 42, 0, 0, 0, 0]);
        let decoded = runend_decode_primitive(ends, values, 3, 4);

        let expected = PrimitiveArray::from_iter(vec![0i32, 0, 42, 0]);
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_sparse_u64_type() {
        // Test sparse optimization with u64 type (larger element size)
        let ends = PrimitiveArray::from_iter([1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let values = PrimitiveArray::from_iter([0u64, 0, 0, 0, 0, 0, 0, 0, 0, 42]);
        let decoded = runend_decode_primitive(ends, values, 0, 10);

        let mut expected_data = vec![0u64; 10];
        expected_data[9] = 42;
        let expected = PrimitiveArray::from_iter(expected_data);
        assert_arrays_eq!(decoded, expected);
    }

    #[test]
    fn decode_sparse_f64_type() {
        // Test sparse optimization with f64 type
        let ends = PrimitiveArray::from_iter([1u32, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        let values =
            PrimitiveArray::from_iter([0.0f64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.5]);
        let decoded = runend_decode_primitive(ends, values, 0, 10);

        let mut expected_data = vec![0.0f64; 10];
        expected_data[9] = 3.5;
        let expected = PrimitiveArray::from_iter(expected_data);
        assert_arrays_eq!(decoded, expected);
    }
}
