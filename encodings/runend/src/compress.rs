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

pub fn runend_decode_typed_bool(
    run_ends: impl Iterator<Item = usize>,
    values: &BitBuffer,
    values_validity: Mask,
    values_nullability: Nullability,
    length: usize,
) -> BoolArray {
    match values_validity {
        Mask::AllTrue(_) => {
            // Pre-allocate buffer with all zeros (false)
            // Only need to fill the true runs - false runs are already 0
            let mut decoded = BitBufferMut::new_unset(length);
            let decoded_bytes = decoded.as_mut_slice();
            let mut current_pos = 0usize;

            for (end, value) in run_ends.zip_eq(values.iter()) {
                // Skip runs that are entirely before the current position
                // Only fill when value is true (false is already 0)
                if end > current_pos && value {
                    fill_bits_true(decoded_bytes, current_pos, end);
                }
                current_pos = end;
            }
            BoolArray::from_bit_buffer(decoded.freeze(), values_nullability.into())
        }
        Mask::AllFalse(_) => {
            BoolArray::from_bit_buffer(BitBuffer::new_unset(length), Validity::AllInvalid)
        }
        Mask::Values(mask) => {
            // Pre-allocate both buffers with zeros
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
                // Skip runs that are entirely before the current position
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
            BoolArray::from_bit_buffer(decoded.freeze(), Validity::from(decoded_validity.freeze()))
        }
    }
}

#[cfg(test)]
mod test {
    use vortex_array::ToCanonical;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;

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
}
