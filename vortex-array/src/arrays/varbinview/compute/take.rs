// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ptr;

use num_traits::AsPrimitive;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::match_each_integer_ptype;
use vortex_error::VortexResult;
use vortex_mask::AllOr;
use vortex_mask::Mask;
use vortex_vector::binaryview::BinaryView;

use crate::Array;
use crate::ArrayRef;
use crate::IntoArray;
use crate::ToCanonical;
use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;
use crate::compute::TakeKernel;
use crate::compute::TakeKernelAdapter;
use crate::register_kernel;
use crate::vtable::ValidityHelper;

/// Take involves creating a new array that references the old array, just with the given set of views.
impl TakeKernel for VarBinViewVTable {
    fn take(&self, array: &VarBinViewArray, indices: &dyn Array) -> VortexResult<ArrayRef> {
        // Compute the new validity.
        let validity = array.validity().take(indices)?;
        let indices = indices.to_primitive();

        let views_buffer = match_each_integer_ptype!(indices.ptype(), |I| {
            take_views(
                array.views(),
                indices.as_slice::<I>(),
                &indices.validity_mask(),
            )
        });

        // SAFETY: taking all components at same indices maintains invariants
        unsafe {
            Ok(VarBinViewArray::new_unchecked(
                views_buffer,
                array.buffers().clone(),
                array
                    .dtype()
                    .union_nullability(indices.dtype().nullability()),
                validity,
            )
            .into_array())
        }
    }
}

register_kernel!(TakeKernelAdapter(VarBinViewVTable).lift());

/// Optimized take implementation for BinaryView arrays.
///
/// This implementation uses direct pointer writes instead of iterator-based collection
/// to minimize function call overhead. BinaryView is exactly 16 bytes (128 bits),
/// which allows efficient copying via `ptr::copy_nonoverlapping`.
fn take_views<I: AsPrimitive<usize>>(
    views: &Buffer<BinaryView>,
    indices: &[I],
    mask: &Mask,
) -> Buffer<BinaryView> {
    let len = indices.len();
    if len == 0 {
        return Buffer::empty();
    }

    // Get a direct slice reference to the views - this deref is not trivial, so we do it once.
    let views_slice = views.as_slice();

    match mask.bit_buffer() {
        AllOr::All => take_views_all_valid(views_slice, indices),
        AllOr::None => {
            // All indices are invalid, return a buffer of default (empty) views.
            BufferMut::full(BinaryView::default(), len).freeze()
        }
        AllOr::Some(validity_buffer) => {
            take_views_with_validity(views_slice, indices, validity_buffer)
        }
    }
}

/// Fast path: all indices are valid, use direct pointer writes.
#[inline]
fn take_views_all_valid<I: AsPrimitive<usize>>(
    views: &[BinaryView],
    indices: &[I],
) -> Buffer<BinaryView> {
    let len = indices.len();
    let mut buffer = BufferMut::<BinaryView>::with_capacity(len);
    let spare = buffer.spare_capacity_mut();
    let src_ptr = views.as_ptr();

    // Process indices with direct pointer writes.
    // BinaryView is 16 bytes (128 bits), Copy, and repr(C) aligned to 16 bytes,
    // making it efficient to copy with ptr::copy_nonoverlapping.
    for (i, idx) in indices.iter().enumerate() {
        let src_idx = idx.as_();
        // SAFETY:
        // - src_ptr.add(src_idx) is valid because indices are within bounds of views
        // - spare[i] is valid because we allocated `len` capacity
        // - BinaryView is Copy so no drop concerns
        unsafe {
            ptr::copy_nonoverlapping(src_ptr.add(src_idx), spare[i].as_mut_ptr(), 1);
        }
    }

    // SAFETY: We initialized exactly `len` elements above.
    unsafe { buffer.set_len(len) };
    buffer.freeze()
}

/// Slow path: some indices may be invalid, check validity for each.
#[inline]
fn take_views_with_validity<I: AsPrimitive<usize>>(
    views: &[BinaryView],
    indices: &[I],
    validity_buffer: &vortex_buffer::BitBuffer,
) -> Buffer<BinaryView> {
    let len = indices.len();
    let mut buffer = BufferMut::<BinaryView>::with_capacity(len);
    let spare = buffer.spare_capacity_mut();
    let src_ptr = views.as_ptr();
    let default_view = BinaryView::default();

    // Iterate through validity bits and indices together.
    for (i, (valid, idx)) in validity_buffer.iter().zip(indices.iter()).enumerate() {
        // SAFETY:
        // - spare[i] is valid because we allocated `len` capacity
        // - src_ptr.add(idx.as_()) is valid when valid is true (indices within bounds)
        // - BinaryView is Copy so no drop concerns
        unsafe {
            if valid {
                ptr::copy_nonoverlapping(src_ptr.add(idx.as_()), spare[i].as_mut_ptr(), 1);
            } else {
                spare[i].write(default_view);
            }
        }
    }

    // SAFETY: We initialized exactly `len` elements above.
    unsafe { buffer.set_len(len) };
    buffer.freeze()
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_buffer::BitBuffer;
    use vortex_buffer::buffer;
    use vortex_dtype::DType;
    use vortex_dtype::Nullability::NonNullable;

    use crate::IntoArray;
    use crate::accessor::ArrayAccessor;
    use crate::array::Array;
    use crate::arrays::PrimitiveArray;
    use crate::arrays::VarBinViewArray;
    use crate::canonical::ToCanonical;
    use crate::compute::conformance::take::test_take_conformance;
    use crate::compute::take;
    use crate::validity::Validity;

    #[test]
    fn take_nullable() {
        let arr = VarBinViewArray::from_iter_nullable_str([
            Some("one"),
            None,
            Some("three"),
            Some("four"),
            None,
            Some("six"),
        ]);

        let taken = take(arr.as_ref(), &buffer![0, 3].into_array()).unwrap();

        assert!(taken.dtype().is_nullable());
        assert_eq!(
            taken.to_varbinview().with_iterator(|it| it
                .map(|v| v.map(|b| unsafe { String::from_utf8_unchecked(b.to_vec()) }))
                .collect::<Vec<_>>()),
            [Some("one".to_string()), Some("four".to_string())]
        );
    }

    #[test]
    fn take_nullable_indices() {
        let arr = VarBinViewArray::from_iter(["one", "two"].map(Some), DType::Utf8(NonNullable));

        let indices = PrimitiveArray::new(
            // Verify that garbage values at NULL indices are ignored.
            buffer![1u64, 999],
            Validity::from(BitBuffer::from(vec![true, false])),
        );

        let taken = take(arr.as_ref(), indices.as_ref()).unwrap();

        assert!(taken.dtype().is_nullable());
        assert_eq!(
            taken.to_varbinview().with_iterator(|it| it
                .map(|v| v.map(|b| unsafe { String::from_utf8_unchecked(b.to_vec()) }))
                .collect::<Vec<_>>()),
            [Some("two".to_string()), None]
        );
    }

    #[rstest]
    #[case(VarBinViewArray::from_iter(
        ["hello", "world", "test", "data", "array"].map(Some),
        DType::Utf8(NonNullable),
    ))]
    #[case(VarBinViewArray::from_iter_nullable_str([
        Some("hello"),
        None,
        Some("test"),
        Some("data"),
        None,
    ]))]
    #[case(VarBinViewArray::from_iter(
        [b"hello".as_slice(), b"world", b"test", b"data", b"array"].map(Some),
        DType::Binary(NonNullable),
    ))]
    #[case(VarBinViewArray::from_iter(["single"].map(Some), DType::Utf8(NonNullable)))]
    fn test_take_varbinview_conformance(#[case] array: VarBinViewArray) {
        test_take_conformance(array.as_ref());
    }
}
