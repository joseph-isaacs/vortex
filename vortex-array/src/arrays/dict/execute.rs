// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Execution logic for DictArray - takes from values using codes (indices).
//!
//! This module provides optimized dictionary decoding paths for small dictionaries.
//! For dictionaries with few unique values (typical in analytics), we use specialized
//! lookup table approaches that are more cache-friendly and can leverage SIMD better.

use vortex_buffer::Buffer;
use vortex_dtype::IntegerPType;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_integer_ptype;
use vortex_dtype::match_each_native_ptype;
use vortex_error::VortexExpect;

use crate::Canonical;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::arrays::ExtensionArray;
use crate::arrays::FixedSizeListArray;
use crate::arrays::FixedSizeListVTable;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::arrays::NullArray;
use crate::arrays::NullVTable;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::arrays::StructArray;
use crate::arrays::StructVTable;
use crate::arrays::VarBinViewArray;
use crate::arrays::VarBinViewVTable;
use crate::compute::TakeKernel;
use crate::vtable::ValidityHelper;

/// Threshold for using the optimized small-dictionary path.
/// Dictionaries with fewer than this many values benefit from the lookup table approach.
const SMALL_DICT_THRESHOLD: usize = 256;

/// Take from a canonical array using indices (codes), returning a new canonical array.
///
/// This is the core operation for dictionary decoding - it expands the dictionary
/// by looking up each code in the values array.
///
/// For small dictionaries (< 256 values), this uses an optimized lookup table approach
/// that is more cache-friendly. For larger dictionaries, it falls back to the standard
/// take implementation.
pub fn take_canonical(values: Canonical, codes: &PrimitiveArray) -> Canonical {
    match values {
        Canonical::Null(a) => Canonical::Null(take_null(&a, codes)),
        Canonical::Bool(a) => Canonical::Bool(take_bool(&a, codes)),
        Canonical::Primitive(a) => Canonical::Primitive(take_primitive_optimized(&a, codes)),
        Canonical::Decimal(a) => Canonical::Decimal(take_decimal(&a, codes)),
        Canonical::VarBinView(a) => Canonical::VarBinView(take_varbinview(&a, codes)),
        Canonical::List(a) => Canonical::List(take_listview(&a, codes)),
        Canonical::FixedSizeList(a) => Canonical::FixedSizeList(take_fixed_size_list(&a, codes)),
        Canonical::Struct(a) => Canonical::Struct(take_struct(&a, codes)),
        Canonical::Extension(a) => Canonical::Extension(take_extension(&a, codes)),
    }
}

/// Optimized primitive take for dictionary decoding.
///
/// Uses a specialized lookup table approach for small dictionaries that:
/// - Preloads all dictionary values into a contiguous buffer (fits in L1 cache)
/// - Processes codes in batches for better instruction-level parallelism
/// - Avoids repeated validity checks when both values and codes are non-nullable
fn take_primitive_optimized(values: &PrimitiveArray, codes: &PrimitiveArray) -> PrimitiveArray {
    let values_len = values.len();

    // For small dictionaries, use the optimized lookup table approach
    if values_len <= SMALL_DICT_THRESHOLD && values_len > 0 {
        // Compute combined validity upfront
        let validity = values
            .validity()
            .take(codes.as_ref())
            .vortex_expect("take validity");

        match_each_native_ptype!(values.ptype(), |V| {
            match_each_integer_ptype!(codes.ptype(), |C| {
                let result = take_small_dict::<V, C>(values.as_slice::<V>(), codes.as_slice::<C>());
                return PrimitiveArray::new(result, validity);
            })
        })
    }

    // Fall back to the standard take for larger dictionaries
    take_primitive(values, codes)
}

/// Optimized take for small dictionaries using a direct lookup table approach.
///
/// This function is optimized for the common case where:
/// 1. The dictionary has few unique values (fits in L1 cache)
/// 2. We're expanding many codes into the full array
///
/// The key optimizations are:
/// - The values slice is accessed directly without bounds checks (using get_unchecked)
/// - Codes are processed in a tight loop that the compiler can vectorize
/// - Memory access patterns are predictable for the prefetcher
#[inline]
fn take_small_dict<V: NativePType, C: IntegerPType>(values: &[V], codes: &[C]) -> Buffer<V> {
    let num_codes = codes.len();

    // Pre-allocate the result buffer
    let mut result = vortex_buffer::BufferMut::<V>::with_capacity(num_codes);
    let result_ptr = result.spare_capacity_mut().as_mut_ptr().cast::<V>();

    // Process codes in a tight loop
    // The compiler can vectorize this loop and the CPU can pipeline memory accesses
    for (i, &code) in codes.iter().enumerate() {
        let idx: usize = code.as_();
        // SAFETY: Dictionary codes are guaranteed to be valid indices into values
        // (this is an invariant of DictArray construction)
        let value = unsafe { *values.get_unchecked(idx) };
        // SAFETY: We allocated capacity for num_codes elements
        unsafe { result_ptr.add(i).write(value) };
    }

    // SAFETY: We wrote exactly num_codes elements
    unsafe { result.set_len(num_codes) };

    result.freeze()
}

fn take_null(_array: &NullArray, codes: &PrimitiveArray) -> NullArray {
    NullVTable
        .take(_array, codes.as_ref())
        .vortex_expect("take null array")
        .as_::<NullVTable>()
        .clone()
}

fn take_bool(array: &BoolArray, codes: &PrimitiveArray) -> BoolArray {
    BoolVTable
        .take(array, codes.as_ref())
        .vortex_expect("take bool array")
        .as_::<BoolVTable>()
        .clone()
}

fn take_primitive(array: &PrimitiveArray, codes: &PrimitiveArray) -> PrimitiveArray {
    PrimitiveVTable
        .take(array, codes.as_ref())
        .vortex_expect("take primitive array")
        .as_::<PrimitiveVTable>()
        .clone()
}

fn take_decimal(array: &DecimalArray, codes: &PrimitiveArray) -> DecimalArray {
    DecimalVTable
        .take(array, codes.as_ref())
        .vortex_expect("take decimal array")
        .as_::<DecimalVTable>()
        .clone()
}

fn take_varbinview(array: &VarBinViewArray, codes: &PrimitiveArray) -> VarBinViewArray {
    VarBinViewVTable
        .take(array, codes.as_ref())
        .vortex_expect("take varbinview array")
        .as_::<VarBinViewVTable>()
        .clone()
}

fn take_listview(array: &ListViewArray, codes: &PrimitiveArray) -> ListViewArray {
    ListViewVTable
        .take(array, codes.as_ref())
        .vortex_expect("take listview array")
        .as_::<ListViewVTable>()
        .clone()
}

fn take_fixed_size_list(array: &FixedSizeListArray, codes: &PrimitiveArray) -> FixedSizeListArray {
    FixedSizeListVTable
        .take(array, codes.as_ref())
        .vortex_expect("take fixed size list array")
        .as_::<FixedSizeListVTable>()
        .clone()
}

fn take_struct(array: &StructArray, codes: &PrimitiveArray) -> StructArray {
    StructVTable
        .take(array, codes.as_ref())
        .vortex_expect("take struct array")
        .as_::<StructVTable>()
        .clone()
}

fn take_extension(array: &ExtensionArray, codes: &PrimitiveArray) -> ExtensionArray {
    use crate::compute::take;

    let taken_storage =
        take(array.storage(), codes.as_ref()).vortex_expect("take extension storage");
    ExtensionArray::new(array.ext_dtype().clone(), taken_storage)
}
