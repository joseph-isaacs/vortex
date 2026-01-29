// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::buffer;
use vortex_dtype::DType;
use vortex_dtype::Nullability;
use vortex_scalar::Scalar;

use super::DictArray;
use crate::Array;
use crate::IntoArray;
use crate::ToCanonical;
use crate::accessor::ArrayAccessor;
use crate::arrays::BoolArray;
use crate::arrays::ListArray;
use crate::arrays::PrimitiveArray;
use crate::arrays::VarBinArray;
use crate::arrays::VarBinViewArray;
use crate::assert_arrays_eq;
use crate::validity::Validity;

#[test]
fn test_slice_into_const_dict() {
    let dict = DictArray::try_new(
        PrimitiveArray::from_option_iter(vec![Some(0u32), None, Some(1)]).to_array(),
        PrimitiveArray::from_option_iter(vec![Some(0i32), Some(1), Some(2)]).to_array(),
    )
    .unwrap();

    assert_eq!(
        Some(Scalar::new(dict.dtype().clone(), 0i32.into())),
        dict.slice(0..1).as_constant()
    );

    assert_eq!(
        Some(Scalar::null(dict.dtype().clone())),
        dict.slice(1..2).as_constant()
    );
}

#[test]
fn test_scalar_at_null_code() {
    let dict = DictArray::try_new(
        PrimitiveArray::from_option_iter(vec![None, Some(0u32), None]).to_array(),
        buffer![1i32].into_array(),
    )
    .unwrap();

    let expected = PrimitiveArray::from_option_iter(vec![None, Some(1i32), None]).into_array();
    assert_arrays_eq!(dict, expected);
}

#[test]
fn test_dict_display() {
    let x = DictArray::try_new(
        buffer![0u8, 0, 0, 1, 0, 3].into_array(),
        VarBinArray::from(vec!["Hello", "你好", "Bonjour", "Hola"]).into_array(),
    )
    .unwrap()
    .into_array();

    assert_eq!(
        x.display_values().to_string(),
        "[\"Hello\", \"Hello\", \"Hello\", \"你好\", \"Hello\", \"Hola\"]"
    )
}

#[test]
fn test_dict_list_dict_display() {
    let elements = DictArray::try_new(
        buffer![0u8, 0, 0, 1, 0, 3, 3, 2].into_array(),
        <VarBinArray as FromIterator<_>>::from_iter([
            Some("Hello"),
            Some("你好"),
            None,
            Some("Bonjour"),
            Some("Hola"),
        ])
        .into_array(),
    )
    .unwrap()
    .into_array();

    assert_eq!(
        elements.display_values().to_string(),
        "[\"Hello\", \"Hello\", \"Hello\", \"你好\", \"Hello\", \"Bonjour\", \"Bonjour\", null]"
    );

    let lists = ListArray::try_new(
        elements,
        buffer![0, 1, 1, 1, 3, 3, 5, 8].into_array(),
        Validity::Array(
            BoolArray::from_iter([true, true, false, true, false, true, true]).into_array(),
        ),
    )
    .unwrap()
    .into_array();

    assert_eq!(
        lists.display_values().to_string(),
        "[[\"Hello\"], [], null, [\"Hello\", \"Hello\"], null, [\"你好\", \"Hello\"], [\"Bonjour\", \"Bonjour\", null]]"
    );

    let x = DictArray::try_new(buffer![6u8, 5, 2, 3, 2, 1].into_array(), lists)
        .unwrap()
        .into_array();

    assert_eq!(
        x.display_values().to_string(),
        "[[\"Bonjour\", \"Bonjour\", null], [\"你好\", \"Hello\"], null, [\"Hello\", \"Hello\"], null, []]"
    )
}

// =============================================================================
// Dictionary Take Correctness Tests
// =============================================================================

mod take_correctness {
    use rstest::rstest;

    use super::*;

    /// Test primitive dictionary canonicalization with small cardinality (4 values).
    #[rstest]
    #[case::uniform(vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1])]
    #[case::sequential(vec![0, 1, 2, 3, 0, 1, 2, 3, 0, 1])]
    #[case::repeated_first(vec![0, 0, 0, 0, 0, 0, 0, 0, 0, 0])]
    #[case::repeated_last(vec![3, 3, 3, 3, 3, 3, 3, 3, 3, 3])]
    #[case::alternating(vec![0, 3, 0, 3, 0, 3, 0, 3, 0, 3])]
    fn test_primitive_dict_small_cardinality(#[case] codes: Vec<u32>) {
        let values: Vec<i64> = vec![100, 200, 300, 400];
        let values_arr = PrimitiveArray::from_iter(values.clone());
        let codes_arr = PrimitiveArray::from_iter(codes.iter().copied());

        let dict = DictArray::try_new(codes_arr.into_array(), values_arr.into_array()).unwrap();
        let canonical = dict.to_canonical().into_array().to_primitive();

        // Verify each element
        let expected: Vec<i64> = codes.iter().map(|&c| values[c as usize]).collect();
        let actual: Vec<i64> = canonical.as_slice::<i64>().to_vec();
        assert_eq!(actual, expected);
    }

    /// Test primitive dictionary with medium cardinality (16 values).
    #[test]
    fn test_primitive_dict_medium_cardinality() {
        let values: Vec<i32> = (0..16).collect();
        let codes: Vec<u32> = (0..100).map(|i| (i % 16) as u32).collect();

        let values_arr = PrimitiveArray::from_iter(values.clone());
        let codes_arr = PrimitiveArray::from_iter(codes.iter().copied());

        let dict = DictArray::try_new(codes_arr.into_array(), values_arr.into_array()).unwrap();
        let canonical = dict.to_canonical().into_array().to_primitive();

        let expected: Vec<i32> = codes.iter().map(|&c| values[c as usize]).collect();
        let actual: Vec<i32> = canonical.as_slice::<i32>().to_vec();
        assert_eq!(actual, expected);
    }

    /// Test primitive dictionary with high cardinality (255 values).
    #[test]
    fn test_primitive_dict_high_cardinality() {
        let values: Vec<u32> = (0..255).collect();
        let codes: Vec<u8> = (0..1000).map(|i| (i % 255) as u8).collect();

        let values_arr = PrimitiveArray::from_iter(values.clone());
        let codes_arr = PrimitiveArray::from_iter(codes.iter().map(|&c| c as u32));

        let dict = DictArray::try_new(codes_arr.into_array(), values_arr.into_array()).unwrap();
        let canonical = dict.to_canonical().into_array().to_primitive();

        let expected: Vec<u32> = codes.iter().map(|&c| values[c as usize]).collect();
        let actual: Vec<u32> = canonical.as_slice::<u32>().to_vec();
        assert_eq!(actual, expected);
    }

    /// Test VarBinView dictionary with short strings (inlined).
    #[test]
    fn test_varbinview_dict_short_strings() {
        let values: Vec<&str> = vec!["a", "bb", "ccc", "dddd"];
        let codes: Vec<u32> = vec![0, 1, 2, 3, 0, 1, 2, 3, 3, 2, 1, 0];

        let values_arr = VarBinViewArray::from_iter(
            values.iter().map(|&s| Some(s)),
            DType::Utf8(Nullability::NonNullable),
        );
        let codes_arr = PrimitiveArray::from_iter(codes.iter().copied());

        let dict = DictArray::try_new(codes_arr.into_array(), values_arr.into_array()).unwrap();
        let canonical = dict.to_canonical().into_array().to_varbinview();

        let expected: Vec<String> = codes
            .iter()
            .map(|&c| values[c as usize].to_string())
            .collect();
        let actual: Vec<String> = canonical.with_iterator(|iter| {
            iter.map(|opt| opt.map(|b| String::from_utf8_lossy(b).to_string()).unwrap())
                .collect()
        });
        assert_eq!(actual, expected);
    }

    /// Test VarBinView dictionary with medium strings (~16 bytes, boundary case).
    #[test]
    fn test_varbinview_dict_medium_strings() {
        let values: Vec<&str> = vec![
            "medium_string_a",
            "medium_string_b",
            "medium_string_c",
            "medium_string_d",
        ];
        let codes: Vec<u32> = vec![0, 1, 2, 3, 3, 2, 1, 0, 0, 0, 3, 3];

        let values_arr = VarBinViewArray::from_iter(
            values.iter().map(|&s| Some(s)),
            DType::Utf8(Nullability::NonNullable),
        );
        let codes_arr = PrimitiveArray::from_iter(codes.iter().copied());

        let dict = DictArray::try_new(codes_arr.into_array(), values_arr.into_array()).unwrap();
        let canonical = dict.to_canonical().into_array().to_varbinview();

        let expected: Vec<String> = codes
            .iter()
            .map(|&c| values[c as usize].to_string())
            .collect();
        let actual: Vec<String> = canonical.with_iterator(|iter| {
            iter.map(|opt| opt.map(|b| String::from_utf8_lossy(b).to_string()).unwrap())
                .collect()
        });
        assert_eq!(actual, expected);
    }

    /// Test VarBinView dictionary with long strings (out-of-line).
    #[test]
    fn test_varbinview_dict_long_strings() {
        let values: Vec<&str> = vec![
            "this_is_a_very_long_string_value_a_123456789",
            "this_is_a_very_long_string_value_b_987654321",
            "this_is_a_very_long_string_value_c_abcdefghi",
            "this_is_a_very_long_string_value_d_zyxwvutsr",
        ];
        let codes: Vec<u32> = vec![0, 1, 2, 3, 2, 1, 0, 3, 3, 3, 0, 0];

        let values_arr = VarBinViewArray::from_iter(
            values.iter().map(|&s| Some(s)),
            DType::Utf8(Nullability::NonNullable),
        );
        let codes_arr = PrimitiveArray::from_iter(codes.iter().copied());

        let dict = DictArray::try_new(codes_arr.into_array(), values_arr.into_array()).unwrap();
        let canonical = dict.to_canonical().into_array().to_varbinview();

        let expected: Vec<String> = codes
            .iter()
            .map(|&c| values[c as usize].to_string())
            .collect();
        let actual: Vec<String> = canonical.with_iterator(|iter| {
            iter.map(|opt| opt.map(|b| String::from_utf8_lossy(b).to_string()).unwrap())
                .collect()
        });
        assert_eq!(actual, expected);
    }

    /// Test dictionary with nullable values in the values array.
    #[test]
    fn test_dict_nullable_values() {
        let values = PrimitiveArray::from_option_iter(vec![Some(10i32), None, Some(30), Some(40)]);
        let codes: Vec<u32> = vec![0, 1, 2, 3, 1, 1, 0, 3];

        let codes_arr = PrimitiveArray::from_iter(codes.iter().copied());

        let dict = DictArray::try_new(codes_arr.into_array(), values.into_array()).unwrap();
        let canonical = dict.to_canonical().into_array().to_primitive();

        let expected = PrimitiveArray::from_option_iter(vec![
            Some(10),
            None,
            Some(30),
            Some(40),
            None,
            None,
            Some(10),
            Some(40),
        ]);
        assert_arrays_eq!(canonical, expected);
    }

    /// Test dictionary with nullable codes.
    #[test]
    fn test_dict_nullable_codes() {
        let values = PrimitiveArray::from_iter(vec![100i32, 200, 300]);
        let codes =
            PrimitiveArray::from_option_iter(vec![Some(0u32), None, Some(2), None, Some(1)]);

        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();
        let canonical = dict.to_canonical().into_array().to_primitive();

        let expected =
            PrimitiveArray::from_option_iter(vec![Some(100i32), None, Some(300), None, Some(200)]);
        assert_arrays_eq!(canonical, expected);
    }

    /// Test dictionary with both nullable codes and nullable values.
    #[test]
    fn test_dict_both_nullable() {
        let values = PrimitiveArray::from_option_iter(vec![Some(10i32), None, Some(30)]);
        let codes = PrimitiveArray::from_option_iter(vec![
            Some(0u32),
            None,
            Some(1),
            Some(2),
            None,
            Some(0),
        ]);

        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();
        let canonical = dict.to_canonical().into_array().to_primitive();

        // code=0 -> 10, code=None -> null, code=1 -> null (value is null), code=2 -> 30
        let expected = PrimitiveArray::from_option_iter(vec![
            Some(10i32),
            None,
            None,
            Some(30),
            None,
            Some(10),
        ]);
        assert_arrays_eq!(canonical, expected);
    }

    /// Test single element dictionary.
    #[test]
    fn test_dict_single_element() {
        let values = PrimitiveArray::from_iter(vec![42i32]);
        let codes = PrimitiveArray::from_iter(vec![0u32, 0, 0, 0, 0]);

        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();
        let canonical = dict.to_canonical().into_array().to_primitive();

        let expected = PrimitiveArray::from_iter(vec![42i32, 42, 42, 42, 42]);
        assert_arrays_eq!(canonical, expected);
    }

    /// Test dictionary with all codes pointing to the same value.
    #[test]
    fn test_dict_all_same_code() {
        let values = PrimitiveArray::from_iter(vec![10i32, 20, 30, 40, 50]);
        let codes = PrimitiveArray::from_iter(vec![2u32, 2, 2, 2, 2, 2, 2, 2]);

        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();
        let canonical = dict.to_canonical().into_array().to_primitive();

        let expected = PrimitiveArray::from_iter(vec![30i32, 30, 30, 30, 30, 30, 30, 30]);
        assert_arrays_eq!(canonical, expected);
    }

    /// Test large dictionary to verify scaling behavior.
    #[test]
    fn test_dict_large_codes_small_values() {
        let values: Vec<i64> = (0..16).map(|i| i * 100).collect();
        let codes: Vec<u32> = (0..10_000).map(|i| (i % 16) as u32).collect();

        let values_arr = PrimitiveArray::from_iter(values.iter().copied());
        let codes_arr = PrimitiveArray::from_iter(codes.iter().copied());

        let dict = DictArray::try_new(codes_arr.into_array(), values_arr.into_array()).unwrap();
        let canonical = dict.to_canonical().into_array().to_primitive();

        assert_eq!(canonical.len(), 10_000);

        // Spot check some values
        let actual = canonical.as_slice::<i64>();
        assert_eq!(actual[0], 0);
        assert_eq!(actual[1], 100);
        assert_eq!(actual[15], 1500);
        assert_eq!(actual[16], 0);
        assert_eq!(actual[9999], 1500); // 9999 % 16 = 15
    }

    /// Test that dict values with f32 type are handled correctly.
    #[test]
    fn test_dict_f32_values() {
        let values = PrimitiveArray::from_iter(vec![1.5f32, 2.5, 3.5, 4.5]);
        let codes = PrimitiveArray::from_iter(vec![0u32, 1, 2, 3, 3, 2, 1, 0]);

        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();
        let canonical = dict.to_canonical().into_array().to_primitive();

        let expected = PrimitiveArray::from_iter(vec![1.5f32, 2.5, 3.5, 4.5, 4.5, 3.5, 2.5, 1.5]);
        assert_arrays_eq!(canonical, expected);
    }

    /// Test that dict values with f64 type are handled correctly.
    #[test]
    fn test_dict_f64_values() {
        let values = PrimitiveArray::from_iter(vec![1.5f64, 2.5, 3.5, 4.5]);
        let codes = PrimitiveArray::from_iter(vec![0u32, 1, 2, 3, 3, 2, 1, 0]);

        let dict = DictArray::try_new(codes.into_array(), values.into_array()).unwrap();
        let canonical = dict.to_canonical().into_array().to_primitive();

        let expected = PrimitiveArray::from_iter(vec![1.5f64, 2.5, 3.5, 4.5, 4.5, 3.5, 2.5, 1.5]);
        assert_arrays_eq!(canonical, expected);
    }
}
