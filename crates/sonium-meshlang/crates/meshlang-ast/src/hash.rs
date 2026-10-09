//! Deterministic structural hashes for incremental compilation and caches.

use serde::{Deserialize, Serialize};
use std::fmt;

/// A platform-independent 256-bit structural hash.
///
/// This is a cache identity, not a cryptographic digest. Four independently
/// seeded FNV-1a lanes make accidental collisions substantially less likely
/// while keeping the pure AST crate small and WASM-friendly.
#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Default,
)]
pub struct StructuralHash(pub [u8; 32]);

/// Pre-pass for [`StructuralHash::of`]: rejects non-finite floats before
/// they reach `serde_json::to_value`, which would silently map `NaN`/`inf`
/// to `null` and make `NaN` hash identically to `null` (and all NaN payloads
/// collide with each other).
struct FiniteChecker;

macro_rules! finite_checker_forward_ok {
    ($($method:ident($($arg:ty),*)),*) => {
        $(fn $method(self, $(_: $arg),*) -> Result<(), serde_json::Error> {
            Ok(())
        })*
    };
}

impl serde::ser::Serializer for FiniteChecker {
    type Ok = ();
    type Error = serde_json::Error;

    finite_checker_forward_ok!(
        serialize_bool(bool),
        serialize_i8(i8),
        serialize_i16(i16),
        serialize_i32(i32),
        serialize_i64(i64),
        serialize_i128(i128),
        serialize_u8(u8),
        serialize_u16(u16),
        serialize_u32(u32),
        serialize_u64(u64),
        serialize_u128(u128),
        serialize_char(char),
        serialize_str(&str),
        serialize_bytes(&[u8]),
        serialize_unit_struct(&'static str),
        serialize_unit_variant(&'static str, u32, &'static str)
    );

    fn serialize_none(self) -> Result<(), serde_json::Error> {
        Ok(())
    }

    fn serialize_unit(self) -> Result<(), serde_json::Error> {
        Ok(())
    }

    fn serialize_f32(self, value: f32) -> Result<(), serde_json::Error> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(serde::ser::Error::custom(
                "non-finite f32 cannot be structurally hashed",
            ))
        }
    }

    fn serialize_f64(self, value: f64) -> Result<(), serde_json::Error> {
        if value.is_finite() {
            Ok(())
        } else {
            Err(serde::ser::Error::custom(
                "non-finite f64 cannot be structurally hashed",
            ))
        }
    }

    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<(), serde_json::Error> {
        value.serialize(self)
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<(), serde_json::Error> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        value: &T,
    ) -> Result<(), serde_json::Error> {
        value.serialize(self)
    }

    fn serialize_seq(self, _len: Option<usize>) -> Result<Self::SerializeSeq, serde_json::Error> {
        Ok(FiniteChecker)
    }

    fn serialize_tuple(self, _len: usize) -> Result<Self::SerializeTuple, serde_json::Error> {
        Ok(FiniteChecker)
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleStruct, serde_json::Error> {
        Ok(FiniteChecker)
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeTupleVariant, serde_json::Error> {
        Ok(FiniteChecker)
    }

    fn serialize_map(self, _len: Option<usize>) -> Result<Self::SerializeMap, serde_json::Error> {
        Ok(FiniteChecker)
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStruct, serde_json::Error> {
        Ok(FiniteChecker)
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        _variant: &'static str,
        _len: usize,
    ) -> Result<Self::SerializeStructVariant, serde_json::Error> {
        Ok(FiniteChecker)
    }

    type SerializeSeq = FiniteChecker;
    type SerializeTuple = FiniteChecker;
    type SerializeTupleStruct = FiniteChecker;
    type SerializeTupleVariant = FiniteChecker;
    type SerializeMap = FiniteChecker;
    type SerializeStruct = FiniteChecker;
    type SerializeStructVariant = FiniteChecker;
}

macro_rules! finite_checker_seq {
    () => {
        fn serialize_element<T: ?Sized + Serialize>(
            &mut self,
            value: &T,
        ) -> Result<(), serde_json::Error> {
            value.serialize(FiniteChecker)
        }

        fn end(self) -> Result<(), serde_json::Error> {
            Ok(())
        }
    };
}

impl serde::ser::SerializeSeq for FiniteChecker {
    type Ok = ();
    type Error = serde_json::Error;
    finite_checker_seq!();
}

impl serde::ser::SerializeTuple for FiniteChecker {
    type Ok = ();
    type Error = serde_json::Error;
    finite_checker_seq!();
}

impl serde::ser::SerializeTupleStruct for FiniteChecker {
    type Ok = ();
    type Error = serde_json::Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        value: &T,
    ) -> Result<(), serde_json::Error> {
        value.serialize(FiniteChecker)
    }

    fn end(self) -> Result<(), serde_json::Error> {
        Ok(())
    }
}

impl serde::ser::SerializeTupleVariant for FiniteChecker {
    type Ok = ();
    type Error = serde_json::Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        value: &T,
    ) -> Result<(), serde_json::Error> {
        value.serialize(FiniteChecker)
    }

    fn end(self) -> Result<(), serde_json::Error> {
        Ok(())
    }
}

impl serde::ser::SerializeMap for FiniteChecker {
    type Ok = ();
    type Error = serde_json::Error;

    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<(), serde_json::Error> {
        key.serialize(FiniteChecker)
    }

    fn serialize_value<T: ?Sized + Serialize>(
        &mut self,
        value: &T,
    ) -> Result<(), serde_json::Error> {
        value.serialize(FiniteChecker)
    }

    fn end(self) -> Result<(), serde_json::Error> {
        Ok(())
    }
}

impl serde::ser::SerializeStruct for FiniteChecker {
    type Ok = ();
    type Error = serde_json::Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> Result<(), serde_json::Error> {
        value.serialize(FiniteChecker)
    }

    fn end(self) -> Result<(), serde_json::Error> {
        Ok(())
    }
}

impl serde::ser::SerializeStructVariant for FiniteChecker {
    type Ok = ();
    type Error = serde_json::Error;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        _key: &'static str,
        value: &T,
    ) -> Result<(), serde_json::Error> {
        value.serialize(FiniteChecker)
    }

    fn end(self) -> Result<(), serde_json::Error> {
        Ok(())
    }
}

impl StructuralHash {
    pub fn of<T: Serialize>(value: &T) -> Result<Self, StructuralHashError> {
        value
            .serialize(FiniteChecker)
            .map_err(StructuralHashError)?;
        let value = serde_json::to_value(value).map_err(StructuralHashError)?;
        let mut canonical = Vec::with_capacity(32);
        canonical.extend_from_slice(b"meshlang-structural-json-v2\0");
        write_canonical_json(&value, &mut canonical).map_err(StructuralHashError)?;
        Ok(Self::from_bytes(&canonical))
    }

    pub fn from_bytes(bytes: &[u8]) -> Self {
        const SEEDS: [u64; 4] = [
            0xcbf2_9ce4_8422_2325,
            0x9e37_79b1_85eb_ca87,
            0xd6e8_feb8_6659_fd93,
            0xa076_1d64_78bd_642f,
        ];
        let mut output = [0_u8; 32];
        for (lane, seed) in SEEDS.into_iter().enumerate() {
            let mut hash = seed;
            for byte in bytes {
                hash ^= u64::from(*byte);
                hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
                hash ^= hash >> 32;
            }
            output[lane * 8..(lane + 1) * 8].copy_from_slice(&hash.to_le_bytes());
        }
        Self(output)
    }

    pub fn to_hex(self) -> String {
        let mut value = String::with_capacity(64);
        for byte in self.0 {
            use std::fmt::Write as _;
            write!(&mut value, "{byte:02x}").expect("writing to String cannot fail");
        }
        value
    }
}

fn write_canonical_json(
    value: &serde_json::Value,
    output: &mut Vec<u8>,
) -> Result<(), serde_json::Error> {
    match value {
        serde_json::Value::Null => output.extend_from_slice(b"null"),
        serde_json::Value::Bool(value) => {
            output.extend_from_slice(if *value { b"true" } else { b"false" });
        }
        serde_json::Value::Number(number) => {
            // Treat the two encodings of zero as the same structural value.
            if number.as_f64().is_some_and(|value| value == 0.0) {
                output.extend_from_slice(b"0");
            } else {
                output.extend_from_slice(number.to_string().as_bytes());
            }
        }
        serde_json::Value::String(value) => {
            output.extend_from_slice(&serde_json::to_vec(value)?);
        }
        serde_json::Value::Array(values) => {
            output.push(b'[');
            for (index, value) in values.iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                write_canonical_json(value, output)?;
            }
            output.push(b']');
        }
        serde_json::Value::Object(values) => {
            let mut entries: Vec<_> = values.iter().collect();
            entries.sort_unstable_by_key(|(key, _)| *key);

            output.push(b'{');
            for (index, (key, value)) in entries.into_iter().enumerate() {
                if index != 0 {
                    output.push(b',');
                }
                output.extend_from_slice(&serde_json::to_vec(key)?);
                output.push(b':');
                write_canonical_json(value, output)?;
            }
            output.push(b'}');
        }
    }
    Ok(())
}

impl fmt::Display for StructuralHash {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.to_hex())
    }
}

#[derive(Debug)]
pub struct StructuralHashError(serde_json::Error);

impl fmt::Display for StructuralHashError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "failed to serialize value for structural hashing: {}",
            self.0
        )
    }
}

impl std::error::Error for StructuralHashError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(&self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn hashes_are_deterministic_and_structural() {
        let first = StructuralHash::of(&vec!["a", "b"]).unwrap();
        let repeated = StructuralHash::of(&vec!["a", "b"]).unwrap();
        let changed = StructuralHash::of(&vec!["b", "a"]).unwrap();
        assert_eq!(first, repeated);
        assert_ne!(first, changed);
        assert_eq!(first.to_hex().len(), 64);
    }

    #[test]
    fn non_finite_floats_are_rejected_instead_of_hashing_as_null() {
        // serde_json maps NaN to null; without the pre-pass NaN and null
        // would hash identically.
        assert!(StructuralHash::of(&f64::NAN).is_err());
        assert!(StructuralHash::of(&f64::INFINITY).is_err());
        assert!(StructuralHash::of(&vec![1.0_f64, f64::NAN]).is_err());
        assert!(StructuralHash::of(&serde_json::Value::Null).is_ok());
        assert!(StructuralHash::of(&0.0_f64).is_ok());
    }

    #[test]
    fn hashes_are_independent_of_object_key_order_and_signed_zero() {
        let mut first = HashMap::new();
        first.insert("left", -0.0_f64);
        first.insert("right", 1.0);

        let mut second = HashMap::new();
        second.insert("right", 1.0_f64);
        second.insert("left", 0.0);

        assert_eq!(
            StructuralHash::of(&first).unwrap(),
            StructuralHash::of(&second).unwrap()
        );
    }
}
