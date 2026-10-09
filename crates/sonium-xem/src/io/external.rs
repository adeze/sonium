//! Dependency-free compatibility reader for external XEM implementations.
//!
//! This module deliberately does not deserialize [`crate::RoomConfig`].  It
//! validates the versioned document envelope using generic JSON values, which
//! makes it useful as a contract test for readers written in another language
//! or for tools that need to forward solver-specific nested payloads.

use serde::de::{self, DeserializeSeed, Deserializer, MapAccess, SeqAccess, Visitor};
use serde_json::{Map, Value};
use std::fmt;

struct RejectDuplicateKeys;

impl<'de> DeserializeSeed<'de> for RejectDuplicateKeys {
    type Value = Value;

    fn deserialize<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(RejectDuplicateKeyValue)
    }
}

struct RejectDuplicateKeyValue;

impl<'de> Visitor<'de> for RejectDuplicateKeyValue {
    type Value = Value;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a JSON value")
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Bool(value))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Number(value.into()))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Number(value.into()))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        serde_json::Number::from_f64(value)
            .map(Value::Number)
            .ok_or_else(|| E::custom("non-finite JSON number"))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::String(value.to_owned()))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::String(value))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Null)
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(Value::Null)
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element_seed(RejectDuplicateKeys)? {
            values.push(value);
        }
        Ok(Value::Array(values))
    }

    fn visit_map<A>(self, mut map: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut fields = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            if fields.contains_key(&key) {
                return Err(de::Error::custom(format!(
                    "duplicate JSON object key {key:?}"
                )));
            }
            let value = map.next_value_seed(RejectDuplicateKeys)?;
            fields.insert(key, value);
        }
        Ok(Value::Object(fields))
    }
}

pub(crate) fn parse_json_reject_duplicate_keys(text: &str) -> Result<Value, serde_json::Error> {
    let mut deserializer = serde_json::Deserializer::from_str(text);
    let value = RejectDuplicateKeys.deserialize(&mut deserializer)?;
    deserializer.end()?;
    Ok(value)
}

/// Failure while reading the language-neutral XEM document envelope.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExternalReaderError {
    /// The input was not valid JSON.
    InvalidJson(String),
    /// The document or schema was not an object.
    ExpectedObject { path: &'static str },
    /// A required envelope member was absent.
    MissingField { path: String },
    /// A field had a JSON type incompatible with the envelope contract.
    InvalidType {
        path: &'static str,
        expected: &'static str,
    },
    /// The schema/document contains a field outside the published envelope.
    UnknownField { path: String },
    /// A document value violates a nested JSON Schema constraint.
    InvalidValue { path: String, reason: String },
    /// The schema or document version is not supported by this reader.
    UnsupportedVersion { version: u32 },
    /// The published schema does not describe the required strict envelope.
    InvalidSchema { reason: &'static str },
}

impl fmt::Display for ExternalReaderError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidJson(reason) => write!(f, "invalid XEM JSON: {reason}"),
            Self::ExpectedObject { path } => write!(f, "{path} must be a JSON object"),
            Self::MissingField { path } => write!(f, "missing required XEM field {path}"),
            Self::InvalidType { path, expected } => {
                write!(f, "XEM field {path} must be {expected}")
            }
            Self::UnknownField { path } => write!(f, "unknown XEM envelope field {path}"),
            Self::InvalidValue { path, reason } => {
                write!(f, "invalid XEM value at {path}: {reason}")
            }
            Self::UnsupportedVersion { version } => {
                write!(f, "unsupported XEM envelope version {version}")
            }
            Self::InvalidSchema { reason } => {
                write!(f, "invalid XEM JSON Schema envelope: {reason}")
            }
        }
    }
}

impl std::error::Error for ExternalReaderError {}

/// A generic, owned XEM document accepted by an external reader.
///
/// The `fields` map intentionally contains [`serde_json::Value`] rather than
/// crate-specific types.  A TypeScript, Python, or C++ consumer can implement
/// the same envelope rules and retain nested BEM/FEM-specific data unchanged.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalRoomDocument {
    /// The exchange schema version, defaulting to v1 when omitted.
    pub schema_version: u32,
    /// All validated top-level fields, including optional solver metadata.
    ///
    /// Kept private so the invariants established by the reader cannot be
    /// invalidated after construction. Use [`Self::fields`] for read-only
    /// inspection and the typed accessors for required envelope members.
    fields: Map<String, Value>,
}

/// A generic, owned multiphysics-problem envelope accepted by an external
/// reader.
///
/// The nested records intentionally remain [`serde_json::Value`] so a
/// consumer written in Python, TypeScript, or another systems language can
/// validate the stable envelope and forward backend-specific records without
/// linking the Rust IR types. The map is private after validation so callers
/// cannot mutate a document into an invalid state.
#[derive(Debug, Clone, PartialEq)]
pub struct ExternalMultiphysicsDocument {
    schema_version: u32,
    fields: Map<String, Value>,
}

impl ExternalMultiphysicsDocument {
    /// Return the validated schema version.
    pub const fn schema_version(&self) -> u32 {
        self.schema_version
    }

    /// Return all validated top-level fields without allowing mutation.
    pub fn fields(&self) -> &Map<String, Value> {
        &self.fields
    }

    /// Return the stable problem identifier.
    ///
    /// Panics if the envelope drifts from the reader contract; prefer
    /// [`Self::try_id`] wherever the document crosses a fallible boundary.
    pub fn id(&self) -> &str {
        self.try_id()
            .expect("validated XEM multiphysics document always contains id")
    }

    /// Fallible [`Self::id`]: validator/accessor drift becomes
    /// [`ExternalReaderError::MissingField`] instead of a panic.
    pub fn try_id(&self) -> Result<&str, ExternalReaderError> {
        self.fields
            .get("id")
            .and_then(Value::as_str)
            .ok_or(ExternalReaderError::MissingField {
                path: "id".to_string(),
            })
    }

    /// Return the domain records without copying them.
    ///
    /// Panics if the envelope drifts from the reader contract; prefer
    /// [`Self::try_domains`] wherever the document crosses a fallible boundary.
    pub fn domains(&self) -> &[Value] {
        self.try_domains()
            .expect("validated XEM multiphysics document always contains domains")
    }

    /// Fallible [`Self::domains`].
    pub fn try_domains(&self) -> Result<&[Value], ExternalReaderError> {
        self.fields
            .get("domains")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .ok_or(ExternalReaderError::MissingField {
                path: "domains".to_string(),
            })
    }

    /// Return the field records without copying them.
    ///
    /// Panics if the envelope drifts from the reader contract; prefer
    /// [`Self::try_fields_array`] wherever the document crosses a fallible
    /// boundary.
    pub fn fields_array(&self) -> &[Value] {
        self.try_fields_array()
            .expect("validated XEM multiphysics document always contains fields")
    }

    /// Fallible [`Self::fields_array`].
    pub fn try_fields_array(&self) -> Result<&[Value], ExternalReaderError> {
        self.fields
            .get("fields")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .ok_or(ExternalReaderError::MissingField {
                path: "fields".to_string(),
            })
    }
}

/// Read a multiphysics-problem document without invoking typed Rust
/// deserialization.
pub fn read_external_multiphysics_document(
    text: &str,
) -> Result<ExternalMultiphysicsDocument, ExternalReaderError> {
    let value = parse_json_reject_duplicate_keys(text)
        .map_err(|error| ExternalReaderError::InvalidJson(error.to_string()))?;
    read_external_multiphysics_value(value)
}

/// Read an already parsed multiphysics-problem document against the published
/// strict v1 schema.
pub fn read_external_multiphysics_value(
    value: Value,
) -> Result<ExternalMultiphysicsDocument, ExternalReaderError> {
    validate_schema_document(&crate::schema::multiphysics_problem_json_schema(), &value)?;
    let Value::Object(fields) = value else {
        return Err(ExternalReaderError::ExpectedObject { path: "$" });
    };
    let schema_version = fields
        .get("schema_version")
        .and_then(Value::as_u64)
        .and_then(|version| u32::try_from(version).ok())
        .ok_or(ExternalReaderError::InvalidType {
            path: "$.schema_version",
            expected: "a non-negative integer representable as u32",
        })?;
    Ok(ExternalMultiphysicsDocument {
        schema_version,
        fields,
    })
}

impl ExternalRoomDocument {
    /// Return all validated top-level fields without allowing mutation.
    pub fn fields(&self) -> &Map<String, Value> {
        &self.fields
    }

    /// Return the required room geometry payload.
    pub fn room(&self) -> &Value {
        self.fields
            .get("room")
            .expect("validated XEM document always contains room")
    }

    /// Return the required source array.
    pub fn sources(&self) -> &[Value] {
        self.fields
            .get("sources")
            .and_then(Value::as_array)
            .expect("validated XEM document always contains sources")
    }

    /// Return the required listening-position array.
    pub fn listening_positions(&self) -> &[Value] {
        self.fields
            .get("listening_positions")
            .and_then(Value::as_array)
            .expect("validated XEM document always contains listening_positions")
    }

    /// Return the required frequency payload.
    pub fn frequencies(&self) -> &Value {
        self.fields
            .get("frequencies")
            .expect("validated XEM document always contains frequencies")
    }
}

/// Read an XEM room document without invoking the typed Rust model.
pub fn read_external_room_document(
    text: &str,
) -> Result<ExternalRoomDocument, ExternalReaderError> {
    let value = parse_json_reject_duplicate_keys(text)
        .map_err(|error| ExternalReaderError::InvalidJson(error.to_string()))?;
    read_external_room_value(value)
}

/// Read an already parsed XEM room document without typed deserialization.
pub fn read_external_room_value(value: Value) -> Result<ExternalRoomDocument, ExternalReaderError> {
    let Value::Object(mut fields) = value else {
        return Err(ExternalReaderError::ExpectedObject { path: "$" });
    };

    let schema_version = match fields.remove("schema_version") {
        None => 1,
        Some(Value::Number(number)) => number
            .as_u64()
            .and_then(|version| u32::try_from(version).ok())
            .ok_or(ExternalReaderError::InvalidType {
                path: "$.schema_version",
                expected: "a non-negative integer",
            })?,
        Some(_) => {
            return Err(ExternalReaderError::InvalidType {
                path: "$.schema_version",
                expected: "a non-negative integer",
            });
        }
    };
    if schema_version != 1 {
        return Err(ExternalReaderError::UnsupportedVersion {
            version: schema_version,
        });
    }

    for field in ["room", "sources", "listening_positions", "frequencies"] {
        if !fields.contains_key(field) {
            return Err(ExternalReaderError::MissingField {
                path: field.to_string(),
            });
        }
    }
    if !fields["room"].is_object() {
        return Err(ExternalReaderError::ExpectedObject { path: "$.room" });
    }
    if !fields["sources"].is_array() {
        return Err(ExternalReaderError::InvalidType {
            path: "$.sources",
            expected: "an array",
        });
    }
    if !fields["listening_positions"].is_array() {
        return Err(ExternalReaderError::InvalidType {
            path: "$.listening_positions",
            expected: "an array",
        });
    }
    if !fields["frequencies"].is_object() {
        return Err(ExternalReaderError::ExpectedObject {
            path: "$.frequencies",
        });
    }

    Ok(ExternalRoomDocument {
        schema_version,
        fields,
    })
}

/// Read and fully validate a room document against the published v1 schema.
///
/// This is the safe entry point for external tools that are about to hand the
/// generic document to a solver. The lenient room reader remains available for
/// forward-compatible envelope inspection when a tool needs to preserve
/// nested solver-specific fields before selecting a schema.
pub fn read_external_room_document_checked(
    text: &str,
) -> Result<ExternalRoomDocument, ExternalReaderError> {
    let value = parse_json_reject_duplicate_keys(text)
        .map_err(|error| ExternalReaderError::InvalidJson(error.to_string()))?;
    read_external_room_value_checked(value)
}

/// Read and migrate a versioned room document against the published schema.
///
/// Version 0 used the same envelope fields but did not carry the explicit v1
/// marker.  The migration inserts that marker before strict validation so
/// external readers and the typed [`crate::RoomConfig`] reader expose the same
/// compatibility policy.  Future versions remain rejected until a dedicated
/// migration is registered.
pub fn read_external_room_document_migrated(
    text: &str,
) -> Result<ExternalRoomDocument, ExternalReaderError> {
    let value = parse_json_reject_duplicate_keys(text)
        .map_err(|error| ExternalReaderError::InvalidJson(error.to_string()))?;
    read_external_room_value_migrated(value)
}

/// Validate and read an already parsed room document using the published
/// schema.
pub fn read_external_room_value_checked(
    value: Value,
) -> Result<ExternalRoomDocument, ExternalReaderError> {
    validate_schema_envelope(&crate::schema::room_config_json_schema(), &value)?;
    read_external_room_value(value)
}

/// Validate and read a room document after applying a registered migration.
pub fn read_external_room_value_migrated(
    mut value: Value,
) -> Result<ExternalRoomDocument, ExternalReaderError> {
    let version = value
        .as_object()
        .and_then(|object| object.get("schema_version"))
        .and_then(Value::as_u64);
    if version == Some(0) {
        let object = value
            .as_object_mut()
            .ok_or(ExternalReaderError::ExpectedObject { path: "$" })?;
        object.insert("schema_version".to_string(), Value::Number(1.into()));
    }
    read_external_room_value_checked(value)
}

/// Validate the complete room document against the generated JSON Schema.
///
/// The validator implements the Draft 2020-12 keywords used by
/// [`crate::schema::room_config_json_schema`]: `type`, `const`, `enum`,
/// `oneOf`, `anyOf`, `allOf`, `required`, `properties`, `additionalProperties`,
/// `items`, length/item bounds, and numeric bounds. Keeping this small
/// implementation dependency-free preserves the crate's WASM build while
/// still making the published cross-language contract enforceable for every
/// nested room/source/boundary field. It is intentionally not advertised as a
/// general-purpose validator for arbitrary schemas.
pub fn validate_schema_envelope(
    schema: &Value,
    document: &Value,
) -> Result<(), ExternalReaderError> {
    validate_schema_document(schema, document)?;
    let external = read_external_room_value(document.clone())?;
    validate_room_semantics(document)?;
    let schema_version = document
        .as_object()
        .and_then(|object| object.get("schema_version"));
    if schema_version.is_some() && external.schema_version != 1 {
        return Err(ExternalReaderError::UnsupportedVersion {
            version: external.schema_version,
        });
    }
    Ok(())
}

/// Validate a strict versioned JSON document against an in-memory schema.
///
/// Unlike [`validate_schema_envelope`], this entry point does not assume the
/// room-configuration payload. It is used by the backend-neutral multiphysics
/// IR and by independent readers that only need envelope/schema guarantees.
pub fn validate_schema_document(
    schema: &Value,
    document: &Value,
) -> Result<(), ExternalReaderError> {
    let Value::Object(schema_object) = schema else {
        return Err(ExternalReaderError::ExpectedObject { path: "$schema" });
    };
    if schema_object.get("$schema")
        != Some(&Value::String(
            "https://json-schema.org/draft/2020-12/schema".to_string(),
        ))
    {
        return Err(ExternalReaderError::InvalidSchema {
            reason: "missing Draft 2020-12 identity",
        });
    }
    if schema_object.get("type") != Some(&Value::String("object".to_string()))
        || schema_object.get("additionalProperties") != Some(&Value::Bool(false))
    {
        return Err(ExternalReaderError::InvalidSchema {
            reason: "root must be a strict object",
        });
    }
    let Some(properties) = schema_object.get("properties").and_then(Value::as_object) else {
        return Err(ExternalReaderError::InvalidSchema {
            reason: "root properties are missing",
        });
    };
    let Some(required) = schema_object.get("required").and_then(Value::as_array) else {
        return Err(ExternalReaderError::InvalidSchema {
            reason: "root required list is missing",
        });
    };
    let Value::Object(document_object) = document else {
        return Err(ExternalReaderError::ExpectedObject { path: "$" });
    };
    for required_field in required {
        let Some(field) = required_field.as_str() else {
            return Err(ExternalReaderError::InvalidSchema {
                reason: "required entries must be strings",
            });
        };
        if !document_object.contains_key(field) {
            return Err(ExternalReaderError::MissingField {
                path: field.to_string(),
            });
        }
        if !properties.contains_key(field) {
            return Err(ExternalReaderError::InvalidSchema {
                reason: "required field is absent from properties",
            });
        }
    }
    // Surface an unsupported numeric version before traversing the payload.
    // This keeps the external reader's compatibility error distinct from a
    // generic `const` mismatch and avoids doing work on a document this reader
    // cannot interpret.
    if let Some(version) = document_object
        .get("schema_version")
        .and_then(Value::as_u64)
        && version != 1
    {
        let version = u32::try_from(version).map_err(|_| ExternalReaderError::InvalidType {
            path: "$.schema_version",
            expected: "a non-negative integer representable as u32",
        })?;
        return Err(ExternalReaderError::UnsupportedVersion { version });
    }
    validate_schema_value(schema, document, "$", true)?;
    if let Some(version_schema) = properties.get("schema_version")
        && version_schema.get("const") != Some(&Value::Number(1.into()))
    {
        return Err(ExternalReaderError::InvalidSchema {
            reason: "schema_version must have const 1",
        });
    }
    Ok(())
}

fn invalid_value(path: &str, reason: impl Into<String>) -> ExternalReaderError {
    ExternalReaderError::InvalidValue {
        path: path.to_string(),
        reason: reason.into(),
    }
}

fn schema_object<'a>(
    schema: &'a Value,
    keyword: &'static str,
) -> Result<&'a Map<String, Value>, ExternalReaderError> {
    schema
        .as_object()
        .ok_or(ExternalReaderError::InvalidSchema { reason: keyword })
}

fn validate_schema_value(
    schema: &Value,
    value: &Value,
    path: &str,
    root: bool,
) -> Result<(), ExternalReaderError> {
    let schema_object = schema_object(schema, "schema entries must be objects")?;

    if let Some(one_of) = schema_object.get("oneOf") {
        let alternatives = one_of
            .as_array()
            .ok_or(ExternalReaderError::InvalidSchema {
                reason: "oneOf must be an array",
            })?;
        let mut matches = 0;
        let mut first_error = None;
        for alternative in alternatives {
            match validate_schema_value(alternative, value, path, false) {
                Ok(()) => matches += 1,
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        if matches != 1 {
            if matches == 0
                && let Some(error @ ExternalReaderError::UnknownField { .. }) = first_error
            {
                return Err(error);
            }
            return Err(invalid_value(
                path,
                format!("oneOf expected exactly one match, found {matches}"),
            ));
        }
    }

    if let Some(any_of) = schema_object.get("anyOf") {
        let alternatives = any_of
            .as_array()
            .ok_or(ExternalReaderError::InvalidSchema {
                reason: "anyOf must be an array",
            })?;
        let mut matched = false;
        let mut first_error = None;
        for alternative in alternatives {
            match validate_schema_value(alternative, value, path, false) {
                Ok(()) => {
                    matched = true;
                    break;
                }
                Err(error) => {
                    first_error.get_or_insert(error);
                }
            }
        }
        if !matched {
            if let Some(error @ ExternalReaderError::UnknownField { .. }) = first_error {
                return Err(error);
            }
            return Err(invalid_value(path, "anyOf did not match any alternative"));
        }
    }

    if let Some(all_of) = schema_object.get("allOf") {
        let alternatives = all_of
            .as_array()
            .ok_or(ExternalReaderError::InvalidSchema {
                reason: "allOf must be an array",
            })?;
        for alternative in alternatives {
            validate_schema_value(alternative, value, path, false)?;
        }
    }

    if let Some(expected) = schema_object.get("type")
        && !type_matches(expected, value)?
    {
        return Err(invalid_value(
            path,
            format!(
                "expected {}, got {}",
                type_description(expected)?,
                value_type(value)
            ),
        ));
    }
    if let Some(expected) = schema_object.get("const")
        && expected != value
    {
        return Err(invalid_value(path, "does not equal the schema const"));
    }
    if let Some(options) = schema_object.get("enum") {
        let options = options
            .as_array()
            .ok_or(ExternalReaderError::InvalidSchema {
                reason: "enum must be an array",
            })?;
        if !options.iter().any(|option| option == value) {
            return Err(invalid_value(
                path,
                "is not one of the permitted enum values",
            ));
        }
    }

    if let Some(required) = schema_object.get("required") {
        let required = required
            .as_array()
            .ok_or(ExternalReaderError::InvalidSchema {
                reason: "required must be an array",
            })?;
        let object = value
            .as_object()
            .ok_or_else(|| invalid_value(path, "must be an object"))?;
        for field in required {
            let field = field.as_str().ok_or(ExternalReaderError::InvalidSchema {
                reason: "required entries must be strings",
            })?;
            if !object.contains_key(field) {
                return Err(ExternalReaderError::MissingField {
                    path: child_path(path, field),
                });
            }
        }
    }

    if let Some(properties) = schema_object.get("properties") {
        let properties = properties
            .as_object()
            .ok_or(ExternalReaderError::InvalidSchema {
                reason: "properties must be an object",
            })?;
        let object = value
            .as_object()
            .ok_or_else(|| invalid_value(path, "must be an object"))?;
        for (field, field_value) in object {
            if let Some(field_schema) = properties.get(field) {
                validate_schema_value(field_schema, field_value, &child_path(path, field), false)?;
            } else if schema_object.get("additionalProperties") == Some(&Value::Bool(false)) {
                return Err(ExternalReaderError::UnknownField {
                    path: child_path(path, field),
                });
            } else if let Some(additional_schema) = schema_object
                .get("additionalProperties")
                .filter(|schema| !schema.is_boolean())
            {
                validate_schema_value(
                    additional_schema,
                    field_value,
                    &child_path(path, field),
                    false,
                )?;
            }
        }
    }

    if let Some(items) = schema_object.get("items") {
        if !items.is_object() {
            return Err(ExternalReaderError::InvalidSchema {
                reason: "items must be a schema object",
            });
        }
        let array = value
            .as_array()
            .ok_or_else(|| invalid_value(path, "must be an array"))?;
        for (index, item) in array.iter().enumerate() {
            validate_schema_value(items, item, &child_path(path, &index.to_string()), false)?;
        }
    }

    if let Some(min_items) = schema_object.get("minItems") {
        let min_items = min_items
            .as_u64()
            .ok_or(ExternalReaderError::InvalidSchema {
                reason: "minItems must be a non-negative integer",
            })?;
        let array = value
            .as_array()
            .ok_or_else(|| invalid_value(path, "must be an array"))?;
        if array.len() < min_items as usize {
            return Err(invalid_value(
                path,
                format!("must contain at least {min_items} items"),
            ));
        }
    }
    if let Some(max_items) = schema_object.get("maxItems") {
        let max_items = max_items
            .as_u64()
            .ok_or(ExternalReaderError::InvalidSchema {
                reason: "maxItems must be a non-negative integer",
            })?;
        let array = value
            .as_array()
            .ok_or_else(|| invalid_value(path, "must be an array"))?;
        if array.len() > max_items as usize {
            return Err(invalid_value(
                path,
                format!("must contain at most {max_items} items"),
            ));
        }
    }
    if let Some(min_length) = schema_object.get("minLength") {
        let min_length = min_length
            .as_u64()
            .ok_or(ExternalReaderError::InvalidSchema {
                reason: "minLength must be a non-negative integer",
            })?;
        let string = value
            .as_str()
            .ok_or_else(|| invalid_value(path, "must be a string"))?;
        if string.chars().count() < min_length as usize {
            return Err(invalid_value(
                path,
                format!("must contain at least {min_length} characters"),
            ));
        }
    }

    validate_numeric_bounds(schema_object, value, path)?;

    if root && schema_object.get("additionalProperties") == Some(&Value::Bool(false)) {
        // The recursive object branch above performs the actual root check;
        // this guard documents that `schema_version` is not a special unknown
        // field and is only allowed because it is present in `properties`.
        debug_assert!(
            schema_object
                .get("properties")
                .and_then(Value::as_object)
                .is_some_and(|properties| properties.contains_key("schema_version"))
        );
    }
    Ok(())
}

fn validate_numeric_bounds(
    schema: &Map<String, Value>,
    value: &Value,
    path: &str,
) -> Result<(), ExternalReaderError> {
    let Some(number) = value.as_f64() else {
        if ["minimum", "exclusiveMinimum", "maximum", "exclusiveMaximum"]
            .iter()
            .any(|keyword| schema.contains_key(*keyword))
        {
            return Err(invalid_value(path, "must be a number"));
        }
        return Ok(());
    };
    check_numeric_bound(
        schema,
        number,
        path,
        "minimum",
        "must be greater than or equal to",
        |a, b| a >= b,
    )?;
    check_numeric_bound(
        schema,
        number,
        path,
        "exclusiveMinimum",
        "must be greater than",
        |a, b| a > b,
    )?;
    check_numeric_bound(
        schema,
        number,
        path,
        "maximum",
        "must be less than or equal to",
        |a, b| a <= b,
    )?;
    check_numeric_bound(
        schema,
        number,
        path,
        "exclusiveMaximum",
        "must be less than",
        |a, b| a < b,
    )?;
    Ok(())
}

fn check_numeric_bound(
    schema: &Map<String, Value>,
    number: f64,
    path: &str,
    keyword: &str,
    text: &str,
    predicate: impl FnOnce(f64, f64) -> bool,
) -> Result<(), ExternalReaderError> {
    let Some(bound) = schema.get(keyword) else {
        return Ok(());
    };
    let bound = bound.as_f64().ok_or(ExternalReaderError::InvalidSchema {
        reason: "numeric bounds must be numbers",
    })?;
    if !predicate(number, bound) {
        return Err(invalid_value(path, format!("{text} {bound}")));
    }
    Ok(())
}

fn type_matches(expected: &Value, value: &Value) -> Result<bool, ExternalReaderError> {
    if let Some(expected) = expected.as_str() {
        return Ok(type_matches_name(expected, value));
    }
    let expected = expected
        .as_array()
        .ok_or(ExternalReaderError::InvalidSchema {
            reason: "type must be a string or array",
        })?;
    expected
        .iter()
        .map(|name| {
            let name = name.as_str().ok_or(ExternalReaderError::InvalidSchema {
                reason: "type entries must be strings",
            })?;
            Ok(type_matches_name(name, value))
        })
        .try_fold(false, |matched, result| Ok(matched || result?))
}

fn type_matches_name(expected: &str, value: &Value) -> bool {
    match expected {
        "object" => value.is_object(),
        "array" => value.is_array(),
        "string" => value.is_string(),
        "number" => value.is_number(),
        "integer" => value.as_i64().is_some() || value.as_u64().is_some(),
        "boolean" => value.is_boolean(),
        "null" => value.is_null(),
        _ => false,
    }
}

fn type_description(expected: &Value) -> Result<String, ExternalReaderError> {
    if let Some(name) = expected.as_str() {
        return Ok(name.to_string());
    }
    let names = expected
        .as_array()
        .ok_or(ExternalReaderError::InvalidSchema {
            reason: "type must be a string or array",
        })?;
    names
        .iter()
        .map(|name| {
            name.as_str()
                .map(str::to_string)
                .ok_or(ExternalReaderError::InvalidSchema {
                    reason: "type entries must be strings",
                })
        })
        .collect::<Result<Vec<_>, _>>()
        .map(|names| names.join(" or "))
}

fn value_type(value: &Value) -> &'static str {
    match value {
        Value::Null => "null",
        Value::Bool(_) => "boolean",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn child_path(parent: &str, child: &str) -> String {
    if parent == "$" {
        format!("$.{child}")
    } else if child.chars().all(|character| character.is_ascii_digit()) {
        format!("{parent}[{child}]")
    } else {
        format!("{parent}.{child}")
    }
}

fn validate_room_semantics(document: &Value) -> Result<(), ExternalReaderError> {
    let object = document
        .as_object()
        .ok_or_else(|| invalid_value("$", "must be an object"))?;
    let frequencies = object
        .get("frequencies")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid_value("$.frequencies", "must be an object"))?;
    validate_frequency_semantics(frequencies)?;

    let sources = object
        .get("sources")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_value("$.sources", "must be an array"))?;
    for (index, source) in sources.iter().enumerate() {
        let path = format!("$.sources[{index}]");
        let source = source
            .as_object()
            .ok_or_else(|| invalid_value(&path, "must be an object"))?;
        if let Some(name) = source.get("name").and_then(Value::as_str)
            && name.trim().is_empty()
        {
            return Err(invalid_value(
                &format!("{path}.name"),
                "must not be empty or whitespace",
            ));
        }
        if let Some(directivity) = source.get("directivity") {
            validate_directivity_semantics(directivity, &format!("{path}.directivity"))?;
        }
        if let Some(crossover) = source.get("crossover") {
            validate_crossover_semantics(crossover, &format!("{path}.crossover"))?;
        }
    }
    Ok(())
}

/// Enforce the same frequency-grid contract as the typed `FrequencyConfig`
/// path.  JSON Schema can express positive endpoints, but not the relationship
/// between bounds, point count, and the generated grid's strict monotonicity.
fn validate_frequency_semantics(
    frequencies: &Map<String, Value>,
) -> Result<(), ExternalReaderError> {
    let min_frequency = frequencies
        .get("min_freq")
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid_value("$.frequencies.min_freq", "must be a number"))?;
    let max_frequency = frequencies
        .get("max_freq")
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid_value("$.frequencies.max_freq", "must be a number"))?;
    if !min_frequency.is_finite() || min_frequency <= 0.0 {
        return Err(invalid_value(
            "$.frequencies.min_freq",
            "must be finite and positive",
        ));
    }
    if !max_frequency.is_finite() || max_frequency <= 0.0 {
        return Err(invalid_value(
            "$.frequencies.max_freq",
            "must be finite and positive",
        ));
    }
    if max_frequency < min_frequency {
        return Err(invalid_value(
            "$.frequencies",
            "max_freq must be greater than or equal to min_freq",
        ));
    }

    let num_points = frequencies
        .get("num_points")
        .and_then(Value::as_u64)
        .ok_or_else(|| invalid_value("$.frequencies.num_points", "must be a non-zero integer"))?;
    let num_points = usize::try_from(num_points)
        .map_err(|_| invalid_value("$.frequencies.num_points", "does not fit in usize"))?;
    if num_points == 0 {
        return Err(invalid_value(
            "$.frequencies.num_points",
            "must be non-zero",
        ));
    }
    if num_points == 1 && min_frequency != max_frequency {
        return Err(invalid_value(
            "$.frequencies",
            "a one-point grid requires min_freq to equal max_freq",
        ));
    }
    if min_frequency == max_frequency && num_points != 1 {
        return Err(invalid_value(
            "$.frequencies",
            "equal bounds require exactly one frequency point",
        ));
    }

    let spacing = frequencies
        .get("spacing")
        .and_then(Value::as_str)
        .unwrap_or("logarithmic");
    if !matches!(spacing, "linear" | "logarithmic") {
        return Err(invalid_value(
            "$.frequencies.spacing",
            "must be 'linear' or 'logarithmic'",
        ));
    }

    if num_points > 1 {
        let denominator = (num_points - 1) as f64;
        let (start, end) = if spacing == "linear" {
            (min_frequency, max_frequency)
        } else {
            (min_frequency.ln(), max_frequency.ln())
        };
        let mut previous = if spacing == "linear" {
            start
        } else {
            start.exp()
        };
        if !previous.is_finite() {
            return Err(invalid_value(
                "$.frequencies",
                "generated grid must contain only finite values",
            ));
        }
        for index in 1..num_points {
            // Match `lin_space`/`log_space` exactly so typed and external
            // readers reject the same rounding-induced duplicate sample.
            let coordinate = start + (end - start) * index as f64 / denominator;
            let current = if spacing == "linear" {
                coordinate
            } else {
                coordinate.exp()
            };
            if !current.is_finite() || current <= previous {
                return Err(invalid_value(
                    "$.frequencies",
                    "generated grid must be finite and strictly increasing",
                ));
            }
            previous = current;
        }
    }
    Ok(())
}

fn validate_crossover_semantics(value: &Value, path: &str) -> Result<(), ExternalReaderError> {
    let Some(object) = value.as_object() else {
        return Err(invalid_value(path, "must be an object"));
    };
    if object.get("type").and_then(Value::as_str) == Some("bandpass") {
        let low = object
            .get("low_cutoff")
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid_value(&format!("{path}.low_cutoff"), "must be a number"))?;
        let high = object
            .get("high_cutoff")
            .and_then(Value::as_f64)
            .ok_or_else(|| invalid_value(&format!("{path}.high_cutoff"), "must be a number"))?;
        if high <= low {
            return Err(invalid_value(
                path,
                "high_cutoff must be greater than low_cutoff",
            ));
        }
    }
    Ok(())
}

fn validate_directivity_semantics(value: &Value, path: &str) -> Result<(), ExternalReaderError> {
    let Some(object) = value.as_object() else {
        return Err(invalid_value(path, "must be an object"));
    };
    match object.get("type").and_then(Value::as_str) {
        Some("custom") => {
            let horizontal = angle_axis(object, "horizontal_angles", path, 0.0, 360.0, false)?;
            let vertical = angle_axis(object, "vertical_angles", path, 0.0, 180.0, true)?;
            let magnitude = object
                .get("magnitude")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid_value(&format!("{path}.magnitude"), "must be an array"))?;
            if magnitude.len() != vertical.len()
                || magnitude.iter().any(|row| {
                    row.as_array()
                        .is_none_or(|row| row.len() != horizontal.len())
                })
            {
                return Err(invalid_value(
                    &format!("{path}.magnitude"),
                    "matrix dimensions must match the angle axes",
                ));
            }
        }
        Some("complex_grid") => {
            let horizontal = angle_axis(object, "horizontal_angles", path, 0.0, 360.0, false)?;
            let vertical = angle_axis(object, "vertical_angles", path, 0.0, 180.0, true)?;
            let frequencies = positive_axis(object, "frequencies", path)?;
            let patterns = object
                .get("patterns")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid_value(&format!("{path}.patterns"), "must be an array"))?;
            if patterns.len() != frequencies.len() {
                return Err(invalid_value(
                    &format!("{path}.patterns"),
                    "one pattern is required for each frequency",
                ));
            }
            for (frequency_index, pattern) in patterns.iter().enumerate() {
                let rows = pattern.as_array().ok_or_else(|| {
                    invalid_value(
                        &format!("{path}.patterns[{frequency_index}]"),
                        "must be an array",
                    )
                })?;
                if rows.len() != vertical.len()
                    || rows.iter().any(|row| {
                        row.as_array()
                            .is_none_or(|row| row.len() != horizontal.len())
                    })
                {
                    return Err(invalid_value(
                        &format!("{path}.patterns[{frequency_index}]"),
                        "matrix dimensions must match the angle axes",
                    ));
                }
            }
        }
        _ => {}
    }
    Ok(())
}

fn angle_axis(
    object: &Map<String, Value>,
    field: &str,
    parent: &str,
    minimum: f64,
    maximum: f64,
    inclusive_maximum: bool,
) -> Result<Vec<f64>, ExternalReaderError> {
    let values = object
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_value(&format!("{parent}.{field}"), "must be an array"))?;
    let values = values
        .iter()
        .map(|value| {
            value.as_f64().ok_or_else(|| {
                invalid_value(&format!("{parent}.{field}"), "must contain only numbers")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.len() < 2
        || values.iter().any(|value| {
            !value.is_finite()
                || *value < minimum
                || (!inclusive_maximum && *value >= maximum)
                || (inclusive_maximum && *value > maximum)
        })
        || values.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(invalid_value(
            &format!("{parent}.{field}"),
            "must be finite, ordered, and inside the supported angular range",
        ));
    }
    Ok(values)
}

fn positive_axis(
    object: &Map<String, Value>,
    field: &str,
    parent: &str,
) -> Result<Vec<f64>, ExternalReaderError> {
    let values = object
        .get(field)
        .and_then(Value::as_array)
        .ok_or_else(|| invalid_value(&format!("{parent}.{field}"), "must be an array"))?;
    let values = values
        .iter()
        .map(|value| {
            value.as_f64().ok_or_else(|| {
                invalid_value(&format!("{parent}.{field}"), "must contain only numbers")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    if values.is_empty()
        || values
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || values.windows(2).any(|pair| pair[1] <= pair[0])
    {
        return Err(invalid_value(
            &format!("{parent}.{field}"),
            "must be finite, positive, and strictly increasing",
        ));
    }
    Ok(values)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::room_config_json_schema;

    #[test]
    fn external_reader_accepts_fixture_without_typed_deserialization() {
        let text = include_str!("../../fixtures/room-config-v1.json");
        let value: Value = serde_json::from_str(text).expect("fixture is JSON");
        validate_schema_envelope(&room_config_json_schema(), &value)
            .expect("fixture matches strict published envelope");
        let document = read_external_room_value(value).expect("external envelope is valid");
        assert_eq!(document.schema_version, 1);
        assert_eq!(document.sources().len(), 1);
        assert_eq!(document.listening_positions().len(), 1);
        assert!(document.room().is_object());
        assert!(document.frequencies().is_object());
    }

    #[test]
    fn external_multiphysics_reader_accepts_golden_without_typed_deserialization() {
        let text = include_str!("../../fixtures/multiphysics-problem-v1.json");
        let document = read_external_multiphysics_document(text)
            .expect("multiphysics fixture matches the published strict envelope");
        assert_eq!(document.schema_version(), 1);
        assert_eq!(document.id(), "cross-language-multiphysics-v1");
        assert_eq!(document.domains().len(), 1);
        assert_eq!(document.fields_array().len(), 1);

        let typed: crate::MultiphysicsProblem =
            serde_json::from_str(text).expect("fixture must remain typed-readable");
        typed
            .validate()
            .expect("fixture must satisfy IR references");
        let encoded = serde_json::to_string(&typed).expect("typed fixture serializes");
        let round_trip = read_external_multiphysics_document(&encoded)
            .expect("typed round-trip must remain externally readable");
        assert_eq!(round_trip.id(), document.id());
        assert_eq!(round_trip.fields_array().len(), 1);
    }

    #[test]
    fn external_multiphysics_reader_rejects_unknown_fields_and_versions() {
        let mut value: Value =
            serde_json::from_str(include_str!("../../fixtures/multiphysics-problem-v1.json"))
                .expect("fixture is JSON");
        value["unexpected"] = Value::Bool(true);
        assert!(matches!(
            read_external_multiphysics_value(value),
            Err(ExternalReaderError::UnknownField { path })
                if path == "$.unexpected"
        ));

        let mut unsupported: Value =
            serde_json::from_str(include_str!("../../fixtures/multiphysics-problem-v1.json"))
                .expect("fixture is JSON");
        unsupported["schema_version"] = Value::Number(2.into());
        assert!(matches!(
            read_external_multiphysics_value(unsupported),
            Err(ExternalReaderError::UnsupportedVersion { version: 2 })
        ));
    }

    #[test]
    fn external_reader_rejects_unknown_root_fields_and_versions() {
        let schema = room_config_json_schema();
        let mut value: Value =
            serde_json::from_str(include_str!("../../fixtures/room-config-v1.json"))
                .expect("fixture is JSON");
        value["unexpected"] = Value::Bool(true);
        assert!(matches!(
            validate_schema_envelope(&schema, &value),
            Err(ExternalReaderError::UnknownField { .. })
        ));

        let mut old = value;
        old.as_object_mut()
            .expect("object fixture")
            .remove("unexpected");
        old["schema_version"] = Value::Number(2.into());
        assert!(matches!(
            read_external_room_value(old),
            Err(ExternalReaderError::UnsupportedVersion { version: 2 })
        ));
    }

    #[test]
    fn migrated_external_reader_accepts_v0_and_emits_v1_document() {
        let mut old: Value =
            serde_json::from_str(include_str!("../../fixtures/room-config-v1.json"))
                .expect("fixture is JSON");
        old["schema_version"] = Value::Number(0.into());

        let document = read_external_room_value_migrated(old)
            .expect("v0 has the registered explicit-marker migration");
        assert_eq!(document.schema_version, 1);
        assert!(!document.fields().contains_key("schema_version"));
        assert_eq!(document.sources().len(), 1);
    }

    #[test]
    fn schema_validation_rejects_unknown_nested_fields() {
        let schema = room_config_json_schema();
        let mut value: Value =
            serde_json::from_str(include_str!("../../fixtures/room-config-v1.json"))
                .expect("fixture is JSON");
        value["room"]["unexpected"] = Value::Bool(true);

        assert!(matches!(
            validate_schema_envelope(&schema, &value),
            Err(ExternalReaderError::UnknownField { path }) if path == "$.room.unexpected"
        ));
    }

    #[test]
    fn checked_external_reader_enforces_nested_schema() {
        let mut value: Value =
            serde_json::from_str(include_str!("../../fixtures/room-config-v1.json"))
                .expect("fixture is JSON");
        value["sources"][0]["position"]["unexpected"] = Value::Bool(true);

        assert!(matches!(
            read_external_room_value_checked(value),
            Err(ExternalReaderError::UnknownField { path })
                if path == "$.sources[0].position.unexpected"
        ));
        read_external_room_document_checked(include_str!("../../fixtures/room-config-v1.json"))
            .expect("the checked reader accepts the published fixture");
    }

    #[test]
    fn schema_validation_checks_nested_types_and_bounds() {
        let schema = room_config_json_schema();
        let mut value: Value =
            serde_json::from_str(include_str!("../../fixtures/room-config-v1.json"))
                .expect("fixture is JSON");
        value["frequencies"]["num_points"] = Value::Number(0.into());

        assert!(matches!(
            validate_schema_envelope(&schema, &value),
            Err(ExternalReaderError::InvalidValue { path, .. })
                if path == "$.frequencies.num_points"
        ));
    }

    #[test]
    fn schema_validation_rejects_degenerate_frequency_grids() {
        let schema = room_config_json_schema();
        let mut value: Value =
            serde_json::from_str(include_str!("../../fixtures/room-config-v1.json"))
                .expect("fixture is JSON");
        value["frequencies"] = serde_json::json!({
            "min_freq": 1_000.0,
            "max_freq": 1_000.0,
            "num_points": 2,
            "spacing": "linear"
        });
        assert!(matches!(
            validate_schema_envelope(&schema, &value),
            Err(ExternalReaderError::InvalidValue { path, .. })
                if path == "$.frequencies"
        ));

        // The endpoints differ, but the generated middle point rounds to an
        // endpoint and would duplicate a sweep sample.
        value["frequencies"] = serde_json::json!({
            "min_freq": 1.0,
            "max_freq": 1.0 + f64::EPSILON,
            "num_points": 3,
            "spacing": "linear"
        });
        assert!(matches!(
            validate_schema_envelope(&schema, &value),
            Err(ExternalReaderError::InvalidValue { path, .. })
                if path == "$.frequencies"
        ));
    }

    #[test]
    fn checked_schema_reader_reports_unknown_numeric_versions() {
        let mut value: Value =
            serde_json::from_str(include_str!("../../fixtures/room-config-v1.json"))
                .expect("fixture is JSON");
        value["schema_version"] = Value::Number(2.into());
        assert!(matches!(
            read_external_room_value_checked(value),
            Err(ExternalReaderError::UnsupportedVersion { version: 2 })
        ));
    }

    #[test]
    fn schema_validation_accepts_complex_directivity_and_solver_details() {
        let schema = room_config_json_schema();
        let mut value: Value =
            serde_json::from_str(include_str!("../../fixtures/room-config-v1.json"))
                .expect("fixture is JSON");
        value["sources"][0]["directivity"] = serde_json::json!({
            "type": "complex_grid",
            "horizontal_angles": [0.0, 180.0],
            "vertical_angles": [0.0, 90.0],
            "frequencies": [500.0],
            "patterns": [[
                [[1.0, 0.0], [0.5, 0.0]],
                [[0.25, 0.1], [0.1, -0.1]]
            ]]
        });
        value["solver"] = serde_json::json!({
            "method": "slfmm",
            "mesh_resolution": 4,
            "gmres": {"max_iter": 20, "restart": 10, "tolerance": 1e-8},
            "ilu": {"method": "tbem", "scanning_degree": "fine", "use_hierarchical": true},
            "fmm": {"fmm_type": "slfmm", "expansion_order": 8, "max_particles_per_leaf": 32},
            "adaptive_integration": true,
            "adaptive_meshing": null
        });

        validate_schema_envelope(&schema, &value)
            .expect("nested complex directivity and solver payload must validate");
    }

    #[test]
    fn malformed_reader_corpus_returns_errors_without_panicking() {
        let corpus = [
            "",
            "null",
            "[]",
            "{}",
            "{\"schema_version\":\"one\"}",
            "{\"schema_version\":999,\"room\":{},\"sources\":[],\"listening_positions\":[],\"frequencies\":{}}",
            "{\"room\":{},\"sources\":[],\"listening_positions\":[],\"frequencies\":{\"min_freq\":NaN}}",
        ];
        for input in corpus {
            let result = std::panic::catch_unwind(|| read_external_room_document_checked(input));
            assert!(
                result.is_ok(),
                "reader panicked for malformed input: {input:?}"
            );
            assert!(
                result.expect("panic checked").is_err(),
                "accepted malformed input: {input:?}"
            );
        }
    }

    #[test]
    fn text_readers_reject_duplicate_keys_before_last_value_wins() {
        let room = r#"{
            "schema_version": 1,
            "room": {"width": 1, "width": 2},
            "sources": [],
            "listening_positions": [],
            "frequencies": {"min_freq": 100, "max_freq": 200, "num_points": 2}
        }"#;
        let room_error = read_external_room_document_checked(room)
            .expect_err("duplicate nested room keys must not be collapsed");
        assert!(matches!(
            room_error,
            ExternalReaderError::InvalidJson(message)
                if message.contains("duplicate JSON object key")
        ));

        let multiphysics = r#"{"id":"first","id":"second"}"#;
        let multiphysics_error = read_external_multiphysics_document(multiphysics)
            .expect_err("duplicate multiphysics envelope keys must not be collapsed");
        assert!(matches!(
            multiphysics_error,
            ExternalReaderError::InvalidJson(message)
                if message.contains("duplicate JSON object key")
        ));
    }

    #[test]
    fn typed_room_readers_share_duplicate_key_boundary() {
        let json = r#"{
            "schema_version": 0,
            "room": {},
            "room": {},
            "sources": [],
            "listening_positions": [],
            "frequencies": {"min_freq": 100, "max_freq": 200, "num_points": 2}
        }"#;

        assert!(crate::RoomConfig::from_json_checked(json).is_err());
        assert!(crate::RoomConfig::from_json_migrated(json).is_err());
    }
}
