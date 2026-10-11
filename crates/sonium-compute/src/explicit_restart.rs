//! Versioned restart serialization for explicit dynamics.
//!
//! Persists [`ExplicitState`](crate::explicit_state::ExplicitState) to a
//! self-describing JSON restart file. Each file carries the frozen
//! [`EXPLICIT_STATE_SCHEMA`](crate::explicit_state::EXPLICIT_STATE_SCHEMA)
//! tag plus a SHA-256 integrity record over the payload, so a restart is
//! rejected unless its schema matches the running code and every payload
//! byte verifies.
//!
//! File layout (single JSON document, deterministic bytes):
//!
//! ```json
//! {"schema": 1, "sha256": "<64 hex chars>", "state": { "<ExplicitState>" }}
//! ```
//!
//! - `schema` must equal `EXPLICIT_STATE_SCHEMA`; anything else fails
//!   closed as a version mismatch before the checksum is even checked.
//! - `sha256` is the lowercase hex digest of the canonical payload bytes,
//!   defined as `serde_json::to_vec` of the `state` value. Writers and
//!   readers hash the same canonical form, so any flipped byte fails.
//! - `state` is the derived `ExplicitState` serialization. `serde_json`
//!   is built with `float_roundtrip`, so every finite `f64` round-trips
//!   bit-exactly; non-finite values are rejected at write time because
//!   JSON cannot represent them.
//!
//! Use [`write_restart`] / [`read_restart`] for files and
//! [`encode_restart`] / [`decode_restart`] for in-memory bytes (handy
//! for tests and throughput benchmarks).

// Rust guideline compliant 2026-02-21

use std::backtrace::Backtrace;
use std::fmt::{Display, Formatter};
use std::fs;
use std::path::Path;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::explicit_state::{EXPLICIT_STATE_SCHEMA, ExplicitState};

/// Restart file envelope coupling schema tag, checksum, and payload.
#[derive(Debug, Serialize, Deserialize)]
struct RestartEnvelope {
    schema: u32,
    sha256: String,
    state: serde_json::Value,
}

/// Private failure modes behind [`RestartError`].
#[derive(Debug)]
enum RestartErrorKind {
    /// Filesystem read/write failure.
    Io(std::io::Error),
    /// Malformed JSON envelope or payload.
    Json(serde_json::Error),
    /// `schema` tag differs from [`EXPLICIT_STATE_SCHEMA`].
    SchemaMismatch { expected: u32, found: u32 },
    /// Payload bytes do not match the recorded SHA-256 digest.
    ChecksumMismatch { expected: String, actual: String },
    /// A state float is NaN or infinite, which JSON cannot encode.
    NonFinite {
        field: &'static str,
        index: Option<usize>,
    },
    /// The state violates [`ExplicitState::is_consistent`].
    Inconsistent { reason: &'static str },
}

/// Restart serialization or deserialization failure.
#[derive(Debug)]
pub struct RestartError {
    kind: RestartErrorKind,
    backtrace: Backtrace,
}

impl RestartError {
    /// Reports whether this error wraps a filesystem failure.
    #[must_use]
    pub const fn is_io(&self) -> bool {
        matches!(self.kind, RestartErrorKind::Io(_))
    }

    /// Reports whether this error wraps malformed JSON.
    #[must_use]
    pub const fn is_json(&self) -> bool {
        matches!(self.kind, RestartErrorKind::Json(_))
    }

    /// Reports whether the `schema` tag mismatched.
    #[must_use]
    pub const fn is_schema_mismatch(&self) -> bool {
        matches!(self.kind, RestartErrorKind::SchemaMismatch { .. })
    }

    /// Reports whether the SHA-256 integrity check failed.
    #[must_use]
    pub const fn is_checksum_mismatch(&self) -> bool {
        matches!(self.kind, RestartErrorKind::ChecksumMismatch { .. })
    }

    /// Reports whether a non-finite float blocked encoding.
    #[must_use]
    pub const fn is_non_finite(&self) -> bool {
        matches!(self.kind, RestartErrorKind::NonFinite { .. })
    }

    /// Reports whether the state layout was inconsistent.
    #[must_use]
    pub const fn is_inconsistent(&self) -> bool {
        matches!(self.kind, RestartErrorKind::Inconsistent { .. })
    }

    /// Returns the expected schema tag for mismatch errors.
    #[must_use]
    pub const fn expected_schema(&self) -> Option<u32> {
        match &self.kind {
            RestartErrorKind::SchemaMismatch { expected, .. } => Some(*expected),
            _ => None,
        }
    }

    /// Returns the schema tag found in the file.
    #[must_use]
    pub const fn found_schema(&self) -> Option<u32> {
        match &self.kind {
            RestartErrorKind::SchemaMismatch { found, .. } => Some(*found),
            _ => None,
        }
    }

    fn from_kind(kind: RestartErrorKind) -> Self {
        Self {
            kind,
            backtrace: Backtrace::capture(),
        }
    }
}

impl Display for RestartError {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        match &self.kind {
            RestartErrorKind::Io(error) => {
                write!(f, "restart file I/O failed: {error}\n{}", self.backtrace)
            }
            RestartErrorKind::Json(error) => {
                write!(
                    f,
                    "restart file is malformed JSON: {error}\n{}",
                    self.backtrace
                )
            }
            RestartErrorKind::SchemaMismatch { expected, found } => write!(
                f,
                "restart schema mismatch: file has {found}, running code expects {expected}\n{}",
                self.backtrace
            ),
            RestartErrorKind::ChecksumMismatch { expected, actual } => write!(
                f,
                "restart checksum mismatch: file records {expected}, payload hashes to {actual}\n{}",
                self.backtrace
            ),
            RestartErrorKind::NonFinite { field, index } => match index {
                Some(index) => write!(
                    f,
                    "restart state has non-finite {field}[{index}]\n{}",
                    self.backtrace
                ),
                None => write!(
                    f,
                    "restart state has non-finite {field}\n{}",
                    self.backtrace
                ),
            },
            RestartErrorKind::Inconsistent { reason } => write!(
                f,
                "restart state is inconsistent: {reason}\n{}",
                self.backtrace
            ),
        }
    }
}

impl std::error::Error for RestartError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match &self.kind {
            RestartErrorKind::Io(error) => Some(error),
            RestartErrorKind::Json(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RestartError {
    fn from(error: std::io::Error) -> Self {
        Self::from_kind(RestartErrorKind::Io(error))
    }
}

impl From<serde_json::Error> for RestartError {
    fn from(error: serde_json::Error) -> Self {
        Self::from_kind(RestartErrorKind::Json(error))
    }
}

/// Rejects states that cannot round-trip bit-exactly through JSON.
fn validate_encodable(state: &ExplicitState) -> Result<(), RestartError> {
    if !state.is_consistent() {
        return Err(RestartError::from_kind(RestartErrorKind::Inconsistent {
            reason: "vector lengths disagree or a mass is non-finite",
        }));
    }
    for (field, values) in [
        ("positions", state.positions.as_slice()),
        ("velocities", state.velocities.as_slice()),
        ("masses", state.masses.as_slice()),
        ("forces", state.forces.as_slice()),
    ] {
        if let Some(index) = values.iter().position(|value| !value.is_finite()) {
            return Err(RestartError::from_kind(RestartErrorKind::NonFinite {
                field,
                index: Some(index),
            }));
        }
    }
    for (field, value) in [("time", state.time), ("dt", state.dt)] {
        if !value.is_finite() {
            return Err(RestartError::from_kind(RestartErrorKind::NonFinite {
                field,
                index: None,
            }));
        }
    }
    Ok(())
}

/// Encodes a state as deterministic restart-file bytes.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_restart::{decode_restart, encode_restart};
/// use sonium_compute::explicit_state::ExplicitState;
///
/// let mut state = ExplicitState::new(2);
/// state.masses.fill(1.5);
/// let bytes = encode_restart(&state).unwrap();
/// assert_eq!(decode_restart(&bytes).unwrap(), state);
/// ```
///
/// # Errors
///
/// Returns [`RestartError`] when the state is inconsistent (ragged
/// vectors), holds a non-finite float, or fails JSON encoding.
pub fn encode_restart(state: &ExplicitState) -> Result<Vec<u8>, RestartError> {
    validate_encodable(state)?;
    // Canonical payload form: both writer and reader hash `to_vec` of
    // the parsed `state` value, so key order and float rendering are
    // fixed regardless of how the envelope was produced.
    let payload = serde_json::to_value(state)?;
    let payload_bytes = serde_json::to_vec(&payload)?;
    let digest = Sha256::digest(&payload_bytes);
    let envelope = RestartEnvelope {
        schema: EXPLICIT_STATE_SCHEMA,
        sha256: format!("{digest:x}"),
        state: payload,
    };
    Ok(serde_json::to_vec(&envelope)?)
}

/// Decodes restart bytes produced by [`encode_restart`].
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_restart::{decode_restart, encode_restart};
/// use sonium_compute::explicit_state::ExplicitState;
///
/// let state = ExplicitState::new(1);
/// let bytes = encode_restart(&state).unwrap();
/// assert_eq!(decode_restart(bytes).unwrap(), state);
/// ```
///
/// # Errors
///
/// Returns [`RestartError`] when the bytes are malformed JSON, the
/// `schema` tag mismatches (checked before the checksum), the SHA-256
/// record disagrees with the payload, or the decoded state is
/// inconsistent.
pub fn decode_restart(bytes: impl AsRef<[u8]>) -> Result<ExplicitState, RestartError> {
    let envelope: RestartEnvelope = serde_json::from_slice(bytes.as_ref())?;
    if envelope.schema != EXPLICIT_STATE_SCHEMA {
        return Err(RestartError::from_kind(RestartErrorKind::SchemaMismatch {
            expected: EXPLICIT_STATE_SCHEMA,
            found: envelope.schema,
        }));
    }
    let payload_bytes = serde_json::to_vec(&envelope.state)?;
    let digest = Sha256::digest(&payload_bytes);
    let actual = format!("{digest:x}");
    if actual != envelope.sha256 {
        return Err(RestartError::from_kind(
            RestartErrorKind::ChecksumMismatch {
                expected: envelope.sha256,
                actual,
            },
        ));
    }
    let state: ExplicitState = serde_json::from_value(envelope.state)?;
    if !state.is_consistent() {
        return Err(RestartError::from_kind(RestartErrorKind::Inconsistent {
            reason: "decoded vector lengths disagree or a mass is non-finite",
        }));
    }
    Ok(state)
}

/// Writes a state to a versioned restart file.
///
/// The full document is encoded in memory first, so a failure never
/// leaves a truncated file behind.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_restart::{read_restart, write_restart};
/// use sonium_compute::explicit_state::ExplicitState;
///
/// let dir = std::env::temp_dir().join(format!("sonium-restart-{}", std::process::id()));
/// std::fs::create_dir_all(&dir).unwrap();
/// let path = dir.join("restart.json");
/// let state = ExplicitState::new(1);
/// write_restart(&path, &state).unwrap();
/// assert_eq!(read_restart(&path).unwrap(), state);
/// std::fs::remove_file(&path).unwrap();
/// ```
///
/// # Errors
///
/// Returns [`RestartError`] when the state cannot be encoded (see
/// [`encode_restart`]) or the file cannot be written.
pub fn write_restart(path: impl AsRef<Path>, state: &ExplicitState) -> Result<(), RestartError> {
    let bytes = encode_restart(state)?;
    fs::write(path.as_ref(), bytes)?;
    Ok(())
}

/// Reads a state back from a restart file.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_restart::{read_restart, write_restart};
/// use sonium_compute::explicit_state::ExplicitState;
///
/// let dir = std::env::temp_dir().join(format!("sonium-restart-{}", std::process::id()));
/// std::fs::create_dir_all(&dir).unwrap();
/// let path = dir.join("restart.json");
/// write_restart(&path, &ExplicitState::new(1)).unwrap();
/// assert_eq!(read_restart(&path).unwrap(), ExplicitState::new(1));
/// std::fs::remove_file(&path).unwrap();
/// ```
///
/// # Errors
///
/// Returns [`RestartError`] when the file cannot be read or its
/// content fails decoding (see [`decode_restart`]).
pub fn read_restart(path: impl AsRef<Path>) -> Result<ExplicitState, RestartError> {
    let bytes = fs::read(path.as_ref())?;
    decode_restart(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Builds a deterministic state with awkward but finite floats.
    fn sample_state(nodes: usize) -> ExplicitState {
        let mut state = ExplicitState::new(nodes);
        for (index, value) in state.positions.iter_mut().enumerate() {
            let signed = if index % 2 == 0 { 1.0 } else { -1.0 };
            *value = signed * (f64::from(u32::try_from(index).unwrap()) + 0.5) / 3.0;
        }
        for (index, value) in state.velocities.iter_mut().enumerate() {
            *value = (f64::from(u32::try_from(index).unwrap()) * 0.0123).sin() * 1.0e3;
        }
        for (index, value) in state.masses.iter_mut().enumerate() {
            *value = 1.0 + f64::from(u32::try_from(index).unwrap()) * 1.0e-300;
        }
        for (index, value) in state.forces.iter_mut().enumerate() {
            *value = -f64::from(u32::try_from(index).unwrap()) * 2.5;
        }
        if nodes > 0 {
            state.positions[0] = f64::MIN_POSITIVE;
            state.velocities[0] = -0.0;
            state.forces[0] = 1.0e300;
        }
        state.time = 1.0 / 3.0;
        state.step = 41;
        state.dt = 1.0e-6;
        state
    }

    fn assert_bit_equal(left: &ExplicitState, right: &ExplicitState) {
        for (name, left_values, right_values) in [
            (
                "positions",
                left.positions.as_slice(),
                right.positions.as_slice(),
            ),
            (
                "velocities",
                left.velocities.as_slice(),
                right.velocities.as_slice(),
            ),
            ("masses", left.masses.as_slice(), right.masses.as_slice()),
            ("forces", left.forces.as_slice(), right.forces.as_slice()),
        ] {
            assert_eq!(left_values.len(), right_values.len(), "{name} length");
            for (index, (left_value, right_value)) in
                left_values.iter().zip(right_values).enumerate()
            {
                assert_eq!(
                    left_value.to_bits(),
                    right_value.to_bits(),
                    "{name}[{index}] bits"
                );
            }
        }
        assert_eq!(left.time.to_bits(), right.time.to_bits(), "time bits");
        assert_eq!(left.step, right.step, "step");
        assert_eq!(left.dt.to_bits(), right.dt.to_bits(), "dt bits");
    }

    #[test]
    fn file_round_trip_is_bit_exact() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("restart.json");
        let state = sample_state(7);
        write_restart(&path, &state).unwrap();
        let back = read_restart(&path).unwrap();
        assert_eq!(back, state);
        assert_bit_equal(&back, &state);
    }

    #[test]
    fn encoding_is_deterministic() {
        let state = sample_state(5);
        let first = encode_restart(&state).unwrap();
        let second = encode_restart(&state).unwrap();
        assert_eq!(first, second);

        let dir = tempfile::tempdir().unwrap();
        let first_path = dir.path().join("first.json");
        let second_path = dir.path().join("second.json");
        write_restart(&first_path, &state).unwrap();
        write_restart(&second_path, &state).unwrap();
        assert_eq!(
            fs::read(&first_path).unwrap(),
            fs::read(&second_path).unwrap()
        );
    }

    #[test]
    fn envelope_carries_schema_tag_and_checksum() {
        let bytes = encode_restart(&sample_state(3)).unwrap();
        let envelope: RestartEnvelope = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(envelope.schema, EXPLICIT_STATE_SCHEMA);
        let payload_bytes = serde_json::to_vec(&envelope.state).unwrap();
        let digest = Sha256::digest(&payload_bytes);
        assert_eq!(envelope.sha256, format!("{digest:x}"));
        assert_eq!(envelope.sha256.len(), 64);
    }

    #[test]
    fn edited_payload_fails_checksum() {
        let bytes = encode_restart(&sample_state(4)).unwrap();
        let mut envelope: RestartEnvelope = serde_json::from_slice(&bytes).unwrap();
        // Valid JSON, stale checksum: the integrity record must catch it.
        envelope.state["positions"][0] = serde_json::json!(12345.6789);
        let tampered = serde_json::to_vec(&envelope).unwrap();
        let error = decode_restart(tampered).unwrap_err();
        assert!(
            error.is_checksum_mismatch(),
            "expected checksum error, got {error}"
        );
    }

    #[test]
    fn flipped_byte_fails_decoding() {
        let mut bytes = encode_restart(&sample_state(4)).unwrap();
        let middle = bytes.len() / 2;
        bytes[middle] ^= 0x01;
        decode_restart(&bytes).unwrap_err();
    }

    #[test]
    fn wrong_schema_is_rejected_first() {
        let bytes = encode_restart(&sample_state(2)).unwrap();
        let mut envelope: RestartEnvelope = serde_json::from_slice(&bytes).unwrap();
        // Bumping the tag also invalidates the checksum, but the
        // version check runs first and must win.
        envelope.schema = EXPLICIT_STATE_SCHEMA + 1;
        let tampered = serde_json::to_vec(&envelope).unwrap();
        let error = decode_restart(tampered).unwrap_err();
        assert!(
            error.is_schema_mismatch(),
            "expected schema error, got {error}"
        );
        assert_eq!(error.expected_schema(), Some(EXPLICIT_STATE_SCHEMA));
        assert_eq!(error.found_schema(), Some(EXPLICIT_STATE_SCHEMA + 1));
    }

    #[test]
    fn missing_file_reports_io() {
        let dir = tempfile::tempdir().unwrap();
        let error = read_restart(dir.path().join("absent.json")).unwrap_err();
        assert!(error.is_io(), "expected I/O error, got {error}");
    }

    #[test]
    fn garbage_reports_json() {
        let error = decode_restart(b"{not valid json").unwrap_err();
        assert!(error.is_json(), "expected JSON error, got {error}");
    }

    #[test]
    fn non_finite_state_is_rejected_at_write() {
        let mut state = sample_state(3);
        state.velocities[1] = f64::NAN;
        let error = encode_restart(&state).unwrap_err();
        assert!(
            error.is_non_finite(),
            "expected non-finite error, got {error}"
        );

        let mut state = sample_state(3);
        state.dt = f64::INFINITY;
        let error = encode_restart(&state).unwrap_err();
        assert!(
            error.is_non_finite(),
            "expected non-finite error, got {error}"
        );

        // No partial file may be left behind by a failed write.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("restart.json");
        write_restart(&path, &state).unwrap_err();
        assert!(!path.exists());
    }

    #[test]
    fn ragged_state_is_rejected() {
        let mut state = sample_state(3);
        state.velocities.pop();
        let error = encode_restart(&state).unwrap_err();
        assert!(
            error.is_inconsistent(),
            "expected inconsistency error, got {error}"
        );

        // A hand-built envelope with a valid checksum but ragged
        // vectors must still fail on the read path.
        let payload = serde_json::to_value(&state).unwrap();
        let payload_bytes = serde_json::to_vec(&payload).unwrap();
        let digest = Sha256::digest(&payload_bytes);
        let envelope = RestartEnvelope {
            schema: EXPLICIT_STATE_SCHEMA,
            sha256: format!("{digest:x}"),
            state: payload,
        };
        let bytes = serde_json::to_vec(&envelope).unwrap();
        let error = decode_restart(bytes).unwrap_err();
        assert!(
            error.is_inconsistent(),
            "expected inconsistency error, got {error}"
        );
    }
}
