//! Strict ingestion boundary for independent PC-fan product measurements.
//!
//! The solver-side correlation and structural-calibration contracts intentionally
//! accept provenance strings so deterministic fixtures can exercise their math.
//! This module is the stronger boundary for real product evidence: it requires
//! instrument/calibration metadata, a raw-artifact checksum, explicit units, and
//! a monotone RPM table before adapting measurements into either contract.

use crate::correlation::{PcFanCorrelationDataset, PcFanCorrelationObservation};
use crate::structural_calibration::{
    PcFanStructuralCalibrationDataset, PcFanStructuralCalibrationObservation,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::fmt;

const CSV_HEADER: &str = "rpm,airflow_cfm,pressure_drop_pa,acoustic_level_db,winding_temperature_c,maximum_stress_pa,thermal_expansion_m,fatigue_alternating_stress_pa";

/// Metadata required to distinguish an instrument record from a synthetic
/// fixture or a solver-generated prediction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PcFanMeasuredProductEvidence {
    /// Product family or model identifier.
    pub product_id: String,
    /// Physical unit serial number or test article identifier.
    pub serial_number: String,
    /// Instrument chain identifier used for the acquisition.
    pub instrument_id: String,
    /// Calibration certificate or traceability record identifier.
    pub calibration_certificate: String,
    /// Acquisition timestamp in an externally agreed UTC representation.
    pub acquisition_utc: String,
    /// URI/path or immutable object identifier for the raw record.
    pub source_uri: String,
    /// SHA-256 checksum of the raw record, lower- or upper-case hex.
    pub raw_data_sha256: String,
}

impl PcFanMeasuredProductEvidence {
    /// Validate that the metadata is sufficiently specific for independent
    /// evidence admission.
    pub fn validate(&self) -> Result<(), PcFanMeasuredProductError> {
        for (field, value) in [
            ("product_id", self.product_id.as_str()),
            ("serial_number", self.serial_number.as_str()),
            ("instrument_id", self.instrument_id.as_str()),
            (
                "calibration_certificate",
                self.calibration_certificate.as_str(),
            ),
            ("acquisition_utc", self.acquisition_utc.as_str()),
            ("source_uri", self.source_uri.as_str()),
        ] {
            if value.trim().is_empty() {
                return Err(PcFanMeasuredProductError::MissingEvidenceField(field));
            }
        }
        if !is_utc_timestamp(&self.acquisition_utc) {
            return Err(PcFanMeasuredProductError::InvalidTimestamp(
                self.acquisition_utc.clone(),
            ));
        }

        if self.raw_data_sha256.len() != 64
            || !self
                .raw_data_sha256
                .bytes()
                .all(|byte| byte.is_ascii_hexdigit())
        {
            return Err(PcFanMeasuredProductError::InvalidChecksum(
                self.raw_data_sha256.clone(),
            ));
        }

        Ok(())
    }

    /// Return the stable evidence provenance label used by downstream
    /// validation reports.
    pub fn provenance(&self) -> String {
        format!(
            "measured-product:{}:{}:{}:{}",
            self.product_id, self.serial_number, self.instrument_id, self.source_uri
        )
    }
}

/// One row of the fixed-unit measured-product record.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanMeasuredProductObservation {
    /// Rotation speed in revolutions per minute.
    pub rpm: f64,
    /// Volume flow in cubic feet per minute.
    pub airflow_cfm: f64,
    /// Pressure rise/drop in pascals.
    pub pressure_drop_pa: f64,
    /// A-weighted or otherwise declared acoustic level in dB.
    pub acoustic_level_db: Option<f64>,
    /// Winding temperature in degrees Celsius.
    pub winding_temperature_c: f64,
    /// Maximum structural stress in pascals.
    pub maximum_stress_pa: f64,
    /// Thermal expansion in metres.
    pub thermal_expansion_m: f64,
    /// Alternating fatigue stress in pascals.
    pub fatigue_alternating_stress_pa: f64,
}

/// Product measurements plus the evidence record that makes them admissible.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanMeasuredProductDataset {
    pub evidence: PcFanMeasuredProductEvidence,
    pub observations: Vec<PcFanMeasuredProductObservation>,
}

impl PcFanMeasuredProductDataset {
    /// Validate metadata, finite physical values, and strict RPM topology.
    pub fn validate(&self, min_points: usize) -> Result<(), PcFanMeasuredProductError> {
        self.evidence.validate()?;
        if min_points == 0 || self.observations.len() < min_points {
            return Err(PcFanMeasuredProductError::TooFewPoints {
                actual: self.observations.len(),
                required: min_points.max(1),
            });
        }

        for (index, observation) in self.observations.iter().enumerate() {
            validate_observation(index, observation)?;
            if let Some(previous) = index
                .checked_sub(1)
                .and_then(|previous| self.observations.get(previous))
                && observation.rpm.total_cmp(&previous.rpm) != std::cmp::Ordering::Greater
            {
                return Err(PcFanMeasuredProductError::NonIncreasingRpm {
                    index,
                    previous: previous.rpm,
                    rpm: observation.rpm,
                });
            }
        }

        Ok(())
    }

    /// Adapt the record into the aeroacoustic correlation contract.
    pub fn to_correlation_dataset(
        &self,
        require_acoustic: bool,
    ) -> Result<PcFanCorrelationDataset, PcFanMeasuredProductError> {
        self.validate(1)?;
        let mut observations = Vec::with_capacity(self.observations.len());
        for (index, observation) in self.observations.iter().enumerate() {
            if require_acoustic && observation.acoustic_level_db.is_none() {
                return Err(PcFanMeasuredProductError::MissingAcoustic { index });
            }
            observations.push(PcFanCorrelationObservation {
                rpm: observation.rpm,
                airflow_cfm: observation.airflow_cfm,
                pressure_rise_pa: observation.pressure_drop_pa,
                acoustic_level_db: observation.acoustic_level_db,
            });
        }
        Ok(PcFanCorrelationDataset {
            provenance: self.evidence.provenance(),
            observations,
        })
    }

    /// Adapt the record into the thermal/structural calibration contract.
    pub fn to_structural_calibration_dataset(
        &self,
    ) -> Result<PcFanStructuralCalibrationDataset, PcFanMeasuredProductError> {
        self.validate(1)?;
        Ok(PcFanStructuralCalibrationDataset {
            provenance: self.evidence.provenance(),
            observations: self
                .observations
                .iter()
                .map(|observation| PcFanStructuralCalibrationObservation {
                    rpm: observation.rpm,
                    winding_temperature_c: observation.winding_temperature_c,
                    maximum_stress_pa: observation.maximum_stress_pa,
                    thermal_expansion_m: observation.thermal_expansion_m,
                    fatigue_alternating_stress_pa: observation.fatigue_alternating_stress_pa,
                })
                .collect(),
        })
    }
}

/// Errors raised before a product record reaches any comparison metric.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanMeasuredProductError {
    MissingEvidenceField(&'static str),
    InvalidTimestamp(String),
    InvalidChecksum(String),
    ChecksumMismatch {
        expected: String,
        actual: String,
    },
    InvalidCsvHeader,
    InvalidCsvRow {
        line: usize,
        fields: usize,
    },
    InvalidNumber {
        line: usize,
        field: &'static str,
    },
    NonFiniteObservation {
        index: usize,
        field: &'static str,
    },
    InvalidPhysicalObservation {
        index: usize,
        field: &'static str,
        value: f64,
    },
    TooFewPoints {
        actual: usize,
        required: usize,
    },
    NonIncreasingRpm {
        index: usize,
        previous: f64,
        rpm: f64,
    },
    MissingAcoustic {
        index: usize,
    },
}

impl fmt::Display for PcFanMeasuredProductError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MissingEvidenceField(field) => {
                write!(formatter, "missing measured-product evidence field {field}")
            }
            Self::InvalidTimestamp(timestamp) => write!(
                formatter,
                "invalid measured-product UTC acquisition timestamp {timestamp:?}"
            ),
            Self::InvalidChecksum(checksum) => {
                write!(formatter, "invalid raw-data SHA-256 checksum {checksum:?}")
            }
            Self::ChecksumMismatch { expected, actual } => write!(
                formatter,
                "measured-product CSV checksum mismatch: expected {expected}, computed {actual}"
            ),
            Self::InvalidCsvHeader => write!(formatter, "invalid measured-product CSV header"),
            Self::InvalidCsvRow { line, fields } => write!(
                formatter,
                "measured-product CSV line {line} has {fields} fields; expected 8"
            ),
            Self::InvalidNumber { line, field } => {
                write!(
                    formatter,
                    "invalid measured-product CSV number at line {line}, field {field}"
                )
            }
            Self::NonFiniteObservation { index, field } => write!(
                formatter,
                "measured-product observation {index} has non-finite {field}"
            ),
            Self::InvalidPhysicalObservation {
                index,
                field,
                value,
            } => write!(
                formatter,
                "measured-product observation {index} has invalid {field}={value}"
            ),
            Self::TooFewPoints { actual, required } => write!(
                formatter,
                "measured-product dataset has {actual} points; requires {required}"
            ),
            Self::NonIncreasingRpm {
                index,
                previous,
                rpm,
            } => write!(
                formatter,
                "measured-product RPM at index {index} is not increasing: {rpm} <= {previous}"
            ),
            Self::MissingAcoustic { index } => write!(
                formatter,
                "measured-product observation {index} is missing acoustic level"
            ),
        }
    }
}

impl std::error::Error for PcFanMeasuredProductError {}

fn is_utc_timestamp(value: &str) -> bool {
    let bytes = value.as_bytes();
    if bytes.len() < 20
        || bytes[4] != b'-'
        || bytes[7] != b'-'
        || bytes[10] != b'T'
        || bytes[13] != b':'
        || bytes[16] != b':'
        || *bytes.last().unwrap_or(&0) != b'Z'
    {
        return false;
    }
    let digit = |index: usize| bytes.get(index).is_some_and(u8::is_ascii_digit);
    for index in [0, 1, 2, 3, 5, 6, 8, 9, 11, 12, 14, 15, 17, 18] {
        if !digit(index) {
            return false;
        }
    }
    let number = |start: usize, end: usize| {
        bytes[start..end]
            .iter()
            .fold(0_u32, |value, &digit| value * 10 + u32::from(digit - b'0'))
    };
    let month = number(5, 7);
    let day = number(8, 10);
    let hour = number(11, 13);
    let minute = number(14, 16);
    let second = number(17, 19);
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || hour > 23
        || minute > 59
        || second > 60
    {
        return false;
    }
    if bytes.len() == 20 {
        return true;
    }
    bytes[19] == b'.'
        && bytes.len() > 21
        && bytes[20..bytes.len() - 1].iter().all(u8::is_ascii_digit)
}

/// Parse the fixed-unit product-measurement CSV schema.
pub fn parse_pc_fan_measured_product_csv(
    csv: &str,
    evidence: PcFanMeasuredProductEvidence,
) -> Result<PcFanMeasuredProductDataset, PcFanMeasuredProductError> {
    evidence.validate()?;
    let mut lines = csv.lines();
    if lines.next().map(str::trim) != Some(CSV_HEADER) {
        return Err(PcFanMeasuredProductError::InvalidCsvHeader);
    }

    let mut observations = Vec::new();
    for (line_index, line) in lines.enumerate() {
        let line_number = line_index + 2;
        if line.trim().is_empty() {
            continue;
        }
        let fields: Vec<_> = line.split(',').map(str::trim).collect();
        if fields.len() != 8 {
            return Err(PcFanMeasuredProductError::InvalidCsvRow {
                line: line_number,
                fields: fields.len(),
            });
        }

        let parse = |field: &'static str, value: &str| {
            value
                .parse::<f64>()
                .map_err(|_| PcFanMeasuredProductError::InvalidNumber {
                    line: line_number,
                    field,
                })
        };
        observations.push(PcFanMeasuredProductObservation {
            rpm: parse("rpm", fields[0])?,
            airflow_cfm: parse("airflow_cfm", fields[1])?,
            pressure_drop_pa: parse("pressure_drop_pa", fields[2])?,
            acoustic_level_db: if fields[3].is_empty() {
                None
            } else {
                Some(parse("acoustic_level_db", fields[3])?)
            },
            winding_temperature_c: parse("winding_temperature_c", fields[4])?,
            maximum_stress_pa: parse("maximum_stress_pa", fields[5])?,
            thermal_expansion_m: parse("thermal_expansion_m", fields[6])?,
            fatigue_alternating_stress_pa: parse("fatigue_alternating_stress_pa", fields[7])?,
        });
    }

    let dataset = PcFanMeasuredProductDataset {
        evidence,
        observations,
    };
    dataset.validate(1)?;
    let actual_checksum = pc_fan_measured_product_csv_sha256(csv);
    if !dataset
        .evidence
        .raw_data_sha256
        .eq_ignore_ascii_case(&actual_checksum)
    {
        return Err(PcFanMeasuredProductError::ChecksumMismatch {
            expected: dataset.evidence.raw_data_sha256.clone(),
            actual: actual_checksum,
        });
    }
    Ok(dataset)
}

/// Return the lowercase SHA-256 digest that measured-product ingestion binds
/// to the exact CSV bytes, including line endings.
pub fn pc_fan_measured_product_csv_sha256(csv: &str) -> String {
    let digest = Sha256::digest(csv.as_bytes());
    digest.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn validate_observation(
    index: usize,
    observation: &PcFanMeasuredProductObservation,
) -> Result<(), PcFanMeasuredProductError> {
    let values = [
        ("rpm", observation.rpm),
        ("airflow_cfm", observation.airflow_cfm),
        ("pressure_drop_pa", observation.pressure_drop_pa),
        ("winding_temperature_c", observation.winding_temperature_c),
        ("maximum_stress_pa", observation.maximum_stress_pa),
        ("thermal_expansion_m", observation.thermal_expansion_m),
        (
            "fatigue_alternating_stress_pa",
            observation.fatigue_alternating_stress_pa,
        ),
    ];
    for (field, value) in values {
        if !value.is_finite() {
            return Err(PcFanMeasuredProductError::NonFiniteObservation { index, field });
        }
    }
    if let Some(value) = observation.acoustic_level_db
        && !value.is_finite()
    {
        return Err(PcFanMeasuredProductError::NonFiniteObservation {
            index,
            field: "acoustic_level_db",
        });
    }
    for (field, value) in [
        ("rpm", observation.rpm),
        ("airflow_cfm", observation.airflow_cfm),
        ("winding_temperature_c", observation.winding_temperature_c),
        ("maximum_stress_pa", observation.maximum_stress_pa),
        ("thermal_expansion_m", observation.thermal_expansion_m),
        (
            "fatigue_alternating_stress_pa",
            observation.fatigue_alternating_stress_pa,
        ),
    ] {
        if value < 0.0 || (field == "rpm" && value <= 0.0) {
            return Err(PcFanMeasuredProductError::InvalidPhysicalObservation {
                index,
                field,
                value,
            });
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const CSV: &str = "rpm,airflow_cfm,pressure_drop_pa,acoustic_level_db,winding_temperature_c,maximum_stress_pa,thermal_expansion_m,fatigue_alternating_stress_pa\n600,42.0,18.0,31.0,35.0,1000000,0.00001,400000\n900,48.0,25.0,36.0,40.0,1100000,0.00002,450000\n1200,52.0,31.0,42.0,46.0,1200000,0.00003,500000\n";

    fn evidence() -> PcFanMeasuredProductEvidence {
        PcFanMeasuredProductEvidence {
            product_id: "fan-120mm-test".into(),
            serial_number: "SN-001".into(),
            instrument_id: "daq-chain-7".into(),
            calibration_certificate: "cal-2026-001".into(),
            acquisition_utc: "2026-08-24T12:00:00Z".into(),
            source_uri: "https://example.invalid/raw/fan-001.csv".into(),
            raw_data_sha256: pc_fan_measured_product_csv_sha256(CSV),
        }
    }

    #[test]
    fn csv_parser_preserves_evidence_and_adapts_both_contracts() {
        let dataset = parse_pc_fan_measured_product_csv(CSV, evidence()).unwrap();
        dataset.validate(3).unwrap();
        let correlation = dataset.to_correlation_dataset(true).unwrap();
        let structural = dataset.to_structural_calibration_dataset().unwrap();
        assert_eq!(correlation.observations.len(), 3);
        assert_eq!(structural.observations.len(), 3);
        assert!(correlation.provenance.starts_with("measured-product:"));
        assert_eq!(structural.observations[2].maximum_stress_pa, 1_200_000.0);
    }

    #[test]
    fn evidence_rejects_blank_metadata_and_bad_checksum() {
        let mut invalid = evidence();
        invalid.instrument_id.clear();
        assert_eq!(
            invalid.validate(),
            Err(PcFanMeasuredProductError::MissingEvidenceField(
                "instrument_id"
            ))
        );
        let mut invalid = evidence();
        invalid.raw_data_sha256 = "not-a-checksum".into();
        assert!(matches!(
            invalid.validate(),
            Err(PcFanMeasuredProductError::InvalidChecksum(_))
        ));

        let mut invalid = evidence();
        invalid.acquisition_utc = "2026-13-24T12:00:00Z".into();
        assert!(matches!(
            invalid.validate(),
            Err(PcFanMeasuredProductError::InvalidTimestamp(_))
        ));
    }

    #[test]
    fn timestamp_validation_accepts_utc_fractional_seconds_only() {
        let mut valid = evidence();
        valid.acquisition_utc = "2026-08-24T12:00:00.123Z".into();
        assert!(valid.validate().is_ok());

        valid.acquisition_utc = "2026-08-24 12:00:00Z".into();
        assert!(matches!(
            valid.validate(),
            Err(PcFanMeasuredProductError::InvalidTimestamp(_))
        ));
    }

    #[test]
    fn parser_rejects_header_rows_nonfinite_values_and_bad_rpm_topology() {
        assert_eq!(
            parse_pc_fan_measured_product_csv("rpm,wrong\n", evidence()),
            Err(PcFanMeasuredProductError::InvalidCsvHeader)
        );
        let malformed = format!("{CSV_HEADER}\n600,42,18,31,35,1e6,1e-5\n");
        assert!(matches!(
            parse_pc_fan_measured_product_csv(&malformed, evidence()),
            Err(PcFanMeasuredProductError::InvalidCsvRow { .. })
        ));
        let nonfinite = CSV.replace("42.0", "NaN");
        assert!(matches!(
            parse_pc_fan_measured_product_csv(&nonfinite, evidence()),
            Err(PcFanMeasuredProductError::NonFiniteObservation { .. })
        ));
        let duplicate = CSV.replace("900,48.0", "600,48.0");
        assert!(matches!(
            parse_pc_fan_measured_product_csv(&duplicate, evidence()),
            Err(PcFanMeasuredProductError::NonIncreasingRpm { .. })
        ));
    }

    #[test]
    fn parser_binds_checksum_to_exact_csv_bytes() {
        let mut invalid = evidence();
        invalid.raw_data_sha256.replace_range(..2, "ff");
        assert!(matches!(
            parse_pc_fan_measured_product_csv(CSV, invalid),
            Err(PcFanMeasuredProductError::ChecksumMismatch { .. })
        ));

        let crlf = CSV.replace('\n', "\r\n");
        assert!(matches!(
            parse_pc_fan_measured_product_csv(&crlf, evidence()),
            Err(PcFanMeasuredProductError::ChecksumMismatch { .. })
        ));
    }

    #[test]
    fn acoustic_adapter_can_require_or_allow_missing_acoustic_data() {
        let csv = CSV.replacen(",31.0,", ",,", 1);
        let mut csv_evidence = evidence();
        csv_evidence.raw_data_sha256 = pc_fan_measured_product_csv_sha256(&csv);
        let dataset = parse_pc_fan_measured_product_csv(&csv, csv_evidence).unwrap();
        assert!(matches!(
            dataset.to_correlation_dataset(true),
            Err(PcFanMeasuredProductError::MissingAcoustic { index: 0 })
        ));
        assert!(dataset.to_correlation_dataset(false).is_ok());
    }

    #[test]
    fn product_dataset_serde_round_trip_preserves_evidence() {
        let dataset = parse_pc_fan_measured_product_csv(CSV, evidence()).unwrap();
        let encoded = serde_json::to_vec(&dataset).unwrap();
        let decoded: PcFanMeasuredProductDataset = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.evidence, dataset.evidence);
        assert_eq!(decoded.observations.len(), dataset.observations.len());
    }
}
