//! Data loading utilities for WASM environment
//!
//! These functions allow loading configuration and measurement data
//! from strings (e.g., from browser FileReader API) without filesystem access.

use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

const MAX_LOADER_INPUT_BYTES: usize = 16 * 1024 * 1024;
const MAX_MEASUREMENT_POINTS: usize = 1_000_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementData {
    pub name: String,
    pub frequency: Vec<f64>,
    pub spl: Vec<f64>,
    pub phase: Option<Vec<f64>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeasurementBundle {
    pub measurements: Vec<MeasurementData>,
    pub default_name: String,
}

/// Parse a frequency response CSV string
///
/// Expected CSV format:
/// ```csv
/// frequency,On Axis,On Axis ( Listening Window),Listening Window,Sound Power
/// 20,-10.5,-8.2,-6.1,-12.3
/// 25,-8.2,-6.1,-4.5,-9.8
/// ```
///
/// The first column must be frequency (Hz). Additional columns are treated
/// as SPL measurements and the first one is used as the default.
///
/// Returns JSON string on success
#[wasm_bindgen]
pub fn parse_measurement_csv(csv_content: &str) -> Result<String, JsValue> {
    if csv_content.len() > MAX_LOADER_INPUT_BYTES {
        return Err(JsValue::from_str("CSV input exceeds the 16 MiB limit"));
    }
    let mut reader = csv::ReaderBuilder::new()
        .has_headers(true)
        .flexible(true)
        .from_reader(csv_content.as_bytes());

    let headers = reader
        .headers()
        .map_err(|e| JsValue::from_str(&format!("CSV header error: {}", e)))?;

    if headers.len() < 2 {
        return Err(JsValue::from_str(
            "CSV must have at least 2 columns: frequency and one measurement",
        ));
    }

    let measurement_name = if headers.len() >= 2 {
        headers.get(1).unwrap_or("On Axis").to_string()
    } else {
        "default".to_string()
    };

    let mut frequencies: Vec<f64> = Vec::new();
    let mut spl_values: Vec<f64> = Vec::new();

    for result in reader.records() {
        if frequencies.len() >= MAX_MEASUREMENT_POINTS {
            return Err(JsValue::from_str(
                "measurement point count exceeds the limit",
            ));
        }
        let record = result.map_err(|e| JsValue::from_str(&format!("CSV parse error: {}", e)))?;

        if record.len() < 2 {
            continue;
        }

        let freq: f64 = record
            .get(0)
            .ok_or_else(|| JsValue::from_str("Missing frequency value"))?
            .parse()
            .map_err(|_| JsValue::from_str("Invalid frequency value"))?;

        if !freq.is_finite() || freq <= 0.0 {
            return Err(JsValue::from_str(
                "Frequency values must be finite and positive",
            ));
        }

        let spl: f64 = record
            .get(1)
            .ok_or_else(|| JsValue::from_str("Missing SPL value"))?
            .parse()
            .map_err(|_| JsValue::from_str("Invalid SPL value"))?;

        if !spl.is_finite() {
            return Err(JsValue::from_str("SPL values must be finite"));
        }

        frequencies.push(freq);
        spl_values.push(spl);
    }

    if frequencies.is_empty() {
        return Err(JsValue::from_str("No valid data rows found in CSV"));
    }

    let measurement = MeasurementData {
        name: measurement_name,
        frequency: frequencies,
        spl: spl_values,
        phase: None,
    };

    serde_json::to_string(&measurement)
        .map_err(|e| JsValue::from_str(&format!("JSON error: {}", e)))
}

/// Parse a measurement bundle JSON string
///
/// JSON format:
/// ```json
/// {
///   "default": "On Axis",
///   "measurements": ["On Axis", "Listening Window"],
///   "data": {
///     "On Axis": [-10.5, -8.2, ...],
///     "Listening Window": [-8.0, -6.5, ...]
///   }
/// }
/// ```
///
/// Returns JSON string of MeasurementBundle
#[wasm_bindgen]
pub fn parse_measurement_bundle_json(json_content: &str) -> Result<String, JsValue> {
    if json_content.len() > MAX_LOADER_INPUT_BYTES {
        return Err(JsValue::from_str(
            "Measurement bundle JSON exceeds the 16 MiB limit",
        ));
    }

    #[derive(Deserialize)]
    struct RawBundle {
        #[serde(default = "default_string")]
        default: String,
        measurements: Vec<String>,
        data: std::collections::HashMap<String, Vec<f64>>,
    }

    fn default_string() -> String {
        "On Axis".to_string()
    }

    let raw: RawBundle = serde_json::from_str(json_content)
        .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;

    if raw.measurements.len() > MAX_MEASUREMENT_POINTS {
        return Err(JsValue::from_str("measurement count exceeds the limit"));
    }

    let default_name = raw.default;

    let mut measurements = Vec::new();
    for name in raw.measurements {
        if let Some(spl) = raw.data.get(&name) {
            if spl.len() > MAX_MEASUREMENT_POINTS {
                return Err(JsValue::from_str(
                    "measurement point count exceeds the limit",
                ));
            }
            if spl.iter().any(|value| !value.is_finite()) {
                return Err(JsValue::from_str("SPL values must be finite"));
            }
            let freq = generate_spinorama_frequencies(spl.len());
            if freq.len() != spl.len() {
                return Err(JsValue::from_str(
                    "measurement point count exceeds supported spinorama frequency range",
                ));
            }
            measurements.push(MeasurementData {
                name: name.clone(),
                frequency: freq,
                spl: spl.clone(),
                phase: None,
            });
        }
    }

    if measurements.is_empty() {
        return Err(JsValue::from_str("No valid measurements found in bundle"));
    }

    let bundle = MeasurementBundle {
        measurements,
        default_name,
    };

    serde_json::to_string(&bundle).map_err(|e| JsValue::from_str(&format!("JSON error: {}", e)))
}

/// Generate standard spinorama frequency points
fn generate_spinorama_frequencies(num_points: usize) -> Vec<f64> {
    let mut freqs = Vec::with_capacity(num_points);

    let base_freqs: [f64; 31] = [
        20.0, 25.0, 31.5, 40.0, 50.0, 63.0, 80.0, 100.0, 125.0, 160.0, 200.0, 250.0, 315.0, 400.0,
        500.0, 630.0, 800.0, 1000.0, 1250.0, 1600.0, 2000.0, 2500.0, 3150.0, 4000.0, 5000.0,
        6300.0, 8000.0, 10000.0, 12500.0, 16000.0, 20000.0,
    ];

    if num_points <= base_freqs.len() {
        freqs.extend(base_freqs.iter().take(num_points).copied());
    } else {
        freqs.extend_from_slice(&base_freqs);
        while freqs.len() < num_points {
            let last = freqs[freqs.len() - 1];
            let next = if last < 20000.0 {
                last * 1.122 * 1.122
            } else {
                20000.0
            };
            if next > 20000.0 || (next - last) < 0.1 {
                break;
            }
            freqs.push(next);
        }
    }

    freqs
}

/// Interpolate measurement to target frequencies
///
/// Uses logarithmic interpolation for smooth results
///
/// Input JSON: { "measurement": {...}, "target_frequencies": [...] }
/// Output JSON: { "measurement": {...} }
#[wasm_bindgen]
pub fn interpolate_measurement(json_input: &str) -> Result<String, JsValue> {
    if json_input.len() > MAX_LOADER_INPUT_BYTES {
        return Err(JsValue::from_str(
            "Measurement interpolation JSON exceeds the 16 MiB limit",
        ));
    }

    #[derive(Deserialize)]
    struct Input {
        measurement: MeasurementData,
        target_frequencies: Vec<f64>,
    }

    let input: Input = serde_json::from_str(json_input)
        .map_err(|e| JsValue::from_str(&format!("JSON parse error: {}", e)))?;

    let measurement = &input.measurement;
    if measurement.frequency.is_empty() {
        return Err(JsValue::from_str("Empty measurement data"));
    }
    if measurement.frequency.len() != measurement.spl.len() {
        return Err(JsValue::from_str(
            "Frequency and SPL arrays must have the same length",
        ));
    }
    if measurement
        .phase
        .as_ref()
        .is_some_and(|phase| phase.len() != measurement.frequency.len())
    {
        return Err(JsValue::from_str(
            "Frequency and phase arrays must have the same length",
        ));
    }
    if measurement.frequency.len() > MAX_MEASUREMENT_POINTS
        || input.target_frequencies.len() > MAX_MEASUREMENT_POINTS
    {
        return Err(JsValue::from_str(
            "measurement point count exceeds the limit",
        ));
    }
    if measurement
        .frequency
        .iter()
        .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
        || measurement.spl.iter().any(|value| !value.is_finite())
        || measurement
            .phase
            .as_ref()
            .is_some_and(|phase| phase.iter().any(|value| !value.is_finite()))
        || input
            .target_frequencies
            .iter()
            .any(|frequency| !frequency.is_finite() || *frequency <= 0.0)
    {
        return Err(JsValue::from_str(
            "measurement frequencies must be finite and positive and SPL values finite",
        ));
    }
    if measurement
        .frequency
        .windows(2)
        .any(|pair| pair[1] <= pair[0])
    {
        return Err(JsValue::from_str(
            "measurement frequencies must be strictly increasing",
        ));
    }

    let mut interpolated_spl = Vec::with_capacity(input.target_frequencies.len());

    for &target_freq in &input.target_frequencies {
        let spl = interpolate_at_frequency(
            &input.measurement.frequency,
            &input.measurement.spl,
            target_freq,
        );
        interpolated_spl.push(spl);
    }

    let result = MeasurementData {
        name: input.measurement.name.clone(),
        frequency: input.target_frequencies,
        spl: interpolated_spl,
        phase: input.measurement.phase.clone(),
    };

    serde_json::to_string(&result).map_err(|e| JsValue::from_str(&format!("JSON error: {}", e)))
}

fn interpolate_at_frequency(freq: &[f64], spl: &[f64], target: f64) -> f64 {
    if freq.is_empty() || spl.is_empty() {
        return 0.0;
    }
    if target <= freq[0] {
        return spl[0];
    }
    if target >= freq[freq.len() - 1] {
        return spl[freq.len() - 1];
    }

    for i in 0..freq.len() - 1 {
        if target >= freq[i] && target < freq[i + 1] {
            let log_f = target.ln();
            let log_f1 = freq[i].ln();
            let log_f2 = freq[i + 1].ln();
            let t = (log_f - log_f1) / (log_f2 - log_f1);
            return spl[i] * (1.0 - t) + spl[i + 1] * t;
        }
    }

    spl[freq.len() - 1]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_simple_csv() {
        let csv = "frequency,SPL\n20,-10.0\n25,-8.0\n30,-6.0";
        let result = parse_measurement_csv(csv).unwrap();
        let data: MeasurementData = serde_json::from_str(&result).unwrap();
        assert_eq!(data.frequency, vec![20.0, 25.0, 30.0]);
        assert_eq!(data.spl, vec![-10.0, -8.0, -6.0]);
    }

    #[test]
    fn test_interpolate() {
        let measurement = MeasurementData {
            name: "test".to_string(),
            frequency: vec![20.0, 100.0, 1000.0, 10000.0, 20000.0],
            spl: vec![-10.0, -5.0, 0.0, -5.0, -10.0],
            phase: None,
        };

        let target = vec![
            20.0, 50.0, 100.0, 200.0, 500.0, 1000.0, 2000.0, 10000.0, 20000.0,
        ];

        let input = serde_json::json!({
            "measurement": measurement,
            "target_frequencies": target,
        });

        let result = interpolate_measurement(&input.to_string()).unwrap();
        let interpolated: MeasurementData = serde_json::from_str(&result).unwrap();

        assert_eq!(interpolated.frequency.len(), 9);
        assert!((interpolated.spl[2] - (-5.0)).abs() < 0.1);
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn interpolation_rejects_mismatched_and_unsorted_measurements() {
        let mismatched = serde_json::json!({
            "measurement": {
                "name": "bad",
                "frequency": [20.0, 100.0],
                "spl": [-1.0],
                "phase": null
            },
            "target_frequencies": [50.0]
        });
        assert!(interpolate_measurement(&mismatched.to_string()).is_err());

        let unsorted = serde_json::json!({
            "measurement": {
                "name": "bad",
                "frequency": [100.0, 20.0],
                "spl": [-1.0, -2.0],
                "phase": null
            },
            "target_frequencies": [50.0]
        });
        assert!(interpolate_measurement(&unsorted.to_string()).is_err());
    }

    #[cfg(target_arch = "wasm32")]
    #[test]
    fn loader_entry_points_reject_oversized_inputs_before_parsing() {
        let oversized = "{".to_string() + &" ".repeat(MAX_LOADER_INPUT_BYTES);
        assert!(parse_measurement_bundle_json(&oversized).is_err());
        assert!(interpolate_measurement(&oversized).is_err());
        assert!(parse_measurement_csv(&"x".repeat(MAX_LOADER_INPUT_BYTES + 1)).is_err());
    }
}
