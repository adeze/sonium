//! Measured two-way crossover exploration. Predictions do not establish acoustic validity.

pub mod storage;

#[cfg(feature = "desktop")]
pub mod desktop;

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_fem::physics::coupled::lumped_circuit::network::{CircuitNetwork, ImpedanceFn};
use std::f64::consts::TAU;

#[derive(Debug, thiserror::Error)]
pub enum DesignerError {
    #[error("{0}")]
    Invalid(String),
    #[error("Circuit solve failed at {frequency_hz} Hz: {message}")]
    Circuit { frequency_hz: f64, message: String },
}

fn invalid(message: impl Into<String>) -> DesignerError {
    DesignerError::Invalid(message.into())
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CurveKind {
    Response,
    Impedance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Sample {
    pub frequency_hz: f64,
    /// Response: dB SPL. Impedance: ohms. Phase is always degrees.
    pub magnitude: f64,
    pub phase_deg: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Curve {
    pub source: String,
    pub samples: Vec<Sample>,
}

impl Curve {
    pub fn parse(
        text: &str,
        kind: CurveKind,
        source: impl Into<String>,
    ) -> Result<Self, DesignerError> {
        let mut samples = Vec::new();
        for (line, row) in text.trim_start_matches('\u{feff}').lines().enumerate() {
            let row = row.trim();
            if row.is_empty() || ["#", ";", "*", "//"].iter().any(|p| row.starts_with(p)) {
                continue;
            }
            let columns: Vec<_> = row.split_whitespace().collect();
            if columns.len() != 3 {
                return Err(invalid(format!(
                    "Line {}: expected frequency, magnitude and phase",
                    line + 1
                )));
            }
            let values: Result<Vec<f64>, _> = columns.iter().map(|v| v.parse()).collect();
            let values =
                values.map_err(|_| invalid(format!("Line {}: invalid number", line + 1)))?;
            samples.push(Sample {
                frequency_hz: values[0],
                magnitude: values[1],
                phase_deg: values[2],
            });
        }
        let curve = Self {
            source: source.into(),
            samples,
        };
        curve.validate(kind)?;
        Ok(curve)
    }

    /// Validates ordered, finite samples and positive impedance magnitudes.
    pub fn validate(&self, kind: CurveKind) -> Result<(), DesignerError> {
        if self.samples.len() < 2 {
            return Err(invalid(format!(
                "{}: at least two samples required",
                self.source
            )));
        }
        let mut previous = 0.0;
        for sample in &self.samples {
            if !sample.frequency_hz.is_finite()
                || sample.frequency_hz <= previous
                || !(TAU * sample.frequency_hz).is_finite()
                || !sample.magnitude.is_finite()
                || !sample.phase_deg.is_finite()
            {
                return Err(invalid(format!(
                    "{}: samples require finite values and strictly increasing positive frequencies",
                    self.source
                )));
            }
            let value = sample.complex(kind);
            if !finite(value) || value.norm() <= 1e-30 {
                return Err(invalid(format!(
                    "{}: magnitude is zero or outside the numerical range",
                    self.source
                )));
            }
            if kind == CurveKind::Impedance && sample.magnitude <= 0.0 {
                return Err(invalid(format!(
                    "{}: impedance magnitude must be positive",
                    self.source
                )));
            }
            previous = sample.frequency_hz;
        }
        Ok(())
    }

    /// Interpolates complex values over logarithmic frequency; never extrapolates.
    /// Rejects invalid curves and non-finite or out-of-range frequencies.
    pub fn interpolate(&self, frequency: f64, kind: CurveKind) -> Result<Complex64, DesignerError> {
        self.validate(kind)?;
        if !frequency.is_finite() {
            return Err(invalid("Interpolation frequency must be finite"));
        }
        let samples = &self.samples;
        if frequency < samples[0].frequency_hz
            || frequency > samples[samples.len() - 1].frequency_hz
        {
            return Err(invalid(
                "Frequency lies outside imported data; extrapolation is unavailable",
            ));
        }
        let upper = samples.partition_point(|sample| sample.frequency_hz < frequency);
        let value = if samples[upper].frequency_hz == frequency {
            samples[upper].complex(kind)
        } else {
            let a = &samples[upper - 1];
            let b = &samples[upper];
            let t = (frequency.ln() - a.frequency_hz.ln())
                / (b.frequency_hz.ln() - a.frequency_hz.ln());
            a.complex(kind) * (1.0 - t) + b.complex(kind) * t
        };
        if !finite(value) || (kind == CurveKind::Impedance && value.norm() <= 1e-30) {
            return Err(invalid(
                "Interpolated impedance is zero or numerically invalid",
            ));
        }
        Ok(value)
    }
}

impl Sample {
    fn complex(&self, kind: CurveKind) -> Complex64 {
        let magnitude = match kind {
            CurveKind::Response => 10.0_f64.powf(self.magnitude / 20.0),
            CurveKind::Impedance => self.magnitude,
        };
        Complex64::from_polar(magnitude, self.phase_deg.rem_euclid(360.0).to_radians())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Driver {
    pub name: String,
    pub response: Option<Curve>,
    pub impedance: Option<Curve>,
    pub invert: bool,
    pub delay_ms: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Crossover {
    pub woofer_resistance_ohm: f64,
    pub woofer_inductance_mh: f64,
    pub woofer_capacitance_uf: f64,
    pub tweeter_resistance_ohm: f64,
    pub tweeter_capacitance_uf: f64,
    pub tweeter_inductance_mh: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Project {
    pub woofer: Driver,
    pub tweeter: Driver,
    pub crossover: Crossover,
    pub reference_voltage_v: f64,
    pub reference_distance_m: f64,
    pub synthetic: bool,
}

impl Default for Project {
    fn default() -> Self {
        let driver = |name: &str| Driver {
            name: name.into(),
            response: None,
            impedance: None,
            invert: false,
            delay_ms: 0.0,
        };
        Self {
            woofer: driver("Woofer"),
            tweeter: driver("Tweeter"),
            crossover: Crossover {
                woofer_resistance_ohm: 0.0,
                woofer_inductance_mh: 0.9,
                woofer_capacitance_uf: 7.0,
                tweeter_resistance_ohm: 0.0,
                tweeter_capacitance_uf: 7.0,
                tweeter_inductance_mh: 0.9,
            },
            reference_voltage_v: 2.83,
            reference_distance_m: 1.0,
            synthetic: false,
        }
    }
}

impl Project {
    /// Explicit synthetic illustration, never imported measurement evidence.
    pub fn demo() -> Self {
        let mut project = Self {
            synthetic: true,
            ..Self::default()
        };
        for driver in [&mut project.woofer, &mut project.tweeter] {
            let curve = |magnitude| Curve {
                source: "Synthetic flat demonstration".into(),
                samples: [20.0, 20_000.0]
                    .map(|frequency_hz| Sample {
                        frequency_hz,
                        magnitude,
                        phase_deg: 0.0,
                    })
                    .to_vec(),
            };
            driver.response = Some(curve(85.0));
            driver.impedance = Some(curve(8.0));
        }
        project.tweeter.invert = true;
        project
    }

    fn curves(&self) -> Result<[&Curve; 4], DesignerError> {
        Ok([
            self.woofer
                .response
                .as_ref()
                .ok_or_else(|| invalid("Import woofer FRD response"))?,
            self.woofer
                .impedance
                .as_ref()
                .ok_or_else(|| invalid("Import woofer ZMA impedance"))?,
            self.tweeter
                .response
                .as_ref()
                .ok_or_else(|| invalid("Import tweeter FRD response"))?,
            self.tweeter
                .impedance
                .as_ref()
                .ok_or_else(|| invalid("Import tweeter ZMA impedance"))?,
        ])
    }

    pub fn validate(&self) -> Result<(), DesignerError> {
        self.validate_draft()?;
        self.curves()?;
        Ok(())
    }

    /// Validate editable drafts without requiring all four imports to be present.
    pub fn validate_draft(&self) -> Result<(), DesignerError> {
        let c = &self.crossover;
        for value in [
            self.reference_voltage_v,
            self.reference_distance_m,
            c.woofer_inductance_mh,
            c.woofer_capacitance_uf,
            c.tweeter_capacitance_uf,
            c.tweeter_inductance_mh,
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(invalid(
                    "Reference voltage/distance and L/C values must be finite and positive",
                ));
            }
        }
        for value in [
            c.woofer_resistance_ohm,
            c.tweeter_resistance_ohm,
            self.woofer.delay_ms,
            self.tweeter.delay_ms,
        ] {
            if !value.is_finite() || value < 0.0 {
                return Err(invalid(
                    "Resistance and delay must be finite and nonnegative",
                ));
            }
        }
        for driver in [&self.woofer, &self.tweeter] {
            if let Some(curve) = &driver.response {
                curve.validate(CurveKind::Response)?;
            }
            if let Some(curve) = &driver.impedance {
                curve.validate(CurveKind::Impedance)?;
            }
        }
        if let Ok(curves) = self.curves() {
            let (low, high) = support(&curves);
            if low >= high {
                return Err(invalid(
                    "All four curves must share a nonzero frequency interval",
                ));
            }
        }
        Ok(())
    }

    pub fn evaluate(&self) -> Result<Evaluation, DesignerError> {
        self.validate()?;
        let curves = self.curves()?;
        let (low, high) = support(&curves);
        let mut output = Evaluation::default();
        for index in 0..256 {
            let frequency = match index {
                0 => low,
                255 => high,
                _ => (low.ln() + (high.ln() - low.ln()) * index as f64 / 255.0)
                    .exp()
                    .clamp(low, high),
            };
            let omega = TAU * frequency;
            let mut network = CircuitNetwork::new();
            let source = network.add_voltage_source(1, 0);
            let c = &self.crossover;
            // Allocate contiguous nodes: zero-ohm series resistors are wires, not approximations.
            let mut next = 2;
            let mut series_node = |resistance, network: &mut CircuitNetwork| {
                if resistance == 0.0 {
                    1
                } else {
                    let node = next;
                    next += 1;
                    network.add_resistor(1, node, resistance);
                    node
                }
            };
            let woofer_series = series_node(c.woofer_resistance_ohm, &mut network);
            let tweeter_series = series_node(c.tweeter_resistance_ohm, &mut network);
            let woofer_node = next;
            let tweeter_node = next + 1;
            network.add_inductor(woofer_series, woofer_node, c.woofer_inductance_mh * 1e-3);
            network.add_capacitor(woofer_node, 0, c.woofer_capacitance_uf * 1e-6);
            network.add_capacitor(
                tweeter_series,
                tweeter_node,
                c.tweeter_capacitance_uf * 1e-6,
            );
            network.add_inductor(tweeter_node, 0, c.tweeter_inductance_mh * 1e-3);
            for (node, curve) in [(woofer_node, curves[1]), (tweeter_node, curves[3])] {
                let z = curve.interpolate(frequency, CurveKind::Impedance)?;
                network.add_impedance_port(node, 0, ImpedanceFn::new(move |_| z));
            }
            let voltage = Complex64::new(self.reference_voltage_v, 0.0);
            let solution =
                network
                    .solve(omega, &[(source, voltage)])
                    .map_err(|e| DesignerError::Circuit {
                        frequency_hz: frequency,
                        message: e.to_string(),
                    })?;
            let branch =
                |driver: &Driver, curve: &Curve, node| -> Result<Complex64, DesignerError> {
                    let phase = -omega * (driver.delay_ms * 1e-3);
                    if !phase.is_finite() {
                        return Err(invalid("Delay phase exceeds numerical range"));
                    }
                    Ok(curve.interpolate(frequency, CurveKind::Response)?
                        * (solution.node_voltage(node) / voltage)
                        * Complex64::from_polar(if driver.invert { -1.0 } else { 1.0 }, phase))
                };
            let woofer = branch(&self.woofer, curves[0], woofer_node)?;
            let tweeter = branch(&self.tweeter, curves[2], tweeter_node)?;
            let sum = woofer + tweeter;
            let impedance = voltage / solution.vsource_current(source);
            if ![woofer, tweeter, sum, impedance].into_iter().all(finite) {
                return Err(invalid("Circuit produced non-finite results"));
            }
            output.frequencies_hz.push(frequency);
            output.woofer_db.push(db(woofer));
            output.tweeter_db.push(db(tweeter));
            output.sum_db.push(db(sum));
            output.sum_phase_deg.push(sum.arg().to_degrees());
            output.input_impedance_ohm.push(impedance.norm());
            output
                .input_impedance_phase_deg
                .push(impedance.arg().to_degrees());
        }
        Ok(output)
    }
}

fn support(curves: &[&Curve; 4]) -> (f64, f64) {
    (
        curves
            .iter()
            .map(|c| c.samples[0].frequency_hz)
            .fold(0.0, f64::max),
        curves
            .iter()
            .map(|c| c.samples[c.samples.len() - 1].frequency_hz)
            .fold(f64::INFINITY, f64::min),
    )
}

fn finite(value: Complex64) -> bool {
    value.re.is_finite() && value.im.is_finite() && value.norm().is_finite()
}

/// A finite display floor also represents exact complex cancellation.
fn db(value: Complex64) -> f64 {
    20.0 * value.norm().max(1e-15).log10()
}

#[derive(Debug, Clone, Default)]
pub struct Evaluation {
    pub frequencies_hz: Vec<f64>,
    pub woofer_db: Vec<f64>,
    pub tweeter_db: Vec<f64>,
    pub sum_db: Vec<f64>,
    pub sum_phase_deg: Vec<f64>,
    pub input_impedance_ohm: Vec<f64>,
    pub input_impedance_phase_deg: Vec<f64>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn public_interpolation_rejects_invalid_inputs_without_panicking() {
        let empty = Curve {
            source: "empty".into(),
            samples: Vec::new(),
        };
        assert!(empty.interpolate(100.0, CurveKind::Response).is_err());
        let curve = Curve::parse("20 85 0\n20000 85 0", CurveKind::Response, "driver").unwrap();
        for frequency in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, 20001.0] {
            assert!(curve.interpolate(frequency, CurveKind::Response).is_err());
        }
    }

    #[test]
    fn imports_require_unambiguous_ordered_finite_three_column_data() {
        let curve = Curve::parse(
            "\u{feff}# FRD\n\n; note\n* note\n// note\n20 85 0\n20000 84 -30",
            CurveKind::Response,
            "driver",
        )
        .unwrap();
        assert_eq!(curve.samples.len(), 2);
        for text in [
            "20 85\n40 85",
            "20 85 0 extra\n40 85 0",
            "40 85 0\n20 85 0",
            "20 85 0\n20 85 0",
            "0 85 0\n20 85 0",
            "20 NaN 0\n40 85 0",
            "20 85 0",
        ] {
            assert!(
                Curve::parse(text, CurveKind::Response, "bad").is_err(),
                "{text}"
            );
        }
        for magnitude in [0.0, -8.0] {
            assert!(
                Curve::parse(
                    &format!("20 {magnitude} 0\n40 8 0"),
                    CurveKind::Impedance,
                    "bad"
                )
                .is_err()
            );
        }
    }

    #[test]
    fn interpolation_crosses_phase_wrap_without_extrapolation() {
        let curve = Curve::parse("100 0 179\n10000 0 -179", CurveKind::Response, "phase").unwrap();
        let middle = curve.interpolate(1000.0, CurveKind::Response).unwrap();
        assert!((middle.arg().abs() - std::f64::consts::PI).abs() < 1e-12);
        assert!(middle.norm() > 0.999);
        assert!(curve.interpolate(99.0, CurveKind::Response).is_err());
        assert!(curve.interpolate(10001.0, CurveKind::Response).is_err());
    }

    #[test]
    fn default_is_empty_and_demo_roundtrips_with_synthetic_provenance() {
        let empty = Project::default();
        assert!(!empty.synthetic && empty.woofer.response.is_none());
        assert!(empty.validate_draft().is_ok());
        assert!(empty.evaluate().is_err());
        let mut invalid_draft = empty.clone();
        invalid_draft.crossover.woofer_inductance_mh = -1.0;
        assert!(invalid_draft.validate_draft().is_err());
        let mut partial = Project::demo();
        partial.tweeter.response = None;
        let partial: Project =
            serde_json::from_str(&serde_json::to_string(&partial).unwrap()).unwrap();
        assert!(partial.validate_draft().is_ok());
        assert!(partial.evaluate().is_err());
        let json = serde_json::to_string(&Project::demo()).unwrap();
        let project: Project = serde_json::from_str(&json).unwrap();
        assert!(project.synthetic);
        assert_eq!(project.evaluate().unwrap().frequencies_hz.len(), 256);
    }

    #[test]
    fn validation_rechecks_loaded_data_components_and_overlap() {
        let mut project = Project::demo();
        project.woofer.impedance.as_mut().unwrap().samples[0].magnitude = 0.0;
        assert!(project.evaluate().is_err());
        let mut project = Project::demo();
        project.crossover.woofer_inductance_mh = 0.0;
        assert!(project.evaluate().is_err());
        let mut project = Project::demo();
        project.tweeter.delay_ms = -1.0;
        assert!(project.evaluate().is_err());
        let mut project = Project::demo();
        for sample in &mut project.tweeter.response.as_mut().unwrap().samples {
            sample.frequency_hz += 20_000.0;
        }
        assert!(project.evaluate().is_err());
    }

    #[test]
    fn shared_support_and_delay_are_applied_without_extrapolation() {
        let mut project = Project::demo();
        let response = project.tweeter.response.as_mut().unwrap();
        response.samples[0].frequency_hz = 100.0;
        response.samples[1].frequency_hz = 10_000.0;
        let baseline = project.evaluate().unwrap();
        assert_eq!(baseline.frequencies_hz[0], 100.0);
        assert_eq!(*baseline.frequencies_hz.last().unwrap(), 10_000.0);
        project.woofer.delay_ms = 0.5;
        let delayed = project.evaluate().unwrap();
        for (before, after) in baseline.woofer_db.iter().zip(&delayed.woofer_db) {
            assert!((before - after).abs() < 1e-9);
        }
        assert!((baseline.sum_db[127] - delayed.sum_db[127]).abs() > 0.1);
        assert_eq!(baseline.input_impedance_ohm, delayed.input_impedance_ohm);
    }

    #[test]
    fn mna_matches_analytic_two_way_transfer_and_delivered_current() {
        for resistance in [0.0, 1.5] {
            let mut project = Project::demo();
            project.crossover.woofer_resistance_ohm = resistance;
            project.crossover.tweeter_resistance_ohm = resistance;
            let result = project.evaluate().unwrap();
            for index in [0, 127, 255] {
                let jw = Complex64::new(0.0, TAU * result.frequencies_hz[index]);
                let c = &project.crossover;
                let parallel = |a: Complex64, b: Complex64| a * b / (a + b);
                let load = Complex64::new(8.0, 0.0);
                let wl = parallel(load, 1.0 / (jw * c.woofer_capacitance_uf * 1e-6));
                let tl = parallel(load, jw * c.tweeter_inductance_mh * 1e-3);
                let wz = resistance + jw * c.woofer_inductance_mh * 1e-3 + wl;
                let tz = resistance + 1.0 / (jw * c.tweeter_capacitance_uf * 1e-6) + tl;
                let amplitude = 10.0_f64.powf(85.0 / 20.0);
                let woofer = wl / wz * amplitude;
                let tweeter = -tl / tz * amplitude;
                let impedance = parallel(wz, tz);
                assert!((result.woofer_db[index] - db(woofer)).abs() < 1e-9);
                assert!((result.tweeter_db[index] - db(tweeter)).abs() < 1e-9);
                assert!((result.sum_db[index] - db(woofer + tweeter)).abs() < 1e-9);
                assert!((result.input_impedance_ohm[index] - impedance.norm()).abs() < 1e-9);
                assert!(
                    (result.input_impedance_phase_deg[index] - impedance.arg().to_degrees()).abs()
                        < 1e-9
                );
            }
            project.reference_voltage_v = 1.0;
            let at_one_volt = project.evaluate().unwrap();
            for (a, b) in result.sum_db.iter().zip(at_one_volt.sum_db) {
                assert!((a - b).abs() < 1e-9);
            }
        }
    }
}
