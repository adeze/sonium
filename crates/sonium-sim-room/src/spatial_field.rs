//! Small independent CPU references for RoomSim spatial-field checks.
//!
//! These references are intentionally separate from the FEM runner. They are
//! useful for bounded correlation tests, but are not room solutions: the
//! `provenance` field identifies the analytical model and callers must not
//! promote its samples to measured or fully solved data.

use num_complex::Complex64;
use sonium_xem::{Point3D, constants::AIR_DENSITY_20C};

/// Origin of a spatial-field reference sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialFieldProvenance {
    /// Outgoing free-field pressure from an ideal omnidirectional monopole.
    AnalyticMonopoleGreenFunction,
}

impl SpatialFieldProvenance {
    /// Stable label for reports and focused correlation evidence.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AnalyticMonopoleGreenFunction => "analytic-monopole-green-function.v1",
        }
    }
}

/// Complex free-field pressure samples at a set of spatial probes.
///
/// `pressures[position_idx][frequency_idx]` uses the outgoing-wave convention
/// `exp(-i k r)`. The reference assumes 20 °C air density and carries the
/// source position, volume velocity, sound speed, and provenance alongside
/// the samples so a comparison cannot silently lose its physical assumptions.
#[derive(Debug, Clone)]
pub struct SpatialFieldReference {
    pub source_position: Point3D,
    pub positions: Vec<Point3D>,
    pub frequencies: Vec<f64>,
    pub pressures: Vec<Vec<Complex64>>,
    pub sound_speed: f64,
    pub source_volume_velocity: f64,
    pub provenance: SpatialFieldProvenance,
}

impl SpatialFieldReference {
    /// Evaluate an ideal monopole Green-function field at `positions`.
    pub fn monopole(
        source_position: Point3D,
        positions: Vec<Point3D>,
        frequencies: Vec<f64>,
        sound_speed: f64,
        source_volume_velocity: f64,
    ) -> Result<Self, String> {
        validate_point(source_position, "source position")?;
        if positions.is_empty() {
            return Err("spatial reference requires at least one probe".to_string());
        }
        for (index, &position) in positions.iter().enumerate() {
            validate_point(position, &format!("probe {index}"))?;
        }
        validate_frequency_grid(&frequencies)?;
        if !sound_speed.is_finite() || sound_speed <= 0.0 {
            return Err("sound speed must be finite and positive".to_string());
        }
        if !source_volume_velocity.is_finite() {
            return Err("source volume velocity must be finite".to_string());
        }

        let pressures = positions
            .iter()
            .map(|&position| {
                let distance = distance(source_position, position);
                if !distance.is_finite() || distance <= 0.0 {
                    return Err(
                        "spatial reference probes must be finite and distinct from the source"
                            .to_string(),
                    );
                }
                Ok(frequencies
                    .iter()
                    .map(|&frequency| {
                        let omega = 2.0 * std::f64::consts::PI * frequency;
                        let amplitude = omega * AIR_DENSITY_20C * source_volume_velocity
                            / (4.0 * std::f64::consts::PI * distance);
                        let phase = -omega * distance / sound_speed;
                        Complex64::from_polar(amplitude, phase)
                    })
                    .collect())
            })
            .collect::<Result<Vec<_>, String>>()?;

        Ok(Self {
            source_position,
            positions,
            frequencies,
            pressures,
            sound_speed,
            source_volume_velocity,
            provenance: SpatialFieldProvenance::AnalyticMonopoleGreenFunction,
        })
    }

    /// Check the response-grid and provenance contract before correlation.
    pub fn validate_layout(&self) -> Result<(), String> {
        validate_point(self.source_position, "source position")?;
        if self.positions.is_empty() {
            return Err("spatial reference has no probes".to_string());
        }
        if self.positions.len() != self.pressures.len() {
            return Err(format!(
                "probe/pressure length mismatch: {} probes for {} rows",
                self.positions.len(),
                self.pressures.len()
            ));
        }
        validate_frequency_grid(&self.frequencies)?;
        if !self.sound_speed.is_finite() || self.sound_speed <= 0.0 {
            return Err("sound speed must be finite and positive".to_string());
        }
        if !self.source_volume_velocity.is_finite() {
            return Err("source volume velocity must be finite".to_string());
        }
        for (position_index, (&position, pressures)) in
            self.positions.iter().zip(&self.pressures).enumerate()
        {
            validate_point(position, &format!("probe {position_index}"))?;
            let distance = distance(self.source_position, position);
            if !distance.is_finite() || distance <= 0.0 {
                return Err(format!(
                    "probe {position_index} must be finite and distinct from the source"
                ));
            }
            if pressures.len() != self.frequencies.len() {
                return Err(format!(
                    "pressure/frequency length mismatch for probe {position_index}: {} samples for {} frequencies",
                    pressures.len(),
                    self.frequencies.len()
                ));
            }
            if pressures
                .iter()
                .any(|pressure| !pressure.re.is_finite() || !pressure.im.is_finite())
            {
                return Err(format!("non-finite pressure at probe {position_index}"));
            }
        }
        Ok(())
    }
}

fn validate_point(point: Point3D, label: &str) -> Result<(), String> {
    if [point.x, point.y, point.z].into_iter().all(f64::is_finite) {
        Ok(())
    } else {
        Err(format!("{label} must contain finite coordinates"))
    }
}

fn validate_frequency_grid(frequencies: &[f64]) -> Result<(), String> {
    if frequencies.is_empty() {
        return Err("spatial reference requires at least one frequency".to_string());
    }
    if frequencies
        .iter()
        .any(|&frequency| !frequency.is_finite() || frequency <= 0.0)
    {
        return Err("frequency grid must contain finite positive values".to_string());
    }
    if frequencies.windows(2).any(|window| window[1] <= window[0]) {
        return Err("frequency grid must be strictly increasing".to_string());
    }
    Ok(())
}

fn distance(a: Point3D, b: Point3D) -> f64 {
    let dx = a.x - b.x;
    let dy = a.y - b.y;
    let dz = a.z - b.z;
    (dx * dx + dy * dy + dz * dz).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn monopole_reference_records_provenance_and_radial_decay() {
        let reference = SpatialFieldReference::monopole(
            Point3D::new(0.0, 0.0, 0.0),
            vec![Point3D::new(1.0, 0.0, 0.0), Point3D::new(2.0, 0.0, 0.0)],
            vec![200.0, 400.0],
            343.0,
            5.88e-4,
        )
        .expect("reference should be valid");

        reference.validate_layout().expect("layout should be valid");
        assert_eq!(
            reference.provenance,
            SpatialFieldProvenance::AnalyticMonopoleGreenFunction
        );
        assert_eq!(
            reference.provenance.as_str(),
            "analytic-monopole-green-function.v1"
        );
        let near = reference.pressures[0][0].norm();
        let far = reference.pressures[1][0].norm();
        assert!((near / far - 2.0).abs() < 1.0e-12);
    }

    #[test]
    fn monopole_reference_rejects_invalid_grid_and_source_probe() {
        let source = Point3D::new(0.0, 0.0, 0.0);
        let probe = Point3D::new(1.0, 0.0, 0.0);
        for frequencies in [vec![], vec![400.0, 200.0], vec![f64::NAN]] {
            assert!(
                SpatialFieldReference::monopole(source, vec![probe], frequencies, 343.0, 5.88e-4,)
                    .is_err()
            );
        }
        assert!(
            SpatialFieldReference::monopole(source, vec![source], vec![200.0], 343.0, 5.88e-4,)
                .is_err()
        );
    }
}
