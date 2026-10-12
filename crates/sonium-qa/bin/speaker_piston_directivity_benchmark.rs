//! Baffled circular-piston radiation/directivity benchmark.
//!
//! Two complementary legs: analytical self-consistency of the benchmark
//! quantities (piston directivity, first null, radiated power, low-frequency
//! radiation impedance scaling), and a numerical baffled-piston radiation
//! computation via the Rayleigh surface integral with quadrature refinement.
//! The numerical far-field pressure is compared against the independent
//! closed-form `2 J1(u) / u` directivity reference, so this binary exercises
//! real numerical radiation rather than formula evaluation alone.

use num_complex::Complex64;
use sonium_fem::physics::acoustics::electroacoustics::lumped_extraction::radiation_impedance_piston_lf;
use sonium_qa::QaResult;
use spec_math::Bessel;
use std::f64::consts::PI;

const RHO: f64 = 1.204;
const C: f64 = 343.0;

// Named acceptance tolerances for the analytical reference checks.
const ON_AXIS_DIRECTIVITY_TOL: f64 = 1.0e-12;
const FIRST_NULL_TOL: f64 = 1.0e-10;
const POWER_SCALING_TOL: f64 = 1.0e-12;
const RADIATION_IMPEDANCE_TOL: f64 = 1.0e-12;

// Numerical Rayleigh-integral radiation leg. The piston surface is integrated
// with a polar midpoint rule at three refinements; the far-field pressure at
// each probe angle is normalized by the on-axis analytical amplitude so the
// error stays meaningful through directivity nulls.
const RAYLEIGH_FIELD_RADIUS: f64 = 20.0;
const RAYLEIGH_PROBE_ANGLES_DEG: [f64; 7] = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0];
// (radial cells, angular cells) per refinement level, plus an overkill grid
// that serves as the self-convergence reference.
const RAYLEIGH_REFINEMENTS: [(usize, usize); 3] = [(12, 48), (24, 96), (48, 192)];
const RAYLEIGH_OVERKILL_GRID: (usize, usize) = (96, 384);
// Fine-grid agreement with the closed-form 2J1/u directivity, which carries
// finite-R error no quadrature can remove. Measured 3.2e-3/5.3e-3 at ka=3,5;
// the bound keeps about a 2x margin.
const RAYLEIGH_MAX_NORMALIZED_ERROR: f64 = 1e-2;
const RAYLEIGH_MAX_SELF_ERROR: f64 = 1.0e-4;

#[derive(Clone, Copy, Debug)]
struct Piston {
    radius: f64,
    velocity: f64,
}

impl Piston {
    fn area(self) -> f64 {
        PI * self.radius * self.radius
    }

    fn directivity(self, ka: f64, theta_rad: f64) -> f64 {
        let x = ka * theta_rad.sin();
        if x.abs() < 1.0e-8 {
            1.0 - x * x / 8.0
        } else {
            2.0 * x.bessel_jv(1.0) / x
        }
    }

    fn radiated_power_lf(self, frequency: f64) -> f64 {
        let omega = 2.0 * PI * frequency;
        let z_acoustic = radiation_impedance_piston_lf(self.radius, omega, RHO, C);
        0.5 * z_acoustic.re * (self.velocity * self.area()).powi(2)
    }
}

fn assert_close(label: &str, actual: f64, expected: f64, tolerance: f64) -> Result<(), String> {
    let error = (actual - expected).abs();
    if error > tolerance {
        return Err(format!(
            "{label}: expected {expected:.9}, got {actual:.9} (abs error {error:.3e})"
        ));
    }
    Ok(())
}

fn validate_directivity_shape(piston: Piston) -> Result<(), String> {
    let ka = 3.0;
    let on_axis = piston.directivity(ka, 0.0);
    let at_30 = piston.directivity(ka, 30.0_f64.to_radians()).abs();
    let at_60 = piston.directivity(ka, 60.0_f64.to_radians()).abs();
    let at_90 = piston.directivity(ka, 90.0_f64.to_radians()).abs();

    assert_close("on-axis directivity", on_axis, 1.0, ON_AXIS_DIRECTIVITY_TOL)?;
    if !(at_30 > at_60 && at_60 > at_90) {
        return Err(format!(
            "piston directivity should narrow monotonically over 0..90 deg for ka=3: D30={at_30:.6}, D60={at_60:.6}, D90={at_90:.6}"
        ));
    }

    println!(
        "  Directivity ka={ka:.1}: D0={on_axis:.6}, |D30|={at_30:.6}, |D60|={at_60:.6}, |D90|={at_90:.6}"
    );
    Ok(())
}

fn validate_first_null(piston: Piston) -> Result<(), String> {
    let first_j1_zero: f64 = 3.8317059702;
    let ka: f64 = 5.0;
    let theta_null = (first_j1_zero / ka).asin();
    let directivity = piston.directivity(ka, theta_null);

    assert_close("first directivity null", directivity, 0.0, FIRST_NULL_TOL)?;
    println!(
        "  First null ka={ka:.1}: theta={:.3} deg, D={directivity:.3e}",
        theta_null.to_degrees()
    );
    Ok(())
}

fn validate_low_frequency_power_scaling(piston: Piston) -> Result<(), String> {
    let p50 = piston.radiated_power_lf(50.0);
    let p100 = piston.radiated_power_lf(100.0);
    let p200 = piston.radiated_power_lf(200.0);

    assert!(p50 > 0.0 && p100 > 0.0 && p200 > 0.0);
    assert_close(
        "radiated power f^2 scaling 50->100",
        p100 / p50,
        4.0,
        POWER_SCALING_TOL,
    )?;
    assert_close(
        "radiated power f^2 scaling 100->200",
        p200 / p100,
        4.0,
        POWER_SCALING_TOL,
    )?;

    println!("  LF radiated power: P50={p50:.6e} W, P100={p100:.6e} W, P200={p200:.6e} W");
    Ok(())
}

fn validate_radiation_impedance_limits(piston: Piston) -> Result<(), String> {
    let omega = 2.0 * PI * 100.0;
    let ka = omega / C * piston.radius;
    let z = radiation_impedance_piston_lf(piston.radius, omega, RHO, C);
    let area = piston.area();
    let z0 = RHO * C / area;
    let expected_re = z0 * ka * ka / 2.0;
    let expected_im = z0 * 8.0 * ka / (3.0 * PI);

    assert_close(
        "radiation resistance low-frequency limit",
        z.re,
        expected_re,
        RADIATION_IMPEDANCE_TOL,
    )?;
    assert_close(
        "radiation reactance low-frequency limit",
        z.im,
        expected_im,
        RADIATION_IMPEDANCE_TOL,
    )?;
    assert!(
        z.im > z.re,
        "low-frequency radiation reactance should dominate"
    );

    println!(
        "  Radiation impedance ka={ka:.4}: Zac={:.6}+j{:.6} Pa s/m^3",
        z.re, z.im
    );
    Ok(())
}

/// Closed-form far-field pressure of a baffled piston with uniform velocity.
///
/// With `e^{iwt}` time dependence the Rayleigh integral reduces in the far
/// field to `p = i rho c k v A e^{-ikR} / (2 pi R) * D(theta)` with the
/// `2 J1(u) / u` directivity factor, `u = k a sin(theta)`.
fn analytical_far_field_pressure(
    piston: Piston,
    wavenumber: f64,
    field_radius: f64,
    theta_rad: f64,
) -> Complex64 {
    let ka = wavenumber * piston.radius;
    let amplitude =
        RHO * C * wavenumber * piston.velocity * piston.area() / (2.0 * PI * field_radius);
    let phase = wavenumber * field_radius;
    let propagator = Complex64::new(phase.cos(), -phase.sin());
    Complex64::new(0.0, amplitude) * propagator * piston.directivity(ka, theta_rad)
}

/// Numerical baffled-piston pressure from the Rayleigh surface integral.
///
/// Integrates `(i w rho v / 2 pi) e^{-ikd} / d` over the piston disk with a
/// polar midpoint rule (`nr` radial by `nphi` angular cells). This is a genuine
/// numerical radiation computation: no closed-form directivity enters.
fn rayleigh_integral_pressure(
    piston: Piston,
    wavenumber: f64,
    field_radius: f64,
    theta_rad: f64,
    nr: usize,
    nphi: usize,
) -> Complex64 {
    let omega = wavenumber * C;
    let prefactor = Complex64::new(0.0, omega * RHO * piston.velocity / (2.0 * PI));
    let field = [
        field_radius * theta_rad.sin(),
        0.0,
        field_radius * theta_rad.cos(),
    ];
    let dr = piston.radius / nr as f64;
    let dphi = 2.0 * PI / nphi as f64;

    let mut integral = Complex64::new(0.0, 0.0);
    for i in 0..nr {
        let r = (i as f64 + 0.5) * dr;
        for j in 0..nphi {
            let phi = (j as f64 + 0.5) * dphi;
            let dx = field[0] - r * phi.cos();
            let dy = field[1] - r * phi.sin();
            let dz = field[2];
            let distance = (dx * dx + dy * dy + dz * dz).sqrt();
            let phase = wavenumber * distance;
            let green = Complex64::new(phase.cos(), -phase.sin()) / distance;
            integral += green * (r * dr * dphi);
        }
    }
    prefactor * integral
}

/// Numerical far-field pressures at every probe angle for one quadrature grid.
fn rayleigh_probe_pressures(
    piston: Piston,
    wavenumber: f64,
    nr: usize,
    nphi: usize,
) -> Result<Vec<Complex64>, String> {
    let mut pressures = Vec::with_capacity(RAYLEIGH_PROBE_ANGLES_DEG.len());
    for angle_deg in RAYLEIGH_PROBE_ANGLES_DEG {
        let theta = angle_deg.to_radians();
        let pressure =
            rayleigh_integral_pressure(piston, wavenumber, RAYLEIGH_FIELD_RADIUS, theta, nr, nphi);
        if !pressure.re.is_finite() || !pressure.im.is_finite() {
            return Err(format!(
                "non-finite Rayleigh pressure at {angle_deg} deg for ka={:.1}",
                wavenumber * piston.radius
            ));
        }
        pressures.push(pressure);
    }
    Ok(pressures)
}

/// Maximum normalized difference between two probe sweeps.
fn max_probe_difference(
    numerical: &[Complex64],
    reference: &[Complex64],
    normalization: f64,
) -> f64 {
    numerical
        .iter()
        .zip(reference.iter())
        .map(|(a, b)| (a - b).norm() / normalization)
        .fold(0.0, f64::max)
}

struct RayleighLegMetrics {
    coarse_self_error: f64,
    fine_self_error: f64,
    fine_reference_error: f64,
}

/// Numerical piston radiation leg: Rayleigh-integral far field with refinement.
///
/// Quadrature convergence is measured against an overkill grid (self-error
/// must contract across the refinement ladder), while absolute accuracy is
/// measured against the independent closed-form far-field reference. The two
/// comparisons stay separate because the closed form itself carries a finite
/// field-radius approximation that no quadrature refinement can remove.
fn validate_numerical_rayleigh_radiation(
    piston: Piston,
    ka: f64,
) -> Result<RayleighLegMetrics, String> {
    let wavenumber = ka / piston.radius;
    let normalization =
        analytical_far_field_pressure(piston, wavenumber, RAYLEIGH_FIELD_RADIUS, 0.0).norm();
    if !normalization.is_finite() || normalization <= 0.0 {
        return Err(format!(
            "non-positive on-axis reference amplitude {normalization:.3e} for ka={ka:.1}"
        ));
    }
    let overkill = rayleigh_probe_pressures(
        piston,
        wavenumber,
        RAYLEIGH_OVERKILL_GRID.0,
        RAYLEIGH_OVERKILL_GRID.1,
    )?;

    let mut self_errors = Vec::with_capacity(RAYLEIGH_REFINEMENTS.len());
    let mut fine_pressures = Vec::new();
    for (nr, nphi) in RAYLEIGH_REFINEMENTS {
        let pressures = rayleigh_probe_pressures(piston, wavenumber, nr, nphi)?;
        let self_error = max_probe_difference(&pressures, &overkill, normalization);
        println!("  Rayleigh ka={ka:.1} grid {nr}x{nphi}: self-error {self_error:.4e}");
        self_errors.push(self_error);
        fine_pressures = pressures;
    }
    for level in 1..self_errors.len() {
        if self_errors[level] >= self_errors[level - 1] {
            return Err(format!(
                "Rayleigh self-error must contract with refinement for ka={ka:.1}: {self_errors:?}"
            ));
        }
    }
    let fine_self = *self_errors.last().expect("at least one refinement level");
    if fine_self > RAYLEIGH_MAX_SELF_ERROR {
        return Err(format!(
            "Rayleigh fine-grid self-error {fine_self:.4e} exceeds {RAYLEIGH_MAX_SELF_ERROR:.1e} for ka={ka:.1}"
        ));
    }

    let reference: Vec<Complex64> = RAYLEIGH_PROBE_ANGLES_DEG
        .iter()
        .map(|angle_deg| {
            analytical_far_field_pressure(
                piston,
                wavenumber,
                RAYLEIGH_FIELD_RADIUS,
                angle_deg.to_radians(),
            )
        })
        .collect();
    let fine_reference_error = max_probe_difference(&fine_pressures, &reference, normalization);
    println!("  Rayleigh ka={ka:.1} fine grid vs closed form: {fine_reference_error:.4e}");
    if fine_reference_error > RAYLEIGH_MAX_NORMALIZED_ERROR {
        return Err(format!(
            "Rayleigh fine-grid error {fine_reference_error:.4e} exceeds {RAYLEIGH_MAX_NORMALIZED_ERROR:.1e} for ka={ka:.1}"
        ));
    }
    Ok(RayleighLegMetrics {
        coarse_self_error: self_errors[0],
        fine_self_error: fine_self,
        fine_reference_error,
    })
}

struct BenchmarkMetrics {
    rayleigh_coarse_self_ka3: f64,
    rayleigh_fine_self_ka3: f64,
    rayleigh_fine_error_ka3: f64,
    rayleigh_coarse_self_ka5: f64,
    rayleigh_fine_self_ka5: f64,
    rayleigh_fine_error_ka5: f64,
}

fn run() -> Result<BenchmarkMetrics, String> {
    let piston = Piston {
        radius: 0.085,
        velocity: 0.01,
    };

    println!("Baffled piston radiation/directivity benchmark:");
    validate_directivity_shape(piston)?;
    validate_first_null(piston)?;
    validate_low_frequency_power_scaling(piston)?;
    validate_radiation_impedance_limits(piston)?;
    println!("Numerical Rayleigh-integral radiation:");
    let ka3 = validate_numerical_rayleigh_radiation(piston, 3.0)?;
    let ka5 = validate_numerical_rayleigh_radiation(piston, 5.0)?;
    Ok(BenchmarkMetrics {
        rayleigh_coarse_self_ka3: ka3.coarse_self_error,
        rayleigh_fine_self_ka3: ka3.fine_self_error,
        rayleigh_fine_error_ka3: ka3.fine_reference_error,
        rayleigh_coarse_self_ka5: ka5.coarse_self_error,
        rayleigh_fine_self_ka5: ka5.fine_self_error,
        rayleigh_fine_error_ka5: ka5.fine_reference_error,
    })
}

fn main() {
    match run() {
        Err(error) => {
            if let Ok(record) =
                QaResult::failed("speaker.piston-directivity-benchmark.v1", error.as_str())
                    .to_json_line()
            {
                println!("QA_RESULT: {record}");
            }
            eprintln!("FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
        Ok(metrics) => {
            let record = QaResult::passed("speaker.piston-directivity-benchmark.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("rayleigh_coarse_self_ka3", metrics.rayleigh_coarse_self_ka3)
                .metric("rayleigh_fine_self_ka3", metrics.rayleigh_fine_self_ka3)
                .metric("rayleigh_fine_error_ka3", metrics.rayleigh_fine_error_ka3)
                .metric("rayleigh_coarse_self_ka5", metrics.rayleigh_coarse_self_ka5)
                .metric("rayleigh_fine_self_ka5", metrics.rayleigh_fine_self_ka5)
                .metric("rayleigh_fine_error_ka5", metrics.rayleigh_fine_error_ka5)
                .maximum("rayleigh_fine_self_ka3", RAYLEIGH_MAX_SELF_ERROR)
                .maximum("rayleigh_fine_self_ka5", RAYLEIGH_MAX_SELF_ERROR)
                .maximum("rayleigh_fine_error_ka3", RAYLEIGH_MAX_NORMALIZED_ERROR)
                .maximum("rayleigh_fine_error_ka5", RAYLEIGH_MAX_NORMALIZED_ERROR)
                .seed("baffled-piston:rayleigh-midpoint:12/24/48:ka3-ka5");
            match record.to_json_line() {
                Ok(json) => println!("QA_RESULT: {json}"),
                Err(error) => {
                    eprintln!("FAILED: {error}");
                    println!("FAILED");
                    std::process::exit(1);
                }
            }
            println!("SUCCESS");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piston_benchmark_quantities_are_stable() {
        run().expect("baffled piston benchmark should satisfy analytic reference checks");
    }

    #[test]
    fn rayleigh_far_field_matches_closed_form_directivity() {
        let piston = Piston {
            radius: 0.085,
            velocity: 0.01,
        };
        let leg = validate_numerical_rayleigh_radiation(piston, 3.0)
            .expect("Rayleigh leg should evaluate");
        assert!(
            leg.fine_reference_error < RAYLEIGH_MAX_NORMALIZED_ERROR,
            "fine Rayleigh error {:.4e} should stay below the release bound",
            leg.fine_reference_error
        );
        assert!(
            leg.fine_self_error < leg.coarse_self_error,
            "Rayleigh self-error should contract with refinement"
        );
    }
}
