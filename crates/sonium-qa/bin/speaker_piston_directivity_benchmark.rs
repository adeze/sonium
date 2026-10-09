//! Baffled circular-piston radiation/directivity benchmark.
//!
//! This validates the analytic benchmark quantities used to check future BEM
//! loudspeaker radiation runs: piston directivity, first null, radiated power,
//! and low-frequency radiation impedance scaling.

use sonium_fem::physics::acoustics::electroacoustics::lumped_extraction::radiation_impedance_piston_lf;
use spec_math::Bessel;
use std::f64::consts::PI;

const RHO: f64 = 1.204;
const C: f64 = 343.0;

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

    assert_close("on-axis directivity", on_axis, 1.0, 1.0e-12)?;
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

    assert_close("first directivity null", directivity, 0.0, 1.0e-10)?;
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
        1.0e-12,
    )?;
    assert_close(
        "radiated power f^2 scaling 100->200",
        p200 / p100,
        4.0,
        1.0e-12,
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
        1.0e-12,
    )?;
    assert_close(
        "radiation reactance low-frequency limit",
        z.im,
        expected_im,
        1.0e-12,
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

fn run() -> Result<(), String> {
    let piston = Piston {
        radius: 0.085,
        velocity: 0.01,
    };

    println!("Baffled piston radiation/directivity benchmark:");
    validate_directivity_shape(piston)?;
    validate_first_null(piston)?;
    validate_low_frequency_power_scaling(piston)?;
    validate_radiation_impedance_limits(piston)?;
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("FAILED: {error}");
        println!("FAILED");
        std::process::exit(1);
    }

    println!("SUCCESS");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn piston_benchmark_quantities_are_stable() {
        run().expect("baffled piston benchmark should satisfy analytic reference checks");
    }
}
