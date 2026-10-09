//! Loudspeaker small-signal vibro-acoustic validation.
//!
//! This benchmark checks the coupled low-frequency chain used by the speaker
//! stack: electrodynamic drive -> mechanical mobility -> baffled-piston
//! radiation load. It is intentionally cheap and analytic so it can live in
//! deterministic QA tiers.

use num_complex::Complex64;
use sonium_fem::physics::acoustics::electroacoustics::lumped_extraction::{
    compute_thiele_small, radiation_impedance_piston_lf,
};
use sonium_fem::physics::acoustics::electroacoustics::transducer::ElectrodynamicCoupling;
use std::f64::consts::PI;

const RHO: f64 = 1.204;
const C: f64 = 343.0;

#[derive(Clone, Copy, Debug)]
struct DriverFixture {
    mms: f64,
    kms: f64,
    rms: f64,
    bl: f64,
    sd: f64,
    re: f64,
    le: f64,
    radius: f64,
}

impl DriverFixture {
    fn reference_midwoofer() -> Self {
        let radius: f64 = 0.065;
        let sd = PI * radius * radius;
        Self {
            mms: 14.0e-3,
            kms: 2600.0,
            rms: 0.85,
            bl: 7.5,
            sd,
            re: 5.7,
            le: 0.42e-3,
            radius,
        }
    }

    fn omega_s(&self) -> f64 {
        (self.kms / self.mms).sqrt()
    }

    fn mechanical_impedance(&self, omega: f64, include_radiation: bool) -> Complex64 {
        let structural = Complex64::new(self.rms, omega * self.mms - self.kms / omega);
        if include_radiation {
            structural + self.mechanical_radiation_impedance(omega)
        } else {
            structural
        }
    }

    fn mechanical_radiation_impedance(&self, omega: f64) -> Complex64 {
        radiation_impedance_piston_lf(self.radius, omega, RHO, C) * self.sd * self.sd
    }

    fn electrodynamic_coupling(&self) -> ElectrodynamicCoupling {
        ElectrodynamicCoupling {
            resistance: self.re,
            inductance: self.le,
            bl: self.bl,
            coil_dofs: vec![(0, 1.0)],
        }
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

fn assert_in_range(label: &str, actual: f64, min: f64, max: f64) -> Result<(), String> {
    if actual < min || actual > max {
        return Err(format!(
            "{label}: expected {min:.6}..{max:.6}, got {actual:.6}"
        ));
    }
    Ok(())
}

fn validate_thiele_small_consistency(driver: DriverFixture) -> Result<(), String> {
    let ts = compute_thiele_small(
        driver.mms, driver.kms, driver.rms, driver.bl, driver.sd, driver.re, driver.le, RHO, C,
    );

    let fs_expected = driver.omega_s() / (2.0 * PI);
    assert_close("fs", ts.fs, fs_expected, 1.0e-12)?;
    assert_close("Mms", ts.mms, driver.mms, 1.0e-15)?;
    assert_close("Kms", ts.kms, driver.kms, 1.0e-12)?;
    assert_close("Rms", ts.rms, driver.rms, 1.0e-15)?;
    assert_close("BL", ts.bl, driver.bl, 1.0e-15)?;
    assert_close("Sd", ts.sd, driver.sd, 1.0e-15)?;
    assert_close("Re", ts.re, driver.re, 1.0e-15)?;
    assert_close("Le", ts.le, driver.le, 1.0e-15)?;

    let qts_identity = ts.qms * ts.qes / (ts.qms + ts.qes);
    assert_close("Qts identity", ts.qts, qts_identity, 1.0e-12)?;
    assert_in_range("midwoofer fs", ts.fs, 60.0, 75.0)?;
    assert_in_range("midwoofer Qts", ts.qts, 0.30, 0.65)?;
    assert_in_range("midwoofer Vas litres", ts.vas * 1000.0, 5.0, 20.0)?;

    println!(
        "  Thiele-Small: fs={:.3} Hz, Qms={:.3}, Qes={:.3}, Qts={:.3}, Vas={:.2} L",
        ts.fs,
        ts.qms,
        ts.qes,
        ts.qts,
        ts.vas * 1000.0
    );
    Ok(())
}

fn validate_motional_impedance_peak(driver: DriverFixture) -> Result<(), String> {
    let coupling = driver.electrodynamic_coupling();
    let omega_s = driver.omega_s();
    let z_mech = driver.mechanical_impedance(omega_s, false);
    let z_total = coupling.total_impedance(omega_s, z_mech);
    let z_blocked = coupling.electrical_impedance(omega_s);
    let expected_peak_re = driver.re + driver.bl * driver.bl / driver.rms;

    assert_close("mechanical resonance reactance", z_mech.im, 0.0, 1.0e-12)?;
    assert_close(
        "resonance impedance real part",
        z_total.re,
        expected_peak_re,
        1.0e-12,
    )?;
    assert!(z_total.re > z_blocked.re + 40.0);

    println!(
        "  Motional impedance: blocked={:.3}+j{:.3} ohm, peak={:.3}+j{:.3} ohm",
        z_blocked.re, z_blocked.im, z_total.re, z_total.im
    );
    Ok(())
}

fn validate_radiation_load_passivity(driver: DriverFixture) -> Result<(), String> {
    let f = 100.0;
    let omega = 2.0 * PI * f;
    let ka = omega / C * driver.radius;
    let z_acoustic = radiation_impedance_piston_lf(driver.radius, omega, RHO, C);
    let z_mechanical = driver.mechanical_radiation_impedance(omega);

    assert!(
        ka < 0.15,
        "fixture must stay in the low-frequency piston range"
    );
    assert!(z_acoustic.re > 0.0, "radiation resistance must be passive");
    assert!(
        z_acoustic.im > 0.0,
        "radiation reactance should be mass-like"
    );
    assert!(z_acoustic.im > z_acoustic.re);

    let expected_mechanical = z_acoustic * driver.sd * driver.sd;
    assert_close(
        "mechanical radiation resistance",
        z_mechanical.re,
        expected_mechanical.re,
        1.0e-15,
    )?;
    assert_close(
        "mechanical radiation reactance",
        z_mechanical.im,
        expected_mechanical.im,
        1.0e-15,
    )?;

    println!(
        "  Radiation load @ {f:.0} Hz: ka={ka:.4}, Zac={:.3}+j{:.3} Pa s/m^3, Zm={:.6}+j{:.6} kg/s",
        z_acoustic.re, z_acoustic.im, z_mechanical.re, z_mechanical.im
    );
    Ok(())
}

fn validate_power_consistency(driver: DriverFixture) -> Result<(), String> {
    let f = 100.0;
    let omega = 2.0 * PI * f;
    let coupling = driver.electrodynamic_coupling();
    let z_mech = driver.mechanical_impedance(omega, true);
    let voltage = Complex64::new(2.83, 0.0);
    let (_current, velocity, displacement) = coupling.solve_frequency(voltage, omega, z_mech);
    let z_rad = driver.mechanical_radiation_impedance(omega);

    let radiated_power = 0.5 * z_rad.re * velocity.norm_sqr();
    let dissipated_mechanical_power = 0.5 * driver.rms * velocity.norm_sqr();

    assert!(velocity.norm() > 0.0);
    assert_close(
        "displacement velocity relation",
        displacement.norm() * omega,
        velocity.norm(),
        1.0e-12,
    )?;
    assert!(radiated_power > 0.0, "radiated power must be positive");
    assert!(dissipated_mechanical_power > radiated_power);

    println!(
        "  Driven response @ {f:.0} Hz: |v|={:.6} m/s, |x|={:.9} m, Prad={:.6e} W",
        velocity.norm(),
        displacement.norm(),
        radiated_power
    );
    Ok(())
}

fn run() -> Result<(), String> {
    let driver = DriverFixture::reference_midwoofer();

    println!("Loudspeaker vibro-acoustic validation:");
    validate_thiele_small_consistency(driver)?;
    validate_motional_impedance_peak(driver)?;
    validate_radiation_load_passivity(driver)?;
    validate_power_consistency(driver)?;
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
    fn validates_thiele_small_fixture() {
        validate_thiele_small_consistency(DriverFixture::reference_midwoofer())
            .expect("Thiele-Small fixture should be internally consistent");
    }

    #[test]
    fn validates_motional_impedance_peak() {
        validate_motional_impedance_peak(DriverFixture::reference_midwoofer())
            .expect("motional impedance peak should match closed form");
    }

    #[test]
    fn validates_radiation_load_passivity() {
        validate_radiation_load_passivity(DriverFixture::reference_midwoofer())
            .expect("radiation impedance should be passive and mass-like");
    }

    #[test]
    fn validates_power_consistency() {
        validate_power_consistency(DriverFixture::reference_midwoofer())
            .expect("driven vibro-acoustic response should conserve signs and units");
    }
}
