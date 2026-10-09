//! Porous/material impedance QA for BEM boundary admittance inputs.

use sonium_bem::materials::{AmbientMedium, PorousLayer};

// This QA binary validates the legacy convenience API on valid inputs,
// so it is a sanctioned user of the deprecated wrappers.
#[allow(deprecated)]
fn validate_porous_layer_absorption_trend() -> Result<(), String> {
    let medium = AmbientMedium::default();
    let layer = PorousLayer::miki(0.08, 12_000.0);
    let low = layer.absorption_coefficient(125.0, medium);
    let mid = layer.absorption_coefficient(500.0, medium);
    let high = layer.absorption_coefficient(2_000.0, medium);

    println!("Miki porous layer absorption: 125 Hz={low:.3}, 500 Hz={mid:.3}, 2 kHz={high:.3}");

    if !(0.0..=1.0).contains(&low) || !(0.0..=1.0).contains(&mid) || !(0.0..=1.0).contains(&high) {
        return Err("absorption coefficient left physical [0, 1] range".into());
    }
    if mid <= low {
        return Err(format!(
            "expected porous absorption to rise from low to mid band, got low={low:.3}, mid={mid:.3}"
        ));
    }
    if high < mid * 0.80 {
        return Err(format!(
            "expected high-band absorption to remain near mid-band value, got mid={mid:.3}, high={high:.3}"
        ));
    }
    Ok(())
}

// Same sanctioned legacy-API use as above.
#[allow(deprecated)]
fn validate_admittance_is_passive_and_frequency_dependent() -> Result<(), String> {
    let medium = AmbientMedium::new(1.21, 343.0);
    let layer = PorousLayer::delany_bazley(0.05, 16_000.0);
    let y_low = layer.normalized_admittance(500.0, medium);
    let y_high = layer.normalized_admittance(4_000.0, medium);

    println!(
        "Delany-Bazley normalized admittance: 500 Hz={:.4}+{:.4}i, 4 kHz={:.4}+{:.4}i",
        y_low.re, y_low.im, y_high.re, y_high.im
    );

    if y_low.re < 0.0 || y_high.re < 0.0 {
        return Err("porous admittance must have non-negative conductance".into());
    }
    if (y_high - y_low).norm() < 0.05 {
        return Err("porous admittance should be meaningfully frequency dependent".into());
    }
    Ok(())
}

fn run() -> Result<(), String> {
    validate_porous_layer_absorption_trend()?;
    validate_admittance_is_passive_and_frequency_dependent()?;
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
    fn validates_material_impedance_models() {
        run().expect("porous material impedance models should remain passive and physical");
    }
}
