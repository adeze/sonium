//! QA: FW-H and Kirchhoff reconstruction of rigid-sphere scattering.
//!
//! Loads exact Mie-series total pressure (rigid surface, zero normal
//! velocity) onto sphere patches and reconstructs the scattered field
//! at ka = 1, 5, 10 through the stationary FW-H evaluator (loading
//! only) and the zero-Mach convected Kirchhoff evaluator (p, grad p
//! with the rigid dp/dr = 0). Both must reproduce the Mie scattered
//! field, which exercises kernel quadrature far beyond the unit-test
//! regime.
//!
//! Convention bridge: `sonium-wave` scattering series use `exp(+ikr)`
//! kernels (outgoing under `exp(-iwt)`, matching `sonium-bem`), while
//! the FEM analogy evaluators use `exp(-ikr)` (outgoing under the
//! documented `exp(+iwt)`). The Mie series is term-by-term the complex
//! conjugate of the physical `exp(+iwt)` solution (real Legendre and
//! Bessel-J factors, `h1 = conj(h2)`), so conjugating both the surface
//! data and the expected field is exact up to series truncation.

use num_complex::Complex64;
use sonium_fem::physics::acoustics::aeroacoustics::{
    FwhKirchhoffPatch, FwhSurfacePatch, stationary_fwh_pressure,
    stationary_kirchhoff_pressure_convected,
};
use sonium_wave::analytical::{sphere_scattering_3d, suggested_sphere_terms_for_grid};
use std::f64::consts::PI;

#[derive(Debug)]
struct QaCheck {
    method: &'static str,
    name: String,
    metric: f64,
    maximum: f64,
}

impl QaCheck {
    fn passed(&self) -> bool {
        self.metric.is_finite() && self.metric <= self.maximum
    }
}

/// Total Mie pressure at (r, theta) for a rigid sphere of radius a,
/// conjugated into the `exp(+iwt)` frame of the analogy evaluators.
fn mie_total(k: f64, radius: f64, terms: usize, r: f64, theta: f64) -> Complex64 {
    sphere_scattering_3d(k, radius, terms, vec![r], vec![theta]).pressure[0].conj()
}

/// Incident unit plane wave along +z in the `exp(+iwt)` frame.
fn incident(k: f64, position: [f64; 3]) -> Complex64 {
    let phase = -k * position[2];
    Complex64::new(phase.cos(), phase.sin())
}

/// Tangential surface gradient of the Mie total via central differences.
/// The radial derivative vanishes by the rigid boundary condition.
fn mie_surface_gradient(k: f64, radius: f64, terms: usize, theta: f64, phi: f64) -> [Complex64; 3] {
    let step = 1e-4;
    let forward = mie_total(k, radius, terms, radius, (theta + step).min(PI));
    let backward = mie_total(k, radius, terms, radius, (theta - step).max(0.0));
    let dpolar = (forward - backward) / (2.0 * step * radius);
    // Polar tangent direction at (theta, phi).
    let tangent = [
        theta.cos() * phi.cos(),
        theta.cos() * phi.sin(),
        -theta.sin(),
    ];
    [
        dpolar * tangent[0],
        dpolar * tangent[1],
        dpolar * tangent[2],
    ]
}

fn run_case(ka: f64) -> Vec<QaCheck> {
    let radius = 1.0;
    let sound_speed = 343.0;
    let density = 1.2;
    let k = ka / radius;
    let omega = k * sound_speed;
    // Observers sit at 8a: the series needs n_max past k*r there.
    let terms = suggested_sphere_terms_for_grid(ka, k * 8.0 * radius);
    // ~16 points per wavelength around the equator.
    let n_phi = (16.0 * ka) as usize + 32;
    let n_mu = (8.0 * ka) as usize + 16;
    let area = 4.0 * PI * radius * radius / (n_mu * n_phi) as f64;

    let mut fwh_patches = Vec::with_capacity(n_mu * n_phi);
    let mut kirchhoff_patches = Vec::with_capacity(n_mu * n_phi);
    for mu_index in 0..n_mu {
        let mu = -1.0 + 2.0 * (mu_index as f64 + 0.5) / n_mu as f64;
        let theta = mu.acos();
        let radial = (1.0 - mu * mu).sqrt();
        for phi_index in 0..n_phi {
            let phi = 2.0 * PI * (phi_index as f64 + 0.5) / n_phi as f64;
            let normal = [radial * phi.cos(), radial * phi.sin(), mu];
            let position = normal.map(|component| radius * component);
            let pressure = mie_total(k, radius, terms, radius, theta);
            let gradient = mie_surface_gradient(k, radius, terms, theta, phi);
            fwh_patches.push(
                FwhSurfacePatch::new(position, normal, area, pressure, Complex64::new(0.0, 0.0))
                    .unwrap(),
            );
            kirchhoff_patches
                .push(FwhKirchhoffPatch::new(position, normal, area, pressure, gradient).unwrap());
        }
    }

    let mut checks = Vec::new();
    for theta in [0.0, PI / 2.0, PI] {
        let observer = [8.0 * radius * theta.sin(), 0.0, 8.0 * radius * theta.cos()];
        let expected = mie_total(k, radius, terms, 8.0 * radius, theta) - incident(k, observer);
        let fwh = stationary_fwh_pressure(&fwh_patches, observer, omega, density, sound_speed)
            .unwrap()
            .total;
        let kirchhoff = stationary_kirchhoff_pressure_convected(
            &kirchhoff_patches,
            observer,
            omega,
            sound_speed,
            [0.0, 0.0, 0.0],
        )
        .unwrap()
        .total;
        let scale = expected.norm().max(1e-30);
        checks.push(QaCheck {
            method: "FW-H Mie",
            name: format!("ka={ka} theta={theta:.2} scattered reconstruction"),
            metric: (fwh - expected).norm() / scale,
            maximum: 5e-2,
        });
        checks.push(QaCheck {
            method: "Kirchhoff",
            name: format!("ka={ka} theta={theta:.2} scattered reconstruction"),
            metric: (kirchhoff - expected).norm() / scale,
            maximum: 5e-2,
        });
    }
    checks
}

fn main() {
    let mut checks = Vec::new();
    for ka in [1.0, 5.0, 10.0] {
        checks.extend(run_case(ka));
    }
    println!("FW-H/Kirchhoff Mie-scattering validation");
    println!(
        "{:<12} {:<48} {:>12} {:>12}  result",
        "method", "check", "metric", "maximum"
    );
    for result in &checks {
        println!(
            "{:<12} {:<48} {:>12.4e} {:>12.4e}  {}",
            result.method,
            result.name,
            result.metric,
            result.maximum,
            if result.passed() { "pass" } else { "fail" }
        );
    }
    let failures: Vec<_> = checks.iter().filter(|result| !result.passed()).collect();
    if failures.is_empty() {
        println!("SUCCESS");
    } else {
        eprintln!("FAILED: {} of {} QA checks", failures.len(), checks.len());
        std::process::exit(1);
    }
}
