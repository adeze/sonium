//! Audit `reviews/audit-sonium-wave-20260921.md`: oracle independence.
//!
//! Comparing a solver against `sonium-wave` is weak when both sides share the
//! same special-function helper. These goldens break that loop: the
//! rigid-sphere Mie coefficients `a_n` and rigid-cylinder coefficients `b_n`
//! were derived by `fixtures/generate_scattering_golden.py` in Python
//! `decimal` at 110 digits (Miller downward first-kind, upward second-kind;
//! only elementary sin/cos/J0/J1/Y0/Y1 seeds from the independent `mpmath`
//! library). No Sonium code participated.
//!
//! The test reconstructs probe pressures from the golden coefficients with
//! the documented series algebra and compares them against the public
//! `sphere_scattering_3d` / `cylinder_scattering_2d` references. The
//! remaining shared inputs (`j_n`/`y_n`/`P_n` tables, `spec_math` Hankel
//! values) each carry their own independent goldens
//! (`tests/special_golden.rs`); what this file pins is the
//! truncation-sensitive scattering physics.
//!
//! Tolerance is `1e-12` relative: both sides evaluate the same algebra in
//! binary64, so only summation-order roundoff separates them.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_wave::analytical::{
    cylinder_scattering_2d, legendre_p, sphere_scattering_3d, spherical_bessel_j,
    spherical_bessel_y, suggested_cylinder_terms, suggested_sphere_terms,
};
use spec_math::Bessel;
use std::f64::consts::PI;

const FIXTURE: &str = include_str!("../fixtures/scattering-golden-v1.json");
const RELATIVE_TOLERANCE: f64 = 1e-12;

#[derive(Debug, Deserialize)]
struct Fixture {
    schema: String,
    case_id: String,
    precision_digits: u32,
    sphere: Vec<SeriesCase>,
    cylinder: Vec<SeriesCase>,
}

#[derive(Debug, Deserialize)]
struct SeriesCase {
    ka: String,
    num_terms: usize,
    coefficients: Vec<ComplexPair>,
}

#[derive(Debug, Deserialize)]
struct ComplexPair {
    re: String,
    im: String,
}

impl ComplexPair {
    fn value(&self) -> Complex64 {
        Complex64::new(
            self.re.parse::<f64>().expect("golden re is a decimal"),
            self.im.parse::<f64>().expect("golden im is a decimal"),
        )
    }
}

fn assert_pressure_close(actual: &[Complex64], expected: &[Complex64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    let scale = expected
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max)
        .max(1e-30);
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            (actual - expected).norm() <= RELATIVE_TOLERANCE * scale,
            "{label}[{index}] = {actual:.6e}, independent golden gives {expected:.6e}"
        );
    }
}

#[test]
fn sphere_scattering_matches_independent_mie_golden() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid golden fixture");
    assert_eq!(fixture.schema, "sonium-wave.scattering-golden.v1");
    assert_eq!(fixture.case_id, "scattering-coefficients-rigid-v1");
    assert!(fixture.precision_digits >= 50);

    for case in &fixture.sphere {
        let ka: f64 = case.ka.parse().expect("ka is a decimal");
        assert_eq!(
            case.num_terms,
            suggested_sphere_terms(ka),
            "golden must pin the suggested truncation for ka={ka}"
        );
        let golden: Vec<Complex64> = case.coefficients.iter().map(ComplexPair::value).collect();

        let r_points = vec![1.5, 3.0];
        let theta_points = vec![0.0, PI / 2.0, PI];
        let solution = sphere_scattering_3d(
            ka,
            1.0,
            case.num_terms,
            r_points.clone(),
            theta_points.clone(),
        );

        let mut expected = Vec::with_capacity(solution.pressure.len());
        for &r in &r_points {
            let kr = ka * r;
            for &theta in &theta_points {
                let cos_theta = theta.cos();
                let mut total = Complex64::new(0.0, 0.0);
                for (n, &a_n) in golden.iter().enumerate() {
                    let n_f64 = n as f64;
                    let prefactor = 2.0 * n_f64 + 1.0;
                    let i_power_n =
                        Complex64::new((n_f64 * PI / 2.0).cos(), (n_f64 * PI / 2.0).sin());
                    let jn = spherical_bessel_j(n, kr);
                    let hn = Complex64::new(jn, spherical_bessel_y(n, kr));
                    let pn = legendre_p(n, cos_theta);
                    total += prefactor * i_power_n * (jn - a_n * hn) * pn;
                }
                expected.push(total);
            }
        }
        assert_pressure_close(&solution.pressure, &expected, &format!("sphere ka={ka}"));
    }
}

#[test]
fn cylinder_scattering_matches_independent_golden() {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid golden fixture");

    for case in &fixture.cylinder {
        let ka: f64 = case.ka.parse().expect("ka is a decimal");
        assert_eq!(
            case.num_terms,
            suggested_cylinder_terms(ka),
            "golden must pin the suggested truncation for ka={ka}"
        );
        let golden: Vec<Complex64> = case.coefficients.iter().map(ComplexPair::value).collect();

        let r_points = vec![1.5, 3.0];
        let theta_points = vec![0.0, PI / 4.0, PI / 2.0, PI];
        let solution = cylinder_scattering_2d(
            ka,
            1.0,
            case.num_terms,
            r_points.clone(),
            theta_points.clone(),
        );

        let mut expected = Vec::with_capacity(solution.pressure.len());
        for &r in &r_points {
            let kr = ka * r;
            let hankels: Vec<Complex64> = (0..case.num_terms)
                .map(|n| Complex64::new(kr.bessel_jv(n as f64), kr.bessel_yv(n as f64)))
                .collect();
            for &theta in &theta_points {
                let incident = Complex64::new((kr * theta.cos()).cos(), (kr * theta.cos()).sin());
                let mut scattered = Complex64::new(0.0, 0.0);
                for (n, (&b_n, &hn)) in golden.iter().zip(&hankels).enumerate() {
                    let epsilon_n = if n == 0 { 1.0 } else { 2.0 };
                    scattered += b_n * hn * (epsilon_n * (n as f64 * theta).cos());
                }
                expected.push(incident + scattered);
            }
        }
        assert_pressure_close(&solution.pressure, &expected, &format!("cylinder ka={ka}"));
    }
}
