//! QA: von Mises truss snap-through via arc-length continuation.
//!
//! Traces the two-bar truss equilibrium path with the production
//! `riks_continuation` driver and checks the limit load, the snap-through
//! branch, and every equilibrium point against the closed form
//! `sonium_qa::truss_von_mises_load`.

// Rust guideline compliant 2026-02-21

use sonium_fem::solver::nonlinear::arc_length::{riks_continuation, ArcLengthConfig};
use sonium_qa::truss_von_mises_load;

fn main() {
    let (span, height, young, area) = (1.0, 0.1, 1.0, 1.0);
    let internal = |u: &[f64]| vec![truss_von_mises_load(u[0], span, height, young, area)];
    let tangent = |u: &[f64], rhs: &[f64]| {
        let h = 1e-8;
        let slope = (internal(&[u[0] + h])[0] - internal(&[u[0] - h])[0]) / (2.0 * h);
        if slope == 0.0 || !slope.is_finite() {
            return Err("singular truss tangent".to_string());
        }
        Ok(vec![rhs[0] / slope])
    };
    let config = ArcLengthConfig {
        initial_step: 0.02,
        max_step: 0.03,
        max_steps: 60,
        psi: 0.1,
        ..Default::default()
    };
    let path = riks_continuation(internal, tangent, &[1.0], &[0.0], 0.0, &config)
        .expect("truss trace must advance");

    // Reference limit load from a fine closed-form scan.
    let mut reference_peak: f64 = 0.0;
    let mut w: f64 = 0.0;
    while w <= height {
        reference_peak = reference_peak.max(truss_von_mises_load(w, span, height, young, area));
        w += 1e-5;
    }
    // First limit point: peak load before the flat position (later peaks
    // belong to the tensile stretching tail, not the snap-through).
    let traced_peak = path
        .points
        .iter()
        .filter(|point| point.u[0] < 1.2 * height)
        .map(|point| point.lambda)
        .fold(f64::NEG_INFINITY, f64::max);
    println!("closed-form limit load: {reference_peak:.6e}");
    println!("traced limit load:       {traced_peak:.6e}");
    println!(
        "steps accepted/rejected: {}/{}",
        path.steps_accepted, path.steps_rejected
    );
    assert!(
        (traced_peak - reference_peak).abs() <= 0.01 * reference_peak,
        "limit load misses the closed form by more than 1%"
    );

    // Every traced point must sit on the closed-form equilibrium curve.
    for point in &path.points {
        let closed = truss_von_mises_load(point.u[0], span, height, young, area);
        assert!(
            (point.lambda - closed).abs() <= 1e-6 + 0.02 * reference_peak,
            "point w={:.4} leaves the equilibrium curve: {} vs {closed}",
            point.u[0],
            point.lambda
        );
    }

    // The trace must snap past the flat position onto the second branch.
    let tip = path.points.last().unwrap();
    println!("tip: w={:.6}, lambda={:.6e}", tip.u[0], tip.lambda);
    assert!(
        tip.u[0] > 1.5 * height,
        "trace stalled before the mirror branch"
    );
    println!("PASS: qa-fem-von-mises-truss");
}
