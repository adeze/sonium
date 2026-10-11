//! Wolfram-derived golden for ALE pure-advection translation.
//!
//! `fixtures/ale_advection.wls` (Wolfram 15.0.0, 50-digit precision)
//! evaluates the exact translation of a two-harmonic periodic profile
//! at the centers of an 800-cell grid (length 2, velocity 1) after a
//! half and a full period. This test replays the checked-in
//! `ale_advection_golden.json` through the production `Advection1d`
//! path — no live engine needed per commit. Regenerate with:
//! `wolframscript -script fixtures/ale_advection.wls > fixtures/ale_advection_golden.json`.

use serde::Deserialize;
use sonium_fem::physics::fluid::ale::Advection1d;

const GOLDEN: &str = include_str!("../fixtures/ale_advection_golden.json");

#[derive(Debug, Deserialize)]
struct Golden {
    schema_version: u32,
    case_id: String,
    cells: usize,
    length: f64,
    velocity: f64,
    cfl: f64,
    times: Vec<f64>,
    exact_half: Vec<f64>,
    exact_full: Vec<f64>,
    tolerances: Tolerances,
}

#[derive(Debug, Deserialize)]
struct Tolerances {
    error_abs: f64,
}

fn profile(x: f64) -> f64 {
    use std::f64::consts::PI;
    (PI * x).sin() + 0.25 * (2.0 * PI * x + 1.0).sin()
}

#[test]
fn wolfram_advection_replays_through_production_path() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden");
    assert_eq!(golden.schema_version, 1);
    assert_eq!(golden.case_id, "fem.ale-advection.v1");
    assert_eq!(golden.times.len(), 2);
    assert_eq!(golden.exact_half.len(), golden.cells);
    assert_eq!(golden.exact_full.len(), golden.cells);

    let mut solver =
        Advection1d::from_function(golden.cells, golden.length, golden.velocity, profile)
            .expect("build solver");
    solver.run(golden.times[0], golden.cfl).expect("run half");
    let half_error = solver
        .values()
        .iter()
        .zip(golden.exact_half.iter())
        .map(|(got, want)| (got - want).abs())
        .fold(0.0, f64::max);
    assert!(
        half_error < golden.tolerances.error_abs,
        "half-period error {half_error}"
    );

    solver
        .run(golden.times[1] - golden.times[0], golden.cfl)
        .expect("run full");
    let full_error = solver
        .values()
        .iter()
        .zip(golden.exact_full.iter())
        .map(|(got, want)| (got - want).abs())
        .fold(0.0, f64::max);
    assert!(
        full_error < golden.tolerances.error_abs,
        "full-period error {full_error}"
    );
}
