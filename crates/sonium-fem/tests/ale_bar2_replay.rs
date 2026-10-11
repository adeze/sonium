//! Wolfram-derived golden for the 1D bar-impact reference.
//!
//! `fixtures/ale_bar2.wls` (Wolfram 15.0.0, 50-digit precision)
//! evaluates the closed-form d'Alembert solution for the bar2_v41
//! copper-bar analog (wall at `x = 0`, free face at `x = L`,
//! separation at `2L/c`): derived scalars plus 20 interior
//! velocity/stress samples and 2 post-separation free-flight points.
//! This test replays the checked-in `ale_bar2_golden.json` through
//! the production `BarImpact` path — no live engine needed per
//! commit. Regenerate with:
//! `wolframscript -script fixtures/ale_bar2.wls > fixtures/ale_bar2_golden.json`.

use serde::Deserialize;
use sonium_fem::physics::fluid::ale::BarImpact;

const GOLDEN: &str = include_str!("../fixtures/ale_bar2_golden.json");

#[derive(Debug, Deserialize)]
struct Golden {
    schema_version: u32,
    case_id: String,
    density: f64,
    young: f64,
    length: f64,
    speed: f64,
    wave_speed: f64,
    contact_stress: f64,
    separation_time: f64,
    samples: Vec<Sample>,
    free_flight: Vec<Sample>,
    energy_drift: f64,
    tolerances: Tolerances,
}

#[derive(Debug, Deserialize)]
struct Sample {
    x: f64,
    t: f64,
    velocity: f64,
    stress: f64,
}

#[derive(Debug, Deserialize)]
struct Tolerances {
    derived_rel: f64,
    velocity_abs: f64,
    stress_rel: f64,
    energy_rel: f64,
}

fn relative_error(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(f64::MIN_POSITIVE)
}

#[test]
fn wolfram_bar_impact_replays_through_production_path() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden");
    assert_eq!(golden.schema_version, 1);
    assert_eq!(golden.case_id, "fem.ale-bar2.v1");
    assert_eq!(golden.samples.len(), 20);
    assert_eq!(golden.free_flight.len(), 2);

    let bar = BarImpact::new(golden.density, golden.young, golden.length, golden.speed)
        .expect("build bar");
    assert_eq!(bar, BarImpact::copper_bar2());
    assert!(relative_error(bar.wave_speed(), golden.wave_speed) < golden.tolerances.derived_rel);
    assert!(
        relative_error(bar.contact_stress(), golden.contact_stress) < golden.tolerances.derived_rel
    );
    assert!(
        relative_error(bar.separation_time(), golden.separation_time)
            < golden.tolerances.derived_rel
    );

    for sample in golden.samples.iter().chain(golden.free_flight.iter()) {
        let velocity = bar.velocity(sample.x, sample.t);
        let stress = bar.stress(sample.x, sample.t);
        assert!(
            (velocity - sample.velocity).abs() < golden.tolerances.velocity_abs,
            "v({:.5}, {:.3e}): got {velocity}, want {}",
            sample.x,
            sample.t,
            sample.velocity
        );
        let scale = sample.stress.abs().max(bar.contact_stress());
        assert!(
            (stress - sample.stress).abs() < golden.tolerances.stress_rel * scale,
            "s({:.5}, {:.3e}): got {stress}, want {}",
            sample.x,
            sample.t,
            sample.stress
        );
    }

    // Golden self-consistency plus an independent Rust-side energy
    // recomputation at half transit (jump on a cell boundary).
    assert!(golden.energy_drift.abs() < golden.tolerances.energy_rel);
    let points = 4000;
    let width = bar.length / points as f64;
    let time = 0.5 * bar.length / bar.wave_speed();
    let mut energy = 0.0;
    for point in 0..points {
        let x = (point as f64 + 0.5) * width;
        let velocity = bar.velocity(x, time);
        let stress = bar.stress(x, time);
        energy +=
            (0.5 * bar.density * velocity * velocity + stress * stress / (2.0 * bar.young)) * width;
    }
    let initial = 0.5 * bar.density * bar.length * bar.speed * bar.speed;
    assert!(relative_error(energy, initial) < golden.tolerances.energy_rel);
}
