//! Replay: native airbag isentropic-orifice relations against the
//! 50-digit Wolfram oracle (`wolfram/airbag_isentrope.wls`).

use serde::Deserialize;
use sonium_fem::physics::fluid::airbag::{Gas, critical_ratio, isentropic_orifice, mixture_gamma};

#[derive(Debug, Deserialize)]
struct VentCase {
    ratio: f64,
    velocity: f64,
    exit_density: f64,
    mass_rate: f64,
    energy_rate: f64,
    temp_ratio: f64,
}

#[derive(Debug, Deserialize)]
struct ReferenceSet {
    schema_version: u32,
    case: String,
    gamma_deck: f64,
    cp_deck: f64,
    gas_constant: f64,
    pressure: f64,
    density: f64,
    area: f64,
    enthalpy: f64,
    critical_ratio_deck: f64,
    critical_ratio_air: f64,
    air_velocity: f64,
    mixture_gamma: f64,
    vents: Vec<VentCase>,
}

fn references() -> ReferenceSet {
    serde_json::from_str(include_str!("../wolfram/goldens/airbag_isentrope.json")).unwrap()
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    let bound = tol * expected.abs().max(1e-300);
    assert!(
        (actual - expected).abs() <= bound,
        "{what}: actual {actual} differs from expected {expected} by more than {bound}"
    );
}

#[test]
fn airbag_isentrope_matches_wolfram_oracle() {
    let oracle = references();
    assert_eq!(oracle.schema_version, 1);
    assert_eq!(oracle.case, "fem.airbag-isentrope.v1");
    assert_eq!(oracle.vents.len(), 6);

    let gas = Gas::new(oracle.gamma_deck, [oracle.cp_deck, 0.0, 0.0]);
    close(
        gas.specific_gas_constant(295.0),
        oracle.gas_constant,
        1e-12,
        "gas constant",
    );
    close(
        critical_ratio(oracle.gamma_deck),
        oracle.critical_ratio_deck,
        1e-12,
        "deck critical ratio",
    );
    close(
        critical_ratio(1.4),
        oracle.critical_ratio_air,
        1e-12,
        "air critical ratio",
    );

    let air = isentropic_orifice(1.4, 2.0e-4, 1.0e-4, 1.2e-9, 840.0, 3.0e5, f64::INFINITY);
    close(air.velocity, oracle.air_velocity, 1e-12, "air velocity");

    let mix = mixture_gamma(
        &[
            (0.3, gas, oracle.gas_constant),
            (0.7, gas, oracle.gas_constant),
        ],
        300.0,
    );
    close(mix, oracle.mixture_gamma, 1e-12, "mixture gamma");

    for vent in &oracle.vents {
        let downstream = oracle.pressure * vent.ratio;
        let flow = isentropic_orifice(
            oracle.gamma_deck,
            oracle.pressure,
            downstream,
            oracle.density,
            oracle.area,
            oracle.enthalpy,
            f64::INFINITY,
        );
        let tag = format!("ratio {}", vent.ratio);
        close(
            flow.velocity,
            vent.velocity,
            1e-12,
            &format!("{tag} velocity"),
        );
        close(
            flow.exit_density,
            vent.exit_density,
            1e-12,
            &format!("{tag} exit density"),
        );
        close(
            flow.mass_rate,
            vent.mass_rate,
            1e-12,
            &format!("{tag} mass rate"),
        );
        close(
            flow.energy_rate,
            vent.energy_rate,
            1e-12,
            &format!("{tag} energy rate"),
        );
        let pcrit = oracle.pressure * oracle.critical_ratio_deck;
        let temp_ratio = (downstream.max(pcrit) / oracle.pressure)
            .powf((oracle.gamma_deck - 1.0) / oracle.gamma_deck);
        close(
            temp_ratio,
            vent.temp_ratio,
            1e-12,
            &format!("{tag} temp ratio"),
        );
    }
}
