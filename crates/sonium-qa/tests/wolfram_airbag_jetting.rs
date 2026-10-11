//! Replay: native airbag jetting geometry against the 50-digit
//! Wolfram oracle (`wolfram/airbag_jetting.wls`).

use serde::Deserialize;
use sonium_fem::physics::fluid::airbag::{Jet, Tabulated};

#[derive(Debug, Deserialize)]
struct JetCase {
    theta_deg: f64,
    dist: f64,
    facing: f64,
    overpressure: f64,
}

#[derive(Debug, Deserialize)]
struct JetNodes {
    n1: [f64; 3],
    n2: [f64; 3],
    n3: [f64; 3],
}

#[derive(Debug, Deserialize)]
struct ReferenceSet {
    schema_version: u32,
    case: String,
    conical: JetNodes,
    dihedral: JetNodes,
    cases: Vec<CaseEntry>,
}

#[derive(Debug, Deserialize)]
struct CaseEntry {
    id: String,
    jet: String,
    laws: String,
    center: [f64; 3],
    normal: [f64; 3],
    t: f64,
    #[serde(flatten)]
    sample: JetCase,
}

fn references() -> ReferenceSet {
    serde_json::from_str(include_str!("../wolfram/goldens/airbag_jetting.json")).unwrap()
}

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    let bound = tol * expected.abs().max(1e-300);
    assert!(
        (actual - expected).abs() <= bound,
        "{what}: actual {actual} differs from expected {expected} by more than {bound}"
    );
}

fn deck_jet(nodes: &JetNodes) -> Jet {
    Jet {
        n1: nodes.n1,
        n2: nodes.n2,
        n3: nodes.n3,
        time_law: Tabulated::constant(1.0),
        pressure_scale: 1.0,
        time_scale: 1.0,
        angle_law: Tabulated::constant(0.01),
        angle_scale: 1.0,
        angle_arg_scale: 1.0,
        dist_law: Tabulated::constant(0.01),
        dist_scale: 1.0,
        dist_arg_scale: 1.0,
        start: 0.0,
        enabled: true,
    }
}

fn slope_jet(nodes: &JetNodes) -> Jet {
    Jet {
        n1: nodes.n1,
        n2: nodes.n2,
        n3: nodes.n3,
        time_law: Tabulated::new(vec![0.0, 10.0], vec![1.0, 3.0]),
        pressure_scale: 2.0,
        time_scale: 1.0,
        angle_law: Tabulated::new(vec![0.0, 90.0, 180.0], vec![1.0, 0.5, 0.0]),
        angle_scale: 0.5,
        angle_arg_scale: 2.0,
        dist_law: Tabulated::new(vec![0.0, 1.0], vec![1.0, 0.5]),
        dist_scale: 3.0,
        dist_arg_scale: 0.1,
        start: 0.0,
        enabled: true,
    }
}

#[test]
fn airbag_jetting_matches_wolfram_oracle() {
    let oracle = references();
    assert_eq!(oracle.schema_version, 1);
    assert_eq!(oracle.case, "fem.airbag-jetting.v1");
    assert_eq!(oracle.cases.len(), 7);

    for entry in &oracle.cases {
        let nodes = match entry.jet.as_str() {
            "conical" => &oracle.conical,
            "dihedral" => &oracle.dihedral,
            other => panic!("unknown jet {}", other),
        };
        let jet = match entry.laws.as_str() {
            "deck" => deck_jet(nodes),
            "slope" => slope_jet(nodes),
            other => panic!("unknown laws {}", other),
        };
        let sample = jet.sample(entry.center, entry.normal, entry.t);
        close(
            sample.theta_deg,
            entry.sample.theta_deg,
            1e-12,
            &format!("{} theta", entry.id),
        );
        close(
            sample.dist,
            entry.sample.dist,
            1e-12,
            &format!("{} dist", entry.id),
        );
        close(
            sample.facing,
            entry.sample.facing,
            1e-12,
            &format!("{} facing", entry.id),
        );
        if entry.sample.overpressure == 0.0 {
            assert!(
                sample.overpressure.abs() < 1e-300,
                "{} overpressure: expected zero, got {}",
                entry.id,
                sample.overpressure
            );
        } else {
            close(
                sample.overpressure,
                entry.sample.overpressure,
                1e-12,
                &format!("{} overpressure", entry.id),
            );
        }
    }
}
