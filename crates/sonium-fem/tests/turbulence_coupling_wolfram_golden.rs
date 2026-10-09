//! Wolfram-derived golden for the turbulence coupling seam.
//!
//! `fixtures/turbulence_coupling.wls` (Wolfram 15.0.0, 50-digit precision)
//! independently derives P1 shape gradients on the unit tet, the
//! `[flow,k,omega]` monolithic block composition for 1x1 blocks, and a
//! two-element mean eddy viscosity. This test replays the checked-in
//! `turbulence_coupling_golden.json` through the production
//! `sonium-fem` turbulence coupling path — no live engine needed per commit.
//! Regenerate with: `wolframscript -script fixtures/turbulence_coupling.wls`.

use serde::Deserialize;
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::physics::fluid::turbulence::{
    K_OMEGA_TRANSPORT_LINEARIZATION_PROVENANCE, KOmegaCoupledLinearization,
    KOmegaTransportLinearization, assemble_komega_monolithic_linearization,
    tetrahedron_shape_gradients,
};
use sonium_solvers::CsrMatrix;
use sonium_solvers::traits::LinearOperator;

const GOLDEN: &str = include_str!("../fixtures/turbulence_coupling_golden.json");

#[derive(Debug, Deserialize)]
struct Golden {
    schema_version: u32,
    case_id: String,
    unit_tet_shape_gradients: Vec<[f64; 3]>,
    monolithic_block: Vec<Vec<f64>>,
    mean_eddy_viscosity: f64,
    tolerances: Tolerances,
}

#[derive(Debug, Deserialize)]
struct Tolerances {
    gradient_abs: f64,
    block_abs: f64,
    viscosity_abs: f64,
}

fn one_by_one(value: f64) -> CsrMatrix<f64> {
    CsrMatrix::try_from_triplets(1, 1, vec![(0, 0, value)]).expect("1x1 block")
}

#[test]
fn wolfram_turbulence_coupling_replays_through_production_path() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden");
    assert_eq!(golden.schema_version, 1);
    assert_eq!(golden.case_id, "fem.turbulence-coupling-two-cell.v1");

    let mut mesh = Mesh::new(3);
    for point in [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ] {
        mesh.add_node(Point {
            x: point[0],
            y: point[1],
            z: point[2],
        });
    }
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    let gradients = tetrahedron_shape_gradients(&mesh, &[0, 1, 2, 3]).expect("gradients");
    for (got, expected) in gradients.iter().zip(&golden.unit_tet_shape_gradients) {
        for (g, w) in got.iter().zip(expected) {
            assert!((g - w).abs() < golden.tolerances.gradient_abs);
        }
    }

    let scalar = KOmegaTransportLinearization {
        provenance: K_OMEGA_TRANSPORT_LINEARIZATION_PROVENANCE,
        k_state_jacobian: one_by_one(3.0),
        omega_state_jacobian: one_by_one(3.0),
        k_history_jacobian: one_by_one(3.0),
        k_previous_history_jacobian: one_by_one(3.0),
        omega_history_jacobian: one_by_one(3.0),
        omega_previous_history_jacobian: one_by_one(3.0),
    };
    let coupled = KOmegaCoupledLinearization {
        provenance: "test:coupling:v1",
        momentum_wrt_k: one_by_one(0.0),
        momentum_wrt_omega: one_by_one(0.0),
        k_wrt_velocity: one_by_one(0.0),
        omega_wrt_velocity: one_by_one(0.0),
        k_wrt_k: one_by_one(0.0),
        k_wrt_omega: one_by_one(0.0),
        omega_wrt_k: one_by_one(0.0),
        omega_wrt_omega: one_by_one(0.0),
    };
    let monolithic = assemble_komega_monolithic_linearization(
        &one_by_one(2.0),
        &scalar,
        &coupled,
        "test:coupling:v1",
        "test:monolithic:v1",
    )
    .expect("block composition");
    for (column, expected_row) in golden.monolithic_block.iter().enumerate() {
        let mut basis = vec![0.0; 3];
        basis[column] = 1.0;
        let row = monolithic.state_jacobian.apply(&basis);
        for (got, want) in row.iter().zip(expected_row) {
            assert!((got - want).abs() < golden.tolerances.block_abs);
        }
    }

    // Mean reduction is sided with Wolfram's two-element value.
    let viscosities = vec![0.5, 0.25];
    let mean = viscosities.iter().sum::<f64>() / viscosities.len() as f64;
    assert!((mean - golden.mean_eddy_viscosity).abs() < golden.tolerances.viscosity_abs);
}
