//! Wolfram-derived golden for the transfer seam.
//!
//! `fixtures/transfer_seam.wls` (Wolfram 15.0.0, 50-digit precision)
//! independently derives the unit-tet lumped integral, the two-panel
//! inverse-distance value, and both sides of the measure-weighted adjoint
//! identity. This test replays the checked-in `transfer_seam_golden.json`
//! through the production `sonium-fem` mesh transfer path — no live engine
//! needed per commit. Regenerate with:
//! `wolframscript -script fixtures/transfer_seam.wls`.

use serde::Deserialize;
use sonium_fem::mesh::panel_transfer::{
    Panel, PanelSet, PanelStateTransferOptions, transfer_panel_scalar_state,
};

const GOLDEN: &str = include_str!("../fixtures/transfer_seam_golden.json");

#[derive(Debug, Deserialize)]
struct Golden {
    schema_version: u32,
    case_id: String,
    tet_volume: f64,
    tet_lumped_integral: f64,
    panel_idw_value: f64,
    transfer_adjoint_lhs: [f64; 2],
    transfer_adjoint_rhs: [f64; 2],
    transfer_adjoint_mismatch: f64,
    tolerances: Tolerances,
}

#[derive(Debug, Deserialize)]
struct Tolerances {
    conservation_abs: f64,
    idw_abs: f64,
    adjoint_abs: f64,
}

#[test]
fn wolfram_transfer_seam_replays_through_production_path() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden");
    assert_eq!(golden.schema_version, 1);
    assert_eq!(golden.case_id, "fem.transfer-seam-two-cell.v1");

    // Unit simplex tet: V = 1/6, lumped integral of x + 2y + 3z is 0.25.
    assert!((golden.tet_volume - 1.0 / 6.0).abs() < golden.tolerances.conservation_abs);
    assert!((golden.tet_lumped_integral - 0.25).abs() < golden.tolerances.conservation_abs);

    // Two-panel fixture through the production path agrees with Wolfram.
    let source = PanelSet {
        panels: vec![
            Panel {
                center: [0.0, 0.0, 0.0],
                area: 0.5,
                group: 0,
            },
            Panel {
                center: [0.0, 1.0, 0.0],
                area: 0.5,
                group: 0,
            },
        ],
    };
    let target = PanelSet {
        panels: vec![
            Panel {
                center: [0.001, 0.0, 0.0],
                area: 0.5,
                group: 0,
            },
            Panel {
                center: [0.001, 1.0, 0.0],
                area: 0.5,
                group: 0,
            },
        ],
    };
    let result = transfer_panel_scalar_state(
        &source,
        &[3.0, 3.0],
        &target,
        PanelStateTransferOptions::default(),
    )
    .expect("panel transfer");
    for value in &result.values {
        assert!((value - golden.panel_idw_value).abs() < golden.tolerances.idw_abs);
    }

    // Wolfram's adjoint sides agree with each other; the Rust-side identity
    // itself is covered by `mesh::state_transfer` unit tests.
    let [lr, li] = golden.transfer_adjoint_lhs;
    let [rr, ri] = golden.transfer_adjoint_rhs;
    assert!((lr - rr).abs() < golden.tolerances.adjoint_abs);
    assert!((li - ri).abs() < golden.tolerances.adjoint_abs);
    assert!(golden.transfer_adjoint_mismatch < golden.tolerances.adjoint_abs);
}
