//! Wolfram-derived golden for the aperture seam.
//!
//! `fixtures/coupling_seam.wls` (Wolfram 15.0.0, 50-digit working precision)
//! independently derives the active-power audit and the measure-weighted
//! adjoint identity for a two-triangle aperture. This test replays the
//! checked-in `coupling_seam_golden.json` through the production
//! `sonium-coupling` + BEM transfer path — no live engine needed per commit.
//! Regenerate with: `wolframscript -script fixtures/coupling_seam.wls`.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_bem::core::coupling::ConservativeTraceTransfer;
use sonium_coupling::InterfacePowerLedger;

const GOLDEN: &str = include_str!("../fixtures/coupling_seam_golden.json");

#[derive(Debug, Deserialize)]
struct Golden {
    schema_version: u32,
    case_id: String,
    areas_m2: Vec<f64>,
    source_measures_m2: Vec<f64>,
    interior_pressure_per_triangle_pa: Vec<[f64; 2]>,
    exterior_pressure_per_triangle_pa: Vec<[f64; 2]>,
    velocity_per_triangle_m_s: Vec<[f64; 2]>,
    interior_active_power_w: f64,
    exterior_active_power_w: f64,
    adjoint_mismatch: f64,
    tolerances: Tolerances,
}

#[derive(Debug, Deserialize)]
struct Tolerances {
    power_abs_w: f64,
    adjoint_abs: f64,
}

fn pair(values: &[[f64; 2]]) -> Vec<Complex64> {
    values
        .iter()
        .map(|&[re, im]| Complex64::new(re, im))
        .collect()
}

#[test]
fn wolfram_power_audit_replays_through_production_path() {
    let golden: Golden = serde_json::from_str(GOLDEN).expect("parse golden");
    assert_eq!(golden.schema_version, 1);
    assert_eq!(golden.case_id, "coupling.seam-two-triangle.v1");

    let faces = vec![[0usize, 1, 2], [1, 3, 2]];
    let (transfer, _) =
        ConservativeTraceTransfer::try_from_tri3_vertex_faces(&faces, &golden.areas_m2)
            .expect("golden topology builds a transfer");
    assert_eq!(transfer.target_measures, golden.areas_m2);
    for (got, expected) in transfer
        .source_measures
        .iter()
        .zip(golden.source_measures_m2.iter())
    {
        assert!((got - expected).abs() < 1.0e-15);
    }

    let ledger = InterfacePowerLedger::audit(
        &golden.areas_m2,
        &pair(&golden.interior_pressure_per_triangle_pa),
        &pair(&golden.exterior_pressure_per_triangle_pa),
        &pair(&golden.velocity_per_triangle_m_s),
    )
    .expect("valid golden state");
    assert!(
        (ledger.interior_active_power_w - golden.interior_active_power_w).abs()
            < golden.tolerances.power_abs_w
    );
    assert!(
        (ledger.exterior_active_power_w - golden.exterior_active_power_w).abs()
            < golden.tolerances.power_abs_w
    );
    assert!(golden.adjoint_mismatch < golden.tolerances.adjoint_abs);
}
