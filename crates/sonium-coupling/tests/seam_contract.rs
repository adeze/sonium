//! Seam contract: BEM conservative transfer + coupling trace schema +
//! interface power ledger composed through `sonium-coupling`.
//!
//! This is the second consumer of the extracted seam (the first is the
//! speaker-modeling product path): it drives the generic contracts directly
//! — forward conservation, the measure-weighted adjoint identity, and the
//! power balance — without any product fixture.

use num_complex::Complex64;
use sonium_bem::core::coupling::ConservativeTraceTransfer;
use sonium_coupling::{ApertureFace, ApertureTrace, InterfacePowerLedger};

/// Two triangles sharing edge 1-2; areas chosen exactly representable.
fn faces() -> Vec<[usize; 3]> {
    vec![[0, 1, 2], [1, 3, 2]]
}

fn areas() -> Vec<f64> {
    vec![0.5, 0.25]
}

fn transfer() -> (ConservativeTraceTransfer, Vec<usize>) {
    ConservativeTraceTransfer::try_from_tri3_vertex_faces(&faces(), &areas())
        .expect("valid test transfer")
}

#[test]
fn transfer_preserves_constants_and_conserves_measure() {
    let (transfer, source_nodes) = transfer();
    assert_eq!(source_nodes, vec![0, 1, 2, 3]);
    let ones = vec![Complex64::new(1.0, -0.5); 4];
    let mapped = transfer.apply(&ones).expect("forward transfer");
    assert_eq!(mapped.len(), 2);
    for value in &mapped {
        assert!((value - ones[0]).norm() < 1.0e-15);
    }
    let source_total: f64 = transfer.source_measures.iter().sum();
    let target_total: f64 = transfer.target_measures.iter().sum();
    assert!((source_total - target_total).abs() < 1.0e-15);
    assert!((target_total - 0.75).abs() < 1.0e-15);
}

#[test]
fn measure_weighted_adjoint_identity_holds() {
    let (transfer, _) = transfer();
    let x = vec![
        Complex64::new(1.0, 0.5),
        Complex64::new(-2.0, 1.0),
        Complex64::new(0.25, -1.5),
        Complex64::new(3.0, 0.0),
    ];
    let y = vec![Complex64::new(0.5, 0.25), Complex64::new(-1.0, 2.0)];
    let tx = transfer.apply(&x).expect("forward");
    let ahy = transfer.apply_adjoint(&y).expect("adjoint");
    // <Tx, y>_{M_t} == <x, T^H y>_{M_s}.
    let left: Complex64 = tx
        .iter()
        .zip(y.iter())
        .zip(transfer.target_measures.iter())
        .map(|((t, y), m)| t.conj() * y * m)
        .sum();
    let right: Complex64 = x
        .iter()
        .zip(ahy.iter())
        .zip(transfer.source_measures.iter())
        .map(|((x, h), m)| x.conj() * h * m)
        .sum();
    assert!((left - right).norm() < 1.0e-12);
}

#[test]
fn trace_schema_matches_transfer_topology() {
    let (transfer, _) = transfer();
    let trace = ApertureTrace {
        faces: faces()
            .into_iter()
            .zip(areas())
            .map(|(nodes, area_m2)| ApertureFace { nodes, area_m2 })
            .collect(),
        bem_elements: vec![11, 17],
        outward_normal_velocity: true,
    };
    trace.validate().expect("valid trace");
    assert_eq!(trace.areas(), transfer.target_measures);
    assert!((trace.total_area() - 0.75).abs() < 1.0e-15);
}

#[test]
fn power_ledger_balances_consistent_interface_state() {
    let (transfer, _) = transfer();
    // Uniform pressure field: interior nodal values transfer to the same
    // per-triangle exterior values, so no power is lost at the interface.
    let nodal = vec![Complex64::new(2.0, 1.0); 4];
    let exterior = transfer.apply(&nodal).expect("forward");
    let interior_per_triangle = exterior.clone();
    let velocity = vec![Complex64::new(0.1, -0.05); 2];
    let ledger = InterfacePowerLedger::audit(
        &transfer.target_measures,
        &interior_per_triangle,
        &exterior,
        &velocity,
    )
    .expect("valid ledger");
    assert_eq!(ledger.conservation_residual_w, 0.0);
    // 0.5 * Re((2+i)(0.1+0.05i)) * 0.75 = 0.5 * 0.15 * 0.75.
    assert!((ledger.interior_active_power_w - 0.05625).abs() < 1.0e-15);
}

#[test]
fn power_ledger_flags_interface_pressure_jump() {
    let (transfer, _) = transfer();
    let interior = vec![Complex64::new(3.0, 0.0); 2];
    let exterior = vec![Complex64::new(1.0, 0.0); 2];
    let velocity = vec![Complex64::new(1.0, 0.0); 2];
    let ledger =
        InterfacePowerLedger::audit(&transfer.target_measures, &interior, &exterior, &velocity)
            .expect("valid ledger");
    // Per-triangle powers 1.5*area vs 0.5*area; residual = 1.0 * 0.75.
    assert!((ledger.conservation_residual_w - 0.75).abs() < 1.0e-12);
    assert!(ledger.relative_conservation_residual > 0.0);
}
