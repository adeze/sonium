//! CPU-only order-2 H(curl) periodic edge-mode contract.

use num_complex::Complex64;
use sonium_fem::boundary::periodic::{
    MpcError, MpcSystem, PeriodicTransform, SectorPeriodicMap,
    expand_order2_oriented_hcurl_constraints,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "fem.hcurl-order2-periodic-edge-contract.v1";

fn main() {
    let result = match run() {
        Ok((constraint_count, recovery_checks, rejection_checks)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("periodic_edge_mode_constraints", constraint_count as f64)
            .minimum("periodic_edge_mode_constraints", 4.0)
            .metric("recovery_checks", recovery_checks as f64)
            .minimum("recovery_checks", 4.0)
            .metric("rejection_checks", rejection_checks as f64)
            .minimum("rejection_checks", 3.0)
            .dimension("affine order-2 Nedelec periodic edge modes")
            .formulation("Floquet phase with odd/even edge-mode orientation parity")
            .input_checksum("periodic-edge-pairs:two-edge-order2-fixture:v1")
            .reference_checksum("analytic/deterministic-fixture:fem-hcurl-order2-periodic-edge-v1")
            .seed("fem-hcurl-order2-periodic-edge:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error),
    };

    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}

fn run() -> Result<(usize, usize, usize), String> {
    let phase = Complex64::new(0.6, 0.8);
    let map = SectorPeriodicMap {
        node_pairs: vec![(0, 3), (1, 4)],
        edge_pairs: vec![(2, 7, -1), (5, 9, 1)],
        face_pairs: vec![],
        transform: PeriodicTransform::IDENTITY,
        phase,
    };
    let constraints = map
        .hcurl_order2_constraints()
        .map_err(|error| error.to_string())?;
    if constraints.len() != 4
        || constraints[0].phase != -phase
        || constraints[1].phase != phase
        || constraints[2].phase != phase
        || constraints[3].phase != phase
    {
        return Err("order-2 periodic edge-mode parity mismatch".into());
    }

    let mpcs: Vec<_> = constraints.iter().copied().map(Into::into).collect();
    let system = MpcSystem::new(20, &mpcs).map_err(|error| error.to_string())?;
    let mut reduced = vec![Complex64::new(0.0, 0.0); system.num_reduced_dofs()];
    for (index, &full_dof) in system.reduced_full_dofs().iter().enumerate() {
        reduced[index] = Complex64::new(full_dof as f64 + 0.25, -0.5);
    }
    let recovered = system
        .recover(&reduced)
        .map_err(|error| error.to_string())?;
    let mut recovery_checks = 0;
    for constraint in &constraints {
        if constraint
            .residual_checked(&recovered)
            .map_err(|error| error.to_string())?
            .norm()
            >= 1.0e-12
        {
            return Err("periodic MPC recovery residual is nonzero".into());
        }
        recovery_checks += 1;
    }

    let mut rejection_checks = 0;
    if !matches!(
        expand_order2_oriented_hcurl_constraints(&[(0, 1, 0)], phase),
        Err(MpcError::InvalidOrientation(0))
    ) {
        return Err("invalid orientation was accepted".into());
    }
    rejection_checks += 1;
    if !matches!(
        expand_order2_oriented_hcurl_constraints(&[(0, 1, 1)], Complex64::new(f64::NAN, 0.0)),
        Err(MpcError::NonFiniteCoefficient { .. })
    ) {
        return Err("non-finite phase was accepted".into());
    }
    rejection_checks += 1;
    if !matches!(
        expand_order2_oriented_hcurl_constraints(&[(usize::MAX, 1, 1)], Complex64::new(1.0, 0.0),),
        Err(MpcError::DofIndexOverflow(usize::MAX))
    ) {
        return Err("overflowing edge index was accepted".into());
    }
    rejection_checks += 1;

    Ok((constraints.len(), recovery_checks, rejection_checks))
}
