//! Assembled-AMG versus matrix-free comparison at matched error.
//!
//! Both paths solve the same 3-D solid study; the P1 assembled-AMG grid is
//! refined until displacement and strain energy match the P2 matrix-free
//! reference. Timings, iterations, and sizes are reported as evidence — no
//! speedup is asserted.

// Rust guideline compliant 2026-02-21

use sonium_sim_structural_mechanics::{
    Solid3DStaticOptions, Solid3DStaticSolver, Solid3DStudy, StructuralPolynomialOrder,
    run_solid3d_static_with_options,
};
use std::time::Instant;

#[test]
fn assembled_amg_matches_matrix_free_at_matched_error() {
    let study = Solid3DStudy::introductory();
    let started = Instant::now();
    let reference = run_solid3d_static_with_options(
        &study,
        Solid3DStaticOptions {
            polynomial_order: StructuralPolynomialOrder::Quadratic,
            solver: Solid3DStaticSolver::MatrixFreeJacobiFgmres,
        },
    )
    .expect("matrix-free reference run");
    let reference_ms = started.elapsed().as_secs_f64() * 1.0e3;
    assert!(reference.converged);
    assert!(reference.max_displacement_m > 0.0);
    assert!(reference.strain_energy_j > 0.0);
    println!(
        "matrix-free P2: {}x{}x{} dofs={} iters={} residual={:.3e} u_max={:.6e} energy={:.6e} ms={:.1}",
        study.geometry.nx,
        study.geometry.ny,
        study.geometry.nz,
        reference.dofs,
        reference.solver_iterations,
        reference.solver_residual,
        reference.max_displacement_m,
        reference.strain_energy_j,
        reference_ms
    );

    let mut resolution = [study.geometry.nx, study.geometry.ny, study.geometry.nz];
    let matched = loop {
        let mut refined = study.clone();
        refined.geometry.nx = resolution[0];
        refined.geometry.ny = resolution[1];
        refined.geometry.nz = resolution[2];
        let started = Instant::now();
        let candidate = run_solid3d_static_with_options(
            &refined,
            Solid3DStaticOptions {
                polynomial_order: StructuralPolynomialOrder::Linear,
                solver: Solid3DStaticSolver::AssembledAmgFgmres,
            },
        )
        .expect("assembled AMG run");
        let candidate_ms = started.elapsed().as_secs_f64() * 1.0e3;
        assert!(candidate.converged);
        let displacement_gap = (candidate.max_displacement_m - reference.max_displacement_m).abs()
            / reference.max_displacement_m;
        let energy_gap = (candidate.strain_energy_j - reference.strain_energy_j).abs()
            / reference.strain_energy_j;
        println!(
            "assembled-AMG P1: {}x{}x{} dofs={} iters={} residual={:.3e} du={:.3e} dE={:.3e} ms={:.1}",
            resolution[0],
            resolution[1],
            resolution[2],
            candidate.dofs,
            candidate.solver_iterations,
            candidate.solver_residual,
            displacement_gap,
            energy_gap,
            candidate_ms
        );
        // 2% matched error: measured gaps halve per refinement (6.8% ->
        // 3.7% -> 1.5% -> 0.9%), so this matches one level below the cap
        // with margin instead of balancing on the refinement ceiling.
        if displacement_gap < 0.02 && energy_gap < 0.02 {
            break candidate;
        }
        resolution = [resolution[0] * 2, resolution[1] * 2, resolution[2] * 2];
        assert!(
            resolution[0] <= 24,
            "P1 assembled-AMG failed to match P2 matrix-free within the refinement budget"
        );
    };

    // Both matched solutions must satisfy tight equilibrium gates.
    for (name, result) in [
        ("matrix-free P2", &reference),
        ("assembled-AMG P1", &matched),
    ] {
        let applied = result
            .applied_force_n
            .iter()
            .map(|f| f * f)
            .sum::<f64>()
            .sqrt();
        assert!(applied > 0.0);
        let relative = result.free_equilibrium_residual_n / applied;
        assert!(
            relative < 1.0e-6,
            "{name}: relative equilibrium residual = {relative}"
        );
    }
}
