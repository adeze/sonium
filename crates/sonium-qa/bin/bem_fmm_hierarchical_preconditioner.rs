//! Checked single-node hierarchical FMM preconditioning for room acoustics.

use sonium_bem::room_acoustics::{
    FmmSolverConfig, Point3D, RoomMesh, Source, SurfaceElement,
    solve_bem_fmm_gmres_hierarchical_with_result,
};
use sonium_qa::{QaResult, QaStatus};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.fmm-hierarchical-preconditioner.v1";

fn tetrahedron_mesh() -> RoomMesh {
    RoomMesh {
        nodes: vec![
            Point3D::new(0.0, 0.0, 0.0),
            Point3D::new(1.0, 0.0, 0.0),
            Point3D::new(0.0, 1.0, 0.0),
            Point3D::new(0.0, 0.0, 1.0),
        ],
        elements: vec![
            SurfaceElement::triangle(0, 1, 2),
            SurfaceElement::triangle(0, 3, 1),
            SurfaceElement::triangle(0, 2, 3),
            SurfaceElement::triangle(1, 3, 2),
        ],
    }
}

fn run() -> Result<(usize, usize, f64), String> {
    let mesh = tetrahedron_mesh();
    let source = Source::omnidirectional(Point3D::new(0.2, 0.2, 0.2), 1.0);
    let config = FmmSolverConfig {
        max_elements_per_leaf: 1,
        n_theta: 4,
        n_phi: 8,
        n_terms: 4,
        ..FmmSolverConfig::default()
    };
    let result = solve_bem_fmm_gmres_hierarchical_with_result(
        &mesh,
        &[source],
        1.0,
        100.0,
        &config,
        100,
        10,
        1.0e-6,
    )?;

    if !result.converged {
        return Err(format!(
            "hierarchical FMM GMRES did not converge after {} iterations (residual {:.3e})",
            result.iterations, result.residual
        ));
    }
    if !result.residual.is_finite() || result.residual > 1.0e-6 {
        return Err(format!(
            "hierarchical FMM full-operator residual {:.3e} exceeds 1e-6",
            result.residual
        ));
    }
    if !result
        .x
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
    {
        return Err("hierarchical FMM returned a non-finite solution".to_string());
    }

    Ok((result.x.len(), result.iterations, result.residual))
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok((dimension, iterations, residual))) => QaResult::passed(CASE_ID)
            .metric("dimension", dimension as f64)
            .metric("iterations", iterations as f64)
            .metric("true_residual", residual)
            .maximum("true_residual", 1.0e-6)
            .input_checksum("analytic/deterministic-fixture:bem-fmm-hierarchical-preconditioner-v1")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-fmm-hierarchical-preconditioner-v1",
            )
            .dimension("3-D tetrahedron; four constant-panel DOFs")
            .reference("independent full-SLFMM residual certification")
            .seed("bem-fmm-hierarchical-preconditioner:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Ok(Err(message)) => QaResult::failed(CASE_ID, message),
        Err(_) => QaResult::failed(
            CASE_ID,
            "panic while running hierarchical FMM preconditioner QA".to_string(),
        ),
    };
    let passed = matches!(&result.status, QaStatus::Passed);
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !passed {
        std::process::exit(1);
    }
}
