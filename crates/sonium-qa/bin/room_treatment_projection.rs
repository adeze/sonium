//! Compile-only contract for fixed-mesh to body-fitted treatment projection.
//!
//! This source checks interpolation and adjoint pullback topology without
//! launching a target FEM reanalysis, QA binary workload, or simulation.

use sonium_sim_room::treatment_projection::{
    BodyFittedTreatmentProjection, optimize_and_promote_body_fitted_treatment,
};

fn main() {
    let _workflow_entry_point = optimize_and_promote_body_fitted_treatment;
    let projection = BodyFittedTreatmentProjection::new(
        &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        &[[0.25, 0.0, 0.0], [0.75, 0.0, 0.0]],
        1.1,
    )
    .expect("compile-only treatment projection fixture must be supported");
    let target = projection
        .project(&[0.2, 0.8])
        .expect("compile-only treatment projection must apply");
    let source_gradient = projection
        .pullback_gradient(&[1.0, 1.0])
        .expect("compile-only treatment projection pullback must apply");
    assert_eq!(target.len(), 2);
    assert_eq!(source_gradient.len(), 2);
    println!("SUCCESS");
}
