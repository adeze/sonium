//! QA: body-fitted, mesh-converged reanalysis of projected motor topology
//! designs (review item D5, `acoustics-20250730.md` §3.5 promotion step 7).
//!
//! Gates:
//!
//! 1. **Golden reproduction.** The original full-steel geometry (default
//!    12x12 driver) through extract -> remesh -> reanalyze at refinement 1
//!    reproduces the golden model BL within 1e-9 relative. The remeshed
//!    geometry and mesh are identical to the golden magnetic mesh at K=1, so
//!    only the linear solver differs (sparse Cholesky vs complex direct LU —
//!    both direct); observed agreement is ~3e-15.
//! 2. **Optimized-design convergence.** A short projected topology run on the
//!    8x8 fixture reanalyzed over the K = 1..16 ladder must report
//!    `mesh_converged` at the default 1 % tolerance. The staircase-vs-fitted
//!    BL discrepancy is printed (not gated): the pole-tip corner singularity
//!    makes the topology-resolution BL a significant under-estimate, which is
//!    exactly what D5 set out to quantify.

use sonium_fem::optimization::InterpolationScheme;
use sonium_sim_speaker_modeling::axisymmetric_linear::{
    AxisymmetricLinearConfig, AxisymmetricLinearModel,
};
use sonium_sim_speaker_modeling::motor_topology::{
    MotorTopologyConfig, ProjectionContinuationConfig, optimize_motor_topology,
};
use sonium_sim_speaker_modeling::projected_reanalysis::{
    ProjectedReanalysisConfig, motor_design_elements, reanalyze_physical_design,
    reanalyze_projected_design,
};
use std::time::Instant;

fn fixture_driver() -> AxisymmetricLinearConfig {
    AxisymmetricLinearConfig {
        magnetic_radial_cells: 8,
        magnetic_axial_cells: 8,
        structural_radial_cells: 6,
        structural_axial_cells: 1,
        acoustic_axial_cells: 4,
        ..AxisymmetricLinearConfig::default()
    }
}

fn main() {
    let start = Instant::now();

    // Gate 1: golden reproduction on the original full-steel geometry.
    let driver = AxisymmetricLinearConfig::default();
    let golden = AxisymmetricLinearModel::build(driver.clone())
        .expect("golden model builds")
        .magnetic_bias
        .distributed_bl_tm;
    let design_elements = motor_design_elements(&driver).expect("design elements");
    let physical = vec![1.0; design_elements.len()];
    let full_steel = reanalyze_physical_design(
        &driver,
        &physical,
        InterpolationScheme::Ramp { penalty: 3.0 },
        0.5,
        &ProjectedReanalysisConfig {
            refinement_factors: vec![1],
            ..ProjectedReanalysisConfig::default()
        },
    )
    .expect("full-steel reanalysis");
    let golden_relative = (full_steel.bl_body_fitted_tm - golden).abs() / golden;
    println!(
        "gate 1 golden reproduction: golden={golden:.12} T·m, reanalysis={:.12} T·m, rel={golden_relative:.3e}",
        full_steel.bl_body_fitted_tm
    );
    assert!(
        golden_relative < 1.0e-9,
        "GATE 1 FAILED: K=1 body-fitted BL deviates from golden BL by {golden_relative:e}"
    );

    // Gate 2: optimized design on the 8x8 fixture, full convergence ladder.
    let driver = fixture_driver();
    let topology = MotorTopologyConfig {
        max_iterations: 12,
        tolerance: 1.0e-12,
        steel_volume_fraction: 0.65,
        projection: Some(ProjectionContinuationConfig {
            beta_initial: 1.0,
            beta_max: 16.0,
            eta: 0.5,
            continuation_interval: 4,
        }),
        ..MotorTopologyConfig::default()
    };
    let result = optimize_motor_topology(&driver, &topology).expect("topology run");
    let report = reanalyze_projected_design(
        &driver,
        &topology,
        &result,
        &ProjectedReanalysisConfig {
            refinement_factors: vec![1, 2, 4, 8, 16],
            ..ProjectedReanalysisConfig::default()
        },
    )
    .expect("projected reanalysis");
    println!(
        "gate 2 optimized design: steel cells {}/{}, gray fraction {:.4}, boundary loops {} ({} holes)",
        report.steel_element_count,
        result.design_elements.len(),
        result.gray_fraction,
        report.boundary_loops.len(),
        report.boundary_loops.iter().filter(|l| l.is_hole).count()
    );
    println!(
        "  staircase (interpolated, projected): {:.6} T·m",
        report.bl_projected_interpolated_tm
    );
    for level in &report.levels {
        println!(
            "  K={:2}: {:6} elements, BL={:.6} T·m, change={}",
            level.refinement_factor,
            level.num_elements,
            level.bl_tm,
            level
                .relative_change_from_previous
                .map(|c| format!("{c:.3e}"))
                .unwrap_or_else(|| "-".to_string())
        );
    }
    println!(
        "  body-fitted converged BL: {:.6} T·m (staircase-vs-fitted discrepancy {:.3e})",
        report.bl_body_fitted_tm, report.relative_discrepancy
    );
    assert!(
        report.mesh_converged,
        "GATE 2 FAILED: body-fitted BL not mesh-converged at 1% over the K=1..16 ladder"
    );

    println!(
        "SUCCESS qa-speaker-projected-reanalysis ({:.1?})",
        start.elapsed()
    );
}
