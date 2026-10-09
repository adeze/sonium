//! Physical CPU FEM mesh-convergence execution and evidence.
//!
//! This module is the execution boundary for a three-level refinement study.
//! Each level is solved on a full physical-room mesh, retains its nodal field,
//! and is compared with the exact listener coordinates used by that solve.
//! The returned report therefore cannot accidentally mix response data from
//! one mesh with metadata from another.

use super::fem_mesh_config::FemMeshConfig;
use super::fem_run_diagnostics::FemRunDiagnostics;
use super::run::run_fem_with_mesh_field_report;
use super::types::FemNodalField;
use super::validate::validate_mesh_config;
use crate::mesh_convergence::{MeshConvergenceReport, PhysicalMeshConvergenceLadder};
use anyhow::Result;
use sonium_xem::RoomSimulation;

/// One requested level in a physical FEM refinement study.
#[derive(Debug, Clone)]
pub struct FemMeshConvergenceSpec {
    /// Stable label written into the convergence evidence.
    pub label: String,
    /// Mesh and solver configuration for this level.
    pub mesh_config: FemMeshConfig,
}

impl FemMeshConvergenceSpec {
    /// Construct a labelled physical FEM refinement level.
    pub fn new(label: impl Into<String>, mesh_config: FemMeshConfig) -> Self {
        Self {
            label: label.into(),
            mesh_config,
        }
    }
}

/// Retained physical result and solver evidence for one refinement level.
#[derive(Debug)]
pub struct FemMeshConvergenceLevelRun {
    /// The requested label and configuration.
    pub spec: FemMeshConvergenceSpec,
    /// Full physical-room nodal field captured by the solve.
    pub field: FemNodalField,
    /// Per-frequency convergence and resource evidence.
    pub diagnostics: FemRunDiagnostics,
}

/// Complete three-level physical FEM refinement result.
#[derive(Debug)]
pub struct FemMeshConvergenceRun {
    /// Coarse-level retained field and diagnostics.
    pub coarse: FemMeshConvergenceLevelRun,
    /// Medium-level retained field and diagnostics.
    pub medium: FemMeshConvergenceLevelRun,
    /// Fine-level retained field and diagnostics.
    pub fine: FemMeshConvergenceLevelRun,
    /// Mesh-count, listener-position, topology, and solved-response evidence.
    pub ladder: PhysicalMeshConvergenceLadder,
    /// Adjacent-level response errors after the physical metadata checks.
    pub report: MeshConvergenceReport,
}

/// Validate the cheap part of a physical three-level FEM study before solving.
pub fn validate_fem_mesh_convergence_specs(
    specs: &[FemMeshConvergenceSpec; 3],
) -> Result<(), String> {
    let labels = specs
        .iter()
        .map(|spec| spec.label.trim())
        .collect::<Vec<_>>();
    if labels.iter().any(|label| label.is_empty()) {
        return Err("FEM mesh-convergence labels must be non-empty".into());
    }
    if labels[0] == labels[1] || labels[1] == labels[2] || labels[0] == labels[2] {
        return Err("FEM mesh-convergence labels must be distinct".into());
    }
    for spec in specs {
        validate_mesh_config(&spec.mesh_config).map_err(|error| error.to_string())?;
    }
    if !(specs[0].mesh_config.grading_factor > specs[1].mesh_config.grading_factor
        && specs[1].mesh_config.grading_factor > specs[2].mesh_config.grading_factor)
    {
        return Err(
            "FEM mesh-convergence grading factors must strictly decrease from coarse to fine"
                .into(),
        );
    }
    Ok(())
}

/// Validate convergence gates before allocating meshes or launching a solve.
fn validate_fem_mesh_convergence_limits(
    listener_position_tolerance_m: f64,
    frequency_tolerance_hz: f64,
    relative_floor_pa: f64,
    max_rms_relative_error: f64,
    max_max_relative_error: f64,
) -> Result<(), String> {
    if !listener_position_tolerance_m.is_finite() || listener_position_tolerance_m < 0.0 {
        return Err(
            "mesh-convergence listener position tolerance must be finite non-negative".into(),
        );
    }
    if !frequency_tolerance_hz.is_finite() || frequency_tolerance_hz < 0.0 {
        return Err("mesh-convergence frequency tolerance must be finite non-negative".into());
    }
    if !relative_floor_pa.is_finite() || relative_floor_pa <= 0.0 {
        return Err("mesh-convergence relative pressure floor must be finite positive".into());
    }
    if !max_rms_relative_error.is_finite() || max_rms_relative_error < 0.0 {
        return Err("mesh-convergence RMS error budget must be finite non-negative".into());
    }
    if !max_max_relative_error.is_finite() || max_max_relative_error < 0.0 {
        return Err("mesh-convergence maximum error budget must be finite non-negative".into());
    }
    Ok(())
}

/// Run and require a converged full-domain CPU FEM refinement ladder.
///
/// The three solves are deliberately sequential: each retains a complete
/// nodal field and diagnostics packet before the next level starts. This is
/// suitable for an explicit, bounded promotion gate; callers that need
/// candidate-level parallelism should schedule separate invocations outside
/// this evidence boundary.
pub fn run_fem_mesh_convergence_ladder(
    simulation: &RoomSimulation,
    specs: &[FemMeshConvergenceSpec; 3],
    listener_position_tolerance_m: f64,
    frequency_tolerance_hz: f64,
    relative_floor_pa: f64,
    max_rms_relative_error: f64,
    max_max_relative_error: f64,
) -> Result<FemMeshConvergenceRun> {
    validate_fem_mesh_convergence_specs(specs).map_err(anyhow::Error::msg)?;
    validate_fem_mesh_convergence_limits(
        listener_position_tolerance_m,
        frequency_tolerance_hz,
        relative_floor_pa,
        max_rms_relative_error,
        max_max_relative_error,
    )
    .map_err(anyhow::Error::msg)?;

    let (coarse_output, coarse_field, coarse_diagnostics) =
        run_fem_with_mesh_field_report(simulation, &specs[0].mesh_config)?;
    let (medium_output, medium_field, medium_diagnostics) =
        run_fem_with_mesh_field_report(simulation, &specs[1].mesh_config)?;
    let (fine_output, fine_field, fine_diagnostics) =
        run_fem_with_mesh_field_report(simulation, &specs[2].mesh_config)?;

    let ladder = PhysicalMeshConvergenceLadder::from_fem_nodal_fields(
        specs[0].label.clone(),
        &coarse_field,
        coarse_output,
        specs[1].label.clone(),
        &medium_field,
        medium_output,
        specs[2].label.clone(),
        &fine_field,
        fine_output,
    )
    .map_err(anyhow::Error::msg)?;
    let report = ladder
        .require_converged(
            listener_position_tolerance_m,
            frequency_tolerance_hz,
            relative_floor_pa,
            max_rms_relative_error,
            max_max_relative_error,
        )
        .map_err(anyhow::Error::msg)?;

    Ok(FemMeshConvergenceRun {
        coarse: FemMeshConvergenceLevelRun {
            spec: specs[0].clone(),
            field: coarse_field,
            diagnostics: coarse_diagnostics,
        },
        medium: FemMeshConvergenceLevelRun {
            spec: specs[1].clone(),
            field: medium_field,
            diagnostics: medium_diagnostics,
        },
        fine: FemMeshConvergenceLevelRun {
            spec: specs[2].clone(),
            field: fine_field,
            diagnostics: fine_diagnostics,
        },
        ladder,
        report,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fem_runner::fem_run_diagnostics::FemPhysicalEvidenceStatus;

    fn specs() -> [FemMeshConvergenceSpec; 3] {
        let mut coarse = FemMeshConfig::default();
        coarse.grading_factor = 10.0;
        coarse.fine_radius = 0.0;
        let mut medium = FemMeshConfig::default();
        medium.grading_factor = 2.0;
        medium.fine_radius = 0.3;
        [
            FemMeshConvergenceSpec::new("coarse", coarse),
            FemMeshConvergenceSpec::new("medium", medium),
            FemMeshConvergenceSpec::new("fine", FemMeshConfig::default()),
        ]
    }

    #[test]
    fn accepts_three_valid_labelled_specs_without_running_a_solver() {
        validate_fem_mesh_convergence_specs(&specs()).unwrap();
    }

    #[test]
    fn rejects_duplicate_or_blank_level_labels_before_expensive_work() {
        let mut duplicate = specs();
        duplicate[2].label = "medium".into();
        assert!(
            validate_fem_mesh_convergence_specs(&duplicate)
                .unwrap_err()
                .contains("distinct")
        );

        let mut blank = specs();
        blank[0].label = "  ".into();
        assert!(
            validate_fem_mesh_convergence_specs(&blank)
                .unwrap_err()
                .contains("non-empty")
        );
    }

    #[test]
    fn rejects_non_refining_grading_before_expensive_work() {
        let mut specs = specs();
        specs[1].mesh_config.grading_factor = 11.0;
        let error = validate_fem_mesh_convergence_specs(&specs).unwrap_err();
        assert!(error.contains("grading factors"));
    }

    #[test]
    fn rejects_invalid_mesh_configuration_before_expensive_work() {
        let mut invalid = specs();
        invalid[1].mesh_config.modal_damping_ratio = f64::NAN;
        assert!(
            validate_fem_mesh_convergence_specs(&invalid)
                .unwrap_err()
                .contains("modal_damping_ratio")
        );
    }

    #[test]
    fn rejects_invalid_convergence_limits_before_expensive_work() {
        let valid = (1.0e-12, 1.0e-9, 1.0e-12, 0.5, 0.5);
        let cases = [
            (
                (f64::NAN, valid.1, valid.2, valid.3, valid.4),
                "listener position tolerance",
            ),
            (
                (valid.0, -1.0, valid.2, valid.3, valid.4),
                "frequency tolerance",
            ),
            (
                (valid.0, valid.1, 0.0, valid.3, valid.4),
                "relative pressure floor",
            ),
            (
                (valid.0, valid.1, valid.2, f64::INFINITY, valid.4),
                "RMS error budget",
            ),
            (
                (valid.0, valid.1, valid.2, valid.3, -0.1),
                "maximum error budget",
            ),
        ];

        for (limits, expected) in cases {
            let error = validate_fem_mesh_convergence_limits(
                limits.0, limits.1, limits.2, limits.3, limits.4,
            )
            .unwrap_err();
            assert!(error.contains(expected), "{error}");
        }

        validate_fem_mesh_convergence_limits(valid.0, valid.1, valid.2, valid.3, valid.4).unwrap();
    }

    #[test]
    fn executes_bounded_three_level_cpu_ladder_and_retains_physical_fields() {
        use sonium_xem::{
            BoundaryConfig, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source,
            SurfaceConfig,
        };

        let mut simulation = RoomSimulation::new(
            RoomGeometry::Rectangular(RectangularRoom::new(2.4, 2.0, 1.8)),
            vec![Source::omnidirectional(Point3D::new(0.4, 0.35, 0.35), 1.0)],
            vec![Point3D::new(1.8, 1.4, 1.2)],
        );
        let matched = SurfaceConfig::Admittance {
            real: 1.0,
            imag: 0.0,
        };
        simulation.boundaries = BoundaryConfig {
            floor: matched.clone(),
            ceiling: matched.clone(),
            walls: matched,
            front_wall: None,
            back_wall: None,
            left_wall: None,
            right_wall: None,
        };
        simulation.frequencies = vec![100.0];

        let specs = specs();
        let run = run_fem_mesh_convergence_ladder(
            &simulation,
            &specs,
            1.0e-12,
            1.0e-9,
            1.0e-12,
            1.0,
            1.0,
        )
        .expect("bounded three-level CPU FEM ladder should execute");

        assert_eq!(run.coarse.field.frequencies, vec![100.0]);
        assert_eq!(run.medium.field.listener_positions.len(), 1);
        assert_eq!(run.fine.field.source_names, vec!["Source".to_string()]);
        assert!(!run.coarse.field.mesh.nodes.is_empty());
        assert!(run.coarse.diagnostics.require_converged().is_ok());
        assert!(run.medium.diagnostics.require_converged().is_ok());
        assert!(run.fine.diagnostics.require_converged().is_ok());
        assert_eq!(
            run.coarse.diagnostics.physical_evidence_status,
            FemPhysicalEvidenceStatus::Complete
        );
        assert_eq!(
            run.medium.diagnostics.physical_evidence_status,
            FemPhysicalEvidenceStatus::Complete
        );
        assert_eq!(
            run.fine.diagnostics.physical_evidence_status,
            FemPhysicalEvidenceStatus::Complete
        );
        for diagnostics in [
            &run.coarse.diagnostics,
            &run.medium.diagnostics,
            &run.fine.diagnostics,
        ] {
            assert!(diagnostics.frequency_stats.iter().all(|stats| {
                stats.operator_work.is_some()
                    && stats.physical_power.is_some()
                    && stats.boundary_flux.is_some()
            }));
        }
        assert!(run.ladder.coarse.node_count < run.ladder.medium.node_count);
        assert!(run.ladder.medium.node_count < run.ladder.fine.node_count);
        assert!(run.report.medium_to_fine_rms_relative_error.is_finite());
        assert!(run.report.medium_to_fine_max_relative_error.is_finite());
    }
}
