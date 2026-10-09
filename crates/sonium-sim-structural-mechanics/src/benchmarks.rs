//! Structural benchmark and QA evidence aggregation.
//!
//! The suite intentionally records what was measured, rather than collapsing
//! every case to a plausible scalar. Current references are analytical or
//! published-formula checks; external CalculiX/Code_Aster/FEniCSx execution is
//! represented as an explicit capability-gated status even though reproducible
//! adapters and versioned input decks are now available.

use crate::{
    AxisymmetricCylinderStudy, Beam2DStudy, BracketShapeObjective, BracketShapeOptimizationStudy,
    BracketStudy, BucklingStudy, CompositeLaminateStudy, ConstrainedTopologyOptimizationStudy,
    ElastoplasticBarStudy, ExternalBenchmarkDefinition, ExternalBenchmarkStatus,
    FatigueDamageStudy, FemFatigue2DStudy, FemTopology2DEnvelopeStudy, FemTopology2DMultiLoadStudy,
    FemTopology2DObjective, FemTopology2DStudy, Hyperelastic3DStudy, PhaseFieldDamageStudy,
    PhaseFieldForceControlledStudy, Plate2DStudy, RandomVibrationStudy, ReliabilityStudy,
    ShellMultiLoadStudy, Solid3DMultiLoadCase, Solid3DMultiLoadStudy, Solid3DStaticOptions,
    Solid3DStaticSolver, Solid3DStudy, Solid3DTopologyObjective, Solid3DTopologyStudy,
    StructuralExecutionProvenance, StructuralPolynomialOrder, StructuralRomStudy,
    ThermoelasticStudy, TopologyOptimizationStudy, TransientSdoFStudy, external_benchmark_status,
    external_solver_status_summary, render_external_decks, run_axisymmetric_cylinder,
    run_beam2d_modal, run_beam2d_static, run_bracket_shape_optimization,
    run_composite_laminate_static, run_constrained_topology_optimization, run_elastoplastic_bar,
    run_fatigue_damage, run_fem_fatigue_2d, run_fem_topology_2d, run_fem_topology_2d_envelope,
    run_fem_topology_2d_multi_load, run_fem_topology_2d_robust, run_hyperelastic_3d,
    run_linear_buckling, run_phase_field_damage, run_phase_field_damage_force_controlled,
    run_plate2d_static, run_random_vibration, run_reliability_study, run_shell_multi_load,
    run_solid3d_multi_load, run_solid3d_static, run_solid3d_static_with_options,
    run_solid3d_topology, run_static_bracket, run_structural_rom, run_thermoelastic_static,
    run_topology_optimization, run_transient_sdof,
};
#[cfg(feature = "meshlang")]
use crate::{MeshlangBracketParameters, compile_meshlang_bracket};
use serde::{Deserialize, Serialize};
use std::time::Instant;

/// Schema version for [`StructuralBenchmarkSuite`].
pub const STRUCTURAL_BENCHMARK_SCHEMA_VERSION: u32 = 2;

/// One benchmark's measured quality gates.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralBenchmarkCase {
    pub name: String,
    pub reference_source: String,
    pub computed_value: f64,
    pub reference_value: f64,
    pub relative_error: f64,
    pub tolerance: f64,
    pub solver_residual: Option<f64>,
    pub equilibrium_residual: Option<f64>,
    pub energy_balance_error: Option<f64>,
    pub gradient_error: Option<f64>,
    pub feasible: bool,
    pub converged: bool,
    pub iterations: usize,
    /// Deterministic sparse-work estimate, not a process RSS measurement.
    pub estimated_memory_bytes: usize,
    pub elapsed_ms: f64,
    pub passed: bool,
    pub applicability: String,
}

/// Complete structural benchmark/QA evidence report.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StructuralBenchmarkSuite {
    pub schema_version: u32,
    pub execution: StructuralExecutionProvenance,
    pub cases: Vec<StructuralBenchmarkCase>,
    pub all_passed: bool,
    /// External adapters may be available without having been executed.
    #[serde(default)]
    pub external_comparison: ExternalBenchmarkStatus,
    pub external_comparison_status: String,
    pub applicability_limits: Vec<String>,
}

/// Run the current analytical benchmark suite.
pub fn run_structural_benchmark_suite()
-> Result<StructuralBenchmarkSuite, crate::StructuralMechanicsError> {
    let mut cases = Vec::new();

    let mut bracket = BracketStudy::introductory_bracket();
    bracket.geometry.nx = 24;
    bracket.geometry.ny = 6;
    let started = Instant::now();
    let result = run_static_bracket(&bracket)?;
    cases.push(case_report(
        "bracket-static-smoke",
        "Euler-Bernoulli cantilever estimate",
        result.average_tip_deflection_m.abs(),
        result.beam_theory_tip_deflection_m.abs(),
        result.beam_theory_relative_error,
        0.35,
        Some(result.solver_residual),
        Some(result.free_equilibrium_residual_n),
        Some(result.energy_balance_relative_error),
        None,
        result.converged,
        result.solver_iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "2D P1 plane-stress bracket; centroid stress is diagnostic only.",
    ));

    let mut axisymmetric = AxisymmetricCylinderStudy::introductory();
    axisymmetric.geometry.radial_elements = 16;
    let started = Instant::now();
    let result = run_axisymmetric_cylinder(&axisymmetric)?;
    let axisymmetric_error = result
        .max_radial_displacement_relative_error
        .max(result.max_radial_stress_relative_error)
        .max(result.max_hoop_stress_relative_error);
    cases.push(case_report(
        "axisymmetric-cylinder-plane-strain",
        "Closed-form Lamé thick-cylinder displacement and stress fields",
        axisymmetric_error,
        0.0,
        axisymmetric_error,
        0.10,
        Some(result.solver_residual),
        Some(result.free_equilibrium_residual_n),
        Some(result.energy_balance_relative_error),
        None,
        result.converged,
        result.solver_iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "2D meridional P1 axisymmetric elasticity with plane-strain axial constraint; general axisymmetric and nonlinear behavior are not claimed.",
    ));

    let started = Instant::now();
    let result = run_composite_laminate_static(&CompositeLaminateStudy::introductory())?;
    let strength_gate = if result.max_tsai_hill_index < 1.0 && result.max_stress_index < 1.0 {
        0.0
    } else {
        1.0
    };
    let composite_error = result
        .analytical_relative_error
        .max(result.load_integration_relative_error)
        .max(strength_gate);
    cases.push(case_report(
        "composite-laminate-orthotropic",
        "Laminate-A analytical unidirectional tension displacement and ply-strength gate",
        composite_error,
        0.0,
        composite_error,
        1.0e-8,
        Some(result.solver_residual),
        Some(result.free_equilibrium_residual_n.max(result.force_balance_residual_n)),
        Some(result.energy_balance_relative_error),
        None,
        result.converged,
        result.solver_iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "P1 orthotropic laminate membrane with material-axis ply stress recovery; unsymmetric bending-coupled layups and general anisotropic solids are not claimed.",
    ));

    let mut bracket_shape = BracketShapeOptimizationStudy::introductory(bracket.clone());
    bracket_shape.base.geometry.nx = 4;
    bracket_shape.base.geometry.ny = 1;
    bracket_shape.iterations = 1;
    bracket_shape.step_size = 0.10;
    bracket_shape.derivative_step_fraction = 2.0e-4;
    bracket_shape.objective = BracketShapeObjective::FrequencyTarget {
        target_frequency_hz: 1_000.0,
        frequency_weight_j_per_hz2: 1.0e-8,
    };
    let started = Instant::now();
    let result = run_bracket_shape_optimization(&bracket_shape)?;
    let objective_gate: f64 = if result.final_objective_j <= result.initial_objective_j {
        0.0
    } else {
        1.0
    };
    let gradient_gate = result.gradient_taylor_relative_error;
    let shape_gate = objective_gate.max(gradient_gate);
    cases.push(case_report(
        "bracket-shape-frequency-target",
        "Assembled first-bending-frequency target on a body-fitted bracket",
        shape_gate,
        0.0,
        shape_gate,
        1.0e-6,
        Some(result.final_static_result.solver_residual),
        Some(result.final_static_result.free_equilibrium_residual_n),
        Some(result.final_static_result.energy_balance_relative_error),
        Some(result.gradient_taylor_relative_error),
        result.final_static_result.converged,
        result.iterations,
        estimate_memory(
            result.final_static_result.dofs,
            result.final_static_result.elements,
        ),
        started.elapsed().as_secs_f64() * 1_000.0,
        "2D P1 body-fitted shape objective; frequency targeting is validated, while general shape derivatives are not claimed.",
    ));

    let mut bracket_buckling = bracket_shape.clone();
    bracket_buckling.base.loads.axial_prestress_force_n = -250.0;
    bracket_buckling.objective = BracketShapeObjective::BucklingTarget {
        target_load_factor: 100.0,
        buckling_weight_j_per_factor2: 1.0e-6,
    };
    let started = Instant::now();
    let result = run_bracket_shape_optimization(&bracket_buckling)?;
    let buckling_gate: f64 = if result.final_objective_j <= result.initial_objective_j {
        0.0
    } else {
        1.0
    };
    let shape_gate = buckling_gate.max(result.gradient_taylor_relative_error);
    cases.push(case_report(
        "bracket-shape-buckling-target",
        "Linearized buckling-factor target on a compressive body-fitted bracket",
        shape_gate,
        0.0,
        shape_gate,
        1.0e-6,
        Some(result.final_static_result.solver_residual),
        Some(result.final_static_result.free_equilibrium_residual_n),
        Some(result.final_static_result.energy_balance_relative_error),
        Some(result.gradient_taylor_relative_error),
        result.final_static_result.converged,
        result.iterations,
        estimate_memory(
            result.final_static_result.dofs,
            result.final_static_result.elements,
        ),
        started.elapsed().as_secs_f64() * 1_000.0,
        "2D P1 body-fitted pre-buckling target; nonlinear post-buckling, imperfections, and general geometric stiffness are not claimed.",
    ));

    let mut beam = Beam2DStudy::introductory();
    beam.geometry.elements = 8;
    let started = Instant::now();
    let result = run_beam2d_static(&beam)?;
    cases.push(case_report(
        "beam2d-static",
        "Euler-Bernoulli/Timoshenko analytical tip deflection",
        result.tip_deflection_m.abs(),
        result.analytical_tip_deflection_m.abs(),
        result.analytical_relative_error,
        0.10,
        Some(result.solver_residual),
        Some(result.free_equilibrium_residual_n),
        Some(result.energy_balance_relative_error),
        None,
        result.converged,
        result.solver_iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "2D beam idealisation; no shear-locking or 3D effects beyond the selected formulation.",
    ));

    let mut beam_modal = Beam2DStudy::introductory();
    beam_modal.geometry.elements = 4;
    beam_modal.modal_modes = 2;
    let started = Instant::now();
    let result = run_beam2d_modal(&beam_modal)?;
    let beta = 1.875_104_068_711_961_f64;
    let reference = beta.powi(2)
        * (beam_modal.material.youngs_modulus_pa * beam_modal.geometry.second_moment_m4()
            / (beam_modal.material.density_kg_m3
                * beam_modal.geometry.area_m2()
                * beam_modal.geometry.length_m.powi(4)))
        .sqrt()
        / (2.0 * std::f64::consts::PI);
    let computed = result.frequencies_hz[0];
    cases.push(case_report(
        "beam2d-modal",
        "Euler-Bernoulli cantilever first natural frequency",
        computed,
        reference,
        (computed - reference).abs() / reference,
        0.10,
        Some(result.residual_norms.iter().copied().fold(0.0, f64::max)),
        None,
        Some(result.mass_normalization_error),
        Some(result.orthogonality_error),
        result.converged,
        result.solver_iterations,
        estimate_memory(result.modes.len() * result.modes[0].len() * 3, beam_modal.geometry.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "Assembled Euler-Bernoulli beam modal solve; first-mode analytical reference and mass/orthogonality gates.",
    ));

    let mut plate = Plate2DStudy::introductory();
    plate.geometry.nx = 40;
    plate.geometry.ny = 40;
    let started = Instant::now();
    let result = run_plate2d_static(&plate)?;
    cases.push(case_report(
        "plate2d-simply-supported",
        "Navier sinusoidal Mindlin plate center deflection",
        result.center_deflection_m.abs(),
        result.analytical_center_deflection_m.abs(),
        result.analytical_relative_error,
        0.02,
        Some(result.solver_residual),
        Some(result.free_equilibrium_residual_n),
        Some(result.energy_balance_relative_error),
        Some(result.load_integration_relative_error),
        result.converged,
        result.solver_iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "2D P1 Mindlin plate with DSG shear treatment; analytical Navier reference, not a shell certification.",
    ));

    let mut shell_multi = ShellMultiLoadStudy::introductory();
    shell_multi.base.geometry.nx = 4;
    shell_multi.base.geometry.ny = 2;
    let started = Instant::now();
    let result = run_shell_multi_load(&shell_multi)?;
    let shell_multi_gate = result.factorization_reused && result.converged;
    let shell_multi_gate_value = if shell_multi_gate { 1.0 } else { 0.0 };
    cases.push(case_report(
        "shell-multi-load",
        "Shared-factorization flat-shell pressure-case verification",
        shell_multi_gate_value,
        1.0,
        1.0 - shell_multi_gate_value,
        0.0,
        Some(
            result
                .cases
                .iter()
                .map(|case| case.solver_residual)
                .fold(0.0, f64::max),
        ),
        Some(
            result
                .cases
                .iter()
                .map(|case| {
                    case.free_equilibrium_residual_n
                        .max(case.force_balance_residual_n)
                })
                .fold(0.0, f64::max),
        ),
        Some(
            result
                .cases
                .iter()
                .map(|case| case.energy_balance_relative_error)
                .fold(0.0, f64::max),
        ),
        None,
        shell_multi_gate,
        result.cases.len(),
        result.factorization_memory_bytes,
        started.elapsed().as_secs_f64() * 1_000.0,
        "Flat P1 triangular CST membrane plus DKT bending shell with shared factorization across pressure RHS cases; shell stress recovery and nonlinear/curved behavior are not claimed.",
    ));

    let thermal = ThermoelasticStudy::introductory();
    let started = Instant::now();
    let result = run_thermoelastic_static(&thermal)?;
    let reference =
        result.expected_corner_displacement_m[0].hypot(result.expected_corner_displacement_m[1]);
    let computed =
        result.actual_corner_displacement_m[0].hypot(result.actual_corner_displacement_m[1]);
    cases.push(case_report(
        "thermoelastic-free-expansion",
        "Uniform-temperature affine free-expansion solution",
        computed,
        reference,
        (computed - reference).abs() / reference.max(1.0e-30),
        1.0e-8,
        Some(result.solver_residual),
        Some(result.free_equilibrium_residual_n),
        Some(result.energy_balance_relative_error),
        Some(result.max_affine_displacement_error_m),
        result.converged,
        result.solver_iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "Plane-stress thermoelastic patch with uniform eigenstrain; free expansion and zero-stress diagnostic.",
    ));

    let solid = Solid3DStudy::introductory();
    let started = Instant::now();
    let result = run_solid3d_static(&solid)?;
    let reference = result.applied_force_n[0].abs() * solid.geometry.length_m
        / (solid.material.youngs_modulus_pa * solid.geometry.loaded_face_area_m2());
    let relative_error = (result.max_displacement_m - reference).abs() / reference.max(1.0e-30);
    cases.push(case_report(
        "solid3d-axial",
        "Uniform axial bar displacement estimate",
        result.max_displacement_m,
        reference,
        relative_error,
        0.20,
        Some(result.solver_residual),
        Some(result.free_equilibrium_residual_n),
        Some(result.energy_balance_relative_error),
        None,
        result.converged,
        result.solver_iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "P1 tetrahedral 3D solid with face traction; coarse mesh estimate.",
    ));

    let mut operator_study = solid.clone();
    operator_study.geometry.nx = 1;
    operator_study.geometry.ny = 1;
    operator_study.geometry.nz = 1;
    let started = Instant::now();
    let result = run_solid3d_static_with_options(
        &operator_study,
        Solid3DStaticOptions {
            polynomial_order: StructuralPolynomialOrder::Linear,
            solver: Solid3DStaticSolver::AssembledAmgFgmres,
        },
    )?;
    let amg_gate = result.converged
        && result.solver == Solid3DStaticSolver::AssembledAmgFgmres
        && result.free_equilibrium_residual_n < 1.0e-6
        && result.force_balance_residual_n < 1.0e-6
        && result.energy_balance_relative_error < 1.0e-9;
    let amg_gate_value = if amg_gate { 1.0 } else { 0.0 };
    cases.push(case_report(
        "solid3d-static-amg-operator",
        "Explicit assembled elasticity AMG-FGMRES operator path",
        amg_gate_value,
        1.0,
        1.0 - amg_gate_value,
        0.0,
        Some(result.solver_residual),
        Some(result.free_equilibrium_residual_n.max(result.force_balance_residual_n)),
        Some(result.energy_balance_relative_error),
        None,
        amg_gate,
        result.solver_iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "Small P1 CPU/f64 operator-selection smoke for assembled elasticity AMG-FGMRES; no large-scale performance or accelerator parity is claimed.",
    ));

    let started = Instant::now();
    let result = run_solid3d_static_with_options(
        &operator_study,
        Solid3DStaticOptions {
            polynomial_order: StructuralPolynomialOrder::Quadratic,
            solver: Solid3DStaticSolver::MatrixFreeJacobiFgmres,
        },
    )?;
    let matrix_free_gate = result.converged
        && result.solver == Solid3DStaticSolver::MatrixFreeJacobiFgmres
        && result.polynomial_order == StructuralPolynomialOrder::Quadratic
        && result.free_equilibrium_residual_n < 1.0e-6
        && result.force_balance_residual_n < 1.0e-6
        && result.energy_balance_relative_error < 1.0e-9;
    let matrix_free_gate_value = if matrix_free_gate { 1.0 } else { 0.0 };
    cases.push(case_report(
        "solid3d-static-matrix-free-p2",
        "Explicit affine-tetrahedral P2 matrix-free Jacobi-FGMRES operator path",
        matrix_free_gate_value,
        1.0,
        1.0 - matrix_free_gate_value,
        0.0,
        Some(result.solver_residual),
        Some(result.free_equilibrium_residual_n.max(result.force_balance_residual_n)),
        Some(result.energy_balance_relative_error),
        None,
        matrix_free_gate,
        result.solver_iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "Small P2 CPU/f64 matrix-free operator-selection smoke with residual/equilibrium/energy gates; no large-scale performance or accelerator parity is claimed.",
    ));

    let solid_multi = Solid3DMultiLoadStudy {
        base: solid.clone(),
        load_cases: vec![
            Solid3DMultiLoadCase {
                name: "axial-service".to_string(),
                force_n: [100.0, 0.0, 0.0],
            },
            Solid3DMultiLoadCase {
                name: "combined-service".to_string(),
                force_n: [60.0, 20.0, -10.0],
            },
        ],
    };
    let started = Instant::now();
    let result = run_solid3d_multi_load(&solid_multi)?;
    let multi_load_gate = result.factorization_reused && result.converged;
    let multi_load_gate_value = if multi_load_gate { 1.0 } else { 0.0 };
    cases.push(case_report(
        "solid3d-multi-load",
        "Shared-factorization P1 tetrahedral load-case verification",
        multi_load_gate_value,
        1.0,
        1.0 - multi_load_gate_value,
        0.0,
        Some(
            result
                .cases
                .iter()
                .map(|case| case.solver_residual)
                .fold(0.0, f64::max),
        ),
        Some(
            result
                .cases
                .iter()
                .map(|case| {
                    case.free_equilibrium_residual_n
                        .max(case.force_balance_residual_n)
                })
                .fold(0.0, f64::max),
        ),
        Some(
            result
                .cases
                .iter()
                .map(|case| case.energy_balance_relative_error)
                .fold(0.0, f64::max),
        ),
        None,
        multi_load_gate,
        result.cases.len(),
        result.factorization_memory_bytes,
        started.elapsed().as_secs_f64() * 1_000.0,
        "Assembled linear P1 3D solid with one constrained sparse factorization reused across traction RHS cases; quadratic/matrix-free and shell multi-load are not claimed.",
    ));

    let mut solid_topology = Solid3DTopologyStudy::introductory();
    solid_topology.base.geometry.nx = 2;
    solid_topology.base.geometry.ny = 1;
    solid_topology.base.geometry.nz = 1;
    solid_topology.volume_fraction = 0.90;
    solid_topology.iterations = 20;
    solid_topology.move_limit = 1.0e-4;
    solid_topology.convergence_tolerance = 1.0e-3;
    solid_topology.allowable_stress_pa = 1.0e14;
    let started = Instant::now();
    let result = run_solid3d_topology(&solid_topology)?;
    let topology_gate = result.converged
        && result.connected
        && result.stress_feasible
        && result.modal_feasible
        && result.first_frequency_hz.is_finite()
        && result.buckling_feasible
        && result.morphology_feasible
        && result.body_fitted_reanalysis
        && result.body_fitted_feasible
        && result.factorization_reused_across_cases
        && result.cases.iter().all(|case| case.converged);
    let topology_gate_value = if topology_gate { 1.0 } else { 0.0 };
    cases.push(case_report(
        "solid3d-topology-simp",
        "Filtered/projected assembled linear-tetrahedral SIMP multi-load compliance, frequency, and static gates",
        topology_gate_value,
        1.0,
        1.0 - topology_gate_value,
        0.0,
        Some(
            result
                .cases
                .iter()
                .map(|case| case.solver_residual)
                .fold(0.0, f64::max),
        ),
        Some(
            result
                .cases
                .iter()
                .map(|case| {
                    case.free_equilibrium_residual_n
                        .max(case.force_balance_residual_n)
                })
                .fold(0.0, f64::max),
        ),
        Some(
            result
                .cases
                .iter()
                .map(|case| case.energy_balance_relative_error)
                .fold(0.0, f64::max),
        ),
        None,
        topology_gate,
        result.iterations.max(1),
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "Bounded CPU/f64 assembled linear-tetrahedral 3D SIMP path with exact maximum-compliance active-case gradients, density filtering/projection, body-fitted reanalysis, final-density first-frequency evidence, and an optional x-compressive linearized buckling gate; nonlinear, mixed-sign geometric stiffness, and accelerator execution are not claimed.",
    ));

    let mut frequency_objective = solid_topology.clone();
    frequency_objective.objective = Solid3DTopologyObjective::FirstFrequency;
    frequency_objective.iterations = 1;
    let started = Instant::now();
    let result = run_solid3d_topology(&frequency_objective)?;
    let frequency_gate = result.final_objective_value.is_finite()
        && result.first_frequency_hz.is_finite()
        && result.first_frequency_hz > 0.0
        && result.objective_improved
        && result.objective_gradient_taylor_relative_error < 1.0e-3
        && result.body_fitted_feasible;
    let frequency_gate_value = if frequency_gate { 1.0 } else { 0.0 };
    cases.push(case_report(
        "solid3d-topology-frequency-objective",
        "Assembled generalized mass/stiffness first-frequency objective and sensitivity gate",
        frequency_gate_value,
        1.0,
        1.0 - frequency_gate_value,
        0.0,
        Some(
            result
                .cases
                .iter()
                .map(|case| case.solver_residual)
                .fold(0.0, f64::max),
        ),
        Some(
            result
                .cases
                .iter()
                .map(|case| {
                    case.free_equilibrium_residual_n
                        .max(case.force_balance_residual_n)
                })
                .fold(0.0, f64::max),
        ),
        Some(
            result
                .cases
                .iter()
                .map(|case| case.energy_balance_relative_error)
                .fold(0.0, f64::max),
        ),
        Some(result.objective_gradient_taylor_relative_error),
        frequency_gate,
        result.iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "Assembled linear-tetrahedral 3D topology with a first-frequency objective differentiated through the generalized eigenproblem; CPU/f64 reference only.",
    ));

    let mut buckling_objective = solid_topology.clone();
    buckling_objective.objective = Solid3DTopologyObjective::BucklingLoadFactor;
    buckling_objective.load_cases[0].force_n = [-100.0, 0.0, 0.0];
    buckling_objective.base.loads.force_n = [-100.0, 0.0, 0.0];
    buckling_objective.iterations = 1;
    let started = Instant::now();
    let result = run_solid3d_topology(&buckling_objective)?;
    let buckling_gate = result.final_objective_value.is_finite()
        && result
            .buckling_load_factor
            .is_some_and(|value| value.is_finite())
        && result.objective_improved
        && result.objective_gradient_taylor_relative_error < 5.0e-2
        && result.body_fitted_feasible;
    let buckling_gate_value = if buckling_gate { 1.0 } else { 0.0 };
    cases.push(case_report(
        "solid3d-topology-buckling-objective",
        "Re-solved linearized initial-stress buckling objective and finite-difference gate",
        buckling_gate_value,
        1.0,
        1.0 - buckling_gate_value,
        0.0,
        Some(
            result
                .cases
                .iter()
                .map(|case| case.solver_residual)
                .fold(0.0, f64::max),
        ),
        Some(
            result
                .cases
                .iter()
                .map(|case| {
                    case.free_equilibrium_residual_n
                        .max(case.force_balance_residual_n)
                })
                .fold(0.0, f64::max),
        ),
        Some(
            result
                .cases
                .iter()
                .map(|case| case.energy_balance_relative_error)
                .fold(0.0, f64::max),
        ),
        Some(result.objective_gradient_taylor_relative_error),
        buckling_gate,
        result.iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "Assembled linear-tetrahedral 3D topology with a compressive linearized buckling objective; bounded finite-difference sensitivity, not nonlinear post-buckling certification.",
    ));

    let hyperelastic = Hyperelastic3DStudy::introductory();
    let started = Instant::now();
    let result = run_hyperelastic_3d(&hyperelastic)?;
    let constitutive_gate = result
        .constitutive_gradient_error
        .max((result.taylor_remainder_ratio - 0.25).abs());
    cases.push(case_report(
        "solid3d-hyperelastic",
        "Homogeneous displacement-controlled Neo-Hookean extension",
        result.max_displacement_m,
        hyperelastic.prescribed_extension_m,
        (result.max_displacement_m - hyperelastic.prescribed_extension_m).abs()
            / hyperelastic.prescribed_extension_m,
        0.05,
        Some(result.final_residual),
        Some(result.final_residual),
        Some(result.relative_energy_error),
        Some(constitutive_gate),
        result.converged,
        result.newton_iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "P1 tetrahedral total-Lagrangian Neo-Hookean solid; displacement control and affine Taylor gates, not post-buckling or fracture.",
    ));

    let phase_field = PhaseFieldDamageStudy::introductory();
    let started = Instant::now();
    let result = run_phase_field_damage(&phase_field)?;
    let damage_reference = result.seed_damage.max(1.0e-12);
    cases.push(case_report(
        "phase-field-damage-staggered",
        "Bounded irreversible phase-field damage path with seeded band",
        result.max_damage,
        damage_reference,
        (result.max_damage - damage_reference).abs() / damage_reference,
        0.50,
        Some(result.final_mechanics_residual),
        Some(result.final_free_equilibrium_residual),
        Some(result.final_crack_surface_energy_j),
        Some(result.max_bound_violation),
        result.converged,
        result.load_steps,
        estimate_memory(result.mechanics_dofs + result.damage_dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "2D P1 staggered linear elasticity/phase-field damage; seeded displacement-controlled evidence, not force-controlled crack propagation.",
    ));

    let force_phase_field = PhaseFieldForceControlledStudy::introductory();
    let started = Instant::now();
    let result = run_phase_field_damage_force_controlled(&force_phase_field)?;
    let damage_reference = result.seed_damage.max(1.0e-12);
    let damage_error = (result.max_damage - damage_reference).abs() / damage_reference;
    let force_history_monotone = result
        .applied_force_history_n
        .windows(2)
        .all(|window| window[1] > window[0]);
    cases.push(case_report(
        "phase-field-damage-force-controlled",
        "Ramped right-edge traction with irreversible seeded damage",
        result.max_damage,
        damage_reference,
        damage_error,
        0.50,
        Some(result.final_mechanics_residual),
        Some(result.final_free_equilibrium_residual),
        Some(result.final_crack_surface_energy_j),
        Some(result.max_bound_violation),
        result.converged && force_history_monotone,
        result.load_steps,
        estimate_memory(result.mechanics_dofs + result.damage_dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "2D P1 force-controlled traction continuation; stable seeded damage evidence, not arc-length post-failure propagation.",
    ));

    let mut buckling = BucklingStudy::introductory();
    buckling.geometry.elements = 8;
    let started = Instant::now();
    let result = run_linear_buckling(&buckling)?;
    cases.push(case_report(
        "beam-buckling-pinned",
        "Euler critical load",
        result.eigenvalue_n,
        result.analytical_euler_load_n,
        result.analytical_relative_error,
        0.10,
        Some(result.residual_norm),
        None,
        None,
        None,
        result.converged,
        result.solver_iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "Linear pre-buckling eigenproblem; no post-buckling or imperfection sensitivity.",
    ));

    let plastic = ElastoplasticBarStudy::introductory();
    let started = Instant::now();
    let result = run_elastoplastic_bar(&plastic)?;
    let stress = plastic.max_force_n / plastic.area_m2;
    let reference = plastic.length_m
        * (stress / plastic.youngs_modulus_pa
            + (stress - plastic.yield_stress_pa) / plastic.isotropic_hardening_pa);
    let relative_error = (result.final_displacement_m - reference).abs() / reference;
    let iterations = result.steps.iter().map(|step| step.newton_iterations).sum();
    cases.push(case_report(
        "bar-elastoplastic",
        "1D return-mapping closed form",
        result.final_displacement_m,
        reference,
        relative_error,
        1.0e-8,
        result.steps.last().map(|step| step.residual_n),
        Some(result.max_equilibrium_residual_n),
        None,
        None,
        result.steps.iter().all(|step| step.converged),
        iterations,
        estimate_memory(plastic.elements + 1, plastic.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "Monotone small-strain 1D elastoplastic bar; no multiaxial plasticity.",
    ));

    let topology = TopologyOptimizationStudy::introductory();
    let started = Instant::now();
    let result = run_topology_optimization(&topology)?;
    let relative_error =
        (result.final_volume_fraction - topology.volume_fraction).abs() / topology.volume_fraction;
    cases.push(case_report(
        "topology-volume-connectivity",
        "Prescribed volume fraction and minimum-density feasibility",
        result.final_volume_fraction,
        topology.volume_fraction,
        relative_error,
        1.0e-6,
        None,
        None,
        None,
        None,
        result.final_connected,
        result.iterations,
        estimate_memory(
            result.final_physical_density.len(),
            result.final_physical_density.len(),
        ),
        started.elapsed().as_secs_f64() * 1_000.0,
        "1D filtered/projected SIMP surrogate; not a 3D manufacturing certification.",
    ));

    let mut fem_topology = FemTopology2DStudy::introductory();
    fem_topology.geometry.nx = 6;
    fem_topology.geometry.ny = 2;
    fem_topology.volume_fraction = 0.95;
    fem_topology.move_limit = 0.01;
    fem_topology.continuation_steps = 1;
    fem_topology.iterations_per_stage = 3;
    fem_topology.filter_radius_m = 0.02;
    fem_topology.maximum_overhang_slope = 100.0;
    let started = Instant::now();
    let result = run_fem_topology_2d(&fem_topology)?;
    cases.push(case_report(
        "fem-topology-2d-reference",
        "P1 FEM compliance with projected/body-fitted reanalysis",
        result.body_fitted_reanalysis_compliance_j,
        result.projected_reanalysis_compliance_j,
        result.body_fitted_reanalysis_relative_error,
        0.25,
        Some(result.solver_residual),
        Some(result.free_equilibrium_residual_n),
        Some(result.energy_balance_relative_error),
        Some(result.gradient_taylor_relative_error),
        result.connected
            && result.buckling_feasible
            && result.minimum_feature_feasible
            && result.overhang_feasible
            && result.stress_feasible
            && result.reanalysis_feasible,
        result.iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "P1 triangular plane-stress FEM topology reference; compressive buckling is only active for x-directed compression.",
    ));

    let mut frequency_objective = fem_topology.clone();
    frequency_objective.objective = FemTopology2DObjective::FirstFrequency;
    frequency_objective.iterations_per_stage = 1;
    let started = Instant::now();
    let result = run_fem_topology_2d(&frequency_objective)?;
    let frequency_gate = result.final_objective_value.is_finite()
        && result.first_frequency_hz.is_finite()
        && result.first_frequency_hz > 0.0
        && result.objective_improved
        && result.gradient_taylor_relative_error < 1.0e-3
        && result.reanalysis_feasible;
    let frequency_gate_value = if frequency_gate { 1.0 } else { 0.0 };
    cases.push(case_report(
        "fem-topology-2d-frequency-objective",
        "Assembled generalized mass/stiffness first-frequency objective and sensitivity gate",
        frequency_gate_value,
        1.0,
        1.0 - frequency_gate_value,
        0.0,
        Some(result.solver_residual),
        Some(result.free_equilibrium_residual_n),
        Some(result.energy_balance_relative_error),
        Some(result.gradient_taylor_relative_error),
        frequency_gate,
        result.iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "P1 plane-stress topology with an assembled first-frequency objective; CPU/f64 reference only.",
    ));

    let mut buckling_objective = fem_topology.clone();
    buckling_objective.objective = FemTopology2DObjective::BucklingLoadFactor;
    buckling_objective.edge_force_n = [-120.0, 0.0];
    buckling_objective.iterations_per_stage = 1;
    let started = Instant::now();
    let result = run_fem_topology_2d(&buckling_objective)?;
    let buckling_gate = result.final_objective_value.is_finite()
        && result
            .buckling_load_factor
            .is_some_and(|value| value.is_finite())
        && result.objective_improved
        && result.gradient_taylor_relative_error < 5.0e-2
        && result.reanalysis_feasible;
    let buckling_gate_value = if buckling_gate { 1.0 } else { 0.0 };
    cases.push(case_report(
        "fem-topology-2d-buckling-objective",
        "Re-solved linearized initial-stress buckling objective and finite-difference gate",
        buckling_gate_value,
        1.0,
        1.0 - buckling_gate_value,
        0.0,
        Some(result.solver_residual),
        Some(result.free_equilibrium_residual_n),
        Some(result.energy_balance_relative_error),
        Some(result.gradient_taylor_relative_error),
        buckling_gate,
        result.iterations,
        estimate_memory(result.dofs, result.elements),
        started.elapsed().as_secs_f64() * 1_000.0,
        "P1 plane-stress topology with a compressive linearized buckling objective; finite-difference sensitivity, not nonlinear post-buckling certification.",
    ));

    let mut multi_load = FemTopology2DMultiLoadStudy::introductory();
    multi_load.topology.geometry.nx = 6;
    multi_load.topology.geometry.ny = 2;
    multi_load.topology.volume_fraction = 0.95;
    multi_load.topology.move_limit = 0.01;
    multi_load.topology.continuation_steps = 1;
    multi_load.topology.iterations_per_stage = 3;
    multi_load.topology.filter_radius_m = 0.02;
    multi_load.topology.maximum_overhang_slope = 100.0;
    let started = Instant::now();
    let result = run_fem_topology_2d_multi_load(&multi_load)?;
    let multi_load_gate = if result.factorization_reused && result.converged {
        1.0
    } else {
        0.0
    };
    let multi_load_solver_residual = result
        .cases
        .iter()
        .map(|case| case.solver_residual)
        .fold(0.0, f64::max);
    let multi_load_equilibrium_residual = result
        .cases
        .iter()
        .map(|case| {
            case.free_equilibrium_residual_n
                .max(case.force_balance_residual_n)
        })
        .fold(0.0, f64::max);
    let multi_load_energy_error = result
        .cases
        .iter()
        .map(|case| case.energy_balance_relative_error)
        .fold(0.0, f64::max);
    cases.push(case_report(
        "fem-topology-2d-multi-load",
        "Shared-factorization load-case verification on accepted FEM topology",
        multi_load_gate,
        1.0,
        1.0 - multi_load_gate,
        0.0,
        Some(multi_load_solver_residual),
        Some(multi_load_equilibrium_residual),
        Some(multi_load_energy_error),
        None,
        result.factorization_reused && result.converged,
        result.cases.len(),
        result.factorization_memory_bytes,
        started.elapsed().as_secs_f64() * 1_000.0,
        "Primary-load topology optimization followed by multi-load FEM verification; envelope optimization is reported separately.",
    ));

    let mut envelope = FemTopology2DEnvelopeStudy::introductory();
    envelope.topology = multi_load.topology.clone();
    envelope.topology.edge_force_n = envelope.load_cases[0].edge_force_n;
    let started = Instant::now();
    let result = run_fem_topology_2d_envelope(&envelope)?;
    let envelope_gate = result.factorization_reused_across_cases
        && result.connected
        && result.reanalysis_feasible
        && result.cases.iter().all(|case| case.converged)
        && result.final_max_compliance_j <= result.initial_max_compliance_j;
    let envelope_gate_value = if envelope_gate { 1.0 } else { 0.0 };
    cases.push(case_report(
        "fem-topology-2d-envelope",
        "Worst-case compliance topology objective across shared load cases",
        envelope_gate_value,
        1.0,
        1.0 - envelope_gate_value,
        0.0,
        Some(
            result
                .cases
                .iter()
                .map(|case| case.solver_residual)
                .fold(0.0, f64::max),
        ),
        Some(
            result
                .cases
                .iter()
                .map(|case| {
                    case.free_equilibrium_residual_n
                        .max(case.force_balance_residual_n)
                })
                .fold(0.0, f64::max),
        ),
        Some(
            result
                .cases
                .iter()
                .map(|case| case.energy_balance_relative_error)
                .fold(0.0, f64::max),
        ),
        Some(result.gradient_taylor_relative_error),
        envelope_gate,
        result.iterations,
        result.factorization_memory_bytes,
        started.elapsed().as_secs_f64() * 1_000.0,
        "Exact max-compliance subgradient topology optimization on a P1 plane-stress mesh; CPU/f64 reference only.",
    ));

    let robust = crate::FemTopology2DRobustStudy {
        envelope,
        allowable_compliance_j: 1.0,
        ..crate::FemTopology2DRobustStudy::introductory()
    };
    let started = Instant::now();
    let result = run_fem_topology_2d_robust(&robust)?;
    let robust_gate = result.factorization_reused_within_scenario
        && result.morphology_feasible
        && result.uncertainty_feasible
        && result.nominal.gradient_taylor_relative_error.is_finite();
    let robust_gate_value = if robust_gate { 1.0 } else { 0.0 };
    cases.push(case_report(
        "fem-topology-2d-robust",
        "Real-FEM erosion/dilation and material/load uncertainty reanalysis",
        robust_gate_value,
        1.0,
        1.0 - robust_gate_value,
        0.0,
        Some(
            result
                .cases
                .iter()
                .map(|case| case.solver_residual)
                .fold(0.0, f64::max),
        ),
        Some(
            result
                .cases
                .iter()
                .map(|case| {
                    case.free_equilibrium_residual_n
                        .max(case.force_balance_residual_n)
                })
                .fold(0.0, f64::max),
        ),
        Some(
            result
                .cases
                .iter()
                .map(|case| case.energy_balance_relative_error)
                .fold(0.0, f64::max),
        ),
        Some(result.gradient_taylor_relative_error),
        robust_gate,
        result.cases.len(),
        estimate_memory(
            (result.nominal.study.topology.geometry.nx + 1)
                * (result.nominal.study.topology.geometry.ny + 1)
                * 2,
            result.nominal.study.topology.geometry.nx
                * result.nominal.study.topology.geometry.ny
                * 2,
        ),
        started.elapsed().as_secs_f64() * 1_000.0,
        "P1 plane-stress FEM envelope followed by graph-based robust morphology and explicit isotropic uncertainty cases; calibrated probabilistic robustness and 3D topology are not claimed.",
    ));

    let started = Instant::now();
    let fatigue = run_fem_fatigue_2d(&FemFatigue2DStudy::introductory())?;
    let fatigue_gate = if fatigue.converged && !fatigue.failed {
        1.0
    } else {
        0.0
    };
    let fatigue_solver_residual = fatigue
        .blocks
        .iter()
        .map(|block| block.solver_residual)
        .fold(0.0, f64::max);
    let fatigue_equilibrium_residual = fatigue
        .blocks
        .iter()
        .map(|block| block.equilibrium_residual_n)
        .fold(0.0, f64::max);
    cases.push(case_report(
        "fem-fatigue-2d-reference",
        "P1 FEM load-state stress recovery and S-N/Miner feasibility gate",
        fatigue_gate,
        1.0,
        1.0 - fatigue_gate,
        0.0,
        Some(fatigue_solver_residual),
        Some(fatigue_equilibrium_residual),
        None,
        None,
        fatigue.converged && !fatigue.failed,
        fatigue.blocks.len() * 2,
        estimate_memory(
            (fatigue.study.geometry.nx + 1) * (fatigue.study.geometry.ny + 1) * 2,
            fatigue.study.geometry.nx * fatigue.study.geometry.ny * 2,
        ),
        started.elapsed().as_secs_f64() * 1_000.0,
        "P1 plane-stress FEM states with scalar Basquin/Miner accumulation; no multiaxial or crack-growth certification.",
    ));

    let started = Instant::now();
    let rom = run_structural_rom(&StructuralRomStudy::introductory())?;
    cases.push(case_report(
        "beam-structural-rom",
        "Residual-certified recycled modal Galerkin query",
        rom.reduced_tip_deflection_m,
        rom.full_order_tip_deflection_m,
        rom.relative_tip_error,
        1.0e-7,
        Some(rom.reduced_relative_residual),
        None,
        Some(rom.full_order_relative_residual),
        None,
        rom.recycled_basis && rom.converged,
        rom.final_basis_dimension,
        estimate_memory(rom.full_order_dofs, rom.full_order_dofs),
        started.elapsed().as_secs_f64() * 1_000.0,
        "Linear beam modal ROM with exact residual estimator and full-order enrichment gate; not a nonlinear or topology ROM.",
    ));

    let constrained_topology = ConstrainedTopologyOptimizationStudy::introductory();
    let started = Instant::now();
    let result = run_constrained_topology_optimization(&constrained_topology)?;
    let gate_value = if result.feasible { 1.0 } else { 0.0 };
    cases.push(case_report(
        "topology-constraint-evidence",
        "All reduced stress/modal/buckling/manufacturing/reanalysis gates",
        gate_value,
        1.0,
        1.0 - gate_value,
        0.0,
        None,
        None,
        None,
        Some(result.projected_reanalysis_relative_error),
        result.robust_result.converged,
        result.robust_result.iterations,
        estimate_memory(
            result.robust_result.final_physical_density.len(),
            result.robust_result.final_physical_density.len(),
        ),
        started.elapsed().as_secs_f64() * 1_000.0,
        "1D constraint evidence; not a 3D stress, modal, buckling, or overhang certification.",
    ));

    let started = Instant::now();
    let transient = run_transient_sdof(&TransientSdoFStudy::introductory())?;
    let transient_energy_scale = transient
        .total_input_work_j
        .abs()
        .max(transient.total_damping_dissipation_j.abs())
        .max(1.0e-30);
    let transient_energy_relative_error =
        transient.max_energy_balance_residual_j / transient_energy_scale;
    cases.push(case_report(
        "transient-sdof-energy",
        "Newmark average-acceleration zero-load energy reference",
        transient.final_displacement_m.abs(),
        0.0,
        transient.final_displacement_m.abs(),
        1.0e-12,
        None,
        Some(transient.max_equilibrium_residual_n),
        Some(transient_energy_relative_error),
        None,
        transient.max_equilibrium_residual_n.is_finite()
            && transient_energy_relative_error.is_finite()
            && transient_energy_relative_error <= 1.0e-8,
        transient.steps.len(),
        estimate_memory(transient.steps.len(), transient.steps.len()),
        started.elapsed().as_secs_f64() * 1_000.0,
        "Single-DOF Newmark average-acceleration transient energy/equilibrium reference; not a distributed transient FEM result.",
    ));

    let started = Instant::now();
    let random_vibration = run_random_vibration(&RandomVibrationStudy::introductory())?;
    let random_vibration_gate = if random_vibration.feasible { 0.0 } else { 1.0 };
    cases.push(case_report(
        "random-vibration-sdof",
        "Exact SDOF PSD transfer and peak-factor displacement gate",
        random_vibration.peak_displacement_m,
        random_vibration.study.allowable_displacement_m,
        random_vibration_gate,
        0.0,
        None,
        None,
        None,
        Some(random_vibration_gate),
        random_vibration.feasible,
        random_vibration.study.frequencies_hz.len(),
        estimate_memory(
            random_vibration.study.frequencies_hz.len(),
            random_vibration.study.frequencies_hz.len(),
        ),
        started.elapsed().as_secs_f64() * 1_000.0,
        "Single-DOF random-vibration PSD screening with a peak factor; not a modal random-vibration FEM certification.",
    ));

    let started = Instant::now();
    let reliability = run_reliability_study(&ReliabilityStudy::introductory())?;
    let expected_failure_probability = 1.0 / reliability.study.load_samples_n.len() as f64;
    let reliability_error = (reliability.failure_probability - expected_failure_probability).abs()
        / expected_failure_probability.max(1.0e-30);
    cases.push(case_report(
        "reliability-empirical-screen",
        "Empirical sample-counting failure probability",
        reliability.failure_probability,
        expected_failure_probability,
        reliability_error,
        1.0e-12,
        None,
        None,
        None,
        Some(reliability_error),
        reliability.feasible,
        reliability.study.load_samples_n.len(),
        estimate_memory(
            reliability.study.load_samples_n.len(),
            reliability.study.load_samples_n.len(),
        ),
        started.elapsed().as_secs_f64() * 1_000.0,
        "Empirical finite-sample reliability screen with a Gaussian-equivalent margin index; not FORM/SORM certification.",
    ));

    let started = Instant::now();
    let fatigue = run_fatigue_damage(&FatigueDamageStudy::introductory())?;
    let fatigue_reference = fatigue
        .blocks
        .iter()
        .map(|block| block.damage_increment)
        .sum::<f64>();
    let fatigue_error =
        (fatigue.total_damage - fatigue_reference).abs() / fatigue_reference.abs().max(1.0e-30);
    cases.push(case_report(
        "fatigue-miner-reference",
        "Basquin S-N block accumulation and Miner damage identity",
        fatigue.total_damage,
        fatigue_reference,
        fatigue_error,
        1.0e-12,
        None,
        None,
        None,
        None,
        !fatigue.failed,
        fatigue.blocks.len(),
        estimate_memory(fatigue.blocks.len(), fatigue.blocks.len()),
        started.elapsed().as_secs_f64() * 1_000.0,
        "Scalar Basquin/Miner fatigue reference; not multiaxial, mean-stress, crack-growth, or calibrated FEM fatigue certification.",
    ));

    let started = Instant::now();
    let external_definition = ExternalBenchmarkDefinition::default();
    let external_decks = render_external_decks(&external_definition)?;
    let adapters_valid = external_decks.len() == 3
        && external_decks.iter().all(|deck| {
            deck.schema_version == external_definition.schema_version
                && deck.benchmark_name == external_definition.name
                && !deck.files.is_empty()
                && deck.files.iter().all(|file| !file.contents.is_empty())
        });
    let adapter_count = external_decks.len() as f64;
    cases.push(case_report(
        "external-adapter-decks",
        "Versioned CalculiX/Code_Aster/FEniCSx benchmark deck generation",
        adapter_count,
        3.0,
        (adapter_count - 3.0).abs() / 3.0,
        0.0,
        None,
        None,
        None,
        None,
        adapters_valid,
        1,
        estimate_memory(
            external_decks
                .iter()
                .map(|deck| {
                    deck.files
                        .iter()
                        .map(|file| file.contents.len())
                        .sum::<usize>()
                })
                .sum(),
            external_decks.len(),
        ),
        started.elapsed().as_secs_f64() * 1_000.0,
        "Deck generation is validated; external execution remains explicit and capability-gated.",
    ));

    #[cfg(feature = "meshlang")]
    {
        let started = Instant::now();
        let artifact = compile_meshlang_bracket(MeshlangBracketParameters::default())?;
        let compiled_bytes = artifact.geo_code.len() as f64;
        cases.push(case_report(
            "cad-meshlang-compile",
            "Deterministic parameterized structural CAD compilation",
            compiled_bytes,
            compiled_bytes,
            0.0,
            0.0,
            None,
            None,
            None,
            None,
            !artifact.provenance_manifest_json.is_empty(),
            1,
            estimate_memory(artifact.geo_code.len(), artifact.source.len()),
            started.elapsed().as_secs_f64() * 1_000.0,
            "MeshLang compiler provenance only; Gmsh volume meshing is an external optional step.",
        ));
    }

    Ok(StructuralBenchmarkSuite {
        schema_version: STRUCTURAL_BENCHMARK_SCHEMA_VERSION,
        execution: StructuralExecutionProvenance::cpu_reference(),
        all_passed: cases.iter().all(|case| case.passed),
        cases,
        external_comparison: external_benchmark_status(),
        external_comparison_status: external_solver_status_summary(),
        applicability_limits: vec![
            "Deck generation and executable discovery do not establish external-solver parity; run_external_benchmark must be invoked explicitly.".to_string(),
            "Peak point/edge stress is not used as an optimization gate.".to_string(),
            "Nonlinear/contact/fatigue evidence is limited to the named reduced models.".to_string(),
            "3D topology evidence is limited to assembled linear tetrahedra with filtered/projected static multi-load compliance, first-frequency and optional x-compressive linearized buckling objectives/gates, graph morphology, and active-element body-fitted reanalysis; nonlinear/post-buckling, mixed-sign geometric stiffness, and accelerator paths remain bounded out.".to_string(),
        ],
    })
}

#[allow(clippy::too_many_arguments)]
fn case_report(
    name: &str,
    reference_source: &str,
    computed_value: f64,
    reference_value: f64,
    relative_error: f64,
    tolerance: f64,
    solver_residual: Option<f64>,
    equilibrium_residual: Option<f64>,
    energy_balance_error: Option<f64>,
    gradient_error: Option<f64>,
    converged: bool,
    iterations: usize,
    estimated_memory_bytes: usize,
    elapsed_ms: f64,
    applicability: &str,
) -> StructuralBenchmarkCase {
    let feasible = relative_error.is_finite()
        && relative_error <= tolerance
        && solver_residual.is_none_or(|value| value.is_finite())
        && equilibrium_residual.is_none_or(|value| value.is_finite())
        && energy_balance_error.is_none_or(|value| value.is_finite())
        && gradient_error.is_none_or(|value| value.is_finite());
    StructuralBenchmarkCase {
        name: name.to_string(),
        reference_source: reference_source.to_string(),
        computed_value,
        reference_value,
        relative_error,
        tolerance,
        solver_residual,
        equilibrium_residual,
        energy_balance_error,
        gradient_error,
        feasible,
        converged,
        iterations,
        estimated_memory_bytes,
        elapsed_ms,
        passed: feasible && converged,
        applicability: applicability.to_string(),
    }
}

fn estimate_memory(dofs: usize, elements: usize) -> usize {
    dofs.saturating_mul(8 * 6) + elements.saturating_mul(8 * 16)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn benchmark_suite_records_quality_gates_and_limits() {
        let suite = run_structural_benchmark_suite().unwrap();
        assert!(suite.all_passed, "{:#?}", suite.cases);
        assert_eq!(suite.schema_version, STRUCTURAL_BENCHMARK_SCHEMA_VERSION);
        assert_eq!(
            suite.execution,
            StructuralExecutionProvenance::cpu_reference()
        );
        #[cfg(not(feature = "meshlang"))]
        assert_eq!(suite.cases.len(), 37);
        #[cfg(feature = "meshlang")]
        assert_eq!(suite.cases.len(), 38);
        assert!(
            suite
                .cases
                .iter()
                .all(|case| case.passed && case.converged && case.iterations > 0)
        );
        assert!(
            suite
                .cases
                .iter()
                .all(|case| case.estimated_memory_bytes > 0 && case.elapsed_ms >= 0.0)
        );
        let axisymmetric = suite
            .cases
            .iter()
            .find(|case| case.name == "axisymmetric-cylinder-plane-strain")
            .expect("axisymmetric cylinder benchmark must be registered");
        assert!(axisymmetric.passed, "{axisymmetric:#?}");
        assert!(axisymmetric.relative_error < 0.10);
        let composite = suite
            .cases
            .iter()
            .find(|case| case.name == "composite-laminate-orthotropic")
            .expect("orthotropic composite benchmark must be registered");
        assert!(composite.passed, "{composite:#?}");
        assert!(composite.relative_error < 1.0e-8);
        for name in [
            "solid3d-static-amg-operator",
            "solid3d-static-matrix-free-p2",
        ] {
            let operator_case = suite
                .cases
                .iter()
                .find(|case| case.name == name)
                .unwrap_or_else(|| panic!("operator benchmark {name} must be registered"));
            assert!(operator_case.passed, "{operator_case:#?}");
        }
        let bracket_shape = suite
            .cases
            .iter()
            .find(|case| case.name == "bracket-shape-frequency-target")
            .expect("frequency-target shape benchmark must be registered");
        assert!(bracket_shape.passed);
        assert!(bracket_shape.gradient_error.unwrap() < 1.0e-6);
        let bracket_buckling = suite
            .cases
            .iter()
            .find(|case| case.name == "bracket-shape-buckling-target")
            .expect("buckling-target shape benchmark must be registered");
        assert!(bracket_buckling.passed);
        assert!(bracket_buckling.gradient_error.unwrap() < 1.0e-6);
        for name in [
            "solid3d-topology-frequency-objective",
            "solid3d-topology-buckling-objective",
        ] {
            let case = suite
                .cases
                .iter()
                .find(|case| case.name == name)
                .unwrap_or_else(|| panic!("benchmark case {name} must be registered"));
            assert!(case.passed, "{case:#?}");
            assert!(case.gradient_error.unwrap().is_finite());
        }
        let hyperelastic = suite
            .cases
            .iter()
            .find(|case| case.name == "solid3d-hyperelastic")
            .expect("hyperelastic benchmark case must be registered");
        assert!(hyperelastic.energy_balance_error.unwrap() < 1.0e-10);
        assert!(hyperelastic.gradient_error.unwrap() < 0.02);
        let phase_field = suite
            .cases
            .iter()
            .find(|case| case.name == "phase-field-damage-staggered")
            .expect("phase-field benchmark case must be registered");
        assert!(phase_field.converged);
        assert!(phase_field.solver_residual.unwrap() < 1.0e-6);
        assert!(phase_field.equilibrium_residual.unwrap() < 1.0e-6);
        let force_phase_field = suite
            .cases
            .iter()
            .find(|case| case.name == "phase-field-damage-force-controlled")
            .expect("force-controlled phase-field benchmark must be registered");
        assert!(force_phase_field.converged);
        assert!(force_phase_field.solver_residual.unwrap() < 1.0e-6);
        assert!(force_phase_field.equilibrium_residual.unwrap() < 1.0e-6);
        for name in [
            "transient-sdof-energy",
            "random-vibration-sdof",
            "reliability-empirical-screen",
            "fatigue-miner-reference",
        ] {
            let case = suite
                .cases
                .iter()
                .find(|case| case.name == name)
                .unwrap_or_else(|| panic!("benchmark case {name} must be registered"));
            assert!(case.passed, "{case:#?}");
        }
        assert!(
            suite
                .external_comparison_status
                .contains("not run by default")
        );
        assert_eq!(
            suite.external_comparison.status,
            crate::ExternalComparisonStatus::NotRun
        );
        assert_eq!(suite.external_comparison.adapters_registered.len(), 3);
        assert!(!suite.external_comparison.parity_passed());
        assert!(!suite.applicability_limits.is_empty());

        let encoded = serde_json::to_vec(&suite).unwrap();
        let decoded: StructuralBenchmarkSuite = serde_json::from_slice(&encoded).unwrap();
        assert_eq!(decoded.external_comparison, suite.external_comparison);

        let mut legacy = serde_json::to_value(&suite).unwrap();
        let legacy_object = legacy.as_object_mut().unwrap();
        legacy_object.remove("external_comparison");
        legacy_object.insert("schema_version".to_string(), serde_json::Value::from(1_u32));
        let legacy_decoded: StructuralBenchmarkSuite = serde_json::from_value(legacy).unwrap();
        assert_eq!(
            legacy_decoded.external_comparison.status,
            crate::ExternalComparisonStatus::NotRun
        );
    }
}
