//! Structural-mechanics example studies inspired by bracket-style FEM workflows.
//!
//! This crate keeps tutorial-scale modeling choices outside `sonium-fem`:
//! geometry intent, boundary/load selections, study sequencing, and compact
//! post-processing live here, while assembly and solvers stay in the FEM and
//! solver crates.

use num_complex::Complex64;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_homogeneous_dirichlet};
use sonium_fem::forms::standard::MassForm;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::{Mesh, MeshTopology, rectangular_mesh_triangles};
use sonium_fem::physics::solid::eigenmodes::build_linear_buckling_problem;
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use sonium_fem::quadrature::gauss_legendre_1d;
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{
    CgConfig, CsrMatrix, LobpcgConfig, MatrixClass, SparseDirectConfig, SparseDirectMethod,
    TripletMatrix, cg, factorize_sparse, lobpcg_generalized,
};
use std::path::{Path, PathBuf};

pub mod architecture;
pub mod axisymmetric;
pub mod beam;
pub mod benchmarks;
pub mod bolted_flange;
pub mod buckling;
#[cfg(feature = "meshlang")]
pub mod cad;
pub mod capabilities;
pub mod composite;
pub mod contact3d;
#[cfg(feature = "cuda")]
pub mod cuda;
pub mod dynamics;
pub mod external;
pub mod families;
pub mod fem_fatigue;
pub mod fem_topology;
pub mod force3d;
pub mod force_contact3d;
pub mod friction_contact3d;
pub mod gallery;
pub mod hyperelastic;
pub mod hyperelastic_convergence;
pub mod material_calibration;
pub mod measured_field_validation;
pub mod measured_validation;
pub mod mixed_up;
pub mod modal_harmonic_response;
pub mod modal_random_vibration;
pub mod nonlinear;
pub mod nonlinear_beam;
pub mod nonlinear_shell;
pub mod nonlinear_transient;
pub mod nonlinear_transient_fem;
pub mod optimization;
pub mod phase_field;
pub mod plate;
pub mod post_buckling;
pub mod preview;
pub mod references;
pub mod rom;
pub mod shell;
pub mod shell_post_buckling;
pub mod solid_convergence;
pub mod solid_harmonic_convergence;
pub mod solid_harmonic_response;
pub mod solid_topology;
pub mod studies;
pub mod thermoelastic;
pub mod transient_fem;
pub mod validation;
pub mod viscoelastic;

pub use architecture::{
    CROSS_ARCH_STRUCTURAL_SCHEMA_VERSION, CrossArchitectureComparison, CrossArchitectureTolerances,
    NativeStructuralArchitectureMetadata, NativeStructuralReferencePacket,
    StructuralArchitectureError, compare_structural_architecture_packets,
    run_native_structural_reference_packet,
};
pub use axisymmetric::{
    AxisymmetricCylinderGeometry, AxisymmetricCylinderMaterial, AxisymmetricCylinderResult,
    AxisymmetricCylinderStudy, run_axisymmetric_cylinder,
};
pub use beam::{
    Beam2DGeometry, Beam2DModalResult, Beam2DStaticResult, Beam2DStudy, run_beam2d_modal,
    run_beam2d_static,
};
pub use benchmarks::{
    STRUCTURAL_BENCHMARK_SCHEMA_VERSION, StructuralBenchmarkCase, StructuralBenchmarkSuite,
    run_structural_benchmark_suite,
};
pub use bolted_flange::{
    BoltedFlangeGeometry, BoltedFlangeMaterial, BoltedFlangeResult, BoltedFlangeStep,
    BoltedFlangeStudy, bolt_axial_stiffness_n_per_m, rotscher_cone_stiffness_n_per_m,
    run_bolted_flange,
};
pub use buckling::{
    BucklingEndCondition, BucklingGeometry, BucklingResult, BucklingStudy, run_linear_buckling,
};
#[cfg(feature = "meshlang")]
pub use cad::{
    MeshlangBracketCompileArtifact, MeshlangBracketParameters,
    MeshlangBracketShapeOptimizationResult, MeshlangBracketShapeOptimizationStudy,
    compile_meshlang_bracket, mesh_meshlang_bracket, run_meshlang_bracket_shape_optimization,
};
pub use capabilities::{
    STRUCTURAL_CAPABILITY_SCHEMA_VERSION, StructuralAnalysisFamily, StructuralBackend,
    StructuralBackendCapability, StructuralExecutionMode, StructuralExecutionProvenance,
    StructuralPrecisionPolicy, structural_backend_capabilities,
};
pub use composite::{
    COMPOSITE_FIXED_BOUNDARY_MARKER, COMPOSITE_LOADED_BOUNDARY_MARKER,
    CompositeLaminateStaticResult, CompositeLaminateStudy, CompositeMembraneGeometry, CompositePly,
    CompositePlyFailureResult, CompositePlyStrength, run_composite_laminate_static,
};
pub use contact3d::{
    HYPERELASTIC_3D_CONTACT_SCHEMA_VERSION, Hyperelastic3DContactResult,
    Hyperelastic3DContactStepResult, Hyperelastic3DContactStudy, run_hyperelastic_3d_contact,
};
#[cfg(feature = "cuda")]
pub use cuda::{
    CudaCancellationToken, CudaDerivativeEvidence, CudaStaticBracketConfig,
    CudaStaticBracketEvidence, CudaStaticBracketResult, StructuralCudaError,
    run_static_bracket_cuda,
};
pub use dynamics::{
    FatigueBlockResult, FatigueDamageResult, FatigueDamageStudy, RandomVibrationResult,
    RandomVibrationStudy, ReliabilityStudy, ReliabilityStudyResult, TransientSdoFResult,
    TransientSdoFStudy, TransientStepResult, run_fatigue_damage, run_random_vibration,
    run_reliability_study, run_transient_sdof,
};
pub use external::{
    EXTERNAL_BENCHMARK_SCHEMA_VERSION, EXTERNAL_BENCHMARK_STATUS_SCHEMA_VERSION,
    ExternalBenchmarkDeck, ExternalBenchmarkDefinition, ExternalBenchmarkStatus,
    ExternalComparisonResult, ExternalComparisonStatus, ExternalDisplacementObservation,
    ExternalDisplacementSource, ExternalExecutionStage, ExternalInputFile, ExternalRunOptions,
    ExternalRunResult, ExternalSolverAvailability, ExternalSolverKind,
    compare_external_displacement, compare_external_observation, discover_external_solver,
    discover_external_solvers, external_benchmark_status, external_solver_status_summary,
    parse_external_displacement, parse_external_displacement_for_parity,
    parse_external_displacement_observation, render_external_deck, render_external_decks,
    run_external_benchmark,
};
pub use families::{
    SOLID3D_FIXED_BOUNDARY_MARKER, SOLID3D_LOADED_BOUNDARY_MARKER, Solid3DGeometry, Solid3DLoads,
    Solid3DModalResult, Solid3DMultiLoadCase, Solid3DMultiLoadCaseResult, Solid3DMultiLoadResult,
    Solid3DMultiLoadStudy, Solid3DStaticOptions, Solid3DStaticResult, Solid3DStaticSolver,
    Solid3DStudy, run_solid3d_modal, run_solid3d_multi_load, run_solid3d_static,
    run_solid3d_static_order, run_solid3d_static_with_options,
};
pub use fem_fatigue::{
    EUROCODE3_CUTOFF_CYCLES, EUROCODE3_KNEE_CYCLES, EUROCODE3_REFERENCE_CYCLES,
    EUROCODE3_SLOPE_ABOVE_KNEE, EUROCODE3_SLOPE_BELOW_KNEE, FemFatigue2DBlockResult,
    FemFatigue2DLoadBlock, FemFatigue2DResult, FemFatigue2DStudy, PublishedSnCurve,
    run_fem_fatigue_2d,
};
pub use fem_topology::{
    FemTopology2DEnvelopeResult, FemTopology2DEnvelopeStudy, FemTopology2DGeometry,
    FemTopology2DLoadCase, FemTopology2DLoadCaseResult, FemTopology2DMaterial,
    FemTopology2DMultiLoadResult, FemTopology2DMultiLoadStudy, FemTopology2DObjective,
    FemTopology2DResult, FemTopology2DRobustCaseResult, FemTopology2DRobustResult,
    FemTopology2DRobustStudy, FemTopology2DStudy, FemTopology2DUncertaintyCase,
    run_fem_topology_2d, run_fem_topology_2d_envelope, run_fem_topology_2d_multi_load,
    run_fem_topology_2d_robust,
};
pub use force_contact3d::{
    HYPERELASTIC_3D_FORCE_CONTACT_SCHEMA_VERSION, Hyperelastic3DForceContactResult,
    Hyperelastic3DForceContactStepResult, Hyperelastic3DForceContactStudy,
    run_hyperelastic_3d_force_contact,
};
pub use force3d::{
    HYPERELASTIC_3D_FORCE_SCHEMA_VERSION, Hyperelastic3DForceResult, Hyperelastic3DForceStepResult,
    Hyperelastic3DForceStudy, run_hyperelastic_3d_force,
};
pub use friction_contact3d::{
    FRICTIONAL_3D_CONTACT_SCHEMA_VERSION, Frictional3DContactResult, Frictional3DContactStudy,
    run_frictional_3d_contact,
};
pub use gallery::{
    FEM_TOPOLOGY_GALLERY_SCHEMA_VERSION, FemTopologyBeforeAfterSummary,
    SOLID3D_TOPOLOGY_GALLERY_SCHEMA_VERSION, Solid3DTopologyBeforeAfterSummary,
    append_fem_topology_before_after_gallery, append_solid3d_topology_before_after_gallery,
};
pub use hyperelastic::{Hyperelastic3DResult, Hyperelastic3DStudy, run_hyperelastic_3d};
pub use hyperelastic_convergence::{
    HYPERELASTIC_3D_CONVERGENCE_SCHEMA_VERSION, Hyperelastic3DMeshConvergenceReport,
    run_hyperelastic_3d_mesh_convergence,
};
pub use material_calibration::{
    CalibratedElastoplasticMaterial, MATERIAL_CALIBRATION_SCHEMA_VERSION,
    MaterialCalibrationModelAssumption, MaterialCalibrationResult, MaterialCalibrationStudy,
    StressStrainObservation, calibrate_material,
};
pub use measured_field_validation::{
    MEASURED_FIELD_VALIDATION_DIGEST_ALGORITHM, MEASURED_FIELD_VALIDATION_SCHEMA_VERSION,
    MeasuredFieldChannel, MeasuredFieldChannelResult, MeasuredFieldEvidenceSource,
    MeasuredFieldPoint, MeasuredFieldQuantity, MeasuredFieldValidationIntegrity,
    MeasuredFieldValidationResult, MeasuredFieldValidationStudy, compare_measured_field_validation,
    measured_field_input_sha256,
};
pub use measured_validation::{
    MEASURED_VALIDATION_DIGEST_ALGORITHM, MEASURED_VALIDATION_SCHEMA_VERSION,
    MeasuredDisplacementPoint, MeasuredValidationEvidenceSource, MeasuredValidationIntegrity,
    MeasuredValidationResult, MeasuredValidationStudy, compare_measured_displacements,
    measured_validation_input_sha256,
};
pub use mixed_up::{
    MixedUpAnnulusGeometry, MixedUpAnnulusResult, MixedUpAnnulusStudy, MixedUpMaterial,
    MixedUpMmsResult, MixedUpMmsStudy, run_mixed_up_annulus, run_mixed_up_mms,
};
pub use modal_harmonic_response::{
    MODAL_HARMONIC_RESPONSE_SCHEMA_VERSION, ModalHarmonicResponsePoint,
    ModalHarmonicResponseResult, ModalHarmonicResponseStudy, run_modal_harmonic_response,
};
pub use modal_random_vibration::{
    MODAL_RANDOM_VIBRATION_SCHEMA_VERSION, ModalRandomVibrationModeResult,
    ModalRandomVibrationResult, ModalRandomVibrationStudy, run_modal_random_vibration,
};
pub use nonlinear::{
    ContactStepResult, ElastoplasticBarResult, ElastoplasticBarStepResult, ElastoplasticBarStudy,
    FrictionlessContactResult, FrictionlessContactStudy, run_elastoplastic_bar,
    run_frictionless_contact,
};
pub use nonlinear_beam::{
    ElasticaTip, NonlinearBeamResult, NonlinearBeamStep, NonlinearBeamStudy,
    beam_column_cantilever_tip, elastica_cantilever_tip, euler_cantilever_buckling_load,
    run_nonlinear_beam,
};
pub use nonlinear_shell::{
    NonlinearShellBucklingResult, NonlinearShellBucklingStudy, NonlinearShellPressureResult,
    NonlinearShellPressureStudy, NonlinearShellStep, NonlinearShellStripResult,
    NonlinearShellStripStudy, plate_bending_stiffness_nm, run_nonlinear_shell_buckling,
    run_nonlinear_shell_pressure, run_nonlinear_shell_strip,
    simply_supported_plate_buckling_load_n_per_m,
};
pub use nonlinear_transient::{
    NONLINEAR_TRANSIENT_SCHEMA_VERSION, NonlinearTransientModelAssumption,
    NonlinearTransientResult, NonlinearTransientStepResult, NonlinearTransientStudy,
    run_nonlinear_transient,
};
pub use nonlinear_transient_fem::{
    NONLINEAR_TRANSIENT_FEM_3D_SCHEMA_VERSION, NonlinearTransientFem3DResult,
    NonlinearTransientFem3DStepResult, NonlinearTransientFem3DStudy,
    run_nonlinear_transient_fem_3d,
};
pub use optimization::{
    BracketShapeDesign, BracketShapeIteration, BracketShapeObjective,
    BracketShapeOptimizationResult, BracketShapeOptimizationStudy,
    ConstrainedTopologyOptimizationResult, ConstrainedTopologyOptimizationStudy,
    RobustTopologyOptimizationResult, RobustTopologyOptimizationStudy, ShapeDesign,
    ShapeIterationResult, ShapeOptimizationResult, ShapeOptimizationStudy, TopologyInterpolation,
    TopologyOptimizationResult, TopologyOptimizationStudy, UncertaintyBarSample, UncertaintyStudy,
    UncertaintyStudyResult, run_bracket_shape_optimization, run_constrained_topology_optimization,
    run_robust_topology_optimization, run_shape_optimization, run_topology_optimization,
    run_uncertainty_study,
};
pub use phase_field::{
    PhaseFieldDamageResult, PhaseFieldDamageStudy, PhaseFieldForceControlledStudy,
    PhaseFieldLoadingControl, run_phase_field_damage, run_phase_field_damage_force_controlled,
};
pub use plate::{
    Plate2DGeometry, Plate2DMaterial, Plate2DStaticResult, Plate2DStudy, PlateShearTreatment,
    run_plate2d_static,
};
pub use post_buckling::{
    POST_BUCKLING_SCHEMA_VERSION, PostBucklingBranch, PostBucklingModelAssumption,
    PostBucklingPoint, PostBucklingResult, PostBucklingStudy, run_post_buckling,
};
pub use preview::{
    STRUCTURAL_PREVIEW_SCHEMA_VERSION, StructuralReducedPreview, bracket_static_preview,
};
pub use references::{
    PUBLISHED_REFERENCE_COMPARISON_SCHEMA_VERSION, PUBLISHED_REFERENCE_MANIFEST_DIGEST_ALGORITHM,
    PUBLISHED_REFERENCE_MANIFEST_SCHEMA_VERSION, PublishedReferenceAuthority,
    PublishedReferenceCase, PublishedReferenceComparisonCase, PublishedReferenceComparisonReport,
    PublishedReferenceManifest, PublishedReferenceManifestIntegrity,
    compare_nafems_suite_to_published_manifest, load_published_reference_manifest,
    published_reference_manifest_sha256,
};
pub use rom::{StructuralRomResult, StructuralRomStudy, run_structural_rom};
pub use shell::{
    SHELL_FIXED_BOUNDARY_MARKER, SHELL_LOADED_BOUNDARY_MARKER, ShellGeometry, ShellMaterial,
    ShellMultiLoadCase, ShellMultiLoadCaseResult, ShellMultiLoadResult, ShellMultiLoadStudy,
    ShellStaticResult, ShellStaticStudy, run_shell_multi_load, run_shell_static,
};
pub use shell_post_buckling::{
    COROTATIONAL_SHELL_POST_BUCKLING_SCHEMA_VERSION, CorotationalShellPostBucklingResult,
    CorotationalShellPostBucklingStep, CorotationalShellPostBucklingStudy,
    run_corotational_shell_post_buckling,
};
pub use solid_convergence::{
    SOLID3D_STATIC_CONVERGENCE_SCHEMA_VERSION, Solid3DStaticMeshConvergenceReport,
    run_solid3d_static_mesh_convergence,
};
pub use solid_harmonic_convergence::{
    SOLID3D_HARMONIC_CONVERGENCE_SCHEMA_VERSION, Solid3DHarmonicMeshConvergenceReport,
    run_solid3d_harmonic_mesh_convergence,
};
pub use solid_harmonic_response::{
    SOLID3D_HARMONIC_RESPONSE_SCHEMA_VERSION, Solid3DHarmonicResponsePoint,
    Solid3DHarmonicResponseResult, Solid3DHarmonicResponseStudy, run_solid3d_harmonic_response,
};
pub use solid_topology::{
    SOLID3D_ROBUST_TOPOLOGY_PROMOTION_SCHEMA_VERSION, SOLID3D_TOPOLOGY_CONVERGENCE_SCHEMA_VERSION,
    SOLID3D_TOPOLOGY_PROMOTION_SCHEMA_VERSION, Solid3DBucklingModel,
    Solid3DRobustTopologyCaseResult, Solid3DRobustTopologyPromotion, Solid3DRobustTopologyResult,
    Solid3DRobustTopologyStudy, Solid3DTopologyCaseResult, Solid3DTopologyConvergenceLevel,
    Solid3DTopologyConvergencePromotion, Solid3DTopologyConvergenceReport,
    Solid3DTopologyObjective, Solid3DTopologyPromotion, Solid3DTopologyResult,
    Solid3DTopologyStudy, Solid3DTopologyUncertaintyCase, promote_solid3d_robust_topology,
    promote_solid3d_topology, promote_solid3d_topology_mesh_convergence,
    run_solid3d_robust_topology, run_solid3d_topology, run_solid3d_topology_mesh_convergence,
};
pub use studies::{
    LoadCaseResult, MultiLoadResult, MultiLoadStudy, ReusedLoadCaseResult, ReusedMultiLoadResult,
    StructuralEnvelope, StructuralLoadCase, run_multi_load_study, run_multi_load_study_reused,
};
pub use thermoelastic::{
    ThermalLoadModel, ThermoelasticGeometry, ThermoelasticMaterialInput, ThermoelasticStaticResult,
    ThermoelasticStudy, run_thermoelastic_static,
};
pub use transient_fem::{
    TRANSIENT_FEM_SCHEMA_VERSION, TransientFemResult, TransientFemStepResult, TransientFemStudy,
    run_transient_fem,
};
pub use validation::{
    NafemsBenchmarkCase, NafemsBenchmarkKind, NafemsBenchmarkSuite,
    STRUCTURAL_VALIDATION_SCHEMA_VERSION, ThickWalledCylinderResult, ThickWalledCylinderStudy,
    ValidationReferenceKind, run_nafems_benchmark_suite, run_thick_walled_cylinder_analytical,
};
pub use viscoelastic::{
    VISCOELASTIC_FIXED_BOUNDARY_MARKER, Viscoelastic2DGeometry, Viscoelastic2DMaterial,
    Viscoelastic2DResult, Viscoelastic2DStudy, run_viscoelastic_2d,
};

/// Boundary marker assigned to the fixed bracket root by Sonium's rectangular mesh.
pub const FIXED_ROOT_MARKER: i32 = 1;
/// Boundary marker assigned to the loaded free end by Sonium's rectangular mesh.
pub const LOADED_TIP_MARKER: i32 = 2;

fn structural_parallel_worker_count(item_count: usize) -> usize {
    item_count.min(rayon::current_num_threads()).max(1)
}

/// DOFs × cases per thread above which multi-case sparse-solve collects
/// (shell / topology / solid-3d load cases) parallelize. One case is a
/// sparse triangular solve plus assembly, so the grain keeps pool dispatch
/// amortized; tutorial-size studies stay serial.
pub(crate) const STRUCTURAL_SOLVE_WORK_PER_THREAD: usize = 65_536;

/// Pointwise values per thread above which cheap pointwise collects (e.g.
/// compliance sums over density cells) parallelize.
pub(crate) const STRUCTURAL_POINTWISE_WORK_PER_THREAD: usize = 131_072;

/// Machine-adaptive parallel gate for multi-case collects.
///
/// A fixed item count cannot be right on every machine: the point where
/// pool dispatch pays off moves with thread count and per-case cost
/// (DOFs × cases). The gate resolves `work_items >= items_per_thread *
/// threads` with `available_parallelism`, honoring
/// `SONIUM_STRUCTURAL_PAR_THRESHOLD` (absolute override for tuning and
/// forced-path tests) and `SONIUM_SERIAL=1` (process-wide serial hatch).
/// Single-threaded processes always run serially.
///
/// The per-thread grains are starting points, not calibrated values: tune
/// them with benches on the target machine.
pub(crate) fn use_structural_parallel(work_items: usize, items_per_thread: usize) -> bool {
    if serial_override() {
        return false;
    }
    if let Some(threshold) = env_threshold("SONIUM_STRUCTURAL_PAR_THRESHOLD") {
        return work_items >= threshold;
    }
    let threads = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1);
    if threads < 2 {
        return false;
    }
    work_items >= items_per_thread.saturating_mul(threads).max(1)
}

fn serial_override() -> bool {
    std::env::var("SONIUM_SERIAL")
        .map(|value| matches!(value.to_ascii_lowercase().as_str(), "1" | "true" | "yes"))
        .unwrap_or(false)
}

fn env_threshold(name: &str) -> Option<usize> {
    std::env::var(name)
        .ok()
        .and_then(|value| value.parse::<usize>().ok())
}

pub(crate) fn parallel_collect_results<T, R, F>(
    items: &[T],
    work_items: usize,
    items_per_thread: usize,
    operation: F,
) -> Result<Vec<R>, StructuralMechanicsError>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> Result<R, StructuralMechanicsError> + Send + Sync,
{
    if !use_structural_parallel(work_items, items_per_thread) {
        return items.iter().map(operation).collect();
    }
    items.par_iter().map(operation).collect()
}

pub(crate) fn parallel_collect_values<T, R, F>(
    items: &[T],
    work_items: usize,
    items_per_thread: usize,
    operation: F,
) -> Vec<R>
where
    T: Sync,
    R: Send,
    F: Fn(&T) -> R + Send + Sync,
{
    if !use_structural_parallel(work_items, items_per_thread) {
        return items.iter().map(operation).collect();
    }
    items.par_iter().map(operation).collect()
}

/// Two-dimensional solid-mechanics reduction for the bracket.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PlaneModel {
    /// Thin bracket with negligible out-of-plane stress.
    PlaneStress,
    /// Long or heavily constrained body with negligible out-of-plane strain.
    PlaneStrain,
}

/// Polynomial order used by the integrated structural bracket FEM path.
///
/// The linear and quadratic paths share the same affine triangular mesh. P2
/// edge DOFs are supplied by `VectorH1Space`; the mesh itself is intentionally
/// not promoted with extra geometric nodes. This is exact only for
/// straight-edge brackets: a curved boundary studied under `Quadratic` keeps
/// faceted-geometry error under a "Quadratic" label until isoparametric
/// mapping lands, so gate P2-curved meshes at the study constructor.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructuralPolynomialOrder {
    #[default]
    Linear,
    Quadratic,
}

impl StructuralPolynomialOrder {
    pub const fn degree(self) -> usize {
        match self {
            Self::Linear => 1,
            Self::Quadratic => 2,
        }
    }

    const fn as_fem(self) -> PolynomialDegree {
        match self {
            Self::Linear => PolynomialDegree::P1,
            Self::Quadratic => PolynomialDegree::P2,
        }
    }
}

/// Structural idealisation recorded in a result's provenance.
///
/// The introductory study currently implements the two-dimensional variants.
/// The other variants are explicit extension points so a result cannot be
/// mistaken for a shell, beam, or three-dimensional solid analysis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructuralModelAssumption {
    PlaneStress,
    PlaneStrain,
    /// Axisymmetric meridional FEM with an explicitly constrained plane-strain axial direction.
    AxisymmetricElasticity,
    /// Linear complex-stiffness plane-stress FEM with a structural loss factor.
    ViscoelasticPlaneStress,
    MindlinPlate,
    CompositeLaminate,
    Shell,
    /// Displacement-controlled geometrically nonlinear triangular shell.
    CorotationalShell,
    Beam,
    Solid3D,
    /// Bolted-flange 3D assembly with pretensioned bolt spring and
    /// augmented-Lagrangian contact.
    BoltedFlangeContact,
    /// Monotone small-strain one-dimensional elastoplastic bar with isotropic hardening.
    ElastoplasticBar,
    /// One-degree-of-freedom frictionless unilateral contact with penalty regularisation.
    FrictionlessContact,
    /// Linear single-degree-of-freedom transient oscillator integrated with Newmark average acceleration.
    TransientSingleDof,
    /// Assembled 2D linear-elastic FEM transient integrated with Newmark average acceleration.
    TransientFem2D,
    /// Stress-life block accumulation using Miner damage summation.
    MinerFatigue,
    /// P1 plane-stress FEM load-state stress amplitudes with Miner accumulation.
    FatiguePlaneStress2D,
    /// One-dimensional analytic shape optimization with body-fitted mesh motion.
    ShapeOptimization1D,
    /// One-dimensional filtered/projected SIMP topology surrogate.
    SimpTopology1D,
    /// Density-based SIMP/RAMP topology on a P1 triangular plane-stress FEM mesh.
    SimpTopology2D,
    /// Density-based SIMP/RAMP topology on an assembled linear-tetrahedral 3D mesh.
    SimpTopology3D,
    /// Continuation-filtered SIMP/RAMP topology evidence.
    RobustSimpRampTopology,
    /// Deterministic uncertainty propagation for a normalized bar stiffness.
    UncertaintyBar,
    /// Body-fitted bracket FEM shape optimization with an adjoint thickness gradient.
    BracketShapeOptimization,
    /// Single-degree-of-freedom PSD random-vibration response.
    RandomVibrationSingleDof,
    /// Independent mass-normalized modal PSD random-vibration response.
    ModalRandomVibration,
    /// Independent mass-normalized modal harmonic response.
    ModalHarmonicResponse,
    /// Direct assembled linear 3-D solid harmonic response.
    Solid3DHarmonicResponse,
    /// Empirical load/resistance reliability constraint study.
    EmpiricalReliability,
    /// Displacement-controlled total-Lagrangian 3D Neo-Hookean solid.
    HyperelasticSolid3D,
    /// Implicit-Newmark transient total-Lagrangian 3D Neo-Hookean FEM.
    NonlinearTransientFem3D,
    /// Staggered 2D linear-elasticity/phase-field damage evidence.
    PhaseFieldDamage2D,
    /// Linear beam Galerkin reduced-order model with an explicit residual gate.
    ReducedOrderBeam,
    /// Mixed displacement-pressure Taylor-Hood linear elasticity.
    MixedUpElasticity,
    /// Load-stepped geometrically nonlinear 2D corotational beam.
    CorotationalBeam,
}

impl From<PlaneModel> for StructuralModelAssumption {
    fn from(model: PlaneModel) -> Self {
        match model {
            PlaneModel::PlaneStress => Self::PlaneStress,
            PlaneModel::PlaneStrain => Self::PlaneStrain,
        }
    }
}

/// Convention used by the assembled harmonic solve.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DampingConvention {
    /// `C = 2 * damping_ratio * omega_1 * M`.
    MassProportionalAtFirstMode,
}

/// Normalization applied to reported eigenvectors.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ModalNormalization {
    /// Raw/euclidean eigenvectors; modal force consumers must not treat these as mass-normalized.
    Euclidean,
    MassNormalized,
}

/// Stress result provenance for the P1 bracket recovery.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StressRecoveryPolicy {
    /// Constant strain/stress recovered at each linear element's centroid.
    ElementCentroid,
}

impl PlaneModel {
    fn as_fem(self) -> PlaneAssumption {
        match self {
            Self::PlaneStress => PlaneAssumption::PlaneStress,
            Self::PlaneStrain => PlaneAssumption::PlaneStrain,
        }
    }
}

/// Solver family recommended for a tutorial study step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StructuralSolverFamily {
    /// Symmetric positive-definite static elasticity solve.
    ConjugateGradient,
    /// Generalized stiffness/mass eigensolve.
    LobpcgGeneralizedEigen,
    /// Implicit Newmark-style structural dynamics.
    NewmarkTimeIntegration,
    /// Linear harmonic solve, optionally accelerated by modal reduction.
    FrequencyDomainModal,
    /// Load-controlled Newton iterations with continuation between converged states.
    NewtonRaphsonContinuation,
    /// Active-set Newton solve for a penalty-regularised unilateral contact law.
    PenaltyContactNewton,
    /// Stress-life cycle block accumulation.
    MinerDamageAccumulation,
    /// Projected gradient design update.
    ProjectedGradient,
    /// Reused normalized stiffness factorization across parameter samples.
    ReusedLinearSolve,
    /// Adjoint compliance gradient with projected shape updates.
    AdjointProjectedGradient,
    /// Spectral random-vibration response integration.
    SpectralRandomVibration,
    /// Empirical reliability sampling.
    ReliabilitySampling,
    /// Staggered phase-field damage with per-element degraded mechanics.
    PhaseFieldStaggeredDamage,
}

/// A named study step in the introductory structural workflow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StructuralStudyStep {
    pub name: String,
    pub purpose: String,
    pub solver: StructuralSolverFamily,
}

/// Bracket dimensions and mesh density.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BracketGeometry {
    pub length_m: f64,
    pub height_m: f64,
    pub thickness_m: f64,
    pub nx: usize,
    pub ny: usize,
}

impl Default for BracketGeometry {
    fn default() -> Self {
        Self {
            length_m: 0.18,
            height_m: 0.036,
            thickness_m: 0.006,
            nx: 36,
            ny: 8,
        }
    }
}

impl BracketGeometry {
    pub fn area_m2(self) -> f64 {
        self.length_m * self.height_m
    }

    pub fn volume_m3(self) -> f64 {
        self.area_m2() * self.thickness_m
    }

    pub fn second_moment_m4(self) -> f64 {
        self.thickness_m * self.height_m.powi(3) / 12.0
    }
}

/// Linear-elastic material and damping inputs for the bracket.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct StructuralMaterial {
    pub youngs_modulus_pa: f64,
    pub poisson_ratio: f64,
    pub density_kg_m3: f64,
    pub damping_ratio: f64,
}

impl Default for StructuralMaterial {
    fn default() -> Self {
        Self {
            youngs_modulus_pa: 70.0e9,
            poisson_ratio: 0.33,
            density_kg_m3: 2700.0,
            damping_ratio: 0.015,
        }
    }
}

impl StructuralMaterial {
    fn as_fem(self) -> IsotropicMaterial {
        IsotropicMaterial::new(self.youngs_modulus_pa, self.poisson_ratio)
    }
}

/// Loads and boundary selections for the tutorial model.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BracketLoads {
    /// Downward total force on the loaded edge.
    pub vertical_tip_force_n: f64,
    /// Axial preload applied at the tip. Positive is tension and negative is
    /// compression. The assembled modal solve includes its geometric
    /// stiffness; compression must remain below the model's buckling limit.
    pub axial_prestress_force_n: f64,
    pub fixed_boundary_marker: i32,
    pub loaded_boundary_marker: i32,
}

impl Default for BracketLoads {
    fn default() -> Self {
        Self {
            vertical_tip_force_n: -120.0,
            axial_prestress_force_n: 250.0,
            fixed_boundary_marker: FIXED_ROOT_MARKER,
            loaded_boundary_marker: LOADED_TIP_MARKER,
        }
    }
}

/// Complete bracket example configuration.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BracketStudy {
    pub name: String,
    pub geometry: BracketGeometry,
    pub material: StructuralMaterial,
    pub plane_model: PlaneModel,
    pub loads: BracketLoads,
    pub frequency_hz: Vec<f64>,
    pub steps: Vec<StructuralStudyStep>,
}

impl BracketStudy {
    /// Introductory bracket workflow mirroring the static, modal, transient,
    /// and frequency-domain progression used in structural-mechanics tutorials.
    pub fn introductory_bracket() -> Self {
        Self {
            name: "introductory-structural-bracket".to_string(),
            geometry: BracketGeometry::default(),
            material: StructuralMaterial::default(),
            plane_model: PlaneModel::PlaneStress,
            loads: BracketLoads::default(),
            frequency_hz: octave_sweep(20.0, 2_000.0, 6),
            steps: vec![
                StructuralStudyStep {
                    name: "linear-static".to_string(),
                    purpose: "Compute displacement, strain, and von Mises stress under a bracket tip load.".to_string(),
                    solver: StructuralSolverFamily::ConjugateGradient,
                },
                StructuralStudyStep {
                    name: "time-dependent-damped".to_string(),
                    purpose: "Represent a damped transient load path using the same stiffness, mass, and damping inputs.".to_string(),
                    solver: StructuralSolverFamily::NewmarkTimeIntegration,
                },
                StructuralStudyStep {
                    name: "eigenfrequency".to_string(),
                    purpose: "Solve the assembled generalized stiffness/mass eigenproblem and report normalized modes.".to_string(),
                    solver: StructuralSolverFamily::LobpcgGeneralizedEigen,
                },
                StructuralStudyStep {
                    name: "frequency-domain-modal".to_string(),
                    purpose: "Solve the assembled complex stiffness/mass frequency response with declared damping.".to_string(),
                    solver: StructuralSolverFamily::FrequencyDomainModal,
                },
            ],
        }
    }

    pub fn mesh(&self) -> Mesh {
        rectangular_mesh_triangles(
            0.0,
            self.geometry.length_m,
            -0.5 * self.geometry.height_m,
            0.5 * self.geometry.height_m,
            self.geometry.nx,
            self.geometry.ny,
        )
    }

    /// State the idealisation used by this study in machine-readable form.
    pub fn model_assumption(&self) -> StructuralModelAssumption {
        self.plane_model.into()
    }
}

/// Static FEM result plus compact post-processing metrics.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct StaticBracketResult {
    pub nodes: usize,
    pub elements: usize,
    pub dofs: usize,
    #[serde(default)]
    pub polynomial_order: StructuralPolynomialOrder,
    pub solver_iterations: usize,
    pub solver_residual: f64,
    pub converged: bool,
    pub max_displacement_m: f64,
    pub average_tip_deflection_m: f64,
    pub beam_theory_tip_deflection_m: f64,
    pub beam_theory_relative_error: f64,
    pub max_von_mises_pa: f64,
    pub strain_energy_j: f64,
    pub applied_force_n: [f64; 2],
    pub reaction_force_n: [f64; 2],
    pub free_equilibrium_residual_n: f64,
    pub force_balance_residual_n: f64,
    pub energy_balance_relative_error: f64,
    pub model_assumption: StructuralModelAssumption,
    pub stress_recovery: StressRecoveryPolicy,
    pub displacement: Vec<[f64; 2]>,
}

/// Coarse/fine evidence for a static study's mesh convergence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshConvergenceReport {
    pub coarse: StaticBracketResult,
    pub refined: StaticBracketResult,
    pub displacement_relative_change: f64,
    pub energy_relative_change: f64,
    pub stress_relative_change: f64,
    /// Whether the centroid-stress diagnostic meets the requested tolerance.
    /// This is reported separately because centroid stress is not a hard
    /// optimization gate under the crate's stress-singularity policy.
    #[serde(default)]
    pub stress_converged: bool,
    pub tolerance: f64,
    pub converged: bool,
}

/// Same-mesh p-refinement evidence from the integrated structural FEM path.
///
/// This report compares actual P1 and P2 solves. It does not compare a P2
/// result against a promoted geometry or against a reduced-order surrogate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolynomialOrderConvergenceReport {
    pub linear: StaticBracketResult,
    pub quadratic: StaticBracketResult,
    pub displacement_relative_change: f64,
    pub energy_relative_change: f64,
    pub stress_relative_change: f64,
    /// Whether the centroid-stress diagnostic meets the requested tolerance.
    /// This remains separate from `converged` for the same singularity-policy
    /// reason as [`MeshConvergenceReport::stress_converged`].
    #[serde(default)]
    pub stress_converged: bool,
    pub tolerance: f64,
    pub converged: bool,
}

/// Failure to produce a physically usable structural-mechanics result.
#[derive(Debug, Clone, PartialEq)]
pub enum StructuralMechanicsError {
    /// The iterative solver exhausted its work without meeting its tolerance.
    NonConvergence { iterations: usize, residual: f64 },
    /// The configured fixed boundary did not select any mesh nodes.
    MissingFixedBoundary { marker: i32 },
    /// The configured load marker did not select any mesh nodes.
    MissingLoadedBoundary { marker: i32 },
    /// Study inputs or an unsupported mesh violate a physical precondition.
    InvalidConfiguration { field: &'static str, value: String },
    /// The assembled mesh cannot support a structural solve.
    InvalidMesh { reason: String },
    /// An assembled modal solve failed to converge.
    ModalNonConvergence { iterations: usize, residual: f64 },
    /// A static solve failed to converge for a named structural model family.
    StaticNonConvergence {
        model: StructuralModelAssumption,
        iterations: usize,
        residual: f64,
    },
    /// A modal or harmonic system is singular or otherwise unusable.
    SolverFailure {
        stage: &'static str,
        message: String,
    },
    /// An external executable required by an explicitly requested comparison is unavailable.
    ExternalSolverUnavailable { solver: String, executable: String },
    /// An external solver process failed to launch or returned a non-zero status.
    ExternalExecutionFailed {
        solver: String,
        status: Option<i32>,
        message: String,
    },
    /// An external benchmark definition or generated deck is invalid.
    ExternalBenchmarkInvalid { field: &'static str, value: String },
    /// A published-reference manifest failed schema or value validation.
    PublishedReferenceManifestInvalid { field: &'static str, value: String },
    /// A published-reference manifest could not be read or decoded.
    PublishedReferenceIo { path: PathBuf, message: String },
    /// External output did not contain a usable displacement result.
    ExternalOutputParseFailed { solver: String, message: String },
    /// A multi-load study must contain at least one load case.
    EmptyLoadCases,
    /// The assembled tangent has lost positive definiteness under compression.
    UnstableModel { eigenvalue: f64 },
    /// A solved result failed a physical validation gate.
    ValidationFailure { check: &'static str, value: f64 },
    /// A nonlinear load increment exhausted its Newton iterations.
    NonlinearNonConvergence {
        model: StructuralModelAssumption,
        load_step: usize,
        iterations: usize,
        residual: f64,
    },
}

impl std::fmt::Display for StructuralMechanicsError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonConvergence {
                iterations,
                residual,
            } => {
                write!(
                    formatter,
                    "static bracket solve did not converge after {iterations} iterations: residual={residual}"
                )
            }
            Self::MissingFixedBoundary { marker } => {
                write!(formatter, "fixed boundary marker {marker} has no nodes")
            }
            Self::MissingLoadedBoundary { marker } => {
                write!(formatter, "loaded boundary marker {marker} has no nodes")
            }
            Self::InvalidConfiguration { field, value } => {
                write!(
                    formatter,
                    "invalid structural configuration {field}={value}"
                )
            }
            Self::InvalidMesh { reason } => write!(formatter, "invalid structural mesh: {reason}"),
            Self::ModalNonConvergence {
                iterations,
                residual,
            } => write!(
                formatter,
                "modal solve did not converge after {iterations} iterations: residual={residual}"
            ),
            Self::StaticNonConvergence {
                model,
                iterations,
                residual,
            } => write!(
                formatter,
                "{model:?} static solve did not converge after {iterations} iterations: residual={residual}"
            ),
            Self::SolverFailure { stage, message } => {
                write!(formatter, "{stage} solver failed: {message}")
            }
            Self::ExternalSolverUnavailable { solver, executable } => write!(
                formatter,
                "external solver {solver} is unavailable (expected executable {executable})"
            ),
            Self::ExternalExecutionFailed {
                solver,
                status,
                message,
            } => write!(
                formatter,
                "external solver {solver} failed with status {status:?}: {message}"
            ),
            Self::ExternalBenchmarkInvalid { field, value } => {
                write!(formatter, "invalid external benchmark {field}={value}")
            }
            Self::PublishedReferenceManifestInvalid { field, value } => write!(
                formatter,
                "invalid published-reference manifest {field}={value}"
            ),
            Self::PublishedReferenceIo { path, message } => write!(
                formatter,
                "could not load published-reference manifest {}: {message}",
                path.display()
            ),
            Self::ExternalOutputParseFailed { solver, message } => write!(
                formatter,
                "could not parse {solver} benchmark output: {message}"
            ),
            Self::EmptyLoadCases => write!(formatter, "multi-load study contains no load cases"),
            Self::UnstableModel { eigenvalue } => {
                write!(
                    formatter,
                    "structural model is unstable: eigenvalue={eigenvalue}"
                )
            }
            Self::ValidationFailure { check, value } => {
                write!(
                    formatter,
                    "structural validation {check} failed: value={value}"
                )
            }
            Self::NonlinearNonConvergence {
                model,
                load_step,
                iterations,
                residual,
            } => write!(
                formatter,
                "{model:?} nonlinear step {load_step} did not converge after {iterations} iterations: residual={residual}"
            ),
        }
    }
}

impl std::error::Error for StructuralMechanicsError {}

fn require_converged(
    converged: bool,
    iterations: usize,
    residual: f64,
) -> Result<(), StructuralMechanicsError> {
    if converged {
        Ok(())
    } else {
        Err(StructuralMechanicsError::NonConvergence {
            iterations,
            residual,
        })
    }
}

/// Reduced modal summary derived from the static compliance.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct ModalSummary {
    #[serde(default)]
    pub polynomial_order: StructuralPolynomialOrder,
    /// Transverse stiffness recovered from the linear static FEM compliance.
    pub static_stiffness_n_per_m: f64,
    /// Modal geometric stiffness contribution from the signed axial preload.
    pub geometric_stiffness_n_per_m: f64,
    /// Generalized modal stiffness used for the assembled frequency.
    pub prestressed_stiffness_n_per_m: f64,
    pub total_mass_kg: f64,
    pub effective_modal_mass_kg: f64,
    pub first_bending_frequency_hz: f64,
    pub damping_ratio: f64,
    pub eigenvalue: f64,
    pub solver_iterations: usize,
    pub solver_residual: f64,
    pub converged: bool,
    pub normalization: ModalNormalization,
    pub mass_normalization_error: f64,
    pub orthogonality_error: f64,
    /// Linearized critical-load multiplier for an explicitly compressive axial
    /// preload. Tension and transverse-only cases report `None`.
    #[serde(default)]
    pub linearized_buckling_load_factor: Option<f64>,
    pub damping_convention: DampingConvention,
    pub model_assumption: StructuralModelAssumption,
}

/// One row in the assembled harmonic response.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct FrequencyResponsePoint {
    pub frequency_hz: f64,
    pub displacement_abs_m: f64,
    pub dynamic_magnification: f64,
    pub phase_lag_rad: f64,
}

/// Full report produced by the introductory example.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BracketExampleReport {
    pub study: BracketStudy,
    pub static_result: StaticBracketResult,
    pub modal: ModalSummary,
    pub frequency_response: Vec<FrequencyResponsePoint>,
    pub execution: StructuralExecutionProvenance,
}

/// Files written by the visualization helper.
#[derive(Debug, Clone, PartialEq)]
pub struct VisualizationArtifacts {
    pub output_dir: PathBuf,
    pub report_json: PathBuf,
    pub result_vtu: PathBuf,
    pub deformed_preview_vtu: PathBuf,
    pub warp_scale: f64,
    pub report: BracketExampleReport,
}

/// Run the full introductory bracket example.
pub fn run_introductory_bracket_example() -> Result<BracketExampleReport, StructuralMechanicsError>
{
    run_bracket_example(BracketStudy::introductory_bracket())
}

/// Run the bracket workflow for an explicitly sized study.
pub fn run_bracket_example(
    study: BracketStudy,
) -> Result<BracketExampleReport, StructuralMechanicsError> {
    let static_result = run_static_bracket(&study)?;
    let modal = solve_first_bending_mode(&study, &static_result)?;
    let frequency_response = solve_harmonic_tip_response(&study, &static_result, &modal)?;
    Ok(BracketExampleReport {
        study,
        static_result,
        modal,
        frequency_response,
        execution: StructuralExecutionProvenance::cpu_reference(),
    })
}

/// Run the introductory bracket example and write ParaView-compatible files.
///
/// Open `deformed_preview_vtu` for an immediate visual, or open `result_vtu`
/// and apply ParaView's "Warp By Vector" filter to the `displacement` field.
pub fn write_introductory_bracket_visualization(
    output_dir: impl AsRef<Path>,
) -> std::io::Result<VisualizationArtifacts> {
    let report = run_introductory_bracket_example()
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    write_bracket_visualization(report, output_dir)
}

/// Write VTU files and a JSON report for a completed bracket example.
pub fn write_bracket_visualization(
    report: BracketExampleReport,
    output_dir: impl AsRef<Path>,
) -> std::io::Result<VisualizationArtifacts> {
    let output_dir = output_dir.as_ref().to_path_buf();
    std::fs::create_dir_all(&output_dir)?;

    let result_vtu = output_dir.join("bracket_result.vtu");
    let deformed_preview_vtu = output_dir.join("bracket_deformed_preview.vtu");
    let report_json = output_dir.join("bracket_report.json");

    let mesh = report.study.mesh();
    let displacement = &report.static_result.displacement;
    let (ux, uy, uz, magnitude) = displacement_fields(displacement);
    let von_mises = element_von_mises(&mesh, &report.study, displacement);
    let warp_scale = visible_warp_scale(&report.study, &report.static_result);

    let mut writer = VtuWriter::new(&mesh);
    writer.add_point_vector("displacement", &ux, &uy, &uz);
    writer.add_point_scalar("displacement_magnitude", &magnitude);
    writer.add_cell_scalar("von_mises_pa", &von_mises);
    writer.write(&result_vtu)?;

    let mut warped = mesh.clone();
    for (node, [dx, dy]) in warped.nodes.iter_mut().zip(displacement) {
        node.x += dx * warp_scale;
        node.y += dy * warp_scale;
    }
    let mut preview = VtuWriter::new(&warped);
    preview.add_point_vector("displacement", &ux, &uy, &uz);
    preview.add_point_scalar("displacement_magnitude", &magnitude);
    preview.add_cell_scalar("von_mises_pa", &von_mises);
    preview.write(&deformed_preview_vtu)?;

    let json = serde_json::to_string_pretty(&report)
        .map_err(|error| std::io::Error::other(error.to_string()))?;
    std::fs::write(&report_json, json)?;

    Ok(VisualizationArtifacts {
        output_dir,
        report_json,
        result_vtu,
        deformed_preview_vtu,
        warp_scale,
        report,
    })
}

/// Assemble and solve the linear static bracket problem.
pub fn run_static_bracket(
    study: &BracketStudy,
) -> Result<StaticBracketResult, StructuralMechanicsError> {
    run_static_bracket_order(study, StructuralPolynomialOrder::Linear)
}

/// Solve the bracket static problem at an explicitly selected polynomial order.
///
/// Modal and harmonic APIs continue to use the linear reference result until
/// their mode-shape and geometric-stiffness post-processing are generalized to
/// the same order.
pub fn run_static_bracket_order(
    study: &BracketStudy,
    polynomial_order: StructuralPolynomialOrder,
) -> Result<StaticBracketResult, StructuralMechanicsError> {
    validate_study(study)?;
    let mesh = study.mesh();
    run_static_bracket_on_mesh_order(study, &mesh, polynomial_order)
}

/// Run the bracket static solve on an already validated/morphed mesh.
///
/// This crate-visible seam lets shape studies perform transactional FEM mesh
/// motion and then reassemble on exactly the body-fitted geometry that was
/// accepted by [`sonium_fem::mesh::MeshMotion`].
pub(crate) fn run_static_bracket_on_mesh(
    study: &BracketStudy,
    mesh: &Mesh,
) -> Result<StaticBracketResult, StructuralMechanicsError> {
    run_static_bracket_on_mesh_order(study, mesh, StructuralPolynomialOrder::Linear)
}

pub(crate) fn run_static_bracket_on_mesh_order(
    study: &BracketStudy,
    mesh: &Mesh,
    polynomial_order: StructuralPolynomialOrder,
) -> Result<StaticBracketResult, StructuralMechanicsError> {
    validate_mesh(mesh, study)?;
    let degree = polynomial_order.as_fem();
    let space = VectorH1Space::new(degree, 2);
    let dofs = space.num_global_dofs(mesh);
    let material = study.material.as_fem();
    let form = ElasticityForm::new_2d(material, study.plane_model.as_fem());
    let matrices = assemble_structural_matrices(study, mesh, &space, &form, degree);
    let stiffness = matrices.stiffness;
    let unconstrained_stiffness = stiffness.clone();

    let mut rhs = vec![0.0; dofs];
    apply_tip_edge_load_order(mesh, study, &space, degree, &mut rhs);
    let external_rhs = rhs.clone();

    let mut system = FemSystem::new(stiffness, rhs);
    try_apply_homogeneous_dirichlet(&mut system, mesh, &space, study.loads.fixed_boundary_marker)
        .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "static boundary",
        message: error.to_string(),
    })?;

    let csr = system.matrix.to_csr();
    let config = CgConfig {
        max_iterations: 8_000,
        tolerance: 1e-9,
        print_interval: 0,
    };
    let solved = cg(&csr, &system.rhs, &config);
    require_converged(solved.converged, solved.iterations, solved.residual)?;

    let displacement = nodal_displacement(&solved.x, mesh.num_nodes());
    let tip_dofs = boundary_component_dofs(mesh, &space, study.loads.loaded_boundary_marker, 1);
    if tip_dofs.is_empty() {
        return Err(StructuralMechanicsError::MissingLoadedBoundary {
            marker: study.loads.loaded_boundary_marker,
        });
    }
    let average_tip_deflection_m =
        tip_dofs.iter().map(|&dof| solved.x[dof]).sum::<f64>() / tip_dofs.len() as f64;
    let max_displacement_m = solved
        .x
        .as_chunks::<2>()
        .0
        .iter()
        .map(|d| (d[0] * d[0] + d[1] * d[1]).sqrt())
        .fold(0.0, f64::max);
    let beam_theory_tip_deflection_m = beam_tip_deflection(study);
    let beam_theory_relative_error =
        (average_tip_deflection_m.abs() - beam_theory_tip_deflection_m.abs()).abs()
            / beam_theory_tip_deflection_m.abs();
    let fixed_dofs = space.boundary_dofs(mesh, study.loads.fixed_boundary_marker);
    let internal_force = triplet_matvec(&unconstrained_stiffness, &solved.x);
    let applied_force_n = component_force(&external_rhs);
    let reaction_force_n = reaction_force(&internal_force, &external_rhs, &fixed_dofs);
    let free_equilibrium_residual_n =
        free_residual_norm(&internal_force, &external_rhs, &system.dirichlet_dofs);
    let force_balance_residual_n = ((applied_force_n[0] + reaction_force_n[0]).powi(2)
        + (applied_force_n[1] + reaction_force_n[1]).powi(2))
    .sqrt();
    let internal_energy = 0.5 * dot(&solved.x, &internal_force);
    let rhs_energy = 0.5 * dot(&solved.x, &external_rhs);
    let energy_balance_relative_error =
        (internal_energy - rhs_energy).abs() / rhs_energy.abs().max(1.0);
    let force_scale = dot(&applied_force_n, &applied_force_n).sqrt().max(1.0);
    let integrated_load_error = ((applied_force_n[0] - study.loads.axial_prestress_force_n)
        .powi(2)
        + (applied_force_n[1] - study.loads.vertical_tip_force_n).powi(2))
    .sqrt();
    if !integrated_load_error.is_finite() || integrated_load_error > 1e-10 * force_scale {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "load_integration",
            value: integrated_load_error,
        });
    }
    if !free_equilibrium_residual_n.is_finite() || free_equilibrium_residual_n > 1e-6 * force_scale
    {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "free_equilibrium",
            value: free_equilibrium_residual_n,
        });
    }
    if !force_balance_residual_n.is_finite() || force_balance_residual_n > 1e-6 * force_scale {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "force_balance",
            value: force_balance_residual_n,
        });
    }
    if !energy_balance_relative_error.is_finite() || energy_balance_relative_error > 1e-9 {
        return Err(StructuralMechanicsError::ValidationFailure {
            check: "energy_identity",
            value: energy_balance_relative_error,
        });
    }

    Ok(StaticBracketResult {
        nodes: mesh.num_nodes(),
        elements: mesh.num_elements(),
        dofs,
        solver_iterations: solved.iterations,
        solver_residual: solved.residual,
        converged: solved.converged,
        max_displacement_m,
        average_tip_deflection_m,
        beam_theory_tip_deflection_m,
        beam_theory_relative_error,
        max_von_mises_pa: max_von_mises_for_order(
            mesh,
            study,
            &space,
            &solved.x,
            polynomial_order,
            &displacement,
        ),
        strain_energy_j: internal_energy,
        applied_force_n,
        reaction_force_n,
        free_equilibrium_residual_n,
        force_balance_residual_n,
        energy_balance_relative_error,
        model_assumption: study.model_assumption(),
        stress_recovery: StressRecoveryPolicy::ElementCentroid,
        polynomial_order,
        displacement,
    })
}

/// Linear system artifacts used by crate-local adjoint shape studies.
pub(crate) struct StaticLinearState {
    pub(crate) stiffness: CsrMatrix<f64>,
    pub(crate) external_rhs: Vec<f64>,
    pub(crate) solution: Vec<f64>,
}

/// Assemble and solve the constrained static system on a supplied mesh.
///
/// This deliberately shares the same boundary/load path as the reported
/// static result. It is kept separate from the serializable result so adjoint
/// studies can reuse the assembled matrix without widening the public study
/// format.
pub(crate) fn solve_static_linear_state(
    study: &BracketStudy,
    mesh: &Mesh,
) -> Result<StaticLinearState, StructuralMechanicsError> {
    validate_mesh(mesh, study)?;
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let dofs = space.num_global_dofs(mesh);
    let form = ElasticityForm::new_2d(study.material.as_fem(), study.plane_model.as_fem());
    let matrices = assemble_structural_matrices(study, mesh, &space, &form, PolynomialDegree::P1);
    let mut rhs = vec![0.0; dofs];
    apply_tip_edge_load(mesh, study, &mut rhs);
    let external_rhs = rhs.clone();
    let mut system = FemSystem::new(matrices.stiffness, rhs);
    try_apply_homogeneous_dirichlet(&mut system, mesh, &space, study.loads.fixed_boundary_marker)
        .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "shape adjoint boundary",
        message: error.to_string(),
    })?;
    let stiffness = system.matrix.to_csr();
    let solved = cg(
        &stiffness,
        &system.rhs,
        &CgConfig {
            max_iterations: 8_000,
            tolerance: 1e-9,
            print_interval: 0,
        },
    );
    require_converged(solved.converged, solved.iterations, solved.residual)?;
    Ok(StaticLinearState {
        stiffness,
        external_rhs,
        solution: solved.x,
    })
}

/// Solve a study on its requested and uniformly refined meshes.
///
/// Displacement and energy are the primary convergence evidence. Centroid
/// stress is reported as a diagnostic only because a point/edge peak is not a
/// regularized quantity suitable for optimization constraints.
pub fn run_static_mesh_convergence(
    study: &BracketStudy,
    tolerance: f64,
) -> Result<MeshConvergenceReport, StructuralMechanicsError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "mesh_convergence_tolerance",
            value: tolerance.to_string(),
        });
    }
    let coarse = run_static_bracket(study)?;
    let mut refined_study = study.clone();
    refined_study.geometry.nx = study.geometry.nx.checked_mul(2).ok_or_else(|| {
        StructuralMechanicsError::InvalidConfiguration {
            field: "nx",
            value: "refinement overflows usize".to_string(),
        }
    })?;
    refined_study.geometry.ny = study.geometry.ny.checked_mul(2).ok_or_else(|| {
        StructuralMechanicsError::InvalidConfiguration {
            field: "ny",
            value: "refinement overflows usize".to_string(),
        }
    })?;
    let refined = run_static_bracket(&refined_study)?;
    let displacement_relative_change = relative_change(
        refined.average_tip_deflection_m,
        coarse.average_tip_deflection_m,
    );
    let energy_relative_change = relative_change(refined.strain_energy_j, coarse.strain_energy_j);
    let stress_relative_change = relative_change(refined.max_von_mises_pa, coarse.max_von_mises_pa);
    Ok(MeshConvergenceReport {
        coarse,
        refined,
        displacement_relative_change,
        energy_relative_change,
        stress_relative_change,
        stress_converged: stress_relative_change <= tolerance,
        tolerance,
        converged: displacement_relative_change <= tolerance && energy_relative_change <= tolerance,
    })
}

/// Compare the same affine mesh solved with linear and quadratic H1 bases.
pub fn run_static_order_convergence(
    study: &BracketStudy,
    tolerance: f64,
) -> Result<PolynomialOrderConvergenceReport, StructuralMechanicsError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "order_convergence_tolerance",
            value: tolerance.to_string(),
        });
    }
    let linear = run_static_bracket_order(study, StructuralPolynomialOrder::Linear)?;
    let quadratic = run_static_bracket_order(study, StructuralPolynomialOrder::Quadratic)?;
    let displacement_relative_change = relative_change(
        quadratic.average_tip_deflection_m,
        linear.average_tip_deflection_m,
    );
    let energy_relative_change = relative_change(quadratic.strain_energy_j, linear.strain_energy_j);
    let stress_relative_change =
        relative_change(quadratic.max_von_mises_pa, linear.max_von_mises_pa);
    Ok(PolynomialOrderConvergenceReport {
        linear,
        quadratic,
        displacement_relative_change,
        energy_relative_change,
        stress_relative_change,
        stress_converged: stress_relative_change <= tolerance,
        tolerance,
        converged: displacement_relative_change <= tolerance && energy_relative_change <= tolerance,
    })
}

fn relative_change(refined: f64, coarse: f64) -> f64 {
    (refined - coarse).abs() / refined.abs().max(f64::EPSILON)
}

/// Solve the first bending mode from the assembled stiffness and mass matrices.
///
/// The axial preload contributes an assembled initial-stress (geometric)
/// stiffness. Positive `N` is tension; compression reduces the frequency.
pub fn solve_first_bending_mode(
    study: &BracketStudy,
    static_result: &StaticBracketResult,
) -> Result<ModalSummary, StructuralMechanicsError> {
    solve_first_bending_mode_order(study, static_result, static_result.polynomial_order)
}

/// Solve the first bending mode at an explicitly selected polynomial order.
pub fn solve_first_bending_mode_order(
    study: &BracketStudy,
    static_result: &StaticBracketResult,
    polynomial_order: StructuralPolynomialOrder,
) -> Result<ModalSummary, StructuralMechanicsError> {
    validate_study(study)?;
    if static_result.polynomial_order != polynomial_order {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "static_result.polynomial_order",
            value: format!("expected {polynomial_order:?}"),
        });
    }
    let mesh = study.mesh();
    validate_mesh(&mesh, study)?;
    let degree = polynomial_order.as_fem();
    let space = VectorH1Space::new(degree, 2);
    let material = study.material.as_fem();
    let form = ElasticityForm::new_2d(material, study.plane_model.as_fem());
    let matrices = assemble_structural_matrices(study, &mesh, &space, &form, degree);
    let fixed_dofs = space.boundary_dofs(&mesh, study.loads.fixed_boundary_marker);
    let (reduced_elastic, free_to_full) = reduce_triplets(&matrices.stiffness, &fixed_dofs);
    let (reduced_mass, _) = reduce_triplets(&matrices.mass, &fixed_dofs);
    let (reduced_geometric, _) = reduce_triplets(&matrices.geometric, &fixed_dofs);
    let reduced_stiffness = add_csr(&reduced_elastic, &reduced_geometric);
    if reduced_stiffness.num_rows == 0 {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "all displacement DOFs are constrained".to_string(),
        });
    }

    let lobpcg = lobpcg_generalized(
        &reduced_stiffness,
        &reduced_mass,
        &LobpcgConfig {
            num_eigenvalues: 2.min(reduced_stiffness.num_rows),
            max_iterations: 2_000,
            tolerance: 1e-8,
            block_padding: 1,
        },
    )
    .map_err(|error| StructuralMechanicsError::SolverFailure {
        stage: "modal",
        message: error.to_string(),
    })?;
    let residual = lobpcg
        .residual_norms
        .first()
        .copied()
        .unwrap_or(f64::INFINITY);
    if !lobpcg.converged {
        return Err(StructuralMechanicsError::ModalNonConvergence {
            iterations: lobpcg.iterations,
            residual,
        });
    }
    let eigenvalue = lobpcg.eigenvalues.first().copied().unwrap_or(f64::NAN);
    if !eigenvalue.is_finite() || eigenvalue <= 0.0 {
        return Err(StructuralMechanicsError::UnstableModel { eigenvalue });
    }

    let mut mode = (0..reduced_stiffness.num_rows)
        .map(|row| lobpcg.eigenvectors[[row, 0]])
        .collect::<Vec<_>>();
    let mass_norm = dot(&mode, &reduced_mass.matvec(&mode)).sqrt();
    if !mass_norm.is_finite() || mass_norm <= f64::EPSILON {
        return Err(StructuralMechanicsError::SolverFailure {
            stage: "modal normalization",
            message: "eigenvector has zero or non-finite mass norm".to_string(),
        });
    }
    for value in &mut mode {
        *value /= mass_norm;
    }
    let mass_normalization_error = (dot(&mode, &reduced_mass.matvec(&mode)) - 1.0).abs();
    let tip_dofs = boundary_component_dofs(&mesh, &space, study.loads.loaded_boundary_marker, 1);
    let tip_count = tip_dofs.len();
    if tip_count == 0 {
        return Err(StructuralMechanicsError::MissingLoadedBoundary {
            marker: study.loads.loaded_boundary_marker,
        });
    }
    let mut influence = vec![0.0; reduced_stiffness.num_rows];
    for full_dof in tip_dofs {
        let reduced = free_to_full
            .iter()
            .position(|&mapped| mapped == full_dof)
            .ok_or_else(|| StructuralMechanicsError::InvalidMesh {
                reason: "loaded boundary is fully constrained".to_string(),
            })?;
        influence[reduced] += 1.0 / tip_count as f64;
    }
    let effective_modal_mass_kg = dot(&mode, &reduced_mass.matvec(&influence)).powi(2);
    let geometric_modal = dot(&mode, &reduced_geometric.matvec(&mode));
    let linearized_buckling_load_factor = if study.loads.axial_prestress_force_n < 0.0 {
        let compressive_geometric = scale_triplet(&matrices.geometric, -1.0);
        let problem =
            build_linear_buckling_problem(&matrices.stiffness, &compressive_geometric, &fixed_dofs);
        let buckling = lobpcg_generalized(
            &problem.stiffness_like,
            &problem.mass_like,
            &LobpcgConfig {
                num_eigenvalues: 1,
                max_iterations: 2_000,
                tolerance: 1.0e-8,
                block_padding: 1,
            },
        )
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage: "bracket linearized buckling",
            message: error.to_string(),
        })?;
        if !buckling.converged {
            return Err(StructuralMechanicsError::ModalNonConvergence {
                iterations: buckling.iterations,
                residual: buckling
                    .residual_norms
                    .first()
                    .copied()
                    .unwrap_or(f64::INFINITY),
            });
        }
        let factor = buckling
            .eigenvalues
            .iter()
            .copied()
            .filter(|value| value.is_finite() && *value > 0.0)
            .fold(f64::INFINITY, f64::min);
        if !factor.is_finite() {
            return Err(StructuralMechanicsError::UnstableModel { eigenvalue: factor });
        }
        Some(factor)
    } else {
        None
    };
    let modal_stiffness = eigenvalue * effective_modal_mass_kg;
    let geometric_stiffness_n_per_m = geometric_modal * effective_modal_mass_kg;
    let force = study.loads.vertical_tip_force_n.abs();
    let compliance = static_result.average_tip_deflection_m.abs() / force;
    let static_stiffness_n_per_m = 1.0 / compliance;
    let total_mass_kg = study.material.density_kg_m3 * study.geometry.volume_m3();

    let mut orthogonality_error = 0.0;
    if lobpcg.eigenvectors.ncols() > 1 {
        let second = (0..reduced_stiffness.num_rows)
            .map(|row| lobpcg.eigenvectors[[row, 1]])
            .collect::<Vec<_>>();
        let second_norm = dot(&second, &reduced_mass.matvec(&second)).sqrt();
        if second_norm > f64::EPSILON {
            orthogonality_error = dot(&mode, &reduced_mass.matvec(&second)) / second_norm;
        }
    }

    Ok(ModalSummary {
        polynomial_order,
        static_stiffness_n_per_m,
        geometric_stiffness_n_per_m,
        prestressed_stiffness_n_per_m: modal_stiffness,
        total_mass_kg,
        effective_modal_mass_kg,
        first_bending_frequency_hz: eigenvalue.sqrt() / std::f64::consts::TAU,
        damping_ratio: study.material.damping_ratio,
        eigenvalue,
        solver_iterations: lobpcg.iterations,
        solver_residual: residual,
        converged: lobpcg.converged,
        normalization: ModalNormalization::MassNormalized,
        mass_normalization_error,
        orthogonality_error,
        linearized_buckling_load_factor,
        damping_convention: DampingConvention::MassProportionalAtFirstMode,
        model_assumption: study.model_assumption(),
    })
}

/// Backward-compatible name for the assembled modal solve.
#[deprecated(note = "use solve_first_bending_mode to handle typed solver errors")]
pub fn estimate_first_bending_mode(
    study: &BracketStudy,
    static_result: &StaticBracketResult,
) -> ModalSummary {
    solve_first_bending_mode(study, static_result)
        .unwrap_or_else(|error| panic!("assembled modal solve failed: {error}"))
}

/// Solve the full assembled complex frequency-response problem.
pub fn solve_harmonic_tip_response(
    study: &BracketStudy,
    static_result: &StaticBracketResult,
    modal: &ModalSummary,
) -> Result<Vec<FrequencyResponsePoint>, StructuralMechanicsError> {
    solve_harmonic_tip_response_order(study, static_result, modal, static_result.polynomial_order)
}

/// Solve the assembled complex frequency response at an explicit polynomial
/// order. Static and modal inputs must carry the same order provenance.
pub fn solve_harmonic_tip_response_order(
    study: &BracketStudy,
    static_result: &StaticBracketResult,
    modal: &ModalSummary,
    polynomial_order: StructuralPolynomialOrder,
) -> Result<Vec<FrequencyResponsePoint>, StructuralMechanicsError> {
    solve_harmonic_tip_response_order_with_threads(
        study,
        static_result,
        modal,
        polynomial_order,
        None,
    )
}

/// Solve the assembled complex frequency response in a deterministic local
/// Rayon pool. Each frequency owns its sparse factorization; only immutable
/// reduced matrices and load mappings are shared between workers.
pub fn solve_harmonic_tip_response_parallel_order(
    study: &BracketStudy,
    static_result: &StaticBracketResult,
    modal: &ModalSummary,
    polynomial_order: StructuralPolynomialOrder,
    num_threads: usize,
) -> Result<Vec<FrequencyResponsePoint>, StructuralMechanicsError> {
    if num_threads == 0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "harmonic_parallel_threads",
            value: num_threads.to_string(),
        });
    }
    solve_harmonic_tip_response_order_with_threads(
        study,
        static_result,
        modal,
        polynomial_order,
        Some(num_threads),
    )
}

fn solve_harmonic_tip_response_order_with_threads(
    study: &BracketStudy,
    static_result: &StaticBracketResult,
    modal: &ModalSummary,
    polynomial_order: StructuralPolynomialOrder,
    parallel_threads: Option<usize>,
) -> Result<Vec<FrequencyResponsePoint>, StructuralMechanicsError> {
    validate_study(study)?;
    if static_result.polynomial_order != polynomial_order {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "static_result.polynomial_order",
            value: format!("expected {polynomial_order:?}"),
        });
    }
    if modal.polynomial_order != polynomial_order {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "modal.polynomial_order",
            value: format!("expected {polynomial_order:?}"),
        });
    }
    let mesh = study.mesh();
    validate_mesh(&mesh, study)?;
    let degree = polynomial_order.as_fem();
    let space = VectorH1Space::new(degree, 2);
    let material = study.material.as_fem();
    let form = ElasticityForm::new_2d(material, study.plane_model.as_fem());
    let matrices = assemble_structural_matrices(study, &mesh, &space, &form, degree);
    let fixed_dofs = space.boundary_dofs(&mesh, study.loads.fixed_boundary_marker);
    let (elastic, free_to_full) = reduce_triplets(&matrices.stiffness, &fixed_dofs);
    let (mass, _) = reduce_triplets(&matrices.mass, &fixed_dofs);
    let (geometric, _) = reduce_triplets(&matrices.geometric, &fixed_dofs);
    let free_count = elastic.num_rows;
    let tip_dofs = boundary_component_dofs(&mesh, &space, study.loads.loaded_boundary_marker, 1);
    if tip_dofs.is_empty() {
        return Err(StructuralMechanicsError::MissingLoadedBoundary {
            marker: study.loads.loaded_boundary_marker,
        });
    }
    let static_tip = static_result.average_tip_deflection_m.abs();
    if static_tip <= f64::EPSILON {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "vertical_tip_force_n",
            value: "must produce a non-zero static tip response".to_string(),
        });
    }
    let mut load = vec![0.0; space.num_global_dofs(&mesh)];
    apply_tip_edge_load_order(&mesh, study, &space, degree, &mut load);
    let reduced_load = free_to_full
        .iter()
        .map(|&full| load[full])
        .collect::<Vec<_>>();
    let damping_scale =
        2.0 * modal.damping_ratio * modal.first_bending_frequency_hz * std::f64::consts::TAU;
    let solver_config = SparseDirectConfig {
        matrix_class: MatrixClass::General,
        method: SparseDirectMethod::SupernodalLu,
        ..SparseDirectConfig::default()
    };

    let frequency_system = HarmonicFrequencySystem {
        elastic: &elastic,
        mass: &mass,
        geometric: &geometric,
        free_count,
        free_to_full: &free_to_full,
        reduced_load: &reduced_load,
        tip_dofs: &tip_dofs,
        static_tip,
        damping_scale,
        vertical_tip_force_n: study.loads.vertical_tip_force_n,
        solver_config: &solver_config,
    };
    let solve = |&frequency_hz| frequency_system.solve(frequency_hz);
    let automatic_threads = harmonic_parallel_worker_count(
        study.frequency_hz.len(),
        free_count,
        elastic.values.len() + mass.values.len() + geometric.values.len(),
    );
    let resolved_threads = parallel_threads.unwrap_or(automatic_threads);
    match resolved_threads {
        0 | 1 => study.frequency_hz.iter().map(solve).collect(),
        num_threads => {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build()
                .map_err(|error| StructuralMechanicsError::SolverFailure {
                    stage: "harmonic parallel execution",
                    message: error.to_string(),
                })?;
            pool.install(|| study.frequency_hz.par_iter().map(solve).collect())
        }
    }
}

fn harmonic_parallel_worker_count(
    frequency_count: usize,
    free_dofs: usize,
    matrix_nonzeros: usize,
) -> usize {
    let cpu_workers = structural_parallel_worker_count(frequency_count);
    // f64 CSR costs ~16 B/nz (8 value + 4-8 column index + amortized row
    // pointer); previously multiplied by 16 twice more (~128 B/nz, ~8x
    // overcount), which throttled harmonic parallelism for no reason.
    let estimated_bytes_per_worker = matrix_nonzeros
        .saturating_mul(16)
        .saturating_add(free_dofs.saturating_mul(16))
        .max(1);
    let memory_workers = structural_available_memory_bytes()
        .map(|bytes| bytes.saturating_mul(3) / 4 / estimated_bytes_per_worker)
        .unwrap_or(cpu_workers)
        .max(1);
    cpu_workers.min(memory_workers)
}

fn structural_available_memory_bytes() -> Option<usize> {
    if let Ok(value) = std::env::var("SONIUM_STRUCTURAL_MEMORY_BUDGET_BYTES")
        && let Ok(bytes) = value.parse::<usize>()
        && bytes > 0
    {
        return Some(bytes);
    }
    let meminfo = std::fs::read_to_string("/proc/meminfo").ok()?;
    let kib = meminfo.lines().find_map(|line| {
        line.strip_prefix("MemAvailable:")?
            .split_whitespace()
            .next()?
            .parse::<usize>()
            .ok()
    })?;
    kib.checked_mul(1024)
}

struct HarmonicFrequencySystem<'a> {
    elastic: &'a CsrMatrix<f64>,
    mass: &'a CsrMatrix<f64>,
    geometric: &'a CsrMatrix<f64>,
    free_count: usize,
    free_to_full: &'a [usize],
    reduced_load: &'a [f64],
    tip_dofs: &'a [usize],
    static_tip: f64,
    damping_scale: f64,
    vertical_tip_force_n: f64,
    solver_config: &'a SparseDirectConfig,
}

impl HarmonicFrequencySystem<'_> {
    fn solve(&self, frequency_hz: f64) -> Result<FrequencyResponsePoint, StructuralMechanicsError> {
        let omega = std::f64::consts::TAU * frequency_hz;
        let mut triplets =
            Vec::with_capacity(self.elastic.nnz() + self.geometric.nnz() + self.mass.nnz());
        append_complex_triplets(&mut triplets, self.elastic, Complex64::new(1.0, 0.0));
        append_complex_triplets(&mut triplets, self.geometric, Complex64::new(1.0, 0.0));
        append_complex_triplets(
            &mut triplets,
            self.mass,
            Complex64::new(-omega * omega, omega * self.damping_scale),
        );
        let operator = CsrMatrix::from_triplets(self.free_count, self.free_count, triplets);
        let factor = factorize_sparse(&operator, self.solver_config, None).map_err(|error| {
            StructuralMechanicsError::SolverFailure {
                stage: "harmonic",
                message: error.to_string(),
            }
        })?;
        let complex_load = self
            .reduced_load
            .iter()
            .map(|&value| Complex64::new(value, 0.0))
            .collect::<Vec<_>>();
        let solution = factor.solve(&complex_load).map_err(|error| {
            StructuralMechanicsError::SolverFailure {
                stage: "harmonic",
                message: error.to_string(),
            }
        })?;
        let tip = self
            .tip_dofs
            .iter()
            .map(|&full_dof| {
                self.free_to_full
                    .iter()
                    .position(|&mapped| mapped == full_dof)
                    .map(|reduced| solution[reduced])
                    .unwrap_or_else(|| Complex64::new(0.0, 0.0))
            })
            .sum::<Complex64>()
            / self.tip_dofs.len() as f64;
        let transfer = tip / self.vertical_tip_force_n;
        Ok(FrequencyResponsePoint {
            frequency_hz,
            displacement_abs_m: tip.norm(),
            dynamic_magnification: tip.norm() / self.static_tip,
            phase_lag_rad: (-transfer.im).atan2(transfer.re),
        })
    }
}

/// Backward-compatible name for the assembled harmonic solve.
#[deprecated(note = "use solve_harmonic_tip_response to handle typed solver errors")]
pub fn harmonic_tip_response(
    study: &BracketStudy,
    static_result: &StaticBracketResult,
    modal: &ModalSummary,
) -> Vec<FrequencyResponsePoint> {
    solve_harmonic_tip_response(study, static_result, modal)
        .unwrap_or_else(|error| panic!("assembled harmonic solve failed: {error}"))
}

/// Build a finite, monotone logarithmic frequency sweep with typed input
/// failures. The returned grid always contains the positive start frequency.
pub fn try_octave_sweep(
    start_hz: f64,
    stop_hz: f64,
    points_per_octave: usize,
) -> Result<Vec<f64>, StructuralMechanicsError> {
    if !start_hz.is_finite() || start_hz <= 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "octave_sweep_start_hz",
            value: start_hz.to_string(),
        });
    }
    if !stop_hz.is_finite() || stop_hz < start_hz {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "octave_sweep_stop_hz",
            value: stop_hz.to_string(),
        });
    }
    if points_per_octave == 0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "octave_sweep_points_per_octave",
            value: points_per_octave.to_string(),
        });
    }

    let octaves = (stop_hz / start_hz).log2();
    let step_count = octaves * points_per_octave as f64;
    if !step_count.is_finite() || step_count >= usize::MAX as f64 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "octave_sweep_size",
            value: step_count.to_string(),
        });
    }
    let steps = step_count.ceil() as usize;
    const MAX_OCTAVE_SWEEP_POINTS: usize = 1_000_000;
    let point_count = steps.saturating_add(1);
    if point_count > MAX_OCTAVE_SWEEP_POINTS {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "octave_sweep_size",
            value: point_count.to_string(),
        });
    }
    let limit = stop_hz * (1.0 + 1e-12);
    let sweep = (0..=steps)
        .map(|i| start_hz * 2.0_f64.powf(i as f64 / points_per_octave as f64))
        .take_while(|&frequency| frequency.is_finite() && frequency <= limit)
        .collect::<Vec<_>>();
    if sweep.is_empty() || sweep.windows(2).any(|pair| pair[1] <= pair[0]) {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "octave_sweep_grid",
            value: format!("{start_hz}..{stop_hz} @ {points_per_octave}/octave"),
        });
    }
    Ok(sweep)
}

/// Backward-compatible octave sweep wrapper. New callers should use
/// [`try_octave_sweep`] to handle invalid inputs without a panic.
pub fn octave_sweep(start_hz: f64, stop_hz: f64, points_per_octave: usize) -> Vec<f64> {
    try_octave_sweep(start_hz, stop_hz, points_per_octave)
        .unwrap_or_else(|error| panic!("invalid octave sweep: {error}"))
}

pub(crate) struct StructuralMatrices {
    stiffness: TripletMatrix<f64>,
    mass: TripletMatrix<f64>,
    geometric: TripletMatrix<f64>,
}

fn validate_study(study: &BracketStudy) -> Result<(), StructuralMechanicsError> {
    let geometry = study.geometry;
    for (field, value) in [
        ("length_m", geometry.length_m),
        ("height_m", geometry.height_m),
        ("thickness_m", geometry.thickness_m),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field,
                value: value.to_string(),
            });
        }
    }
    if geometry.nx == 0 || geometry.ny == 0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "mesh_density",
            value: format!("nx={}, ny={}", geometry.nx, geometry.ny),
        });
    }
    let material = study.material;
    if !material.youngs_modulus_pa.is_finite() || material.youngs_modulus_pa <= 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "youngs_modulus_pa",
            value: material.youngs_modulus_pa.to_string(),
        });
    }
    if !material.poisson_ratio.is_finite()
        || material.poisson_ratio <= -1.0
        || material.poisson_ratio >= 0.5
    {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "poisson_ratio",
            value: material.poisson_ratio.to_string(),
        });
    }
    if !material.density_kg_m3.is_finite() || material.density_kg_m3 <= 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "density_kg_m3",
            value: material.density_kg_m3.to_string(),
        });
    }
    if !material.damping_ratio.is_finite() || material.damping_ratio < 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "damping_ratio",
            value: material.damping_ratio.to_string(),
        });
    }
    if !study.loads.vertical_tip_force_n.is_finite()
        || study.loads.vertical_tip_force_n.abs() <= f64::EPSILON
    {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "vertical_tip_force_n",
            value: study.loads.vertical_tip_force_n.to_string(),
        });
    }
    if !study.loads.axial_prestress_force_n.is_finite() {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "axial_prestress_force_n",
            value: study.loads.axial_prestress_force_n.to_string(),
        });
    }
    if study.loads.fixed_boundary_marker == study.loads.loaded_boundary_marker {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "boundary_markers",
            value: "fixed and loaded markers must differ".to_string(),
        });
    }
    for &frequency in &study.frequency_hz {
        if !frequency.is_finite() || frequency < 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "frequency_hz",
                value: frequency.to_string(),
            });
        }
    }
    Ok(())
}

pub(crate) fn validate_mesh(
    mesh: &Mesh,
    study: &BracketStudy,
) -> Result<(), StructuralMechanicsError> {
    if mesh.dimension != 2 {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: format!("expected a 2D mesh, got dimension {}", mesh.dimension),
        });
    }
    if mesh.nodes.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "mesh contains no nodes".to_string(),
        });
    }
    if mesh.elements.is_empty() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: "mesh contains no elements".to_string(),
        });
    }

    for (index, point) in mesh.nodes.iter().enumerate() {
        if !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite() {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: format!("node {index} has non-finite coordinates"),
            });
        }
    }

    // Validate all index-bearing connectivity before calling helpers such as
    // `vertices` or `nodes_on_marker`, which intentionally assume a coherent
    // mesh and therefore use direct indexing.
    for (index, element) in mesh.elements.iter().enumerate() {
        if element.nodes.len() < element.element_type.num_vertices() {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: format!(
                    "element {index} has {} nodes, fewer than the {} required vertices",
                    element.nodes.len(),
                    element.element_type.num_vertices()
                ),
            });
        }
        if let Some(&node) = element.nodes.iter().find(|&&node| node >= mesh.nodes.len()) {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: format!("element {index} references missing node {node}"),
            });
        }
    }
    for (index, boundary) in mesh.boundaries.iter().enumerate() {
        if boundary.element_idx >= mesh.elements.len() {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: format!(
                    "boundary {index} references missing element {}",
                    boundary.element_idx
                ),
            });
        }
        if let Some(&node) = boundary
            .nodes
            .iter()
            .find(|&&node| node >= mesh.nodes.len())
        {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: format!("boundary {index} references missing node {node}"),
            });
        }
    }
    if let Err(errors) = mesh.validate_connectivity() {
        return Err(StructuralMechanicsError::InvalidMesh {
            reason: format!("mesh connectivity validation failed: {errors:?}"),
        });
    }

    if nodes_on_marker(mesh, study.loads.fixed_boundary_marker).is_empty() {
        return Err(StructuralMechanicsError::MissingFixedBoundary {
            marker: study.loads.fixed_boundary_marker,
        });
    }
    if nodes_on_marker(mesh, study.loads.loaded_boundary_marker).is_empty() {
        return Err(StructuralMechanicsError::MissingLoadedBoundary {
            marker: study.loads.loaded_boundary_marker,
        });
    }
    for (index, element) in mesh.elements.iter().enumerate() {
        let nodes = element.vertices();
        if nodes.len() != 3 {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: format!("element {index} is not a triangle"),
            });
        }
        let p0 = mesh.nodes[nodes[0]];
        let p1 = mesh.nodes[nodes[1]];
        let p2 = mesh.nodes[nodes[2]];
        let twice_area = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
        if !twice_area.is_finite() || twice_area.abs() <= 1e-15 {
            return Err(StructuralMechanicsError::InvalidMesh {
                reason: format!("element {index} has zero or non-finite area"),
            });
        }
    }
    Ok(())
}

pub(crate) fn assemble_structural_matrices(
    study: &BracketStudy,
    mesh: &Mesh,
    space: &VectorH1Space,
    form: &ElasticityForm,
    degree: PolynomialDegree,
) -> StructuralMatrices {
    let dofs = space.num_global_dofs(mesh);
    let quadrature_order = degree.degree() + 1;
    let mut stiffness = assemble_bilinear(form, space, space, mesh, quadrature_order);
    for value in &mut stiffness.values {
        *value *= study.geometry.thickness_m;
    }

    // The legacy assemble_mass API sizes its result from mesh vertices and is
    // therefore P1-only. The generic vector MassForm uses the same P1/P2
    // space and produces a correctly sized consistent mass matrix.
    let mut mass = assemble_bilinear(&MassForm, space, space, mesh, quadrature_order);
    for value in &mut mass.values {
        *value *= study.material.density_kg_m3 * study.geometry.thickness_m;
    }

    let mut geometric = TripletMatrix::with_capacity(dofs, dofs, mesh.num_elements() * 144);
    let axial_stress = study.loads.axial_prestress_force_n
        / (study.geometry.height_m * study.geometry.thickness_m);
    // Three-point triangle quadrature integrates the P1/P2 geometric form
    // exactly: sigma_xx is constant and each basis-gradient product has
    // degree zero (P1) or two (P2) on the reference triangle.
    let geometric_quadrature = [
        (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
    ];
    for (elem_idx, element) in mesh.elements.iter().enumerate() {
        let nodes = element.vertices();
        let p0 = mesh.nodes[nodes[0]];
        let p1 = mesh.nodes[nodes[1]];
        let p2 = mesh.nodes[nodes[2]];
        let determinant = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
        let local_to_global = space.local_to_global(mesh, elem_idx);
        for &(xi, eta, weight) in &geometric_quadrature {
            let tabulation = space.tabulate(mesh, elem_idx, xi, eta, 0.0);
            let factor = axial_stress * determinant.abs() * study.geometry.thickness_m * weight;
            for (row, &row_dof) in local_to_global.iter().enumerate() {
                let row_component = row % 2;
                let row_gradient_x = tabulation.derivatives[row * 2];
                for (column, &column_dof) in local_to_global.iter().enumerate() {
                    if row_component != column % 2 {
                        continue;
                    }
                    let value = factor * row_gradient_x * tabulation.derivatives[column * 2];
                    geometric.push(row_dof, column_dof, value);
                }
            }
        }
    }

    StructuralMatrices {
        stiffness,
        mass,
        geometric,
    }
}

pub(crate) fn reduce_triplets(
    matrix: &TripletMatrix<f64>,
    constrained: &[usize],
) -> (CsrMatrix<f64>, Vec<usize>) {
    let mut is_constrained = vec![false; matrix.num_rows];
    for &dof in constrained {
        if dof < is_constrained.len() {
            is_constrained[dof] = true;
        }
    }
    let mut free_to_full = Vec::new();
    let mut full_to_free = vec![usize::MAX; matrix.num_rows];
    for (full, constrained) in is_constrained.iter().copied().enumerate() {
        if !constrained {
            full_to_free[full] = free_to_full.len();
            free_to_full.push(full);
        }
    }
    let mut reduced = Vec::with_capacity(matrix.nnz());
    for index in 0..matrix.nnz() {
        let row = full_to_free[matrix.rows[index]];
        let col = full_to_free[matrix.cols[index]];
        if row != usize::MAX && col != usize::MAX {
            reduced.push((row, col, matrix.values[index]));
        }
    }
    (
        CsrMatrix::from_triplets(free_to_full.len(), free_to_full.len(), reduced),
        free_to_full,
    )
}

fn add_csr(lhs: &CsrMatrix<f64>, rhs: &CsrMatrix<f64>) -> CsrMatrix<f64> {
    let mut triplets = Vec::with_capacity(lhs.values.len() + rhs.values.len());
    for matrix in [lhs, rhs] {
        for row in 0..matrix.num_rows {
            for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                triplets.push((row, matrix.col_indices[index], matrix.values[index]));
            }
        }
    }
    CsrMatrix::from_triplets(lhs.num_rows, lhs.num_cols, triplets)
}

fn scale_triplet(matrix: &TripletMatrix<f64>, scale: f64) -> TripletMatrix<f64> {
    let mut scaled = TripletMatrix::with_capacity(matrix.num_rows, matrix.num_cols, matrix.nnz());
    for index in 0..matrix.nnz() {
        scaled.push(
            matrix.rows[index],
            matrix.cols[index],
            scale * matrix.values[index],
        );
    }
    scaled
}

fn append_complex_triplets(
    output: &mut Vec<(usize, usize, Complex64)>,
    matrix: &CsrMatrix<f64>,
    scale: Complex64,
) {
    for row in 0..matrix.num_rows {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            output.push((
                row,
                matrix.col_indices[index],
                Complex64::new(matrix.values[index], 0.0) * scale,
            ));
        }
    }
}

fn triplet_matvec(matrix: &TripletMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    let mut result = vec![0.0; matrix.num_rows];
    for index in 0..matrix.nnz() {
        result[matrix.rows[index]] += matrix.values[index] * vector[matrix.cols[index]];
    }
    result
}

fn dot(lhs: &[f64], rhs: &[f64]) -> f64 {
    lhs.iter().zip(rhs).map(|(a, b)| a * b).sum()
}

fn component_force(vector: &[f64]) -> [f64; 2] {
    vector
        .as_chunks::<2>()
        .0
        .iter()
        .fold([0.0; 2], |mut total, pair| {
            total[0] += pair[0];
            total[1] += pair[1];
            total
        })
}

fn reaction_force(internal: &[f64], external: &[f64], fixed_dofs: &[usize]) -> [f64; 2] {
    fixed_dofs.iter().fold([0.0; 2], |mut total, &dof| {
        total[dof % 2] += internal[dof] - external[dof];
        total
    })
}

fn free_residual_norm(
    internal: &[f64],
    external: &[f64],
    dirichlet: &std::collections::HashSet<usize>,
) -> f64 {
    internal
        .iter()
        .zip(external)
        .enumerate()
        .filter(|(dof, _)| !dirichlet.contains(dof))
        .map(|(_, (lhs, rhs))| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn apply_tip_edge_load(mesh: &Mesh, study: &BracketStudy, rhs: &mut [f64]) {
    apply_tip_edge_load_components(
        mesh,
        study,
        rhs,
        study.loads.axial_prestress_force_n,
        study.loads.vertical_tip_force_n,
    );
}

pub(crate) fn apply_tip_edge_load_order(
    mesh: &Mesh,
    study: &BracketStudy,
    space: &VectorH1Space,
    degree: PolynomialDegree,
    rhs: &mut [f64],
) {
    match degree {
        PolynomialDegree::P1 => apply_tip_edge_load(mesh, study, rhs),
        PolynomialDegree::P2 => apply_tip_edge_load_p2(mesh, study, space, rhs),
        PolynomialDegree::P3 => unreachable!("structural bracket supports P1 and P2 only"),
    }
}

fn apply_tip_edge_load_p2(
    mesh: &Mesh,
    study: &BracketStudy,
    space: &VectorH1Space,
    rhs: &mut [f64],
) {
    debug_assert_eq!(space.num_components(), 2);
    let topology = MeshTopology::build(mesh);
    let quadrature = gauss_legendre_1d(3);
    let traction_y =
        study.loads.vertical_tip_force_n / (study.geometry.height_m * study.geometry.thickness_m);
    let traction_x = study.loads.axial_prestress_force_n
        / (study.geometry.height_m * study.geometry.thickness_m);
    let edge_dof_offset = mesh.num_nodes();

    for boundary in &mesh.boundaries {
        if boundary.marker != study.loads.loaded_boundary_marker || boundary.nodes.len() != 2 {
            continue;
        }
        let node0 = boundary.nodes[0];
        let node1 = boundary.nodes[1];
        let a = mesh.nodes[node0];
        let b = mesh.nodes[node1];
        let length = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt();
        let edge_key = if node0 < node1 {
            (node0, node1)
        } else {
            (node1, node0)
        };
        let Some(&edge_index) = topology.edge_map.get(&edge_key) else {
            continue;
        };
        let midpoint = edge_dof_offset + edge_index;

        for point in &quadrature {
            let t = 0.5 * (point.xi() + 1.0);
            let n0 = (1.0 - t) * (1.0 - 2.0 * t);
            let n_mid = 4.0 * t * (1.0 - t);
            let n1 = t * (2.0 * t - 1.0);
            let scale = 0.5 * point.weight * length * study.geometry.thickness_m;
            for (dof, shape) in [(node0 * 2, n0), (midpoint * 2, n_mid), (node1 * 2, n1)] {
                rhs[dof] += traction_x * shape * scale;
                rhs[dof + 1] += traction_y * shape * scale;
            }
        }
    }
}

fn apply_tip_edge_load_components(
    mesh: &Mesh,
    study: &BracketStudy,
    rhs: &mut [f64],
    force_x: f64,
    force_y: f64,
) {
    let traction_y = force_y / (study.geometry.height_m * study.geometry.thickness_m);
    let traction_x = force_x / (study.geometry.height_m * study.geometry.thickness_m);
    for boundary in &mesh.boundaries {
        if boundary.marker != study.loads.loaded_boundary_marker || boundary.nodes.len() != 2 {
            continue;
        }
        let a = mesh.nodes[boundary.nodes[0]];
        let b = mesh.nodes[boundary.nodes[1]];
        let length = ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt();
        let scale = 0.5 * length * study.geometry.thickness_m;
        for &node in &boundary.nodes {
            rhs[node * 2] += traction_x * scale;
            rhs[node * 2 + 1] += traction_y * scale;
        }
    }
}

fn nodes_on_marker(mesh: &Mesh, marker: i32) -> Vec<usize> {
    let mut nodes = Vec::new();
    for boundary in &mesh.boundaries {
        if boundary.marker == marker {
            nodes.extend(boundary.nodes.iter().copied());
        }
    }
    nodes.sort_unstable();
    nodes.dedup();
    nodes
}

fn boundary_component_dofs(
    mesh: &Mesh,
    space: &VectorH1Space,
    marker: i32,
    component: usize,
) -> Vec<usize> {
    let components = space.num_components();
    space
        .boundary_dofs(mesh, marker)
        .into_iter()
        .filter(|dof| dof % components == component)
        .collect()
}

fn nodal_displacement(x: &[f64], node_count: usize) -> Vec<[f64; 2]> {
    (0..node_count)
        .map(|node| [x[node * 2], x[node * 2 + 1]])
        .collect()
}

fn displacement_fields(displacement: &[[f64; 2]]) -> (Vec<f64>, Vec<f64>, Vec<f64>, Vec<f64>) {
    let ux: Vec<f64> = displacement.iter().map(|d| d[0]).collect();
    let uy: Vec<f64> = displacement.iter().map(|d| d[1]).collect();
    let uz = vec![0.0; displacement.len()];
    let magnitude: Vec<f64> = displacement
        .iter()
        .map(|[ux, uy]| (ux * ux + uy * uy).sqrt())
        .collect();
    (ux, uy, uz, magnitude)
}

fn visible_warp_scale(study: &BracketStudy, result: &StaticBracketResult) -> f64 {
    if result.max_displacement_m <= f64::EPSILON {
        1.0
    } else {
        0.18 * study.geometry.length_m / result.max_displacement_m
    }
}

fn beam_tip_deflection(study: &BracketStudy) -> f64 {
    let force = study.loads.vertical_tip_force_n;
    let l = study.geometry.length_m;
    let e = study.material.youngs_modulus_pa;
    let i = study.geometry.second_moment_m4();
    force * l.powi(3) / (3.0 * e * i)
}

fn element_von_mises(mesh: &Mesh, study: &BracketStudy, displacement: &[[f64; 2]]) -> Vec<f64> {
    let c = match study.plane_model {
        PlaneModel::PlaneStress => study.material.as_fem().voigt_2d_plane_stress(),
        PlaneModel::PlaneStrain => study.material.as_fem().voigt_2d_plane_strain(),
    };

    mesh.elements
        .iter()
        .map(|element| {
            let nodes = element.vertices();
            let grads = triangle_shape_gradients(mesh, nodes);
            let mut dux_dx = 0.0;
            let mut dux_dy = 0.0;
            let mut duy_dx = 0.0;
            let mut duy_dy = 0.0;

            for (local, &node) in nodes.iter().enumerate() {
                let [ux, uy] = displacement[node];
                dux_dx += ux * grads[local][0];
                dux_dy += ux * grads[local][1];
                duy_dx += uy * grads[local][0];
                duy_dy += uy * grads[local][1];
            }

            let strain = [dux_dx, duy_dy, dux_dy + duy_dx];
            let sx = c[0] * strain[0] + c[1] * strain[1] + c[2] * strain[2];
            let sy = c[3] * strain[0] + c[4] * strain[1] + c[5] * strain[2];
            let txy = c[6] * strain[0] + c[7] * strain[1] + c[8] * strain[2];

            match study.plane_model {
                PlaneModel::PlaneStress => (sx * sx - sx * sy + sy * sy + 3.0 * txy * txy).sqrt(),
                PlaneModel::PlaneStrain => {
                    let nu = study.material.poisson_ratio;
                    let sz = nu * (sx + sy);
                    (0.5 * ((sx - sy).powi(2) + (sy - sz).powi(2) + (sz - sx).powi(2))
                        + 3.0 * txy * txy)
                        .sqrt()
                }
            }
        })
        .collect()
}

fn max_von_mises(mesh: &Mesh, study: &BracketStudy, displacement: &[[f64; 2]]) -> f64 {
    element_von_mises(mesh, study, displacement)
        .into_iter()
        .fold(0.0, f64::max)
}

fn max_von_mises_for_order(
    mesh: &Mesh,
    study: &BracketStudy,
    space: &VectorH1Space,
    solution: &[f64],
    polynomial_order: StructuralPolynomialOrder,
    vertex_displacement: &[[f64; 2]],
) -> f64 {
    if polynomial_order == StructuralPolynomialOrder::Linear {
        return max_von_mises(mesh, study, vertex_displacement);
    }

    (0..mesh.num_elements())
        .map(|elem_idx| element_von_mises_from_space(mesh, study, space, solution, elem_idx))
        .fold(0.0, f64::max)
}

fn element_von_mises_from_space(
    mesh: &Mesh,
    study: &BracketStudy,
    space: &VectorH1Space,
    solution: &[f64],
    elem_idx: usize,
) -> f64 {
    let tabulation = space.tabulate(mesh, elem_idx, 1.0 / 3.0, 1.0 / 3.0, 0.0);
    let local_to_global = space.local_to_global(mesh, elem_idx);
    let mut gradient = [[0.0; 2]; 2];
    for (local_dof, &global_dof) in local_to_global.iter().enumerate() {
        let component = local_dof % 2;
        let value = solution[global_dof];
        gradient[component][0] += value * tabulation.derivatives[local_dof * 2];
        gradient[component][1] += value * tabulation.derivatives[local_dof * 2 + 1];
    }

    let c = match study.plane_model {
        PlaneModel::PlaneStress => study.material.as_fem().voigt_2d_plane_stress(),
        PlaneModel::PlaneStrain => study.material.as_fem().voigt_2d_plane_strain(),
    };
    let strain = [
        gradient[0][0],
        gradient[1][1],
        gradient[0][1] + gradient[1][0],
    ];
    let sx = c[0] * strain[0] + c[1] * strain[1] + c[2] * strain[2];
    let sy = c[3] * strain[0] + c[4] * strain[1] + c[5] * strain[2];
    let txy = c[6] * strain[0] + c[7] * strain[1] + c[8] * strain[2];

    match study.plane_model {
        PlaneModel::PlaneStress => (sx * sx - sx * sy + sy * sy + 3.0 * txy * txy).sqrt(),
        PlaneModel::PlaneStrain => {
            let nu = study.material.poisson_ratio;
            let sz = nu * (sx + sy);
            (0.5 * ((sx - sy).powi(2) + (sy - sz).powi(2) + (sz - sx).powi(2)) + 3.0 * txy * txy)
                .sqrt()
        }
    }
}

fn triangle_shape_gradients(mesh: &Mesh, nodes: &[usize]) -> [[f64; 2]; 3] {
    assert_eq!(nodes.len(), 3);
    let p0 = mesh.nodes[nodes[0]];
    let p1 = mesh.nodes[nodes[1]];
    let p2 = mesh.nodes[nodes[2]];
    let det = (p1.x - p0.x) * (p2.y - p0.y) - (p2.x - p0.x) * (p1.y - p0.y);
    [
        [(p1.y - p2.y) / det, (p2.x - p1.x) / det],
        [(p2.y - p0.y) / det, (p0.x - p2.x) / det],
        [(p0.y - p1.y) / det, (p1.x - p0.x) / det],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    // Single test for the parallel gate and both collect paths: process env
    // is shared by all test threads, so env-mutating checks must not run
    // concurrently with each other.
    #[test]
    fn structural_parallel_policy() {
        // SAFETY: only path selection reads these keys, and both the serial
        // and parallel collect paths are exactly correct, so any concurrent
        // non-env test can at most take the other correct path.
        unsafe {
            let prev_threshold = std::env::var_os("SONIUM_STRUCTURAL_PAR_THRESHOLD");
            let prev_serial = std::env::var_os("SONIUM_SERIAL");
            std::env::remove_var("SONIUM_STRUCTURAL_PAR_THRESHOLD");
            std::env::remove_var("SONIUM_SERIAL");

            let threads = std::thread::available_parallelism()
                .map(usize::from)
                .unwrap_or(1);
            if threads < 2 {
                assert!(!use_structural_parallel(
                    usize::MAX / 2,
                    STRUCTURAL_SOLVE_WORK_PER_THREAD
                ));
            } else {
                // Tiny tutorial-size work stays serial; huge work
                // parallelizes.
                assert!(!use_structural_parallel(
                    1,
                    STRUCTURAL_SOLVE_WORK_PER_THREAD
                ));
                assert!(use_structural_parallel(
                    usize::MAX / 2,
                    STRUCTURAL_SOLVE_WORK_PER_THREAD
                ));
                // The decision scales with threads, not a fixed count.
                assert!(use_structural_parallel(
                    threads * STRUCTURAL_SOLVE_WORK_PER_THREAD,
                    STRUCTURAL_SOLVE_WORK_PER_THREAD
                ));
                assert!(!use_structural_parallel(
                    threads * STRUCTURAL_SOLVE_WORK_PER_THREAD - 1,
                    STRUCTURAL_SOLVE_WORK_PER_THREAD
                ));
            }

            // Forced-serial collect matches the serial reference, and the
            // forced-parallel collect (threshold 1) agrees exactly.
            let items: Vec<usize> = (0..32).collect();
            let expected: Vec<usize> = items.iter().map(|x| x * 2).collect();
            std::env::set_var("SONIUM_SERIAL", "1");
            let serial: Vec<usize> = parallel_collect_values(&items, usize::MAX, 1, |x| x * 2);
            assert_eq!(serial, expected);
            std::env::remove_var("SONIUM_SERIAL");
            std::env::set_var("SONIUM_STRUCTURAL_PAR_THRESHOLD", "1");
            let parallel: Vec<usize> = parallel_collect_values(&items, usize::MAX, 1, |x| x * 2);
            assert_eq!(parallel, expected);

            match prev_threshold {
                Some(value) => std::env::set_var("SONIUM_STRUCTURAL_PAR_THRESHOLD", value),
                None => std::env::remove_var("SONIUM_STRUCTURAL_PAR_THRESHOLD"),
            }
            match prev_serial {
                Some(value) => std::env::set_var("SONIUM_SERIAL", value),
                None => std::env::remove_var("SONIUM_SERIAL"),
            }
        }
    }

    #[test]
    fn non_convergence_preserves_solver_status() {
        assert_eq!(
            require_converged(false, 37, 2.5e-4),
            Err(StructuralMechanicsError::NonConvergence {
                iterations: 37,
                residual: 2.5e-4,
            })
        );
        assert_eq!(require_converged(true, 4, 1.0e-12), Ok(()));
    }

    #[test]
    fn introductory_study_follows_static_dynamic_modal_frequency_sequence() {
        let study = BracketStudy::introductory_bracket();
        let names: Vec<&str> = study.steps.iter().map(|step| step.name.as_str()).collect();
        assert_eq!(
            names,
            vec![
                "linear-static",
                "time-dependent-damped",
                "eigenfrequency",
                "frequency-domain-modal"
            ]
        );
        assert!(study.frequency_hz.len() > 10);
    }

    #[test]
    fn static_bracket_deflects_down_and_recovers_stress() {
        let mut study = BracketStudy::introductory_bracket();
        study.geometry.nx = 24;
        study.geometry.ny = 6;

        let result = run_static_bracket(&study).unwrap();

        assert!(result.converged);
        assert!(result.average_tip_deflection_m < 0.0);
        assert!(result.max_displacement_m > 0.0);
        assert!(result.max_von_mises_pa > 0.0);
        assert!(result.strain_energy_j > 0.0);
        assert!(result.beam_theory_relative_error < 0.35);
        assert!((result.applied_force_n[0] - study.loads.axial_prestress_force_n).abs() < 1e-10);
        assert!((result.applied_force_n[1] - study.loads.vertical_tip_force_n).abs() < 1e-10);
        let force_scale = (result.applied_force_n[0].powi(2) + result.applied_force_n[1].powi(2))
            .sqrt()
            .max(1.0);
        assert!(result.force_balance_residual_n < 1e-6 * force_scale);
        assert!(result.free_equilibrium_residual_n < 1e-6 * force_scale);
        assert!(result.energy_balance_relative_error < 1e-12);
        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::PlaneStress
        );
        assert_eq!(
            result.stress_recovery,
            StressRecoveryPolicy::ElementCentroid
        );
    }

    #[test]
    fn assembled_modal_result_is_mass_normalized() {
        let mut study = BracketStudy::introductory_bracket();
        study.geometry.nx = 12;
        study.geometry.ny = 3;
        let static_result = run_static_bracket(&study).unwrap();
        let modal = solve_first_bending_mode(&study, &static_result).unwrap();

        assert!(modal.eigenvalue > 0.0);
        assert!(modal.first_bending_frequency_hz > 0.0);
        assert!(modal.solver_residual.is_finite());
        assert!(modal.mass_normalization_error < 1e-10);
        assert!(modal.orthogonality_error.abs() < 1e-8);
        assert_eq!(modal.normalization, ModalNormalization::MassNormalized);
        assert_eq!(
            modal.damping_convention,
            DampingConvention::MassProportionalAtFirstMode
        );
    }

    #[test]
    fn zero_frequency_harmonic_response_matches_static_limit() {
        let mut study = BracketStudy::introductory_bracket();
        study.geometry.nx = 12;
        study.geometry.ny = 3;
        study.loads.axial_prestress_force_n = 0.0;
        study.frequency_hz = vec![0.0];
        let static_result = run_static_bracket(&study).unwrap();
        let modal = solve_first_bending_mode(&study, &static_result).unwrap();
        let response = solve_harmonic_tip_response(&study, &static_result, &modal).unwrap();

        assert_eq!(response.len(), 1);
        assert!(
            (response[0].displacement_abs_m - static_result.average_tip_deflection_m.abs())
                / static_result.average_tip_deflection_m.abs()
                < 1e-8
        );
        assert!(response[0].phase_lag_rad.abs() < 1e-10);
    }

    #[test]
    fn quadratic_modal_and_harmonic_paths_preserve_static_limit() {
        let mut study = BracketStudy::introductory_bracket();
        study.geometry.nx = 8;
        study.geometry.ny = 2;
        study.loads.axial_prestress_force_n = 0.0;
        study.frequency_hz = vec![0.0];
        let static_result =
            run_static_bracket_order(&study, StructuralPolynomialOrder::Quadratic).unwrap();
        let modal = solve_first_bending_mode_order(
            &study,
            &static_result,
            StructuralPolynomialOrder::Quadratic,
        )
        .unwrap();
        let response = solve_harmonic_tip_response_order(
            &study,
            &static_result,
            &modal,
            StructuralPolynomialOrder::Quadratic,
        )
        .unwrap();

        assert_eq!(modal.polynomial_order, StructuralPolynomialOrder::Quadratic);
        assert!(modal.eigenvalue > 0.0);
        assert!(modal.mass_normalization_error < 1e-10);
        assert!(modal.orthogonality_error.abs() < 1e-8);
        assert_eq!(response.len(), 1);
        assert!(
            (response[0].displacement_abs_m - static_result.average_tip_deflection_m.abs())
                / static_result.average_tip_deflection_m.abs()
                < 1e-8
        );
        assert!(response[0].phase_lag_rad.abs() < 1e-10);
    }

    #[test]
    fn parallel_harmonic_sweep_preserves_order_and_serial_values() {
        let mut study = BracketStudy::introductory_bracket();
        study.geometry.nx = 4;
        study.geometry.ny = 1;
        study.loads.axial_prestress_force_n = 0.0;
        study.frequency_hz = vec![0.0, 250.0, 1_000.0];
        let static_result = run_static_bracket(&study).unwrap();
        let modal = solve_first_bending_mode(&study, &static_result).unwrap();
        let serial = solve_harmonic_tip_response(&study, &static_result, &modal).unwrap();
        let parallel = solve_harmonic_tip_response_parallel_order(
            &study,
            &static_result,
            &modal,
            StructuralPolynomialOrder::Linear,
            2,
        )
        .unwrap();

        assert_eq!(parallel.len(), serial.len());
        for (expected, actual) in serial.iter().zip(&parallel) {
            assert_eq!(actual.frequency_hz, expected.frequency_hz);
            assert!((actual.displacement_abs_m - expected.displacement_abs_m).abs() < 1e-12);
            assert!((actual.dynamic_magnification - expected.dynamic_magnification).abs() < 1e-12);
            assert!((actual.phase_lag_rad - expected.phase_lag_rad).abs() < 1e-12);
        }

        let error = solve_harmonic_tip_response_parallel_order(
            &study,
            &static_result,
            &modal,
            StructuralPolynomialOrder::Linear,
            0,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            StructuralMechanicsError::InvalidConfiguration {
                field: "harmonic_parallel_threads",
                ..
            }
        ));
    }

    #[test]
    fn modal_order_mismatch_is_rejected_before_assembly() {
        let mut study = BracketStudy::introductory_bracket();
        study.geometry.nx = 4;
        study.geometry.ny = 1;
        let static_result = run_static_bracket(&study).unwrap();
        let error = solve_first_bending_mode_order(
            &study,
            &static_result,
            StructuralPolynomialOrder::Quadratic,
        )
        .unwrap_err();
        assert!(matches!(
            error,
            StructuralMechanicsError::InvalidConfiguration {
                field: "static_result.polynomial_order",
                ..
            }
        ));
    }

    #[test]
    fn invalid_boundary_marker_is_reported_without_panicking() {
        let mut study = BracketStudy::introductory_bracket();
        study.loads.fixed_boundary_marker = 99;
        assert_eq!(
            run_static_bracket(&study),
            Err(StructuralMechanicsError::MissingFixedBoundary { marker: 99 })
        );
    }

    #[test]
    fn invalid_element_connectivity_returns_typed_mesh_error() {
        let study = BracketStudy::introductory_bracket();
        let mut mesh = study.mesh();
        let missing_node = mesh.nodes.len();
        mesh.elements[0].nodes[0] = missing_node;

        let error = run_static_bracket_on_mesh(&study, &mesh).unwrap_err();
        assert!(matches!(
            error,
            StructuralMechanicsError::InvalidMesh { ref reason }
                if reason.contains("missing node")
        ));
    }

    #[test]
    fn invalid_boundary_connectivity_returns_typed_mesh_error() {
        let study = BracketStudy::introductory_bracket();
        let mut mesh = study.mesh();
        let missing_node = mesh.nodes.len();
        mesh.boundaries[0].nodes[0] = missing_node;

        let error = run_static_bracket_on_mesh(&study, &mesh).unwrap_err();
        assert!(matches!(
            error,
            StructuralMechanicsError::InvalidMesh { ref reason }
                if reason.contains("boundary") && reason.contains("missing node")
        ));
    }

    #[test]
    fn boundary_ownership_mismatch_returns_typed_mesh_error() {
        let study = BracketStudy::introductory_bracket();
        let mut mesh = study.mesh();
        let boundary = &mesh.boundaries[0];
        let owner_nodes = mesh.elements[boundary.element_idx].nodes.clone();
        let replacement = (0..mesh.nodes.len())
            .find(|node| !owner_nodes.contains(node))
            .expect("fixture must contain a node outside the boundary owner");
        mesh.boundaries[0].nodes[0] = replacement;

        let error = run_static_bracket_on_mesh(&study, &mesh).unwrap_err();
        assert!(matches!(
            error,
            StructuralMechanicsError::InvalidMesh { ref reason }
                if reason.contains("connectivity") && reason.contains("BoundaryNodeSetMismatch")
        ));
    }

    #[test]
    fn mesh_convergence_reports_refined_evidence() {
        let mut study = BracketStudy::introductory_bracket();
        study.geometry.nx = 8;
        study.geometry.ny = 2;
        let report = run_static_mesh_convergence(&study, 0.2).unwrap();

        assert!(report.refined.nodes > report.coarse.nodes);
        assert!(report.displacement_relative_change.is_finite());
        assert!(report.energy_relative_change.is_finite());
        assert!(report.stress_relative_change.is_finite());
        assert_eq!(
            report.stress_converged,
            report.stress_relative_change <= report.tolerance
        );
    }

    #[test]
    fn quadratic_static_path_uses_edge_dofs_and_preserves_balance() {
        let mut study = BracketStudy::introductory_bracket();
        study.geometry.nx = 8;
        study.geometry.ny = 2;

        let linear = run_static_bracket_order(&study, StructuralPolynomialOrder::Linear).unwrap();
        let quadratic =
            run_static_bracket_order(&study, StructuralPolynomialOrder::Quadratic).unwrap();

        assert_eq!(linear.polynomial_order, StructuralPolynomialOrder::Linear);
        assert_eq!(
            quadratic.polynomial_order,
            StructuralPolynomialOrder::Quadratic
        );
        assert!(quadratic.dofs > linear.dofs);
        assert!(quadratic.max_displacement_m > 0.0);
        assert!(quadratic.max_von_mises_pa > 0.0);
        assert!(quadratic.strain_energy_j > 0.0);
        assert!(quadratic.free_equilibrium_residual_n < 1.0e-6);
        assert!(quadratic.force_balance_residual_n < 1.0e-6);
        assert!(quadratic.energy_balance_relative_error < 1.0e-12);
        assert!(
            (quadratic.applied_force_n[0] - study.loads.axial_prestress_force_n).abs() < 1.0e-10
        );
        assert!((quadratic.applied_force_n[1] - study.loads.vertical_tip_force_n).abs() < 1.0e-10);
    }

    #[test]
    fn order_convergence_report_contains_actual_linear_and_quadratic_solves() {
        let mut study = BracketStudy::introductory_bracket();
        study.geometry.nx = 8;
        study.geometry.ny = 2;
        let report = run_static_order_convergence(&study, 0.2).unwrap();

        assert_eq!(
            report.linear.polynomial_order,
            StructuralPolynomialOrder::Linear
        );
        assert_eq!(
            report.quadratic.polynomial_order,
            StructuralPolynomialOrder::Quadratic
        );
        assert!(report.quadratic.dofs > report.linear.dofs);
        assert!(report.displacement_relative_change.is_finite());
        assert!(report.energy_relative_change.is_finite());
        assert!(report.stress_relative_change.is_finite());
        assert_eq!(
            report.stress_converged,
            report.stress_relative_change <= report.tolerance
        );
    }

    #[test]
    fn assembled_frequency_response_peaks_near_modal_frequency() {
        let mut study = BracketStudy::introductory_bracket();
        study.frequency_hz = octave_sweep(20.0, 5_000.0, 12);
        let static_result = run_static_bracket(&study).unwrap();
        let modal = solve_first_bending_mode(&study, &static_result).unwrap();
        let response = solve_harmonic_tip_response(&study, &static_result, &modal).unwrap();
        let peak = response
            .iter()
            .max_by(|a, b| {
                a.dynamic_magnification
                    .partial_cmp(&b.dynamic_magnification)
                    .unwrap()
            })
            .unwrap();

        assert!(modal.first_bending_frequency_hz > 0.0);
        assert!(peak.dynamic_magnification > 1.0);
    }

    #[test]
    fn checked_octave_sweep_rejects_invalid_inputs_and_returns_ordered_grid() {
        assert!(matches!(
            try_octave_sweep(0.0, 100.0, 12),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "octave_sweep_start_hz",
                ..
            })
        ));
        assert!(matches!(
            try_octave_sweep(100.0, f64::NAN, 12),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "octave_sweep_stop_hz",
                ..
            })
        ));
        assert!(matches!(
            try_octave_sweep(100.0, 200.0, 0),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "octave_sweep_points_per_octave",
                ..
            })
        ));
        assert!(matches!(
            try_octave_sweep(1.0, 1.0e308, 2_000),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "octave_sweep_size",
                ..
            })
        ));

        let sweep = try_octave_sweep(20.0, 2_000.0, 6).expect("valid checked octave sweep");
        assert_eq!(sweep.first().copied(), Some(20.0));
        assert!(
            sweep
                .last()
                .is_some_and(|&frequency| frequency <= 2_000.0 * (1.0 + 1e-12))
        );
        assert!(sweep.windows(2).all(|pair| pair[1] > pair[0]));
    }

    #[test]
    fn axial_prestress_changes_frequency_with_the_correct_sign() {
        let mut unloaded = BracketStudy::introductory_bracket();
        unloaded.loads.axial_prestress_force_n = 0.0;
        let static_result = run_static_bracket(&unloaded).unwrap();
        let unloaded_mode = solve_first_bending_mode(&unloaded, &static_result).unwrap();

        let mut tensioned = unloaded.clone();
        tensioned.loads.axial_prestress_force_n = 10_000.0;
        let tensioned_mode = solve_first_bending_mode(&tensioned, &static_result).unwrap();

        let mut compressed = unloaded.clone();
        compressed.loads.axial_prestress_force_n = -10_000.0;
        let compressed_mode = solve_first_bending_mode(&compressed, &static_result).unwrap();

        assert!(tensioned_mode.geometric_stiffness_n_per_m > 0.0);
        assert!(compressed_mode.geometric_stiffness_n_per_m < 0.0);
        assert!(
            tensioned_mode.first_bending_frequency_hz > unloaded_mode.first_bending_frequency_hz
        );
        assert!(
            compressed_mode.first_bending_frequency_hz < unloaded_mode.first_bending_frequency_hz
        );
    }

    #[test]
    fn writes_paraview_visualization_artifacts() {
        let dir = std::env::temp_dir().join(format!(
            "sonium_structural_visualization_{}",
            std::process::id()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let artifacts = write_introductory_bracket_visualization(&dir).unwrap();

        assert!(artifacts.report_json.exists());
        assert!(artifacts.result_vtu.exists());
        assert!(artifacts.deformed_preview_vtu.exists());
        assert!(artifacts.warp_scale > 1.0);

        let vtu = std::fs::read_to_string(&artifacts.result_vtu).unwrap();
        assert!(vtu.contains("displacement"));
        assert!(vtu.contains("von_mises_pa"));

        let _ = std::fs::remove_dir_all(&dir);
    }
}
