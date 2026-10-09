//! Propeller aeroacoustic simulation helpers.
//!
//! This crate keeps application-level propeller design studies outside the FEM
//! core. The first model is an axisymmetric source-noise surrogate for shrouded
//! propellers, with explicit modifiers for toroidal loop blades and bio-inspired
//! leading/trailing-edge serrations.

pub mod aeroelastic_reference;
pub mod axisymmetric_flow_reference;
pub mod axisymmetric_meridional_flow_reference;
pub mod blade3d;
pub mod blade_element;
pub mod body_fitted_flow_reference;
pub mod centrifugal_modal_reference;
pub mod compressibility_reference;
pub mod coupled_swirl_transport_reference;
#[cfg(feature = "cuda")]
pub mod cuda_adapter;
pub mod duct_shroud_reference;
pub mod dynamic_stall_reference;
pub mod execution_evidence;
pub mod farassat_fwh_reference;
pub mod fatigue_reference;
pub mod fidelity_promotion_reference;
pub mod free_wake_reference;
pub mod fwh_reference;
pub mod gallery;
pub mod harmonic_signal_reference;
pub mod installation_reference;
pub mod load_coupled_free_wake;
pub mod mean_flow_fwh_reference;
pub mod measured_calibration_reference;
pub mod measured_correlation_reference;
pub mod measured_dataset_attestation_reference;
pub mod measured_dataset_reference;
pub mod measured_uncertainty_reference;
pub mod mesh_convergence;
pub mod mesh_state_transfer;
pub mod motor_matching;
pub mod moving_fwh_reference;
pub mod multi_rotor_bem_reference;
pub mod multi_rotor_reference;
pub mod nonlinear_aeroelastic_reference;
pub mod operating_envelope;
pub mod optimization3d;
pub mod panel_load_reference;
pub mod panel_refinement_reference;
pub(crate) mod parallelism;
pub mod provenance;
pub mod radiation_reference;
pub mod robust_operating_reference;
pub mod source_model;
pub mod structural_reference;
pub mod swirl_transport_reference;
pub mod transient_free_wake_reference;
pub mod transient_load_coupled_free_wake;
pub mod viscous_core_reference;
pub mod vortex_lattice_reference;
pub mod wake_convergence;
pub mod wake_reference;

pub use gallery::{GalleryBundleValidation, validate_gallery_bundle, write_gallery_bundle};

pub use mean_flow_fwh_reference::{
    MeanFlowFwhElementResult, MeanFlowFwhError, MeanFlowFwhOptions, MeanFlowFwhResult,
    evaluate_mean_flow_fwh_reference,
};
pub use measured_calibration_reference::{
    MeasuredCalibrationError, MeasuredCalibrationOptions, MeasuredCalibrationPoint,
    MeasuredCalibrationResult, MeasuredCalibrationSample, calibrate_measured_response,
};
pub use measured_correlation_reference::{
    MeasuredHarmonicCorrelationError, MeasuredHarmonicCorrelationOptions,
    MeasuredHarmonicCorrelationPoint, MeasuredHarmonicCorrelationResult, MeasuredHarmonicPoint,
    correlate_measured_harmonics,
};
pub use measured_dataset_attestation_reference::{
    MEASURED_DATASET_ATTESTATION_SCHEMA_V1, MeasuredDatasetAttestation,
    MeasuredDatasetAttestationError, MeasuredDatasetTrustAnchor, MeasuredDatasetTrustPolicy,
    measured_dataset_attestation_payload,
};
pub use measured_dataset_reference::{
    MEASURED_CALIBRATION_DATASET_SCHEMA_V1, MeasuredCalibrationDataset,
    MeasuredCalibrationDatasetError, MeasuredCalibrationDatasetMetadata,
};
pub use measured_uncertainty_reference::{
    MeasuredUncertaintyError, MeasuredUncertaintyOptions, MeasuredUncertaintyPoint,
    MeasuredUncertaintyResult, MeasuredUncertaintySample, evaluate_measured_uncertainty,
};
pub use mesh_state_transfer::{
    PanelStateTransferError, PanelStateTransferOptions, PanelStateTransferResult,
    transfer_panel_scalar_state,
};
pub use moving_fwh_reference::{
    MovingFwhElementHistory, MovingFwhElementResult, MovingFwhError, MovingFwhOptions,
    MovingFwhResult, MovingFwhSample, evaluate_moving_fwh_reference,
};
pub use transient_free_wake_reference::{
    TransientFreeWakeError, TransientFreeWakeOptions, TransientFreeWakeResult,
    TransientFreeWakeStep, advance_prescribed_free_wake,
    advance_prescribed_free_wake_with_shedding,
};
pub use transient_load_coupled_free_wake::{
    TransientLoadCoupledFreeWakeError, TransientLoadCoupledFreeWakeOptions,
    TransientLoadCoupledFreeWakeResult, TransientLoadCoupledFreeWakeStep,
    evaluate_transient_load_coupled_free_wake,
};

pub use aeroelastic_reference::{
    BladeAeroelasticError, BladeAeroelasticOptions, BladeAeroelasticResult,
    BladeAeroelasticStation, evaluate_blade_aeroelastic_reference,
};
pub use axisymmetric_flow_reference::{
    AXISYMMETRIC_FLOW_REFERENCE_SCHEMA_V1, AXISYMMETRIC_ROTATING_WALL_SCHEMA_V1,
    AxisymmetricFlowError, AxisymmetricFlowOptions, AxisymmetricFlowResult,
    AxisymmetricFlowSection, AxisymmetricRotatingWallOptions, AxisymmetricRotatingWallResult,
    AxisymmetricRotatingWallStation, evaluate_axisymmetric_rotating_wall,
    evaluate_axisymmetric_weighted_flow, solve_axisymmetric_weighted_flow,
};
pub use axisymmetric_meridional_flow_reference::{
    AXISYMMETRIC_MERIDIONAL_FLOW_REFERENCE_SCHEMA_V1, AxisymmetricMeridionalFlowError,
    AxisymmetricMeridionalFlowResult, AxisymmetricMeridionalForceFn,
    solve_axisymmetric_meridional_flow, solve_axisymmetric_meridional_flow_with_centrifugal_source,
};
pub use blade_element::{
    AirfoilPolar, AirfoilPolarPoint, BladeElementBemConvergence, BladeElementBemConvergenceLevel,
    BladeElementBemOptions, BladeElementBemResult, BladeElementBemStrip, BladeElementError,
    BladeElementExternalInflow, BladeElementFidelityComparison, BladeElementFidelityError,
    BladeElementFidelityOptions, BladeElementModel, BladeElementOperatingPoint,
    BladeElementRankEvidence, BladeElementRankMetric, BladeElementRankOrder, BladeElementResult,
    BladeElementStation, BladeElementStripLoad, assess_direct_bem_rank_preservation,
    compare_direct_and_bem, compare_direct_and_bem_multipoint,
};
pub use blade3d::{
    Blade3dConfig, BladeMesh, BladeMeshError, BladePanel, LOOP_CLOSURE_FRACTION_RANGE,
    PanelAcousticSummary, Point3, TIP_RADIUS_FRACTION, compare_panel_acoustics,
};
pub use body_fitted_flow_reference::{
    BODY_FITTED_FLOW_REFERENCE_SCHEMA_V1, BodyFittedDuctStation, BodyFittedFlowError,
    BodyFittedFlowNode, BodyFittedFlowOptions, BodyFittedFlowResult, BodyFittedRotatingWallModel,
    BodyFittedTurbulenceModel, solve_body_fitted_duct_flow,
};
pub use centrifugal_modal_reference::{
    CentrifugalModalError, CentrifugalModalOptions, CentrifugalModalResult,
    evaluate_centrifugal_stiffened_modal_reference,
};
pub use compressibility_reference::{
    CompressibilityError, CompressibilityOptions, CompressibilityResult,
    evaluate_compressibility_reference,
};
pub use coupled_swirl_transport_reference::{
    BODY_FITTED_COUPLED_SWIRL_TRANSPORT_SCHEMA_V1, BodyFittedCoupledSwirlError,
    BodyFittedCoupledSwirlOptions, BodyFittedCoupledSwirlResult,
    solve_body_fitted_coupled_swirl_transport,
};
#[cfg(feature = "cuda")]
pub use cuda_adapter::{
    PROPELLER_CUDA_ACOUSTIC_SWEEP_SCHEMA_V1, PROPELLER_CUDA_ADAPTER_SCHEMA_V1,
    PROPELLER_CUDA_VLM_ACTION_SCHEMA_V1, PropellerCudaAcousticHarmonic,
    PropellerCudaAcousticSweepResult, PropellerCudaAdapterError, PropellerCudaPanelResult,
    PropellerCudaPanelSession, PropellerCudaVortexLatticeActionResult,
    PropellerCudaVortexLatticeSession, evaluate_cuda_panel_bem,
};
pub use duct_shroud_reference::{
    DuctShroudError, DuctShroudFlowResult, DuctShroudFlowStation, DuctShroudOptions,
    DuctShroudStation, evaluate_duct_shroud_reference,
};
pub use dynamic_stall_reference::{
    DynamicStallError, DynamicStallInput, DynamicStallOptions, DynamicStallResult,
    DynamicStallSample, evaluate_dynamic_stall_reference,
};
pub use execution_evidence::{
    EXECUTION_EVIDENCE_SCHEMA_V1, PropellerArtifactEvidence, PropellerBackend,
    PropellerCapabilityCell, PropellerCapabilityMatrix, PropellerCapabilityMatrixError,
    PropellerCapabilityRequest, PropellerCapabilityResult, PropellerExecutionBatch,
    PropellerExecutionDirection, PropellerExecutionProvenance, PropellerHostMetadata,
    PropellerPrecision, PropellerUnsupportedCapability, PropellerWorkload, sha256_hex,
};
pub use farassat_fwh_reference::{
    FarassatFwhElementResult, FarassatFwhError, FarassatFwhOptions, FarassatFwhResult,
    evaluate_farassat_1a_reference,
};
pub use fatigue_reference::{
    FatigueCycle, FatigueCycleResult, FatigueError, FatigueOptions, FatigueResult,
    evaluate_s_n_fatigue_reference,
};
pub use fidelity_promotion_reference::{
    FidelityPromotionError, FidelityPromotionOptions, FidelityPromotionPoint,
    FidelityPromotionResult, FidelityPromotionSample, evaluate_fidelity_promotion,
};
pub use free_wake_reference::{
    SteadyFreeWakeError, SteadyFreeWakeFilament, SteadyFreeWakeNode, SteadyFreeWakeOptions,
    SteadyFreeWakeResult, evaluate_steady_free_wake,
};
pub use fwh_reference::{
    FwhComplexPressure, FwhError, FwhOptions, FwhResult, FwhSurfaceElement, evaluate_fwh_reference,
};
pub use harmonic_signal_reference::{
    HarmonicSignalComponent, HarmonicSignalComponentResult, HarmonicSignalError,
    HarmonicSignalOptions, HarmonicSignalResult, evaluate_harmonic_signal_reference,
};
pub use installation_reference::{
    ImageReflectionCoefficient, InstallationDirectivityResult, InstallationReferenceError,
    evaluate_ground_image_directivity,
};
pub use load_coupled_free_wake::{
    LoadCoupledFreeWakeError, LoadCoupledFreeWakeOptions, LoadCoupledFreeWakeResult,
    evaluate_load_coupled_steady_free_wake,
};
pub use mesh_convergence::{
    PropellerMeshConvergence, PropellerMeshConvergenceError, PropellerMeshConvergenceLevel,
    evaluate_mesh_convergence,
};
pub use motor_matching::{
    MotorMatchingError, MotorPropellerMatchCondition, MotorPropellerMatchOptions,
    MotorPropellerMatchResult, MotorTorqueSpeedCurve, MotorTorqueSpeedPoint,
    evaluate_motor_propeller_match,
};
pub use multi_rotor_bem_reference::{
    MultiRotorBemError, MultiRotorBemOptions, MultiRotorBemResult, MultiRotorBemRotor,
    MultiRotorBemRotorResult, evaluate_multi_rotor_bem,
};
pub use multi_rotor_reference::{
    MultiRotorReferenceError, MultiRotorReferenceOptions, MultiRotorReferenceResult,
    MultiRotorReferenceRotor, MultiRotorReferenceRotorResult, evaluate_multi_rotor_reference,
};
pub use nonlinear_aeroelastic_reference::{
    NonlinearAeroelasticModalError, NonlinearAeroelasticModalOptions,
    NonlinearAeroelasticModalResult, evaluate_nonlinear_aeroelastic_modal,
};
pub use operating_envelope::{
    AIR_GAMMA, AIR_GAS_CONSTANT_J_KG_K, PropellerMultipointOperatingPoint,
    PropellerOperatingCondition, PropellerOperatingEnvelope, PropellerOperatingEnvelopeError,
    PropellerOperatingPoint,
};
pub use optimization3d::{
    BladeShapeGradient, BladeShapeTaylorReport, BladeShapeVariables, OptimizedBlade,
    Propeller3dOptConfig, Propeller3dOptError, PropellerActuatorDiskReference,
    PropellerActuatorDiskReferenceError, PropellerLoadPowerSummary, baseline_blade_shape,
    estimate_shape_gradient, evaluate_blade_shape, optimize_blade_shape, pareto_frontier,
    taylor_test_shape_derivative, try_optimize_blade_shape, validate_shape_trust_region,
};
pub use panel_load_reference::{
    PanelLoadProjection, PanelLoadProjectionError, PanelLoadProjectionOptions,
    PanelLoadProjectionResult, project_bem_loads_to_panel_mesh,
};
pub use panel_refinement_reference::{
    PANEL_REFINEMENT_REFERENCE_SCHEMA_V1, PanelRefinementError, PanelRefinementObservation,
    PanelRefinementOptions, PanelRefinementPoint, PanelRefinementResult, evaluate_panel_refinement,
};
pub use provenance::{PropellerModelFidelity, PropellerResultProvenance};
pub use radiation_reference::{
    RotatingDipoleComplexPressure, RotatingDipoleReference, RotatingDipoleReferenceError,
};
pub use robust_operating_reference::{
    PropellerUncertaintyScenario, RobustOperatingError, RobustOperatingOptions,
    RobustOperatingResult, RobustOperatingScenarioResult, evaluate_robust_operating_envelope,
};
pub use source_model::{
    COMBINED_REDUCTION_PRIOR_DB, EdgeTreatment, EmpiricalReductionPriorDb, EvaluationMode,
    HarmonicNoise, LE_SERRATION_REDUCTION_PRIOR_DB, PropellerDesign, PropellerPlanform,
    PropellerSpectrum, PropellerSpectrumError, PropellerSpectrumProvenance,
    SERRATION_AMPLITUDE_CHORD_FRACTION_RANGE, SERRATION_SPAN_FRACTION_RANGE,
    SERRATION_WAVELENGTH_CHORD_FRACTION_RANGE, SerrationGeometry, SpectrumDirectivityConvention,
    SpectrumPhaseConvention, SpectrumPressureConvention, TE_SERRATION_FIXED_RPM_PRIOR_DB,
    TE_SERRATION_MATCHED_THRUST_PRIOR_DB, TOROIDAL_REDUCTION_PRIOR_DB,
};
pub use structural_reference::{
    BladeStructuralError, BladeStructuralOptions, BladeStructuralResult, BladeStructuralStation,
    evaluate_blade_structural_reference,
};
pub use swirl_transport_reference::{
    BODY_FITTED_SWIRL_TRANSPORT_SCHEMA_V1, BodyFittedSwirlError, BodyFittedSwirlNode,
    BodyFittedSwirlOptions, BodyFittedSwirlResult, solve_body_fitted_swirl_transport,
};
pub use viscous_core_reference::{
    ViscousCoreError, ViscousCoreOptions, ViscousCoreResult, ViscousCoreStep,
    evaluate_viscous_core_evolution,
};
pub use vortex_lattice_reference::{
    VortexLatticeError, VortexLatticeLinearSystem, VortexLatticeOptions, VortexLatticePanelResult,
    VortexLatticeResult, assemble_vortex_lattice_system, evaluate_vortex_lattice_reference,
};
pub use wake_convergence::{
    SteadyFreeWakeConvergence, SteadyFreeWakeConvergenceError, SteadyFreeWakeConvergenceLevel,
    SteadyFreeWakeConvergenceOptions, SteadyFreeWakeDiscretization,
    evaluate_steady_free_wake_convergence,
};
pub use wake_reference::{
    PrescribedHelicalWakeError, PrescribedHelicalWakeOptions, PrescribedHelicalWakeResult,
    PrescribedHelicalWakeStation, evaluate_prescribed_helical_wake,
};
