//! # BEM: Boundary Element Method Library
//!
//! High-performance, memory-efficient BEM solver for acoustic scattering problems.
//!
//! ## Features
//!
//! - Parallel execution with Rayon (memory-efficient, no async overhead)
//! - Comprehensive analytical validation (1D, 2D, 3D)
//! - JSON output for visualization
//!

#![warn(missing_docs)]
#![warn(clippy::all)]
#![allow(clippy::too_many_arguments)] // Scientific code often has many parameters

// pub mod analytical; // Removed - use sonium_wave::analytical instead
pub mod core;
pub mod materials;
pub mod numcalc;
pub mod room_acoustics;
pub mod testing;

// Re-exports
pub use core::admission::{
    BemAdmissionError, BemResource, BemResourceBudget, BemResourceEstimate, BemResourceRequest,
};
pub use core::assembly::{
    AcaVariant, BoundaryOperatorAssemblyError, BoundaryOperators, CompressedTransmissionSolution,
    CompressedTransmissionTbemSystem, GalerkinAssemblyError, GalerkinDoubleLayerOperators,
    GalerkinHypersingularOperators, GalerkinQuad4TransmissionError,
    GalerkinQuadratureAdmissionError, GalerkinQuadraturePolicy, GalerkinTransmissionAssemblyError,
    GalerkinTransmissionCompressionError, GalerkinTransmissionFieldError,
    GalerkinTransmissionMaterialDerivativeError, GalerkinTransmissionOperators,
    GalerkinTransmissionProjectionError, GalerkinTransmissionRhsError,
    GalerkinTransmissionSolveError, HMatrixVelocityOperator, LowRankPrecision,
    MatrixDerivativeError, MergeSizeError, MissingSourceDof, MlfmmAssemblyError,
    MlfmmNearFieldError, MlfmmRowActionError, MlfmmWavenumberDerivative, SlfmmAssemblyError,
    SlfmmNearFieldError, SlfmmRowActionError, SlfmmWavenumberDerivative,
    StabilizedTbemVelocityOperator, TbemDerivativeError, TbemQuadraturePairError,
    TbemQuadraturePolicy, TbemVelocityOperatorError, TransmissionAssemblyError,
    TransmissionCompressionError, TransmissionMaterialDerivativeError,
    TransmissionMaterialParameter, TransmissionRhsError, TransmissionSide, TransmissionTbemSystem,
    apply_tbem_stabilized_exact_wavenumber_derivative, assemble_boundary_operators,
    assemble_boundary_operators_with_basis, assemble_boundary_operators_with_basis_shared_dofs,
    assemble_galerkin_double_layer_with_basis, assemble_galerkin_double_layer_with_global_dofs,
    assemble_galerkin_hypersingular_with_global_dofs,
    assemble_galerkin_muller_transmission_material_derivative_with_global_dofs,
    assemble_galerkin_muller_transmission_quad4_pair,
    assemble_galerkin_muller_transmission_quad4_pair_material_derivative,
    assemble_galerkin_muller_transmission_quad4_pair_wavenumber_derivative,
    assemble_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs,
    assemble_galerkin_muller_transmission_with_global_dofs,
    assemble_galerkin_muller_transmission_with_global_dofs_checked, assemble_galerkin_single_layer,
    assemble_galerkin_single_layer_with_basis, assemble_galerkin_single_layer_with_global_dofs,
    build_directional_mlfmm_system_stabilized_checked,
    build_directional_mlfmm_system_stabilized_with_basis_checked,
    build_directional_mlfmm_wavenumber_derivative, build_hmatrix_density_derivative,
    build_hmatrix_stabilized_density_derivative, build_hmatrix_stabilized_wavenumber_derivative,
    build_hmatrix_wavenumber_derivative, build_hmatrix_with_variant_precision_and_beta,
    build_mlfmm_system_stabilized_checked, build_mlfmm_system_stabilized_with_basis_checked,
    build_mlfmm_wavenumber_derivative, build_slfmm_system_stabilized_checked,
    build_slfmm_system_stabilized_with_basis_checked, build_slfmm_wavenumber_derivative,
    build_slfmm_wavenumber_derivative_with_basis_checked, build_tbem_system_checked,
    build_tbem_system_stabilized_density_derivative,
    build_tbem_system_stabilized_impedance_density_derivative,
    build_tbem_system_stabilized_impedance_speed_derivative,
    build_tbem_system_stabilized_speed_derivative,
    build_tbem_system_stabilized_wavenumber_derivative, build_transmission_tbem_system,
    build_transmission_tbem_system_from_xem, build_transmission_tbem_system_with_basis,
    build_transmission_tbem_system_with_continuous_quadratic,
    build_transmission_tbem_system_with_continuous_quadratic_from_xem,
    compress_galerkin_muller_transmission_material_derivative_with_global_dofs_checked,
    compress_galerkin_muller_transmission_quad4_material_derivative_with_global_dofs,
    compress_galerkin_muller_transmission_quad4_material_derivative_with_global_dofs_checked,
    compress_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs,
    compress_galerkin_muller_transmission_quad4_with_global_dofs,
    compress_galerkin_muller_transmission_quad4_with_global_dofs_checked,
    compress_galerkin_muller_transmission_with_global_dofs,
    compress_galerkin_muller_transmission_with_global_dofs_checked,
    compute_matrix_entry_wavenumber_derivative, compute_regular_matrix_entry_wavenumber_derivative,
    tbem_wavenumber_objective_gradient, try_build_tbem_system_with_beta,
    validate_mlfmm_constant_panel_layout, validate_slfmm_constant_panel_layout,
};
#[cfg(feature = "cuda")]
pub use core::assembly::{
    CudaTbemAssemblyError, CudaTbemAssemblyWorkspace, build_quad4_cuda_regular_quadrature,
    build_tbem_system_with_beta_cuda, build_tbem_system_with_beta_cuda_checked,
    build_tri3_cuda_singular_quadrature,
};
pub use core::axisymmetric::{
    AxisymmetricBemError, AxisymmetricPenetrableProfile, AxisymmetricPenetrableProfileSolution,
    AxisymmetricPenetrableSphere, AxisymmetricPenetrableSphereSolution,
    AxisymmetricProfileTransmissionAdjointSolution, AxisymmetricRigidSphere,
    AxisymmetricRigidSphereSolution, AxisymmetricTransmissionAdjointSolution,
};
pub use core::bem_solver::{
    BemAdjointSolution, BemLoadSolution, BemOperatorCache, BemResultError, BoundaryConditionType,
};
pub use core::contracts::{
    AdjointContractReport, OperatorAccuracyReport, OperatorContractError, PassivityContractReport,
    ReciprocityContractReport, TransmissionPowerBalanceReport, TransmissionReciprocityReport,
    check_adjoint_contract, check_block_rhs_contract, check_passivity_contract,
    check_reciprocity_contract, check_transmission_passivity_contract,
    check_transmission_power_balance, check_transmission_reciprocity_contract,
    check_transmission_reciprocity_operator, constant_panel_power, require_operator_accuracy,
};
pub use core::conventions::{
    ACOUSTIC_CONVENTION, BEM_CONVENTION_CONTRACT_VERSION, BemConvention, BemConventionContract,
    BemConventionError, BemDomain, BemProblemError, BoundaryTraceConvention, BurtonMillerSign,
    NormalVelocityDirection, PressureTrace,
};
pub use core::coupling::{
    ConservativeTraceTransfer, CoupledPreconditionerKind, CoupledResidualReport,
    CoupledTraceOperator, CoupledTracePreconditioner, CoupledTraceSystem, CouplingConvention,
    CouplingError,
};
pub use core::distributed::{
    DistributedBemConfig, DistributedLinearOperator, DistributedLocalPreconditioner,
    DistributedRowPlan, DistributedVectorExchange, InProcessVectorExchange, PartitionStrategy,
    plan_distributed_rows, plan_distributed_rows_with_dof_count, try_block_gmres_distributed,
    try_block_gmres_distributed_preconditioned, try_gmres_distributed,
    try_gmres_distributed_preconditioned,
};
pub use core::integration::{
    regular_integration_fixed_order_source_point_derivative,
    regular_integration_fixed_order_wavenumber_derivative,
    regular_integration_wavenumber_derivative, singular_integration_wavenumber_derivative,
    singular_integration_wavenumber_derivative_with_params,
};
pub use core::mesh::{
    BemCacheInvalidation, BemCacheReuse, BemReuseContext, BemReuseDecision, BemReuseError,
    BemReusePolicy, BemReuseRejection, MeshLineage, MeshLineageError, MeshUpdatePlan,
    assess_bem_cache_reuse, classify_mesh_update, relative_frequency_delta,
};
pub use core::nested_coupling::{
    NestedTraceEdge, NestedTraceOperator, NestedTracePreconditioner, NestedTracePreconditionerKind,
    NestedTraceRegion,
};
#[cfg(feature = "cuda")]
pub use core::postprocess::compute_scattered_field_cuda_checked;
pub use core::postprocess::{
    FieldEvaluationError, FieldPoint, build_constant_field_operator,
    compute_scattered_field_checked, compute_scattered_field_gradient_checked,
    compute_total_field_checked, compute_total_field_gradient_checked,
};
pub use core::reduced_2d::{
    TwoDBemError, TwoDBoundaryPanel, TwoDFluidCylinder, TwoDFluidCylinderSolution, TwoDFourierMode,
    TwoDPlaneWave, TwoDRigidCylinder, TwoDRigidCylinderSolution, TwoDTransmissionAdjointSolution,
};
pub use core::scheduling::{
    BemBatchSchedule, BemBatchTask, BemBatchWorkload, BemMemoryBudget, BemSchedulingError,
    plan_bem_batch,
};
pub use core::types::{BasisOrder, PhysicsParams, PhysicsParamsError};
pub use materials::{
    AmbientMedium, FluidInterface, LayeredPorousSurface, MaterialError, PlaneWaveTransmission,
    PorousLayer, PorousModel,
};
pub use sonium_wave::analytical;
pub use testing::*;

/// Separated Quad4 Galerkin single-layer block assembly.
pub use core::assembly::galerkin::assemble_galerkin_single_layer_quad4_pair;

/// Library version
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Git commit hash (set during build)
pub const GIT_HASH: &str = env!("GIT_HASH");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_version() {
        assert!(!VERSION.is_empty());
    }
}
