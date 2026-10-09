//! Pure Rust BEM (Boundary Element Method) Solver
//!
//! This module provides a complete BEM solver for acoustic scattering problems,
//! supporting Traditional BEM (TBEM), Single-Level Fast Multipole (SLFMM),
//! and Multi-Level Fast Multipole (MLFMM) methods.
//!
//! ## Architecture
//!
//! - `types`: Core data structures (Mesh, Element, PhysicsParams)
//! - `constants`: Physical and integration constants
//! - `greens`: Green's function computations (via math-wave)
//! - `mesh`: Mesh loading, element operations, and mesh generators
//! - `integration`: Numerical quadrature (Gauss-Legendre, singular)
//! - `assembly`: BEM matrix assembly (TBEM, SLFMM, MLFMM)
//! - `solver`: Linear solvers (Direct, CGS, BiCGSTAB)
//! - `incident`: Incident field computation (plane waves, point sources)
//! - `postprocess`: Result computation at evaluation points
//! - `io`: Input/output (NC.inp format, JSON)
//! - `bem_solver`: High-level API for solving BEM problems
//! - `algebra`: Pure Rust linear algebra fallbacks for WASM portability
//! - `parallel`: Portable parallel iteration (works with native, WASM, or sequential)

pub mod admission;
pub mod array;
pub mod assembly;
pub mod axisymmetric;
pub mod bem_solver;
pub mod constants;
pub mod contracts;
pub mod conventions;
pub mod coupling;
pub mod dimension;
pub mod distributed;
pub mod incident;
pub mod integration;
pub mod io;
pub mod mesh;
pub mod nested_coupling;
pub mod parallel;
pub mod postprocess;
pub mod reduced_2d;
pub mod scheduling;
pub mod solver;
pub mod types;

// Re-exports for convenience
pub use admission::{
    BemAdmissionError, BemResource, BemResourceBudget, BemResourceEstimate, BemResourceRequest,
};
pub use axisymmetric::{
    AxisymmetricBemError, AxisymmetricPenetrableProfile, AxisymmetricPenetrableProfileSolution,
    AxisymmetricPenetrableSphere, AxisymmetricPenetrableSphereSolution,
    AxisymmetricProfileTransmissionAdjointSolution, AxisymmetricRigidSphere,
    AxisymmetricRigidSphereSolution, AxisymmetricTransmissionAdjointSolution,
};
pub use bem_solver::{
    AssemblyMethod, BemAdjointSolution, BemError, BemLoadSolution, BemOperatorCache, BemProblem,
    BemResultError, BemSolution, BemSolver, BoundaryConditionType, SolverMethod,
};
pub use constants::PhysicsParams;
pub use conventions::ACOUSTIC_CONVENTION;
pub use coupling::{
    ConservativeTraceTransfer, CoupledPreconditionerKind, CoupledResidualReport,
    CoupledTraceOperator, CoupledTracePreconditioner, CoupledTraceSystem, CouplingConvention,
    CouplingError,
};
pub use dimension::{CpuBemDimension, CpuBemDimensionError, admit_cpu_bem_problem};
pub use distributed::{
    DistributedBemConfig, DistributedLinearOperator, DistributedLocalPreconditioner,
    DistributedRowPlan, DistributedVectorExchange, InProcessVectorExchange, PartitionStrategy,
    plan_distributed_rows, plan_distributed_rows_with_dof_count, try_gmres_distributed,
    try_gmres_distributed_preconditioned,
};
pub use incident::{IncidentField, IncidentFieldError};
pub use nested_coupling::{
    NestedTraceEdge, NestedTraceOperator, NestedTracePreconditioner, NestedTracePreconditionerKind,
    NestedTraceRegion,
};
pub use scheduling::{
    BemBatchSchedule, BemBatchTask, BemBatchWorkload, BemMemoryBudget, BemSchedulingError,
    plan_bem_batch,
};
pub use types::*;
