//! Physical-dimension admission for the CPU BEM backend.
//!
//! The current CPU kernels integrate three-dimensional surface elements
//! (`Tri3`/`Quad4`).  A planar 2-D boundary or an axisymmetric meridional
//! boundary must not be silently routed through those kernels: its Green
//! function and measure are different. The reduced axisymmetric ring-source
//! path lives in [`super::axisymmetric`]; this boundary protects only the
//! surface-mesh [`BemProblem`] dispatcher.

use thiserror::Error;

use crate::core::bem_solver::{BemError, BemProblem};

/// Physical reduction requested from the CPU BEM backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CpuBemDimension {
    /// A genuine two-dimensional line-boundary formulation.
    TwoDimensional,
    /// An axisymmetric meridional formulation with azimuthal weighting.
    Axisymmetric,
    /// The supported three-dimensional surface formulation.
    ThreeDimensional,
}

impl std::fmt::Display for CpuBemDimension {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::TwoDimensional => "2-D",
            Self::Axisymmetric => "axisymmetric",
            Self::ThreeDimensional => "3-D",
        })
    }
}

/// Failure to admit a CPU BEM problem at a requested physical dimension.
#[derive(Debug, Error)]
pub enum CpuBemDimensionError {
    /// The requested reduction is incompatible with the surface-mesh API.
    #[error(
        "CPU BEM does not support {requested} admission: only {supported} surface kernels are available"
    )]
    Unsupported {
        /// Requested physical reduction.
        requested: CpuBemDimension,
        /// Dimension implemented by the current kernels.
        supported: CpuBemDimension,
    },
    /// The supported 3-D problem failed the normal checked BEM boundary.
    #[error("invalid 3-D CPU BEM problem: {0}")]
    InvalidProblem(#[source] BemError),
}

/// Admit a problem before a CPU BEM assembly is dispatched.
///
/// This function is intentionally a hard boundary: reduced requests return a
/// typed error instead of falling back to the 3-D surface kernels. A 3-D
/// request is checked with the same validation used by the high-level solver.
/// Axisymmetric callers must construct [`super::AxisymmetricRigidSphere`].
pub fn admit_cpu_bem_problem(
    problem: &BemProblem,
    requested: CpuBemDimension,
) -> Result<(), CpuBemDimensionError> {
    if requested != CpuBemDimension::ThreeDimensional {
        return Err(CpuBemDimensionError::Unsupported {
            requested,
            supported: CpuBemDimension::ThreeDimensional,
        });
    }

    problem
        .validate()
        .map_err(CpuBemDimensionError::InvalidProblem)
}
