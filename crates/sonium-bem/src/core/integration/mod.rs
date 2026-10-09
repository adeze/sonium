//! Numerical integration for BEM
//!
//! Provides Gauss-Legendre quadrature for regular and singular element integration.
//!
//! ## Module Organization
//!
//! - [`gauss`] - Gauss-Legendre quadrature rules for 1D, triangles, quads, and spheres
//! - [`singular`] - Singular integration for self-element (when source = field element)
//! - [`regular`] - Regular integration for non-self elements with adaptive subdivision
//! - [`duffy`] - Duffy transform and collocation singular integration helpers

pub mod duffy;
pub mod gauss;
pub mod lebedev;
pub mod regular;
pub mod singular;

pub use lebedev::{
    LebedevError, lebedev_quadrature, point_count, smallest_order_for_degree, supported_orders,
    try_lebedev_quadrature,
};

pub use duffy::{
    GalerkinIntegrationResult, GalerkinQuad4WeakResult, duffy_self_triangle,
    duffy_self_triangle_constant,
    galerkin_quad4_regular_weak_form_wavenumber_derivative_with_basis,
    galerkin_quad4_regular_weak_form_with_basis, galerkin_single_layer_regular_quad4_dispatch,
    galerkin_single_layer_regular_quad4_with_basis,
    galerkin_single_layer_regular_quad4_with_basis_scalar, galerkin_single_layer_regular_triangle,
    galerkin_single_layer_regular_triangle_with_basis,
    galerkin_single_layer_self_quad4_wavenumber_derivative_with_basis,
    galerkin_single_layer_self_quad4_with_basis, galerkin_single_layer_self_triangle,
    galerkin_single_layer_self_triangle_with_basis,
    galerkin_single_layer_touching_quad4_wavenumber_derivative_with_basis,
    galerkin_single_layer_touching_quad4_with_basis, sauter_schwab_common_edge,
    sauter_schwab_common_vertex, sauter_schwab_galerkin_common_edge,
    sauter_schwab_galerkin_common_vertex, singular_integration_dispatch,
};
pub use gauss::*;
pub use regular::{
    HIGH_ACCURACY_THRESHOLD, QUASI_SINGULAR_THRESHOLD, integrate_g_only, integrate_h_only,
    optimal_quadrature_order, quasi_singular_integration, regular_integration,
    regular_integration_checked, regular_integration_error_estimate,
    regular_integration_fixed_order, regular_integration_fixed_order_source_point_derivative,
    regular_integration_fixed_order_wavenumber_derivative, regular_integration_linear,
    regular_integration_linear_with_normals, regular_integration_wavenumber_derivative,
    regular_integration_with_normals,
};
pub use singular::{
    MAX_SUBELEMENTS, QuadratureParams, Subelement, generate_subelements, singular_integration,
    singular_integration_checked, singular_integration_curved, singular_integration_error_estimate,
    singular_integration_linear, singular_integration_linear_curved,
    singular_integration_linear_with_params, singular_integration_wavenumber_derivative,
    singular_integration_wavenumber_derivative_with_params, singular_integration_with_params,
};

pub(crate) use regular::{
    regular_integration_linear_fixed_order_wavenumber_derivative,
    regular_integration_linear_fixed_order_wavenumber_derivative_with_normals,
    regular_integration_linear_wavenumber_derivative_with_normals,
};
pub(crate) use singular::{
    singular_integration_linear_curved_wavenumber_derivative,
    singular_integration_linear_wavenumber_derivative,
};
