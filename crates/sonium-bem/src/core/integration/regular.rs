//! Regular (non-singular) element integration
//!
//! Adaptive integration for elements when source and field elements are different.

use num_complex::Complex64;
use sonium_solvers::dense::DenseMatrix;
use std::f64::consts::PI;

use crate::core::integration::gauss::{quad_quadrature, triangle_quadrature};
use crate::core::integration::singular::generate_subelements;
use crate::core::mesh::element::{basis_values, cross_product, normalize, quadrature_geometry};
use crate::core::types::{
    BasisOrder, ElementType, IntegrationEstimate, IntegrationResult, LinearIntegrationResult,
    PhysicsParams, QuadratureConvergenceError, QuadratureOrderError,
};

/// Perform regular integration over a field element
///
/// This implements NC_RegularIntegration from the C++ code.
/// Uses adaptive subelement subdivision based on distance from source point.
///
/// # Arguments
/// * `source_point` - Source point (collocation point)
/// * `source_normal` - Unit normal at source point
/// * `element_coords` - Node coordinates of the field element (num_nodes × 3)
/// * `element_type` - Triangle or quad element
/// * `element_area` - Area of the field element
/// * `physics` - Physics parameters (wave number, etc.)
/// * `bc_values` - Boundary condition values at element nodes (for RHS)
/// * `bc_type` - Boundary condition type (0=velocity, 1=pressure)
/// * `compute_rhs` - Whether to compute RHS contribution
///
/// # Returns
/// IntegrationResult with G, H, H^T, E integrals and RHS contribution
pub fn regular_integration(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    element_area: f64,
    physics: &PhysicsParams,
    bc_values: Option<&[Complex64]>,
    bc_type: i32,
    compute_rhs: bool,
) -> IntegrationResult {
    regular_integration_with_normals(
        source_point,
        source_normal,
        element_coords,
        element_type,
        element_area,
        physics,
        bc_values,
        bc_type,
        compute_rhs,
        None,
    )
}

/// Perform regular integration with optional isoparametric curved normals
///
/// When `node_normals` is `Some`, the field-element unit normal at each quadrature
/// point is replaced by the interpolated (and re-normalized) normal built from the
/// per-node normals. The Jacobian (surface element dS) always comes from the flat
/// cross-product so that the area measure is unchanged.
pub fn regular_integration_with_normals(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    element_area: f64,
    physics: &PhysicsParams,
    bc_values: Option<&[Complex64]>,
    bc_type: i32,
    compute_rhs: bool,
    node_normals: Option<&[[f64; 3]]>,
) -> IntegrationResult {
    regular_integration_with_normals_mode(
        source_point,
        source_normal,
        element_coords,
        element_type,
        element_area,
        physics,
        bc_values,
        bc_type,
        compute_rhs,
        node_normals,
        false,
    )
}

/// Analytic wavenumber derivative of the adaptive regular/curved integral.
/// Geometry and the adaptive subdivision topology are held fixed at the
/// supplied physics point; the kernel phase, H/E factors, and Burton--Miller
/// beta in the optional RHS are differentiated analytically.
pub fn regular_integration_wavenumber_derivative(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    element_area: f64,
    physics: &PhysicsParams,
    bc_values: Option<&[Complex64]>,
    bc_type: i32,
    compute_rhs: bool,
    node_normals: Option<&[[f64; 3]]>,
) -> IntegrationResult {
    regular_integration_with_normals_mode(
        source_point,
        source_normal,
        element_coords,
        element_type,
        element_area,
        physics,
        bc_values,
        bc_type,
        compute_rhs,
        node_normals,
        true,
    )
}

fn regular_integration_with_normals_mode(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    element_area: f64,
    physics: &PhysicsParams,
    bc_values: Option<&[Complex64]>,
    bc_type: i32,
    compute_rhs: bool,
    node_normals: Option<&[[f64; 3]]>,
    wavenumber_derivative: bool,
) -> IntegrationResult {
    let wavruim = physics.harmonic_factor * physics.wave_number;
    let k2 = physics.wave_number * physics.wave_number;

    let mut result = IntegrationResult::default();

    // Generate adaptive subelements
    let subelements =
        generate_subelements(source_point, element_coords, element_type, element_area);

    // Local coordinates of element vertices (standard vertex mapping)
    // Triangle: N0 at (0,0), N1 at (1,0), N2 at (0,1)
    // Quad: bilinear on [-1,1]²
    let (_csi_nodes, _eta_nodes): (Vec<f64>, Vec<f64>) = match element_type {
        ElementType::Tri3 => (vec![0.0, 1.0, 0.0], vec![0.0, 0.0, 1.0]),
        ElementType::Quad4 => (vec![1.0, -1.0, -1.0, 1.0], vec![1.0, 1.0, -1.0, -1.0]),
    };

    // Process each subelement
    for subel in &subelements {
        let xice = subel.xi_center;
        let etce = subel.eta_center;
        let fase = subel.factor;
        let fase2 = fase * fase;
        let iforie = (fase.abs() - 1.0).abs() < 1e-10; // Not subdivided

        // Get quadrature points for this subelement
        let quad_points = get_quadrature_points(element_type, subel.gauss_order);

        for (csi_gau, eta_gau, wei_gau) in quad_points {
            // Transform quadrature point to original element coordinates
            let (xio, eto, weih2) = if iforie {
                (csi_gau, eta_gau, wei_gau)
            } else if let Some(tri_verts) = &subel.tri_vertices {
                // For triangular subelements: use affine transformation with shape functions
                // Reference triangle point (ξ, η) maps to:
                // xio = v0.0 * (1-ξ-η) + v1.0 * ξ + v2.0 * η
                // eto = v0.1 * (1-ξ-η) + v1.1 * ξ + v2.1 * η
                let l0 = 1.0 - csi_gau - eta_gau; // barycentric coord for v0
                let xio = tri_verts[0].0 * l0 + tri_verts[1].0 * csi_gau + tri_verts[2].0 * eta_gau;
                let eto = tri_verts[0].1 * l0 + tri_verts[1].1 * csi_gau + tri_verts[2].1 * eta_gau;

                // Jacobian of the affine map = 2 * area of subtriangle
                // Area = 0.5 * |det([v1-v0, v2-v0])|
                let dx1 = tri_verts[1].0 - tri_verts[0].0;
                let dy1 = tri_verts[1].1 - tri_verts[0].1;
                let dx2 = tri_verts[2].0 - tri_verts[0].0;
                let dy2 = tri_verts[2].1 - tri_verts[0].1;
                let det = (dx1 * dy2 - dx2 * dy1).abs();
                // det = 2 * area of subtriangle in local coords
                // The reference triangle has area 0.5, so Jacobian = det
                let weih2 = wei_gau * det;

                (xio, eto, weih2)
            } else {
                // For quad subelements: use center + scale transformation
                (
                    xice + csi_gau * fase,
                    etce + eta_gau * fase,
                    wei_gau * fase2,
                )
            };

            // Compute shape functions, Jacobian and position at quadrature point
            let params = compute_parameters(element_coords, element_type, xio, eto);

            let (crd_poi, jacobian, el_norm) = if node_normals.is_some() {
                quadrature_geometry(
                    element_coords,
                    &params.shape_fn,
                    &params.shape_ds,
                    &params.shape_dt,
                    node_normals,
                )
            } else {
                (params.position, params.jacobian, params.normal)
            };

            let wga = weih2 * jacobian;

            // Compute distance vector and kernels
            let diff_fsp = [
                crd_poi[0] - source_point[0],
                crd_poi[1] - source_point[1],
                crd_poi[2] - source_point[2],
            ];

            let (unit_r, dis_fsp) = normalize(diff_fsp);

            if dis_fsp < 1e-15 {
                continue;
            }

            // G kernel: exp(ikr)/(4πr)
            let re1 = wavruim * dis_fsp;
            let re2 = wga / (4.0 * PI * dis_fsp);
            let zg = sonium_fastmath::exp_i(re1) * re2;
            let dzg = zg * Complex64::new(0.0, physics.harmonic_factor * dis_fsp);

            // H kernel (double layer): ∂G/∂n_y
            let z1 = Complex64::new(-1.0 / dis_fsp, wavruim);
            let zhh_base = zg * z1;
            let dzhh_base = dzg * z1 + zg * Complex64::new(0.0, physics.harmonic_factor);
            let re1_h = unit_r[0] * el_norm[0] + unit_r[1] * el_norm[1] + unit_r[2] * el_norm[2];
            let zhh = if wavenumber_derivative {
                dzhh_base * re1_h
            } else {
                zhh_base * re1_h
            };

            // H^T kernel (adjoint double layer): ∂G/∂n_x
            let re2_h = -(unit_r[0] * source_normal[0]
                + unit_r[1] * source_normal[1]
                + unit_r[2] * source_normal[2]);
            let zht = if wavenumber_derivative {
                dzhh_base * re2_h
            } else {
                zhh_base * re2_h
            };

            // E kernel (hypersingular): ∂²G/(∂n_x ∂n_y)
            let rq = re1_h * re2_h;
            let nx_dot_ny = source_normal[0] * el_norm[0]
                + source_normal[1] * el_norm[1]
                + source_normal[2] * el_norm[2];
            let dq = dis_fsp * dis_fsp;

            let ze_factor = Complex64::new(
                (3.0 / dq - k2) * rq + nx_dot_ny / dq,
                -wavruim / dis_fsp * (3.0 * rq + nx_dot_ny),
            );
            let dze_factor = Complex64::new(
                -2.0 * physics.wave_number * rq,
                -physics.harmonic_factor / dis_fsp * (3.0 * rq + nx_dot_ny),
            );
            let ze = if wavenumber_derivative {
                dzg * ze_factor + zg * dze_factor
            } else {
                zg * ze_factor
            };

            // Accumulate integrals
            result.g_integral += if wavenumber_derivative { dzg } else { zg };
            result.dg_dn_integral += zhh;
            result.dg_dnx_integral += zht;
            result.d2g_dnxdny_integral += ze;

            // RHS contribution
            if compute_rhs && let Some(bc) = bc_values {
                // Interpolate BC at quadrature point
                let mut zbgao = Complex64::new(0.0, 0.0);
                for (i, &n) in params.shape_fn.iter().enumerate() {
                    if i < bc.len() {
                        zbgao += bc[i] * n;
                    }
                }

                let gamma = physics.gamma();
                let tau = physics.tau;
                let beta = physics.burton_miller_beta();
                let beta_derivative = physics.burton_miller_beta_wavenumber_derivative();

                if bc_type == 0 {
                    // Velocity BC: known v contributes through G and K^T kernels
                    let contribution = if wavenumber_derivative {
                        dzg * gamma * tau + zht * beta + (zhh_base * re2_h) * beta_derivative
                    } else {
                        zg * gamma * tau + zht * beta
                    };
                    result.rhs_contribution += contribution * zbgao;
                } else if bc_type == 1 {
                    // Pressure BC (Dirichlet): known p contributes through K and E kernels
                    // Moves (K + β*E)·p_known from LHS to RHS (positive sign)
                    let contribution = if wavenumber_derivative {
                        zhh * gamma * tau + ze * beta + (zhh_base * re1_h) * beta_derivative
                    } else {
                        zhh * gamma * tau + ze * beta
                    };
                    result.rhs_contribution += contribution * zbgao;
                }
            }
        }
    }

    result
}

/// Perform regular integration for linear elements, returning per-basis-function results
///
/// Each IntegrationResult in the output corresponds to one basis function of the field element.
/// The kernel integrals are weighted by the shape function value at each quadrature point:
///   result[j].g_integral = ∫ N_j(y_q) * G(x, y_q) dS_y
///
/// # Arguments
/// * `source_point` - Collocation point (one of the Gauss points inside the source element)
/// * `source_normal` - Unit normal at the source collocation point
/// * `element_coords` - Node coordinates of the field element
/// * `element_type` - Triangle or quad element
/// * `element_area` - Area of the field element
/// * `physics` - Physics parameters
/// * `num_basis` - Number of basis functions (3 for Tri3, 4 for Quad4)
pub fn regular_integration_linear(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    element_area: f64,
    physics: &PhysicsParams,
    num_basis: usize,
) -> LinearIntegrationResult {
    regular_integration_linear_with_normals(
        source_point,
        source_normal,
        element_coords,
        element_type,
        element_area,
        physics,
        num_basis,
        None,
    )
}

/// Perform regular integration for linear elements with optional curved normals
pub fn regular_integration_linear_with_normals(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    element_area: f64,
    physics: &PhysicsParams,
    num_basis: usize,
    node_normals: Option<&[[f64; 3]]>,
) -> LinearIntegrationResult {
    let wavruim = physics.harmonic_factor * physics.wave_number;
    let k2 = physics.wave_number * physics.wave_number;

    let mut basis_results: Vec<IntegrationResult> = (0..num_basis)
        .map(|_| IntegrationResult::default())
        .collect();

    // Generate adaptive subelements
    let subelements =
        generate_subelements(source_point, element_coords, element_type, element_area);

    // Process each subelement
    for subel in &subelements {
        let xice = subel.xi_center;
        let etce = subel.eta_center;
        let fase = subel.factor;
        let fase2 = fase * fase;
        let iforie = (fase.abs() - 1.0).abs() < 1e-10;

        let quad_points = get_quadrature_points(element_type, subel.gauss_order);

        for (csi_gau, eta_gau, wei_gau) in quad_points {
            let (xio, eto, weih2) = if iforie {
                (csi_gau, eta_gau, wei_gau)
            } else if let Some(tri_verts) = &subel.tri_vertices {
                let l0 = 1.0 - csi_gau - eta_gau;
                let xio = tri_verts[0].0 * l0 + tri_verts[1].0 * csi_gau + tri_verts[2].0 * eta_gau;
                let eto = tri_verts[0].1 * l0 + tri_verts[1].1 * csi_gau + tri_verts[2].1 * eta_gau;

                let dx1 = tri_verts[1].0 - tri_verts[0].0;
                let dy1 = tri_verts[1].1 - tri_verts[0].1;
                let dx2 = tri_verts[2].0 - tri_verts[0].0;
                let dy2 = tri_verts[2].1 - tri_verts[0].1;
                let det = (dx1 * dy2 - dx2 * dy1).abs();
                let weih2 = wei_gau * det;

                (xio, eto, weih2)
            } else {
                (
                    xice + csi_gau * fase,
                    etce + eta_gau * fase,
                    wei_gau * fase2,
                )
            };

            // Compute shape functions, Jacobian and position at quadrature point
            let params = compute_parameters(element_coords, element_type, xio, eto);

            let (crd_poi, jacobian, el_norm) = if node_normals.is_some() {
                quadrature_geometry(
                    element_coords,
                    &params.shape_fn,
                    &params.shape_ds,
                    &params.shape_dt,
                    node_normals,
                )
            } else {
                (params.position, params.jacobian, params.normal)
            };

            let wga = weih2 * jacobian;

            // Compute distance vector and kernels
            let diff_fsp = [
                crd_poi[0] - source_point[0],
                crd_poi[1] - source_point[1],
                crd_poi[2] - source_point[2],
            ];

            let (unit_r, dis_fsp) = normalize(diff_fsp);

            if dis_fsp < 1e-15 {
                continue;
            }

            // G kernel: exp(ikr)/(4πr)
            let re1 = wavruim * dis_fsp;
            let re2 = wga / (4.0 * PI * dis_fsp);
            let zg = sonium_fastmath::exp_i(re1) * re2;

            // H kernel (double layer): ∂G/∂n_y
            let z1 = Complex64::new(-1.0 / dis_fsp, wavruim);
            let zhh_base = zg * z1;
            let re1_h = unit_r[0] * el_norm[0] + unit_r[1] * el_norm[1] + unit_r[2] * el_norm[2];
            let zhh = zhh_base * re1_h;

            // H^T kernel (adjoint double layer): ∂G/∂n_x
            let re2_h = -(unit_r[0] * source_normal[0]
                + unit_r[1] * source_normal[1]
                + unit_r[2] * source_normal[2]);
            let zht = zhh_base * re2_h;

            // E kernel (hypersingular): ∂²G/(∂n_x ∂n_y)
            let rq = re1_h * re2_h;
            let nx_dot_ny = source_normal[0] * el_norm[0]
                + source_normal[1] * el_norm[1]
                + source_normal[2] * el_norm[2];
            let dq = dis_fsp * dis_fsp;

            let ze_factor = Complex64::new(
                (3.0 / dq - k2) * rq + nx_dot_ny / dq,
                -wavruim / dis_fsp * (3.0 * rq + nx_dot_ny),
            );
            let ze = zg * ze_factor;

            let basis_values = panel_basis_values(element_type, xio, eto, num_basis);

            // Weight each kernel by the collocation-basis value and accumulate
            for b in 0..num_basis {
                let n_b = basis_values[b];
                basis_results[b].g_integral += zg * n_b;
                basis_results[b].dg_dn_integral += zhh * n_b;
                basis_results[b].dg_dnx_integral += zht * n_b;
                basis_results[b].d2g_dnxdny_integral += ze * n_b;
            }
        }
    }

    LinearIntegrationResult { basis_results }
}

/// Adaptive regular linear-element integration differentiated with respect to
/// the acoustic wavenumber.  The subdivision and quadrature schedule is
/// exactly the one used by [`regular_integration_linear_with_normals`], so a
/// material derivative does not introduce a parameter-dependent quadrature
/// mismatch.
pub(crate) fn regular_integration_linear_wavenumber_derivative_with_normals(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    element_area: f64,
    physics: &PhysicsParams,
    num_basis: usize,
    node_normals: Option<&[[f64; 3]]>,
) -> LinearIntegrationResult {
    let phase_sign = physics.harmonic_factor;
    let k = physics.wave_number;
    let dk2_dk = 2.0 * k;
    let mut basis_results = vec![IntegrationResult::default(); num_basis];
    let subelements =
        generate_subelements(source_point, element_coords, element_type, element_area);

    for subel in &subelements {
        let factor = subel.factor;
        let factor_squared = factor * factor;
        let is_full_panel = (factor.abs() - 1.0).abs() < 1.0e-10;
        for (xi_g, eta_g, weight_g) in get_quadrature_points(element_type, subel.gauss_order) {
            let (xi, eta, mapped_weight) = if is_full_panel {
                (xi_g, eta_g, weight_g)
            } else if let Some(tri_vertices) = &subel.tri_vertices {
                let l0 = 1.0 - xi_g - eta_g;
                let xi =
                    tri_vertices[0].0 * l0 + tri_vertices[1].0 * xi_g + tri_vertices[2].0 * eta_g;
                let eta =
                    tri_vertices[0].1 * l0 + tri_vertices[1].1 * xi_g + tri_vertices[2].1 * eta_g;
                let dx1 = tri_vertices[1].0 - tri_vertices[0].0;
                let dy1 = tri_vertices[1].1 - tri_vertices[0].1;
                let dx2 = tri_vertices[2].0 - tri_vertices[0].0;
                let dy2 = tri_vertices[2].1 - tri_vertices[0].1;
                let determinant = (dx1 * dy2 - dx2 * dy1).abs();
                (xi, eta, weight_g * determinant)
            } else {
                (
                    subel.xi_center + factor * xi_g,
                    subel.eta_center + factor * eta_g,
                    weight_g * factor_squared,
                )
            };

            let params = compute_parameters(element_coords, element_type, xi, eta);
            let (position, jacobian, element_normal) = if node_normals.is_some() {
                quadrature_geometry(
                    element_coords,
                    &params.shape_fn,
                    &params.shape_ds,
                    &params.shape_dt,
                    node_normals,
                )
            } else {
                (params.position, params.jacobian, params.normal)
            };
            let weighted_jacobian = mapped_weight * jacobian;
            let diff = [
                position[0] - source_point[0],
                position[1] - source_point[1],
                position[2] - source_point[2],
            ];
            let (unit_r, distance) = normalize(diff);
            if distance < 1.0e-15 {
                continue;
            }

            let phase = phase_sign * k * distance;
            let g = sonium_fastmath::exp_i(phase) * (weighted_jacobian / (4.0 * PI * distance));
            let dg_dk = g * Complex64::new(0.0, phase_sign * distance);

            let r_dot_ny = unit_r[0] * element_normal[0]
                + unit_r[1] * element_normal[1]
                + unit_r[2] * element_normal[2];
            let r_dot_nx = -(unit_r[0] * source_normal[0]
                + unit_r[1] * source_normal[1]
                + unit_r[2] * source_normal[2]);
            let kernel_derivative_factor = Complex64::new(0.0, phase_sign);
            let h_factor = Complex64::new(-1.0 / distance, phase_sign * k);
            let dh_base_dk = dg_dk * h_factor + g * kernel_derivative_factor;
            let dh_dk = dh_base_dk * r_dot_ny;
            let dht_dk = dh_base_dk * r_dot_nx;

            let normal_product = r_dot_ny * r_dot_nx;
            let source_field_normal = source_normal[0] * element_normal[0]
                + source_normal[1] * element_normal[1]
                + source_normal[2] * element_normal[2];
            let distance_squared = distance * distance;
            let e_factor = Complex64::new(
                (3.0 / distance_squared - k * k) * normal_product
                    + source_field_normal / distance_squared,
                -phase_sign * k / distance * (3.0 * normal_product + source_field_normal),
            );
            let de_factor = Complex64::new(
                -dk2_dk * normal_product,
                -phase_sign / distance * (3.0 * normal_product + source_field_normal),
            );
            let de_dk = dg_dk * e_factor + g * de_factor;

            for (result, basis_value) in
                basis_results
                    .iter_mut()
                    .zip(panel_basis_values(element_type, xi, eta, num_basis))
            {
                result.g_integral += dg_dk * basis_value;
                result.dg_dn_integral += dh_dk * basis_value;
                result.dg_dnx_integral += dht_dk * basis_value;
                result.d2g_dnxdny_integral += de_dk * basis_value;
            }
        }
    }
    LinearIntegrationResult { basis_results }
}

fn panel_basis_values(element_type: ElementType, s: f64, t: f64, num_basis: usize) -> Vec<f64> {
    let basis = match (element_type, num_basis) {
        (ElementType::Tri3, 3) | (ElementType::Quad4, 4) => BasisOrder::Linear,
        (ElementType::Tri3, 6) | (ElementType::Quad4, 9) => BasisOrder::Quadratic,
        _ => panic!("unsupported discontinuous panel basis layout"),
    };
    basis_values(element_type, basis, s, t)
}

/// Get quadrature points for element type and order
fn get_quadrature_points(element_type: ElementType, order: usize) -> Vec<(f64, f64, f64)> {
    match element_type {
        ElementType::Tri3 => triangle_quadrature(order),
        ElementType::Quad4 => quad_quadrature(order),
    }
}

struct ElementParameters {
    shape_fn: Vec<f64>,
    shape_ds: Vec<f64>,
    shape_dt: Vec<f64>,
    jacobian: f64,
    normal: [f64; 3],
    position: [f64; 3],
}

/// Compute shape functions, Jacobian, normal and position at local coordinates
fn compute_parameters(
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    s: f64,
    t: f64,
) -> ElementParameters {
    let num_nodes = element_type.num_nodes();

    let (shape_fn, shape_ds, shape_dt) = match element_type {
        ElementType::Tri3 => {
            // Triangle area coordinates: standard vertex mapping
            // N0 = 1-s-t (vertex at (0,0))
            // N1 = s (vertex at (1,0))
            // N2 = t (vertex at (0,1))
            let shape_fn = vec![1.0 - s - t, s, t];
            let shape_ds = vec![-1.0, 1.0, 0.0];
            let shape_dt = vec![-1.0, 0.0, 1.0];
            (shape_fn, shape_ds, shape_dt)
        }
        ElementType::Quad4 => {
            // Bilinear quad on [-1,1]²
            let s1 = 0.25 * (s + 1.0);
            let s2 = 0.25 * (s - 1.0);
            let t1 = t + 1.0;
            let t2 = t - 1.0;

            let shape_fn = vec![s1 * t1, -s2 * t1, s2 * t2, -s1 * t2];
            let shape_ds = vec![
                0.25 * (t + 1.0),
                -0.25 * (t + 1.0),
                0.25 * (t - 1.0),
                -0.25 * (t - 1.0),
            ];
            let shape_dt = vec![
                0.25 * (s + 1.0),
                0.25 * (1.0 - s),
                0.25 * (s - 1.0),
                -0.25 * (s + 1.0),
            ];
            (shape_fn, shape_ds, shape_dt)
        }
    };

    // Compute global position and tangent vectors
    let mut crd_poi = [0.0_f64; 3];
    let mut dx_ds = [0.0_f64; 3];
    let mut dx_dt = [0.0_f64; 3];

    for i in 0..num_nodes {
        for j in 0..3 {
            crd_poi[j] += shape_fn[i] * element_coords[[i, j]];
            dx_ds[j] += shape_ds[i] * element_coords[[i, j]];
            dx_dt[j] += shape_dt[i] * element_coords[[i, j]];
        }
    }

    // Normal = dx_ds × dx_dt
    let n_arr = cross_product(dx_ds, dx_dt);
    let jacobian = (n_arr[0] * n_arr[0] + n_arr[1] * n_arr[1] + n_arr[2] * n_arr[2]).sqrt();

    let el_norm = if jacobian > 1e-15 {
        [
            n_arr[0] / jacobian,
            n_arr[1] / jacobian,
            n_arr[2] / jacobian,
        ]
    } else {
        [0.0; 3]
    };

    ElementParameters {
        shape_fn,
        shape_ds,
        shape_dt,
        jacobian,
        normal: el_norm,
        position: crd_poi,
    }
}

/// Threshold for quasi-singular integration (distance/element_size ratio)
/// NumCalc uses approximately 3.0
pub const QUASI_SINGULAR_THRESHOLD: f64 = 3.0;

/// High-accuracy threshold requiring 13-point quadrature
pub const HIGH_ACCURACY_THRESHOLD: f64 = 2.0;

/// Determine the optimal quadrature order based on distance to element
///
/// Returns the recommended Gauss order for triangle quadrature.
/// - ratio < 2.0: 13-point (very near-singular)
/// - ratio < 3.0: 7-point (quasi-singular)
/// - ratio >= 3.0: 4-point (regular)
pub fn optimal_quadrature_order(distance: f64, element_size: f64) -> usize {
    let ratio = distance / element_size;
    if ratio < HIGH_ACCURACY_THRESHOLD {
        4 // Maps to 13-point in triangle_quadrature
    } else if ratio < QUASI_SINGULAR_THRESHOLD {
        3 // Maps to 7-point in triangle_quadrature
    } else {
        2 // Maps to 4-point in triangle_quadrature
    }
}

/// Perform quasi-singular integration with adaptive quadrature order
///
/// This function determines the appropriate quadrature order based on the
/// distance between source point and element, using higher-order rules
/// for near-singular cases.
///
/// # Arguments
/// * `source_point` - Source point (collocation point)
/// * `source_normal` - Unit normal at source point
/// * `element_coords` - Node coordinates of the field element (num_nodes × 3)
/// * `element_type` - Triangle or quad element
/// * `element_center` - Center of the element
/// * `element_area` - Area of the field element
/// * `physics` - Physics parameters (wave number, etc.)
///
/// # Returns
/// IntegrationResult with G, H, H^T, E integrals
pub fn quasi_singular_integration(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    element_center: &[f64; 3],
    element_area: f64,
    physics: &PhysicsParams,
) -> IntegrationResult {
    // Compute distance from source to element center
    let mut dist_sq = 0.0;
    for i in 0..3 {
        let d = element_center[i] - source_point[i];
        dist_sq += d * d;
    }
    let distance = dist_sq.sqrt();
    let element_size = element_area.sqrt();
    let ratio = distance / element_size;

    // Choose integration strategy based on distance ratio
    if ratio < HIGH_ACCURACY_THRESHOLD {
        // Very close: use adaptive subdivision with high-order quadrature
        regular_integration(
            source_point,
            source_normal,
            element_coords,
            element_type,
            element_area,
            physics,
            None,
            0,
            false,
        )
    } else if ratio < QUASI_SINGULAR_THRESHOLD {
        // Quasi-singular: use high-order quadrature without subdivision
        let order = optimal_quadrature_order(distance, element_size);
        regular_integration_fixed_order(
            source_point,
            source_normal,
            element_coords,
            element_type,
            physics,
            order,
        )
    } else {
        // Regular far-field: use standard 4-point quadrature
        regular_integration_fixed_order(
            source_point,
            source_normal,
            element_coords,
            element_type,
            physics,
            2, // 4-point quadrature
        )
    }
}

/// Compute element integration with fixed quadrature order (no adaptation)
///
/// This is a simpler version for far-field elements where adaptation is not needed.
pub fn regular_integration_fixed_order(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    gauss_order: usize,
) -> IntegrationResult {
    let wavruim = physics.harmonic_factor * physics.wave_number;
    let k2 = physics.wave_number * physics.wave_number;

    let mut result = IntegrationResult::default();

    // Get quadrature points
    let quad_points = get_quadrature_points(element_type, gauss_order);

    for (xi, eta, weight) in quad_points {
        // Compute shape functions, Jacobian and position
        let params = compute_parameters(element_coords, element_type, xi, eta);
        let jacobian = params.jacobian;
        let el_norm = params.normal;
        let crd_poi = params.position;

        let wga = weight * jacobian;

        // Distance vector
        let diff_fsp = [
            crd_poi[0] - source_point[0],
            crd_poi[1] - source_point[1],
            crd_poi[2] - source_point[2],
        ];

        let (unit_r, dis_fsp) = normalize(diff_fsp);

        if dis_fsp < 1e-15 {
            continue;
        }

        // G kernel
        let kr = wavruim * dis_fsp;
        let g_scaled = wga / (4.0 * PI * dis_fsp);
        let zg = sonium_fastmath::exp_i(kr) * g_scaled;

        // H and H^T kernels
        let z1 = Complex64::new(-1.0 / dis_fsp, wavruim);
        let zhh_base = zg * z1;
        let r_dot_ny = unit_r[0] * el_norm[0] + unit_r[1] * el_norm[1] + unit_r[2] * el_norm[2];
        let r_dot_nx = -(unit_r[0] * source_normal[0]
            + unit_r[1] * source_normal[1]
            + unit_r[2] * source_normal[2]);
        let zhh = zhh_base * r_dot_ny;
        let zht = zhh_base * r_dot_nx;

        // E kernel
        let rq = r_dot_ny * r_dot_nx;
        let nx_dot_ny = source_normal[0] * el_norm[0]
            + source_normal[1] * el_norm[1]
            + source_normal[2] * el_norm[2];
        let dq = dis_fsp * dis_fsp;

        let ze_factor = Complex64::new(
            (3.0 / dq - k2) * rq + nx_dot_ny / dq,
            -wavruim / dis_fsp * (3.0 * rq + nx_dot_ny),
        );
        let ze = zg * ze_factor;

        // Accumulate
        result.g_integral += zg;
        result.dg_dn_integral += zhh;
        result.dg_dnx_integral += zht;
        result.d2g_dnxdny_integral += ze;
    }

    result
}

/// Evaluate the analytic derivative of the fixed-order regular kernels with
/// respect to the acoustic wavenumber.
///
/// The geometry, quadrature weights, normals, and element basis are held
/// fixed.  This returns the derivative of the four kernel channels in the
/// same layout as [`IntegrationResult`]; `rhs_contribution` is zero because
/// this entry point does not assemble a boundary-condition contribution.
/// Callers assembling a Burton--Miller RHS must additionally differentiate
/// its explicit `beta(k)` factor and incident field.
///
/// Keeping this derivative beside the production fixed-order evaluator makes
/// it usable by dense, H-matrix, and FMM validation code without introducing
/// finite differences into the numerical kernel itself.
/// Evaluate regular kernels at a fixed order while retaining panel basis
/// weights. This is the higher-order counterpart of
/// [`regular_integration_fixed_order`] and is intentionally crate-visible so
/// checked TBEM admission can estimate every basis coefficient, rather than
/// only the unweighted panel integral.
pub(crate) fn regular_integration_linear_fixed_order(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    num_basis: usize,
    gauss_order: usize,
) -> LinearIntegrationResult {
    let wavruim = physics.harmonic_factor * physics.wave_number;
    let k2 = physics.wave_number * physics.wave_number;
    let mut basis_results = vec![IntegrationResult::default(); num_basis];

    for (xi, eta, weight) in get_quadrature_points(element_type, gauss_order) {
        let params = compute_parameters(element_coords, element_type, xi, eta);
        let (unit_r, distance) = normalize([
            params.position[0] - source_point[0],
            params.position[1] - source_point[1],
            params.position[2] - source_point[2],
        ]);
        if distance < 1e-15 {
            continue;
        }

        let wga = weight * params.jacobian;
        let zg = sonium_fastmath::exp_i(wavruim * distance) * (wga / (4.0 * PI * distance));
        let z1 = Complex64::new(-1.0 / distance, wavruim);
        let zhh_base = zg * z1;
        let r_dot_ny = unit_r[0] * params.normal[0]
            + unit_r[1] * params.normal[1]
            + unit_r[2] * params.normal[2];
        let r_dot_nx = -(unit_r[0] * source_normal[0]
            + unit_r[1] * source_normal[1]
            + unit_r[2] * source_normal[2]);
        let zhh = zhh_base * r_dot_ny;
        let zht = zhh_base * r_dot_nx;
        let rq = r_dot_ny * r_dot_nx;
        let nx_dot_ny = source_normal[0] * params.normal[0]
            + source_normal[1] * params.normal[1]
            + source_normal[2] * params.normal[2];
        let distance_squared = distance * distance;
        let ze = zg
            * Complex64::new(
                (3.0 / distance_squared - k2) * rq + nx_dot_ny / distance_squared,
                -wavruim / distance * (3.0 * rq + nx_dot_ny),
            );

        for (result, basis_value) in
            basis_results
                .iter_mut()
                .zip(panel_basis_values(element_type, xi, eta, num_basis))
        {
            result.g_integral += zg * basis_value;
            result.dg_dn_integral += zhh * basis_value;
            result.dg_dnx_integral += zht * basis_value;
            result.d2g_dnxdny_integral += ze * basis_value;
        }
    }

    LinearIntegrationResult { basis_results }
}

/// Evaluates fixed-order regular integration and its wavenumber derivative.
pub fn regular_integration_fixed_order_wavenumber_derivative(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    gauss_order: usize,
) -> IntegrationResult {
    let phase_sign = physics.harmonic_factor;
    let k = physics.wave_number;
    let dk2_dk = 2.0 * k;
    let mut derivative = IntegrationResult::default();

    for (xi, eta, weight) in get_quadrature_points(element_type, gauss_order) {
        let params = compute_parameters(element_coords, element_type, xi, eta);
        let wga = weight * params.jacobian;
        let diff = [
            params.position[0] - source_point[0],
            params.position[1] - source_point[1],
            params.position[2] - source_point[2],
        ];
        let (unit_r, distance) = normalize(diff);
        if distance < 1e-15 {
            continue;
        }

        let phase = phase_sign * k * distance;
        let g = sonium_fastmath::exp_i(phase) * (wga / (4.0 * PI * distance));
        let dg_dk = g * Complex64::new(0.0, phase_sign * distance);

        let r_dot_ny = unit_r[0] * params.normal[0]
            + unit_r[1] * params.normal[1]
            + unit_r[2] * params.normal[2];
        let r_dot_nx = -(unit_r[0] * source_normal[0]
            + unit_r[1] * source_normal[1]
            + unit_r[2] * source_normal[2]);
        let d_kernel_factor = Complex64::new(0.0, phase_sign);
        let h_factor = Complex64::new(-1.0 / distance, phase_sign * k);
        let dh_dk = (dg_dk * h_factor + g * d_kernel_factor) * r_dot_ny;
        let dht_dk = (dg_dk * h_factor + g * d_kernel_factor) * r_dot_nx;

        let rq = r_dot_ny * r_dot_nx;
        let nx_dot_ny = source_normal[0] * params.normal[0]
            + source_normal[1] * params.normal[1]
            + source_normal[2] * params.normal[2];
        let distance_squared = distance * distance;
        let e_factor = Complex64::new(
            (3.0 / distance_squared - k * k) * rq + nx_dot_ny / distance_squared,
            -phase_sign * k / distance * (3.0 * rq + nx_dot_ny),
        );
        let de_factor = Complex64::new(
            -dk2_dk * rq,
            -phase_sign / distance * (3.0 * rq + nx_dot_ny),
        );
        let de_dk = dg_dk * e_factor + g * de_factor;

        derivative.g_integral += dg_dk;
        derivative.dg_dn_integral += dh_dk;
        derivative.dg_dnx_integral += dht_dk;
        derivative.d2g_dnxdny_integral += de_dk;
    }

    derivative
}

/// Evaluate the fixed-order analytic wavenumber derivative while retaining
/// one result per higher-order panel basis function.
///
/// This is the basis-weighted counterpart of
/// [`regular_integration_fixed_order_wavenumber_derivative`].  Geometry,
/// quadrature weights, normals, and basis functions are held fixed; only the
/// Helmholtz kernels (including their wavenumber-dependent factors) are
/// differentiated.  Keeping the derivative in the same per-basis layout as
/// [`regular_integration_linear_fixed_order`] lets compressed higher-order
/// operators use the analytic derivative kernel instead of a perturbed
/// re-assembly.
pub(crate) fn regular_integration_linear_fixed_order_wavenumber_derivative(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    num_basis: usize,
    gauss_order: usize,
) -> LinearIntegrationResult {
    regular_integration_linear_fixed_order_wavenumber_derivative_with_normals(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        num_basis,
        gauss_order,
        None,
    )
}

/// Fixed-order regular linear derivative with optional interpolated normals.
pub(crate) fn regular_integration_linear_fixed_order_wavenumber_derivative_with_normals(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    num_basis: usize,
    gauss_order: usize,
    node_normals: Option<&[[f64; 3]]>,
) -> LinearIntegrationResult {
    let phase_sign = physics.harmonic_factor;
    let k = physics.wave_number;
    let dk2_dk = 2.0 * k;
    let mut basis_results = vec![IntegrationResult::default(); num_basis];

    for (xi, eta, weight) in get_quadrature_points(element_type, gauss_order) {
        let params = compute_parameters(element_coords, element_type, xi, eta);
        let (position, jacobian, normal) = if node_normals.is_some() {
            quadrature_geometry(
                element_coords,
                &params.shape_fn,
                &params.shape_ds,
                &params.shape_dt,
                node_normals,
            )
        } else {
            (params.position, params.jacobian, params.normal)
        };
        let wga = weight * jacobian;
        let diff = [
            position[0] - source_point[0],
            position[1] - source_point[1],
            position[2] - source_point[2],
        ];
        let (unit_r, distance) = normalize(diff);
        if distance < 1e-15 {
            continue;
        }

        let phase = phase_sign * k * distance;
        let g = sonium_fastmath::exp_i(phase) * (wga / (4.0 * PI * distance));
        let dg_dk = g * Complex64::new(0.0, phase_sign * distance);

        let r_dot_ny = unit_r[0] * normal[0] + unit_r[1] * normal[1] + unit_r[2] * normal[2];
        let r_dot_nx = -(unit_r[0] * source_normal[0]
            + unit_r[1] * source_normal[1]
            + unit_r[2] * source_normal[2]);
        let d_kernel_factor = Complex64::new(0.0, phase_sign);
        let h_factor = Complex64::new(-1.0 / distance, phase_sign * k);
        let dh_dk = (dg_dk * h_factor + g * d_kernel_factor) * r_dot_ny;
        let dht_dk = (dg_dk * h_factor + g * d_kernel_factor) * r_dot_nx;

        let rq = r_dot_ny * r_dot_nx;
        let nx_dot_ny = source_normal[0] * normal[0]
            + source_normal[1] * normal[1]
            + source_normal[2] * normal[2];
        let distance_squared = distance * distance;
        let e_factor = Complex64::new(
            (3.0 / distance_squared - k * k) * rq + nx_dot_ny / distance_squared,
            -phase_sign * k / distance * (3.0 * rq + nx_dot_ny),
        );
        let de_factor = Complex64::new(
            -dk2_dk * rq,
            -phase_sign / distance * (3.0 * rq + nx_dot_ny),
        );
        let de_dk = dg_dk * e_factor + g * de_factor;

        for (result, basis_value) in
            basis_results
                .iter_mut()
                .zip(panel_basis_values(element_type, xi, eta, num_basis))
        {
            result.g_integral += dg_dk * basis_value;
            result.dg_dn_integral += dh_dk * basis_value;
            result.dg_dnx_integral += dht_dk * basis_value;
            result.d2g_dnxdny_integral += de_dk * basis_value;
        }
    }

    LinearIntegrationResult { basis_results }
}

/// Evaluate analytic derivatives of the fixed-order regular kernels with
/// respect to the three source/collocation-point coordinates.
///
/// The field element geometry, source normal, and quadrature geometry are
/// held fixed. This is the geometry derivative for a collocation-point
/// parameterization; derivatives of element Jacobians, normals, or topology
/// require a shape-calculus layer above this kernel contract.
pub fn regular_integration_fixed_order_source_point_derivative(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    gauss_order: usize,
) -> [IntegrationResult; 3] {
    let wavruim = physics.harmonic_factor * physics.wave_number;
    let k2 = physics.wave_number * physics.wave_number;
    let mut derivative = std::array::from_fn(|_| IntegrationResult::default());

    for (xi, eta, weight) in get_quadrature_points(element_type, gauss_order) {
        let params = compute_parameters(element_coords, element_type, xi, eta);
        let wga = weight * params.jacobian;
        let diff = [
            params.position[0] - source_point[0],
            params.position[1] - source_point[1],
            params.position[2] - source_point[2],
        ];
        let (unit_r, distance) = normalize(diff);
        if distance < 1e-15 {
            continue;
        }

        let g = sonium_fastmath::exp_i(wavruim * distance) * (wga / (4.0 * PI * distance));
        let z1 = Complex64::new(-1.0 / distance, wavruim);
        let r_dot_ny = unit_r[0] * params.normal[0]
            + unit_r[1] * params.normal[1]
            + unit_r[2] * params.normal[2];
        let r_dot_nx = -(unit_r[0] * source_normal[0]
            + unit_r[1] * source_normal[1]
            + unit_r[2] * source_normal[2]);
        let rq = r_dot_ny * r_dot_nx;
        let nx_dot_ny = source_normal[0] * params.normal[0]
            + source_normal[1] * params.normal[1]
            + source_normal[2] * params.normal[2];
        let distance_squared = distance * distance;
        let e_factor = Complex64::new(
            (3.0 / distance_squared - k2) * rq + nx_dot_ny / distance_squared,
            -wavruim / distance * (3.0 * rq + nx_dot_ny),
        );

        for axis in 0..3 {
            // Differentiate with respect to r first, then apply
            // d r / d source_point = -r_hat.
            let dr = unit_r[axis];
            let dg_dr = g * z1 * dr;
            let d_r_dot_ny = (params.normal[axis] - r_dot_ny * unit_r[axis]) / distance;
            let d_r_dot_nx = (-source_normal[axis] - r_dot_nx * unit_r[axis]) / distance;
            let d_rq = d_r_dot_ny * r_dot_nx + r_dot_ny * d_r_dot_nx;
            let d_a = -6.0 * dr / (distance * distance_squared);
            let d_b = -2.0 * dr / (distance * distance_squared);
            let d_c = wavruim * dr / distance_squared;
            let a = 3.0 / distance_squared - k2;
            let c = -wavruim / distance;
            let d_e_factor = Complex64::new(
                d_a * rq + a * d_rq + d_b * nx_dot_ny,
                d_c * (3.0 * rq + nx_dot_ny) + c * 3.0 * d_rq,
            );
            // source_point moves opposite to r, hence the minus sign.
            let dg = -dg_dr;
            let dh = -(dg_dr * z1 * r_dot_ny
                + g * (dr / (distance * distance)) * r_dot_ny
                + g * z1 * d_r_dot_ny);
            let dht = -(dg_dr * z1 * r_dot_nx
                + g * (dr / (distance * distance)) * r_dot_nx
                + g * z1 * d_r_dot_nx);
            let de = -(dg_dr * e_factor + g * d_e_factor);
            derivative[axis].g_integral += dg;
            derivative[axis].dg_dn_integral += dh;
            derivative[axis].dg_dnx_integral += dht;
            derivative[axis].d2g_dnxdny_integral += de;
        }
    }

    derivative
}

/// Compare two fixed-order regular evaluations, including the hypersingular term.
///
/// The helper is also suitable for near-singular and curved-element validation
/// when the caller supplies the corresponding fixed-order evaluations. Existing
/// production integration entry points remain unchanged.
pub fn regular_integration_error_estimate(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    coarse_order: usize,
    fine_order: usize,
) -> Result<IntegrationEstimate, QuadratureOrderError> {
    if coarse_order == 0
        || fine_order <= coarse_order
        || fine_order > crate::core::integration::gauss::N_GAUORDER
    {
        return Err(QuadratureOrderError {
            coarse_order,
            fine_order,
        });
    }

    let coarse = regular_integration_fixed_order(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        coarse_order,
    );
    let fine = regular_integration_fixed_order(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        fine_order,
    );
    Ok(IntegrationEstimate::from_results(coarse, fine))
}

/// Evaluate a regular panel with an explicit adjacent-order convergence gate.
///
/// The returned value is the finer-order result only after every kernel
/// channel (single layer, both double layers, hypersingular term, and RHS)
/// satisfies the requested mixed absolute/relative budget. This checked
/// entry point keeps the historical infallible/adaptive API available while
/// giving production assembly callers a typed failure instead of silently
/// accepting an under-resolved quadrature estimate.
pub fn regular_integration_checked(
    source_point: &[f64; 3],
    source_normal: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    coarse_order: usize,
    fine_order: usize,
    absolute_tolerance: f64,
    relative_tolerance: f64,
) -> Result<IntegrationResult, QuadratureConvergenceError> {
    let estimate = regular_integration_error_estimate(
        source_point,
        source_normal,
        element_coords,
        element_type,
        physics,
        coarse_order,
        fine_order,
    )
    .map_err(QuadratureConvergenceError::InvalidOrder)?;
    estimate.require_converged(absolute_tolerance, relative_tolerance)?;
    Ok(estimate.fine)
}

/// Compute G kernel only (single layer potential) for efficiency
///
/// Used when only G is needed (e.g., for FMM far-field approximation).
pub fn integrate_g_only(
    source_point: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    gauss_order: usize,
) -> Complex64 {
    let wavruim = physics.harmonic_factor * physics.wave_number;
    let mut result = Complex64::new(0.0, 0.0);

    let quad_points = get_quadrature_points(element_type, gauss_order);

    for (xi, eta, weight) in quad_points {
        let params = compute_parameters(element_coords, element_type, xi, eta);
        let jacobian = params.jacobian;
        let crd_poi = params.position;

        let wga = weight * jacobian;

        let mut diff = [0.0_f64; 3];
        for i in 0..3 {
            diff[i] = crd_poi[i] - source_point[i];
        }
        let r = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();

        if r < 1e-15 {
            continue;
        }

        let kr = wavruim * r;
        let g_scaled = wga / (4.0 * PI * r);
        result += sonium_fastmath::exp_i(kr) * g_scaled;
    }

    result
}

/// Compute H kernel only (double layer potential)
///
/// Used when only the double layer is needed.
pub fn integrate_h_only(
    source_point: &[f64; 3],
    element_coords: &DenseMatrix<f64>,
    element_type: ElementType,
    physics: &PhysicsParams,
    gauss_order: usize,
) -> Complex64 {
    let wavruim = physics.harmonic_factor * physics.wave_number;
    let mut result = Complex64::new(0.0, 0.0);

    let quad_points = get_quadrature_points(element_type, gauss_order);

    for (xi, eta, weight) in quad_points {
        let params = compute_parameters(element_coords, element_type, xi, eta);
        let jacobian = params.jacobian;
        let el_norm = params.normal;
        let crd_poi = params.position;

        let wga = weight * jacobian;

        let mut diff = [0.0_f64; 3];
        for i in 0..3 {
            diff[i] = crd_poi[i] - source_point[i];
        }
        let r = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();

        if r < 1e-15 {
            continue;
        }

        let kr = wavruim * r;
        let g_scaled = wga / (4.0 * PI * r);
        let zg = sonium_fastmath::exp_i(kr) * g_scaled;

        let z1 = Complex64::new(-1.0 / r, wavruim);
        let r_dot_n: f64 = diff
            .iter()
            .zip(el_norm.iter())
            .map(|(d, n)| d * n)
            .sum::<f64>()
            / r;

        result += zg * z1 * r_dot_n;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_triangle() -> DenseMatrix<f64> {
        DenseMatrix::from_row_major_vec(3, 3, vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    }

    fn make_test_quad() -> DenseMatrix<f64> {
        DenseMatrix::from_row_major_vec(
            4,
            3,
            vec![0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0],
        )
    }

    #[test]
    fn test_regular_integration_far_field() {
        let coords = make_test_triangle();
        let source = [10.0, 0.0, 0.0]; // Far from element
        let normal = [0.0, 0.0, 1.0];

        let physics = PhysicsParams::new(1000.0, 343.0, 1.21, false);

        let result = regular_integration(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            0.5,
            &physics,
            None,
            0,
            false,
        );

        // G integral should be finite and small (far field)
        assert!(result.g_integral.norm().is_finite());
        assert!(result.g_integral.norm() < 0.1);
    }

    #[test]
    fn test_regular_integration_fixed_order() {
        let coords = make_test_triangle();
        let source = [5.0, 5.0, 0.0];
        let normal = [0.0, 0.0, 1.0];

        let physics = PhysicsParams::new(1000.0, 343.0, 1.21, false);

        let result = regular_integration_fixed_order(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &physics,
            4,
        );

        assert!(result.g_integral.norm().is_finite());
    }

    #[test]
    fn fixed_order_wavenumber_derivative_matches_taylor_slope() {
        let coords = make_test_triangle();
        let source = [2.0, 1.5, 1.25];
        let normal = [0.0, 0.0, 1.0];
        let speed = 343.0;
        let density = 1.21;
        let k = 2.3;
        let physics = PhysicsParams::new(k * speed / (2.0 * PI), speed, density, false);
        let derivative = regular_integration_fixed_order_wavenumber_derivative(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &physics,
            6,
        );

        let evaluate = |delta: f64| {
            let perturbed =
                PhysicsParams::new((k + delta) * speed / (2.0 * PI), speed, density, false);
            regular_integration_fixed_order(
                &source,
                &normal,
                &coords,
                ElementType::Tri3,
                &perturbed,
                6,
            )
        };
        let step = 1.0e-5;
        let plus = evaluate(step);
        let minus = evaluate(-step);
        let central = |a: Complex64, b: Complex64| (a - b) / (2.0 * step);

        for (actual, expected) in [
            (
                central(plus.g_integral, minus.g_integral),
                derivative.g_integral,
            ),
            (
                central(plus.dg_dn_integral, minus.dg_dn_integral),
                derivative.dg_dn_integral,
            ),
            (
                central(plus.dg_dnx_integral, minus.dg_dnx_integral),
                derivative.dg_dnx_integral,
            ),
            (
                central(plus.d2g_dnxdny_integral, minus.d2g_dnxdny_integral),
                derivative.d2g_dnxdny_integral,
            ),
        ] {
            let error = (actual - expected).norm();
            let scale = expected.norm().max(1.0);
            assert!(
                error / scale < 1.0e-8,
                "derivative error={error:e}, expected={expected:?}"
            );
        }
    }

    #[test]
    fn adaptive_wavenumber_derivative_matches_central_difference() {
        let coords = make_test_triangle();
        let source = [2.0, 1.5, 1.25];
        let normal = [0.0, 0.0, 1.0];
        let speed = 343.0;
        let density = 1.21;
        let k = 2.3;
        let physics = PhysicsParams::new(k * speed / (2.0 * PI), speed, density, false);
        let derivative = regular_integration_wavenumber_derivative(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            0.5,
            &physics,
            None,
            0,
            false,
            None,
        );
        let evaluate = |delta: f64| {
            let perturbed =
                PhysicsParams::new((k + delta) * speed / (2.0 * PI), speed, density, false);
            regular_integration(
                &source,
                &normal,
                &coords,
                ElementType::Tri3,
                0.5,
                &perturbed,
                None,
                0,
                false,
            )
        };
        let step = 1.0e-5;
        let plus = evaluate(step);
        let minus = evaluate(-step);
        let central = |a: Complex64, b: Complex64| (a - b) / (2.0 * step);
        for (actual, expected) in [
            (
                central(plus.g_integral, minus.g_integral),
                derivative.g_integral,
            ),
            (
                central(plus.dg_dn_integral, minus.dg_dn_integral),
                derivative.dg_dn_integral,
            ),
            (
                central(plus.dg_dnx_integral, minus.dg_dnx_integral),
                derivative.dg_dnx_integral,
            ),
            (
                central(plus.d2g_dnxdny_integral, minus.d2g_dnxdny_integral),
                derivative.d2g_dnxdny_integral,
            ),
        ] {
            let error = (actual - expected).norm();
            let scale = expected.norm().max(1.0);
            assert!(
                error / scale < 1.0e-8,
                "adaptive derivative error={error:e}, expected={expected:?}"
            );
        }
    }

    #[test]
    fn source_point_geometry_derivative_matches_central_difference() {
        let coords = make_test_triangle();
        let source = [2.0, 1.5, 1.25];
        let normal = [0.0, 0.0, 1.0];
        let physics = PhysicsParams::new(900.0, 343.0, 1.21, false);
        let analytic = regular_integration_fixed_order_source_point_derivative(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &physics,
            6,
        );
        let step = 1.0e-6;
        for axis in 0..3 {
            let mut plus_source = source;
            plus_source[axis] += step;
            let mut minus_source = source;
            minus_source[axis] -= step;
            let plus = regular_integration_fixed_order(
                &plus_source,
                &normal,
                &coords,
                ElementType::Tri3,
                &physics,
                6,
            );
            let minus = regular_integration_fixed_order(
                &minus_source,
                &normal,
                &coords,
                ElementType::Tri3,
                &physics,
                6,
            );
            let central = |a: Complex64, b: Complex64| (a - b) / (2.0 * step);
            for (actual, expected) in [
                (
                    central(plus.g_integral, minus.g_integral),
                    analytic[axis].g_integral,
                ),
                (
                    central(plus.dg_dn_integral, minus.dg_dn_integral),
                    analytic[axis].dg_dn_integral,
                ),
                (
                    central(plus.dg_dnx_integral, minus.dg_dnx_integral),
                    analytic[axis].dg_dnx_integral,
                ),
                (
                    central(plus.d2g_dnxdny_integral, minus.d2g_dnxdny_integral),
                    analytic[axis].d2g_dnxdny_integral,
                ),
            ] {
                assert!((actual - expected).norm() < 1.0e-8);
            }
        }
    }

    #[test]
    fn test_regular_quadrature_error_estimate_compares_adjacent_orders() {
        let coords = make_test_triangle();
        let source = [2.0, 2.0, 1.0];
        let normal = [0.0, 0.0, 1.0];
        let physics = PhysicsParams::new(1000.0, 343.0, 1.21, false);

        let estimate = regular_integration_error_estimate(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &physics,
            3,
            4,
        )
        .expect("adjacent orders should be accepted");

        assert!(estimate.error.max_absolute.is_finite());
        assert!(estimate.error.max_relative.is_finite());
        assert!(estimate.is_converged(1.0, 1.0));
        assert!(
            regular_integration_error_estimate(
                &source,
                &normal,
                &coords,
                ElementType::Tri3,
                &physics,
                4,
                3,
            )
            .is_err()
        );
        assert!(
            regular_integration_error_estimate(
                &source,
                &normal,
                &coords,
                ElementType::Tri3,
                &physics,
                4,
                crate::core::integration::gauss::N_GAUORDER + 1,
            )
            .is_err()
        );
    }

    #[test]
    fn checked_regular_quadrature_enforces_mixed_error_budget() {
        let coords = make_test_triangle();
        let source = [2.0, 2.0, 1.0];
        let normal = [0.0, 0.0, 1.0];
        let physics = PhysicsParams::new(1000.0, 343.0, 1.21, false);

        let accepted = regular_integration_checked(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &physics,
            3,
            4,
            1.0,
            1.0,
        )
        .expect("the broad mixed budget should accept this estimate");
        assert!(accepted.g_integral.norm().is_finite());

        let rejected = regular_integration_checked(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &physics,
            3,
            4,
            0.0,
            0.0,
        )
        .expect_err("zero budget must reject a non-identical adjacent estimate");
        assert!(matches!(
            rejected,
            QuadratureConvergenceError::NotConverged { .. }
        ));

        let invalid_order = regular_integration_checked(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &physics,
            4,
            3,
            1.0,
            1.0,
        )
        .expect_err("a non-increasing order pair must be typed failure");
        assert!(matches!(
            invalid_order,
            QuadratureConvergenceError::InvalidOrder(_)
        ));
    }

    #[test]
    fn near_singular_regular_rule_converges_toward_high_order_reference() {
        // A source just off the panel is regular but strongly peaked. This
        // guards the production order-estimate path against a hidden
        // same-rule fallback: increasing the actual quadrature order must
        // improve the single-layer action toward an independently chosen
        // high-order reference.
        let coords = make_test_triangle();
        let source = [0.25, 0.25, 0.1];
        let normal = [0.0, 0.0, 1.0];
        let physics = PhysicsParams::new(1000.0, 343.0, 1.21, false);
        let reference = regular_integration_fixed_order(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &physics,
            64,
        )
        .g_integral;

        let error = |order| {
            (regular_integration_fixed_order(
                &source,
                &normal,
                &coords,
                ElementType::Tri3,
                &physics,
                order,
            )
            .g_integral
                - reference)
                .norm()
        };
        let order_4 = error(4);
        let order_8 = error(8);
        let order_16 = error(16);
        let order_32 = error(32);

        assert!(
            order_8 < order_4,
            "order 8 error={order_8:e}, order 4={order_4:e}"
        );
        assert!(
            order_16 < order_8,
            "order 16 error={order_16:e}, order 8={order_8:e}"
        );
        assert!(
            order_32 < order_16,
            "order 32 error={order_32:e}, order 16={order_16:e}"
        );
        assert!(order_32 < 1.0e-10, "order 32 error={order_32:e}");
        let dispatched = quasi_singular_integration(
            &source,
            &normal,
            &coords,
            ElementType::Tri3,
            &[1.0 / 3.0, 1.0 / 3.0, 0.0],
            0.5,
            &physics,
        );
        assert!(
            dispatched.g_integral.re.is_finite() && dispatched.g_integral.im.is_finite(),
            "near-singular dispatch returned a non-finite single-layer integral"
        );
    }

    #[test]
    fn test_integrate_g_only() {
        let coords = make_test_triangle();
        let source = [2.0, 2.0, 1.0];

        let physics = PhysicsParams::new(1000.0, 343.0, 1.21, false);

        let g = integrate_g_only(&source, &coords, ElementType::Tri3, &physics, 4);

        assert!(g.norm().is_finite());
        assert!(g.norm() > 0.0);
    }

    #[test]
    fn test_quad_element_integration() {
        let coords = make_test_quad();
        let source = [5.0, 0.5, 0.0];
        let normal = [0.0, 0.0, 1.0];

        let physics = PhysicsParams::new(1000.0, 343.0, 1.21, false);

        let result = regular_integration_fixed_order(
            &source,
            &normal,
            &coords,
            ElementType::Quad4,
            &physics,
            4,
        );

        assert!(result.g_integral.norm().is_finite());
    }

    #[test]
    fn test_integration_symmetry() {
        // For a symmetric setup, H^T at point A should equal H at symmetric point
        let coords = make_test_triangle();
        let source1 = [0.5, 0.5, 1.0];
        let normal1 = [0.0, 0.0, 1.0];
        let source2 = [0.5, 0.5, -1.0];
        let normal2 = [0.0, 0.0, -1.0];

        let physics = PhysicsParams::new(1000.0, 343.0, 1.21, false);

        let result1 = regular_integration_fixed_order(
            &source1,
            &normal1,
            &coords,
            ElementType::Tri3,
            &physics,
            4,
        );

        let result2 = regular_integration_fixed_order(
            &source2,
            &normal2,
            &coords,
            ElementType::Tri3,
            &physics,
            4,
        );

        // G should be equal (symmetric sources)
        assert!((result1.g_integral.norm() - result2.g_integral.norm()).abs() < 1e-10);
    }

    #[test]
    fn test_compute_parameters_triangle() {
        let coords = make_test_triangle();

        let params = compute_parameters(&coords, ElementType::Tri3, 0.5, 0.25);

        // Shape functions should sum to 1
        let sum: f64 = params.shape_fn.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Jacobian should be 2 * area = 1
        assert!((params.jacobian - 1.0).abs() < 1e-10);

        // Normal should be (0, 0, 1)
        assert!(params.normal[2].abs() > 0.99);

        // Position check
        assert!((params.position[0] - 0.5).abs() < 1e-10);
        assert!((params.position[1] - 0.25).abs() < 1e-10);
    }

    #[test]
    fn test_compute_parameters_quad() {
        let coords = make_test_quad();

        let params = compute_parameters(&coords, ElementType::Quad4, 0.0, 0.0);

        // Shape functions should sum to 1
        let sum: f64 = params.shape_fn.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // Normal should be (0, 0, 1)
        assert!(params.normal[2].abs() > 0.99);
    }
}
