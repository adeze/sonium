//! Shape function evaluation for finite elements
//!
//! Provides a unified interface for evaluating basis functions and their
//! gradients at arbitrary points within reference elements.

use super::lagrange::*;
use crate::mesh::ElementType;

use std::cell::UnsafeCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::thread_local;

thread_local! {
    static SHAPE_CACHE: UnsafeCell<HashMap<CachedShapeKey, ShapeValues>> =
        UnsafeCell::new(HashMap::new());
}

#[derive(Clone, Debug)]
struct CachedShapeKey {
    elem_type: ElementType,
    degree: PolynomialDegree,
    xi: f64,
    eta: f64,
    zeta: f64,
}

impl PartialEq for CachedShapeKey {
    fn eq(&self, other: &Self) -> bool {
        self.elem_type == other.elem_type
            && self.degree == other.degree
            && self.xi.to_bits() == other.xi.to_bits()
            && self.eta.to_bits() == other.eta.to_bits()
            && self.zeta.to_bits() == other.zeta.to_bits()
    }
}

impl Eq for CachedShapeKey {}

impl Hash for CachedShapeKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.elem_type.hash(state);
        self.degree.hash(state);
        self.xi.to_bits().hash(state);
        self.eta.to_bits().hash(state);
        self.zeta.to_bits().hash(state);
    }
}

/// Shape functions evaluated at a point.
///
/// Gradients are stored as a flat array: `[g0_x, g0_y, g0_z, g1_x, g1_y, g1_z, ...]`
/// with `grad_dim` components per DOF.
#[derive(Debug, Clone)]
pub struct ShapeValues {
    /// Basis function values [N_0, N_1, ..., N_n]
    pub values: Vec<f64>,
    /// Flattened gradients: [dN_0/dxi, dN_0/deta, dN_0/dzeta, dN_1/dxi, ...]
    pub gradients: Vec<f64>,
    /// Number of spatial dimensions for each gradient
    pub grad_dim: usize,
}

/// Typed failure for shape-function evaluation at a public FEM boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeEvaluationError {
    /// The requested element/degree pair has no implemented basis contract.
    UnsupportedElementDegree {
        /// Element family requested by the caller.
        element_type: ElementType,
        /// Polynomial degree requested by the caller.
        degree: PolynomialDegree,
    },
    /// Reference coordinates must be finite before they enter the cache or
    /// basis polynomials.
    NonFiniteReferencePoint,
}

impl std::fmt::Display for ShapeEvaluationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsupportedElementDegree {
                element_type,
                degree,
            } => write!(
                f,
                "shape functions for {element_type:?} degree {degree:?} are not implemented. \
                     Supported: Triangle(P1,P2), Quadrilateral(P1,P2), \
                     Tetrahedron(P1,P2,P3), Hexahedron(P1,P2)"
            ),
            Self::NonFiniteReferencePoint => {
                f.write_str("shape-function reference coordinates must be finite")
            }
        }
    }
}

impl std::error::Error for ShapeEvaluationError {}

/// Evaluate shape functions for an element at a reference point.
/// Results are cached per (ElementType, PolynomialDegree, xi, eta, zeta).
pub fn evaluate_shape(
    element_type: ElementType,
    degree: PolynomialDegree,
    xi: f64,
    eta: f64,
    zeta: f64,
) -> ShapeValues {
    try_evaluate_shape(element_type, degree, xi, eta, zeta)
        .unwrap_or_else(|error| panic!("{error}"))
}

/// Checked shape-function evaluation.
///
/// Callers that accept user-selected element families should use this API so
/// unsupported high-order combinations and non-finite reference points are
/// reported as typed failures rather than panics. [`evaluate_shape`] remains
/// as the historical compatibility wrapper.
pub fn try_evaluate_shape(
    element_type: ElementType,
    degree: PolynomialDegree,
    xi: f64,
    eta: f64,
    zeta: f64,
) -> Result<ShapeValues, ShapeEvaluationError> {
    if !xi.is_finite() || !eta.is_finite() || !zeta.is_finite() {
        return Err(ShapeEvaluationError::NonFiniteReferencePoint);
    }
    let supported = matches!(
        (element_type, degree),
        (
            ElementType::Triangle,
            PolynomialDegree::P1 | PolynomialDegree::P2 | PolynomialDegree::P3
        ) | (
            ElementType::Quadrilateral,
            PolynomialDegree::P1 | PolynomialDegree::P2 | PolynomialDegree::P3
        ) | (
            ElementType::Tetrahedron,
            PolynomialDegree::P1 | PolynomialDegree::P2 | PolynomialDegree::P3
        ) | (
            ElementType::Hexahedron,
            PolynomialDegree::P1 | PolynomialDegree::P2 | PolynomialDegree::P3
        )
    );
    if !supported {
        return Err(ShapeEvaluationError::UnsupportedElementDegree {
            element_type,
            degree,
        });
    }

    Ok(evaluate_shape_cached(element_type, degree, xi, eta, zeta))
}

fn evaluate_shape_cached(
    element_type: ElementType,
    degree: PolynomialDegree,
    xi: f64,
    eta: f64,
    zeta: f64,
) -> ShapeValues {
    let key = CachedShapeKey {
        elem_type: element_type,
        degree,
        xi,
        eta,
        zeta,
    };

    SHAPE_CACHE.with(|cache| {
        let cache = unsafe { &mut *cache.get() };
        if let Some(cached) = cache.get(&key) {
            return cached.clone();
        }

        let result = compute_shape_values(element_type, degree, xi, eta, zeta);
        cache.insert(key, result.clone());
        result
    })
}

fn compute_shape_values(
    element_type: ElementType,
    degree: PolynomialDegree,
    xi: f64,
    eta: f64,
    zeta: f64,
) -> ShapeValues {
    match (element_type, degree) {
        (ElementType::Triangle, PolynomialDegree::P1) => {
            let vals = p1_triangle(xi, eta);
            let grads = p1_triangle_grad();
            let grad_dim = 2;
            let n = grads.len();
            let mut gradients = Vec::with_capacity(n * grad_dim);
            for g in &grads {
                gradients.extend_from_slice(g);
            }
            ShapeValues {
                values: vals.to_vec(),
                gradients,
                grad_dim,
            }
        }
        (ElementType::Triangle, PolynomialDegree::P2) => {
            let vals = p2_triangle(xi, eta);
            let grads = p2_triangle_grad(xi, eta);
            let grad_dim = 2;
            let n = grads.len();
            let mut gradients = Vec::with_capacity(n * grad_dim);
            for g in &grads {
                gradients.extend_from_slice(g);
            }
            ShapeValues {
                values: vals.to_vec(),
                gradients,
                grad_dim,
            }
        }
        (ElementType::Triangle, PolynomialDegree::P3) => {
            let vals = p3_triangle(xi, eta);
            let grads = p3_triangle_grad(xi, eta);
            ShapeValues {
                values: vals.to_vec(),
                gradients: flatten_grads(&grads),
                grad_dim: 2,
            }
        }
        (ElementType::Quadrilateral, PolynomialDegree::P1) => {
            let vals = q1_quadrilateral(xi, eta);
            let grads = q1_quadrilateral_grad(xi, eta);
            let grad_dim = 2;
            let n = grads.len();
            let mut gradients = Vec::with_capacity(n * grad_dim);
            for g in &grads {
                gradients.extend_from_slice(g);
            }
            ShapeValues {
                values: vals.to_vec(),
                gradients,
                grad_dim,
            }
        }
        (ElementType::Quadrilateral, PolynomialDegree::P3) => {
            let vals = q3_quadrilateral(xi, eta);
            let grads = q3_quadrilateral_grad(xi, eta);
            ShapeValues {
                values: vals.to_vec(),
                gradients: flatten_grads(&grads),
                grad_dim: 2,
            }
        }
        (ElementType::Tetrahedron, PolynomialDegree::P1) => {
            let vals = p1_tetrahedron(xi, eta, zeta);
            let grads = p1_tetrahedron_grad();
            let grad_dim = 3;
            let n = grads.len();
            let mut gradients = Vec::with_capacity(n * grad_dim);
            for g in &grads {
                gradients.extend_from_slice(g);
            }
            ShapeValues {
                values: vals.to_vec(),
                gradients,
                grad_dim,
            }
        }
        (ElementType::Tetrahedron, PolynomialDegree::P2) => {
            let vals = p2_tetrahedron(xi, eta, zeta);
            let grads = p2_tetrahedron_grad(xi, eta, zeta);
            ShapeValues {
                values: vals.to_vec(),
                gradients: flatten_grads(&grads),
                grad_dim: 3,
            }
        }
        (ElementType::Tetrahedron, PolynomialDegree::P3) => {
            let vals = p3_tetrahedron(xi, eta, zeta);
            let grads = p3_tetrahedron_grad(xi, eta, zeta);
            ShapeValues {
                values: vals.to_vec(),
                gradients: flatten_grads(&grads),
                grad_dim: 3,
            }
        }
        (ElementType::Quadrilateral, PolynomialDegree::P2) => {
            let vals = q2_quadrilateral(xi, eta);
            let grads = q2_quadrilateral_grad(xi, eta);
            ShapeValues {
                values: vals.to_vec(),
                gradients: flatten_grads(&grads),
                grad_dim: 2,
            }
        }
        (ElementType::Hexahedron, PolynomialDegree::P1) => {
            let vals = q1_hexahedron(xi, eta, zeta);
            let grads = q1_hexahedron_grad(xi, eta, zeta);
            let grad_dim = 3;
            let n = grads.len();
            let mut gradients = Vec::with_capacity(n * grad_dim);
            for g in &grads {
                gradients.extend_from_slice(g);
            }
            ShapeValues {
                values: vals.to_vec(),
                gradients,
                grad_dim,
            }
        }
        (ElementType::Hexahedron, PolynomialDegree::P2) => {
            let vals = q2_hexahedron(xi, eta, zeta);
            let grads = q2_hexahedron_grad(xi, eta, zeta);
            ShapeValues {
                values: vals.to_vec(),
                gradients: flatten_grads(&grads),
                grad_dim: 3,
            }
        }
        (ElementType::Hexahedron, PolynomialDegree::P3) => {
            let vals = q3_hexahedron(xi, eta, zeta);
            let grads = q3_hexahedron_grad(xi, eta, zeta);
            ShapeValues {
                values: vals.to_vec(),
                gradients: flatten_grads(&grads),
                grad_dim: 3,
            }
        }
    }
}

fn flatten_grads<const N: usize, const D: usize>(grads: &[[f64; D]; N]) -> Vec<f64> {
    let mut out = Vec::with_capacity(N * D);
    for g in grads {
        out.extend_from_slice(g);
    }
    out
}

/// Jacobian matrix for coordinate transformation
/// Maps reference element coordinates to physical coordinates
#[derive(Debug, Clone)]
pub struct Jacobian {
    /// Jacobian matrix entries (row-major)
    /// 2D: [[dx/dxi, dx/deta], [dy/dxi, dy/deta]]
    /// 3D: [[dx/dxi, dx/deta, dx/dzeta], ...]
    pub matrix: Vec<Vec<f64>>,
    /// Determinant of Jacobian
    pub det: f64,
    /// Inverse of Jacobian (for gradient transformation) — stored as flat [row*dim + col]
    pub inverse: Vec<f64>,
    /// Dimension of the Jacobian (2 or 3)
    pub dim: usize,
}

impl Jacobian {
    /// Compute 2D Jacobian from flat shape gradients and node coordinates.
    /// grad_ref: flat gradient array [g0_x, g0_y, g1_x, g1_y, ...], grad_dim=2.
    pub fn from_2d(grad_ref: &[f64], grad_dim: usize, coords: &[[f64; 2]]) -> Self {
        let mut j = [[0.0; 2]; 2];

        for (i, g) in grad_ref.chunks(grad_dim).enumerate() {
            j[0][0] += g[0] * coords[i][0];
            j[0][1] += g[1] * coords[i][0];
            j[1][0] += g[0] * coords[i][1];
            j[1][1] += g[1] * coords[i][1];
        }

        let det = j[0][0] * j[1][1] - j[0][1] * j[1][0];
        let inv_det = 1.0 / det;

        let inverse = vec![
            j[1][1] * inv_det,
            -j[0][1] * inv_det,
            -j[1][0] * inv_det,
            j[0][0] * inv_det,
        ];

        Self {
            matrix: vec![vec![j[0][0], j[0][1]], vec![j[1][0], j[1][1]]],
            det,
            inverse,
            dim: 2,
        }
    }

    /// Compute 3D Jacobian from flat shape gradients and node coordinates.
    pub fn from_3d(grad_ref: &[f64], grad_dim: usize, coords: &[[f64; 3]]) -> Self {
        let mut j = [[0.0; 3]; 3];

        for (i, g) in grad_ref.chunks(grad_dim).enumerate() {
            for k in 0..3 {
                j[0][k] += g[k] * coords[i][0];
                j[1][k] += g[k] * coords[i][1];
                j[2][k] += g[k] * coords[i][2];
            }
        }

        let det = j[0][0] * (j[1][1] * j[2][2] - j[1][2] * j[2][1])
            - j[0][1] * (j[1][0] * j[2][2] - j[1][2] * j[2][0])
            + j[0][2] * (j[1][0] * j[2][1] - j[1][1] * j[2][0]);

        let inv_det = 1.0 / det;

        let inverse = vec![
            (j[1][1] * j[2][2] - j[1][2] * j[2][1]) * inv_det,
            (j[0][2] * j[2][1] - j[0][1] * j[2][2]) * inv_det,
            (j[0][1] * j[1][2] - j[0][2] * j[1][1]) * inv_det,
            (j[1][2] * j[2][0] - j[1][0] * j[2][2]) * inv_det,
            (j[0][0] * j[2][2] - j[0][2] * j[2][0]) * inv_det,
            (j[0][2] * j[1][0] - j[0][0] * j[1][2]) * inv_det,
            (j[1][0] * j[2][1] - j[1][1] * j[2][0]) * inv_det,
            (j[0][1] * j[2][0] - j[0][0] * j[2][1]) * inv_det,
            (j[0][0] * j[1][1] - j[0][1] * j[1][0]) * inv_det,
        ];

        Self {
            matrix: vec![
                vec![j[0][0], j[0][1], j[0][2]],
                vec![j[1][0], j[1][1], j[1][2]],
                vec![j[2][0], j[2][1], j[2][2]],
            ],
            det,
            inverse,
            dim: 3,
        }
    }

    /// Transform reference gradient to physical gradient: grad_phys = J^{-T} * grad_ref.
    /// grad_ref: flat array with dim components.
    /// Returns flat array with dim components.
    pub fn transform_gradient(&self, grad_ref: &[f64]) -> Vec<f64> {
        let dim = self.dim;
        let mut result = vec![0.0; dim];

        for (i, result_i) in result.iter_mut().enumerate() {
            for (j, &grad_j) in grad_ref.iter().enumerate() {
                *result_i += self.inverse[j * dim + i] * grad_j;
            }
        }

        result
    }

    /// Transform reference gradient in-place into pre-allocated output.
    pub fn transform_gradient_into(&self, grad_ref: &[f64], out: &mut [f64]) {
        let dim = self.dim;
        for (i, out_i) in out.iter_mut().enumerate() {
            *out_i = 0.0;
            for (j, &grad_j) in grad_ref.iter().enumerate() {
                *out_i += self.inverse[j * dim + i] * grad_j;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};

    fn hash_key(key: &CachedShapeKey) -> u64 {
        let mut hasher = DefaultHasher::new();
        key.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn test_evaluate_shape_p1_triangle() {
        let shape = evaluate_shape(ElementType::Triangle, PolynomialDegree::P1, 0.25, 0.25, 0.0);
        assert_eq!(shape.values.len(), 3);
        assert_eq!(shape.gradients.len(), 6); // 3 DOFs × 2 components

        // Partition of unity
        let sum: f64 = shape.values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_evaluate_shape_p2_triangle_kronecker_and_gradient_sum() {
        let nodes = [
            (0.0, 0.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (0.5, 0.0),
            (0.5, 0.5),
            (0.0, 0.5),
        ];

        for (node_idx, &(xi, eta)) in nodes.iter().enumerate() {
            let shape = evaluate_shape(ElementType::Triangle, PolynomialDegree::P2, xi, eta, 0.0);
            assert_eq!(shape.values.len(), 6);
            assert_eq!(shape.gradients.len(), 12);
            assert_eq!(shape.grad_dim, 2);

            for (basis_idx, &value) in shape.values.iter().enumerate() {
                let expected = if basis_idx == node_idx { 1.0 } else { 0.0 };
                assert!(
                    (value - expected).abs() < 1e-14,
                    "Triangle P2 basis {basis_idx} at node {node_idx}: got {value}, expected {expected}"
                );
            }

            let grad_sum =
                shape
                    .gradients
                    .chunks(shape.grad_dim)
                    .fold([0.0; 2], |mut acc, grad| {
                        acc[0] += grad[0];
                        acc[1] += grad[1];
                        acc
                    });
            assert!(grad_sum[0].abs() < 1e-14);
            assert!(grad_sum[1].abs() < 1e-14);
        }
    }

    #[test]
    fn cached_shape_key_hash_matches_equality() {
        let key = CachedShapeKey {
            elem_type: ElementType::Triangle,
            degree: PolynomialDegree::P1,
            xi: 0.25,
            eta: 0.25,
            zeta: 0.0,
        };
        let same_key = CachedShapeKey { ..key.clone() };
        let nearby_key = CachedShapeKey {
            xi: f64::from_bits(key.xi.to_bits() + 1),
            ..key.clone()
        };

        assert_eq!(key, same_key);
        assert_eq!(hash_key(&key), hash_key(&same_key));
        assert_ne!(key, nearby_key);

        let mut cache = HashMap::new();
        cache.insert(key.clone(), "exact");
        cache.insert(nearby_key.clone(), "nearby");

        assert_eq!(cache.get(&key), Some(&"exact"));
        assert_eq!(cache.get(&same_key), Some(&"exact"));
        assert_eq!(cache.get(&nearby_key), Some(&"nearby"));
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn test_evaluate_shape_p2_tetrahedron_partition_of_unity() {
        let shape = evaluate_shape(
            ElementType::Tetrahedron,
            PolynomialDegree::P2,
            0.2,
            0.3,
            0.1,
        );
        assert_eq!(shape.values.len(), 10);
        assert_eq!(shape.gradients.len(), 30);

        let sum: f64 = shape.values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_evaluate_shape_p2_tetrahedron_kronecker_and_gradient_sum() {
        let nodes = [
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
            (0.5, 0.0, 0.0),
            (0.0, 0.5, 0.0),
            (0.0, 0.0, 0.5),
            (0.5, 0.5, 0.0),
            (0.5, 0.0, 0.5),
            (0.0, 0.5, 0.5),
        ];

        for (node_idx, &(xi, eta, zeta)) in nodes.iter().enumerate() {
            let shape = evaluate_shape(
                ElementType::Tetrahedron,
                PolynomialDegree::P2,
                xi,
                eta,
                zeta,
            );
            assert_eq!(shape.values.len(), 10);
            assert_eq!(shape.gradients.len(), 30);
            assert_eq!(shape.grad_dim, 3);

            for (basis_idx, &value) in shape.values.iter().enumerate() {
                let expected = if basis_idx == node_idx { 1.0 } else { 0.0 };
                assert!(
                    (value - expected).abs() < 1e-14,
                    "Tetrahedron P2 basis {basis_idx} at node {node_idx}: got {value}, expected {expected}"
                );
            }

            let grad_sum =
                shape
                    .gradients
                    .chunks(shape.grad_dim)
                    .fold([0.0; 3], |mut acc, grad| {
                        acc[0] += grad[0];
                        acc[1] += grad[1];
                        acc[2] += grad[2];
                        acc
                    });
            assert!(grad_sum[0].abs() < 1e-14);
            assert!(grad_sum[1].abs() < 1e-14);
            assert!(grad_sum[2].abs() < 1e-14);
        }
    }

    #[test]
    fn test_p3_shapes_are_supported_and_partition_unity() {
        let triangle =
            try_evaluate_shape(ElementType::Triangle, PolynomialDegree::P3, 0.21, 0.27, 0.0)
                .expect("P3 triangle shape evaluation");
        assert_eq!(triangle.values.len(), 10);
        assert_eq!(triangle.gradients.len(), 20);
        assert!((triangle.values.iter().sum::<f64>() - 1.0).abs() < 1e-13);
        let triangle_gradient_sum =
            triangle
                .gradients
                .chunks(2)
                .fold([0.0; 2], |mut sum, gradient| {
                    sum[0] += gradient[0];
                    sum[1] += gradient[1];
                    sum
                });
        assert!(triangle_gradient_sum.iter().all(|sum| sum.abs() < 1e-13));

        let quadrilateral = try_evaluate_shape(
            ElementType::Quadrilateral,
            PolynomialDegree::P3,
            -0.19,
            0.31,
            0.0,
        )
        .expect("P3 quadrilateral shape evaluation");
        assert_eq!(quadrilateral.values.len(), 16);
        assert_eq!(quadrilateral.gradients.len(), 32);
        assert!((quadrilateral.values.iter().sum::<f64>() - 1.0).abs() < 1e-13);
        let quadrilateral_gradient_sum =
            quadrilateral
                .gradients
                .chunks(2)
                .fold([0.0; 2], |mut sum, gradient| {
                    sum[0] += gradient[0];
                    sum[1] += gradient[1];
                    sum
                });
        assert!(
            quadrilateral_gradient_sum
                .iter()
                .all(|sum| sum.abs() < 1e-13)
        );

        let hexahedron = try_evaluate_shape(
            ElementType::Hexahedron,
            PolynomialDegree::P3,
            0.13,
            -0.23,
            0.37,
        )
        .expect("P3 hexahedron shape evaluation");
        assert_eq!(hexahedron.values.len(), 64);
        assert_eq!(hexahedron.gradients.len(), 192);
        assert!((hexahedron.values.iter().sum::<f64>() - 1.0).abs() < 1e-13);
        let hexahedron_gradient_sum =
            hexahedron
                .gradients
                .chunks(3)
                .fold([0.0; 3], |mut sum, gradient| {
                    sum[0] += gradient[0];
                    sum[1] += gradient[1];
                    sum[2] += gradient[2];
                    sum
                });
        assert!(hexahedron_gradient_sum.iter().all(|sum| sum.abs() < 1e-13));
    }

    #[test]
    fn checked_shape_evaluation_rejects_unsupported_and_nonfinite_inputs() {
        assert!(matches!(
            try_evaluate_shape(
                ElementType::Triangle,
                PolynomialDegree::P3,
                f64::NAN,
                0.0,
                0.0
            ),
            Err(ShapeEvaluationError::NonFiniteReferencePoint)
        ));
        assert!(matches!(
            try_evaluate_shape(
                ElementType::Triangle,
                PolynomialDegree::P1,
                f64::NAN,
                0.0,
                0.0
            ),
            Err(ShapeEvaluationError::NonFiniteReferencePoint)
        ));
    }

    #[test]
    fn test_evaluate_shape_q2_quadrilateral_kronecker() {
        let nodes = [
            (-1.0, -1.0),
            (1.0, -1.0),
            (1.0, 1.0),
            (-1.0, 1.0),
            (0.0, -1.0),
            (1.0, 0.0),
            (0.0, 1.0),
            (-1.0, 0.0),
            (0.0, 0.0),
        ];
        for (i, &(xi, eta)) in nodes.iter().enumerate() {
            let shape = evaluate_shape(
                ElementType::Quadrilateral,
                PolynomialDegree::P2,
                xi,
                eta,
                0.0,
            );
            assert_eq!(shape.values.len(), 9);
            assert_eq!(shape.gradients.len(), 18);
            assert_eq!(shape.grad_dim, 2);
            for (j, &v) in shape.values.iter().enumerate() {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (v - expected).abs() < 1e-13,
                    "Q2 quad N_{j} at node {i}: got {v}, expected {expected}"
                );
            }
        }
    }

    #[test]
    fn test_evaluate_shape_q2_hexahedron_partition_and_kronecker() {
        // Partition of unity at a random interior point.
        let shape = evaluate_shape(
            ElementType::Hexahedron,
            PolynomialDegree::P2,
            0.21,
            -0.37,
            0.55,
        );
        assert_eq!(shape.values.len(), 27);
        assert_eq!(shape.gradients.len(), 81);
        assert_eq!(shape.grad_dim, 3);
        let sum: f64 = shape.values.iter().sum();
        assert!((sum - 1.0).abs() < 1e-12, "Q2 hex sum = {sum}");

        // Kronecker delta check at the volume centre.
        let centre = evaluate_shape(ElementType::Hexahedron, PolynomialDegree::P2, 0.0, 0.0, 0.0);
        for (j, &v) in centre.values.iter().enumerate() {
            let expected = if j == 26 { 1.0 } else { 0.0 };
            assert!(
                (v - expected).abs() < 1e-13,
                "Q2 hex N_{j} at centre = {v}, expected {expected}"
            );
        }
    }

    #[test]
    fn test_jacobian_2d_unit_triangle() {
        // Unit triangle with vertices (0,0), (1,0), (0,1)
        let coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        // Flat gradients: [g0_x, g0_y, g1_x, g1_y, g2_x, g2_y]
        let grads = vec![-1.0, -1.0, 1.0, 0.0, 0.0, 1.0];

        let jac = Jacobian::from_2d(&grads, 2, &coords);

        // For unit triangle, J should be identity
        assert!((jac.matrix[0][0] - 1.0).abs() < 1e-14);
        assert!(jac.matrix[0][1].abs() < 1e-14);
        assert!(jac.matrix[1][0].abs() < 1e-14);
        assert!((jac.matrix[1][1] - 1.0).abs() < 1e-14);
        assert!((jac.det - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_jacobian_2d_scaled_triangle() {
        // Scaled triangle with vertices (0,0), (2,0), (0,2)
        let coords = [[0.0, 0.0], [2.0, 0.0], [0.0, 2.0]];
        let grads = vec![-1.0, -1.0, 1.0, 0.0, 0.0, 1.0];

        let jac = Jacobian::from_2d(&grads, 2, &coords);

        // Jacobian should be 2*I
        assert!((jac.matrix[0][0] - 2.0).abs() < 1e-14);
        assert!((jac.matrix[1][1] - 2.0).abs() < 1e-14);
        assert!((jac.det - 4.0).abs() < 1e-14);
    }

    #[test]
    fn test_gradient_transformation() {
        // For unit triangle, gradient should be unchanged
        let coords = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]];
        let grads = vec![-1.0, -1.0, 1.0, 0.0, 0.0, 1.0];

        let jac = Jacobian::from_2d(&grads, 2, &coords);
        let ref_grad = vec![1.0, 2.0];
        let phys_grad = jac.transform_gradient(&ref_grad);

        assert!((phys_grad[0] - 1.0).abs() < 1e-14);
        assert!((phys_grad[1] - 2.0).abs() < 1e-14);
    }
}
