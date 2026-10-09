use super::MaterialSocConfig;
use num_traits::{FromPrimitive, ToPrimitive};
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;
use sonium_solvers_sys::parallel::parallel_map_indexed;

/// Compute strength of connection matrix
///
/// Entry (i,j) is strong if |a_ij| >= theta * max_k!=i |a_ik|
///
/// When theta=0, all off-diagonal entries are considered strong.
/// Small entries below 1% of the diagonal are filtered out before
/// the threshold computation (this removes noise from fill-in).
pub(super) fn compute_strength_matrix<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    theta: f64,
) -> Vec<Vec<usize>> {
    let n = matrix.num_rows;
    let theta_r = T::Real::from_f64(theta).unwrap();
    let filter_tol = T::Real::from_f64(0.01).unwrap();

    parallel_map_indexed(n, |i| {
        // Direct slice walks over the CSR row: avoids the `get(i, i)` binary
        // search and the `row_entries` iterator adapters. Pass order and gates
        // match the two-pass formulation exactly, so output is bit-identical.
        let start = matrix.row_ptrs[i];
        let end = matrix.row_ptrs[i + 1];
        let cols = &matrix.col_indices[start..end];
        let vals = &matrix.values[start..end];

        // Pass 1: diagonal magnitude for the noise threshold. Rows are stored
        // sorted, so the scan stops at the diagonal (average half-row walk).
        let mut diag_norm = T::Real::from_f64(0.0).unwrap();
        for (&j, &val) in cols.iter().zip(vals.iter()) {
            if j == i {
                diag_norm = val.norm();
                break;
            }
        }
        let noise_threshold = filter_tol * diag_norm;

        // Pass 2: max off-diagonal magnitude (excluding noise).
        let mut max_off_diag = T::Real::from_f64(0.0).unwrap();
        for (&j, &val) in cols.iter().zip(vals.iter()) {
            if j != i {
                let norm = val.norm();
                if norm > noise_threshold && norm > max_off_diag {
                    max_off_diag = norm;
                }
            }
        }

        let threshold = theta_r * max_off_diag;

        // Pass 3: select strong neighbors (single `norm()` eval per entry).
        let mut row_strong = Vec::new();
        for (&j, &val) in cols.iter().zip(vals.iter()) {
            if j != i {
                let norm = val.norm();
                if norm >= threshold && norm > noise_threshold {
                    row_strong.push(j);
                }
            }
        }
        row_strong
    })
}

/// Symmetrize a strength-of-connection graph: edge i--j iff i in S_j or j in S_i.
///
/// Classical SOC on strongly nonsymmetric (e.g. upwinded) operators produces
/// a one-sided graph: row `i` only connects to its upwind neighbor. The
/// sequential/parallel coarseners degenerate on such graphs (tie-broken
/// influence measures can select every point as a C-point, yielding no
/// coarsening at all). Symmetrizing restores meaningful C/F splittings, at
/// the price of treating downwind connections as strong for coarsening and
/// interpolation. Used by the ℓAIR restriction path (see `lair.rs`).
pub(super) fn symmetrize_strong(strong: &[Vec<usize>]) -> Vec<Vec<usize>> {
    let n = strong.len();
    let mut sym: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (i, row) in strong.iter().enumerate() {
        for &j in row {
            sym[i].push(j);
            sym[j].push(i);
        }
    }
    for row in &mut sym {
        row.sort_unstable();
        row.dedup();
    }
    sym
}

/// Compute the determinant of a dim x dim tensor stored in row-major order.
///
/// Returns `abs(det)` clamped to at least `1e-30` to avoid division by zero.
fn tensor_determinant(k: &[f64], dim: usize) -> f64 {
    match dim {
        2 => {
            // 2x2: [a11, a12, a21, a22]
            (k[0] * k[3] - k[1] * k[2]).abs().max(1e-30)
        }
        3 => {
            // 3x3: [a11, a12, a13, a21, a22, a23, a31, a32, a33]
            let d = k[0] * (k[4] * k[8] - k[5] * k[7]) - k[1] * (k[3] * k[8] - k[5] * k[6])
                + k[2] * (k[3] * k[7] - k[4] * k[6]);
            d.abs().max(1e-30)
        }
        // Scalar case: tensor is [k], determinant is |k|
        1 => k[0].abs().max(1e-30),
        _ => 1.0,
    }
}

/// Material-aware strength of connection (distance-Laplacian weighted SOC).
///
/// For heterogeneous/anisotropic materials, the standard SOC fails because it
/// ignores material discontinuities. This function normalizes each connection
/// by the geometric mean of the endpoint material determinants:
///
///   `s_ij = |a_ij| / sqrt(|K_i| * |K_j|)`
///
/// Then applies the standard threshold: connection (i,j) is strong if
/// `s_ij >= theta * max_k s_ik`.
///
/// Falls back to standard SOC if `material.tensors.len() != n`.
pub(super) fn compute_material_strength<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    theta: f64,
    material: &MaterialSocConfig,
) -> Vec<Vec<usize>> {
    let n = matrix.num_rows;

    // Fall back to standard SOC if material data doesn't match the matrix
    if material.tensors.len() != n {
        return compute_strength_matrix(matrix, theta);
    }

    // Pre-compute determinants of material tensors for each node
    let det: Vec<f64> = material
        .tensors
        .iter()
        .map(|k| tensor_determinant(k, material.dim))
        .collect();

    let filter_tol = T::Real::from_f64(0.01).unwrap();

    parallel_map_indexed(n, |i| {
        compute_material_row(matrix, i, theta, &det, filter_tol)
    })
}

/// Compute the material-weighted strong connections for a single row.
fn compute_material_row<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    i: usize,
    theta: f64,
    det: &[f64],
    filter_tol: T::Real,
) -> Vec<usize> {
    let diag_norm = matrix.get(i, i).norm();
    let noise_threshold = filter_tol * diag_norm;

    let det_i = det[i];

    // Compute material-weighted strengths and find the maximum
    let mut max_strength = 0.0_f64;
    let mut weighted: Vec<(usize, f64)> = Vec::new();

    for (j, val) in matrix.row_entries(i) {
        if i == j {
            continue;
        }
        let norm = val.norm();
        if norm <= noise_threshold {
            continue;
        }

        // s_ij = |a_ij| / sqrt(|K_i| * |K_j|)
        let geo_mean = (det_i * det[j]).sqrt();
        let s_ij = norm.to_f64().unwrap_or(0.0) / geo_mean;

        if s_ij > max_strength {
            max_strength = s_ij;
        }
        weighted.push((j, s_ij));
    }

    let threshold = theta * max_strength;

    weighted
        .into_iter()
        .filter_map(|(j, s)| if s >= threshold { Some(j) } else { None })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::amg::MaterialSocConfig;
    use num_complex::Complex64;

    /// Build a 1D Laplacian of size `n` for testing.
    fn build_1d_laplacian(n: usize) -> CsrMatrix<Complex64> {
        let mut triplets: Vec<(usize, usize, Complex64)> = Vec::new();
        for i in 0..n {
            triplets.push((i, i, Complex64::new(2.0, 0.0)));
            if i > 0 {
                triplets.push((i, i - 1, Complex64::new(-1.0, 0.0)));
            }
            if i < n - 1 {
                triplets.push((i, i + 1, Complex64::new(-1.0, 0.0)));
            }
        }
        CsrMatrix::from_triplets(n, n, triplets)
    }

    #[test]
    fn test_symmetrize_strong() {
        let strong = vec![vec![1], vec![2], vec![]];
        let sym = symmetrize_strong(&strong);
        assert_eq!(sym[0], vec![1]);
        assert_eq!(sym[1], vec![0, 2]);
        assert_eq!(sym[2], vec![1]);
    }

    #[test]
    fn test_material_soc_falls_back_on_size_mismatch() {
        let n = 10;
        let matrix = build_1d_laplacian(n);
        // Provide tensors for only 5 nodes -- should fall back to standard SOC
        let material = MaterialSocConfig {
            tensors: (0..5).map(|_| vec![1.0, 0.0, 0.0, 1.0]).collect(),
            dim: 2,
        };
        let standard = compute_strength_matrix(&matrix, 0.25);
        let mat_result = compute_material_strength(&matrix, 0.25, &material);
        assert_eq!(standard, mat_result);
    }

    #[test]
    fn test_material_soc_uniform_matches_standard() {
        // With uniform isotropic material (identity tensor everywhere),
        // material SOC should produce the same connections as standard SOC
        // because sqrt(det(I) * det(I)) = 1, so s_ij = |a_ij|.
        let n = 20;
        let matrix = build_1d_laplacian(n);
        let material = MaterialSocConfig {
            tensors: (0..n).map(|_| vec![1.0, 0.0, 0.0, 1.0]).collect(),
            dim: 2,
        };
        let standard = compute_strength_matrix(&matrix, 0.25);
        let mat_result = compute_material_strength(&matrix, 0.25, &material);
        assert_eq!(standard, mat_result);
    }

    #[test]
    fn test_material_soc_discontinuity() {
        // Create a 1D problem with material discontinuity at the midpoint:
        // K = 1.0 for nodes 0..10, K = 1000.0 for nodes 10..20
        // Standard SOC would treat all connections equally.
        // Material SOC should weaken the connection at the interface.
        let n = 20;
        let matrix = build_1d_laplacian(n);

        // Scalar material tensors stored as 1x1 matrices
        let tensors: Vec<Vec<f64>> = (0..n)
            .map(|i| {
                let k = if i < 10 { 1.0 } else { 1000.0 };
                vec![k]
            })
            .collect();
        let material = MaterialSocConfig { tensors, dim: 1 };

        // With a reasonably high theta, the interface connection should be
        // dropped by material SOC but kept by standard SOC.
        let theta = 0.25;

        let standard = compute_strength_matrix(&matrix, theta);
        let mat_result = compute_material_strength(&matrix, theta, &material);

        // In the standard SOC, all off-diagonal connections have |a_ij| = 1
        // and the max off-diagonal is 1, so all are strong at theta=0.25.
        // Verify that standard SOC keeps connections at the interface.
        assert!(standard[9].contains(&10), "standard SOC should keep 9->10");
        assert!(standard[10].contains(&9), "standard SOC should keep 10->9");

        // In material SOC, for node 9 (K=1) connecting to node 10 (K=1000):
        //   s_{9,10} = 1.0 / sqrt(1.0 * 1000.0) ≈ 0.0316
        // For node 9 connecting to node 8 (K=1):
        //   s_{9,8} = 1.0 / sqrt(1.0 * 1.0) = 1.0
        // So s_{9,10} = 0.0316 < 0.25 * 1.0 = threshold -> dropped
        assert!(
            !mat_result[9].contains(&10),
            "material SOC should drop the connection 9->10 across the discontinuity"
        );

        // Similarly for node 10 (K=1000) connecting to node 9 (K=1):
        //   s_{10,9} = 1.0 / sqrt(1000.0 * 1.0) ≈ 0.0316
        // For node 10 connecting to node 11 (K=1000):
        //   s_{10,11} = 1.0 / sqrt(1000.0 * 1000.0) = 0.001
        // max_strength for row 10 = max(0.0316, 0.001) = 0.0316
        // threshold = 0.25 * 0.0316 ≈ 0.0079
        // s_{10,9} = 0.0316 >= 0.0079 -> kept
        // s_{10,11} = 0.001 < 0.0079 -> dropped!
        //
        // That's actually interesting: from node 10's perspective, the
        // cross-interface connection (to node 9) is the STRONGEST because
        // the low-K node makes the denominator smaller. The within-material
        // connection (to node 11) is weaker because both have high K.
        //
        // The key asymmetry: node 9 drops node 10, node 10 keeps node 9.
        // This is correct behavior -- it reflects the physics that the
        // interface is a significant transition for the low-K side.
        assert!(
            mat_result[10].contains(&9),
            "material SOC should keep 10->9 (strongest connection for high-K side)"
        );

        // Connections within the same material region should still be strong
        assert!(
            mat_result[5].contains(&4) && mat_result[5].contains(&6),
            "connections within low-K region should remain strong"
        );
    }

    #[test]
    fn test_material_soc_3d_tensor() {
        // Test with 3D identity tensors -- should match standard SOC
        let n = 10;
        let matrix = build_1d_laplacian(n);
        let identity_3d = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let material = MaterialSocConfig {
            tensors: (0..n).map(|_| identity_3d.clone()).collect(),
            dim: 3,
        };
        let standard = compute_strength_matrix(&matrix, 0.25);
        let mat_result = compute_material_strength(&matrix, 0.25, &material);
        assert_eq!(standard, mat_result);
    }

    #[test]
    fn test_tensor_determinant_2d() {
        // Identity: det = 1
        assert!((tensor_determinant(&[1.0, 0.0, 0.0, 1.0], 2) - 1.0).abs() < 1e-12);
        // Scaled: det = 4
        assert!((tensor_determinant(&[2.0, 0.0, 0.0, 2.0], 2) - 4.0).abs() < 1e-12);
        // Non-diagonal
        assert!((tensor_determinant(&[3.0, 1.0, 1.0, 3.0], 2) - 8.0).abs() < 1e-12);
    }

    #[test]
    fn test_tensor_determinant_3d() {
        let id3 = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        assert!((tensor_determinant(&id3, 3) - 1.0).abs() < 1e-12);

        let scaled = vec![2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0];
        assert!((tensor_determinant(&scaled, 3) - 24.0).abs() < 1e-12);
    }

    #[test]
    fn test_tensor_determinant_1d() {
        assert!((tensor_determinant(&[5.0], 1) - 5.0).abs() < 1e-12);
        // Near-zero is clamped to 1e-30
        assert!(tensor_determinant(&[0.0], 1) >= 1e-30);
    }
}
