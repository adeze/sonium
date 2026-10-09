//! Discrete Kirchhoff Triangle (DKT) plate bending element
//!
//! 9 DOFs per element: `[w1, θx1, θy1, w2, θx2, θy2, w3, θx3, θy3]`
//!
//! The DKT element imposes the Kirchhoff constraint (zero transverse shear)
//! discretely at edge midpoints, producing a thin-plate element free of
//! shear locking.
//!
//! Reference: Batoz, Bathe & Ho (1980), "A study of three-node triangular
//! plate bending elements", IJNME 15.

/// Compute the 9×9 DKT bending stiffness matrix in local coordinates.
///
/// `coords`: `[(x1,y1), (x2,y2), (x3,y3)]` in local frame.
/// `db`: bending constitutive matrix (3×3, row-major) = `D_b`.
/// `area`: triangle area.
///
/// Returns the 9×9 stiffness matrix (row-major).
pub fn dkt_bending_stiffness(coords: &[(f64, f64); 3], db: &[f64; 9], area: f64) -> [f64; 81] {
    // Edge vectors and lengths
    let (x1, y1) = coords[0];
    let (x2, y2) = coords[1];
    let (x3, y3) = coords[2];

    let x12 = x1 - x2;
    let x23 = x2 - x3;
    let x31 = x3 - x1;
    let y12 = y1 - y2;
    let y23 = y2 - y3;
    let y31 = y3 - y1;

    let l12_sq = x12 * x12 + y12 * y12;
    let l23_sq = x23 * x23 + y23 * y23;
    let l31_sq = x31 * x31 + y31 * y31;

    // DKT coefficients (Batoz et al. 1980, Eqs. 20-25)
    let a4 = -x23 / l23_sq;
    let a5 = -x31 / l31_sq;
    let a6 = -x12 / l12_sq;

    let b4 = 0.75 * x23 * y23 / l23_sq;
    let b5 = 0.75 * x31 * y31 / l31_sq;
    let b6 = 0.75 * x12 * y12 / l12_sq;

    let c4 = (0.25 * x23 * x23 - 0.5 * y23 * y23) / l23_sq;
    let c5 = (0.25 * x31 * x31 - 0.5 * y31 * y31) / l31_sq;
    let c6 = (0.25 * x12 * x12 - 0.5 * y12 * y12) / l12_sq;

    let d4 = -y23 / l23_sq;
    let d5 = -y31 / l31_sq;
    let d6 = -y12 / l12_sq;

    let e4 = (0.25 * y23 * y23 - 0.5 * x23 * x23) / l23_sq;
    let e5 = (0.25 * y31 * y31 - 0.5 * x31 * x31) / l31_sq;
    let e6 = (0.25 * y12 * y12 - 0.5 * x12 * x12) / l12_sq;

    // 3-point Gauss quadrature on the triangle
    // Points: (1/6, 1/6), (2/3, 1/6), (1/6, 2/3)
    let gauss_pts: [(f64, f64, f64); 3] = [
        (1.0 / 6.0, 1.0 / 6.0, 1.0 / 3.0),
        (2.0 / 3.0, 1.0 / 6.0, 1.0 / 3.0),
        (1.0 / 6.0, 2.0 / 3.0, 1.0 / 3.0),
    ];

    let mut k = [0.0; 81];

    for &(xi, eta, w) in &gauss_pts {
        let bb = dkt_b_matrix(
            xi, eta, a4, a5, a6, b4, b5, b6, c4, c5, c6, d4, d5, d6, e4, e5, e6, y12, y23, y31,
            x12, x23, x31, area,
        );

        // K += w * A * Bb^T * Db * Bb
        let mut db_bb = [0.0; 27]; // Db * Bb (3×9)
        for i in 0..3 {
            for j in 0..9 {
                for p in 0..3 {
                    db_bb[i * 9 + j] += db[i * 3 + p] * bb[p * 9 + j];
                }
            }
        }

        for i in 0..9 {
            for j in 0..9 {
                let mut val = 0.0;
                for p in 0..3 {
                    val += bb[p * 9 + i] * db_bb[p * 9 + j];
                }
                k[i * 9 + j] += w * area * val;
            }
        }
    }

    k
}

/// Compute the 3×9 curvature-displacement (Bb) matrix at area coordinates (xi, eta).
///
/// The curvature vector is `[κ_xx, κ_yy, 2κ_xy]`.
/// The DOF vector is `[w1, θx1, θy1, w2, θx2, θy2, w3, θx3, θy3]`.
#[allow(clippy::too_many_arguments)]
fn dkt_b_matrix(
    xi: f64,
    eta: f64,
    a4: f64,
    a5: f64,
    a6: f64,
    b4: f64,
    b5: f64,
    b6: f64,
    c4: f64,
    c5: f64,
    c6: f64,
    d4: f64,
    d5: f64,
    d6: f64,
    e4: f64,
    e5: f64,
    e6: f64,
    y12: f64,
    _y23: f64,
    y31: f64,
    x12: f64,
    _x23: f64,
    x31: f64,
    area: f64,
) -> [f64; 27] {
    // Natural derivatives of the 6-node quadratic shapes at (xi, eta),
    // order [N1, N2, N3, N4(mid 2-3), N5(mid 3-1), N6(mid 1-2)] with
    // (xi, eta) = (L2, L3) and zeta = L1.
    let zeta = 1.0 - xi - eta;
    let dndxi = [
        1.0 - 4.0 * zeta,
        4.0 * xi - 1.0,
        0.0,
        4.0 * eta,
        -4.0 * eta,
        4.0 * (zeta - xi),
    ];
    let dndeta = [
        1.0 - 4.0 * zeta,
        0.0,
        4.0 * eta - 1.0,
        4.0 * xi,
        4.0 * (zeta - eta),
        -4.0 * xi,
    ];
    // Rotation interpolants Hx, Hy (Batoz et al. 1980, Eqs. 21-25) evaluated
    // on a shape vector N. H is linear in N, so passing dN/dxi (resp.
    // dN/deta) yields the natural derivatives directly. DOF order per node
    // triple: [w, θx, θy] with right-hand-rule rotations; sides 4/5/6 sit
    // opposite nodes 1/2/3, so node i couples the two adjacent sides.
    let eval_h = |n: &[f64; 6]| -> ([f64; 9], [f64; 9]) {
        let hx = [
            1.5 * (a6 * n[5] - a5 * n[4]),
            b5 * n[4] + b6 * n[5],
            n[0] - c5 * n[4] - c6 * n[5],
            1.5 * (a4 * n[3] - a6 * n[5]),
            b6 * n[5] + b4 * n[3],
            n[1] - c6 * n[5] - c4 * n[3],
            1.5 * (a5 * n[4] - a4 * n[3]),
            b4 * n[3] + b5 * n[4],
            n[2] - c4 * n[3] - c5 * n[4],
        ];
        let hy = [
            1.5 * (d6 * n[5] - d5 * n[4]),
            -n[0] + e5 * n[4] + e6 * n[5],
            -(b5 * n[4] + b6 * n[5]),
            1.5 * (d4 * n[3] - d6 * n[5]),
            -n[1] + e6 * n[5] + e4 * n[3],
            -(b6 * n[5] + b4 * n[3]),
            1.5 * (d5 * n[4] - d4 * n[3]),
            -n[2] + e4 * n[3] + e5 * n[4],
            -(b4 * n[3] + b5 * n[4]),
        ];
        (hx, hy)
    };
    let (hx_xi, hy_xi) = eval_h(&dndxi);
    let (hx_eta, hy_eta) = eval_h(&dndeta);
    // Physical derivatives with (xi, eta) = (L2, L3):
    // d/dx = (b2 d/dxi + b3 d/deta)/2A, d/dy = (c2 d/dxi + c3 d/deta)/2A
    // with b2 = y31, b3 = y12, c2 = -x31, c3 = -x12.
    let inv_2a = 1.0 / (2.0 * area);
    let mut bb = [0.0; 27]; // 3 rows × 9 cols
    for j in 0..9 {
        // kappa_x = d(beta_x)/dx
        bb[j] = inv_2a * (y31 * hx_xi[j] + y12 * hx_eta[j]);
        // kappa_y = d(beta_y)/dy
        bb[9 + j] = inv_2a * (-x31 * hy_xi[j] - x12 * hy_eta[j]);
        // 2*kappa_xy = d(beta_x)/dy + d(beta_y)/dx
        bb[18 + j] =
            inv_2a * (-x31 * hx_xi[j] - x12 * hx_eta[j] + y31 * hy_xi[j] + y12 * hy_eta[j]);
    }
    bb
}

/// Compute the 3×9 DKT curvature B-matrix at a given point (xi, eta) in area coordinates.
///
/// Public wrapper for external use (e.g. elastoplastic shell through-thickness integration).
/// Curvature vector: `[κ_xx, κ_yy, 2κ_xy]`, DOFs: `[w1,θx1,θy1, w2,θx2,θy2, w3,θx3,θy3]`.
pub fn dkt_b_matrix_at(coords: &[(f64, f64); 3], area: f64, xi: f64, eta: f64) -> [f64; 27] {
    let (x1, y1) = coords[0];
    let (x2, y2) = coords[1];
    let (x3, y3) = coords[2];

    let x12 = x1 - x2;
    let x23 = x2 - x3;
    let x31 = x3 - x1;
    let y12 = y1 - y2;
    let y23 = y2 - y3;
    let y31 = y3 - y1;

    let l12_sq = x12 * x12 + y12 * y12;
    let l23_sq = x23 * x23 + y23 * y23;
    let l31_sq = x31 * x31 + y31 * y31;

    let a4 = -x23 / l23_sq;
    let a5 = -x31 / l31_sq;
    let a6 = -x12 / l12_sq;
    let b4 = 0.75 * x23 * y23 / l23_sq;
    let b5 = 0.75 * x31 * y31 / l31_sq;
    let b6 = 0.75 * x12 * y12 / l12_sq;
    let c4 = (0.25 * x23 * x23 - 0.5 * y23 * y23) / l23_sq;
    let c5 = (0.25 * x31 * x31 - 0.5 * y31 * y31) / l31_sq;
    let c6 = (0.25 * x12 * x12 - 0.5 * y12 * y12) / l12_sq;
    let d4 = -y23 / l23_sq;
    let d5 = -y31 / l31_sq;
    let d6 = -y12 / l12_sq;
    let e4 = (0.25 * y23 * y23 - 0.5 * x23 * x23) / l23_sq;
    let e5 = (0.25 * y31 * y31 - 0.5 * x31 * x31) / l31_sq;
    let e6 = (0.25 * y12 * y12 - 0.5 * x12 * x12) / l12_sq;

    dkt_b_matrix(
        xi, eta, a4, a5, a6, b4, b5, b6, c4, c5, c6, d4, d5, d6, e4, e5, e6, y12, y23, y31, x12,
        x23, x31, area,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dkt_stiffness_symmetry() {
        let coords = [(0.0_f64, 0.0), (1.0, 0.0), (0.5, 0.8)];
        let area = (0.5
            * ((coords[1].0 - coords[0].0) * (coords[2].1 - coords[0].1)
                - (coords[2].0 - coords[0].0) * (coords[1].1 - coords[0].1)))
            .abs();

        // Unit bending constitutive matrix for testing
        let db = [1.0, 0.3, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.35];
        let k = dkt_bending_stiffness(&coords, &db, area);

        for i in 0..9 {
            for j in i + 1..9 {
                let diff = (k[i * 9 + j] - k[j * 9 + i]).abs();
                let scale = k[i * 9 + i].abs().max(1e-10);
                assert!(
                    diff / scale < 1e-10,
                    "DKT stiffness not symmetric at ({i},{j}): {} vs {}",
                    k[i * 9 + j],
                    k[j * 9 + i]
                );
            }
        }
    }

    #[test]
    fn test_dkt_stiffness_positive_diagonal() {
        let coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let area = 0.5;
        let db = [1.0, 0.3, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.35];
        let k = dkt_bending_stiffness(&coords, &db, area);

        for i in 0..9 {
            assert!(
                k[i * 9 + i] > 0.0,
                "DKT diagonal [{i}] = {} should be positive",
                k[i * 9 + i]
            );
        }
    }

    #[test]
    fn test_dkt_constant_curvature_patch() {
        // Kirchhoff patch fields on a non-right triangle (right-hand-rule
        // rotations: θx = ∂w/∂y, θy = −∂w/∂x). Batoz's sign convention is
        // sagging-negative (κ = −∂²w), so a unit imposed curvature must
        // reproduce exactly −1 in its slot and 0 elsewhere, at every point.
        // This pins the H-table/chain-rule transcription end to end.
        let coords = [(0.0f64, 0.0), (0.25, 0.0), (0.25, 0.3)];
        let area = 0.5 * 0.25 * 0.3;
        let cases: [([f64; 9], [f64; 3]); 3] = [
            // κxx = −1 from w = x²/2.
            (
                [0.0, 0.0, 0.0, 0.03125, 0.0, -0.25, 0.03125, 0.0, -0.25],
                [-1.0, 0.0, 0.0],
            ),
            // κyy = −1 from w = y²/2.
            (
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.045, 0.3, 0.0],
                [0.0, -1.0, 0.0],
            ),
            // 2κxy = −2 from w = xy.
            (
                [0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.075, 0.25, -0.3],
                [0.0, 0.0, -2.0],
            ),
        ];
        for points in [(1.0 / 3.0, 1.0 / 3.0), (0.6, 0.2), (0.1, 0.7)] {
            let bb = dkt_b_matrix_at(&coords, area, points.0, points.1);
            for (dofs, want) in &cases {
                let mut got = [0.0; 3];
                for row in 0..3 {
                    for column in 0..9 {
                        got[row] += bb[row * 9 + column] * dofs[column];
                    }
                }
                for row in 0..3 {
                    assert!(
                        (got[row] - want[row]).abs() < 1e-9,
                        "patch {want:?} at ({}, {}): got {got:?}",
                        points.0,
                        points.1
                    );
                }
            }
        }
    }

    #[test]
    fn test_dkt_rigid_modes_have_zero_curvature() {
        let coords = [(0.0f64, 0.0), (0.25, 0.0), (0.25, 0.3)];
        let area = 0.5 * 0.25 * 0.3;
        // Rigid translation w = 1 and rigid rotations about x and y.
        let modes = [
            [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.3, 1.0, 0.0],
            [0.0, 0.0, 1.0, -0.25, 0.0, 1.0, -0.25, 0.0, 1.0],
        ];
        let bb = dkt_b_matrix_at(&coords, area, 1.0 / 3.0, 1.0 / 3.0);
        for dofs in &modes {
            for row in 0..3 {
                let mut value = 0.0;
                for column in 0..9 {
                    value += bb[row * 9 + column] * dofs[column];
                }
                assert!(
                    value.abs() < 1e-9,
                    "rigid mode {dofs:?} gives curvature {value} in row {row}"
                );
            }
        }
    }

    #[test]
    fn test_dkt_stiffness_scales_with_constitutive() {
        let coords = [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0)];
        let area = 0.5;
        let db1 = [1.0, 0.3, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0, 0.35];
        let db2 = [2.0, 0.6, 0.0, 0.6, 2.0, 0.0, 0.0, 0.0, 0.70];

        let k1 = dkt_bending_stiffness(&coords, &db1, area);
        let k2 = dkt_bending_stiffness(&coords, &db2, area);

        for i in 0..81 {
            if k1[i].abs() > 1e-15 {
                let ratio = k2[i] / k1[i];
                assert!(
                    (ratio - 2.0).abs() < 1e-10,
                    "DKT scaling: k2/k1 = {ratio} at index {i}"
                );
            }
        }
    }
}
