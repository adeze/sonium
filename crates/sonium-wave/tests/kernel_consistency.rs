//! Cross-consistency tests for the 3D Helmholtz kernels.
//!
//! These assert exact algebraic relations between kernels that must hold for
//! any non-coincident pair, independent of series expansions or fixtures:
//! the adjoint (source-normal) derivative is the negated direct
//! (field-normal) derivative when both normals coincide, and every checked
//! `try_*` twin rejects coincidence while the bare kernel reports the
//! documented infinite sentinel.

use sonium_wave::Point;
use sonium_wave::special::helmholtz::{
    all_kernels_3d, greens_function_3d, try_all_kernels_3d, try_greens_function_3d,
};

#[test]
fn adjoint_derivative_is_negated_direct_derivative_for_shared_normal() {
    let source = Point::new_3d(0.0, 0.0, 0.0);
    let field = Point::new_3d(0.4, -0.3, 0.5);
    let normal = [0.0, 0.0, 1.0];
    let k = 2.0 * std::f64::consts::PI / 0.5;

    let (g, dg_dny, dg_dnx, _d2g) = all_kernels_3d(&source, &field, &normal, &normal, k);
    assert!(g.is_finite());
    // ∂G/∂n_x = −∂G/∂n_y when n_x == n_y (see `all_kernels_3d`).
    assert!(
        (dg_dnx + dg_dny).norm() < 1e-12 * dg_dny.norm().max(1e-30),
        "adjoint consistency violated: dg_dnx={dg_dnx:?}, dg_dny={dg_dny:?}"
    );

    // Same relation at a second, closer pair (stronger near-field terms).
    let near = Point::new_3d(1e-6, 2e-6, -1e-6);
    let (g2, dg_dny2, dg_dnx2, d2g2) = all_kernels_3d(&source, &near, &normal, &normal, k);
    assert!(g2.is_finite() && d2g2.is_finite());
    assert!((dg_dnx2 + dg_dny2).norm() < 1e-9 * dg_dny2.norm().max(1e-30));
}

#[test]
fn coincident_pair_is_sentinel_or_error_never_silent_nan() {
    let p = Point::new_3d(1.0, 2.0, 3.0);
    let k = 5.0;

    // Bare kernel: documented infinite sentinel at coincidence.
    let g = greens_function_3d(0.0, k);
    assert!(g.re.is_infinite() && g.im == 0.0);

    // Checked twins fail closed.
    assert!(try_greens_function_3d(0.0, k).is_err());
    assert!(try_greens_function_3d(f64::NAN, k).is_err());
    assert!(try_greens_function_3d(1.0, f64::NAN).is_err());
    let n = [0.0, 0.0, 1.0];
    assert!(try_all_kernels_3d(&p, &p, &n, &n, k).is_err());
}
