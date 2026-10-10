//! QA: tapered 3-D beam vs stepped and tapered closed forms.
//!
//! Covers prismatic exactness (tapered with equal end sections reproduces
//! the straight element), the stepped-cantilever closed form, the linearly
//! tapered cantilever (unit-load integral with cubic I(x)), a Timoshenko
//! cross-check against a fine prismatic reference, and mesh convergence.
//! Taper varies the in-plane width; out-of-plane height stays constant.

// Rust guideline compliant 2026-02-21

use sonium_fem::physics::solid::beam::{BeamElement3D, Section3D, TaperedBeamElement3D};

/// Solve a dense system in place with partial pivoting.
fn dense_solve(matrix: &mut [f64], rhs: &mut [f64], n: usize) {
    for col in 0..n {
        let mut pivot = col;
        let mut best = matrix[col * n + col].abs();
        for row in col + 1..n {
            let candidate = matrix[row * n + col].abs();
            if candidate > best {
                best = candidate;
                pivot = row;
            }
        }
        assert!(best > 1.0e-300, "singular QA system");
        if pivot != col {
            for k in col..n {
                matrix.swap(col * n + k, pivot * n + k);
            }
            rhs.swap(col, pivot);
        }
        let diag = matrix[col * n + col];
        for row in col + 1..n {
            let factor = matrix[row * n + col] / diag;
            matrix[row * n + col] = 0.0;
            for k in col + 1..n {
                matrix[row * n + k] -= factor * matrix[col * n + k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for k in row + 1..n {
            sum -= matrix[row * n + k] * rhs[k];
        }
        rhs[row] = sum / matrix[row * n + row];
    }
}

/// Tip deflection of a tapered cantilever with `n` elements (EB regime).
fn tapered_tip(n: usize) -> f64 {
    let e = 200.0e9;
    let b0 = 0.05;
    let b1 = 0.025;
    let h = 0.05;
    let l = 1.0;
    let dx = l / n as f64;
    let width_at = |i: usize| b0 + (b1 - b0) * i as f64 / n as f64;
    let elements: Vec<TaperedBeamElement3D> = (0..n)
        .map(|i| {
            TaperedBeamElement3D::new(
                [i, i + 1],
                [i as f64 * dx, 0.0, 0.0],
                [(i + 1) as f64 * dx, 0.0, 0.0],
                Section3D::rectangular(e, 0.3, width_at(i), h, 7800.0, false),
                Section3D::rectangular(e, 0.3, width_at(i + 1), h, 7800.0, false),
                [0.0, 1.0, 0.0],
            )
        })
        .collect();
    cantilever_tip_y(&elements, 1000.0)
}

/// Transverse tip deflection of a clamped tapered assembly under tip load.
fn cantilever_tip_y(elements: &[TaperedBeamElement3D], load: f64) -> f64 {
    let n_dofs = (elements.len() + 1) * 6;
    let mut k = vec![0.0; n_dofs * n_dofs];
    for elem in elements {
        let ke = elem.stiffness_global();
        let dofs = elem.global_dofs();
        for (i, &gi) in dofs.iter().enumerate() {
            for (j, &gj) in dofs.iter().enumerate() {
                k[gi * n_dofs + gj] += ke[i * 12 + j];
            }
        }
    }
    let free: Vec<usize> = (6..n_dofs).collect();
    let m = free.len();
    let mut matrix = vec![0.0; m * m];
    let mut rhs = vec![0.0; m];
    for (i, &gi) in free.iter().enumerate() {
        for (j, &gj) in free.iter().enumerate() {
            matrix[i * m + j] = k[gi * n_dofs + gj];
        }
    }
    rhs[m - 5] = load;
    dense_solve(&mut matrix, &mut rhs, m);
    rhs[m - 5]
}

/// Hand closed form for the linear width taper, b(x) = b0 (1 - a x/L).
///
/// Integrates P (L-x)^2/(E I(x)) with I(x) = h b(x)^3/12 in closed form:
/// F(t) = ln t + 2c/t - c^2/(2t^2) with c = 1 - a.
fn tapered_closed_form(l: f64, p: f64, e: f64, h: f64, b0: f64, alpha: f64) -> f64 {
    let c = 1.0 - alpha;
    let f = |t: f64| t.ln() + 2.0 * c / t - c * c / (2.0 * t * t);
    let i0 = h * b0.powi(3) / 12.0;
    p * l.powi(3) / (e * i0 * alpha.powi(3)) * (f(1.0) - f(c))
}

fn test_prismatic_exactness() {
    let section = Section3D::rectangular(200.0e9, 0.3, 0.05, 0.05, 7800.0, true);
    let tapered = TaperedBeamElement3D::new(
        [0, 1],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        section,
        section,
        [0.0, 1.0, 0.0],
    );
    let straight = BeamElement3D::new(
        [0, 1],
        [0.0, 0.0, 0.0],
        [2.0, 0.0, 0.0],
        section,
        [0.0, 1.0, 0.0],
    );
    let kt = tapered.stiffness_local();
    let ks = straight.stiffness_local();
    let scale = ks.iter().fold(0.0_f64, |m, &v| m.max(v.abs()));
    let worst = kt
        .iter()
        .zip(ks.iter())
        .map(|(a, b)| (a - b).abs() / scale)
        .fold(0.0_f64, f64::max);
    println!("prismatic exactness: worst entry rel={worst:.2e}");
    assert!(worst < 1.0e-12, "prismatic mismatch {worst:.3e}");
}

fn test_stepped_reference() {
    // Two-span stepped cantilever: tip segment (L2, I2) on root segment
    // (L1, I1); superposition gives the tip deflection in closed form.
    let e = 200.0e9;
    let l1 = 0.6;
    let l2 = 0.4;
    let root = Section3D::rectangular(e, 0.3, 0.05, 0.05, 7800.0, false);
    let tip = Section3D::rectangular(e, 0.3, 0.025, 0.05, 7800.0, false);
    let e1 = TaperedBeamElement3D::new(
        [0, 1],
        [0.0, 0.0, 0.0],
        [l1, 0.0, 0.0],
        root,
        root,
        [0.0, 1.0, 0.0],
    );
    let e2 = TaperedBeamElement3D::new(
        [1, 2],
        [l1, 0.0, 0.0],
        [l1 + l2, 0.0, 0.0],
        tip,
        tip,
        [0.0, 1.0, 0.0],
    );
    let fem = cantilever_tip_y(&[e1, e2], 1000.0);
    // Superposition with the junction moment M = P l2: the tip segment
    // bends as a cantilever from the junction while the junction itself
    // deflects and rotates under shear P plus moment M. Reduces to
    // P L^3/(3EI) for I1 = I2 (binomial identity).
    let p = 1000.0;
    let expected = p * l2.powi(3) / (3.0 * e * tip.iz) + p * l1.powi(3) / (3.0 * e * root.iz)
        + p * l1 * l1 * l2 / (e * root.iz)
        + p * l1 * l2 * l2 / (e * root.iz);
    let error = (fem - expected).abs() / expected;
    println!("stepped: FEM={fem:.12e} exact={expected:.12e} rel={error:.2e}");
    assert!(error < 1.0e-9, "stepped error {error:.3e}");
}

fn test_linear_taper() {
    let expected = tapered_closed_form(1.0, 1000.0, 200.0e9, 0.05, 0.05, 0.5);
    let fem = tapered_tip(16);
    let error = (fem - expected).abs() / expected;
    println!("linear taper: FEM={fem:.12e} exact={expected:.12e} rel={error:.2e}");
    assert!(error < 2.0e-3, "taper error {error:.3e}");
    println!("GOLDEN beam_tapered_tip fem={fem:.17e} hand={expected:.17e}");
}

fn test_timoshenko_cross_check() {
    // Tapered Timoshenko assembly vs a fine prismatic midpoint reference.
    let e = 200.0e9;
    let b0 = 0.05;
    let b1 = 0.025;
    let h = 0.05;
    let l = 1.0;
    let n = 128;
    let dx = l / n as f64;
    let width_at = |x: f64| b0 + (b1 - b0) * x / l;
    let tapered: Vec<TaperedBeamElement3D> = (0..n)
        .map(|i| {
            TaperedBeamElement3D::new(
                [i, i + 1],
                [i as f64 * dx, 0.0, 0.0],
                [(i + 1) as f64 * dx, 0.0, 0.0],
                Section3D::rectangular(e, 0.3, width_at(i as f64 * dx), h, 7800.0, true),
                Section3D::rectangular(e, 0.3, width_at((i + 1) as f64 * dx), h, 7800.0, true),
                [0.0, 1.0, 0.0],
            )
        })
        .collect();
    let fem = cantilever_tip_y(&tapered, 1000.0);
    // Reference: 256 prismatic segments with midpoint sections.
    let m = 256;
    let seg = l / m as f64;
    let mut k = vec![0.0; (m + 1) * 6 * ((m + 1) * 6)];
    let n_dofs = (m + 1) * 6;
    for i in 0..m {
        let mid = (i as f64 + 0.5) * seg;
        let section = Section3D::rectangular(e, 0.3, width_at(mid), h, 7800.0, true);
        let elem = BeamElement3D::new(
            [i, i + 1],
            [i as f64 * seg, 0.0, 0.0],
            [(i + 1) as f64 * seg, 0.0, 0.0],
            section,
            [0.0, 1.0, 0.0],
        );
        let ke = elem.stiffness_global();
        let dofs = elem.global_dofs();
        for (a, &ga) in dofs.iter().enumerate() {
            for (b, &gb) in dofs.iter().enumerate() {
                k[ga * n_dofs + gb] += ke[a * 12 + b];
            }
        }
    }
    let free: Vec<usize> = (6..n_dofs).collect();
    let f = free.len();
    let mut matrix = vec![0.0; f * f];
    let mut rhs = vec![0.0; f];
    for (i, &gi) in free.iter().enumerate() {
        for (j, &gj) in free.iter().enumerate() {
            matrix[i * f + j] = k[gi * n_dofs + gj];
        }
    }
    rhs[f - 5] = 1000.0;
    dense_solve(&mut matrix, &mut rhs, f);
    let reference = rhs[f - 5];
    let error = (fem - reference).abs() / reference;
    println!("timoshenko x-check: tapered={fem:.12e} ref={reference:.12e} rel={error:.2e}");
    assert!(error < 1.0e-4, "timoshenko error {error:.3e}");
}

fn test_convergence() {
    let expected = tapered_closed_form(1.0, 1000.0, 200.0e9, 0.05, 0.05, 0.5);
    let err8 = (tapered_tip(8) - expected).abs() / expected;
    let err32 = (tapered_tip(32) - expected).abs() / expected;
    println!("convergence: err8={err8:.2e} err32={err32:.2e} ratio={:.1}", err8 / err32);
    assert!(err32 < err8 / 8.0, "no quadratic convergence");
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        test_prismatic_exactness();
        test_stepped_reference();
        test_linear_taper();
        test_timoshenko_cross_check();
        test_convergence();
    });
    match result {
        Ok(()) => println!("SUCCESS"),
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
