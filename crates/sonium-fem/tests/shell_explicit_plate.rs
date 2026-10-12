//! Phase 7 lane 7B: explicit shell dynamics on a clamped plate.
//!
//! Small-mesh (4x4) durable cover for `shell::explicit`: relaxation
//! converges, matches the dense static solve, lands near the
//! Timoshenko closed form, and the lumped mass audit is exact. The
//! `qa-fem-shell-explicit` leg repeats this at gate mesh (10x10).

// Rust guideline compliant 2026-02-21

use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::shell::ShellSection;
use sonium_fem::physics::solid::shell::explicit::{
    DynamicRelaxation, ShellExplicitModel, lumped_masses,
};
use sonium_fem::physics::solid::shell::{ShellElement, assemble_shell_load};

const GRID: usize = 4;

fn plate(section: ShellSection) -> (Vec<ShellElement>, usize) {
    let step = 1.0 / GRID as f64;
    let point = |i: usize, j: usize| Point::new_3d(i as f64 * step, j as f64 * step, 0.0);
    let id = |i: usize, j: usize| j * (GRID + 1) + i;
    let mut elements = Vec::new();
    for j in 0..GRID {
        for i in 0..GRID {
            elements.push(ShellElement::from_points(
                [id(i, j), id(i + 1, j), id(i + 1, j + 1)],
                &point(i, j),
                &point(i + 1, j),
                &point(i + 1, j + 1),
                section,
            ));
            elements.push(ShellElement::from_points(
                [id(i, j), id(i + 1, j + 1), id(i, j + 1)],
                &point(i, j),
                &point(i + 1, j + 1),
                &point(i, j + 1),
                section,
            ));
        }
    }
    (elements, (GRID + 1) * (GRID + 1))
}

#[test]
fn shell_explicit_plate_relaxes_to_static_answer() {
    let section = ShellSection::new(200.0e9, 0.3, 0.02, 7800.0);
    let (elements, n_nodes) = plate(section);
    let n_dofs = n_nodes * 6;

    // Mass audit: translational lumps recover rho A t exactly.
    let masses = lumped_masses(&elements, n_dofs).unwrap();
    let mut transl = 0.0;
    for node in 0..n_nodes {
        transl += masses[node * 6] + masses[node * 6 + 1] + masses[node * 6 + 2];
    }
    assert!((transl - 3.0 * 7800.0 * 0.02).abs() / (3.0 * 7800.0 * 0.02) < 1e-12);

    let mut model = ShellExplicitModel::elastic(elements.clone(), n_dofs).unwrap();
    for j in 0..=GRID {
        for i in 0..=GRID {
            if i == 0 || j == 0 || i == GRID || j == GRID {
                model.fix_node(j * (GRID + 1) + i);
            }
        }
    }
    let load = assemble_shell_load(&elements, n_dofs, |_| -1000.0);
    let critical = model.elastic_critical_dt(200).unwrap();
    assert!(critical > 0.0);
    let driver = DynamicRelaxation::new(0.5 * critical, 1.0e-3, 60_000).unwrap();
    let out = driver.relax(&mut model, &load).unwrap();
    assert!(
        out.converged,
        "relaxation stalled: residual {:.3e} after {} steps",
        out.residual_norm, out.steps
    );

    let answer = model.static_solve(&load).unwrap();
    let center = (GRID / 2) * (GRID + 1) + GRID / 2;
    let w_relax = out.displacement[center * 6 + 2];
    let w_static = answer[center * 6 + 2];
    assert!(
        (w_relax - w_static).abs() < 0.03 * w_static.abs(),
        "relax {w_relax:.6e} vs static {w_static:.6e}"
    );

    // Coarse-mesh closed-form check (Timoshenko Table 8, nu = 0.3).
    let rigidity = 200.0e9 * 0.02f64.powi(3) / (12.0 * (1.0 - 0.09));
    let w_analytic = -0.00126 * 1000.0 / rigidity;
    assert!(
        (w_static - w_analytic).abs() < 0.15 * w_analytic.abs(),
        "static {w_static:.6e} vs analytic {w_analytic:.6e}"
    );
    assert!(w_static < 0.0, "pressure must deflect downward");
}
