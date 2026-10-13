//! QA: FEM elasticity — infinite plate with circular hole under uniform tension.
//!
//! Analytical (Kirsch): sigma_theta at r=a, theta=pi/2 equals 3*sigma0.
//! We apply far-field displacement BCs on the outer boundary and check
//! displacement and stress against Kirsch fields on refined logarithmic meshes.
//! Increasing the outer radius separately checks the far-field truncation error.

// Rust guideline compliant 2026-02-21

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::annular_mesh_triangles;
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_solvers::{CsrMatrix, GmresConfig, IluPreconditioner, gmres_preconditioned};

fn apply_dirichlet_csr(
    csr: &mut CsrMatrix<f64>,
    rhs: &mut [f64],
    prescribed_dofs: &[usize],
    prescribed_values: &[f64],
) {
    for (&dof, &value) in prescribed_dofs.iter().zip(prescribed_values) {
        for row in 0..csr.num_rows {
            if row == dof {
                continue;
            }
            for index in csr.row_ptrs[row]..csr.row_ptrs[row + 1] {
                if csr.col_indices[index] == dof {
                    rhs[row] -= csr.values[index] * value;
                    csr.values[index] = 0.0;
                }
            }
        }
        for index in csr.row_ptrs[dof]..csr.row_ptrs[dof + 1] {
            csr.values[index] = if csr.col_indices[index] == dof {
                1.0
            } else {
                0.0
            };
        }
        rhs[dof] = value;
    }
}

/// Compute sigma_xx for a P1 triangle given nodal displacements.
fn element_stress_xx(
    nodes: &[sonium_fem::mesh::Point],
    u: &[f64], // [u1x, u1y, u2x, u2y, u3x, u3y]
    e_mod: f64,
    nu: f64,
) -> f64 {
    let x1 = nodes[0].x;
    let y1 = nodes[0].y;
    let x2 = nodes[1].x;
    let y2 = nodes[1].y;
    let x3 = nodes[2].x;
    let y3 = nodes[2].y;

    // Shape gradients require the signed Jacobian for either node ordering.
    let area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1));
    if area.abs() < 1e-30 {
        return 0.0;
    }

    let inv_2a = 1.0 / (2.0 * area);

    // B-matrix (3 x 6) for plane stress
    let b11 = (y2 - y3) * inv_2a;
    let _b12 = 0.0;
    let b13 = (y3 - y1) * inv_2a;
    let _b14 = 0.0;
    let b15 = (y1 - y2) * inv_2a;
    let _b16 = 0.0;

    let _b21 = 0.0;
    let b22 = (x3 - x2) * inv_2a;
    let _b23 = 0.0;
    let b24 = (x1 - x3) * inv_2a;
    let _b25 = 0.0;
    let b26 = (x2 - x1) * inv_2a;

    let b31 = (x3 - x2) * inv_2a;
    let b32 = (y2 - y3) * inv_2a;
    let b33 = (x1 - x3) * inv_2a;
    let b34 = (y3 - y1) * inv_2a;
    let b35 = (x2 - x1) * inv_2a;
    let b36 = (y1 - y2) * inv_2a;

    let ex = b11 * u[0] + b13 * u[2] + b15 * u[4];
    let ey = b22 * u[1] + b24 * u[3] + b26 * u[5];
    let _gxy = b31 * u[0] + b32 * u[1] + b33 * u[2] + b34 * u[3] + b35 * u[4] + b36 * u[5];

    let factor = e_mod / (1.0 - nu * nu);
    factor * (ex + nu * ey)
}

const CASE_ID: &str = "fem.elasticity-plate-hole-kirsch-convergence.v1";
const HOLE_RADIUS: f64 = 0.5;
const TENSION: f64 = 1.0e6;
const YOUNGS_MODULUS: f64 = 200e9;
const POISSON_RATIO: f64 = 0.3;

// Cartesian displacement follows by integrating the plane-stress Kirsch
// strains. The rigid translation and rotation constants are zero.
fn kirsch_displacement(x: f64, y: f64) -> [f64; 2] {
    let r = x.hypot(y);
    let theta = y.atan2(x);
    let a2 = HOLE_RADIUS.powi(2);
    let a4 = a2 * a2;
    let nu = POISSON_RATIO;
    let scale = TENSION / (2.0 * YOUNGS_MODULUS);
    let ur = scale
        * ((1.0 - nu) * r
            + (1.0 + nu) * a2 / r
            + ((1.0 + nu) * r + 4.0 * a2 / r - (1.0 + nu) * a4 / r.powi(3)) * (2.0 * theta).cos());
    let ut = -scale
        * ((1.0 + nu) * r + 2.0 * (1.0 - nu) * a2 / r + (1.0 + nu) * a4 / r.powi(3))
        * (2.0 * theta).sin();
    [
        ur * theta.cos() - ut * theta.sin(),
        ur * theta.sin() + ut * theta.cos(),
    ]
}

fn kirsch_stress_xx(x: f64, y: f64) -> f64 {
    let r = x.hypot(y);
    let theta = y.atan2(x);
    let ar2 = (HOLE_RADIUS / r).powi(2);
    let ar4 = ar2 * ar2;
    let rr = 0.5 * TENSION * ((1.0 - ar2) + (1.0 - 4.0 * ar2 + 3.0 * ar4) * (2.0 * theta).cos());
    let tt = 0.5 * TENSION * ((1.0 + ar2) - (1.0 + 3.0 * ar4) * (2.0 * theta).cos());
    let rt = -0.5 * TENSION * (1.0 + 2.0 * ar2 - 3.0 * ar4) * (2.0 * theta).sin();
    rr * theta.cos().powi(2) + tt * theta.sin().powi(2) - 2.0 * rt * theta.sin() * theta.cos()
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    displacement_error: f64,
    stress_error: f64,
    concentration_error: f64,
    true_residual: f64,
}

fn solve_case(radial: usize, angular: usize, outer_radius: f64) -> Result<Metrics, String> {
    let a = HOLE_RADIUS;
    let b = outer_radius;
    let mut mesh = annular_mesh_triangles(0.0, 0.0, a, b, radial, angular);
    // Keep the near-hole resolution controlled when the outer radius increases.
    for node in &mut mesh.nodes {
        let r = node.x.hypot(node.y);
        let t = (r - a) / (b - a);
        let mapped_radius = a * (b / a).powf(t);
        node.x *= mapped_radius / r;
        node.y *= mapped_radius / r;
    }
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let n_dofs = space.num_global_dofs(&mesh);
    let material = IsotropicMaterial::new(YOUNGS_MODULUS, POISSON_RATIO);
    let form = ElasticityForm::new_2d(material, PlaneAssumption::PlaneStress);
    let stiffness = assemble_bilinear(&form, &space, &space, &mesh, 1);
    let mut csr = stiffness.to_csr();
    let mut rhs = vec![0.0; n_dofs];
    let mut prescribed = Vec::new();
    let mut prescribed_vals = Vec::new();
    for (i, node) in mesh.nodes.iter().enumerate() {
        if (node.x.hypot(node.y) - b).abs() < b * 1e-10 {
            prescribed.extend([i * 2, i * 2 + 1]);
            prescribed_vals.extend([
                TENSION * node.x / YOUNGS_MODULUS,
                -POISSON_RATIO * TENSION * node.y / YOUNGS_MODULUS,
            ]);
        }
    }
    apply_dirichlet_csr(&mut csr, &mut rhs, &prescribed, &prescribed_vals);
    let precond = IluPreconditioner::from_csr(&csr);
    let result = gmres_preconditioned(
        &csr,
        &precond,
        &rhs,
        &GmresConfig {
            max_iterations: 5000,
            restart: 200,
            tolerance: 1e-11,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    let true_residual = sonium_qa::relative_residual_norm(&csr, &result.x, &rhs);
    if !result.converged || !true_residual.is_finite() || true_residual > 1e-8 {
        return Err(format!(
            "plate-hole solve failed: converged={}, true residual={true_residual:.3e}",
            result.converged
        ));
    }
    // Restrict field norms to the same near-hole annulus r<=2a at every level.
    let mut displacement_error = 0.0;
    let mut displacement_norm = 0.0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        if node.x.hypot(node.y) > 2.0 * a {
            continue;
        }
        let exact = kirsch_displacement(node.x, node.y);
        for j in 0..2 {
            displacement_error += (result.x[2 * i + j] - exact[j]).powi(2);
            displacement_norm += exact[j].powi(2);
        }
    }
    let mut stress_error = 0.0;
    let mut stress_norm = 0.0;
    let mut maximum_stress = 0.0_f64;
    for element in &mesh.elements {
        let indices = element.vertices();
        let points = indices.iter().map(|&i| mesh.nodes[i]).collect::<Vec<_>>();
        let cx = points.iter().map(|p| p.x).sum::<f64>() / 3.0;
        let cy = points.iter().map(|p| p.y).sum::<f64>() / 3.0;
        if cx.hypot(cy) > 2.0 * a {
            continue;
        }
        let displacement = indices
            .iter()
            .flat_map(|&i| [result.x[2 * i], result.x[2 * i + 1]])
            .collect::<Vec<_>>();
        let actual = element_stress_xx(&points, &displacement, YOUNGS_MODULUS, POISSON_RATIO);
        let exact = kirsch_stress_xx(cx, cy);
        let area = 0.5
            * ((points[1].x - points[0].x) * (points[2].y - points[0].y)
                - (points[2].x - points[0].x) * (points[1].y - points[0].y))
                .abs();
        stress_error += area * (actual - exact).powi(2);
        stress_norm += area * exact.powi(2);
        maximum_stress = maximum_stress.max(actual);
    }
    let metrics = Metrics {
        displacement_error: (displacement_error / displacement_norm).sqrt(),
        stress_error: (stress_error / stress_norm).sqrt(),
        concentration_error: (maximum_stress / TENSION - 3.0).abs() / 3.0,
        true_residual,
    };
    println!(
        "plate-hole radial={radial} angular={angular} b/a={:.0}: displacement={:.4e}, stress={:.4e}, Kt error={:.4e}, residual={:.3e}",
        b / a,
        metrics.displacement_error,
        metrics.stress_error,
        metrics.concentration_error,
        true_residual
    );
    Ok(metrics)
}

fn run() -> Result<sonium_qa::QaResult, String> {
    let coarse = solve_case(32, 64, 10.0)?;
    let medium = solve_case(64, 128, 10.0)?;
    let fine = solve_case(128, 256, 10.0)?;
    // log(40)/log(20)=1.231: 160 radial cells retain the near-hole spacing
    // of the 128-cell b/a=20 mesh when the outer radius doubles.
    let expanded = solve_case(160, 256, 20.0)?;
    let displacement_contraction = (medium.displacement_error / coarse.displacement_error)
        .max(fine.displacement_error / medium.displacement_error);
    let stress_contraction =
        (medium.stress_error / coarse.stress_error).max(fine.stress_error / medium.stress_error);
    let domain_displacement_contraction = expanded.displacement_error / fine.displacement_error;
    let domain_stress_contraction = expanded.stress_error / fine.stress_error;
    let maximum_residual = coarse
        .true_residual
        .max(medium.true_residual)
        .max(fine.true_residual)
        .max(expanded.true_residual);
    // Resolved P1 stresses converge at first order; displacements at second
    // order until the independently tested outer-boundary truncation dominates.
    Ok(sonium_qa::QaResult::passed(CASE_ID)
        .requested_backend("cpu").actual_backend("cpu")
        .seed("kirsch:a0.5:sigma1e6:E200e9:nu0.3:log-radial32-64-128:outer-radius10-20")
        .reference("Kirsch plane-stress displacement and Cartesian sigma_xx; far-field uniaxial tension")
        .reference_checksum("analytic:kirsch-plane-stress:v1")
        .formulation("native P1 triangular elasticity; near-hole displacement and area-weighted centroid stress accuracy; mesh and outer-radius refinement")
        .metric("displacement_relative_l2",expanded.displacement_error).maximum("displacement_relative_l2",0.005)
        .metric("stress_relative_l2",expanded.stress_error).maximum("stress_relative_l2",0.02)
        .metric("concentration_relative_error",expanded.concentration_error).maximum("concentration_relative_error",0.01)
        .metric("mesh_displacement_contraction",displacement_contraction).maximum("mesh_displacement_contraction",0.85)
        .metric("mesh_stress_contraction",stress_contraction).maximum("mesh_stress_contraction",0.85)
        .metric("domain_displacement_contraction",domain_displacement_contraction).maximum("domain_displacement_contraction",0.9)
        .metric("domain_stress_contraction",domain_stress_contraction).maximum("domain_stress_contraction",0.95)
        .metric("maximum_true_residual",maximum_residual).maximum("maximum_true_residual",1e-8)
        .metric("numerical_solve_executed",4.0).minimum("numerical_solve_executed",4.0))
}

fn main() {
    match run().and_then(|record| {
        record.release_eligible()?;
        record.to_json_line()
    }) {
        Ok(line) => {
            println!("QA_RESULT: {line}");
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(line) = sonium_qa::QaResult::failed(CASE_ID, &error).to_json_line() {
                println!("QA_RESULT: {line}");
            }
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovered_stress_is_independent_of_triangle_orientation() {
        use sonium_fem::mesh::Point;
        for points in [
            [
                Point::new_2d(0.0, 0.0),
                Point::new_2d(1.0, 0.0),
                Point::new_2d(0.0, 1.0),
            ],
            [
                Point::new_2d(0.0, 0.0),
                Point::new_2d(0.0, 1.0),
                Point::new_2d(1.0, 0.0),
            ],
        ] {
            let displacement = points
                .iter()
                .flat_map(|p| {
                    [
                        TENSION * p.x / YOUNGS_MODULUS,
                        -POISSON_RATIO * TENSION * p.y / YOUNGS_MODULUS,
                    ]
                })
                .collect::<Vec<_>>();
            let stress = element_stress_xx(&points, &displacement, YOUNGS_MODULUS, POISSON_RATIO);
            assert!((stress / TENSION - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn kirsch_reference_recovers_hole_concentration_and_far_field() {
        assert!((kirsch_stress_xx(0.0, HOLE_RADIUS) / TENSION - 3.0).abs() < 1e-12);
        assert!(kirsch_stress_xx(HOLE_RADIUS, 0.0).abs() / TENSION < 1e-12);
        assert!((kirsch_stress_xx(1e6, 1e6) / TENSION - 1.0).abs() < 1e-10);
    }

    #[test]
    fn kirsch_displacement_gradients_recover_stress() {
        let h = 1e-5;
        for (x, y) in [(0.8, 0.3), (-0.7, 0.6), (0.2, -1.1)] {
            let plus_x = kirsch_displacement(x + h, y);
            let minus_x = kirsch_displacement(x - h, y);
            let plus_y = kirsch_displacement(x, y + h);
            let minus_y = kirsch_displacement(x, y - h);
            let ex = (plus_x[0] - minus_x[0]) / (2.0 * h);
            let ey = (plus_y[1] - minus_y[1]) / (2.0 * h);
            let actual = YOUNGS_MODULUS / (1.0 - POISSON_RATIO.powi(2)) * (ex + POISSON_RATIO * ey);
            assert!((actual - kirsch_stress_xx(x, y)).abs() / TENSION < 1e-8);
        }
    }
}
