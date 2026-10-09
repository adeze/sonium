//! Literal Sonium reproduction of Wolfram verification case
//! `SolidMechanics-FEM-Stationary-3D-0001`.
//!
//! A 50 x 10 x 20 mm steel block is fixed at x=0 and pulled by a total
//! 10 kN force at x=50 mm. Wolfram checks 0.012 mm elongation and a uniform
//! 50 N/mm^2 axial/von-Mises stress away from the clamp.

use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_homogeneous_dirichlet, apply_traction_vector},
    mesh::{Mesh, box_mesh_tetrahedra},
    physics::solid::elasticity::{ElasticityForm, IsotropicMaterial},
    spaces::{FunctionSpace, VectorH1Space},
};
use sonium_solvers::SparseLuFactorization;

const LENGTH_MM: f64 = 50.0;
const WIDTH_MM: f64 = 10.0;
const HEIGHT_MM: f64 = 20.0;
const YOUNG_MODULUS_N_MM2: f64 = 210_000.0;
const POISSON_RATIO: f64 = 0.3;
const TOTAL_FORCE_N: f64 = 10_000.0;
const TRACTION_N_MM2: f64 = TOTAL_FORCE_N / (WIDTH_MM * HEIGHT_MM);
const REFERENCE_ELONGATION_MM: f64 = 0.012;
const REFERENCE_STRESS_N_MM2: f64 = 50.0;

#[derive(Debug, Clone, Copy)]
struct Stress {
    xx: f64,
    yy: f64,
    zz: f64,
    xy: f64,
    yz: f64,
    zx: f64,
}

impl Stress {
    fn von_mises(self) -> f64 {
        (0.5 * ((self.xx - self.yy).powi(2)
            + (self.yy - self.zz).powi(2)
            + (self.zz - self.xx).powi(2))
            + 3.0 * (self.xy.powi(2) + self.yz.powi(2) + self.zx.powi(2)))
        .sqrt()
    }
}

fn inverse_3x3(matrix: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let [[a, b, c], [d, e, f], [g, h, i]] = matrix;
    let determinant = a * (e * i - f * h) - b * (d * i - f * g) + c * (d * h - e * g);
    assert!(determinant.abs() > 1.0e-14, "nondegenerate tetrahedron");
    let scale = 1.0 / determinant;
    [
        [
            (e * i - f * h) * scale,
            (c * h - b * i) * scale,
            (b * f - c * e) * scale,
        ],
        [
            (f * g - d * i) * scale,
            (a * i - c * g) * scale,
            (c * d - a * f) * scale,
        ],
        [
            (d * h - e * g) * scale,
            (b * g - a * h) * scale,
            (a * e - b * d) * scale,
        ],
    ]
}

fn element_stress(mesh: &Mesh, nodes: &[usize], displacement: &[f64]) -> Stress {
    assert_eq!(nodes.len(), 4);
    let points = [
        &mesh.nodes[nodes[0]],
        &mesh.nodes[nodes[1]],
        &mesh.nodes[nodes[2]],
        &mesh.nodes[nodes[3]],
    ];
    let jacobian = [
        [
            points[1].x - points[0].x,
            points[2].x - points[0].x,
            points[3].x - points[0].x,
        ],
        [
            points[1].y - points[0].y,
            points[2].y - points[0].y,
            points[3].y - points[0].y,
        ],
        [
            points[1].z - points[0].z,
            points[2].z - points[0].z,
            points[3].z - points[0].z,
        ],
    ];
    let inverse = inverse_3x3(jacobian);
    let mut gradients = [[0.0; 3]; 4];
    gradients[1] = inverse[0];
    gradients[2] = inverse[1];
    gradients[3] = inverse[2];
    gradients[0] = [
        -gradients[1][0] - gradients[2][0] - gradients[3][0],
        -gradients[1][1] - gradients[2][1] - gradients[3][1],
        -gradients[1][2] - gradients[2][2] - gradients[3][2],
    ];

    let mut displacement_gradient = [[0.0; 3]; 3];
    for (local, &node) in nodes.iter().enumerate() {
        for component in 0..3 {
            for coordinate in 0..3 {
                displacement_gradient[component][coordinate] +=
                    displacement[3 * node + component] * gradients[local][coordinate];
            }
        }
    }
    let strain_xx = displacement_gradient[0][0];
    let strain_yy = displacement_gradient[1][1];
    let strain_zz = displacement_gradient[2][2];
    let strain_xy = 0.5 * (displacement_gradient[0][1] + displacement_gradient[1][0]);
    let strain_yz = 0.5 * (displacement_gradient[1][2] + displacement_gradient[2][1]);
    let strain_zx = 0.5 * (displacement_gradient[2][0] + displacement_gradient[0][2]);
    let material = IsotropicMaterial::new(YOUNG_MODULUS_N_MM2, POISSON_RATIO);
    let lambda = material.lambda();
    let mu = material.mu();
    let trace = strain_xx + strain_yy + strain_zz;
    Stress {
        xx: lambda * trace + 2.0 * mu * strain_xx,
        yy: lambda * trace + 2.0 * mu * strain_yy,
        zz: lambda * trace + 2.0 * mu * strain_zz,
        xy: 2.0 * mu * strain_xy,
        yz: 2.0 * mu * strain_yz,
        zx: 2.0 * mu * strain_zx,
    }
}

#[test]
fn sonium_reproduces_wolfram_uniform_3d_tension() {
    // Every tetrahedral edge is below Wolfram's requested 10 mm maximum.
    let mut mesh = box_mesh_tetrahedra(0.0, LENGTH_MM, 0.0, WIDTH_MM, 0.0, HEIGHT_MM, 20, 4, 8);
    for boundary in &mut mesh.boundaries {
        if boundary
            .nodes
            .iter()
            .all(|&node| mesh.nodes[node].x.abs() < 1.0e-12)
        {
            boundary.marker = 1;
        } else if boundary
            .nodes
            .iter()
            .all(|&node| (mesh.nodes[node].x - LENGTH_MM).abs() < 1.0e-12)
        {
            boundary.marker = 2;
        } else {
            boundary.marker = 3;
        }
    }
    let space = VectorH1Space::new(PolynomialDegree::P1, 3);
    let stiffness = assemble_bilinear(
        &ElasticityForm::new_3d(IsotropicMaterial::new(YOUNG_MODULUS_N_MM2, POISSON_RATIO)),
        &space,
        &space,
        &mesh,
        2,
    );
    let original = stiffness.to_csr();
    let mut system = FemSystem::new(stiffness, vec![0.0; space.num_global_dofs(&mesh)]);
    apply_traction_vector(&mut system, &mesh, &space, 2, &|_, _, _| {
        vec![TRACTION_N_MM2, 0.0, 0.0]
    });
    let unconstrained_rhs = system.rhs.clone();
    let applied_force = unconstrained_rhs.iter().step_by(3).sum::<f64>();
    assert!((applied_force - TOTAL_FORCE_N).abs() < 1.0e-9);
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 1);
    let displacement = SparseLuFactorization::factorize(&system.matrix.to_csr())
        .expect("3D tension factorization")
        .solve(&system.rhs)
        .expect("3D tension solve");

    let maximum_elongation = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, point)| (point.x - LENGTH_MM).abs() < 1.0e-12)
        .map(|(node, _)| displacement[3 * node])
        .fold(f64::NEG_INFINITY, f64::max);
    let elongation_error =
        ((maximum_elongation - REFERENCE_ELONGATION_MM) / REFERENCE_ELONGATION_MM).abs();
    assert!(
        elongation_error < 0.02,
        "Sonium elongation {maximum_elongation:.9} mm versus Wolfram {REFERENCE_ELONGATION_MM:.9} mm, relative error {elongation_error:.3e}"
    );

    let internal_force = original.matvec(&displacement);
    let clamp_reaction = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(_, point)| point.x.abs() < 1.0e-12)
        .map(|(node, _)| internal_force[3 * node] - unconstrained_rhs[3 * node])
        .sum::<f64>();
    assert!(
        (clamp_reaction + TOTAL_FORCE_N).abs() / TOTAL_FORCE_N < 1.0e-10,
        "reaction {clamp_reaction:.9} N must balance {TOTAL_FORCE_N:.9} N"
    );

    let mut maximum_axial_error: f64 = 0.0;
    let mut maximum_von_mises_error: f64 = 0.0;
    for element in &mesh.elements {
        let nodes = element.vertices();
        if nodes.len() != 4 {
            continue;
        }
        let centroid_x = nodes.iter().map(|&node| mesh.nodes[node].x).sum::<f64>() / 4.0;
        if centroid_x < 30.0 {
            continue;
        }
        let stress = element_stress(&mesh, nodes, &displacement);
        maximum_axial_error = maximum_axial_error.max((stress.xx - REFERENCE_STRESS_N_MM2).abs());
        maximum_von_mises_error =
            maximum_von_mises_error.max((stress.von_mises() - REFERENCE_STRESS_N_MM2).abs());
    }
    assert!(
        maximum_axial_error < 0.04,
        "far-field sigma_xx maximum absolute error {maximum_axial_error:.6} N/mm^2"
    );
    assert!(
        maximum_von_mises_error < 0.05,
        "far-field von-Mises maximum absolute error {maximum_von_mises_error:.6} N/mm^2"
    );
}
