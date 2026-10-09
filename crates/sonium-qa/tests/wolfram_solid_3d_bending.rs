//! Literal Sonium reproduction of Wolfram verification case
//! `SolidMechanics-FEM-Stationary-3D-0003`.
//!
//! A 200 x 40 x 40 mm steel cantilever is fixed at x=0 and loaded by a
//! total -10 kN z force on x=200. Wolfram probes |w(200,20,0)| and checks
//! it against 25/42 mm with a 2% tolerance. The Sonium port uses quadratic
//! tetrahedra and a consistent quadratic traction trace.

use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_homogeneous_dirichlet, apply_traction_vector},
    mesh::box_mesh_tetrahedra,
    physics::solid::elasticity::{ElasticityForm, IsotropicMaterial},
    spaces::{FunctionSpace, VectorH1Space},
};
use sonium_solvers::SparseLuFactorization;

const LENGTH_MM: f64 = 200.0;
const WIDTH_MM: f64 = 40.0;
const HEIGHT_MM: f64 = 40.0;
const YOUNG_MODULUS_N_MM2: f64 = 210_000.0;
const POISSON_RATIO: f64 = 0.3;
const TOTAL_FORCE_N: f64 = 10_000.0;
const TRACTION_Z_N_MM2: f64 = -TOTAL_FORCE_N / (WIDTH_MM * HEIGHT_MM);
const REFERENCE_DEFLECTION_MM: f64 = 25.0 / 42.0;

struct BendingResult {
    deflection: f64,
    applied_force_z: f64,
    clamp_reaction_z: f64,
}

fn solve(degree: PolynomialDegree) -> BendingResult {
    // The even y subdivision puts the official (200,20,0) probe on a vertex.
    let mut mesh = box_mesh_tetrahedra(0.0, LENGTH_MM, 0.0, WIDTH_MM, 0.0, HEIGHT_MM, 10, 2, 2);

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

    let space = VectorH1Space::new(degree, 3);
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
        vec![0.0, 0.0, TRACTION_Z_N_MM2]
    });
    let unconstrained_rhs = system.rhs.clone();

    let applied_force_z = unconstrained_rhs.iter().skip(2).step_by(3).sum::<f64>();
    apply_homogeneous_dirichlet(&mut system, &mesh, &space, 1);
    let displacement = SparseLuFactorization::factorize(&system.matrix.to_csr())
        .expect("3D bending factorization")
        .solve(&system.rhs)
        .expect("3D bending solve");
    let residual = system
        .residual_report(&displacement, 1.0e-10, 1.0e-10)
        .expect("3D bending residual");
    assert!(
        residual.converged,
        "3D bending relative residual {}",
        residual.relative_residual
    );

    let probe_node = mesh
        .nodes
        .iter()
        .position(|point| {
            (point.x - LENGTH_MM).abs() < 1.0e-12
                && (point.y - WIDTH_MM / 2.0).abs() < 1.0e-12
                && point.z.abs() < 1.0e-12
        })
        .expect("official Wolfram probe must be a mesh node");
    let deflection = displacement[3 * probe_node + 2].abs();
    let internal_force = original.matvec(&displacement);
    let clamp_reaction_z = space
        .boundary_dofs(&mesh, 1)
        .into_iter()
        .filter(|dof| dof % 3 == 2)
        .map(|dof| internal_force[dof] - unconstrained_rhs[dof])
        .sum::<f64>();
    BendingResult {
        deflection,
        applied_force_z,
        clamp_reaction_z,
    }
}

#[test]
fn sonium_matches_wolfram_end_loaded_3d_bending() {
    let linear = solve(PolynomialDegree::P1);
    let quadratic = solve(PolynomialDegree::P2);
    assert!(
        (quadratic.applied_force_z + TOTAL_FORCE_N).abs() < 1.0e-9,
        "integrated end load {:.9} N versus expected -{TOTAL_FORCE_N:.9} N",
        quadratic.applied_force_z
    );
    assert!(
        (quadratic.clamp_reaction_z - TOTAL_FORCE_N).abs() / TOTAL_FORCE_N < 1.0e-10,
        "clamp reaction {:.9} N does not balance {TOTAL_FORCE_N:.9} N",
        quadratic.clamp_reaction_z
    );
    let linear_error =
        ((linear.deflection - REFERENCE_DEFLECTION_MM) / REFERENCE_DEFLECTION_MM).abs();
    let relative_error =
        ((quadratic.deflection - REFERENCE_DEFLECTION_MM) / REFERENCE_DEFLECTION_MM).abs();
    assert!(
        relative_error < 0.02,
        "P2 Sonium {:.9} mm versus Wolfram {REFERENCE_DEFLECTION_MM:.9} mm, relative error {relative_error:.3e}",
        quadratic.deflection
    );
    assert!(
        relative_error < linear_error,
        "P2 order refinement must improve the published-reference error: P1 {linear_error:.3e}, P2 {relative_error:.3e}"
    );
}
