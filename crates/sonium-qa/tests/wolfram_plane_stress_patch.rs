//! Literal Sonium reproduction of Wolfram verification case
//! `SolidMechanics-FEM-Stationary-2D-PlaneStress-0001`.
//!
//! Wolfram specifies the eight nodes, three Q1 quadrilaterals, material,
//! roller/fixed constraints, top load, nodal displacement reference, and
//! reaction-force reference. No geometry or mesh approximation is involved.

use sonium_fem::{
    assembly::assemble_bilinear,
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, apply_traction_vector, try_apply_homogeneous_dirichlet_dofs},
    mesh::{BoundaryType, ElementType, Mesh, Point},
    physics::solid::elasticity::{ElasticityForm, IsotropicMaterial, PlaneAssumption},
    spaces::{FunctionSpace, VectorH1Space},
};
use sonium_solvers::SparseLuFactorization;

const YOUNG_MODULUS: f64 = 3.0e6;
const POISSON_RATIO: f64 = 0.2;
const THICKNESS: f64 = 4.0;
const TOP_TRACTION_PER_THICKNESS: f64 = -50.0;

const COORDINATES: [[f64; 2]; 8] = [
    [0.0, 5.0],
    [0.0, 12.0],
    [6.0, 0.0],
    [6.0, 5.0],
    [20.0, 0.0],
    [20.0, 12.0],
    [54.0, 0.0],
    [54.0, 12.0],
];

const QUADS: [[usize; 4]; 3] = [[0, 3, 5, 1], [2, 4, 5, 3], [4, 6, 7, 5]];

const REFERENCE_DISPLACEMENTS: [[f64; 2]; 8] = [
    [0.0, -0.0183155],
    [0.0, -0.0183204],
    [0.00275915, -0.0166486],
    [0.00114552, -0.0164634],
    [0.00305003, -0.0113566],
    [-0.00210128, -0.0116254],
    [0.0, 0.0],
    [0.0, 0.0],
];

fn official_mesh() -> Mesh {
    let mut mesh = Mesh::new(2);
    for [x, y] in COORDINATES {
        mesh.add_node(Point::new_2d(x, y));
    }
    for nodes in QUADS {
        mesh.add_element(ElementType::Quadrilateral, nodes.to_vec());
    }
    mesh.detect_boundaries();
    mesh.set_boundary_condition(BoundaryType::Neumann, 2, |points| {
        points.iter().all(|point| (point.y - 12.0).abs() < 1.0e-12)
    });
    mesh
}

#[test]
fn sonium_reproduces_wolfram_three_quad_plane_stress_patch() {
    let mesh = official_mesh();
    let space = VectorH1Space::new(PolynomialDegree::P1, 2);
    let stiffness = assemble_bilinear(
        &ElasticityForm::new_2d(
            IsotropicMaterial::new(YOUNG_MODULUS, POISSON_RATIO),
            PlaneAssumption::PlaneStress,
        ),
        &space,
        &space,
        &mesh,
        2,
    );
    let original = stiffness.to_csr();
    let mut system = FemSystem::new(stiffness, vec![0.0; space.num_global_dofs(&mesh)]);

    // Wolfram multiplies both the plane-stress stiffness and its boundary
    // load by thickness=4. Solving the per-unit-thickness system is identical;
    // physical forces below are multiplied by THICKNESS for comparison.
    apply_traction_vector(&mut system, &mesh, &space, 2, &|_, _, _| {
        vec![0.0, TOP_TRACTION_PER_THICKNESS]
    });
    let unconstrained_rhs = system.rhs.clone();
    let physical_load_y = THICKNESS * unconstrained_rhs.iter().skip(1).step_by(2).sum::<f64>();
    assert!((physical_load_y + 10_800.0).abs() < 1.0e-9);

    // x=0 is an x-only roller; x=54 fixes both displacement components.
    try_apply_homogeneous_dirichlet_dofs(&mut system, &[0, 2, 12, 13, 14, 15])
        .expect("apply official component-wise constraints");
    let displacement = SparseLuFactorization::factorize(&system.matrix.to_csr())
        .expect("plane-stress patch factorization")
        .solve(&system.rhs)
        .expect("plane-stress patch solve");

    let maximum_error = REFERENCE_DISPLACEMENTS
        .iter()
        .enumerate()
        .flat_map(|(node, reference)| {
            [
                (displacement[2 * node] - reference[0]).abs(),
                (displacement[2 * node + 1] - reference[1]).abs(),
            ]
        })
        .fold(0.0, f64::max);
    assert!(
        maximum_error < 1.0e-7,
        "maximum nodal displacement error {maximum_error:.3e} exceeds Wolfram's 1e-7 gate"
    );

    let internal_force = original.matvec(&displacement);
    // Wolfram reports constrained positions component-major: roller/fixed x
    // reactions first, then the two fixed-face y reactions. Sonium stores
    // vector DOFs node-major, so select them in Wolfram's published order.
    let wolfram_reaction_order = [0, 2, 12, 14, 13, 15];
    let physical_reactions = wolfram_reaction_order
        .map(|dof| THICKNESS * (internal_force[dof] - unconstrained_rhs[dof]));
    let reference_reactions = [-7931.98, 10360.8, -19673.0, 17244.2, 5839.98, 4960.02];
    let maximum_reaction_error = physical_reactions
        .iter()
        .zip(reference_reactions)
        .map(|(actual, reference)| (actual - reference).abs())
        .fold(0.0, f64::max);
    assert!(
        maximum_reaction_error <= 0.1,
        "maximum reaction error {maximum_reaction_error:.6} exceeds Wolfram's 0.1 gate"
    );
    let total_reaction_y = physical_reactions[4] + physical_reactions[5];
    assert!((total_reaction_y + physical_load_y).abs() < 1.0e-9);
}
