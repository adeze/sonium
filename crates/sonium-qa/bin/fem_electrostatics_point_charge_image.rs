//! Solved 3D point charge above a grounded plane, with field refinement.

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_dirichlet_values};
use sonium_fem::forms::LaplacianForm;
use sonium_fem::mesh::unit_cube_tetrahedra;
use sonium_fem::physics::electromagnetics::electrostatic::{
    EPSILON_0, ElectrostaticSolvedField, PointChargeSource, assemble_point_charge_source,
    grounded_plane_image_potential,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::{CgConfig, DiagonalPreconditioner, pcg};
use std::f64::consts::PI;

const CASE_ID: &str = "fem.electrostatics-point-charge-image.v1";
const CHARGE: f64 = 1e-9;
const POSITION: [f64; 3] = [0.0, 0.0, 0.5];
// A fixed exclusion ball makes all refinement levels measure the same
// regular part of the solution. Point-source energy is singular in 3D.
const EXCLUSION_RADIUS: f64 = 0.3;

fn reference(point: [f64; 3]) -> (f64, [f64; 3]) {
    let direct: [f64; 3] = std::array::from_fn(|d| point[d] - POSITION[d]);
    let image = [point[0], point[1], point[2] + POSITION[2]];
    let r = direct.iter().map(|v| v * v).sum::<f64>().sqrt();
    let ri = image.iter().map(|v| v * v).sum::<f64>().sqrt();
    let coefficient = CHARGE / (4.0 * PI * EPSILON_0);
    (
        coefficient * (1.0 / r - 1.0 / ri),
        std::array::from_fn(|d| coefficient * (direct[d] / r.powi(3) - image[d] / ri.powi(3))),
    )
}

#[derive(Debug)]
struct ImageMetrics {
    potential_error: f64,
    field_error: f64,
    residual: f64,
    boundary_error: f64,
    charge_error: f64,
    free_dofs: usize,
}

fn solve_image(n: usize) -> Result<ImageMetrics, String> {
    let mut mesh = unit_cube_tetrahedra(n);
    for node in &mut mesh.nodes {
        node.x = 2.0 * node.x - 1.0;
        node.y = 2.0 * node.y - 1.0;
    }
    // Every exterior face gets exact infinite-half-space data. The z=0 face
    // is grounded, and the other faces impose a transparent analytic trace,
    // eliminating outer-domain truncation from this discretization gate.
    for face in &mut mesh.boundaries {
        face.marker = 1;
    }
    let space = H1Space::new(PolynomialDegree::P1);
    let raw = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 2);
    let physical_rhs =
        assemble_point_charge_source(&PointChargeSource::new(POSITION, CHARGE), &mesh, &space);
    let charge_error = (physical_rhs.iter().sum::<f64>() - CHARGE).abs() / CHARGE;
    if !charge_error.is_finite() || charge_error > 1e-12 {
        return Err("3D point charge was not assembled exactly once".into());
    }
    // Divide the full equation by permittivity before solver admission.
    let rhs: Vec<f64> = physical_rhs.iter().map(|v| v / EPSILON_0).collect();
    let unconstrained = raw.to_csr();
    let mut system = FemSystem::new(raw, rhs.clone());
    try_apply_dirichlet_values(
        &mut system,
        &mesh,
        &space,
        1,
        &|x, y, z| reference([x, y, z]).0,
        None,
    )
    .map_err(|e| format!("image boundary data: {e}"))?;
    let boundary = space.boundary_dofs(&mesh, 1);
    let free_dofs = mesh.num_nodes() - boundary.len();
    if free_dofs == 0 {
        return Err("image-charge mesh has no free unknowns".into());
    }
    let csr = system.matrix.to_csr();
    let preconditioner =
        DiagonalPreconditioner::try_from_csr(&csr).map_err(|e| format!("image Jacobi: {e}"))?;
    let result = pcg(
        &csr,
        &preconditioner,
        &system.rhs,
        &CgConfig {
            max_iterations: 3000,
            tolerance: 1e-11,
            print_interval: 0,
        },
    );
    if !result.converged || result.x.iter().any(|v| !v.is_finite()) {
        return Err(format!("image solve did not converge: {:?}", result.status));
    }
    let mut is_boundary = vec![false; mesh.num_nodes()];
    let mut boundary_error = 0.0_f64;
    for &node in &boundary {
        is_boundary[node] = true;
        let p = mesh.nodes[node];
        boundary_error = boundary_error.max((result.x[node] - reference([p.x, p.y, p.z]).0).abs());
    }
    // Independently recompute the physical equation on free DOFs, before BCs.
    let action = unconstrained.matvec(&result.x);
    let numerator: f64 = (0..rhs.len())
        .filter(|&i| !is_boundary[i])
        .map(|i| (action[i] - rhs[i]).powi(2))
        .sum();
    let rhs_norm: f64 = rhs.iter().map(|v| v * v).sum();
    let residual = (numerator / rhs_norm).sqrt();
    let field = ElectrostaticSolvedField::new(&mesh, &space, &result.x, EPSILON_0)
        .map_err(|e| e.to_string())?;
    let mut potential_error_squared = 0.0;
    let mut potential_norm_squared = 0.0;
    let mut field_error_squared = 0.0;
    let mut field_norm_squared = 0.0;
    let mut measured_volume = 0.0;
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, element);
        for point in QuadratureRule::new(mesh.elements[element].element_type, 3).iter() {
            let tab = space.tabulate(&mesh, element, point.xi(), point.eta(), point.zeta());
            let distance = (0..3)
                .map(|d| (tab.physical_coords[d] - POSITION[d]).powi(2))
                .sum::<f64>()
                .sqrt();
            if distance < EXCLUSION_RADIUS {
                continue;
            }
            let (expected, expected_field) = reference(tab.physical_coords);
            let actual: f64 = dofs
                .iter()
                .enumerate()
                .map(|(local, &global)| result.x[global] * tab.value(local))
                .sum();
            let actual_field = field
                .electric_field_at_reference(element, point.xi(), point.eta(), point.zeta())
                .map_err(|e| e.to_string())?;
            let weight = point.weight * tab.det_j;
            potential_error_squared += weight * (actual - expected).powi(2);
            potential_norm_squared += weight * expected.powi(2);
            field_error_squared += weight
                * (0..3)
                    .map(|d| (actual_field[d] - expected_field[d]).powi(2))
                    .sum::<f64>();
            field_norm_squared += weight * expected_field.iter().map(|v| v * v).sum::<f64>();
            measured_volume += weight;
        }
    }
    if measured_volume <= 0.0 {
        return Err("image-charge error norm measured no regular volume".into());
    }
    Ok(ImageMetrics {
        potential_error: (potential_error_squared / potential_norm_squared).sqrt(),
        field_error: (field_error_squared / field_norm_squared).sqrt(),
        residual,
        boundary_error,
        charge_error,
        free_dofs,
    })
}

fn run() -> Result<QaResult, String> {
    let probe = [0.2, 0.0, 0.25];
    let helper_error = (grounded_plane_image_potential(probe, POSITION, CHARGE, EPSILON_0)
        - reference(probe).0)
        .abs();
    let coarse = solve_image(12)?;
    println!("image n=12: {coarse:?}");
    let medium = solve_image(24)?;
    println!("image n=24: {medium:?}");
    let fine = solve_image(48)?;
    println!("image n=48: {fine:?}");
    let potential_rate = (medium.potential_error / fine.potential_error).log2();
    let field_rate = (medium.field_error / fine.field_error).log2();
    if !(coarse.potential_error > medium.potential_error
        && medium.potential_error > fine.potential_error
        && coarse.field_error > medium.field_error
        && medium.field_error > fine.field_error)
    {
        return Err("image-charge potential/field errors do not contract".into());
    }
    for (name, error, bound) in [
        ("potential", fine.potential_error, 0.01),
        ("electric field", fine.field_error, 0.1),
        ("physical residual", fine.residual, 1e-9),
        ("boundary", fine.boundary_error, 1e-9),
        ("source charge", fine.charge_error, 1e-12),
        ("image helper", helper_error, 1e-12),
    ] {
        if !error.is_finite() || error > bound {
            return Err(format!("{name} error {error:.3e} exceeds {bound:.3e}"));
        }
    }
    if !potential_rate.is_finite()
        || potential_rate < 1.0
        || !field_rate.is_finite()
        || field_rate < 0.8
    {
        return Err(
            "image-charge potential/field convergence rates below expected regular-domain bounds"
                .into(),
        );
    }
    Ok(QaResult::passed(CASE_ID).requested_backend("cpu").actual_backend("cpu")
        .metric("coarse_potential_error",coarse.potential_error).metric("medium_potential_error",medium.potential_error)
        .metric("fine_potential_error",fine.potential_error).maximum("fine_potential_error",0.01)
        .metric("coarse_field_error",coarse.field_error).metric("medium_field_error",medium.field_error)
        .metric("fine_field_error",fine.field_error).maximum("fine_field_error",0.1)
        .metric("potential_refinement_rate",potential_rate).minimum("potential_refinement_rate",1.0)
        .metric("field_refinement_rate",field_rate).minimum("field_refinement_rate",0.8)
        .metric("physical_residual",fine.residual).maximum("physical_residual",1e-9)
        .metric("boundary_error",fine.boundary_error).maximum("boundary_error",1e-9)
        .metric("source_charge_error",fine.charge_error).maximum("source_charge_error",1e-12)
        .metric("image_helper_error",helper_error).maximum("image_helper_error",1e-12)
        .metric("free_dofs",fine.free_dofs as f64).minimum("free_dofs",1.0)
        .metric("singular_exclusion_radius",EXCLUSION_RADIUS)
        .reference("independent Coulomb point charge minus mirror charge; potential and electric field")
        .reference_checksum("analytic:grounded-plane-point-source:v1")
        .input_checksum("tetra-half-space-box:n12-n24-n48:q1nC:z0.5:v1")
        .dimension("3D [-1,1]^2 x [0,1], charge inside at (0,0,0.5)")
        .formulation("production point-charge source, P1 Poisson assembly, grounded-plane Dirichlet and analytic exterior traces")
        .tolerance("potential/field quadrature errors outside fixed 0.3m source ball; spatial contraction and true residual"))
}

fn main() {
    let result = match run() {
        Ok(record) => record,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("FAILED: {error}");
            std::process::exit(1);
        }
    }
    if result.status == QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
