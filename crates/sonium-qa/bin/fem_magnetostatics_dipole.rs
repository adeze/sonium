//! Solved 3D interior magnetic dipole, with regular-domain field refinement.

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_dirichlet_values};
use sonium_fem::mesh::unit_cube_tetrahedra;
use sonium_fem::physics::electromagnetics::magnetostatics::{
    PermeabilityForm, PointDipoleSource, assemble_point_dipole_source, compute_b_field,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::{CgConfig, DiagonalPreconditioner, pcg};
use std::f64::consts::PI;

const CASE_ID: &str = "fem.magnetostatics-dipole.v1";
const MU: f64 = 4.0e-7 * PI;
const POSITION: [f64; 3] = [0.0; 3];
// All three components are active; this catches ignored out-of-plane loads.
const MOMENT: [f64; 3] = [0.4, -0.7, 1.0];
// Fixed exclusion: a point dipole has singular total field energy in 3D.
const EXCLUSION_RADIUS: f64 = 0.5;

fn reference(point: [f64; 3]) -> (f64, [f64; 3]) {
    let offset: [f64; 3] = std::array::from_fn(|d| point[d] - POSITION[d]);
    let radius = offset.iter().map(|v| v * v).sum::<f64>().sqrt();
    let dot = (0..3).map(|d| MOMENT[d] * offset[d]).sum::<f64>();
    (
        dot / (4.0 * PI * radius.powi(3)),
        std::array::from_fn(|d| {
            MU / (4.0 * PI) * (3.0 * offset[d] * dot / radius.powi(5) - MOMENT[d] / radius.powi(3))
        }),
    )
}

#[derive(Debug)]
struct DipoleMetrics {
    potential_error: f64,
    field_error: f64,
    residual: f64,
    boundary_error: f64,
    moment_error: f64,
    free_dofs: usize,
}

fn solve_dipole(n: usize) -> Result<DipoleMetrics, String> {
    let mut mesh = unit_cube_tetrahedra(n);
    for node in &mut mesh.nodes {
        node.x = 2.0 * node.x - 1.0;
        node.y = 2.0 * node.y - 1.0;
        node.z = 2.0 * node.z - 1.0;
    }
    // Exact exterior dipole traces remove finite-box truncation error.
    // The singular dipole is inside, with every interior potential unknown solved.
    for face in &mut mesh.boundaries {
        face.marker = 1;
    }
    let space = H1Space::new(PolynomialDegree::P1);
    let raw = assemble_bilinear(&PermeabilityForm::from_mu(1.0), &space, &space, &mesh, 2);
    // Divide the full permeability-weighted system by mu. The production
    // source moment is integrated B_r = mu * physical magnetic moment.
    let physical_rhs = assemble_point_dipole_source(
        &PointDipoleSource::new(POSITION, MOMENT.map(|m| MU * m)),
        &mesh,
        &space,
    );
    let rhs: Vec<f64> = physical_rhs.iter().map(|v| v / MU).collect();
    let mut moment_error = rhs.iter().sum::<f64>().abs();
    for (axis, &expected) in MOMENT.iter().enumerate() {
        let measured: f64 = rhs
            .iter()
            .zip(&mesh.nodes)
            .map(|(v, p)| v * [p.x, p.y, p.z][axis])
            .sum();
        moment_error = moment_error.max((measured - expected).abs());
    }
    if !moment_error.is_finite() || moment_error > 1e-12 {
        return Err("3D signed dipole moment was not assembled exactly once".into());
    }
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
    .map_err(|e| format!("dipole boundary data: {e}"))?;
    let boundary = space.boundary_dofs(&mesh, 1);
    let free_dofs = mesh.num_nodes() - boundary.len();
    if free_dofs == 0 {
        return Err("image-charge mesh has no free unknowns".into());
    }
    let csr = system.matrix.to_csr();
    let preconditioner =
        DiagonalPreconditioner::try_from_csr(&csr).map_err(|e| format!("dipole Jacobi: {e}"))?;
    let result = pcg(
        &csr,
        &preconditioner,
        &system.rhs,
        &CgConfig {
            max_iterations: 3000,
            tolerance: 1e-10,
            print_interval: 0,
        },
    );
    if !result.converged || result.x.iter().any(|v| !v.is_finite()) {
        return Err(format!(
            "dipole solve did not converge: {:?}",
            result.status
        ));
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
            let values: Vec<f64> = dofs.iter().map(|&i| result.x[i]).collect();
            let actual_field = compute_b_field(&values, &tab, MU, &[0.0; 3]);
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
        return Err("dipole error norm measured no regular volume".into());
    }
    Ok(DipoleMetrics {
        potential_error: (potential_error_squared / potential_norm_squared).sqrt(),
        field_error: (field_error_squared / field_norm_squared).sqrt(),
        residual,
        boundary_error,
        moment_error,
        free_dofs,
    })
}

fn run() -> Result<QaResult, String> {
    let coarse = solve_dipole(12)?;
    println!("dipole n=12: {coarse:?}");
    let medium = solve_dipole(24)?;
    println!("dipole n=24: {medium:?}");
    let fine = solve_dipole(48)?;
    println!("dipole n=48: {fine:?}");
    let potential_rate = (medium.potential_error / fine.potential_error).log2();
    let field_rate = (medium.field_error / fine.field_error).log2();
    if !(coarse.potential_error > medium.potential_error
        && medium.potential_error > fine.potential_error
        && coarse.field_error > medium.field_error
        && medium.field_error > fine.field_error)
    {
        return Err("dipole potential/field errors do not contract".into());
    }
    for (name, error, bound) in [
        ("potential", fine.potential_error, 0.01),
        ("magnetic field", fine.field_error, 0.1),
        ("physical residual", fine.residual, 1e-9),
        ("boundary", fine.boundary_error, 1e-9),
        ("source moment", fine.moment_error, 1e-12),
    ] {
        if !error.is_finite() || error > bound {
            return Err(format!("{name} error {error:.3e} exceeds {bound:.3e}"));
        }
    }
    if !potential_rate.is_finite()
        || potential_rate < 1.8
        || !field_rate.is_finite()
        || field_rate < 0.8
    {
        return Err(
            "dipole potential/field convergence rates below expected regular-domain bounds".into(),
        );
    }
    Ok(QaResult::passed(CASE_ID).requested_backend("cpu").actual_backend("cpu")
        .metric("coarse_potential_error",coarse.potential_error).metric("medium_potential_error",medium.potential_error)
        .metric("fine_potential_error",fine.potential_error).maximum("fine_potential_error",0.01)
        .metric("coarse_field_error",coarse.field_error).metric("medium_field_error",medium.field_error)
        .metric("fine_field_error",fine.field_error).maximum("fine_field_error",0.1)
        .metric("potential_refinement_rate",potential_rate).minimum("potential_refinement_rate",1.8)
        .metric("field_refinement_rate",field_rate).minimum("field_refinement_rate",0.8)
        .metric("physical_residual",fine.residual).maximum("physical_residual",1e-9)
        .metric("boundary_error",fine.boundary_error).maximum("boundary_error",1e-9)
        .metric("source_moment_error",fine.moment_error).maximum("source_moment_error",1e-12)
        .metric("free_dofs",fine.free_dofs as f64).minimum("free_dofs",1.0)
        .metric("singular_exclusion_radius",EXCLUSION_RADIUS)
        .reference("independent 3D point magnetic dipole: phi=m.r/(4pi r^3), B=mu*(3r(m.r)/r^5-m/r^3)/(4pi)")
        .reference_checksum("analytic:interior-3d-magnetic-dipole:v1")
        .input_checksum("tetra-cube:n12-n24-n48:moment0.4,-0.7,1:mu0:v1")
        .dimension("3D [-1,1]^3; singular point dipole at origin; fixed regular region r>=0.5")
        .formulation("production tetrahedral point-dipole load, permeability Laplacian, P1 scalar potential and recovered B; analytic exterior traces")
        .tolerance("regular-domain potential <=1%, magnetic field <=10%, contraction and potential/field rates>=1.8/0.8; physical residual<=1e-9"))
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
