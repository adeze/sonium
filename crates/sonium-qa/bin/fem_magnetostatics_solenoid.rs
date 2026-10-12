//! Solved infinite-solenoid vector potential and magnetic-field refinement.

use sonium_fem::assembly::{assemble_bilinear, assemble_linear};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, try_apply_dirichlet_values};
use sonium_fem::mesh::rectangular_mesh_triangles;
use sonium_fem::physics::electromagnetics::magnetostatics::{
    AxisymmetricAThetaForm, AzimuthalCurrentDensityForm, MU_0,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::direct::SparseLuFactorization;
use std::f64::consts::PI;

const CASE_ID: &str = "fem.magnetostatics-solenoid.v1";
const INNER_RADIUS: f64 = 0.1;
const OUTER_RADIUS: f64 = 0.2;
const DOMAIN_RADIUS: f64 = 0.3;
const LENGTH: f64 = 0.12;
const AMPERE_TURNS_PER_LENGTH: f64 = 2000.0;
const CURRENT_DENSITY: f64 = AMPERE_TURNS_PER_LENGTH / (OUTER_RADIUS - INNER_RADIUS);

// Ampere's law for an infinitely long winding of finite radial thickness.
// The exterior trace is exact and the axial faces have zero natural flux.
fn reference(r: f64) -> (f64, f64) {
    let a = INNER_RADIUS;
    let b = OUTER_RADIUS;
    let bz = MU_0 * CURRENT_DENSITY * (b - r.max(a)).max(0.0);
    let integral = if r < a {
        (b - a) * r * r / 2.0
    } else {
        let t = r.min(b);
        (b - a) * a * a / 2.0 + b * (t * t - a * a) / 2.0 - (t.powi(3) - a.powi(3)) / 3.0
    };
    let potential = if r == 0.0 {
        0.0
    } else {
        MU_0 * CURRENT_DENSITY * integral / r
    };
    (potential, bz)
}

#[derive(Debug)]
struct Metrics {
    potential_error: f64,
    field_error: f64,
    bore_flux_error: f64,
    energy_error: f64,
    residual: f64,
    boundary_error: f64,
    free_dofs: usize,
}

fn solve(n: usize) -> Result<Metrics, String> {
    let mut mesh = rectangular_mesh_triangles(0.0, DOMAIN_RADIUS, 0.0, LENGTH, n, n / 6);
    // Axis and outer radius are essential; axial faces retain the natural
    // infinite-length condition dA/dz=0, so interior field is not prescribed.
    for face in &mut mesh.boundaries {
        face.marker = if face.nodes.iter().all(|&i| mesh.nodes[i].x.abs() < 1e-12) {
            2
        } else if face
            .nodes
            .iter()
            .all(|&i| (mesh.nodes[i].x - DOMAIN_RADIUS).abs() < 1e-12)
        {
            3
        } else {
            1
        };
    }
    let space = H1Space::new(PolynomialDegree::P1);
    let mut matrix =
        assemble_bilinear(&AxisymmetricAThetaForm::new(MU_0), &space, &space, &mesh, 4);
    let source = AzimuthalCurrentDensityForm::new(|r, _| {
        if r > INNER_RADIUS && r < OUTER_RADIUS {
            CURRENT_DENSITY
        } else {
            0.0
        }
    });
    let mut rhs = assemble_linear(&source, &space, &mesh, 4);
    if !rhs.iter().any(|v| v.abs() > 0.0) {
        return Err("solenoid current source is empty".into());
    }
    // Normalize the complete physical equation by reluctivity before BC rows.
    for value in &mut matrix.values {
        *value *= MU_0;
    }
    for value in &mut rhs {
        *value *= MU_0;
    }
    let raw = matrix.to_csr();
    let physical_rhs = rhs.clone();
    let mut system = FemSystem::new(matrix, rhs);
    for marker in [2, 3] {
        try_apply_dirichlet_values(
            &mut system,
            &mesh,
            &space,
            marker,
            &|r, _, _| reference(r).0,
            None,
        )
        .map_err(|e| e.to_string())?;
    }
    let mut constrained = vec![false; mesh.num_nodes()];
    for marker in [2, 3] {
        for dof in space.boundary_dofs(&mesh, marker) {
            constrained[dof] = true;
        }
    }
    let factor =
        SparseLuFactorization::factorize(&system.matrix.to_csr()).map_err(|e| e.to_string())?;
    let values = factor.solve(&system.rhs).map_err(|e| e.to_string())?;
    if values.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite solenoid potential".into());
    }
    let boundary_error = mesh
        .nodes
        .iter()
        .enumerate()
        .filter(|(i, _)| constrained[*i])
        .map(|(i, p)| (values[i] - reference(p.x).0).abs())
        .fold(0.0_f64, f64::max);
    let action = raw.matvec(&values);
    let residual = action
        .iter()
        .zip(&physical_rhs)
        .enumerate()
        .filter(|(i, _)| !constrained[*i])
        .map(|(_, (&a, &b))| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / physical_rhs.iter().map(|v| v * v).sum::<f64>().sqrt();
    let (
        mut potential_error,
        mut potential_norm,
        mut field_error,
        mut field_norm,
        mut bore_flux,
        mut energy,
    ) = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, element);
        for qp in QuadratureRule::new(mesh.elements[element].element_type, 4).iter() {
            let tab = space.tabulate(&mesh, element, qp.xi(), qp.eta(), 0.0);
            let r = tab.physical_coords[0];
            let potential = dofs
                .iter()
                .enumerate()
                .map(|(i, &d)| values[d] * tab.value(i))
                .sum::<f64>();
            let dr = dofs
                .iter()
                .enumerate()
                .map(|(i, &d)| values[d] * tab.grad(i, 0))
                .sum::<f64>();
            let dz = dofs
                .iter()
                .enumerate()
                .map(|(i, &d)| values[d] * tab.grad(i, 1))
                .sum::<f64>();
            let bz = dr + potential / r;
            let br = -dz;
            let (exact_a, exact_b) = reference(r);
            let weight = 2.0 * PI * r * qp.weight * tab.det_j;
            potential_error += weight * (potential - exact_a).powi(2);
            potential_norm += weight * exact_a.powi(2);
            field_error += weight * ((bz - exact_b).powi(2) + br * br);
            field_norm += weight * exact_b.powi(2);
            if r < INNER_RADIUS {
                bore_flux += weight * bz / LENGTH;
            }
            energy += weight * (bz * bz + br * br) / (2.0 * MU_0);
        }
    }
    let a = INNER_RADIUS;
    let b = OUTER_RADIUS;
    let primitive = |r: f64| b * b * r * r / 2.0 - 2.0 * b * r.powi(3) / 3.0 + r.powi(4) / 4.0;
    let exact_energy = PI
        * LENGTH
        * MU_0
        * CURRENT_DENSITY.powi(2)
        * ((b - a).powi(2) * a * a / 2.0 + primitive(b) - primitive(a));
    let exact_flux = PI * a * a * MU_0 * AMPERE_TURNS_PER_LENGTH;
    Ok(Metrics {
        potential_error: (potential_error / potential_norm).sqrt(),
        field_error: (field_error / field_norm).sqrt(),
        bore_flux_error: (bore_flux / exact_flux - 1.0).abs(),
        energy_error: (energy / exact_energy - 1.0).abs(),
        residual,
        boundary_error,
        free_dofs: constrained.iter().filter(|v| !**v).count(),
    })
}

fn run() -> Result<QaResult, String> {
    let coarse = solve(12)?;
    println!("solenoid n=12: {coarse:?}");
    let medium = solve(24)?;
    println!("solenoid n=24: {medium:?}");
    let fine = solve(48)?;
    println!("solenoid n=48: {fine:?}");
    if !(coarse.potential_error > medium.potential_error
        && medium.potential_error > fine.potential_error
        && coarse.field_error > medium.field_error
        && medium.field_error > fine.field_error)
    {
        return Err("solenoid potential/field errors do not contract".into());
    }
    let potential_rate = (medium.potential_error / fine.potential_error).log2();
    let field_rate = (medium.field_error / fine.field_error).log2();
    let mut record = QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu");
    for (key, value, bound) in [
        ("potential_error", fine.potential_error, 0.002),
        ("magnetic_field_error", fine.field_error, 0.02),
        ("bore_flux_error", fine.bore_flux_error, 0.002),
        ("magnetic_energy_error", fine.energy_error, 0.002),
        ("physical_residual", fine.residual, 1e-10),
        ("boundary_error", fine.boundary_error, 1e-12),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{key} {value:.3e} exceeds {bound:.3e}"));
        }
        record = record.metric(key, value).maximum(key, bound);
    }
    if !potential_rate.is_finite()
        || potential_rate < 1.5
        || !field_rate.is_finite()
        || field_rate < 0.8
    {
        return Err("solenoid refinement rate below P1 accuracy bounds".into());
    }
    Ok(record.metric("coarse_field_error",coarse.field_error).metric("medium_field_error",medium.field_error)
        .metric("potential_refinement_rate",potential_rate).minimum("potential_refinement_rate",1.5)
        .metric("field_refinement_rate",field_rate).minimum("field_refinement_rate",0.8)
        .metric("free_dofs",fine.free_dofs as f64).minimum("free_dofs",1.0)
        .reference("Ampere law for infinite solenoid with finite radial winding thickness; Bz(r), enclosed flux, magnetic energy")
        .reference_checksum("analytic:infinite-thick-winding-solenoid:v1")
        .input_checksum("axisymmetric:r0-0.3:z0-0.12:winding0.1-0.2:2000Am-1:n12-24-48:v1")
        .seed("axisymmetric-solenoid:p1:n12-24-48")
        .dimension("axisymmetric r-z winding and air, infinite-length natural axial boundaries")
        .formulation("production AxisymmetricAThetaForm and AzimuthalCurrentDensityForm, solved Atheta with recovered Br/Bz")
        .tolerance("weighted field/potential quadrature norms and refinement, independent flux/energy and physical residual"))
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
    if result.status != QaStatus::Passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
