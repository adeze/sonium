//! Numerical conductor slabs using the production complex H(curl) operator.

use num_complex::Complex64;
use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::mesh::topology::MeshTopology;
use sonium_fem::physics::electromagnetics::eddy_current::{ConductorMaterial, EddyCurrentForm};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;
#[path = "hcurl_solve.rs"]
mod hcurl_solve;
use std::collections::BTreeSet;
use std::f64::consts::PI;

#[derive(Debug)]
struct SlabMetrics {
    field_error: f64,
    residual: f64,
    impedance_error: f64,
    skin_depth_error: f64,
    phase_error: f64,
    free_dofs: usize,
}

fn solve_slab(n: usize, sheet: bool) -> Result<SlabMetrics, String> {
    let material = ConductorMaterial::copper();
    let frequency = if sheet { 1.0e6 } else { 1.0e3 };
    let omega = 2.0 * PI * frequency;
    let mu = material.permeability();
    // Independent reference derived from q^2 = i omega mu sigma.
    let delta = (2.0 / (omega * mu * material.conductivity)).sqrt();
    let q = Complex64::new(1.0, 1.0) / delta;
    let length = if sheet { 0.5 * delta } else { 6.0 * delta };
    let width = 0.25 * delta;
    let exact = |x: f64| {
        if sheet {
            (q * (length - x)).cosh() / (q * length).cosh()
        } else {
            (-q * x).exp()
        }
    };
    let mesh = rectangular_mesh_triangles(0.0, length, 0.0, width, n, n / 8);
    let space = HcurlSpace::new();
    let operator = assemble_bilinear(
        &EddyCurrentForm::new(&material, frequency),
        &space,
        &space,
        &mesh,
        4,
    )
    .to_csr();
    let solved = hcurl_solve::solve_transverse(
        &mesh,
        &operator,
        if sheet { &[1, 3, 4] } else { &[1, 2, 3, 4] },
        exact,
    )?;
    let solution = solved.values;
    let residual = solved.residual;
    let free_dofs = solved.free_dofs;
    let topology = MeshTopology::build(&mesh);
    let prescribed: BTreeSet<usize> = [1, 2, 3, 4]
        .iter()
        .flat_map(|&marker| space.boundary_dofs(&mesh, marker))
        .collect();
    let mut error_squared = 0.0;
    let mut norm_squared = 0.0;
    let mut integrated_a = Complex64::new(0.0, 0.0);
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, element);
        let rule = QuadratureRule::new(mesh.elements[element].element_type, 5);
        for point in rule.iter() {
            let tab = space.tabulate(&mesh, element, point.xi(), point.eta(), point.zeta());
            let mut actual = [Complex64::new(0.0, 0.0); 2];
            for (local, &global) in dofs.iter().enumerate() {
                for (component, value) in actual.iter_mut().enumerate() {
                    *value += solution[global] * tab.values[2 * local + component];
                }
            }
            let expected = exact(tab.physical_coords[0]);
            let weight = point.weight * tab.det_j;
            error_squared += weight * (actual[0].norm_sqr() + (actual[1] - expected).norm_sqr());
            norm_squared += weight * expected.norm_sqr();
            integrated_a += weight * actual[1] / width;
        }
    }
    let field_error = (error_squared / norm_squared).sqrt();
    let expected_integral = if sheet {
        (q * length).tanh() / q
    } else {
        (1.0 - (-q * length).exp()) / q
    };
    // Surface E divided by integrated sheet current: Z = 1/(sigma integral A).
    let actual_z = 1.0 / (material.conductivity * integrated_a);
    let expected_z = 1.0 / (material.conductivity * expected_integral);
    let impedance_error = (actual_z - expected_z).norm() / expected_z.norm();
    let mut skin_depth_error = 0.0;
    let mut phase_error = 0.0;
    if !sheet {
        let edge = topology
            .edges
            .iter()
            .enumerate()
            .filter(|(edge, (a, b))| {
                !prescribed.contains(edge) && (mesh.nodes[*a].x - mesh.nodes[*b].x).abs() < 1e-15
            })
            .min_by(|(_, (a, _)), (_, (b, _))| {
                (mesh.nodes[*a].x - delta)
                    .abs()
                    .total_cmp(&(mesh.nodes[*b].x - delta).abs())
            })
            .ok_or("no interior transverse edge")?;
        let (a, b) = *edge.1;
        let x = mesh.nodes[a].x;
        let value = solution[edge.0] / (mesh.nodes[b].y - mesh.nodes[a].y);
        let solved_delta = -x / value.norm().ln();
        skin_depth_error = (solved_delta - delta).abs() / delta;
        phase_error = (value.arg() + x / delta).abs();
    }
    Ok(SlabMetrics {
        field_error,
        residual,
        impedance_error,
        skin_depth_error,
        phase_error,
        free_dofs,
    })
}

pub(super) fn numerical_gate(sheet: bool) -> Result<QaResult, String> {
    let coarse = solve_slab(16, sheet)?;
    let medium = solve_slab(32, sheet)?;
    let fine = solve_slab(64, sheet)?;
    let rate = (medium.field_error / fine.field_error).log2();
    if !(coarse.field_error > medium.field_error && medium.field_error > fine.field_error)
        || !rate.is_finite()
        || rate < 0.8
    {
        return Err(format!(
            "conductor field does not converge: {:?}, {:?}, {:?}",
            coarse, medium, fine
        ));
    }
    // First-order edge interpolation has measured fine L2 errors 2.85%
    // (six-depth slab) and 0.38% (half-depth sheet). Preserve modest margin
    // while requiring contraction and the theoretical first-order rate.
    let field_bound = if sheet { 0.006 } else { 0.04 };
    let impedance_bound = if sheet { 3e-5 } else { 1.5e-3 };
    for (name, error, bound) in [
        ("complex field", fine.field_error, field_bound),
        ("physical residual", fine.residual, 1e-10),
        ("complex impedance", fine.impedance_error, impedance_bound),
    ] {
        if !error.is_finite() || error > bound {
            return Err(format!("{name} error {error:.3e} exceeds {bound:.3e}"));
        }
    }
    if !sheet
        && (!fine.skin_depth_error.is_finite()
            || fine.skin_depth_error > 1e-3
            || !fine.phase_error.is_finite()
            || fine.phase_error > 1e-3)
    {
        return Err(format!("skin-depth/phase check failed: {fine:?}"));
    }
    let id = if sheet {
        "fem.eddy-current-thin-sheet.v1"
    } else {
        "fem.eddy-current-1d-skin-effect.v1"
    };
    let mut record = QaResult::passed(id)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("coarse_complex_l2_error", coarse.field_error)
        .metric("medium_complex_l2_error", medium.field_error)
        .metric("fine_complex_l2_error", fine.field_error)
        .maximum("fine_complex_l2_error", field_bound)
        .metric("l2_refinement_rate", rate)
        .minimum("l2_refinement_rate", 0.8)
        .metric("physical_residual", fine.residual)
        .maximum("physical_residual", 1e-10)
        .metric("complex_impedance_error", fine.impedance_error)
        .maximum("complex_impedance_error", impedance_bound)
        .metric("free_edge_dofs", fine.free_dofs as f64)
        .minimum("free_edge_dofs", 1.0)
        .reference(
            "independent exponential half-space / cosh finite-sheet complex diffusion solution",
        )
        .formulation("production EddyCurrentForm, Nedelec edge elements, nonzero tangential traces")
        .seed("copper:slab:n16-n32-n64:complex-hcurl:v1")
        .input_checksum(if sheet { "copper:1MHz:half-depth-sheet:n16-n32-n64:v1" } else { "copper:1kHz:six-depth-slab:n16-n32-n64:v1" })
        .reference_checksum("analytic:complex-conductor-diffusion:exp-cosh:v1")
        .dimension("2-D copper conductor, transverse vector potential varying through thickness")
        .tolerance("first-order complex field convergence; independently recomputed residual and integrated-current impedance");
    if !sheet {
        record = record
            .metric("skin_depth_error", fine.skin_depth_error)
            .maximum("skin_depth_error", 1e-3)
            .metric("phase_error_radians", fine.phase_error)
            .maximum("phase_error_radians", 1e-3);
    }
    Ok(record)
}
