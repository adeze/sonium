//! Transverse Maxwell plane wave: identities and a solved complex field.

#[path = "support/hcurl_solve.rs"]
mod hcurl_solve;

use num_complex::Complex64;
use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::mesh::{rectangular_mesh_triangles, topology::MeshTopology};
use sonium_fem::physics::electromagnetics::maxwell::{
    EdgeMassForm, MaxwellForm, incident_plane_wave_e,
};
use sonium_fem::quadrature::QuadratureRule;
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::{QaResult, QaStatus};
use std::f64::consts::PI;

const CASE_ID: &str = "fem.maxwell-plane-wave.v1";
// A quarter-wavelength box keeps k below its first transverse resonance.
const SIDE: f64 = 0.25;
const WAVENUMBER: f64 = 2.0 * PI;

#[derive(Debug)]
struct WaveMetrics {
    field_error: f64,
    curl_error: f64,
    residual: f64,
    gauss_error: f64,
    free_dofs: usize,
}

fn solve_wave(n: usize) -> Result<WaveMetrics, String> {
    let mesh = rectangular_mesh_triangles(0.0, SIDE, 0.0, SIDE, n, n);
    let space = HcurlSpace::new();
    let form = MaxwellForm::from_coefficients(1.0, Complex64::new(-WAVENUMBER.powi(2), 0.0));
    let operator = assemble_bilinear(&form, &space, &space, &mesh, 4).to_csr();
    let exact = |x: f64| Complex64::from_polar(1.0, WAVENUMBER * x);
    let solved = hcurl_solve::solve_transverse(&mesh, &operator, &[1, 2, 3, 4], exact)?;
    let mut field_squared = 0.0;
    let mut curl_squared = 0.0;
    let mut volume = 0.0;
    for element in 0..mesh.num_elements() {
        let dofs = space.local_to_global(&mesh, element);
        for point in QuadratureRule::new(mesh.elements[element].element_type, 5).iter() {
            let tab = space.tabulate(&mesh, element, point.xi(), point.eta(), point.zeta());
            let mut field = [Complex64::new(0.0, 0.0); 2];
            let mut curl = Complex64::new(0.0, 0.0);
            for (local, &global) in dofs.iter().enumerate() {
                for (component, value) in field.iter_mut().enumerate() {
                    *value += solved.values[global] * tab.values[2 * local + component];
                }
                curl += solved.values[global] * tab.derivatives[local];
            }
            let reference = exact(tab.physical_coords[0]);
            let reference_curl = Complex64::new(0.0, WAVENUMBER) * reference;
            let weight = point.weight * tab.det_j;
            field_squared += weight * (field[0].norm_sqr() + (field[1] - reference).norm_sqr());
            curl_squared += weight * (curl - reference_curl).norm_sqr();
            volume += weight;
        }
    }
    // Gauss law tested weakly against interior nodal gradients: (E,grad phi)=0.
    // This includes the normal flux jumps between edge elements.
    let mass = assemble_bilinear(&EdgeMassForm, &space, &space, &mesh, 4).to_csr();
    let mass_field: Vec<Complex64> = (0..mass.num_rows)
        .map(|row| {
            mass.row_entries(row)
                .map(|(column, value)| solved.values[column] * value)
                .sum()
        })
        .collect();
    let mass_norm = mass_field.iter().map(|v| v.norm_sqr()).sum::<f64>().sqrt();
    let topology = MeshTopology::build(&mesh);
    let mut gauss_error = 0.0_f64;
    let mut gauss_tests = 0;
    for (node, point) in mesh.nodes.iter().enumerate() {
        if point.x <= 1e-12
            || point.x >= SIDE - 1e-12
            || point.y <= 1e-12
            || point.y >= SIDE - 1e-12
        {
            continue;
        }
        let mut weak_divergence = Complex64::new(0.0, 0.0);
        let mut gradient_squared = 0.0;
        for (edge, &(a, b)) in topology.edges.iter().enumerate() {
            let difference =
                (if b == node { 1.0 } else { 0.0 }) - (if a == node { 1.0 } else { 0.0 });
            weak_divergence += mass_field[edge] * difference;
            gradient_squared += difference * difference;
        }
        gauss_error =
            gauss_error.max(weak_divergence.norm() / (mass_norm * gradient_squared.sqrt()));
        gauss_tests += 1;
    }
    if gauss_tests == 0 || !gauss_error.is_finite() {
        return Err("no valid weak Gauss-law tests".into());
    }
    Ok(WaveMetrics {
        field_error: (field_squared / volume).sqrt(),
        curl_error: (curl_squared / (WAVENUMBER.powi(2) * volume)).sqrt(),
        residual: solved.residual,
        gauss_error,
        free_dofs: solved.free_dofs,
    })
}

fn run() -> Result<QaResult, String> {
    let wave = incident_plane_wave_e(
        [
            Complex64::new(0.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(0.0, 0.0),
        ],
        [1.0, 0.0, 0.0],
        WAVENUMBER,
    );
    let closure = (wave(0.0, 0.0, 0.0)[1] - wave(1.0, 0.0, 0.0)[1]).norm();
    // Independently differentiate the incident helper: curl E = i k E_y e_z,
    // div E = 0, curl curl E = k^2 E. The source-free wave is transverse.
    let h = 1e-4;
    let center = wave(0.1, 0.0, 0.0)[1];
    let left = wave(0.1 - h, 0.0, 0.0)[1];
    let right = wave(0.1 + h, 0.0, 0.0)[1];
    let curl_identity_error =
        ((right - left) / (2.0 * h) - Complex64::new(0.0, WAVENUMBER) * center).norm() / WAVENUMBER;
    let curl_curl_identity_error =
        (-(right - 2.0 * center + left) / h.powi(2) - WAVENUMBER.powi(2) * center).norm()
            / WAVENUMBER.powi(2);
    let longitudinal = wave(0.1, 0.0, 0.0)[0].norm();
    if closure > 1e-12
        || longitudinal != 0.0
        || curl_identity_error > 1e-6
        || curl_curl_identity_error > 1e-6
    {
        return Err("incident helper violates transverse Maxwell identities".into());
    }
    let coarse = solve_wave(8)?;
    let medium = solve_wave(16)?;
    let fine = solve_wave(32)?;
    let field_rate = (medium.field_error / fine.field_error).log2();
    let curl_rate = (medium.curl_error / fine.curl_error).log2();
    if !(coarse.field_error > medium.field_error
        && medium.field_error > fine.field_error
        && coarse.curl_error > medium.curl_error
        && medium.curl_error > fine.curl_error)
    {
        return Err(format!(
            "plane wave does not refine: {coarse:?}; {medium:?}; {fine:?}"
        ));
    }
    // Lowest-order edge fields and curls converge at first order. These
    // quarter-wave meshes must resolve both errors to 2% or better.
    for (name, value, bound) in [
        ("field", fine.field_error, 0.02),
        ("curl", fine.curl_error, 0.02),
        ("residual", fine.residual, 1e-10),
        ("weak Gauss law", fine.gauss_error, 1e-10),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name} error {value:.3e} exceeds {bound:.3e}"));
        }
    }
    if !field_rate.is_finite() || field_rate < 0.8 || !curl_rate.is_finite() || curl_rate < 0.8 {
        return Err("plane-wave field/curl refinement rate below 0.8".into());
    }
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("coarse_field_error", coarse.field_error)
        .metric("medium_field_error", medium.field_error)
        .metric("fine_field_error", fine.field_error)
        .maximum("fine_field_error", 0.02)
        .metric("fine_curl_error", fine.curl_error)
        .maximum("fine_curl_error", 0.02)
        .metric("field_refinement_rate", field_rate)
        .minimum("field_refinement_rate", 0.8)
        .metric("curl_refinement_rate", curl_rate)
        .minimum("curl_refinement_rate", 0.8)
        .metric("physical_residual", fine.residual)
        .maximum("physical_residual", 1e-10)
        .metric("weak_gauss_error", fine.gauss_error)
        .maximum("weak_gauss_error", 1e-10)
        .metric("free_edge_dofs", fine.free_dofs as f64)
        .minimum("free_edge_dofs", 1.0)
        .metric("phase_closure_error", closure)
        .maximum("phase_closure_error", 1e-12)
        .metric("longitudinal_component", longitudinal)
        .maximum("longitudinal_component", 0.0)
        .metric("curl_identity_error", curl_identity_error)
        .maximum("curl_identity_error", 1e-6)
        .metric("curl_curl_identity_error", curl_curl_identity_error)
        .maximum("curl_curl_identity_error", 1e-6)
        .reference(
            "transverse exp(i k x) plane wave, analytic curls and weak source-free Gauss law",
        )
        .dimension("2-D quarter-wavelength square, complex transverse vector field")
        .formulation("production MaxwellForm; Nedelec edge fields and exact tangential traces")
        .reference_checksum("analytic:transverse-plane-wave:exp-ikx:v1")
        .input_checksum("quarter-wave-square:n8-n16-n32:transverse:v1"))
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
