//! Curved Quad4 basis-aware MLFMM proxy-normal contract.

use num_complex::Complex64;
use sonium_bem::core::assembly::mlfmm::build_mlfmm_system_stabilized_with_basis_checked;
use sonium_bem::core::integration::unit_sphere_quadrature;
use sonium_bem::core::{
    BasisOrder, BoundaryCondition, Cluster, ClusterLevel, Element, ElementProperty, ElementType,
    PhysicsParams,
};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "bem.mlfmm-curved-quad4-normals.v1";

fn run() -> Result<(f64, f64), String> {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.4],
        [0.0, 1.0, 0.0],
    ];
    let element = Element {
        connectivity: vec![0, 1, 2, 3],
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0; 3]; 4],
        center: [0.5, 0.5, 0.1],
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses: vec![0, 1, 2, 3],
    };
    let mut cluster = Cluster::new(element.center);
    cluster.element_indices = vec![0];
    cluster.num_elements = 1;
    let mut level = ClusterLevel::new(1);
    level.theta_points = 4;
    level.phi_points = 4;
    level.expansion_terms = 16;
    level.clusters.push(cluster);

    let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
    let system = build_mlfmm_system_stabilized_with_basis_checked(
        std::slice::from_ref(&element),
        &nodes,
        vec![level],
        &physics,
        BasisOrder::Linear,
        -1.0,
        1.0,
    )
    .map_err(|error| format!("curved Quad4 MLFMM build failed: {error}"))?;

    let (sphere_coords, _) = unit_sphere_quadrature(4, 4);
    let points = element.collocation_points(BasisOrder::Linear, &nodes);
    let normals = element.collocation_normals(BasisOrder::Linear, &nodes);
    let normal_variation = (normals[0][2] - normals[1][2]).abs();
    let matrix = &system.s_matrices[0].matrices[0];
    let coord = sphere_coords[0];
    let ik = Complex64::new(0.0, physics.wave_number);
    let gamma_tau = Complex64::new(-physics.gamma() * physics.tau, 0.0);
    let beta = -physics.burton_miller_beta();
    let mut max_error: f64 = 0.0;
    for local in 0..4 {
        let diff = [
            points[local][0] - element.center[0],
            points[local][1] - element.center[1],
            points[local][2] - element.center[2],
        ];
        let s_dot_diff = coord[0] * diff[0] + coord[1] * diff[1] + coord[2] * diff[2];
        let s_dot_normal = coord[0] * normals[local][0]
            + coord[1] * normals[local][1]
            + coord[2] * normals[local][2];
        let expected = (gamma_tau + beta * (ik * s_dot_normal))
            * sonium_fastmath::exp_i(physics.wave_number * s_dot_diff);
        let error = (matrix[[local, 0]] - expected).norm();
        if !error.is_finite() {
            return Err(format!(
                "non-finite proxy-normal error at local DOF {local}"
            ));
        }
        max_error = max_error.max(error);
    }
    if normal_variation <= 1.0e-6 {
        return Err(format!(
            "curved Quad4 normal variation too small: {normal_variation}"
        ));
    }
    if max_error > 1.0e-12 {
        return Err(format!("curved Quad4 proxy-normal error {max_error:.3e}"));
    }
    Ok((normal_variation, max_error))
}

fn main() {
    let result = match run() {
        Ok((normal_variation, max_error)) => QaResult::passed(CASE_ID)
            .formulation("CPU basis-aware MLFMM curved Quad4 collocation-normal action")
            .seed("bem-mlfmm-curved-quad4-normals:cpu:v1")
            .metric("normal_variation", normal_variation)
            .metric("max_proxy_normal_error", max_error)
            .maximum("max_proxy_normal_error", 1.0e-12),
        Err(error) => QaResult::failed(CASE_ID, error)
            .formulation("CPU basis-aware MLFMM curved Quad4 collocation-normal action")
            .seed("bem-mlfmm-curved-quad4-normals:cpu:v1"),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if result.status == QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
