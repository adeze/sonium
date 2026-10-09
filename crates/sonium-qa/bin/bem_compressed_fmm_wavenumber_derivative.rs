//! Bounded SLFMM/MLFMM compressed wavenumber-derivative action contract.
//!
//! This QA case exercises the factorized derivative actions directly.  It
//! uses two separated Tri3 panels and never launches a solver or simulator.

use num_complex::Complex64;
use sonium_bem::core::assembly::mlfmm::{
    DirectionalMlfmmConfig, build_cluster_tree, build_directional_mlfmm_system_stabilized_checked,
    build_directional_mlfmm_wavenumber_derivative, build_mlfmm_system_stabilized_checked,
    build_mlfmm_wavenumber_derivative,
};
use sonium_bem::core::assembly::slfmm::{
    build_slfmm_system_stabilized_checked, build_slfmm_wavenumber_derivative,
};
use sonium_bem::core::{
    BoundaryCondition, Cluster, Element, ElementProperty, ElementType, PhysicsParams,
};
use sonium_qa::QaResult;

const CASE_ID: &str = "bem.compressed-fmm-wavenumber-derivative.v1";

fn fixture() -> (Vec<Element>, Vec<[f64; 3]>, Vec<Cluster>) {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [4.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [4.0, 1.0, 0.0],
    ];
    let element = |connectivity: Vec<usize>, center: [f64; 3], dof| Element {
        connectivity,
        element_type: ElementType::Tri3,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0; 3]; 3],
        center,
        area: 0.5,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses: vec![dof],
    };
    let elements = vec![
        element(vec![0, 1, 2], [1.0 / 3.0, 1.0 / 3.0, 0.0], 0),
        element(vec![3, 4, 5], [13.0 / 3.0, 1.0 / 3.0, 0.0], 1),
    ];
    let mut left = Cluster::new(elements[0].center);
    left.element_indices = vec![0];
    left.num_elements = 1;
    left.near_clusters = vec![0];
    left.far_clusters = vec![1];
    let mut right = Cluster::new(elements[1].center);
    right.element_indices = vec![1];
    right.num_elements = 1;
    right.near_clusters = vec![1];
    right.far_clusters = vec![0];
    (elements, nodes, vec![left, right])
}

fn relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    let numerator = actual
        .iter()
        .zip(expected.iter())
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = expected
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(1.0e-30);
    numerator / denominator
}

fn run() -> Result<(f64, f64, f64), String> {
    let (elements, nodes, clusters) = fixture();
    let physics = PhysicsParams::new(100.0, 343.0, 1.21, false);
    let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
    let beta_scale = 0.2;
    let x = vec![Complex64::new(0.7, -0.2), Complex64::new(-0.3, 0.4)];
    let step = 1.0e-4;
    let make_physics = |wavenumber: f64| {
        PhysicsParams::new(
            wavenumber * physics.speed_of_sound / (2.0 * std::f64::consts::PI),
            physics.speed_of_sound,
            physics.density,
            false,
        )
    };
    let plus_physics = make_physics(physics.wave_number + step);
    let minus_physics = make_physics(physics.wave_number - step);

    let slfmm_base = build_slfmm_system_stabilized_checked(
        &elements, &nodes, &clusters, &physics, 4, 8, 4, sign, beta_scale,
    )
    .map_err(|error| error.to_string())?;
    let slfmm_derivative = build_slfmm_wavenumber_derivative(
        &elements, &nodes, &clusters, &physics, 4, 8, 4, sign, beta_scale,
    )
    .map_err(|error| error.to_string())?;
    let slfmm_plus = build_slfmm_system_stabilized_checked(
        &elements,
        &nodes,
        &clusters,
        &plus_physics,
        4,
        8,
        4,
        sign,
        beta_scale,
    )
    .map_err(|error| error.to_string())?
    .matvec(&x);
    let slfmm_minus = build_slfmm_system_stabilized_checked(
        &elements,
        &nodes,
        &clusters,
        &minus_physics,
        4,
        8,
        4,
        sign,
        beta_scale,
    )
    .map_err(|error| error.to_string())?
    .matvec(&x);
    let slfmm_expected: Vec<_> = slfmm_plus
        .iter()
        .zip(slfmm_minus.iter())
        .map(|(plus, minus)| (*plus - *minus) / (2.0 * step))
        .collect();
    let slfmm_error = relative_error(&slfmm_derivative.apply(&slfmm_base, &x), &slfmm_expected);

    let cluster_levels = build_cluster_tree(&elements, 1, &physics);
    let mlfmm_base = build_mlfmm_system_stabilized_checked(
        &elements,
        &nodes,
        cluster_levels.clone(),
        &physics,
        sign,
        beta_scale,
    )
    .map_err(|error| error.to_string())?;
    let mlfmm_derivative = build_mlfmm_wavenumber_derivative(
        &elements,
        &nodes,
        cluster_levels.clone(),
        &physics,
        sign,
        beta_scale,
    )
    .map_err(|error| error.to_string())?;
    let mlfmm_plus = build_mlfmm_system_stabilized_checked(
        &elements,
        &nodes,
        cluster_levels.clone(),
        &plus_physics,
        sign,
        beta_scale,
    )
    .map_err(|error| error.to_string())?
    .matvec(&x);
    let mlfmm_minus = build_mlfmm_system_stabilized_checked(
        &elements,
        &nodes,
        cluster_levels,
        &minus_physics,
        sign,
        beta_scale,
    )
    .map_err(|error| error.to_string())?
    .matvec(&x);
    let mlfmm_expected: Vec<_> = mlfmm_plus
        .iter()
        .zip(mlfmm_minus.iter())
        .map(|(plus, minus)| (*plus - *minus) / (2.0 * step))
        .collect();
    let mlfmm_error = relative_error(&mlfmm_derivative.apply(&mlfmm_base, &x), &mlfmm_expected);
    let directional_levels = build_cluster_tree(&elements, 1, &physics);
    let directional_config = DirectionalMlfmmConfig::default();
    let directional_base = build_directional_mlfmm_system_stabilized_checked(
        &elements,
        &nodes,
        directional_levels.clone(),
        &physics,
        sign,
        beta_scale,
        directional_config,
    )
    .map_err(|error| error.to_string())?;
    let directional_derivative = build_directional_mlfmm_wavenumber_derivative(
        &elements,
        &nodes,
        directional_levels.clone(),
        &physics,
        sign,
        beta_scale,
        directional_config,
    )
    .map_err(|error| error.to_string())?;
    let directional_plus = build_directional_mlfmm_system_stabilized_checked(
        &elements,
        &nodes,
        directional_levels.clone(),
        &plus_physics,
        sign,
        beta_scale,
        directional_config,
    )
    .map_err(|error| error.to_string())?
    .matvec(&x);
    let directional_minus = build_directional_mlfmm_system_stabilized_checked(
        &elements,
        &nodes,
        directional_levels,
        &minus_physics,
        sign,
        beta_scale,
        directional_config,
    )
    .map_err(|error| error.to_string())?
    .matvec(&x);
    let directional_expected: Vec<_> = directional_plus
        .iter()
        .zip(directional_minus.iter())
        .map(|(plus, minus)| (*plus - *minus) / (2.0 * step))
        .collect();
    let directional_error = relative_error(
        &directional_derivative.apply(&directional_base, &x),
        &directional_expected,
    );
    if !slfmm_error.is_finite() || !mlfmm_error.is_finite() || !directional_error.is_finite() {
        return Err("non-finite compressed derivative error".to_string());
    }
    if slfmm_error > 2.0e-3 || mlfmm_error > 2.0e-3 || directional_error > 2.0e-3 {
        return Err(format!(
            "compressed FMM derivative error exceeds tolerance: SLFMM={slfmm_error:e}, MLFMM={mlfmm_error:e}, directional={directional_error:e}"
        ));
    }
    Ok((slfmm_error, mlfmm_error, directional_error))
}

fn main() {
    let result = match run() {
        Ok((slfmm_error, mlfmm_error, directional_error)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("slfmm_relative_error", slfmm_error)
            .metric("mlfmm_relative_error", mlfmm_error)
            .metric("directional_relative_error", directional_error)
            .maximum("slfmm_relative_error", 2.0e-3)
            .maximum("mlfmm_relative_error", 2.0e-3)
            .maximum("directional_relative_error", 2.0e-3)
            .reference("independent centered compressed SLFMM/MLFMM actions")
            .dimension("two separated constant Tri3 panels; Complex64 action")
            .formulation("fixed-topology compressed Burton-Miller wavenumber derivative")
            .seed("bem-compressed-fmm-wavenumber-derivative:cpu:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .reference("independent centered compressed SLFMM/MLFMM actions")
            .dimension("two separated constant Tri3 panels; Complex64 action")
            .formulation("fixed-topology compressed Burton-Miller wavenumber derivative")
            .seed("bem-compressed-fmm-wavenumber-derivative:cpu:v1"),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if result.status == sonium_qa::QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
