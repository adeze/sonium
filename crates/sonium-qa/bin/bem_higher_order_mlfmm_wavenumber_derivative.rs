//! Tiny CPU contract for basis-weighted MLFMM wavenumber derivative actions.

use num_complex::Complex64;
use sonium_bem::core::assembly::mlfmm::{
    DirectionalMlfmmConfig, build_directional_mlfmm_system_stabilized_with_basis_checked,
    build_directional_mlfmm_wavenumber_derivative_with_basis_checked,
    build_mlfmm_system_stabilized_with_basis_checked,
    build_mlfmm_wavenumber_derivative_with_basis_checked,
};
use sonium_bem::core::{
    BasisOrder, BoundaryCondition, Cluster, ClusterLevel, Element, ElementProperty, ElementType,
    PhysicsParams,
};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "bem.higher-order-mlfmm-wavenumber-derivative.v1";

fn fixture() -> (Vec<Element>, Vec<[f64; 3]>, Vec<ClusterLevel>) {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [4.0, 0.0, 0.0],
        [5.0, 0.0, 0.0],
        [4.0, 1.0, 0.0],
    ];
    let make_element = |connectivity: Vec<usize>, center: [f64; 3], dofs: Vec<usize>| Element {
        connectivity,
        element_type: ElementType::Tri3,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 3],
        center,
        area: 0.5,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses: dofs,
    };
    let elements = vec![
        make_element(vec![0, 1, 2], [1.0 / 3.0, 1.0 / 3.0, 0.0], (0..6).collect()),
        make_element(
            vec![3, 4, 5],
            [13.0 / 3.0, 1.0 / 3.0, 0.0],
            (6..12).collect(),
        ),
    ];

    let mut root = Cluster::new([2.5, 1.0 / 3.0, 0.0]);
    root.element_indices = vec![0, 1];
    root.num_elements = 2;
    root.sons = vec![0, 1];
    root.radius = 2.5;
    root.level = 0;

    let mut left = Cluster::new(elements[0].center);
    left.element_indices = vec![0];
    left.num_elements = 1;
    left.near_clusters = vec![0];
    left.far_clusters = vec![1];
    left.father = Some(0);
    left.radius = 0.75;
    left.level = 1;

    let mut right = Cluster::new(elements[1].center);
    right.element_indices = vec![1];
    right.num_elements = 1;
    right.near_clusters = vec![1];
    right.far_clusters = vec![0];
    right.father = Some(0);
    right.radius = 0.75;
    right.level = 1;

    let mut root_level = ClusterLevel::new(1);
    root_level.expansion_terms = 4;
    root_level.theta_points = 4;
    root_level.phi_points = 8;
    root_level.clusters.push(root);
    let mut leaf_level = ClusterLevel::new(2);
    leaf_level.expansion_terms = 4;
    leaf_level.theta_points = 4;
    leaf_level.phi_points = 8;
    leaf_level.clusters.extend([left, right]);
    (elements, nodes, vec![root_level, leaf_level])
}

fn physics_at_wavenumber(wavenumber: f64) -> PhysicsParams {
    PhysicsParams::new(
        wavenumber * 343.0 / (2.0 * std::f64::consts::PI),
        343.0,
        1.21,
        false,
    )
}

fn backend_error(directional: bool) -> Result<f64, String> {
    let (elements, nodes, levels) = fixture();
    let physics = physics_at_wavenumber(1.8);
    let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
    let beta_scale = PhysicsParams::optimal_beta_scale(physics.wave_number);
    let config = DirectionalMlfmmConfig {
        min_kr: 0.0,
        min_active_points: 1,
        ..Default::default()
    };
    let build = |candidate: &PhysicsParams, levels: Vec<ClusterLevel>| {
        if directional {
            build_directional_mlfmm_system_stabilized_with_basis_checked(
                &elements,
                &nodes,
                levels,
                candidate,
                BasisOrder::Quadratic,
                sign,
                beta_scale,
                config,
            )
        } else {
            build_mlfmm_system_stabilized_with_basis_checked(
                &elements,
                &nodes,
                levels,
                candidate,
                BasisOrder::Quadratic,
                sign,
                beta_scale,
            )
        }
    };
    let base = build(&physics, levels.clone()).map_err(|error| error.to_string())?;
    let derivative = if directional {
        build_directional_mlfmm_wavenumber_derivative_with_basis_checked(
            &elements,
            &nodes,
            levels.clone(),
            &physics,
            BasisOrder::Quadratic,
            sign,
            beta_scale,
            config,
        )
    } else {
        build_mlfmm_wavenumber_derivative_with_basis_checked(
            &elements,
            &nodes,
            levels.clone(),
            &physics,
            BasisOrder::Quadratic,
            sign,
            beta_scale,
        )
    }?;
    let x: Vec<_> = (0..12)
        .map(|index| Complex64::new(0.2 + index as f64 * 0.11, -0.1 + index as f64 * 0.07))
        .collect();
    let analytic = derivative.apply(&base, &x);
    let step = 1.0e-5;
    let plus = build(
        &physics_at_wavenumber(physics.wave_number + step),
        levels.clone(),
    )
    .map_err(|error| error.to_string())?
    .matvec(&x);
    let minus = build(&physics_at_wavenumber(physics.wave_number - step), levels)
        .map_err(|error| error.to_string())?
        .matvec(&x);
    let (error, scale) = analytic.iter().zip(plus.iter().zip(minus.iter())).fold(
        (0.0, 0.0),
        |(error, scale), (actual, (plus, minus))| {
            let expected = (*plus - *minus) / (2.0 * step);
            (
                error + (*actual - expected).norm_sqr(),
                scale + expected.norm_sqr(),
            )
        },
    );
    Ok(error.sqrt() / scale.sqrt().max(1.0e-12))
}

fn main() {
    let result = match (backend_error(false), backend_error(true)) {
        (Ok(full), Ok(directional)) if full.is_finite() && directional.is_finite() => {
            QaResult::passed(CASE_ID)
                .formulation("CPU quadratic basis-weighted multilevel MLFMM/directional-MLFMM wavenumber derivative action")
                .seed("bem-higher-order-mlfmm-wavenumber-derivative:cpu:v1")
                .metric("full_relative_error", full)
                .metric("directional_relative_error", directional)
                .maximum("full_relative_error", 3.0e-3)
                .maximum("directional_relative_error", 3.0e-3)
        }
        (Ok(full), Ok(directional)) => QaResult::failed(
            CASE_ID,
            format!("non-finite derivative errors full={full} directional={directional}"),
        ),
        (Err(error), _) | (_, Err(error)) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if !matches!(&result.status, QaStatus::Passed) {
        std::process::exit(1);
    }
}
