use num_complex::Complex64;
use sonium_bem::core::assembly::mlfmm::{
    DirectionalMlfmmConfig, build_directional_mlfmm_system_stabilized_with_basis_checked,
    build_directional_mlfmm_wavenumber_derivative_with_basis_checked,
    build_mlfmm_system_stabilized_with_basis_checked,
    build_mlfmm_wavenumber_derivative_with_basis_checked,
};
use sonium_bem::core::{
    AssemblyMethod, BasisContinuity, BasisOrder, BemProblem, BemSolver, BoundaryCondition, Cluster,
    ClusterLevel, Element, ElementProperty, ElementType, PhysicsParams, SolverMethod,
};

fn fixture() -> (Vec<Element>, Vec<[f64; 3]>, Vec<ClusterLevel>) {
    let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
    let element = Element {
        connectivity: vec![0, 1, 2],
        element_type: ElementType::Tri3,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 3],
        center: [1.0 / 3.0, 1.0 / 3.0, 0.0],
        area: 0.5,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses: (0..6).collect(),
    };
    let mut cluster = Cluster::new(element.center);
    cluster.element_indices = vec![0];
    cluster.num_elements = 1;
    let mut level = ClusterLevel::new(1);
    level.clusters.push(cluster);
    (vec![element], nodes, vec![level])
}

fn physics_at_wavenumber(wavenumber: f64) -> PhysicsParams {
    PhysicsParams::new(
        wavenumber * 343.0 / (2.0 * std::f64::consts::PI),
        343.0,
        1.21,
        false,
    )
}

fn assert_basis_derivative_matches_centered_difference(directional: bool) {
    let (elements, nodes, levels) = fixture();
    let physics = physics_at_wavenumber(1.8);
    let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
    let beta_scale = PhysicsParams::optimal_beta_scale(physics.wave_number);
    let config = DirectionalMlfmmConfig {
        min_kr: 1.0e9,
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
    let base = build(&physics, levels.clone()).expect("basis MLFMM value assembly");
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
    }
    .expect("basis MLFMM derivative assembly");
    let x: Vec<_> = (0..6)
        .map(|index| Complex64::new(0.2 + index as f64 * 0.11, -0.1 + index as f64 * 0.07))
        .collect();
    let analytic = derivative.apply(&base, &x);
    let step = 1.0e-5;
    let plus = build(
        &physics_at_wavenumber(physics.wave_number + step),
        levels.clone(),
    )
    .expect("positive basis MLFMM assembly")
    .matvec(&x);
    let minus = build(&physics_at_wavenumber(physics.wave_number - step), levels)
        .expect("negative basis MLFMM assembly")
        .matvec(&x);
    let finite_difference: Vec<_> = plus
        .iter()
        .zip(minus.iter())
        .map(|(plus, minus)| (*plus - *minus) / (2.0 * step))
        .collect();
    let error = analytic
        .iter()
        .zip(finite_difference.iter())
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let scale = finite_difference
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(1.0e-12);
    assert!(
        error / scale < 2.0e-4,
        "relative derivative error={}",
        error / scale
    );
}

#[test]
fn higher_order_mlfmm_wavenumber_derivative_matches_centered_difference() {
    assert_basis_derivative_matches_centered_difference(false);
}

#[test]
fn higher_order_directional_mlfmm_wavenumber_derivative_matches_centered_difference() {
    assert_basis_derivative_matches_centered_difference(true);
}

#[test]
fn high_level_quadratic_mlfmm_wavenumber_gradient_uses_checked_dispatch() {
    let problem = BemProblem::rigid_sphere_scattering_custom(0.1, 30.0, 343.0, 1.21, 3, 3);
    let active_elements = problem
        .mesh
        .elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .count();
    let dimension = active_elements * 6;
    let rhs: Vec<_> = (0..dimension)
        .map(|index| Complex64::new(0.3 + index as f64 * 0.01, -0.2 + index as f64 * 0.02))
        .collect();
    let solver = BemSolver::new()
        .with_basis_order(BasisOrder::Quadratic)
        .with_basis_continuity(BasisContinuity::Discontinuous)
        .with_assembly_method(AssemblyMethod::Mlfmm)
        .with_solver_method(SolverMethod::Gmres)
        .with_tolerance(1.0e-5)
        .with_max_iterations(40);
    let gradient = solver
        .solve_wavenumber_objective_gradient(&problem, &rhs, &rhs)
        .expect("higher-order MLFMM objective-gradient dispatch");
    assert!(gradient.is_finite());
}

fn multilevel_fixture() -> (Vec<Element>, Vec<[f64; 3]>, Vec<ClusterLevel>) {
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

fn assert_multilevel_analytic_derivative_matches_centered_difference(directional: bool) {
    let (elements, nodes, levels) = multilevel_fixture();
    let physics = physics_at_wavenumber(1.8);
    let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
    let beta_scale = PhysicsParams::optimal_beta_scale(physics.wave_number);
    let x: Vec<_> = (0..12)
        .map(|index| Complex64::new(0.2 + index as f64 * 0.011, -0.1 + index as f64 * 0.007))
        .collect();
    let config = DirectionalMlfmmConfig {
        min_kr: 0.0,
        min_active_points: 1,
        ..Default::default()
    };

    let (base, derivative) = if directional {
        (
            build_directional_mlfmm_system_stabilized_with_basis_checked(
                &elements,
                &nodes,
                levels.clone(),
                &physics,
                BasisOrder::Quadratic,
                sign,
                beta_scale,
                config,
            )
            .expect("multilevel directional MLFMM value"),
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
            .expect("multilevel directional MLFMM derivative"),
        )
    } else {
        (
            build_mlfmm_system_stabilized_with_basis_checked(
                &elements,
                &nodes,
                levels.clone(),
                &physics,
                BasisOrder::Quadratic,
                sign,
                beta_scale,
            )
            .expect("multilevel MLFMM value"),
            build_mlfmm_wavenumber_derivative_with_basis_checked(
                &elements,
                &nodes,
                levels.clone(),
                &physics,
                BasisOrder::Quadratic,
                sign,
                beta_scale,
            )
            .expect("multilevel MLFMM derivative"),
        )
    };
    assert!(
        base.d_matrices
            .iter()
            .flatten()
            .any(|entry| !entry.diagonal.is_empty())
    );
    assert!(
        derivative
            .d_matrices
            .iter()
            .flatten()
            .any(|entry| { entry.diagonal.iter().any(|value| value.norm() > 1.0e-12) })
    );
    assert!(derivative.t_matrices.iter().any(|level| {
        level
            .matrices
            .iter()
            .any(|matrix| matrix.as_slice().iter().any(|value| value.norm() > 1.0e-12))
    }));

    let analytic = derivative.apply(&base, &x);
    let step = 1.0e-5;
    let make_system = |wavenumber: f64| {
        let candidate = physics_at_wavenumber(wavenumber);
        if directional {
            build_directional_mlfmm_system_stabilized_with_basis_checked(
                &elements,
                &nodes,
                levels.clone(),
                &candidate,
                BasisOrder::Quadratic,
                PhysicsParams::optimal_formulation_sign(candidate.wave_number),
                PhysicsParams::optimal_beta_scale(candidate.wave_number),
                config,
            )
            .expect("directional perturbed MLFMM value")
        } else {
            build_mlfmm_system_stabilized_with_basis_checked(
                &elements,
                &nodes,
                levels.clone(),
                &candidate,
                BasisOrder::Quadratic,
                PhysicsParams::optimal_formulation_sign(candidate.wave_number),
                PhysicsParams::optimal_beta_scale(candidate.wave_number),
            )
            .expect("perturbed MLFMM value")
        }
    };
    let plus = make_system(physics.wave_number + step).matvec(&x);
    let minus = make_system(physics.wave_number - step).matvec(&x);
    let finite_difference: Vec<_> = plus
        .iter()
        .zip(minus.iter())
        .map(|(plus, minus)| (*plus - *minus) / (2.0 * step))
        .collect();
    let error = analytic
        .iter()
        .zip(finite_difference.iter())
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let scale = finite_difference
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(1.0e-12);
    assert!(
        error / scale < 3.0e-3,
        "multilevel derivative relative error={}",
        error / scale
    );
}

#[test]
fn higher_order_multilevel_mlfmm_analytic_factors_match_centered_difference() {
    assert_multilevel_analytic_derivative_matches_centered_difference(false);
}

#[test]
fn higher_order_multilevel_directional_mlfmm_analytic_factors_match_centered_difference() {
    assert_multilevel_analytic_derivative_matches_centered_difference(true);
}
