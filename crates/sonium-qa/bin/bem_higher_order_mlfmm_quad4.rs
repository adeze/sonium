//! Checked basis-weighted full/directional MLFMM admission for quadratic Quad4.
//!
//! This is a bounded source/type-check contract.  It intentionally uses a
//! single deterministic panel and must not be treated as a large simulation.

use num_complex::Complex64;
use sonium_bem::core::assembly::mlfmm::{
    DirectionalMlfmmConfig, build_cluster_tree,
    build_directional_mlfmm_system_stabilized_with_basis_checked,
    build_mlfmm_system_stabilized_with_basis_checked,
};
use sonium_bem::core::bem_solver::{
    AssemblyMethod, BemPreconditioner, BemProblem, BemSolver, BoundaryConditionType, SolverMethod,
};
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_cubed_sphere_quad_mesh;
use sonium_bem::core::types::{
    BasisOrder, BoundaryCondition, Cluster, ClusterLevel, Element, ElementProperty, ElementType,
    PhysicsParams,
};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::traits::LinearOperator;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.higher-order-mlfmm-quad4.v1";

#[derive(Clone, Copy, Debug)]
enum ReferenceAction {
    Forward,
    Transpose,
    Hermitian,
}

fn relative_error(expected: &[Complex64], actual: &[Complex64]) -> f64 {
    let numerator = expected
        .iter()
        .zip(actual)
        .map(|(left, right)| (*left - *right).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = expected
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    numerator / denominator
}

fn dense_action(
    matrix: &DenseMatrix<Complex64>,
    input: &[Complex64],
    action: ReferenceAction,
) -> Vec<Complex64> {
    let mut output = vec![Complex64::new(0.0, 0.0); matrix.nrows()];
    match action {
        ReferenceAction::Forward => matrix.matvec_into(input, &mut output),
        ReferenceAction::Transpose => matrix.matvec_transpose_into(input, &mut output),
        ReferenceAction::Hermitian => matrix.matvec_hermitian_into(input, &mut output),
    }
    output
}

fn far_field_reference(
    physics: &PhysicsParams,
    sign: f64,
    beta_scale: f64,
) -> Result<(f64, f64, usize, usize), String> {
    let mut nodes = Vec::with_capacity(8);
    let mut elements = Vec::with_capacity(2);
    for (offset, dof_start) in [(0.0, 0_usize), (20.0, 9_usize)] {
        let base = nodes.len();
        nodes.extend([
            [offset, 0.0, 0.0],
            [offset + 1.0, 0.0, 0.0],
            [offset + 1.0, 1.0, 0.0],
            [offset, 1.0, 0.0],
        ]);
        elements.push(Element {
            connectivity: vec![base, base + 1, base + 2, base + 3],
            element_type: ElementType::Quad4,
            property: ElementProperty::Surface,
            normal: [0.0, 0.0, 1.0],
            node_normals: vec![[0.0, 0.0, 1.0]; 4],
            center: [offset + 0.5, 0.5, 0.0],
            area: 1.0,
            boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
            group: 0,
            dof_addresses: (dof_start..dof_start + 9).collect(),
        });
    }

    let mut root = Cluster::new([10.5, 0.5, 0.0]);
    root.element_indices = vec![0, 1];
    root.num_elements = 2;
    root.radius = 10.5;
    root.level = 0;
    root.sons = vec![0, 1];

    let mut left = Cluster::new(elements[0].center);
    left.element_indices = vec![0];
    left.num_elements = 1;
    left.radius = 0.5;
    left.level = 1;
    left.father = Some(0);
    left.near_clusters = vec![0];
    left.far_clusters = vec![1];

    let mut right = Cluster::new(elements[1].center);
    right.element_indices = vec![1];
    right.num_elements = 1;
    right.radius = 0.5;
    right.level = 1;
    right.father = Some(0);
    right.near_clusters = vec![1];
    right.far_clusters = vec![0];

    let mut root_level = ClusterLevel::new(1);
    root_level.theta_points = 12;
    root_level.phi_points = 24;
    root_level.expansion_terms = 12;
    root_level.clusters = vec![root];
    let mut leaf_level = ClusterLevel::new(2);
    leaf_level.theta_points = 12;
    leaf_level.phi_points = 24;
    leaf_level.expansion_terms = 12;
    leaf_level.clusters = vec![left, right];

    let dense = sonium_bem::core::assembly::tbem::build_tbem_system_quadratic_stabilized(
        &elements, &nodes, physics, sign, beta_scale,
    );
    let levels = vec![root_level, leaf_level];
    let mlfmm = build_mlfmm_system_stabilized_with_basis_checked(
        &elements,
        &nodes,
        levels.clone(),
        physics,
        BasisOrder::Quadratic,
        sign,
        beta_scale,
    )
    .map_err(|error| format!("two-panel quadratic MLFMM reference build failed: {error}"))?;
    let directional = build_directional_mlfmm_system_stabilized_with_basis_checked(
        &elements,
        &nodes,
        levels,
        physics,
        BasisOrder::Quadratic,
        sign,
        beta_scale,
        DirectionalMlfmmConfig::default(),
    )
    .map_err(|error| format!("two-panel directional MLFMM reference build failed: {error}"))?;
    let far_entries = mlfmm.d_matrices.iter().flatten().count();
    if far_entries == 0 {
        return Err("two-panel reference fixture did not admit a far-field translation".into());
    }
    let input: Vec<_> = (0..dense.num_dofs)
        .map(|index| Complex64::new(0.2 + index as f64 * 0.01, -0.1 + index as f64 * 0.02))
        .collect();
    let mut maximum_error = 0.0_f64;
    let mut maximum_directional_error = 0.0_f64;
    for action in [
        ReferenceAction::Forward,
        ReferenceAction::Transpose,
        ReferenceAction::Hermitian,
    ] {
        let expected = dense_action(&dense.matrix, &input, action);
        let actual = match action {
            ReferenceAction::Forward => mlfmm.matvec(&input),
            ReferenceAction::Transpose => mlfmm.matvec_transpose(&input),
            ReferenceAction::Hermitian => mlfmm.apply_hermitian(&input),
        };
        if actual
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(format!(
                "{action:?} MLFMM action returned non-finite output"
            ));
        }
        maximum_error = maximum_error.max(relative_error(&expected, &actual));

        let directional_actual = match action {
            ReferenceAction::Forward => directional.matvec(&input),
            ReferenceAction::Transpose => directional.matvec_transpose(&input),
            ReferenceAction::Hermitian => directional.apply_hermitian(&input),
        };
        if directional_actual
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        {
            return Err(format!(
                "directional {action:?} MLFMM action returned non-finite output"
            ));
        }
        maximum_directional_error =
            maximum_directional_error.max(relative_error(&expected, &directional_actual));
    }
    if directional.translation_stats().active_points == 0 {
        return Err(
            "two-panel reference fixture did not retain directional far-field points".into(),
        );
    }
    Ok((
        maximum_error,
        maximum_directional_error,
        far_entries,
        directional.translation_stats().active_points,
    ))
}

fn run() -> Result<(usize, f64, f64, usize, usize, usize, f64, f64, usize, usize), String> {
    let nodes = vec![
        [-1.0, -1.0, 0.0],
        [1.0, -1.0, 0.0],
        [1.0, 1.0, 0.0],
        [-1.0, 1.0, 0.0],
    ];
    let elements = vec![Element {
        connectivity: vec![0, 1, 2, 3],
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center: [0.0, 0.0, 0.0],
        area: 4.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses: (0..9).collect(),
    }];
    let physics = PhysicsParams::new(120.0, 343.0, 1.21, false);
    let (
        mlfmm_reference_error,
        directional_reference_error,
        far_entries,
        directional_reference_active_points,
    ) = far_field_reference(&physics, -1.0, 1.0)?;
    let cluster_levels = build_cluster_tree(&elements, 1, &physics);

    let full = build_mlfmm_system_stabilized_with_basis_checked(
        &elements,
        &nodes,
        cluster_levels.clone(),
        &physics,
        BasisOrder::Quadratic,
        -1.0,
        1.0,
    )
    .map_err(|error| format!("full quadratic Quad4 MLFMM failed: {error}"))?;
    let directional = build_directional_mlfmm_system_stabilized_with_basis_checked(
        &elements,
        &nodes,
        cluster_levels,
        &physics,
        BasisOrder::Quadratic,
        -1.0,
        1.0,
        DirectionalMlfmmConfig::default(),
    )
    .map_err(|error| format!("directional quadratic Quad4 MLFMM failed: {error}"))?;

    let input = (0..9)
        .map(|index| Complex64::new(0.2 + index as f64 / 13.0, -0.1 + index as f64 / 17.0))
        .collect::<Vec<_>>();
    let full_output = full.matvec(&input);
    let directional_output = directional.matvec(&input);
    let difference = full_output
        .iter()
        .zip(&directional_output)
        .map(|(full, directional)| (full - directional).norm())
        .fold(0.0_f64, f64::max);
    if full_output
        .iter()
        .chain(&directional_output)
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("quadratic Quad4 MLFMM action returned non-finite output".to_string());
    }
    if full.num_dofs != 9 || directional.num_dofs != 9 {
        return Err("quadratic Quad4 MLFMM did not preserve nine global DOFs".to_string());
    }

    let problem = BemProblem {
        mesh: generate_cubed_sphere_quad_mesh(0.1, 1),
        physics: physics.clone(),
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    };
    let solver = BemSolver::new()
        .with_continuous_quadratic()
        .with_assembly_method(AssemblyMethod::Mlfmm)
        .with_solver_method(SolverMethod::Gmres)
        .with_preconditioner(BemPreconditioner::Jacobi)
        .with_tolerance(1.0e-8)
        .with_max_iterations(300);
    let solution = solver
        .solve(&problem)
        .map_err(|error| format!("continuous quadratic Quad4 MLFMM solve failed: {error}"))?;
    let field = solution.evaluate_pressure(&[0.0, 0.0, 0.2]);
    if solution.num_dofs() != 26 || !field.re.is_finite() || !field.im.is_finite() {
        return Err("continuous quadratic Quad4 MLFMM solve returned invalid output".to_string());
    }

    Ok((
        full.num_dofs,
        difference,
        full.translation_stats().active_fraction(),
        full.translation_stats().active_points,
        directional.translation_stats().active_points,
        solution.num_dofs(),
        mlfmm_reference_error,
        directional_reference_error,
        far_entries,
        directional_reference_active_points,
    ))
}

fn main() {
    let started = Instant::now();
    let result = match catch_unwind(AssertUnwindSafe(run)) {
        Ok(Ok((
            dofs,
            action_error,
            full_active_fraction,
            full_active_points,
            directional_active_points,
            continuous_dofs,
            mlfmm_reference_error,
            directional_reference_error,
            far_entries,
            directional_reference_active_points,
        ))) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("dofs", dofs as f64)
            .metric("action_error", action_error)
            .metric("full_active_fraction", full_active_fraction)
            .metric("full_active_points", full_active_points as f64)
            .metric(
                "directional_active_points",
                directional_active_points as f64,
            )
            .metric("continuous_dofs", continuous_dofs as f64)
            .metric("mlfmm_reference_error", mlfmm_reference_error)
            .metric("directional_reference_error", directional_reference_error)
            .metric("far_field_entries", far_entries as f64)
            .metric(
                "directional_reference_active_points",
                directional_reference_active_points as f64,
            )
            .maximum("action_error", 1.0e-10)
            .maximum("mlfmm_reference_error", 1.0e-8)
            .maximum("directional_reference_error", 1.0e-3)
            .minimum("directional_reference_active_points", 1.0)
            .dimension(
                "3-D quadratic Quad4 full/directional builder plus continuous six-panel solve",
            )
            .formulation("basis-weighted full and directional MLFMM proxy expansion")
            .reference_checksum("analytic/deterministic-fixture:bem-higher-order-mlfmm-quad4-v1")
            .seed("bem-higher-order-mlfmm-quad4:cpu:v1"),
        Ok(Err(error)) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .dimension(
                "3-D quadratic Quad4 full/directional builder plus continuous six-panel solve",
            )
            .seed("bem-higher-order-mlfmm-quad4:cpu:v1"),
        Err(_) => QaResult::failed(CASE_ID, "quadratic Quad4 MLFMM QA panicked")
            .requested_backend("cpu")
            .actual_backend("cpu")
            .dimension(
                "3-D quadratic Quad4 full/directional builder plus continuous six-panel solve",
            )
            .seed("bem-higher-order-mlfmm-quad4:cpu:v1"),
    }
    .elapsed_ms(started.elapsed().as_millis() as u64);
    println!("{}", result.to_json_line().expect("serialize QA result"));
}
