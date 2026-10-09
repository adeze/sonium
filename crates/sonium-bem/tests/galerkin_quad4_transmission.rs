//! Focused separated isoparametric Quad4 Galerkin Müller coverage.

use num_complex::Complex64;
use sonium_bem::core::IncidentField;
use sonium_bem::core::assembly::common::get_element_coords;
use sonium_bem::core::integration::{
    galerkin_single_layer_self_quad4_wavenumber_derivative_with_basis,
    galerkin_single_layer_self_quad4_with_basis,
    galerkin_single_layer_touching_quad4_wavenumber_derivative_with_basis,
    galerkin_single_layer_touching_quad4_with_basis,
};
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::{
    BasisOrder, GalerkinAssemblyError, GalerkinQuad4TransmissionError,
    GalerkinQuadratureAdmissionError, GalerkinQuadraturePolicy, PhysicsParams,
    TransmissionMaterialParameter, TransmissionSide,
    assemble_galerkin_hypersingular_with_global_dofs,
    assemble_galerkin_muller_transmission_material_derivative_with_global_dofs,
    assemble_galerkin_muller_transmission_quad4_pair,
    assemble_galerkin_muller_transmission_quad4_pair_wavenumber_derivative,
    assemble_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs,
    assemble_galerkin_muller_transmission_with_global_dofs,
    assemble_galerkin_muller_transmission_with_global_dofs_checked,
    compress_galerkin_muller_transmission_quad4_material_derivative_with_global_dofs,
    compress_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs,
    compress_galerkin_muller_transmission_quad4_with_global_dofs_checked,
    compress_galerkin_muller_transmission_with_global_dofs,
    compress_galerkin_muller_transmission_with_global_dofs_checked,
};
use sonium_solvers::iterative::GmresConfig;

fn quad(connectivity: Vec<usize>, dof_addresses: Vec<usize>, center: [f64; 3]) -> Element {
    Element {
        connectivity,
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center,
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses,
    }
}

fn media() -> (PhysicsParams, PhysicsParams) {
    (
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("exterior medium"),
        PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, true).expect("interior medium"),
    )
}

fn pair_nodes(curved: bool) -> (Vec<Element>, Vec<[f64; 3]>) {
    let source_bend = if curved { 0.3 } else { 0.0 };
    let field_bend = if curved { 0.3 } else { 0.0 };
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, source_bend],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.2],
        [1.0, 0.0, 1.2],
        [1.0, 1.0, 1.2 + field_bend],
        [0.0, 1.0, 1.2],
    ];
    let elements = vec![
        quad(vec![0, 1, 2, 3], (0..9).collect(), [0.5, 0.5, 0.0]),
        quad(vec![4, 5, 6, 7], (0..9).collect(), [0.5, 0.5, 1.2]),
    ];
    (elements, nodes)
}

fn closed_cube() -> (Vec<Element>, Vec<[f64; 3]>) {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let connectivity = [
        vec![0, 1, 2, 3],
        vec![4, 7, 6, 5],
        vec![0, 4, 5, 1],
        vec![1, 5, 6, 2],
        vec![2, 6, 7, 3],
        vec![3, 7, 4, 0],
    ];
    let elements = connectivity
        .into_iter()
        .map(|connectivity| {
            let dof_addresses = connectivity.clone();
            quad(connectivity, dof_addresses, [0.5, 0.5, 0.5])
        })
        .collect();
    (elements, nodes)
}

fn closed_cube_quadratic() -> (Vec<Element>, Vec<[f64; 3]>) {
    let (mut elements, nodes) = closed_cube();
    let quadratic_dofs = [
        vec![0, 1, 2, 3, 8, 9, 10, 11, 20],
        vec![4, 7, 6, 5, 15, 14, 13, 12, 21],
        vec![0, 4, 5, 1, 16, 12, 17, 8, 22],
        vec![1, 5, 6, 2, 17, 13, 18, 9, 23],
        vec![2, 6, 7, 3, 18, 14, 19, 10, 24],
        vec![3, 7, 4, 0, 19, 15, 16, 11, 25],
    ];
    for (element, dofs) in elements.iter_mut().zip(quadratic_dofs) {
        element.dof_addresses = dofs;
    }
    (elements, nodes)
}

fn assert_finite(matrix: &sonium_solvers::dense::DenseMatrix<Complex64>) {
    assert!(
        matrix
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
}

fn with_wavenumber(medium: &PhysicsParams, wave_number: f64) -> PhysicsParams {
    let mut perturbed = medium.clone();
    perturbed.speed_of_sound = perturbed.omega / wave_number;
    perturbed.wave_number = wave_number;
    perturbed.wave_length = perturbed.speed_of_sound / perturbed.frequency;
    perturbed.pressure_factor = perturbed.density * perturbed.omega * perturbed.harmonic_factor;
    perturbed
}

#[test]
fn open_quad4_hypersingular_matches_direct_finite_part_for_both_orders() {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let physics = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physics");

    for (basis, local_basis_count) in [(BasisOrder::Linear, 4_usize), (BasisOrder::Quadratic, 9)] {
        let element = quad(
            vec![0, 1, 2, 3],
            (0..local_basis_count).collect(),
            [0.5, 0.5, 0.0],
        );
        let operators = assemble_galerkin_hypersingular_with_global_dofs(
            std::slice::from_ref(&element),
            &nodes,
            &physics,
            basis,
            2,
            local_basis_count,
        )
        .expect("open Quad4 hypersingular assembly");
        assert_finite(&operators.hypersingular);

        let coords = get_element_coords(&element, &nodes);
        let direct =
            galerkin_single_layer_self_quad4_with_basis(&coords, &physics, local_basis_count, 2);
        for row in 0..local_basis_count {
            for column in 0..local_basis_count {
                let direct_value = (direct.basis_results[row][column].d2g_dnxdny_integral
                    + direct.basis_results[column][row].d2g_dnxdny_integral)
                    * 0.5;
                let error = (operators.hypersingular[[row, column]] - direct_value).norm();
                assert!(error < 1.0e-10, "open Quad4 finite-part error={error}");
            }
        }
    }
}

#[test]
fn open_quad4_hypersingular_compressed_matches_dense() {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let elements = vec![quad(vec![0, 1, 2, 3], vec![0, 1, 2, 3], [0.5, 0.5, 0.0])];
    let exterior = PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid exterior");
    let interior = PhysicsParams::try_new(420.0, 290.0, 1.55, false).expect("valid interior");
    let dense = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        6,
        4,
    )
    .expect("open Quad4 dense Müller assembly");
    let incident = IncidentField::plane_wave_z();
    let (pressure, velocity) = dense
        .project_incident_trace_coefficients(&elements, &nodes, &incident, &exterior, 2)
        .expect("open Quad4 incident projection");
    assert_eq!(pressure.len(), 4);
    assert_eq!(velocity.len(), 4);
    // Dense and compressed operators must use the same quadrature rule for
    // this action parity check; compression is not a quadrature refinement.
    let compressed = compress_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        6,
        4,
        2,
        1.0e-10,
        8,
    )
    .expect("open Quad4 compressed Müller assembly");
    let probe = (0..dense.total_dofs())
        .map(|index| Complex64::new(0.2 + index as f64 * 0.01, -0.1))
        .collect::<Vec<_>>();
    let dense_action = dense.matrix().matvec(&probe);
    let compressed_action = compressed
        .apply_checked(&probe)
        .expect("open Quad4 compressed action");
    let error = dense_action
        .iter()
        .zip(&compressed_action)
        .map(|(expected, actual)| (*expected - *actual).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        error < 1.0e-9,
        "open Quad4 compressed action error {error:e}"
    );

    let rhs = dense
        .incident_rhs_from_trace_coefficients(&pressure, &velocity)
        .expect("open Quad4 incident RHS");
    let dense_solution = dense.solve(&rhs, 1.0e-8).expect("open Quad4 dense solve");
    let mut config = GmresConfig::default();
    config.max_iterations = 80;
    config.restart = 16;
    config.tolerance = 1.0e-10;
    config.exact_residual_check_interval = 1;
    let mut nearfield = sonium_solvers::dense::DenseMatrix::zeros(8, 8);
    for index in 0..8 {
        nearfield[[index, index]] = Complex64::new(1.0, 0.0);
    }
    let compressed_solution = compressed
        .solve_with_ilu(&rhs, &nearfield, &config, 1.0e-8)
        .expect("open Quad4 compressed solve");
    let points = [[0.2, 0.3, 1.8], [0.7, 0.4, 1.2]];
    let dense_field = dense
        .evaluate_exterior_field_checked(
            &elements,
            &nodes,
            &dense_solution,
            &points,
            &incident,
            &exterior,
            2,
        )
        .expect("open Quad4 dense field");
    let compressed_field = dense
        .evaluate_exterior_field_from_compressed_solution_checked(
            &compressed_solution,
            &elements,
            &nodes,
            &points,
            &incident,
            &exterior,
            2,
        )
        .expect("open Quad4 compressed field");
    let field_error = dense_field
        .iter()
        .zip(&compressed_field)
        .map(|(expected, actual)| (expected.p_total - actual.p_total).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        field_error < 1.0e-7,
        "open Quad4 compressed field error {field_error:e}"
    );
}

#[test]
fn curved_quad4_compressed_field_matches_dense_solution() {
    let (elements, mut nodes) = closed_cube_quadratic();
    nodes[6][2] += 0.3;
    let (exterior, interior) = media();
    let operators = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        2,
        26,
    )
    .expect("curved Quad4 dense Müller assembly");
    let compressed = compress_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        2,
        26,
        4,
        1.0e-9,
        16,
    )
    .expect("curved Quad4 compressed Müller assembly");

    let incident = IncidentField::plane_wave_z();
    let (pressure, velocity) = operators
        .project_incident_trace_coefficients(&elements, &nodes, &incident, &exterior, 2)
        .expect("curved Quad4 incident projection");
    let rhs = operators
        .incident_rhs_from_trace_coefficients(&pressure, &velocity)
        .expect("curved Quad4 incident RHS");
    let dense_solution = operators
        .solve(&rhs, 1.0e-8)
        .expect("curved Quad4 dense solve");

    let mut config = GmresConfig::default();
    config.max_iterations = 120;
    config.restart = 32;
    config.tolerance = 1.0e-10;
    config.exact_residual_check_interval = 1;
    let mut nearfield =
        sonium_solvers::dense::DenseMatrix::zeros(operators.total_dofs(), operators.total_dofs());
    for index in 0..operators.total_dofs() {
        nearfield[[index, index]] = Complex64::new(1.0, 0.0);
    }
    let compressed_solution = compressed
        .solve_with_ilu(&rhs, &nearfield, &config, 1.0e-8)
        .expect("curved Quad4 compressed solve");

    let points = [[0.2, 0.3, 1.8], [1.4, 0.4, 0.8]];
    let dense_field = operators
        .evaluate_exterior_field_checked(
            &elements,
            &nodes,
            &dense_solution,
            &points,
            &incident,
            &exterior,
            2,
        )
        .expect("curved Quad4 dense field evaluation");
    let compressed_field = operators
        .evaluate_exterior_field_from_compressed_solution_checked(
            &compressed_solution,
            &elements,
            &nodes,
            &points,
            &incident,
            &exterior,
            2,
        )
        .expect("curved Quad4 compressed field evaluation");

    let field_error = compressed_field
        .iter()
        .zip(&dense_field)
        .map(|(actual, expected)| (actual.p_total - expected.p_total).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        field_error <= 2.0e-6,
        "curved Quad4 compressed field error {field_error:e}"
    );
    assert!(
        compressed_field
            .iter()
            .all(|value| { value.p_total.re.is_finite() && value.p_total.im.is_finite() })
    );
}

#[test]
fn checked_curved_quad4_compression_certifies_against_admitted_source() {
    let (elements, mut nodes) = closed_cube_quadratic();
    nodes[6][2] += 0.3;
    let (exterior, interior) = media();
    let policy = GalerkinQuadraturePolicy {
        coarse_order: 1,
        fine_order: 2,
        absolute_tolerance: 1.0e6,
        relative_tolerance: 1.0e6,
    };
    let compressed = compress_galerkin_muller_transmission_quad4_with_global_dofs_checked(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        26,
        policy,
        4,
        1.0e-1,
        1,
    )
    .expect("checked curved Quad4 compression");
    let dense = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        policy.fine_order,
        26,
    )
    .expect("admitted fine curved Quad4 source");
    let incident = IncidentField::plane_wave_z();
    let (pressure, velocity) = dense
        .project_incident_trace_coefficients(&elements, &nodes, &incident, &exterior, 2)
        .expect("curved Quad4 incident projection");
    let rhs = dense
        .incident_rhs_from_trace_coefficients(&pressure, &velocity)
        .expect("curved Quad4 incident RHS");
    let mut config = GmresConfig::default();
    config.max_iterations = 40;
    config.restart = 8;
    config.tolerance = 1.0e-10;
    config.exact_residual_check_interval = 1;
    let mut nearfield = sonium_solvers::dense::DenseMatrix::zeros(52, 52);
    for index in 0..52 {
        nearfield[[index, index]] = Complex64::new(1.0, 0.0);
    }
    let solution = compressed
        .solve_with_ilu(&rhs, &nearfield, &config, 1.0e-8)
        .expect("exact-source-certified curved Quad4 solve");
    let mut residual_squared = 0.0;
    let mut rhs_squared = 0.0;
    for row in 0..dense.total_dofs() {
        let action = (0..dense.total_dofs())
            .map(|column| dense.matrix()[[row, column]] * solution.values()[column])
            .sum::<Complex64>();
        residual_squared += (action - rhs[row]).norm_sqr();
        rhs_squared += rhs[row].norm_sqr();
    }
    let exact_residual = residual_squared.sqrt() / rhs_squared.sqrt().max(f64::MIN_POSITIVE);
    assert!(solution.relative_residual() <= 1.0e-8);
    assert!(
        exact_residual <= 1.0e-8,
        "exact residual {exact_residual:e}"
    );
}

#[test]
fn curved_quad4_checked_compression_rejects_invalid_policy_before_assembly() {
    let (elements, nodes) = closed_cube();
    let (exterior, interior) = media();
    let error = compress_galerkin_muller_transmission_quad4_with_global_dofs_checked(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        8,
        GalerkinQuadraturePolicy {
            coarse_order: 2,
            fine_order: 2,
            absolute_tolerance: 1.0e-8,
            relative_tolerance: 1.0e-6,
        },
        4,
        1.0e-8,
        16,
    )
    .expect_err("equal coarse/fine orders must be rejected before assembly");
    assert!(matches!(
        error,
        sonium_bem::GalerkinTransmissionCompressionError::Quadrature(
            GalerkinQuadratureAdmissionError::Policy(_)
        )
    ));
}

#[test]
fn curved_quad4_transmission_block_is_finite_and_uses_isoparametric_geometry() {
    let (exterior, interior) = media();
    let (curved_elements, curved_nodes) = pair_nodes(true);
    let mut curved_linear_elements = curved_elements.clone();
    for element in &mut curved_linear_elements {
        element.dof_addresses = (0..4).collect();
    }
    let curved = assemble_galerkin_muller_transmission_quad4_pair(
        &curved_linear_elements[0],
        &curved_linear_elements[1],
        &curved_nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        6,
    )
    .expect("curved Quad4 transmission block");
    assert_eq!(curved.shape(), (8, 8));
    assert_finite(&curved);
    assert!(curved.as_slice().iter().any(|value| value.norm() > 1.0e-12));

    let (flat_elements, flat_nodes) = pair_nodes(false);
    let mut flat_linear_elements = flat_elements.clone();
    for element in &mut flat_linear_elements {
        element.dof_addresses = (0..4).collect();
    }
    let flat = assemble_galerkin_muller_transmission_quad4_pair(
        &flat_linear_elements[0],
        &flat_linear_elements[1],
        &flat_nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        6,
    )
    .expect("flat Quad4 transmission block");
    let geometry_delta = curved
        .as_slice()
        .iter()
        .zip(flat.as_slice())
        .map(|(curved, flat)| (curved - flat).norm())
        .fold(0.0, f64::max);
    assert!(
        geometry_delta > 1.0e-6,
        "curved map was not used: {geometry_delta}"
    );

    let quadratic = assemble_galerkin_muller_transmission_quad4_pair(
        &curved_elements[0],
        &curved_elements[1],
        &curved_nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        6,
    )
    .expect("biquadratic Quad4 transmission block");
    assert_eq!(quadratic.shape(), (18, 18));
    assert_finite(&quadratic);
}

#[test]
fn curved_quad4_transmission_accepts_touching_pair_and_rejects_bad_layout() {
    let (exterior, interior) = media();
    let (mut elements, nodes) = pair_nodes(true);
    for element in &mut elements {
        element.dof_addresses = (0..4).collect();
    }
    elements[1].connectivity = vec![1, 4, 5, 2];
    let touching = assemble_galerkin_muller_transmission_quad4_pair(
        &elements[0],
        &elements[1],
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        6,
    )
    .expect("mapped curved transmission touching pair");
    assert_eq!(touching.shape(), (8, 8));
    assert_finite(&touching);

    elements[1].connectivity = vec![2, 4, 5, 6];
    let vertex_touching = assemble_galerkin_muller_transmission_quad4_pair(
        &elements[0],
        &elements[1],
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        6,
    )
    .expect("mapped curved transmission common-vertex pair");
    assert_eq!(vertex_touching.shape(), (8, 8));
    assert_finite(&vertex_touching);

    elements[1].connectivity = vec![4, 5, 6, 7];
    elements[0].dof_addresses.pop();
    let bad_layout = assemble_galerkin_muller_transmission_quad4_pair(
        &elements[0],
        &elements[1],
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        6,
    )
    .expect_err("curved transmission must reject a short local basis map");
    assert!(matches!(
        bad_layout,
        GalerkinQuad4TransmissionError::InvalidBasisDofs {
            element: 0,
            dofs: 3,
            expected: 4,
        }
    ));
}

#[test]
fn curved_quad4_transmission_accepts_self_pair_for_both_orders() {
    let (exterior, interior) = media();
    let (mut elements, nodes) = pair_nodes(true);

    elements[0].dof_addresses = (0..4).collect();
    let linear = assemble_galerkin_muller_transmission_quad4_pair(
        &elements[0],
        &elements[0],
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        6,
    )
    .expect("mapped curved transmission self pair (linear)");
    assert_eq!(linear.shape(), (8, 8));
    assert_finite(&linear);
    assert!(linear.as_slice().iter().any(|value| value.norm() > 1.0e-12));

    elements[0].dof_addresses = (0..9).collect();
    let quadratic = assemble_galerkin_muller_transmission_quad4_pair(
        &elements[0],
        &elements[0],
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Quadratic,
        6,
    )
    .expect("mapped curved transmission self pair (quadratic)");
    assert_eq!(quadratic.shape(), (18, 18));
    assert_finite(&quadratic);
    assert!(
        quadratic
            .as_slice()
            .iter()
            .any(|value| value.norm() > 1.0e-12)
    );
}

#[test]
fn closed_quad4_global_muller_assembles_self_and_touching_pairs() {
    let (exterior, interior) = media();
    let (elements, nodes) = closed_cube();
    let operators = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        8,
    )
    .expect("closed Quad4 Müller assembly");
    assert_eq!(operators.matrix().shape(), (16, 16));
    assert_finite(operators.matrix());
}

#[test]
fn closed_quad4_global_wavenumber_derivative_matches_centered_reassembly() {
    let (exterior, interior) = media();
    let (elements, nodes) = closed_cube();
    let analytic =
        assemble_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs(
            &elements,
            &nodes,
            &exterior,
            &interior,
            BasisOrder::Linear,
            2,
            8,
            TransmissionSide::Exterior,
        )
        .expect("global Quad4 analytic wavenumber derivative");
    let step = (exterior.wave_number.abs() * 1.0e-5).max(1.0e-7);
    let plus = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &with_wavenumber(&exterior, exterior.wave_number + step),
        &interior,
        BasisOrder::Linear,
        2,
        8,
    )
    .expect("global plus perturbation");
    let minus = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &with_wavenumber(&exterior, exterior.wave_number - step),
        &interior,
        BasisOrder::Linear,
        2,
        8,
    )
    .expect("global minus perturbation");
    let mut max_error = 0.0_f64;
    let mut max_reference = 0.0_f64;
    for ((analytic, plus), minus) in analytic
        .as_slice()
        .iter()
        .zip(plus.matrix().as_slice())
        .zip(minus.matrix().as_slice())
    {
        let reference = (*plus - *minus) / (2.0 * step);
        max_error = max_error.max((*analytic - reference).norm());
        max_reference = max_reference.max(reference.norm());
    }
    assert!(
        max_error <= 2.0e-3 * (1.0 + max_reference),
        "global Quad4 derivative error {max_error:e} reference {max_reference:e}"
    );

    let compressed =
        compress_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs(
            &elements,
            &nodes,
            &exterior,
            &interior,
            BasisOrder::Linear,
            2,
            8,
            TransmissionSide::Exterior,
            2,
            1.0e-6,
            8,
        )
        .expect("compressed global Quad4 derivative");
    let trial = vec![Complex64::new(0.03, -0.01); compressed.total_dofs()];
    let compressed_action = sonium_solvers::traits::LinearOperator::apply(&compressed, &trial);
    let dense_action = analytic.dot(&trial);
    let action_error = compressed_action
        .iter()
        .zip(dense_action.iter())
        .map(|(actual, expected)| (*actual - *expected).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        action_error <= 1.0e-6,
        "compressed action error {action_error:e}"
    );
}

#[test]
fn open_quad4_global_wavenumber_derivative_matches_direct_reassembly() {
    let (exterior, interior) = media();
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
    ];
    let elements = vec![quad(vec![0, 1, 2, 3], (0..4).collect(), [0.5, 0.5, 0.0])];
    let analytic =
        assemble_galerkin_muller_transmission_quad4_wavenumber_derivative_with_global_dofs(
            &elements,
            &nodes,
            &exterior,
            &interior,
            BasisOrder::Linear,
            2,
            4,
            TransmissionSide::Exterior,
        )
        .expect("open Quad4 analytic wavenumber derivative");
    let step = (exterior.wave_number.abs() * 1.0e-5).max(1.0e-7);
    let plus = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &with_wavenumber(&exterior, exterior.wave_number + step),
        &interior,
        BasisOrder::Linear,
        2,
        4,
    )
    .expect("open Quad4 plus perturbation");
    let minus = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &with_wavenumber(&exterior, exterior.wave_number - step),
        &interior,
        BasisOrder::Linear,
        2,
        4,
    )
    .expect("open Quad4 minus perturbation");
    let mut max_error = 0.0_f64;
    let mut max_reference = 0.0_f64;
    for ((analytic, plus), minus) in analytic
        .as_slice()
        .iter()
        .zip(plus.matrix().as_slice())
        .zip(minus.matrix().as_slice())
    {
        let reference = (*plus - *minus) / (2.0 * step);
        max_error = max_error.max((*analytic - reference).norm());
        max_reference = max_reference.max(reference.norm());
    }
    assert!(
        max_error <= 2.0e-3 * (1.0 + max_reference),
        "open Quad4 derivative error {max_error:e} reference {max_reference:e}"
    );
}

#[test]
fn checked_global_quad4_quadrature_policy_is_a_real_admission_gate() {
    let (exterior, interior) = media();
    let (elements, nodes) = closed_cube();
    let permissive = GalerkinQuadraturePolicy {
        coarse_order: 1,
        fine_order: 2,
        absolute_tolerance: 1.0e6,
        relative_tolerance: 1.0e6,
    };
    let operators = assemble_galerkin_muller_transmission_with_global_dofs_checked(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        8,
        permissive,
    )
    .expect("permissive adjacent-order policy should admit closed cube");
    assert_eq!(operators.matrix().shape(), (16, 16));
    assert_finite(operators.matrix());
    let compressed = compress_galerkin_muller_transmission_with_global_dofs_checked(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        8,
        permissive,
        2,
        1.0e-6,
        4,
    )
    .expect("checked compression should inherit permissive quadrature gate");
    assert_eq!(compressed.total_dofs(), 16);
    let trial = vec![Complex64::new(0.1, -0.02); compressed.total_dofs()];
    let action = sonium_solvers::traits::LinearOperator::apply(&compressed, &trial);
    assert!(
        action
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let strict = GalerkinQuadraturePolicy {
        absolute_tolerance: 0.0,
        relative_tolerance: 0.0,
        ..permissive
    };
    let error = assemble_galerkin_muller_transmission_with_global_dofs_checked(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        8,
        strict,
    )
    .expect_err("zero adjacent-order budget must fail closed");
    assert!(matches!(
        error,
        GalerkinQuadratureAdmissionError::NotConverged(_)
    ));

    let compression_error = compress_galerkin_muller_transmission_with_global_dofs_checked(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        8,
        strict,
        2,
        1.0e-6,
        4,
    )
    .expect_err("strict compression policy must fail before ACA");
    assert!(matches!(
        compression_error,
        sonium_bem::GalerkinTransmissionCompressionError::Quadrature(
            GalerkinQuadratureAdmissionError::NotConverged(_)
        )
    ));

    let invalid = GalerkinQuadraturePolicy {
        coarse_order: 2,
        fine_order: 2,
        ..permissive
    };
    let error = assemble_galerkin_muller_transmission_with_global_dofs_checked(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        8,
        invalid,
    )
    .expect_err("non-increasing policy must be rejected before assembly");
    assert!(matches!(error, GalerkinQuadratureAdmissionError::Policy(_)));
}

#[test]
fn closed_quad4_quadratic_hypersingular_rejects_vertex_edge_dof_alias() {
    let (mut elements, nodes) = closed_cube_quadratic();
    // Vertex DOFs are 0..8.  The four edge DOFs follow in a stable
    // undirected-edge numbering.  The (0,1) midpoint is deliberately given
    // vertex-6's address in both incident panels; neither incident panel owns
    // vertex 6, so the local duplicate check cannot catch this alias.
    elements[0].dof_addresses[4] = 6;
    elements[2].dof_addresses[7] = 6;

    let error = assemble_galerkin_hypersingular_with_global_dofs(
        &elements,
        &nodes,
        &PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("physics"),
        BasisOrder::Quadratic,
        2,
        26,
    )
    .expect_err("vertex/edge global DOF alias must fail closed");
    assert!(matches!(
        error,
        GalerkinAssemblyError::HypersingularGlobalDofEntityConflict {
            element: 1,
            local: 2,
            dof: 6,
        }
    ));
}

#[test]
fn closed_quad4_quadratic_hypersingular_rejects_center_dof_alias() {
    let (mut elements, nodes) = closed_cube_quadratic();
    // The top panel has no vertex 0, so this alias is invisible to its local
    // duplicate check but collides with the bottom panel's vertex trace DOF.
    elements[1].dof_addresses[8] = 0;
    let error = assemble_galerkin_hypersingular_with_global_dofs(
        &elements,
        &nodes,
        &PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("physics"),
        BasisOrder::Quadratic,
        2,
        26,
    )
    .expect_err("element-center DOF alias must fail closed");
    assert!(matches!(
        error,
        GalerkinAssemblyError::HypersingularGlobalDofEntityConflict {
            element: 1,
            local: 8,
            dof: 0,
        }
    ));
}

#[test]
fn closed_quad4_hypersingular_rejects_inconsistent_winding() {
    let (mut elements, nodes) = closed_cube();
    // Reverse the bottom panel's local order and its matching linear DOFs.
    // Its shared (0, 1) edge now has the same directed traversal as the
    // adjacent side panel, so the weak-form conormal cancellation contract
    // must reject the layout before integration.
    elements[0].connectivity = vec![0, 3, 2, 1];
    elements[0].dof_addresses = vec![0, 3, 2, 1];

    let error = assemble_galerkin_hypersingular_with_global_dofs(
        &elements,
        &nodes,
        &PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("physics"),
        BasisOrder::Linear,
        2,
        8,
    )
    .expect_err("inconsistent closed Quad4 winding must fail before integration");

    assert!(matches!(
        error,
        GalerkinAssemblyError::InconsistentHypersingularOrientation { .. }
    ));
}

#[test]
fn closed_quad4_hypersingular_rejects_out_of_range_node() {
    let (mut elements, nodes) = closed_cube();
    elements[0].connectivity[2] = nodes.len();

    let error = assemble_galerkin_hypersingular_with_global_dofs(
        &elements,
        &nodes,
        &PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("physics"),
        BasisOrder::Linear,
        2,
        8,
    )
    .expect_err("out-of-range Quad4 node must be rejected before coordinate lookup");

    assert!(matches!(
        error,
        GalerkinAssemblyError::NodeOutOfRange {
            element: 0,
            local: 2,
            node,
            node_count,
        } if node == nodes.len() && node_count == nodes.len()
    ));
}

#[test]
fn closed_quad4_muller_projects_solves_and_evaluates_field() {
    let (exterior, interior) = media();
    let (elements, nodes) = closed_cube();
    let operators = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        8,
    )
    .expect("closed Quad4 Müller assembly");
    let incident = IncidentField::plane_wave_z();
    let rhs = operators
        .incident_rhs_from_field(&elements, &nodes, &incident, &exterior, 2)
        .expect("Quad4 incident projection and RHS");
    let solution = operators.solve(&rhs, 1.0e-6).expect("Quad4 Müller solve");
    let field = operators
        .evaluate_exterior_field_checked(
            &elements,
            &nodes,
            &solution,
            &[[0.1, -0.2, 0.8], [0.5, 0.5, 1.8]],
            &incident,
            &exterior,
            2,
        )
        .expect("Quad4 exterior field evaluation");
    assert_eq!(field.len(), 2);
    assert!(field.iter().all(|value| {
        value.p_incident.re.is_finite()
            && value.p_incident.im.is_finite()
            && value.p_scattered.re.is_finite()
            && value.p_scattered.im.is_finite()
            && value.p_total.re.is_finite()
            && value.p_total.im.is_finite()
    }));

    let mut mixed = elements.clone();
    mixed[0].element_type = ElementType::Tri3;
    let mixed_error = operators
        .project_incident_trace_coefficients(&mixed, &nodes, &incident, &exterior, 2)
        .expect_err("mixed Tri3/Quad4 projection must fail closed");
    assert!(matches!(
        mixed_error,
        sonium_bem::GalerkinTransmissionProjectionError::Layout(
            GalerkinAssemblyError::UnsupportedElementType { element: 0, .. }
        )
    ));

    let mut malformed = elements.clone();
    malformed[0].connectivity.pop();
    let malformed_error = operators
        .project_incident_trace_coefficients(&malformed, &nodes, &incident, &exterior, 2)
        .expect_err("malformed Quad4 connectivity must fail closed");
    assert!(matches!(
        malformed_error,
        sonium_bem::GalerkinTransmissionProjectionError::Layout(
            GalerkinAssemblyError::InvalidQuad4Connectivity { element: 0, .. }
        )
    ));
}
#[test]
fn quad4_kernel_native_compression_matches_dense_global_action_for_both_orders() {
    let (exterior, interior) = media();
    for (basis, global_dof_count) in [(BasisOrder::Linear, 8), (BasisOrder::Quadratic, 26)] {
        let (elements, nodes) = match basis {
            BasisOrder::Linear => closed_cube(),
            BasisOrder::Quadratic => closed_cube_quadratic(),
            BasisOrder::Constant => unreachable!(),
        };
        let dense = assemble_galerkin_muller_transmission_with_global_dofs(
            &elements,
            &nodes,
            &exterior,
            &interior,
            basis,
            2,
            global_dof_count,
        )
        .expect("dense closed Quad4 Müller assembly");
        let compressed = compress_galerkin_muller_transmission_with_global_dofs(
            &elements,
            &nodes,
            &exterior,
            &interior,
            basis,
            2,
            global_dof_count,
            4,
            1.0e-10,
            64,
        )
        .expect("kernel-native Quad4 compression");
        let trial: Vec<Complex64> = (0..compressed.total_dofs())
            .map(|index| Complex64::new(0.07 * (index as f64 + 1.0), -0.03))
            .collect();
        let compressed_action = sonium_solvers::traits::LinearOperator::apply(&compressed, &trial);
        let dense_action = (0..dense.matrix().nrows())
            .map(|row| {
                (0..dense.matrix().ncols())
                    .map(|column| dense.matrix()[[row, column]] * trial[column])
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        assert!(
            compressed_action
                .iter()
                .all(|value| { value.re.is_finite() && value.im.is_finite() })
        );
        let relative_error = compressed_action
            .iter()
            .zip(dense_action.iter())
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0_f64, f64::max)
            / dense_action
                .iter()
                .map(|value| value.norm())
                .fold(0.0_f64, f64::max)
                .max(f64::MIN_POSITIVE);
        assert!(
            relative_error < 2.0e-7,
            "{basis:?} kernel-native action error {relative_error:e}"
        );
    }
}

#[test]
fn self_quad4_wavenumber_derivative_matches_centered_reference() {
    let (exterior, _) = media();
    let (elements, nodes) = pair_nodes(false);
    let coords = get_element_coords(&elements[0], &nodes);
    let gauss_order = 2;
    let analytic = galerkin_single_layer_self_quad4_wavenumber_derivative_with_basis(
        &coords,
        &exterior,
        4,
        gauss_order,
    );
    let step = (exterior.wave_number.abs() * 1.0e-5).max(1.0e-7);
    let plus = galerkin_single_layer_self_quad4_with_basis(
        &coords,
        &with_wavenumber(&exterior, exterior.wave_number + step),
        4,
        gauss_order,
    );
    let minus = galerkin_single_layer_self_quad4_with_basis(
        &coords,
        &with_wavenumber(&exterior, exterior.wave_number - step),
        4,
        gauss_order,
    );
    let mut max_error = 0.0_f64;
    let mut max_reference = 0.0_f64;
    for (analytic_row, (plus_row, minus_row)) in analytic
        .basis_results
        .iter()
        .zip(plus.basis_results.iter().zip(minus.basis_results.iter()))
    {
        for ((analytic, plus), minus) in analytic_row
            .iter()
            .zip(plus_row.iter())
            .zip(minus_row.iter())
        {
            let reference = (plus.g_integral - minus.g_integral) / (2.0 * step);
            max_error = max_error.max((analytic.g_integral - reference).norm());
            max_reference = max_reference.max(reference.norm());
        }
    }
    assert!(
        max_error <= 5.0e-4 * (1.0 + max_reference),
        "self Quad4 analytic derivative error {max_error:e} reference {max_reference:e}"
    );
}

#[test]
fn touching_quad4_wavenumber_derivative_matches_centered_reference() {
    let (exterior, _) = media();
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.2],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.2],
    ];
    let source = quad(vec![0, 1, 2, 3], (0..4).collect(), [0.5, 0.5, 0.05]);
    let field = quad(vec![1, 4, 5, 2], (4..8).collect(), [1.5, 0.5, 0.05]);
    let source_coords = get_element_coords(&source, &nodes);
    let field_coords = get_element_coords(&field, &nodes);
    let gauss_order = 2;
    let analytic = galerkin_single_layer_touching_quad4_wavenumber_derivative_with_basis(
        &source_coords,
        &field_coords,
        &source.connectivity,
        &field.connectivity,
        &exterior,
        4,
        4,
        gauss_order,
    );
    let step = (exterior.wave_number.abs() * 1.0e-5).max(1.0e-7);
    let plus = galerkin_single_layer_touching_quad4_with_basis(
        &source_coords,
        &field_coords,
        &source.connectivity,
        &field.connectivity,
        &with_wavenumber(&exterior, exterior.wave_number + step),
        4,
        4,
        gauss_order,
    );
    let minus = galerkin_single_layer_touching_quad4_with_basis(
        &source_coords,
        &field_coords,
        &source.connectivity,
        &field.connectivity,
        &with_wavenumber(&exterior, exterior.wave_number - step),
        4,
        4,
        gauss_order,
    );
    let mut max_error = 0.0_f64;
    let mut max_reference = 0.0_f64;
    for (analytic_row, (plus_row, minus_row)) in analytic
        .basis_results
        .iter()
        .zip(plus.basis_results.iter().zip(minus.basis_results.iter()))
    {
        for ((analytic, plus), minus) in analytic_row
            .iter()
            .zip(plus_row.iter())
            .zip(minus_row.iter())
        {
            let reference = (plus.g_integral - minus.g_integral) / (2.0 * step);
            max_error = max_error.max((analytic.g_integral - reference).norm());
            max_reference = max_reference.max(reference.norm());
        }
    }
    assert!(
        max_error <= 5.0e-4 * (1.0 + max_reference),
        "touching Quad4 analytic derivative error {max_error:e} reference {max_reference:e}"
    );
}

#[test]
fn separated_quad4_wavenumber_derivative_matches_centered_reference() {
    let (exterior, interior) = media();
    let (mut elements, nodes) = pair_nodes(true);
    for element in &mut elements {
        element.dof_addresses = (0..4).collect();
    }
    let source = &elements[0];
    let field = &elements[1];
    let gauss_order = 2;
    let analytic = assemble_galerkin_muller_transmission_quad4_pair_wavenumber_derivative(
        source,
        field,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        gauss_order,
        TransmissionSide::Exterior,
    )
    .expect("analytic separated Quad4 wavenumber derivative");

    let step = (exterior.wave_number.abs() * 1.0e-5).max(1.0e-7);
    let plus_medium = with_wavenumber(&exterior, exterior.wave_number + step);
    let minus_medium = with_wavenumber(&exterior, exterior.wave_number - step);
    let plus = assemble_galerkin_muller_transmission_quad4_pair(
        source,
        field,
        &nodes,
        &plus_medium,
        &interior,
        BasisOrder::Linear,
        gauss_order,
    )
    .expect("plus perturbation");
    let minus = assemble_galerkin_muller_transmission_quad4_pair(
        source,
        field,
        &nodes,
        &minus_medium,
        &interior,
        BasisOrder::Linear,
        gauss_order,
    )
    .expect("minus perturbation");
    let mut max_error = 0.0_f64;
    let mut max_reference = 0.0_f64;
    for ((analytic, plus), minus) in analytic
        .as_slice()
        .iter()
        .zip(plus.as_slice())
        .zip(minus.as_slice())
    {
        let reference = (*plus - *minus) / (2.0 * step);
        max_error = max_error.max((*analytic - reference).norm());
        max_reference = max_reference.max(reference.norm());
    }
    assert!(
        max_error <= 5.0e-4 * (1.0 + max_reference),
        "analytic Quad4 wavenumber derivative error {max_error:e} reference {max_reference:e}"
    );
}

#[test]
fn self_quad4_transmission_wavenumber_derivative_matches_centered_reference() {
    let (exterior, interior) = media();
    let (mut elements, nodes) = pair_nodes(true);
    for element in &mut elements {
        element.dof_addresses = (0..4).collect();
    }
    let source = &elements[0];
    let gauss_order = 2;
    let analytic = assemble_galerkin_muller_transmission_quad4_pair_wavenumber_derivative(
        source,
        source,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        gauss_order,
        TransmissionSide::Exterior,
    )
    .expect("analytic self Quad4 wavenumber derivative");

    let step = (exterior.wave_number.abs() * 1.0e-5).max(1.0e-7);
    let plus = assemble_galerkin_muller_transmission_quad4_pair(
        source,
        source,
        &nodes,
        &with_wavenumber(&exterior, exterior.wave_number + step),
        &interior,
        BasisOrder::Linear,
        gauss_order,
    )
    .expect("plus perturbation");
    let minus = assemble_galerkin_muller_transmission_quad4_pair(
        source,
        source,
        &nodes,
        &with_wavenumber(&exterior, exterior.wave_number - step),
        &interior,
        BasisOrder::Linear,
        gauss_order,
    )
    .expect("minus perturbation");

    let mut max_error = 0.0_f64;
    let mut max_reference = 0.0_f64;
    for ((analytic, plus), minus) in analytic
        .as_slice()
        .iter()
        .zip(plus.as_slice())
        .zip(minus.as_slice())
    {
        let reference = (*plus - *minus) / (2.0 * step);
        max_error = max_error.max((*analytic - reference).norm());
        max_reference = max_reference.max(reference.norm());
    }
    assert!(
        max_error <= 5.0e-4 * (1.0 + max_reference),
        "self Quad4 analytic derivative error {max_error:e} reference {max_reference:e}"
    );
}

#[test]
fn quad4_kernel_native_material_derivative_matches_dense_action() {
    let (exterior, interior) = media();
    let (mut elements, nodes) = pair_nodes(true);
    for (offset, element) in [0, 4].into_iter().zip(&mut elements) {
        element.dof_addresses = (offset..offset + 4).collect();
    }
    let policy = GalerkinQuadraturePolicy {
        coarse_order: 2,
        fine_order: 3,
        absolute_tolerance: 1.0e-6,
        relative_tolerance: 1.0e-4,
    };
    let trial: Vec<Complex64> = (0..16)
        .map(|index| Complex64::new(0.05 * (index as f64 + 1.0), -0.02))
        .collect();

    for (side, parameter) in [
        (
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
        ),
        (
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::SpeedOfSound,
        ),
        (
            TransmissionSide::Interior,
            TransmissionMaterialParameter::Wavenumber,
        ),
        (
            TransmissionSide::Interior,
            TransmissionMaterialParameter::SpeedOfSound,
        ),
        (
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Density,
        ),
        (
            TransmissionSide::Interior,
            TransmissionMaterialParameter::Density,
        ),
    ] {
        let dense = assemble_galerkin_muller_transmission_material_derivative_with_global_dofs(
            &elements,
            &nodes,
            &exterior,
            &interior,
            BasisOrder::Linear,
            policy.fine_order,
            8,
            side,
            parameter,
        )
        .expect("dense Quad4 material derivative");
        let compressed =
            compress_galerkin_muller_transmission_quad4_material_derivative_with_global_dofs(
                &elements,
                &nodes,
                &exterior,
                &interior,
                BasisOrder::Linear,
                policy.fine_order,
                8,
                side,
                parameter,
                4,
                1.0e-10,
                64,
            )
            .expect("kernel-native Quad4 material derivative");
        let compressed_action = sonium_solvers::traits::LinearOperator::apply(&compressed, &trial);
        let dense_action = (0..dense.nrows())
            .map(|row| {
                (0..dense.ncols())
                    .map(|column| dense[[row, column]] * trial[column])
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        let relative_error = compressed_action
            .iter()
            .zip(dense_action.iter())
            .map(|(actual, expected)| (*actual - *expected).norm())
            .fold(0.0_f64, f64::max)
            / dense_action
                .iter()
                .map(|value| value.norm())
                .fold(0.0_f64, f64::max)
                .max(f64::MIN_POSITIVE);
        assert!(
            relative_error < 2.0e-6,
            "{side:?} {parameter:?} Quad4 material derivative action error {relative_error:e}"
        );
    }
}

#[test]
fn quad4_global_material_derivatives_match_centered_reassembly() {
    let (exterior, interior) = media();
    let (mut elements, nodes) = pair_nodes(true);
    for (offset, element) in [0, 4].into_iter().zip(&mut elements) {
        element.dof_addresses = (offset..offset + 4).collect();
    }

    let perturb = |medium: &PhysicsParams, parameter: TransmissionMaterialParameter, value: f64| {
        let mut perturbed = medium.clone();
        match parameter {
            TransmissionMaterialParameter::Wavenumber => {
                perturbed.speed_of_sound = perturbed.omega / value;
            }
            TransmissionMaterialParameter::SpeedOfSound => {
                perturbed.speed_of_sound = value;
            }
            TransmissionMaterialParameter::Density => {
                perturbed.density = value;
            }
        }
        perturbed.wave_number = perturbed.omega / perturbed.speed_of_sound;
        perturbed.wave_length = perturbed.speed_of_sound / perturbed.frequency;
        perturbed.pressure_factor = perturbed.density * perturbed.omega * perturbed.harmonic_factor;
        perturbed
    };
    let step = |medium: &PhysicsParams, parameter: TransmissionMaterialParameter| match parameter {
        TransmissionMaterialParameter::Wavenumber => {
            (medium.wave_number.abs() * 1.0e-5).max(1.0e-7)
        }
        TransmissionMaterialParameter::SpeedOfSound => {
            (medium.speed_of_sound.abs() * 1.0e-5).max(1.0e-5)
        }
        TransmissionMaterialParameter::Density => (medium.density.abs() * 1.0e-5).max(1.0e-5),
    };

    for (side, parameter) in [
        (
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
        ),
        (
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::SpeedOfSound,
        ),
        (
            TransmissionSide::Interior,
            TransmissionMaterialParameter::Wavenumber,
        ),
        (
            TransmissionSide::Interior,
            TransmissionMaterialParameter::SpeedOfSound,
        ),
        (
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Density,
        ),
        (
            TransmissionSide::Interior,
            TransmissionMaterialParameter::Density,
        ),
    ] {
        let selected = match side {
            TransmissionSide::Exterior => &exterior,
            TransmissionSide::Interior => &interior,
        };
        let delta = step(selected, parameter);
        let base_value = match parameter {
            TransmissionMaterialParameter::Wavenumber => selected.wave_number,
            TransmissionMaterialParameter::SpeedOfSound => selected.speed_of_sound,
            TransmissionMaterialParameter::Density => selected.density,
        };
        let plus_value = base_value + delta;
        let minus_value = base_value - delta;
        let (exterior_plus, interior_plus) = match side {
            TransmissionSide::Exterior => {
                (perturb(&exterior, parameter, plus_value), interior.clone())
            }
            TransmissionSide::Interior => {
                (exterior.clone(), perturb(&interior, parameter, plus_value))
            }
        };
        let (exterior_minus, interior_minus) = match side {
            TransmissionSide::Exterior => {
                (perturb(&exterior, parameter, minus_value), interior.clone())
            }
            TransmissionSide::Interior => {
                (exterior.clone(), perturb(&interior, parameter, minus_value))
            }
        };
        let plus = assemble_galerkin_muller_transmission_with_global_dofs(
            &elements,
            &nodes,
            &exterior_plus,
            &interior_plus,
            BasisOrder::Linear,
            3,
            8,
        )
        .expect("plus Quad4 material assembly");
        let minus = assemble_galerkin_muller_transmission_with_global_dofs(
            &elements,
            &nodes,
            &exterior_minus,
            &interior_minus,
            BasisOrder::Linear,
            3,
            8,
        )
        .expect("minus Quad4 material assembly");
        let analytic = assemble_galerkin_muller_transmission_material_derivative_with_global_dofs(
            &elements,
            &nodes,
            &exterior,
            &interior,
            BasisOrder::Linear,
            3,
            8,
            side,
            parameter,
        )
        .expect("analytic Quad4 material derivative");
        let mut max_error = 0.0_f64;
        let mut max_reference = 0.0_f64;
        for ((actual, plus), minus) in analytic
            .as_slice()
            .iter()
            .zip(plus.matrix().as_slice())
            .zip(minus.matrix().as_slice())
        {
            let reference = (*plus - *minus) / (2.0 * delta);
            max_error = max_error.max((*actual - reference).norm());
            max_reference = max_reference.max(reference.norm());
        }
        assert!(
            max_error <= 5.0e-3 * (1.0 + max_reference),
            "{side:?} {parameter:?} Quad4 derivative error {max_error:e} reference {max_reference:e}"
        );
    }
}
