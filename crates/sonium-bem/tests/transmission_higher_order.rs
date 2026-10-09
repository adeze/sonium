use num_complex::Complex64;
use sonium_bem::check_transmission_reciprocity_contract;
use sonium_bem::core::assembly::{
    BoundaryOperatorAssemblyError, TbemQuadraturePolicy, TransmissionAssemblyError,
    assemble_boundary_operators_with_basis, assemble_boundary_operators_with_basis_shared_dofs,
    build_transmission_tbem_system_with_basis, build_transmission_tbem_system_with_basis_checked,
    build_transmission_tbem_system_with_basis_checked_from_xem,
    build_transmission_tbem_system_with_continuous_quadratic,
    build_transmission_tbem_system_with_continuous_quadratic_checked,
    build_transmission_tbem_system_with_continuous_quadratic_checked_from_xem,
    build_transmission_tbem_system_with_continuous_quadratic_from_xem,
};
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::{generate_cubed_sphere_quad_mesh, generate_icosphere_mesh};
use sonium_bem::core::solver::transmission::{
    solve_incident_transmission, solve_transmission_adjoint_many, solve_transmission_adjoint_system,
};
use sonium_bem::core::types::{BasisOrder, PhysicsParams};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::GmresConfig;
use sonium_wave::analytical::FluidSphereTransmission3d;
use sonium_xem::{AcousticMedium, ConventionRegistry, CoordinateFrame, HarmonicConvention};
use std::f64::consts::PI;

fn media() -> (PhysicsParams, PhysicsParams) {
    (
        PhysicsParams::new(420.0, 343.0, 1.21, false),
        PhysicsParams::new(420.0, 1_480.0, 1_000.0, true),
    )
}

#[test]
fn constant_transmission_matrix_satisfies_weighted_reciprocity_contract() {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let (exterior, interior) = media();
    let system = build_transmission_tbem_system_with_basis(
        &mesh.elements,
        &mesh.nodes,
        exterior.clone(),
        interior,
        BasisOrder::Constant,
    )
    .expect("constant transmission assembly");
    let probes = (0..3)
        .map(|seed| {
            (0..system.total_dofs())
                .map(|index| {
                    let phase = (index + 1 + seed * 3) as f64;
                    Complex64::new(0.15 + 0.03 * phase, -0.2 + 0.017 * (phase + seed as f64))
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    let report = check_transmission_reciprocity_contract(
        system.matrix(),
        system.interface_dofs(),
        exterior.omega,
        exterior.harmonic_factor,
        &probes,
        1.0e-6,
    )
    .expect("constant Müller matrix should satisfy weighted reciprocity");
    assert_eq!(report.probes, 3);
    assert_eq!(report.pairs, 3);
    assert!(report.relative_error <= 1.0e-6, "{report:?}");
}

#[test]
fn checked_continuous_quadratic_transmission_admits_both_medium_quadrature_paths() {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let (exterior, interior) = media();
    let permissive = TbemQuadraturePolicy {
        absolute_tolerance: 1.0e3,
        relative_tolerance: 1.0e3,
        ..TbemQuadraturePolicy::default()
    };
    let system = build_transmission_tbem_system_with_continuous_quadratic_checked(
        &mesh.elements,
        &mesh.nodes,
        exterior.clone(),
        interior.clone(),
        permissive,
    )
    .expect("permissive higher-order quadrature admission");
    assert_eq!(system.basis_order(), BasisOrder::Quadratic);
    let panel_local = build_transmission_tbem_system_with_basis_checked(
        &mesh.elements,
        &mesh.nodes,
        exterior.clone(),
        interior.clone(),
        BasisOrder::Linear,
        permissive,
    )
    .expect("panel-local higher-order quadrature admission");
    assert_eq!(panel_local.basis_order(), BasisOrder::Linear);
    let typed_panel_local = build_transmission_tbem_system_with_basis_checked_from_xem(
        &mesh.elements,
        &mesh.nodes,
        AcousticMedium::try_new(420.0, 343.0, 1.21).expect("incident medium"),
        AcousticMedium::try_new(420.0, 1_480.0, 1_000.0).expect("transmitted medium"),
        BasisOrder::Linear,
        permissive,
    )
    .expect("typed-XEM panel-local quadrature admission");
    assert_eq!(typed_panel_local.basis_order(), BasisOrder::Linear);

    let invalid = TbemQuadraturePolicy {
        regular_coarse_order: 0,
        ..TbemQuadraturePolicy::default()
    };
    let error = build_transmission_tbem_system_with_continuous_quadratic_checked(
        &mesh.elements,
        &mesh.nodes,
        exterior.clone(),
        interior.clone(),
        invalid,
    )
    .expect_err("invalid higher-order quadrature policy must fail closed");
    assert!(matches!(
        error,
        TransmissionAssemblyError::Quadrature {
            medium: "exterior",
            ..
        }
    ));

    let incident = AcousticMedium::try_new(
        exterior.frequency,
        exterior.speed_of_sound,
        exterior.density,
    )
    .expect("incident medium");
    let transmitted = AcousticMedium::try_new(
        interior.frequency,
        interior.speed_of_sound,
        interior.density,
    )
    .expect("transmitted medium");
    build_transmission_tbem_system_with_continuous_quadratic_checked_from_xem(
        &mesh.elements,
        &mesh.nodes,
        incident,
        transmitted,
        permissive,
    )
    .expect("typed-XEM higher-order quadrature admission");
}

#[test]
fn linear_transmission_assembles_and_evaluates_field() {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let original_dofs: Vec<Vec<usize>> = mesh
        .elements
        .iter()
        .map(|element| element.dof_addresses.clone())
        .collect();
    let (exterior, interior) = media();
    let system = build_transmission_tbem_system_with_basis(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
        BasisOrder::Linear,
    )
    .expect("linear transmission assembly");

    assert_eq!(system.basis_order(), BasisOrder::Linear);
    assert_eq!(system.interface_dofs(), mesh.elements.len() * 3);
    assert!(
        mesh.elements
            .iter()
            .zip(original_dofs)
            .all(|(element, original)| element.dof_addresses == original)
    );
    assert!(
        system
            .matrix()
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let incident = IncidentField::plane_wave_z();
    let solution =
        solve_incident_transmission(&system, &incident, 1.0e-9).expect("linear transmission solve");
    assert!(solution.relative_residual <= 1.0e-9);
    let field = solution
        .evaluate_exterior_field_checked(&system, &[[0.0, 0.0, 0.8]], &incident)
        .expect("linear exterior field evaluation");
    assert_eq!(field.len(), 1);
    assert!(field[0].p_total.re.is_finite() && field[0].p_total.im.is_finite());
}

#[test]
fn higher_order_operator_assembly_rejects_wrong_local_dof_count() {
    let mut mesh = generate_icosphere_mesh(0.2, 1);
    mesh.elements[0].dof_addresses = vec![0];
    let physics = PhysicsParams::new(420.0, 343.0, 1.21, false);
    let error = assemble_boundary_operators_with_basis(
        &mesh.elements,
        &mesh.nodes,
        &physics,
        BasisOrder::Linear,
    )
    .expect_err("linear operator assembly must reject constant local layout");
    assert!(matches!(
        error,
        BoundaryOperatorAssemblyError::BasisDofCount {
            element: 0,
            actual: 1,
            expected: 3,
            basis: BasisOrder::Linear,
        }
    ));
}

#[test]
fn constant_operator_assembly_rejects_wrong_local_dof_count() {
    let mut mesh = generate_icosphere_mesh(0.2, 0);
    mesh.elements[0].dof_addresses.clear();
    let physics = PhysicsParams::new(420.0, 343.0, 1.21, false);

    let error = assemble_boundary_operators_with_basis(
        &mesh.elements,
        &mesh.nodes,
        &physics,
        BasisOrder::Constant,
    )
    .expect_err("constant operator assembly must reject an empty local DOF list");
    assert!(matches!(
        error,
        BoundaryOperatorAssemblyError::BasisDofCount {
            element: 0,
            actual: 0,
            expected: 1,
            basis: BasisOrder::Constant,
        }
    ));
}

#[test]
fn constant_operator_assembly_rejects_out_of_range_node() {
    let mut mesh = generate_icosphere_mesh(0.2, 0);
    let invalid_node = mesh.nodes.len();
    mesh.elements[0].connectivity[0] = invalid_node;
    let physics = PhysicsParams::new(420.0, 343.0, 1.21, false);

    let error = assemble_boundary_operators_with_basis(
        &mesh.elements,
        &mesh.nodes,
        &physics,
        BasisOrder::Constant,
    )
    .expect_err("constant operator assembly must reject invalid connectivity");
    assert!(matches!(
        error,
        BoundaryOperatorAssemblyError::NodeOutOfRange {
            element: 0,
            local: 0,
            node,
            node_count,
        } if node == invalid_node && node_count == mesh.nodes.len()
    ));
}

#[test]
fn shared_constant_operator_assembly_rejects_duplicate_dof() {
    let mut mesh = generate_icosphere_mesh(0.2, 0);
    let duplicate = mesh.elements[0].dof_addresses[0];
    mesh.elements[1].dof_addresses[0] = duplicate;
    let physics = PhysicsParams::new(420.0, 343.0, 1.21, false);

    let error = assemble_boundary_operators_with_basis_shared_dofs(
        &mesh.elements,
        &mesh.nodes,
        &physics,
        BasisOrder::Constant,
    )
    .expect_err("constant shared-DOF assembly must reject duplicate addresses");
    assert!(matches!(
        error,
        BoundaryOperatorAssemblyError::DuplicateDof {
            first_element: 0,
            second_element: 1,
            dof,
        } if dof == duplicate
    ));
}

#[test]
fn quadratic_transmission_matrix_has_expected_block_layout() {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let (exterior, interior) = media();
    let system = build_transmission_tbem_system_with_basis(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
        BasisOrder::Quadratic,
    )
    .expect("quadratic transmission assembly");

    let expected_dofs = mesh.elements.len() * 6;
    assert_eq!(system.basis_order(), BasisOrder::Quadratic);
    assert_eq!(system.interface_dofs(), expected_dofs);
    assert_eq!(
        system.matrix().shape(),
        (2 * expected_dofs, 2 * expected_dofs)
    );
    assert!(
        system
            .matrix()
            .as_slice()
            .iter()
            .any(|value| { value.norm() > 0.0 && value.re.is_finite() && value.im.is_finite() })
    );
}

#[test]
fn continuous_quadratic_transmission_uses_shared_rows_and_solves() {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let original_dofs: Vec<Vec<usize>> = mesh
        .elements
        .iter()
        .map(|element| element.dof_addresses.clone())
        .collect();
    let (exterior, interior) = media();
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
    )
    .expect("continuous quadratic transmission assembly");

    assert_eq!(system.basis_order(), BasisOrder::Quadratic);
    assert!(system.interface_dofs() < mesh.elements.len() * 6);
    assert!(
        mesh.elements
            .iter()
            .zip(original_dofs)
            .all(|(element, original)| element.dof_addresses == original)
    );
    assert_eq!(
        system.matrix().shape(),
        (system.total_dofs(), system.total_dofs())
    );
    assert!(
        system
            .matrix()
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let incident = IncidentField::plane_wave_z();
    let solution = solve_incident_transmission(&system, &incident, 1.0e-8)
        .expect("continuous quadratic transmission solve");
    assert!(solution.relative_residual <= 1.0e-8);
    let field = solution
        .evaluate_exterior_field_checked(&system, &[[0.0, 0.0, 0.8]], &incident)
        .expect("continuous quadratic exterior field");
    assert_eq!(field.len(), 1);
    assert!(field[0].p_total.re.is_finite() && field[0].p_total.im.is_finite());
}

#[test]
fn continuous_quadratic_transmission_matches_independent_mie_field() {
    let radius = 0.2;
    let frequency = 420.0;
    let exterior_speed = 343.0;
    let interior_speed = 1_480.0;
    let exterior_density = 1.21;
    let interior_density = 1_000.0;
    let mesh = generate_icosphere_mesh(radius, 1);
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        PhysicsParams::new(frequency, exterior_speed, exterior_density, false),
        PhysicsParams::new(frequency, interior_speed, interior_density, true),
    )
    .expect("continuous quadratic transmission assembly");
    let incident = IncidentField::plane_wave_z();
    let solution = solve_incident_transmission(&system, &incident, 1.0e-8)
        .expect("continuous quadratic transmission solve");
    let evaluation_radius = 0.8;
    let angles: [f64; 8] = [0.2, 0.55, 0.9, 1.25, 1.6, 2.0, 2.4, 2.8];
    let points: Vec<[f64; 3]> = angles
        .iter()
        .map(|&theta| {
            [
                evaluation_radius * theta.sin(),
                0.0,
                evaluation_radius * theta.cos(),
            ]
        })
        .collect();
    let actual = solution
        .evaluate_exterior_field_checked(&system, &points, &incident)
        .expect("continuous quadratic exterior field");
    let mie = FluidSphereTransmission3d::try_new(
        radius,
        2.0 * PI * frequency / exterior_speed,
        2.0 * PI * frequency / interior_speed,
        exterior_density,
        interior_density,
        32,
    )
    .expect("fluid-sphere Mie reference");
    let max_relative_error = actual
        .iter()
        .zip(angles)
        .map(|(field, theta)| {
            let reference = mie
                .exterior_pressure(evaluation_radius, theta)
                .expect("valid Mie evaluation");
            (field.p_total - reference).norm()
                / field
                    .p_total
                    .norm()
                    .max(reference.norm())
                    .max(f64::MIN_POSITIVE)
        })
        .fold(0.0, f64::max);
    assert!(
        max_relative_error < 0.15,
        "continuous quadratic transmission Mie error {max_relative_error:.3e}"
    );
}

#[test]
fn continuous_quad4_transmission_accepts_curved_node_normals() {
    let mesh = generate_cubed_sphere_quad_mesh(0.2, 1);
    let (exterior, interior) = media();
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
    )
    .expect("curved Quad4 continuous transmission assembly");

    assert_eq!(system.basis_order(), BasisOrder::Quadratic);
    assert!(system.interface_dofs() < mesh.elements.len() * 9);
    assert!(
        system
            .matrix()
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let incident = IncidentField::plane_wave_z();
    let solution = solve_incident_transmission(&system, &incident, 1.0e-8)
        .expect("curved Quad4 continuous transmission solve");
    assert!(solution.relative_residual <= 1.0e-8);
}

#[test]
fn continuous_quadratic_transmission_adjoint_has_certified_single_and_batch_solves() {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let (exterior, interior) = media();
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
    )
    .expect("continuous quadratic transmission assembly");

    let dimension = system.total_dofs();
    let rhs = (0..dimension)
        .map(|index| {
            num_complex::Complex64::new(0.17 + index as f64 / 31.0, -0.09 + index as f64 / 43.0)
        })
        .collect::<Vec<_>>();
    let second_rhs = rhs
        .iter()
        .enumerate()
        .map(|(index, value)| *value + num_complex::Complex64::new(index as f64 / 67.0, 0.03))
        .collect::<Vec<_>>();

    let adjoint = solve_transmission_adjoint_system(&system, &rhs, 1.0e-8)
        .expect("continuous quadratic adjoint solve");
    assert!(adjoint.relative_residual <= 1.0e-8);

    let batch = solve_transmission_adjoint_many(&system, &[rhs.clone(), second_rhs], 1.0e-8)
        .expect("continuous quadratic adjoint batch solve");
    assert_eq!(batch.len(), 2);
    assert!(
        batch
            .iter()
            .all(|solution| solution.relative_residual <= 1.0e-8)
    );
    assert_eq!(batch[0].values(), adjoint.values());
    let adjoint_field = adjoint
        .evaluate_exterior_adjoint_pressure_checked(&system, &[[0.0, 0.0, 0.8]])
        .expect("continuous quadratic adjoint field");
    assert_eq!(adjoint_field.len(), 1);
    assert!(
        adjoint_field
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    assert!(
        adjoint
            .evaluate_exterior_adjoint_pressure_checked(&system, &[[f64::NAN, 0.0, 0.8]])
            .is_err()
    );

    let trial = (0..dimension)
        .map(|index| {
            num_complex::Complex64::new(-0.11 + index as f64 / 53.0, 0.08 - index as f64 / 71.0)
        })
        .collect::<Vec<_>>();
    let matrix_trial = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| system.matrix()[[row, column]] * trial[column])
                .sum::<num_complex::Complex64>()
        })
        .collect::<Vec<_>>();
    let lhs = matrix_trial
        .iter()
        .zip(adjoint.values())
        .map(|(left, right)| left.conj() * right)
        .sum::<num_complex::Complex64>();
    let rhs_pairing = trial
        .iter()
        .zip(&rhs)
        .map(|(left, right)| left.conj() * right)
        .sum::<num_complex::Complex64>();
    let pairing_error =
        (lhs - rhs_pairing).norm() / lhs.norm().max(rhs_pairing.norm()).max(f64::MIN_POSITIVE);
    assert!(
        pairing_error <= 1.0e-8,
        "Hermitian pairing error {pairing_error:.3e}"
    );
}

#[test]
fn continuous_quadratic_xem_adapter_preserves_convention_and_matrix_contract() {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let incident = AcousticMedium::try_new(420.0, 343.0, 1.21).expect("incident medium");
    let transmitted = AcousticMedium::try_new(420.0, 1_480.0, 1_000.0).expect("transmitted medium");
    let typed = build_transmission_tbem_system_with_continuous_quadratic_from_xem(
        &mesh.elements,
        &mesh.nodes,
        incident,
        transmitted,
    )
    .expect("typed continuous quadratic assembly");
    let direct = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        PhysicsParams::new(420.0, 343.0, 1.21, false),
        PhysicsParams::new(420.0, 1_480.0, 1_000.0, true),
    )
    .expect("direct continuous quadratic assembly");
    assert_eq!(typed.interface_dofs(), direct.interface_dofs());
    assert_eq!(typed.matrix().shape(), direct.matrix().shape());
    let max_matrix_difference = typed
        .matrix()
        .as_slice()
        .iter()
        .zip(direct.matrix().as_slice())
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0, f64::max);
    assert!(
        max_matrix_difference <= 1.0e-12,
        "typed/direct matrix difference {max_matrix_difference:.3e}"
    );
}

#[test]
fn continuous_quadratic_xem_adapter_rejects_mixed_harmonic_conventions() {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let incident = AcousticMedium::try_new(420.0, 343.0, 1.21).expect("incident medium");
    let mut transmitted =
        AcousticMedium::try_new(420.0, 1_480.0, 1_000.0).expect("transmitted medium");
    transmitted.conventions = ConventionRegistry {
        harmonic_time: HarmonicConvention::ExpPlusIwt,
        coordinate_frame: CoordinateFrame::WorldRightHanded,
        outward_normals: true,
        rms_amplitudes: true,
    };
    let error = build_transmission_tbem_system_with_continuous_quadratic_from_xem(
        &mesh.elements,
        &mesh.nodes,
        incident,
        transmitted,
    )
    .expect_err("mixed harmonic conventions must fail closed");
    assert!(error.to_string().contains("harmonic"));
}

#[test]
fn compressed_transmission_operator_preserves_forward_and_adjoint_actions() {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let (exterior, interior) = media();
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
    )
    .expect("continuous quadratic transmission assembly");
    let compressed = system
        .compress(48, 1.0e-2, 48)
        .expect("compressed transmission construction");
    assert!(compressed.compressed_block_count() > 0);
    assert!(compressed.maximum_block_relative_error().is_finite());

    let dimension = system.total_dofs();
    let trial = (0..dimension)
        .map(|index| {
            num_complex::Complex64::new(-0.13 + index as f64 / 47.0, 0.07 - index as f64 / 61.0)
        })
        .collect::<Vec<_>>();
    let dense_action = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| system.matrix()[[row, column]] * trial[column])
                .sum::<num_complex::Complex64>()
        })
        .collect::<Vec<_>>();
    let compressed_action = sonium_solvers::traits::LinearOperator::apply(&compressed, &trial);
    let forward_error = compressed_action
        .iter()
        .zip(&dense_action)
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max)
        / dense_action
            .iter()
            .map(|value| value.norm())
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);
    assert!(
        forward_error <= 2.0e-2,
        "forward compression error {forward_error:.3e}"
    );

    let dense_adjoint = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| system.matrix()[[column, row]].conj() * trial[column])
                .sum::<num_complex::Complex64>()
        })
        .collect::<Vec<_>>();
    let compressed_adjoint = compressed.apply_hermitian(&trial);
    let adjoint_error = compressed_adjoint
        .iter()
        .zip(&dense_adjoint)
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max)
        / dense_adjoint
            .iter()
            .map(|value| value.norm())
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);
    assert!(
        adjoint_error <= 2.0e-2,
        "adjoint compression error {adjoint_error:.3e}"
    );

    let dense_transpose = (0..dimension)
        .map(|row| {
            (0..dimension)
                .map(|column| system.matrix()[[column, row]] * trial[column])
                .sum::<num_complex::Complex64>()
        })
        .collect::<Vec<_>>();
    let compressed_transpose = compressed.apply_transpose(&trial);
    let transpose_error = compressed_transpose
        .iter()
        .zip(&dense_transpose)
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max)
        / dense_transpose
            .iter()
            .map(|value| value.norm())
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);
    assert!(
        transpose_error <= 2.0e-2,
        "transpose compression error {transpose_error:.3e}"
    );

    assert!(matches!(
        system.compress(0, 1.0e-2, 48),
        Err(sonium_bem::core::assembly::TransmissionCompressionError::InvalidBlockSize)
    ));
}

#[test]
fn compressed_transmission_checked_actions_reject_bad_inputs() {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        PhysicsParams::new(420.0, 343.0, 1.21, false),
        PhysicsParams::new(420.0, 1_480.0, 1_000.0, true),
    )
    .expect("continuous quadratic transmission assembly");
    let compressed = system
        .compress(48, 1.0e-2, 48)
        .expect("transmission compression");

    let short = vec![num_complex::Complex64::new(0.0, 0.0); compressed.total_dofs() - 1];
    for (operation, result) in [
        ("forward", compressed.apply_checked(&short)),
        ("transpose", compressed.apply_transpose_checked(&short)),
        ("Hermitian", compressed.apply_hermitian_checked(&short)),
    ] {
        assert!(
            matches!(
                result,
                Err(sonium_bem::core::assembly::TransmissionCompressionError::Dimension {
                    operation: actual,
                    ..
                }) if actual == operation
            ),
            "{operation} must reject a short action vector"
        );
    }

    let mut non_finite = vec![num_complex::Complex64::new(0.0, 0.0); compressed.total_dofs()];
    non_finite[3] = num_complex::Complex64::new(f64::NAN, 0.0);
    assert!(matches!(
        compressed.apply_checked(&non_finite),
        Err(
            sonium_bem::core::assembly::TransmissionCompressionError::NonFiniteInput {
                operation: "forward",
                index: 3,
            }
        )
    ));
}

#[test]
fn compressed_transmission_forward_and_adjoint_ilu_solves_are_certified() {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let (exterior, interior) = media();
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &mesh.elements,
        &mesh.nodes,
        exterior,
        interior,
    )
    .expect("continuous quadratic transmission system");
    let compressed = system
        .compress(48, 1.0e-2, 48)
        .expect("compressed transmission system");
    let dimension = compressed.total_dofs();
    let on_demand = sonium_bem::core::assembly::CompressedTransmissionTbemSystem::from_entry(
        system.interface_dofs(),
        dimension,
        48,
        1.0e-2,
        48,
        |row, column| system.matrix()[[row, column]],
    )
    .expect("on-demand compressed transmission system");
    assert_eq!(
        on_demand.compressed_block_count(),
        compressed.compressed_block_count()
    );
    let rhs = (0..dimension)
        .map(|index| num_complex::Complex64::new(0.17 + index as f64 / 43.0, -0.11))
        .collect::<Vec<_>>();
    let second_rhs = rhs
        .iter()
        .enumerate()
        .map(|(index, value)| *value + num_complex::Complex64::new(0.03 * index as f64, 0.07))
        .collect::<Vec<_>>();
    let config = GmresConfig::<f64> {
        max_iterations: 120,
        restart: 32,
        tolerance: 1.0e-10,
        exact_residual_check_interval: 1,
        ..GmresConfig::default()
    };
    let mut nearfield = DenseMatrix::zeros(dimension, dimension);
    for index in 0..dimension {
        nearfield[[index, index]] = num_complex::Complex64::new(1.0, 0.0);
    }
    let forward = compressed
        .solve_with_ilu(&rhs, &nearfield, &config, 1.0e-8)
        .expect("compressed forward solve");
    assert!(forward.relative_residual() <= 1.0e-8);
    assert!(forward.iterations() > 0);
    let on_demand_action = on_demand
        .apply_checked(&rhs)
        .expect("on-demand compressed action");
    let dense_source_action = compressed
        .apply_checked(&rhs)
        .expect("dense-source compressed action");
    let on_demand_action_error = on_demand_action
        .iter()
        .zip(&dense_source_action)
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max);
    assert!(on_demand_action_error <= 1.0e-12);

    let forward_batch = compressed
        .solve_many_with_ilu(
            &[rhs.clone(), second_rhs.clone()],
            &nearfield,
            &config,
            1.0e-8,
        )
        .expect("compressed forward batch");
    assert_eq!(forward_batch.len(), 2);
    assert!(
        forward_batch
            .iter()
            .all(|solution| solution.relative_residual() <= 1.0e-8)
    );
    assert_eq!(forward_batch[0].values(), forward.values());

    let hermitian = nearfield.clone();
    let adjoint_rhs = (0..dimension)
        .map(|index| num_complex::Complex64::new(-0.08, 0.13 + index as f64 / 59.0))
        .collect::<Vec<_>>();
    let adjoint = compressed
        .solve_hermitian_with_ilu(&adjoint_rhs, &hermitian, &config, 1.0e-8)
        .expect("compressed adjoint solve");
    assert!(adjoint.relative_residual() <= 1.0e-8);
    let adjoint_batch = compressed
        .solve_hermitian_many_with_ilu(
            &[adjoint_rhs.clone(), rhs.clone()],
            &hermitian,
            &config,
            1.0e-8,
        )
        .expect("compressed adjoint batch");
    assert_eq!(adjoint_batch.len(), 2);
    assert_eq!(adjoint_batch[0].values(), adjoint.values());
    assert!(
        adjoint_batch
            .iter()
            .all(|solution| solution.relative_residual() <= 1.0e-8)
    );

    assert!(matches!(
        compressed.solve_with_ilu(
            &rhs,
            &DenseMatrix::zeros(dimension - 1, dimension - 1),
            &config,
            1.0e-8,
        ),
        Err(sonium_bem::core::assembly::TransmissionCompressionError::NearFieldDimension { .. })
    ));
    let mut non_finite_rhs = rhs;
    non_finite_rhs[2] = num_complex::Complex64::new(f64::NAN, 0.0);
    assert!(matches!(
        compressed.solve_with_ilu(&non_finite_rhs, &nearfield, &config, 1.0e-8),
        Err(sonium_bem::core::assembly::TransmissionCompressionError::NonFiniteInput { .. })
    ));
    assert!(matches!(
        sonium_bem::core::assembly::CompressedTransmissionTbemSystem::from_entry(
            1,
            3,
            2,
            1.0e-2,
            4,
            |_row, _column| num_complex::Complex64::new(0.0, 0.0),
        ),
        Err(
            sonium_bem::core::assembly::TransmissionCompressionError::InvalidSystemDimension { .. }
        )
    ));
    assert!(matches!(
        sonium_bem::core::assembly::CompressedTransmissionTbemSystem::from_entry(
            system.interface_dofs(),
            dimension,
            48,
            1.0e-2,
            48,
            |_row, _column| num_complex::Complex64::new(f64::NAN, 0.0),
        ),
        Err(sonium_bem::core::assembly::TransmissionCompressionError::NonFiniteMatrix { .. })
    ));
}
