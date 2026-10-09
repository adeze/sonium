use num_complex::Complex64;
use sonium_bem::core::assembly::{
    GalerkinTransmissionAssemblyError, GalerkinTransmissionMaterialDerivativeError,
    assemble_galerkin_muller_transmission_material_derivative_with_global_dofs,
    assemble_galerkin_muller_transmission_with_global_dofs,
    compress_galerkin_muller_transmission_material_derivative_with_global_dofs_checked,
};
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::{
    BasisOrder, GalerkinQuadraturePolicy, PhysicsParams, TransmissionMaterialParameter,
    TransmissionSide,
};
use sonium_solvers::dense::DenseMatrix;

const FREQUENCY_HZ: f64 = 420.0;
const RADIUS_M: f64 = 0.2;
const GAUSS_ORDER: usize = 4;

fn setup_mesh() -> (Vec<sonium_bem::core::types::Element>, Vec<[f64; 3]>, usize) {
    let mut mesh = generate_icosphere_mesh(RADIUS_M, 0);
    for element in &mut mesh.elements {
        element.dof_addresses = element.connectivity.clone();
    }
    let global_dof_count = mesh.nodes.len();
    (mesh.elements, mesh.nodes, global_dof_count)
}

fn media() -> (PhysicsParams, PhysicsParams) {
    (
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false).expect("valid exterior"),
        PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true).expect("valid interior"),
    )
}

fn perturb_medium(
    medium: &PhysicsParams,
    value: f64,
    parameter: TransmissionMaterialParameter,
) -> PhysicsParams {
    let (speed, density) = match parameter {
        TransmissionMaterialParameter::Wavenumber => (medium.omega / value, medium.density),
        TransmissionMaterialParameter::SpeedOfSound => (value, medium.density),
        TransmissionMaterialParameter::Density => (medium.speed_of_sound, value),
    };
    PhysicsParams::try_new(FREQUENCY_HZ, speed, density, medium.tau < 0.0)
        .expect("valid perturbed medium")
}

fn centered_step(medium: &PhysicsParams, parameter: TransmissionMaterialParameter) -> f64 {
    match parameter {
        TransmissionMaterialParameter::Wavenumber => {
            let step = (medium.wave_number.abs() * 1.0e-5).max(1.0e-7);
            step.min(medium.wave_number * 0.1)
        }
        TransmissionMaterialParameter::SpeedOfSound => {
            (medium.speed_of_sound.abs() * 1.0e-5).max(1.0e-5)
        }
        TransmissionMaterialParameter::Density => (medium.density.abs() * 1.0e-5).max(1.0e-5),
    }
}

fn max_matrix_difference(lhs: &DenseMatrix<Complex64>, rhs: &DenseMatrix<Complex64>) -> f64 {
    assert_eq!(lhs.shape(), rhs.shape());
    lhs.as_slice()
        .iter()
        .zip(rhs.as_slice())
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0, f64::max)
}

fn direct_centered_derivative(
    elements: &[sonium_bem::core::types::Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    side: TransmissionSide,
    parameter: TransmissionMaterialParameter,
    global_dof_count: usize,
) -> DenseMatrix<Complex64> {
    let selected = match side {
        TransmissionSide::Exterior => exterior,
        TransmissionSide::Interior => interior,
    };
    let step = centered_step(selected, parameter);
    let plus_value = match parameter {
        TransmissionMaterialParameter::Wavenumber => selected.wave_number + step,
        TransmissionMaterialParameter::SpeedOfSound => selected.speed_of_sound + step,
        TransmissionMaterialParameter::Density => selected.density + step,
    };
    let minus_value = match parameter {
        TransmissionMaterialParameter::Wavenumber => selected.wave_number - step,
        TransmissionMaterialParameter::SpeedOfSound => selected.speed_of_sound - step,
        TransmissionMaterialParameter::Density => selected.density - step,
    };
    let (exterior_plus, interior_plus) = match side {
        TransmissionSide::Exterior => (
            perturb_medium(exterior, plus_value, parameter),
            interior.clone(),
        ),
        TransmissionSide::Interior => (
            exterior.clone(),
            perturb_medium(interior, plus_value, parameter),
        ),
    };
    let (exterior_minus, interior_minus) = match side {
        TransmissionSide::Exterior => (
            perturb_medium(exterior, minus_value, parameter),
            interior.clone(),
        ),
        TransmissionSide::Interior => (
            exterior.clone(),
            perturb_medium(interior, minus_value, parameter),
        ),
    };
    let plus = assemble_galerkin_muller_transmission_with_global_dofs(
        elements,
        nodes,
        &exterior_plus,
        &interior_plus,
        BasisOrder::Linear,
        GAUSS_ORDER,
        global_dof_count,
    )
    .expect("plus Galerkin assembly");
    let minus = assemble_galerkin_muller_transmission_with_global_dofs(
        elements,
        nodes,
        &exterior_minus,
        &interior_minus,
        BasisOrder::Linear,
        GAUSS_ORDER,
        global_dof_count,
    )
    .expect("minus Galerkin assembly");
    let mut derivative = DenseMatrix::zeros(plus.matrix().nrows(), plus.matrix().ncols());
    for row in 0..derivative.nrows() {
        for column in 0..derivative.ncols() {
            derivative[[row, column]] =
                (plus.matrix()[[row, column]] - minus.matrix()[[row, column]]) / (2.0 * step);
        }
    }
    derivative
}

#[test]
fn affine_tri3_galerkin_material_derivatives_match_centered_reassembly() {
    let (elements, nodes, global_dof_count) = setup_mesh();
    let (exterior, interior) = media();
    for (side, parameter) in [
        (
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::SpeedOfSound,
        ),
        (
            TransmissionSide::Interior,
            TransmissionMaterialParameter::Density,
        ),
        (
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
        ),
    ] {
        let derivative =
            assemble_galerkin_muller_transmission_material_derivative_with_global_dofs(
                &elements,
                &nodes,
                &exterior,
                &interior,
                BasisOrder::Linear,
                GAUSS_ORDER,
                global_dof_count,
                side,
                parameter,
            )
            .expect("Galerkin material derivative");
        let expected = direct_centered_derivative(
            &elements,
            &nodes,
            &exterior,
            &interior,
            side,
            parameter,
            global_dof_count,
        );
        assert!(
            max_matrix_difference(&derivative, &expected) < 1.0e-10,
            "{side:?} {parameter:?} derivative differs from centered reassembly"
        );
    }
}

#[test]
fn galerkin_material_derivative_rejects_invalid_medium_and_basis() {
    let (elements, nodes, global_dof_count) = setup_mesh();
    let (mut exterior, interior) = media();
    exterior.density = f64::NAN;
    assert!(matches!(
        assemble_galerkin_muller_transmission_material_derivative_with_global_dofs(
            &elements,
            &nodes,
            &exterior,
            &interior,
            BasisOrder::Linear,
            GAUSS_ORDER,
            global_dof_count,
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Density,
        ),
        Err(GalerkinTransmissionMaterialDerivativeError::InvalidMedium {
            side: TransmissionSide::Exterior,
            ..
        })
    ));

    let (exterior, interior) = media();
    assert!(matches!(
        assemble_galerkin_muller_transmission_material_derivative_with_global_dofs(
            &elements,
            &nodes,
            &exterior,
            &interior,
            BasisOrder::Constant,
            GAUSS_ORDER,
            global_dof_count,
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::SpeedOfSound,
        ),
        Err(GalerkinTransmissionMaterialDerivativeError::Assembly(
            GalerkinTransmissionAssemblyError::UnsupportedBasis { .. }
        ))
    ));
}

#[test]
fn compressed_galerkin_material_derivative_matches_dense_action() {
    let (elements, nodes, global_dof_count) = setup_mesh();
    let (exterior, interior) = media();
    let policy = GalerkinQuadraturePolicy {
        coarse_order: 2,
        fine_order: 3,
        absolute_tolerance: 30.0,
        relative_tolerance: 5.0e-2,
    };
    let compressed =
        compress_galerkin_muller_transmission_material_derivative_with_global_dofs_checked(
            &elements,
            &nodes,
            &exterior,
            &interior,
            BasisOrder::Linear,
            global_dof_count,
            policy,
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
            4,
            1.0e-8,
            8,
        )
        .expect("compressed Galerkin material derivative");
    let dense = assemble_galerkin_muller_transmission_material_derivative_with_global_dofs(
        &elements,
        &nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        policy.fine_order,
        global_dof_count,
        TransmissionSide::Exterior,
        TransmissionMaterialParameter::Wavenumber,
    )
    .expect("dense Galerkin material derivative");
    let input: Vec<Complex64> = (0..2 * global_dof_count)
        .map(|index| Complex64::new(0.2 + index as f64 * 0.01, -0.1 + index as f64 * 0.003))
        .collect();
    let actual = compressed
        .apply_checked(&input)
        .expect("compressed derivative action");
    let expected: Vec<Complex64> = (0..2 * global_dof_count)
        .map(|row| {
            (0..2 * global_dof_count)
                .map(|column| dense[[row, column]] * input[column])
                .sum()
        })
        .collect();
    let error = actual
        .iter()
        .zip(expected.iter())
        .map(|(value, reference)| (value - reference).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        error <= 2.0e-7,
        "compressed Galerkin derivative error {error:.3e}"
    );
}

#[test]
fn curved_quad4_pair_material_derivative_is_finite() {
    let nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.3],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.4],
        [1.0, 0.0, 1.4],
        [1.0, 1.0, 1.7],
        [0.0, 1.0, 1.4],
    ];
    let make_element = |connectivity: Vec<usize>| Element {
        connectivity,
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center: [0.5, 0.5, 0.0],
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]),
        group: 0,
        dof_addresses: vec![0, 1, 2, 3],
    };
    let source = make_element(vec![0, 1, 2, 3]);
    let field = make_element(vec![4, 5, 6, 7]);
    let (exterior, interior) = media();
    let derivative =
        sonium_bem::assemble_galerkin_muller_transmission_quad4_pair_material_derivative(
            &source,
            &field,
            &nodes,
            &exterior,
            &interior,
            BasisOrder::Linear,
            4,
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::SpeedOfSound,
        )
        .expect("curved Quad4 pair material derivative");
    assert_eq!(derivative.shape(), (8, 8));
    assert!(
        derivative
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let mut invalid_exterior = exterior.clone();
    invalid_exterior.density = f64::NAN;
    assert!(matches!(
        sonium_bem::assemble_galerkin_muller_transmission_quad4_pair_material_derivative(
            &source,
            &field,
            &nodes,
            &invalid_exterior,
            &interior,
            BasisOrder::Linear,
            4,
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Density,
        ),
        Err(GalerkinTransmissionMaterialDerivativeError::InvalidMedium {
            side: TransmissionSide::Exterior,
            ..
        })
    ));
}
