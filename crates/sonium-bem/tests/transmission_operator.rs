// Independent physical/kernel checks for the production Müller transmission
// operator.
//
// The reference below deliberately evaluates the off-diagonal 3-D Green
// kernels with an independent tensor-product/Duffy quadrature.  It does not
// call sonium-bem's integration routines, so the test checks both the
// medium-weighted block wiring and the production quadrature path.

use num_complex::Complex64;
use sonium_bem::core::distributed::plan_distributed_rows;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::Element;
use sonium_bem::{
    BasisOrder, PhysicsParams, TransmissionAssemblyError, TransmissionMaterialDerivativeError,
    TransmissionMaterialParameter, TransmissionSide, build_transmission_tbem_system,
    build_transmission_tbem_system_from_xem, build_transmission_tbem_system_with_basis,
    build_transmission_tbem_system_with_continuous_quadratic,
};
use sonium_solvers::traits::LinearOperator;
use sonium_xem::AcousticMedium;

const FREQUENCY_HZ: f64 = 420.0;
const RADIUS_M: f64 = 0.2;
const MAX_BLOCK_RELATIVE_ERROR: f64 = 2.0e-7;

fn setup_mesh() -> (Vec<Element>, Vec<[f64; 3]>) {
    let mesh = generate_icosphere_mesh(RADIUS_M, 1);
    let mut elements = mesh.elements;
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![index];
    }
    (elements, mesh.nodes)
}

fn setup_small_mesh() -> (Vec<Element>, Vec<[f64; 3]>) {
    let mesh = generate_icosphere_mesh(RADIUS_M, 0);
    let mut elements = mesh.elements;
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![index];
    }
    (elements, mesh.nodes)
}

fn gauss_unit_interval() -> ([f64; 8], [f64; 8]) {
    (
        [
            0.0198550717512319,
            0.1016667612931866,
            0.2372337950418355,
            0.4082826787521751,
            0.5917173212478249,
            0.7627662049581645,
            0.8983332387068134,
            0.9801449282487681,
        ],
        [
            0.0506142681451881,
            0.1111905172266872,
            0.1568533229389436,
            0.1813418916891809,
            0.1813418916891809,
            0.1568533229389436,
            0.1111905172266872,
            0.0506142681451881,
        ],
    )
}

fn reference_panel_kernels(
    source: &Element,
    field: &Element,
    nodes: &[[f64; 3]],
    physics: &PhysicsParams,
) -> [Complex64; 4] {
    let a = nodes[field.connectivity[0]];
    let b = nodes[field.connectivity[1]];
    let c = nodes[field.connectivity[2]];
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let jacobian = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    let (points, weights) = gauss_unit_interval();
    let mut result = [Complex64::default(); 4];

    for (t, &wt) in points.iter().zip(weights.iter()) {
        for (s, &ws) in points.iter().zip(weights.iter()) {
            let u = *t;
            let v = (1.0 - *t) * *s;
            let point = [
                a[0] + u * ab[0] + v * ac[0],
                a[1] + u * ab[1] + v * ac[1],
                a[2] + u * ab[2] + v * ac[2],
            ];
            let diff = [
                point[0] - source.center[0],
                point[1] - source.center[1],
                point[2] - source.center[2],
            ];
            let radius = (diff[0] * diff[0] + diff[1] * diff[1] + diff[2] * diff[2]).sqrt();
            if radius <= 1.0e-14 {
                continue;
            }
            let unit = [diff[0] / radius, diff[1] / radius, diff[2] / radius];
            let weight = wt * ws * (1.0 - *t) * jacobian;
            let phase = physics.harmonic_factor * physics.wave_number * radius;
            let green = Complex64::new(phase.cos(), phase.sin())
                * (weight / (4.0 * std::f64::consts::PI * radius));
            let gradient_factor =
                Complex64::new(-1.0 / radius, physics.harmonic_factor * physics.wave_number);
            let field_normal_projection =
                unit[0] * field.normal[0] + unit[1] * field.normal[1] + unit[2] * field.normal[2];
            let source_normal_projection = -(unit[0] * source.normal[0]
                + unit[1] * source.normal[1]
                + unit[2] * source.normal[2]);
            let field_kernel = green * gradient_factor * field_normal_projection;
            let source_kernel = green * gradient_factor * source_normal_projection;
            let rq = field_normal_projection * source_normal_projection;
            let normal_dot = source.normal[0] * field.normal[0]
                + source.normal[1] * field.normal[1]
                + source.normal[2] * field.normal[2];
            let radius_squared = radius * radius;
            let hypersingular_factor = Complex64::new(
                (3.0 / radius_squared - physics.wave_number * physics.wave_number) * rq
                    + normal_dot / radius_squared,
                -physics.harmonic_factor * physics.wave_number / radius * (3.0 * rq + normal_dot),
            );

            result[0] += green;
            result[1] += field_kernel;
            result[2] += source_kernel;
            result[3] += green * hypersingular_factor;
        }
    }
    result
}

fn relative_error(actual: Complex64, expected: Complex64) -> f64 {
    (actual - expected).norm() / actual.norm().max(expected.norm()).max(f64::MIN_POSITIVE)
}

fn central_difference_matrix(
    plus: &sonium_bem::TransmissionTbemSystem,
    minus: &sonium_bem::TransmissionTbemSystem,
    step: f64,
) -> Vec<Complex64> {
    plus.matrix()
        .as_slice()
        .iter()
        .zip(minus.matrix().as_slice())
        .map(|(plus, minus)| (*plus - *minus) / (2.0 * step))
        .collect()
}

fn max_matrix_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| {
            (*actual - *expected).norm() / actual.norm().max(expected.norm()).max(1.0e-3)
        })
        .fold(0.0, f64::max)
}

fn rebuild_with_mediums(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: PhysicsParams,
    interior: PhysicsParams,
) -> sonium_bem::TransmissionTbemSystem {
    build_transmission_tbem_system(elements, nodes, exterior, interior)
        .expect("finite transmission rebuild")
}

#[test]
fn transmission_material_derivatives_match_speed_and_density_finite_differences_v1() {
    let (elements, nodes) = setup_small_mesh();
    let exterior =
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false).expect("valid exterior physics");
    let interior = PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
        .expect("valid interior physics");
    let system = rebuild_with_mediums(&elements, &nodes, exterior.clone(), interior.clone());

    let speed_step = 1.0e-2;
    let exterior_speed_plus = rebuild_with_mediums(
        &elements,
        &nodes,
        PhysicsParams::try_new(
            FREQUENCY_HZ,
            exterior.speed_of_sound + speed_step,
            exterior.density,
            false,
        )
        .expect("exterior speed plus"),
        interior.clone(),
    );
    let exterior_speed_minus = rebuild_with_mediums(
        &elements,
        &nodes,
        PhysicsParams::try_new(
            FREQUENCY_HZ,
            exterior.speed_of_sound - speed_step,
            exterior.density,
            false,
        )
        .expect("exterior speed minus"),
        interior.clone(),
    );
    let exterior_speed_fd =
        central_difference_matrix(&exterior_speed_plus, &exterior_speed_minus, speed_step);
    let exterior_speed = system
        .matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::SpeedOfSound,
        )
        .expect("exterior speed derivative");
    assert!(
        max_matrix_error(exterior_speed.as_slice(), &exterior_speed_fd) <= 4.0e-5,
        "exterior speed derivative mismatch"
    );

    let exterior_wavenumber = system
        .matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
        )
        .expect("exterior wavenumber derivative");
    let speed_scale = -exterior.wave_number / exterior.speed_of_sound;
    let scaled_wavenumber: Vec<Complex64> = exterior_wavenumber
        .as_slice()
        .iter()
        .map(|value| *value * speed_scale)
        .collect();
    assert!(
        max_matrix_error(exterior_speed.as_slice(), &scaled_wavenumber) <= 2.0e-12,
        "speed derivative must follow dk/dc chain rule"
    );

    let interior_speed_plus = rebuild_with_mediums(
        &elements,
        &nodes,
        exterior.clone(),
        PhysicsParams::try_new(
            FREQUENCY_HZ,
            interior.speed_of_sound + speed_step,
            interior.density,
            true,
        )
        .expect("interior speed plus"),
    );
    let interior_speed_minus = rebuild_with_mediums(
        &elements,
        &nodes,
        exterior.clone(),
        PhysicsParams::try_new(
            FREQUENCY_HZ,
            interior.speed_of_sound - speed_step,
            interior.density,
            true,
        )
        .expect("interior speed minus"),
    );
    let interior_speed_fd =
        central_difference_matrix(&interior_speed_plus, &interior_speed_minus, speed_step);
    let interior_speed = system
        .matrix_material_derivative(
            TransmissionSide::Interior,
            TransmissionMaterialParameter::SpeedOfSound,
        )
        .expect("interior speed derivative");
    assert!(
        max_matrix_error(interior_speed.as_slice(), &interior_speed_fd) <= 4.0e-5,
        "interior speed derivative mismatch"
    );

    let density_step = 1.0e-5;
    let exterior_density_plus = rebuild_with_mediums(
        &elements,
        &nodes,
        PhysicsParams::try_new(
            FREQUENCY_HZ,
            exterior.speed_of_sound,
            exterior.density + density_step,
            false,
        )
        .expect("exterior density plus"),
        interior.clone(),
    );
    let exterior_density_minus = rebuild_with_mediums(
        &elements,
        &nodes,
        PhysicsParams::try_new(
            FREQUENCY_HZ,
            exterior.speed_of_sound,
            exterior.density - density_step,
            false,
        )
        .expect("exterior density minus"),
        interior.clone(),
    );
    let exterior_density_fd = central_difference_matrix(
        &exterior_density_plus,
        &exterior_density_minus,
        density_step,
    );
    let exterior_density = system
        .matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Density,
        )
        .expect("exterior density derivative");
    let exterior_density_error =
        max_matrix_error(exterior_density.as_slice(), &exterior_density_fd);
    assert!(
        exterior_density_error <= 2.0e-8,
        "exterior density derivative mismatch: {exterior_density_error:.3e}"
    );

    let interior_density_plus = rebuild_with_mediums(
        &elements,
        &nodes,
        exterior.clone(),
        PhysicsParams::try_new(
            FREQUENCY_HZ,
            interior.speed_of_sound,
            interior.density + density_step,
            true,
        )
        .expect("interior density plus"),
    );
    let interior_density_minus = rebuild_with_mediums(
        &elements,
        &nodes,
        exterior.clone(),
        PhysicsParams::try_new(
            FREQUENCY_HZ,
            interior.speed_of_sound,
            interior.density - density_step,
            true,
        )
        .expect("interior density minus"),
    );
    let interior_density_fd = central_difference_matrix(
        &interior_density_plus,
        &interior_density_minus,
        density_step,
    );
    let interior_density = system
        .matrix_material_derivative(
            TransmissionSide::Interior,
            TransmissionMaterialParameter::Density,
        )
        .expect("interior density derivative");
    let interior_density_error =
        max_matrix_error(interior_density.as_slice(), &interior_density_fd);
    assert!(
        interior_density_error <= 2.0e-8,
        "interior density derivative mismatch: {interior_density_error:.3e}"
    );
}

#[test]
fn transmission_material_derivatives_validate_and_cover_higher_order_systems_v1() {
    let (elements, nodes) = setup_small_mesh();
    let exterior =
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false).expect("valid exterior physics");
    let interior = PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
        .expect("valid interior physics");
    let mut system = rebuild_with_mediums(&elements, &nodes, exterior.clone(), interior.clone());
    system.exterior.density = f64::NAN;
    assert!(matches!(
        system.matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Density,
        ),
        Err(TransmissionMaterialDerivativeError::InvalidMedium {
            side: TransmissionSide::Exterior,
            ..
        })
    ));

    let mut higher_order_elements = elements;
    for (index, element) in higher_order_elements.iter_mut().enumerate() {
        element.dof_addresses = vec![3 * index, 3 * index + 1, 3 * index + 2];
    }
    let higher_order = build_transmission_tbem_system_with_basis(
        &higher_order_elements,
        &nodes,
        exterior,
        interior,
        BasisOrder::Linear,
    )
    .expect("linear transmission system");
    let higher_order_wavenumber = higher_order
        .matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
        )
        .expect("linear higher-order wavenumber derivative");
    assert!(
        higher_order_wavenumber
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );
    let higher_order_speed_step = 343.0e-5;
    let higher_order_speed_plus = build_transmission_tbem_system_with_basis(
        &higher_order_elements,
        &nodes,
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0 + higher_order_speed_step, 1.21, false)
            .expect("linear higher-order speed plus"),
        PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
            .expect("linear higher-order interior plus"),
        BasisOrder::Linear,
    )
    .expect("linear higher-order speed plus system");
    let higher_order_speed_minus = build_transmission_tbem_system_with_basis(
        &higher_order_elements,
        &nodes,
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0 - higher_order_speed_step, 1.21, false)
            .expect("linear higher-order speed minus"),
        PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
            .expect("linear higher-order interior minus"),
        BasisOrder::Linear,
    )
    .expect("linear higher-order speed minus system");
    let higher_order_speed_fd = central_difference_matrix(
        &higher_order_speed_plus,
        &higher_order_speed_minus,
        higher_order_speed_step,
    );
    let higher_order_speed = higher_order
        .matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::SpeedOfSound,
        )
        .expect("linear higher-order speed derivative");
    assert!(
        max_matrix_error(higher_order_speed.as_slice(), &higher_order_speed_fd) <= 2.0e-8,
        "linear higher-order speed derivative mismatch"
    );

    let higher_order_density = higher_order
        .matrix_material_derivative(
            TransmissionSide::Interior,
            TransmissionMaterialParameter::Density,
        )
        .expect("linear higher-order density derivative");
    assert!(
        higher_order_density
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let (continuous_elements, continuous_nodes) = setup_small_mesh();
    let continuous = build_transmission_tbem_system_with_continuous_quadratic(
        &continuous_elements,
        &continuous_nodes,
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false)
            .expect("continuous exterior physics"),
        PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
            .expect("continuous interior physics"),
    )
    .expect("continuous quadratic transmission system");
    let continuous_speed = continuous
        .matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::SpeedOfSound,
        )
        .expect("continuous quadratic speed derivative");
    assert!(
        continuous_speed
            .as_slice()
            .iter()
            .all(|value| value.re.is_finite() && value.im.is_finite())
    );

    let continuous_speed_step = 343.0e-5;
    let continuous_plus = build_transmission_tbem_system_with_continuous_quadratic(
        &continuous_elements,
        &continuous_nodes,
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0 + continuous_speed_step, 1.21, false)
            .expect("continuous plus exterior physics"),
        PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
            .expect("continuous plus interior physics"),
    )
    .expect("continuous plus quadratic transmission system");
    let continuous_minus = build_transmission_tbem_system_with_continuous_quadratic(
        &continuous_elements,
        &continuous_nodes,
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0 - continuous_speed_step, 1.21, false)
            .expect("continuous minus exterior physics"),
        PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
            .expect("continuous minus interior physics"),
    )
    .expect("continuous minus quadratic transmission system");
    let continuous_speed_fd =
        central_difference_matrix(&continuous_plus, &continuous_minus, continuous_speed_step);
    let continuous_speed_error =
        max_matrix_error(continuous_speed.as_slice(), &continuous_speed_fd);
    assert!(
        continuous_speed_error <= 2.0e-8,
        "continuous quadratic speed derivative mismatch: {continuous_speed_error:.3e}"
    );
}

#[test]
fn compressed_transmission_material_derivative_matches_dense_action_v1() {
    let (elements, nodes) = setup_small_mesh();
    let exterior =
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false).expect("valid exterior physics");
    let interior = PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
        .expect("valid interior physics");
    let system = build_transmission_tbem_system(&elements, &nodes, exterior, interior)
        .expect("small transmission system");

    let dense_derivative = system
        .matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
        )
        .expect("dense material derivative");
    let compressed = system
        .compress_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
            4,
            1.0e-8,
            8,
        )
        .expect("compressed material derivative");
    let input: Vec<Complex64> = (0..system.total_dofs())
        .map(|index| Complex64::new(0.2 + index as f64 * 0.013, -0.1 + index as f64 * 0.007))
        .collect();
    let actual = compressed
        .apply_checked(&input)
        .expect("valid compressed derivative action");
    let expected: Vec<Complex64> = (0..system.total_dofs())
        .map(|row| {
            (0..system.total_dofs())
                .map(|column| dense_derivative[[row, column]] * input[column])
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
        "compressed derivative action error {error:.3e}"
    );

    let mut linear_elements = elements;
    for (index, element) in linear_elements.iter_mut().enumerate() {
        element.dof_addresses = vec![3 * index, 3 * index + 1, 3 * index + 2];
    }
    let linear = build_transmission_tbem_system_with_basis(
        &linear_elements,
        &nodes,
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false).expect("valid exterior"),
        PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true).expect("valid interior"),
        BasisOrder::Linear,
    )
    .expect("linear transmission system");
    let linear_dense = linear
        .matrix_material_derivative(
            TransmissionSide::Interior,
            TransmissionMaterialParameter::Density,
        )
        .expect("linear dense material derivative");
    let linear_compressed = linear
        .compress_material_derivative(
            TransmissionSide::Interior,
            TransmissionMaterialParameter::Density,
            8,
            1.0e-8,
            8,
        )
        .expect("linear compressed material derivative");
    let linear_input: Vec<Complex64> = (0..linear.total_dofs())
        .map(|index| Complex64::new(0.1 + index as f64 * 0.003, 0.05 - index as f64 * 0.002))
        .collect();
    let linear_actual = linear_compressed
        .apply_checked(&linear_input)
        .expect("valid linear compressed derivative action");
    let linear_expected: Vec<Complex64> = (0..linear.total_dofs())
        .map(|row| {
            (0..linear.total_dofs())
                .map(|column| linear_dense[[row, column]] * linear_input[column])
                .sum()
        })
        .collect();
    let linear_error = linear_actual
        .iter()
        .zip(linear_expected.iter())
        .map(|(value, reference)| (value - reference).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        linear_error <= 2.0e-7,
        "linear compressed derivative action error {linear_error:.3e}"
    );
}

#[test]
fn kernel_native_transmission_material_derivative_matches_dense_action_v1() {
    let (elements, nodes) = setup_small_mesh();
    let exterior =
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false).expect("valid exterior physics");
    let interior = PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
        .expect("valid interior physics");
    let system = build_transmission_tbem_system(&elements, &nodes, exterior, interior)
        .expect("small transmission system");
    let input: Vec<Complex64> = (0..system.total_dofs())
        .map(|index| Complex64::new(0.2 + index as f64 * 0.013, -0.1 + index as f64 * 0.007))
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
            TransmissionMaterialParameter::Density,
        ),
    ] {
        let dense = system
            .matrix_material_derivative(side, parameter)
            .expect("dense material derivative");
        let compressed = system
            .compress_kernel_native_material_derivative(side, parameter, 4, 1.0e-8, 8)
            .expect("kernel-native compressed material derivative");
        let actual = compressed
            .apply_checked(&input)
            .expect("valid compressed derivative action");
        let expected: Vec<Complex64> = (0..system.total_dofs())
            .map(|row| {
                (0..system.total_dofs())
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
            "kernel-native compressed derivative action error for {side:?}/{parameter:?}: {error:.3e}"
        );

        let transpose_actual = compressed
            .apply_transpose_checked(&input)
            .expect("valid compressed transpose derivative action");
        let transpose_expected: Vec<Complex64> = (0..system.total_dofs())
            .map(|row| {
                (0..system.total_dofs())
                    .map(|column| dense[[column, row]] * input[column])
                    .sum()
            })
            .collect();
        let transpose_error = transpose_actual
            .iter()
            .zip(transpose_expected.iter())
            .map(|(value, reference)| (value - reference).norm())
            .fold(0.0_f64, f64::max);
        assert!(
            transpose_error <= 2.0e-7,
            "kernel-native compressed transpose derivative action error for {side:?}/{parameter:?}: {transpose_error:.3e}"
        );
    }
}

#[test]
fn kernel_native_higher_order_transmission_value_and_derivative_match_dense_v1() {
    let (elements, nodes) = setup_small_mesh();
    let exterior =
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false).expect("valid exterior physics");
    let interior = PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
        .expect("valid interior physics");
    let system = build_transmission_tbem_system_with_basis(
        &elements,
        &nodes,
        exterior,
        interior,
        BasisOrder::Linear,
    )
    .expect("linear transmission system");
    let compressed_derivative = system
        .compress_kernel_native_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
            4,
            1.0e-8,
            8,
        )
        .expect("kernel-native higher-order derivative");
    let dense_derivative = system
        .matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
        )
        .expect("dense higher-order derivative");
    let input: Vec<Complex64> = (0..system.total_dofs())
        .map(|index| Complex64::new(0.2 + index as f64 * 0.003, -0.1 + index as f64 * 0.002))
        .collect();
    let actual = compressed_derivative
        .apply_checked(&input)
        .expect("higher-order derivative action");
    let expected: Vec<Complex64> = (0..system.total_dofs())
        .map(|row| {
            (0..system.total_dofs())
                .map(|column| dense_derivative[[row, column]] * input[column])
                .sum()
        })
        .collect();
    let derivative_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(value, reference)| (value - reference).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        derivative_error <= 2.0e-7,
        "higher-order kernel-native derivative action error {derivative_error:.3e}"
    );

    let compressed_value = system
        .compress_kernel_native(4, 1.0e-8, 8)
        .expect("kernel-native higher-order value");
    let actual = compressed_value
        .apply_checked(&input)
        .expect("higher-order value action");
    let expected: Vec<Complex64> = (0..system.total_dofs())
        .map(|row| {
            (0..system.total_dofs())
                .map(|column| system.matrix()[[row, column]] * input[column])
                .sum()
        })
        .collect();
    let value_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(value, reference)| (value - reference).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        value_error <= 2.0e-7,
        "higher-order kernel-native value action error {value_error:.3e}"
    );
}

#[test]
fn kernel_native_quadratic_transmission_material_derivative_matches_dense_v1() {
    let (elements, nodes) = setup_small_mesh();
    let exterior =
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false).expect("valid exterior physics");
    let interior = PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
        .expect("valid interior physics");
    let system = build_transmission_tbem_system_with_basis(
        &elements,
        &nodes,
        exterior,
        interior,
        BasisOrder::Quadratic,
    )
    .expect("quadratic transmission system");
    let compressed = system
        .compress_kernel_native_material_derivative(
            TransmissionSide::Interior,
            TransmissionMaterialParameter::Density,
            8,
            1.0e-8,
            8,
        )
        .expect("quadratic kernel-native density derivative");
    let dense = system
        .matrix_material_derivative(
            TransmissionSide::Interior,
            TransmissionMaterialParameter::Density,
        )
        .expect("quadratic dense density derivative");
    let input: Vec<Complex64> = (0..system.total_dofs())
        .map(|index| Complex64::new(0.1 + index as f64 * 0.001, -0.04 + index as f64 * 0.0007))
        .collect();
    let actual = compressed
        .apply_checked(&input)
        .expect("quadratic density derivative action");
    let expected: Vec<Complex64> = (0..system.total_dofs())
        .map(|row| {
            (0..system.total_dofs())
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
        "quadratic derivative action error {error:.3e}"
    );
}

#[test]
fn kernel_native_shared_quadratic_transmission_value_and_derivative_match_dense_v1() {
    let (elements, nodes) = setup_small_mesh();
    let exterior =
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false).expect("valid exterior physics");
    let interior = PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
        .expect("valid interior physics");
    let system = build_transmission_tbem_system_with_continuous_quadratic(
        &elements, &nodes, exterior, interior,
    )
    .expect("continuous quadratic transmission system");
    let input: Vec<Complex64> = (0..system.total_dofs())
        .map(|index| Complex64::new(0.08 + index as f64 * 0.002, -0.03 + index as f64 * 0.001))
        .collect();

    let compressed = system
        .compress_kernel_native(system.total_dofs(), 1.0e-8, 8)
        .expect("shared quadratic kernel-native value");
    let actual = compressed
        .apply_checked(&input)
        .expect("shared quadratic value action");
    let expected: Vec<Complex64> = (0..system.total_dofs())
        .map(|row| {
            (0..system.total_dofs())
                .map(|column| system.matrix()[[row, column]] * input[column])
                .sum()
        })
        .collect();
    let value_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(value, reference)| (value - reference).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        value_error <= 2.0e-7,
        "shared quadratic value action error {value_error:.3e}"
    );

    let compressed = system
        .compress_kernel_native_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
            system.total_dofs(),
            1.0e-8,
            8,
        )
        .expect("shared quadratic kernel-native derivative");
    let dense = system
        .matrix_material_derivative(
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Wavenumber,
        )
        .expect("shared quadratic dense derivative");
    let actual = compressed
        .apply_checked(&input)
        .expect("shared quadratic derivative action");
    let expected: Vec<Complex64> = (0..system.total_dofs())
        .map(|row| {
            (0..system.total_dofs())
                .map(|column| dense[[row, column]] * input[column])
                .sum()
        })
        .collect();
    let derivative_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(value, reference)| (value - reference).norm())
        .fold(0.0_f64, f64::max);
    assert!(
        derivative_error <= 2.0e-7,
        "shared quadratic derivative action error {derivative_error:.3e}"
    );
}

#[test]
fn typed_transmission_operator_matches_independent_kernel_blocks_v1() {
    let (elements, nodes) = setup_mesh();
    let incident = AcousticMedium::try_new(FREQUENCY_HZ, 343.0, 1.21).expect("incident medium");
    let transmitted =
        AcousticMedium::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0).expect("transmitted medium");
    let system = build_transmission_tbem_system_from_xem(&elements, &nodes, incident, transmitted)
        .expect("typed transmission assembly");

    let source_index = 0;
    let field_index = elements
        .iter()
        .enumerate()
        .skip(1)
        .find(|(_, field)| {
            let dx = field.center[0] - elements[source_index].center[0];
            let dy = field.center[1] - elements[source_index].center[1];
            let dz = field.center[2] - elements[source_index].center[2];
            (dx * dx + dy * dy + dz * dz).sqrt() > 0.25
        })
        .map(|(index, _)| index)
        .expect("icosphere must contain a well-separated panel pair");

    let exterior =
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false).expect("exterior physics");
    let interior =
        PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true).expect("interior physics");
    let exterior_reference = reference_panel_kernels(
        &elements[source_index],
        &elements[field_index],
        &nodes,
        &exterior,
    );
    let interior_reference = reference_panel_kernels(
        &elements[source_index],
        &elements[field_index],
        &nodes,
        &interior,
    );
    let i_omega = Complex64::new(0.0, exterior.harmonic_factor * exterior.omega);
    let expected = [
        interior_reference[1] + exterior_reference[1],
        -(i_omega
            * (interior.density * interior_reference[0]
                + exterior.density * exterior_reference[0])),
        interior_reference[3] / interior.density + exterior_reference[3] / exterior.density,
        -i_omega * (interior_reference[2] + exterior_reference[2]),
    ];
    let n = system.interface_dofs();
    let actual = [
        system.matrix()[[source_index, field_index]],
        system.matrix()[[source_index, n + field_index]],
        system.matrix()[[n + source_index, field_index]],
        system.matrix()[[n + source_index, n + field_index]],
    ];
    for (index, (actual, expected)) in actual.into_iter().zip(expected).enumerate() {
        let error = relative_error(actual, expected);
        assert!(
            error <= MAX_BLOCK_RELATIVE_ERROR,
            "Müller block {index} relative error {error:.3e} exceeds {MAX_BLOCK_RELATIVE_ERROR:.3e}"
        );
    }
}

#[test]
fn kernel_native_transmission_compression_matches_dense_action_v1() {
    let (elements, nodes) = setup_small_mesh();
    let exterior =
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false).expect("valid exterior physics");
    let interior = PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
        .expect("valid interior physics");
    let system = build_transmission_tbem_system(&elements, &nodes, exterior, interior)
        .expect("small transmission system");
    let compressed = system
        .compress_kernel_native(4, 1.0e-8, 8)
        .expect("kernel-native compressed transmission system");
    let input: Vec<Complex64> = (0..system.total_dofs())
        .map(|index| Complex64::new(0.2 + index as f64 * 0.013, -0.1 + index as f64 * 0.007))
        .collect();
    let actual = compressed
        .apply_checked(&input)
        .expect("finite kernel-native action");
    let expected: Vec<Complex64> = (0..system.total_dofs())
        .map(|row| {
            (0..system.total_dofs())
                .map(|column| system.matrix()[[row, column]] * input[column])
                .sum()
        })
        .collect();
    let max_error = actual
        .iter()
        .zip(expected.iter())
        .map(|(value, reference)| (*value - *reference).norm())
        .fold(0.0_f64, f64::max);
    let scale = expected
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    assert!(
        max_error / scale <= 2.0e-7,
        "kernel-native action mismatch {:.3e}",
        max_error / scale
    );

    let transpose_input: Vec<Complex64> = (0..system.total_dofs())
        .map(|index| Complex64::new(-0.15 + index as f64 * 0.009, 0.04 - index as f64 * 0.005))
        .collect();
    let transpose_actual = compressed
        .apply_transpose_checked(&transpose_input)
        .expect("finite kernel-native transpose action");
    let transpose_expected: Vec<Complex64> = (0..system.total_dofs())
        .map(|row| {
            (0..system.total_dofs())
                .map(|column| system.matrix()[[column, row]] * transpose_input[column])
                .sum()
        })
        .collect();
    let transpose_error = transpose_actual
        .iter()
        .zip(transpose_expected.iter())
        .map(|(value, reference)| (*value - *reference).norm())
        .fold(0.0_f64, f64::max);
    let transpose_scale = transpose_expected
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    assert!(
        transpose_error / transpose_scale <= 2.0e-7,
        "kernel-native transpose mismatch {:.3e}",
        transpose_error / transpose_scale
    );
}

#[test]
fn distributed_transmission_row_actions_match_global_forward_and_transpose_v1() {
    let (elements, nodes) = setup_small_mesh();
    let exterior =
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false).expect("valid exterior physics");
    let interior = PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
        .expect("valid interior physics");
    let system = build_transmission_tbem_system(&elements, &nodes, exterior, interior)
        .expect("small transmission system");
    let compressed = system
        .compress_kernel_native(4, 1.0e-8, 8)
        .expect("kernel-native compressed transmission system");
    let input: Vec<Complex64> = (0..system.total_dofs())
        .map(|index| Complex64::new(0.1 + index as f64 * 0.011, -0.03 + index as f64 * 0.004))
        .collect();
    let global_forward = compressed
        .apply_checked(&input)
        .expect("global forward action");
    let global_transpose = compressed
        .apply_transpose_checked(&input)
        .expect("global transpose action");
    let plan = plan_distributed_rows(&elements, 4).expect("deterministic row plan");
    let rows_by_rank = plan
        .transmission_trace_rows(system.interface_dofs())
        .expect("transmission row ownership");
    for rows in rows_by_rank {
        let local_forward = compressed
            .apply_rows_checked(&input, &rows)
            .expect("local forward action");
        let local_transpose = compressed
            .apply_transpose_rows_checked(&input, &rows)
            .expect("local transpose action");
        assert_eq!(local_forward.len(), rows.len());
        for (slot, &row) in rows.iter().enumerate() {
            assert!(
                (local_forward[slot] - global_forward[row]).norm() <= 1.0e-10,
                "forward row {row} mismatch: {}",
                (local_forward[slot] - global_forward[row]).norm()
            );
            assert!(
                (local_transpose[slot] - global_transpose[row]).norm() <= 1.0e-10,
                "transpose row {row} mismatch: {}",
                (local_transpose[slot] - global_transpose[row]).norm()
            );
        }
    }
    let error = compressed
        .apply_rows_checked(&input, &[system.total_dofs()])
        .expect_err("out-of-range row must fail closed");
    assert!(matches!(
        error,
        sonium_bem::TransmissionCompressionError::InvalidRow { .. }
    ));
}

#[test]
fn compressed_transmission_hermitian_row_blocks_match_global_actions_v1() {
    let (elements, nodes) = setup_mesh();
    let exterior =
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false).expect("exterior medium");
    let interior =
        PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true).expect("interior medium");
    let system = build_transmission_tbem_system(&elements, &nodes, exterior, interior)
        .expect("transmission assembly");
    let compressed = system
        .compress_kernel_native(4, 1.0e-8, 8)
        .expect("kernel-native compression");

    let mut input = sonium_solvers::dense::DenseMatrix::zeros(system.total_dofs(), 3);
    for row in 0..input.nrows() {
        for rhs in 0..input.ncols() {
            input[[row, rhs]] = Complex64::new(
                0.07 + 0.011 * row as f64 + 0.013 * rhs as f64,
                -0.02 + 0.004 * row as f64 - 0.009 * rhs as f64,
            );
        }
    }
    let rows = [system.total_dofs() - 1, 1, system.total_dofs() - 1];
    let selected = compressed
        .apply_hermitian_rows_block_checked(&input, &rows)
        .expect("Hermitian row block action");
    assert_eq!(selected.nrows(), rows.len());
    assert_eq!(selected.ncols(), input.ncols());

    for rhs in 0..input.ncols() {
        let column = (0..input.nrows())
            .map(|row| input[[row, rhs]])
            .collect::<Vec<_>>();
        let expected = compressed.apply_hermitian(&column);
        let scalar_selected = compressed
            .apply_hermitian_rows_checked(&column, &rows)
            .expect("Hermitian scalar row action");
        for (slot, &row) in rows.iter().enumerate() {
            assert!(
                (selected[[slot, rhs]] - expected[row]).norm() <= 1.0e-12,
                "Hermitian row {row}, RHS {rhs} mismatch"
            );
            assert!(
                (scalar_selected[slot] - selected[[slot, rhs]]).norm() <= 1.0e-12,
                "Hermitian scalar/block row {row}, RHS {rhs} mismatch"
            );
        }
    }

    let wrong_rows = sonium_solvers::dense::DenseMatrix::zeros(system.total_dofs() - 1, 1);
    assert!(matches!(
        compressed.apply_hermitian_rows_block_checked(&wrong_rows, &rows),
        Err(sonium_bem::TransmissionCompressionError::Dimension { .. })
    ));
    assert!(matches!(
        compressed.apply_hermitian_rows_block_checked(&input, &[system.total_dofs()]),
        Err(sonium_bem::TransmissionCompressionError::InvalidRow { .. })
    ));
    assert!(matches!(
        compressed.apply_hermitian_rows_checked(&input.column_to_vec(0), &[system.total_dofs()]),
        Err(sonium_bem::TransmissionCompressionError::InvalidRow { .. })
    ));
}

#[test]
fn typed_transmission_admission_rejects_mixed_frequency_before_assembly() {
    let (elements, nodes) = setup_mesh();
    let incident = AcousticMedium::try_new(FREQUENCY_HZ, 343.0, 1.21).expect("incident medium");
    let transmitted = AcousticMedium::try_new(FREQUENCY_HZ + 1.0, 1_480.0, 1_000.0)
        .expect("individually valid transmitted medium");
    let error = build_transmission_tbem_system_from_xem(&elements, &nodes, incident, transmitted)
        .expect_err("mixed-frequency media must be rejected");
    assert!(matches!(error, TransmissionAssemblyError::Interface(_)));
}

pub fn typed_transmission_adapter_matches_raw_operator_and_medium_reference_v1() {
    let (elements, nodes) = setup_mesh();
    let incident = AcousticMedium::try_new(FREQUENCY_HZ, 343.0, 1.21).expect("incident medium");
    let transmitted =
        AcousticMedium::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0).expect("transmitted medium");

    let typed = build_transmission_tbem_system_from_xem(&elements, &nodes, incident, transmitted)
        .expect("typed transmission assembly");
    let raw = build_transmission_tbem_system(
        &elements,
        &nodes,
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false).expect("raw exterior physics"),
        PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
            .expect("raw transmitted physics"),
    )
    .expect("raw transmission assembly");

    assert_eq!(typed.interface_dofs(), raw.interface_dofs());
    let mut maximum_relative_error = 0.0_f64;
    for row in 0..typed.total_dofs() {
        for column in 0..typed.total_dofs() {
            maximum_relative_error = maximum_relative_error.max(relative_error(
                typed.matrix()[[row, column]],
                raw.matrix()[[row, column]],
            ));
        }
    }
    assert!(
        maximum_relative_error <= 1.0e-14,
        "typed/raw transmission matrix relative error {maximum_relative_error:.3e} exceeds 1e-14"
    );

    // Independent dimensional reference: the typed medium must carry the
    // same signed outgoing wavenumber as k = 2πf/c under the default XEM
    // convention before BEM assembles medium-weighted blocks.
    let expected_incident_wavenumber = 2.0 * std::f64::consts::PI * FREQUENCY_HZ / 343.0;
    let expected_transmitted_wavenumber = 2.0 * std::f64::consts::PI * FREQUENCY_HZ / 1_480.0;
    assert!((incident.wavenumber() - expected_incident_wavenumber).abs() <= 1.0e-14);
    assert!((transmitted.wavenumber() - expected_transmitted_wavenumber).abs() <= 1.0e-14);
    assert!((typed.exterior.wave_number - expected_incident_wavenumber).abs() <= 1.0e-14);
    assert!((typed.interior.wave_number - expected_transmitted_wavenumber).abs() <= 1.0e-14);
}

#[test]
fn test_typed_transmission_adapter_matches_raw_operator_and_medium_reference_v1() {
    typed_transmission_adapter_matches_raw_operator_and_medium_reference_v1();
}

#[test]
fn compressed_transmission_rejects_non_finite_aca_error_v1() {
    let error = sonium_bem::core::assembly::CompressedTransmissionTbemSystem::from_entry(
        2,
        4,
        2,
        1.0e-2,
        1,
        |row, column| {
            if row < 2 && column >= 2 {
                Complex64::new(f64::MAX, 0.0)
            } else {
                Complex64::new(1.0 + (row + column) as f64, 0.0)
            }
        },
    )
    .expect_err("finite entries with overflowing ACA norm must fail closed");
    assert!(matches!(
        error,
        sonium_bem::TransmissionCompressionError::NonFiniteBlock { .. }
    ));
}

#[test]
fn compressed_transmission_checked_actions_reject_non_finite_outputs_v1() {
    let compressed = sonium_bem::core::assembly::CompressedTransmissionTbemSystem::from_entry(
        2,
        4,
        2,
        1.0e-2,
        2,
        |_row, _column| Complex64::new(2.0, 0.0),
    )
    .expect("finite compressed operator");
    let input = vec![Complex64::new(f64::MAX, 0.0); 4];
    for (operation, result) in [
        ("forward", compressed.apply_checked(&input)),
        ("transpose", compressed.apply_transpose_checked(&input)),
        ("Hermitian", compressed.apply_hermitian_checked(&input)),
    ] {
        assert!(
            matches!(
                result,
                Err(sonium_bem::TransmissionCompressionError::NonFiniteOutput {
                    operation: actual,
                    ..
                }) if actual == operation
            ),
            "{operation} action must reject overflowed output"
        );
    }

    let mut block_input = sonium_solvers::dense::DenseMatrix::zeros(4, 2);
    for row in 0..4 {
        block_input[[row, 0]] = input[row];
        block_input[[row, 1]] = input[row];
    }
    assert!(matches!(
        compressed.apply_hermitian_block_checked(&block_input),
        Err(sonium_bem::TransmissionCompressionError::NonFiniteOutput { .. })
    ));
    assert!(matches!(
        compressed.apply_rows_checked(&input, &[0, 1]),
        Err(sonium_bem::TransmissionCompressionError::NonFiniteOutput { .. })
    ));
    assert!(matches!(
        compressed.apply_transpose_rows_block_checked(&block_input, &[0, 1]),
        Err(sonium_bem::TransmissionCompressionError::NonFiniteOutput { .. })
    ));
}

#[test]
fn compressed_transmission_rank_truncated_far_block_action_has_bounded_error_v1() {
    let entry = |row: usize, column: usize| {
        let value = match (row, column) {
            (0, 0) | (1, 1) | (2, 2) | (3, 3) => 2.0,
            (0, 2) => 1.0,
            (1, 3) => 0.2,
            (2, 0) => 0.5,
            (3, 1) => 0.1,
            _ => 0.0,
        };
        Complex64::new(value, 0.0)
    };
    let compressed = sonium_bem::core::assembly::CompressedTransmissionTbemSystem::from_entry(
        2, 4, 2, 0.75, 1, entry,
    )
    .expect("rank-truncated far-block operator");
    assert_eq!(compressed.compressed_block_count(), 2);

    let input = [
        Complex64::new(1.0, -0.2),
        Complex64::new(-0.5, 0.7),
        Complex64::new(0.8, 0.1),
        Complex64::new(-1.2, -0.4),
    ];
    let mut dense_action = vec![Complex64::default(); 4];
    for row in 0..4 {
        for column in 0..4 {
            dense_action[row] += entry(row, column) * input[column];
        }
    }
    let compressed_action = compressed.apply(&input);
    let error = compressed_action
        .iter()
        .zip(&dense_action)
        .map(|(actual, reference)| (*actual - *reference).norm())
        .fold(0.0, f64::max)
        / dense_action
            .iter()
            .map(|value| value.norm())
            .fold(0.0, f64::max)
            .max(f64::MIN_POSITIVE);
    assert!(error.is_finite());
    assert!(error <= 0.75, "rank-truncated far-block error {error:e}");
}
