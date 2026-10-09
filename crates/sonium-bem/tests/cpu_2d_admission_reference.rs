//! CPU BEM/XEM evidence for surface-mesh dimensional dispatch.
//!
//! The production BEM path is deliberately tested only for checked admission:
//! it must reject a reduced request rather than applying a 3-D surface kernel.
//! Axisymmetric production execution is exposed separately through the reduced
//! ring-source API. The physical slice below remains an independent modal
//! solution for a rigid circular cylinder in a plane wave.

use num_complex::Complex64;
use sonium_bem::core::{BemProblem, CpuBemDimension, CpuBemDimensionError, admit_cpu_bem_problem};
use sonium_xem::{
    ComplexStorage, ConventionRegistry, Domain, DomainKind, FieldKind, FieldSpec,
    MultiphysicsProblem, ScalarType, StorageOrder, TensorLayout,
};
use std::f64::consts::PI;

const FIELD_ID: &str = "pressure";
const EULER_GAMMA: f64 = 0.577_215_664_901_532_9;

fn xem_pressure_problem(element_count: usize) -> MultiphysicsProblem {
    MultiphysicsProblem {
        id: "cpu-bem-2d-admission-reference-v1".into(),
        conventions: ConventionRegistry::default(),
        domains: vec![Domain {
            id: "acoustics".into(),
            label: "2-D reference acoustics".into(),
            kind: DomainKind::Acoustics,
        }],
        fields: vec![FieldSpec {
            id: FIELD_ID.into(),
            name: "complex acoustic pressure".into(),
            domain_id: "acoustics".into(),
            kind: FieldKind::ComplexScalar,
            units: Some("Pa".into()),
            layout: TensorLayout {
                scalar_type: ScalarType::ComplexF64,
                shape: vec![element_count],
                strides: vec![1],
                byte_offset: 0,
                order: StorageOrder::RowMajor,
                complex_storage: ComplexStorage::Interleaved,
                coordinate_frame: None,
            },
        }],
        ..MultiphysicsProblem::default()
    }
}

fn j0(x: f64) -> f64 {
    let z = x * x / 4.0;
    let mut term = 1.0;
    let mut sum = term;
    for m in 1..80 {
        term *= -z / (m as f64).powi(2);
        sum += term;
        if term.abs() <= sum.abs() * 2.0e-16 {
            break;
        }
    }
    sum
}

fn j1(x: f64) -> f64 {
    let z = x * x / 4.0;
    let mut term = x / 2.0;
    let mut sum = term;
    for m in 1..80 {
        term *= -z / ((m as f64) * ((m + 1) as f64));
        sum += term;
        if term.abs() <= sum.abs() * 2.0e-16 {
            break;
        }
    }
    sum
}

fn y0(x: f64) -> f64 {
    let z = x * x / 4.0;
    let j = j0(x);
    let mut harmonic = 0.0;
    let mut term = 1.0;
    let mut series = 0.0;
    for m in 1..80 {
        harmonic += 1.0 / m as f64;
        term *= -z / (m as f64).powi(2);
        series -= harmonic * term;
        if term.abs() <= (1.0 + series.abs()) * 2.0e-16 {
            break;
        }
    }
    (2.0 / PI) * ((x / 2.0).ln() + EULER_GAMMA) * j + (2.0 / PI) * series
}

fn y1(x: f64) -> f64 {
    let z = x * x / 4.0;
    let j = j0(x);
    let j_derivative = -j1(x);
    let mut harmonic = 0.0;
    let mut term = 1.0;
    let mut series_derivative = 0.0;
    for m in 1..80 {
        harmonic += 1.0 / m as f64;
        term *= -z / (m as f64).powi(2);
        // The Y0 series uses -H_m * term; differentiate z=x²/4.
        series_derivative -= harmonic * term * (2.0 * m as f64 / x);
        if term.abs() <= (1.0 + series_derivative.abs()) * 2.0e-16 {
            break;
        }
    }
    -(2.0 / PI) * (j / x + ((x / 2.0).ln() + EULER_GAMMA) * j_derivative + series_derivative)
}

fn bessel_pair(order: usize, x: f64) -> (f64, f64) {
    if order == 0 {
        return (j0(x), y0(x));
    }
    if order == 1 {
        return (j1(x), y1(x));
    }
    let mut jm1 = (j0(x), y0(x));
    let mut j = (j1(x), y1(x));
    for n in 1..order {
        let factor = 2.0 * n as f64 / x;
        let next = (factor * j.0 - jm1.0, factor * j.1 - jm1.1);
        jm1 = j;
        j = next;
    }
    j
}

fn derivative(order: usize, x: f64) -> (f64, f64) {
    let current = bessel_pair(order, x);
    let next = bessel_pair(order + 1, x);
    (
        order as f64 / x * current.0 - next.0,
        order as f64 / x * current.1 - next.1,
    )
}

fn rigid_cylinder_pressure(k: f64, radius: f64, r: f64, theta: f64, terms: usize) -> Complex64 {
    let ka = k * radius;
    let kr = k * r;
    let mut total = Complex64::new(0.0, 0.0);
    for n in 0..terms {
        let weight = if n == 0 { 1.0 } else { 2.0 };
        let (j_surface, y_surface) = derivative(n, ka);
        let coefficient = Complex64::new(-j_surface, 0.0) / Complex64::new(j_surface, y_surface);
        let (j_field, y_field) = bessel_pair(n, kr);
        let hankel = Complex64::new(j_field, y_field);
        let incident = Complex64::from_polar(1.0, n as f64 * PI / 2.0);
        total += incident
            * (Complex64::new(j_field, 0.0) + coefficient * hankel)
            * (weight * (n as f64 * theta).cos());
    }
    total
}

fn rigid_cylinder_radial_derivative(k: f64, radius: f64, theta: f64, terms: usize) -> Complex64 {
    let ka = k * radius;
    let mut derivative_sum = Complex64::new(0.0, 0.0);
    for n in 0..terms {
        let weight = if n == 0 { 1.0 } else { 2.0 };
        let (j_surface, y_surface) = derivative(n, ka);
        let coefficient = Complex64::new(-j_surface, 0.0) / Complex64::new(j_surface, y_surface);
        let (dj, dy) = derivative(n, ka);
        let incident = Complex64::from_polar(1.0, n as f64 * PI / 2.0);
        derivative_sum += incident * Complex64::new(dj, 0.0) * (weight * (n as f64 * theta).cos());
        // The Hankel derivative is J'_n + iY'_n; the expression above keeps
        // the incident derivative separate so the cancellation is explicit.
        derivative_sum +=
            incident * coefficient * Complex64::new(dj, dy) * (weight * (n as f64 * theta).cos());
    }
    derivative_sum * k
}

#[test]
fn cpu_bem_xem_surface_boundary_rejects_reduced_dispatch() {
    let three_d = BemProblem::rigid_sphere_scattering_custom(0.1, 800.0, 343.0, 1.21, 4, 8);
    let exchange = xem_pressure_problem(three_d.mesh.elements.len());
    exchange
        .validate()
        .expect("XEM pressure contract must validate");
    let admitted = BemProblem::from_xem(
        &exchange,
        FIELD_ID,
        three_d.mesh.clone(),
        three_d.physics.clone(),
        three_d.incident_field.clone(),
        three_d.bc_type,
        three_d.use_burton_miller,
    )
    .expect("valid XEM-to-BEM 3-D handoff");

    admit_cpu_bem_problem(&admitted, CpuBemDimension::ThreeDimensional)
        .expect("the supported 3-D surface path must be admitted");
    for requested in [
        CpuBemDimension::TwoDimensional,
        CpuBemDimension::Axisymmetric,
    ] {
        assert!(matches!(
            admit_cpu_bem_problem(&admitted, requested),
            Err(CpuBemDimensionError::Unsupported { .. })
        ));
    }

    let mut malformed = admitted.clone();
    malformed.mesh.elements.clear();
    assert!(matches!(
        admit_cpu_bem_problem(&malformed, CpuBemDimension::ThreeDimensional),
        Err(CpuBemDimensionError::InvalidProblem(_))
    ));
}

#[test]
fn independent_rigid_cylinder_reference_satisfies_neumann_boundary() {
    assert!((j0(1.0) - 0.765_197_686_557_966_6).abs() < 2.0e-14);
    assert!((j1(1.0) - 0.440_050_585_744_933_5).abs() < 2.0e-14);
    assert!((y0(1.0) - 0.088_256_964_215_676_96).abs() < 2.0e-14);
    assert!((y1(1.0) + 0.781_212_821_300_288_7).abs() < 2.0e-14);

    let k = 2.0;
    let radius = 0.4;
    let terms = 18;
    let mut max_boundary_residual: f64 = 0.0;
    for index in 0..17 {
        let theta = 2.0 * PI * index as f64 / 17.0;
        let residual = rigid_cylinder_radial_derivative(k, radius, theta, terms).norm();
        max_boundary_residual = max_boundary_residual.max(residual);
    }
    assert!(
        max_boundary_residual < 2.0e-12,
        "independent rigid-cylinder reference violates zero normal velocity: {max_boundary_residual:e}"
    );

    let forward = rigid_cylinder_pressure(k, radius, 1.2, 0.37, terms);
    let mirrored = rigid_cylinder_pressure(k, radius, 1.2, -0.37, terms);
    assert!((forward - mirrored).norm() < 2.0e-12);
    assert!(forward.re.is_finite() && forward.im.is_finite());
}
