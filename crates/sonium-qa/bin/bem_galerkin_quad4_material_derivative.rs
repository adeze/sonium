//! Bounded CPU QA for on-demand curved Quad4 material-derivative actions.

use num_complex::Complex64;
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::{
    BasisOrder, PhysicsParams, TransmissionMaterialParameter, TransmissionSide,
    assemble_galerkin_muller_transmission_material_derivative_with_global_dofs,
    assemble_galerkin_muller_transmission_with_global_dofs,
    compress_galerkin_muller_transmission_quad4_material_derivative_with_global_dofs,
};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "bem.galerkin-quad4-material-derivative.v1";

fn element(connectivity: Vec<usize>, dof_addresses: Vec<usize>, center: [f64; 3]) -> Element {
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

fn perturb(
    medium: &PhysicsParams,
    parameter: TransmissionMaterialParameter,
    value: f64,
) -> PhysicsParams {
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
}

fn centered_step(medium: &PhysicsParams, parameter: TransmissionMaterialParameter) -> f64 {
    match parameter {
        TransmissionMaterialParameter::Wavenumber => {
            (medium.wave_number.abs() * 1.0e-5).max(1.0e-7)
        }
        TransmissionMaterialParameter::SpeedOfSound => {
            (medium.speed_of_sound.abs() * 1.0e-5).max(1.0e-5)
        }
        TransmissionMaterialParameter::Density => (medium.density.abs() * 1.0e-5).max(1.0e-5),
    }
}

fn max_relative_difference(lhs: &DenseMatrix<Complex64>, rhs: &DenseMatrix<Complex64>) -> f64 {
    let max_error = lhs
        .as_slice()
        .iter()
        .zip(rhs.as_slice())
        .map(|(actual, reference)| (actual - reference).norm())
        .fold(0.0_f64, f64::max);
    let max_reference = lhs
        .as_slice()
        .iter()
        .zip(rhs.as_slice())
        .map(|(actual, reference)| actual.norm().max(reference.norm()))
        .fold(f64::MIN_POSITIVE, f64::max);
    max_error / max_reference
}

fn run() -> Result<(f64, f64, usize), String> {
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
    let elements = vec![
        element(vec![0, 1, 2, 3], vec![0, 1, 2, 3], [0.5, 0.5, 0.0]),
        element(vec![4, 5, 6, 7], vec![4, 5, 6, 7], [0.5, 0.5, 1.4]),
    ];
    let exterior =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;
    let interior =
        PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, true).map_err(|error| error.to_string())?;
    let trial: Vec<Complex64> = (0..16)
        .map(|index| Complex64::new(0.05 * (index as f64 + 1.0), -0.02))
        .collect();
    let mut maximum_relative_error = 0.0_f64;
    let mut maximum_centered_error = 0.0_f64;
    let mut minimum_compressed_blocks = usize::MAX;
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
            3,
            8,
            side,
            parameter,
        )
        .map_err(|error| error.to_string())?;
        let selected = match side {
            TransmissionSide::Exterior => &exterior,
            TransmissionSide::Interior => &interior,
        };
        let delta = centered_step(selected, parameter);
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
        .map_err(|error| error.to_string())?;
        let minus = assemble_galerkin_muller_transmission_with_global_dofs(
            &elements,
            &nodes,
            &exterior_minus,
            &interior_minus,
            BasisOrder::Linear,
            3,
            8,
        )
        .map_err(|error| error.to_string())?;
        let centered = DenseMatrix::from_fn(dense.nrows(), dense.ncols(), |row, column| {
            (plus.matrix()[[row, column]] - minus.matrix()[[row, column]]) / (2.0 * delta)
        });
        let centered_error = max_relative_difference(&dense, &centered);
        if !centered_error.is_finite() || centered_error > 5.0e-3 {
            return Err(format!(
                "{side:?} {parameter:?} analytic-centered error {centered_error:e}"
            ));
        }
        maximum_centered_error = maximum_centered_error.max(centered_error);
        let compressed =
            compress_galerkin_muller_transmission_quad4_material_derivative_with_global_dofs(
                &elements,
                &nodes,
                &exterior,
                &interior,
                BasisOrder::Linear,
                3,
                8,
                side,
                parameter,
                4,
                1.0e-10,
                64,
            )
            .map_err(|error| error.to_string())?;
        let actual = sonium_solvers::traits::LinearOperator::apply(&compressed, &trial);
        let expected = (0..dense.nrows())
            .map(|row| {
                (0..dense.ncols())
                    .map(|column| dense[[row, column]] * trial[column])
                    .sum::<Complex64>()
            })
            .collect::<Vec<_>>();
        let error = actual
            .iter()
            .zip(expected.iter())
            .map(|(value, reference)| (*value - *reference).norm())
            .fold(0.0_f64, f64::max)
            / expected
                .iter()
                .map(|value| value.norm())
                .fold(0.0_f64, f64::max)
                .max(f64::MIN_POSITIVE);
        if !error.is_finite() || error > 2.0e-6 {
            return Err(format!(
                "{side:?} {parameter:?} compressed action error {error:e}"
            ));
        }
        maximum_relative_error = maximum_relative_error.max(error);
        minimum_compressed_blocks =
            minimum_compressed_blocks.min(compressed.compressed_block_count());
    }
    Ok((
        maximum_relative_error,
        maximum_centered_error,
        minimum_compressed_blocks,
    ))
}

fn main() {
    let started = Instant::now();
    let record = match run() {
        Ok((maximum_relative_error, maximum_centered_error, minimum_compressed_blocks)) => {
            QaResult::passed(CASE_ID)
                .metric("maximum_relative_action_error", maximum_relative_error)
                .metric("maximum_relative_centered_error", maximum_centered_error)
                .metric(
                    "minimum_compressed_blocks",
                    minimum_compressed_blocks as f64,
                )
                .input_checksum(
                    "analytic/deterministic-fixture:bem-galerkin-quad4-material-derivative-v1",
                )
                .reference_checksum(
                    "analytic/deterministic-fixture:bem-galerkin-quad4-material-derivative-v1",
                )
                .dimension("3-D curved separated Quad4 two-trace material derivatives")
                .formulation("on-demand isoparametric pair kernels with deterministic block ACA")
                .tolerance("analytic-centered relative error <= 5e-3; compressed action <= 2e-6")
                .seed("bem-galerkin-quad4-material-derivative:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64)
        }
        Err(error) => QaResult::failed(CASE_ID, error)
            .input_checksum(
                "analytic/deterministic-fixture:bem-galerkin-quad4-material-derivative-v1",
            )
            .reference_checksum(
                "analytic/deterministic-fixture:bem-galerkin-quad4-material-derivative-v1",
            )
            .dimension("3-D curved separated Quad4 two-trace material derivatives")
            .seed("bem-galerkin-quad4-material-derivative:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
    };
    println!(
        "QA_RESULT: {}",
        record.to_json_line().expect("serialize QA result")
    );
    if record.status != sonium_qa::QaStatus::Passed {
        std::process::exit(1);
    }
}
