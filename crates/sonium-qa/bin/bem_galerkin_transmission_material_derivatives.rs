use num_complex::Complex64;
use sonium_bem::core::assembly::{
    GalerkinTransmissionAssemblyError, GalerkinTransmissionMaterialDerivativeError,
    assemble_galerkin_muller_transmission_material_derivative_with_global_dofs,
    assemble_galerkin_muller_transmission_with_global_dofs,
};
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::types::Element;
use sonium_bem::{BasisOrder, PhysicsParams, TransmissionMaterialParameter, TransmissionSide};
use sonium_qa::QaResult;
use sonium_solvers::dense::DenseMatrix;
use std::time::Instant;

const CASE_ID: &str = "bem.galerkin-transmission-material-derivatives.v1";
const FREQUENCY_HZ: f64 = 420.0;
const GAUSS_ORDER: usize = 4;

fn setup_mesh() -> (Vec<Element>, Vec<[f64; 3]>, usize) {
    let mut mesh = generate_icosphere_mesh(0.2, 0);
    for element in &mut mesh.elements {
        element.dof_addresses = element.connectivity.clone();
    }
    let global_dof_count = mesh.nodes.len();
    (mesh.elements, mesh.nodes, global_dof_count)
}

fn media() -> Result<(PhysicsParams, PhysicsParams), String> {
    Ok((
        PhysicsParams::try_new(FREQUENCY_HZ, 343.0, 1.21, false)
            .map_err(|error| error.to_string())?,
        PhysicsParams::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0, true)
            .map_err(|error| error.to_string())?,
    ))
}

fn step(medium: &PhysicsParams, parameter: TransmissionMaterialParameter) -> f64 {
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

fn perturb(
    medium: &PhysicsParams,
    value: f64,
    parameter: TransmissionMaterialParameter,
) -> Result<PhysicsParams, String> {
    let (speed, density) = match parameter {
        TransmissionMaterialParameter::Wavenumber => (medium.omega / value, medium.density),
        TransmissionMaterialParameter::SpeedOfSound => (value, medium.density),
        TransmissionMaterialParameter::Density => (medium.speed_of_sound, value),
    };
    PhysicsParams::try_new(FREQUENCY_HZ, speed, density, medium.tau < 0.0)
        .map_err(|error| error.to_string())
}

fn max_error(lhs: &DenseMatrix<Complex64>, rhs: &DenseMatrix<Complex64>) -> f64 {
    lhs.as_slice()
        .iter()
        .zip(rhs.as_slice())
        .map(|(left, right)| (*left - *right).norm())
        .fold(0.0, f64::max)
}

fn direct_centered(
    elements: &[Element],
    nodes: &[[f64; 3]],
    exterior: &PhysicsParams,
    interior: &PhysicsParams,
    side: TransmissionSide,
    parameter: TransmissionMaterialParameter,
    global_dof_count: usize,
) -> Result<DenseMatrix<Complex64>, String> {
    let selected = match side {
        TransmissionSide::Exterior => exterior,
        TransmissionSide::Interior => interior,
    };
    let delta = step(selected, parameter);
    let value = match parameter {
        TransmissionMaterialParameter::Wavenumber => selected.wave_number,
        TransmissionMaterialParameter::SpeedOfSound => selected.speed_of_sound,
        TransmissionMaterialParameter::Density => selected.density,
    };
    let build = |value: f64| -> Result<_, String> {
        let (exterior, interior) = match side {
            TransmissionSide::Exterior => (perturb(exterior, value, parameter)?, interior.clone()),
            TransmissionSide::Interior => (exterior.clone(), perturb(interior, value, parameter)?),
        };
        assemble_galerkin_muller_transmission_with_global_dofs(
            elements,
            nodes,
            &exterior,
            &interior,
            BasisOrder::Linear,
            GAUSS_ORDER,
            global_dof_count,
        )
        .map(|operators| operators.matrix().clone())
        .map_err(|error| error.to_string())
    };
    let plus = build(value + delta)?;
    let minus = build(value - delta)?;
    let mut result = DenseMatrix::zeros(plus.nrows(), plus.ncols());
    for row in 0..result.nrows() {
        for column in 0..result.ncols() {
            result[[row, column]] = (plus[[row, column]] - minus[[row, column]]) / (2.0 * delta);
        }
    }
    Ok(result)
}

fn run() -> Result<(f64, f64, f64), String> {
    let (elements, nodes, global_dof_count) = setup_mesh();
    let (exterior, interior) = media()?;
    let mut worst_error: f64 = 0.0;
    let mut finite_entries = 0usize;
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
            .map_err(|error| error.to_string())?;
        finite_entries += derivative
            .as_slice()
            .iter()
            .filter(|value| value.re.is_finite() && value.im.is_finite())
            .count();
        let expected = direct_centered(
            &elements,
            &nodes,
            &exterior,
            &interior,
            side,
            parameter,
            global_dof_count,
        )?;
        worst_error = worst_error.max(max_error(&derivative, &expected));
    }

    let mut invalid = exterior.clone();
    invalid.density = f64::NAN;
    let invalid_rejected = matches!(
        assemble_galerkin_muller_transmission_material_derivative_with_global_dofs(
            &elements,
            &nodes,
            &invalid,
            &interior,
            BasisOrder::Linear,
            GAUSS_ORDER,
            global_dof_count,
            TransmissionSide::Exterior,
            TransmissionMaterialParameter::Density,
        ),
        Err(GalerkinTransmissionMaterialDerivativeError::InvalidMedium { .. })
    );
    let unsupported_rejected = matches!(
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
    );
    if !invalid_rejected || !unsupported_rejected {
        return Err("Galerkin derivative admission checks did not reject invalid inputs".into());
    }
    Ok((worst_error, finite_entries as f64, global_dof_count as f64))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((worst_error, finite_entries, global_dof_count)) => QaResult::passed(CASE_ID)
            .input_checksum("bem:icosphere:subdivision-0:closed-conforming-linear-tri3")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-galerkin-transmission-material-derivatives-v1",
            )
            .dimension("3-D CPU affine-Tri3 Galerkin Müller material derivative")
            .formulation("dense shared-DOF centered material derivatives")
            .tolerance("all derivative entries finite; centered reassembly error <= 1e-10")
            .minimum("finite_entries", 1.0)
            .minimum("global_dof_count", 1.0)
            .maximum("worst_centered_error", 1.0e-10)
            .metric("worst_centered_error", worst_error)
            .metric("finite_entries", finite_entries)
            .metric("global_dof_count", global_dof_count)
            .seed("bem-galerkin-transmission-material-derivatives:icosphere-level-0:v1")
            .elapsed_ms(started.elapsed().as_millis() as u64),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
}
