//! Bounded CPU evidence for conforming quadratic Müller transmission.

use sonium_bem::core::assembly::build_transmission_tbem_system_with_continuous_quadratic;
use sonium_bem::core::assembly::build_transmission_tbem_system_with_continuous_quadratic_from_xem;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::{generate_cubed_sphere_quad_mesh, generate_icosphere_mesh};
use sonium_bem::core::solver::transmission::{
    solve_incident_transmission, solve_transmission_adjoint_many,
};
use sonium_bem::core::types::PhysicsParams;
use sonium_qa::{QaResult, QaStatus};
use sonium_wave::analytical::FluidSphereTransmission3d;
use sonium_xem::AcousticMedium;
use std::f64::consts::PI;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-continuous-quadratic.v1";

fn run() -> Result<(usize, usize, f64, f64, f64, f64, f64, f64), String> {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let system = build_transmission_tbem_system_with_continuous_quadratic_from_xem(
        &mesh.elements,
        &mesh.nodes,
        AcousticMedium::try_new(420.0, 343.0, 1.21).map_err(|error| error.to_string())?,
        AcousticMedium::try_new(420.0, 1_480.0, 1_000.0).map_err(|error| error.to_string())?,
    )
    .map_err(|error| error.to_string())?;

    if system.interface_dofs() >= mesh.elements.len() * 6 {
        return Err("continuous topology did not reduce shared DOFs".to_string());
    }
    if system
        .matrix()
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("continuous transmission matrix contains non-finite values".to_string());
    }

    let incident = IncidentField::plane_wave_z();
    let solution = solve_incident_transmission(&system, &incident, 1.0e-8)
        .map_err(|error| error.to_string())?;
    let field = solution
        .evaluate_exterior_field_checked(&system, &[[0.0, 0.0, 0.8]], &incident)
        .map_err(|error| error.to_string())?;
    if field
        .iter()
        .any(|point| !point.p_total.re.is_finite() || !point.p_total.im.is_finite())
    {
        return Err("continuous exterior field contains non-finite values".to_string());
    }
    let angles: [f64; 8] = [0.2, 0.55, 0.9, 1.25, 1.6, 2.0, 2.4, 2.8];
    let evaluation_radius = 0.8;
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
    let evaluated = solution
        .evaluate_exterior_field_checked(&system, &points, &incident)
        .map_err(|error| error.to_string())?;
    let mie = FluidSphereTransmission3d::try_new(
        0.2,
        2.0 * PI * 420.0 / 343.0,
        2.0 * PI * 420.0 / 1_480.0,
        1.21,
        1_000.0,
        32,
    )
    .map_err(|error| error.to_string())?;
    let max_relative_error = evaluated
        .iter()
        .zip(angles)
        .map(|(field, theta)| {
            let reference = mie
                .exterior_pressure(evaluation_radius, theta)
                .map_err(|error| error.to_string())?;
            Ok::<f64, String>(
                (field.p_total - reference).norm()
                    / field
                        .p_total
                        .norm()
                        .max(reference.norm())
                        .max(f64::MIN_POSITIVE),
            )
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .fold(0.0, f64::max);
    if max_relative_error >= 0.15 {
        return Err(format!(
            "continuous transmission Mie field error {max_relative_error:.3e}"
        ));
    }
    let quad_mesh = generate_cubed_sphere_quad_mesh(0.2, 1);
    let quad_system = build_transmission_tbem_system_with_continuous_quadratic(
        &quad_mesh.elements,
        &quad_mesh.nodes,
        PhysicsParams::new(420.0, 343.0, 1.21, false),
        PhysicsParams::new(420.0, 1_480.0, 1_000.0, true),
    )
    .map_err(|error| error.to_string())?;
    let quad_solution = solve_incident_transmission(&quad_system, &incident, 1.0e-8)
        .map_err(|error| error.to_string())?;
    if quad_solution.relative_residual > 1.0e-8 {
        return Err(format!(
            "curved Quad4 transmission residual {:.3e}",
            quad_solution.relative_residual
        ));
    }

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
    let adjoints = solve_transmission_adjoint_many(&system, &[rhs, second_rhs], 1.0e-8)
        .map_err(|error| error.to_string())?;
    if adjoints.len() != 2 {
        return Err(format!(
            "continuous quadratic adjoint batch returned {} columns",
            adjoints.len()
        ));
    }
    let adjoint_residual = adjoints[0].relative_residual;
    let adjoint_batch_residual = adjoints
        .iter()
        .map(|solution| solution.relative_residual)
        .fold(0.0, f64::max);
    let adjoint_field = adjoints[0]
        .evaluate_exterior_adjoint_pressure_checked(&system, &[[0.0, 0.0, 0.8]])
        .map_err(|error| error.to_string())?;
    let adjoint_field_norm = adjoint_field
        .iter()
        .map(|value| value.norm())
        .fold(0.0, f64::max);
    if !adjoint_field_norm.is_finite() || adjoint_field_norm <= 0.0 {
        return Err("continuous quadratic adjoint field was non-finite or zero".to_string());
    }
    let invalid_adjoint_field_rejected = adjoints[0]
        .evaluate_exterior_adjoint_pressure_checked(&system, &[[f64::NAN, 0.0, 0.8]])
        .is_err();
    if !invalid_adjoint_field_rejected {
        return Err("invalid quadratic adjoint field point was accepted".to_string());
    }

    Ok((
        system.interface_dofs(),
        quad_system.interface_dofs(),
        solution.relative_residual,
        max_relative_error,
        adjoint_residual,
        adjoint_batch_residual,
        adjoint_field_norm,
        f64::from(invalid_adjoint_field_rejected),
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((
            interface_dofs,
            quad4_dofs,
            residual,
            mie_error,
            adjoint_residual,
            adjoint_batch_residual,
            adjoint_field_norm,
            invalid_adjoint_field_rejected,
        )) => QaResult::passed(CASE_ID)
            .metric("interface_dofs", interface_dofs as f64)
            .metric("quad4_interface_dofs", quad4_dofs as f64)
            .metric("relative_residual", residual)
            .metric("max_relative_mie_error", mie_error)
            .metric("adjoint_residual", adjoint_residual)
            .metric("adjoint_batch_residual", adjoint_batch_residual)
            .metric("adjoint_field_norm", adjoint_field_norm)
            .metric(
                "invalid_adjoint_field_rejected",
                invalid_adjoint_field_rejected,
            )
            .requested_backend("cpu")
            .actual_backend("cpu")
            .input_checksum("bem:icosphere:subdivision-1:continuous-quadratic-transmission")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-transmission-continuous-quadratic-v1",
            )
            .tolerance("relative residual <= 1e-8")
            .dimension("3-D dense CPU conforming quadratic Müller transmission")
            .seed("bem-transmission-continuous-quadratic:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(message) => QaResult::failed(CASE_ID, message)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .input_checksum("bem:icosphere:subdivision-1:continuous-quadratic-transmission")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-transmission-continuous-quadratic-v1",
            )
            .dimension("3-D dense CPU conforming quadratic Müller transmission")
            .seed("bem-transmission-continuous-quadratic:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
