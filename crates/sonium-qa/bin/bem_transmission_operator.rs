//! Release-facing contract for the bounded constant-panel Müller operator.

use num_complex::Complex64;
use sonium_bem::build_transmission_tbem_system_from_xem;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::solver::solve_incident_transmission;
use sonium_bem::core::types::Element;
use sonium_qa::QaResult;
use sonium_xem::AcousticMedium;
use std::time::Instant;

const CASE_ID: &str = "bem.transmission-operator-contracts.v1";
const FREQUENCY_HZ: f64 = 420.0;

fn run() -> Result<(f64, f64, f64, f64, f64, f64), String> {
    let mesh = generate_icosphere_mesh(0.2, 1);
    let mut elements: Vec<Element> = mesh.elements;
    for (index, element) in elements.iter_mut().enumerate() {
        element.dof_addresses = vec![index];
    }

    let incident = AcousticMedium::try_new(FREQUENCY_HZ, 343.0, 1.21)
        .map_err(|error| format!("incident medium: {error}"))?;
    let transmitted = AcousticMedium::try_new(FREQUENCY_HZ, 1_480.0, 1_000.0)
        .map_err(|error| format!("transmitted medium: {error}"))?;
    let system =
        build_transmission_tbem_system_from_xem(&elements, &mesh.nodes, incident, transmitted)
            .map_err(|error| format!("transmission assembly: {error}"))?;

    let incident_rhs = system
        .incident_rhs(&IncidentField::plane_wave_z())
        .map_err(|error| format!("incident RHS: {error}"))?;
    if incident_rhs.len() != system.total_dofs()
        || incident_rhs
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("incident RHS was not finite and dimension-matched".into());
    }

    let incident_solution =
        solve_incident_transmission(&system, &IncidentField::plane_wave_z(), 1.0e-10)
            .map_err(|error| format!("incident transmission solve: {error}"))?;
    let incident_residual = incident_solution.relative_residual;
    if !incident_residual.is_finite() || incident_residual > 1.0e-10 {
        return Err(format!(
            "incident transmission residual {incident_residual:.3e} exceeds bound"
        ));
    }

    let exterior_field = incident_solution
        .evaluate_exterior_field_checked(
            &system,
            &[[0.0, 0.0, 1.0], [0.3, 0.0, 0.0]],
            &IncidentField::plane_wave_z(),
        )
        .map_err(|error| format!("exterior field evaluation: {error}"))?;
    if exterior_field.iter().any(|point| {
        [point.p_incident, point.p_scattered, point.p_total]
            .iter()
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    }) {
        return Err("exterior field contains non-finite pressure".into());
    }

    let exterior_field_points = exterior_field.len() as f64;

    let n = elements
        .iter()
        .filter(|element| !element.property.is_evaluation())
        .count();
    if system.interface_dofs() != n || system.total_dofs() != 2 * n {
        return Err(format!(
            "unexpected interface dimensions: reported {} / {}, expected {} / {}",
            system.interface_dofs(),
            system.total_dofs(),
            n,
            2 * n
        ));
    }

    let mut finite_entries = 0.0;
    let mut matrix_l1 = 0.0;
    for row in 0..system.total_dofs() {
        for column in 0..system.total_dofs() {
            let value: Complex64 = system.matrix()[[row, column]];
            if !value.re.is_finite() || !value.im.is_finite() {
                return Err(format!("non-finite matrix entry at ({row}, {column})"));
            }
            finite_entries += 1.0;
            matrix_l1 += value.norm();
        }
    }
    if matrix_l1 <= 0.0 {
        return Err("transmission matrix is identically zero".into());
    }

    let mixed_frequency = AcousticMedium::try_new(FREQUENCY_HZ + 1.0, 1_480.0, 1_000.0)
        .map_err(|error| format!("mixed-frequency medium: {error}"))?;
    let rejection_checks =
        build_transmission_tbem_system_from_xem(&elements, &mesh.nodes, incident, mixed_frequency)
            .is_err() as u8 as f64;
    if rejection_checks < 1.0 {
        return Err("mixed-frequency transmission input was not rejected".into());
    }

    Ok((
        n as f64,
        finite_entries,
        matrix_l1,
        rejection_checks,
        incident_residual,
        exterior_field_points,
    ))
}

fn main() {
    let started = Instant::now();
    match std::panic::catch_unwind(run) {
        Ok(Ok((
            interface_dofs,
            finite_entries,
            matrix_l1,
            rejection_checks,
            incident_residual,
            exterior_field_points,
        ))) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("interface_dofs", interface_dofs)
                .metric("finite_matrix_entries", finite_entries)
                .metric("matrix_l1_norm", matrix_l1)
        .metric("typed_rejection_checks", rejection_checks)
        .metric("incident_solve_relative_residual", incident_residual)
        .metric("exterior_field_points", exterior_field_points)
                .minimum("interface_dofs", 1.0)
                .minimum("finite_matrix_entries", 1.0)
                .minimum("matrix_l1_norm", f64::MIN_POSITIVE)
            .minimum("typed_rejection_checks", 1.0)
        .maximum("incident_solve_relative_residual", 1.0e-10)
        .minimum("exterior_field_points", 2.0)
                .input_checksum("analytic/deterministic-fixture:bem-transmission-operator-v1")
                .reference("independent off-diagonal kernel fixture and typed XEM admission")
                .reference_checksum("analytic/deterministic-fixture:bem-transmission-operator-v1")
                .dimension("3D closed icosphere, constant-panel two-medium interface")
                .formulation("dense constant-panel Müller transmission operator")
                .tolerance("finite 2N-by-2N matrix, nonzero norm, mixed-frequency rejection")
                .seed("bem-transmission-operator:icosphere-level-1:v1")
                .reference(
                    "independent off-diagonal kernel fixture, typed mixed-frequency rejection, incident-trace RHS finiteness, and solved residual",
                )
                .formulation("dense constant-panel Müller transmission contract")
                .tolerance(
                    "finite matrix entries; matrix L1 norm > 0; mixed-frequency rejection dimension-matched finite incident RHS and solve residual <= 1e-10",
                )
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("transmission QA result must serialize")
            );
            println!("SUCCESS");
        }
        Ok(Err(error)) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("transmission QA failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
        Err(_) => {
            let record = QaResult::failed(CASE_ID, "panic during transmission QA")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("transmission panic result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
