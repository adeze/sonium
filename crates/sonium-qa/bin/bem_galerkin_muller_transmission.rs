//! Bounded CPU evidence for the affine-Tri3 Galerkin Müller transmission block.

use std::time::Instant;

use sonium_bem::core::IncidentField;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::{
    BasisOrder, PhysicsParams, assemble_galerkin_muller_transmission_with_global_dofs,
    compress_galerkin_muller_transmission_with_global_dofs,
};
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::dense::DenseMatrix;
use sonium_solvers::iterative::GmresConfig;

const CASE_ID: &str = "bem.galerkin-muller-transmission.v1";

fn run() -> Result<
    (
        usize,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        usize,
        usize,
    ),
    String,
> {
    let mesh = generate_icosphere_mesh(0.2, 0);
    let mut elements = mesh.elements;
    let mut unique_nodes = elements
        .iter()
        .flat_map(|element| element.connectivity.iter().copied())
        .collect::<Vec<_>>();
    unique_nodes.sort_unstable();
    unique_nodes.dedup();
    for element in &mut elements {
        element.dof_addresses = element
            .connectivity
            .iter()
            .map(|node| unique_nodes.binary_search(node).expect("node is present"))
            .collect();
    }
    let global_dof_count = unique_nodes.len();
    let exterior =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;
    let interior =
        PhysicsParams::try_new(420.0, 290.0, 1.55, false).map_err(|error| error.to_string())?;
    let operators = assemble_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &mesh.nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        global_dof_count,
    )
    .map_err(|error| error.to_string())?;
    let matrix = operators.matrix();
    let compressed = operators
        .compress(2, 1.0e-8, 2)
        .map_err(|error| error.to_string())?;
    let compression_probe = (0..operators.total_dofs())
        .map(|index| num_complex::Complex64::new(0.2 + index as f64 * 0.01, -0.1))
        .collect::<Vec<_>>();
    let dense_action = matrix.matvec(&compression_probe);
    let compressed_action = compressed
        .apply_checked(&compression_probe)
        .map_err(|error| error.to_string())?;
    let compressed_action_error = dense_action
        .iter()
        .zip(&compressed_action)
        .map(|(dense, compressed)| (*dense - *compressed).norm())
        .fold(0.0_f64, f64::max);
    if !compressed_action_error.is_finite() || compressed_action_error > 1.0e-7 {
        return Err(format!(
            "compressed Galerkin action error {compressed_action_error:e} exceeds 1e-7"
        ));
    }
    let kernel_native = compress_galerkin_muller_transmission_with_global_dofs(
        &elements,
        &mesh.nodes,
        &exterior,
        &interior,
        BasisOrder::Linear,
        2,
        global_dof_count,
        2,
        1.0e-8,
        2,
    )
    .map_err(|error| error.to_string())?;
    let kernel_native_action = kernel_native
        .apply_checked(&compression_probe)
        .map_err(|error| error.to_string())?;
    let kernel_native_action_error = dense_action
        .iter()
        .zip(&kernel_native_action)
        .map(|(dense, compressed)| (*dense - *compressed).norm())
        .fold(0.0_f64, f64::max);
    if !kernel_native_action_error.is_finite() || kernel_native_action_error > 1.0e-7 {
        return Err(format!(
            "kernel-native Galerkin action error {kernel_native_action_error:e} exceeds 1e-7"
        ));
    }
    let mut compressed_config = GmresConfig::default();
    compressed_config.max_iterations = 80;
    compressed_config.restart = 24;
    compressed_config.tolerance = 1.0e-10;
    compressed_config.exact_residual_check_interval = 1;
    let mut nearfield = DenseMatrix::zeros(operators.total_dofs(), operators.total_dofs());
    for index in 0..operators.total_dofs() {
        nearfield[[index, index]] = num_complex::Complex64::new(1.0, 0.0);
    }
    if matrix
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("Galerkin Müller matrix contains a non-finite entry".to_string());
    }
    let pressure_block_norm = (0..global_dof_count)
        .flat_map(|row| (0..global_dof_count).map(move |column| matrix[[row, column]].norm_sqr()))
        .sum::<f64>()
        .sqrt();
    let velocity_block_norm = (0..global_dof_count)
        .flat_map(|row| {
            (0..global_dof_count).map(move |column| {
                matrix[[global_dof_count + row, global_dof_count + column]].norm_sqr()
            })
        })
        .sum::<f64>()
        .sqrt();
    if !pressure_block_norm.is_finite()
        || !velocity_block_norm.is_finite()
        || pressure_block_norm <= 0.0
        || velocity_block_norm <= 0.0
    {
        return Err("Galerkin Müller diagonal blocks must be finite and nonzero".to_string());
    }
    let pressure_trace = (0..global_dof_count)
        .map(|index| num_complex::Complex64::new(0.1 + index as f64 * 0.01, -0.02))
        .collect::<Vec<_>>();
    let velocity_trace = (0..global_dof_count)
        .map(|index| num_complex::Complex64::new(-0.03, 0.05 + index as f64 * 0.02))
        .collect::<Vec<_>>();
    let rhs = operators
        .incident_rhs_from_trace_coefficients(&pressure_trace, &velocity_trace)
        .map_err(|error| error.to_string())?;
    let solution = operators
        .solve(&rhs, 1.0e-8)
        .map_err(|error| error.to_string())?;
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    let residual = (0..operators.total_dofs())
        .map(|row| {
            let action = (0..operators.total_dofs())
                .map(|column| matrix[[row, column]] * solution[column])
                .sum::<num_complex::Complex64>();
            (action - rhs[row]).norm_sqr()
        })
        .sum::<f64>()
        .sqrt()
        / rhs_norm.max(1.0e-30);
    if !residual.is_finite() || residual > 1.0e-8 {
        return Err(format!(
            "Galerkin Müller solve residual {residual:.3e} exceeds 1e-8"
        ));
    }
    let incident = IncidentField::plane_wave_z();
    let projected_rhs = operators
        .incident_rhs_from_field(&elements, &mesh.nodes, &incident, &exterior, 3)
        .map_err(|error| error.to_string())?;
    let projected_solution = operators
        .solve(&projected_rhs, 1.0e-8)
        .map_err(|error| error.to_string())?;
    let batch_rhs = vec![rhs.clone(), projected_rhs.clone()];
    let forward_batch = operators
        .solve_many(&batch_rhs, 1.0e-8)
        .map_err(|error| error.to_string())?;
    let adjoint_batch = operators
        .solve_adjoint_many(&batch_rhs, 1.0e-8)
        .map_err(|error| error.to_string())?;
    let mut batch_forward_error = 0.0_f64;
    let mut batch_adjoint_error = 0.0_f64;
    for (column, batch_solution) in forward_batch.iter().enumerate() {
        let scalar_solution = operators
            .solve(&batch_rhs[column], 1.0e-8)
            .map_err(|error| error.to_string())?;
        batch_forward_error = batch_forward_error.max(
            batch_solution
                .iter()
                .zip(&scalar_solution)
                .map(|(batch, scalar)| (batch - scalar).norm())
                .fold(0.0_f64, f64::max),
        );
        let scalar_adjoint = operators
            .solve_adjoint(&batch_rhs[column], 1.0e-8)
            .map_err(|error| error.to_string())?;
        batch_adjoint_error = batch_adjoint_error.max(
            adjoint_batch[column]
                .iter()
                .zip(&scalar_adjoint)
                .map(|(batch, scalar)| (batch - scalar).norm())
                .fold(0.0_f64, f64::max),
        );
    }
    if !batch_forward_error.is_finite()
        || !batch_adjoint_error.is_finite()
        || batch_forward_error > 1.0e-11
        || batch_adjoint_error > 1.0e-11
    {
        return Err(format!(
            "Galerkin Müller batch parity errors {batch_forward_error:.3e}/{batch_adjoint_error:.3e} exceed 1e-11"
        ));
    }
    let projected_rhs_norm = projected_rhs
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let projected_residual = (0..operators.total_dofs())
        .map(|row| {
            let action = (0..operators.total_dofs())
                .map(|column| operators.matrix()[[row, column]] * projected_solution[column])
                .sum::<num_complex::Complex64>();
            (action - projected_rhs[row]).norm_sqr()
        })
        .sum::<f64>()
        .sqrt()
        / projected_rhs_norm.max(1.0e-30);
    if !projected_residual.is_finite() || projected_residual > 1.0e-8 {
        return Err(format!(
            "projected Galerkin Müller residual {projected_residual:.3e} exceeds 1e-8"
        ));
    }
    let field = operators
        .evaluate_exterior_field_checked(
            &elements,
            &mesh.nodes,
            &projected_solution,
            &[[0.0, 0.0, 0.8], [0.31, -0.12, 0.77]],
            &incident,
            &exterior,
            3,
        )
        .map_err(|error| error.to_string())?;
    if field.len() != 2
        || field.iter().any(|value| {
            !value.p_incident.re.is_finite()
                || !value.p_incident.im.is_finite()
                || !value.p_scattered.re.is_finite()
                || !value.p_scattered.im.is_finite()
                || !value.p_total.re.is_finite()
                || !value.p_total.im.is_finite()
        })
    {
        return Err("Galerkin Müller exterior field contains non-finite values".to_string());
    }
    let maximum_field_magnitude = field
        .iter()
        .map(|value| value.p_total.norm())
        .fold(0.0, f64::max);
    if !maximum_field_magnitude.is_finite() || maximum_field_magnitude <= 0.0 {
        return Err("Galerkin Müller exterior field is empty or zero".to_string());
    }
    let compressed_solution = kernel_native
        .solve_with_ilu(&projected_rhs, &nearfield, &compressed_config, 1.0e-8)
        .map_err(|error| error.to_string())?;
    let compressed_field = operators
        .evaluate_exterior_field_from_compressed_solution_checked(
            &compressed_solution,
            &elements,
            &mesh.nodes,
            &[[0.0, 0.0, 0.8], [0.31, -0.12, 0.77]],
            &incident,
            &exterior,
            3,
        )
        .map_err(|error| error.to_string())?;
    let compressed_field_error = field
        .iter()
        .zip(&compressed_field)
        .map(|(dense, compressed)| (dense.p_total - compressed.p_total).norm())
        .fold(0.0_f64, f64::max);
    if !compressed_field_error.is_finite() || compressed_field_error > 1.0e-7 {
        return Err(format!(
            "compressed Galerkin field error {compressed_field_error:e} exceeds 1e-7"
        ));
    }
    Ok((
        operators.total_dofs(),
        pressure_block_norm.max(velocity_block_norm),
        residual,
        projected_residual,
        batch_forward_error,
        batch_adjoint_error,
        maximum_field_magnitude,
        compressed_action_error,
        kernel_native_action_error,
        compressed_field_error,
        compressed.compressed_block_count(),
        kernel_native.compressed_block_count(),
    ))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((
            dimension,
            block_norm,
            residual,
            projected_residual,
            batch_forward_error,
            batch_adjoint_error,
            maximum_field_magnitude,
            compressed_action_error,
            kernel_native_action_error,
            compressed_field_error,
            compressed_blocks,
            kernel_native_blocks,
        )) => QaResult::passed(CASE_ID)
            .input_checksum("bem:icosphere:subdivision-0:closed-conforming-linear-tri3")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-galerkin-muller-transmission-v1",
            )
            .dimension(format!(
                "3-D CPU {dimension}-DOF two-medium Galerkin Müller operator"
            ))
            .metric("maximum_diagonal_block_norm", block_norm)
            .minimum("maximum_diagonal_block_norm", f64::MIN_POSITIVE)
            .metric("relative_solve_residual", residual)
            .maximum("relative_solve_residual", 1.0e-8)
            .metric("projected_incident_rhs_residual", projected_residual)
            .maximum("projected_incident_rhs_residual", 1.0e-8)
            .metric("batch_forward_parity_error", batch_forward_error)
            .maximum("batch_forward_parity_error", 1.0e-11)
            .metric("batch_adjoint_parity_error", batch_adjoint_error)
            .maximum("batch_adjoint_parity_error", 1.0e-11)
            .metric("maximum_exterior_field_magnitude", maximum_field_magnitude)
            .minimum("maximum_exterior_field_magnitude", f64::MIN_POSITIVE)
            .metric("compressed_action_error", compressed_action_error)
            .maximum("compressed_action_error", 1.0e-7)
            .metric("kernel_native_action_error", kernel_native_action_error)
            .maximum("kernel_native_action_error", 1.0e-7)
            .metric("compressed_field_error", compressed_field_error)
            .maximum("compressed_field_error", 1.0e-7)
            .metric("compressed_blocks", compressed_blocks as f64)
            .minimum("compressed_blocks", 1.0)
            .metric("kernel_native_blocks", kernel_native_blocks as f64)
            .minimum("kernel_native_blocks", 1.0)
            .seed("bem-galerkin-muller-transmission:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if !matches!(result.status, QaStatus::Passed) {
        std::process::exit(1);
    }
}
