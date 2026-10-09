//! Bounded CPU evidence for conforming affine-Tri3 Galerkin hypersingular W.

use std::time::Instant;

use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::{
    BasisOrder, GalerkinAssemblyError, PhysicsParams,
    assemble_galerkin_hypersingular_with_global_dofs,
};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "bem.galerkin-hypersingular-weak-form.v1";

fn run() -> Result<(usize, f64, bool), String> {
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
    let physics =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).map_err(|error| error.to_string())?;
    let operators = assemble_galerkin_hypersingular_with_global_dofs(
        &elements,
        &mesh.nodes,
        &physics,
        BasisOrder::Linear,
        3,
        unique_nodes.len(),
    )
    .map_err(|error| error.to_string())?;
    if operators
        .hypersingular
        .as_slice()
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("Galerkin hypersingular matrix contains a non-finite entry".to_string());
    }
    let maximum_asymmetry = (0..unique_nodes.len())
        .flat_map(|row| (0..unique_nodes.len()).map(move |column| (row, column)))
        .map(|(row, column)| {
            (operators.hypersingular[[row, column]] - operators.hypersingular[[column, row]]).norm()
        })
        .fold(0.0_f64, f64::max);
    if maximum_asymmetry > 1.0e-12 {
        return Err(format!(
            "Galerkin hypersingular reciprocity error {maximum_asymmetry:.3e} exceeds 1e-12"
        ));
    }
    let mut malformed = elements.clone();
    malformed[0].connectivity[0] = mesh.nodes.len();
    let node_rejection = matches!(
        assemble_galerkin_hypersingular_with_global_dofs(
            &malformed,
            &mesh.nodes,
            &physics,
            BasisOrder::Linear,
            3,
            unique_nodes.len(),
        ),
        Err(GalerkinAssemblyError::NodeOutOfRange { .. })
    );
    if !node_rejection {
        return Err("Galerkin hypersingular node-bound rejection was not typed".into());
    }

    Ok((unique_nodes.len(), maximum_asymmetry, node_rejection))
}

fn main() {
    let started = Instant::now();
    let result = match run() {
        Ok((dimension, maximum_asymmetry, node_rejection)) => QaResult::passed(CASE_ID)
            .input_checksum("bem:icosphere:subdivision-0:closed-conforming-linear-tri3")
            .reference_checksum(
                "analytic/deterministic-fixture:bem-galerkin-hypersingular-weak-form-v1",
            )
            .dimension(format!(
                "3-D CPU {dimension}-DOF closed conforming P1 Tri3 weak W"
            ))
            .metric("maximum_asymmetry", maximum_asymmetry)
            .metric("node_rejection", if node_rejection { 1.0 } else { 0.0 })
            .maximum("maximum_asymmetry", 1.0e-12)
            .seed("bem-galerkin-hypersingular-weak-form:cpu:v1")
            .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX)),
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA result"));
    if !matches!(result.status, QaStatus::Passed) {
        std::process::exit(1);
    }
}
