//! CPU-only MeshLang-to-FEM mixed-order scalar H1 state-transfer reference.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{
    Mesh, MeshlangTransferError, apply_h1_mixed_state_transfer, unit_square_triangles,
};
use sonium_fem::spaces::H1Space;
use sonium_qa::QaResult;
use std::collections::BTreeMap;
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-h1-mixed-transfer-reference.v1";

fn node_entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn entity_map(mesh: &Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|index| (node_entity(index), index))
        .collect()
}

fn linear_field(point: sonium_fem::mesh::Point) -> f64 {
    0.75 + 1.2 * point.x - 0.8 * point.y
}

fn quadratic_field(point: sonium_fem::mesh::Point) -> f64 {
    0.5 + 1.2 * point.x - 0.8 * point.y + 1.5 * point.x * point.x - 0.4 * point.x * point.y
        + 0.6 * point.y * point.y
}

fn interpolation_plan(mesh: &Mesh) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..mesh.num_nodes())
            .map(|index| IrStateTransferEntry {
                entity: node_entity(index),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

fn transfer_error(
    source_mesh: &Mesh,
    target_mesh: &Mesh,
    source_space: &H1Space,
    target_space: &H1Space,
    field: fn(sonium_fem::mesh::Point) -> f64,
) -> Result<f64, String> {
    let source_values = source_mesh
        .nodes
        .iter()
        .copied()
        .map(field)
        .collect::<Vec<_>>();
    let transferred = apply_h1_mixed_state_transfer::<&'static str, _>(
        &interpolation_plan(target_mesh),
        source_mesh,
        target_mesh,
        source_space,
        target_space,
        &source_values,
        &entity_map(source_mesh),
        &entity_map(target_mesh),
        |_, _, _| Ok(0.0),
    )
    .map_err(|error| error.to_string())?;

    Ok(target_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(index, point)| (transferred[&node_entity(index)] - field(*point)).abs())
        .fold(0.0_f64, f64::max))
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let p1_source = unit_square_triangles(1);
    let p2_target = unit_square_triangles(2).to_p2();
    let p2_source = p2_target.clone();
    let p1_target = unit_square_triangles(1);

    let p1_to_p2 = transfer_error(
        &p1_source,
        &p2_target,
        &H1Space::new(PolynomialDegree::P1),
        &H1Space::new(PolynomialDegree::P2),
        linear_field,
    )?;
    let p2_to_p1 = transfer_error(
        &p2_source,
        &p1_target,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P1),
        quadratic_field,
    )?;

    let mismatch = apply_h1_mixed_state_transfer::<&'static str, _>(
        &interpolation_plan(&p1_target),
        &p1_source,
        &p1_target,
        &H1Space::new(PolynomialDegree::P2),
        &H1Space::new(PolynomialDegree::P1),
        &vec![0.0; p1_source.num_nodes()],
        &entity_map(&p1_source),
        &entity_map(&p1_target),
        |_, _, _| Ok(0.0),
    );
    if !matches!(mismatch, Err(MeshlangTransferError::InvalidH1Space { .. })) {
        return Err("mixed-order degree/geometry mismatch was not rejected".to_string());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("p1_to_p2_max_error", p1_to_p2)
        .maximum("p1_to_p2_max_error", 1.0e-12)
        .metric("p2_to_p1_max_error", p2_to_p1)
        .maximum("p2_to_p1_max_error", 1.0e-12)
        .metric("rejection_checks", 1.0)
        .minimum("rejection_checks", 1.0)
        .dimension("2-D affine triangular mixed-order H1 P1/P2 transfer")
        .formulation("canonical MeshLang plan through checked FEM nodal interpolation")
        .reference("analytic linear and quadratic fields")
        .reference_checksum("analytic-p1-p2-fields:meshlang-fem-h1-mixed-v1")
        .input_checksum("unit-square-triangles-p1-p2-affine:meshlang-h1-mixed-v1")
        .seed("meshlang-fem-h1-mixed-transfer:cpu:v1")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!("{}", result.to_json_line().expect("serialize QA JSON"));
    if !matches!(result.status, sonium_qa::QaStatus::Passed) {
        std::process::exit(1);
    }
}
