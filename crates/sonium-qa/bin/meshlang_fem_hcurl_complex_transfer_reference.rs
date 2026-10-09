//! CPU-only reference for the complex MeshLang H(curl) consumer.

use std::collections::BTreeMap;
use std::time::Instant;

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use num_complex::Complex64;
use sonium_fem::mesh::{
    MeshTopology, MeshlangTransferError, apply_complex_hcurl_order1_state_transfer,
    unit_square_triangles,
};
use sonium_fem::spaces::HcurlSpace;
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "meshlang.fem-hcurl-complex-transfer-reference.v1";

fn edge_id(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn edge_value(mesh: &sonium_fem::mesh::Mesh, edge: (usize, usize)) -> f64 {
    let p0 = mesh.nodes[edge.0];
    let p1 = mesh.nodes[edge.1];
    (p1.x - p0.x) * (1.0 + 0.25 * (p0.x + p1.x)) + (p1.y - p0.y) * (0.5 - 0.75 * (p0.y + p1.y))
}

fn plan(mesh: &sonium_fem::mesh::Mesh, action: IrStateTransferAction) -> IrStateTransferPlan {
    let edge_count = MeshTopology::build(mesh).edges.len();
    IrStateTransferPlan {
        responsibility: match action {
            IrStateTransferAction::Conservative => StateTransferResponsibility::Conservative,
            _ => StateTransferResponsibility::Interpolate,
        },
        entries: (0..edge_count)
            .map(|index| IrStateTransferEntry {
                entity: edge_id(index),
                action,
            })
            .collect(),
    }
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let mesh = unit_square_triangles(1);
    let space = HcurlSpace::new();
    let edges = MeshTopology::build(&mesh).edges;
    let values = edges
        .iter()
        .map(|&edge| {
            let real = edge_value(&mesh, edge);
            Complex64::new(real, -0.5 * real)
        })
        .collect::<Vec<_>>();
    let entities = (0..edges.len())
        .map(|index| (edge_id(index), index))
        .collect::<BTreeMap<_, _>>();
    let transferred = apply_complex_hcurl_order1_state_transfer::<&'static str, _>(
        &plan(&mesh, IrStateTransferAction::Interpolate),
        &mesh,
        &mesh,
        &space,
        &space,
        &values,
        &entities,
        &entities,
        |_entity, _action, _previous| Err("unexpected callback"),
    )
    .map_err(|error| error.to_string())?;
    let max_error = values
        .iter()
        .enumerate()
        .map(|(index, value)| (transferred[&edge_id(index)] - value).norm())
        .fold(0.0_f64, f64::max);
    if max_error > 1.0e-12 {
        return Err(format!(
            "complex H(curl) error {max_error} exceeds tolerance"
        ));
    }

    let rejection = apply_complex_hcurl_order1_state_transfer::<&'static str, _>(
        &plan(&mesh, IrStateTransferAction::Conservative),
        &mesh,
        &mesh,
        &space,
        &space,
        &values,
        &entities,
        &entities,
        |_entity, _action, _previous| Err("must not run"),
    );
    if !matches!(
        rejection,
        Err(MeshlangTransferError::UnsupportedAction {
            action: IrStateTransferAction::Conservative,
            ..
        })
    ) {
        return Err("scalar conservative rejection was not typed".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("edge_dofs", values.len() as f64)
        .metric("complex_component_checks", (values.len() * 2) as f64)
        .metric("max_complex_error", max_error)
        .metric("rejection_checks", 1.0)
        .reference("same-mesh complex Nedelec edge-moment parity")
        .reference_checksum("analytic/deterministic-fixture:meshlang-fem-hcurl-complex-v1")
        .dimension("2-D affine triangular order-1 H(curl), Complex64")
        .formulation("independent checked real/imaginary Piola transfer")
        .tolerance("maximum complex edge-moment error <= 1e-12")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    let passed = result.status == QaStatus::Passed;
    println!(
        "{}",
        result.to_json_line().expect("serialize QA result JSON")
    );
    if !passed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
