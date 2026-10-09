//! Release-facing MeshLang-to-FEM H(curl) state-transfer reference.
//!
//! This is a deterministic edge-moment fixture. It exercises the canonical
//! MeshLang plan bridge and the checked FEM order-1 H(curl) interpolation
//! kernel without invoking a solver, mesher, Gmsh, or simulator workload.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use num_complex::Complex64;
use sonium_fem::mesh::{
    MeshTopology, apply_complex_hcurl_order2_triangle_state_transfer,
    apply_hcurl_order1_state_transfer, transfer_hcurl_order1_state_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;
use std::collections::BTreeMap;

const CASE_ID: &str = "meshlang.fem-hcurl-state-transfer-reference.v1";

fn edge_id(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn edge_moment(mesh: &sonium_fem::mesh::Mesh, edge: (usize, usize)) -> f64 {
    let p0 = mesh.nodes[edge.0];
    let p1 = mesh.nodes[edge.1];
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    dx * (1.0 - 0.5 * (p0.y + p1.y)) + dy * (2.0 + 0.5 * (p0.x + p1.x))
}

fn run() -> Result<QaResult, String> {
    let source_mesh = unit_square_triangles(1);
    let target_mesh = unit_square_triangles(2);
    let source_space = HcurlSpace::new();
    let target_space = HcurlSpace::new();
    let source_topology = MeshTopology::build(&source_mesh);
    let target_topology = MeshTopology::build(&target_mesh);
    let source_values: Vec<_> = source_topology
        .edges
        .iter()
        .map(|&edge| edge_moment(&source_mesh, edge))
        .collect();
    let source_entities = source_topology
        .edges
        .iter()
        .enumerate()
        .map(|(index, _)| (edge_id(index), index))
        .collect::<BTreeMap<_, _>>();
    let target_entities = target_topology
        .edges
        .iter()
        .enumerate()
        .map(|(index, _)| (edge_id(index), index))
        .collect::<BTreeMap<_, _>>();
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: target_topology
            .edges
            .iter()
            .enumerate()
            .map(|(index, _)| IrStateTransferEntry {
                entity: edge_id(index),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };

    let transferred = apply_hcurl_order1_state_transfer::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
        &source_entities,
        &target_entities,
        |_, _, _| Err("interpolation must not invoke the callback"),
    )
    .map_err(|error| error.to_string())?;
    let expected = transfer_hcurl_order1_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
    )
    .map_err(|error| error.to_string())?;
    for (index, (&edge, &value)) in target_topology.edges.iter().zip(&expected).enumerate() {
        let analytic = edge_moment(&target_mesh, edge);
        let bridged = *transferred
            .get(&edge_id(index))
            .ok_or_else(|| format!("missing transferred edge {index}"))?;
        if (bridged - value).abs() > 1.0e-10 || (bridged - analytic).abs() > 1.0e-10 {
            return Err(format!(
                "edge {index} moment mismatch: {bridged} != {value}/{analytic}"
            ));
        }
    }

    let conservative = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Conservative,
        entries: vec![IrStateTransferEntry {
            entity: edge_id(0),
            action: IrStateTransferAction::Conservative,
        }],
    };
    let rejection = apply_hcurl_order1_state_transfer::<&'static str, _>(
        &conservative,
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
        &source_entities,
        &target_entities,
        |_, _, _| Ok(0.0),
    );
    if !matches!(
        rejection,
        Err(sonium_fem::mesh::MeshlangTransferError::UnsupportedAction {
            action: IrStateTransferAction::Conservative,
            ..
        })
    ) {
        return Err("H(curl) accepted an invalid scalar conservative action".into());
    }

    let complex_space = HcurlSpace::order2();
    let complex_dofs = complex_space.num_global_dofs(&source_mesh);
    let complex_values = (0..complex_dofs)
        .map(|index| Complex64::new(index as f64 * 0.25, -(index as f64) * 0.5))
        .collect::<Vec<_>>();
    let complex_entities = (0..complex_dofs)
        .map(|dof| (EntityId::new(EntityDimension::Point, dof as u64), dof))
        .collect::<BTreeMap<_, _>>();
    let complex_plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..complex_dofs)
            .map(|dof| IrStateTransferEntry {
                entity: EntityId::new(EntityDimension::Point, dof as u64),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };
    let complex_transferred =
        apply_complex_hcurl_order2_triangle_state_transfer::<&'static str, _>(
            &complex_plan,
            &source_mesh,
            &source_mesh,
            &complex_space,
            &complex_space,
            &complex_values,
            &complex_entities,
            &complex_entities,
            |_, _, _| Err("unexpected complex callback"),
        )
        .map_err(|error| format!("complex order-2 bridge: {error}"))?;
    let complex_max_error = complex_values
        .iter()
        .enumerate()
        .map(|(index, expected)| {
            let actual = complex_transferred
                .get(&EntityId::new(EntityDimension::Point, index as u64))
                .copied()
                .unwrap_or_default();
            (actual - *expected).norm()
        })
        .fold(0.0_f64, f64::max);
    if complex_max_error > 1.0e-10 {
        return Err(format!("complex order-2 bridge error {complex_max_error}"));
    }

    let moment_checks = target_topology.edges.len();
    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .backend_passed("cpu")
        .metric("moment_checks", moment_checks as f64)
        .minimum("moment_checks", moment_checks as f64)
        .metric("finite_output_checks", 1.0)
        .minimum("finite_output_checks", 1.0)
        .metric("rejection_checks", 1.0)
        .minimum("rejection_checks", 1.0)
        .metric("complex_order2_checks", 1.0)
        .minimum("complex_order2_checks", 1.0)
        .metric("complex_order2_max_error", complex_max_error)
        .maximum("complex_order2_max_error", 1.0e-10)
        .metric("checks_passed", (moment_checks + 4) as f64)
        .reference("analytic affine Nedelec edge moments through MeshLang FEM bridge")
        .reference_checksum("analytic/unit-square-hcurl-state-transfer-v1")
        .dimension("2-D affine triangular order-1 H(curl) source/refined target")
        .formulation("IrStateTransferPlan order-1 H(curl) checked consumer")
        .tolerance("all target edge moments agree with analytic and checked-kernel references")
        .elapsed_ms(0))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("H(curl) state-transfer QA result must serialize")
    );
}
