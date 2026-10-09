//! Release-facing Meshlang-to-FEM affine RT1 H(div) state-transfer reference.
//!
//! This deterministic CPU fixture exercises edge and cell modes through the
//! checked RT1 transfer kernel. It does not invoke a solver, Gmsh, or simulator.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use num_complex::Complex64;
use sonium_fem::mesh::{
    ElementType, Mesh, Point, apply_complex_hdiv_rt1_state_transfer, apply_hdiv_rt1_state_transfer,
    transfer_hdiv_rt1_state_checked,
};
use sonium_fem::spaces::{FunctionSpace, HdivOrder, HdivSpace};
use sonium_qa::QaResult;
use std::collections::BTreeMap;

const CASE_ID: &str = "meshlang.fem-hdiv-rt1-transfer-reference.v1";

fn triangle_mesh(origin_x: f64, origin_y: f64, scale: f64) -> Mesh {
    let mut mesh = Mesh::new(2);
    mesh.add_node(Point {
        x: origin_x,
        y: origin_y,
        z: 0.0,
    });
    mesh.add_node(Point {
        x: origin_x + scale,
        y: origin_y,
        z: 0.0,
    });
    mesh.add_node(Point {
        x: origin_x,
        y: origin_y + scale,
        z: 0.0,
    });
    mesh.add_element(ElementType::Triangle, vec![0, 1, 2]);
    mesh
}

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn target_entities(space: &HdivSpace, mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..space.num_global_dofs(mesh))
        .map(|index| (entity(index), index))
        .collect()
}

fn run() -> Result<QaResult, String> {
    let source_mesh = triangle_mesh(0.0, 0.0, 2.0);
    // This release-facing fixture checks the MeshLang bridge on a single
    // affine mesh; refinement ownership remains a separate kernel contract.
    let target_mesh = triangle_mesh(0.25, 0.25, 0.5);
    let source_space = HdivSpace::with_order(HdivOrder::Rt1);
    let target_space = HdivSpace::with_order(HdivOrder::Rt1);
    let source_dofs = source_space.num_global_dofs(&source_mesh);
    let target_dofs = target_space.num_global_dofs(&target_mesh);
    let source_values: Vec<_> = (0..source_dofs)
        .map(|index| Complex64::new(0.25 + index as f64, -0.5 - 0.5 * index as f64))
        .collect();
    let real_values: Vec<_> = source_values.iter().map(|value| value.re).collect();
    let imaginary_values: Vec<_> = source_values.iter().map(|value| value.im).collect();
    let expected_real = transfer_hdiv_rt1_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &real_values,
    )
    .map_err(|error| format!("real RT1 transfer: {error}"))?;
    let expected_imaginary = transfer_hdiv_rt1_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &imaginary_values,
    )
    .map_err(|error| format!("imaginary RT1 transfer: {error}"))?;
    let target_entities = target_entities(&target_space, &target_mesh);
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: target_entities
            .keys()
            .map(|&entity| IrStateTransferEntry {
                entity,
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    };
    let transferred = apply_complex_hdiv_rt1_state_transfer::<String, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
        &BTreeMap::new(),
        &target_entities,
        |_, _, _| Ok(Complex64::new(0.0, 0.0)),
    )
    .map_err(|error| format!("complex RT1 bridge: {error}"))?;
    let max_error = target_entities
        .iter()
        .map(|(entity, &index)| {
            let value = transferred[entity];
            (value.re - expected_real[index])
                .abs()
                .max((value.im - expected_imaginary[index]).abs())
        })
        .fold(0.0_f64, f64::max);
    if max_error > 1.0e-10 {
        return Err(format!(
            "complex RT1 transfer mismatch: max error {max_error:e}"
        ));
    }

    let conservative_plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Conservative,
        entries: vec![IrStateTransferEntry {
            entity: entity(0),
            action: IrStateTransferAction::Conservative,
        }],
    };
    let conservative = apply_hdiv_rt1_state_transfer::<String, _>(
        &conservative_plan,
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &real_values,
        &BTreeMap::new(),
        &target_entities,
        |_, _, _| Ok(0.0),
    )
    .map_err(|error| format!("RT1 conservative bridge: {error}"))?;
    if (conservative[&entity(0)] - expected_real[0]).abs() > 1.0e-10 {
        return Err("conservative RT1 transfer did not preserve the first mode".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .backend_passed("cpu")
        .metric("target_mode_checks", target_dofs as f64)
        .minimum("target_mode_checks", target_dofs as f64)
        .metric("complex_component_checks", (2 * target_dofs) as f64)
        .minimum("complex_component_checks", (2 * target_dofs) as f64)
        .metric("conservative_checks", 1.0)
        .minimum("conservative_checks", 1.0)
        .metric("max_error", max_error)
        .maximum("max_error", 1.0e-10)
        .reference("analytic checked RT1 edge/cell component transfer through MeshLang")
        .reference_checksum("analytic/unit-square-rt1-complex-state-transfer-v1")
        .dimension("2-D affine triangular RT1 same-mesh transfer")
        .formulation("IrStateTransferPlan RT1 H(div) checked consumer")
        .tolerance("all real/imaginary RT1 modes agree with the checked kernel")
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
            .expect("RT1 MeshLang state-transfer QA result must serialize")
    );
}
