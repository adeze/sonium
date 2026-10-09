//! CPU-only MeshLang-to-FEM mixed-order triangular H(curl) reference.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::mesh::{
    apply_hcurl_mixed_state_transfer, transfer_hcurl_mixed_triangle_state_checked,
    unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HcurlSpace};
use sonium_qa::QaResult;
use std::collections::BTreeMap;
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-hcurl-mixed-transfer-reference.v1";

fn dof_entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn entities(dofs: usize) -> BTreeMap<EntityId, usize> {
    (0..dofs).map(|dof| (dof_entity(dof), dof)).collect()
}

fn plan(dofs: usize) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: (0..dofs)
            .map(|dof| IrStateTransferEntry {
                entity: dof_entity(dof),
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

fn curved_triangle_mesh(midpoint_offset: f64) -> sonium_fem::mesh::Mesh {
    let mut mesh = unit_square_triangles(1).to_p2();
    let midpoint = mesh.elements[0].nodes[3];
    mesh.nodes[midpoint].y += midpoint_offset;
    mesh
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let mesh = unit_square_triangles(1);
    let cases = [
        (HcurlSpace::new(), HcurlSpace::order2()),
        (HcurlSpace::order2(), HcurlSpace::new()),
    ];
    let mut max_error = 0.0_f64;
    let mut curved_max_error = 0.0_f64;
    let mut total_source_dofs = 0_usize;
    let mut total_target_dofs = 0_usize;

    for (source_space, target_space) in cases {
        let source_dofs = source_space.num_global_dofs(&mesh);
        let target_dofs = target_space.num_global_dofs(&mesh);
        let source_values: Vec<_> = (0..source_dofs)
            .map(|dof| 0.25 + 0.125 * dof as f64 - 0.01 * (dof as f64).powi(2))
            .collect();
        let expected = transfer_hcurl_mixed_triangle_state_checked(
            &mesh,
            &mesh,
            &source_space,
            &target_space,
            &source_values,
        )
        .map_err(|error| error.to_string())?;
        let source_entities = entities(source_dofs);
        let target_entities = entities(target_dofs);
        let actual = apply_hcurl_mixed_state_transfer::<String, _>(
            &plan(target_dofs),
            &mesh,
            &mesh,
            &source_space,
            &target_space,
            &source_values,
            &source_entities,
            &target_entities,
            |_, _, _| -> Result<f64, String> {
                Err("interpolation callback unexpectedly invoked".into())
            },
        )
        .map_err(|error| error.to_string())?;
        if actual.len() != target_dofs {
            return Err(format!(
                "mixed bridge returned {} values, expected {target_dofs}",
                actual.len()
            ));
        }
        for (dof, expected_value) in expected.iter().enumerate() {
            let error = (actual[&dof_entity(dof)] - expected_value).abs();
            max_error = max_error.max(error);
        }
        total_source_dofs += source_dofs;
        total_target_dofs += target_dofs;
    }
    let curved_cases = [
        (
            unit_square_triangles(1),
            curved_triangle_mesh(0.08),
            HcurlSpace::new(),
            HcurlSpace::order2(),
        ),
        (
            curved_triangle_mesh(-0.08),
            unit_square_triangles(1),
            HcurlSpace::order2(),
            HcurlSpace::new(),
        ),
    ];
    for (source_mesh, target_mesh, source_space, target_space) in curved_cases {
        let source_dofs = source_space.num_global_dofs(&source_mesh);
        let target_dofs = target_space.num_global_dofs(&target_mesh);
        let source_values: Vec<_> = (0..source_dofs)
            .map(|dof| 0.17 + 0.09 * dof as f64 - 0.003 * (dof as f64).powi(2))
            .collect();
        let expected = transfer_hcurl_mixed_triangle_state_checked(
            &source_mesh,
            &target_mesh,
            &source_space,
            &target_space,
            &source_values,
        )
        .map_err(|error| error.to_string())?;
        let source_entities = entities(source_dofs);
        let target_entities = entities(target_dofs);
        let actual = apply_hcurl_mixed_state_transfer::<String, _>(
            &plan(target_dofs),
            &source_mesh,
            &target_mesh,
            &source_space,
            &target_space,
            &source_values,
            &source_entities,
            &target_entities,
            |_, _, _| -> Result<f64, String> {
                Err("curved interpolation callback unexpectedly invoked".into())
            },
        )
        .map_err(|error| error.to_string())?;
        if actual.len() != target_dofs {
            return Err(format!(
                "curved mixed bridge returned {} values, expected {target_dofs}",
                actual.len()
            ));
        }
        for (dof, expected_value) in expected.iter().enumerate() {
            let error = (actual[&dof_entity(dof)] - expected_value).abs();
            curved_max_error = curved_max_error.max(error);
            max_error = max_error.max(error);
        }
        total_source_dofs += source_dofs;
        total_target_dofs += target_dofs;
    }

    if max_error > 1.0e-11 {
        return Err(format!(
            "mixed H(curl) bridge error {max_error:e} exceeds tolerance"
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .metric("max_hcurl_mixed_error", max_error)
        .metric("curved_cases", 2.0)
        .metric("curved_max_error", curved_max_error)
        .metric("source_dofs_both_directions", total_source_dofs as f64)
        .metric("target_dofs_both_directions", total_target_dofs as f64)
        .dimension("2-D affine and curved mixed-order triangular H(curl)")
        .input_checksum("unit-square-and-curved-p2-triangles:meshlang-hcurl-mixed-v2")
        .reference_checksum("checked-kernel-bridge-affine-curved:meshlang-hcurl-mixed-v2")
        .seed("meshlang-fem-hcurl-mixed-transfer:cpu:v1")
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
