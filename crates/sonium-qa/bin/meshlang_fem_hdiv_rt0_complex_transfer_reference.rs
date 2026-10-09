//! Bounded release reference for complex RT0 H(div) MeshLang transfer.
//!
//! This fixture exercises only the FEM transfer bridge and checked RT0 kernel;
//! it does not invoke a solver, mesher, Gmsh, or simulation workload.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use num_complex::Complex64;
use sonium_fem::mesh::{
    apply_complex_hdiv_rt0_state_transfer, transfer_hdiv_rt0_state_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HdivSpace};
use sonium_qa::QaResult;
use std::collections::BTreeMap;

const CASE_ID: &str = "meshlang.fem-hdiv-rt0-complex-transfer-reference.v1";

fn edge_id(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn run() -> Result<QaResult, String> {
    let source_mesh = unit_square_triangles(1);
    let target_mesh = unit_square_triangles(2);
    let source_space = HdivSpace::new();
    let target_space = HdivSpace::new();
    let source_dofs = source_space.num_global_dofs(&source_mesh);
    let target_dofs = target_space.num_global_dofs(&target_mesh);
    let source_values: Vec<_> = (0..source_dofs)
        .map(|index| Complex64::new(0.25 + index as f64, -0.5 - 2.0 * index as f64))
        .collect();
    let real_source: Vec<_> = source_values.iter().map(|value| value.re).collect();
    let imaginary_source: Vec<_> = source_values.iter().map(|value| value.im).collect();
    let expected_real = transfer_hdiv_rt0_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &real_source,
    )
    .map_err(|error| format!("real RT0 transfer: {error}"))?;
    let expected_imaginary = transfer_hdiv_rt0_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &imaginary_source,
    )
    .map_err(|error| format!("imaginary RT0 transfer: {error}"))?;

    let target_entities = (0..target_dofs)
        .map(|index| (edge_id(index), index))
        .collect::<BTreeMap<_, _>>();
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
    let transferred = apply_complex_hdiv_rt0_state_transfer::<String, _>(
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
    .map_err(|error| format!("complex MeshLang RT0 bridge: {error}"))?;

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
            "complex RT0 transfer mismatch: max error {max_error:e}"
        ));
    }

    Ok(QaResult::passed(CASE_ID)
        .actual_backend("cpu")
        .backend_passed("cpu")
        .metric("target_flux_moment_checks", target_dofs as f64)
        .minimum("target_flux_moment_checks", target_dofs as f64)
        .metric("component_checks", (2 * target_dofs) as f64)
        .minimum("component_checks", (2 * target_dofs) as f64)
        .metric("max_error", max_error)
        .maximum("max_error", 1.0e-10)
        .reference("analytic checked RT0 real/imaginary component transfer")
        .reference_checksum("analytic/unit-square-rt0-complex-state-transfer-v1")
        .dimension("2-D affine triangular complex RT0")
        .formulation("IrStateTransferPlan complex H(div) checked consumer")
        .tolerance("real and imaginary flux moments agree checked RT0 kernel")
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
            .expect("complex RT0 MeshLang QA result must serialize")
    );
}
