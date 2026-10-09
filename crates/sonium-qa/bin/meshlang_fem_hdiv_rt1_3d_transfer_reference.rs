//! CPU-only MeshLang-to-FEM affine 3-D RT1 H(div) transfer reference.
//!
//! This fixture exercises real and Complex64 edge/face/cell moments through
//! the checked RT1 transfer bridge on a tetrahedral source/target pair.  It
//! does not invoke a solver, Gmsh, simulator, or broad QA workload.

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
use std::time::Instant;

const CASE_ID: &str = "meshlang.fem-hdiv-rt1-3d-transfer-reference.v1";

fn tetrahedron_mesh(origin: [f64; 3], scale: f64) -> Mesh {
    let mut mesh = Mesh::new(3);
    mesh.add_node(Point::new_3d(origin[0], origin[1], origin[2]));
    mesh.add_node(Point::new_3d(origin[0] + scale, origin[1], origin[2]));
    mesh.add_node(Point::new_3d(origin[0], origin[1] + scale, origin[2]));
    mesh.add_node(Point::new_3d(origin[0], origin[1], origin[2] + scale));
    mesh.add_element(ElementType::Tetrahedron, vec![0, 1, 2, 3]);
    mesh
}

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn entities(space: &HdivSpace, mesh: &Mesh) -> BTreeMap<EntityId, usize> {
    (0..space.num_global_dofs(mesh))
        .map(|index| (entity(index), index))
        .collect()
}

fn interpolate_plan(target: &BTreeMap<EntityId, usize>) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: target
            .keys()
            .map(|&entity| IrStateTransferEntry {
                entity,
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

fn run() -> Result<QaResult, String> {
    let started = Instant::now();
    let source_mesh = tetrahedron_mesh([0.0, 0.0, 0.0], 2.0);
    let target_mesh = tetrahedron_mesh([0.2, 0.15, 0.1], 0.6);
    let source_space = HdivSpace::with_order(HdivOrder::Rt1);
    let target_space = HdivSpace::with_order(HdivOrder::Rt1);
    let source_dofs = source_space.num_global_dofs(&source_mesh);
    let target_dofs = target_space.num_global_dofs(&target_mesh);
    if source_dofs != 15 || target_dofs != 15 {
        return Err(format!(
            "unexpected affine tetrahedral RT1 layout: source={source_dofs}, target={target_dofs}"
        ));
    }

    let source_values: Vec<f64> = (0..source_dofs)
        .map(|index| 0.25 + 0.5 * index as f64)
        .collect();
    let real_expected = transfer_hdiv_rt1_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
    )
    .map_err(|error| format!("real RT1 kernel: {error}"))?;

    let complex_source: Vec<Complex64> = (0..source_dofs)
        .map(|index| Complex64::new(0.25 + index as f64, -0.75 - 0.25 * index as f64))
        .collect();
    let complex_real_values: Vec<f64> = complex_source.iter().map(|value| value.re).collect();
    let imaginary_values: Vec<f64> = complex_source.iter().map(|value| value.im).collect();
    let complex_real_expected = transfer_hdiv_rt1_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &complex_real_values,
    )
    .map_err(|error| format!("complex-real RT1 kernel: {error}"))?;
    let imaginary_expected = transfer_hdiv_rt1_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &imaginary_values,
    )
    .map_err(|error| format!("imaginary RT1 kernel: {error}"))?;

    let target_entities = entities(&target_space, &target_mesh);
    let plan = interpolate_plan(&target_entities);
    let transferred = apply_hdiv_rt1_state_transfer::<String, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
        &BTreeMap::new(),
        &target_entities,
        |_, _, _| Err("interpolation must not invoke callback".into()),
    )
    .map_err(|error| format!("real RT1 bridge: {error}"))?;
    let complex_transferred = apply_complex_hdiv_rt1_state_transfer::<String, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &complex_source,
        &BTreeMap::new(),
        &target_entities,
        |_, _, _| Err("complex interpolation must not invoke callback".into()),
    )
    .map_err(|error| format!("complex RT1 bridge: {error}"))?;

    let mut max_error = 0.0_f64;
    let mut max_entity = None;
    let mut complex_checks = 0usize;
    for (&entity, &index) in &target_entities {
        let real_error = (transferred[&entity] - real_expected[index]).abs();
        if real_error > max_error {
            max_error = real_error;
            max_entity = Some((index, "real"));
        }
        let value = complex_transferred[&entity];
        let complex_real_error = (value.re - complex_real_expected[index]).abs();
        if complex_real_error > max_error {
            max_error = complex_real_error;
            max_entity = Some((index, "complex-real"));
        }
        let complex_imag_error = (value.im - imaginary_expected[index]).abs();
        if complex_imag_error > max_error {
            max_error = complex_imag_error;
            max_entity = Some((index, "complex-imag"));
        }
        complex_checks += 2;
    }
    if max_error > 1.0e-10 {
        return Err(format!(
            "3-D RT1 bridge error {max_error:e} exceeds tolerance at {max_entity:?}"
        ));
    }

    let mismatch = apply_hdiv_rt1_state_transfer::<String, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &HdivSpace::new(),
        &target_space,
        &source_values,
        &BTreeMap::new(),
        &target_entities,
        |_, _, _| Ok(0.0),
    )
    .expect_err("RT0/RT1 mismatch must be rejected");
    if !mismatch.to_string().contains("RT1") {
        return Err(format!("unexpected order rejection: {mismatch}"));
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .backend_passed("cpu")
        .metric("target_mode_checks", target_dofs as f64)
        .minimum("target_mode_checks", target_dofs as f64)
        .metric("complex_component_checks", complex_checks as f64)
        .minimum("complex_component_checks", complex_checks as f64)
        .metric("rejection_checks", 1.0)
        .minimum("rejection_checks", 1.0)
        .metric("max_error", max_error)
        .maximum("max_error", 1.0e-10)
        .reference("affine tetrahedral RT1 edge, face, and cell moments")
        .reference_checksum("analytic/affine-tetrahedron-rt1-3d-meshlang-v1")
        .dimension("3-D affine tetrahedral RT1 H(div) MeshLang bridge")
        .formulation("real and Complex64 checked RT1 transfer with order rejection")
        .elapsed_ms(started.elapsed().as_millis() as u64))
}

fn main() {
    let result = match run() {
        Ok(result) => result,
        Err(error) => QaResult::failed(CASE_ID, error),
    };
    println!(
        "{}",
        result.to_json_line().expect("serialize RT1 QA result")
    );
}
