//! Release-facing FEM consumer for a canonical MeshLang state-transfer plan.

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use num_complex::Complex64;
use sonium_fem::mesh::meshlang_transfer::apply_complex_nodal_state_transfer;
use sonium_fem::mesh::meshlang_transfer::apply_nodal_state_transfer;
use sonium_fem::mesh::{create_interpolation_data_checked, unit_square_triangles};
use sonium_qa::{QaResult, QaStatus};
use std::collections::BTreeMap;

const CASE_ID: &str = "meshlang.fem-state-transfer-reference.v1";

fn id(index: u64) -> EntityId {
    EntityId::new(EntityDimension::Point, index)
}

fn run() -> Result<QaResult, String> {
    let mesh = unit_square_triangles(1);
    let data = create_interpolation_data_checked(&mesh, &mesh)
        .map_err(|error| format!("interpolation metadata: {error}"))?;
    let source_values: Vec<_> = mesh
        .nodes
        .iter()
        .map(|point| 1.0 + point.x + 2.0 * point.y)
        .collect();
    let source_entities: BTreeMap<_, _> = (0..mesh.num_nodes())
        .map(|node| (id(node as u64), node))
        .collect();
    let target_entities = source_entities.clone();
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::ApplicationDefined,
        entries: vec![
            IrStateTransferEntry {
                entity: id(0),
                action: IrStateTransferAction::Preserve,
            },
            IrStateTransferEntry {
                entity: id(1),
                action: IrStateTransferAction::Interpolate,
            },
            IrStateTransferEntry {
                entity: id(2),
                action: IrStateTransferAction::Conservative,
            },
            IrStateTransferEntry {
                entity: id(3),
                action: IrStateTransferAction::Initialize,
            },
            IrStateTransferEntry {
                entity: id(4),
                action: IrStateTransferAction::ApplicationDefined,
            },
            IrStateTransferEntry {
                entity: id(5),
                action: IrStateTransferAction::Drop,
            },
        ],
    };
    let transferred = apply_nodal_state_transfer::<&'static str, _>(
        &plan,
        &mesh,
        &mesh,
        &data,
        &source_values,
        &source_entities,
        &target_entities,
        |_, action, source| match action {
            IrStateTransferAction::Initialize => Ok(7.5),
            IrStateTransferAction::ApplicationDefined => Ok(source.unwrap_or(5.0) + 0.5),
            _ => Err("unexpected callback action"),
        },
    )
    .map_err(|error| error.to_string())?;
    let complex_source: Vec<_> = source_values
        .iter()
        .map(|value| Complex64::new(*value, -*value))
        .collect();
    let complex = apply_complex_nodal_state_transfer::<&'static str, _>(
        &plan,
        &mesh,
        &mesh,
        &data,
        &complex_source,
        &source_entities,
        &target_entities,
        |_, action, source| match action {
            IrStateTransferAction::Initialize => Ok(Complex64::new(7.5, -7.5)),
            IrStateTransferAction::ApplicationDefined => {
                Ok(source.unwrap_or(Complex64::new(5.0, -5.0)) + Complex64::new(0.5, -0.5))
            }
            _ => Err("unexpected callback action"),
        },
    )
    .map_err(|error| error.to_string())?;

    let mut checks = 0_u64;
    let mut check = |condition: bool, message: &str| -> Result<(), String> {
        checks += 1;
        if condition {
            Ok(())
        } else {
            Err(message.to_owned())
        }
    };
    check(
        transferred.get(&id(0)) == source_values.first(),
        "preserve action changed the source value",
    )?;
    check(
        transferred.get(&id(1)) == source_values.get(1),
        "interpolation action disagrees on an identical mesh",
    )?;
    check(
        transferred.get(&id(2)) == source_values.get(2),
        "conservative action disagrees on an identical mesh",
    )?;
    check(
        transferred.get(&id(3)) == Some(&7.5),
        "initialize callback result missing",
    )?;
    check(
        transferred.get(&id(4)) == Some(&5.5),
        "application-defined callback result missing",
    )?;
    check(
        !transferred.contains_key(&id(5)) && transferred.values().all(|value| value.is_finite()),
        "drop or finite-output contract failed",
    )?;
    check(
        complex.get(&id(0)) == complex_source.first()
            && complex.get(&id(1)) == complex_source.get(1)
            && complex
                .values()
                .all(|value| value.re.is_finite() && value.im.is_finite()),
        "complex FEM state-transfer contract failed",
    )?;

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .backend_passed("cpu")
        .metric("checks", checks as f64)
        .minimum("checks", 7.0)
        .metric("output_entities", transferred.len() as f64)
        .minimum("output_entities", 5.0)
        .metric("source_nodes", source_values.len() as f64)
        .minimum("source_nodes", 4.0)
        .reference("existing checked FEM nodal and conservative interpolation kernels")
        .reference_checksum("analytic/identity-mesh:meshlang-fem-state-transfer-v1")
        .dimension("2-D unit-square triangular nodal field")
        .formulation("MeshLang IrStateTransferPlan to FEM checked consumer")
        .tolerance("all six action checks exact; all output values finite")
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
            .expect("state-transfer QA result must serialize")
    );
    if result.status != QaStatus::Passed {
        std::process::exit(1);
    }
}
