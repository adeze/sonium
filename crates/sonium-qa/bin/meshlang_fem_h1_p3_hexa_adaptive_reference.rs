use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::refinement::{
    coarsen_elements_checked, refine_elements_high_order_with_history_checked,
};
use sonium_fem::mesh::{
    ElementType, Mesh, Point, RefinementTransferDirection,
    apply_h1_p3_state_transfer_with_refinement_history,
};
use sonium_fem::spaces::H1Space;
use sonium_qa::QaResult;
use std::collections::{BTreeMap, BTreeSet};

const CASE_ID: &str = "meshlang.fem-h1-p3-hexa-adaptive-reference.v1";

const P3_HEX_LATTICE: [[usize; 3]; 64] = [
    [0, 0, 0],
    [3, 0, 0],
    [3, 3, 0],
    [0, 3, 0],
    [0, 0, 3],
    [3, 0, 3],
    [3, 3, 3],
    [0, 3, 3],
    [1, 0, 0],
    [2, 0, 0],
    [3, 1, 0],
    [3, 2, 0],
    [2, 3, 0],
    [1, 3, 0],
    [0, 2, 0],
    [0, 1, 0],
    [1, 0, 3],
    [2, 0, 3],
    [3, 1, 3],
    [3, 2, 3],
    [2, 3, 3],
    [1, 3, 3],
    [0, 2, 3],
    [0, 1, 3],
    [0, 0, 1],
    [0, 0, 2],
    [3, 0, 1],
    [3, 0, 2],
    [3, 3, 1],
    [3, 3, 2],
    [0, 3, 1],
    [0, 3, 2],
    [1, 1, 0],
    [2, 1, 0],
    [2, 2, 0],
    [1, 2, 0],
    [1, 1, 3],
    [2, 1, 3],
    [2, 2, 3],
    [1, 2, 3],
    [1, 0, 1],
    [2, 0, 1],
    [2, 0, 2],
    [1, 0, 2],
    [1, 3, 1],
    [2, 3, 1],
    [2, 3, 2],
    [1, 3, 2],
    [0, 1, 1],
    [0, 2, 1],
    [0, 2, 2],
    [0, 1, 2],
    [3, 1, 1],
    [3, 2, 1],
    [3, 2, 2],
    [3, 1, 2],
    [1, 1, 1],
    [2, 1, 1],
    [2, 2, 1],
    [1, 2, 1],
    [1, 1, 2],
    [2, 1, 2],
    [2, 2, 2],
    [1, 2, 2],
];

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Point, index as u64)
}

fn p3_hexahedron() -> Mesh {
    let mut mesh = Mesh::new(3);
    for [i, j, k] in P3_HEX_LATTICE {
        mesh.add_node(Point::new_3d(
            i as f64 / 3.0,
            j as f64 / 3.0,
            k as f64 / 3.0,
        ));
    }
    mesh.add_element(ElementType::Hexahedron, (0..64).collect());
    mesh
}

fn cubic_value(point: Point) -> f64 {
    0.4 + 1.1 * point.x - 0.7 * point.y + 0.8 * point.x * point.x - 0.3 * point.x * point.y
        + 0.5 * point.y * point.y
        + 0.2 * point.x * point.x * point.x
        - 0.15 * point.x * point.y * point.y
}

fn all_entities(mesh: &Mesh) -> BTreeMap<EntityId, usize> {
    (0..mesh.num_nodes())
        .map(|index| (entity(index), index))
        .collect()
}

fn active_entities(mesh: &Mesh) -> BTreeMap<EntityId, usize> {
    mesh.elements
        .iter()
        .flat_map(|element| element.nodes.iter().copied())
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(|index| (entity(index), index))
        .collect()
}

fn interpolation_plan(entities: &BTreeMap<EntityId, usize>) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::Interpolate,
        entries: entities
            .keys()
            .copied()
            .map(|entity| IrStateTransferEntry {
                entity,
                action: IrStateTransferAction::Interpolate,
            })
            .collect(),
    }
}

fn run() -> Result<QaResult, String> {
    let coarse_mesh = p3_hexahedron();
    let mut fine_mesh = coarse_mesh.clone();
    let (refinement, history) =
        refine_elements_high_order_with_history_checked(&mut fine_mesh, &[0])
            .map_err(|error| error.to_string())?;
    if refinement.new_elements.len() != 8 || refinement.new_nodes.len() != 279 {
        return Err(format!(
            "unexpected P3 hexa refinement counts: children={}, new_nodes={}",
            refinement.new_elements.len(),
            refinement.new_nodes.len()
        ));
    }

    let source_entities = all_entities(&coarse_mesh);
    let fine_entities = all_entities(&fine_mesh);
    let source_values = coarse_mesh
        .nodes
        .iter()
        .copied()
        .map(cubic_value)
        .collect::<Vec<_>>();
    let fine_values = apply_h1_p3_state_transfer_with_refinement_history::<&'static str, _>(
        &interpolation_plan(&fine_entities),
        &coarse_mesh,
        &fine_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
        &source_values,
        &history,
        RefinementTransferDirection::Refine,
        &source_entities,
        &fine_entities,
        |_, _, _| Ok(0.0),
    )
    .map_err(|error| error.to_string())?;
    let refine_error = fine_mesh
        .nodes
        .iter()
        .enumerate()
        .map(|(node, point)| (fine_values[&entity(node)] - cubic_value(*point)).abs())
        .fold(0.0_f64, f64::max);

    let refined_mesh = fine_mesh.clone();
    let parent_id = history.parents[0].element.id;
    coarsen_elements_checked(&mut fine_mesh, &history, &[parent_id])
        .map_err(|error| error.to_string())?;
    let restored_entities = active_entities(&fine_mesh);
    let restored = apply_h1_p3_state_transfer_with_refinement_history::<&'static str, _>(
        &interpolation_plan(&restored_entities),
        &refined_mesh,
        &fine_mesh,
        &H1Space::new(PolynomialDegree::P3),
        &H1Space::new(PolynomialDegree::P3),
        &(0..refined_mesh.num_nodes())
            .map(|node| fine_values[&entity(node)])
            .collect::<Vec<_>>(),
        &history,
        RefinementTransferDirection::Coarsen,
        &all_entities(&refined_mesh),
        &restored_entities,
        |_, _, _| Ok(0.0),
    )
    .map_err(|error| error.to_string())?;
    let coarsen_error = restored_entities
        .values()
        .map(|&node| (restored[&entity(node)] - cubic_value(fine_mesh.nodes[node])).abs())
        .fold(0.0_f64, f64::max);

    if !refine_error.is_finite()
        || !coarsen_error.is_finite()
        || refine_error >= 1.0e-9
        || coarsen_error >= 1.0e-9
    {
        return Err(format!(
            "P3 hexa transfer error too large: refine={refine_error:e}, coarsen={coarsen_error:e}"
        ));
    }
    Ok(QaResult::passed(CASE_ID)
        .metric("children", refinement.new_elements.len() as f64)
        .metric("new_nodes", refinement.new_nodes.len() as f64)
        .metric("refine_error", refine_error)
        .metric("coarsen_error", coarsen_error))
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
