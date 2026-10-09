use meshlang_ast::{RemeshResponsibility, StateTransferResponsibility, VersionedCanonicalIr};
use meshlang_compiler::compile_to_geo_full;

fn ir(source: &str, file: &str) -> meshlang_ast::CanonicalGeometryIr {
    let result = compile_to_geo_full(source, file).expect("MeshLang source must compile");
    serde_json::from_str::<VersionedCanonicalIr>(&result.canonical_ir_json)
        .expect("versioned canonical IR")
        .ir
}

#[test]
fn numeric_design_perturbation_preserves_entity_topology() {
    let first = ir(
        "param width = 1.0 geometry model { let body = Box(width, 1.0, 1.0) }",
        "topology-diff-stable-a.mlg",
    );
    let second = ir(
        "param width = 1.25 geometry model { let body = Box(width, 1.0, 1.0) }",
        "topology-diff-stable-b.mlg",
    );
    let diff = first.topology_diff(&second);
    assert!(diff.added_entities.is_empty());
    assert!(diff.removed_entities.is_empty());
    assert!(diff.modified_entities.is_empty());
    assert!(!diff.topology_changed());
    assert!(diff.values_changed());
    assert_ne!(
        first.structural_hash().unwrap(),
        second.structural_hash().unwrap()
    );
}

#[test]
fn compile_time_branch_change_reports_added_and_removed_entities() {
    let first = ir(
        "geometry model { if true { let shape = Box(1.0, 1.0, 1.0) } else { let shape = Sphere(r: 0.5) } }",
        "topology-diff-branch-box.mlg",
    );
    let second = ir(
        "geometry model { if false { let shape = Box(1.0, 1.0, 1.0) } else { let shape = Sphere(r: 0.5) } }",
        "topology-diff-branch-sphere.mlg",
    );
    let diff = first.topology_diff(&second);
    assert!(diff.topology_changed());
    assert!(
        !diff.modified_entities.is_empty()
            || !diff.added_entities.is_empty()
            || !diff.removed_entities.is_empty()
    );
}

#[test]
fn dynamic_branch_membership_is_part_of_topology_contract() {
    let first = ir(
        "param enabled = true geometry model { if enabled { let primary = Box(1.0, 1.0, 1.0) } else { let fallback = Sphere(r: 0.5) } }",
        "topology-diff-dynamic-a.mlg",
    );
    let second = ir(
        "param enabled = false geometry model { if enabled { let primary = Box(1.0, 1.0, 1.0) } else { let fallback = Sphere(r: 0.5) } }",
        "topology-diff-dynamic-b.mlg",
    );
    let diff = first.topology_diff(&second);
    assert_eq!(first.controls.len(), second.controls.len());
    assert!(diff.values_changed() || diff.topology_changed());
}

#[test]
fn marker_membership_changes_are_reported_without_false_topology_changes() {
    let first = ir(
        "geometry model { let left = Box(1.0, 1.0, 1.0) let right = Box(1.0, 1.0, 1.0) |> translate(2.0, 0.0, 0.0) } physical_region volume selected { entities: model.left }",
        "topology-diff-marker-left.mlg",
    );
    let second = ir(
        "geometry model { let left = Box(1.0, 1.0, 1.0) let right = Box(1.0, 1.0, 1.0) |> translate(2.0, 0.0, 0.0) } physical_region volume selected { entities: model.right }",
        "topology-diff-marker-right.mlg",
    );
    let diff = first.topology_diff(&second);
    assert!(!diff.topology_changed());
    assert!(diff.markers_changed());
    assert_eq!(diff.changed_physical_groups.len(), 1);
}

#[test]
fn design_change_contract_assigns_remesh_and_state_transfer() {
    let source = |initial: f64| {
        format!(
            "design width {{ type: continuous initial: {initial} lower: 0.5 upper: 2.0 }} geometry model {{ let body = Box(width, 1.0, 1.0) }}"
        )
    };
    let first = ir(&source(1.0), "change-contract-design-a.mlg");
    let second = ir(&source(1.25), "change-contract-design-b.mlg");
    let contract = first.change_contract(&second);
    assert!(contract.diff.values_changed());
    assert_eq!(contract.remesh, RemeshResponsibility::Global);
    assert_eq!(
        contract.state_transfer,
        StateTransferResponsibility::Interpolate
    );
    assert!(contract.validate().is_ok());
}

#[test]
fn topology_change_without_sensitivity_requires_application_policy() {
    let first = ir(
        "geometry model { if true { let shape = Box(1.0, 1.0, 1.0) } }",
        "change-contract-topology-a.mlg",
    );
    let second = ir(
        "geometry model { if false { let shape = Box(1.0, 1.0, 1.0) } else { let shape = Sphere(r: 0.5) } }",
        "change-contract-topology-b.mlg",
    );
    let contract = first.change_contract(&second);
    assert!(contract.diff.topology_changed());
    assert_eq!(contract.remesh, RemeshResponsibility::TopologyDependent);
    assert_eq!(
        contract.state_transfer,
        StateTransferResponsibility::ApplicationDefined
    );
    assert!(contract.validate().is_ok());
}

#[test]
fn marker_only_change_does_not_force_remesh_or_transfer() {
    let first = ir(
        "geometry model { let left = Box(1.0, 1.0, 1.0) let right = Box(1.0, 1.0, 1.0) |> translate(2.0, 0.0, 0.0) } physical_region volume selected { entities: model.left }",
        "change-contract-marker-a.mlg",
    );
    let second = ir(
        "geometry model { let left = Box(1.0, 1.0, 1.0) let right = Box(1.0, 1.0, 1.0) |> translate(2.0, 0.0, 0.0) } physical_region volume selected { entities: model.right }",
        "change-contract-marker-b.mlg",
    );
    let contract = first.change_contract(&second);
    assert!(!contract.diff.topology_changed());
    assert!(contract.diff.markers_changed());
    assert_eq!(contract.remesh, RemeshResponsibility::NotRequired);
    assert_eq!(contract.state_transfer, StateTransferResponsibility::None);
    assert!(contract.validate().is_ok());
}

#[test]
fn change_contract_validation_accepts_design_topology_and_marker_packets() {
    let design_a = ir(
        "design width { type: continuous initial: 1.0 lower: 0.5 upper: 2.0 } geometry model { let body = Box(width, 1.0, 1.0) }",
        "change-contract-validate-design-a.mlg",
    );
    let design_b = ir(
        "design width { type: continuous initial: 1.25 lower: 0.5 upper: 2.0 } geometry model { let body = Box(width, 1.0, 1.0) }",
        "change-contract-validate-design-b.mlg",
    );
    assert!(design_a.change_contract(&design_b).validate().is_ok());

    let topology_a = ir(
        "geometry model { if true { let shape = Box(1.0, 1.0, 1.0) } else { let shape = Sphere(r: 0.5) } }",
        "change-contract-validate-topology-a.mlg",
    );
    let topology_b = ir(
        "geometry model { if false { let shape = Box(1.0, 1.0, 1.0) } else { let shape = Sphere(r: 0.5) } }",
        "change-contract-validate-topology-b.mlg",
    );
    assert!(topology_a.change_contract(&topology_b).validate().is_ok());

    let marker_a = ir(
        "geometry model { let body = Box(1.0, 1.0, 1.0) } physical_region volume left { entities: model.body }",
        "change-contract-validate-marker-a.mlg",
    );
    let marker_b = ir(
        "geometry model { let body = Box(1.0, 1.0, 1.0) } physical_region volume right { entities: model.body }",
        "change-contract-validate-marker-b.mlg",
    );
    assert!(marker_a.change_contract(&marker_b).validate().is_ok());
}

#[test]
fn change_contract_validation_rejects_downgrade_and_unsorted_hand_offs() {
    let first = ir(
        "geometry model { if true { let shape = Box(1.0, 1.0, 1.0) } }",
        "change-contract-invalid-a.mlg",
    );
    let second = ir(
        "geometry model { if false { let shape = Box(1.0, 1.0, 1.0) } else { let shape = Sphere(r: 0.5) } }",
        "change-contract-invalid-b.mlg",
    );
    let mut topology = first.change_contract(&second);
    topology.remesh = RemeshResponsibility::Global;
    assert_eq!(
        topology.validate(),
        Err("topology changes require topology-dependent remesh")
    );

    let mut unsorted = first.change_contract(&first);
    unsorted.affected_variables = vec![meshlang_ast::SymbolId(2), meshlang_ast::SymbolId(1)];
    assert_eq!(
        unsorted.validate(),
        Err("change-contract variables must be strictly sorted")
    );
}
