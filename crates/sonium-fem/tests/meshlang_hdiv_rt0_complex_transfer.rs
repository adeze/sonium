#![cfg(feature = "meshlang")]

use meshlang_ast::{
    EntityDimension, EntityId, IrStateTransferAction, IrStateTransferEntry, IrStateTransferPlan,
    StateTransferResponsibility,
};
use num_complex::Complex64;
use sonium_fem::mesh::{
    apply_complex_hdiv_rt0_state_transfer, transfer_hdiv_rt0_state_checked, unit_square_triangles,
};
use sonium_fem::spaces::{FunctionSpace, HdivSpace};
use std::collections::BTreeMap;

fn entity(index: usize) -> EntityId {
    EntityId::new(EntityDimension::Curve, index as u64)
}

fn entities(space: &HdivSpace, mesh: &sonium_fem::mesh::Mesh) -> BTreeMap<EntityId, usize> {
    (0..space.num_global_dofs(mesh))
        .map(|index| (entity(index), index))
        .collect()
}

#[test]
fn complex_rt0_transfer_matches_independent_component_kernels() {
    let source_mesh = unit_square_triangles(1);
    let target_mesh = unit_square_triangles(2);
    let source_space = HdivSpace::new();
    let target_space = HdivSpace::new();
    let source_dofs = source_space.num_global_dofs(&source_mesh);
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
    .expect("real RT0 transfer should succeed");
    let expected_imaginary = transfer_hdiv_rt0_state_checked(
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &imaginary_source,
    )
    .expect("imaginary RT0 transfer should succeed");

    let target_entities = entities(&target_space, &target_mesh);
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
    let transferred = apply_complex_hdiv_rt0_state_transfer::<&'static str, _>(
        &plan,
        &source_mesh,
        &target_mesh,
        &source_space,
        &target_space,
        &source_values,
        &BTreeMap::new(),
        &target_entities,
        |_, _, _| Err("callback must not run for interpolation"),
    )
    .expect("complex RT0 MeshLang bridge should succeed");

    for (index, (&entity, _)) in target_entities.iter().enumerate() {
        let value = transferred[&entity];
        assert!((value.re - expected_real[index]).abs() < 1.0e-10);
        assert!((value.im - expected_imaginary[index]).abs() < 1.0e-10);
    }
}

#[test]
fn complex_rt0_bridge_preserves_callback_and_rejects_non_finite_source() {
    let mesh = unit_square_triangles(1);
    let space = HdivSpace::new();
    let source_dofs = space.num_global_dofs(&mesh);
    let source_entities = BTreeMap::from([(entity(0), 0usize)]);
    let target_entities = BTreeMap::new();
    let source_values = vec![Complex64::new(1.0, -2.0); source_dofs];
    let plan = IrStateTransferPlan {
        responsibility: StateTransferResponsibility::ApplicationDefined,
        entries: vec![
            IrStateTransferEntry {
                entity: entity(0),
                action: IrStateTransferAction::Preserve,
            },
            IrStateTransferEntry {
                entity: entity(1),
                action: IrStateTransferAction::Initialize,
            },
        ],
    };
    let transferred = apply_complex_hdiv_rt0_state_transfer::<&'static str, _>(
        &plan,
        &mesh,
        &mesh,
        &space,
        &space,
        &source_values,
        &source_entities,
        &target_entities,
        |callback_entity, action, source| {
            assert_eq!(action, IrStateTransferAction::Initialize);
            assert_eq!(callback_entity, entity(1));
            assert_eq!(source, None);
            Ok(Complex64::new(3.0, 4.0))
        },
    )
    .expect("preserve and initialize should succeed");
    assert_eq!(transferred[&entity(0)], source_values[0]);
    assert_eq!(transferred[&entity(1)], Complex64::new(3.0, 4.0));

    let mut non_finite = source_values;
    non_finite[0].im = f64::NAN;
    let error = apply_complex_hdiv_rt0_state_transfer::<&'static str, _>(
        &IrStateTransferPlan {
            responsibility: StateTransferResponsibility::None,
            entries: vec![IrStateTransferEntry {
                entity: entity(0),
                action: IrStateTransferAction::Preserve,
            }],
        },
        &mesh,
        &mesh,
        &space,
        &space,
        &non_finite,
        &source_entities,
        &target_entities,
        |_, _, _| Ok(Complex64::new(0.0, 0.0)),
    )
    .expect_err("non-finite complex source must be rejected");
    assert!(matches!(
        error,
        sonium_fem::mesh::MeshlangTransferError::NonFiniteSource { .. }
    ));
}
