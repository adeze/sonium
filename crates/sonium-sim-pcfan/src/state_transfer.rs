//! Checked state transfer across topology-changing resolved-CFD meshes.
//!
//! The production resolved PC-fan solve currently uses topology-preserving
//! ALE motion.  This module provides the missing boundary for a future
//! remeshing path: a complete source state is transferred to a new P1 or
//! transient P2/P1 tetrahedral mesh, or the operation fails without returning
//! a partial state. The BDF mean-flow levels plus checked P1 k--omega history
//! can be transferred, but transported turbulence equations,
//! topology-dependent operators, and adjoint history artifacts are
//! intentionally reassembled rather than copied.
//!
//! The numeric transfer layer lives in [`sonium_fem::mesh::state_transfer`];
//! this module keeps the PC-fan report shapes, provenance strings, and public
//! names as thin wrappers and aliases so existing callers are unaffected.

use sonium_fem::mesh::Mesh;
use sonium_fem::physics::fluid::stokes::time_stepping::BdfState;

pub use sonium_fem::mesh::state_transfer::{
    CfdStateTransfer as PcFanCfdStateTransfer, CfdStateTransferError as PcFanCfdStateTransferError,
    CfdStateTransferReport as PcFanCfdStateTransferReport,
    KOmegaBdfState as PcFanCfdKOmegaBdfState,
    KOmegaBdfStateAdjointTransfer as PcFanCfdKOmegaBdfStateAdjointTransfer,
    KOmegaBdfStateAdjointTransferChain as PcFanCfdKOmegaBdfStateAdjointTransferChain,
    KOmegaBdfStateAdjointTransferChainReport as PcFanCfdKOmegaBdfStateAdjointTransferChainReport,
    KOmegaBdfStateTransfer as PcFanCfdKOmegaBdfStateTransfer,
    KOmegaBdfStateTransferChain as PcFanCfdKOmegaBdfStateTransferChain,
    KOmegaBdfStateTransferChainReport as PcFanCfdKOmegaBdfStateTransferChainReport,
    KOmegaBdfStateTransferReport as PcFanCfdKOmegaBdfStateTransferReport,
    KOmegaStateTransferError as PcFanCfdKOmegaStateTransferError,
    P2P1StateAdjointTransfer as PcFanCfdP2P1StateAdjointTransfer,
    P2P1StateAdjointTransferChain as PcFanCfdP2P1StateAdjointTransferChain,
    P2P1StateAdjointTransferChainReport as PcFanCfdP2P1StateAdjointTransferChainReport,
    P2P1StateTransfer as PcFanCfdP2P1StateTransfer,
    P2P1StateTransferChain as PcFanCfdP2P1StateTransferChain,
    P2P1StateTransferChainReport as PcFanCfdP2P1StateTransferChainReport,
    P2P1StateTransferReport as PcFanCfdP2P1StateTransferReport,
    transfer_komega_bdf_state as transfer_pc_fan_cfd_komega_bdf_state,
    transfer_komega_bdf_state_adjoint as transfer_pc_fan_cfd_komega_bdf_state_adjoint,
    transfer_p1_cfd_state as transfer_pc_fan_cfd_state,
    transfer_p2p1_bdf_state as transfer_pc_fan_cfd_p2p1_bdf_state,
    transfer_p2p1_bdf_state_adjoint as transfer_pc_fan_cfd_p2p1_bdf_state_adjoint,
    transfer_p2p1_state as transfer_pc_fan_cfd_p2p1_state,
    transfer_p2p1_state_adjoint as transfer_pc_fan_cfd_p2p1_state_adjoint,
};
use sonium_fem::mesh::state_transfer::{
    transfer_komega_bdf_state_adjoint_chain, transfer_komega_bdf_state_chain,
    transfer_p2p1_bdf_state_adjoint_chain, transfer_p2p1_bdf_state_chain,
};

/// Transfer a BDF state through an ordered topology-changing mesh chain.
///
/// Thin wrapper stamping the PC-fan handoff provenance over the reusable
/// [`sonium_fem::mesh::state_transfer`] chain.
pub fn transfer_pc_fan_cfd_p2p1_bdf_state_chain(
    mesh_chain: &[&Mesh],
    source_state: &BdfState,
) -> Result<PcFanCfdP2P1StateTransferChain, PcFanCfdStateTransferError> {
    transfer_p2p1_bdf_state_chain(
        mesh_chain,
        source_state,
        "pcfan:topology-changing-p2p1-bdf-state-transfer:v1",
    )
}

/// Apply BDF state-adjoint transfer through a mesh chain in reverse order.
///
/// Thin wrapper stamping the PC-fan transpose provenance over the reusable
/// [`sonium_fem::mesh::state_transfer`] chain.
pub fn transfer_pc_fan_cfd_p2p1_bdf_state_adjoint_chain(
    mesh_chain: &[&Mesh],
    terminal_adjoint: &BdfState,
) -> Result<PcFanCfdP2P1StateAdjointTransferChain, PcFanCfdStateTransferError> {
    transfer_p2p1_bdf_state_adjoint_chain(
        mesh_chain,
        terminal_adjoint,
        "pcfan:topology-changing-p2p1-bdf-state-adjoint-transfer:v1",
    )
}

/// Transfer P1 k--omega history through an ordered topology-changing mesh
/// chain. Thin wrapper stamping the PC-fan history provenance.
pub fn transfer_pc_fan_cfd_komega_bdf_state_chain(
    mesh_chain: &[&Mesh],
    source_state: &PcFanCfdKOmegaBdfState,
) -> Result<PcFanCfdKOmegaBdfStateTransferChain, PcFanCfdKOmegaStateTransferError> {
    transfer_komega_bdf_state_chain(
        mesh_chain,
        source_state,
        "pcfan:topology-changing-komega-bdf-history-transfer:v1",
    )
}

/// Apply k--omega BDF state-adjoint transfer through a mesh chain in reverse.
/// Thin wrapper stamping the PC-fan transpose provenance.
pub fn transfer_pc_fan_cfd_komega_bdf_state_adjoint_chain(
    mesh_chain: &[&Mesh],
    terminal_adjoint: &PcFanCfdKOmegaBdfState,
) -> Result<PcFanCfdKOmegaBdfStateAdjointTransferChain, PcFanCfdKOmegaStateTransferError> {
    transfer_komega_bdf_state_adjoint_chain(
        mesh_chain,
        terminal_adjoint,
        "pcfan:topology-changing-komega-bdf-state-adjoint-transfer:v1",
    )
}
#[cfg(test)]
mod tests {
    use super::*;
    use sonium_fem::basis::PolynomialDegree;
    use sonium_fem::mesh::InterpolationError;
    use sonium_fem::mesh::generators::box_mesh_tetrahedra;
    use sonium_fem::mesh::unit_cube_tetrahedra;
    use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};

    fn source_state(mesh: &Mesh) -> (Vec<[f64; 3]>, Vec<f64>) {
        let velocity = mesh
            .nodes
            .iter()
            .map(|point| {
                [
                    point.x + 2.0 * point.y,
                    -point.z + 0.5,
                    point.x - point.y + point.z,
                ]
            })
            .collect();
        let pressure = mesh
            .nodes
            .iter()
            .map(|point| 2.0 * point.x - point.y + 3.0 * point.z)
            .collect();
        (velocity, pressure)
    }

    fn affine_p2_p1_state(mesh: &Mesh) -> (Vec<f64>, Vec<f64>) {
        let velocity_space = H1Space::new(PolynomialDegree::P2);
        let velocity_coordinates = velocity_space
            .dof_coordinates(mesh)
            .expect("P2 tetrahedral H1 DOFs have coordinates");
        let velocity = velocity_coordinates
            .iter()
            .flat_map(|point| {
                [
                    point[0] + 2.0 * point[1],
                    -point[2] + 0.5,
                    point[0] - point[1] + point[2],
                ]
            })
            .collect();

        let pressure_space = H1Space::new(PolynomialDegree::P1);
        let pressure_coordinates = pressure_space
            .dof_coordinates(mesh)
            .expect("P1 tetrahedral H1 DOFs have coordinates");
        let pressure = pressure_coordinates
            .iter()
            .map(|point| 2.0 * point[0] - point[1] + 3.0 * point[2])
            .collect();
        (velocity, pressure)
    }

    #[test]
    fn constant_state_is_preserved_across_topology_change() {
        let source = unit_cube_tetrahedra(1);
        let target = unit_cube_tetrahedra(2);
        let velocity = vec![[1.25, -0.5, 3.0]; source.num_nodes()];
        let pressure = vec![101_325.0; source.num_nodes()];

        let transferred =
            transfer_pc_fan_cfd_state(&source, &target, &velocity, &pressure).unwrap();

        for value in &transferred.velocity_m_s {
            assert!((value[0] - 1.25).abs() < 1.0e-12);
            assert!((value[1] + 0.5).abs() < 1.0e-12);
            assert!((value[2] - 3.0).abs() < 1.0e-12);
        }
        assert!(
            transferred
                .pressure_pa
                .iter()
                .all(|value| (*value - 101_325.0).abs() < 1.0e-9)
        );
        assert!(transferred.report.max_velocity_correction_m_s < 1.0e-12);
        assert!(transferred.report.max_pressure_correction_pa < 1.0e-9);
    }

    #[test]
    fn consistent_negative_fem_orientation_is_supported() {
        let source = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1);
        let target = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 2, 2);

        let velocity = vec![[2.0, -1.0, 0.5]; source.num_nodes()];
        let pressure = vec![42.0; source.num_nodes()];
        let transferred = transfer_pc_fan_cfd_state(&source, &target, &velocity, &pressure)
            .expect("a consistent FEM orientation convention is valid");

        assert!(transferred.velocity_m_s.iter().all(|value| {
            value
                .iter()
                .zip([2.0, -1.0, 0.5])
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-12)
        }));
        assert!(
            transferred
                .pressure_pa
                .iter()
                .all(|value| (*value - 42.0).abs() < 1.0e-12)
        );
    }

    #[test]
    fn affine_state_remains_finite_and_conservative() {
        let source = unit_cube_tetrahedra(1);
        let target = unit_cube_tetrahedra(2);
        let (velocity, pressure) = source_state(&source);

        let transferred =
            transfer_pc_fan_cfd_state(&source, &target, &velocity, &pressure).unwrap();

        for (point, value) in target.nodes.iter().zip(&transferred.velocity_m_s) {
            let expected = [
                point.x + 2.0 * point.y,
                -point.z + 0.5,
                point.x - point.y + point.z,
            ];
            for component in 0..3 {
                assert!((value[component] - expected[component]).abs() < 1.0e-10);
            }
        }
        for (point, value) in target.nodes.iter().zip(&transferred.pressure_pa) {
            let expected = 2.0 * point.x - point.y + 3.0 * point.z;
            assert!((value - expected).abs() < 1.0e-10);
        }
        assert!(transferred.report.source_volume_m3 > 0.0);
        assert!(
            (transferred.report.source_volume_m3 - transferred.report.target_volume_m3).abs()
                < 1.0e-12
        );
        assert!(transferred.report.max_velocity_correction_m_s < 1.0e-10);
        assert!(transferred.report.max_pressure_correction_pa < 1.0e-10);
    }

    #[test]
    fn vector_component_ordering_is_preserved() {
        let source = unit_cube_tetrahedra(1);
        let target = unit_cube_tetrahedra(2);
        let velocity = vec![[10.0, 20.0, 30.0]; source.num_nodes()];
        let pressure = vec![0.0; source.num_nodes()];

        let transferred =
            transfer_pc_fan_cfd_state(&source, &target, &velocity, &pressure).unwrap();

        assert!(
            transferred
                .velocity_m_s
                .iter()
                .all(|value| *value == [10.0, 20.0, 30.0])
        );
    }

    #[test]
    fn malformed_source_lengths_are_rejected() {
        let source = unit_cube_tetrahedra(1);
        let target = unit_cube_tetrahedra(2);
        let pressure = vec![0.0; source.num_nodes()];

        assert!(matches!(
            transfer_pc_fan_cfd_state(&source, &target, &[], &pressure),
            Err(PcFanCfdStateTransferError::SourceVelocityCount { .. })
        ));
        let velocity = vec![[0.0; 3]; source.num_nodes()];
        assert!(matches!(
            transfer_pc_fan_cfd_state(&source, &target, &velocity, &[]),
            Err(PcFanCfdStateTransferError::SourcePressureCount { .. })
        ));
    }

    #[test]
    fn non_finite_source_is_rejected_before_interpolation() {
        let source = unit_cube_tetrahedra(1);
        let target = unit_cube_tetrahedra(2);
        let mut velocity = vec![[0.0; 3]; source.num_nodes()];
        velocity[0][1] = f64::NAN;
        let pressure = vec![0.0; source.num_nodes()];
        assert!(matches!(
            transfer_pc_fan_cfd_state(&source, &target, &velocity, &pressure),
            Err(PcFanCfdStateTransferError::NonFiniteSourceVelocity {
                node: 0,
                component: 1
            })
        ));

        let velocity = vec![[0.0; 3]; source.num_nodes()];
        let mut pressure = vec![0.0; source.num_nodes()];
        pressure[2] = f64::INFINITY;
        assert!(matches!(
            transfer_pc_fan_cfd_state(&source, &target, &velocity, &pressure),
            Err(PcFanCfdStateTransferError::NonFiniteSourcePressure { node: 2 })
        ));
    }

    #[test]
    fn target_outside_source_domain_fails_closed() {
        let source = unit_cube_tetrahedra(1);
        let mut target = unit_cube_tetrahedra(2);
        target.nodes[0].x = -0.25;
        target.invalidate_caches();
        let velocity = vec![[0.0; 3]; source.num_nodes()];
        let pressure = vec![0.0; source.num_nodes()];

        let result = transfer_pc_fan_cfd_state(&source, &target, &velocity, &pressure);
        assert!(matches!(
            result,
            Err(PcFanCfdStateTransferError::Interpolation(
                InterpolationError::TargetOutside { .. }
            ))
        ));
        assert_eq!(target.nodes[0].x, -0.25);
    }

    #[test]
    fn mismatched_domain_volume_fails_closed() {
        let source = unit_cube_tetrahedra(1);
        let mut target = unit_cube_tetrahedra(2);
        for point in &mut target.nodes {
            point.x = 0.125 + 0.75 * point.x;
            point.y = 0.125 + 0.75 * point.y;
            point.z = 0.125 + 0.75 * point.z;
        }
        target.invalidate_caches();
        let velocity = vec![[0.0; 3]; source.num_nodes()];
        let pressure = vec![0.0; source.num_nodes()];

        assert!(matches!(
            transfer_pc_fan_cfd_state(&source, &target, &velocity, &pressure),
            Err(PcFanCfdStateTransferError::Interpolation(
                InterpolationError::ConservationMismatch { .. }
            ))
        ));
    }

    #[test]
    fn inverted_mesh_is_rejected() {
        let source = unit_cube_tetrahedra(1);
        let mut target = unit_cube_tetrahedra(1);
        target.elements[0].nodes.swap(0, 1);
        target.invalidate_caches();
        let velocity = vec![[0.0; 3]; source.num_nodes()];
        let pressure = vec![0.0; source.num_nodes()];

        assert!(matches!(
            transfer_pc_fan_cfd_state(&source, &target, &velocity, &pressure),
            Err(PcFanCfdStateTransferError::InvalidMesh { role: "target", .. })
        ));
    }

    #[test]
    fn p2_p1_affine_state_is_exact_across_topology_change() {
        let source = unit_cube_tetrahedra(1);
        let target = unit_cube_tetrahedra(2);
        let (velocity, pressure) = affine_p2_p1_state(&source);

        let transferred =
            transfer_pc_fan_cfd_p2p1_state(&source, &target, &velocity, &pressure).unwrap();
        let (expected_velocity, expected_pressure) = affine_p2_p1_state(&target);

        assert_eq!(transferred.velocity_dofs.len(), expected_velocity.len());
        assert_eq!(transferred.pressure_dofs.len(), expected_pressure.len());
        for (actual, expected) in transferred.velocity_dofs.iter().zip(expected_velocity) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
        for (actual, expected) in transferred.pressure_dofs.iter().zip(expected_pressure) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
        assert_eq!(
            transferred.report.target_velocity_dof_count,
            transferred.velocity_dofs.len()
        );
        assert_eq!(
            transferred.report.target_pressure_dof_count,
            transferred.pressure_dofs.len()
        );
    }

    #[test]
    fn p2_p1_consistent_negative_fem_orientation_is_supported() {
        let source = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 1, 1, 1);
        let target = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 2, 2, 2);
        let (velocity, pressure) = affine_p2_p1_state(&source);

        let transferred = transfer_pc_fan_cfd_p2p1_state(&source, &target, &velocity, &pressure)
            .expect("a consistent negative FEM orientation is valid");
        let (expected_velocity, expected_pressure) = affine_p2_p1_state(&target);

        assert!(
            transferred
                .velocity_dofs
                .iter()
                .zip(expected_velocity)
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10)
        );
        assert!(
            transferred
                .pressure_dofs
                .iter()
                .zip(expected_pressure)
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-10)
        );
    }

    #[test]
    fn p2_p1_malformed_and_non_finite_state_is_rejected() {
        let source = unit_cube_tetrahedra(1);
        let target = unit_cube_tetrahedra(2);
        let (velocity, pressure) = affine_p2_p1_state(&source);

        assert!(matches!(
            transfer_pc_fan_cfd_p2p1_state(&source, &target, &[], &pressure),
            Err(PcFanCfdStateTransferError::SourceVelocityDofCount { .. })
        ));

        let mut non_finite_velocity = velocity.clone();
        non_finite_velocity[4] = f64::NAN;
        assert!(matches!(
            transfer_pc_fan_cfd_p2p1_state(&source, &target, &non_finite_velocity, &pressure),
            Err(PcFanCfdStateTransferError::NonFiniteSourceVelocityDof { dof: 4 })
        ));

        let mut non_finite_pressure = pressure;
        non_finite_pressure[2] = f64::INFINITY;
        assert!(matches!(
            transfer_pc_fan_cfd_p2p1_state(&source, &target, &velocity, &non_finite_pressure),
            Err(PcFanCfdStateTransferError::NonFiniteSourcePressureDof { dof: 2 })
        ));

        let target_velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let target_pressure_space = H1Space::new(PolynomialDegree::P1);
        let target_velocity = vec![0.0; target_velocity_space.num_global_dofs(&target)];
        let target_pressure = vec![0.0; target_pressure_space.num_global_dofs(&target)];
        assert!(matches!(
            transfer_pc_fan_cfd_p2p1_state_adjoint(
                &source,
                &target,
                &target_velocity[..target_velocity.len() - 1],
                &target_pressure,
            ),
            Err(PcFanCfdStateTransferError::TargetVelocityDofCount { .. })
        ));

        let mut non_finite_target_velocity = target_velocity;
        non_finite_target_velocity[4] = f64::NAN;
        assert!(matches!(
            transfer_pc_fan_cfd_p2p1_state_adjoint(
                &source,
                &target,
                &non_finite_target_velocity,
                &target_pressure,
            ),
            Err(PcFanCfdStateTransferError::NonFiniteTargetVelocityDof { dof: 4 })
        ));
    }

    #[test]
    fn p2_p1_mismatched_domain_volume_fails_closed() {
        let source = unit_cube_tetrahedra(1);
        let mut target = unit_cube_tetrahedra(2);
        for point in &mut target.nodes {
            point.x = 0.125 + 0.75 * point.x;
            point.y = 0.125 + 0.75 * point.y;
            point.z = 0.125 + 0.75 * point.z;
        }
        target.invalidate_caches();
        let (velocity, pressure) = affine_p2_p1_state(&source);

        assert!(matches!(
            transfer_pc_fan_cfd_p2p1_state(&source, &target, &velocity, &pressure),
            Err(PcFanCfdStateTransferError::InvalidMesh { role: "target", .. })
        ));
    }

    #[test]
    fn p2_p1_bdf_state_transfers_current_and_previous_velocity_levels() {
        let source = unit_cube_tetrahedra(1);
        let target = unit_cube_tetrahedra(2);
        let (velocity, pressure) = affine_p2_p1_state(&source);
        let velocity_prev = velocity
            .iter()
            .map(|value| 0.5 * value + 1.25)
            .collect::<Vec<_>>();
        let source_state = BdfState {
            velocity,
            velocity_prev,
            pressure,
            time: 0.0125,
            step: 7,
        };

        let transferred =
            transfer_pc_fan_cfd_p2p1_bdf_state(&source, &target, &source_state).unwrap();
        let (expected_velocity, expected_pressure) = affine_p2_p1_state(&target);

        assert_eq!(transferred.time, source_state.time);
        assert_eq!(transferred.step, source_state.step);
        for (actual, expected) in transferred.velocity.iter().zip(&expected_velocity) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
        for (actual, expected) in transferred
            .velocity_prev
            .iter()
            .zip(expected_velocity.iter().map(|value| 0.5 * value + 1.25))
        {
            assert!((actual - expected).abs() < 1.0e-10);
        }
        for (actual, expected) in transferred.pressure.iter().zip(expected_pressure) {
            assert!((actual - expected).abs() < 1.0e-10);
        }
    }

    #[test]
    fn p2_p1_bdf_state_rejects_invalid_history_and_time_metadata() {
        let source = unit_cube_tetrahedra(1);
        let target = unit_cube_tetrahedra(2);
        let (velocity, pressure) = affine_p2_p1_state(&source);
        let mut source_state = BdfState::from_steady(velocity, pressure);

        source_state.velocity_prev.pop();
        assert!(matches!(
            transfer_pc_fan_cfd_p2p1_bdf_state(&source, &target, &source_state),
            Err(PcFanCfdStateTransferError::SourceVelocityDofCount { .. })
        ));

        let (velocity, pressure) = affine_p2_p1_state(&source);
        let source_state = BdfState {
            velocity,
            velocity_prev: vec![],
            pressure,
            time: f64::NAN,
            step: 0,
        };
        assert!(matches!(
            transfer_pc_fan_cfd_p2p1_bdf_state(&source, &target, &source_state),
            Err(PcFanCfdStateTransferError::NonFiniteSourceTime)
        ));
    }

    #[test]
    fn p2_p1_bdf_state_chain_transfers_each_remesh_and_requires_reassembly() {
        let first = unit_cube_tetrahedra(1);
        let second = unit_cube_tetrahedra(2);
        let third = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3, 2, 2);
        let (velocity, pressure) = affine_p2_p1_state(&first);
        let source_state = BdfState {
            velocity: velocity.clone(),
            velocity_prev: velocity.iter().map(|value| 0.25 * value + 0.75).collect(),
            pressure,
            time: 0.125,
            step: 11,
        };

        let chain =
            transfer_pc_fan_cfd_p2p1_bdf_state_chain(&[&first, &second, &third], &source_state)
                .expect("both topology-changing transitions should transfer");

        assert_eq!(chain.states.len(), 2);
        assert_eq!(chain.report.stage_count, 2);
        assert_eq!(chain.report.stages.len(), 2);
        assert!(chain.report.requires_operator_reassembly);
        assert_eq!(
            chain.report.provenance,
            "pcfan:topology-changing-p2p1-bdf-state-transfer:v1"
        );
        for state in &chain.states {
            assert_eq!(state.time, source_state.time);
            assert_eq!(state.step, source_state.step);
            assert!(state.velocity.iter().all(|value| value.is_finite()));
            assert!(state.velocity_prev.iter().all(|value| value.is_finite()));
            assert!(state.pressure.iter().all(|value| value.is_finite()));
        }

        assert!(matches!(
            transfer_pc_fan_cfd_p2p1_bdf_state_chain(&[&first], &source_state),
            Err(PcFanCfdStateTransferError::InvalidMesh {
                role: "mesh chain",
                ..
            })
        ));
    }

    fn komega_state(mesh: &Mesh) -> PcFanCfdKOmegaBdfState {
        let coordinates = H1Space::new(PolynomialDegree::P1)
            .dof_coordinates(mesh)
            .expect("P1 tetrahedral H1 DOF coordinates");
        let k = coordinates
            .iter()
            .map(|point| 0.2 + point[0] + 2.0 * point[1] + point[2])
            .collect::<Vec<_>>();
        let k_prev = k.iter().map(|value| 0.5 * value + 0.1).collect();
        let omega = coordinates
            .iter()
            .map(|point| 1.0 + point[0] + point[1] + point[2])
            .collect::<Vec<_>>();
        let omega_prev = omega.iter().map(|value| 0.75 * value).collect();
        PcFanCfdKOmegaBdfState {
            k,
            k_prev,
            omega,
            omega_prev,
            time: 0.125,
            step: 11,
        }
    }

    #[test]
    fn komega_bdf_history_preserves_affine_fields_and_metadata() {
        let source = unit_cube_tetrahedra(1);
        let target = unit_cube_tetrahedra(2);
        let source_state = komega_state(&source);
        let transferred = transfer_pc_fan_cfd_komega_bdf_state(&source, &target, &source_state)
            .expect("admissible P1 k-omega history should transfer");
        let expected = komega_state(&target);

        for (actual, expected) in transferred
            .state
            .k
            .iter()
            .chain(&transferred.state.k_prev)
            .chain(&transferred.state.omega)
            .chain(&transferred.state.omega_prev)
            .zip(
                expected
                    .k
                    .iter()
                    .chain(&expected.k_prev)
                    .chain(&expected.omega)
                    .chain(&expected.omega_prev),
            )
        {
            assert!((actual - expected).abs() < 1.0e-12);
        }
        assert_eq!(transferred.state.time, expected.time);
        assert_eq!(transferred.state.step, expected.step);
        assert_eq!(
            transferred.report.source_scalar_dof_count,
            source_state.k.len()
        );
        assert_eq!(transferred.report.target_scalar_dof_count, expected.k.len());
        assert!(transferred.report.minimum_k > 0.0);
        assert!(transferred.report.minimum_omega > 0.0);
    }

    #[test]
    fn komega_bdf_history_rejects_bad_shape_non_finite_and_nonphysical_values() {
        let source = unit_cube_tetrahedra(1);
        let target = unit_cube_tetrahedra(2);
        let mut state = komega_state(&source);
        state.k.pop();
        assert!(matches!(
            transfer_pc_fan_cfd_komega_bdf_state(&source, &target, &state),
            Err(PcFanCfdKOmegaStateTransferError::FieldDofCount { field: "k", .. })
        ));

        let mut state = komega_state(&source);
        state.omega_prev[0] = f64::NAN;
        assert!(matches!(
            transfer_pc_fan_cfd_komega_bdf_state(&source, &target, &state),
            Err(PcFanCfdKOmegaStateTransferError::NonFiniteField { field: "omega", .. })
        ));

        let mut state = komega_state(&source);
        state.k[0] = -f64::EPSILON;
        assert!(matches!(
            transfer_pc_fan_cfd_komega_bdf_state(&source, &target, &state),
            Err(PcFanCfdKOmegaStateTransferError::InvalidFieldValue { field: "k", .. })
        ));

        let mut state = komega_state(&source);
        state.omega[0] = 0.0;
        assert!(matches!(
            transfer_pc_fan_cfd_komega_bdf_state(&source, &target, &state),
            Err(PcFanCfdKOmegaStateTransferError::InvalidFieldValue { field: "omega", .. })
        ));

        let mut state = komega_state(&source);
        state.time = f64::NAN;
        assert!(matches!(
            transfer_pc_fan_cfd_komega_bdf_state(&source, &target, &state),
            Err(PcFanCfdKOmegaStateTransferError::NonFiniteSourceTime)
        ));
    }

    #[test]
    fn komega_bdf_history_chain_transfers_each_stage_and_requires_reassembly() {
        let first = unit_cube_tetrahedra(1);
        let second = unit_cube_tetrahedra(2);
        let third = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3, 2, 2);
        let source_state = komega_state(&first);
        let chain =
            transfer_pc_fan_cfd_komega_bdf_state_chain(&[&first, &second, &third], &source_state)
                .expect("both topology-changing scalar-history stages should transfer");

        assert_eq!(chain.states.len(), 2);
        assert_eq!(chain.report.stage_count, 2);
        assert_eq!(chain.report.stages.len(), 2);
        assert!(chain.report.requires_operator_reassembly);
        assert_eq!(
            chain.report.provenance,
            "pcfan:topology-changing-komega-bdf-history-transfer:v1"
        );
        for state in &chain.states {
            assert_eq!(state.time, source_state.time);
            assert_eq!(state.step, source_state.step);
            assert!(
                state
                    .k
                    .iter()
                    .chain(&state.k_prev)
                    .all(|value| *value >= 0.0)
            );
            assert!(
                state
                    .omega
                    .iter()
                    .chain(&state.omega_prev)
                    .all(|value| *value > 0.0)
            );
        }
    }

    fn dot_values(left: &[f64], right: &[f64]) -> f64 {
        assert_eq!(left.len(), right.len());
        left.iter().zip(right).map(|(a, b)| a * b).sum()
    }

    #[test]
    fn p2_p1_state_adjoint_is_the_exact_single_stage_transpose() {
        let source = unit_cube_tetrahedra(1);
        let target = unit_cube_tetrahedra(2);
        let source_velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let source_pressure_space = H1Space::new(PolynomialDegree::P1);
        let target_velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let target_pressure_space = H1Space::new(PolynomialDegree::P1);

        let source_velocity: Vec<f64> = (0..source_velocity_space.num_global_dofs(&source))
            .map(|index| 0.25 + 0.13 * index as f64)
            .collect();
        let source_pressure: Vec<f64> = (0..source_pressure_space.num_global_dofs(&source))
            .map(|index| -0.7 + 0.19 * index as f64)
            .collect();
        let target_velocity_adjoint: Vec<f64> = (0..target_velocity_space.num_global_dofs(&target))
            .map(|index| -0.4 + 0.07 * index as f64)
            .collect();
        let target_pressure_adjoint: Vec<f64> = (0..target_pressure_space.num_global_dofs(&target))
            .map(|index| 0.6 - 0.11 * index as f64)
            .collect();

        let forward =
            transfer_pc_fan_cfd_p2p1_state(&source, &target, &source_velocity, &source_pressure)
                .expect("forward P2/P1 transfer should succeed");
        let adjoint = transfer_pc_fan_cfd_p2p1_state_adjoint(
            &source,
            &target,
            &target_velocity_adjoint,
            &target_pressure_adjoint,
        )
        .expect("P2/P1 transpose transfer should succeed");

        let target_pairing = dot_values(&forward.velocity_dofs, &target_velocity_adjoint)
            + dot_values(&forward.pressure_dofs, &target_pressure_adjoint);
        let source_pairing = dot_values(&source_velocity, &adjoint.velocity_dofs)
            + dot_values(&source_pressure, &adjoint.pressure_dofs);
        assert!((target_pairing - source_pairing).abs() < 1.0e-10);
        assert_eq!(
            adjoint.report.source_velocity_dof_count,
            source_velocity.len()
        );
        assert_eq!(
            adjoint.report.target_velocity_dof_count,
            target_velocity_adjoint.len()
        );
    }

    #[test]
    fn p2_p1_bdf_adjoint_chain_is_the_transpose_of_composed_transfer() {
        let first = unit_cube_tetrahedra(1);
        let second = unit_cube_tetrahedra(2);
        let third = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3, 2, 2);
        let first_velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
        let first_pressure_space = H1Space::new(PolynomialDegree::P1);
        let source_state = BdfState {
            velocity: (0..first_velocity_space.num_global_dofs(&first))
                .map(|index| 0.2 + 0.03 * index as f64)
                .collect(),
            velocity_prev: (0..first_velocity_space.num_global_dofs(&first))
                .map(|index| -0.1 + 0.05 * index as f64)
                .collect(),
            pressure: (0..first_pressure_space.num_global_dofs(&first))
                .map(|index| 0.8 - 0.02 * index as f64)
                .collect(),
            time: 0.25,
            step: 4,
        };
        let forward =
            transfer_pc_fan_cfd_p2p1_bdf_state_chain(&[&first, &second, &third], &source_state)
                .expect("forward BDF chain should succeed");
        let terminal = forward.states.last().expect("terminal forward state");
        let target_adjoint = BdfState {
            velocity: terminal
                .velocity
                .iter()
                .enumerate()
                .map(|(index, _)| -0.35 + 0.017 * index as f64)
                .collect(),
            velocity_prev: terminal
                .velocity_prev
                .iter()
                .enumerate()
                .map(|(index, _)| 0.45 - 0.013 * index as f64)
                .collect(),
            pressure: terminal
                .pressure
                .iter()
                .enumerate()
                .map(|(index, _)| -0.2 + 0.031 * index as f64)
                .collect(),
            time: terminal.time,
            step: terminal.step,
        };
        let reverse = transfer_pc_fan_cfd_p2p1_bdf_state_adjoint_chain(
            &[&first, &second, &third],
            &target_adjoint,
        )
        .expect("reverse BDF chain should succeed");
        let source_adjoint = reverse.states.last().expect("source adjoint state");

        let terminal_pairing = dot_values(&terminal.velocity, &target_adjoint.velocity)
            + dot_values(&terminal.velocity_prev, &target_adjoint.velocity_prev)
            + dot_values(&terminal.pressure, &target_adjoint.pressure);
        let source_pairing = dot_values(&source_state.velocity, &source_adjoint.velocity)
            + dot_values(&source_state.velocity_prev, &source_adjoint.velocity_prev)
            + dot_values(&source_state.pressure, &source_adjoint.pressure);
        assert!((terminal_pairing - source_pairing).abs() < 1.0e-8);
        assert_eq!(reverse.states.len(), 2);
        assert_eq!(reverse.report.stage_count, 2);
        assert_eq!(
            reverse.report.provenance,
            "pcfan:topology-changing-p2p1-bdf-state-adjoint-transfer:v1"
        );
    }

    #[test]
    fn komega_adjoint_chain_preserves_pairing_without_positivity_assumption() {
        let first = unit_cube_tetrahedra(1);
        let second = unit_cube_tetrahedra(2);
        let third = box_mesh_tetrahedra(0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 3, 2, 2);
        let source_state = komega_state(&first);
        let forward =
            transfer_pc_fan_cfd_komega_bdf_state_chain(&[&first, &second, &third], &source_state)
                .expect("forward k--omega chain should succeed");
        let terminal = forward.states.last().expect("terminal scalar state");
        let target_adjoint = PcFanCfdKOmegaBdfState {
            k: terminal
                .k
                .iter()
                .enumerate()
                .map(|(index, _)| -0.3 + 0.01 * index as f64)
                .collect(),
            k_prev: terminal
                .k_prev
                .iter()
                .enumerate()
                .map(|(index, _)| 0.2 - 0.015 * index as f64)
                .collect(),
            omega: terminal
                .omega
                .iter()
                .enumerate()
                .map(|(index, _)| -0.1 + 0.02 * index as f64)
                .collect(),
            omega_prev: terminal
                .omega_prev
                .iter()
                .enumerate()
                .map(|(index, _)| 0.4 - 0.01 * index as f64)
                .collect(),
            time: terminal.time,
            step: terminal.step,
        };
        let reverse = transfer_pc_fan_cfd_komega_bdf_state_adjoint_chain(
            &[&first, &second, &third],
            &target_adjoint,
        )
        .expect("reverse k--omega chain should succeed");
        let source_adjoint = reverse.states.last().expect("source scalar adjoint");

        let terminal_pairing = dot_values(&terminal.k, &target_adjoint.k)
            + dot_values(&terminal.k_prev, &target_adjoint.k_prev)
            + dot_values(&terminal.omega, &target_adjoint.omega)
            + dot_values(&terminal.omega_prev, &target_adjoint.omega_prev);
        let source_pairing = dot_values(&source_state.k, &source_adjoint.k)
            + dot_values(&source_state.k_prev, &source_adjoint.k_prev)
            + dot_values(&source_state.omega, &source_adjoint.omega)
            + dot_values(&source_state.omega_prev, &source_adjoint.omega_prev);
        assert!((terminal_pairing - source_pairing).abs() < 1.0e-10);
        assert!(source_adjoint.k.iter().any(|value| *value < 0.0));
        assert_eq!(reverse.states.len(), 2);
        assert_eq!(
            reverse.report.provenance,
            "pcfan:topology-changing-komega-bdf-state-adjoint-transfer:v1"
        );
    }
}
