//! Checked scalar state transfer between panelized 3-D propeller meshes.
//!
//! This module supplies the remesh/state-transfer seam required by the
//! application-level geometry workflow. The inverse-distance interpolation
//! core lives in [`sonium_fem::mesh::panel_transfer`]; this module keeps the
//! `BladeMesh` topology checks (planform, edge treatment), the product panel
//! mapping, and the public names as a thin wrapper.

use crate::{BladeMesh, BladePanel};

use sonium_fem::mesh::panel_transfer::{Panel, PanelSet, transfer_panel_scalar_state as transfer};
pub use sonium_fem::mesh::panel_transfer::{
    PanelStateTransferError, PanelStateTransferOptions, PanelStateTransferResult,
};

fn panel_center(panel: &BladePanel) -> [f64; 3] {
    let mut center = [0.0; 3];
    for point in panel.vertices {
        center[0] += point.x * 0.25;
        center[1] += point.y * 0.25;
        center[2] += point.z * 0.25;
    }
    center
}

fn as_panel_set(mesh: &BladeMesh) -> PanelSet {
    PanelSet {
        panels: mesh
            .panels
            .iter()
            .map(|panel| Panel {
                center: panel_center(panel),
                area: panel.area,
                group: panel.blade_index,
            })
            .collect(),
    }
}

/// Transfer one scalar state per panel between compatible 3-D meshes.
///
/// Product wrapper enforcing the blade-topology contract (planform and edge
/// treatment must match) before delegating to the reusable
/// [`sonium_fem::mesh::panel_transfer`] core.
pub fn transfer_panel_scalar_state(
    source_mesh: &BladeMesh,
    source_values: &[f64],
    target_mesh: &BladeMesh,
    options: PanelStateTransferOptions,
) -> Result<PanelStateTransferResult, PanelStateTransferError> {
    if source_mesh.planform != target_mesh.planform {
        return Err(PanelStateTransferError::TopologyMismatch { field: "planform" });
    }
    if source_mesh.edge_treatment != target_mesh.edge_treatment {
        return Err(PanelStateTransferError::TopologyMismatch {
            field: "edge_treatment",
        });
    }
    transfer(
        &as_panel_set(source_mesh),
        source_values,
        &as_panel_set(target_mesh),
        options,
    )
}
