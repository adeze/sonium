//! Checked H(curl)/H(div) hanging-edge and hanging-face constraints.
//!
//! The order-1 Nédélec edge DOF is an oriented tangential line moment.  On an
//! affine coarse edge, restriction to a fine subedge is therefore the signed
//! parameter-length fraction of the coarse moment.  Higher-order edge moments
//! and H(div) face moments need distinct trace matrices and are intentionally
//! admitted by separate packets.

use super::periodic::{MpcError, MpcSystem, MpcTerm, MultiPointConstraint};
use crate::mesh::{ElementType, Face, Mesh, topology::MeshTopology, types::MeshConnectivityError};
use crate::spaces::{FunctionSpace, HcurlSpace, HdivOrder, HdivSpace, SpaceValidationError};
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MixedHangingError {
    #[error("mixed hanging tolerance must be finite and positive")]
    InvalidTolerance(f64),
    #[error("mixed hanging constraints require the supported mesh dimension")]
    UnsupportedDimension(usize),
    #[error("mixed hanging-edge constraints require the supported lowest order")]
    UnsupportedOrder(usize),
    #[error("invalid mesh for mixed hanging edges: {errors:?}")]
    InvalidMesh { errors: Vec<MeshConnectivityError> },
    #[error("invalid mixed finite-element space")]
    InvalidSpace(#[source] SpaceValidationError),
    #[error("affine triangular-face hanging constraints require affine P1 elements")]
    UnsupportedGeometry,
    #[error("RT1 H(div) hanging-edge constraints require triangular elements")]
    UnsupportedEdgeTopology,
    #[error("3-D H(curl) hanging-edge constraints require tetrahedral elements")]
    UnsupportedHcurlTopology,
    #[error("3-D H(div) hanging constraints require triangular faces")]
    UnsupportedFaceTopology,
    #[error("RT1 triangular-face trace map is singular for fine face {face}")]
    SingularFaceTrace { face: usize },
    #[error("fine edge {fine_edge} has ambiguous containing coarse edge")]
    AmbiguousContainingEdge { fine_edge: usize },
    #[error("fine face {face} has ambiguous containing coarse face")]
    AmbiguousContainingFace { face: usize },
    #[error("fine edge {fine_edge} has no containing coarse edge")]
    MissingContainingEdge { fine_edge: usize },
    #[error("mixed hanging-edge MPC construction failed")]
    Mpc(#[source] MpcError),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HcurlHangingEdgeConstraint {
    pub slave_edge: usize,
    pub coarse_edge: usize,
    pub signed_fraction: f64,
}

/// A checked order-1 Nédélec restriction from a coarse triangular face to a
/// fine edge contained in that face. The three coefficients map the coarse
/// face's canonical edge moments to the fine edge's canonical line moment.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HcurlOrder1HangingFaceConstraint {
    pub slave_edge: usize,
    pub coarse_face: usize,
    pub coarse_edges: [usize; 3],
    pub coefficients: [f64; 3],
}

/// A checked order-2 Nédélec restriction from a coarse triangular face to a
/// fine edge contained in that face. The eight masters are the six coarse
/// edge modes followed by the two coarse face-interior modes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HcurlOrder2HangingFaceEdgeConstraint {
    pub slave_edge: usize,
    pub coarse_face: usize,
    pub coarse_dofs: [usize; 8],
    pub coefficients: [[f64; 8]; 2],
}

/// A checked order-2 Nédélec restriction for the two face-interior modes of
/// a fine triangular face contained in a coarse triangular face. The eight
/// masters are the six coarse edge modes followed by the two coarse face
/// modes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HcurlOrder2HangingFaceInteriorConstraint {
    pub slave_face: usize,
    pub slave_dofs: [usize; 2],
    pub coarse_face: usize,
    pub coarse_dofs: [usize; 8],
    pub coefficients: [[f64; 8]; 2],
}

/// A checked order-2 Nédélec edge-trace restriction.
///
/// The two edge modes use the tangential traces `1` and `1 - 2t` in the
/// canonical edge coordinate. `coefficients[slave_mode][coarse_mode]` maps
/// coarse edge coefficients to the corresponding fine-edge coefficients.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HcurlOrder2HangingEdgeConstraint {
    pub slave_edge: usize,
    pub coarse_edge: usize,
    pub signed_fraction: f64,
    pub fine_start_parameter: f64,
    pub coefficients: [[f64; 2]; 2],
}

/// A checked RT1 H(div) restriction from one coarse edge to a fine subedge.
///
/// The two edge moments are the constant and first centered normal-flux
/// moments. `coefficients[fine_mode][coarse_mode]` maps the coarse edge
/// moments to the corresponding fine-edge moments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdivRt1HangingEdgeConstraint {
    pub slave_edge: usize,
    pub coarse_edge: usize,
    pub signed_fraction: f64,
    pub fine_start_parameter: f64,
    pub coefficients: [[f64; 2]; 2],
}

/// A checked RT1 H(div) restriction from one coarse triangular face to a
/// fine triangular subface. The three rows and columns correspond to the
/// constant and two affine face normal-flux moments used by the production
/// tetrahedral RT1 tabulator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdivRt1HangingFaceConstraint {
    pub slave_face: usize,
    pub coarse_face: usize,
    pub signed_fraction: f64,
    pub coefficients: [[f64; 3]; 3],
}

impl HcurlHangingEdgeConstraint {
    pub fn to_mpc(self) -> MultiPointConstraint {
        MultiPointConstraint {
            slave: self.slave_edge,
            masters: vec![MpcTerm {
                dof: self.coarse_edge,
                coefficient: num_complex::Complex64::new(self.signed_fraction, 0.0),
            }],
        }
    }
}

impl HcurlOrder1HangingFaceConstraint {
    /// Convert the face restriction into one checked multi-master MPC row.
    pub fn to_mpc(self) -> MultiPointConstraint {
        MultiPointConstraint {
            slave: self.slave_edge,
            masters: self
                .coarse_edges
                .into_iter()
                .zip(self.coefficients)
                .map(|(dof, coefficient)| MpcTerm {
                    dof,
                    coefficient: num_complex::Complex64::new(coefficient, 0.0),
                })
                .collect(),
        }
    }
}

impl HcurlOrder2HangingFaceEdgeConstraint {
    /// Convert the two fine-edge modes into checked multi-master MPC rows.
    pub fn to_mpcs(self) -> [MultiPointConstraint; 2] {
        std::array::from_fn(|fine_mode| MultiPointConstraint {
            slave: self.slave_edge * 2 + fine_mode,
            masters: self
                .coarse_dofs
                .into_iter()
                .zip(self.coefficients[fine_mode])
                .map(|(dof, coefficient)| MpcTerm {
                    dof,
                    coefficient: num_complex::Complex64::new(coefficient, 0.0),
                })
                .collect(),
        })
    }
}

impl HcurlOrder2HangingFaceInteriorConstraint {
    /// Convert the two fine-face modes into checked multi-master MPC rows.
    pub fn to_mpcs(self) -> [MultiPointConstraint; 2] {
        std::array::from_fn(|fine_mode| MultiPointConstraint {
            slave: self.slave_dofs[fine_mode],
            masters: self
                .coarse_dofs
                .into_iter()
                .zip(self.coefficients[fine_mode])
                .map(|(dof, coefficient)| MpcTerm {
                    dof,
                    coefficient: num_complex::Complex64::new(coefficient, 0.0),
                })
                .collect(),
        })
    }
}

/// Detect order-1 Nédélec constraints for fine 2-D subedges.
pub fn hcurl_order1_hanging_edge_constraints_2d(
    mesh: &Mesh,
    space: &HcurlSpace,
    tolerance: f64,
) -> Result<Vec<HcurlHangingEdgeConstraint>, MixedHangingError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(MixedHangingError::InvalidTolerance(tolerance));
    }
    if mesh.dimension != 2 {
        return Err(MixedHangingError::UnsupportedDimension(mesh.dimension));
    }
    if space.order != 1 {
        return Err(MixedHangingError::UnsupportedOrder(space.order));
    }
    let topology = MeshTopology::try_build(mesh)
        .map_err(|errors| MixedHangingError::InvalidMesh { errors })?;
    space
        .validate(mesh)
        .map_err(MixedHangingError::InvalidSpace)?;

    let edge_lengths = topology
        .edges
        .iter()
        .map(|&(first, second)| edge_length(mesh, first, second))
        .collect::<Vec<_>>();
    if edge_lengths
        .iter()
        .any(|length| !length.is_finite() || *length <= tolerance)
    {
        return Err(MixedHangingError::InvalidMesh { errors: Vec::new() });
    }

    let mut constraints = Vec::new();
    for fine_edge in 0..topology.edges.len() {
        let (fine_first, fine_second) = topology.edges[fine_edge];
        let mut candidates = Vec::new();
        for coarse_edge in 0..topology.edges.len() {
            if coarse_edge == fine_edge
                || edge_lengths[coarse_edge] <= edge_lengths[fine_edge] + tolerance
            {
                continue;
            }
            let (coarse_first, coarse_second) = topology.edges[coarse_edge];
            let Some(first_parameter) =
                edge_parameter(mesh, coarse_first, coarse_second, fine_first, tolerance)
            else {
                continue;
            };
            let Some(second_parameter) =
                edge_parameter(mesh, coarse_first, coarse_second, fine_second, tolerance)
            else {
                continue;
            };
            let fraction = second_parameter - first_parameter;
            if fraction.abs() * edge_lengths[coarse_edge]
                <= edge_lengths[fine_edge] + 10.0 * tolerance
                && fraction.is_finite()
                && fraction.abs() > tolerance
            {
                candidates.push((coarse_edge, fraction));
            }
        }

        if candidates.is_empty() {
            continue;
        }
        candidates.sort_by(|left, right| {
            edge_lengths[right.0]
                .total_cmp(&edge_lengths[left.0])
                .then_with(|| left.0.cmp(&right.0))
        });
        if candidates.len() > 1
            && (edge_lengths[candidates[0].0] - edge_lengths[candidates[1].0]).abs() <= tolerance
        {
            return Err(MixedHangingError::AmbiguousContainingEdge { fine_edge });
        }
        let (coarse_edge, fraction) = candidates[0];
        constraints.push(HcurlHangingEdgeConstraint {
            slave_edge: fine_edge,
            coarse_edge,
            signed_fraction: fraction,
        });
    }

    Ok(constraints)
}

/// Build the checked order-1 H(curl) hanging-edge MPC system.
pub fn hcurl_order1_hanging_edge_mpc_system_2d(
    mesh: &Mesh,
    space: &HcurlSpace,
    tolerance: f64,
) -> Result<MpcSystem, MixedHangingError> {
    let constraints = hcurl_order1_hanging_edge_constraints_2d(mesh, space, tolerance)?;
    let mpcs = constraints
        .iter()
        .copied()
        .map(HcurlHangingEdgeConstraint::to_mpc)
        .collect::<Vec<_>>();
    MpcSystem::new(space.num_global_dofs(mesh), &mpcs).map_err(MixedHangingError::Mpc)
}

/// Detect order-1 Nédélec constraints on affine 3-D tetrahedral subedges.
///
/// Order-1 H(curl) has only oriented line moments on edges, so the same
/// signed parameter-length restriction applies in 3-D. The tetrahedral-only
/// admission keeps mixed and curved topology out of this bounded packet.
pub fn hcurl_order1_hanging_edge_constraints_3d(
    mesh: &Mesh,
    space: &HcurlSpace,
    tolerance: f64,
) -> Result<Vec<HcurlHangingEdgeConstraint>, MixedHangingError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(MixedHangingError::InvalidTolerance(tolerance));
    }
    if mesh.dimension != 3 {
        return Err(MixedHangingError::UnsupportedDimension(mesh.dimension));
    }
    if space.order != 1 {
        return Err(MixedHangingError::UnsupportedOrder(space.order));
    }
    if mesh
        .elements
        .iter()
        .any(|element| element.nodes.len() != element.element_type.num_vertices())
    {
        return Err(MixedHangingError::UnsupportedGeometry);
    }
    if mesh
        .elements
        .iter()
        .any(|element| element.element_type != ElementType::Tetrahedron)
    {
        return Err(MixedHangingError::UnsupportedHcurlTopology);
    }
    let topology = MeshTopology::try_build(mesh)
        .map_err(|errors| MixedHangingError::InvalidMesh { errors })?;
    space
        .validate(mesh)
        .map_err(MixedHangingError::InvalidSpace)?;
    let edge_lengths = topology
        .edges
        .iter()
        .map(|&(first, second)| edge_length_3d(mesh, first, second))
        .collect::<Vec<_>>();
    if edge_lengths
        .iter()
        .any(|length| !length.is_finite() || *length <= tolerance)
    {
        return Err(MixedHangingError::InvalidMesh { errors: Vec::new() });
    }

    let mut constraints = Vec::new();
    for fine_edge in 0..topology.edges.len() {
        let (fine_first, fine_second) = topology.edges[fine_edge];
        let mut candidates = Vec::new();
        for coarse_edge in 0..topology.edges.len() {
            if coarse_edge == fine_edge
                || edge_lengths[coarse_edge] <= edge_lengths[fine_edge] + tolerance
            {
                continue;
            }
            let (coarse_first, coarse_second) = topology.edges[coarse_edge];
            let Some(first_parameter) =
                edge_parameter_3d(mesh, coarse_first, coarse_second, fine_first, tolerance)
            else {
                continue;
            };
            let Some(second_parameter) =
                edge_parameter_3d(mesh, coarse_first, coarse_second, fine_second, tolerance)
            else {
                continue;
            };
            let signed_fraction = second_parameter - first_parameter;
            if signed_fraction.is_finite()
                && signed_fraction.abs() > tolerance
                && signed_fraction.abs() * edge_lengths[coarse_edge]
                    <= edge_lengths[fine_edge]
                        + 10.0 * tolerance * edge_lengths[coarse_edge].max(1.0)
            {
                candidates.push((coarse_edge, signed_fraction));
            }
        }
        if candidates.is_empty() {
            continue;
        }
        candidates.sort_by(|left, right| {
            edge_lengths[right.0]
                .total_cmp(&edge_lengths[left.0])
                .then_with(|| left.0.cmp(&right.0))
        });
        if candidates.len() > 1
            && (edge_lengths[candidates[0].0] - edge_lengths[candidates[1].0]).abs() <= tolerance
        {
            return Err(MixedHangingError::AmbiguousContainingEdge { fine_edge });
        }
        let (coarse_edge, signed_fraction) = candidates[0];
        constraints.push(HcurlHangingEdgeConstraint {
            slave_edge: fine_edge,
            coarse_edge,
            signed_fraction,
        });
    }
    Ok(constraints)
}

/// Build the checked order-1 3-D H(curl) hanging-edge MPC system.
pub fn hcurl_order1_hanging_edge_mpc_system_3d(
    mesh: &Mesh,
    space: &HcurlSpace,
    tolerance: f64,
) -> Result<MpcSystem, MixedHangingError> {
    let constraints = hcurl_order1_hanging_edge_constraints_3d(mesh, space, tolerance)?;
    let mpcs = constraints
        .iter()
        .copied()
        .map(HcurlHangingEdgeConstraint::to_mpc)
        .collect::<Vec<_>>();
    MpcSystem::new(space.num_global_dofs(mesh), &mpcs).map_err(MixedHangingError::Mpc)
}

/// Detect order-1 Nédélec tangential line-moment restrictions on affine 3-D
/// triangular hanging faces.
///
/// Unlike the edge-only packet, this also catches fine edges in the interior
/// of a coarse triangular face. Each fine canonical edge moment is computed
/// by integrating the production, Piola-tabulated coarse edge basis along the
/// fine segment. Only affine tetrahedral meshes with triangular faces are
/// admitted; higher-order and curved face maps need separate trace contracts.
pub fn hcurl_order1_hanging_tri_face_constraints_3d(
    mesh: &Mesh,
    space: &HcurlSpace,
    tolerance: f64,
) -> Result<Vec<HcurlOrder1HangingFaceConstraint>, MixedHangingError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(MixedHangingError::InvalidTolerance(tolerance));
    }
    if mesh.dimension != 3 {
        return Err(MixedHangingError::UnsupportedDimension(mesh.dimension));
    }
    if space.order != 1 {
        return Err(MixedHangingError::UnsupportedOrder(space.order));
    }
    if mesh
        .elements
        .iter()
        .any(|element| element.nodes.len() != element.element_type.num_vertices())
    {
        return Err(MixedHangingError::UnsupportedGeometry);
    }
    if mesh
        .elements
        .iter()
        .any(|element| element.element_type != ElementType::Tetrahedron)
    {
        return Err(MixedHangingError::UnsupportedHcurlTopology);
    }

    let topology = MeshTopology::try_build(mesh)
        .map_err(|errors| MixedHangingError::InvalidMesh { errors })?;
    space
        .validate(mesh)
        .map_err(MixedHangingError::InvalidSpace)?;
    let face_nodes = topology
        .faces
        .iter()
        .map(|face| match face {
            Face::Triangle(nodes) => Ok(*nodes),
            Face::Quad(_) => Err(MixedHangingError::UnsupportedFaceTopology),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let face_area_vectors = face_nodes
        .iter()
        .map(|&nodes| triangle_area_vector(mesh, nodes))
        .collect::<Vec<_>>();
    if face_area_vectors.iter().any(|vector| {
        !vector.iter().all(|value| value.is_finite())
            || !vector_norm(*vector).is_finite()
            || vector_norm(*vector) <= tolerance
    }) {
        return Err(MixedHangingError::InvalidMesh { errors: Vec::new() });
    }

    let mut constraints = Vec::new();
    for fine_edge in 0..topology.edges.len() {
        let (fine_first, fine_second) = topology.edges[fine_edge];
        let fine_length = edge_length_3d(mesh, fine_first, fine_second);
        if !fine_length.is_finite() || fine_length <= tolerance {
            return Err(MixedHangingError::InvalidMesh { errors: Vec::new() });
        }

        let mut candidates = Vec::new();
        for coarse_face in 0..face_nodes.len() {
            let coarse_edges = face_edge_indices(&topology, face_nodes[coarse_face])
                .ok_or(MixedHangingError::UnsupportedFaceTopology)?;
            if coarse_edges.contains(&fine_edge)
                || !triangle_contains_edge(
                    mesh,
                    face_nodes[coarse_face],
                    fine_first,
                    fine_second,
                    tolerance,
                )
            {
                continue;
            }
            candidates.push((
                coarse_face,
                vector_norm(face_area_vectors[coarse_face]),
                coarse_edges,
            ));
        }
        if candidates.is_empty() {
            continue;
        }
        candidates.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        if candidates.len() > 1 && (candidates[0].1 - candidates[1].1).abs() <= tolerance {
            return Err(MixedHangingError::AmbiguousContainingFace {
                face: candidates[0].0,
            });
        }
        let (coarse_face, _, coarse_edges) = candidates[0];
        let Some(coefficients) = hcurl_order1_face_edge_restriction(
            mesh,
            space,
            &topology,
            coarse_face,
            coarse_edges,
            (fine_first, fine_second),
            tolerance,
        ) else {
            return Err(MixedHangingError::SingularFaceTrace { face: coarse_face });
        };
        constraints.push(HcurlOrder1HangingFaceConstraint {
            slave_edge: fine_edge,
            coarse_face,
            coarse_edges,
            coefficients,
        });
    }

    Ok(constraints)
}

/// Build the checked order-1 H(curl) triangular-face MPC system in 3-D.
pub fn hcurl_order1_hanging_tri_face_mpc_system_3d(
    mesh: &Mesh,
    space: &HcurlSpace,
    tolerance: f64,
) -> Result<MpcSystem, MixedHangingError> {
    let constraints = hcurl_order1_hanging_tri_face_constraints_3d(mesh, space, tolerance)?;
    let mpcs = constraints
        .iter()
        .copied()
        .map(HcurlOrder1HangingFaceConstraint::to_mpc)
        .collect::<Vec<_>>();
    MpcSystem::new(space.num_global_dofs(mesh), &mpcs).map_err(MixedHangingError::Mpc)
}

/// Detect order-2 Nédélec edge-mode restrictions on affine 3-D triangular
/// hanging faces.
///
/// Each fine edge mode is recovered from the production order-2 coarse face
/// trace: the first row is the constant line moment and the second row is the
/// centered first moment. The eight coarse masters are the six edge modes and
/// two face-interior modes owned by the containing coarse tetrahedral face.
/// Face-interior fine modes are intentionally a separate follow-up contract.
pub fn hcurl_order2_hanging_tri_face_edge_constraints_3d(
    mesh: &Mesh,
    space: &HcurlSpace,
    tolerance: f64,
) -> Result<Vec<HcurlOrder2HangingFaceEdgeConstraint>, MixedHangingError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(MixedHangingError::InvalidTolerance(tolerance));
    }
    if mesh.dimension != 3 {
        return Err(MixedHangingError::UnsupportedDimension(mesh.dimension));
    }
    if space.order != 2 {
        return Err(MixedHangingError::UnsupportedOrder(space.order));
    }
    if mesh
        .elements
        .iter()
        .any(|element| element.nodes.len() != element.element_type.num_vertices())
    {
        return Err(MixedHangingError::UnsupportedGeometry);
    }
    if mesh
        .elements
        .iter()
        .any(|element| element.element_type != ElementType::Tetrahedron)
    {
        return Err(MixedHangingError::UnsupportedHcurlTopology);
    }

    let topology = MeshTopology::try_build(mesh)
        .map_err(|errors| MixedHangingError::InvalidMesh { errors })?;
    space
        .validate(mesh)
        .map_err(MixedHangingError::InvalidSpace)?;
    let face_nodes = topology
        .faces
        .iter()
        .map(|face| match face {
            Face::Triangle(nodes) => Ok(*nodes),
            Face::Quad(_) => Err(MixedHangingError::UnsupportedFaceTopology),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let face_area_vectors = face_nodes
        .iter()
        .map(|&nodes| triangle_area_vector(mesh, nodes))
        .collect::<Vec<_>>();
    if face_area_vectors.iter().any(|vector| {
        !vector.iter().all(|value| value.is_finite())
            || !vector_norm(*vector).is_finite()
            || vector_norm(*vector) <= tolerance
    }) {
        return Err(MixedHangingError::InvalidMesh { errors: Vec::new() });
    }

    let mut constraints = Vec::new();
    for fine_edge in 0..topology.edges.len() {
        let (fine_first, fine_second) = topology.edges[fine_edge];
        let fine_length = edge_length_3d(mesh, fine_first, fine_second);
        if !fine_length.is_finite() || fine_length <= tolerance {
            return Err(MixedHangingError::InvalidMesh { errors: Vec::new() });
        }

        let mut candidates = Vec::new();
        for coarse_face in 0..face_nodes.len() {
            let coarse_edges = face_edge_indices(&topology, face_nodes[coarse_face])
                .ok_or(MixedHangingError::UnsupportedFaceTopology)?;
            if coarse_edges.contains(&fine_edge)
                || !triangle_contains_edge(
                    mesh,
                    face_nodes[coarse_face],
                    fine_first,
                    fine_second,
                    tolerance,
                )
            {
                continue;
            }
            candidates.push((
                coarse_face,
                vector_norm(face_area_vectors[coarse_face]),
                coarse_edges,
            ));
        }
        if candidates.is_empty() {
            continue;
        }
        candidates.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        if candidates.len() > 1 && (candidates[0].1 - candidates[1].1).abs() <= tolerance {
            return Err(MixedHangingError::AmbiguousContainingFace {
                face: candidates[0].0,
            });
        }
        let (coarse_face, _, coarse_edges) = candidates[0];
        let (coarse_element, coarse_local_face) =
            rt1_face_owner(&topology, coarse_face).ok_or(MixedHangingError::UnsupportedGeometry)?;
        let element_dofs = space.local_to_global(mesh, coarse_element);
        let local_edges = topology.element_edge_indices(coarse_element);
        let mut coarse_local_dofs = [0usize; 8];
        for (edge_mode, &coarse_edge) in coarse_edges.iter().enumerate() {
            let local_edge = local_edges
                .iter()
                .position(|&edge| edge == coarse_edge)
                .ok_or(MixedHangingError::UnsupportedGeometry)?;
            coarse_local_dofs[2 * edge_mode] = 2 * local_edge;
            coarse_local_dofs[2 * edge_mode + 1] = 2 * local_edge + 1;
        }
        coarse_local_dofs[6] = 12 + 2 * coarse_local_face;
        coarse_local_dofs[7] = coarse_local_dofs[6] + 1;
        let coarse_dofs = coarse_local_dofs.map(|local_dof| element_dofs[local_dof]);
        let Some(coefficients) = hcurl_order2_face_edge_restriction(
            mesh,
            space,
            coarse_element,
            coarse_local_dofs,
            (fine_first, fine_second),
            tolerance,
        ) else {
            return Err(MixedHangingError::SingularFaceTrace { face: coarse_face });
        };
        constraints.push(HcurlOrder2HangingFaceEdgeConstraint {
            slave_edge: fine_edge,
            coarse_face,
            coarse_dofs,
            coefficients,
        });
    }

    Ok(constraints)
}

/// Build the checked order-2 H(curl) triangular-face edge MPC system in 3-D.
pub fn hcurl_order2_hanging_tri_face_edge_mpc_system_3d(
    mesh: &Mesh,
    space: &HcurlSpace,
    tolerance: f64,
) -> Result<MpcSystem, MixedHangingError> {
    let constraints = hcurl_order2_hanging_tri_face_edge_constraints_3d(mesh, space, tolerance)?;
    let mpcs = constraints
        .iter()
        .flat_map(|constraint| (*constraint).to_mpcs())
        .collect::<Vec<_>>();
    MpcSystem::new(space.num_global_dofs(mesh), &mpcs).map_err(MixedHangingError::Mpc)
}

/// Detect the two order-2 Nédélec face-interior restrictions on affine 3-D
/// triangular hanging faces.
///
/// Fine edge traces are removed first using the independently defined order-2
/// line moments. The remaining tangential field is projected onto the two
/// production fine-face bubble functions with a degree-5 triangular rule,
/// yielding a checked 2×8 map from the six coarse edge modes and two coarse
/// face modes. Curved and higher-order geometry remain outside this contract.
pub fn hcurl_order2_hanging_tri_face_interior_constraints_3d(
    mesh: &Mesh,
    space: &HcurlSpace,
    tolerance: f64,
) -> Result<Vec<HcurlOrder2HangingFaceInteriorConstraint>, MixedHangingError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(MixedHangingError::InvalidTolerance(tolerance));
    }
    if mesh.dimension != 3 {
        return Err(MixedHangingError::UnsupportedDimension(mesh.dimension));
    }
    if space.order != 2 {
        return Err(MixedHangingError::UnsupportedOrder(space.order));
    }
    if mesh
        .elements
        .iter()
        .any(|element| element.nodes.len() != element.element_type.num_vertices())
    {
        return Err(MixedHangingError::UnsupportedGeometry);
    }
    if mesh
        .elements
        .iter()
        .any(|element| element.element_type != ElementType::Tetrahedron)
    {
        return Err(MixedHangingError::UnsupportedHcurlTopology);
    }

    let topology = MeshTopology::try_build(mesh)
        .map_err(|errors| MixedHangingError::InvalidMesh { errors })?;
    space
        .validate(mesh)
        .map_err(MixedHangingError::InvalidSpace)?;
    let face_nodes = topology
        .faces
        .iter()
        .map(|face| match face {
            Face::Triangle(nodes) => Ok(*nodes),
            Face::Quad(_) => Err(MixedHangingError::UnsupportedFaceTopology),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let face_area_vectors = face_nodes
        .iter()
        .map(|&nodes| triangle_area_vector(mesh, nodes))
        .collect::<Vec<_>>();
    if face_area_vectors.iter().any(|vector| {
        !vector.iter().all(|value| value.is_finite())
            || !vector_norm(*vector).is_finite()
            || vector_norm(*vector) <= tolerance
    }) {
        return Err(MixedHangingError::InvalidMesh { errors: Vec::new() });
    }

    let mut constraints = Vec::new();
    for fine_face in 0..face_nodes.len() {
        let fine_area = vector_norm(face_area_vectors[fine_face]);
        let mut candidates = Vec::new();
        for coarse_face in 0..face_nodes.len() {
            if coarse_face == fine_face
                || vector_norm(face_area_vectors[coarse_face]) <= fine_area + tolerance
                || !triangle_contains_face(
                    mesh,
                    face_nodes[coarse_face],
                    face_nodes[fine_face],
                    tolerance,
                )
            {
                continue;
            }
            candidates.push((coarse_face, vector_norm(face_area_vectors[coarse_face])));
        }
        if candidates.is_empty() {
            continue;
        }
        candidates.sort_by(|left, right| {
            right
                .1
                .total_cmp(&left.1)
                .then_with(|| left.0.cmp(&right.0))
        });
        if candidates.len() > 1 && (candidates[0].1 - candidates[1].1).abs() <= tolerance {
            return Err(MixedHangingError::AmbiguousContainingFace { face: fine_face });
        }
        let coarse_face = candidates[0].0;
        let coarse_edges = face_edge_indices(&topology, face_nodes[coarse_face])
            .ok_or(MixedHangingError::UnsupportedFaceTopology)?;
        let (coarse_element, coarse_local_face) =
            rt1_face_owner(&topology, coarse_face).ok_or(MixedHangingError::UnsupportedGeometry)?;
        let element_dofs = space.local_to_global(mesh, coarse_element);
        let local_edges = topology.element_edge_indices(coarse_element);
        let mut coarse_local_dofs = [0usize; 8];
        for (edge_mode, &coarse_edge) in coarse_edges.iter().enumerate() {
            let local_edge = local_edges
                .iter()
                .position(|&edge| edge == coarse_edge)
                .ok_or(MixedHangingError::UnsupportedGeometry)?;
            coarse_local_dofs[2 * edge_mode] = 2 * local_edge;
            coarse_local_dofs[2 * edge_mode + 1] = 2 * local_edge + 1;
        }
        coarse_local_dofs[6] = 12 + 2 * coarse_local_face;
        coarse_local_dofs[7] = coarse_local_dofs[6] + 1;
        let coarse_dofs = coarse_local_dofs.map(|local_dof| element_dofs[local_dof]);
        let (fine_element, fine_local_face) =
            rt1_face_owner(&topology, fine_face).ok_or(MixedHangingError::UnsupportedGeometry)?;
        let fine_element_dofs = space.local_to_global(mesh, fine_element);
        let fine_local_face_dofs = [12 + 2 * fine_local_face, 12 + 2 * fine_local_face + 1];
        let slave_dofs = fine_local_face_dofs.map(|local_dof| fine_element_dofs[local_dof]);
        let Some(coefficients) = hcurl_order2_face_interior_restriction(
            mesh,
            space,
            &topology,
            coarse_element,
            coarse_local_dofs,
            fine_element,
            fine_local_face,
            fine_face,
            tolerance,
        ) else {
            return Err(MixedHangingError::SingularFaceTrace { face: fine_face });
        };
        constraints.push(HcurlOrder2HangingFaceInteriorConstraint {
            slave_face: fine_face,
            slave_dofs,
            coarse_face,
            coarse_dofs,
            coefficients,
        });
    }

    Ok(constraints)
}

/// Build the checked order-2 H(curl) triangular-face interior MPC system.
pub fn hcurl_order2_hanging_tri_face_interior_mpc_system_3d(
    mesh: &Mesh,
    space: &HcurlSpace,
    tolerance: f64,
) -> Result<MpcSystem, MixedHangingError> {
    let constraints =
        hcurl_order2_hanging_tri_face_interior_constraints_3d(mesh, space, tolerance)?;
    let mpcs = constraints
        .iter()
        .flat_map(|constraint| (*constraint).to_mpcs())
        .collect::<Vec<_>>();
    MpcSystem::new(space.num_global_dofs(mesh), &mpcs).map_err(MixedHangingError::Mpc)
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdivRt0HangingEdgeConstraint {
    pub slave_edge: usize,
    pub coarse_edge: usize,
    pub signed_fraction: f64,
}

impl HdivRt0HangingEdgeConstraint {
    pub fn to_mpc(self) -> MultiPointConstraint {
        MultiPointConstraint {
            slave: self.slave_edge,
            masters: vec![MpcTerm {
                dof: self.coarse_edge,
                coefficient: num_complex::Complex64::new(self.signed_fraction, 0.0),
            }],
        }
    }
}

impl HcurlOrder2HangingEdgeConstraint {
    /// Convert both fine-edge trace modes into checked MPC rows.
    pub fn to_mpcs(self) -> [MultiPointConstraint; 2] {
        [
            MultiPointConstraint {
                slave: self.slave_edge * 2,
                masters: vec![
                    MpcTerm {
                        dof: self.coarse_edge * 2,
                        coefficient: num_complex::Complex64::new(self.coefficients[0][0], 0.0),
                    },
                    MpcTerm {
                        dof: self.coarse_edge * 2 + 1,
                        coefficient: num_complex::Complex64::new(self.coefficients[0][1], 0.0),
                    },
                ],
            },
            MultiPointConstraint {
                slave: self.slave_edge * 2 + 1,
                masters: vec![
                    MpcTerm {
                        dof: self.coarse_edge * 2,
                        coefficient: num_complex::Complex64::new(self.coefficients[1][0], 0.0),
                    },
                    MpcTerm {
                        dof: self.coarse_edge * 2 + 1,
                        coefficient: num_complex::Complex64::new(self.coefficients[1][1], 0.0),
                    },
                ],
            },
        ]
    }
}

impl HdivRt1HangingEdgeConstraint {
    /// Convert both fine-edge RT1 moment restrictions into checked MPC rows.
    pub fn to_mpcs(self) -> [MultiPointConstraint; 2] {
        [
            MultiPointConstraint {
                slave: self.slave_edge * 2,
                masters: vec![
                    MpcTerm {
                        dof: self.coarse_edge * 2,
                        coefficient: num_complex::Complex64::new(self.coefficients[0][0], 0.0),
                    },
                    MpcTerm {
                        dof: self.coarse_edge * 2 + 1,
                        coefficient: num_complex::Complex64::new(self.coefficients[0][1], 0.0),
                    },
                ],
            },
            MultiPointConstraint {
                slave: self.slave_edge * 2 + 1,
                masters: vec![
                    MpcTerm {
                        dof: self.coarse_edge * 2,
                        coefficient: num_complex::Complex64::new(self.coefficients[1][0], 0.0),
                    },
                    MpcTerm {
                        dof: self.coarse_edge * 2 + 1,
                        coefficient: num_complex::Complex64::new(self.coefficients[1][1], 0.0),
                    },
                ],
            },
        ]
    }
}

impl HdivRt1HangingFaceConstraint {
    /// Convert the three fine-face moment restrictions into checked MPC rows.
    pub fn to_mpcs(self) -> [MultiPointConstraint; 3] {
        std::array::from_fn(|fine_mode| MultiPointConstraint {
            slave: self.slave_face * 3 + fine_mode,
            masters: (0..3)
                .map(|coarse_mode| MpcTerm {
                    dof: self.coarse_face * 3 + coarse_mode,
                    coefficient: num_complex::Complex64::new(
                        self.coefficients[fine_mode][coarse_mode],
                        0.0,
                    ),
                })
                .collect(),
        })
    }
}

/// Detect order-2 Nédélec restrictions for affine 2-D hanging edges.
pub fn hcurl_order2_hanging_edge_constraints_2d(
    mesh: &Mesh,
    space: &HcurlSpace,
    tolerance: f64,
) -> Result<Vec<HcurlOrder2HangingEdgeConstraint>, MixedHangingError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(MixedHangingError::InvalidTolerance(tolerance));
    }
    if mesh.dimension != 2 {
        return Err(MixedHangingError::UnsupportedDimension(mesh.dimension));
    }
    if space.order != 2 {
        return Err(MixedHangingError::UnsupportedOrder(space.order));
    }
    if mesh
        .elements
        .iter()
        .any(|element| element.nodes.len() != element.element_type.num_vertices())
    {
        return Err(MixedHangingError::UnsupportedGeometry);
    }
    let topology = MeshTopology::try_build(mesh)
        .map_err(|errors| MixedHangingError::InvalidMesh { errors })?;
    space
        .validate(mesh)
        .map_err(MixedHangingError::InvalidSpace)?;

    let edge_lengths = topology
        .edges
        .iter()
        .map(|&(first, second)| edge_length(mesh, first, second))
        .collect::<Vec<_>>();
    if edge_lengths
        .iter()
        .any(|length| !length.is_finite() || *length <= tolerance)
    {
        return Err(MixedHangingError::InvalidMesh { errors: Vec::new() });
    }

    let mut constraints = Vec::new();
    for fine_edge in 0..topology.edges.len() {
        let (fine_first, fine_second) = topology.edges[fine_edge];
        let mut candidates = Vec::new();
        for coarse_edge in 0..topology.edges.len() {
            if coarse_edge == fine_edge
                || edge_lengths[coarse_edge] <= edge_lengths[fine_edge] + tolerance
            {
                continue;
            }
            let (coarse_first, coarse_second) = topology.edges[coarse_edge];
            let Some(first_parameter) =
                edge_parameter(mesh, coarse_first, coarse_second, fine_first, tolerance)
            else {
                continue;
            };
            let Some(second_parameter) =
                edge_parameter(mesh, coarse_first, coarse_second, fine_second, tolerance)
            else {
                continue;
            };
            let signed_fraction = second_parameter - first_parameter;
            let fraction = signed_fraction.abs();
            if !signed_fraction.is_finite()
                || fraction <= tolerance
                || (fraction * edge_lengths[coarse_edge] - edge_lengths[fine_edge]).abs()
                    > 10.0 * tolerance * edge_lengths[coarse_edge].max(1.0)
            {
                continue;
            }
            candidates.push((coarse_edge, first_parameter, signed_fraction));
        }

        if candidates.is_empty() {
            continue;
        }
        candidates.sort_by(|left, right| {
            edge_lengths[right.0]
                .total_cmp(&edge_lengths[left.0])
                .then_with(|| left.0.cmp(&right.0))
        });
        if candidates.len() > 1
            && (edge_lengths[candidates[0].0] - edge_lengths[candidates[1].0]).abs() <= tolerance
        {
            return Err(MixedHangingError::AmbiguousContainingEdge { fine_edge });
        }

        let (coarse_edge, first_parameter, signed_fraction) = candidates[0];
        let fraction = signed_fraction.abs();
        let start = first_parameter.clamp(0.0, 1.0);
        let fraction_squared = fraction * fraction;
        let coefficients = [
            [
                signed_fraction,
                signed_fraction * (1.0 - 2.0 * start) - fraction_squared,
            ],
            [0.0, fraction_squared],
        ];
        if !coefficients
            .iter()
            .flatten()
            .all(|coefficient| coefficient.is_finite())
        {
            return Err(MixedHangingError::InvalidMesh { errors: Vec::new() });
        }
        constraints.push(HcurlOrder2HangingEdgeConstraint {
            slave_edge: fine_edge,
            coarse_edge,
            signed_fraction,
            fine_start_parameter: start,
            coefficients,
        });
    }
    Ok(constraints)
}

/// Build the checked order-2 H(curl) hanging-edge MPC system.
pub fn hcurl_order2_hanging_edge_mpc_system_2d(
    mesh: &Mesh,
    space: &HcurlSpace,
    tolerance: f64,
) -> Result<MpcSystem, MixedHangingError> {
    let constraints = hcurl_order2_hanging_edge_constraints_2d(mesh, space, tolerance)?;
    let mpcs = constraints
        .iter()
        .flat_map(|constraint| (*constraint).to_mpcs())
        .collect::<Vec<_>>();
    MpcSystem::new(space.num_global_dofs(mesh), &mpcs).map_err(MixedHangingError::Mpc)
}

/// Detect RT1 H(div) restrictions on affine 2-D triangular hanging edges.
///
/// RT1 edge moments use the normal traces `1 / L` and `3t / L`, with `t` the
/// canonical edge coordinate in `[-1, 1]`. If a fine edge maps to the coarse
/// edge by `t = t0 + b*s`, the restriction matrix is
/// `[[b, 3*b*t0], [0, b^2]]`, including the sign from reversed orientation.
pub fn hdiv_rt1_hanging_edge_constraints_2d(
    mesh: &Mesh,
    space: &HdivSpace,
    tolerance: f64,
) -> Result<Vec<HdivRt1HangingEdgeConstraint>, MixedHangingError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(MixedHangingError::InvalidTolerance(tolerance));
    }
    if mesh.dimension != 2 {
        return Err(MixedHangingError::UnsupportedDimension(mesh.dimension));
    }
    if space.order() != HdivOrder::Rt1 {
        return Err(MixedHangingError::UnsupportedOrder(
            space.order().polynomial_degree(),
        ));
    }
    if mesh
        .elements
        .iter()
        .any(|element| element.nodes.len() != element.element_type.num_vertices())
    {
        return Err(MixedHangingError::UnsupportedGeometry);
    }
    if mesh
        .elements
        .iter()
        .any(|element| element.element_type != crate::mesh::ElementType::Triangle)
    {
        return Err(MixedHangingError::UnsupportedEdgeTopology);
    }

    let topology = MeshTopology::try_build(mesh)
        .map_err(|errors| MixedHangingError::InvalidMesh { errors })?;
    space
        .validate(mesh)
        .map_err(MixedHangingError::InvalidSpace)?;
    let edge_lengths = topology
        .edges
        .iter()
        .map(|&(first, second)| edge_length(mesh, first, second))
        .collect::<Vec<_>>();
    if edge_lengths
        .iter()
        .any(|length| !length.is_finite() || *length <= tolerance)
    {
        return Err(MixedHangingError::InvalidMesh { errors: Vec::new() });
    }

    let mut constraints = Vec::new();
    for fine_edge in 0..topology.edges.len() {
        let (fine_first, fine_second) = topology.edges[fine_edge];
        let mut candidates = Vec::new();
        for coarse_edge in 0..topology.edges.len() {
            if coarse_edge == fine_edge
                || edge_lengths[coarse_edge] <= edge_lengths[fine_edge] + tolerance
            {
                continue;
            }
            let (coarse_first, coarse_second) = topology.edges[coarse_edge];
            let Some(first_parameter) =
                edge_parameter(mesh, coarse_first, coarse_second, fine_first, tolerance)
            else {
                continue;
            };
            let Some(second_parameter) =
                edge_parameter(mesh, coarse_first, coarse_second, fine_second, tolerance)
            else {
                continue;
            };
            let signed_fraction = second_parameter - first_parameter;
            if signed_fraction.abs() * edge_lengths[coarse_edge]
                <= edge_lengths[fine_edge] + 10.0 * tolerance * edge_lengths[coarse_edge].max(1.0)
                && signed_fraction.is_finite()
                && signed_fraction.abs() > tolerance
            {
                candidates.push((coarse_edge, signed_fraction, first_parameter));
            }
        }
        if candidates.is_empty() {
            continue;
        }
        candidates.sort_by(|left, right| {
            edge_lengths[right.0]
                .total_cmp(&edge_lengths[left.0])
                .then_with(|| left.0.cmp(&right.0))
        });
        if candidates.len() > 1
            && (edge_lengths[candidates[0].0] - edge_lengths[candidates[1].0]).abs() <= tolerance
        {
            return Err(MixedHangingError::AmbiguousContainingEdge { fine_edge });
        }
        let (coarse_edge, signed_fraction, first_parameter) = candidates[0];
        let start = first_parameter.clamp(0.0, 1.0);
        let coefficients = [
            [signed_fraction, 3.0 * signed_fraction * (2.0 * start - 1.0)],
            [0.0, signed_fraction * signed_fraction],
        ];
        if !coefficients
            .iter()
            .flatten()
            .all(|coefficient| coefficient.is_finite())
        {
            return Err(MixedHangingError::InvalidMesh { errors: Vec::new() });
        }
        constraints.push(HdivRt1HangingEdgeConstraint {
            slave_edge: fine_edge,
            coarse_edge,
            signed_fraction,
            fine_start_parameter: start,
            coefficients,
        });
    }
    Ok(constraints)
}

/// Build the checked RT1 H(div) hanging-edge MPC system.
pub fn hdiv_rt1_hanging_edge_mpc_system_2d(
    mesh: &Mesh,
    space: &HdivSpace,
    tolerance: f64,
) -> Result<MpcSystem, MixedHangingError> {
    let constraints = hdiv_rt1_hanging_edge_constraints_2d(mesh, space, tolerance)?;
    let mpcs = constraints
        .iter()
        .flat_map(|constraint| (*constraint).to_mpcs())
        .collect::<Vec<_>>();
    MpcSystem::new(space.num_global_dofs(mesh), &mpcs).map_err(MixedHangingError::Mpc)
}

/// Detect RT0 H(div) normal-flux restrictions on affine 2-D hanging edges.
pub fn hdiv_rt0_hanging_edge_constraints_2d(
    mesh: &Mesh,
    space: &HdivSpace,
    tolerance: f64,
) -> Result<Vec<HdivRt0HangingEdgeConstraint>, MixedHangingError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(MixedHangingError::InvalidTolerance(tolerance));
    }
    if mesh.dimension != 2 {
        return Err(MixedHangingError::UnsupportedDimension(mesh.dimension));
    }
    if space.order() != HdivOrder::Rt0 {
        return Err(MixedHangingError::UnsupportedOrder(
            space.order().polynomial_degree(),
        ));
    }
    let topology = MeshTopology::try_build(mesh)
        .map_err(|errors| MixedHangingError::InvalidMesh { errors })?;
    space
        .validate(mesh)
        .map_err(MixedHangingError::InvalidSpace)?;

    let edge_lengths = topology
        .edges
        .iter()
        .map(|&(first, second)| edge_length(mesh, first, second))
        .collect::<Vec<_>>();
    if edge_lengths
        .iter()
        .any(|length| !length.is_finite() || *length <= tolerance)
    {
        return Err(MixedHangingError::InvalidMesh { errors: Vec::new() });
    }

    let mut constraints = Vec::new();
    for fine_edge in 0..topology.edges.len() {
        let (fine_first, fine_second) = topology.edges[fine_edge];
        let mut candidates = Vec::new();
        for coarse_edge in 0..topology.edges.len() {
            if coarse_edge == fine_edge
                || edge_lengths[coarse_edge] <= edge_lengths[fine_edge] + tolerance
            {
                continue;
            }
            let (coarse_first, coarse_second) = topology.edges[coarse_edge];
            let Some(first_parameter) =
                edge_parameter(mesh, coarse_first, coarse_second, fine_first, tolerance)
            else {
                continue;
            };
            let Some(second_parameter) =
                edge_parameter(mesh, coarse_first, coarse_second, fine_second, tolerance)
            else {
                continue;
            };
            let fraction = second_parameter - first_parameter;
            if fraction.abs() * edge_lengths[coarse_edge]
                <= edge_lengths[fine_edge] + 10.0 * tolerance
                && fraction.is_finite()
                && fraction.abs() > tolerance
            {
                candidates.push((coarse_edge, fraction));
            }
        }
        if candidates.is_empty() {
            continue;
        }
        candidates.sort_by(|left, right| {
            edge_lengths[right.0]
                .total_cmp(&edge_lengths[left.0])
                .then_with(|| left.0.cmp(&right.0))
        });
        if candidates.len() > 1
            && (edge_lengths[candidates[0].0] - edge_lengths[candidates[1].0]).abs() <= tolerance
        {
            return Err(MixedHangingError::AmbiguousContainingEdge { fine_edge });
        }
        let (coarse_edge, fraction) = candidates[0];
        constraints.push(HdivRt0HangingEdgeConstraint {
            slave_edge: fine_edge,
            coarse_edge,
            signed_fraction: fraction,
        });
    }
    Ok(constraints)
}

/// Build the checked RT0 H(div) hanging-edge MPC system.
pub fn hdiv_rt0_hanging_edge_mpc_system_2d(
    mesh: &Mesh,
    space: &HdivSpace,
    tolerance: f64,
) -> Result<MpcSystem, MixedHangingError> {
    let constraints = hdiv_rt0_hanging_edge_constraints_2d(mesh, space, tolerance)?;
    let mpcs = constraints
        .iter()
        .copied()
        .map(HdivRt0HangingEdgeConstraint::to_mpc)
        .collect::<Vec<_>>();
    MpcSystem::new(space.num_global_dofs(mesh), &mpcs).map_err(MixedHangingError::Mpc)
}

/// A signed RT0 H(div) triangular-face restriction on an affine 3-D mesh.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HdivRt0HangingFaceConstraint {
    pub slave_face: usize,
    pub coarse_face: usize,
    pub signed_fraction: f64,
}

impl HdivRt0HangingFaceConstraint {
    /// Convert the face restriction into a one-master MPC row.
    pub fn to_mpc(self) -> MultiPointConstraint {
        MultiPointConstraint {
            slave: self.slave_face,
            masters: vec![MpcTerm {
                dof: self.coarse_face,
                coefficient: num_complex::Complex64::new(self.signed_fraction, 0.0),
            }],
        }
    }
}

/// Detect RT0 H(div) normal-flux restrictions on affine 3-D triangular faces.
///
/// Each fine triangular face must lie in, and be contained by, a larger coarse
/// triangular face. The coefficient is the signed ratio of their oriented
/// area vectors, so canonical face orientation is preserved in the MPC.
pub fn hdiv_rt0_hanging_tri_face_constraints_3d(
    mesh: &Mesh,
    space: &HdivSpace,
    tolerance: f64,
) -> Result<Vec<HdivRt0HangingFaceConstraint>, MixedHangingError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(MixedHangingError::InvalidTolerance(tolerance));
    }
    if mesh.dimension != 3 {
        return Err(MixedHangingError::UnsupportedDimension(mesh.dimension));
    }
    if space.order() != HdivOrder::Rt0 {
        return Err(MixedHangingError::UnsupportedOrder(
            space.order().polynomial_degree(),
        ));
    }
    if mesh
        .elements
        .iter()
        .any(|element| element.nodes.len() != element.element_type.num_vertices())
    {
        return Err(MixedHangingError::UnsupportedGeometry);
    }

    let topology = MeshTopology::try_build(mesh)
        .map_err(|errors| MixedHangingError::InvalidMesh { errors })?;
    space
        .validate(mesh)
        .map_err(MixedHangingError::InvalidSpace)?;
    if topology
        .faces
        .iter()
        .any(|face| !matches!(face, Face::Triangle(_)))
    {
        return Err(MixedHangingError::UnsupportedFaceTopology);
    }

    let face_nodes = topology
        .faces
        .iter()
        .map(|face| match face {
            Face::Triangle(nodes) => Ok(*nodes),
            Face::Quad(_) => Err(MixedHangingError::UnsupportedFaceTopology),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let area_vectors = face_nodes
        .iter()
        .map(|&nodes| triangle_area_vector(mesh, nodes))
        .collect::<Vec<_>>();
    if area_vectors.iter().any(|vector| {
        !vector.iter().all(|value| value.is_finite())
            || !vector_norm(*vector).is_finite()
            || vector_norm(*vector) <= tolerance
    }) {
        return Err(MixedHangingError::InvalidMesh { errors: Vec::new() });
    }

    let mut constraints = Vec::new();
    for fine_face in 0..face_nodes.len() {
        let fine_area = vector_norm(area_vectors[fine_face]);
        let mut candidates = Vec::new();
        for coarse_face in 0..face_nodes.len() {
            let coarse_area = vector_norm(area_vectors[coarse_face]);
            if coarse_face == fine_face || coarse_area <= fine_area + tolerance {
                continue;
            }
            if !triangle_contains_face(
                mesh,
                face_nodes[coarse_face],
                face_nodes[fine_face],
                tolerance,
            ) {
                continue;
            }
            let coarse_area_squared =
                vector_dot(area_vectors[coarse_face], area_vectors[coarse_face]);
            let signed_fraction = vector_dot(area_vectors[fine_face], area_vectors[coarse_face])
                / coarse_area_squared;
            let area_tolerance = 10.0 * tolerance * coarse_area.sqrt().max(tolerance);
            if signed_fraction.is_finite()
                && signed_fraction.abs() > tolerance
                && signed_fraction.abs() * coarse_area <= fine_area + area_tolerance
            {
                candidates.push((coarse_face, signed_fraction));
            }
        }
        if candidates.is_empty() {
            continue;
        }
        candidates.sort_by(|left, right| {
            vector_norm(area_vectors[right.0])
                .total_cmp(&vector_norm(area_vectors[left.0]))
                .then_with(|| left.0.cmp(&right.0))
        });
        if candidates.len() > 1
            && (vector_norm(area_vectors[candidates[0].0])
                - vector_norm(area_vectors[candidates[1].0]))
            .abs()
                <= tolerance
        {
            return Err(MixedHangingError::AmbiguousContainingFace { face: fine_face });
        }
        let (coarse_face, signed_fraction) = candidates[0];
        constraints.push(HdivRt0HangingFaceConstraint {
            slave_face: fine_face,
            coarse_face,
            signed_fraction,
        });
    }

    Ok(constraints)
}

/// Build the checked RT0 H(div) triangular-face MPC system in 3-D.
pub fn hdiv_rt0_hanging_tri_face_mpc_system_3d(
    mesh: &Mesh,
    space: &HdivSpace,
    tolerance: f64,
) -> Result<MpcSystem, MixedHangingError> {
    let constraints = hdiv_rt0_hanging_tri_face_constraints_3d(mesh, space, tolerance)?;
    let mpcs = constraints
        .iter()
        .copied()
        .map(HdivRt0HangingFaceConstraint::to_mpc)
        .collect::<Vec<_>>();
    MpcSystem::new(space.num_global_dofs(mesh), &mpcs).map_err(MixedHangingError::Mpc)
}

/// Detect RT1 H(div) normal-flux restrictions on affine 3-D tetrahedral
/// meshes with triangular hanging faces.
///
/// The restriction coefficients are computed from the production RT1
/// tetrahedral basis and the independent three-mode face moment functionals.
/// This keeps local face frames, Piola mapping, and the topology-owned normal
/// orientation in the contract instead of assuming that all tetrahedra use
/// the same vertex ordering.
pub fn hdiv_rt1_hanging_tri_face_constraints_3d(
    mesh: &Mesh,
    space: &HdivSpace,
    tolerance: f64,
) -> Result<Vec<HdivRt1HangingFaceConstraint>, MixedHangingError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(MixedHangingError::InvalidTolerance(tolerance));
    }
    if mesh.dimension != 3 {
        return Err(MixedHangingError::UnsupportedDimension(mesh.dimension));
    }
    if space.order() != HdivOrder::Rt1 {
        return Err(MixedHangingError::UnsupportedOrder(
            space.order().polynomial_degree(),
        ));
    }
    if mesh
        .elements
        .iter()
        .any(|element| element.nodes.len() != element.element_type.num_vertices())
    {
        return Err(MixedHangingError::UnsupportedGeometry);
    }
    if mesh
        .elements
        .iter()
        .any(|element| element.element_type != ElementType::Tetrahedron)
    {
        return Err(MixedHangingError::UnsupportedFaceTopology);
    }

    let topology = MeshTopology::try_build(mesh)
        .map_err(|errors| MixedHangingError::InvalidMesh { errors })?;
    space
        .validate(mesh)
        .map_err(MixedHangingError::InvalidSpace)?;
    let face_nodes = topology
        .faces
        .iter()
        .map(|face| match face {
            Face::Triangle(nodes) => Ok(*nodes),
            Face::Quad(_) => Err(MixedHangingError::UnsupportedFaceTopology),
        })
        .collect::<Result<Vec<_>, _>>()?;
    let area_vectors = face_nodes
        .iter()
        .map(|&nodes| triangle_area_vector(mesh, nodes))
        .collect::<Vec<_>>();
    if area_vectors.iter().any(|vector| {
        !vector.iter().all(|value| value.is_finite()) || vector_norm(*vector) <= tolerance
    }) {
        return Err(MixedHangingError::InvalidMesh { errors: Vec::new() });
    }

    let mut constraints = Vec::new();
    for fine_face in 0..face_nodes.len() {
        let fine_area = vector_norm(area_vectors[fine_face]);
        let mut candidates = Vec::new();
        for coarse_face in 0..face_nodes.len() {
            if coarse_face == fine_face {
                continue;
            }
            let coarse_area = vector_norm(area_vectors[coarse_face]);
            if coarse_area <= fine_area + tolerance
                || !triangle_contains_face(
                    mesh,
                    face_nodes[coarse_face],
                    face_nodes[fine_face],
                    tolerance,
                )
            {
                continue;
            }
            let coarse_area_squared =
                vector_dot(area_vectors[coarse_face], area_vectors[coarse_face]);
            let signed_fraction = vector_dot(area_vectors[fine_face], area_vectors[coarse_face])
                / coarse_area_squared;
            let area_tolerance = 10.0 * tolerance * coarse_area.sqrt().max(tolerance);
            if signed_fraction.is_finite()
                && signed_fraction.abs() > tolerance
                && signed_fraction.abs() * coarse_area <= fine_area + area_tolerance
            {
                candidates.push((coarse_face, signed_fraction));
            }
        }
        if candidates.is_empty() {
            continue;
        }
        candidates.sort_by(|left, right| {
            vector_norm(area_vectors[right.0])
                .total_cmp(&vector_norm(area_vectors[left.0]))
                .then_with(|| left.0.cmp(&right.0))
        });
        if candidates.len() > 1
            && (vector_norm(area_vectors[candidates[0].0])
                - vector_norm(area_vectors[candidates[1].0]))
            .abs()
                <= tolerance
        {
            return Err(MixedHangingError::AmbiguousContainingFace { face: fine_face });
        }
        let (coarse_face, signed_fraction) = candidates[0];
        let coefficients =
            rt1_face_restriction_matrix(mesh, space, &topology, coarse_face, fine_face, tolerance)
                .ok_or(MixedHangingError::SingularFaceTrace { face: fine_face })?;
        if !coefficients
            .iter()
            .flatten()
            .all(|coefficient| coefficient.is_finite())
        {
            return Err(MixedHangingError::InvalidMesh { errors: Vec::new() });
        }
        constraints.push(HdivRt1HangingFaceConstraint {
            slave_face: fine_face,
            coarse_face,
            signed_fraction,
            coefficients,
        });
    }
    Ok(constraints)
}

/// Build the checked RT1 triangular-face MPC system.
pub fn hdiv_rt1_hanging_tri_face_mpc_system_3d(
    mesh: &Mesh,
    space: &HdivSpace,
    tolerance: f64,
) -> Result<MpcSystem, MixedHangingError> {
    let constraints = hdiv_rt1_hanging_tri_face_constraints_3d(mesh, space, tolerance)?;
    let mpcs = constraints
        .iter()
        .flat_map(|constraint| (*constraint).to_mpcs())
        .collect::<Vec<_>>();
    MpcSystem::new(space.num_global_dofs(mesh), &mpcs).map_err(MixedHangingError::Mpc)
}

const RT1_TETRA_FACE_RULE: [(f64, f64, f64); 3] = [
    (1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0),
    (2.0 / 3.0, 1.0 / 6.0, 1.0 / 6.0),
    (1.0 / 6.0, 2.0 / 3.0, 1.0 / 6.0),
];

fn rt1_face_owner(topology: &MeshTopology, face: usize) -> Option<(usize, usize)> {
    topology
        .element_faces
        .iter()
        .enumerate()
        .find_map(|(element, faces)| {
            faces
                .iter()
                .position(|&candidate| candidate == face)
                .map(|local_face| (element, local_face))
        })
}

fn rt1_face_reference_point(face: usize, a: f64, b: f64) -> [f64; 3] {
    match face {
        0 => [a, b, 0.0],
        1 => [a, 0.0, b],
        2 => [0.0, a, b],
        3 => [a, b, 1.0 - a - b],
        _ => unreachable!("tetrahedron has four faces"),
    }
}

fn rt1_face_monomial(face: usize, mode: usize, point: [f64; 3]) -> f64 {
    match mode {
        0 => 1.0,
        1 => match face {
            0 | 1 => point[0],
            2 | 3 => point[1],
            _ => unreachable!("tetrahedron has four faces"),
        },
        2 => match face {
            0 => point[1],
            1..=3 => point[2],
            _ => unreachable!("tetrahedron has four faces"),
        },
        _ => unreachable!("RT1 triangular face has three modes"),
    }
}

fn rt1_face_reference_tangents(face: usize) -> ([f64; 3], [f64; 3]) {
    match face {
        0 => ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        1 => ([1.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
        2 => ([0.0, 1.0, 0.0], [0.0, 0.0, 1.0]),
        3 => ([1.0, 0.0, -1.0], [0.0, 1.0, -1.0]),
        _ => unreachable!("tetrahedron has four faces"),
    }
}

fn rt1_face_opposite_vertex(face: usize) -> [f64; 3] {
    match face {
        0 => [0.0, 0.0, 1.0],
        1 => [0.0, 1.0, 0.0],
        2 => [1.0, 0.0, 0.0],
        3 => [0.0, 0.0, 0.0],
        _ => unreachable!("tetrahedron has four faces"),
    }
}

fn tetra_affine_geometry(mesh: &Mesh, element: usize) -> Option<([f64; 3], [[f64; 3]; 3], f64)> {
    let nodes = &mesh.elements[element].nodes;
    if nodes.len() != 4 {
        return None;
    }
    let origin = mesh.nodes[nodes[0]];
    let p1 = mesh.nodes[nodes[1]];
    let p2 = mesh.nodes[nodes[2]];
    let p3 = mesh.nodes[nodes[3]];
    let jacobian = [
        [p1.x - origin.x, p2.x - origin.x, p3.x - origin.x],
        [p1.y - origin.y, p2.y - origin.y, p3.y - origin.y],
        [p1.z - origin.z, p2.z - origin.z, p3.z - origin.z],
    ];
    let determinant = matrix_det_3(jacobian);
    if !determinant.is_finite() || determinant.abs() <= f64::EPSILON {
        return None;
    }
    Some(([origin.x, origin.y, origin.z], jacobian, determinant))
}

fn matrix_det_3(matrix: [[f64; 3]; 3]) -> f64 {
    matrix[0][0] * (matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1])
        - matrix[0][1] * (matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0])
        + matrix[0][2] * (matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0])
}

fn matrix_vec_3(matrix: [[f64; 3]; 3], vector: [f64; 3]) -> [f64; 3] {
    [
        vector_dot(matrix[0], vector),
        vector_dot(matrix[1], vector),
        vector_dot(matrix[2], vector),
    ]
}

fn physical_from_reference(
    origin: [f64; 3],
    jacobian: [[f64; 3]; 3],
    reference: [f64; 3],
) -> [f64; 3] {
    let mapped = matrix_vec_3(jacobian, reference);
    [
        origin[0] + mapped[0],
        origin[1] + mapped[1],
        origin[2] + mapped[2],
    ]
}

fn reference_from_physical(
    origin: [f64; 3],
    jacobian: [[f64; 3]; 3],
    determinant: f64,
    physical: [f64; 3],
) -> Option<[f64; 3]> {
    let rhs = [
        physical[0] - origin[0],
        physical[1] - origin[1],
        physical[2] - origin[2],
    ];
    let c0 = [jacobian[0][0], jacobian[1][0], jacobian[2][0]];
    let c1 = [jacobian[0][1], jacobian[1][1], jacobian[2][1]];
    let c2 = [jacobian[0][2], jacobian[1][2], jacobian[2][2]];
    let cross_12 = vector_cross(c1, c2);
    let cross_rhs_2 = vector_cross(rhs, c2);
    let cross_1_rhs = vector_cross(c1, rhs);
    let reference = [
        vector_dot(rhs, cross_12) / determinant,
        vector_dot(c0, cross_rhs_2) / determinant,
        vector_dot(c0, cross_1_rhs) / determinant,
    ];
    reference
        .iter()
        .all(|value| value.is_finite())
        .then_some(reference)
}

fn vector_cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn rt1_face_surface_geometry(
    origin: [f64; 3],
    jacobian: [[f64; 3]; 3],
    face: usize,
) -> Option<([f64; 3], f64)> {
    let (reference_tangent_a, reference_tangent_b) = rt1_face_reference_tangents(face);
    let tangent_a = matrix_vec_3(jacobian, reference_tangent_a);
    let tangent_b = matrix_vec_3(jacobian, reference_tangent_b);
    let mut area_vector = vector_cross(tangent_a, tangent_b);
    let face_origin =
        physical_from_reference(origin, jacobian, rt1_face_reference_point(face, 0.0, 0.0));
    let opposite = physical_from_reference(origin, jacobian, rt1_face_opposite_vertex(face));
    if vector_dot(
        area_vector,
        [
            opposite[0] - face_origin[0],
            opposite[1] - face_origin[1],
            opposite[2] - face_origin[2],
        ],
    ) > 0.0
    {
        area_vector = [-area_vector[0], -area_vector[1], -area_vector[2]];
    }
    let surface_jacobian = vector_norm(area_vector);
    if !surface_jacobian.is_finite() || surface_jacobian <= f64::EPSILON {
        return None;
    }
    Some((
        [
            area_vector[0] / surface_jacobian,
            area_vector[1] / surface_jacobian,
            area_vector[2] / surface_jacobian,
        ],
        surface_jacobian,
    ))
}

fn matrix_inverse_2(matrix: [[f64; 2]; 2], tolerance: f64) -> Option<[[f64; 2]; 2]> {
    let determinant = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    let scale = matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(1.0);
    if !determinant.is_finite() || determinant.abs() <= tolerance * scale.powi(2) {
        return None;
    }
    Some([
        [matrix[1][1] / determinant, -matrix[0][1] / determinant],
        [-matrix[1][0] / determinant, matrix[0][0] / determinant],
    ])
}

fn matrix_inverse_3(matrix: [[f64; 3]; 3], tolerance: f64) -> Option<[[f64; 3]; 3]> {
    let determinant = matrix_det_3(matrix);
    let scale = matrix
        .iter()
        .flatten()
        .map(|value| value.abs())
        .fold(0.0, f64::max)
        .max(1.0);
    if !determinant.is_finite() || determinant.abs() <= tolerance * scale.powi(3) {
        return None;
    }
    let cofactors = [
        [
            matrix[1][1] * matrix[2][2] - matrix[1][2] * matrix[2][1],
            -(matrix[1][0] * matrix[2][2] - matrix[1][2] * matrix[2][0]),
            matrix[1][0] * matrix[2][1] - matrix[1][1] * matrix[2][0],
        ],
        [
            -(matrix[0][1] * matrix[2][2] - matrix[0][2] * matrix[2][1]),
            matrix[0][0] * matrix[2][2] - matrix[0][2] * matrix[2][0],
            -(matrix[0][0] * matrix[2][1] - matrix[0][1] * matrix[2][0]),
        ],
        [
            matrix[0][1] * matrix[1][2] - matrix[0][2] * matrix[1][1],
            -(matrix[0][0] * matrix[1][2] - matrix[0][2] * matrix[1][0]),
            matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0],
        ],
    ];
    Some(std::array::from_fn(|row| {
        std::array::from_fn(|column| cofactors[column][row] / determinant)
    }))
}

fn rt1_face_restriction_matrix(
    mesh: &Mesh,
    space: &HdivSpace,
    topology: &MeshTopology,
    coarse_face: usize,
    fine_face: usize,
    tolerance: f64,
) -> Option<[[f64; 3]; 3]> {
    let (coarse_element, coarse_local_face) = rt1_face_owner(topology, coarse_face)?;
    let (fine_element, fine_local_face) = rt1_face_owner(topology, fine_face)?;
    let (coarse_origin, coarse_jacobian, coarse_determinant) =
        tetra_affine_geometry(mesh, coarse_element)?;
    let (fine_origin, fine_jacobian, _) = tetra_affine_geometry(mesh, fine_element)?;
    let (fine_normal, fine_surface_jacobian) =
        rt1_face_surface_geometry(fine_origin, fine_jacobian, fine_local_face)?;
    let mut raw = [[0.0; 3]; 3];
    let mut fine_functionals = [[0.0; 3]; 3];

    for &(a, b, weight) in &RT1_TETRA_FACE_RULE {
        let fine_reference = rt1_face_reference_point(fine_local_face, a, b);
        let physical = physical_from_reference(fine_origin, fine_jacobian, fine_reference);
        let coarse_reference =
            reference_from_physical(coarse_origin, coarse_jacobian, coarse_determinant, physical)?;
        let fine_tabulation = space.tabulate(
            mesh,
            fine_element,
            fine_reference[0],
            fine_reference[1],
            fine_reference[2],
        );
        let coarse_tabulation = space.tabulate(
            mesh,
            coarse_element,
            coarse_reference[0],
            coarse_reference[1],
            coarse_reference[2],
        );
        for mode in 0..3 {
            let monomial = rt1_face_monomial(fine_local_face, mode, fine_reference);
            let measure = weight * fine_surface_jacobian * monomial;
            for (fine_mode, slot) in fine_functionals[mode].iter_mut().enumerate() {
                let local_dof = 3 * fine_local_face + fine_mode;
                let value = [
                    fine_tabulation.vector_value(local_dof, 0),
                    fine_tabulation.vector_value(local_dof, 1),
                    fine_tabulation.vector_value(local_dof, 2),
                ];
                *slot += measure * vector_dot(value, fine_normal);
            }
            for (coarse_mode, slot) in raw[mode].iter_mut().enumerate() {
                let local_dof = 3 * coarse_local_face + coarse_mode;
                let value = [
                    coarse_tabulation.vector_value(local_dof, 0),
                    coarse_tabulation.vector_value(local_dof, 1),
                    coarse_tabulation.vector_value(local_dof, 2),
                ];
                *slot += measure * vector_dot(value, fine_normal);
            }
        }
    }

    let inverse = matrix_inverse_3(fine_functionals, tolerance)?;
    let coefficients: [[f64; 3]; 3] = std::array::from_fn(|fine_mode| {
        std::array::from_fn(|coarse_mode| {
            (0..3)
                .map(|moment| inverse[fine_mode][moment] * raw[moment][coarse_mode])
                .sum()
        })
    });
    coefficients
        .iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(coefficients)
}

fn triangle_area_vector(mesh: &Mesh, nodes: [usize; 3]) -> [f64; 3] {
    let first = mesh.nodes[nodes[0]];
    let second = mesh.nodes[nodes[1]];
    let third = mesh.nodes[nodes[2]];
    let first_edge = [second.x - first.x, second.y - first.y, second.z - first.z];
    let second_edge = [third.x - first.x, third.y - first.y, third.z - first.z];
    [
        first_edge[1] * second_edge[2] - first_edge[2] * second_edge[1],
        first_edge[2] * second_edge[0] - first_edge[0] * second_edge[2],
        first_edge[0] * second_edge[1] - first_edge[1] * second_edge[0],
    ]
}

fn vector_dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn tangential_vector_dot(left: [f64; 3], right: [f64; 3], unit_normal: [f64; 3]) -> f64 {
    let left_normal = vector_dot(left, unit_normal);
    let right_normal = vector_dot(right, unit_normal);
    vector_dot(
        [
            left[0] - left_normal * unit_normal[0],
            left[1] - left_normal * unit_normal[1],
            left[2] - left_normal * unit_normal[2],
        ],
        [
            right[0] - right_normal * unit_normal[0],
            right[1] - right_normal * unit_normal[1],
            right[2] - right_normal * unit_normal[2],
        ],
    )
}

fn vector_norm(vector: [f64; 3]) -> f64 {
    vector_dot(vector, vector).sqrt()
}

fn triangle_contains_face(
    mesh: &Mesh,
    coarse: [usize; 3],
    fine: [usize; 3],
    tolerance: f64,
) -> bool {
    let origin = mesh.nodes[coarse[0]];
    let first = mesh.nodes[coarse[1]];
    let second = mesh.nodes[coarse[2]];
    let basis_first = [first.x - origin.x, first.y - origin.y, first.z - origin.z];
    let basis_second = [
        second.x - origin.x,
        second.y - origin.y,
        second.z - origin.z,
    ];
    let normal = [
        basis_first[1] * basis_second[2] - basis_first[2] * basis_second[1],
        basis_first[2] * basis_second[0] - basis_first[0] * basis_second[2],
        basis_first[0] * basis_second[1] - basis_first[1] * basis_second[0],
    ];
    let normal_squared = vector_dot(normal, normal);
    if !normal_squared.is_finite() || normal_squared <= tolerance * tolerance {
        return false;
    }
    let basis_first_squared = vector_dot(basis_first, basis_first);
    let basis_second_squared = vector_dot(basis_second, basis_second);
    let basis_dot = vector_dot(basis_first, basis_second);
    let denominator = basis_first_squared * basis_second_squared - basis_dot * basis_dot;
    if !denominator.is_finite() || denominator <= tolerance * tolerance {
        return false;
    }
    let parameter_tolerance = tolerance
        / basis_first_squared
            .max(basis_second_squared)
            .sqrt()
            .max(tolerance);

    fine.iter().all(|&node| {
        let point = mesh.nodes[node];
        let offset = [point.x - origin.x, point.y - origin.y, point.z - origin.z];
        let plane_distance = vector_dot(offset, normal).abs() / normal_squared.sqrt();
        if !plane_distance.is_finite() || plane_distance > tolerance {
            return false;
        }
        let first_parameter = (basis_second_squared * vector_dot(offset, basis_first)
            - basis_dot * vector_dot(offset, basis_second))
            / denominator;
        let second_parameter = (basis_first_squared * vector_dot(offset, basis_second)
            - basis_dot * vector_dot(offset, basis_first))
            / denominator;
        let third_parameter = 1.0 - first_parameter - second_parameter;
        first_parameter >= -parameter_tolerance
            && second_parameter >= -parameter_tolerance
            && third_parameter >= -parameter_tolerance
            && first_parameter <= 1.0 + parameter_tolerance
            && second_parameter <= 1.0 + parameter_tolerance
            && third_parameter <= 1.0 + parameter_tolerance
    })
}

fn edge_parameter(
    mesh: &Mesh,
    coarse_first: usize,
    coarse_second: usize,
    point: usize,
    tolerance: f64,
) -> Option<f64> {
    let first = mesh.nodes[coarse_first];
    let second = mesh.nodes[coarse_second];
    let candidate = mesh.nodes[point];
    let direction = [second.x - first.x, second.y - first.y];
    let offset = [candidate.x - first.x, candidate.y - first.y];
    let length_squared = direction[0] * direction[0] + direction[1] * direction[1];
    if !length_squared.is_finite() || length_squared <= tolerance * tolerance {
        return None;
    }
    let parameter = (offset[0] * direction[0] + offset[1] * direction[1]) / length_squared;
    let cross = offset[0] * direction[1] - offset[1] * direction[0];
    let distance = cross.abs() / length_squared.sqrt();
    let margin = tolerance / length_squared.sqrt().max(tolerance);
    (parameter.is_finite()
        && distance <= tolerance
        && parameter >= -margin
        && parameter <= 1.0 + margin)
        .then_some(parameter.clamp(0.0, 1.0))
}

fn edge_parameter_3d(
    mesh: &Mesh,
    coarse_first: usize,
    coarse_second: usize,
    point: usize,
    tolerance: f64,
) -> Option<f64> {
    let first = mesh.nodes[coarse_first];
    let second = mesh.nodes[coarse_second];
    let candidate = mesh.nodes[point];
    let direction = [second.x - first.x, second.y - first.y, second.z - first.z];
    let offset = [
        candidate.x - first.x,
        candidate.y - first.y,
        candidate.z - first.z,
    ];
    let length_squared = vector_dot(direction, direction);
    if !length_squared.is_finite() || length_squared <= tolerance * tolerance {
        return None;
    }
    let parameter = vector_dot(offset, direction) / length_squared;
    let distance = vector_norm(vector_cross(offset, direction)) / length_squared.sqrt();
    let margin = tolerance / length_squared.sqrt().max(tolerance);
    (parameter.is_finite()
        && distance.is_finite()
        && distance <= tolerance
        && parameter >= -margin
        && parameter <= 1.0 + margin)
        .then_some(parameter.clamp(0.0, 1.0))
}

fn hcurl_order2_face_edge_restriction(
    mesh: &Mesh,
    space: &HcurlSpace,
    coarse_element: usize,
    coarse_local_dofs: [usize; 8],
    fine_edge: (usize, usize),
    _tolerance: f64,
) -> Option<[[f64; 8]; 2]> {
    let (origin, jacobian, determinant) = tetra_affine_geometry(mesh, coarse_element)?;
    let first = mesh.nodes[fine_edge.0];
    let second = mesh.nodes[fine_edge.1];
    let direction = [second.x - first.x, second.y - first.y, second.z - first.z];
    let quadrature_offset = 0.5 / 3.0_f64.sqrt();
    let mut coefficients = [[0.0; 8]; 2];
    for (parameter, weight) in [
        (0.5 - quadrature_offset, 0.5),
        (0.5 + quadrature_offset, 0.5),
    ] {
        let physical = [
            first.x + parameter * direction[0],
            first.y + parameter * direction[1],
            first.z + parameter * direction[2],
        ];
        let reference = reference_from_physical(origin, jacobian, determinant, physical)?;
        let tabulation = space
            .try_tabulate(
                mesh,
                coarse_element,
                reference[0],
                reference[1],
                reference[2],
            )
            .ok()?;
        for (mode, &local_dof) in coarse_local_dofs.iter().enumerate() {
            let value = [
                tabulation.vector_value(local_dof, 0),
                tabulation.vector_value(local_dof, 1),
                tabulation.vector_value(local_dof, 2),
            ];
            let line_value = vector_dot(value, direction);
            coefficients[0][mode] += weight * line_value;
            coefficients[1][mode] += weight * 3.0 * line_value * (1.0 - 2.0 * parameter);
        }
    }
    coefficients
        .iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(coefficients)
}

#[allow(clippy::excessive_precision)]
const HCURL_FACE_RULE: [(f64, f64, f64); 7] = [
    (1.0 / 3.0, 1.0 / 3.0, 0.1125),
    (
        0.47014206410511505,
        0.47014206410511505,
        0.06619707639425309,
    ),
    (0.47014206410511505, 0.0597158717897699, 0.06619707639425309),
    (0.0597158717897699, 0.47014206410511505, 0.06619707639425309),
    (
        0.10128650732345634,
        0.10128650732345634,
        0.06296959027241358,
    ),
    (0.10128650732345634, 0.7974269853530872, 0.06296959027241358),
    (0.7974269853530872, 0.10128650732345634, 0.06296959027241358),
];

#[allow(clippy::too_many_arguments)]
fn hcurl_order2_face_interior_restriction(
    mesh: &Mesh,
    space: &HcurlSpace,
    topology: &MeshTopology,
    coarse_element: usize,
    coarse_local_dofs: [usize; 8],
    fine_element: usize,
    fine_local_face: usize,
    fine_face: usize,
    tolerance: f64,
) -> Option<[[f64; 8]; 2]> {
    let (coarse_origin, coarse_jacobian, coarse_determinant) =
        tetra_affine_geometry(mesh, coarse_element)?;
    let (fine_origin, fine_jacobian, _) = tetra_affine_geometry(mesh, fine_element)?;
    let (fine_area_vector, fine_surface_jacobian) =
        rt1_face_surface_geometry(fine_origin, fine_jacobian, fine_local_face)?;
    let fine_unit_normal = fine_area_vector;
    let fine_nodes = match topology.face(fine_face) {
        Face::Triangle(nodes) => *nodes,
        Face::Quad(_) => return None,
    };
    let fine_edges = face_edge_indices(topology, fine_nodes)?;
    let fine_local_edges = topology.element_edge_indices(fine_element);
    let fine_edge_local_dofs = fine_edges.map(|edge| {
        let local_edge = fine_local_edges
            .iter()
            .position(|&candidate| candidate == edge)?;
        Some([2 * local_edge, 2 * local_edge + 1])
    });
    if fine_edge_local_dofs.iter().any(Option::is_none) {
        return None;
    }
    let fine_edge_local_dofs = fine_edge_local_dofs.map(Option::unwrap);
    let mut edge_coefficients = [[[0.0; 8]; 2]; 3];
    for (index, &edge) in fine_edges.iter().enumerate() {
        edge_coefficients[index] = hcurl_order2_face_edge_restriction(
            mesh,
            space,
            coarse_element,
            coarse_local_dofs,
            topology.edges[edge],
            tolerance,
        )?;
    }

    let fine_bubble_dofs = [12 + 2 * fine_local_face, 12 + 2 * fine_local_face + 1];
    let mut gram = [[0.0; 2]; 2];
    let mut raw = [[0.0; 8]; 2];
    for &(a, b, weight) in &HCURL_FACE_RULE {
        let fine_reference = rt1_face_reference_point(fine_local_face, a, b);
        let physical = physical_from_reference(fine_origin, fine_jacobian, fine_reference);
        let coarse_reference =
            reference_from_physical(coarse_origin, coarse_jacobian, coarse_determinant, physical)?;
        let fine_tabulation = space
            .try_tabulate(
                mesh,
                fine_element,
                fine_reference[0],
                fine_reference[1],
                fine_reference[2],
            )
            .ok()?;
        let coarse_tabulation = space
            .try_tabulate(
                mesh,
                coarse_element,
                coarse_reference[0],
                coarse_reference[1],
                coarse_reference[2],
            )
            .ok()?;
        let fine_bubbles = fine_bubble_dofs.map(|local_dof| {
            [
                fine_tabulation.vector_value(local_dof, 0),
                fine_tabulation.vector_value(local_dof, 1),
                fine_tabulation.vector_value(local_dof, 2),
            ]
        });
        let measure = weight * fine_surface_jacobian;
        for moment in 0..2 {
            for fine_mode in 0..2 {
                gram[moment][fine_mode] += measure
                    * tangential_vector_dot(
                        fine_bubbles[moment],
                        fine_bubbles[fine_mode],
                        fine_unit_normal,
                    );
            }
        }
        for coarse_mode in 0..8 {
            let mut residual = [
                coarse_tabulation.vector_value(coarse_local_dofs[coarse_mode], 0),
                coarse_tabulation.vector_value(coarse_local_dofs[coarse_mode], 1),
                coarse_tabulation.vector_value(coarse_local_dofs[coarse_mode], 2),
            ];
            for edge_index in 0..3 {
                for edge_mode in 0..2 {
                    let coefficient = edge_coefficients[edge_index][edge_mode][coarse_mode];
                    let local_dof = fine_edge_local_dofs[edge_index][edge_mode];
                    for (component, value) in residual.iter_mut().enumerate() {
                        *value -= coefficient * fine_tabulation.vector_value(local_dof, component);
                    }
                }
            }
            for moment in 0..2 {
                raw[moment][coarse_mode] += measure
                    * tangential_vector_dot(residual, fine_bubbles[moment], fine_unit_normal);
            }
        }
    }

    let inverse = matrix_inverse_2(gram, tolerance)?;
    let coefficients: [[f64; 8]; 2] = std::array::from_fn(|fine_mode| {
        std::array::from_fn(|coarse_mode| {
            (0..2)
                .map(|moment| inverse[fine_mode][moment] * raw[moment][coarse_mode])
                .sum()
        })
    });
    coefficients
        .iter()
        .flatten()
        .all(|value| value.is_finite())
        .then_some(coefficients)
}

fn face_edge_indices(topology: &MeshTopology, nodes: [usize; 3]) -> Option<[usize; 3]> {
    let pairs = [
        (nodes[0], nodes[1]),
        (nodes[1], nodes[2]),
        (nodes[2], nodes[0]),
    ];
    let mut edges = [0usize; 3];
    for (index, (first, second)) in pairs.into_iter().enumerate() {
        let key = if first < second {
            (first, second)
        } else {
            (second, first)
        };
        edges[index] = topology.edges.iter().position(|&edge| edge == key)?;
    }
    edges.sort_unstable();
    Some(edges)
}

fn triangle_contains_edge(
    mesh: &Mesh,
    coarse: [usize; 3],
    first_node: usize,
    second_node: usize,
    tolerance: f64,
) -> bool {
    let origin = mesh.nodes[coarse[0]];
    let first = mesh.nodes[coarse[1]];
    let second = mesh.nodes[coarse[2]];
    let basis_first = [first.x - origin.x, first.y - origin.y, first.z - origin.z];
    let basis_second = [
        second.x - origin.x,
        second.y - origin.y,
        second.z - origin.z,
    ];
    let normal = vector_cross(basis_first, basis_second);
    let normal_squared = vector_dot(normal, normal);
    let basis_first_squared = vector_dot(basis_first, basis_first);
    let basis_second_squared = vector_dot(basis_second, basis_second);
    let basis_dot = vector_dot(basis_first, basis_second);
    let denominator = basis_first_squared * basis_second_squared - basis_dot * basis_dot;
    if !normal_squared.is_finite()
        || normal_squared <= tolerance * tolerance
        || !denominator.is_finite()
        || denominator <= tolerance * tolerance
    {
        return false;
    }
    let parameter_tolerance = tolerance
        / basis_first_squared
            .max(basis_second_squared)
            .sqrt()
            .max(tolerance);
    [first_node, second_node].into_iter().all(|node| {
        let point = mesh.nodes[node];
        let offset = [point.x - origin.x, point.y - origin.y, point.z - origin.z];
        let plane_distance = vector_dot(offset, normal).abs() / normal_squared.sqrt();
        if !plane_distance.is_finite() || plane_distance > tolerance {
            return false;
        }
        let first_parameter = (basis_second_squared * vector_dot(offset, basis_first)
            - basis_dot * vector_dot(offset, basis_second))
            / denominator;
        let second_parameter = (basis_first_squared * vector_dot(offset, basis_second)
            - basis_dot * vector_dot(offset, basis_first))
            / denominator;
        let third_parameter = 1.0 - first_parameter - second_parameter;
        first_parameter >= -parameter_tolerance
            && second_parameter >= -parameter_tolerance
            && third_parameter >= -parameter_tolerance
            && first_parameter <= 1.0 + parameter_tolerance
            && second_parameter <= 1.0 + parameter_tolerance
            && third_parameter <= 1.0 + parameter_tolerance
    })
}

fn hcurl_order1_face_edge_restriction(
    mesh: &Mesh,
    space: &HcurlSpace,
    topology: &MeshTopology,
    coarse_face: usize,
    coarse_edges: [usize; 3],
    fine_edge: (usize, usize),
    _tolerance: f64,
) -> Option<[f64; 3]> {
    let (coarse_element, _) = rt1_face_owner(topology, coarse_face)?;
    let (origin, jacobian, determinant) = tetra_affine_geometry(mesh, coarse_element)?;
    let local_edges = topology.element_edge_indices(coarse_element);
    let local_dofs =
        coarse_edges.map(|edge| local_edges.iter().position(|&candidate| candidate == edge));
    if local_dofs.iter().any(Option::is_none) {
        return None;
    }
    let local_dofs = local_dofs.map(Option::unwrap);
    let first = mesh.nodes[fine_edge.0];
    let second = mesh.nodes[fine_edge.1];
    let direction = [second.x - first.x, second.y - first.y, second.z - first.z];
    let quadrature_offset = 0.5 / 3.0_f64.sqrt();
    let mut coefficients = [0.0; 3];
    for (parameter, weight) in [
        (0.5 - quadrature_offset, 0.5),
        (0.5 + quadrature_offset, 0.5),
    ] {
        let physical = [
            first.x + parameter * direction[0],
            first.y + parameter * direction[1],
            first.z + parameter * direction[2],
        ];
        let reference = reference_from_physical(origin, jacobian, determinant, physical)?;
        let tabulation = space
            .try_tabulate(
                mesh,
                coarse_element,
                reference[0],
                reference[1],
                reference[2],
            )
            .ok()?;
        for (mode, &local_dof) in local_dofs.iter().enumerate() {
            let value = [
                tabulation.vector_value(local_dof, 0),
                tabulation.vector_value(local_dof, 1),
                tabulation.vector_value(local_dof, 2),
            ];
            coefficients[mode] += weight * vector_dot(value, direction);
        }
    }
    coefficients
        .iter()
        .all(|value| value.is_finite())
        .then_some(coefficients)
}

fn edge_length(mesh: &Mesh, first: usize, second: usize) -> f64 {
    let first = mesh.nodes[first];
    let second = mesh.nodes[second];
    ((second.x - first.x).powi(2) + (second.y - first.y).powi(2)).sqrt()
}

fn edge_length_3d(mesh: &Mesh, first: usize, second: usize) -> f64 {
    let first = mesh.nodes[first];
    let second = mesh.nodes[second];
    ((second.x - first.x).powi(2) + (second.y - first.y).powi(2) + (second.z - first.z).powi(2))
        .sqrt()
}
