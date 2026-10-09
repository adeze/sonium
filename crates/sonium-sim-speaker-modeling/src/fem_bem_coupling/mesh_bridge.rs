//! FEM boundary-surface → BEM mesh conversion.
//!
//! The exterior BEM model of a cabinet is built directly from the boundary
//! faces of the interior tetrahedral FEM mesh, so the BEM constant elements
//! align 1:1 with the FEM boundary faces on the aperture. Each BEM element
//! inherits the FEM boundary marker: aperture faces become prescribed-velocity
//! elements, everything else stays rigid.
//!
//! Face winding is oriented so the BEM element normal points out of the FEM
//! volume (into the exterior air), using the owning tetrahedron's apex node
//! as the inward reference. This matches the outward-normal convention of the
//! TBEM sphere generators, so the same `sign = 1.0` stabilized assembly used
//! by `sonium-sim-speaker` applies unchanged.

use num_complex::Complex64;
use serde::{Deserialize, Serialize};
use sonium_bem::core::types::{
    BoundaryCondition, Element as BemElement, ElementProperty, ElementType as BemElementType,
    Mesh as BemMesh,
};
use sonium_fem::mesh::Mesh as FemMesh;
use std::collections::{BTreeMap, HashMap};

use crate::product_acoustic_mesh::{
    ProductInteriorMesh, ProductInteriorMeshConfig, ProductInteriorMeshError,
    ProductInteriorPatchKind, build_body_fitted_sealed_product_interior_mesh,
    build_ported_product_interior_mesh, build_sealed_product_interior_mesh,
};
use crate::product_design::{ProductDesign, ProductDesignError};

/// Errors raised while bridging FEM boundary surfaces to a BEM mesh.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MeshBridgeError {
    #[error("no FEM boundary faces match markers {markers:?}")]
    EmptySurface { markers: Vec<i32> },
    #[error("FEM boundary face {face} has {nodes} nodes; only triangular faces are supported")]
    NonTriangularFace { face: usize, nodes: usize },
    #[error("FEM boundary face {face} references invalid owning element {element}")]
    InvalidOwningElement { face: usize, element: usize },
    #[error("FEM boundary face {face} has zero area")]
    DegenerateFace { face: usize },
}

/// Exterior BEM surface extracted from tagged FEM boundary faces.
#[derive(Debug, Clone)]
pub struct BemInterfaceMesh {
    /// BEM mesh with constant (Tri3) elements. Aperture elements carry a
    /// prescribed-velocity boundary condition (initially zero), all other
    /// elements are rigid.
    pub mesh: BemMesh,
    /// BEM element indices of the aperture faces (FEM `aperture_marker`).
    pub aperture_elements: Vec<usize>,
    /// BEM element index of every collected FEM boundary face, in the same
    /// order as [`fem_boundary_faces`].
    pub fem_boundary_to_bem: Vec<usize>,
    /// FEM boundary-face index for every BEM element (inverse map).
    pub bem_to_fem_boundary: Vec<usize>,
    /// FEM boundary-face indices that were collected, in collection order.
    fem_boundary_faces: Vec<usize>,
}

impl BemInterfaceMesh {
    /// FEM boundary-face indices that were collected, in collection order.
    pub fn fem_boundary_faces(&self) -> &[usize] {
        &self.fem_boundary_faces
    }

    /// Number of aperture elements (the DtN dimension `m`).
    pub fn num_aperture_elements(&self) -> usize {
        self.aperture_elements.len()
    }

    /// Total aperture area from the BEM element areas.
    pub fn aperture_area(&self) -> f64 {
        self.aperture_elements
            .iter()
            .map(|&idx| self.mesh.elements[idx].area)
            .sum()
    }
}

/// One typed FEM-to-BEM trace for a semantic product boundary patch.
///
/// Interior and exterior meshes are generated from the same FEM boundary
/// faces. Keeping the correspondence by patch prevents callers from
/// reconstructing traces from marker integers or element insertion order.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductInterfaceTrace {
    pub patch: ProductInteriorPatchKind,
    pub fem_boundary_faces: Vec<usize>,
    pub bem_elements: Vec<usize>,
    pub area_m2: f64,
}

/// Serialized topology and trace evidence for a canonical product coupling
/// mesh. The FEM/BEM coordinates remain owned by [`ProductCouplingMesh`]; this
/// record preserves the exact identity needed to audit or replay a coupling
/// handoff without serializing solver-owned frequency state.
pub const PRODUCT_COUPLING_MESH_EVIDENCE_SCHEMA_VERSION: u32 = 1;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProductCouplingMeshEvidence {
    pub schema_version: u32,
    pub geometry_fingerprint: String,
    pub interior_node_count: usize,
    pub interior_element_count: usize,
    pub interior_boundary_face_count: usize,
    pub exterior_node_count: usize,
    pub exterior_element_count: usize,
    pub aperture_element_count: usize,
    pub traces: Vec<ProductInterfaceTrace>,
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum ProductCouplingMeshEvidenceError {
    #[error("unsupported product coupling evidence schema {actual}, expected {expected}")]
    SchemaVersion { actual: u32, expected: u32 },
    #[error("product coupling evidence has an empty geometry fingerprint")]
    EmptyGeometryFingerprint,
    #[error("product coupling evidence contains non-finite or negative trace area")]
    InvalidTraceArea,
    #[error(
        "product coupling evidence trace {patch:?} references invalid FEM boundary face {face}"
    )]
    InvalidTraceFace {
        patch: ProductInteriorPatchKind,
        face: usize,
    },
    #[error(
        "product coupling evidence trace {patch:?} references invalid exterior BEM element {element}"
    )]
    InvalidTraceElement {
        patch: ProductInteriorPatchKind,
        element: usize,
    },
    #[error("product coupling evidence contains mismatched trace lengths for {patch:?}")]
    TraceLengthMismatch { patch: ProductInteriorPatchKind },
    #[error("product coupling evidence contains duplicate or unsorted trace {patch:?}")]
    TraceOrderMismatch { patch: ProductInteriorPatchKind },
    #[error("product coupling evidence geometry fingerprint does not match the mesh")]
    GeometryFingerprintMismatch,
    #[error("product coupling evidence {field} count mismatch: expected {expected}, got {actual}")]
    CountMismatch {
        field: &'static str,
        expected: usize,
        actual: usize,
    },
    #[error("product coupling evidence trace count mismatch: expected {expected}, got {actual}")]
    TraceCountMismatch { expected: usize, actual: usize },
    #[error("product coupling evidence trace {patch:?} does not match the mesh")]
    TraceMismatch { patch: ProductInteriorPatchKind },
}

impl ProductCouplingMeshEvidence {
    /// Validate schema-local invariants before comparing against a mesh.
    pub fn validate(&self) -> Result<(), ProductCouplingMeshEvidenceError> {
        if self.schema_version != PRODUCT_COUPLING_MESH_EVIDENCE_SCHEMA_VERSION {
            return Err(ProductCouplingMeshEvidenceError::SchemaVersion {
                actual: self.schema_version,
                expected: PRODUCT_COUPLING_MESH_EVIDENCE_SCHEMA_VERSION,
            });
        }
        if self.geometry_fingerprint.is_empty() {
            return Err(ProductCouplingMeshEvidenceError::EmptyGeometryFingerprint);
        }

        let mut previous = None;
        for trace in &self.traces {
            if !trace.area_m2.is_finite() || trace.area_m2 < 0.0 {
                return Err(ProductCouplingMeshEvidenceError::InvalidTraceArea);
            }
            if let Some(&face) = trace
                .fem_boundary_faces
                .iter()
                .find(|&&face| face >= self.interior_boundary_face_count)
            {
                return Err(ProductCouplingMeshEvidenceError::InvalidTraceFace {
                    patch: trace.patch,
                    face,
                });
            }
            if let Some(&element) = trace
                .bem_elements
                .iter()
                .find(|&&element| element >= self.exterior_element_count)
            {
                return Err(ProductCouplingMeshEvidenceError::InvalidTraceElement {
                    patch: trace.patch,
                    element,
                });
            }
            // Composite internal patches (the solid port tube and braces) are
            // intentionally absent from the exterior BEM surface, so they
            // retain FEM faces without a BEM mapping.
            if !trace.bem_elements.is_empty()
                && trace.fem_boundary_faces.len() != trace.bem_elements.len()
            {
                return Err(ProductCouplingMeshEvidenceError::TraceLengthMismatch {
                    patch: trace.patch,
                });
            }
            if previous.is_some_and(|patch| patch >= trace.patch) {
                return Err(ProductCouplingMeshEvidenceError::TraceOrderMismatch {
                    patch: trace.patch,
                });
            }
            previous = Some(trace.patch);
        }
        Ok(())
    }
}

/// Canonical body-fitted product coupling mesh.
///
/// This is the reusable M0/M4 artifact: one validated interior product mesh,
/// the exterior BEM surface extracted from that mesh, and typed patch traces.
/// It intentionally owns geometry/topology only; frequency-dependent
/// operators and driver states remain owned by the monolithic solver.
#[derive(Debug, Clone)]
pub struct ProductCouplingMesh {
    pub interior: ProductInteriorMesh,
    pub exterior: BemInterfaceMesh,
    pub traces: BTreeMap<ProductInteriorPatchKind, ProductInterfaceTrace>,
    pub geometry_fingerprint: String,
}

/// Errors raised while constructing the canonical product coupling mesh.
#[derive(Debug, thiserror::Error)]
pub enum ProductCouplingMeshError {
    #[error(transparent)]
    InvalidProduct(#[from] ProductDesignError),
    #[error(transparent)]
    Interior(#[from] ProductInteriorMeshError),
    #[error(transparent)]
    Bridge(#[from] MeshBridgeError),
    #[error("required product interface trace is empty for patch {patch:?}")]
    EmptyRequiredTrace { patch: ProductInteriorPatchKind },
}

impl ProductCouplingMesh {
    /// Build the canonical artifact using the product's resolved lowering.
    pub fn build(
        design: &ProductDesign,
        config: ProductInteriorMeshConfig,
    ) -> Result<Self, ProductCouplingMeshError> {
        design.validate()?;
        let interior = if design.waveguide.is_some() {
            build_body_fitted_sealed_product_interior_mesh(design, config)?
        } else if design.cabinet.port.is_some() {
            build_ported_product_interior_mesh(design, config)?
        } else if design.drivers.len() == 1 && !design.drivers[0].is_axis_aligned() {
            build_body_fitted_sealed_product_interior_mesh(design, config)?
        } else {
            build_sealed_product_interior_mesh(design, config)?
        };
        Self::from_interior(design, interior)
    }

    /// Attach an exterior BEM surface and typed traces to an existing
    /// validated interior product mesh.
    pub fn from_interior(
        design: &ProductDesign,
        interior: ProductInteriorMesh,
    ) -> Result<Self, ProductCouplingMeshError> {
        design.validate()?;
        let aperture_marker = interior
            .patch(ProductInteriorPatchKind::PortMouth)
            .map_or(i32::MIN, |patch| patch.marker);
        let wall_markers = interior
            .patches
            .iter()
            .filter(|patch| {
                !matches!(
                    patch.kind,
                    ProductInteriorPatchKind::PortMouth
                        | ProductInteriorPatchKind::PortWall
                        | ProductInteriorPatchKind::WaveguideWall { .. }
                        | ProductInteriorPatchKind::BraceWall { .. }
                )
            })
            .map(|patch| patch.marker)
            .collect::<Vec<_>>();
        let exterior = build_exterior_bem_mesh(&interior.mesh, aperture_marker, &wall_markers)?;

        let fem_to_bem = exterior
            .fem_boundary_faces()
            .iter()
            .copied()
            .zip(exterior.fem_boundary_to_bem.iter().copied())
            .collect::<HashMap<_, _>>();
        let mut traces = BTreeMap::new();
        for patch in &interior.patches {
            let bem_elements = patch
                .boundary_faces
                .iter()
                .filter_map(|face| fem_to_bem.get(face).copied())
                .collect::<Vec<_>>();
            let required = matches!(
                patch.kind,
                ProductInteriorPatchKind::DriverRear { .. } | ProductInteriorPatchKind::PortMouth
            );
            if required && bem_elements.is_empty() {
                return Err(ProductCouplingMeshError::EmptyRequiredTrace { patch: patch.kind });
            }
            traces.insert(
                patch.kind,
                ProductInterfaceTrace {
                    patch: patch.kind,
                    fem_boundary_faces: patch.boundary_faces.clone(),
                    bem_elements,
                    area_m2: patch.area_m2,
                },
            );
        }

        Ok(Self {
            geometry_fingerprint: format!("product-coupling-v1:{}", interior.geometry_fingerprint),
            interior,
            exterior,
            traces,
        })
    }

    /// Return the trace for a semantic patch, if that patch exists.
    pub fn trace(&self, patch: ProductInteriorPatchKind) -> Option<&ProductInterfaceTrace> {
        self.traces.get(&patch)
    }

    /// Create the versioned serialized evidence companion for this mesh.
    pub fn evidence(&self) -> ProductCouplingMeshEvidence {
        ProductCouplingMeshEvidence {
            schema_version: PRODUCT_COUPLING_MESH_EVIDENCE_SCHEMA_VERSION,
            geometry_fingerprint: self.geometry_fingerprint.clone(),
            interior_node_count: self.interior.mesh.nodes.len(),
            interior_element_count: self.interior.mesh.elements.len(),
            interior_boundary_face_count: self.interior.mesh.boundaries.len(),
            exterior_node_count: self.exterior.mesh.nodes.len(),
            exterior_element_count: self.exterior.mesh.elements.len(),
            aperture_element_count: self.exterior.aperture_elements.len(),
            traces: self.traces.values().cloned().collect(),
        }
    }

    /// Revalidate serialized evidence against the live FEM/BEM topology.
    ///
    /// Trace indices are compared exactly; only the derived floating-point
    /// patch area receives a small serialization round-trip tolerance.
    pub fn validate_evidence(
        &self,
        evidence: &ProductCouplingMeshEvidence,
    ) -> Result<(), ProductCouplingMeshEvidenceError> {
        evidence.validate()?;
        if evidence.geometry_fingerprint != self.geometry_fingerprint {
            return Err(ProductCouplingMeshEvidenceError::GeometryFingerprintMismatch);
        }

        let counts = [
            (
                "interior_node_count",
                self.interior.mesh.nodes.len(),
                evidence.interior_node_count,
            ),
            (
                "interior_element_count",
                self.interior.mesh.elements.len(),
                evidence.interior_element_count,
            ),
            (
                "interior_boundary_face_count",
                self.interior.mesh.boundaries.len(),
                evidence.interior_boundary_face_count,
            ),
            (
                "exterior_node_count",
                self.exterior.mesh.nodes.len(),
                evidence.exterior_node_count,
            ),
            (
                "exterior_element_count",
                self.exterior.mesh.elements.len(),
                evidence.exterior_element_count,
            ),
            (
                "aperture_element_count",
                self.exterior.aperture_elements.len(),
                evidence.aperture_element_count,
            ),
        ];
        for (field, expected, actual) in counts {
            if expected != actual {
                return Err(ProductCouplingMeshEvidenceError::CountMismatch {
                    field,
                    expected,
                    actual,
                });
            }
        }
        if self.traces.len() != evidence.traces.len() {
            return Err(ProductCouplingMeshEvidenceError::TraceCountMismatch {
                expected: self.traces.len(),
                actual: evidence.traces.len(),
            });
        }

        for (expected, actual) in self.traces.values().zip(&evidence.traces) {
            let area_scale = expected.area_m2.abs().max(actual.area_m2.abs()).max(1.0);
            let area_matches = (expected.area_m2 - actual.area_m2).abs() <= 1.0e-12 * area_scale;
            if expected.patch != actual.patch
                || expected.fem_boundary_faces != actual.fem_boundary_faces
                || expected.bem_elements != actual.bem_elements
                || !area_matches
            {
                return Err(ProductCouplingMeshEvidenceError::TraceMismatch {
                    patch: actual.patch,
                });
            }
        }
        Ok(())
    }
}

/// Build the canonical product coupling mesh.
pub fn build_product_coupling_mesh(
    design: &ProductDesign,
    config: ProductInteriorMeshConfig,
) -> Result<ProductCouplingMesh, ProductCouplingMeshError> {
    ProductCouplingMesh::build(design, config)
}

/// Build an exterior BEM mesh from all FEM boundary faces whose marker is the
/// aperture marker or one of `wall_markers`.
///
/// Every collected face becomes one constant BEM element; the element order
/// follows the FEM boundary-face order. Faces are re-wound when necessary so
/// BEM normals point out of the FEM volume.
pub fn build_exterior_bem_mesh(
    fem_mesh: &FemMesh,
    aperture_marker: i32,
    wall_markers: &[i32],
) -> Result<BemInterfaceMesh, MeshBridgeError> {
    let mut collected = Vec::new();
    for (face_idx, face) in fem_mesh.boundaries.iter().enumerate() {
        if face.marker == aperture_marker || wall_markers.contains(&face.marker) {
            collected.push(face_idx);
        }
    }
    if collected.is_empty() {
        let mut markers = wall_markers.to_vec();
        markers.push(aperture_marker);
        return Err(MeshBridgeError::EmptySurface { markers });
    }

    // Compact node set: only nodes referenced by the collected faces.
    let mut fem_to_bem_node: HashMap<usize, usize> = HashMap::new();
    let mut bem_nodes: Vec<[f64; 3]> = Vec::new();
    let mut elements = Vec::with_capacity(collected.len());
    let mut aperture_elements = Vec::new();
    let mut bem_to_fem_boundary = Vec::with_capacity(collected.len());

    for &face_idx in &collected {
        let face = &fem_mesh.boundaries[face_idx];
        if face.nodes.len() != 3 {
            return Err(MeshBridgeError::NonTriangularFace {
                face: face_idx,
                nodes: face.nodes.len(),
            });
        }

        let mut conn = [0usize; 3];
        for (local, &fem_node) in face.nodes.iter().enumerate() {
            let bem_node = *fem_to_bem_node.entry(fem_node).or_insert_with(|| {
                let p = &fem_mesh.nodes[fem_node];
                bem_nodes.push([p.x, p.y, p.z]);
                bem_nodes.len() - 1
            });
            conn[local] = bem_node;
        }

        // Orient the face so its winding normal points away from the owning
        // tetrahedron's apex (out of the FEM volume).
        let tet = fem_mesh.elements.get(face.element_idx).ok_or(
            MeshBridgeError::InvalidOwningElement {
                face: face_idx,
                element: face.element_idx,
            },
        )?;
        let apex = tet
            .nodes
            .iter()
            .find(|n| !face.nodes.contains(n))
            .copied()
            .ok_or(MeshBridgeError::InvalidOwningElement {
                face: face_idx,
                element: face.element_idx,
            })?;
        let apex_p = &fem_mesh.nodes[apex];
        let p0 = bem_nodes[conn[0]];
        let p1 = bem_nodes[conn[1]];
        let p2 = bem_nodes[conn[2]];
        let normal = triangle_normal(p0, p1, p2);
        let centroid = [
            (p0[0] + p1[0] + p2[0]) / 3.0,
            (p0[1] + p1[1] + p2[1]) / 3.0,
            (p0[2] + p1[2] + p2[2]) / 3.0,
        ];
        let outward = [
            centroid[0] - apex_p.x,
            centroid[1] - apex_p.y,
            centroid[2] - apex_p.z,
        ];
        let mut normal = normal;
        if normal[0] * outward[0] + normal[1] * outward[1] + normal[2] * outward[2] < 0.0 {
            conn.swap(1, 2);
            normal = [-normal[0], -normal[1], -normal[2]];
        }
        let norm_len =
            (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if norm_len < 1e-30 {
            return Err(MeshBridgeError::DegenerateFace { face: face_idx });
        }
        let unit_normal = [
            normal[0] / norm_len,
            normal[1] / norm_len,
            normal[2] / norm_len,
        ];
        let area = 0.5 * norm_len;

        let elem_index = elements.len();
        if face.marker == aperture_marker {
            aperture_elements.push(elem_index);
        }
        let zero = Complex64::new(0.0, 0.0);
        elements.push(BemElement {
            connectivity: conn.to_vec(),
            element_type: BemElementType::Tri3,
            property: ElementProperty::Surface,
            normal: unit_normal,
            node_normals: vec![unit_normal; 3],
            center: centroid,
            area,
            boundary_condition: BoundaryCondition::Velocity(vec![zero]),
            group: 0,
            dof_addresses: vec![elem_index],
        });
        bem_to_fem_boundary.push(face_idx);
    }

    let num_nodes = bem_nodes.len();
    let num_elements = elements.len();
    let mesh = BemMesh {
        nodes: bem_nodes,
        elements,
        external_node_numbers: (0..num_nodes as i32).collect(),
        external_element_numbers: (0..num_elements as i32).collect(),
        num_boundary_nodes: num_nodes,
        num_evaluation_nodes: 0,
        num_boundary_elements: num_elements,
        num_evaluation_elements: 0,
        symmetry_planes: [0, 0, 0],
        symmetry_coordinates: [0.0, 0.0, 0.0],
        num_reflections: 0,
        node_to_elements: Vec::new(),
    };

    Ok(BemInterfaceMesh {
        mesh,
        aperture_elements,
        fem_boundary_to_bem: (0..num_elements).collect(),
        bem_to_fem_boundary,
        fem_boundary_faces: collected,
    })
}

/// Unnormalized triangle normal `(p1 − p0) × (p2 − p0)`; its length is twice
/// the triangle area.
fn triangle_normal(p0: [f64; 3], p1: [f64; 3], p2: [f64; 3]) -> [f64; 3] {
    let v1 = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v2 = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    [
        v1[1] * v2[2] - v1[2] * v2[1],
        v1[2] * v2[0] - v1[0] * v2[2],
        v1[0] * v2[1] - v1[1] * v2[0],
    ]
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_fem::mesh::generators::unit_cube_tetrahedra;

    #[test]
    fn cube_boundary_converts_with_outward_normals() {
        let mesh = unit_cube_tetrahedra(1);
        let markers: Vec<i32> = mesh.boundaries.iter().map(|b| b.marker).collect();
        let wall_markers: Vec<i32> = {
            let mut m = markers.clone();
            m.sort_unstable();
            m.dedup();
            m
        };
        let bem = build_exterior_bem_mesh(&mesh, -999, &wall_markers).unwrap();
        assert_eq!(bem.mesh.elements.len(), mesh.boundaries.len());
        assert!(bem.aperture_elements.is_empty());

        // Outward check: normal must point away from the cube centre.
        let centre = [0.5, 0.5, 0.5];
        let mut total_area = 0.0;
        for elem in &bem.mesh.elements {
            let d = [
                elem.center[0] - centre[0],
                elem.center[1] - centre[1],
                elem.center[2] - centre[2],
            ];
            let dot = elem.normal[0] * d[0] + elem.normal[1] * d[1] + elem.normal[2] * d[2];
            assert!(dot > 0.0, "element normal points inward: {elem:?}");
            let n_len =
                (elem.normal[0].powi(2) + elem.normal[1].powi(2) + elem.normal[2].powi(2)).sqrt();
            assert!((n_len - 1.0).abs() < 1e-12);
            total_area += elem.area;
        }
        // Unit cube surface area = 6.
        assert!((total_area - 6.0).abs() < 1e-9, "area = {total_area}");
    }

    #[test]
    fn aperture_elements_are_tracked() {
        let mut mesh = unit_cube_tetrahedra(1);
        // Re-marker all z = 0 faces as aperture.
        for face in &mut mesh.boundaries {
            if face.nodes.iter().all(|&n| mesh.nodes[n].z.abs() < 1e-12) {
                face.marker = 42;
            }
        }
        let wall_markers = [0, 1, 2, 3, 4, 5, 6];
        let bem = build_exterior_bem_mesh(&mesh, 42, &wall_markers).unwrap();
        assert!(!bem.aperture_elements.is_empty());
        let aperture_area: f64 = bem
            .aperture_elements
            .iter()
            .map(|&i| bem.mesh.elements[i].area)
            .sum();
        assert!((aperture_area - 1.0).abs() < 1e-9, "area = {aperture_area}");
    }
}
