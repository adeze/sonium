//! Topology tracker: resolves dot-access paths like `cabinet.shell.faces.top`
//! to concrete Gmsh entity tags after CSG operations.
//!
//! When a shape is created in Gmsh (e.g., `Box(1) = {0,0,0, 1,1,1}`), it has
//! known topology (6 faces, 12 edges, 8 vertices). After CSG booleans, Gmsh
//! renumbers entities. This module:
//!
//! 1. Records the canonical topology of each primitive shape
//! 2. Tracks how CSG operations modify the entity map
//! 3. Resolves named topology paths to tag sets
//!
//! Until we can query Gmsh at compile time (which requires running it), we use
//! a **deferred resolution** strategy: emit the topology queries as Gmsh API
//! calls that run at mesh-generation time.

use std::collections::HashMap;

/// Canonical face names for box primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoxFace {
    /// -Z face
    Bottom,
    /// +Z face
    Top,
    /// -Y face
    Back,
    /// +Y face
    Front,
    /// -X face
    Left,
    /// +X face
    Right,
}

impl BoxFace {
    pub fn all() -> &'static [BoxFace] {
        &[
            BoxFace::Bottom,
            BoxFace::Top,
            BoxFace::Back,
            BoxFace::Front,
            BoxFace::Left,
            BoxFace::Right,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            BoxFace::Bottom => "bottom",
            BoxFace::Top => "top",
            BoxFace::Back => "back",
            BoxFace::Front => "front",
            BoxFace::Left => "left",
            BoxFace::Right => "right",
        }
    }

    /// Gmsh OpenCASCADE Box face ordering (0-indexed surface offset from volume tag).
    /// OCC convention: bottom=1, top=2, front=3, back=4, left=5, right=6
    /// (relative to the volume tag, actual tags depend on creation order)
    pub fn occ_index(&self) -> usize {
        match self {
            BoxFace::Bottom => 0,
            BoxFace::Top => 1,
            BoxFace::Front => 2,
            BoxFace::Back => 3,
            BoxFace::Left => 4,
            BoxFace::Right => 5,
        }
    }

    pub fn from_name(name: &str) -> Option<BoxFace> {
        match name {
            "bottom" => Some(BoxFace::Bottom),
            "top" => Some(BoxFace::Top),
            "back" => Some(BoxFace::Back),
            "front" => Some(BoxFace::Front),
            "left" => Some(BoxFace::Left),
            "right" => Some(BoxFace::Right),
            _ => None,
        }
    }
}

/// Canonical face names for cylinder primitives.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum CylinderFace {
    Bottom,
    Top,
    Lateral,
}

impl CylinderFace {
    pub fn all() -> &'static [CylinderFace] {
        &[
            CylinderFace::Bottom,
            CylinderFace::Top,
            CylinderFace::Lateral,
        ]
    }

    pub fn name(&self) -> &'static str {
        match self {
            CylinderFace::Bottom => "bottom",
            CylinderFace::Top => "top",
            CylinderFace::Lateral => "lateral",
        }
    }

    pub fn from_name(name: &str) -> Option<CylinderFace> {
        match name {
            "bottom" => Some(CylinderFace::Bottom),
            "top" => Some(CylinderFace::Top),
            "lateral" => Some(CylinderFace::Lateral),
            _ => None,
        }
    }
}

/// Known topology for a shape entity.
#[derive(Debug, Clone)]
pub enum ShapeTopology {
    Box {
        volume_tag: i32,
        /// Mapping of canonical face name → surface tag.
        /// Populated at creation time for primitives, or deferred after CSG.
        face_tags: HashMap<BoxFace, i32>,
    },
    Cylinder {
        volume_tag: i32,
        face_tags: HashMap<CylinderFace, i32>,
    },
    Sphere {
        volume_tag: i32,
        surface_tag: i32,
    },
    /// After CSG, we don't know the exact face mapping anymore.
    /// We store the operation info so we can emit runtime queries.
    CsgResult {
        volume_tag: i32,
        /// The operands that went into this result.
        left_name: String,
        right_name: String,
        op: CsgOp,
    },
    /// Generic solid with unknown face topology (e.g., imported STEP).
    Unknown {
        volume_tag: i32,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum CsgOp {
    Difference,
    Union,
    Intersection,
}

impl ShapeTopology {
    pub fn volume_tag(&self) -> i32 {
        match self {
            ShapeTopology::Box { volume_tag, .. } => *volume_tag,
            ShapeTopology::Cylinder { volume_tag, .. } => *volume_tag,
            ShapeTopology::Sphere { volume_tag, .. } => *volume_tag,
            ShapeTopology::CsgResult { volume_tag, .. } => *volume_tag,
            ShapeTopology::Unknown { volume_tag } => *volume_tag,
        }
    }
}

/// The topology tracker maintains the shape topology for all named entities.
#[derive(Debug, Default)]
pub struct TopologyTracker {
    /// Maps qualified entity names to their topology info.
    shapes: HashMap<String, ShapeTopology>,
}

impl TopologyTracker {
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a newly created box primitive.
    /// OCC creates surfaces with tags volume_base_tag * 6 - 5 .. volume_base_tag * 6
    /// but in practice the tags are sequential from the global surface counter.
    /// We use deferred resolution for safety.
    pub fn register_box(&mut self, name: &str, volume_tag: i32) {
        self.shapes.insert(
            name.to_string(),
            ShapeTopology::Box {
                volume_tag,
                face_tags: HashMap::new(), // deferred
            },
        );
    }

    pub fn register_cylinder(&mut self, name: &str, volume_tag: i32) {
        self.shapes.insert(
            name.to_string(),
            ShapeTopology::Cylinder {
                volume_tag,
                face_tags: HashMap::new(),
            },
        );
    }

    pub fn register_sphere(&mut self, name: &str, volume_tag: i32) {
        self.shapes.insert(
            name.to_string(),
            ShapeTopology::Sphere {
                volume_tag,
                surface_tag: 0, // deferred
            },
        );
    }

    pub fn register_csg_result(
        &mut self,
        name: &str,
        volume_tag: i32,
        left_name: &str,
        right_name: &str,
        op: CsgOp,
    ) {
        self.shapes.insert(
            name.to_string(),
            ShapeTopology::CsgResult {
                volume_tag,
                left_name: left_name.to_string(),
                right_name: right_name.to_string(),
                op,
            },
        );
    }

    pub fn register_unknown(&mut self, name: &str, volume_tag: i32) {
        self.shapes
            .insert(name.to_string(), ShapeTopology::Unknown { volume_tag });
    }

    /// Preserve the known primitive/topology class when a pipeline aliases an
    /// existing entity before applying an in-place transform.
    pub fn register_alias(&mut self, name: &str, source: &str) {
        if let Some(shape) = self.shapes.get(source).cloned() {
            self.shapes.insert(name.to_string(), shape);
        }
    }

    pub fn lookup(&self, name: &str) -> Option<&ShapeTopology> {
        self.shapes.get(name)
    }

    /// Resolve a topology path like `cabinet.shell.faces.top` to a
    /// TopologyQuery that can be emitted as Gmsh API calls.
    pub fn resolve_path(&self, parts: &[&str]) -> Result<TopologyQuery, String> {
        if parts.is_empty() {
            return Err("empty topology path".into());
        }

        // Find the shape entity. Try progressively longer prefixes.
        // e.g., for ["cabinet", "shell", "faces", "top"],
        // try "cabinet.shell", then "cabinet"
        let mut shape_name = None;
        let mut remaining = parts;

        for i in (1..=parts.len()).rev() {
            let candidate = parts[..i].join(".");
            if self.shapes.contains_key(&candidate) {
                shape_name = Some(candidate);
                remaining = &parts[i..];
                break;
            }
        }

        let shape_name =
            shape_name.ok_or_else(|| format!("unknown entity '{}'", parts.join(".")))?;

        let shape = self.shapes.get(&shape_name).unwrap();

        if remaining.is_empty() {
            // Reference to the solid itself
            return Ok(TopologyQuery::Volume {
                tag: shape.volume_tag(),
            });
        }

        match remaining {
            ["faces"] => Ok(TopologyQuery::AllFaces {
                volume_tag: shape.volume_tag(),
            }),
            ["faces", face_name] => {
                // Try to resolve statically for known primitives
                match shape {
                    ShapeTopology::Box {
                        volume_tag,
                        face_tags,
                    } => {
                        if let Some(bf) = BoxFace::from_name(face_name) {
                            if let Some(&tag) = face_tags.get(&bf) {
                                Ok(TopologyQuery::Surface { tag })
                            } else {
                                // Deferred: emit runtime query
                                Ok(TopologyQuery::BoxFaceDeferred {
                                    volume_tag: *volume_tag,
                                    face: bf,
                                })
                            }
                        } else {
                            Err(format!("unknown box face '{}'", face_name))
                        }
                    }
                    ShapeTopology::Cylinder {
                        volume_tag,
                        face_tags,
                    } => {
                        if let Some(cf) = CylinderFace::from_name(face_name) {
                            if let Some(&tag) = face_tags.get(&cf) {
                                Ok(TopologyQuery::Surface { tag })
                            } else {
                                Ok(TopologyQuery::CylinderFaceDeferred {
                                    volume_tag: *volume_tag,
                                    face: cf,
                                })
                            }
                        } else {
                            Err(format!("unknown cylinder face '{}'", face_name))
                        }
                    }
                    ShapeTopology::Sphere {
                        volume_tag,
                        surface_tag,
                    } => {
                        if *face_name == "outer" || *face_name == "surface" {
                            if *surface_tag != 0 {
                                Ok(TopologyQuery::Surface { tag: *surface_tag })
                            } else {
                                Ok(TopologyQuery::AllFaces {
                                    volume_tag: *volume_tag,
                                })
                            }
                        } else {
                            Err(format!("sphere has no face '{}'", face_name))
                        }
                    }
                    _ => {
                        // CSG result or unknown: all face access is deferred
                        Ok(TopologyQuery::NamedFaceDeferred {
                            volume_tag: shape.volume_tag(),
                            face_name: face_name.to_string(),
                        })
                    }
                }
            }
            ["edges"] => Ok(TopologyQuery::AllEdges {
                volume_tag: shape.volume_tag(),
            }),
            ["vertices"] => Ok(TopologyQuery::AllVertices {
                volume_tag: shape.volume_tag(),
            }),
            _ => Err(format!(
                "unsupported topology access '.{}'",
                remaining.join(".")
            )),
        }
    }
}

/// Result of topology resolution — either a concrete tag or a deferred query.
#[derive(Debug, Clone)]
pub enum TopologyQuery {
    /// A known volume tag.
    Volume { tag: i32 },
    /// A known surface tag.
    Surface { tag: i32 },
    /// All faces of a volume (runtime: `gmsh.model.getBoundary([(3, tag)])`).
    AllFaces { volume_tag: i32 },
    /// All edges of a volume.
    AllEdges { volume_tag: i32 },
    /// All vertices of a volume.
    AllVertices { volume_tag: i32 },
    /// A specific box face, resolved at runtime via bounding box comparison.
    BoxFaceDeferred { volume_tag: i32, face: BoxFace },
    /// A specific cylinder face, resolved at runtime.
    CylinderFaceDeferred { volume_tag: i32, face: CylinderFace },
    /// A face on a CSG result or unknown shape, resolved by name at runtime.
    NamedFaceDeferred { volume_tag: i32, face_name: String },
    /// Set difference of two face sets.
    FaceSetDifference {
        left: Box<TopologyQuery>,
        right: Box<TopologyQuery>,
    },
    /// Set union of two face sets.
    FaceSetUnion {
        left: Box<TopologyQuery>,
        right: Box<TopologyQuery>,
    },
}

impl TopologyQuery {
    /// Emit this query as Gmsh .geo code.
    /// Returns a string that evaluates to a surface/entity list in .geo syntax.
    pub fn to_geo(&self) -> String {
        match self {
            TopologyQuery::Volume { tag } => format!("Volume{{{}}}", tag),
            TopologyQuery::Surface { tag } => format!("Surface{{{}}}", tag),
            TopologyQuery::AllFaces { volume_tag } => {
                format!("Boundary{{ Volume{{{}}}; }}", volume_tag)
            }
            TopologyQuery::AllEdges { volume_tag } => {
                format!("Boundary{{ Boundary{{ Volume{{{}}}; }}; }}", volume_tag)
            }
            TopologyQuery::AllVertices { volume_tag } => {
                format!(
                    "Boundary{{ Boundary{{ Boundary{{ Volume{{{}}}; }}; }}; }}",
                    volume_tag
                )
            }
            TopologyQuery::BoxFaceDeferred { volume_tag, face } => {
                // Use Gmsh's Boundary + bounding box to identify faces.
                // For .geo, we emit the face index comment and use Boundary.
                format!(
                    "Boundary{{ Volume{{{}}}; }} /* .{} */",
                    volume_tag,
                    face.name()
                )
            }
            TopologyQuery::CylinderFaceDeferred { volume_tag, face } => {
                format!(
                    "Boundary{{ Volume{{{}}}; }} /* .{} */",
                    volume_tag,
                    face.name()
                )
            }
            TopologyQuery::NamedFaceDeferred {
                volume_tag,
                face_name,
            } => {
                format!(
                    "Boundary{{ Volume{{{}}}; }} /* .{} */",
                    volume_tag, face_name
                )
            }
            TopologyQuery::FaceSetDifference { left, right } => {
                // .geo doesn't have set operations on entities directly,
                // so we emit a comment and the full boundary for now
                format!("/* {} - {} */", left.to_geo(), right.to_geo())
            }
            TopologyQuery::FaceSetUnion { left, right } => {
                format!("/* {} | {} */", left.to_geo(), right.to_geo())
            }
        }
    }

    /// Emit this query as Gmsh Python API code.
    /// Returns a Python expression string.
    pub fn to_python(&self) -> String {
        match self {
            TopologyQuery::Volume { tag } => format!("[(3, {})]", tag),
            TopologyQuery::Surface { tag } => format!("[(2, {})]", tag),
            TopologyQuery::AllFaces { volume_tag } => {
                format!(
                    "gmsh.model.getBoundary([(3, {})], oriented=False)",
                    volume_tag
                )
            }
            TopologyQuery::AllEdges { volume_tag } => {
                format!(
                    "gmsh.model.getBoundary(gmsh.model.getBoundary([(3, {})], oriented=False), oriented=False)",
                    volume_tag
                )
            }
            TopologyQuery::AllVertices { volume_tag } => {
                format!(
                    "gmsh.model.getBoundary(gmsh.model.getBoundary(gmsh.model.getBoundary([(3, {})], oriented=False), oriented=False), oriented=False)",
                    volume_tag
                )
            }
            TopologyQuery::BoxFaceDeferred { volume_tag, face } => {
                // Emit a helper that finds the face by its normal direction
                let (coord_idx, sign) = match face {
                    BoxFace::Bottom => (2, "min"),
                    BoxFace::Top => (2, "max"),
                    BoxFace::Back => (1, "min"),
                    BoxFace::Front => (1, "max"),
                    BoxFace::Left => (0, "min"),
                    BoxFace::Right => (0, "max"),
                };
                format!(
                    "mlg_find_box_face({}, {}, '{}')",
                    volume_tag, coord_idx, sign
                )
            }
            TopologyQuery::CylinderFaceDeferred { volume_tag, face } => {
                format!("mlg_find_cyl_face({}, '{}')", volume_tag, face.name())
            }
            TopologyQuery::NamedFaceDeferred {
                volume_tag,
                face_name,
            } => {
                format!("mlg_find_face({}, '{}')", volume_tag, face_name)
            }
            TopologyQuery::FaceSetDifference { left, right } => {
                format!("mlg_face_diff({}, {})", left.to_python(), right.to_python())
            }
            TopologyQuery::FaceSetUnion { left, right } => {
                format!("{} + {}", left.to_python(), right.to_python())
            }
        }
    }

    /// Emit the Python helper functions required by deferred topology queries.
    pub fn python_helpers() -> &'static str {
        r#"
def mlg_find_box_face(vol_tag, coord_idx, side):
    """Find a box face by its extremal coordinate."""
    faces = gmsh.model.getBoundary([(3, vol_tag)], oriented=False)
    result = []
    for dim, tag in faces:
        bb = gmsh.model.getBoundingBox(dim, tag)
        # bb = [xmin, ymin, zmin, xmax, ymax, zmax]
        lo = bb[coord_idx]
        hi = bb[coord_idx + 3]
        vol_bb = gmsh.model.getBoundingBox(3, vol_tag)
        vol_lo = vol_bb[coord_idx]
        vol_hi = vol_bb[coord_idx + 3]
        tol = (vol_hi - vol_lo) * 1e-6
        if side == 'min' and abs(hi - vol_lo) < tol:
            result.append((dim, tag))
        elif side == 'max' and abs(lo - vol_hi) < tol:
            result.append((dim, tag))
    return result

def mlg_find_cyl_face(vol_tag, face_name):
    """Find a cylinder face by name (top/bottom/lateral)."""
    faces = gmsh.model.getBoundary([(3, vol_tag)], oriented=False)
    if face_name == 'lateral':
        # Lateral face is the one with the largest area
        max_area = -1
        best = None
        for dim, tag in faces:
            mass = gmsh.model.occ.getMass(dim, tag)
            if mass > max_area:
                max_area = mass
                best = (dim, tag)
        return [best] if best else []
    else:
        # top/bottom: pick the cap axis from the cylinder's largest extent,
        # so rotated cylinders are not treated as Z-aligned.
        vol_bb = gmsh.model.getBoundingBox(3, vol_tag)
        extents = [vol_bb[i + 3] - vol_bb[i] for i in range(3)]
        axis_idx = max(range(3), key=lambda i: extents[i])
        target = vol_bb[axis_idx] if face_name == 'bottom' else vol_bb[axis_idx + 3]
        result = []
        for dim, tag in faces:
            bb = gmsh.model.getBoundingBox(dim, tag)
            center = (bb[axis_idx] + bb[axis_idx + 3]) / 2.0
            tol = max(extents[axis_idx] * 1e-6, 1e-12)
            if abs(center - target) < tol:
                result.append((dim, tag))
        return result

def mlg_face_diff(faces_a, faces_b):
    """Set difference of two face lists."""
    tags_b = set(t for _, t in faces_b)
    return [(d, t) for d, t in faces_a if t not in tags_b]
"#
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_resolve_box() {
        let mut tracker = TopologyTracker::new();
        tracker.register_box("cabinet.shell", 1);

        let q = tracker
            .resolve_path(&["cabinet", "shell", "faces"])
            .unwrap();
        match q {
            TopologyQuery::AllFaces { volume_tag } => assert_eq!(volume_tag, 1),
            _ => panic!("expected AllFaces"),
        }

        let q = tracker
            .resolve_path(&["cabinet", "shell", "faces", "top"])
            .unwrap();
        match q {
            TopologyQuery::BoxFaceDeferred { volume_tag, face } => {
                assert_eq!(volume_tag, 1);
                assert_eq!(face, BoxFace::Top);
            }
            _ => panic!("expected BoxFaceDeferred"),
        }
    }

    #[test]
    fn test_resolve_cylinder() {
        let mut tracker = TopologyTracker::new();
        tracker.register_cylinder("cabinet.port", 2);

        let q = tracker
            .resolve_path(&["cabinet", "port", "faces", "lateral"])
            .unwrap();
        match q {
            TopologyQuery::CylinderFaceDeferred { volume_tag, face } => {
                assert_eq!(volume_tag, 2);
                assert_eq!(face, CylinderFace::Lateral);
            }
            _ => panic!("expected CylinderFaceDeferred"),
        }
    }

    #[test]
    fn test_resolve_csg_result() {
        let mut tracker = TopologyTracker::new();
        tracker.register_csg_result(
            "cabinet.solid",
            3,
            "cabinet.shell",
            "cabinet.port",
            CsgOp::Difference,
        );

        let q = tracker
            .resolve_path(&["cabinet", "solid", "faces"])
            .unwrap();
        match q {
            TopologyQuery::AllFaces { volume_tag } => assert_eq!(volume_tag, 3),
            _ => panic!("expected AllFaces"),
        }
    }

    #[test]
    fn test_unknown_path() {
        let tracker = TopologyTracker::new();
        let result = tracker.resolve_path(&["nonexistent", "faces"]);
        assert!(result.is_err());
    }

    #[test]
    fn test_geo_output() {
        let q = TopologyQuery::AllFaces { volume_tag: 5 };
        assert_eq!(q.to_geo(), "Boundary{ Volume{5}; }");
    }

    #[test]
    fn test_python_output() {
        let q = TopologyQuery::AllFaces { volume_tag: 5 };
        assert_eq!(
            q.to_python(),
            "gmsh.model.getBoundary([(3, 5)], oriented=False)"
        );
    }
}
