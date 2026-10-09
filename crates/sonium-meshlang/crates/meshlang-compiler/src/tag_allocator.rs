//! Gmsh entity tag allocator.
//!
//! Gmsh identifies everything by integer tags. This module maps MeshLang
//! named entities to auto-allocated tags, and emits a `.mlg.map` file
//! for round-tripping.
//!
//! For the `.geo` backend, tags are allocated at **Gmsh runtime** via
//! `newv`/`news`/`newc`/`newp` to avoid collisions with OCC-internal tags.
//! Each entity is stored as a `.geo` variable (e.g. `__v_1`) whose value
//! is determined at runtime.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use meshlang_ast::{CanonicalGeometryIr, SourceReference};

/// Gmsh entity dimension.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GmshDim {
    Point = 0,
    Curve = 1,
    Surface = 2,
    Volume = 3,
}

impl GmshDim {
    /// The Gmsh `.geo` keyword for this dimension in commands.
    pub fn keyword(self) -> &'static str {
        match self {
            GmshDim::Point => "Point",
            GmshDim::Curve => "Curve",
            GmshDim::Surface => "Surface",
            GmshDim::Volume => "Volume",
        }
    }

    /// The Gmsh `.geo` `new*` function for runtime tag allocation.
    pub fn new_tag_fn(self) -> &'static str {
        match self {
            GmshDim::Point => "newp",
            GmshDim::Curve => "newc",
            GmshDim::Surface => "news",
            GmshDim::Volume => "newv",
        }
    }
}

/// A Gmsh entity reference: dimension + tag.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct GmshEntity {
    pub dim: GmshDim,
    pub tag: i32,
}

/// A physical group in Gmsh.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GmshPhysicalGroup {
    pub dim: GmshDim,
    pub tag: i32,
    pub name: String,
    pub entities: Vec<i32>,
    /// Source locations that declared or transformed this group. Kept as an
    /// additive field so older `.mlg.map` readers can continue to ignore it.
    #[serde(default)]
    pub provenance: Vec<SourceReference>,
}

/// Bidirectional mapping between MeshLang names and Gmsh tags.
///
/// In the `.geo` backend, each entity also gets a `.geo` variable name
/// (the `geo_var` field in `GeoEntity`) that holds the runtime tag.
#[derive(Debug, Default)]
pub struct TagAllocator {
    next_tag: [i32; 4], // one counter per dimension (for compile-time IDs)
    /// MeshLang dotted path → Gmsh entity
    name_to_entity: HashMap<String, GmshEntity>,
    /// Canonical-IR assignments installed before backend emission.
    planned_entities: HashMap<String, GmshEntity>,
    /// Gmsh entity → MeshLang name
    entity_to_name: HashMap<GmshEntity, String>,
    /// MeshLang name → .geo variable name for runtime tag reference
    name_to_geo_var: HashMap<String, String>,
    /// Physical groups
    physical_groups: Vec<GmshPhysicalGroup>,
    next_physical_tag: i32,
}

impl TagAllocator {
    pub fn new() -> Self {
        Self {
            next_tag: [1, 1, 1, 1],
            name_to_entity: HashMap::new(),
            planned_entities: HashMap::new(),
            entity_to_name: HashMap::new(),
            name_to_geo_var: HashMap::new(),
            physical_groups: Vec::new(),
            next_physical_tag: 1,
        }
    }

    /// Allocate a new entity tag for the given dimension and MeshLang name.
    /// Returns a compile-time ID (sequential) used for internal tracking.
    /// Also generates a sanitized `.geo` variable name.
    pub fn alloc(&mut self, dim: GmshDim, name: &str) -> GmshEntity {
        // A canonical-IR plan can reserve the stable ID before backend
        // emission. Reusing it here keeps `.mlg.map` IDs independent from
        // backend-only temporary entities.
        if let Some(entity) = self.planned_entities.get(name).copied() {
            if entity.dim == dim {
                self.name_to_entity.insert(name.to_string(), entity);
                self.entity_to_name.insert(entity, name.to_string());
                self.name_to_geo_var
                    .insert(name.to_string(), sanitize_geo_var(name));
                return entity;
            }
        }
        if let Some(entity) = self.name_to_entity.get(name).copied() {
            if entity.dim == dim {
                return entity;
            }
        }
        let idx = dim as usize;
        let tag = self.next_tag[idx];
        self.next_tag[idx] += 1;

        let entity = GmshEntity { dim, tag };
        self.name_to_entity.insert(name.to_string(), entity);
        self.entity_to_name.insert(entity, name.to_string());

        // Generate a .geo variable name: replace dots with underscores
        let geo_var = sanitize_geo_var(name);
        self.name_to_geo_var.insert(name.to_string(), geo_var);
        entity
    }

    /// Reserve the exact dimension-qualified ID assigned by canonical IR.
    ///
    /// Plans are installed before emission, so subsequently allocated
    /// backend temporaries start after every public entity ID.
    pub fn reserve(&mut self, dim: GmshDim, name: &str, tag: i32) -> Result<(), String> {
        if tag <= 0 {
            return Err(format!(
                "canonical entity '{}' has non-positive ID {tag}",
                name
            ));
        }
        let entity = GmshEntity { dim, tag };
        if let Some(existing) = self.planned_entities.get(name) {
            if *existing != entity {
                return Err(format!(
                    "canonical entity '{}' is planned as both {:?} and {:?}",
                    name, existing, entity
                ));
            }
            return Ok(());
        }
        if let Some(existing_name) = self
            .planned_entities
            .iter()
            .find_map(|(planned_name, planned)| (*planned == entity).then_some(planned_name))
        {
            return Err(format!(
                "canonical entity ({:?}, {}) is assigned to both '{}' and '{}'",
                dim, tag, existing_name, name
            ));
        }

        self.planned_entities.insert(name.to_owned(), entity);
        let next = &mut self.next_tag[dim as usize];
        *next = (*next).max(tag.saturating_add(1));
        Ok(())
    }

    /// Allocate a tag without a name (for intermediate geometry).
    pub fn alloc_anonymous(&mut self, dim: GmshDim) -> GmshEntity {
        let idx = dim as usize;
        let tag = self.next_tag[idx];
        self.next_tag[idx] += 1;
        GmshEntity { dim, tag }
    }

    /// Look up a Gmsh entity by its MeshLang path.
    pub fn lookup(&self, name: &str) -> Option<GmshEntity> {
        self.name_to_entity.get(name).copied()
    }

    /// Look up the .geo variable name for a MeshLang entity.
    pub fn geo_var(&self, name: &str) -> Option<&str> {
        self.name_to_geo_var.get(name).map(|s| s.as_str())
    }

    /// Look up a MeshLang name by Gmsh entity.
    pub fn reverse_lookup(&self, entity: GmshEntity) -> Option<&str> {
        self.entity_to_name.get(&entity).map(|s| s.as_str())
    }

    /// Reassign a name to point to a different tag (e.g., after copy or fragment).
    pub fn reassign(&mut self, name: &str, new_tag: i32, dim: GmshDim) {
        // Remove stale reverse-lookup for the old entity this name pointed to
        if let Some(old_entity) = self.name_to_entity.get(name).copied() {
            self.entity_to_name.remove(&old_entity);
        }
        let entity = GmshEntity { dim, tag: new_tag };
        self.name_to_entity.insert(name.to_string(), entity);
        self.entity_to_name.insert(entity, name.to_string());
    }

    /// Reassign a name and also update its .geo variable reference.
    pub fn reassign_geo_var(&mut self, name: &str, geo_var: &str, dim: GmshDim) {
        let old_entity = self.name_to_entity.get(name).copied();
        let planned = self
            .planned_entities
            .get(name)
            .copied()
            .filter(|entity| entity.dim == dim);
        let entity = match (planned, old_entity) {
            (Some(entity), Some(old)) => {
                self.entity_to_name.remove(&old);
                entity
            }
            (Some(entity), None) => entity,
            // A runtime operation changed the concrete Gmsh tag, not the
            // identity of the MeshLang entity. Keep its stable compiler ID.
            (None, Some(entity)) if entity.dim == dim => entity,
            (None, Some(entity)) => {
                self.entity_to_name.remove(&entity);
                self.alloc_anonymous(dim)
            }
            (None, None) => self.alloc_anonymous(dim),
        };
        self.name_to_entity.insert(name.to_string(), entity);
        self.entity_to_name.insert(entity, name.to_string());
        self.name_to_geo_var
            .insert(name.to_string(), geo_var.to_string());
    }

    /// Validate the bidirectional entity map before serializing an artifact.
    pub fn validate_reverse_mappings(&self) -> Result<(), String> {
        if self.name_to_entity.len() != self.entity_to_name.len() {
            return Err(format!(
                "entity map cardinality mismatch: {} names, {} reverse entries",
                self.name_to_entity.len(),
                self.entity_to_name.len()
            ));
        }

        for (name, entity) in &self.name_to_entity {
            match self.entity_to_name.get(entity) {
                Some(reverse_name) if reverse_name == name => {}
                Some(reverse_name) => {
                    return Err(format!(
                        "entity ({:?}, {}) maps to both '{}' and '{}'",
                        entity.dim, entity.tag, name, reverse_name
                    ));
                }
                None => {
                    return Err(format!(
                        "entity ({:?}, {}) for '{}' has no reverse mapping",
                        entity.dim, entity.tag, name
                    ));
                }
            }
        }

        Ok(())
    }

    /// Register a physical group.
    pub fn add_physical_group(&mut self, dim: GmshDim, name: &str, entities: Vec<i32>) -> i32 {
        let tag = self.next_physical_tag;
        self.next_physical_tag += 1;
        self.physical_groups.push(GmshPhysicalGroup {
            dim,
            tag,
            name: name.to_string(),
            entities,
            provenance: Vec::new(),
        });
        tag
    }

    /// Get all physical groups.
    pub fn physical_groups(&self) -> &[GmshPhysicalGroup] {
        &self.physical_groups
    }

    /// Export the tag map as JSON (for the .mlg.map sidecar file).
    pub fn export_map(&self) -> TagMap {
        let mut entries: Vec<TagMapEntry> = self
            .name_to_entity
            .iter()
            .map(|(name, entity)| TagMapEntry {
                name: name.clone(),
                dim: entity.dim,
                tag: entity.tag,
                provenance: Vec::new(),
            })
            .collect();
        entries.sort_by(|a, b| {
            (a.dim as u8, a.tag, a.name.as_str()).cmp(&(b.dim as u8, b.tag, b.name.as_str()))
        });

        TagMap {
            version: "1.0".to_string(),
            entities: entries,
            physical_groups: self.physical_groups.clone(),
        }
    }

    /// Export the tag map with canonical source provenance attached to every
    /// public entity and physical group. Backend tags remain runtime details;
    /// the canonical IDs and source references make the sidecar useful for
    /// diagnostics and remeshing consumers.
    pub fn export_map_with_provenance(&self, ir: &CanonicalGeometryIr) -> TagMap {
        let mut map = self.export_map();
        for entry in &mut map.entities {
            if let Some(node) = ir.entities.iter().find(|node| node.name == entry.name) {
                entry.provenance = node.provenance.clone();
            }
        }
        for group in &mut map.physical_groups {
            if let Some(source) = ir
                .physical_groups
                .iter()
                .find(|candidate| candidate.name == group.name)
            {
                group.provenance = source.provenance.clone();
            }
        }
        map
    }
}

/// Convert a MeshLang dotted path to a valid Gmsh variable name.
/// Gmsh interprets dots as option categories, so we replace them.
fn sanitize_geo_var(name: &str) -> String {
    name.replace('.', "__")
}

/// Serializable tag map for the `.mlg.map` sidecar file.
#[derive(Debug, Serialize, Deserialize)]
pub struct TagMap {
    pub version: String,
    pub entities: Vec<TagMapEntry>,
    pub physical_groups: Vec<GmshPhysicalGroup>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct TagMapEntry {
    pub name: String,
    pub dim: GmshDim,
    pub tag: i32,
    #[serde(default)]
    pub provenance: Vec<SourceReference>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn gmsh_dim_serializes_as_name() {
        assert_eq!(
            serde_json::to_value(GmshDim::Surface).unwrap(),
            json!("Surface")
        );
    }

    #[test]
    fn exported_tag_map_uses_named_dimensions() {
        let mut tags = TagAllocator::new();
        tags.alloc(GmshDim::Volume, "cabinet");
        tags.add_physical_group(GmshDim::Surface, "walls", vec![1, 2]);

        let json = serde_json::to_value(tags.export_map()).unwrap();
        assert_eq!(json["entities"][0]["dim"], json!("Volume"));
        assert_eq!(json["physical_groups"][0]["dim"], json!("Surface"));
    }

    #[test]
    fn runtime_reassignment_preserves_unique_stable_ids() {
        let mut tags = TagAllocator::new();
        let first = tags.alloc(GmshDim::Volume, "first");
        let second = tags.alloc(GmshDim::Volume, "second");

        tags.reassign_geo_var("first", "first_result", GmshDim::Volume);
        tags.reassign_geo_var("second", "second_result", GmshDim::Volume);

        assert_eq!(tags.lookup("first"), Some(first));
        assert_eq!(tags.lookup("second"), Some(second));
        assert_ne!(tags.lookup("first"), tags.lookup("second"));
        assert!(tags
            .export_map()
            .entities
            .iter()
            .all(|entity| entity.tag > 0));
        tags.validate_reverse_mappings().unwrap();
    }

    #[test]
    fn runtime_dimension_change_allocates_a_new_dimension_qualified_id() {
        let mut tags = TagAllocator::new();
        let surface = tags.alloc(GmshDim::Surface, "panel");

        tags.reassign_geo_var("panel", "extrude_result", GmshDim::Volume);

        let volume = tags.lookup("panel").unwrap();
        assert_eq!(surface.dim, GmshDim::Surface);
        assert_eq!(volume.dim, GmshDim::Volume);
        assert!(volume.tag > 0);
        assert!(tags.reverse_lookup(surface).is_none());
        assert_eq!(tags.reverse_lookup(volume), Some("panel"));
        tags.validate_reverse_mappings().unwrap();
    }

    #[test]
    fn exported_tag_map_is_dimension_and_id_sorted() {
        let mut tags = TagAllocator::new();
        tags.alloc(GmshDim::Volume, "z_volume");
        tags.alloc(GmshDim::Surface, "surface");
        tags.alloc(GmshDim::Volume, "a_volume");

        let entries = tags.export_map().entities;
        let keys: Vec<_> = entries
            .iter()
            .map(|entry| (entry.dim as u8, entry.tag, entry.name.as_str()))
            .collect();
        assert_eq!(
            keys,
            vec![
                (GmshDim::Surface as u8, 1, "surface"),
                (GmshDim::Volume as u8, 1, "z_volume"),
                (GmshDim::Volume as u8, 2, "a_volume"),
            ]
        );
    }

    #[test]
    fn tag_map_provenance_is_additive_for_legacy_sidecars() {
        let legacy = r#"{
            "version": "1.0",
            "entities": [{"name": "body", "dim": "Volume", "tag": 1}],
            "physical_groups": []
        }"#;
        let map: TagMap = serde_json::from_str(legacy).unwrap();
        assert!(map.entities[0].provenance.is_empty());

        let source = meshlang_ast::SourceMetadata::for_source("body.mlg", "geometry body {}");
        let mut ir = meshlang_ast::CanonicalGeometryIr::new(source);
        ir.entities.push(meshlang_ast::GeometryNode {
            id: meshlang_ast::EntityId::new(meshlang_ast::EntityDimension::Volume, 1),
            name: "body".into(),
            operation: meshlang_ast::GeometryOperation::Primitive {
                kind: meshlang_ast::PrimitiveKind::Box,
            },
            inputs: Vec::new(),
            arguments: Vec::new(),
            provenance: vec![meshlang_ast::SourceReference {
                source: ir.source.identity.id,
                node: meshlang_ast::NodeId(7),
                span: meshlang_ast::span::Span::with_file(0, 1, 0),
            }],
        });
        let mut allocator = TagAllocator::new();
        allocator.alloc(GmshDim::Volume, "body");
        let map = allocator.export_map_with_provenance(&ir);
        assert_eq!(map.entities[0].provenance.len(), 1);
    }

    #[test]
    fn canonical_plan_survives_backend_dimension_change() {
        let mut tags = TagAllocator::new();
        tags.reserve(GmshDim::Volume, "room.extruded", 7).unwrap();

        // The backend temporarily creates the surface under the result name,
        // then replaces it with the swept volume.
        let surface = tags.alloc(GmshDim::Surface, "room.extruded");
        assert_ne!(surface.tag, 7);
        tags.reassign_geo_var("room.extruded", "extrude_result", GmshDim::Volume);

        assert_eq!(
            tags.lookup("room.extruded"),
            Some(GmshEntity {
                dim: GmshDim::Volume,
                tag: 7,
            })
        );
        tags.validate_reverse_mappings().unwrap();
    }
}
