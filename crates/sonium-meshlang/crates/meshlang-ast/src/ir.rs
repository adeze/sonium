//! Canonical backend-neutral geometry IR.
//!
//! Surface syntax is lowered into this model before any Gmsh GEO/Python
//! emission. The IR contains resolved dimensions, stable IDs, and provenance;
//! it intentionally contains neither AST expressions nor backend snippets.

use crate::hash::{StructuralHash, StructuralHashError};
use crate::metadata::{EntityDimension, InequalityRelation, ObjectiveSense, ProjectionMethod};
use crate::provenance::{SourceIdentity, SourceMetadata, SourceReference};
use crate::units::Dimension;
use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

pub const CURRENT_IR_SCHEMA_VERSION: IrSchemaVersion = IrSchemaVersion(1);
/// Version of the persistence envelope around a canonical IR artifact.
///
/// This is deliberately separate from [`IrSchemaVersion`], which describes
/// the backend-neutral IR model itself. Keeping the two boundaries distinct
/// lets old raw IR payloads migrate into an explicit document envelope without
/// silently changing IR semantics.
pub const LEGACY_IR_DOCUMENT_VERSION: IrDocumentVersion = IrDocumentVersion(0);
pub const CURRENT_IR_DOCUMENT_VERSION: IrDocumentVersion = IrDocumentVersion(1);
pub const DEFAULT_MAX_SERIALIZED_IR_BYTES: usize = 16 * 1024 * 1024;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(transparent)]
pub struct IrSchemaVersion(pub u32);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct IrDocumentVersion(pub u32);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct SymbolId(pub u64);

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EntityId {
    pub dimension: EntityDimension,
    pub index: u64,
}

impl EntityId {
    pub const fn new(dimension: EntityDimension, index: u64) -> Self {
        Self { dimension, index }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[serde(transparent)]
pub struct PhysicalGroupId(pub u64);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalGeometryIr {
    pub schema_version: IrSchemaVersion,
    pub source: SourceMetadata,
    /// All source identities referenced by provenance records, including
    /// imported modules. The metadata identity is always retained first.
    #[serde(default)]
    pub source_identities: Vec<SourceIdentity>,
    pub parameters: Vec<IrParameter>,
    pub entities: Vec<GeometryNode>,
    /// Dynamic geometry control-flow is retained as an ordered, backend-
    /// neutral branch plan.  Native backends must preserve the first-match
    /// semantics represented here rather than selecting a branch while
    /// lowering.
    #[serde(default)]
    pub controls: Vec<IrConditionalControl>,
    /// Runtime-dynamic geometry loops in source order.
    #[serde(default)]
    pub loops: Vec<IrLoopControl>,
    pub physical_groups: Vec<PhysicalGroup>,
    pub boundaries: Vec<IrBoundary>,
    pub interfaces: Vec<IrInterface>,
    pub periodic_pairs: Vec<IrPeriodicPair>,
    pub coordinate_systems: Vec<IrCoordinateSystem>,
    pub material_regions: Vec<IrMaterialRegion>,
    pub fields: Vec<IrGeometryField>,
    pub mesh: IrMeshControls,
    pub optimization: IrOptimizationModel,
    pub uncertainties: Vec<IrUncertainty>,
    pub sensitivity: SensitivityManifest,
}

impl CanonicalGeometryIr {
    pub fn new(source: SourceMetadata) -> Self {
        let source_identity = source.identity.clone();
        Self {
            schema_version: CURRENT_IR_SCHEMA_VERSION,
            source,
            source_identities: vec![source_identity],
            parameters: Vec::new(),
            entities: Vec::new(),
            controls: Vec::new(),
            loops: Vec::new(),
            physical_groups: Vec::new(),
            boundaries: Vec::new(),
            interfaces: Vec::new(),
            periodic_pairs: Vec::new(),
            coordinate_systems: Vec::new(),
            material_regions: Vec::new(),
            fields: Vec::new(),
            mesh: IrMeshControls::default(),
            optimization: IrOptimizationModel::default(),
            uncertainties: Vec::new(),
            sensitivity: SensitivityManifest::default(),
        }
    }

    /// Ensure older serialized IR payloads retain the root source identity.
    pub fn ensure_source_identity(&mut self) {
        if !self
            .source_identities
            .iter()
            .any(|identity| identity == &self.source.identity)
        {
            self.source_identities
                .insert(0, self.source.identity.clone());
        }
    }

    pub fn structural_hash(&self) -> Result<StructuralHash, StructuralHashError> {
        StructuralHash::of(self)
    }

    /// Compare the backend-neutral topology of two IR candidates.
    ///
    /// Numeric argument changes are deliberately ignored: a design-variable
    /// perturbation can change a mesh metric or primitive size while retaining
    /// the same entity IDs and operation graph.  Added/removed entities,
    /// operation/input changes, or changed dynamic-branch membership are
    /// reported so a meshing consumer can make an explicit remesh decision.
    pub fn topology_diff(&self, candidate: &Self) -> IrTopologyDiff {
        let current = self
            .entities
            .iter()
            .map(|node| (node.id, node))
            .collect::<BTreeMap<_, _>>();
        let next = candidate
            .entities
            .iter()
            .map(|node| (node.id, node))
            .collect::<BTreeMap<_, _>>();

        let added_entities = next
            .keys()
            .filter(|id| !current.contains_key(id))
            .copied()
            .collect();
        let removed_entities = current
            .keys()
            .filter(|id| !next.contains_key(id))
            .copied()
            .collect();
        let modified_entities = current
            .keys()
            .filter_map(|id| {
                let before = current.get(id)?;
                let after = next.get(id)?;
                (topology_node_signature(before) != topology_node_signature(after)).then_some(*id)
            })
            .collect();
        let changed_geometry_values = current
            .keys()
            .filter_map(|id| {
                let before = current.get(id)?;
                let after = next.get(id)?;
                (geometry_node_value_signature(before) != geometry_node_value_signature(after))
                    .then_some(*id)
            })
            .collect();

        let changed_controls = self
            .controls
            .iter()
            .zip(candidate.controls.iter())
            .enumerate()
            .filter_map(|(index, (before, after))| {
                (control_topology_signature(before) != control_topology_signature(after))
                    .then_some(index)
            })
            .chain(
                self.controls.len().min(candidate.controls.len())
                    ..self.controls.len().max(candidate.controls.len()),
            )
            .collect();

        let current_groups = self
            .physical_groups
            .iter()
            .map(|group| (group.id, group))
            .collect::<BTreeMap<_, _>>();
        let candidate_groups = candidate
            .physical_groups
            .iter()
            .map(|group| (group.id, group))
            .collect::<BTreeMap<_, _>>();
        let changed_physical_groups = current_groups
            .keys()
            .filter_map(|id| {
                let before = current_groups.get(id)?;
                let after = candidate_groups.get(id)?;
                let before_signature = StructuralHash::of(&(
                    before.name.as_str(),
                    before.dimension,
                    &before.entities,
                    before.material,
                ))
                .ok();
                let after_signature = StructuralHash::of(&(
                    after.name.as_str(),
                    after.dimension,
                    &after.entities,
                    after.material,
                ))
                .ok();
                (before_signature != after_signature).then_some(*id)
            })
            .chain(
                candidate_groups
                    .keys()
                    .filter(|id| !current_groups.contains_key(id))
                    .copied(),
            )
            .chain(
                current_groups
                    .keys()
                    .filter(|id| !candidate_groups.contains_key(id))
                    .copied(),
            )
            .collect();

        let current_parameters = self
            .parameters
            .iter()
            .map(|parameter| (parameter.id, parameter))
            .collect::<BTreeMap<_, _>>();
        let candidate_parameters = candidate
            .parameters
            .iter()
            .map(|parameter| (parameter.id, parameter))
            .collect::<BTreeMap<_, _>>();
        let current_variables = self
            .optimization
            .variables
            .iter()
            .map(|variable| (variable.id, variable))
            .collect::<BTreeMap<_, _>>();
        let candidate_variables = candidate
            .optimization
            .variables
            .iter()
            .map(|variable| (variable.id, variable))
            .collect::<BTreeMap<_, _>>();
        let changed_parameters = current_parameters
            .keys()
            .filter_map(|id| {
                let before = current_parameters.get(id)?;
                let after = candidate_parameters.get(id)?;
                let before_signature = StructuralHash::of(&(&before.value, before.dimension)).ok();
                let after_signature = StructuralHash::of(&(&after.value, after.dimension)).ok();
                (before_signature != after_signature).then_some(*id)
            })
            .chain(
                candidate_parameters
                    .keys()
                    .filter(|id| !current_parameters.contains_key(id))
                    .copied(),
            )
            .chain(
                current_parameters
                    .keys()
                    .filter(|id| !candidate_parameters.contains_key(id))
                    .copied(),
            )
            .chain(current_variables.keys().filter_map(|id| {
                let before = current_variables.get(id)?;
                let after = candidate_variables.get(id)?;
                let before_signature = StructuralHash::of(&before.initial).ok();
                let after_signature = StructuralHash::of(&after.initial).ok();
                (before_signature != after_signature).then_some(*id)
            }))
            .chain(
                candidate_variables
                    .keys()
                    .filter(|id| !current_variables.contains_key(id))
                    .copied(),
            )
            .chain(
                current_variables
                    .keys()
                    .filter(|id| !candidate_variables.contains_key(id))
                    .copied(),
            )
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();

        IrTopologyDiff {
            added_entities,
            removed_entities,
            modified_entities,
            changed_geometry_values,
            changed_controls,
            changed_physical_groups,
            changed_parameters,
        }
    }

    /// Build the explicit remesh/state-transfer contract for a candidate IR.
    ///
    /// Sensitivity entries are matched by changed parameters, entity graph
    /// nodes, and physical groups. A topology change without a matching entry
    /// deliberately falls back to `TopologyDependent` plus
    /// `ApplicationDefined`, requiring the downstream consumer to opt in
    /// rather than silently transferring state.
    pub fn change_contract(&self, candidate: &Self) -> IrChangeContract {
        let diff = self.topology_diff(candidate);
        let changed_entities = diff
            .added_entities
            .iter()
            .chain(&diff.removed_entities)
            .chain(&diff.modified_entities)
            .chain(&diff.changed_geometry_values)
            .copied()
            .collect::<BTreeSet<_>>();
        let changed_groups = diff
            .changed_physical_groups
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();
        let changed_parameters = diff
            .changed_parameters
            .iter()
            .copied()
            .collect::<BTreeSet<_>>();

        let mut affected_variables = BTreeSet::new();
        let mut remesh = RemeshResponsibility::NotRequired;
        let mut state_transfer = StateTransferResponsibility::None;
        for entry in &self.sensitivity.entries {
            let affected = entry
                .parameters
                .iter()
                .any(|parameter| changed_parameters.contains(parameter))
                || entry
                    .entities
                    .iter()
                    .any(|entity| changed_entities.contains(entity))
                || entry
                    .physical_groups
                    .iter()
                    .any(|group| changed_groups.contains(group));
            if affected {
                affected_variables.insert(entry.variable);
                remesh = higher_remesh_responsibility(remesh, entry.remesh);
                state_transfer =
                    higher_state_transfer_responsibility(state_transfer, entry.state_transfer);
            }
        }

        if diff.topology_changed() && affected_variables.is_empty() {
            remesh = RemeshResponsibility::TopologyDependent;
            state_transfer = StateTransferResponsibility::ApplicationDefined;
        }

        IrChangeContract {
            diff,
            affected_variables: affected_variables.into_iter().collect(),
            remesh,
            state_transfer,
        }
    }

    /// Build an entity-level transfer plan for a candidate remesh.
    ///
    /// Stable unchanged IDs are preserved, removed IDs are dropped, new IDs
    /// are initialized, and changed IDs receive the strongest transfer action
    /// declared by the sensitivity/change contract. Numerical interpolation or
    /// conservation remains the responsibility of the selected consumer.
    pub fn state_transfer_plan(&self, candidate: &Self) -> IrStateTransferPlan {
        let contract = self.change_contract(candidate);
        let diff = &contract.diff;
        let old_ids = self
            .entities
            .iter()
            .map(|node| node.id)
            .collect::<BTreeSet<_>>();
        let new_ids = candidate
            .entities
            .iter()
            .map(|node| node.id)
            .collect::<BTreeSet<_>>();
        let changed = diff
            .modified_entities
            .iter()
            .chain(&diff.changed_geometry_values)
            .copied()
            .collect::<BTreeSet<_>>();
        let control_change = !diff.changed_controls.is_empty();
        let mut entries = old_ids
            .union(&new_ids)
            .copied()
            .map(|entity| {
                let action = if !new_ids.contains(&entity) {
                    IrStateTransferAction::Drop
                } else if !old_ids.contains(&entity) {
                    action_for_new_entity(contract.state_transfer)
                } else if changed.contains(&entity) || control_change {
                    action_for_changed_entity(contract.state_transfer)
                } else {
                    IrStateTransferAction::Preserve
                };
                IrStateTransferEntry { entity, action }
            })
            .collect::<Vec<_>>();
        entries.sort_by_key(|entry| entry.entity);
        IrStateTransferPlan {
            responsibility: contract.state_transfer,
            entries,
        }
    }

    /// Validate backend-independent invariants before code generation.
    pub fn validate(&self) -> Result<(), Vec<IrValidationError>> {
        let mut errors = Vec::new();
        if self.schema_version != CURRENT_IR_SCHEMA_VERSION {
            errors.push(IrValidationError::UnsupportedSchema(self.schema_version));
        }

        let mut entity_ids = BTreeSet::new();
        for node in &self.entities {
            if node.id.index == u64::MAX {
                errors.push(IrValidationError::SentinelEntityId(node.id));
            }
            if !entity_ids.insert(node.id) {
                errors.push(IrValidationError::DuplicateEntityId(node.id));
            }
        }

        let entity_nodes = self
            .entities
            .iter()
            .map(|node| (node.id, node))
            .collect::<BTreeMap<_, _>>();
        fn validate_parent_chain(
            entity: EntityId,
            nodes: &BTreeMap<EntityId, &GeometryNode>,
            visiting: &mut BTreeSet<EntityId>,
            completed: &mut BTreeSet<EntityId>,
            errors: &mut Vec<IrValidationError>,
        ) {
            if completed.contains(&entity) {
                return;
            }
            if !visiting.insert(entity) {
                errors.push(IrValidationError::EntityParentCycle { entity });
                return;
            }
            if let Some(node) = nodes.get(&entity) {
                for parent in &node.inputs {
                    if nodes.contains_key(parent) {
                        validate_parent_chain(*parent, nodes, visiting, completed, errors);
                    }
                }
            }
            visiting.remove(&entity);
            completed.insert(entity);
        }
        let mut visiting = BTreeSet::new();
        let mut completed = BTreeSet::new();
        for node in &self.entities {
            validate_parent_chain(
                node.id,
                &entity_nodes,
                &mut visiting,
                &mut completed,
                &mut errors,
            );
        }

        for (control_index, control) in self.controls.iter().enumerate() {
            if let Some(parent) = control.parent
                && parent >= control_index
            {
                errors.push(IrValidationError::InvalidConditionalParent {
                    control: control_index,
                    parent,
                });
            }
            if let Some(parent_loop) = control.parent_loop
                && parent_loop >= self.loops.len()
            {
                errors.push(IrValidationError::InvalidConditionalLoopParent {
                    control: control_index,
                    parent_loop,
                });
            }
            let mut branch_entities = BTreeSet::new();
            for (branch_index, branch) in control.branches.iter().enumerate() {
                if branch.condition.is_none() && branch_index + 1 != control.branches.len() {
                    errors.push(IrValidationError::InvalidConditionalBranchOrder {
                        control: control_index,
                        branch: branch_index,
                    });
                }
                for entity in &branch.entities {
                    if !branch_entities.insert(*entity) {
                        errors.push(IrValidationError::DuplicateConditionalEntity {
                            control: control_index,
                            entity: *entity,
                        });
                    }
                    if !entity_ids.contains(entity) {
                        errors.push(IrValidationError::UnknownConditionalEntity {
                            referenced: *entity,
                        });
                    }
                }
            }
        }

        for (loop_index, loop_control) in self.loops.iter().enumerate() {
            if let Some(parent) = loop_control.parent
                && parent >= loop_index
            {
                errors.push(IrValidationError::InvalidLoopParent { loop_index, parent });
            }
            if let Some(parent_control) = loop_control.parent_control
                && parent_control >= self.controls.len()
            {
                errors.push(IrValidationError::InvalidLoopConditionalParent {
                    loop_index,
                    parent_control,
                });
            }
            if loop_control.clauses.is_empty() {
                errors.push(IrValidationError::EmptyLoopClauses { loop_index });
            }
            let mut loop_entities = BTreeSet::new();
            for entity in &loop_control.entities {
                if !loop_entities.insert(*entity) {
                    errors.push(IrValidationError::DuplicateLoopEntity {
                        loop_index,
                        entity: *entity,
                    });
                }
                if !entity_ids.contains(entity) {
                    errors.push(IrValidationError::UnknownLoopEntity {
                        loop_index,
                        entity: *entity,
                    });
                }
            }
            for clause in &loop_control.clauses {
                if clause.name.trim().is_empty() {
                    errors.push(IrValidationError::EmptyLoopVariable { loop_index });
                }
            }
        }

        for node in &self.entities {
            for input in &node.inputs {
                if !entity_ids.contains(input) {
                    errors.push(IrValidationError::UnknownEntityReference {
                        owner: node.id,
                        referenced: *input,
                    });
                }
            }
            if matches!(node.operation, GeometryOperation::Boolean { .. }) {
                if node.inputs.len() < 2 {
                    errors.push(IrValidationError::BooleanArity(node.id));
                }
                for input in &node.inputs {
                    if input.dimension != node.id.dimension {
                        errors.push(IrValidationError::DimensionMismatch {
                            owner: node.id,
                            referenced: *input,
                        });
                    }
                }
            }
        }

        let mut group_ids = BTreeSet::new();
        for group in &self.physical_groups {
            if !group_ids.insert(group.id) {
                errors.push(IrValidationError::DuplicatePhysicalGroupId(group.id));
            }
            for entity in &group.entities {
                if entity.dimension != group.dimension {
                    errors.push(IrValidationError::PhysicalGroupDimension {
                        group: group.id,
                        entity: *entity,
                    });
                }
                if !entity_ids.contains(entity) {
                    errors.push(IrValidationError::UnknownPhysicalGroupEntity {
                        group: group.id,
                        entity: *entity,
                    });
                }
            }
        }

        for boundary in &self.boundaries {
            for entity in &boundary.entities {
                if entity.dimension != boundary.dimension {
                    errors.push(IrValidationError::BoundaryDimension {
                        name: boundary.name.clone(),
                        entity: *entity,
                    });
                }
                if !entity_ids.contains(entity) {
                    errors.push(IrValidationError::UnknownBoundaryEntity {
                        name: boundary.name.clone(),
                        entity: *entity,
                    });
                }
            }
        }

        for pair in &self.periodic_pairs {
            validate_known_entity("periodic source", pair.source, &entity_ids, &mut errors);
            validate_known_entity("periodic target", pair.target, &entity_ids, &mut errors);
            validate_pair_dimensions(pair.source, pair.target, &mut errors);
        }
        for interface in &self.interfaces {
            validate_known_entity(
                &format!("interface '{}' primary", interface.name),
                interface.primary,
                &entity_ids,
                &mut errors,
            );
            validate_known_entity(
                &format!("interface '{}' secondary", interface.name),
                interface.secondary,
                &entity_ids,
                &mut errors,
            );
            validate_pair_dimensions(interface.primary, interface.secondary, &mut errors);
        }
        for region in &self.material_regions {
            validate_entity_slice(
                &format!("material region '{}'", region.name),
                &region.entities,
                &entity_ids,
                &mut errors,
            );
        }
        let known_field_ids = self
            .fields
            .iter()
            .map(|field| field.id)
            .collect::<BTreeSet<_>>();
        for field in &self.fields {
            match &field.kind {
                IrGeometryFieldKind::Topology { region, .. } => validate_entity_slice(
                    &format!("topology field '{}'", field.name),
                    region,
                    &entity_ids,
                    &mut errors,
                ),
                IrGeometryFieldKind::Projected {
                    geometry,
                    source_field,
                    ..
                } => {
                    validate_entity_slice(
                        &format!("projected field '{}'", field.name),
                        geometry,
                        &entity_ids,
                        &mut errors,
                    );
                    if !known_field_ids.contains(source_field) {
                        errors.push(IrValidationError::UnknownProjectedSourceField {
                            field: *source_field,
                        });
                    }
                }
                IrGeometryFieldKind::LevelSet { .. } => {}
            }
        }
        for (index, metric) in self.mesh.metrics.iter().enumerate() {
            validate_entity_slice(
                &format!("mesh metric {index}"),
                &metric.target,
                &entity_ids,
                &mut errors,
            );
        }
        for (index, layer) in self.mesh.boundary_layers.iter().enumerate() {
            validate_entity_slice(
                &format!("boundary layer {index}"),
                &layer.target,
                &entity_ids,
                &mut errors,
            );
        }
        for (index, refine) in self.mesh.refinements.iter().enumerate() {
            match &refine.kind {
                IrRefineKind::Near { entity, .. }
                | IrRefineKind::On { entity }
                | IrRefineKind::In { region: entity } => validate_known_entity(
                    &format!("refinement {index}"),
                    *entity,
                    &entity_ids,
                    &mut errors,
                ),
                IrRefineKind::Field { .. } => {}
            }
        }
        for (index, transfinite) in self.mesh.transfinite.iter().enumerate() {
            let (entity, expected) = match &transfinite.kind {
                IrTransfiniteKind::Curve { entity, .. } => (*entity, EntityDimension::Curve),
                IrTransfiniteKind::Surface { entity, .. } => (*entity, EntityDimension::Surface),
                IrTransfiniteKind::Volume { entity, .. } => (*entity, EntityDimension::Volume),
            };
            validate_known_entity(
                &format!("transfinite {index}"),
                entity,
                &entity_ids,
                &mut errors,
            );
            if entity.dimension != expected {
                errors.push(IrValidationError::MeshControlDimension {
                    context: format!("transfinite {index}"),
                    expected,
                    found: entity.dimension,
                    entity,
                });
            }
        }
        for (index, embed) in self.mesh.embeds.iter().enumerate() {
            validate_known_entity(
                &format!("embed {index} source"),
                embed.source,
                &entity_ids,
                &mut errors,
            );
            if embed.source.dimension != embed.source_dimension {
                errors.push(IrValidationError::MeshControlDimension {
                    context: format!("embed {index} source"),
                    expected: embed.source_dimension,
                    found: embed.source.dimension,
                    entity: embed.source,
                });
            }
            if embed.target.dimension != embed.target_dimension {
                errors.push(IrValidationError::MeshControlDimension {
                    context: format!("embed {index} target"),
                    expected: embed.target_dimension,
                    found: embed.target.dimension,
                    entity: embed.target,
                });
            }
            validate_known_entity(
                &format!("embed {index} target"),
                embed.target,
                &entity_ids,
                &mut errors,
            );
        }
        for (index, compound) in self.mesh.compounds.iter().enumerate() {
            validate_entity_slice(
                &format!("compound {index}"),
                &compound.entities,
                &entity_ids,
                &mut errors,
            );
            let Some(first) = compound.entities.first().copied() else {
                errors.push(IrValidationError::MeshCompoundEmpty { index });
                continue;
            };
            for entity in compound.entities.iter().copied().skip(1) {
                if entity.dimension != first.dimension {
                    errors.push(IrValidationError::MeshCompoundDimensionMismatch {
                        index,
                        expected: first.dimension,
                        found: entity.dimension,
                        entity,
                    });
                }
            }
        }
        for region in &self.optimization.regions {
            validate_entity_slice(
                &format!("design region '{}'", region.name),
                &region.entities,
                &entity_ids,
                &mut errors,
            );
        }
        for constraint in &self.optimization.constraints {
            match &constraint.constraint {
                IrConstraint::VolumeFraction { region, .. }
                | IrConstraint::MinimumFeature { region, .. } => validate_entity_slice(
                    constraint
                        .name
                        .as_deref()
                        .unwrap_or("optimization constraint"),
                    region,
                    &entity_ids,
                    &mut errors,
                ),
                _ => {}
            }
        }
        let known_parameter_ids = self
            .parameters
            .iter()
            .map(|parameter| parameter.id)
            .chain(
                self.optimization
                    .variables
                    .iter()
                    .map(|variable| variable.id),
            )
            .collect::<BTreeSet<_>>();
        let known_variable_ids = self
            .optimization
            .variables
            .iter()
            .map(|variable| variable.id)
            .collect::<BTreeSet<_>>();
        for entry in &self.sensitivity.entries {
            if !known_variable_ids.contains(&entry.variable) {
                errors.push(IrValidationError::UnknownSensitivityVariable {
                    variable: entry.variable,
                });
            }
            for parameter in &entry.parameters {
                if !known_parameter_ids.contains(parameter) {
                    errors.push(IrValidationError::UnknownSensitivityParameter {
                        variable: entry.variable,
                        parameter: *parameter,
                    });
                }
            }
            validate_entity_slice(
                &format!("sensitivity variable {}", entry.variable.0),
                &entry.entities,
                &entity_ids,
                &mut errors,
            );
            for group in &entry.physical_groups {
                if !group_ids.contains(group) {
                    errors.push(IrValidationError::UnknownSensitivityPhysicalGroup {
                        variable: entry.variable,
                        group: *group,
                    });
                }
            }
        }

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Validate source ancestry required by backend sidecars and diagnostics.
    ///
    /// Structural validation intentionally remains usable for synthetic IR
    /// fixtures, while compiler-produced IR must pass this stricter boundary
    /// before native backend emission.
    pub fn validate_provenance(&self) -> Result<(), Vec<IrValidationError>> {
        let mut errors = Vec::new();
        for node in &self.entities {
            if node.provenance.is_empty() {
                errors.push(IrValidationError::MissingEntityProvenance { entity: node.id });
                continue;
            }
            if node
                .provenance
                .iter()
                .any(|reference| reference.span.start > reference.span.end)
            {
                errors.push(IrValidationError::InvalidEntityProvenanceSpan { entity: node.id });
            }
        }

        for parameter in &self.parameters {
            validate_provenance_records(
                format!("parameter '{}'", parameter.name),
                &parameter.provenance,
                &mut errors,
            );
        }
        for (control_index, control) in self.controls.iter().enumerate() {
            validate_provenance_records(
                format!("conditional control {control_index}"),
                &control.provenance,
                &mut errors,
            );
            for (branch_index, branch) in control.branches.iter().enumerate() {
                validate_provenance_records(
                    format!("conditional control {control_index} branch {branch_index}"),
                    &branch.provenance,
                    &mut errors,
                );
            }
        }
        for (loop_index, loop_control) in self.loops.iter().enumerate() {
            validate_provenance_records(
                format!("loop {loop_index}"),
                &loop_control.provenance,
                &mut errors,
            );
            for (clause_index, clause) in loop_control.clauses.iter().enumerate() {
                validate_provenance_records(
                    format!("loop {loop_index} clause {clause_index}"),
                    &clause.provenance,
                    &mut errors,
                );
            }
        }
        for group in &self.physical_groups {
            validate_provenance_records(
                format!("physical group {}", group.id.0),
                &group.provenance,
                &mut errors,
            );
        }
        for boundary in &self.boundaries {
            validate_provenance_records(
                format!("boundary '{}'", boundary.name),
                &boundary.provenance,
                &mut errors,
            );
        }
        for interface in &self.interfaces {
            validate_provenance_records(
                format!("interface '{}'", interface.name),
                &interface.provenance,
                &mut errors,
            );
        }
        for (pair_index, pair) in self.periodic_pairs.iter().enumerate() {
            validate_provenance_records(
                format!("periodic pair {pair_index}"),
                &pair.provenance,
                &mut errors,
            );
        }
        for coordinate_system in &self.coordinate_systems {
            validate_provenance_records(
                format!("coordinate system '{}'", coordinate_system.name),
                &coordinate_system.provenance,
                &mut errors,
            );
        }
        for region in &self.material_regions {
            validate_provenance_records(
                format!("material region '{}'", region.name),
                &region.provenance,
                &mut errors,
            );
        }
        for field in &self.fields {
            validate_provenance_records(
                format!("geometry field '{}'", field.name),
                &field.provenance,
                &mut errors,
            );
        }
        for (metric_index, metric) in self.mesh.metrics.iter().enumerate() {
            validate_provenance_records(
                format!("mesh metric {metric_index}"),
                &metric.provenance,
                &mut errors,
            );
        }
        for (layer_index, layer) in self.mesh.boundary_layers.iter().enumerate() {
            validate_provenance_records(
                format!("boundary layer {layer_index}"),
                &layer.provenance,
                &mut errors,
            );
        }
        for (refine_index, refine) in self.mesh.refinements.iter().enumerate() {
            validate_provenance_records(
                format!("refinement {refine_index}"),
                &refine.provenance,
                &mut errors,
            );
        }
        for (transfinite_index, transfinite) in self.mesh.transfinite.iter().enumerate() {
            validate_provenance_records(
                format!("transfinite {transfinite_index}"),
                &transfinite.provenance,
                &mut errors,
            );
        }
        for (embed_index, embed) in self.mesh.embeds.iter().enumerate() {
            validate_provenance_records(
                format!("embed {embed_index}"),
                &embed.provenance,
                &mut errors,
            );
        }
        for (compound_index, compound) in self.mesh.compounds.iter().enumerate() {
            validate_provenance_records(
                format!("compound {compound_index}"),
                &compound.provenance,
                &mut errors,
            );
        }
        for (variable_index, variable) in self.optimization.variables.iter().enumerate() {
            validate_provenance_records(
                format!("optimization variable {variable_index}"),
                &variable.provenance,
                &mut errors,
            );
        }
        for (objective_index, objective) in self.optimization.objectives.iter().enumerate() {
            validate_provenance_records(
                format!("optimization objective {objective_index}"),
                &objective.provenance,
                &mut errors,
            );
        }
        for (constraint_index, constraint) in self.optimization.constraints.iter().enumerate() {
            validate_provenance_records(
                format!("optimization constraint {constraint_index}"),
                &constraint.provenance,
                &mut errors,
            );
        }
        for (region_index, region) in self.optimization.regions.iter().enumerate() {
            validate_provenance_records(
                format!("design region {region_index}"),
                &region.provenance,
                &mut errors,
            );
        }
        for uncertainty in &self.uncertainties {
            validate_provenance_records(
                format!("uncertainty '{}'", uncertainty.name),
                &uncertainty.provenance,
                &mut errors,
            );
        }
        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }
}

/// Versioned persistence boundary for a canonical IR artifact.
///
/// Raw `CanonicalGeometryIr` JSON predates this type and is accepted only by
/// [`VersionedCanonicalIr::from_json_slice`] as a labelled migration input.
/// New callers should use this envelope so future migrations cannot be
/// mistaken for a compatible raw serde payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct VersionedCanonicalIr {
    pub document_version: IrDocumentVersion,
    pub ir: CanonicalGeometryIr,
}

impl VersionedCanonicalIr {
    pub fn current(ir: CanonicalGeometryIr) -> Self {
        Self {
            document_version: CURRENT_IR_DOCUMENT_VERSION,
            ir,
        }
    }

    pub fn to_json_vec(&self) -> Result<Vec<u8>, IrPersistenceError> {
        if self.document_version != CURRENT_IR_DOCUMENT_VERSION {
            return Err(IrPersistenceError::UnsupportedDocumentVersion(
                self.document_version,
            ));
        }
        self.ir.validate().map_err(IrPersistenceError::InvalidIr)?;
        serde_json::to_vec(self).map_err(IrPersistenceError::Json)
    }

    pub fn from_json_slice(bytes: &[u8]) -> Result<(Self, IrMigrationReport), IrPersistenceError> {
        Self::from_json_slice_with_limit(bytes, DEFAULT_MAX_SERIALIZED_IR_BYTES)
    }

    pub fn from_json_slice_with_limit(
        bytes: &[u8],
        max_bytes: usize,
    ) -> Result<(Self, IrMigrationReport), IrPersistenceError> {
        if bytes.len() > max_bytes {
            return Err(IrPersistenceError::ResourceLimit {
                actual: bytes.len(),
                maximum: max_bytes,
            });
        }
        let value = serde_json::from_slice(bytes).map_err(IrPersistenceError::Json)?;
        let (migrated, report) = migrate_ir_value(value)?;
        let mut document = serde_json::from_value::<VersionedCanonicalIr>(migrated)
            .map_err(IrPersistenceError::Json)?;
        document.ir.ensure_source_identity();
        document
            .ir
            .validate()
            .map_err(IrPersistenceError::InvalidIr)?;
        Ok((document, report))
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IrMigrationReport {
    pub steps: Vec<IrMigrationStep>,
}

impl IrMigrationReport {
    pub fn was_migrated(&self) -> bool {
        !self.steps.is_empty()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IrMigrationStep {
    pub from: IrDocumentVersion,
    pub to: IrDocumentVersion,
    pub description: &'static str,
}

/// Migrate an explicit canonical-IR envelope or a legacy raw IR value.
pub fn migrate_ir_value(value: Value) -> Result<(Value, IrMigrationReport), IrPersistenceError> {
    let Value::Object(mut object) = value else {
        return Err(IrPersistenceError::InvalidEnvelope);
    };
    if object.contains_key("ir") {
        let version = object
            .remove("document_version")
            .ok_or(IrPersistenceError::MissingDocumentVersion)
            .and_then(|value| {
                serde_json::from_value::<IrDocumentVersion>(value).map_err(IrPersistenceError::Json)
            })?;
        let ir = object.remove("ir").ok_or(IrPersistenceError::MissingIr)?;
        if !object.is_empty() {
            return Err(IrPersistenceError::InvalidEnvelope);
        }
        if version != CURRENT_IR_DOCUMENT_VERSION {
            return Err(IrPersistenceError::UnsupportedDocumentVersion(version));
        }
        return Ok((
            ir_envelope(CURRENT_IR_DOCUMENT_VERSION, ir),
            IrMigrationReport::default(),
        ));
    }

    // The historical persistence format was a bare CanonicalGeometryIr. Its
    // internal `schema_version` remains part of the IR payload and is checked
    // after deserialization/validation above.
    let ir = Value::Object(object);
    Ok((
        ir_envelope(CURRENT_IR_DOCUMENT_VERSION, ir),
        IrMigrationReport {
            steps: vec![IrMigrationStep {
                from: LEGACY_IR_DOCUMENT_VERSION,
                to: CURRENT_IR_DOCUMENT_VERSION,
                description: "wrap legacy raw canonical IR in the explicit versioned IR envelope",
            }],
        },
    ))
}

fn ir_envelope(version: IrDocumentVersion, ir: Value) -> Value {
    let mut object = Map::new();
    object.insert(
        "document_version".into(),
        Value::Number(serde_json::Number::from(version.0)),
    );
    object.insert("ir".into(), ir);
    Value::Object(object)
}

#[derive(Debug)]
pub enum IrPersistenceError {
    Json(serde_json::Error),
    InvalidEnvelope,
    MissingDocumentVersion,
    MissingIr,
    UnsupportedDocumentVersion(IrDocumentVersion),
    ResourceLimit { actual: usize, maximum: usize },
    InvalidIr(Vec<IrValidationError>),
}

impl fmt::Display for IrPersistenceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(error) => write!(f, "invalid serialized canonical IR: {error}"),
            Self::InvalidEnvelope => {
                f.write_str("serialized canonical IR must be an object envelope")
            }
            Self::MissingDocumentVersion => {
                f.write_str("canonical IR envelope has no document_version")
            }
            Self::MissingIr => f.write_str("canonical IR envelope has no ir payload"),
            Self::UnsupportedDocumentVersion(version) => {
                write!(f, "unsupported canonical IR document version {}", version.0)
            }
            Self::ResourceLimit { actual, maximum } => write!(
                f,
                "serialized canonical IR is {actual} bytes, exceeding the {maximum}-byte limit"
            ),
            Self::InvalidIr(errors) => write!(
                f,
                "canonical IR validation failed with {} error(s)",
                errors.len()
            ),
        }
    }
}

impl std::error::Error for IrPersistenceError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(error) => Some(error),
            _ => None,
        }
    }
}

fn validate_pair_dimensions(
    first: EntityId,
    second: EntityId,
    errors: &mut Vec<IrValidationError>,
) {
    if first.dimension != second.dimension {
        errors.push(IrValidationError::PairDimensionMismatch { first, second });
    }
}

fn validate_known_entity(
    context: &str,
    entity: EntityId,
    known: &BTreeSet<EntityId>,
    errors: &mut Vec<IrValidationError>,
) {
    if !known.contains(&entity) {
        errors.push(IrValidationError::UnknownReferencedEntity {
            context: context.to_owned(),
            entity,
        });
    }
}

fn validate_entity_slice(
    context: &str,
    entities: &[EntityId],
    known: &BTreeSet<EntityId>,
    errors: &mut Vec<IrValidationError>,
) {
    for entity in entities {
        validate_known_entity(context, *entity, known, errors);
    }
}

fn validate_provenance_records(
    context: impl Into<String>,
    provenance: &[SourceReference],
    errors: &mut Vec<IrValidationError>,
) {
    let context = context.into();
    if provenance.is_empty() {
        errors.push(IrValidationError::MissingProvenance { context });
    } else if provenance
        .iter()
        .any(|reference| reference.span.start > reference.span.end)
    {
        errors.push(IrValidationError::InvalidProvenanceSpan { context });
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrParameter {
    pub id: SymbolId,
    pub name: String,
    pub value: IrValue,
    pub dimension: Dimension,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrValue {
    Integer(i64),
    Scalar {
        value: f64,
        dimension: Dimension,
    },
    Boolean(bool),
    Text(String),
    Vector(Vec<IrValue>),
    Parameter(SymbolId),
    Entity(EntityId),
    Unary {
        op: IrUnaryOperator,
        operand: Box<IrValue>,
    },
    Binary {
        op: IrBinaryOperator,
        left: Box<IrValue>,
        right: Box<IrValue>,
    },
    Call {
        function: SymbolId,
        arguments: Vec<IrValue>,
    },
    Index {
        collection: Box<IrValue>,
        index: Box<IrValue>,
    },
    Field {
        object: Box<IrValue>,
        field: SymbolId,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IrUnaryOperator {
    Negate,
    LogicalNot,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IrBinaryOperator {
    Add,
    Subtract,
    Multiply,
    Divide,
    Modulo,
    Power,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    LogicalAnd,
    LogicalOr,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryNode {
    pub id: EntityId,
    /// Resolved, module-qualified surface name used for diagnostics and stable
    /// frontend-to-backend symbol lookup.
    pub name: String,
    pub operation: GeometryOperation,
    pub inputs: Vec<EntityId>,
    pub arguments: Vec<IrArgument>,
    pub provenance: Vec<SourceReference>,
}

/// Backend-neutral result of comparing two canonical IR topology graphs.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct IrTopologyDiff {
    pub added_entities: Vec<EntityId>,
    pub removed_entities: Vec<EntityId>,
    pub modified_entities: Vec<EntityId>,
    pub changed_geometry_values: Vec<EntityId>,
    pub changed_controls: Vec<usize>,
    pub changed_physical_groups: Vec<PhysicalGroupId>,
    pub changed_parameters: Vec<SymbolId>,
}

/// Explicit candidate-change hand-off for optimization and remeshing clients.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IrChangeContract {
    pub diff: IrTopologyDiff,
    pub affected_variables: Vec<SymbolId>,
    pub remesh: RemeshResponsibility,
    pub state_transfer: StateTransferResponsibility,
}

impl IrChangeContract {
    /// Validate the change hand-off before a remesher or state consumer uses it.
    ///
    /// The compiler currently constructs this contract canonically, but keeping
    /// the invariant at the consumer boundary prevents hand-written, persisted,
    /// or cross-language contracts from silently downgrading a topology change
    /// to a numerical transfer.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self
            .affected_variables
            .windows(2)
            .any(|pair| pair[0] >= pair[1])
        {
            return Err("change-contract variables must be strictly sorted");
        }
        let no_change = self.diff.added_entities.is_empty()
            && self.diff.removed_entities.is_empty()
            && self.diff.modified_entities.is_empty()
            && self.diff.changed_geometry_values.is_empty()
            && self.diff.changed_controls.is_empty()
            && self.diff.changed_physical_groups.is_empty()
            && self.diff.changed_parameters.is_empty();
        if no_change
            && (self.remesh != RemeshResponsibility::NotRequired
                || self.state_transfer != StateTransferResponsibility::None)
        {
            return Err("unchanged topology cannot request remesh or state transfer");
        }
        if self.diff.topology_changed() && self.remesh != RemeshResponsibility::TopologyDependent {
            return Err("topology changes require topology-dependent remesh");
        }
        if self.diff.topology_changed()
            && self.state_transfer != StateTransferResponsibility::ApplicationDefined
        {
            return Err("topology changes require application-defined state transfer");
        }
        if !self.diff.topology_changed() && self.remesh == RemeshResponsibility::TopologyDependent {
            return Err("topology-dependent remesh requires a topology change");
        }
        Ok(())
    }
}

/// Executable entity-level hand-off for a remesh/state-transfer consumer.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IrStateTransferPlan {
    pub responsibility: StateTransferResponsibility,
    pub entries: Vec<IrStateTransferEntry>,
}

/// Action a downstream consumer must take for one stable entity ID.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IrStateTransferAction {
    Preserve,
    Interpolate,
    Conservative,
    ApplicationDefined,
    Initialize,
    Drop,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IrStateTransferEntry {
    pub entity: EntityId,
    pub action: IrStateTransferAction,
}

impl IrStateTransferPlan {
    /// Look up the deterministic action for one entity.
    pub fn action_for(&self, entity: EntityId) -> Option<IrStateTransferAction> {
        self.entries
            .binary_search_by_key(&entity, |entry| entry.entity)
            .ok()
            .map(|index| self.entries[index].action)
    }

    /// Validate uniqueness and canonical entity ordering before consumption.
    pub fn validate(&self) -> Result<(), &'static str> {
        if self
            .entries
            .windows(2)
            .any(|pair| pair[0].entity >= pair[1].entity)
        {
            return Err("state-transfer entries must be strictly entity-sorted");
        }
        for entry in &self.entries {
            let required = match entry.action {
                IrStateTransferAction::Interpolate => {
                    Some(StateTransferResponsibility::Interpolate)
                }
                IrStateTransferAction::Conservative => {
                    Some(StateTransferResponsibility::Conservative)
                }
                IrStateTransferAction::ApplicationDefined => {
                    Some(StateTransferResponsibility::ApplicationDefined)
                }
                IrStateTransferAction::Preserve
                | IrStateTransferAction::Initialize
                | IrStateTransferAction::Drop => None,
            };
            if required.is_some_and(|required| {
                state_transfer_rank(required) > state_transfer_rank(self.responsibility)
            }) {
                return Err("state-transfer action exceeds declared responsibility");
            }
        }
        Ok(())
    }
}

impl IrTopologyDiff {
    /// Whether the candidate requires a topology-aware remesh.
    pub fn topology_changed(&self) -> bool {
        !(self.added_entities.is_empty()
            && self.removed_entities.is_empty()
            && self.modified_entities.is_empty()
            && self.changed_controls.is_empty())
    }

    /// Whether physical-group membership or marker metadata changed.
    pub fn markers_changed(&self) -> bool {
        !self.changed_physical_groups.is_empty()
    }

    /// Whether numeric parameter or geometry argument values changed.
    pub fn values_changed(&self) -> bool {
        !self.changed_parameters.is_empty() || !self.changed_geometry_values.is_empty()
    }
}

fn higher_remesh_responsibility(
    left: RemeshResponsibility,
    right: RemeshResponsibility,
) -> RemeshResponsibility {
    if remesh_rank(right) > remesh_rank(left) {
        right
    } else {
        left
    }
}

fn remesh_rank(value: RemeshResponsibility) -> u8 {
    match value {
        RemeshResponsibility::NotRequired => 0,
        RemeshResponsibility::Local => 1,
        RemeshResponsibility::Global => 2,
        RemeshResponsibility::TopologyDependent => 3,
    }
}

fn higher_state_transfer_responsibility(
    left: StateTransferResponsibility,
    right: StateTransferResponsibility,
) -> StateTransferResponsibility {
    if state_transfer_rank(right) > state_transfer_rank(left) {
        right
    } else {
        left
    }
}

fn state_transfer_rank(value: StateTransferResponsibility) -> u8 {
    match value {
        StateTransferResponsibility::None => 0,
        StateTransferResponsibility::Interpolate => 1,
        StateTransferResponsibility::Conservative => 2,
        StateTransferResponsibility::ApplicationDefined => 3,
    }
}

fn action_for_new_entity(responsibility: StateTransferResponsibility) -> IrStateTransferAction {
    match responsibility {
        StateTransferResponsibility::ApplicationDefined => {
            IrStateTransferAction::ApplicationDefined
        }
        _ => IrStateTransferAction::Initialize,
    }
}

fn action_for_changed_entity(responsibility: StateTransferResponsibility) -> IrStateTransferAction {
    match responsibility {
        StateTransferResponsibility::None => IrStateTransferAction::Initialize,
        StateTransferResponsibility::Interpolate => IrStateTransferAction::Interpolate,
        StateTransferResponsibility::Conservative => IrStateTransferAction::Conservative,
        StateTransferResponsibility::ApplicationDefined => {
            IrStateTransferAction::ApplicationDefined
        }
    }
}

fn topology_node_signature(node: &GeometryNode) -> Option<StructuralHash> {
    StructuralHash::of(&(node.operation.clone(), node.inputs.clone())).ok()
}

fn geometry_node_value_signature(node: &GeometryNode) -> Option<StructuralHash> {
    StructuralHash::of(&(node.operation.clone(), node.inputs.clone(), &node.arguments)).ok()
}

fn control_topology_signature(
    control: &IrConditionalControl,
) -> Vec<(Option<usize>, Vec<EntityId>)> {
    control
        .branches
        .iter()
        .map(|branch| (control.parent, branch.entities.clone()))
        .collect()
}

/// One dynamic geometry control-flow plan.  Branches are ordered exactly as
/// written in source; the first branch whose condition is true is selected.
/// `None` is the unconditional final `else` branch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrConditionalControl {
    /// Parent dynamic conditional, when nested in another conditional.
    #[serde(default)]
    pub parent: Option<usize>,
    /// Parent runtime loop, when this conditional is nested in a loop.
    /// The separate index keeps the two control collections independently
    /// append-only and preserves compatibility with older IR documents.
    #[serde(default)]
    pub parent_loop: Option<usize>,
    pub branches: Vec<IrConditionalBranch>,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrConditionalBranch {
    pub condition: Option<IrValue>,
    pub entities: Vec<EntityId>,
    pub provenance: Vec<SourceReference>,
}

/// One runtime-dynamic geometry loop. Clauses are ordered as written and
/// form a Cartesian product; the body is represented by the listed canonical
/// entity IDs. Backends may choose their native loop syntax, but must preserve
/// clause order and the inclusive-end contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrLoopControl {
    /// Parent runtime loop, when nested in another loop.
    #[serde(default)]
    pub parent: Option<usize>,
    /// Parent dynamic conditional, when this loop is nested in a branch.
    #[serde(default)]
    pub parent_control: Option<usize>,
    pub clauses: Vec<IrLoopClause>,
    pub entities: Vec<EntityId>,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrLoopClause {
    pub variable: SymbolId,
    pub name: String,
    pub start: IrValue,
    pub end: IrValue,
    pub inclusive: bool,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrArgument {
    pub name: IrArgumentName,
    pub value: IrValue,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IrArgumentName {
    Origin,
    Start,
    End,
    Center,
    Points,
    Sections,
    Ruled,
    Size,
    Radius,
    RadiusStart,
    RadiusEnd,
    MajorRadius,
    MinorRadius,
    Height,
    Direction,
    Angle,
    Axis,
    Count,
    Offset,
    Scale,
    Plane,
    Matrix,
    Layers,
    Recombine,
    Thickness,
    Distance,
    Tolerance,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeometryOperation {
    Primitive {
        kind: PrimitiveKind,
    },
    Boolean {
        kind: BooleanOperation,
    },
    Transform {
        kind: TransformOperation,
    },
    Sweep {
        kind: SweepOperation,
    },
    Array,
    Compound,
    /// Extract a lower-dimensional topological boundary (faces, edges, or
    /// vertices) from a named parent entity.
    Boundary {
        target: EntityDimension,
        /// Optional backend-neutral selector such as `top`, `bottom`, or
        /// `front`, applied within the extracted boundary set.
        selector: Option<String>,
    },
    Project {
        method: ProjectionMethod,
    },
    Shell,
    Fillet,
    Chamfer,
    Twist,
    Copy,
    Fragment,
    Coherence,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum PrimitiveKind {
    Point,
    Line,
    Arc,
    Spline,
    BSpline,
    Polygon,
    Rectangle,
    Disk,
    Box,
    Sphere,
    Cylinder,
    Cone,
    Torus,
    Wedge,
    Loft,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BooleanOperation {
    Union,
    Difference,
    Intersection,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum TransformOperation {
    Translate,
    Rotate,
    Scale,
    Mirror,
    Affine,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SweepOperation {
    Extrude,
    Revolve,
    Pipe,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalGroup {
    pub id: PhysicalGroupId,
    pub name: String,
    pub dimension: EntityDimension,
    pub entities: Vec<EntityId>,
    pub material: Option<SymbolId>,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrBoundary {
    pub name: String,
    pub dimension: EntityDimension,
    pub entities: Vec<EntityId>,
    pub kind: IrBoundaryKind,
    pub settings: Vec<IrProperty>,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrBoundaryKind {
    Dirichlet,
    Neumann,
    Robin,
    Impedance,
    Symmetry,
    Periodic,
    Named(SymbolId),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrInterface {
    pub name: String,
    pub primary: EntityId,
    pub secondary: EntityId,
    pub kind: IrInterfaceKind,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrInterfaceKind {
    Conforming,
    Contact,
    Mortar,
    FluidStructure,
    Named(SymbolId),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrPeriodicPair {
    pub source: EntityId,
    pub target: EntityId,
    pub transform: IrTransform,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrTransform {
    /// Row-major homogeneous 4x4 transform.
    pub coefficients: [f64; 16],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrCoordinateSystem {
    pub id: SymbolId,
    pub name: String,
    pub origin: [f64; 3],
    /// Row-major orthonormal basis.
    pub basis: [f64; 9],
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrMaterialRegion {
    pub name: String,
    pub material: SymbolId,
    pub entities: Vec<EntityId>,
    pub properties: Vec<IrProperty>,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrGeometryField {
    pub id: SymbolId,
    pub name: String,
    pub kind: IrGeometryFieldKind,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrGeometryFieldKind {
    Topology {
        region: Vec<EntityId>,
        initial_density: IrValue,
        filter_radius: Option<IrValue>,
    },
    LevelSet {
        field: IrValue,
        iso_value: IrValue,
    },
    Projected {
        geometry: Vec<EntityId>,
        source_field: SymbolId,
        method: ProjectionMethod,
        tolerance: Option<IrValue>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrProperty {
    pub name: String,
    pub value: IrValue,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IrMeshControls {
    /// Mesh-block scopes are retained even though the current backend-neutral
    /// controls are stored in one aggregate collection.
    pub targets: Vec<String>,
    pub options: Vec<IrProperty>,
    pub metrics: Vec<IrMeshMetric>,
    pub boundary_layers: Vec<IrBoundaryLayer>,
    pub refinements: Vec<IrRefineControl>,
    pub transfinite: Vec<IrTransfiniteControl>,
    pub embeds: Vec<IrEmbedControl>,
    pub compounds: Vec<IrCompoundControl>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrMeshMetric {
    pub target: Vec<EntityId>,
    pub tensor: IrMetricTensor,
    pub coordinate_system: Option<SymbolId>,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrMetricTensor {
    Isotropic(f64),
    /// Symmetric tensor `[xx, yy, zz, xy, xz, yz]`.
    Anisotropic([f64; 6]),
    Hessian {
        field: IrValue,
        minimum_size: f64,
        maximum_size: f64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrBoundaryLayer {
    pub target: Vec<EntityId>,
    pub first_layer_height: f64,
    pub total_thickness: f64,
    pub growth_ratio: f64,
    pub number_of_layers: Option<u32>,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrRefineControl {
    pub kind: IrRefineKind,
    pub body: Vec<IrProperty>,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrRefineKind {
    Near {
        entity: EntityId,
        args: Vec<IrProperty>,
    },
    On {
        entity: EntityId,
    },
    In {
        region: EntityId,
    },
    Field {
        args: Vec<IrProperty>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrTransfiniteControl {
    pub kind: IrTransfiniteKind,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrTransfiniteKind {
    Curve {
        entity: EntityId,
        settings: Vec<IrProperty>,
    },
    Surface {
        entity: EntityId,
        settings: Vec<IrProperty>,
    },
    Volume {
        entity: EntityId,
        settings: Vec<IrProperty>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrEmbedControl {
    pub source: EntityId,
    pub target: EntityId,
    pub source_dimension: EntityDimension,
    pub target_dimension: EntityDimension,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrCompoundControl {
    pub entities: Vec<EntityId>,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct IrOptimizationModel {
    pub variables: Vec<IrDesignVariable>,
    pub objectives: Vec<IrObjective>,
    pub constraints: Vec<IrConstraintRecord>,
    pub regions: Vec<IrDesignRegion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrConstraintRecord {
    pub name: Option<String>,
    pub constraint: IrConstraint,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrDesignVariable {
    pub id: SymbolId,
    pub name: String,
    pub kind: IrDesignVariableKind,
    pub initial: IrValue,
    pub bounds: Option<IrBounds>,
    pub distribution: Option<IrDistribution>,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IrDesignVariableKind {
    Continuous,
    Integer,
    Boolean,
    Categorical,
    Shape,
    Topology,
    LevelSet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrBounds {
    pub lower: IrValue,
    pub upper: IrValue,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrDistribution {
    Deterministic(IrValue),
    Uniform {
        lower: IrValue,
        upper: IrValue,
    },
    Normal {
        mean: IrValue,
        standard_deviation: IrValue,
    },
    LogNormal {
        mean: IrValue,
        standard_deviation: IrValue,
    },
    Discrete {
        values: Vec<IrValue>,
        weights: Vec<f64>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrObjective {
    pub name: Option<String>,
    pub sense: ObjectiveSense,
    pub value: IrValue,
    pub weight: f64,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrConstraint {
    Equality {
        left: IrValue,
        right: IrValue,
        tolerance: f64,
    },
    Inequality {
        left: IrValue,
        relation: InequalityRelation,
        right: IrValue,
    },
    Range {
        value: IrValue,
        bounds: IrBounds,
    },
    VolumeFraction {
        region: Vec<EntityId>,
        maximum: IrValue,
    },
    MinimumFeature {
        region: Vec<EntityId>,
        radius: IrValue,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrUncertainty {
    pub id: SymbolId,
    pub name: String,
    pub distribution: IrDistribution,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IrDesignRegion {
    pub name: String,
    pub kind: IrDesignRegionKind,
    pub entities: Vec<EntityId>,
    pub annotations: Vec<IrManufacturingAnnotation>,
    pub provenance: Vec<SourceReference>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum IrDesignRegionKind {
    Design,
    NonDesign,
    KeepOut,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IrManufacturingAnnotation {
    MinimumThickness(f64),
    MinimumRadius(f64),
    Overhang {
        maximum_angle: f64,
        direction: [f64; 3],
    },
    SymmetryPlane {
        origin: [f64; 3],
        normal: [f64; 3],
    },
    SymmetryAxis {
        origin: [f64; 3],
        direction: [f64; 3],
    },
    /// Rotational/cyclic design symmetry around an axis. `sectors` is the
    /// number of equal angular sectors in one full turn and is retained in
    /// the canonical IR so consumers can constrain or replicate the design
    /// region during meshing and optimization.
    SymmetryCyclic {
        axis: [f64; 3],
        sectors: u32,
    },
    DrawDirection([f64; 3]),
    Frozen,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SensitivityManifest {
    pub entries: Vec<SensitivityEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensitivityEntry {
    pub variable: SymbolId,
    pub parameters: Vec<SymbolId>,
    pub entities: Vec<EntityId>,
    pub physical_groups: Vec<PhysicalGroupId>,
    pub remesh: RemeshResponsibility,
    pub state_transfer: StateTransferResponsibility,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum RemeshResponsibility {
    NotRequired,
    Local,
    Global,
    TopologyDependent,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum StateTransferResponsibility {
    None,
    Interpolate,
    Conservative,
    ApplicationDefined,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrValidationError {
    UnsupportedSchema(IrSchemaVersion),
    SentinelEntityId(EntityId),
    DuplicateEntityId(EntityId),
    UnknownEntityReference {
        owner: EntityId,
        referenced: EntityId,
    },
    /// Entity inputs must form an acyclic parent graph.
    EntityParentCycle {
        entity: EntityId,
    },
    UnknownConditionalEntity {
        referenced: EntityId,
    },
    /// A backend-visible entity must retain at least one source ancestor.
    MissingEntityProvenance {
        entity: EntityId,
    },
    /// An entity source ancestor may not contain an inverted byte span.
    InvalidEntityProvenanceSpan {
        entity: EntityId,
    },
    /// Every backend-visible non-entity record must retain source ancestry.
    MissingProvenance {
        context: String,
    },
    /// A non-entity source ancestor may not contain an inverted byte span.
    InvalidProvenanceSpan {
        context: String,
    },
    /// A dynamic conditional may only refer to an earlier control as parent.
    InvalidConditionalParent {
        control: usize,
        parent: usize,
    },
    InvalidConditionalLoopParent {
        control: usize,
        parent_loop: usize,
    },
    /// An unconditional `else` branch must be the final branch.
    InvalidConditionalBranchOrder {
        control: usize,
        branch: usize,
    },
    /// An entity may belong to only one alternative branch of a control.
    DuplicateConditionalEntity {
        control: usize,
        entity: EntityId,
    },
    InvalidLoopParent {
        loop_index: usize,
        parent: usize,
    },
    InvalidLoopConditionalParent {
        loop_index: usize,
        parent_control: usize,
    },
    EmptyLoopClauses {
        loop_index: usize,
    },
    DuplicateLoopEntity {
        loop_index: usize,
        entity: EntityId,
    },
    UnknownLoopEntity {
        loop_index: usize,
        entity: EntityId,
    },
    EmptyLoopVariable {
        loop_index: usize,
    },
    BooleanArity(EntityId),
    DimensionMismatch {
        owner: EntityId,
        referenced: EntityId,
    },
    DuplicatePhysicalGroupId(PhysicalGroupId),
    PhysicalGroupDimension {
        group: PhysicalGroupId,
        entity: EntityId,
    },
    UnknownPhysicalGroupEntity {
        group: PhysicalGroupId,
        entity: EntityId,
    },
    BoundaryDimension {
        name: String,
        entity: EntityId,
    },
    UnknownBoundaryEntity {
        name: String,
        entity: EntityId,
    },
    PairDimensionMismatch {
        first: EntityId,
        second: EntityId,
    },
    UnknownReferencedEntity {
        context: String,
        entity: EntityId,
    },
    UnknownSensitivityPhysicalGroup {
        variable: SymbolId,
        group: PhysicalGroupId,
    },
    UnknownSensitivityVariable {
        variable: SymbolId,
    },
    UnknownSensitivityParameter {
        variable: SymbolId,
        parameter: SymbolId,
    },
    UnknownProjectedSourceField {
        field: SymbolId,
    },
    MeshControlDimension {
        context: String,
        expected: EntityDimension,
        found: EntityDimension,
        entity: EntityId,
    },
    /// A mesh compound control contained no entities.
    MeshCompoundEmpty {
        index: usize,
    },
    /// A mesh compound control mixed curves, surfaces, or volumes.
    MeshCompoundDimensionMismatch {
        index: usize,
        expected: EntityDimension,
        found: EntityDimension,
        entity: EntityId,
    },
}

impl fmt::Display for IrValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid canonical geometry IR: {self:?}")
    }
}

impl std::error::Error for IrValidationError {}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: EntityId, operation: GeometryOperation, inputs: Vec<EntityId>) -> GeometryNode {
        GeometryNode {
            id,
            name: format!("fixture::entity_{}", id.index),
            operation,
            inputs,
            arguments: Vec::new(),
            provenance: Vec::new(),
        }
    }

    #[test]
    fn canonical_ir_validates_and_hashes_deterministically() {
        let mut ir = CanonicalGeometryIr::new(SourceMetadata::for_source("fixture.mlg", "box"));
        let first = EntityId::new(EntityDimension::Volume, 1);
        let second = EntityId::new(EntityDimension::Volume, 2);
        let union = EntityId::new(EntityDimension::Volume, 3);
        ir.entities.push(node(
            first,
            GeometryOperation::Primitive {
                kind: PrimitiveKind::Box,
            },
            Vec::new(),
        ));
        ir.entities.push(node(
            second,
            GeometryOperation::Primitive {
                kind: PrimitiveKind::Sphere,
            },
            Vec::new(),
        ));
        ir.entities.push(node(
            union,
            GeometryOperation::Boolean {
                kind: BooleanOperation::Union,
            },
            vec![first, second],
        ));
        ir.physical_groups.push(PhysicalGroup {
            id: PhysicalGroupId(1),
            name: "domain".into(),
            dimension: EntityDimension::Volume,
            entities: vec![union],
            material: None,
            provenance: Vec::new(),
        });
        ir.validate().unwrap();
        assert_eq!(ir.structural_hash().unwrap(), ir.structural_hash().unwrap());

        let round_trip: CanonicalGeometryIr =
            serde_json::from_slice(&serde_json::to_vec(&ir).unwrap()).unwrap();
        assert_eq!(
            ir.structural_hash().unwrap(),
            round_trip.structural_hash().unwrap()
        );
    }

    #[test]
    fn versioned_canonical_ir_round_trips_and_migrates_raw_ir() {
        let ir = CanonicalGeometryIr::new(SourceMetadata::for_source("fixture.mlg", "box"));
        let document = VersionedCanonicalIr::current(ir.clone());
        let bytes = document.to_json_vec().unwrap();
        let (round_trip, report) = VersionedCanonicalIr::from_json_slice(&bytes).unwrap();
        assert!(!report.was_migrated());
        assert_eq!(
            round_trip.ir.structural_hash().unwrap(),
            ir.structural_hash().unwrap()
        );

        let legacy = serde_json::to_vec(&ir).unwrap();
        let (migrated, report) = VersionedCanonicalIr::from_json_slice(&legacy).unwrap();
        assert!(report.was_migrated());
        assert_eq!(report.steps[0].from, LEGACY_IR_DOCUMENT_VERSION);
        assert_eq!(report.steps[0].to, CURRENT_IR_DOCUMENT_VERSION);
        assert_eq!(
            migrated.ir.structural_hash().unwrap(),
            ir.structural_hash().unwrap()
        );
    }

    #[test]
    fn versioned_canonical_ir_rejects_invalid_versions_limits_and_ir() {
        let invalid_version = serde_json::json!({
            "document_version": 99,
            "ir": CanonicalGeometryIr::new(SourceMetadata::default()),
        });
        let error =
            VersionedCanonicalIr::from_json_slice(&serde_json::to_vec(&invalid_version).unwrap())
                .unwrap_err();
        assert!(matches!(
            error,
            IrPersistenceError::UnsupportedDocumentVersion(IrDocumentVersion(99))
        ));

        let error = VersionedCanonicalIr::from_json_slice_with_limit(b"{}", 1).unwrap_err();
        assert!(matches!(error, IrPersistenceError::ResourceLimit { .. }));

        let mut invalid = CanonicalGeometryIr::new(SourceMetadata::default());
        invalid.entities.push(node(
            EntityId::new(EntityDimension::Volume, 1),
            GeometryOperation::Boolean {
                kind: BooleanOperation::Union,
            },
            Vec::new(),
        ));
        let raw = serde_json::to_vec(&invalid).unwrap();
        let error = VersionedCanonicalIr::from_json_slice(&raw).unwrap_err();
        assert!(matches!(error, IrPersistenceError::InvalidIr(_)));
    }

    #[test]
    fn canonical_ir_rejects_entity_parent_cycles() {
        let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
        let first = EntityId::new(EntityDimension::Volume, 1);
        let second = EntityId::new(EntityDimension::Volume, 2);
        ir.entities.push(node(
            first,
            GeometryOperation::Transform {
                kind: TransformOperation::Translate,
            },
            vec![second],
        ));
        ir.entities.push(node(
            second,
            GeometryOperation::Transform {
                kind: TransformOperation::Translate,
            },
            vec![first],
        ));
        let errors = ir.validate().expect_err("cyclic entity parents must fail");
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, IrValidationError::EntityParentCycle { .. }))
        );
    }

    #[test]
    fn canonical_ir_rejects_unknown_and_dimension_wrong_boolean_inputs() {
        let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
        let output = EntityId::new(EntityDimension::Volume, 1);
        let missing_surface = EntityId::new(EntityDimension::Surface, 2);
        ir.entities.push(node(
            output,
            GeometryOperation::Boolean {
                kind: BooleanOperation::Difference,
            },
            vec![output, missing_surface],
        ));
        let errors = ir.validate().unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            IrValidationError::UnknownEntityReference { referenced, .. }
                if *referenced == missing_surface
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            IrValidationError::DimensionMismatch { referenced, .. }
                if *referenced == missing_surface
        )));
    }

    #[test]
    fn canonical_ir_rejects_empty_and_mixed_dimension_compounds() {
        let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
        let curve = EntityId::new(EntityDimension::Curve, 1);
        let surface = EntityId::new(EntityDimension::Surface, 2);
        ir.entities.push(node(
            curve,
            GeometryOperation::Primitive {
                kind: PrimitiveKind::Line,
            },
            Vec::new(),
        ));
        ir.entities.push(node(
            surface,
            GeometryOperation::Primitive {
                kind: PrimitiveKind::Rectangle,
            },
            Vec::new(),
        ));
        ir.mesh.compounds.push(IrCompoundControl {
            entities: Vec::new(),
            provenance: Vec::new(),
        });
        ir.mesh.compounds.push(IrCompoundControl {
            entities: vec![curve, surface],
            provenance: Vec::new(),
        });

        let errors = ir.validate().expect_err("invalid compound controls");
        assert!(
            errors
                .iter()
                .any(|error| matches!(error, IrValidationError::MeshCompoundEmpty { index: 0 }))
        );
        assert!(errors.iter().any(|error| matches!(
            error,
            IrValidationError::MeshCompoundDimensionMismatch {
                index: 1,
                expected: EntityDimension::Curve,
                found: EntityDimension::Surface,
                entity,
            } if *entity == surface
        )));
    }

    #[test]
    fn canonical_ir_rejects_unknown_metadata_and_sensitivity_references() {
        let mut ir = CanonicalGeometryIr::new(SourceMetadata::default());
        let known = EntityId::new(EntityDimension::Surface, 1);
        let missing = EntityId::new(EntityDimension::Surface, 2);
        ir.entities.push(node(
            known,
            GeometryOperation::Primitive {
                kind: PrimitiveKind::Rectangle,
            },
            Vec::new(),
        ));
        ir.interfaces.push(IrInterface {
            name: "coupling".into(),
            primary: known,
            secondary: missing,
            kind: IrInterfaceKind::Conforming,
            provenance: Vec::new(),
        });
        ir.fields.push(IrGeometryField {
            id: SymbolId(1),
            name: "projected".into(),
            kind: IrGeometryFieldKind::Projected {
                geometry: vec![known],
                source_field: SymbolId(99),
                method: ProjectionMethod::IsoSurface,
                tolerance: None,
            },
            provenance: Vec::new(),
        });
        ir.sensitivity.entries.push(SensitivityEntry {
            variable: SymbolId(7),
            parameters: Vec::new(),
            entities: vec![missing],
            physical_groups: vec![PhysicalGroupId(9)],
            remesh: RemeshResponsibility::Global,
            state_transfer: StateTransferResponsibility::Interpolate,
        });

        let errors = ir.validate().unwrap_err();
        assert!(errors.iter().any(|error| matches!(
            error,
            IrValidationError::UnknownReferencedEntity { entity, .. } if *entity == missing
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            IrValidationError::UnknownSensitivityPhysicalGroup { group, .. }
                if *group == PhysicalGroupId(9)
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            IrValidationError::UnknownSensitivityVariable { variable }
                if *variable == SymbolId(7)
        )));
        assert!(errors.iter().any(|error| matches!(
            error,
            IrValidationError::UnknownProjectedSourceField { field }
                if *field == SymbolId(99)
        )));
    }
}
