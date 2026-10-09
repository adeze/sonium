//! Backend-neutral intermediate representation for coupled multiphysics data.
//!
//! The IR contains metadata and references to field buffers rather than owning
//! solver arrays. A BEM, FEM, GPU, or WASM backend can therefore consume the
//! same schema without copying a large field just to exchange its provenance.

use crate::conventions::{ConventionRegistry, CoordinateFrame, SchemaVersion};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

/// Current version of the multiphysics IR payload.
pub const MULTIPHYSICS_IR_VERSION: u32 = 1;

/// A stable, backend-neutral multiphysics problem description.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MultiphysicsProblem {
    /// Schema version of this document.
    pub schema_version: SchemaVersion,
    /// Stable problem identifier.
    pub id: String,
    /// Shared phasor, orientation, and amplitude conventions.
    pub conventions: ConventionRegistry,
    /// Physical domains represented by the problem.
    pub domains: Vec<Domain>,
    /// Geometric or material regions within those domains.
    pub regions: Vec<Region>,
    /// Unknown or derived fields.
    pub fields: Vec<FieldSpec>,
    /// Boundary conditions applied to fields.
    pub boundary_conditions: Vec<BoundaryCondition>,
    /// Conditions coupling two regions across an interface.
    pub interface_conditions: Vec<InterfaceCondition>,
    /// Constitutive or source laws referenced by regions and conditions.
    pub material_laws: Vec<MaterialLaw>,
    /// Field-to-field or domain-to-domain couplings.
    pub couplings: Vec<Coupling>,
    /// Shape/topology design variables.
    pub design_variables: Vec<DesignVariable>,
    /// Optimization objectives.
    pub objectives: Vec<Objective>,
    /// Optimization or physical constraints.
    pub constraints: Vec<Constraint>,
    /// Origin and fidelity of the problem data.
    pub provenance: Provenance,
}

/// Wire representation for a problem received through serde.
///
/// The IR remains convenient to construct in-process, while external payloads
/// must pass the same schema/reference validation required at solver entry.
#[derive(Debug, Deserialize)]
struct MultiphysicsProblemWire {
    schema_version: SchemaVersion,
    id: String,
    conventions: ConventionRegistry,
    domains: Vec<Domain>,
    regions: Vec<Region>,
    fields: Vec<FieldSpec>,
    boundary_conditions: Vec<BoundaryCondition>,
    interface_conditions: Vec<InterfaceCondition>,
    material_laws: Vec<MaterialLaw>,
    couplings: Vec<Coupling>,
    design_variables: Vec<DesignVariable>,
    objectives: Vec<Objective>,
    constraints: Vec<Constraint>,
    provenance: Provenance,
}

impl<'de> Deserialize<'de> for MultiphysicsProblem {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = MultiphysicsProblemWire::deserialize(deserializer)?;
        let problem = Self {
            schema_version: wire.schema_version,
            id: wire.id,
            conventions: wire.conventions,
            domains: wire.domains,
            regions: wire.regions,
            fields: wire.fields,
            boundary_conditions: wire.boundary_conditions,
            interface_conditions: wire.interface_conditions,
            material_laws: wire.material_laws,
            couplings: wire.couplings,
            design_variables: wire.design_variables,
            objectives: wire.objectives,
            constraints: wire.constraints,
            provenance: wire.provenance,
        };
        problem
            .validate()
            .map_err(|error| <D::Error as serde::de::Error>::custom(error.to_string()))?;
        Ok(problem)
    }
}

impl Default for MultiphysicsProblem {
    fn default() -> Self {
        Self {
            schema_version: SchemaVersion(MULTIPHYSICS_IR_VERSION),
            id: String::new(),
            conventions: ConventionRegistry::default(),
            domains: Vec::new(),
            regions: Vec::new(),
            fields: Vec::new(),
            boundary_conditions: Vec::new(),
            interface_conditions: Vec::new(),
            material_laws: Vec::new(),
            couplings: Vec::new(),
            design_variables: Vec::new(),
            objectives: Vec::new(),
            constraints: Vec::new(),
            provenance: Provenance::default(),
        }
    }
}

/// Validate an IR problem before handing it to a solver backend.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum IrValidationError {
    /// An entity has no stable identifier.
    EmptyId { kind: &'static str },
    /// An identifier occurs more than once in the same entity namespace.
    DuplicateId { kind: &'static str, id: String },
    /// A reference points to an entity that is not present.
    MissingReference { kind: &'static str, id: String },
    /// A field buffer layout cannot describe a strided tensor.
    InvalidLayout(String),
    /// A coupled-model value or metadata field violates its semantic contract.
    InvalidValue { kind: &'static str, reason: String },
    /// The payload has an unsupported schema version.
    UnsupportedSchemaVersion(SchemaVersion),
}

impl std::fmt::Display for IrValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmptyId { kind } => write!(f, "{kind} has an empty id"),
            Self::DuplicateId { kind, id } => write!(f, "duplicate {kind} id: {id}"),
            Self::MissingReference { kind, id } => write!(f, "missing {kind} reference: {id}"),
            Self::InvalidLayout(message) => write!(f, "invalid field layout: {message}"),
            Self::InvalidValue { kind, reason } => write!(f, "invalid {kind}: {reason}"),
            Self::UnsupportedSchemaVersion(version) => {
                write!(
                    f,
                    "unsupported multiphysics IR schema version: {}",
                    version.0
                )
            }
        }
    }
}

impl std::error::Error for IrValidationError {}

impl MultiphysicsProblem {
    /// Validate schema, IDs, references, and field layouts.
    pub fn validate(&self) -> Result<(), IrValidationError> {
        if self.id.trim().is_empty() {
            return Err(IrValidationError::EmptyId { kind: "problem" });
        }
        if self.schema_version.0 != MULTIPHYSICS_IR_VERSION {
            return Err(IrValidationError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        validate_ids("domain", self.domains.iter().map(|item| item.id.as_str()))?;
        validate_ids("region", self.regions.iter().map(|item| item.id.as_str()))?;
        validate_ids("field", self.fields.iter().map(|item| item.id.as_str()))?;
        validate_ids(
            "material law",
            self.material_laws.iter().map(|item| item.id.as_str()),
        )?;
        validate_ids(
            "boundary condition",
            self.boundary_conditions.iter().map(|item| item.id.as_str()),
        )?;
        validate_ids(
            "interface condition",
            self.interface_conditions
                .iter()
                .map(|item| item.id.as_str()),
        )?;
        validate_ids(
            "coupling",
            self.couplings.iter().map(|item| item.id.as_str()),
        )?;
        validate_ids(
            "design variable",
            self.design_variables.iter().map(|item| item.id.as_str()),
        )?;
        validate_ids(
            "objective",
            self.objectives.iter().map(|item| item.id.as_str()),
        )?;
        validate_ids(
            "constraint",
            self.constraints.iter().map(|item| item.id.as_str()),
        )?;
        for field in &self.fields {
            field.layout.validate()?;
            if !self
                .domains
                .iter()
                .any(|domain| domain.id == field.domain_id)
            {
                return Err(IrValidationError::MissingReference {
                    kind: "domain",
                    id: field.domain_id.clone(),
                });
            }
        }
        for region in &self.regions {
            if !self
                .domains
                .iter()
                .any(|domain| domain.id == region.domain_id)
            {
                return Err(IrValidationError::MissingReference {
                    kind: "domain",
                    id: region.domain_id.clone(),
                });
            }
            if let Some(material_id) = &region.material_law_id
                && !self.material_laws.iter().any(|law| &law.id == material_id)
            {
                return Err(IrValidationError::MissingReference {
                    kind: "material law",
                    id: material_id.clone(),
                });
            }
        }
        for law in &self.material_laws {
            if law
                .parameters
                .iter()
                .any(|(name, value)| name.trim().is_empty() || !value.is_finite())
            {
                return Err(IrValidationError::InvalidValue {
                    kind: "material law",
                    reason: format!(
                        "{} contains an empty parameter name or non-finite parameter",
                        law.id
                    ),
                });
            }
        }
        for condition in &self.boundary_conditions {
            require_reference(
                "region",
                &condition.region_id,
                self.regions.iter().map(|item| item.id.as_str()),
            )?;
            require_reference(
                "field",
                &condition.field_id,
                self.fields.iter().map(|item| item.id.as_str()),
            )?;
            if let Some(value) = &condition.value {
                validate_data_reference(value)?;
            }
        }
        for condition in &self.interface_conditions {
            require_reference(
                "region",
                &condition.left_region_id,
                self.regions.iter().map(|item| item.id.as_str()),
            )?;
            require_reference(
                "region",
                &condition.right_region_id,
                self.regions.iter().map(|item| item.id.as_str()),
            )?;
            if condition.left_region_id == condition.right_region_id {
                return Err(IrValidationError::InvalidValue {
                    kind: "interface condition",
                    reason: format!("{} must connect two distinct regions", condition.id),
                });
            }
            if condition.field_ids.is_empty() {
                return Err(IrValidationError::InvalidValue {
                    kind: "interface condition",
                    reason: format!("{} must reference at least one field", condition.id),
                });
            }
            for field_id in &condition.field_ids {
                require_reference(
                    "field",
                    field_id,
                    self.fields.iter().map(|item| item.id.as_str()),
                )?;
            }
            if let Some(transfer) = &condition.transfer {
                validate_transfer_metadata(transfer)?;
            }
        }
        for coupling in &self.couplings {
            require_reference(
                "field",
                &coupling.source_field_id,
                self.fields.iter().map(|item| item.id.as_str()),
            )?;
            require_reference(
                "field",
                &coupling.target_field_id,
                self.fields.iter().map(|item| item.id.as_str()),
            )?;
            if coupling.source_field_id == coupling.target_field_id {
                return Err(IrValidationError::InvalidValue {
                    kind: "coupling",
                    reason: format!("{} must connect distinct fields", coupling.id),
                });
            }
            if coupling
                .conservation
                .as_deref()
                .is_some_and(|quantity| quantity.trim().is_empty())
            {
                return Err(IrValidationError::InvalidValue {
                    kind: "coupling",
                    reason: format!("{} has an empty conserved quantity", coupling.id),
                });
            }
        }
        for design in &self.design_variables {
            if design.region_ids.is_empty() {
                return Err(IrValidationError::InvalidValue {
                    kind: "design variable",
                    reason: format!("{} must reference at least one region", design.id),
                });
            }
            for region_id in &design.region_ids {
                require_reference(
                    "region",
                    region_id,
                    self.regions.iter().map(|item| item.id.as_str()),
                )?;
            }
            validate_bounds("design variable", &design.id, design.bounds)?;
        }
        for objective in &self.objectives {
            if objective.target.trim().is_empty() || !objective.weight.is_finite() {
                return Err(IrValidationError::InvalidValue {
                    kind: "objective",
                    reason: format!("{} has an empty target or non-finite weight", objective.id),
                });
            }
        }
        for constraint in &self.constraints {
            if constraint.target.trim().is_empty() {
                return Err(IrValidationError::InvalidValue {
                    kind: "constraint",
                    reason: format!("{} has an empty target", constraint.id),
                });
            }
            validate_bounds("constraint", &constraint.id, constraint.bounds)?;
        }
        validate_provenance(&self.provenance, &self.regions, &self.fields)?;
        Ok(())
    }
}

fn require_reference<'a>(
    kind: &'static str,
    id: &str,
    ids: impl IntoIterator<Item = &'a str>,
) -> Result<(), IrValidationError> {
    if ids.into_iter().any(|candidate| candidate == id) {
        Ok(())
    } else {
        Err(IrValidationError::MissingReference {
            kind,
            id: id.to_owned(),
        })
    }
}

fn validate_bounds(
    kind: &'static str,
    id: &str,
    bounds: Option<[f64; 2]>,
) -> Result<(), IrValidationError> {
    if let Some([lower, upper]) = bounds
        && (!lower.is_finite() || !upper.is_finite() || lower > upper)
    {
        return Err(IrValidationError::InvalidValue {
            kind,
            reason: format!("{id} has invalid bounds"),
        });
    }
    Ok(())
}

fn validate_data_reference(reference: &DataReference) -> Result<(), IrValidationError> {
    let has_expression = reference
        .expression
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty());
    let has_uri = reference
        .uri
        .as_deref()
        .is_some_and(|value| !value.trim().is_empty());
    if !has_expression && !has_uri {
        return Err(IrValidationError::InvalidValue {
            kind: "data reference",
            reason: "an expression or URI is required".into(),
        });
    }
    Ok(())
}

fn validate_transfer_metadata(transfer: &TransferMetadata) -> Result<(), IrValidationError> {
    if transfer.method.trim().is_empty()
        || transfer
            .conserved_quantity
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
    {
        return Err(IrValidationError::InvalidValue {
            kind: "transfer metadata",
            reason: "method and conserved quantity names must be non-empty".into(),
        });
    }
    Ok(())
}

fn validate_provenance(
    provenance: &Provenance,
    regions: &[Region],
    fields: &[FieldSpec],
) -> Result<(), IrValidationError> {
    validate_provenance_metadata(provenance)?;
    for mask in &provenance.design_region_masks {
        require_reference(
            "region",
            &mask.region_id,
            regions.iter().map(|item| item.id.as_str()),
        )?;
        if let Some(field_id) = &mask.field_id {
            require_reference(
                "field",
                field_id,
                fields.iter().map(|item| item.id.as_str()),
            )?;
        }
        if mask.semantics.trim().is_empty() {
            return Err(IrValidationError::InvalidValue {
                kind: "design region mask",
                reason: "semantics must be non-empty".into(),
            });
        }
    }
    Ok(())
}

fn validate_provenance_metadata(provenance: &Provenance) -> Result<(), IrValidationError> {
    if let Some(execution) = &provenance.execution {
        execution.validate()?;
    }
    if let Some(lineage) = &provenance.mesh_lineage
        && (lineage.mesh_id.trim().is_empty()
            || lineage.relation.trim().is_empty()
            || lineage
                .parent_mesh_id
                .as_deref()
                .is_some_and(|value| value.trim().is_empty()))
    {
        return Err(IrValidationError::InvalidValue {
            kind: "mesh lineage",
            reason: "mesh id and relation must be non-empty".into(),
        });
    }
    if let Some(transfer) = &provenance.state_transfer
        && (transfer.source_mesh_id.trim().is_empty()
            || transfer.target_mesh_id.trim().is_empty()
            || transfer.method.trim().is_empty())
    {
        return Err(IrValidationError::InvalidValue {
            kind: "state transfer",
            reason: "source, target, and method must be non-empty".into(),
        });
    }
    if provenance.derivative_convention.method == DerivativeMethod::None
        && provenance.derivative_convention.includes_mesh_motion
    {
        return Err(IrValidationError::InvalidValue {
            kind: "derivative convention",
            reason: "mesh-motion terms require a derivative method".into(),
        });
    }
    Ok(())
}

fn validate_ids<'a>(
    kind: &'static str,
    ids: impl IntoIterator<Item = &'a str>,
) -> Result<(), IrValidationError> {
    let mut seen = std::collections::HashSet::new();
    for id in ids {
        if id.trim().is_empty() {
            return Err(IrValidationError::EmptyId { kind });
        }
        if !seen.insert(id) {
            return Err(IrValidationError::DuplicateId {
                kind,
                id: id.to_owned(),
            });
        }
    }
    Ok(())
}

impl MultiphysicsProblem {
    /// Find a field declaration by its stable identifier.
    pub fn field(&self, field_id: &str) -> Option<&FieldSpec> {
        self.fields.iter().find(|field| field.id == field_id)
    }

    /// Validate and return the canonical complex acoustic-pressure field.
    ///
    /// BEM and Helmholtz FEM both exchange complex pressure in pascals.  Keep
    /// that contract in the XEM layer so consumers cannot accidentally accept
    /// a structurally valid scalar field from a structural domain, a real
    /// buffer, or an unlabelled/non-pressure quantity and only discover the
    /// mismatch after assembly or result serialization.
    pub fn acoustic_complex_pressure_field(
        &self,
        field_id: &str,
    ) -> Result<&FieldSpec, IrValidationError> {
        self.validate()?;
        let field = self
            .field(field_id)
            .ok_or_else(|| IrValidationError::MissingReference {
                kind: "problem field",
                id: field_id.to_owned(),
            })?;
        let acoustics_domain = self
            .domains
            .iter()
            .find(|domain| matches!(domain.kind, DomainKind::Acoustics));
        if acoustics_domain.is_none_or(|domain| domain.id != field.domain_id) {
            return Err(IrValidationError::InvalidValue {
                kind: "field",
                reason: format!("field `{field_id}` must belong to an acoustics domain"),
            });
        }
        if !matches!(field.kind, FieldKind::ComplexScalar)
            || field.layout.scalar_type != ScalarType::ComplexF64
        {
            return Err(IrValidationError::InvalidValue {
                kind: "field",
                reason: format!("field `{field_id}` must use ComplexScalar/ComplexF64 storage"),
            });
        }
        let pressure_units = field.units.as_deref().is_some_and(|units| {
            units.trim().eq_ignore_ascii_case("pa")
                || units.trim().eq_ignore_ascii_case("pascal")
                || units.trim().eq_ignore_ascii_case("pascals")
        });
        if !pressure_units {
            return Err(IrValidationError::InvalidValue {
                kind: "field",
                reason: format!("field `{field_id}` must declare pressure units (Pa)"),
            });
        }
        Ok(field)
    }
}

/// Physical or mathematical domain in a coupled problem.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Domain {
    /// Stable identifier.
    pub id: String,
    /// Human-readable label.
    pub label: String,
    /// Governing-physics family.
    pub kind: DomainKind,
}

/// Governing-physics family for a domain.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DomainKind {
    /// Linear or nonlinear acoustics.
    Acoustics,
    /// Solid or structural mechanics.
    Structural,
    /// Heat or mass transport.
    Thermal,
    /// Fluid or aeroacoustic flow.
    Fluid,
    /// Maxwell or quasi-static electromagnetics.
    Electromagnetic,
    /// A physics family not yet standardized by XEM.
    Custom(String),
}

/// Geometric/material region belonging to a domain.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Region {
    /// Stable identifier.
    pub id: String,
    /// Owning domain identifier.
    pub domain_id: String,
    /// Optional human-readable label.
    pub label: Option<String>,
    /// Parent CAD/geometry entity identifier.
    pub parent_geometry_entity: Option<String>,
    /// Referenced constitutive law.
    pub material_law_id: Option<String>,
}

/// A field exchanged between physics modules or stored in a result.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldSpec {
    /// Stable field identifier.
    pub id: String,
    /// Human-readable field name.
    pub name: String,
    /// Owning domain identifier.
    pub domain_id: String,
    /// Physical role of the field.
    pub kind: FieldKind,
    /// Optional SI or domain-specific unit string.
    pub units: Option<String>,
    /// Shape, scalar type, and memory layout of the values.
    pub layout: TensorLayout,
}

/// Semantic role of a field.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FieldKind {
    /// Scalar real-valued field.
    Scalar,
    /// Scalar complex phasor field.
    ComplexScalar,
    /// Vector-valued field.
    Vector,
    /// Complex vector-valued field.
    ComplexVector,
    /// Higher-order tensor field.
    Tensor,
    /// Application-specific field role.
    Custom(String),
}

/// Boundary condition attached to a region and field.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BoundaryCondition {
    /// Stable condition identifier.
    pub id: String,
    /// Region on which the condition is applied.
    pub region_id: String,
    /// Field constrained by the condition.
    pub field_id: String,
    /// Mathematical condition family.
    pub kind: BoundaryConditionKind,
    /// Optional symbolic expression or buffer reference for its value.
    pub value: Option<DataReference>,
}

/// Boundary condition family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum BoundaryConditionKind {
    /// Essential/value condition.
    Dirichlet,
    /// Natural/flux condition.
    Neumann,
    /// Mixed condition.
    Robin,
    /// Acoustic impedance or modal condition.
    Impedance,
    /// Periodic identification.
    Periodic,
    /// Application-specific condition.
    Custom(String),
}

/// Coupling condition between two regions across an interface.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InterfaceCondition {
    /// Stable condition identifier.
    pub id: String,
    /// First region identifier.
    pub left_region_id: String,
    /// Second region identifier.
    pub right_region_id: String,
    /// Coupled field identifiers in left/right order.
    pub field_ids: Vec<String>,
    /// Interface law.
    pub kind: InterfaceConditionKind,
    /// Transfer/conservation metadata.
    pub transfer: Option<TransferMetadata>,
}

/// Interface law family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum InterfaceConditionKind {
    /// Continuity of a field and its flux.
    Continuity,
    /// Conservative mortar or projection transfer.
    Mortar,
    /// Contact or impedance interface.
    Impedance,
    /// Application-specific interface law.
    Custom(String),
}

/// Constitutive or source law.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MaterialLaw {
    /// Stable identifier.
    pub id: String,
    /// Law family.
    pub kind: MaterialLawKind,
    /// Deterministic scalar parameters in SI units where applicable.
    #[serde(default)]
    pub parameters: BTreeMap<String, f64>,
}

/// Material-law family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum MaterialLawKind {
    /// Constant isotropic coefficients.
    Isotropic,
    /// Direction-dependent coefficients.
    Anisotropic,
    /// Frequency-dependent coefficients.
    FrequencyDependent,
    /// Symbolic or user-defined law.
    Expression(String),
}

/// Field or domain coupling edge.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Coupling {
    /// Stable coupling identifier.
    pub id: String,
    /// Source field identifier.
    pub source_field_id: String,
    /// Target field identifier.
    pub target_field_id: String,
    /// Coupling family.
    pub kind: CouplingKind,
    /// Whether the transfer is required to conserve the named quantity.
    pub conservation: Option<String>,
}

/// Coupling family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum CouplingKind {
    /// Strong monolithic residual/Jacobian coupling.
    Monolithic,
    /// Partitioned fixed-point or staggered coupling.
    Partitioned,
    /// One-way source transfer.
    OneWay,
    /// Application-specific coupling.
    Custom(String),
}

/// A validated field-coupling graph derived from a multiphysics problem.
///
/// Monolithic edges describe a single residual/Jacobian system and therefore
/// do not impose an execution order. Partitioned, one-way, and custom edges
/// are directed dependencies and must form a DAG before a partitioned solver
/// can schedule them.
#[derive(Debug, Clone, PartialEq)]
pub struct CouplingGraph {
    field_ids: Vec<String>,
    couplings: Vec<Coupling>,
}

/// Failure while constructing or scheduling a coupling graph.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CouplingGraphError {
    /// The source problem failed its schema/reference validation.
    InvalidProblem(String),
    /// The source or target field is not in the graph.
    MissingField(String),
    /// Two edges reuse the same stable coupling identifier.
    DuplicateCoupling(String),
    /// Directed partitioned dependencies contain a cycle.
    Cycle(Vec<String>),
}

impl std::fmt::Display for CouplingGraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidProblem(reason) => write!(f, "invalid coupling problem: {reason}"),
            Self::MissingField(id) => write!(f, "coupling graph references missing field {id}"),
            Self::DuplicateCoupling(id) => write!(f, "duplicate coupling id {id}"),
            Self::Cycle(nodes) => write!(f, "partitioned coupling cycle: {}", nodes.join(" -> ")),
        }
    }
}

impl std::error::Error for CouplingGraphError {}

impl CouplingGraph {
    /// Validate and build a graph from a multiphysics problem.
    pub fn from_problem(problem: &MultiphysicsProblem) -> Result<Self, CouplingGraphError> {
        problem
            .validate()
            .map_err(|error| CouplingGraphError::InvalidProblem(error.to_string()))?;
        let field_ids = problem
            .fields
            .iter()
            .map(|field| field.id.clone())
            .collect::<Vec<_>>();
        let field_set = field_ids.iter().collect::<std::collections::HashSet<_>>();
        let mut coupling_ids = std::collections::HashSet::new();
        for coupling in &problem.couplings {
            if !field_set.contains(&coupling.source_field_id)
                || !field_set.contains(&coupling.target_field_id)
            {
                return Err(CouplingGraphError::MissingField(
                    if !field_set.contains(&coupling.source_field_id) {
                        coupling.source_field_id.clone()
                    } else {
                        coupling.target_field_id.clone()
                    },
                ));
            }
            if !coupling_ids.insert(coupling.id.as_str()) {
                return Err(CouplingGraphError::DuplicateCoupling(coupling.id.clone()));
            }
        }
        Ok(Self {
            field_ids,
            couplings: problem.couplings.clone(),
        })
    }

    /// Field identifiers in stable problem order.
    pub fn field_ids(&self) -> &[String] {
        &self.field_ids
    }

    /// Coupling edges in stable problem order.
    pub fn couplings(&self) -> &[Coupling] {
        &self.couplings
    }

    /// Return outgoing edges from a field.
    pub fn outgoing(&self, field_id: &str) -> Vec<&Coupling> {
        self.couplings
            .iter()
            .filter(|coupling| coupling.source_field_id == field_id)
            .collect()
    }

    /// Return a deterministic execution order for partitioned dependencies.
    ///
    /// Monolithic edges are omitted because their fields are solved together.
    /// The returned order contains every field, including isolated and
    /// monolithic-only fields.
    pub fn partitioned_order(&self) -> Result<Vec<String>, CouplingGraphError> {
        let mut indegree = std::collections::BTreeMap::new();
        let mut outgoing = std::collections::BTreeMap::<String, Vec<String>>::new();
        for field_id in &self.field_ids {
            indegree.insert(field_id.clone(), 0usize);
        }
        for coupling in &self.couplings {
            if matches!(coupling.kind, CouplingKind::Monolithic) {
                continue;
            }
            outgoing
                .entry(coupling.source_field_id.clone())
                .or_default()
                .push(coupling.target_field_id.clone());
            *indegree
                .get_mut(&coupling.target_field_id)
                .expect("graph construction validates target fields") += 1;
        }

        let mut ready = indegree
            .iter()
            .filter_map(|(field_id, &degree)| (degree == 0).then_some(field_id.clone()))
            .collect::<std::collections::BTreeSet<_>>();
        let mut order = Vec::with_capacity(self.field_ids.len());
        while let Some(field_id) = ready.pop_first() {
            order.push(field_id.clone());
            if let Some(targets) = outgoing.get(&field_id) {
                for target in targets {
                    let degree = indegree
                        .get_mut(target)
                        .expect("graph construction validates target fields");
                    *degree -= 1;
                    if *degree == 0 {
                        ready.insert(target.clone());
                    }
                }
            }
        }
        if order.len() != self.field_ids.len() {
            return Err(CouplingGraphError::Cycle(
                indegree
                    .into_iter()
                    .filter_map(|(field_id, degree)| (degree > 0).then_some(field_id))
                    .collect(),
            ));
        }
        Ok(order)
    }
}

/// Shape or topology design variable.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignVariable {
    /// Stable identifier.
    pub id: String,
    /// Design-variable family.
    pub kind: DesignVariableKind,
    /// Regions controlled by this variable.
    pub region_ids: Vec<String>,
    /// Optional lower/upper bounds.
    pub bounds: Option<[f64; 2]>,
    /// Derivative convention for this variable.
    pub derivative_convention: DerivativeConvention,
}

/// Shape/topology design-variable family.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DesignVariableKind {
    /// CAD or mesh geometry displacement.
    Shape,
    /// Element density or material interpolation value.
    Density,
    /// Projected/filtered topology field.
    ProjectedDensity,
    /// Discrete topology or region selection.
    Topology,
    /// Application-specific design variable.
    Custom(String),
}

/// Optimization objective.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Objective {
    /// Stable identifier.
    pub id: String,
    /// Field or metric being optimized.
    pub target: String,
    /// Minimize or maximize.
    pub sense: ObjectiveSense,
    /// Scalar weighting in a composite objective.
    pub weight: f64,
}

/// Objective direction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ObjectiveSense {
    /// Minimize the target.
    Minimize,
    /// Maximize the target.
    Maximize,
}

/// Optimization or physical constraint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Constraint {
    /// Stable identifier.
    pub id: String,
    /// Field or metric constrained.
    pub target: String,
    /// Optional lower and upper limits.
    pub bounds: Option<[f64; 2]>,
}

/// Fidelity and lineage information shared by problems and results.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Provenance {
    /// Fidelity category of the payload.
    pub fidelity: FidelityTag,
    /// Producer or measurement pipeline.
    pub producer: Option<String>,
    /// Code revision that generated the payload.
    pub code_revision: Option<String>,
    /// Execution environment and numerical settings for generated artifacts.
    #[serde(default)]
    pub execution: Option<ExecutionMetadata>,
    /// Parent geometry/design entity, when derived from one.
    pub parent_geometry_entity: Option<String>,
    /// Design-region masks used to generate the payload.
    #[serde(default)]
    pub design_region_masks: Vec<DesignRegionMask>,
    /// Mesh refinement/remeshing lineage.
    pub mesh_lineage: Option<MeshLineage>,
    /// State transfer metadata after remeshing or reduction.
    pub state_transfer: Option<StateTransferMetadata>,
    /// Derivative semantics for attached sensitivities.
    pub derivative_convention: DerivativeConvention,
}

impl Provenance {
    /// Validate the provenance required for a release-quality artifact.
    ///
    /// The ordinary IR validator accepts intentionally small in-memory
    /// fixtures with omitted provenance.  Exporters and QA promotion gates
    /// should call this stricter entry point: every generated or measured
    /// artifact must identify its producer and code/data revision, and a
    /// remeshed or transferred state must carry an explicit lineage record.
    pub fn validate_for_release(&self) -> Result<(), IrValidationError> {
        validate_provenance_metadata(self)?;
        if self
            .producer
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
        {
            return Err(IrValidationError::InvalidValue {
                kind: "provenance",
                reason: "release artifacts require a non-empty producer".into(),
            });
        }
        if self.fidelity != FidelityTag::Measured && self.execution.is_none() {
            return Err(IrValidationError::InvalidValue {
                kind: "provenance",
                reason: "generated release artifacts require execution metadata".into(),
            });
        }
        if !matches!(
            self.fidelity,
            FidelityTag::Analytical | FidelityTag::Measured
        ) && self.mesh_lineage.is_none()
        {
            return Err(IrValidationError::InvalidValue {
                kind: "provenance",
                reason: "discretized release artifacts require mesh lineage".into(),
            });
        }
        if self
            .code_revision
            .as_deref()
            .is_none_or(|value| value.trim().is_empty())
        {
            return Err(IrValidationError::InvalidValue {
                kind: "provenance",
                reason: "release artifacts require a non-empty code_revision".into(),
            });
        }
        if self
            .parent_geometry_entity
            .as_deref()
            .is_some_and(|value| value.trim().is_empty())
        {
            return Err(IrValidationError::InvalidValue {
                kind: "provenance",
                reason: "parent_geometry_entity must be non-empty when present".into(),
            });
        }
        let mut regions = BTreeSet::new();
        for mask in &self.design_region_masks {
            if !regions.insert(mask.region_id.as_str()) {
                return Err(IrValidationError::DuplicateId {
                    kind: "design region mask",
                    id: mask.region_id.clone(),
                });
            }
        }
        if let Some(lineage) = &self.mesh_lineage
            && lineage
                .parent_mesh_id
                .as_deref()
                .is_some_and(|parent| parent == lineage.mesh_id)
        {
            return Err(IrValidationError::InvalidValue {
                kind: "mesh lineage",
                reason: "parent mesh must differ from current mesh".into(),
            });
        }
        if let Some(transfer) = &self.state_transfer
            && transfer.source_mesh_id == transfer.target_mesh_id
        {
            return Err(IrValidationError::InvalidValue {
                kind: "state transfer",
                reason: "source and target meshes must differ".into(),
            });
        }
        Ok(())
    }
}

impl Default for Provenance {
    fn default() -> Self {
        Self {
            fidelity: FidelityTag::Simulated,
            producer: None,
            code_revision: None,
            execution: None,
            parent_geometry_entity: None,
            design_region_masks: Vec::new(),
            mesh_lineage: None,
            state_transfer: None,
            derivative_convention: DerivativeConvention::default(),
        }
    }
}

/// Platform, solver, and numerical settings needed to reproduce an artifact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ExecutionMetadata {
    /// Operating system or runtime platform that produced the artifact.
    pub platform: String,
    /// Solver or pipeline identity.
    pub solver: String,
    /// Named finite, non-negative numerical tolerances or setup controls.
    #[serde(default)]
    pub tolerances: BTreeMap<String, f64>,
}

impl ExecutionMetadata {
    /// Validate reproducibility metadata supplied by a generating backend.
    pub fn validate(&self) -> Result<(), IrValidationError> {
        if self.platform.trim().is_empty() || self.solver.trim().is_empty() {
            return Err(IrValidationError::InvalidValue {
                kind: "execution metadata",
                reason: "platform and solver must be non-empty".into(),
            });
        }
        if self.tolerances.is_empty() {
            return Err(IrValidationError::InvalidValue {
                kind: "execution metadata",
                reason: "release artifacts require at least one tolerance or setup control".into(),
            });
        }
        for (name, value) in &self.tolerances {
            if name.trim().is_empty() || !value.is_finite() || *value < 0.0 {
                return Err(IrValidationError::InvalidValue {
                    kind: "execution metadata",
                    reason: "tolerance names must be non-empty and values finite/non-negative"
                        .into(),
                });
            }
        }
        Ok(())
    }
}

/// Fidelity category for a field, problem, or result.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum FidelityTag {
    /// Closed-form or analytical reference.
    Analytical,
    /// Reduced-order or modal model.
    Reduced,
    /// Numerical simulation.
    Simulated,
    /// Extrapolated beyond the calibrated domain.
    Extrapolated,
    /// Artificially generated data.
    Synthetic,
    /// Physical measurement.
    Measured,
}

/// Mask or density field identifying a design region.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignRegionMask {
    /// Region identifier.
    pub region_id: String,
    /// Optional field carrying the mask values.
    pub field_id: Option<String>,
    /// Semantics such as `active`, `frozen`, `eroded`, or `dilated`.
    pub semantics: String,
}

/// Parent/child mesh relationship.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshLineage {
    /// Current mesh identifier.
    pub mesh_id: String,
    /// Parent mesh identifier, if this is a refinement/remesh.
    pub parent_mesh_id: Option<String>,
    /// Human-readable relation such as `refined`, `coarsened`, or `remeshed`.
    pub relation: String,
}

/// State transfer performed between two discretizations.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateTransferMetadata {
    /// Source mesh identifier.
    pub source_mesh_id: String,
    /// Target mesh identifier.
    pub target_mesh_id: String,
    /// Transfer method, e.g. conservative mortar or interpolation.
    pub method: String,
    /// Quantity claimed to be conserved, when applicable.
    pub conserved_quantity: Option<String>,
}

/// Derivative semantics for geometry, material, source, or state sensitivities.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DerivativeConvention {
    /// Differentiation method.
    pub method: DerivativeMethod,
    /// Coordinates in which derivatives are expressed.
    pub coordinate_frame: CoordinateFrame,
    /// Whether mesh-motion terms are included.
    pub includes_mesh_motion: bool,
}

impl Default for DerivativeConvention {
    fn default() -> Self {
        Self {
            method: DerivativeMethod::None,
            coordinate_frame: CoordinateFrame::WorldRightHanded,
            includes_mesh_motion: false,
        }
    }
}

/// Differentiation method.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DerivativeMethod {
    /// No derivative is attached.
    None,
    /// Analytic derivative of the governing model.
    Analytic,
    /// Adjoint derivative.
    Adjoint,
    /// Algorithmic differentiation.
    Algorithmic,
    /// Finite-difference derivative.
    FiniteDifference,
    /// Complex-step derivative.
    ComplexStep,
}

/// Data layout for a scalar, complex, vector, or tensor buffer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TensorLayout {
    /// Scalar storage type.
    pub scalar_type: ScalarType,
    /// Extent in each dimension.
    pub shape: Vec<usize>,
    /// Element strides in each dimension.
    pub strides: Vec<usize>,
    /// Byte offset into the referenced buffer.
    pub byte_offset: u64,
    /// Ordering used for contiguous arrays.
    pub order: StorageOrder,
    /// Complex-number storage convention.
    pub complex_storage: ComplexStorage,
    /// Coordinate frame for vector/tensor components, if applicable.
    pub coordinate_frame: Option<CoordinateFrame>,
}

impl Default for TensorLayout {
    fn default() -> Self {
        Self {
            scalar_type: ScalarType::F64,
            shape: Vec::new(),
            strides: Vec::new(),
            byte_offset: 0,
            order: StorageOrder::RowMajor,
            complex_storage: ComplexStorage::Interleaved,
            coordinate_frame: None,
        }
    }
}

impl TensorLayout {
    /// Return the number of logical elements in this tensor.
    pub fn element_count(&self) -> Result<usize, IrValidationError> {
        self.shape.iter().try_fold(1usize, |count, &extent| {
            count
                .checked_mul(extent)
                .ok_or_else(|| IrValidationError::InvalidLayout("element count overflows".into()))
        })
    }

    /// Return the minimum byte range required to store this tensor.
    ///
    /// The calculation honors explicit strides, so it remains correct for a
    /// strided view rather than assuming `element_count * scalar_size`.
    pub fn storage_byte_length(&self) -> Result<u64, IrValidationError> {
        self.validate()?;
        if self.shape.contains(&0) {
            return Ok(0);
        }
        let scalar_bytes = match self.scalar_type {
            ScalarType::F32 => 4_u64,
            ScalarType::F64 => 8_u64,
            ScalarType::ComplexF32 => 8_u64,
            ScalarType::ComplexF64 => 16_u64,
        };
        let last_element = self.shape.iter().zip(&self.strides).try_fold(
            self.byte_offset,
            |offset, (&extent, &stride)| {
                let extent = u64::try_from(extent).map_err(|_| {
                    IrValidationError::InvalidLayout("tensor extent does not fit in u64".into())
                })?;
                let stride = u64::try_from(stride).map_err(|_| {
                    IrValidationError::InvalidLayout("tensor stride does not fit in u64".into())
                })?;
                let contribution = extent
                    .checked_sub(1)
                    .and_then(|last| last.checked_mul(stride))
                    .ok_or_else(|| {
                        IrValidationError::InvalidLayout("storage offset overflows".into())
                    })?;
                offset.checked_add(contribution).ok_or_else(|| {
                    IrValidationError::InvalidLayout("storage offset overflows".into())
                })
            },
        )?;
        last_element
            .checked_add(1)
            .and_then(|elements| elements.checked_mul(scalar_bytes))
            .ok_or_else(|| IrValidationError::InvalidLayout("storage byte length overflows".into()))
    }

    /// Validate rank/stride consistency and checked byte-offset arithmetic.
    pub fn validate(&self) -> Result<(), IrValidationError> {
        if self.shape.len() != self.strides.len() {
            return Err(IrValidationError::InvalidLayout(
                "shape and strides must have equal rank".into(),
            ));
        }
        if self
            .shape
            .iter()
            .zip(self.strides.iter())
            .any(|(&extent, &stride)| extent > 0 && stride == 0)
        {
            return Err(IrValidationError::InvalidLayout(
                "non-empty dimensions must have non-zero strides".into(),
            ));
        }
        if matches!(self.scalar_type, ScalarType::F32 | ScalarType::F64)
            && self.complex_storage == ComplexStorage::Split
        {
            return Err(IrValidationError::InvalidLayout(
                "split storage is only valid for complex scalar types".into(),
            ));
        }
        if self.order != StorageOrder::Strided && self.shape.iter().all(|&extent| extent > 0) {
            let mut expected = 1usize;
            match self.order {
                StorageOrder::RowMajor => {
                    for index in (0..self.shape.len()).rev() {
                        if self.strides[index] != expected {
                            return Err(IrValidationError::InvalidLayout(
                                "contiguous tensor strides do not match the declared storage order"
                                    .into(),
                            ));
                        }
                        expected = expected.checked_mul(self.shape[index]).ok_or_else(|| {
                            IrValidationError::InvalidLayout("stride overflows".into())
                        })?;
                    }
                }
                StorageOrder::ColumnMajor => {
                    for index in 0..self.shape.len() {
                        if self.strides[index] != expected {
                            return Err(IrValidationError::InvalidLayout(
                                "contiguous tensor strides do not match the declared storage order"
                                    .into(),
                            ));
                        }
                        expected = expected.checked_mul(self.shape[index]).ok_or_else(|| {
                            IrValidationError::InvalidLayout("stride overflows".into())
                        })?;
                    }
                }
                StorageOrder::Strided => unreachable!(),
            }
        }
        let _ = self
            .shape
            .iter()
            .zip(self.strides.iter())
            .try_fold(self.byte_offset, |offset, (&extent, &stride)| {
                if extent == 0 {
                    return Some(offset);
                }
                let extent = u64::try_from(extent).ok()?;
                let stride = u64::try_from(stride).ok()?;
                let last = extent.checked_sub(1)?.checked_mul(stride)?;
                offset.checked_add(last)
            })
            .ok_or_else(|| IrValidationError::InvalidLayout("buffer offset overflows".into()))?;
        Ok(())
    }
}

/// Borrowed typed view over a field buffer owned by a solver or application.
///
/// The view owns neither the layout nor the values and therefore performs no
/// allocation or copy. It is intentionally limited to logically contiguous
/// values; strided/external buffers remain representable by [`TensorLayout`]
/// and [`DataReference`] and can be adapted by a backend-specific view.
#[derive(Debug, Clone, Copy)]
pub struct BorrowedFieldView<'a, T> {
    layout: &'a TensorLayout,
    values: &'a [T],
}

impl<'a, T: XemScalar> BorrowedFieldView<'a, T> {
    /// Create a checked view over an existing typed slice.
    pub fn new(layout: &'a TensorLayout, values: &'a [T]) -> Result<Self, IrValidationError> {
        layout.validate()?;
        if layout.order == StorageOrder::Strided {
            return Err(IrValidationError::InvalidLayout(
                "BorrowedFieldView requires a contiguous layout".into(),
            ));
        }
        if layout.scalar_type != T::SCALAR_TYPE {
            return Err(IrValidationError::InvalidLayout(format!(
                "typed view element type {:?} does not match declared scalar type {:?}",
                T::SCALAR_TYPE,
                layout.scalar_type
            )));
        }
        if layout.element_count()? != values.len() {
            return Err(IrValidationError::InvalidLayout(
                "typed view length does not match tensor shape".into(),
            ));
        }
        Ok(Self { layout, values })
    }

    /// Return the metadata without taking ownership.
    pub const fn layout(&self) -> &'a TensorLayout {
        self.layout
    }

    /// Return the borrowed values without copying.
    pub const fn values(&self) -> &'a [T] {
        self.values
    }
}

/// Allocation-free view over a time-history tensor.
///
/// A history is only exposed when each time frame is physically contiguous in
/// the supplied slice. This prevents a caller from accidentally interpreting
/// a strided snapshot as a dense frame; backends that own a strided device
/// buffer can continue using [`TensorLayout`] and their native adapter.
#[derive(Debug, Clone, Copy)]
pub struct BorrowedHistoryView<'a, T> {
    layout: &'a TensorLayout,
    values: &'a [T],
    time_axis: usize,
    frame_len: usize,
}

impl<'a, T: XemScalar> BorrowedHistoryView<'a, T> {
    /// Create a checked view with `time_axis` identifying the snapshot axis.
    pub fn new(
        layout: &'a TensorLayout,
        values: &'a [T],
        time_axis: usize,
    ) -> Result<Self, IrValidationError> {
        layout.validate()?;
        if layout.order == StorageOrder::Strided {
            return Err(IrValidationError::InvalidLayout(
                "BorrowedHistoryView requires a contiguous layout".into(),
            ));
        }
        // Reuse the field-view checks so histories enforce the same scalar
        // type contract as one-shot fields before examining frame strides.
        BorrowedFieldView::new(layout, values)?;
        let time_steps = layout.shape.get(time_axis).copied().ok_or_else(|| {
            IrValidationError::InvalidLayout("history time axis is out of bounds".into())
        })?;
        if time_steps == 0 {
            return Err(IrValidationError::InvalidLayout(
                "history time axis must be non-empty".into(),
            ));
        }
        let element_count = layout.element_count()?;
        if element_count != values.len() {
            return Err(IrValidationError::InvalidLayout(
                "typed history length does not match tensor shape".into(),
            ));
        }
        let frame_len = element_count / time_steps;
        if layout.strides[time_axis] != frame_len {
            return Err(IrValidationError::InvalidLayout(
                "history frames are not contiguous for the selected time axis".into(),
            ));
        }
        Ok(Self {
            layout,
            values,
            time_axis,
            frame_len,
        })
    }

    /// Tensor metadata retained by the view.
    pub const fn layout(&self) -> &'a TensorLayout {
        self.layout
    }

    /// Axis containing snapshots.
    pub const fn time_axis(&self) -> usize {
        self.time_axis
    }

    /// Number of snapshots in the history.
    pub fn time_steps(&self) -> usize {
        self.layout.shape[self.time_axis]
    }

    /// Number of scalar values in one snapshot.
    pub const fn frame_len(&self) -> usize {
        self.frame_len
    }

    /// Borrow one snapshot without copying.
    pub fn frame(&self, index: usize) -> Option<&'a [T]> {
        let start = index.checked_mul(self.frame_len)?;
        let end = start.checked_add(self.frame_len)?;
        self.values.get(start..end)
    }
}

/// Scalar element type in a field buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScalarType {
    /// 32-bit real scalar.
    F32,
    /// 64-bit real scalar.
    F64,
    /// 32-bit complex scalar.
    ComplexF32,
    /// 64-bit complex scalar.
    ComplexF64,
}

/// Rust element types that have a lossless XEM [`ScalarType`] representation.
///
/// Keeping this mapping at the borrowed-view boundary prevents a caller from
/// pairing, for example, a `Complex64` slice with a real `F64` layout merely
/// because both happen to have a compatible element count.
pub trait XemScalar {
    /// Serialized scalar type represented by this Rust element.
    const SCALAR_TYPE: ScalarType;
}

impl XemScalar for f32 {
    const SCALAR_TYPE: ScalarType = ScalarType::F32;
}

impl XemScalar for f64 {
    const SCALAR_TYPE: ScalarType = ScalarType::F64;
}

impl XemScalar for num_complex::Complex32 {
    const SCALAR_TYPE: ScalarType = ScalarType::ComplexF32;
}

impl XemScalar for num_complex::Complex64 {
    const SCALAR_TYPE: ScalarType = ScalarType::ComplexF64;
}

/// Contiguous or strided storage order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StorageOrder {
    /// C/row-major order.
    RowMajor,
    /// Fortran/column-major order.
    ColumnMajor,
    /// Explicit strides describe the order.
    Strided,
}

/// Representation of complex values in a buffer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ComplexStorage {
    /// Real and imaginary parts alternate per value.
    Interleaved,
    /// Real and imaginary planes are stored separately.
    Split,
}

/// Reference to an inline expression or externally owned field buffer.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataReference {
    /// Optional symbolic expression for small controls.
    pub expression: Option<String>,
    /// External URI/path for a large buffer.
    pub uri: Option<String>,
    /// Byte range in the external buffer.
    pub byte_offset: u64,
    /// Number of bytes in the referenced range.
    pub byte_length: u64,
}

/// A result payload with field buffers and provenance.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct MultiphysicsResult {
    /// Schema version of this document.
    pub schema_version: SchemaVersion,
    /// Problem identifier that produced the result.
    pub problem_id: String,
    /// Result fields, referencing owned or external buffers.
    pub fields: Vec<FieldResult>,
    /// Fidelity and mesh/design lineage.
    pub provenance: Provenance,
    /// Uncertainty and sample metadata.
    pub uncertainty: Option<UncertaintyMetadata>,
    /// Measurement or model calibration metadata.
    pub calibration: Option<CalibrationMetadata>,
}

/// Wire representation for untrusted result artifacts.
#[derive(Debug, Deserialize)]
struct MultiphysicsResultWire {
    schema_version: SchemaVersion,
    problem_id: String,
    fields: Vec<FieldResult>,
    provenance: Provenance,
    uncertainty: Option<UncertaintyMetadata>,
    calibration: Option<CalibrationMetadata>,
}

impl<'de> Deserialize<'de> for MultiphysicsResult {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let wire = MultiphysicsResultWire::deserialize(deserializer)?;
        let result = Self {
            schema_version: wire.schema_version,
            problem_id: wire.problem_id,
            fields: wire.fields,
            provenance: wire.provenance,
            uncertainty: wire.uncertainty,
            calibration: wire.calibration,
        };
        result
            .validate()
            .map_err(|error| <D::Error as serde::de::Error>::custom(error.to_string()))?;
        Ok(result)
    }
}

impl MultiphysicsResult {
    /// Build a result envelope for one field declared by a validated problem.
    ///
    /// The envelope carries only metadata and a [`DataReference`].  A backend
    /// can pair it with [`BorrowedResultField`] to expose its existing typed
    /// buffer without copying it, or serialize the reference for a streaming
    /// writer.  Keeping construction here makes field identifiers, layouts,
    /// schema versions, and provenance consistent across BEM/FEM consumers.
    pub fn from_problem_field(
        problem: &MultiphysicsProblem,
        field_id: &str,
        data: DataReference,
    ) -> Result<Self, IrValidationError> {
        problem.validate()?;
        let field = problem
            .field(field_id)
            .ok_or_else(|| IrValidationError::MissingReference {
                kind: "problem field",
                id: field_id.to_owned(),
            })?;
        let result = Self {
            schema_version: problem.schema_version,
            problem_id: problem.id.clone(),
            fields: vec![FieldResult {
                field_id: field.id.clone(),
                layout: field.layout.clone(),
                data,
            }],
            provenance: problem.provenance.clone(),
            uncertainty: None,
            calibration: None,
        };
        result.validate()?;
        Ok(result)
    }

    /// Validate result schema, field identifiers, layouts, and buffer ranges.
    pub fn validate(&self) -> Result<(), IrValidationError> {
        if self.problem_id.trim().is_empty() {
            return Err(IrValidationError::EmptyId {
                kind: "result problem",
            });
        }
        if self.schema_version.0 != MULTIPHYSICS_IR_VERSION {
            return Err(IrValidationError::UnsupportedSchemaVersion(
                self.schema_version,
            ));
        }
        let mut ids = std::collections::HashSet::new();
        for field in &self.fields {
            if field.field_id.trim().is_empty() {
                return Err(IrValidationError::EmptyId {
                    kind: "result field",
                });
            }
            if !ids.insert(field.field_id.as_str()) {
                return Err(IrValidationError::DuplicateId {
                    kind: "result field",
                    id: field.field_id.clone(),
                });
            }
            field.layout.validate()?;
            validate_data_reference(&field.data)?;
            let required_bytes = field.layout.storage_byte_length()?;
            if field.data.byte_length < required_bytes {
                return Err(IrValidationError::InvalidLayout(format!(
                    "result field `{}` data range is {} bytes but layout requires at least {}",
                    field.field_id, field.data.byte_length, required_bytes
                )));
            }
            field
                .data
                .byte_offset
                .checked_add(field.data.byte_length)
                .ok_or_else(|| IrValidationError::InvalidLayout("data range overflows".into()))?;
        }
        validate_provenance_metadata(&self.provenance)?;
        if let Some(uncertainty) = &self.uncertainty {
            uncertainty.validate()?;
        }
        if let Some(calibration) = &self.calibration {
            calibration.validate()?;
        }
        Ok(())
    }

    /// Validate a result before release or promotion.
    ///
    /// In addition to ordinary schema validation, this requires reproducible
    /// producer/revision/execution provenance for generated data and a
    /// calibration record for measured data. This keeps compact in-memory
    /// fixtures possible through [`Self::validate`] while preventing them from
    /// being accidentally promoted as traceable artifacts.
    pub fn validate_for_release(&self) -> Result<(), IrValidationError> {
        self.validate()?;
        self.provenance.validate_for_release()?;
        if self.provenance.fidelity == FidelityTag::Measured && self.calibration.is_none() {
            return Err(IrValidationError::InvalidValue {
                kind: "calibration metadata",
                reason: "measured release artifacts require calibration metadata".into(),
            });
        }
        Ok(())
    }
}

fn validate_distribution(name: &str, distribution: &Distribution) -> Result<(), IrValidationError> {
    let valid = match distribution {
        Distribution::Deterministic { value } => value.is_finite(),
        Distribution::Normal {
            mean,
            standard_deviation,
        } => mean.is_finite() && standard_deviation.is_finite() && *standard_deviation >= 0.0,
        Distribution::Uniform { minimum, maximum } => {
            minimum.is_finite() && maximum.is_finite() && minimum <= maximum
        }
        Distribution::LogNormal {
            log_mean,
            log_standard_deviation,
        } => {
            log_mean.is_finite()
                && log_standard_deviation.is_finite()
                && *log_standard_deviation >= 0.0
        }
        Distribution::Empirical { samples } => {
            !samples.is_empty() && samples.iter().all(|sample| sample.is_finite())
        }
    };
    if valid {
        Ok(())
    } else {
        Err(IrValidationError::InvalidValue {
            kind: "uncertainty distribution",
            reason: format!("{name} has invalid parameters"),
        })
    }
}

/// Field values in a result document.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FieldResult {
    /// Field identifier from the problem.
    pub field_id: String,
    /// Layout of the referenced values.
    pub layout: TensorLayout,
    /// Location of the values without embedding a backend-specific array type.
    pub data: DataReference,
}

/// A zero-copy typed view that is tied to a serialized result field.
///
/// The result metadata is borrowed rather than owned so this type cannot
/// outlive the document that declares the layout.  Values are borrowed from
/// the backend-owned slice and are checked against the declared contiguous
/// element count at construction.
#[derive(Debug, Clone, Copy)]
pub struct BorrowedResultField<'a, T> {
    field: &'a FieldResult,
    view: BorrowedFieldView<'a, T>,
}

impl<'a, T: XemScalar> BorrowedResultField<'a, T> {
    /// Adapt an existing typed slice to a field in a validated result.
    pub fn from_result(
        result: &'a MultiphysicsResult,
        field_id: &str,
        values: &'a [T],
    ) -> Result<Self, IrValidationError> {
        result.validate()?;
        let field = result
            .fields
            .iter()
            .find(|field| field.field_id == field_id)
            .ok_or_else(|| IrValidationError::MissingReference {
                kind: "result field",
                id: field_id.to_owned(),
            })?;
        let view = BorrowedFieldView::new(&field.layout, values)?;
        Ok(Self { field, view })
    }

    /// Stable result-field identifier.
    pub fn field_id(&self) -> &str {
        &self.field.field_id
    }

    /// Result metadata for this field.
    pub const fn metadata(&self) -> &'a FieldResult {
        self.field
    }

    /// Borrowed allocation-free values and layout.
    pub const fn view(&self) -> BorrowedFieldView<'a, T>
    where
        T: Copy,
    {
        self.view
    }
}

/// Uncertainty metadata for a result or design sample.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UncertaintyMetadata {
    /// Sample identifier in a sweep or ensemble.
    pub sample_id: Option<String>,
    /// Correlation group shared by related random variables.
    pub correlation_group: Option<String>,
    /// Random variables and their distributions.
    #[serde(default)]
    pub variables: BTreeMap<String, Distribution>,
}

impl UncertaintyMetadata {
    /// Validate sample and correlation identifiers plus distribution support.
    pub fn validate(&self) -> Result<(), IrValidationError> {
        for (label, value) in [
            ("sample id", self.sample_id.as_deref()),
            ("correlation group", self.correlation_group.as_deref()),
        ] {
            if value.is_some_and(|value| value.trim().is_empty()) {
                return Err(IrValidationError::InvalidValue {
                    kind: "uncertainty metadata",
                    reason: format!("{label} must be non-empty when present"),
                });
            }
        }
        for (name, distribution) in &self.variables {
            if name.trim().is_empty() {
                return Err(IrValidationError::InvalidValue {
                    kind: "uncertainty variable",
                    reason: "variable names must be non-empty".into(),
                });
            }
            validate_distribution(name, distribution)?;
        }
        Ok(())
    }
}

/// Probability distribution descriptor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Distribution {
    /// Deterministic value.
    Deterministic { value: f64 },
    /// Normal distribution.
    Normal { mean: f64, standard_deviation: f64 },
    /// Uniform distribution.
    Uniform { minimum: f64, maximum: f64 },
    /// Log-normal distribution.
    LogNormal {
        log_mean: f64,
        log_standard_deviation: f64,
    },
    /// Empirical samples.
    Empirical { samples: Vec<f64> },
}

/// Calibration and measurement traceability metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CalibrationMetadata {
    /// Instrument or reference solver identifier.
    pub instrument_or_reference: String,
    /// Calibration dataset identifier.
    pub dataset_id: Option<String>,
    /// Coordinate frame of the calibration data.
    pub coordinate_frame: CoordinateFrame,
    /// Optional calibration notes or revision.
    pub notes: Option<String>,
}

impl CalibrationMetadata {
    /// Validate calibration traceability and frame metadata.
    pub fn validate(&self) -> Result<(), IrValidationError> {
        if self.instrument_or_reference.trim().is_empty() {
            return Err(IrValidationError::InvalidValue {
                kind: "calibration metadata",
                reason: "instrument_or_reference must be non-empty".into(),
            });
        }
        if self
            .dataset_id
            .as_deref()
            .is_some_and(|dataset| dataset.trim().is_empty())
        {
            return Err(IrValidationError::InvalidValue {
                kind: "calibration metadata",
                reason: "dataset_id must be non-empty when present".into(),
            });
        }
        if self
            .notes
            .as_deref()
            .is_some_and(|notes| notes.contains('\0'))
        {
            return Err(IrValidationError::InvalidValue {
                kind: "calibration metadata",
                reason: "notes must not contain NUL".into(),
            });
        }
        Ok(())
    }
}

/// Transfer metadata for an interface or coupling operator.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TransferMetadata {
    /// Transfer operator family.
    pub method: String,
    /// Quantity whose integral/power is conserved, when applicable.
    pub conserved_quantity: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_complex::Complex64;

    fn problem_fixture() -> MultiphysicsProblem {
        MultiphysicsProblem {
            id: "coupled-room".into(),
            domains: vec![Domain {
                id: "acoustic".into(),
                label: "Exterior acoustics".into(),
                kind: DomainKind::Acoustics,
            }],
            regions: vec![Region {
                id: "air".into(),
                domain_id: "acoustic".into(),
                label: Some("Air".into()),
                parent_geometry_entity: Some("cad:volume:1".into()),
                material_law_id: Some("air-law".into()),
            }],
            fields: vec![FieldSpec {
                id: "pressure".into(),
                name: "complex pressure".into(),
                domain_id: "acoustic".into(),
                kind: FieldKind::ComplexScalar,
                units: Some("Pa".into()),
                layout: TensorLayout {
                    scalar_type: ScalarType::ComplexF64,
                    shape: vec![128, 3],
                    strides: vec![3, 1],
                    ..TensorLayout::default()
                },
            }],
            material_laws: vec![MaterialLaw {
                id: "air-law".into(),
                kind: MaterialLawKind::Isotropic,
                parameters: BTreeMap::from([(String::from("speed_of_sound"), 343.0)]),
            }],
            ..MultiphysicsProblem::default()
        }
    }

    #[test]
    fn ir_validates_and_round_trips() {
        let problem = problem_fixture();
        problem.validate().expect("fixture should be valid");
        let json = serde_json::to_string(&problem).expect("serialize IR");
        let decoded: MultiphysicsProblem = serde_json::from_str(&json).expect("deserialize IR");
        assert_eq!(decoded, problem);
    }

    #[test]
    fn problem_serde_admission_rejects_invalid_ir_metadata() {
        let problem = problem_fixture();
        let encoded = serde_json::to_value(&problem).expect("serialize valid IR problem");

        let mut blank_problem_id = encoded.clone();
        blank_problem_id["id"] = serde_json::json!("  ");
        let error = serde_json::from_value::<MultiphysicsProblem>(blank_problem_id)
            .expect_err("blank stable problem ID must be rejected");
        assert!(error.to_string().contains("problem has an empty id"));

        let mut blank_parameter = encoded.clone();
        blank_parameter["material_laws"][0]["parameters"][""] = serde_json::json!(1.0);
        let error = serde_json::from_value::<MultiphysicsProblem>(blank_parameter)
            .expect_err("blank material parameter names must be rejected");
        assert!(error.to_string().contains("empty parameter name"));

        let mut blank_conservation = encoded;
        blank_conservation["couplings"] = serde_json::json!([{
            "id": "pressure-output",
            "source_field_id": "pressure",
            "target_field_id": "pressure-output",
            "kind": "OneWay",
            "conservation": "\t"
        }]);
        let mut output_field = blank_conservation["fields"][0].clone();
        output_field["id"] = serde_json::json!("pressure-output");
        output_field["name"] = serde_json::json!("output pressure");
        blank_conservation["fields"]
            .as_array_mut()
            .expect("fields array")
            .push(output_field);
        let error = serde_json::from_value::<MultiphysicsProblem>(blank_conservation)
            .expect_err("blank conservation metadata must be rejected");
        assert!(error.to_string().contains("empty conserved quantity"));
    }

    #[test]
    fn ir_rejects_missing_references_and_bad_layouts() {
        let mut problem = problem_fixture();
        problem.fields[0].domain_id = "missing".into();
        assert!(matches!(
            problem.validate(),
            Err(IrValidationError::MissingReference { kind: "domain", .. })
        ));

        let layout = TensorLayout {
            shape: vec![2],
            strides: vec![],
            ..TensorLayout::default()
        };
        assert!(matches!(
            layout.validate(),
            Err(IrValidationError::InvalidLayout(_))
        ));

        let noncontiguous = TensorLayout {
            shape: vec![2, 3],
            strides: vec![1, 2],
            ..TensorLayout::default()
        };
        assert!(matches!(
            noncontiguous.validate(),
            Err(IrValidationError::InvalidLayout(message))
                if message.contains("declared storage order")
        ));
    }

    #[test]
    fn acoustic_pressure_field_contract_rejects_wrong_domain_type_and_units() {
        let problem = problem_fixture();
        assert_eq!(
            problem
                .acoustic_complex_pressure_field("pressure")
                .expect("fixture pressure field should be consumable")
                .id,
            "pressure"
        );

        let mut invalid = problem.clone();
        invalid.fields[0].units = Some("m/s".into());
        assert!(matches!(
            invalid.acoustic_complex_pressure_field("pressure"),
            Err(IrValidationError::InvalidValue { kind: "field", reason })
                if reason.contains("pressure units")
        ));

        invalid = problem.clone();
        invalid.fields[0].kind = FieldKind::Scalar;
        assert!(matches!(
            invalid.acoustic_complex_pressure_field("pressure"),
            Err(IrValidationError::InvalidValue { kind: "field", reason })
                if reason.contains("ComplexScalar")
        ));

        invalid = problem;
        invalid.fields[0].domain_id = "structural".into();
        invalid.domains.push(Domain {
            id: "structural".into(),
            label: "Structure".into(),
            kind: DomainKind::Structural,
        });
        assert!(matches!(
            invalid.acoustic_complex_pressure_field("pressure"),
            Err(IrValidationError::InvalidValue { kind: "field", reason })
                if reason.contains("acoustics domain")
        ));
    }

    #[test]
    fn ir_validates_couplings_interfaces_designs_and_provenance() {
        let mut problem = problem_fixture();
        problem.regions.push(Region {
            id: "liner".into(),
            domain_id: "acoustic".into(),
            label: None,
            parent_geometry_entity: None,
            material_law_id: None,
        });
        problem.interface_conditions.push(InterfaceCondition {
            id: "air-liner".into(),
            left_region_id: "air".into(),
            right_region_id: "liner".into(),
            field_ids: vec!["pressure".into()],
            kind: InterfaceConditionKind::Mortar,
            transfer: Some(TransferMetadata {
                method: "conservative-mortar".into(),
                conserved_quantity: Some("power".into()),
            }),
        });
        problem.boundary_conditions.push(BoundaryCondition {
            id: "pressure-bc".into(),
            region_id: "air".into(),
            field_id: "pressure".into(),
            kind: BoundaryConditionKind::Dirichlet,
            value: Some(DataReference {
                expression: Some("0".into()),
                uri: None,
                byte_offset: 0,
                byte_length: 0,
            }),
        });
        problem.design_variables.push(DesignVariable {
            id: "shape".into(),
            kind: DesignVariableKind::Shape,
            region_ids: vec!["air".into()],
            bounds: Some([-1.0, 1.0]),
            derivative_convention: DerivativeConvention {
                method: DerivativeMethod::Adjoint,
                includes_mesh_motion: true,
                ..DerivativeConvention::default()
            },
        });
        problem.couplings.push(Coupling {
            id: "pressure-self-check".into(),
            source_field_id: "pressure".into(),
            target_field_id: "pressure".into(),
            kind: CouplingKind::Custom("diagnostic".into()),
            conservation: None,
        });
        assert!(matches!(
            problem.validate(),
            Err(IrValidationError::InvalidValue {
                kind: "coupling",
                ..
            })
        ));

        problem.couplings[0].target_field_id = "pressure-output".into();
        assert!(matches!(
            problem.validate(),
            Err(IrValidationError::MissingReference { kind: "field", .. })
        ));

        problem.couplings.clear();
        problem
            .provenance
            .design_region_masks
            .push(DesignRegionMask {
                region_id: "air".into(),
                field_id: Some("missing-mask".into()),
                semantics: "active".into(),
            });
        assert!(matches!(
            problem.validate(),
            Err(IrValidationError::MissingReference { kind: "field", .. })
        ));
    }

    #[test]
    fn ir_rejects_nonfinite_materials_bounds_and_external_metadata() {
        let mut problem = problem_fixture();
        problem.material_laws[0]
            .parameters
            .insert("loss".into(), f64::NAN);
        assert!(matches!(
            problem.validate(),
            Err(IrValidationError::InvalidValue {
                kind: "material law",
                ..
            })
        ));

        let mut result = MultiphysicsResult {
            schema_version: SchemaVersion(MULTIPHYSICS_IR_VERSION),
            problem_id: "coupled-room".into(),
            fields: vec![FieldResult {
                field_id: "pressure".into(),
                layout: TensorLayout {
                    shape: vec![1],
                    strides: vec![1],
                    ..TensorLayout::default()
                },
                data: DataReference {
                    expression: None,
                    uri: None,
                    byte_offset: 0,
                    byte_length: 0,
                },
            }],
            provenance: Provenance::default(),
            uncertainty: None,
            calibration: None,
        };
        assert!(matches!(
            result.validate(),
            Err(IrValidationError::InvalidValue {
                kind: "data reference",
                ..
            })
        ));
        result.fields[0].data.uri = Some("mem://pressure".into());
        result.fields[0].data.byte_length = 8;
        result.uncertainty = Some(UncertaintyMetadata {
            sample_id: None,
            correlation_group: None,
            variables: BTreeMap::from([(
                "temperature".into(),
                Distribution::Normal {
                    mean: 293.15,
                    standard_deviation: -1.0,
                },
            )]),
        });
        assert!(matches!(
            result.validate(),
            Err(IrValidationError::InvalidValue {
                kind: "uncertainty distribution",
                ..
            })
        ));
    }

    #[test]
    fn coupling_graph_orders_partitioned_fields_and_rejects_cycles() {
        let mut problem = problem_fixture();
        problem.fields.extend([
            FieldSpec {
                id: "temperature".into(),
                name: "temperature".into(),
                domain_id: "acoustic".into(),
                kind: FieldKind::Scalar,
                units: Some("K".into()),
                layout: TensorLayout {
                    shape: vec![128],
                    strides: vec![1],
                    ..TensorLayout::default()
                },
            },
            FieldSpec {
                id: "displacement".into(),
                name: "displacement".into(),
                domain_id: "acoustic".into(),
                kind: FieldKind::Vector,
                units: Some("m".into()),
                layout: TensorLayout {
                    shape: vec![128, 3],
                    strides: vec![3, 1],
                    ..TensorLayout::default()
                },
            },
        ]);
        problem.couplings = vec![
            Coupling {
                id: "pressure-temperature".into(),
                source_field_id: "pressure".into(),
                target_field_id: "temperature".into(),
                kind: CouplingKind::Partitioned,
                conservation: None,
            },
            Coupling {
                id: "temperature-displacement".into(),
                source_field_id: "temperature".into(),
                target_field_id: "displacement".into(),
                kind: CouplingKind::OneWay,
                conservation: None,
            },
        ];
        problem.validate().expect("graph fixture is valid");
        let graph = CouplingGraph::from_problem(&problem).expect("graph builds");
        assert_eq!(
            graph.partitioned_order().expect("DAG has an order"),
            ["pressure", "temperature", "displacement"]
        );
        assert_eq!(graph.outgoing("pressure").len(), 1);

        problem.couplings.push(Coupling {
            id: "displacement-pressure-monolithic".into(),
            source_field_id: "displacement".into(),
            target_field_id: "pressure".into(),
            kind: CouplingKind::Monolithic,
            conservation: None,
        });
        let graph = CouplingGraph::from_problem(&problem).expect("monolithic cycles are allowed");
        assert_eq!(
            graph
                .partitioned_order()
                .expect("monolithic edge is not a dependency"),
            ["pressure", "temperature", "displacement"]
        );

        problem.couplings.last_mut().expect("edge exists").kind = CouplingKind::Partitioned;
        let graph = CouplingGraph::from_problem(&problem).expect("fields still exist");
        assert!(matches!(
            graph.partitioned_order(),
            Err(CouplingGraphError::Cycle(nodes)) if nodes == ["displacement", "pressure", "temperature"]
        ));
    }

    #[test]
    fn result_keeps_external_buffer_metadata_without_owning_values() {
        let result = MultiphysicsResult {
            schema_version: SchemaVersion(MULTIPHYSICS_IR_VERSION),
            problem_id: "coupled-room".into(),
            fields: vec![FieldResult {
                field_id: "pressure".into(),
                layout: TensorLayout {
                    scalar_type: ScalarType::ComplexF64,
                    shape: vec![4],
                    strides: vec![1],
                    ..TensorLayout::default()
                },
                data: DataReference {
                    expression: None,
                    uri: Some("mem://shared/pressure".into()),
                    byte_offset: 64,
                    byte_length: 64,
                },
            }],
            provenance: Provenance {
                fidelity: FidelityTag::Measured,
                ..Provenance::default()
            },
            uncertainty: Some(UncertaintyMetadata {
                sample_id: Some("sample-7".into()),
                correlation_group: Some("mic-array-a".into()),
                variables: BTreeMap::from([(
                    String::from("temperature"),
                    Distribution::Normal {
                        mean: 293.15,
                        standard_deviation: 0.2,
                    },
                )]),
            }),
            calibration: None,
        };
        result.validate().expect("result metadata should validate");
        let json = serde_json::to_string(&result).expect("serialize result");
        assert!(json.contains("mem://shared/pressure"));
        assert!(json.contains("sample-7"));
    }

    #[test]
    fn result_serde_admission_rejects_invalid_buffer_ranges() {
        let problem = problem_fixture();
        let result = MultiphysicsResult::from_problem_field(
            &problem,
            "pressure",
            DataReference {
                expression: None,
                uri: Some("mem://solver/pressure".into()),
                byte_offset: 0,
                byte_length: 128 * 3 * std::mem::size_of::<Complex64>() as u64,
            },
        )
        .expect("fixture result must be valid");
        let encoded = serde_json::to_value(&result).expect("serialize valid result");
        let decoded: MultiphysicsResult =
            serde_json::from_value(encoded.clone()).expect("valid result must deserialize");
        assert_eq!(decoded, result);

        let mut truncated = encoded;
        truncated["fields"][0]["data"]["byte_length"] = serde_json::json!(0);
        let error = serde_json::from_value::<MultiphysicsResult>(truncated)
            .expect_err("truncated result buffers must be rejected");
        assert!(error.to_string().contains("requires at least"));
    }

    #[test]
    fn borrowed_field_view_does_not_copy_and_checks_shape() {
        let layout = TensorLayout {
            scalar_type: ScalarType::F64,
            shape: vec![2, 2],
            strides: vec![2, 1],
            ..TensorLayout::default()
        };
        assert_eq!(layout.storage_byte_length().unwrap(), 32);
        let values = [1.0, 2.0, 3.0, 4.0];
        let view = BorrowedFieldView::new(&layout, &values).expect("valid view");
        assert_eq!(view.values().as_ptr(), values.as_ptr());
        assert!(BorrowedFieldView::new(&layout, &values[..3]).is_err());

        let complex_layout = TensorLayout {
            scalar_type: ScalarType::ComplexF64,
            shape: vec![2],
            strides: vec![1],
            ..TensorLayout::default()
        };
        assert!(matches!(
            BorrowedFieldView::new(&complex_layout, &[1.0_f64, 2.0]),
            Err(IrValidationError::InvalidLayout(message))
                if message.contains("does not match")
        ));
    }

    #[test]
    fn borrowed_history_view_exposes_contiguous_frames_without_copying() {
        let layout = TensorLayout {
            scalar_type: ScalarType::F64,
            shape: vec![2, 3],
            strides: vec![3, 1],
            ..TensorLayout::default()
        };
        let values = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let history = BorrowedHistoryView::new(&layout, &values, 0).expect("valid history");
        assert_eq!(history.time_steps(), 2);
        assert_eq!(history.frame_len(), 3);
        assert_eq!(history.frame(1).unwrap(), &[4.0, 5.0, 6.0]);
        assert_eq!(history.frame(2), None);
        assert_eq!(history.frame(0).unwrap().as_ptr(), values.as_ptr());
    }

    #[test]
    fn borrowed_history_view_rejects_noncontiguous_time_axis() {
        let layout = TensorLayout {
            scalar_type: ScalarType::F64,
            shape: vec![2, 3],
            strides: vec![3, 1],
            ..TensorLayout::default()
        };
        let values = [0.0; 6];
        let error = BorrowedHistoryView::new(&layout, &values, 1).unwrap_err();
        assert!(
            matches!(error, IrValidationError::InvalidLayout(message) if message.contains("contiguous"))
        );
    }

    #[test]
    fn result_envelope_and_borrowed_result_field_share_layout_without_copying() {
        let problem = problem_fixture();
        let result = MultiphysicsResult::from_problem_field(
            &problem,
            "pressure",
            DataReference {
                expression: None,
                uri: Some("mem://solver/pressure".into()),
                byte_offset: 0,
                byte_length: 128 * 3 * std::mem::size_of::<Complex64>() as u64,
            },
        )
        .expect("problem field should produce a valid result envelope");
        let values = vec![Complex64::new(1.0, -0.5); 128 * 3];
        let field = BorrowedResultField::from_result(&result, "pressure", &values)
            .expect("typed values should match the declared result layout");
        assert_eq!(field.field_id(), "pressure");
        assert_eq!(field.view().values().as_ptr(), values.as_ptr());
        assert!(matches!(
            BorrowedResultField::from_result(&result, "pressure", &values[..3]),
            Err(IrValidationError::InvalidLayout(_))
        ));
        assert!(matches!(
            BorrowedResultField::from_result(&result, "missing", &values),
            Err(IrValidationError::MissingReference {
                kind: "result field",
                ..
            })
        ));
        let mut truncated = result.clone();
        truncated.fields[0].data.byte_length = 1;
        assert!(matches!(
            truncated.validate(),
            Err(IrValidationError::InvalidLayout(message)) if message.contains("requires at least")
        ));
    }

    #[test]
    fn release_provenance_requires_reproducible_identity_and_lineage() {
        let mut provenance = Provenance::default();
        assert!(provenance.validate_for_release().is_err());
        provenance.producer = Some("sonium-bem".into());
        provenance.code_revision = Some("abc123".into());
        provenance.execution = Some(ExecutionMetadata {
            platform: "macos-aarch64".into(),
            solver: "bem-gmres".into(),
            tolerances: BTreeMap::from([("relative_residual".into(), 1.0e-8)]),
        });
        provenance.mesh_lineage = Some(MeshLineage {
            mesh_id: "mesh-2".into(),
            parent_mesh_id: Some("mesh-2".into()),
            relation: "refined".into(),
        });
        assert!(matches!(
            provenance.validate_for_release(),
            Err(IrValidationError::InvalidValue {
                kind: "mesh lineage",
                ..
            })
        ));
        provenance.mesh_lineage.as_mut().unwrap().parent_mesh_id = Some("mesh-1".into());
        assert!(provenance.validate_for_release().is_ok());
    }

    #[test]
    fn release_result_requires_execution_or_measurement_calibration() {
        let result = MultiphysicsResult {
            schema_version: SchemaVersion(MULTIPHYSICS_IR_VERSION),
            problem_id: "release-fixture".into(),
            fields: vec![FieldResult {
                field_id: "pressure".into(),
                layout: TensorLayout {
                    shape: vec![2],
                    strides: vec![1],
                    order: StorageOrder::RowMajor,
                    scalar_type: ScalarType::ComplexF64,
                    complex_storage: ComplexStorage::Interleaved,
                    ..TensorLayout::default()
                },
                data: DataReference {
                    expression: None,
                    uri: Some("mem://release/pressure".into()),
                    byte_offset: 0,
                    byte_length: 32,
                },
            }],
            provenance: Provenance {
                producer: Some("sonium-bem".into()),
                code_revision: Some("abc123".into()),
                mesh_lineage: Some(MeshLineage {
                    mesh_id: "mesh-1".into(),
                    parent_mesh_id: None,
                    relation: "generated".into(),
                }),
                execution: Some(ExecutionMetadata {
                    platform: "linux-x86_64".into(),
                    solver: "gmres".into(),
                    tolerances: BTreeMap::from([("relative_residual".into(), 1.0e-8)]),
                }),
                ..Provenance::default()
            },
            uncertainty: None,
            calibration: None,
        };
        result
            .validate_for_release()
            .expect("traceable generated result");

        let mut missing_execution = result.clone();
        missing_execution.provenance.execution = None;
        assert!(matches!(
            missing_execution.validate_for_release(),
            Err(IrValidationError::InvalidValue {
                kind: "provenance",
                ..
            })
        ));

        let mut missing_mesh = result.clone();
        missing_mesh.provenance.mesh_lineage = None;
        assert!(matches!(
            missing_mesh.validate_for_release(),
            Err(IrValidationError::InvalidValue {
                kind: "provenance",
                ..
            })
        ));

        let mut measured = result;
        measured.provenance.fidelity = FidelityTag::Measured;
        measured.provenance.execution = None;
        assert!(matches!(
            measured.validate_for_release(),
            Err(IrValidationError::InvalidValue {
                kind: "calibration metadata",
                ..
            })
        ));
        measured.calibration = Some(CalibrationMetadata {
            instrument_or_reference: "calibrated-microphone".into(),
            dataset_id: Some("measurement-42".into()),
            coordinate_frame: CoordinateFrame::WorldRightHanded,
            notes: None,
        });
        measured
            .validate_for_release()
            .expect("calibrated measured result");
    }

    #[test]
    fn uncertainty_and_calibration_metadata_reject_empty_identifiers() {
        let uncertainty = UncertaintyMetadata {
            sample_id: Some("  ".into()),
            correlation_group: None,
            variables: BTreeMap::new(),
        };
        assert!(matches!(
            uncertainty.validate(),
            Err(IrValidationError::InvalidValue {
                kind: "uncertainty metadata",
                ..
            })
        ));
        let calibration = CalibrationMetadata {
            instrument_or_reference: "mic".into(),
            dataset_id: Some("  ".into()),
            coordinate_frame: CoordinateFrame::WorldRightHanded,
            notes: None,
        };
        assert!(calibration.validate().is_err());
    }
}
