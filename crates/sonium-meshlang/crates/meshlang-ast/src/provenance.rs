//! Lossless source identity, comments, stable node IDs, and transform provenance.

use crate::hash::StructuralHash;
use crate::span::Span;
use serde::{Deserialize, Serialize};

/// Version of the stable MeshLang provenance contract.
pub const PROVENANCE_CONTRACT_VERSION: u32 = 1;

#[derive(
    Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash, Default,
)]
#[serde(transparent)]
pub struct SourceId(pub u64);

impl SourceId {
    /// Construct a deterministic source identity from a logical display name.
    /// The name is data only; the AST never interprets it as a filesystem path.
    pub fn named(display_name: &str) -> Self {
        let hash = StructuralHash::from_bytes(display_name.as_bytes());
        let mut bytes = [0_u8; 8];
        bytes.copy_from_slice(&hash.0[..8]);
        Self(u64::from_le_bytes(bytes))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceIdentity {
    pub id: SourceId,
    pub display_name: String,
    pub content_hash: Option<StructuralHash>,
}

impl SourceIdentity {
    pub fn named(display_name: impl Into<String>) -> Self {
        let display_name = display_name.into();
        Self {
            id: SourceId::named(&display_name),
            display_name,
            content_hash: None,
        }
    }

    pub fn with_content(mut self, source: &str) -> Self {
        self.content_hash = Some(StructuralHash::from_bytes(source.as_bytes()));
        self
    }

    /// Validate that the serialized identity is deterministic for its name.
    pub fn validate(&self) -> Result<(), SourceIdentityError> {
        if self.display_name.trim().is_empty() {
            return Err(SourceIdentityError::EmptyDisplayName);
        }
        let expected = SourceId::named(&self.display_name);
        if self.id != expected {
            return Err(SourceIdentityError::IdMismatch {
                expected,
                actual: self.id,
            });
        }
        Ok(())
    }
}

/// Error from validating a serialized source identity.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SourceIdentityError {
    /// A source identity must have a stable logical name.
    EmptyDisplayName,
    /// The ID does not match the deterministic hash of the display name.
    IdMismatch {
        expected: SourceId,
        actual: SourceId,
    },
}

impl Default for SourceIdentity {
    fn default() -> Self {
        Self::named("<anonymous>")
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub struct LanguageVersion {
    pub major: u16,
    pub minor: u16,
}

impl LanguageVersion {
    pub const CURRENT: Self = Self { major: 1, minor: 0 };
}

impl Default for LanguageVersion {
    fn default() -> Self {
        Self::CURRENT
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct SourceMetadata {
    pub identity: SourceIdentity,
    pub language_version: LanguageVersion,
}

impl SourceMetadata {
    pub fn for_source(display_name: impl Into<String>, source: &str) -> Self {
        Self {
            identity: SourceIdentity::named(display_name).with_content(source),
            language_version: LanguageVersion::CURRENT,
        }
    }
}

/// The source-side portion of a compiler artifact's provenance contract.
///
/// Unlike [`crate::SourceFile`], this deliberately omits declarations: a
/// backend artifact needs enough information to map emitted entities and
/// diagnostics back to source files, comments, and transformed nodes without
/// duplicating the complete frontend AST.  It is serde-stable so consumers can
/// round-trip the sidecar independently of the backend that produced it.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceMap {
    pub metadata: SourceMetadata,
    #[serde(default)]
    pub source_identities: Vec<SourceIdentity>,
    pub comments: Vec<Comment>,
    pub provenance: ProvenanceMap,
    pub span: Span,
}

impl SourceMap {
    pub fn from_source_file(source_file: &crate::SourceFile) -> Self {
        let mut source_identities = source_file.source_identities.clone();
        if !source_identities
            .iter()
            .any(|identity| identity == &source_file.metadata.identity)
        {
            source_identities.insert(0, source_file.metadata.identity.clone());
        }
        Self {
            metadata: source_file.metadata.clone(),
            source_identities,
            comments: source_file.comments.clone(),
            provenance: source_file.provenance.clone(),
            span: source_file.span,
        }
    }

    /// Validate source identity and provenance invariants for a sidecar map.
    pub fn validate(&self) -> Result<(), ProvenanceValidationError> {
        let mut source_ids = std::collections::BTreeSet::new();
        for identity in &self.source_identities {
            identity
                .validate()
                .map_err(ProvenanceValidationError::InvalidSourceIdentity)?;
            if !source_ids.insert(identity.id) {
                return Err(ProvenanceValidationError::DuplicateSourceIdentity {
                    source: identity.id,
                });
            }
        }
        if !self
            .source_identities
            .iter()
            .any(|identity| identity == &self.metadata.identity)
        {
            return Err(ProvenanceValidationError::MissingMetadataIdentity);
        }
        if self.span.start > self.span.end {
            return Err(ProvenanceValidationError::InvalidSpan {
                source: self.metadata.identity.id,
                node: NodeId(0),
                span: self.span,
            });
        }
        for comment in &self.comments {
            if comment.span.start > comment.span.end {
                return Err(ProvenanceValidationError::InvalidCommentSpan { span: comment.span });
            }
        }
        self.provenance.validate()?;
        for record in &self.provenance.records {
            for reference in &record.sources {
                if !source_ids.contains(&reference.source) {
                    return Err(ProvenanceValidationError::UnknownSourceReference {
                        node: record.node,
                        source: reference.source,
                    });
                }
            }
        }
        Ok(())
    }

    /// Return a deterministic, validated source map for persistence or API
    /// sidecars.
    pub fn canonicalized(&self) -> Result<Self, ProvenanceValidationError> {
        let mut canonical = self.clone();
        canonical.source_identities.sort_by(|left, right| {
            (left.id.0, left.display_name.as_str()).cmp(&(right.id.0, right.display_name.as_str()))
        });
        canonical.source_identities.dedup();
        canonical.comments.sort_by(|left, right| {
            (
                left.span.file_id,
                left.span.start,
                left.span.end,
                comment_kind_order(left.kind),
                left.target.map(|node| node.0),
                left.text.as_str(),
            )
                .cmp(&(
                    right.span.file_id,
                    right.span.start,
                    right.span.end,
                    comment_kind_order(right.kind),
                    right.target.map(|node| node.0),
                    right.text.as_str(),
                ))
        });
        canonical.provenance = canonical.provenance.canonicalized()?;
        canonical.validate()?;
        Ok(canonical)
    }

    /// Canonicalize this source map in place.
    pub fn canonicalize(&mut self) -> Result<(), ProvenanceValidationError> {
        *self = self.canonicalized()?;
        Ok(())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Hash)]
#[serde(transparent)]
pub struct NodeId(pub u64);

impl NodeId {
    /// Derive the identity used by source/provenance records.
    ///
    /// This is intentionally source-sensitive: a node keeps its identity
    /// when the same source span is reparsed, but inserting text before it
    /// changes the span and therefore the ID. Consumers that need identity
    /// across source edits or AST transforms must use a semantic entity ID or
    /// add an explicit provenance mapping; a raw `NodeId` is not an edit-
    /// stable identity.
    pub fn from_source_span(source: SourceId, kind: &str, span: Span) -> Self {
        let mut bytes = Vec::with_capacity(24 + kind.len());
        bytes.extend_from_slice(&source.0.to_le_bytes());
        bytes.extend_from_slice(&span.file_id.to_le_bytes());
        bytes.extend_from_slice(&(span.start as u64).to_le_bytes());
        bytes.extend_from_slice(&(span.end as u64).to_le_bytes());
        bytes.extend_from_slice(kind.as_bytes());
        let hash = StructuralHash::from_bytes(&bytes);
        let mut id = [0_u8; 8];
        id.copy_from_slice(&hash.0[..8]);
        Self(u64::from_le_bytes(id))
    }
}

/// Common identity contract for AST nodes without inflating every node with a
/// separately allocated metadata object.
pub trait AstNode {
    fn span(&self) -> Span;
    fn node_kind(&self) -> &'static str;

    fn node_id(&self, source: &SourceIdentity) -> NodeId {
        NodeId::from_source_span(source.id, self.node_kind(), self.span())
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum CommentKind {
    Line,
    Block,
    DocLine,
    DocBlock,
}

fn comment_kind_order(kind: CommentKind) -> u8 {
    match kind {
        CommentKind::Line => 0,
        CommentKind::Block => 1,
        CommentKind::DocLine => 2,
        CommentKind::DocBlock => 3,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Comment {
    pub kind: CommentKind,
    pub text: String,
    pub span: Span,
    /// The documented node, when this is a doc comment.
    pub target: Option<NodeId>,
}

impl Comment {
    pub fn new(kind: CommentKind, text: impl Into<String>, span: Span) -> Self {
        Self {
            kind,
            text: text.into(),
            span,
            target: None,
        }
    }

    pub fn documenting(mut self, target: NodeId) -> Self {
        self.target = Some(target);
        self
    }

    pub fn is_doc(&self) -> bool {
        matches!(self.kind, CommentKind::DocLine | CommentKind::DocBlock)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SourceReference {
    pub source: SourceId,
    pub node: NodeId,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProvenanceOperation {
    Parsed,
    Desugared,
    Expanded,
    Projected,
    Transformed,
    Migrated { from_schema: u32 },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvenanceRecord {
    pub node: NodeId,
    pub sources: Vec<SourceReference>,
    pub operation: ProvenanceOperation,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ProvenanceMap {
    pub records: Vec<ProvenanceRecord>,
}

/// Error from validating provenance ancestry and source spans.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProvenanceValidationError {
    /// Every emitted node must retain at least one source ancestor.
    EmptySourceAncestry { node: NodeId },
    /// Source byte ranges must be ordered.
    InvalidSpan {
        source: SourceId,
        node: NodeId,
        span: Span,
    },
    /// Comment byte ranges must be ordered before a sidecar is persisted.
    InvalidCommentSpan { span: Span },
    /// A single record must not repeat an identical source reference.
    DuplicateSourceReference { node: NodeId, source: SourceId },
    /// Canonicalization should remove exact duplicate records.
    DuplicateRecord { node: NodeId },
    /// All serialized source identities must include the map metadata source.
    MissingMetadataIdentity,
    /// Source identities must be unique by stable ID.
    DuplicateSourceIdentity { source: SourceId },
    /// Every ancestry reference must resolve to a serialized source identity.
    UnknownSourceReference { node: NodeId, source: SourceId },
    /// A source identity failed its deterministic-ID check.
    InvalidSourceIdentity(SourceIdentityError),
}

impl ProvenanceMap {
    pub fn for_node(&self, node: NodeId) -> impl Iterator<Item = &ProvenanceRecord> {
        self.records
            .iter()
            .filter(move |record| record.node == node)
    }

    pub fn push(&mut self, record: ProvenanceRecord) {
        self.records.push(record);
    }

    /// Validate ancestry, span ordering, and duplicate invariants.
    pub fn validate(&self) -> Result<(), ProvenanceValidationError> {
        let mut seen_records: Vec<&ProvenanceRecord> = Vec::new();
        for record in &self.records {
            if record.sources.is_empty() {
                return Err(ProvenanceValidationError::EmptySourceAncestry { node: record.node });
            }
            if seen_records.contains(&record) {
                return Err(ProvenanceValidationError::DuplicateRecord { node: record.node });
            }
            seen_records.push(record);
            let mut seen_sources = std::collections::BTreeSet::new();
            for reference in &record.sources {
                if reference.span.start > reference.span.end {
                    return Err(ProvenanceValidationError::InvalidSpan {
                        source: reference.source,
                        node: reference.node,
                        span: reference.span,
                    });
                }
                let key = (
                    reference.source.0,
                    reference.node.0,
                    reference.span.start,
                    reference.span.end,
                    reference.span.file_id,
                );
                if !seen_sources.insert(key) {
                    return Err(ProvenanceValidationError::DuplicateSourceReference {
                        node: record.node,
                        source: reference.source,
                    });
                }
            }
        }
        Ok(())
    }

    /// Return a deterministic, deduplicated provenance map.
    pub fn canonicalized(&self) -> Result<Self, ProvenanceValidationError> {
        let mut canonical = self.clone();
        for record in &mut canonical.records {
            record.sources.sort_by_key(|reference| {
                (
                    reference.source.0,
                    reference.node.0,
                    reference.span.start,
                    reference.span.end,
                    reference.span.file_id,
                )
            });
            record.sources.dedup();
        }
        canonical.records.sort_by_key(|record| {
            let operation = match &record.operation {
                ProvenanceOperation::Parsed => (0_u8, 0_u32),
                ProvenanceOperation::Desugared => (1, 0),
                ProvenanceOperation::Expanded => (2, 0),
                ProvenanceOperation::Projected => (3, 0),
                ProvenanceOperation::Transformed => (4, 0),
                ProvenanceOperation::Migrated { from_schema } => (5, *from_schema),
            };
            let sources = record
                .sources
                .iter()
                .map(|reference| {
                    (
                        reference.source.0,
                        reference.node.0,
                        reference.span.start,
                        reference.span.end,
                        reference.span.file_id,
                    )
                })
                .collect::<Vec<_>>();
            (record.node.0, operation, sources)
        });
        canonical.records.dedup();
        canonical.validate()?;
        Ok(canonical)
    }

    /// Canonicalize this map in place.
    pub fn canonicalize(&mut self) -> Result<(), ProvenanceValidationError> {
        *self = self.canonicalized()?;
        Ok(())
    }
}

macro_rules! impl_ast_node {
    ($($type:ty => $kind:literal),+ $(,)?) => {
        $(
            impl AstNode for $type {
                fn span(&self) -> Span {
                    self.span
                }

                fn node_kind(&self) -> &'static str {
                    $kind
                }
            }
        )+
    };
}

impl_ast_node!(
    crate::SourceFile => "source_file",
    crate::ParamDecl => "param_decl",
    crate::ConstDecl => "const_decl",
    crate::ImportDecl => "import_decl",
    crate::FnDecl => "fn_decl",
    crate::ParamDef => "param_def",
    crate::GeometryBlock => "geometry_block",
    crate::MeshBlock => "mesh_block",
    crate::RefineStatement => "refine_statement",
    crate::BoundaryLayerStatement => "boundary_layer_statement",
    crate::TransfiniteStatement => "transfinite_statement",
    crate::EmbedStatement => "embed_statement",
    crate::PeriodicStatement => "periodic_statement",
    crate::CompoundStatement => "compound_statement",
    crate::PhysicsBlock => "physics_block",
    crate::GroupDecl => "group_decl",
    crate::ExportBlock => "export_block",
    crate::StructDecl => "struct_decl",
    crate::StructField => "struct_field",
    crate::Block => "block",
    crate::LetStatement => "let_statement",
    crate::ForStatement => "for_statement",
    crate::ForClause => "for_clause",
    crate::RangeExpr => "range_expr",
    crate::IfStatement => "if_statement",
    crate::LiteralExpr => "literal_expr",
    crate::ParenExpr => "paren_expr",
    crate::BinaryExpr => "binary_expr",
    crate::UnaryExpr => "unary_expr",
    crate::PipeExpr => "pipe_expr",
    crate::FieldAccessExpr => "field_access_expr",
    crate::IndexExpr => "index_expr",
    crate::ConstructorExpr => "constructor_expr",
    crate::ArrayExpr => "array_expr",
    crate::StructInitExpr => "struct_init_expr",
    crate::StructInitField => "struct_init_field",
    crate::CallExpr => "call_expr",
    crate::Argument => "argument",
    crate::Ident => "ident",
    crate::KeyValue => "key_value",
    crate::DesignVariableDecl => "design_variable_decl",
    crate::DesignBounds => "design_bounds",
    crate::UncertaintyDecl => "uncertainty_decl",
    crate::ObjectiveDecl => "objective_decl",
    crate::ConstraintDecl => "constraint_decl",
    crate::DesignRegionDecl => "design_region_decl",
    crate::PhysicalRegionDecl => "physical_region_decl",
    crate::BoundaryDecl => "boundary_decl",
    crate::InterfaceDecl => "interface_decl",
    crate::CoordinateSystemDecl => "coordinate_system_decl",
    crate::MaterialRegionDecl => "material_region_decl",
    crate::GeometryFieldDecl => "geometry_field_decl",
    crate::MeshMetricStatement => "mesh_metric_statement",
);

impl AstNode for crate::Declaration {
    fn span(&self) -> Span {
        match self {
            Self::Param(node) => node.span,
            Self::Const(node) => node.span,
            Self::Import(node) => node.span,
            Self::Function(node) => node.span,
            Self::Geometry(node) => node.span,
            Self::Mesh(node) => node.span,
            Self::Physics(node) => node.span,
            Self::Export(node) => node.span,
            Self::Struct(node) => node.span,
            Self::DesignVariable(node) => node.span,
            Self::Uncertainty(node) => node.span,
            Self::Objective(node) => node.span,
            Self::Constraint(node) => node.span,
            Self::DesignRegion(node) => node.span,
            Self::PhysicalRegion(node) => node.span,
            Self::Boundary(node) => node.span,
            Self::Interface(node) => node.span,
            Self::CoordinateSystem(node) => node.span,
            Self::MaterialRegion(node) => node.span,
            Self::GeometryField(node) => node.span,
        }
    }

    fn node_kind(&self) -> &'static str {
        match self {
            Self::Param(_) => "declaration:param",
            Self::Const(_) => "declaration:const",
            Self::Import(_) => "declaration:import",
            Self::Function(_) => "declaration:function",
            Self::Geometry(_) => "declaration:geometry",
            Self::Mesh(_) => "declaration:mesh",
            Self::Physics(_) => "declaration:physics",
            Self::Export(_) => "declaration:export",
            Self::Struct(_) => "declaration:struct",
            Self::DesignVariable(_) => "declaration:design_variable",
            Self::Uncertainty(_) => "declaration:uncertainty",
            Self::Objective(_) => "declaration:objective",
            Self::Constraint(_) => "declaration:constraint",
            Self::DesignRegion(_) => "declaration:design_region",
            Self::PhysicalRegion(_) => "declaration:physical_region",
            Self::Boundary(_) => "declaration:boundary",
            Self::Interface(_) => "declaration:interface",
            Self::CoordinateSystem(_) => "declaration:coordinate_system",
            Self::MaterialRegion(_) => "declaration:material_region",
            Self::GeometryField(_) => "declaration:geometry_field",
        }
    }
}

impl AstNode for crate::Statement {
    fn span(&self) -> Span {
        match self {
            Self::Let(node) => node.span,
            Self::Expr(node) => node.span(),
            Self::For(node) => node.span,
            Self::If(node) => node.span,
            Self::KeyValue(node) => node.span,
            Self::Group(node) => node.span,
        }
    }

    fn node_kind(&self) -> &'static str {
        match self {
            Self::Let(_) => "statement:let",
            Self::Expr(_) => "statement:expr",
            Self::For(_) => "statement:for",
            Self::If(_) => "statement:if",
            Self::KeyValue(_) => "statement:key_value",
            Self::Group(_) => "statement:group",
        }
    }
}

impl AstNode for crate::Expr {
    fn span(&self) -> Span {
        match self {
            Self::Ident(node) => node.span,
            Self::Literal(node) => node.span,
            Self::Binary(node) => node.span,
            Self::Unary(node) => node.span,
            Self::Pipe(node) => node.span,
            Self::FieldAccess(node) => node.span,
            Self::Index(node) => node.span,
            Self::Constructor(node) => node.span,
            Self::Call(node) => node.span,
            Self::Paren(node) => node.span,
            Self::Array(node) => node.span,
            Self::StructInit(node) => node.span,
            Self::If(node) => node.span,
        }
    }

    fn node_kind(&self) -> &'static str {
        match self {
            Self::Ident(_) => "expr:ident",
            Self::Literal(_) => "expr:literal",
            Self::Binary(_) => "expr:binary",
            Self::Unary(_) => "expr:unary",
            Self::Pipe(_) => "expr:pipe",
            Self::FieldAccess(_) => "expr:field_access",
            Self::Index(_) => "expr:index",
            Self::Constructor(_) => "expr:constructor",
            Self::Call(_) => "expr:call",
            Self::Paren(_) => "expr:paren",
            Self::Array(_) => "expr:array",
            Self::StructInit(_) => "expr:struct_init",
            Self::If(_) => "expr:if",
        }
    }
}

impl AstNode for crate::MeshStatement {
    fn span(&self) -> Span {
        match self {
            Self::KeyValue(node) => node.span,
            Self::Refine(node) => node.span,
            Self::BoundaryLayer(node) => node.span,
            Self::Transfinite(node) => node.span,
            Self::Embed(node) => node.span,
            Self::Periodic(node) => node.span,
            Self::Compound(node) => node.span,
            Self::Metric(node) => node.span,
        }
    }

    fn node_kind(&self) -> &'static str {
        match self {
            Self::KeyValue(_) => "mesh:key_value",
            Self::Refine(_) => "mesh:refine",
            Self::BoundaryLayer(_) => "mesh:boundary_layer",
            Self::Transfinite(_) => "mesh:transfinite",
            Self::Embed(_) => "mesh:embed",
            Self::Periodic(_) => "mesh:periodic",
            Self::Compound(_) => "mesh:compound",
            Self::Metric(_) => "mesh:metric",
        }
    }
}

impl AstNode for crate::PhysicsStatement {
    fn span(&self) -> Span {
        match self {
            Self::Group(node) => node.span,
            Self::Let(node) => node.span,
        }
    }

    fn node_kind(&self) -> &'static str {
        match self {
            Self::Group(_) => "physics:group",
            Self::Let(_) => "physics:let",
        }
    }
}

impl AstNode for crate::ElseBranch {
    fn span(&self) -> Span {
        match self {
            Self::Else(node) => node.span,
            Self::ElseIf(node) => node.span,
        }
    }

    fn node_kind(&self) -> &'static str {
        match self {
            Self::Else(_) => "else_branch:else",
            Self::ElseIf(_) => "else_branch:else_if",
        }
    }
}

impl AstNode for crate::types::TypeAnnotation {
    fn span(&self) -> Span {
        match self {
            Self::Named(node) => node.span,
            Self::FixedArray { span, .. }
            | Self::DynArray { span, .. }
            | Self::Generic { span, .. } => *span,
        }
    }

    fn node_kind(&self) -> &'static str {
        match self {
            Self::Named(_) => "type:named",
            Self::FixedArray { .. } => "type:fixed_array",
            Self::DynArray { .. } => "type:dynamic_array",
            Self::Generic { .. } => "type:generic",
        }
    }
}

impl_ast_node!(crate::types::NamedType => "named_type");

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_ids_are_stable_and_source_sensitive() {
        let first = SourceIdentity::named("fixture.mlg");
        let same = SourceIdentity::named("fixture.mlg");
        let other = SourceIdentity::named("other.mlg");
        let span = Span::with_file(5, 10, 2);
        assert_eq!(
            NodeId::from_source_span(first.id, "expr", span),
            NodeId::from_source_span(same.id, "expr", span)
        );
        assert_ne!(
            NodeId::from_source_span(first.id, "expr", span),
            NodeId::from_source_span(other.id, "expr", span)
        );
    }
}
