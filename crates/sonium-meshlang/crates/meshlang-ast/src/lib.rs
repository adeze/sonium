//! MeshLang Abstract Syntax Tree
//!
//! This crate defines the typed AST that the parser produces and the compiler consumes.
//! Every node carries a `Span` for error reporting.

pub mod hash;
pub mod ir;
pub mod metadata;
pub mod provenance;
pub mod schema;
pub mod span;
pub mod state_transfer;
pub mod tooling;
pub mod types;
pub mod units;
pub mod visitor;

pub use hash::{StructuralHash, StructuralHashError};
pub use ir::*;
pub use metadata::*;
pub use provenance::*;
pub use schema::{
    AstSchemaVersion, CURRENT_AST_SCHEMA_VERSION, DEFAULT_MAX_SERIALIZED_AST_BYTES,
    LEGACY_AST_SCHEMA_VERSION, MigrationReport, MigrationStep, SchemaError, VersionedAst,
};
pub use state_transfer::{
    IrStateTransferPlanChain, SpatialInterpolationError, StateTransferApplyError,
    StateTransferChainApplyError, StateTransferChainError, StateTransferReverseError,
};
pub use tooling::{
    ToolingArtifacts, ToolingValidationError, generate_tooling_artifacts, json_schema,
    validate_tooling_artifacts,
};

use serde::{Deserialize, Serialize};
use span::Span;
use types::TypeAnnotation;
use units::Unit;

/// Declare a serde AST sum type and generate its tooling variant inventory
/// from the same tokens. Exhaustive visitor matches then make an added variant
/// a compile error until traversal is updated.
macro_rules! ast_enum {
    (
        $(#[$enum_meta:meta])*
        pub enum $name:ident {
            $(
                $(#[$variant_meta:meta])*
                $variant:ident($payload:ty)
            ),+ $(,)?
        }
    ) => {
        $(#[$enum_meta])*
        pub enum $name {
            $(
                $(#[$variant_meta])*
                $variant($payload),
            )+
        }

        impl $name {
            pub const VARIANT_NAMES: &'static [&'static str] = &[$(stringify!($variant)),+];
            pub const VARIANT_PAYLOADS: &'static [&'static str] = &[$(stringify!($payload)),+];
        }
    };
}

// ── Top Level ────────────────────────────────────────────────────────────────

/// A complete MeshLang source file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceFile {
    pub declarations: Vec<Declaration>,
    #[serde(default)]
    pub metadata: SourceMetadata,
    #[serde(default)]
    pub comments: Vec<Comment>,
    /// All source identities referenced by this file and any flattened
    /// imports.  The parser inserts the root identity; the compiler's import
    /// flattener extends the catalog without changing declaration spans.
    #[serde(default)]
    pub source_identities: Vec<SourceIdentity>,
    #[serde(default)]
    pub provenance: ProvenanceMap,
    pub span: Span,
}

impl SourceFile {
    pub fn new(declarations: Vec<Declaration>, span: Span) -> Self {
        Self {
            declarations,
            metadata: SourceMetadata::default(),
            comments: Vec::new(),
            source_identities: Vec::new(),
            provenance: ProvenanceMap::default(),
            span,
        }
    }

    pub fn with_source_identity(mut self, identity: SourceIdentity) -> Self {
        self.metadata.identity = identity.clone();
        self.add_source_identity(identity);
        self
    }

    pub fn with_source_metadata(mut self, metadata: SourceMetadata) -> Self {
        self.metadata = metadata;
        self.add_source_identity(self.metadata.identity.clone());
        self
    }

    pub fn push_comment(&mut self, comment: Comment) {
        self.comments.push(comment);
    }

    pub fn extend_comments(&mut self, comments: impl IntoIterator<Item = Comment>) {
        self.comments.extend(comments);
    }

    /// Add a source identity once, preserving first-seen deterministic order.
    pub fn add_source_identity(&mut self, identity: SourceIdentity) {
        if !self
            .source_identities
            .iter()
            .any(|known| known == &identity)
        {
            self.source_identities.push(identity);
        }
    }

    pub fn extend_source_identities(
        &mut self,
        identities: impl IntoIterator<Item = SourceIdentity>,
    ) {
        for identity in identities {
            self.add_source_identity(identity);
        }
    }

    pub fn structural_hash(&self) -> Result<StructuralHash, StructuralHashError> {
        StructuralHash::of(self)
    }
}

ast_enum! {
    // Keeping declaration payloads inline avoids an allocation for every
    // declaration; move to arenas/boxing only if AST-clone profiling warrants it.
    #[allow(clippy::large_enum_variant)]
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Declaration {
        Param(ParamDecl),
        Const(ConstDecl),
        Import(ImportDecl),
        Function(FnDecl),
        Geometry(GeometryBlock),
        Mesh(MeshBlock),
        Physics(PhysicsBlock),
        Export(ExportBlock),
        Struct(StructDecl),
        DesignVariable(DesignVariableDecl),
        Uncertainty(UncertaintyDecl),
        Objective(ObjectiveDecl),
        Constraint(ConstraintDecl),
        DesignRegion(DesignRegionDecl),
        PhysicalRegion(PhysicalRegionDecl),
        Boundary(BoundaryDecl),
        Interface(InterfaceDecl),
        CoordinateSystem(CoordinateSystemDecl),
        MaterialRegion(MaterialRegionDecl),
        GeometryField(GeometryFieldDecl),
    }
}

// ── Parameters & Constants ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamDecl {
    pub name: Ident,
    pub type_ann: Option<TypeAnnotation>,
    pub value: Expr,
    pub unit: Option<Unit>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstDecl {
    pub name: Ident,
    pub type_ann: Option<TypeAnnotation>,
    pub value: Expr,
    pub unit: Option<Unit>,
    pub span: Span,
}

// ── Imports ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImportDecl {
    pub path: String,
    pub kind: ImportKind,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImportKind {
    /// `import "file.mlg"`
    Whole,
    /// `import "file.mlg" as alias`
    Aliased(Ident),
    /// `import { A, B } from "file.mlg"`
    Selective(Vec<Ident>),
}

// ── Functions ────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FnDecl {
    pub is_pub: bool,
    pub name: Ident,
    pub params: Vec<ParamDef>,
    pub return_type: Option<TypeAnnotation>,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamDef {
    pub name: Ident,
    pub type_ann: TypeAnnotation,
    pub default: Option<Expr>,
    pub span: Span,
}

// ── Geometry ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryBlock {
    pub is_pub: bool,
    pub name: Ident,
    pub params: Vec<ParamDef>,
    /// Named entity results exported by a parameterized geometry template.
    ///
    /// A template with ports is instantiated into a private canonical scope;
    /// callers access a port through the instance, e.g. `frame.inlet`.
    /// Keeping this optional/defaulted preserves deserialization of the v2
    /// envelope emitted before named template ports were introduced.
    #[serde(default)]
    pub ports: Vec<ParamDef>,
    pub body: Block,
    pub span: Span,
}

// ── Mesh Control ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshBlock {
    pub target: Option<Ident>,
    pub statements: Vec<MeshStatement>,
    pub span: Span,
}

ast_enum! {
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum MeshStatement {
        KeyValue(KeyValue),
        Refine(RefineStatement),
        BoundaryLayer(BoundaryLayerStatement),
        Transfinite(TransfiniteStatement),
        Embed(EmbedStatement),
        Periodic(PeriodicStatement),
        Compound(CompoundStatement),
        Metric(MeshMetricStatement),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RefineStatement {
    pub kind: RefineKind,
    pub body: Vec<KeyValue>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RefineKind {
    /// `refine near(entity, distance: x) { ... }`
    Near { entity: Expr, args: Vec<KeyValue> },
    /// `refine on(entity) { ... }`
    On { entity: Expr },
    /// `refine in(region) { ... }`
    In { region: Expr },
    /// `refine field(key: val, ...) { ... }`
    Field { args: Vec<KeyValue> },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryLayerStatement {
    pub target: Expr,
    pub body: Vec<KeyValue>,
    pub span: Span,
}

// ── Transfinite / Embed / Periodic / Compound ───────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransfiniteStatement {
    /// `curve`, `surface`, or `volume`
    pub kind: TransfiniteKind,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransfiniteKind {
    /// `transfinite curve(entity, nodes: N, type: "Progression", coef: 1.1)`
    Curve {
        entity: Expr,
        settings: Vec<KeyValue>,
    },
    /// `transfinite surface(entity)` or `transfinite surface(entity, corners: [...])`
    Surface {
        entity: Expr,
        settings: Vec<KeyValue>,
    },
    /// `transfinite volume(entity)`
    Volume {
        entity: Expr,
        settings: Vec<KeyValue>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbedStatement {
    /// Dimension of entities to embed (0=point, 1=curve)
    pub embed_dim: EmbedDim,
    /// Entity to embed
    pub source: Expr,
    /// Target entity (containing volume or surface)
    pub target: Expr,
    /// Dimension of target (2=surface, 3=volume)
    pub target_dim: EmbedDim,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum EmbedDim {
    Point,
    Curve,
    Surface,
    Volume,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PeriodicStatement {
    /// Dimension of the periodic entities
    pub dim: PeriodicDim,
    /// Source entity
    pub source: Expr,
    /// Master entity
    pub target: Expr,
    /// Affine transform (translation vector or full 4x4 matrix)
    pub transform: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PeriodicDim {
    Curve,
    Surface,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompoundStatement {
    /// Entities to compound-mesh across
    pub entities: Vec<Expr>,
    pub span: Span,
}

// ── Physics ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsBlock {
    pub statements: Vec<PhysicsStatement>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PhysicsStatement {
    Group(GroupDecl),
    Let(LetStatement),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroupDecl {
    pub name: Ident,
    pub bc_type: Ident,
    pub value: Expr,
    pub span: Span,
}

// ── Export ────────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExportBlock {
    pub name: Option<String>,
    pub settings: Vec<KeyValue>,
    pub span: Span,
}

// ── Structs ──────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructDecl {
    pub is_pub: bool,
    pub name: Ident,
    pub fields: Vec<StructField>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructField {
    pub name: Ident,
    pub type_ann: TypeAnnotation,
    pub span: Span,
}

// ── Statements ───────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Block {
    pub statements: Vec<Statement>,
    pub span: Span,
}

ast_enum! {
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Statement {
        Let(LetStatement),
        Expr(Expr),
        For(ForStatement),
        If(IfStatement),
        KeyValue(KeyValue),
        Group(GroupDecl),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LetStatement {
    pub name: Ident,
    pub type_ann: Option<TypeAnnotation>,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForStatement {
    pub clauses: Vec<ForClause>,
    pub body: Block,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForClause {
    pub var: Ident,
    pub range: RangeExpr,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RangeExpr {
    pub start: Box<Expr>,
    pub end: Box<Expr>,
    pub inclusive: bool,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IfStatement {
    pub condition: Expr,
    pub then_block: Block,
    pub else_block: Option<ElseBranch>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ElseBranch {
    Else(Block),
    ElseIf(Box<IfStatement>),
}

// ── Expressions ──────────────────────────────────────────────────────────────

ast_enum! {
    #[derive(Debug, Clone, Serialize, Deserialize)]
    pub enum Expr {
    /// `identifier`
        Ident(Ident),

        /// `42`, `3.14`
        Literal(LiteralExpr),

    /// `a + b`, `a - b`, `a * b`, etc.
        Binary(BinaryExpr),

    /// `-x`, `!b`
        Unary(UnaryExpr),

    /// `a |> translate(x, y, z)`
        Pipe(PipeExpr),

    /// `a.b.c`
        FieldAccess(FieldAccessExpr),

    /// `a[0]`
        Index(IndexExpr),

    /// `TypeName(args...)`
        Constructor(ConstructorExpr),

    /// `func_name(args...)` — lowercase function call
        Call(CallExpr),

        /// `(expr)`
        Paren(ParenExpr),

    /// `[a, b, c]`
        Array(ArrayExpr),

    /// `MyStruct { field: value, ... }`
        StructInit(StructInitExpr),

    /// `if cond { a } else { b }`
        If(Box<IfStatement>),
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LiteralExpr {
    pub value: Literal,
    pub span: Span,
}

impl LiteralExpr {
    pub fn new(value: Literal, span: Span) -> Self {
        Self { value, span }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParenExpr {
    pub value: Box<Expr>,
    pub span: Span,
}

impl ParenExpr {
    pub fn new(value: Expr, span: Span) -> Self {
        Self {
            value: Box::new(value),
            span,
        }
    }
}

impl Expr {
    pub fn literal(value: Literal, span: Span) -> Self {
        Self::Literal(LiteralExpr::new(value, span))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinaryExpr {
    pub left: Box<Expr>,
    pub op: BinaryOp,
    pub right: Box<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum BinaryOp {
    // Arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Pow,

    // CSG
    CsgUnion,      // |
    CsgDifference, // - (context-dependent with Sub)
    CsgIntersect,  // &

    // Comparison
    Eq,
    Ne,
    Lt,
    Gt,
    Le,
    Ge,

    // Logical
    And,
    Or,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnaryExpr {
    pub op: UnaryOp,
    pub operand: Box<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,
    Not,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PipeExpr {
    pub value: Box<Expr>,
    pub function: Ident,
    pub args: Vec<Argument>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldAccessExpr {
    pub object: Box<Expr>,
    pub field: Ident,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IndexExpr {
    pub object: Box<Expr>,
    pub index: Box<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstructorExpr {
    pub type_name: Ident,
    pub args: Vec<Argument>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArrayExpr {
    pub elements: Vec<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructInitExpr {
    pub type_name: Ident,
    pub fields: Vec<StructInitField>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructInitField {
    pub name: Ident,
    pub value: Expr,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CallExpr {
    pub function: Ident,
    pub args: Vec<Argument>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Argument {
    pub name: Option<Ident>,
    pub value: Expr,
    pub span: Span,
}

// ── Shared Primitives ────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ident {
    pub name: String,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Literal {
    Int(i64),
    Float(f64),
    String(String),
    Bool(bool),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct KeyValue {
    pub key: Ident,
    pub value: Expr,
    pub unit: Option<Unit>,
    pub span: Span,
}
