//! Type annotations and the type system.

use serde::{Deserialize, Serialize};

use crate::span::Span;

/// A type as written in source code.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TypeAnnotation {
    /// Simple named type: `float`, `Solid`, `Length`, etc.
    Named(NamedType),
    /// Fixed-size array: `[float; 3]`
    FixedArray {
        element: Box<TypeAnnotation>,
        size: u64,
        span: Span,
    },
    /// Dynamic array: `[Point]`
    DynArray {
        element: Box<TypeAnnotation>,
        span: Span,
    },
    /// Generic: `Map<String, Float>`
    Generic {
        name: String,
        params: Vec<TypeAnnotation>,
        span: Span,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NamedType {
    pub name: String,
    pub span: Span,
}

/// Resolved types used during type checking.
#[derive(Debug, Clone, PartialEq)]
pub enum ResolvedType {
    // Scalars
    Int,
    Float,
    Bool,
    String,

    // Dimensional (float with physical dimension)
    Dimensional(crate::units::Dimension),

    // Geometry
    Point,
    Curve,
    Surface,
    Solid,
    FaceSet,
    EdgeSet,

    // Collections
    Array(Box<ResolvedType>),
    FixedArray(Box<ResolvedType>, u64),

    // User-defined
    Struct(String),

    // Unknown / not yet inferred
    Inferred,

    // Error sentinel
    Error,
}

impl ResolvedType {
    /// Can this type participate in CSG operations?
    pub fn is_csg_operand(&self) -> bool {
        matches!(self, ResolvedType::Surface | ResolvedType::Solid)
    }

    pub fn entity_dimension(&self) -> Option<crate::metadata::EntityDimension> {
        use crate::metadata::EntityDimension;
        match self {
            Self::Point => Some(EntityDimension::Point),
            Self::Curve | Self::EdgeSet => Some(EntityDimension::Curve),
            Self::Surface | Self::FaceSet => Some(EntityDimension::Surface),
            Self::Solid => Some(EntityDimension::Volume),
            _ => None,
        }
    }

    /// Is this a geometry type?
    pub fn is_geometry(&self) -> bool {
        matches!(
            self,
            ResolvedType::Point
                | ResolvedType::Curve
                | ResolvedType::Surface
                | ResolvedType::Solid
                | ResolvedType::FaceSet
                | ResolvedType::EdgeSet
        )
    }

    /// Is this a numeric type (for arithmetic)?
    pub fn is_numeric(&self) -> bool {
        matches!(
            self,
            ResolvedType::Int | ResolvedType::Float | ResolvedType::Dimensional(_)
        )
    }

    pub fn display_name(&self) -> String {
        match self {
            ResolvedType::Int => "int".into(),
            ResolvedType::Float => "float".into(),
            ResolvedType::Bool => "bool".into(),
            ResolvedType::String => "string".into(),
            ResolvedType::Dimensional(d) => d.display_name().into(),
            ResolvedType::Point => "point".into(),
            ResolvedType::Curve => "curve".into(),
            ResolvedType::Surface => "surface".into(),
            ResolvedType::Solid => "solid".into(),
            ResolvedType::FaceSet => "face_set".into(),
            ResolvedType::EdgeSet => "edge_set".into(),
            ResolvedType::Array(inner) => format!("[{}]", inner.display_name()),
            ResolvedType::FixedArray(inner, n) => format!("[{}; {}]", inner.display_name(), n),
            ResolvedType::Struct(name) => name.clone(),
            ResolvedType::Inferred => "<inferred>".into(),
            ResolvedType::Error => "<error>".into(),
        }
    }
}
