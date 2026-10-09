//! Typed design, physics, manufacturing, and mesh metadata.
//!
//! These types describe modelling intent only. They contain no backend names,
//! solver callbacks, or filesystem semantics.

use crate::span::Span;
use crate::units::Unit;
use crate::{Expr, Ident, KeyValue};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum EntityDimension {
    Point,
    Curve,
    Surface,
    Volume,
}

impl EntityDimension {
    pub const fn topological_dimension(self) -> u8 {
        match self {
            Self::Point => 0,
            Self::Curve => 1,
            Self::Surface => 2,
            Self::Volume => 3,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignVariableDecl {
    pub name: Ident,
    pub variable_type: DesignVariableType,
    pub initial: Expr,
    pub bounds: Option<DesignBounds>,
    pub unit: Option<Unit>,
    pub distribution: Option<Distribution>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DesignVariableType {
    Continuous,
    Integer,
    Boolean,
    Categorical { values: Vec<Expr> },
    Shape,
    Topology,
    LevelSet,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignBounds {
    pub lower: Expr,
    pub upper: Expr,
    pub lower_inclusive: bool,
    pub upper_inclusive: bool,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Distribution {
    Deterministic {
        value: Expr,
    },
    Uniform {
        lower: Expr,
        upper: Expr,
    },
    Normal {
        mean: Expr,
        standard_deviation: Expr,
    },
    LogNormal {
        mean: Expr,
        standard_deviation: Expr,
    },
    Discrete {
        values: Vec<Expr>,
        weights: Vec<Expr>,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UncertaintyDecl {
    pub name: Ident,
    pub distribution: Distribution,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ObjectiveDecl {
    pub name: Option<Ident>,
    pub sense: ObjectiveSense,
    pub expression: Expr,
    pub weight: Option<Expr>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ObjectiveSense {
    Minimize,
    Maximize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstraintDecl {
    pub name: Option<Ident>,
    pub kind: ConstraintKind,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConstraintKind {
    Equality {
        left: Expr,
        right: Expr,
        tolerance: Option<Expr>,
    },
    Inequality {
        left: Expr,
        relation: InequalityRelation,
        right: Expr,
    },
    Range {
        expression: Expr,
        bounds: DesignBounds,
    },
    VolumeFraction {
        region: Expr,
        maximum: Expr,
    },
    MinimumFeature {
        region: Expr,
        radius: Expr,
    },
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum InequalityRelation {
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DesignRegionDecl {
    pub name: Ident,
    pub region: Expr,
    pub kind: DesignRegionKind,
    pub annotations: Vec<ManufacturingAnnotation>,
    pub span: Span,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum DesignRegionKind {
    Design,
    NonDesign,
    KeepOut,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ManufacturingAnnotation {
    MinimumThickness {
        value: Expr,
        unit: Option<Unit>,
    },
    MinimumRadius {
        value: Expr,
        unit: Option<Unit>,
    },
    Overhang {
        maximum_angle: Expr,
        direction: Expr,
    },
    Symmetry {
        kind: SymmetryKind,
    },
    DrawDirection {
        direction: Expr,
    },
    Frozen,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SymmetryKind {
    Plane { origin: Expr, normal: Expr },
    Axis { origin: Expr, direction: Expr },
    Cyclic { axis: Expr, sectors: Expr },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicalRegionDecl {
    pub name: Ident,
    pub dimension: EntityDimension,
    pub entities: Expr,
    pub material: Option<Ident>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundaryDecl {
    pub name: Ident,
    pub dimension: EntityDimension,
    pub entities: Expr,
    pub kind: BoundaryKind,
    pub settings: Vec<KeyValue>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BoundaryKind {
    Dirichlet,
    Neumann,
    Robin,
    Impedance,
    Symmetry,
    Periodic,
    Named(Ident),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InterfaceDecl {
    pub name: Ident,
    pub dimension: EntityDimension,
    pub primary: Expr,
    pub secondary: Expr,
    pub kind: InterfaceKind,
    pub settings: Vec<KeyValue>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InterfaceKind {
    Conforming,
    Contact,
    Mortar,
    FluidStructure,
    Named(Ident),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinateSystemDecl {
    pub name: Ident,
    pub kind: CoordinateSystemKind,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CoordinateSystemKind {
    Cartesian {
        origin: Expr,
        axes: Vec<Expr>,
    },
    Cylindrical {
        origin: Expr,
        axis: Expr,
        reference: Expr,
    },
    Spherical {
        origin: Expr,
        polar_axis: Expr,
        reference: Expr,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialRegionDecl {
    pub name: Ident,
    pub region: Expr,
    pub material: Ident,
    pub properties: Vec<KeyValue>,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeometryFieldDecl {
    pub name: Ident,
    pub kind: GeometryFieldKind,
    pub span: Span,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GeometryFieldKind {
    Topology {
        region: Expr,
        initial_density: Expr,
        filter_radius: Option<Expr>,
    },
    LevelSet {
        field: Expr,
        iso_value: Expr,
    },
    Projected {
        geometry: Expr,
        provenance: ProjectedGeometryProvenance,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProjectedGeometryProvenance {
    pub source_field: Ident,
    pub method: ProjectionMethod,
    pub tolerance: Option<Expr>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ProjectionMethod {
    IsoSurface,
    SignedDistance,
    ClosestPoint,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MeshMetricStatement {
    pub target: Expr,
    pub metric: MeshMetricKind,
    pub coordinate_system: Option<Ident>,
    pub span: Span,
}

// Inline expression payloads keep traversal allocation-free. Boxing is a
// profiling-driven optimization rather than a default AST storage policy.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MeshMetricKind {
    Isotropic {
        size: Expr,
    },
    /// Symmetric tensor components `[xx, yy, zz, xy, xz, yz]`.
    Anisotropic {
        components: Vec<Expr>,
    },
    Hessian {
        field: Expr,
        minimum_size: Expr,
        maximum_size: Expr,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Literal, StructuralHash};

    #[derive(Serialize)]
    struct MetadataSnapshotCorpus {
        variable_types: Vec<DesignVariableType>,
        distributions: Vec<Distribution>,
        constraints: Vec<ConstraintKind>,
        annotations: Vec<ManufacturingAnnotation>,
        symmetries: Vec<SymmetryKind>,
        coordinate_systems: Vec<CoordinateSystemKind>,
        fields: Vec<GeometryFieldKind>,
        metrics: Vec<MeshMetricKind>,
        boundaries: Vec<BoundaryKind>,
        interfaces: Vec<InterfaceKind>,
    }

    fn expr(value: i64) -> Expr {
        Expr::literal(Literal::Int(value), Span::DUMMY)
    }

    fn ident(name: &str) -> Ident {
        Ident {
            name: name.into(),
            span: Span::DUMMY,
        }
    }

    #[test]
    fn every_nested_metadata_variant_has_a_stable_snapshot() {
        let bounds = || DesignBounds {
            lower: expr(1),
            upper: expr(2),
            lower_inclusive: true,
            upper_inclusive: false,
            span: Span::DUMMY,
        };
        let corpus = MetadataSnapshotCorpus {
            variable_types: vec![
                DesignVariableType::Continuous,
                DesignVariableType::Integer,
                DesignVariableType::Boolean,
                DesignVariableType::Categorical {
                    values: vec![expr(1), expr(2)],
                },
                DesignVariableType::Shape,
                DesignVariableType::Topology,
                DesignVariableType::LevelSet,
            ],
            distributions: vec![
                Distribution::Deterministic { value: expr(1) },
                Distribution::Uniform {
                    lower: expr(1),
                    upper: expr(2),
                },
                Distribution::Normal {
                    mean: expr(1),
                    standard_deviation: expr(2),
                },
                Distribution::LogNormal {
                    mean: expr(1),
                    standard_deviation: expr(2),
                },
                Distribution::Discrete {
                    values: vec![expr(1)],
                    weights: vec![expr(2)],
                },
            ],
            constraints: vec![
                ConstraintKind::Equality {
                    left: expr(1),
                    right: expr(2),
                    tolerance: Some(expr(3)),
                },
                ConstraintKind::Inequality {
                    left: expr(1),
                    relation: InequalityRelation::LessEqual,
                    right: expr(2),
                },
                ConstraintKind::Range {
                    expression: expr(1),
                    bounds: bounds(),
                },
                ConstraintKind::VolumeFraction {
                    region: expr(1),
                    maximum: expr(2),
                },
                ConstraintKind::MinimumFeature {
                    region: expr(1),
                    radius: expr(2),
                },
            ],
            annotations: vec![
                ManufacturingAnnotation::MinimumThickness {
                    value: expr(1),
                    unit: None,
                },
                ManufacturingAnnotation::MinimumRadius {
                    value: expr(1),
                    unit: None,
                },
                ManufacturingAnnotation::Overhang {
                    maximum_angle: expr(1),
                    direction: expr(2),
                },
                ManufacturingAnnotation::Symmetry {
                    kind: SymmetryKind::Plane {
                        origin: expr(1),
                        normal: expr(2),
                    },
                },
                ManufacturingAnnotation::DrawDirection { direction: expr(1) },
                ManufacturingAnnotation::Frozen,
            ],
            symmetries: vec![
                SymmetryKind::Plane {
                    origin: expr(1),
                    normal: expr(2),
                },
                SymmetryKind::Axis {
                    origin: expr(1),
                    direction: expr(2),
                },
                SymmetryKind::Cyclic {
                    axis: expr(1),
                    sectors: expr(2),
                },
            ],
            coordinate_systems: vec![
                CoordinateSystemKind::Cartesian {
                    origin: expr(1),
                    axes: vec![expr(2), expr(3), expr(4)],
                },
                CoordinateSystemKind::Cylindrical {
                    origin: expr(1),
                    axis: expr(2),
                    reference: expr(3),
                },
                CoordinateSystemKind::Spherical {
                    origin: expr(1),
                    polar_axis: expr(2),
                    reference: expr(3),
                },
            ],
            fields: vec![
                GeometryFieldKind::Topology {
                    region: expr(1),
                    initial_density: expr(2),
                    filter_radius: Some(expr(3)),
                },
                GeometryFieldKind::LevelSet {
                    field: expr(1),
                    iso_value: expr(2),
                },
                GeometryFieldKind::Projected {
                    geometry: expr(1),
                    provenance: ProjectedGeometryProvenance {
                        source_field: ident("density"),
                        method: ProjectionMethod::SignedDistance,
                        tolerance: Some(expr(2)),
                    },
                },
            ],
            metrics: vec![
                MeshMetricKind::Isotropic { size: expr(1) },
                MeshMetricKind::Anisotropic {
                    components: (1..=6).map(expr).collect(),
                },
                MeshMetricKind::Hessian {
                    field: expr(1),
                    minimum_size: expr(2),
                    maximum_size: expr(3),
                },
            ],
            boundaries: vec![
                BoundaryKind::Dirichlet,
                BoundaryKind::Neumann,
                BoundaryKind::Robin,
                BoundaryKind::Impedance,
                BoundaryKind::Symmetry,
                BoundaryKind::Periodic,
                BoundaryKind::Named(ident("custom")),
            ],
            interfaces: vec![
                InterfaceKind::Conforming,
                InterfaceKind::Contact,
                InterfaceKind::Mortar,
                InterfaceKind::FluidStructure,
                InterfaceKind::Named(ident("custom")),
            ],
        };
        let hash = StructuralHash::of(&corpus).unwrap().to_hex();
        assert_eq!(
            hash,
            "180bb3615ea9bb46accfdcb6f930830f106722608a29a6597dfb2a81f3ae366c"
        );
    }
}
