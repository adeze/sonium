//! Type checker: validates types, units, and dimensional consistency.
//!
//! Runs after parsing, before codegen. Reports diagnostics for:
//! - Unit mismatches in arithmetic
//! - CSG operations on non-solid types
//! - Undefined references
//! - Unused geometry
//! - Mesh sizing warnings (elements per wavelength)

use meshlang_ast::types::ResolvedType;
use meshlang_ast::units::Dimension;
use meshlang_ast::*;
use std::collections::{HashMap, HashSet};

/// A diagnostic message from type checking.
#[derive(Debug, Clone)]
pub struct Diagnostic {
    pub level: DiagLevel,
    pub message: String,
    pub span: meshlang_ast::span::Span,
    pub hint: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DiagLevel {
    Error,
    Warning,
    Info,
}

impl std::fmt::Display for Diagnostic {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let prefix = match self.level {
            DiagLevel::Error => "error",
            DiagLevel::Warning => "warning",
            DiagLevel::Info => "info",
        };
        write!(f, "{}: {}", prefix, self.message)?;
        if let Some(ref hint) = self.hint {
            write!(f, "\n  = hint: {}", hint)?;
        }
        Ok(())
    }
}

/// Type environment: maps names to resolved types.
#[derive(Debug, Default)]
struct TypeEnv {
    bindings: HashMap<String, TypeInfo>,
}

#[derive(Debug, Clone)]
struct TypeInfo {
    resolved_type: ResolvedType,
    dimension: Option<Dimension>,
    geometry_template: Option<String>,
}

impl TypeEnv {
    fn new() -> Self {
        Self::default()
    }

    fn bind(&mut self, name: &str, info: TypeInfo) {
        self.bindings.insert(name.to_string(), info);
    }

    fn lookup(&self, name: &str) -> Option<&TypeInfo> {
        self.bindings.get(name)
    }
}

/// The type checker.
pub struct TypeChecker {
    diagnostics: Vec<Diagnostic>,
    env: TypeEnv,
    /// Track which geometry blocks are referenced by mesh/physics/export.
    geometry_used: HashMap<String, bool>,
    /// Struct field definitions: struct name → [(field_name, field_type)].
    struct_fields: HashMap<String, Vec<(String, ResolvedType)>>,
    /// Declared typed ports for geometry templates.
    geometry_ports: HashMap<String, HashMap<String, ResolvedType>>,
}

impl Default for TypeChecker {
    fn default() -> Self {
        Self::new()
    }
}

impl TypeChecker {
    pub fn new() -> Self {
        let mut env = TypeEnv::new();

        // Register built-in constructors
        for name in &["Box", "Sphere", "Cylinder", "Cone", "Torus", "Wedge"] {
            env.bind(
                name,
                TypeInfo {
                    resolved_type: ResolvedType::Solid,
                    dimension: None,
                    geometry_template: None,
                },
            );
        }
        for name in &["Rectangle", "Circle", "Polygon"] {
            env.bind(
                name,
                TypeInfo {
                    resolved_type: ResolvedType::Surface,
                    dimension: None,
                    geometry_template: None,
                },
            );
        }

        // Built-in transforms return the same type as input
        for name in &[
            "translate",
            "rotate",
            "scale",
            "mirror",
            "extrude",
            "revolve",
            "sweep",
            "copy",
            "fragment",
            "shell",
            "twist",
            "coherence",
            "fillet",
            "chamfer",
            "array",
            "affine",
        ] {
            env.bind(
                name,
                TypeInfo {
                    resolved_type: ResolvedType::Solid,
                    dimension: None,
                    geometry_template: None,
                },
            );
        }

        // Axis constants
        for name in &["X", "Y", "Z"] {
            env.bind(
                name,
                TypeInfo {
                    resolved_type: ResolvedType::Inferred,
                    dimension: None,
                    geometry_template: None,
                },
            );
        }

        Self {
            diagnostics: Vec::new(),
            env,
            geometry_used: HashMap::new(),
            struct_fields: HashMap::new(),
            geometry_ports: HashMap::new(),
        }
    }

    /// Run type checking on a source file. Returns diagnostics.
    pub fn check(&mut self, file: &SourceFile) -> Vec<Diagnostic> {
        // First pass: register all top-level declarations
        for decl in &file.declarations {
            self.register_declaration(decl);
        }

        // Second pass: check types and units
        for decl in &file.declarations {
            self.check_declaration(decl);
        }

        // Check for unused geometry
        for (name, used) in &self.geometry_used {
            if !used {
                self.diagnostics.push(Diagnostic {
                    level: DiagLevel::Warning,
                    message: format!("geometry '{}' defined but never referenced", name),
                    span: meshlang_ast::span::Span::DUMMY,
                    hint: Some("remove it or add it to a mesh/physics/export block".into()),
                });
            }
        }

        self.diagnostics.clone()
    }

    fn register_declaration(&mut self, decl: &Declaration) {
        match decl {
            Declaration::Param(p) => {
                let dim = p.unit.as_ref().map(|u| u.dimension);
                let rt = if let Some(d) = dim {
                    ResolvedType::Dimensional(d)
                } else {
                    self.infer_expr_type(&p.value)
                };
                self.env.bind(
                    &p.name.name,
                    TypeInfo {
                        resolved_type: rt,
                        dimension: dim,
                        geometry_template: None,
                    },
                );
            }
            Declaration::Const(c) => {
                let dim = c.unit.as_ref().map(|u| u.dimension);
                let rt = if let Some(d) = dim {
                    ResolvedType::Dimensional(d)
                } else {
                    self.infer_expr_type(&c.value)
                };
                self.env.bind(
                    &c.name.name,
                    TypeInfo {
                        resolved_type: rt,
                        dimension: dim,
                        geometry_template: None,
                    },
                );
            }
            Declaration::Geometry(g) => {
                self.geometry_used.insert(g.name.name.clone(), false);
                let ports = g
                    .ports
                    .iter()
                    .map(|port| {
                        (
                            port.name.name.clone(),
                            self.resolve_type_annotation(&port.type_ann),
                        )
                    })
                    .collect();
                self.geometry_ports.insert(g.name.name.clone(), ports);
                self.env.bind(
                    &g.name.name,
                    TypeInfo {
                        resolved_type: ResolvedType::Solid,
                        dimension: None,
                        geometry_template: None,
                    },
                );
            }
            Declaration::Function(f) => {
                self.env.bind(
                    &f.name.name,
                    TypeInfo {
                        resolved_type: ResolvedType::Inferred,
                        dimension: None,
                        geometry_template: None,
                    },
                );
            }
            Declaration::Struct(s) => {
                // Register the struct type constructor
                self.env.bind(
                    &s.name.name,
                    TypeInfo {
                        resolved_type: ResolvedType::Struct(s.name.name.clone()),
                        dimension: None,
                        geometry_template: None,
                    },
                );
                // Store field definitions for field access checking
                let fields: Vec<(String, ResolvedType)> = s
                    .fields
                    .iter()
                    .map(|f| {
                        let rt = self.resolve_type_annotation(&f.type_ann);
                        (f.name.name.clone(), rt)
                    })
                    .collect();
                self.struct_fields.insert(s.name.name.clone(), fields);
            }
            Declaration::DesignVariable(variable) => {
                let resolved_type = match variable.variable_type {
                    DesignVariableType::Continuous
                    | DesignVariableType::Shape
                    | DesignVariableType::Topology
                    | DesignVariableType::LevelSet => {
                        variable.unit.as_ref().map_or(ResolvedType::Float, |unit| {
                            ResolvedType::Dimensional(unit.dimension)
                        })
                    }
                    DesignVariableType::Integer => ResolvedType::Int,
                    DesignVariableType::Boolean => ResolvedType::Bool,
                    DesignVariableType::Categorical { .. } => ResolvedType::Inferred,
                };
                self.env.bind(
                    &variable.name.name,
                    TypeInfo {
                        resolved_type,
                        dimension: variable.unit.as_ref().map(|unit| unit.dimension),
                        geometry_template: None,
                    },
                );
            }
            Declaration::Uncertainty(uncertainty) => self.env.bind(
                &uncertainty.name.name,
                TypeInfo {
                    resolved_type: ResolvedType::Inferred,
                    dimension: None,
                    geometry_template: None,
                },
            ),
            Declaration::PhysicalRegion(region) => {
                self.bind_entity_metadata(&region.name.name, region.dimension)
            }
            Declaration::Boundary(boundary) => {
                self.bind_entity_metadata(&boundary.name.name, boundary.dimension)
            }
            Declaration::Interface(interface) => {
                self.bind_entity_metadata(&interface.name.name, interface.dimension)
            }
            Declaration::DesignRegion(region) => self.bind_metadata_value(&region.name.name),
            Declaration::CoordinateSystem(system) => self.bind_metadata_value(&system.name.name),
            Declaration::MaterialRegion(region) => self.bind_metadata_value(&region.name.name),
            Declaration::GeometryField(field) => self.env.bind(
                &field.name.name,
                TypeInfo {
                    resolved_type: ResolvedType::Inferred,
                    dimension: None,
                    geometry_template: None,
                },
            ),
            _ => {}
        }
    }

    fn check_declaration(&mut self, decl: &Declaration) {
        match decl {
            Declaration::Param(p) => {
                self.check_expr(&p.value);
            }
            Declaration::Const(c) => {
                self.check_expr(&c.value);
            }
            Declaration::Geometry(g) => {
                self.check_geometry(g);
            }
            Declaration::Function(function) => {
                let mut shadowed = Vec::new();
                for parameter in &function.params {
                    let info = TypeInfo {
                        resolved_type: self.resolve_type_annotation(&parameter.type_ann),
                        dimension: None,
                        geometry_template: None,
                    };
                    let previous = self.env.bindings.insert(parameter.name.name.clone(), info);
                    shadowed.push((parameter.name.name.clone(), previous));
                }

                if function.body.statements.is_empty() {
                    self.diagnostics.push(Diagnostic {
                        level: DiagLevel::Error,
                        message: format!(
                            "inline function '{}' must contain at least one statement",
                            function.name.name
                        ),
                        span: function.body.span,
                        hint: Some("use let bindings followed by a final expression".into()),
                    });
                } else {
                    // Inline functions use the same checked control-flow grammar
                    // as geometry blocks.  Lowering still requires a value-yielding
                    // final statement, but rejecting `if`/`for` here made valid
                    // compile-time geometry functions impossible to express.
                    self.check_control_block(&function.body);
                    if !block_has_inline_value(&function.body) {
                        self.diagnostics.push(Diagnostic {
                            level: DiagLevel::Error,
                            message: format!(
                                "function '{}' must end in a value-producing statement",
                                function.name.name
                            ),
                            span: function.body.span,
                            hint: Some(
                                "use a final expression, let binding, if/else, or compile-time for"
                                    .into(),
                            ),
                        });
                    }
                }

                for (name, previous) in shadowed.into_iter().rev() {
                    if let Some(previous) = previous {
                        self.env.bindings.insert(name, previous);
                    } else {
                        self.env.bindings.remove(&name);
                    }
                }
            }
            Declaration::Mesh(m) => {
                // Mark target geometry as used
                if let Some(ref target) = m.target {
                    if let Some(used) = self.geometry_used.get_mut(&target.name) {
                        *used = true;
                    } else {
                        self.diagnostics.push(Diagnostic {
                            level: DiagLevel::Error,
                            message: format!(
                                "mesh block references undefined geometry '{}'",
                                target.name
                            ),
                            span: target.span,
                            hint: None,
                        });
                    }
                }

                // Check mesh sizing warnings
                self.check_mesh_statements(m);
                self.check_mesh_sizing(m);
            }
            Declaration::Physics(p) => {
                for stmt in &p.statements {
                    if let PhysicsStatement::Group(g) = stmt {
                        self.check_expr(&g.value);
                    }
                }
            }
            Declaration::DesignVariable(variable) => {
                self.check_expr(&variable.initial);
                if let Some(bounds) = &variable.bounds {
                    self.check_expr(&bounds.lower);
                    self.check_expr(&bounds.upper);
                }
                if let DesignVariableType::Categorical { values } = &variable.variable_type {
                    for value in values {
                        self.check_expr(value);
                    }
                }
                if let Some(distribution) = &variable.distribution {
                    self.check_distribution(distribution);
                }
                self.check_design_variable_semantics(variable);
            }
            Declaration::Uncertainty(uncertainty) => {
                self.check_distribution(&uncertainty.distribution);
            }
            Declaration::Objective(objective) => {
                self.check_expr(&objective.expression);
                if let Some(weight) = &objective.weight {
                    self.check_expr(weight);
                }
            }
            Declaration::Constraint(constraint) => self.check_constraint(&constraint.kind),
            Declaration::DesignRegion(region) => {
                self.check_expr(&region.region);
                for annotation in &region.annotations {
                    self.check_manufacturing_annotation(annotation);
                }
            }
            Declaration::PhysicalRegion(region) => self.check_expr(&region.entities),
            Declaration::Boundary(boundary) => {
                self.check_expr(&boundary.entities);
                for setting in &boundary.settings {
                    self.check_expr(&setting.value);
                }
            }
            Declaration::Interface(interface) => {
                self.check_expr(&interface.primary);
                self.check_expr(&interface.secondary);
                for setting in &interface.settings {
                    self.check_expr(&setting.value);
                }
            }
            Declaration::CoordinateSystem(system) => match &system.kind {
                CoordinateSystemKind::Cartesian { origin, axes } => {
                    self.check_expr(origin);
                    for axis in axes {
                        self.check_expr(axis);
                    }
                }
                CoordinateSystemKind::Cylindrical {
                    origin,
                    axis,
                    reference,
                }
                | CoordinateSystemKind::Spherical {
                    origin,
                    polar_axis: axis,
                    reference,
                } => {
                    self.check_expr(origin);
                    self.check_expr(axis);
                    self.check_expr(reference);
                }
            },
            Declaration::MaterialRegion(region) => {
                self.check_expr(&region.region);
                for property in &region.properties {
                    self.check_expr(&property.value);
                }
            }
            Declaration::GeometryField(field) => match &field.kind {
                GeometryFieldKind::Topology {
                    region,
                    initial_density,
                    filter_radius,
                } => {
                    self.check_expr(region);
                    self.check_expr(initial_density);
                    self.require_unit_interval_constant(
                        initial_density,
                        "topology-field initial density must lie in [0, 1]",
                    );
                    if let Some(filter_radius) = filter_radius {
                        self.check_expr(filter_radius);
                        self.require_positive_constant(
                            filter_radius,
                            "topology-field filter radius must be positive",
                        );
                    }
                }
                GeometryFieldKind::LevelSet { field, iso_value } => {
                    self.check_expr(field);
                    self.check_expr(iso_value);
                }
                GeometryFieldKind::Projected {
                    geometry,
                    provenance,
                } => {
                    self.check_expr(geometry);
                    if let Some(tolerance) = &provenance.tolerance {
                        self.check_expr(tolerance);
                        self.require_non_negative_constant(
                            tolerance,
                            "projected-field provenance tolerance must be non-negative",
                        );
                    }
                }
            },
            _ => {}
        }
    }

    fn bind_entity_metadata(&mut self, name: &str, dimension: EntityDimension) {
        let resolved_type = match dimension {
            EntityDimension::Point => ResolvedType::Point,
            EntityDimension::Curve => ResolvedType::Curve,
            EntityDimension::Surface => ResolvedType::Surface,
            EntityDimension::Volume => ResolvedType::Solid,
        };
        self.env.bind(
            name,
            TypeInfo {
                resolved_type,
                dimension: None,
                geometry_template: None,
            },
        );
    }

    fn bind_metadata_value(&mut self, name: &str) {
        self.env.bind(
            name,
            TypeInfo {
                resolved_type: ResolvedType::Inferred,
                dimension: None,
                geometry_template: None,
            },
        );
    }

    fn check_distribution(&mut self, distribution: &Distribution) {
        match distribution {
            Distribution::Deterministic { value } => self.check_expr(value),
            Distribution::Uniform { lower, upper } => {
                self.check_expr(lower);
                self.check_expr(upper);
                self.check_strictly_ordered_constants(
                    lower,
                    upper,
                    "uniform distribution lower bound must be less than its upper bound",
                );
            }
            Distribution::Normal {
                mean,
                standard_deviation,
            }
            | Distribution::LogNormal {
                mean,
                standard_deviation,
            } => {
                self.check_expr(mean);
                self.check_expr(standard_deviation);
                self.require_positive_constant(
                    standard_deviation,
                    "distribution standard deviation must be positive",
                );
            }
            Distribution::Discrete { values, weights } => {
                for value in values {
                    self.check_expr(value);
                }
                for weight in weights {
                    self.check_expr(weight);
                }
                if values.is_empty() {
                    self.semantic_error(
                        meshlang_ast::span::Span::DUMMY,
                        "discrete distribution requires at least one value",
                    );
                }
                if values.len() != weights.len() {
                    self.semantic_error(
                        meshlang_ast::span::Span::DUMMY,
                        "discrete distribution values and weights must have equal lengths",
                    );
                }
                let weights_are_static =
                    weights.iter().all(|weight| static_number(weight).is_some());
                if weights_are_static {
                    let total = weights
                        .iter()
                        .filter_map(static_number)
                        .inspect(|weight| {
                            if *weight < 0.0 {
                                self.semantic_error(
                                    meshlang_ast::span::Span::DUMMY,
                                    "discrete distribution weights must be non-negative",
                                );
                            }
                        })
                        .sum::<f64>();
                    if !total.is_finite() || total <= 0.0 {
                        self.semantic_error(
                            meshlang_ast::span::Span::DUMMY,
                            "discrete distribution weights must have a finite positive sum",
                        );
                    }
                }
            }
        }
    }

    fn check_constraint(&mut self, constraint: &ConstraintKind) {
        match constraint {
            ConstraintKind::Equality {
                left,
                right,
                tolerance,
            } => {
                self.check_expr(left);
                self.check_expr(right);
                if let Some(tolerance) = tolerance {
                    self.check_expr(tolerance);
                    self.require_non_negative_constant(
                        tolerance,
                        "equality constraint tolerance must be non-negative",
                    );
                }
            }
            ConstraintKind::Inequality { left, right, .. } => {
                self.check_expr(left);
                self.check_expr(right);
            }
            ConstraintKind::Range { expression, bounds } => {
                self.check_expr(expression);
                self.check_expr(&bounds.lower);
                self.check_expr(&bounds.upper);
                self.check_bounds_semantics(bounds, "constraint range");
            }
            ConstraintKind::VolumeFraction { region, maximum } => {
                self.check_expr(region);
                self.check_expr(maximum);
                self.require_unit_interval_constant(
                    maximum,
                    "volume-fraction constraint maximum must lie in [0, 1]",
                );
            }
            ConstraintKind::MinimumFeature { region, radius } => {
                self.check_expr(region);
                self.check_expr(radius);
                self.require_positive_constant(
                    radius,
                    "minimum-feature constraint radius must be positive",
                );
            }
        }
    }

    fn check_manufacturing_annotation(&mut self, annotation: &ManufacturingAnnotation) {
        match annotation {
            ManufacturingAnnotation::MinimumThickness { value, .. }
            | ManufacturingAnnotation::MinimumRadius { value, .. } => {
                self.check_expr(value);
                self.require_positive_constant(
                    value,
                    "manufacturing minimum thickness/radius must be positive",
                );
            }
            ManufacturingAnnotation::Overhang {
                maximum_angle,
                direction,
            } => {
                self.check_expr(maximum_angle);
                self.check_expr(direction);
            }
            ManufacturingAnnotation::Symmetry { kind } => match kind {
                SymmetryKind::Plane { origin, normal } => {
                    self.check_expr(origin);
                    self.check_expr(normal);
                }
                SymmetryKind::Axis { origin, direction } => {
                    self.check_expr(origin);
                    self.check_expr(direction);
                }
                SymmetryKind::Cyclic { axis, sectors } => {
                    self.check_expr(axis);
                    self.check_expr(sectors);
                    if let Some(sectors) = static_number(sectors) {
                        if !sectors.is_finite() || sectors < 2.0 || sectors.fract() != 0.0 {
                            self.semantic_error(
                                meshlang_ast::span::Span::DUMMY,
                                "cyclic symmetry sectors must be an integer of at least two",
                            );
                        }
                    }
                }
            },
            ManufacturingAnnotation::DrawDirection { direction } => self.check_expr(direction),
            ManufacturingAnnotation::Frozen => {}
        }
    }

    fn check_design_variable_semantics(&mut self, variable: &DesignVariableDecl) {
        if let Some(bounds) = &variable.bounds {
            self.check_bounds_semantics(bounds, "design variable");
            if let (Some(initial), Some(lower), Some(upper)) = (
                static_number(&variable.initial),
                static_number(&bounds.lower),
                static_number(&bounds.upper),
            ) {
                if initial < lower || initial > upper {
                    self.semantic_error(
                        variable.initial.span(),
                        "design variable initial value must lie within its declared bounds",
                    );
                }
            }
        }

        match &variable.variable_type {
            DesignVariableType::Integer => {
                self.require_integral_constant(
                    &variable.initial,
                    "integer design variable initial value must be integral",
                );
                if let Some(bounds) = &variable.bounds {
                    self.require_integral_constant(
                        &bounds.lower,
                        "integer design variable lower bound must be integral",
                    );
                    self.require_integral_constant(
                        &bounds.upper,
                        "integer design variable upper bound must be integral",
                    );
                }
            }
            DesignVariableType::Boolean if variable.bounds.is_some() => self.semantic_error(
                variable.span,
                "boolean design variables cannot declare numeric bounds",
            ),
            DesignVariableType::Categorical { values } if values.is_empty() => self.semantic_error(
                variable.span,
                "categorical design variables require at least one category",
            ),
            _ => {}
        }
    }

    fn check_bounds_semantics(&mut self, bounds: &DesignBounds, context: &str) {
        if let (Some(lower), Some(upper)) =
            (static_number(&bounds.lower), static_number(&bounds.upper))
        {
            if !lower.is_finite()
                || !upper.is_finite()
                || lower > upper
                || (lower == upper && !(bounds.lower_inclusive && bounds.upper_inclusive))
            {
                self.semantic_error(
                    bounds.span,
                    &format!(
                        "{context} bounds must be finite and ordered (equal bounds require both endpoints inclusive)"
                    ),
                );
            }
        }
    }

    fn check_strictly_ordered_constants(&mut self, lower: &Expr, upper: &Expr, message: &str) {
        if let (Some(lower_value), Some(upper_value)) = (static_number(lower), static_number(upper))
        {
            if !lower_value.is_finite() || !upper_value.is_finite() || lower_value >= upper_value {
                self.semantic_error(lower.span(), message);
            }
        }
    }

    fn require_positive_constant(&mut self, expr: &Expr, message: &str) {
        if let Some(value) = static_number(expr) {
            if !value.is_finite() || value <= 0.0 {
                self.semantic_error(expr.span(), message);
            }
        }
    }

    fn require_non_negative_constant(&mut self, expr: &Expr, message: &str) {
        if let Some(value) = static_number(expr) {
            if !value.is_finite() || value < 0.0 {
                self.semantic_error(expr.span(), message);
            }
        }
    }

    fn require_unit_interval_constant(&mut self, expr: &Expr, message: &str) {
        if let Some(value) = static_number(expr) {
            if !value.is_finite() || !(0.0..=1.0).contains(&value) {
                self.semantic_error(expr.span(), message);
            }
        }
    }

    fn require_integral_constant(&mut self, expr: &Expr, message: &str) {
        if let Some(value) = static_number(expr) {
            if !value.is_finite() || value.fract() != 0.0 {
                self.semantic_error(expr.span(), message);
            }
        }
    }

    fn semantic_error(&mut self, span: meshlang_ast::span::Span, message: &str) {
        self.diagnostics.push(Diagnostic {
            level: DiagLevel::Error,
            message: message.into(),
            span,
            hint: None,
        });
    }

    fn check_geometry(&mut self, geom: &GeometryBlock) {
        let let_names: HashSet<String> = geom
            .body
            .statements
            .iter()
            .filter_map(|stmt| match stmt {
                Statement::Let(l) => Some(l.name.name.clone()),
                _ => None,
            })
            .collect();
        let mut declared = HashSet::new();
        let mut shadowed = Vec::new();
        for parameter in &geom.params {
            let previous = self.env.bindings.insert(
                parameter.name.name.clone(),
                TypeInfo {
                    resolved_type: self.resolve_type_annotation(&parameter.type_ann),
                    dimension: None,
                    geometry_template: None,
                },
            );
            shadowed.push((parameter.name.name.clone(), previous));
        }

        for stmt in &geom.body.statements {
            if let Statement::Let(l) = stmt {
                self.check_forward_geometry_refs(&l.value, &let_names, &declared);
                self.check_statement(stmt);

                let rt = self.infer_expr_type(&l.value);
                let geometry_template = geometry_template_for_expr(&l.value, &self.geometry_ports);
                let previous = self.env.bindings.insert(
                    l.name.name.clone(),
                    TypeInfo {
                        resolved_type: rt,
                        dimension: None,
                        geometry_template,
                    },
                );
                shadowed.push((l.name.name.clone(), previous));
                declared.insert(l.name.name.clone());
            } else {
                self.check_statement(stmt);
            }
        }
        for (name, previous) in shadowed.into_iter().rev() {
            if let Some(previous) = previous {
                self.env.bindings.insert(name, previous);
            } else {
                self.env.bindings.remove(&name);
            }
        }
    }

    fn check_mesh_statements(&mut self, mesh: &MeshBlock) {
        for statement in &mesh.statements {
            match statement {
                MeshStatement::KeyValue(value) => self.check_expr(&value.value),
                MeshStatement::Refine(refine) => {
                    match &refine.kind {
                        RefineKind::Near { entity, args } => {
                            self.check_expr(entity);
                            for value in args {
                                self.check_expr(&value.value);
                            }
                        }
                        RefineKind::On { entity } => self.check_expr(entity),
                        RefineKind::In { region } => self.check_expr(region),
                        RefineKind::Field { args } => {
                            for value in args {
                                self.check_expr(&value.value);
                            }
                        }
                    }
                    for value in &refine.body {
                        self.check_expr(&value.value);
                    }
                }
                MeshStatement::BoundaryLayer(layer) => {
                    self.check_expr(&layer.target);
                    for value in &layer.body {
                        self.check_expr(&value.value);
                    }
                }
                MeshStatement::Transfinite(statement) => {
                    let (entity, settings) = match &statement.kind {
                        TransfiniteKind::Curve { entity, settings }
                        | TransfiniteKind::Surface { entity, settings }
                        | TransfiniteKind::Volume { entity, settings } => (entity, settings),
                    };
                    self.check_expr(entity);
                    for value in settings {
                        self.check_expr(&value.value);
                    }
                }
                MeshStatement::Embed(embed) => {
                    self.check_expr(&embed.source);
                    self.check_expr(&embed.target);
                }
                MeshStatement::Periodic(periodic) => {
                    self.check_expr(&periodic.source);
                    self.check_expr(&periodic.target);
                    self.check_expr(&periodic.transform);
                }
                MeshStatement::Compound(compound) => {
                    for entity in &compound.entities {
                        self.check_expr(entity);
                    }
                }
                MeshStatement::Metric(metric) => {
                    self.check_expr(&metric.target);
                    match &metric.metric {
                        MeshMetricKind::Isotropic { size } => self.check_expr(size),
                        MeshMetricKind::Anisotropic { components } => {
                            for component in components {
                                self.check_expr(component);
                            }
                        }
                        MeshMetricKind::Hessian {
                            field,
                            minimum_size,
                            maximum_size,
                        } => {
                            self.check_expr(field);
                            self.check_expr(minimum_size);
                            self.check_expr(maximum_size);
                        }
                    }
                    if let Some(system) = &metric.coordinate_system {
                        if self.env.lookup(&system.name).is_none() {
                            self.diagnostics.push(Diagnostic {
                                level: DiagLevel::Error,
                                message: format!(
                                    "mesh metric references undefined coordinate system '{}'",
                                    system.name
                                ),
                                span: system.span,
                                hint: None,
                            });
                        }
                    }
                }
            }
        }
    }

    fn check_statement(&mut self, stmt: &Statement) {
        match stmt {
            Statement::Let(l) => {
                self.check_expr(&l.value);

                // CSG check: both operands of `-`, `|`, `&` must be
                // same-dimensional surfaces or solids.
                if let Expr::Binary(b) = &l.value {
                    if is_csg_op(b.op) && self.is_csg_binary(b) {
                        let lt = self.infer_expr_type(&b.left);
                        let rt = self.infer_expr_type(&b.right);

                        if !lt.is_csg_operand() && lt != ResolvedType::Inferred {
                            self.diagnostics.push(Diagnostic {
                                level: DiagLevel::Error,
                                message: format!(
                                    "left operand of CSG operation has type '{}', expected 'surface' or 'solid'",
                                    lt.display_name()
                                ),
                                span: b.span,
                                hint: None,
                            });
                        }
                        if !rt.is_csg_operand() && rt != ResolvedType::Inferred {
                            self.diagnostics.push(Diagnostic {
                                level: DiagLevel::Error,
                                message: format!(
                                    "right operand of CSG operation has type '{}', expected 'surface' or 'solid'",
                                    rt.display_name()
                                ),
                                span: b.span,
                                hint: None,
                            });
                        }
                        if lt.is_csg_operand() && rt.is_csg_operand() && lt != rt {
                            self.diagnostics.push(Diagnostic {
                                level: DiagLevel::Error,
                                message: format!(
                                    "CSG operands must have the same dimension, got '{}' and '{}'",
                                    lt.display_name(),
                                    rt.display_name()
                                ),
                                span: b.span,
                                hint: Some(
                                    "combine surfaces with surfaces or solids with solids".into(),
                                ),
                            });
                        }
                    }
                }
            }
            Statement::Expr(e) => {
                self.check_expr(e);
            }
            Statement::For(statement) => {
                for clause in &statement.clauses {
                    self.check_expr(&clause.range.start);
                    self.check_expr(&clause.range.end);
                }

                let mut shadowed = Vec::new();
                for clause in &statement.clauses {
                    let previous = self.env.bindings.insert(
                        clause.var.name.clone(),
                        TypeInfo {
                            resolved_type: ResolvedType::Int,
                            dimension: None,
                            geometry_template: None,
                        },
                    );
                    shadowed.push((clause.var.name.clone(), previous));
                }
                self.check_control_block(&statement.body);
                for (name, previous) in shadowed.into_iter().rev() {
                    if let Some(previous) = previous {
                        self.env.bindings.insert(name, previous);
                    } else {
                        self.env.bindings.remove(&name);
                    }
                }
            }
            Statement::If(statement) => self.check_if_statement(statement),
            _ => {}
        }
    }

    fn check_if_statement(&mut self, statement: &IfStatement) {
        self.check_expr(&statement.condition);
        self.check_control_block(&statement.then_block);
        match &statement.else_block {
            Some(ElseBranch::Else(block)) => self.check_control_block(block),
            Some(ElseBranch::ElseIf(statement)) => self.check_if_statement(statement),
            None => {}
        }
    }

    /// Check statements in a control-flow block while retaining the same
    /// binding metadata that top-level geometry statements receive. In
    /// particular, a local `Frame()` binding must retain its template name so
    /// `frame.surface_port` resolves to the declared `Surface` port rather
    /// than falling back to the template's primary `Solid` result.
    fn check_control_block(&mut self, block: &Block) {
        let mut shadowed = Vec::new();
        for statement in &block.statements {
            self.check_statement(statement);
            if let Statement::Let(binding) = statement {
                let previous = self.env.bindings.insert(
                    binding.name.name.clone(),
                    self.type_info_for_binding(&binding.value),
                );
                shadowed.push((binding.name.name.clone(), previous));
            }
        }

        for (name, previous) in shadowed.into_iter().rev() {
            if let Some(previous) = previous {
                self.env.bindings.insert(name, previous);
            } else {
                self.env.bindings.remove(&name);
            }
        }
    }

    fn type_info_for_binding(&self, expr: &Expr) -> TypeInfo {
        TypeInfo {
            resolved_type: self.infer_expr_type(expr),
            dimension: None,
            geometry_template: geometry_template_for_expr(expr, &self.geometry_ports),
        }
    }

    fn check_expr(&mut self, expr: &Expr) {
        match expr {
            Expr::Binary(b) => {
                self.check_expr(&b.left);
                self.check_expr(&b.right);

                // Check unit compatibility for arithmetic
                // For Sub, only check units if not in a CSG (geometry) context
                let is_arith = matches!(b.op, BinaryOp::Add)
                    || (b.op == BinaryOp::Sub && !self.is_csg_binary(b));
                if is_arith {
                    let ld = self.infer_dimension(&b.left);
                    let rd = self.infer_dimension(&b.right);

                    if let (Some(ld), Some(rd)) = (ld, rd) {
                        if !ld.is_compatible(&rd) {
                            self.diagnostics.push(Diagnostic {
                                level: DiagLevel::Error,
                                message: format!(
                                    "unit mismatch: cannot {} {} and {}",
                                    if b.op == BinaryOp::Add {
                                        "add"
                                    } else {
                                        "subtract"
                                    },
                                    ld.display_name(),
                                    rd.display_name()
                                ),
                                span: b.span,
                                hint: Some(format!(
                                    "left has dimension {}, right has dimension {}",
                                    ld.display_name(),
                                    rd.display_name()
                                )),
                            });
                        }
                    }
                }

                if matches!(b.op, BinaryOp::Mul | BinaryOp::Div) {
                    if let (Some(ld), Some(rd)) = (
                        self.infer_dimension(&b.left),
                        self.infer_dimension(&b.right),
                    ) {
                        let result = match b.op {
                            BinaryOp::Mul => ld.checked_mul(rd),
                            BinaryOp::Div => ld.checked_div(rd),
                            _ => unreachable!(),
                        };
                        if result.is_err() {
                            self.diagnostics.push(Diagnostic {
                                level: DiagLevel::Error,
                                message: "dimension exponent overflow in arithmetic expression"
                                    .into(),
                                span: b.span,
                                hint: Some(
                                    "reduce the dimensional exponent before combining values"
                                        .into(),
                                ),
                            });
                        }
                    }
                }
            }
            Expr::Pipe(p) => {
                self.check_expr(&p.value);
                for arg in &p.args {
                    self.check_expr(&arg.value);
                }
                self.check_pipe_transform(p);
            }
            Expr::Constructor(c) => {
                for arg in &c.args {
                    self.check_expr(&arg.value);
                }
                self.check_constructor_args(c);
            }
            Expr::Call(c) => {
                for arg in &c.args {
                    self.check_expr(&arg.value);
                }
            }
            Expr::Unary(u) => self.check_expr(&u.operand),
            Expr::FieldAccess(fa) => self.check_expr(&fa.object),
            Expr::Index(idx) => {
                self.check_expr(&idx.object);
                self.check_expr(&idx.index);
            }
            Expr::Paren(inner) => self.check_expr(&inner.value),
            Expr::Array(arr) => {
                for elem in &arr.elements {
                    self.check_expr(elem);
                }
            }
            Expr::StructInit(si) => {
                for field in &si.fields {
                    self.check_expr(&field.value);
                }
                if self.env.lookup(&si.type_name.name).is_none() {
                    self.diagnostics.push(Diagnostic {
                        level: DiagLevel::Error,
                        message: format!("undefined struct type '{}'", si.type_name.name),
                        span: si.type_name.span,
                        hint: None,
                    });
                }
            }
            Expr::Ident(id)
                if self.env.lookup(&id.name).is_none() && !is_builtin_keyword(&id.name) =>
            {
                self.diagnostics.push(Diagnostic {
                    level: DiagLevel::Warning,
                    message: format!("'{}' may be undefined", id.name),
                    span: id.span,
                    hint: None,
                });
            }
            _ => {}
        }
    }

    fn check_forward_geometry_refs(
        &mut self,
        expr: &Expr,
        let_names: &HashSet<String>,
        declared: &HashSet<String>,
    ) {
        let mut refs = Vec::new();
        collect_ident_refs(expr, &mut refs);
        refs.sort();
        refs.dedup();

        for name in refs {
            if let_names.contains(&name) && !declared.contains(&name) {
                self.diagnostics.push(Diagnostic {
                    level: DiagLevel::Error,
                    message: format!("geometry let '{}' referenced before declaration", name),
                    span: meshlang_ast::span::Span::DUMMY,
                    hint: Some("move the referenced let binding before this expression".into()),
                });
            }
        }
    }

    fn check_constructor_args(&mut self, c: &ConstructorExpr) {
        match c.type_name.name.as_str() {
            "Box" => self.expect_positional_constructor(c, &["x", "y", "z"]),
            "Wedge" => self.expect_positional_constructor(c, &["x", "y", "z", "ltx"]),
            "Rectangle" => self.expect_positional_constructor(c, &["x", "y"]),
            "Point" => self.expect_positional_constructor(c, &["x", "y", "z"]),
            "Arc" => self.expect_positional_constructor(c, &["start", "center", "end"]),
            "Sphere" | "Disk" | "Circle" => self.expect_named_constructor(c, &["r"]),
            "Cylinder" => self.expect_named_constructor(c, &["r", "h"]),
            "Cone" => self.expect_named_constructor(c, &["r1", "r2", "h"]),
            "Torus" => self.expect_named_constructor(c, &["r_major", "r_minor"]),
            "Line" => self.expect_named_constructor(c, &["from", "to"]),
            "Polygon" => self.expect_named_constructor(c, &["points"]),
            "Loft" => self.check_loft_constructor(c),
            "Spline" | "BSpline" if c.args.len() < 2 => {
                self.constructor_error(c, "expected at least 2 point arguments");
            }
            _ => {}
        }
    }

    fn expect_positional_constructor(&mut self, c: &ConstructorExpr, names: &[&str]) {
        if c.args.iter().any(|arg| arg.name.is_some()) {
            self.constructor_error(
                c,
                &format!("expects positional arguments ({})", names.join(", ")),
            );
            return;
        }

        if c.args.len() != names.len() {
            self.constructor_error(
                c,
                &format!(
                    "expects {} positional arguments ({}) but got {}",
                    names.len(),
                    names.join(", "),
                    c.args.len()
                ),
            );
        }
    }

    fn expect_named_constructor(&mut self, c: &ConstructorExpr, required: &[&str]) {
        if c.args.len() != required.len() || c.args.iter().any(|arg| arg.name.is_none()) {
            self.constructor_error(
                c,
                &format!("expects named arguments ({})", required.join(", ")),
            );
            return;
        }

        let mut seen = HashSet::new();
        for arg in &c.args {
            let name = arg
                .name
                .as_ref()
                .map(|n| n.name.as_str())
                .unwrap_or_default();
            if !required.contains(&name) {
                self.constructor_error(
                    c,
                    &format!(
                        "has unknown argument '{}'; expected ({})",
                        name,
                        required.join(", ")
                    ),
                );
                return;
            }
            if !seen.insert(name) {
                self.constructor_error(c, &format!("has duplicate argument '{}'", name));
                return;
            }
        }

        for name in required {
            if !seen.contains(name) {
                self.constructor_error(c, &format!("is missing required argument '{}'", name));
                return;
            }
        }
    }

    fn check_loft_constructor(&mut self, c: &ConstructorExpr) {
        if c.args.is_empty() {
            self.constructor_error(c, "expects at least one section");
            return;
        }

        if !c.args.iter().any(|arg| arg.name.is_some()) {
            return;
        }

        const ALLOWED: &[&str] = &["sections", "ruled"];
        let mut seen_sections = false;
        let mut seen = HashSet::new();
        for arg in &c.args {
            let Some(ident) = &arg.name else {
                self.constructor_error(c, "mixes positional and named arguments");
                return;
            };
            let name = ident.name.as_str();
            if !ALLOWED.contains(&name) {
                self.constructor_error(
                    c,
                    &format!(
                        "has unknown argument '{}'; expected one of ({})",
                        name,
                        ALLOWED.join(", ")
                    ),
                );
                return;
            }
            if !seen.insert(name) {
                self.constructor_error(c, &format!("has duplicate argument '{}'", name));
                return;
            }
            if name == "sections" {
                seen_sections = true;
            }
        }

        if !seen_sections {
            self.constructor_error(c, "is missing required argument 'sections'");
        }
    }

    fn constructor_error(&mut self, c: &ConstructorExpr, detail: &str) {
        self.diagnostics.push(Diagnostic {
            level: DiagLevel::Error,
            message: format!("constructor '{}' {}", c.type_name.name, detail),
            span: c.span,
            hint: None,
        });
    }

    fn check_mesh_sizing(&mut self, mesh: &MeshBlock) {
        // Look for target_frequency + medium_speed + elements_per_wavelength
        // or just default_size, and warn if element size is too coarse
        let mut default_size: Option<f64> = None;
        let mut target_freq: Option<f64> = None;
        let mut medium_speed: Option<f64> = None;
        let mut epw: Option<f64> = None;

        for stmt in &mesh.statements {
            if let MeshStatement::KeyValue(kv) = stmt {
                let val = try_eval_const_expr(&kv.value);
                match kv.key.name.as_str() {
                    "default_size" => default_size = val,
                    "target_frequency" => target_freq = val,
                    "medium_speed" => medium_speed = val,
                    "elements_per_wavelength" => epw = val,
                    _ => {}
                }
            }
        }

        // If we have frequency-based sizing, check the ratio
        if let (Some(freq), Some(speed)) = (target_freq, medium_speed) {
            let wavelength = speed / freq;
            let _elem_per_wl = epw.unwrap_or(6.0);

            if let Some(size) = default_size {
                let actual_epw = wavelength / size;
                if actual_epw < 3.0 {
                    self.diagnostics.push(Diagnostic {
                        level: DiagLevel::Warning,
                        message: format!(
                            "mesh may be too coarse: {:.1} elements per wavelength at {:.0} Hz (minimum recommended: 6)",
                            actual_epw, freq
                        ),
                        span: mesh.span,
                        hint: Some(format!(
                            "wavelength = {:.4} m, element size = {:.4} m, ratio = {:.1}",
                            wavelength, size, actual_epw
                        )),
                    });
                }
            }
        }
    }

    fn infer_expr_type(&self, expr: &Expr) -> ResolvedType {
        match expr {
            Expr::Literal(LiteralExpr {
                value: Literal::Int(_),
                ..
            }) => ResolvedType::Int,
            Expr::Literal(LiteralExpr {
                value: Literal::Float(_),
                ..
            }) => ResolvedType::Float,
            Expr::Literal(LiteralExpr {
                value: Literal::Bool(_),
                ..
            }) => ResolvedType::Bool,
            Expr::Literal(LiteralExpr {
                value: Literal::String(_),
                ..
            }) => ResolvedType::String,
            Expr::Ident(id) => {
                if let Some(info) = self.env.lookup(&id.name) {
                    info.resolved_type.clone()
                } else {
                    ResolvedType::Inferred
                }
            }
            Expr::Constructor(c) => {
                if let Some(info) = self.env.lookup(&c.type_name.name) {
                    info.resolved_type.clone()
                } else {
                    ResolvedType::Inferred
                }
            }
            Expr::Binary(b) if self.is_csg_binary(b) => {
                let left = self.infer_expr_type(&b.left);
                if left.is_csg_operand() {
                    left
                } else {
                    ResolvedType::Solid
                }
            }
            Expr::Binary(b) => {
                let lt = self.infer_expr_type(&b.left);
                let rt = self.infer_expr_type(&b.right);

                match b.op {
                    BinaryOp::Add | BinaryOp::Sub => {
                        // Result has same dimension as operands (if compatible)
                        if let ResolvedType::Dimensional(d) = &lt {
                            ResolvedType::Dimensional(*d)
                        } else {
                            lt
                        }
                    }
                    BinaryOp::Mul => match (&lt, &rt) {
                        (ResolvedType::Dimensional(ld), ResolvedType::Dimensional(rd)) => ld
                            .checked_mul(*rd)
                            .map(ResolvedType::Dimensional)
                            .unwrap_or(ResolvedType::Inferred),
                        (ResolvedType::Dimensional(d), _) | (_, ResolvedType::Dimensional(d)) => {
                            ResolvedType::Dimensional(*d)
                        }
                        _ => ResolvedType::Float,
                    },
                    BinaryOp::Div => match (&lt, &rt) {
                        (ResolvedType::Dimensional(ld), ResolvedType::Dimensional(rd)) => {
                            match ld.checked_div(*rd) {
                                Ok(result_dim) if result_dim.is_dimensionless() => {
                                    ResolvedType::Float
                                }
                                Ok(result_dim) => ResolvedType::Dimensional(result_dim),
                                Err(_) => ResolvedType::Inferred,
                            }
                        }
                        (ResolvedType::Dimensional(d), _) => ResolvedType::Dimensional(*d),
                        _ => ResolvedType::Float,
                    },
                    BinaryOp::Eq
                    | BinaryOp::Ne
                    | BinaryOp::Lt
                    | BinaryOp::Gt
                    | BinaryOp::Le
                    | BinaryOp::Ge
                    | BinaryOp::And
                    | BinaryOp::Or => ResolvedType::Bool,
                    _ => ResolvedType::Float,
                }
            }
            Expr::Unary(u) => self.infer_expr_type(&u.operand),
            Expr::Pipe(p) => self.infer_pipe_type(p),
            Expr::StructInit(si) => {
                if let Some(info) = self.env.lookup(&si.type_name.name) {
                    info.resolved_type.clone()
                } else {
                    ResolvedType::Inferred
                }
            }
            Expr::FieldAccess(fa) => {
                // .faces → FaceSet, .edges → EdgeSet, etc.
                match fa.field.name.as_str() {
                    "faces" => ResolvedType::FaceSet,
                    "edges" => ResolvedType::EdgeSet,
                    "vertices" => ResolvedType::Point,
                    // Named face access returns Surface
                    "top" | "bottom" | "front" | "back" | "left" | "right" | "lateral"
                    | "outer" | "inner" => ResolvedType::Surface,
                    _ => {
                        if let Expr::Ident(object) = fa.object.as_ref() {
                            if let Some(info) = self.env.lookup(&object.name) {
                                if let Some(template) = &info.geometry_template {
                                    return self
                                        .geometry_ports
                                        .get(template)
                                        .and_then(|ports| ports.get(&fa.field.name))
                                        .cloned()
                                        .unwrap_or(ResolvedType::Inferred);
                                }
                            }
                        }
                        let parent = self.infer_expr_type(&fa.object);
                        match &parent {
                            ResolvedType::Struct(name) => {
                                if let Some(fields) = self.struct_fields.get(name) {
                                    fields
                                        .iter()
                                        .find(|(n, _)| n == &fa.field.name)
                                        .map(|(_, t)| t.clone())
                                        .unwrap_or(ResolvedType::Inferred)
                                } else {
                                    ResolvedType::Inferred
                                }
                            }
                            _ if parent.is_geometry() => ResolvedType::Solid,
                            _ => ResolvedType::Inferred,
                        }
                    }
                }
            }
            Expr::Call(c) => {
                if let Some(info) = self.env.lookup(&c.function.name) {
                    info.resolved_type.clone()
                } else {
                    ResolvedType::Inferred
                }
            }
            Expr::Paren(inner) => self.infer_expr_type(&inner.value),
            _ => ResolvedType::Inferred,
        }
    }

    fn infer_pipe_type(&self, pipe: &PipeExpr) -> ResolvedType {
        match pipe.function.name.as_str() {
            "extrude" | "revolve" | "sweep" | "twist" | "shell" | "fillet" | "chamfer"
            | "fragment" => ResolvedType::Solid,
            "translate" | "rotate" | "scale" | "mirror" | "copy" | "coherence" => {
                self.infer_expr_type(&pipe.value)
            }
            "project_iso_surface" | "project_signed_distance" | "project_closest_point" => {
                self.infer_expr_type(&pipe.value)
            }
            _ => ResolvedType::Inferred,
        }
    }

    fn check_pipe_transform(&mut self, pipe: &PipeExpr) {
        let input = self.infer_expr_type(&pipe.value);
        let expected = match pipe.function.name.as_str() {
            "extrude" | "revolve" | "twist" => Some(ResolvedType::Surface),
            "shell" | "fillet" | "chamfer" | "fragment" => Some(ResolvedType::Solid),
            _ => None,
        };

        if let Some(expected) = expected {
            if input != expected && input != ResolvedType::Inferred {
                self.diagnostics.push(Diagnostic {
                    level: DiagLevel::Error,
                    message: format!(
                        "transform '{}' expects {}, got {}",
                        pipe.function.name,
                        expected.display_name(),
                        input.display_name()
                    ),
                    span: pipe.span,
                    hint: None,
                });
            }
        }
    }

    fn is_csg_binary(&self, b: &BinaryExpr) -> bool {
        match b.op {
            BinaryOp::CsgUnion | BinaryOp::CsgDifference | BinaryOp::CsgIntersect => true,
            BinaryOp::Sub => {
                let lt = self.infer_expr_type(&b.left);
                let rt = self.infer_expr_type(&b.right);
                lt.is_csg_operand()
                    || rt.is_csg_operand()
                    || lt == ResolvedType::Inferred
                    || rt == ResolvedType::Inferred
            }
            _ => false,
        }
    }

    fn resolve_type_annotation(&self, ann: &meshlang_ast::types::TypeAnnotation) -> ResolvedType {
        use meshlang_ast::types::TypeAnnotation;
        match ann {
            TypeAnnotation::Named(n) => match n.name.as_str() {
                "int" | "Int" => ResolvedType::Int,
                "float" | "Float" => ResolvedType::Float,
                "bool" | "Bool" => ResolvedType::Bool,
                "string" | "String" => ResolvedType::String,
                "point" | "Point" => ResolvedType::Point,
                "curve" | "Curve" => ResolvedType::Curve,
                "surface" | "Surface" => ResolvedType::Surface,
                "solid" | "Solid" => ResolvedType::Solid,
                other => {
                    if self.struct_fields.contains_key(other) {
                        ResolvedType::Struct(other.to_string())
                    } else {
                        ResolvedType::Inferred
                    }
                }
            },
            TypeAnnotation::FixedArray { element, size, .. } => {
                ResolvedType::FixedArray(Box::new(self.resolve_type_annotation(element)), *size)
            }
            TypeAnnotation::DynArray { element, .. } => {
                ResolvedType::Array(Box::new(self.resolve_type_annotation(element)))
            }
            TypeAnnotation::Generic { .. } => ResolvedType::Inferred,
        }
    }

    fn infer_dimension(&self, expr: &Expr) -> Option<Dimension> {
        match expr {
            Expr::Ident(id) => self.env.lookup(&id.name).and_then(|info| info.dimension),
            Expr::Binary(b) => {
                let ld = self.infer_dimension(&b.left);
                let rd = self.infer_dimension(&b.right);

                match b.op {
                    BinaryOp::Add | BinaryOp::Sub => {
                        // Both must have same dimension
                        ld.or(rd)
                    }
                    BinaryOp::Mul => match (ld, rd) {
                        (Some(l), Some(r)) => l.checked_mul(r).ok(),
                        (Some(d), None) | (None, Some(d)) => Some(d),
                        _ => None,
                    },
                    BinaryOp::Div => match (ld, rd) {
                        (Some(l), Some(r)) => l
                            .checked_div(r)
                            .ok()
                            .and_then(|result| (!result.is_dimensionless()).then_some(result)),
                        (Some(d), None) => Some(d),
                        _ => None,
                    },
                    _ => None,
                }
            }
            Expr::Unary(u) => self.infer_dimension(&u.operand),
            Expr::Paren(inner) => self.infer_dimension(&inner.value),
            _ => None,
        }
    }
}

fn block_has_inline_value(block: &Block) -> bool {
    block
        .statements
        .last()
        .is_some_and(statement_has_inline_value)
}

fn statement_has_inline_value(statement: &Statement) -> bool {
    match statement {
        Statement::Expr(_) | Statement::Let(_) => true,
        Statement::If(statement) => {
            block_has_inline_value(&statement.then_block)
                && match &statement.else_block {
                    Some(ElseBranch::Else(block)) => block_has_inline_value(block),
                    Some(ElseBranch::ElseIf(statement)) => {
                        statement_has_inline_value(&Statement::If((**statement).clone()))
                    }
                    None => false,
                }
        }
        Statement::For(statement) => block_has_inline_value(&statement.body),
        Statement::KeyValue(_) | Statement::Group(_) => false,
    }
}

fn geometry_template_for_expr(
    expr: &Expr,
    geometry_ports: &HashMap<String, HashMap<String, ResolvedType>>,
) -> Option<String> {
    match expr {
        Expr::Constructor(constructor)
            if geometry_ports.contains_key(&constructor.type_name.name) =>
        {
            Some(constructor.type_name.name.clone())
        }
        Expr::Paren(paren) => geometry_template_for_expr(&paren.value, geometry_ports),
        _ => None,
    }
}

fn is_csg_op(op: BinaryOp) -> bool {
    matches!(
        op,
        BinaryOp::CsgUnion | BinaryOp::CsgDifference | BinaryOp::CsgIntersect | BinaryOp::Sub
    )
}

fn collect_ident_refs(expr: &Expr, refs: &mut Vec<String>) {
    match expr {
        Expr::Ident(id) => refs.push(id.name.clone()),
        Expr::Binary(b) => {
            collect_ident_refs(&b.left, refs);
            collect_ident_refs(&b.right, refs);
        }
        Expr::Unary(u) => collect_ident_refs(&u.operand, refs),
        Expr::Pipe(p) => {
            collect_ident_refs(&p.value, refs);
            for arg in &p.args {
                collect_ident_refs(&arg.value, refs);
            }
        }
        Expr::Constructor(c) => {
            for arg in &c.args {
                collect_ident_refs(&arg.value, refs);
            }
        }
        Expr::Call(c) => {
            for arg in &c.args {
                collect_ident_refs(&arg.value, refs);
            }
        }
        Expr::FieldAccess(fa) => collect_ident_refs(&fa.object, refs),
        Expr::Index(idx) => {
            collect_ident_refs(&idx.object, refs);
            collect_ident_refs(&idx.index, refs);
        }
        Expr::Paren(inner) => collect_ident_refs(&inner.value, refs),
        Expr::Array(arr) => {
            for elem in &arr.elements {
                collect_ident_refs(elem, refs);
            }
        }
        Expr::StructInit(si) => {
            for field in &si.fields {
                collect_ident_refs(&field.value, refs);
            }
        }
        _ => {}
    }
}

fn is_builtin_keyword(name: &str) -> bool {
    matches!(
        name,
        "X" | "Y"
            | "Z"
            | "XY"
            | "XZ"
            | "YZ"
            | "msh4"
            | "msh2"
            | "vtk"
            | "stl"
            | "obj"
            | "nastran"
            | "abaqus"
            | "med"
            | "delaunay"
            | "frontal"
            | "meshadapt"
            | "pack_parallelograms"
            | "hxt"
            | "true"
            | "false"
            | "pi"
            | "PI"
    )
}

fn try_eval_const_expr(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Literal(LiteralExpr {
            value: Literal::Float(v),
            ..
        }) => Some(*v),
        Expr::Literal(LiteralExpr {
            value: Literal::Int(v),
            ..
        }) => Some(*v as f64),
        Expr::Ident(id) if id.name == "pi" || id.name == "PI" => Some(std::f64::consts::PI),
        Expr::Binary(b) => {
            let l = try_eval_const_expr(&b.left)?;
            let r = try_eval_const_expr(&b.right)?;
            match b.op {
                BinaryOp::Add => Some(l + r),
                BinaryOp::Sub => Some(l - r),
                BinaryOp::Mul => Some(l * r),
                BinaryOp::Div if r != 0.0 => Some(l / r),
                BinaryOp::Pow => Some(l.powf(r)),
                _ => None,
            }
        }
        Expr::Unary(u) if u.op == UnaryOp::Neg => try_eval_const_expr(&u.operand).map(|v| -v),
        Expr::Paren(inner) => try_eval_const_expr(&inner.value),
        _ => None,
    }
}

fn static_number(expr: &Expr) -> Option<f64> {
    try_eval_const_expr(expr)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check_src(src: &str) -> Vec<Diagnostic> {
        let ast = meshlang_parser::parse(src, "test.mlg").unwrap();
        let mut checker = TypeChecker::new();
        checker.check(&ast)
    }

    #[test]
    fn test_no_errors_on_valid() {
        let diags = check_src(
            r#"
            param size = 0.01
            geometry box_test {
                let outer = Box(1.0, 2.0, 3.0)
            }
            mesh for box_test {
                default_size: 0.01
            }
            export {
                format: msh4
                filename: "test.msh"
            }
        "#,
        );
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.level == DiagLevel::Error)
            .collect();
        assert!(errors.is_empty(), "unexpected errors: {:?}", errors);
    }

    #[test]
    fn test_unused_geometry_warning() {
        let diags = check_src(
            r#"
            geometry unused_box {
                let outer = Box(1.0, 2.0, 3.0)
            }
        "#,
        );
        let warnings: Vec<_> = diags
            .iter()
            .filter(|d| d.level == DiagLevel::Warning && d.message.contains("unused_box"))
            .collect();
        assert!(!warnings.is_empty(), "expected unused geometry warning");
    }

    #[test]
    fn test_undefined_mesh_target() {
        let diags = check_src(
            r#"
            mesh for nonexistent {
                default_size: 0.01
            }
        "#,
        );
        let errors: Vec<_> = diags
            .iter()
            .filter(|d| d.level == DiagLevel::Error && d.message.contains("nonexistent"))
            .collect();
        assert!(!errors.is_empty(), "expected undefined geometry error");
    }

    #[test]
    fn geometry_locals_do_not_leak_between_blocks() {
        let diagnostics = check_src(
            r#"
                geometry first { let private_shape = Box(1.0, 1.0, 1.0) }
                geometry second { let result = private_shape }
            "#,
        );
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.message.contains("private_shape") && diagnostic.message.contains("undefined")
        }));
    }

    #[test]
    fn mesh_metrics_require_a_declared_coordinate_system() {
        let diagnostics = check_src(
            r#"
                geometry model { let panel = Rectangle(1.0, 1.0) }
                mesh for model {
                    metric model.panel {
                        kind: isotropic
                        size: 0.1
                        coordinate_system: missing
                    }
                }
            "#,
        );
        assert!(diagnostics.iter().any(|diagnostic| {
            diagnostic.level == DiagLevel::Error
                && diagnostic
                    .message
                    .contains("undefined coordinate system 'missing'")
        }));
    }

    #[test]
    fn optimization_metadata_rejects_invalid_static_numeric_contracts() {
        let diagnostics = check_src(
            r#"
                geometry domain { let body = Box(1.0, 1.0, 1.0) }
                design bad_bounds {
                    type: continuous,
                    initial: 3.0,
                    lower: 2.0,
                    upper: 1.0
                }
                design bad_integer {
                    type: integer,
                    initial: 1.5,
                    lower: 0.0,
                    upper: 3.5
                }
                uncertainty backwards {
                    distribution: uniform(lower: 1.0, upper: 0.0)
                }
                uncertainty invalid_normal {
                    distribution: normal(mean: 0.0, standard_deviation: 0.0)
                }
                uncertainty bad_discrete {
                    distribution: discrete(values: [1, 2], weights: [1, -1, 0])
                }
                constraint fraction {
                    kind: volume_fraction,
                    region: domain.body,
                    maximum: 1.1
                }
                constraint feature {
                    kind: minimum_feature,
                    region: domain.body,
                    radius: 0.0
                }
                design_region manufacturing {
                    kind: design,
                    region: domain.body,
                    annotations: [minimum_thickness(value: 0.0)]
                }
                field density {
                    kind: topology,
                    region: domain.body,
                    initial_density: -0.1,
                    filter_radius: 0.0
                }
            "#,
        );
        let messages = diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.level == DiagLevel::Error)
            .map(|diagnostic| diagnostic.message.as_str())
            .collect::<Vec<_>>();
        for expected in [
            "design variable bounds must be finite and ordered",
            "design variable initial value must lie within its declared bounds",
            "integer design variable initial value must be integral",
            "integer design variable upper bound must be integral",
            "uniform distribution lower bound must be less than its upper bound",
            "distribution standard deviation must be positive",
            "discrete distribution values and weights must have equal lengths",
            "discrete distribution weights must have a finite positive sum",
            "volume-fraction constraint maximum must lie in [0, 1]",
            "minimum-feature constraint radius must be positive",
            "manufacturing minimum thickness/radius must be positive",
            "topology-field initial density must lie in [0, 1]",
            "topology-field filter radius must be positive",
        ] {
            assert!(
                messages.iter().any(|message| message.contains(expected)),
                "missing semantic error {expected:?}; got {messages:?}"
            );
        }
    }

    #[test]
    fn optimization_metadata_accepts_dynamic_values_and_well_formed_constants() {
        let diagnostics = check_src(
            r#"
                param density_seed = 0.5
                param radius = 0.02
                geometry domain { let body = Box(1.0, 1.0, 1.0) }
                design valid {
                    type: integer,
                    initial: 2,
                    lower: 0,
                    upper: 4,
                    distribution: discrete(values: [0, 2, 4], weights: [0.2, 0.5, 0.3])
                }
                uncertainty tolerance {
                    distribution: normal(mean: density_seed, standard_deviation: 0.1)
                }
                constraint fraction {
                    kind: volume_fraction,
                    region: domain.body,
                    maximum: density_seed
                }
                design_region manufacturing {
                    kind: design,
                    region: domain.body,
                    annotations: [minimum_radius(value: radius)]
                }
                field density {
                    kind: topology,
                    region: domain.body,
                    initial_density: density_seed,
                    filter_radius: radius
                }
            "#,
        );
        let errors = diagnostics
            .iter()
            .filter(|diagnostic| diagnostic.level == DiagLevel::Error)
            .collect::<Vec<_>>();
        assert!(errors.is_empty(), "unexpected semantic errors: {errors:?}");
    }

    #[test]
    fn optimization_semantic_errors_block_backend_emission() {
        let error = crate::compile_to_geo(
            r#"
                geometry domain { let body = Box(1.0, 1.0, 1.0) }
                field density {
                    kind: topology,
                    region: domain.body,
                    initial_density: 2.0,
                    filter_radius: -0.1
                }
            "#,
            "invalid-optimization.mlg",
        )
        .expect_err("invalid optimization metadata must not reach GEO emission");
        let crate::CompileError::TypeError { message } = error else {
            panic!("expected typed semantic failure, got {error:?}");
        };
        assert!(message.contains("topology-field initial density must lie in [0, 1]"));
        assert!(message.contains("topology-field filter radius must be positive"));
    }
}
