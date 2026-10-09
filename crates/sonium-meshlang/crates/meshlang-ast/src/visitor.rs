//! Visitor pattern for AST traversal.
//!
//! Implement `Visitor` for analysis/transform passes (type checking, lowering, etc.)

use crate::*;

/// Visitor trait for immutable AST traversal.
pub trait Visitor: Sized {
    fn visit_source_file(&mut self, file: &SourceFile) {
        walk_source_file(self, file);
    }

    fn visit_declaration(&mut self, decl: &Declaration) {
        walk_declaration(self, decl);
    }

    fn visit_param_decl(&mut self, decl: &ParamDecl) {
        self.visit_expr(&decl.value);
    }
    fn visit_const_decl(&mut self, decl: &ConstDecl) {
        self.visit_expr(&decl.value);
    }
    fn visit_import_decl(&mut self, _decl: &ImportDecl) {}
    fn visit_fn_decl(&mut self, decl: &FnDecl) {
        for param in &decl.params {
            if let Some(default) = &param.default {
                self.visit_expr(default);
            }
        }
        for statement in &decl.body.statements {
            self.visit_statement(statement);
        }
    }
    fn visit_geometry_block(&mut self, block: &GeometryBlock) {
        for param in &block.params {
            if let Some(default) = &param.default {
                self.visit_expr(default);
            }
        }
        for port in &block.ports {
            if let Some(default) = &port.default {
                self.visit_expr(default);
            }
        }
        for statement in &block.body.statements {
            self.visit_statement(statement);
        }
    }
    fn visit_mesh_block(&mut self, block: &MeshBlock) {
        for statement in &block.statements {
            walk_mesh_statement(self, statement);
        }
    }
    fn visit_physics_block(&mut self, block: &PhysicsBlock) {
        for statement in &block.statements {
            match statement {
                PhysicsStatement::Group(group) => self.visit_group_decl(group),
                PhysicsStatement::Let(statement) => self.visit_let_statement(statement),
            }
        }
    }
    fn visit_export_block(&mut self, block: &ExportBlock) {
        for setting in &block.settings {
            self.visit_expr(&setting.value);
        }
    }
    fn visit_struct_decl(&mut self, _decl: &StructDecl) {}
    fn visit_design_variable_decl(&mut self, decl: &DesignVariableDecl) {
        self.visit_expr(&decl.initial);
        if let Some(bounds) = &decl.bounds {
            walk_design_bounds(self, bounds);
        }
        if let Some(distribution) = &decl.distribution {
            walk_distribution(self, distribution);
        }
        if let DesignVariableType::Categorical { values } = &decl.variable_type {
            walk_exprs(self, values);
        }
    }
    fn visit_uncertainty_decl(&mut self, decl: &UncertaintyDecl) {
        walk_distribution(self, &decl.distribution);
    }
    fn visit_objective_decl(&mut self, decl: &ObjectiveDecl) {
        self.visit_expr(&decl.expression);
        if let Some(weight) = &decl.weight {
            self.visit_expr(weight);
        }
    }
    fn visit_constraint_decl(&mut self, decl: &ConstraintDecl) {
        walk_constraint(self, &decl.kind);
    }
    fn visit_design_region_decl(&mut self, decl: &DesignRegionDecl) {
        self.visit_expr(&decl.region);
        for annotation in &decl.annotations {
            walk_manufacturing_annotation(self, annotation);
        }
    }
    fn visit_physical_region_decl(&mut self, decl: &PhysicalRegionDecl) {
        self.visit_expr(&decl.entities);
    }
    fn visit_boundary_decl(&mut self, decl: &BoundaryDecl) {
        self.visit_expr(&decl.entities);
        walk_key_values(self, &decl.settings);
    }
    fn visit_interface_decl(&mut self, decl: &InterfaceDecl) {
        self.visit_expr(&decl.primary);
        self.visit_expr(&decl.secondary);
        walk_key_values(self, &decl.settings);
    }
    fn visit_coordinate_system_decl(&mut self, decl: &CoordinateSystemDecl) {
        walk_coordinate_system(self, &decl.kind);
    }
    fn visit_material_region_decl(&mut self, decl: &MaterialRegionDecl) {
        self.visit_expr(&decl.region);
        walk_key_values(self, &decl.properties);
    }
    fn visit_geometry_field_decl(&mut self, decl: &GeometryFieldDecl) {
        walk_geometry_field(self, &decl.kind);
    }

    fn visit_statement(&mut self, stmt: &Statement) {
        walk_statement(self, stmt);
    }

    fn visit_let_statement(&mut self, stmt: &LetStatement) {
        self.visit_expr(&stmt.value);
    }
    fn visit_for_statement(&mut self, stmt: &ForStatement) {
        for clause in &stmt.clauses {
            self.visit_expr(&clause.range.start);
            self.visit_expr(&clause.range.end);
        }
        for statement in &stmt.body.statements {
            self.visit_statement(statement);
        }
    }
    fn visit_if_statement(&mut self, stmt: &IfStatement) {
        self.visit_expr(&stmt.condition);
        for statement in &stmt.then_block.statements {
            self.visit_statement(statement);
        }
        if let Some(branch) = &stmt.else_block {
            match branch {
                ElseBranch::Else(block) => {
                    for statement in &block.statements {
                        self.visit_statement(statement);
                    }
                }
                ElseBranch::ElseIf(statement) => self.visit_if_statement(statement),
            }
        }
    }

    fn visit_expr(&mut self, expr: &Expr) {
        walk_expr(self, expr);
    }

    fn visit_binary_expr(&mut self, expr: &BinaryExpr) {
        self.visit_expr(&expr.left);
        self.visit_expr(&expr.right);
    }
    fn visit_unary_expr(&mut self, expr: &UnaryExpr) {
        self.visit_expr(&expr.operand);
    }
    fn visit_pipe_expr(&mut self, expr: &PipeExpr) {
        self.visit_expr(&expr.value);
        for argument in &expr.args {
            self.visit_expr(&argument.value);
        }
    }
    fn visit_constructor_expr(&mut self, expr: &ConstructorExpr) {
        for argument in &expr.args {
            self.visit_expr(&argument.value);
        }
    }
    fn visit_call_expr(&mut self, expr: &CallExpr) {
        for argument in &expr.args {
            self.visit_expr(&argument.value);
        }
    }
    fn visit_struct_init_expr(&mut self, expr: &StructInitExpr) {
        for field in &expr.fields {
            self.visit_expr(&field.value);
        }
    }
    fn visit_field_access(&mut self, expr: &FieldAccessExpr) {
        self.visit_expr(&expr.object);
    }
    fn visit_group_decl(&mut self, decl: &GroupDecl) {
        self.visit_expr(&decl.value);
    }
}

pub fn walk_source_file<V: Visitor>(v: &mut V, file: &SourceFile) {
    for decl in &file.declarations {
        v.visit_declaration(decl);
    }
}

pub fn walk_declaration<V: Visitor>(v: &mut V, decl: &Declaration) {
    match decl {
        Declaration::Param(d) => v.visit_param_decl(d),
        Declaration::Const(d) => v.visit_const_decl(d),
        Declaration::Import(d) => v.visit_import_decl(d),
        Declaration::Function(d) => v.visit_fn_decl(d),
        Declaration::Geometry(b) => v.visit_geometry_block(b),
        Declaration::Mesh(b) => v.visit_mesh_block(b),
        Declaration::Physics(b) => v.visit_physics_block(b),
        Declaration::Export(b) => v.visit_export_block(b),
        Declaration::Struct(d) => v.visit_struct_decl(d),
        Declaration::DesignVariable(d) => v.visit_design_variable_decl(d),
        Declaration::Uncertainty(d) => v.visit_uncertainty_decl(d),
        Declaration::Objective(d) => v.visit_objective_decl(d),
        Declaration::Constraint(d) => v.visit_constraint_decl(d),
        Declaration::DesignRegion(d) => v.visit_design_region_decl(d),
        Declaration::PhysicalRegion(d) => v.visit_physical_region_decl(d),
        Declaration::Boundary(d) => v.visit_boundary_decl(d),
        Declaration::Interface(d) => v.visit_interface_decl(d),
        Declaration::CoordinateSystem(d) => v.visit_coordinate_system_decl(d),
        Declaration::MaterialRegion(d) => v.visit_material_region_decl(d),
        Declaration::GeometryField(d) => v.visit_geometry_field_decl(d),
    }
}

pub fn walk_statement<V: Visitor>(v: &mut V, stmt: &Statement) {
    match stmt {
        Statement::Let(s) => v.visit_let_statement(s),
        Statement::Expr(e) => v.visit_expr(e),
        Statement::For(s) => v.visit_for_statement(s),
        Statement::If(s) => v.visit_if_statement(s),
        Statement::KeyValue(key_value) => v.visit_expr(&key_value.value),
        Statement::Group(g) => v.visit_group_decl(g),
    }
}

/// Walk every expression contained in a mesh-control statement.
pub fn walk_mesh_statement<V: Visitor>(v: &mut V, statement: &MeshStatement) {
    match statement {
        MeshStatement::KeyValue(setting) => v.visit_expr(&setting.value),
        MeshStatement::Refine(refine) => {
            match &refine.kind {
                RefineKind::Near { entity, args } => {
                    v.visit_expr(entity);
                    walk_key_values(v, args);
                }
                RefineKind::On { entity } => v.visit_expr(entity),
                RefineKind::In { region } => v.visit_expr(region),
                RefineKind::Field { args } => walk_key_values(v, args),
            }
            walk_key_values(v, &refine.body);
        }
        MeshStatement::BoundaryLayer(layer) => {
            v.visit_expr(&layer.target);
            walk_key_values(v, &layer.body);
        }
        MeshStatement::Transfinite(transfinite) => match &transfinite.kind {
            TransfiniteKind::Curve { entity, settings }
            | TransfiniteKind::Surface { entity, settings }
            | TransfiniteKind::Volume { entity, settings } => {
                v.visit_expr(entity);
                walk_key_values(v, settings);
            }
        },
        MeshStatement::Embed(embed) => {
            v.visit_expr(&embed.source);
            v.visit_expr(&embed.target);
        }
        MeshStatement::Periodic(periodic) => {
            v.visit_expr(&periodic.source);
            v.visit_expr(&periodic.target);
            v.visit_expr(&periodic.transform);
        }
        MeshStatement::Compound(compound) => {
            for entity in &compound.entities {
                v.visit_expr(entity);
            }
        }
        MeshStatement::Metric(metric) => {
            v.visit_expr(&metric.target);
            match &metric.metric {
                MeshMetricKind::Isotropic { size } => v.visit_expr(size),
                MeshMetricKind::Anisotropic { components } => walk_exprs(v, components),
                MeshMetricKind::Hessian {
                    field,
                    minimum_size,
                    maximum_size,
                } => {
                    v.visit_expr(field);
                    v.visit_expr(minimum_size);
                    v.visit_expr(maximum_size);
                }
            }
        }
    }
}

fn walk_key_values<V: Visitor>(v: &mut V, settings: &[KeyValue]) {
    for setting in settings {
        v.visit_expr(&setting.value);
    }
}

fn walk_exprs<V: Visitor>(v: &mut V, expressions: &[Expr]) {
    for expression in expressions {
        v.visit_expr(expression);
    }
}

fn walk_design_bounds<V: Visitor>(v: &mut V, bounds: &DesignBounds) {
    v.visit_expr(&bounds.lower);
    v.visit_expr(&bounds.upper);
}

fn walk_distribution<V: Visitor>(v: &mut V, distribution: &Distribution) {
    match distribution {
        Distribution::Deterministic { value } => v.visit_expr(value),
        Distribution::Uniform { lower, upper } => {
            v.visit_expr(lower);
            v.visit_expr(upper);
        }
        Distribution::Normal {
            mean,
            standard_deviation,
        }
        | Distribution::LogNormal {
            mean,
            standard_deviation,
        } => {
            v.visit_expr(mean);
            v.visit_expr(standard_deviation);
        }
        Distribution::Discrete { values, weights } => {
            walk_exprs(v, values);
            walk_exprs(v, weights);
        }
    }
}

fn walk_constraint<V: Visitor>(v: &mut V, constraint: &ConstraintKind) {
    match constraint {
        ConstraintKind::Equality {
            left,
            right,
            tolerance,
        } => {
            v.visit_expr(left);
            v.visit_expr(right);
            if let Some(tolerance) = tolerance {
                v.visit_expr(tolerance);
            }
        }
        ConstraintKind::Inequality { left, right, .. } => {
            v.visit_expr(left);
            v.visit_expr(right);
        }
        ConstraintKind::Range { expression, bounds } => {
            v.visit_expr(expression);
            walk_design_bounds(v, bounds);
        }
        ConstraintKind::VolumeFraction { region, maximum } => {
            v.visit_expr(region);
            v.visit_expr(maximum);
        }
        ConstraintKind::MinimumFeature { region, radius } => {
            v.visit_expr(region);
            v.visit_expr(radius);
        }
    }
}

fn walk_manufacturing_annotation<V: Visitor>(v: &mut V, annotation: &ManufacturingAnnotation) {
    match annotation {
        ManufacturingAnnotation::MinimumThickness { value, .. }
        | ManufacturingAnnotation::MinimumRadius { value, .. } => v.visit_expr(value),
        ManufacturingAnnotation::Overhang {
            maximum_angle,
            direction,
        } => {
            v.visit_expr(maximum_angle);
            v.visit_expr(direction);
        }
        ManufacturingAnnotation::Symmetry { kind } => match kind {
            SymmetryKind::Plane { origin, normal } => {
                v.visit_expr(origin);
                v.visit_expr(normal);
            }
            SymmetryKind::Axis { origin, direction } => {
                v.visit_expr(origin);
                v.visit_expr(direction);
            }
            SymmetryKind::Cyclic { axis, sectors } => {
                v.visit_expr(axis);
                v.visit_expr(sectors);
            }
        },
        ManufacturingAnnotation::DrawDirection { direction } => v.visit_expr(direction),
        ManufacturingAnnotation::Frozen => {}
    }
}

fn walk_coordinate_system<V: Visitor>(v: &mut V, system: &CoordinateSystemKind) {
    match system {
        CoordinateSystemKind::Cartesian { origin, axes } => {
            v.visit_expr(origin);
            walk_exprs(v, axes);
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
            v.visit_expr(origin);
            v.visit_expr(axis);
            v.visit_expr(reference);
        }
    }
}

fn walk_geometry_field<V: Visitor>(v: &mut V, field: &GeometryFieldKind) {
    match field {
        GeometryFieldKind::Topology {
            region,
            initial_density,
            filter_radius,
        } => {
            v.visit_expr(region);
            v.visit_expr(initial_density);
            if let Some(filter_radius) = filter_radius {
                v.visit_expr(filter_radius);
            }
        }
        GeometryFieldKind::LevelSet { field, iso_value } => {
            v.visit_expr(field);
            v.visit_expr(iso_value);
        }
        GeometryFieldKind::Projected {
            geometry,
            provenance,
        } => {
            v.visit_expr(geometry);
            if let Some(tolerance) = &provenance.tolerance {
                v.visit_expr(tolerance);
            }
        }
    }
}

pub fn walk_expr<V: Visitor>(v: &mut V, expr: &Expr) {
    match expr {
        Expr::Ident(_) | Expr::Literal(_) => {}
        Expr::Binary(e) => v.visit_binary_expr(e),
        Expr::Unary(e) => v.visit_unary_expr(e),
        Expr::Pipe(e) => v.visit_pipe_expr(e),
        Expr::FieldAccess(e) => v.visit_field_access(e),
        Expr::Index(e) => {
            v.visit_expr(&e.object);
            v.visit_expr(&e.index);
        }
        Expr::Constructor(e) => v.visit_constructor_expr(e),
        Expr::Call(e) => v.visit_call_expr(e),
        Expr::Paren(paren) => v.visit_expr(&paren.value),
        Expr::Array(a) => {
            for elem in &a.elements {
                v.visit_expr(elem);
            }
        }
        Expr::StructInit(e) => v.visit_struct_init_expr(e),
        Expr::If(s) => v.visit_if_statement(s),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CountingVisitor {
        expressions: usize,
    }

    impl Visitor for CountingVisitor {
        fn visit_expr(&mut self, expression: &Expr) {
            self.expressions += 1;
            walk_expr(self, expression);
        }
    }

    #[test]
    fn default_visitor_recurses_through_nested_expressions() {
        let expression = Expr::Binary(BinaryExpr {
            left: Box::new(Expr::literal(Literal::Int(1), Span::DUMMY)),
            op: BinaryOp::Add,
            right: Box::new(Expr::Unary(UnaryExpr {
                op: UnaryOp::Neg,
                operand: Box::new(Expr::literal(Literal::Int(2), Span::DUMMY)),
                span: Span::DUMMY,
            })),
            span: Span::DUMMY,
        });
        let mut visitor = CountingVisitor { expressions: 0 };
        visitor.visit_expr(&expression);
        assert_eq!(visitor.expressions, 4);
    }

    fn literal(value: i64) -> Expr {
        Expr::literal(Literal::Int(value), Span::DUMMY)
    }

    fn ident(name: &str) -> Ident {
        Ident {
            name: name.into(),
            span: Span::DUMMY,
        }
    }

    fn setting(value: i64) -> KeyValue {
        KeyValue {
            key: ident("value"),
            value: literal(value),
            unit: None,
            span: Span::DUMMY,
        }
    }

    fn argument(value: i64) -> Argument {
        Argument {
            name: Some(ident("value")),
            value: literal(value),
            span: Span::DUMMY,
        }
    }

    #[test]
    fn visitor_reaches_every_expression_variant_and_nested_child() {
        let expressions = vec![
            Expr::Ident(ident("identifier")),
            literal(0),
            Expr::Binary(BinaryExpr {
                left: Box::new(literal(1)),
                op: BinaryOp::Add,
                right: Box::new(literal(2)),
                span: Span::DUMMY,
            }),
            Expr::Unary(UnaryExpr {
                op: UnaryOp::Neg,
                operand: Box::new(literal(3)),
                span: Span::DUMMY,
            }),
            Expr::Pipe(PipeExpr {
                value: Box::new(literal(4)),
                function: ident("translate"),
                args: vec![argument(5)],
                span: Span::DUMMY,
            }),
            Expr::FieldAccess(FieldAccessExpr {
                object: Box::new(literal(6)),
                field: ident("x"),
                span: Span::DUMMY,
            }),
            Expr::Index(IndexExpr {
                object: Box::new(literal(7)),
                index: Box::new(literal(8)),
                span: Span::DUMMY,
            }),
            Expr::Constructor(ConstructorExpr {
                type_name: ident("Point"),
                args: vec![argument(9)],
                span: Span::DUMMY,
            }),
            Expr::Call(CallExpr {
                function: ident("size"),
                args: vec![argument(10)],
                span: Span::DUMMY,
            }),
            Expr::Paren(ParenExpr::new(literal(11), Span::DUMMY)),
            Expr::Array(ArrayExpr {
                elements: vec![literal(12), literal(13)],
                span: Span::DUMMY,
            }),
            Expr::StructInit(StructInitExpr {
                type_name: ident("Frame"),
                fields: vec![StructInitField {
                    name: ident("origin"),
                    value: literal(14),
                    span: Span::DUMMY,
                }],
                span: Span::DUMMY,
            }),
            Expr::If(Box::new(IfStatement {
                condition: literal(15),
                then_block: Block {
                    statements: vec![Statement::Expr(literal(16))],
                    span: Span::DUMMY,
                },
                else_block: None,
                span: Span::DUMMY,
            })),
        ];

        let mut visitor = CountingVisitor { expressions: 0 };
        assert_eq!(Expr::VARIANT_NAMES.len(), expressions.len());
        for expression in &expressions {
            visitor.visit_expr(expression);
        }

        // The 13 inventory variants are all exercised, including every nested
        // expression edge (29 visits total). This catches a visitor branch
        // that compiles but silently drops a child expression.
        assert_eq!(visitor.expressions, 29);
    }

    #[test]
    fn visitor_reaches_every_mesh_statement_branch() {
        let mesh = MeshBlock {
            target: None,
            statements: vec![
                MeshStatement::Refine(RefineStatement {
                    kind: RefineKind::On { entity: literal(1) },
                    body: vec![setting(2)],
                    span: Span::DUMMY,
                }),
                MeshStatement::Refine(RefineStatement {
                    kind: RefineKind::In { region: literal(3) },
                    body: vec![setting(4)],
                    span: Span::DUMMY,
                }),
                MeshStatement::Refine(RefineStatement {
                    kind: RefineKind::Field {
                        args: vec![setting(5)],
                    },
                    body: vec![setting(6)],
                    span: Span::DUMMY,
                }),
                MeshStatement::Transfinite(TransfiniteStatement {
                    kind: TransfiniteKind::Surface {
                        entity: literal(7),
                        settings: vec![setting(8)],
                    },
                    span: Span::DUMMY,
                }),
                MeshStatement::Transfinite(TransfiniteStatement {
                    kind: TransfiniteKind::Volume {
                        entity: literal(9),
                        settings: vec![setting(10)],
                    },
                    span: Span::DUMMY,
                }),
                MeshStatement::Metric(MeshMetricStatement {
                    target: literal(11),
                    metric: MeshMetricKind::Anisotropic {
                        components: vec![literal(12), literal(13)],
                    },
                    coordinate_system: Some(ident("local")),
                    span: Span::DUMMY,
                }),
                MeshStatement::Metric(MeshMetricStatement {
                    target: literal(14),
                    metric: MeshMetricKind::Hessian {
                        field: literal(15),
                        minimum_size: literal(16),
                        maximum_size: literal(17),
                    },
                    coordinate_system: None,
                    span: Span::DUMMY,
                }),
            ],
            span: Span::DUMMY,
        };

        let mut visitor = CountingVisitor { expressions: 0 };
        visitor.visit_mesh_block(&mesh);

        // Three refine branches, two transfinite dimensions, and both
        // non-isotropic metric forms are covered (17 expression visits).
        assert_eq!(visitor.expressions, 17);
    }

    #[test]
    fn default_visitor_recurses_through_declaration_and_mesh_bodies() {
        let mut visitor = CountingVisitor { expressions: 0 };
        visitor.visit_param_decl(&ParamDecl {
            name: ident("parameter"),
            type_ann: None,
            value: literal(1),
            unit: None,
            span: Span::DUMMY,
        });
        visitor.visit_fn_decl(&FnDecl {
            is_pub: false,
            name: ident("function"),
            params: Vec::new(),
            return_type: None,
            body: Block {
                statements: vec![Statement::Expr(literal(2))],
                span: Span::DUMMY,
            },
            span: Span::DUMMY,
        });
        visitor.visit_geometry_block(&GeometryBlock {
            is_pub: false,
            name: ident("geometry"),
            params: Vec::new(),
            ports: vec![ParamDef {
                name: ident("outlet"),
                type_ann: TypeAnnotation::Named(crate::types::NamedType {
                    name: "Solid".into(),
                    span: Span::DUMMY,
                }),
                default: Some(literal(30)),
                span: Span::DUMMY,
            }],
            body: Block {
                statements: vec![Statement::Let(LetStatement {
                    name: ident("shape"),
                    type_ann: None,
                    value: literal(3),
                    span: Span::DUMMY,
                })],
                span: Span::DUMMY,
            },
            span: Span::DUMMY,
        });
        visitor.visit_mesh_block(&MeshBlock {
            target: None,
            statements: vec![
                MeshStatement::KeyValue(setting(4)),
                MeshStatement::Refine(RefineStatement {
                    kind: RefineKind::Near {
                        entity: literal(15),
                        args: vec![setting(16)],
                    },
                    body: vec![setting(17)],
                    span: Span::DUMMY,
                }),
                MeshStatement::BoundaryLayer(BoundaryLayerStatement {
                    target: literal(18),
                    body: vec![setting(19)],
                    span: Span::DUMMY,
                }),
                MeshStatement::Transfinite(TransfiniteStatement {
                    kind: TransfiniteKind::Curve {
                        entity: literal(20),
                        settings: vec![setting(21)],
                    },
                    span: Span::DUMMY,
                }),
                MeshStatement::Embed(EmbedStatement {
                    embed_dim: EmbedDim::Curve,
                    source: literal(5),
                    target: literal(6),
                    target_dim: EmbedDim::Surface,
                    span: Span::DUMMY,
                }),
                MeshStatement::Periodic(PeriodicStatement {
                    dim: PeriodicDim::Surface,
                    source: literal(7),
                    target: literal(8),
                    transform: literal(9),
                    span: Span::DUMMY,
                }),
                MeshStatement::Compound(CompoundStatement {
                    entities: vec![literal(10), literal(11)],
                    span: Span::DUMMY,
                }),
                MeshStatement::Metric(MeshMetricStatement {
                    target: literal(22),
                    metric: MeshMetricKind::Isotropic { size: literal(23) },
                    coordinate_system: None,
                    span: Span::DUMMY,
                }),
            ],
            span: Span::DUMMY,
        });
        visitor.visit_physics_block(&PhysicsBlock {
            statements: vec![
                PhysicsStatement::Group(GroupDecl {
                    name: ident("wall"),
                    bc_type: ident("neumann"),
                    value: literal(12),
                    span: Span::DUMMY,
                }),
                PhysicsStatement::Let(LetStatement {
                    name: ident("coefficient"),
                    type_ann: None,
                    value: literal(13),
                    span: Span::DUMMY,
                }),
            ],
            span: Span::DUMMY,
        });
        visitor.visit_export_block(&ExportBlock {
            name: None,
            settings: vec![setting(14)],
            span: Span::DUMMY,
        });

        assert_eq!(visitor.expressions, 24);
    }

    #[test]
    fn exhaustive_visitor_reaches_every_metadata_declaration() {
        let declarations = vec![
            Declaration::DesignVariable(DesignVariableDecl {
                name: ident("shape"),
                variable_type: DesignVariableType::Categorical {
                    values: vec![literal(6), literal(7)],
                },
                initial: literal(1),
                bounds: Some(DesignBounds {
                    lower: literal(2),
                    upper: literal(3),
                    lower_inclusive: true,
                    upper_inclusive: true,
                    span: Span::DUMMY,
                }),
                unit: None,
                distribution: Some(Distribution::Uniform {
                    lower: literal(4),
                    upper: literal(5),
                }),
                span: Span::DUMMY,
            }),
            Declaration::Uncertainty(UncertaintyDecl {
                name: ident("load"),
                distribution: Distribution::Deterministic { value: literal(8) },
                span: Span::DUMMY,
            }),
            Declaration::Objective(ObjectiveDecl {
                name: Some(ident("mass")),
                sense: ObjectiveSense::Minimize,
                expression: literal(9),
                weight: Some(literal(10)),
                span: Span::DUMMY,
            }),
            Declaration::Constraint(ConstraintDecl {
                name: Some(ident("volume")),
                kind: ConstraintKind::Equality {
                    left: literal(11),
                    right: literal(12),
                    tolerance: Some(literal(13)),
                },
                span: Span::DUMMY,
            }),
            Declaration::DesignRegion(DesignRegionDecl {
                name: ident("domain"),
                region: literal(14),
                kind: DesignRegionKind::Design,
                annotations: vec![
                    ManufacturingAnnotation::MinimumThickness {
                        value: literal(15),
                        unit: None,
                    },
                    ManufacturingAnnotation::Overhang {
                        maximum_angle: literal(16),
                        direction: literal(17),
                    },
                    ManufacturingAnnotation::Symmetry {
                        kind: SymmetryKind::Plane {
                            origin: literal(18),
                            normal: literal(19),
                        },
                    },
                    ManufacturingAnnotation::DrawDirection {
                        direction: literal(20),
                    },
                    ManufacturingAnnotation::Frozen,
                ],
                span: Span::DUMMY,
            }),
            Declaration::PhysicalRegion(PhysicalRegionDecl {
                name: ident("fluid"),
                dimension: EntityDimension::Volume,
                entities: literal(21),
                material: None,
                span: Span::DUMMY,
            }),
            Declaration::Boundary(BoundaryDecl {
                name: ident("wall"),
                dimension: EntityDimension::Surface,
                entities: literal(22),
                kind: BoundaryKind::Neumann,
                settings: vec![setting(23)],
                span: Span::DUMMY,
            }),
            Declaration::Interface(InterfaceDecl {
                name: ident("coupling"),
                dimension: EntityDimension::Surface,
                primary: literal(24),
                secondary: literal(25),
                kind: InterfaceKind::FluidStructure,
                settings: vec![setting(26)],
                span: Span::DUMMY,
            }),
            Declaration::CoordinateSystem(CoordinateSystemDecl {
                name: ident("local"),
                kind: CoordinateSystemKind::Cartesian {
                    origin: literal(27),
                    axes: vec![literal(28), literal(29), literal(30)],
                },
                span: Span::DUMMY,
            }),
            Declaration::MaterialRegion(MaterialRegionDecl {
                name: ident("steel_region"),
                region: literal(31),
                material: ident("steel"),
                properties: vec![setting(32)],
                span: Span::DUMMY,
            }),
            Declaration::GeometryField(GeometryFieldDecl {
                name: ident("density"),
                kind: GeometryFieldKind::Topology {
                    region: literal(33),
                    initial_density: literal(34),
                    filter_radius: Some(literal(35)),
                },
                span: Span::DUMMY,
            }),
        ];
        let mut visitor = CountingVisitor { expressions: 0 };
        visitor.visit_source_file(&SourceFile::new(declarations, Span::DUMMY));
        assert_eq!(visitor.expressions, 35);
    }
}
