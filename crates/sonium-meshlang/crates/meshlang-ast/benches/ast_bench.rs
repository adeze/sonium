//! AST size, structural-hash and transfer-application benchmarks.
//!
//! Audit follow-up (Stage 1, item 3): the AST package had no benchmark
//! target, so tree allocation, hashing, serialization and transfer-map
//! costs had no platform measurements. Three groups, kept separate because
//! they scale differently:
//!
//! - `ast_size`: parse a generated N-entity source, then report node count
//!   (exact [`Visitor`] count) and serialized bytes alongside build time.
//! - `structural_hash`: [`StructuralHash::of`] over the same trees.
//! - `transfer_chain` / `transfer_weighted`: plan-chain apply and weighted
//!   f64 transfer over N-entry plans.
//!
//! Sizes print once per N at setup (`ASTSIZE entities=.. nodes=.. bytes=..`)
//! so scaling is auditable from the bench log; criterion reports time with
//! per-element throughput.

use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use meshlang_ast::visitor::{
    Visitor, walk_declaration, walk_expr, walk_mesh_statement, walk_source_file, walk_statement,
};
use meshlang_ast::*;
use meshlang_parser::parse;

/// Exact node counter: every `visit_*` counts one and replicates the
/// default traversal (private `walk_*` helpers are inlined as `visit_expr`
/// sequences, which is all they do).
#[derive(Default)]
struct NodeCounter(usize);

impl NodeCounter {
    fn count(file: &SourceFile) -> usize {
        let mut counter = Self::default();
        counter.visit_source_file(file);
        counter.0
    }
}

impl Visitor for NodeCounter {
    fn visit_source_file(&mut self, file: &SourceFile) {
        self.0 += 1;
        walk_source_file(self, file);
    }
    fn visit_declaration(&mut self, decl: &Declaration) {
        self.0 += 1;
        walk_declaration(self, decl);
    }
    fn visit_param_decl(&mut self, decl: &ParamDecl) {
        self.0 += 1;
        self.visit_expr(&decl.value);
    }
    fn visit_const_decl(&mut self, decl: &ConstDecl) {
        self.0 += 1;
        self.visit_expr(&decl.value);
    }
    fn visit_import_decl(&mut self, _decl: &ImportDecl) {
        self.0 += 1;
    }
    fn visit_fn_decl(&mut self, decl: &FnDecl) {
        self.0 += 1;
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
        self.0 += 1;
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
        self.0 += 1;
        for statement in &block.statements {
            walk_mesh_statement(self, statement);
        }
    }
    fn visit_physics_block(&mut self, block: &PhysicsBlock) {
        self.0 += 1;
        for statement in &block.statements {
            match statement {
                PhysicsStatement::Group(group) => self.visit_group_decl(group),
                PhysicsStatement::Let(statement) => self.visit_let_statement(statement),
            }
        }
    }
    fn visit_export_block(&mut self, block: &ExportBlock) {
        self.0 += 1;
        for setting in &block.settings {
            self.visit_expr(&setting.value);
        }
    }
    fn visit_struct_decl(&mut self, _decl: &StructDecl) {
        self.0 += 1;
    }
    fn visit_design_variable_decl(&mut self, decl: &DesignVariableDecl) {
        self.0 += 1;
        self.visit_expr(&decl.initial);
        if let Some(bounds) = &decl.bounds {
            self.visit_expr(&bounds.lower);
            self.visit_expr(&bounds.upper);
        }
        if let Some(distribution) = &decl.distribution {
            visit_distribution(self, distribution);
        }
        if let DesignVariableType::Categorical { values } = &decl.variable_type {
            for value in values {
                self.visit_expr(value);
            }
        }
    }
    fn visit_uncertainty_decl(&mut self, decl: &UncertaintyDecl) {
        self.0 += 1;
        visit_distribution(self, &decl.distribution);
    }
    fn visit_objective_decl(&mut self, decl: &ObjectiveDecl) {
        self.0 += 1;
        self.visit_expr(&decl.expression);
        if let Some(weight) = &decl.weight {
            self.visit_expr(weight);
        }
    }
    fn visit_constraint_decl(&mut self, decl: &ConstraintDecl) {
        self.0 += 1;
        match &decl.kind {
            ConstraintKind::Equality {
                left,
                right,
                tolerance,
            } => {
                self.visit_expr(left);
                self.visit_expr(right);
                if let Some(tolerance) = tolerance {
                    self.visit_expr(tolerance);
                }
            }
            ConstraintKind::Inequality { left, right, .. } => {
                self.visit_expr(left);
                self.visit_expr(right);
            }
            ConstraintKind::Range { expression, bounds } => {
                self.visit_expr(expression);
                self.visit_expr(&bounds.lower);
                self.visit_expr(&bounds.upper);
            }
            ConstraintKind::VolumeFraction { region, maximum } => {
                self.visit_expr(region);
                self.visit_expr(maximum);
            }
            ConstraintKind::MinimumFeature { region, radius } => {
                self.visit_expr(region);
                self.visit_expr(radius);
            }
        }
    }
    fn visit_design_region_decl(&mut self, decl: &DesignRegionDecl) {
        self.0 += 1;
        self.visit_expr(&decl.region);
        for annotation in &decl.annotations {
            match annotation {
                ManufacturingAnnotation::MinimumThickness { value, .. }
                | ManufacturingAnnotation::MinimumRadius { value, .. } => self.visit_expr(value),
                ManufacturingAnnotation::Overhang {
                    maximum_angle,
                    direction,
                } => {
                    self.visit_expr(maximum_angle);
                    self.visit_expr(direction);
                }
                ManufacturingAnnotation::Symmetry { kind } => match kind {
                    SymmetryKind::Plane { origin, normal } => {
                        self.visit_expr(origin);
                        self.visit_expr(normal);
                    }
                    SymmetryKind::Axis { origin, direction } => {
                        self.visit_expr(origin);
                        self.visit_expr(direction);
                    }
                    SymmetryKind::Cyclic { axis, sectors } => {
                        self.visit_expr(axis);
                        self.visit_expr(sectors);
                    }
                },
                ManufacturingAnnotation::DrawDirection { direction } => self.visit_expr(direction),
                ManufacturingAnnotation::Frozen => {}
            }
        }
    }
    fn visit_physical_region_decl(&mut self, decl: &PhysicalRegionDecl) {
        self.0 += 1;
        self.visit_expr(&decl.entities);
    }
    fn visit_boundary_decl(&mut self, decl: &BoundaryDecl) {
        self.0 += 1;
        self.visit_expr(&decl.entities);
        for setting in &decl.settings {
            self.visit_expr(&setting.value);
        }
    }
    fn visit_interface_decl(&mut self, decl: &InterfaceDecl) {
        self.0 += 1;
        self.visit_expr(&decl.primary);
        self.visit_expr(&decl.secondary);
        for setting in &decl.settings {
            self.visit_expr(&setting.value);
        }
    }
    fn visit_coordinate_system_decl(&mut self, decl: &CoordinateSystemDecl) {
        self.0 += 1;
        match &decl.kind {
            CoordinateSystemKind::Cartesian { origin, axes } => {
                self.visit_expr(origin);
                for axis in axes {
                    self.visit_expr(axis);
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
                self.visit_expr(origin);
                self.visit_expr(axis);
                self.visit_expr(reference);
            }
        }
    }
    fn visit_material_region_decl(&mut self, decl: &MaterialRegionDecl) {
        self.0 += 1;
        self.visit_expr(&decl.region);
        for setting in &decl.properties {
            self.visit_expr(&setting.value);
        }
    }
    fn visit_geometry_field_decl(&mut self, decl: &GeometryFieldDecl) {
        self.0 += 1;
        match &decl.kind {
            GeometryFieldKind::Topology {
                region,
                initial_density,
                filter_radius,
            } => {
                self.visit_expr(region);
                self.visit_expr(initial_density);
                if let Some(filter_radius) = filter_radius {
                    self.visit_expr(filter_radius);
                }
            }
            GeometryFieldKind::LevelSet { field, iso_value } => {
                self.visit_expr(field);
                self.visit_expr(iso_value);
            }
            GeometryFieldKind::Projected {
                geometry,
                provenance,
            } => {
                self.visit_expr(geometry);
                if let Some(tolerance) = &provenance.tolerance {
                    self.visit_expr(tolerance);
                }
            }
        }
    }
    fn visit_statement(&mut self, stmt: &Statement) {
        self.0 += 1;
        walk_statement(self, stmt);
    }
    fn visit_let_statement(&mut self, stmt: &LetStatement) {
        self.0 += 1;
        self.visit_expr(&stmt.value);
    }
    fn visit_for_statement(&mut self, stmt: &ForStatement) {
        self.0 += 1;
        for clause in &stmt.clauses {
            self.visit_expr(&clause.range.start);
            self.visit_expr(&clause.range.end);
        }
        for statement in &stmt.body.statements {
            self.visit_statement(statement);
        }
    }
    fn visit_if_statement(&mut self, stmt: &IfStatement) {
        self.0 += 1;
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
        self.0 += 1;
        walk_expr(self, expr);
    }
    fn visit_binary_expr(&mut self, expr: &BinaryExpr) {
        self.0 += 1;
        self.visit_expr(&expr.left);
        self.visit_expr(&expr.right);
    }
    fn visit_unary_expr(&mut self, expr: &UnaryExpr) {
        self.0 += 1;
        self.visit_expr(&expr.operand);
    }
    fn visit_pipe_expr(&mut self, expr: &PipeExpr) {
        self.0 += 1;
        self.visit_expr(&expr.value);
        for argument in &expr.args {
            self.visit_expr(&argument.value);
        }
    }
    fn visit_constructor_expr(&mut self, expr: &ConstructorExpr) {
        self.0 += 1;
        for argument in &expr.args {
            self.visit_expr(&argument.value);
        }
    }
    fn visit_call_expr(&mut self, expr: &CallExpr) {
        self.0 += 1;
        for argument in &expr.args {
            self.visit_expr(&argument.value);
        }
    }
    fn visit_struct_init_expr(&mut self, expr: &StructInitExpr) {
        self.0 += 1;
        for field in &expr.fields {
            self.visit_expr(&field.value);
        }
    }
    fn visit_field_access(&mut self, expr: &FieldAccessExpr) {
        self.0 += 1;
        self.visit_expr(&expr.object);
    }
    fn visit_group_decl(&mut self, decl: &GroupDecl) {
        self.0 += 1;
        self.visit_expr(&decl.value);
    }
}

/// Inline replica of the private `walk_distribution` helper (visit every
/// sampled expression).
fn visit_distribution(counter: &mut NodeCounter, distribution: &Distribution) {
    match distribution {
        Distribution::Deterministic { value } => counter.visit_expr(value),
        Distribution::Uniform { lower, upper } => {
            counter.visit_expr(lower);
            counter.visit_expr(upper);
        }
        Distribution::Normal {
            mean,
            standard_deviation,
        }
        | Distribution::LogNormal {
            mean,
            standard_deviation,
        } => {
            counter.visit_expr(mean);
            counter.visit_expr(standard_deviation);
        }
        Distribution::Discrete { values, weights } => {
            for value in values {
                counter.visit_expr(value);
            }
            for weight in weights {
                counter.visit_expr(weight);
            }
        }
    }
}

/// Generated N-entity fragment-array source (same idiom as the compiler
/// bench working set): one base box plus N translated sphere inclusions.
fn fragment_source(entities: usize) -> String {
    let mut source = String::with_capacity(entities * 96);
    source.push_str("geometry Main {\n  let base = Box(10.0, 1.0, 1.0)\n");
    for index in 0..entities {
        writeln!(
            source,
            "  let inclusion_{index} = Sphere(r: 0.01) |> translate({}, 0.5, 0.5)",
            index as f64 * 0.02
        )
        .unwrap();
    }
    source.push_str("  let result = base |> fragment([");
    for index in 0..entities {
        if index != 0 {
            source.push_str(", ");
        }
        write!(source, "inclusion_{index}").unwrap();
    }
    source.push_str("])\n}\n");
    source
}

fn entity_id(index: u64) -> EntityId {
    EntityId::new(EntityDimension::Volume, index)
}

fn preserve_plan(entries: usize) -> IrStateTransferPlan {
    IrStateTransferPlan {
        responsibility: StateTransferResponsibility::ApplicationDefined,
        entries: (0..entries as u64)
            .map(|index| IrStateTransferEntry {
                entity: entity_id(index),
                action: IrStateTransferAction::Preserve,
            })
            .collect(),
    }
}

fn criterion_benchmark(criterion: &mut Criterion) {
    let sizes = [64usize, 256, 1024];

    let mut size_group = criterion.benchmark_group("ast_size");
    for entities in sizes {
        let source = fragment_source(entities);
        let ast = parse(&source, "ast_bench.mlg").expect("bench source must parse");
        let nodes = NodeCounter::count(&ast);
        let bytes = serde_json::to_vec(&ast).expect("AST must serialize").len();
        eprintln!("ASTSIZE entities={entities} nodes={nodes} bytes={bytes}");
        size_group.throughput(Throughput::Elements(nodes as u64));
        size_group.bench_with_input(BenchmarkId::from_parameter(entities), &entities, |b, _| {
            b.iter(|| {
                let ast =
                    parse(black_box(&source), "ast_bench.mlg").expect("bench source must parse");
                let nodes = NodeCounter::count(&ast);
                let bytes = serde_json::to_vec(&ast).expect("AST must serialize").len();
                black_box((nodes, bytes));
            });
        });
    }
    size_group.finish();

    let mut hash_group = criterion.benchmark_group("structural_hash");
    for entities in sizes {
        let source = fragment_source(entities);
        let ast = parse(&source, "ast_bench.mlg").expect("bench source must parse");
        let bytes = serde_json::to_vec(&ast).expect("AST must serialize").len();
        hash_group.throughput(Throughput::Bytes(bytes as u64));
        hash_group.bench_with_input(BenchmarkId::from_parameter(entities), &ast, |b, ast| {
            b.iter(|| {
                let hash = StructuralHash::of(black_box(ast)).expect("AST must hash");
                black_box(hash);
            });
        });
    }
    hash_group.finish();

    let mut chain_group = criterion.benchmark_group("transfer_chain");
    for entities in sizes {
        let plan = preserve_plan(entities);
        let chain = IrStateTransferPlanChain::new(vec![plan]).expect("bench plan must validate");
        let source: BTreeMap<EntityId, f64> = (0..entities as u64)
            .map(|index| (entity_id(index), index as f64))
            .collect();
        chain_group.throughput(Throughput::Elements(entities as u64));
        chain_group.bench_with_input(BenchmarkId::from_parameter(entities), &entities, |b, _| {
            b.iter(|| {
                let result = chain
                    .apply(black_box(&source), |_, entity, _, _| {
                        Ok::<_, ()>(entity.index as f64)
                    })
                    .expect("bench transfer must apply");
                black_box(result);
            });
        });
    }
    chain_group.finish();

    let mut weighted_group = criterion.benchmark_group("transfer_weighted");
    for entities in sizes {
        let plan = IrStateTransferPlan {
            responsibility: StateTransferResponsibility::ApplicationDefined,
            entries: (0..entities as u64)
                .map(|index| IrStateTransferEntry {
                    entity: entity_id(index),
                    action: IrStateTransferAction::Interpolate,
                })
                .collect(),
        };
        let source: BTreeMap<EntityId, f64> = (0..entities as u64)
            .map(|index| (entity_id(index), index as f64))
            .collect();
        // Four neighbors at 0.25 each: interpolation weights sum to one.
        let weights: BTreeMap<EntityId, Vec<(EntityId, f64)>> = (0..entities as u64)
            .map(|index| {
                let neighbors = (0..4)
                    .map(|k| (entity_id((index + k as u64) % entities as u64), 0.25))
                    .collect();
                (entity_id(index), neighbors)
            })
            .collect();
        weighted_group.throughput(Throughput::Elements(entities as u64));
        weighted_group.bench_with_input(
            BenchmarkId::from_parameter(entities),
            &entities,
            |b, _| {
                b.iter(|| {
                    let result = plan
                        .apply_weighted_f64(
                            black_box(&source),
                            black_box(&weights),
                            |entity, _, _| Ok::<_, ()>(entity.index as f64),
                        )
                        .expect("bench transfer must apply");
                    black_box(result);
                });
            },
        );
    }
    weighted_group.finish();
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
