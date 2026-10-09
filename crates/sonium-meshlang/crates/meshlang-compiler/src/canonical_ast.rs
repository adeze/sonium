//! Materialize the backend-facing geometry view from canonical IR.
//!
//! The parser AST remains useful for diagnostics, exports, and metadata
//! spelling, but geometry emitted by the native backends is reconstructed from
//! the validated IR. This keeps lowering decisions (entity order, CSG inputs,
//! dimensions, and inlined calls) out of the two backend traversals.

use crate::error::CompileError;
use crate::{HessianMetricProvider, HessianMetricRequest};
use meshlang_ast::span::Span;
use meshlang_ast::*;
use std::collections::HashMap;

pub(crate) fn materialize_geometry(
    source: &SourceFile,
    ir: &CanonicalGeometryIr,
) -> Result<SourceFile, CompileError> {
    ir.validate().map_err(|errors| {
        CompileError::Internal(
            errors
                .into_iter()
                .map(|error| error.to_string())
                .collect::<Vec<_>>()
                .join("; "),
        )
    })?;

    let mut result = source.clone();
    let mut declarations = Vec::with_capacity(source.declarations.len());
    let nodes_by_id = ir
        .entities
        .iter()
        .map(|node| (node.id, node))
        .collect::<HashMap<_, _>>();
    let entity_names = ir
        .entities
        .iter()
        .map(|node| (node.id, backend_local_name(&node.name)))
        .collect::<HashMap<_, _>>();
    let nodes_by_name = ir
        .entities
        .iter()
        .filter(|node| !is_internal_name(&node.name))
        .map(|node| (node.name.clone(), node))
        .collect::<HashMap<_, _>>();

    for declaration in &source.declarations {
        match declaration {
            Declaration::Geometry(geometry)
                if geometry.params.is_empty() && geometry.ports.is_empty() =>
            {
                let mut materialized = geometry.clone();
                let mut control_index = 0;
                materialized.body.statements = materialize_block(
                    &geometry.body,
                    &geometry.name.name,
                    &nodes_by_name,
                    &entity_names,
                    &nodes_by_id,
                    ir,
                    &HashMap::new(),
                    &[],
                    &[],
                    &mut control_index,
                )?;
                declarations.push(Declaration::Geometry(materialized));
            }
            Declaration::Mesh(mesh)
                if mesh.statements.iter().any(|statement| {
                    matches!(
                        statement,
                        MeshStatement::Refine(_)
                            | MeshStatement::BoundaryLayer(_)
                            | MeshStatement::Transfinite(_)
                            | MeshStatement::Embed(_)
                            | MeshStatement::Periodic(_)
                            | MeshStatement::Compound(_)
                            | MeshStatement::Metric(_)
                    )
                }) =>
            {
                // Mesh refine/metric/boundary/transfinite/embed/periodic/compound controls are lowered into the canonical IR and
                // emitted once by each backend's canonical-control pass. Keep
                // the source AST copy free of the original expressions so
                // aliases, loop substitutions, and template names cannot
                // diverge between the two backends.
                let mut materialized = mesh.clone();
                let mut retained = Vec::with_capacity(materialized.statements.len());
                for statement in &materialized.statements {
                    match statement {
                        // Hessian metrics are evaluated by the in-process provider
                        // before source emission. Keep a canonicalized AST copy so
                        // the provider sees the same entity/value plan as both
                        // native backends; isotropic and anisotropic metrics are
                        // emitted directly from canonical IR below.
                        MeshStatement::Metric(metric)
                            if matches!(metric.metric, MeshMetricKind::Hessian { .. }) =>
                        {
                            let canonical = ir
                                .mesh
                                .metrics
                                .iter()
                                .find(|candidate| {
                                    candidate
                                        .provenance
                                        .iter()
                                        .any(|reference| reference.span == metric.span)
                                })
                                .ok_or_else(|| {
                                    CompileError::Internal(
                                        "Hessian mesh metric missing canonical provenance".into(),
                                    )
                                })?;
                            retained.push(MeshStatement::Metric(materialize_mesh_metric_control(
                                canonical,
                                ir,
                                &entity_names,
                            )?));
                        }
                        MeshStatement::Refine(_)
                        | MeshStatement::BoundaryLayer(_)
                        | MeshStatement::Transfinite(_)
                        | MeshStatement::Embed(_)
                        | MeshStatement::Periodic(_)
                        | MeshStatement::Compound(_)
                        | MeshStatement::Metric(_) => {}
                        other => retained.push(other.clone()),
                    }
                }
                materialized.statements = retained;
                declarations.push(Declaration::Mesh(materialized));
            }
            _ => declarations.push(declaration.clone()),
        }
    }
    result.declarations = declarations;
    Ok(result)
}

/// Materialize a geometry block while retaining source scalar bindings and
/// runtime control flow. Canonical nodes replace only entity-producing lets.
#[allow(clippy::too_many_arguments)]
fn materialize_block(
    block: &Block,
    geometry_name: &str,
    nodes_by_name: &HashMap<String, &GeometryNode>,
    entity_names: &HashMap<EntityId, String>,
    nodes_by_id: &HashMap<EntityId, &GeometryNode>,
    ir: &CanonicalGeometryIr,
    parent_values: &HashMap<String, Expr>,
    loop_suffix: &[i64],
    branch_path: &[(usize, usize)],
    control_index: &mut usize,
) -> Result<Vec<Statement>, CompileError> {
    // Keep compile-time scalar bindings in the same materialization scope as
    // loop substitutions. Lowering already resolves these values when it
    // selects canonical entities; carrying them here prevents the source AST
    // from re-introducing discarded branches in either backend.
    let mut values = parent_values.clone();
    let mut output = Vec::new();
    for statement in &block.statements {
        match statement {
            Statement::Let(binding) => {
                let name = materialized_local_name(&binding.name.name, loop_suffix, branch_path);
                let qualified_name = format!("{geometry_name}.{name}");
                let source_value = substitute_loop_expr(&binding.value, &values);
                // A ported template owns multiple canonical entities below
                // its instance binding (for example `model.frame.body` and
                // `model.frame.inlet`). There is deliberately no synthetic
                // backend constructor for `Frame(...)`: materialize every
                // public descendant into collision-free local bindings so both
                // native code generators consume the same canonical plan.
                let instance_prefix = format!("{qualified_name}.");
                let instance_nodes = ir
                    .entities
                    .iter()
                    .filter(|node| {
                        !is_internal_name(&node.name) && node.name.starts_with(&instance_prefix)
                    })
                    .collect::<Vec<_>>();
                if !instance_nodes.is_empty() {
                    for node in instance_nodes {
                        output.push(Statement::Let(LetStatement {
                            name: Ident {
                                name: entity_names[&node.id].clone(),
                                span: binding.name.span,
                            },
                            type_ann: None,
                            value: node_expression(node, entity_names, nodes_by_id, ir)?,
                            span: binding.span,
                        }));
                    }
                    if nodes_by_name.get(&qualified_name).is_none() {
                        continue;
                    }
                }
                let value = if let Some(node) = nodes_by_name.get(&qualified_name) {
                    // A function/control-flow result can depend on canonical
                    // helper entities that have no source-level binding (for
                    // example `result = choose(flag)` where the selected
                    // branch owns the primitive).  Emit those dependencies
                    // before the public result so both native backends see a
                    // complete, ordered plan.
                    emit_missing_entity_dependencies(
                        node,
                        &mut output,
                        entity_names,
                        nodes_by_id,
                        ir,
                    )?;
                    let canonical = node_expression(node, entity_names, nodes_by_id, ir)?;
                    // Older IR schemas carry only numeric SymbolIds for calls,
                    // not the original spelling of built-in functions such as
                    // Cos/Sin. Keep the source expression when round-tripping
                    // would otherwise produce an opaque `__symbol_*` name.
                    if contains_placeholder_symbol(&canonical) {
                        source_value.clone()
                    } else {
                        canonical
                    }
                } else {
                    source_value.clone()
                };
                output.push(Statement::Let(LetStatement {
                    name: Ident {
                        name: name.clone(),
                        span: binding.name.span,
                    },
                    type_ann: binding.type_ann.clone(),
                    value,
                    span: binding.span,
                }));
                // Entity-producing lets are represented by canonical nodes;
                // scalar lets remain available for constant folding of later
                // conditions in this block. Keep both spellings for loop
                // materialization, where the emitted binding receives a
                // deterministic suffix but source references retain the
                // original local name.
                if nodes_by_name.get(&qualified_name).is_none() {
                    values.insert(binding.name.name.clone(), source_value.clone());
                    if name != binding.name.name {
                        values.insert(name, source_value);
                    }
                }
            }
            Statement::For(for_statement) => {
                let Some(ranges) = syntactic_for_ranges(&for_statement.clauses, &values) else {
                    let loop_control = ir
                        .loops
                        .iter()
                        .find(|control| {
                            control
                                .provenance
                                .iter()
                                .any(|reference| reference.span == for_statement.span)
                        })
                        .ok_or_else(|| {
                            CompileError::Internal(
                                "dynamic geometry loop is missing canonical loop control".into(),
                            )
                        })?;
                    if loop_control.clauses.len() != for_statement.clauses.len() {
                        return Err(CompileError::Internal(
                            "canonical dynamic loop clause count disagrees with source".into(),
                        ));
                    }
                    let mut materialized = for_statement.clone();
                    for clause in &mut materialized.clauses {
                        *clause.range.start = substitute_loop_expr(&clause.range.start, &values);
                        *clause.range.end = substitute_loop_expr(&clause.range.end, &values);
                    }
                    let mut loop_values = values.clone();
                    for clause in &materialized.clauses {
                        loop_values
                            .insert(clause.var.name.clone(), Expr::Ident(clause.var.clone()));
                    }
                    materialized.body.statements = materialize_block(
                        &for_statement.body,
                        geometry_name,
                        nodes_by_name,
                        entity_names,
                        nodes_by_id,
                        ir,
                        &loop_values,
                        loop_suffix,
                        branch_path,
                        control_index,
                    )?;
                    output.push(Statement::For(materialized));
                    continue;
                };
                let mut expanded = materialize_for_product(
                    for_statement,
                    &ranges,
                    0,
                    &mut Vec::new(),
                    geometry_name,
                    nodes_by_name,
                    entity_names,
                    nodes_by_id,
                    ir,
                    &values,
                    loop_suffix,
                    branch_path,
                    control_index,
                )?;
                chain_loop_volumes(&mut expanded, geometry_name, nodes_by_name);
                output.extend(expanded);
            }
            Statement::If(if_statement) => {
                if let Some(condition) = syntactic_const_bool(&if_statement.condition, &values) {
                    let selected = if condition {
                        Some(&if_statement.then_block)
                    } else {
                        select_else_block(if_statement.else_block.as_ref(), &values)
                    };
                    if let Some(selected) = selected {
                        output.extend(materialize_block(
                            selected,
                            geometry_name,
                            nodes_by_name,
                            entity_names,
                            nodes_by_id,
                            ir,
                            &values,
                            loop_suffix,
                            branch_path,
                            control_index,
                        )?);
                    } else if !condition && if_statement.else_block.is_some() {
                        // A constant-false first branch can lead into a
                        // dynamic `else if`; materialize the live chain
                        // without retaining the statically discarded branch.
                        let Some(ElseBranch::ElseIf(dynamic)) = if_statement.else_block.as_ref()
                        else {
                            unreachable!("missing dynamic else-if branch");
                        };
                        output.push(Statement::If(materialize_dynamic_if(
                            dynamic,
                            geometry_name,
                            nodes_by_name,
                            entity_names,
                            nodes_by_id,
                            ir,
                            &values,
                            loop_suffix,
                            branch_path,
                            *control_index,
                            0,
                            control_index,
                        )?));
                        *control_index += 1;
                    }
                } else {
                    let control = *control_index;
                    *control_index += 1;
                    output.push(Statement::If(materialize_dynamic_if(
                        if_statement,
                        geometry_name,
                        nodes_by_name,
                        entity_names,
                        nodes_by_id,
                        ir,
                        &values,
                        loop_suffix,
                        branch_path,
                        control,
                        0,
                        control_index,
                    )?));
                }
            }
            Statement::Expr(expression) => {
                output.push(Statement::Expr(substitute_loop_expr(expression, &values)))
            }
            other => output.push(other.clone()),
        }
    }
    Ok(output)
}

fn emit_missing_entity_dependencies(
    node: &GeometryNode,
    output: &mut Vec<Statement>,
    entity_names: &HashMap<EntityId, String>,
    nodes_by_id: &HashMap<EntityId, &GeometryNode>,
    ir: &CanonicalGeometryIr,
) -> Result<(), CompileError> {
    for input in &node.inputs {
        let Some(input_node) = nodes_by_id.get(input).copied() else {
            return Err(CompileError::Internal(format!(
                "canonical node '{}' references missing input {:?}",
                node.name, input
            )));
        };
        // Internal canonical nodes are deliberately inlined by
        // `entity_expression`; materializing them as extra bindings would
        // duplicate primitives. Public dependencies, on the other hand,
        // need an explicit helper when no source binding introduced them.
        if is_internal_name(&input_node.name) {
            continue;
        }
        emit_missing_entity_dependencies(input_node, output, entity_names, nodes_by_id, ir)?;
        let local_name = backend_local_name(&input_node.name);
        if output.iter().any(|statement| {
            matches!(statement, Statement::Let(binding) if binding.name.name == local_name)
        }) {
            continue;
        }
        output.push(Statement::Let(LetStatement {
            name: Ident {
                name: local_name,
                span: node_source_span(input_node),
            },
            type_ann: None,
            value: node_expression(input_node, entity_names, nodes_by_id, ir)?,
            span: node_source_span(input_node),
        }));
    }
    Ok(())
}

fn materialized_local_name(
    local: &str,
    loop_suffix: &[i64],
    branch_path: &[(usize, usize)],
) -> String {
    let local = if loop_suffix.is_empty() {
        local.to_owned()
    } else {
        format!(
            "{}_{}",
            local,
            loop_suffix
                .iter()
                .map(i64::to_string)
                .collect::<Vec<_>>()
                .join("_")
        )
    };
    if branch_path.is_empty() {
        return local;
    }
    let prefix = branch_path
        .iter()
        .map(|(control, branch)| format!("__if_{control}_{branch}"))
        .collect::<Vec<_>>()
        .join(".");
    format!("{prefix}.{local}")
}

#[allow(clippy::too_many_arguments)]
fn materialize_dynamic_if(
    statement: &IfStatement,
    geometry_name: &str,
    nodes_by_name: &HashMap<String, &GeometryNode>,
    entity_names: &HashMap<EntityId, String>,
    nodes_by_id: &HashMap<EntityId, &GeometryNode>,
    ir: &CanonicalGeometryIr,
    parent_values: &HashMap<String, Expr>,
    loop_suffix: &[i64],
    parent_path: &[(usize, usize)],
    control: usize,
    branch: usize,
    control_index: &mut usize,
) -> Result<IfStatement, CompileError> {
    let mut then_path = parent_path.to_vec();
    then_path.push((control, branch));
    let mut then_block = statement.then_block.clone();
    then_block.statements = materialize_block(
        &statement.then_block,
        geometry_name,
        nodes_by_name,
        entity_names,
        nodes_by_id,
        ir,
        parent_values,
        loop_suffix,
        &then_path,
        control_index,
    )?;

    let else_block = match statement.else_block.as_ref() {
        Some(ElseBranch::Else(block)) => {
            let mut else_path = parent_path.to_vec();
            else_path.push((control, branch + 1));
            let mut materialized = block.clone();
            materialized.statements = materialize_block(
                block,
                geometry_name,
                nodes_by_name,
                entity_names,
                nodes_by_id,
                ir,
                parent_values,
                loop_suffix,
                &else_path,
                control_index,
            )?;
            Some(ElseBranch::Else(materialized))
        }
        Some(ElseBranch::ElseIf(next)) => {
            Some(ElseBranch::ElseIf(Box::new(materialize_dynamic_if(
                next,
                geometry_name,
                nodes_by_name,
                entity_names,
                nodes_by_id,
                ir,
                parent_values,
                loop_suffix,
                parent_path,
                control,
                branch + 1,
                control_index,
            )?)))
        }
        None => None,
    };

    let mut result = statement.clone();
    result.condition = substitute_loop_expr(&statement.condition, parent_values);
    result.then_block = then_block;
    result.else_block = else_block;
    Ok(result)
}

#[allow(clippy::too_many_arguments)]
fn materialize_for_product(
    statement: &ForStatement,
    ranges: &[(i64, i64)],
    depth: usize,
    indices: &mut Vec<i64>,
    geometry_name: &str,
    nodes_by_name: &HashMap<String, &GeometryNode>,
    entity_names: &HashMap<EntityId, String>,
    nodes_by_id: &HashMap<EntityId, &GeometryNode>,
    ir: &CanonicalGeometryIr,
    parent_values: &HashMap<String, Expr>,
    parent_suffix: &[i64],
    branch_path: &[(usize, usize)],
    control_index: &mut usize,
) -> Result<Vec<Statement>, CompileError> {
    if depth == ranges.len() {
        let mut values = parent_values.clone();
        for (clause, value) in statement.clauses.iter().zip(indices.iter().copied()) {
            values.insert(
                clause.var.name.clone(),
                Expr::literal(Literal::Int(value), clause.var.span),
            );
        }
        let mut suffix = parent_suffix.to_vec();
        suffix.extend(indices.iter().copied());
        return materialize_block(
            &statement.body,
            geometry_name,
            nodes_by_name,
            entity_names,
            nodes_by_id,
            ir,
            &values,
            &suffix,
            branch_path,
            control_index,
        );
    }
    let mut output = Vec::new();
    for value in ranges[depth].0..ranges[depth].1 {
        indices.push(value);
        output.extend(materialize_for_product(
            statement,
            ranges,
            depth + 1,
            indices,
            geometry_name,
            nodes_by_name,
            entity_names,
            nodes_by_id,
            ir,
            parent_values,
            parent_suffix,
            branch_path,
            control_index,
        )?);
        indices.pop();
    }
    Ok(output)
}

fn chain_loop_volumes(
    statements: &mut [Statement],
    geometry_name: &str,
    nodes_by_name: &HashMap<String, &GeometryNode>,
) {
    let volume_names = statements
        .iter()
        .filter_map(|statement| match statement {
            Statement::Let(binding) => {
                let qualified = format!("{geometry_name}.{}", binding.name.name);
                nodes_by_name
                    .get(&qualified)
                    .filter(|node| node.id.dimension == EntityDimension::Volume)
                    .map(|_| binding.name.name.clone())
            }
            _ => None,
        })
        .collect::<Vec<_>>();
    if volume_names.len() < 2 {
        return;
    }
    let mut previous = None;
    let mut previous_span = None;
    for statement in statements {
        let Statement::Let(binding) = statement else {
            continue;
        };
        if !volume_names.iter().any(|name| name == &binding.name.name) {
            continue;
        }
        if let Some(previous_name) = previous {
            let right = binding.value.clone();
            binding.value = Expr::Binary(BinaryExpr {
                left: Box::new(Expr::Ident(Ident {
                    name: previous_name,
                    span: previous_span.unwrap_or(binding.span),
                })),
                op: BinaryOp::CsgUnion,
                right: Box::new(right),
                span: binding.span,
            });
        }
        previous = Some(binding.name.name.clone());
        previous_span = Some(binding.span);
    }
}

fn syntactic_for_ranges(
    clauses: &[ForClause],
    values: &HashMap<String, Expr>,
) -> Option<Vec<(i64, i64)>> {
    clauses
        .iter()
        .map(|clause| {
            let start = syntactic_const_int(&clause.range.start, values)?;
            let mut end = syntactic_const_int(&clause.range.end, values)?;
            if clause.range.inclusive {
                end = end.checked_add(1)?;
            }
            Some((start, end))
        })
        .collect()
}

fn select_else_block<'a>(
    branch: Option<&'a ElseBranch>,
    values: &HashMap<String, Expr>,
) -> Option<&'a Block> {
    match branch {
        Some(ElseBranch::Else(block)) => Some(block),
        Some(ElseBranch::ElseIf(statement)) => {
            let condition = syntactic_const_bool(&statement.condition, values)?;
            if condition {
                Some(&statement.then_block)
            } else {
                select_else_block(statement.else_block.as_ref(), values)
            }
        }
        None => None,
    }
}

const MAX_CONST_EVAL_DEPTH: usize = 64;

fn syntactic_const_int(expr: &Expr, values: &HashMap<String, Expr>) -> Option<i64> {
    syntactic_const_int_inner(expr, values, 0)
}

fn syntactic_const_int_inner(
    expr: &Expr,
    values: &HashMap<String, Expr>,
    depth: usize,
) -> Option<i64> {
    if depth > MAX_CONST_EVAL_DEPTH {
        return None;
    }
    match expr {
        Expr::Literal(LiteralExpr {
            value: Literal::Int(value),
            ..
        }) => Some(*value),
        Expr::Literal(LiteralExpr {
            value: Literal::Float(value),
            ..
        }) => Some(*value as i64),
        Expr::Ident(identifier) => values
            .get(&identifier.name)
            .and_then(|value| syntactic_const_int_inner(value, values, depth + 1)),
        Expr::Unary(unary) if unary.op == UnaryOp::Neg => {
            syntactic_const_int_inner(&unary.operand, values, depth + 1)?.checked_neg()
        }
        Expr::Binary(binary) => {
            let left = syntactic_const_int_inner(&binary.left, values, depth + 1)?;
            let right = syntactic_const_int_inner(&binary.right, values, depth + 1)?;
            match binary.op {
                BinaryOp::Add => left.checked_add(right),
                BinaryOp::Sub => left.checked_sub(right),
                BinaryOp::Mul => left.checked_mul(right),
                BinaryOp::Div if right != 0 => left.checked_div(right),
                _ => None,
            }
        }
        Expr::Paren(paren) => syntactic_const_int_inner(&paren.value, values, depth + 1),
        _ => None,
    }
}

fn syntactic_const_bool(expr: &Expr, values: &HashMap<String, Expr>) -> Option<bool> {
    syntactic_const_bool_inner(expr, values, 0)
}

fn syntactic_const_bool_inner(
    expr: &Expr,
    values: &HashMap<String, Expr>,
    depth: usize,
) -> Option<bool> {
    if depth > MAX_CONST_EVAL_DEPTH {
        return None;
    }
    match expr {
        Expr::Literal(LiteralExpr {
            value: Literal::Bool(value),
            ..
        }) => Some(*value),
        Expr::Ident(identifier) => values
            .get(&identifier.name)
            .and_then(|value| syntactic_const_bool_inner(value, values, depth + 1)),
        Expr::Unary(unary) if unary.op == UnaryOp::Not => {
            syntactic_const_bool_inner(&unary.operand, values, depth + 1).map(|value| !value)
        }
        Expr::Binary(binary) => {
            if let (Some(left), Some(right)) = (
                syntactic_const_int_inner(&binary.left, values, depth + 1),
                syntactic_const_int_inner(&binary.right, values, depth + 1),
            ) {
                return match binary.op {
                    BinaryOp::Eq => Some(left == right),
                    BinaryOp::Ne => Some(left != right),
                    BinaryOp::Lt => Some(left < right),
                    BinaryOp::Gt => Some(left > right),
                    BinaryOp::Le => Some(left <= right),
                    BinaryOp::Ge => Some(left >= right),
                    _ => None,
                };
            }
            match binary.op {
                BinaryOp::And => Some(
                    syntactic_const_bool_inner(&binary.left, values, depth + 1)?
                        && syntactic_const_bool_inner(&binary.right, values, depth + 1)?,
                ),
                BinaryOp::Or => Some(
                    syntactic_const_bool_inner(&binary.left, values, depth + 1)?
                        || syntactic_const_bool_inner(&binary.right, values, depth + 1)?,
                ),
                _ => None,
            }
        }
        Expr::Paren(paren) => syntactic_const_bool_inner(&paren.value, values, depth + 1),
        _ => None,
    }
}

fn contains_placeholder_symbol(expr: &Expr) -> bool {
    match expr {
        Expr::Ident(identifier) => identifier.name.starts_with("__symbol_"),
        Expr::Binary(binary) => {
            contains_placeholder_symbol(&binary.left) || contains_placeholder_symbol(&binary.right)
        }
        Expr::Unary(unary) => contains_placeholder_symbol(&unary.operand),
        Expr::Pipe(pipe) => {
            contains_placeholder_symbol(&pipe.value)
                || pipe
                    .args
                    .iter()
                    .any(|argument| contains_placeholder_symbol(&argument.value))
        }
        Expr::FieldAccess(field) => contains_placeholder_symbol(&field.object),
        Expr::Index(index) => {
            contains_placeholder_symbol(&index.object) || contains_placeholder_symbol(&index.index)
        }
        Expr::Constructor(constructor) => constructor
            .args
            .iter()
            .any(|argument| contains_placeholder_symbol(&argument.value)),
        Expr::Call(call) => {
            call.function.name.starts_with("__symbol_")
                || call
                    .args
                    .iter()
                    .any(|argument| contains_placeholder_symbol(&argument.value))
        }
        Expr::Paren(paren) => contains_placeholder_symbol(&paren.value),
        Expr::Array(array) => array.elements.iter().any(contains_placeholder_symbol),
        Expr::StructInit(value) => value
            .fields
            .iter()
            .any(|field| contains_placeholder_symbol(&field.value)),
        Expr::If(_) | Expr::Literal(_) => false,
    }
}

fn substitute_loop_expr(expr: &Expr, values: &HashMap<String, Expr>) -> Expr {
    match expr {
        Expr::Ident(identifier) => values
            .get(&identifier.name)
            .cloned()
            .unwrap_or_else(|| expr.clone()),
        Expr::Binary(binary) => Expr::Binary(BinaryExpr {
            left: Box::new(substitute_loop_expr(&binary.left, values)),
            op: binary.op,
            right: Box::new(substitute_loop_expr(&binary.right, values)),
            span: binary.span,
        }),
        Expr::Unary(unary) => Expr::Unary(UnaryExpr {
            op: unary.op,
            operand: Box::new(substitute_loop_expr(&unary.operand, values)),
            span: unary.span,
        }),
        Expr::Pipe(pipe) => Expr::Pipe(PipeExpr {
            value: Box::new(substitute_loop_expr(&pipe.value, values)),
            function: pipe.function.clone(),
            args: pipe
                .args
                .iter()
                .map(|argument| Argument {
                    name: argument.name.clone(),
                    value: substitute_loop_expr(&argument.value, values),
                    span: argument.span,
                })
                .collect(),
            span: pipe.span,
        }),
        Expr::FieldAccess(field) => Expr::FieldAccess(FieldAccessExpr {
            object: Box::new(substitute_loop_expr(&field.object, values)),
            field: field.field.clone(),
            span: field.span,
        }),
        Expr::Index(index) => Expr::Index(IndexExpr {
            object: Box::new(substitute_loop_expr(&index.object, values)),
            index: Box::new(substitute_loop_expr(&index.index, values)),
            span: index.span,
        }),
        Expr::Constructor(constructor) => Expr::Constructor(ConstructorExpr {
            type_name: constructor.type_name.clone(),
            args: constructor
                .args
                .iter()
                .map(|argument| Argument {
                    name: argument.name.clone(),
                    value: substitute_loop_expr(&argument.value, values),
                    span: argument.span,
                })
                .collect(),
            span: constructor.span,
        }),
        Expr::Call(call) => Expr::Call(CallExpr {
            function: call.function.clone(),
            args: call
                .args
                .iter()
                .map(|argument| Argument {
                    name: argument.name.clone(),
                    value: substitute_loop_expr(&argument.value, values),
                    span: argument.span,
                })
                .collect(),
            span: call.span,
        }),
        Expr::Paren(paren) => Expr::Paren(ParenExpr::new(
            substitute_loop_expr(&paren.value, values),
            paren.span,
        )),
        Expr::Array(array) => Expr::Array(ArrayExpr {
            elements: array
                .elements
                .iter()
                .map(|element| substitute_loop_expr(element, values))
                .collect(),
            span: array.span,
        }),
        Expr::StructInit(value) => Expr::StructInit(StructInitExpr {
            type_name: value.type_name.clone(),
            fields: value
                .fields
                .iter()
                .map(|field| StructInitField {
                    name: field.name.clone(),
                    value: substitute_loop_expr(&field.value, values),
                    span: field.span,
                })
                .collect(),
            span: value.span,
        }),
        Expr::If(statement) => Expr::If(Box::new(substitute_loop_if_statement(statement, values))),
        Expr::Literal(_) => expr.clone(),
    }
}

fn substitute_loop_if_statement(
    statement: &IfStatement,
    values: &HashMap<String, Expr>,
) -> IfStatement {
    IfStatement {
        condition: substitute_loop_expr(&statement.condition, values),
        then_block: substitute_loop_block(&statement.then_block, values),
        else_block: statement.else_block.as_ref().map(|branch| match branch {
            ElseBranch::Else(block) => ElseBranch::Else(substitute_loop_block(block, values)),
            ElseBranch::ElseIf(statement) => {
                ElseBranch::ElseIf(Box::new(substitute_loop_if_statement(statement, values)))
            }
        }),
        span: statement.span,
    }
}

fn substitute_loop_block(block: &Block, values: &HashMap<String, Expr>) -> Block {
    Block {
        statements: block
            .statements
            .iter()
            .map(|statement| substitute_loop_statement(statement, values))
            .collect(),
        span: block.span,
    }
}

fn substitute_loop_statement(statement: &Statement, values: &HashMap<String, Expr>) -> Statement {
    match statement {
        Statement::Let(binding) => Statement::Let(LetStatement {
            name: binding.name.clone(),
            type_ann: binding.type_ann.clone(),
            value: substitute_loop_expr(&binding.value, values),
            span: binding.span,
        }),
        Statement::Expr(expression) => Statement::Expr(substitute_loop_expr(expression, values)),
        Statement::For(for_statement) => Statement::For(ForStatement {
            clauses: for_statement
                .clauses
                .iter()
                .map(|clause| ForClause {
                    var: clause.var.clone(),
                    range: RangeExpr {
                        start: Box::new(substitute_loop_expr(&clause.range.start, values)),
                        end: Box::new(substitute_loop_expr(&clause.range.end, values)),
                        inclusive: clause.range.inclusive,
                        span: clause.range.span,
                    },
                    span: clause.span,
                })
                .collect(),
            body: substitute_loop_block(&for_statement.body, values),
            span: for_statement.span,
        }),
        Statement::If(if_statement) => {
            Statement::If(substitute_loop_if_statement(if_statement, values))
        }
        Statement::KeyValue(key_value) => Statement::KeyValue(KeyValue {
            key: key_value.key.clone(),
            value: substitute_loop_expr(&key_value.value, values),
            unit: key_value.unit.clone(),
            span: key_value.span,
        }),
        Statement::Group(group) => Statement::Group(GroupDecl {
            name: group.name.clone(),
            bc_type: group.bc_type.clone(),
            value: substitute_loop_expr(&group.value, values),
            span: group.span,
        }),
    }
}

/// Check backend capabilities before AST reconstruction reaches a text
/// emitter. The canonical IR deliberately models features that are useful to
/// optimization and backend-neutral consumers. Projection operations are
/// admitted here because the text backends preserve them as explicit
/// metadata-only aliases; they do not pretend to create native Gmsh fields.
#[cfg(test)]
pub(crate) fn validate_backend_capabilities(ir: &CanonicalGeometryIr) -> Result<(), CompileError> {
    validate_backend_capabilities_with_hessian_provider(ir, false)
}

pub(crate) fn validate_backend_capabilities_with_hessian_provider(
    ir: &CanonicalGeometryIr,
    has_hessian_provider: bool,
) -> Result<(), CompileError> {
    for node in &ir.entities {
        if let GeometryOperation::Compound = node.operation {
            return Err(CompileError::Unsupported(
                "canonical compound geometry is not supported by the native backends".into(),
            ));
        }
    }

    if !has_hessian_provider
        && ir
            .mesh
            .metrics
            .iter()
            .any(|metric| matches!(&metric.tensor, IrMetricTensor::Hessian { .. }))
    {
        return Err(CompileError::Unsupported(
            "Hessian metric evaluation requires an in-process field provider".into(),
        ));
    }

    Ok(())
}

/// Resolve Hessian mesh metrics once, before either backend sees the AST.
/// This is deliberately separate from text emission so a provider cannot
/// produce backend-specific geometry semantics.
pub(crate) fn materialize_hessian_metrics(
    source: &mut SourceFile,
    provider: Option<&dyn HessianMetricProvider>,
) -> Result<(), CompileError> {
    for declaration in &mut source.declarations {
        let Declaration::Mesh(mesh) = declaration else {
            continue;
        };
        for statement in &mut mesh.statements {
            let MeshStatement::Metric(metric) = statement else {
                continue;
            };
            let MeshMetricKind::Hessian {
                field,
                minimum_size,
                maximum_size,
            } = &metric.metric
            else {
                continue;
            };
            let provider = provider.ok_or_else(|| {
                CompileError::Unsupported(
                    "Hessian metric evaluation requires an in-process field provider".into(),
                )
            })?;
            let components = provider
                .resolve(HessianMetricRequest {
                    target: &metric.target,
                    field,
                    minimum_size,
                    maximum_size,
                    coordinate_system: metric.coordinate_system.as_ref(),
                })
                .map_err(|error| CompileError::TypeError {
                    message: format!("Hessian metric provider failed: {error}"),
                })?;
            metric.metric = MeshMetricKind::Anisotropic {
                components: components.into_iter().collect(),
            };
        }
    }
    Ok(())
}

fn node_source_span(node: &GeometryNode) -> Span {
    node.provenance
        .iter()
        .map(|source| source.span)
        .find(|span| *span != Span::DUMMY)
        .unwrap_or(Span::DUMMY)
}

fn node_expression(
    node: &GeometryNode,
    entity_names: &HashMap<EntityId, String>,
    nodes_by_id: &HashMap<EntityId, &GeometryNode>,
    ir: &CanonicalGeometryIr,
) -> Result<Expr, CompileError> {
    let span = node_source_span(node);
    let entity = |id: EntityId| entity_expression(id, entity_names, nodes_by_id, ir, span);
    let value = |value: &IrValue| ir_value_expression(value, entity_names, nodes_by_id, ir, span);
    match &node.operation {
        GeometryOperation::Primitive { kind } => {
            let mut args = node
                .arguments
                .iter()
                .enumerate()
                .map(|(index, argument)| {
                    let name = primitive_argument_name(*kind, argument.name, index, span);
                    Ok(Argument {
                        name,
                        value: value(&argument.value)?,
                        span,
                    })
                })
                .collect::<Result<Vec<_>, CompileError>>()?;
            append_entity_arguments(*kind, &node.inputs, &entity, &mut args, span)?;
            Ok(Expr::Constructor(ConstructorExpr {
                type_name: Ident {
                    name: primitive_name(*kind).to_owned(),
                    span,
                },
                args,
                span,
            }))
        }
        GeometryOperation::Boolean { kind } => {
            let mut inputs = node.inputs.iter().copied();
            let Some(first) = inputs.next() else {
                return Err(CompileError::Internal(format!(
                    "canonical boolean '{}' has no inputs",
                    node.name
                )));
            };
            inputs.try_fold(entity(first)?, |left, input| {
                Ok(Expr::Binary(BinaryExpr {
                    left: Box::new(left),
                    op: boolean_operator(*kind),
                    right: Box::new(entity(input)?),
                    span,
                }))
            })
        }
        GeometryOperation::Boundary { selector, .. } => {
            let input = node.inputs.first().copied().ok_or_else(|| {
                CompileError::Internal(format!("canonical boundary '{}' has no input", node.name))
            })?;
            let field = match node.id.dimension {
                EntityDimension::Surface => "faces",
                EntityDimension::Curve => "edges",
                EntityDimension::Point => "vertices",
                EntityDimension::Volume => "volumes",
            };
            let base = Expr::FieldAccess(FieldAccessExpr {
                object: Box::new(entity(input)?),
                field: Ident {
                    name: field.into(),
                    span,
                },
                span,
            });
            Ok(selector.as_ref().map_or(base.clone(), |selector| {
                Expr::FieldAccess(FieldAccessExpr {
                    object: Box::new(base),
                    field: Ident {
                        name: selector.clone(),
                        span,
                    },
                    span,
                })
            }))
        }
        GeometryOperation::Transform { kind } => pipe_expression(
            transform_name(*kind),
            &node.inputs,
            &node.arguments,
            &entity,
            &value,
            span,
        ),
        GeometryOperation::Sweep {
            kind: SweepOperation::Pipe,
        } => {
            let input = node.inputs.first().copied().ok_or_else(|| {
                CompileError::Internal(format!("canonical sweep '{}' has no input", node.name))
            })?;
            let path = node.inputs.get(1).copied().ok_or_else(|| {
                CompileError::Internal(format!("canonical sweep '{}' has no path", node.name))
            })?;
            let mut args = vec![Argument {
                name: Some(Ident {
                    name: "path".into(),
                    span,
                }),
                value: entity(path)?,
                span,
            }];
            args.extend(
                node.arguments
                    .iter()
                    .enumerate()
                    .map(|(index, argument)| {
                        Ok(Argument {
                            name: backend_argument_name(
                                "sweep",
                                argument.name,
                                index,
                                node.arguments.len(),
                                span,
                            ),
                            value: value(&argument.value)?,
                            span,
                        })
                    })
                    .collect::<Result<Vec<_>, CompileError>>()?,
            );
            Ok(Expr::Pipe(PipeExpr {
                value: Box::new(entity(input)?),
                function: Ident {
                    name: "sweep".into(),
                    span,
                },
                args,
                span,
            }))
        }
        GeometryOperation::Sweep { kind } => pipe_expression(
            sweep_name(*kind),
            &node.inputs,
            &node.arguments,
            &entity,
            &value,
            span,
        ),
        GeometryOperation::Array => Ok(Expr::Array(ArrayExpr {
            elements: node
                .inputs
                .iter()
                .copied()
                .map(entity)
                .collect::<Result<Vec<_>, _>>()?,
            span,
        })),
        GeometryOperation::Compound => Err(CompileError::Unsupported(
            "canonical compound geometry is not supported by the native backends".into(),
        )),
        GeometryOperation::Project { method } => pipe_expression(
            projection_name(*method),
            &node.inputs,
            &node.arguments,
            &entity,
            &value,
            span,
        ),
        GeometryOperation::Shell
        | GeometryOperation::Fillet
        | GeometryOperation::Chamfer
        | GeometryOperation::Twist
        | GeometryOperation::Copy
        | GeometryOperation::Coherence => pipe_expression(
            operation_name(&node.operation),
            &node.inputs,
            &node.arguments,
            &entity,
            &value,
            span,
        ),
        GeometryOperation::Fragment => {
            let input = node.inputs.first().copied().ok_or_else(|| {
                CompileError::Internal(format!("canonical fragment '{}' has no input", node.name))
            })?;
            let tools = node.inputs[1..]
                .iter()
                .copied()
                .map(entity)
                .collect::<Result<Vec<_>, _>>()?;
            if tools.is_empty() {
                return Err(CompileError::Internal(format!(
                    "canonical fragment '{}' has no tool",
                    node.name
                )));
            }
            let tool_value = if tools.len() == 1 {
                tools.into_iter().next().expect("checked non-empty")
            } else {
                Expr::Array(ArrayExpr {
                    elements: tools,
                    span,
                })
            };
            Ok(Expr::Pipe(PipeExpr {
                value: Box::new(entity(input)?),
                function: Ident {
                    name: "fragment".into(),
                    span,
                },
                args: vec![Argument {
                    name: None,
                    value: tool_value,
                    span,
                }],
                span,
            }))
        }
    }
}

fn entity_expression(
    id: EntityId,
    entity_names: &HashMap<EntityId, String>,
    nodes_by_id: &HashMap<EntityId, &GeometryNode>,
    ir: &CanonicalGeometryIr,
    span: Span,
) -> Result<Expr, CompileError> {
    let node = nodes_by_id.get(&id).copied().ok_or_else(|| {
        CompileError::Internal(format!(
            "canonical entity {} has no geometry node",
            id.index
        ))
    })?;
    if !is_internal_name(&node.name) {
        let name = entity_names.get(&id).ok_or_else(|| {
            CompileError::Internal(format!(
                "canonical entity '{}' has no public name",
                node.name
            ))
        })?;
        return Ok(Expr::Ident(Ident {
            name: name.clone(),
            span,
        }));
    }
    node_expression(node, entity_names, nodes_by_id, ir)
}

fn pipe_expression(
    name: &str,
    inputs: &[EntityId],
    arguments: &[IrArgument],
    entity: &impl Fn(EntityId) -> Result<Expr, CompileError>,
    value: &impl Fn(&IrValue) -> Result<Expr, CompileError>,
    span: Span,
) -> Result<Expr, CompileError> {
    let input = inputs.first().copied().ok_or_else(|| {
        CompileError::Internal(format!("canonical operation '{name}' has no input"))
    })?;
    Ok(Expr::Pipe(PipeExpr {
        value: Box::new(entity(input)?),
        function: Ident {
            name: name.to_owned(),
            span,
        },
        args: arguments
            .iter()
            .enumerate()
            .map(|(index, argument)| {
                Ok(Argument {
                    name: backend_argument_name(name, argument.name, index, arguments.len(), span),
                    value: value(&argument.value)?,
                    span,
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()?,
        span,
    }))
}

/// Reconstruct the argument spelling required by the existing backend
/// contracts from normalized IR argument names. The canonical IR does not
/// retain whether a source argument was positional.
fn backend_argument_name(
    operation: &str,
    argument: IrArgumentName,
    index: usize,
    _argument_count: usize,
    span: Span,
) -> Option<Ident> {
    let name = match operation {
        "translate" | "extrude" | "mirror" | "scale" => return None,
        "rotate" | "revolve" => match argument {
            IrArgumentName::Angle => "angle",
            IrArgumentName::Axis => "axis",
            _ => argument_name(argument),
        },
        "fillet" if matches!(argument, IrArgumentName::Radius) => "r",
        "chamfer" if matches!(argument, IrArgumentName::Distance) => "d",
        "twist" if matches!(argument, IrArgumentName::Direction) => ["dx", "dy", "dz"]
            .get(index)
            .copied()
            .unwrap_or("direction"),
        "array" if index == 0 => "count",
        "array" => "offset",
        _ => argument_name(argument),
    };
    Some(Ident {
        name: name.to_owned(),
        span,
    })
}

fn ir_value_expression(
    value: &IrValue,
    entity_names: &HashMap<EntityId, String>,
    nodes_by_id: &HashMap<EntityId, &GeometryNode>,
    ir: &CanonicalGeometryIr,
    span: Span,
) -> Result<Expr, CompileError> {
    Ok(match value {
        IrValue::Integer(value) => Expr::literal(Literal::Int(*value), span),
        IrValue::Scalar { value, .. } => Expr::literal(Literal::Float(*value), span),
        IrValue::Boolean(value) => Expr::literal(Literal::Bool(*value), span),
        IrValue::Text(value) => Expr::literal(Literal::String(value.clone()), span),
        IrValue::Entity(entity) => entity_expression(*entity, entity_names, nodes_by_id, ir, span)?,
        IrValue::Parameter(parameter) => Expr::Ident(Ident {
            name: symbol_name(*parameter, ir),
            span,
        }),
        IrValue::Vector(values) => Expr::Array(ArrayExpr {
            elements: values
                .iter()
                .map(|value| ir_value_expression(value, entity_names, nodes_by_id, ir, span))
                .collect::<Result<Vec<_>, _>>()?,
            span,
        }),
        IrValue::Unary { op, operand } => Expr::Unary(UnaryExpr {
            op: match op {
                IrUnaryOperator::Negate => UnaryOp::Neg,
                IrUnaryOperator::LogicalNot => UnaryOp::Not,
            },
            operand: Box::new(ir_value_expression(
                operand,
                entity_names,
                nodes_by_id,
                ir,
                span,
            )?),
            span,
        }),
        IrValue::Binary { op, left, right } => Expr::Binary(BinaryExpr {
            left: Box::new(ir_value_expression(
                left,
                entity_names,
                nodes_by_id,
                ir,
                span,
            )?),
            op: binary_operator(*op),
            right: Box::new(ir_value_expression(
                right,
                entity_names,
                nodes_by_id,
                ir,
                span,
            )?),
            span,
        }),
        IrValue::Call {
            function,
            arguments,
        } => Expr::Call(CallExpr {
            function: Ident {
                name: symbol_name(*function, ir),
                span,
            },
            args: arguments
                .iter()
                .map(|value| {
                    Ok(Argument {
                        name: None,
                        value: ir_value_expression(value, entity_names, nodes_by_id, ir, span)?,
                        span,
                    })
                })
                .collect::<Result<Vec<_>, CompileError>>()?,
            span,
        }),
        IrValue::Index { collection, index } => Expr::Index(IndexExpr {
            object: Box::new(ir_value_expression(
                collection,
                entity_names,
                nodes_by_id,
                ir,
                span,
            )?),
            index: Box::new(ir_value_expression(
                index,
                entity_names,
                nodes_by_id,
                ir,
                span,
            )?),
            span,
        }),
        IrValue::Field { object, field } => Expr::FieldAccess(FieldAccessExpr {
            object: Box::new(ir_value_expression(
                object,
                entity_names,
                nodes_by_id,
                ir,
                span,
            )?),
            field: Ident {
                name: symbol_name(*field, ir),
                span,
            },
            span,
        }),
    })
}

/// Reconstruct a transfinite mesh control from canonical IR without
/// consulting the source mesh block. Entity values are rendered as proper
/// field-access expressions so both native backends resolve the same
/// dimension-qualified canonical name.
pub(crate) fn materialize_transfinite_control(
    control: &IrTransfiniteControl,
    ir: &CanonicalGeometryIr,
    entity_names: &HashMap<EntityId, String>,
) -> Result<TransfiniteStatement, CompileError> {
    let span = control
        .provenance
        .first()
        .map(|reference| reference.span)
        .unwrap_or(Span::DUMMY);
    let nodes_by_id = ir
        .entities
        .iter()
        .map(|node| (node.id, node))
        .collect::<HashMap<_, _>>();
    let materialize_entity = |entity| {
        mesh_control_value_expression(
            &IrValue::Entity(entity),
            ir,
            entity_names,
            &nodes_by_id,
            span,
        )
    };
    let materialize_settings = |settings: &[IrProperty]| {
        settings
            .iter()
            .map(|property| {
                Ok(KeyValue {
                    key: Ident {
                        name: property.name.clone(),
                        span,
                    },
                    value: mesh_control_value_expression(
                        &property.value,
                        ir,
                        entity_names,
                        &nodes_by_id,
                        span,
                    )?,
                    unit: None,
                    span,
                })
            })
            .collect::<Result<Vec<_>, CompileError>>()
    };

    let kind = match &control.kind {
        IrTransfiniteKind::Curve { entity, settings } => TransfiniteKind::Curve {
            entity: materialize_entity(*entity)?,
            settings: materialize_settings(settings)?,
        },
        IrTransfiniteKind::Surface { entity, settings } => TransfiniteKind::Surface {
            entity: materialize_entity(*entity)?,
            settings: materialize_settings(settings)?,
        },
        IrTransfiniteKind::Volume { entity, settings } => TransfiniteKind::Volume {
            entity: materialize_entity(*entity)?,
            settings: materialize_settings(settings)?,
        },
    };
    Ok(TransfiniteStatement { kind, span })
}

/// Reconstruct a boundary-layer control from canonical IR. The source
/// statement is intentionally not reused: lowering has already normalized
/// aliases, units, inferred layer counts, and the total thickness.
pub(crate) fn materialize_boundary_layer_control(
    control: &IrBoundaryLayer,
    ir: &CanonicalGeometryIr,
    entity_names: &HashMap<EntityId, String>,
) -> Result<BoundaryLayerStatement, CompileError> {
    let span = control
        .provenance
        .first()
        .map(|reference| reference.span)
        .unwrap_or(Span::DUMMY);
    let nodes_by_id = ir
        .entities
        .iter()
        .map(|node| (node.id, node))
        .collect::<HashMap<_, _>>();

    if control.target.is_empty() {
        return Err(CompileError::Internal(
            "canonical boundary-layer control has no target entities".into(),
        ));
    }
    let target_value = if control.target.len() == 1 {
        IrValue::Entity(control.target[0])
    } else {
        IrValue::Vector(
            control
                .target
                .iter()
                .copied()
                .map(IrValue::Entity)
                .collect(),
        )
    };
    let target =
        mesh_control_value_expression(&target_value, ir, entity_names, &nodes_by_id, span)?;
    let mut body = vec![
        canonical_boundary_layer_property("first_height", control.first_layer_height, span),
        canonical_boundary_layer_property("thickness", control.total_thickness, span),
    ];
    if let Some(layers) = control.number_of_layers {
        body.push(KeyValue {
            key: Ident {
                name: "layers".into(),
                span,
            },
            value: Expr::literal(Literal::Int(i64::from(layers)), span),
            unit: None,
            span,
        });
    }
    body.push(canonical_boundary_layer_property(
        "growth_rate",
        control.growth_ratio,
        span,
    ));

    Ok(BoundaryLayerStatement { target, body, span })
}

/// Reconstruct a refine control from canonical IR. The source statement is
/// deliberately not reused: aliases, topology selectors, units, and
/// substituted control-flow values have already been resolved by lowering.
pub(crate) fn materialize_refine_control(
    control: &IrRefineControl,
    ir: &CanonicalGeometryIr,
    entity_names: &HashMap<EntityId, String>,
) -> Result<RefineStatement, CompileError> {
    let span = control
        .provenance
        .first()
        .map(|reference| reference.span)
        .unwrap_or(Span::DUMMY);
    let nodes_by_id = ir
        .entities
        .iter()
        .map(|node| (node.id, node))
        .collect::<HashMap<_, _>>();
    let entity = |id: EntityId| {
        mesh_control_value_expression(&IrValue::Entity(id), ir, entity_names, &nodes_by_id, span)
    };
    let properties = |properties: &[IrProperty]| {
        materialize_mesh_control_properties(properties, ir, entity_names, &nodes_by_id, span)
    };
    let region = |id: EntityId| {
        // `refine in` needs the canonical Box/translate constructor rather
        // than a local binding because both text backends derive field bounds
        // from that expression without evaluating geometry.
        let node = nodes_by_id.get(&id).copied().ok_or_else(|| {
            CompileError::Internal(format!(
                "canonical refine region references unknown entity {id:?}"
            ))
        })?;
        node_expression(node, entity_names, &nodes_by_id, ir)
    };

    let kind = match &control.kind {
        IrRefineKind::Near {
            entity: target,
            args,
        } => RefineKind::Near {
            entity: entity(*target)?,
            args: properties(args)?,
        },
        IrRefineKind::On { entity: target } => RefineKind::On {
            entity: entity(*target)?,
        },
        IrRefineKind::In { region: target } => RefineKind::In {
            region: region(*target)?,
        },
        IrRefineKind::Field { args } => RefineKind::Field {
            args: properties(args)?,
        },
    };
    Ok(RefineStatement {
        kind,
        body: properties(&control.body)?,
        span,
    })
}

/// Reconstruct a mesh metric from canonical IR. Hessian metrics are retained
/// in the backend AST only so the configured in-process provider can resolve
/// them before source emission; native isotropic/anisotropic metrics are
/// emitted directly from this canonical statement by each backend.
pub(crate) fn materialize_mesh_metric_control(
    metric: &IrMeshMetric,
    ir: &CanonicalGeometryIr,
    entity_names: &HashMap<EntityId, String>,
) -> Result<MeshMetricStatement, CompileError> {
    let span = metric
        .provenance
        .first()
        .map(|reference| reference.span)
        .unwrap_or(Span::DUMMY);
    let nodes_by_id = ir
        .entities
        .iter()
        .map(|node| (node.id, node))
        .collect::<HashMap<_, _>>();
    if metric.target.is_empty() {
        return Err(CompileError::Internal(
            "canonical mesh metric has no target entities".into(),
        ));
    }
    let target_value = if metric.target.len() == 1 {
        IrValue::Entity(metric.target[0])
    } else {
        IrValue::Vector(metric.target.iter().copied().map(IrValue::Entity).collect())
    };
    let target =
        mesh_control_value_expression(&target_value, ir, entity_names, &nodes_by_id, span)?;
    let metric_kind = match &metric.tensor {
        IrMetricTensor::Isotropic(size) => MeshMetricKind::Isotropic {
            size: Expr::literal(Literal::Float(*size), span),
        },
        IrMetricTensor::Anisotropic(components) => MeshMetricKind::Anisotropic {
            components: components
                .iter()
                .map(|component| Expr::literal(Literal::Float(*component), span))
                .collect(),
        },
        IrMetricTensor::Hessian {
            field,
            minimum_size,
            maximum_size,
        } => MeshMetricKind::Hessian {
            field: mesh_control_value_expression(field, ir, entity_names, &nodes_by_id, span)?,
            minimum_size: Expr::literal(Literal::Float(*minimum_size), span),
            maximum_size: Expr::literal(Literal::Float(*maximum_size), span),
        },
    };
    let coordinate_system = metric.coordinate_system.map(|symbol| Ident {
        name: symbol_name(symbol, ir),
        span,
    });
    Ok(MeshMetricStatement {
        target,
        metric: metric_kind,
        coordinate_system,
        span,
    })
}

fn materialize_mesh_control_properties(
    properties: &[IrProperty],
    ir: &CanonicalGeometryIr,
    entity_names: &HashMap<EntityId, String>,
    nodes_by_id: &HashMap<EntityId, &GeometryNode>,
    span: Span,
) -> Result<Vec<KeyValue>, CompileError> {
    properties
        .iter()
        .map(|property| {
            Ok(KeyValue {
                key: Ident {
                    name: property.name.clone(),
                    span,
                },
                value: mesh_control_value_expression(
                    &property.value,
                    ir,
                    entity_names,
                    nodes_by_id,
                    span,
                )?,
                unit: None,
                span,
            })
        })
        .collect()
}

fn canonical_boundary_layer_property(name: &str, value: f64, span: Span) -> KeyValue {
    KeyValue {
        key: Ident {
            name: name.into(),
            span,
        },
        value: Expr::literal(Literal::Float(value), span),
        unit: None,
        span,
    }
}

fn mesh_control_value_expression(
    value: &IrValue,
    ir: &CanonicalGeometryIr,
    entity_names: &HashMap<EntityId, String>,
    nodes_by_id: &HashMap<EntityId, &GeometryNode>,
    span: Span,
) -> Result<Expr, CompileError> {
    Ok(match value {
        IrValue::Integer(value) => Expr::literal(Literal::Int(*value), span),
        IrValue::Scalar { value, .. } => Expr::literal(Literal::Float(*value), span),
        IrValue::Boolean(value) => Expr::literal(Literal::Bool(*value), span),
        IrValue::Text(value) => Expr::literal(Literal::String(value.clone()), span),
        IrValue::Entity(entity) => {
            mesh_control_entity_expression(*entity, entity_names, nodes_by_id, ir, span)?
        }
        IrValue::Parameter(parameter) => Expr::Ident(Ident {
            name: symbol_name(*parameter, ir),
            span,
        }),
        IrValue::Vector(values) => Expr::Array(ArrayExpr {
            elements: values
                .iter()
                .map(|value| {
                    mesh_control_value_expression(value, ir, entity_names, nodes_by_id, span)
                })
                .collect::<Result<Vec<_>, _>>()?,
            span,
        }),
        IrValue::Unary { op, operand } => Expr::Unary(UnaryExpr {
            op: match op {
                IrUnaryOperator::Negate => UnaryOp::Neg,
                IrUnaryOperator::LogicalNot => UnaryOp::Not,
            },
            operand: Box::new(mesh_control_value_expression(
                operand,
                ir,
                entity_names,
                nodes_by_id,
                span,
            )?),
            span,
        }),
        IrValue::Binary { op, left, right } => Expr::Binary(BinaryExpr {
            left: Box::new(mesh_control_value_expression(
                left,
                ir,
                entity_names,
                nodes_by_id,
                span,
            )?),
            op: binary_operator(*op),
            right: Box::new(mesh_control_value_expression(
                right,
                ir,
                entity_names,
                nodes_by_id,
                span,
            )?),
            span,
        }),
        IrValue::Call {
            function,
            arguments,
        } => Expr::Call(CallExpr {
            function: Ident {
                name: symbol_name(*function, ir),
                span,
            },
            args: arguments
                .iter()
                .map(|argument| {
                    Ok(Argument {
                        name: None,
                        value: mesh_control_value_expression(
                            argument,
                            ir,
                            entity_names,
                            nodes_by_id,
                            span,
                        )?,
                        span,
                    })
                })
                .collect::<Result<Vec<_>, CompileError>>()?,
            span,
        }),
        IrValue::Index { collection, index } => Expr::Index(IndexExpr {
            object: Box::new(mesh_control_value_expression(
                collection,
                ir,
                entity_names,
                nodes_by_id,
                span,
            )?),
            index: Box::new(mesh_control_value_expression(
                index,
                ir,
                entity_names,
                nodes_by_id,
                span,
            )?),
            span,
        }),
        IrValue::Field { object, field } => Expr::FieldAccess(FieldAccessExpr {
            object: Box::new(mesh_control_value_expression(
                object,
                ir,
                entity_names,
                nodes_by_id,
                span,
            )?),
            field: Ident {
                name: symbol_name(*field, ir),
                span,
            },
            span,
        }),
    })
}

fn mesh_control_entity_expression(
    entity: EntityId,
    entity_names: &HashMap<EntityId, String>,
    nodes_by_id: &HashMap<EntityId, &GeometryNode>,
    ir: &CanonicalGeometryIr,
    span: Span,
) -> Result<Expr, CompileError> {
    let node = nodes_by_id.get(&entity).copied().ok_or_else(|| {
        CompileError::Internal(format!(
            "canonical transfinite control references unknown entity {:?}",
            entity
        ))
    })?;
    if is_internal_name(&node.name) {
        return node_expression(node, entity_names, nodes_by_id, ir);
    }
    let name = entity_names.get(&entity).ok_or_else(|| {
        CompileError::Internal(format!(
            "canonical transfinite control entity '{}' has no name",
            node.name
        ))
    })?;
    let mut parts = name.split('.');
    let first = parts.next().unwrap_or(name);
    let mut expression = Expr::Ident(Ident {
        name: first.to_owned(),
        span,
    });
    for part in parts {
        expression = Expr::FieldAccess(FieldAccessExpr {
            object: Box::new(expression),
            field: Ident {
                name: part.to_owned(),
                span,
            },
            span,
        });
    }
    Ok(expression)
}

fn append_entity_arguments(
    kind: PrimitiveKind,
    inputs: &[EntityId],
    entity: &impl Fn(EntityId) -> Result<Expr, CompileError>,
    arguments: &mut Vec<Argument>,
    span: Span,
) -> Result<(), CompileError> {
    let names: &[&str] = match kind {
        PrimitiveKind::Line => &["from", "to"],
        PrimitiveKind::Arc => &["start", "center", "end"],
        PrimitiveKind::Polygon => &["points"],
        PrimitiveKind::Loft => &["sections"],
        _ => &[],
    };
    if names == ["points"] || names == ["sections"] {
        if !inputs.is_empty() {
            let mut elements = Vec::new();
            for input in inputs.iter().copied() {
                match entity(input)? {
                    Expr::Array(array) => elements.extend(array.elements),
                    expression => elements.push(expression),
                }
            }
            arguments.push(Argument {
                name: Some(Ident {
                    name: names[0].to_owned(),
                    span,
                }),
                value: Expr::Array(ArrayExpr { elements, span }),
                span,
            });
        }
    } else if matches!(kind, PrimitiveKind::Spline | PrimitiveKind::BSpline) {
        for input in inputs.iter().copied() {
            arguments.push(Argument {
                name: None,
                value: entity(input)?,
                span,
            });
        }
    } else {
        for (input, name) in inputs.iter().copied().zip(names) {
            arguments.push(Argument {
                name: Some(Ident {
                    name: (*name).to_owned(),
                    span,
                }),
                value: entity(input)?,
                span,
            });
        }
    }
    Ok(())
}

fn primitive_argument_name(
    kind: PrimitiveKind,
    name: IrArgumentName,
    _index: usize,
    span: Span,
) -> Option<Ident> {
    let named = matches!(
        kind,
        PrimitiveKind::Sphere
            | PrimitiveKind::Disk
            | PrimitiveKind::Cylinder
            | PrimitiveKind::Cone
            | PrimitiveKind::Torus
            | PrimitiveKind::Polygon
    ) || (matches!(kind, PrimitiveKind::Loft) && matches!(name, IrArgumentName::Ruled));
    named.then(|| Ident {
        name: argument_name(name).to_owned(),
        span,
    })
}

fn symbol_name(symbol: SymbolId, ir: &CanonicalGeometryIr) -> String {
    ir.parameters
        .iter()
        .find(|parameter| parameter.id == symbol)
        .map(|parameter| parameter.name.clone())
        .or_else(|| {
            ir.optimization
                .variables
                .iter()
                .find(|variable| variable.id == symbol)
                .map(|variable| variable.name.clone())
        })
        .or_else(|| {
            ir.loops.iter().find_map(|loop_control| {
                loop_control
                    .clauses
                    .iter()
                    .find(|clause| clause.variable == symbol)
                    .map(|clause| clause.name.clone())
            })
        })
        .unwrap_or_else(|| format!("__symbol_{}", symbol.0))
}

fn backend_local_name(name: &str) -> String {
    name.split_once('.')
        .map_or_else(|| name.to_owned(), |(_, local)| local.to_owned())
}

fn is_internal_name(name: &str) -> bool {
    name.split('.')
        .any(|component| component.starts_with("__ir_"))
}

fn primitive_name(kind: PrimitiveKind) -> &'static str {
    match kind {
        PrimitiveKind::Point => "Point",
        PrimitiveKind::Line => "Line",
        PrimitiveKind::Arc => "Arc",
        PrimitiveKind::Spline => "Spline",
        PrimitiveKind::BSpline => "BSpline",
        PrimitiveKind::Polygon => "Polygon",
        PrimitiveKind::Rectangle => "Rectangle",
        PrimitiveKind::Disk => "Disk",
        PrimitiveKind::Box => "Box",
        PrimitiveKind::Sphere => "Sphere",
        PrimitiveKind::Cylinder => "Cylinder",
        PrimitiveKind::Cone => "Cone",
        PrimitiveKind::Torus => "Torus",
        PrimitiveKind::Wedge => "Wedge",
        PrimitiveKind::Loft => "Loft",
    }
}

fn boolean_operator(operation: BooleanOperation) -> BinaryOp {
    match operation {
        BooleanOperation::Union => BinaryOp::CsgUnion,
        BooleanOperation::Difference => BinaryOp::CsgDifference,
        BooleanOperation::Intersection => BinaryOp::CsgIntersect,
    }
}

fn binary_operator(operation: IrBinaryOperator) -> BinaryOp {
    match operation {
        IrBinaryOperator::Add => BinaryOp::Add,
        IrBinaryOperator::Subtract => BinaryOp::Sub,
        IrBinaryOperator::Multiply => BinaryOp::Mul,
        IrBinaryOperator::Divide => BinaryOp::Div,
        IrBinaryOperator::Modulo => BinaryOp::Mod,
        IrBinaryOperator::Power => BinaryOp::Pow,
        IrBinaryOperator::Equal => BinaryOp::Eq,
        IrBinaryOperator::NotEqual => BinaryOp::Ne,
        IrBinaryOperator::Less => BinaryOp::Lt,
        IrBinaryOperator::LessEqual => BinaryOp::Le,
        IrBinaryOperator::Greater => BinaryOp::Gt,
        IrBinaryOperator::GreaterEqual => BinaryOp::Ge,
        IrBinaryOperator::LogicalAnd => BinaryOp::And,
        IrBinaryOperator::LogicalOr => BinaryOp::Or,
    }
}

fn argument_name(name: IrArgumentName) -> &'static str {
    match name {
        IrArgumentName::Origin => "origin",
        IrArgumentName::Start => "start",
        IrArgumentName::End => "end",
        IrArgumentName::Center => "center",
        IrArgumentName::Points => "points",
        IrArgumentName::Sections => "sections",
        IrArgumentName::Ruled => "ruled",
        IrArgumentName::Size => "size",
        IrArgumentName::Radius => "r",
        IrArgumentName::RadiusStart => "r1",
        IrArgumentName::RadiusEnd => "r2",
        IrArgumentName::MajorRadius => "r_major",
        IrArgumentName::MinorRadius => "r_minor",
        IrArgumentName::Height => "h",
        IrArgumentName::Direction => "direction",
        IrArgumentName::Angle => "angle",
        IrArgumentName::Axis => "axis",
        IrArgumentName::Count => "count",
        IrArgumentName::Offset => "offset",
        IrArgumentName::Scale => "factor",
        IrArgumentName::Plane => "plane",
        IrArgumentName::Matrix => "matrix",
        IrArgumentName::Layers => "layers",
        IrArgumentName::Recombine => "recombine",
        IrArgumentName::Thickness => "thickness",
        IrArgumentName::Distance => "distance",
        IrArgumentName::Tolerance => "tolerance",
    }
}

#[cfg(test)]
#[allow(clippy::items_after_test_module)]
mod tests {
    use super::*;

    fn geometry_binding<'a>(file: &'a SourceFile, name: &str) -> &'a LetStatement {
        let Declaration::Geometry(geometry) = file
            .declarations
            .iter()
            .find(|declaration| matches!(declaration, Declaration::Geometry(_)))
            .expect("geometry declaration")
        else {
            unreachable!();
        };
        geometry
            .body
            .statements
            .iter()
            .find_map(|statement| match statement {
                Statement::Let(binding) if binding.name.name == name => Some(binding),
                _ => None,
            })
            .expect("geometry binding")
    }

    #[test]
    fn materialize_replaces_direct_function_calls_with_canonical_geometry() {
        let source = r#"
            fn make_body(radius: Float) -> Solid {
                Sphere(r: radius)
            }
            geometry model {
                let body = make_body(0.5)
            }
        "#;
        let parsed = meshlang_parser::parse(source, "materialize.mlg").unwrap();
        let ir = crate::lower_ir::lower_to_canonical_ir(&parsed).unwrap();
        let materialized = materialize_geometry(&parsed, &ir).unwrap();
        let binding = geometry_binding(&materialized, "body");

        let Expr::Constructor(constructor) = &binding.value else {
            panic!(
                "canonical materialization left a source-level call: {:?}",
                binding.value
            );
        };
        assert_eq!(constructor.type_name.name, "Sphere");
        assert!(constructor
            .args
            .iter()
            .any(|argument| { argument.name.as_ref().is_some_and(|name| name.name == "r") }));
    }

    #[test]
    fn materialize_generated_geometry_keeps_canonical_source_span() {
        let source = r#"
        geometry model {
            /// body source
            let body = Box(1.0, 2.0, 3.0)
            /// moved source
            let moved = body |> translate(1.0, 2.0, 3.0)
        }
        "#;
        let parsed = meshlang_parser::parse(source, "materialize-spans.mlg").unwrap();
        let ir = crate::lower_ir::lower_to_canonical_ir(&parsed).unwrap();
        let materialized = materialize_geometry(&parsed, &ir).unwrap();
        let source_body = geometry_binding(&parsed, "body");
        let source_moved = geometry_binding(&parsed, "moved");
        let body = geometry_binding(&materialized, "body");
        let moved = geometry_binding(&materialized, "moved");
        let Expr::Constructor(body_constructor) = &body.value else {
            panic!("expected canonical body constructor");
        };
        let Expr::Pipe(moved_pipe) = &moved.value else {
            panic!("expected canonical moved pipe");
        };
        assert_ne!(body_constructor.span, Span::DUMMY);
        assert_ne!(moved_pipe.span, Span::DUMMY);
        assert_eq!(body_constructor.span, source_body.value.span());
        assert_eq!(moved_pipe.span, source_moved.value.span());
        assert_eq!(body_constructor.span.file_id, body.span.file_id);
        assert_eq!(moved_pipe.span.file_id, moved.span.file_id);
        assert!(body_constructor.args.iter().all(|argument| {
            argument
                .name
                .as_ref()
                .is_none_or(|name| name.span == body_constructor.span)
        }));
        assert!(moved_pipe.args.iter().all(|argument| {
            argument
                .name
                .as_ref()
                .is_none_or(|name| name.span == moved_pipe.span)
        }));
    }

    #[test]
    fn generated_helper_bindings_keep_dependency_source_span() {
        let source = r#"
            geometry model {
                let body = Box(1.0, 1.0, 1.0)
                let moved = body |> translate(1.0, 2.0, 3.0)
            }
        "#;
        let parsed = meshlang_parser::parse(source, "materialize-helper-span.mlg").unwrap();
        let ir = crate::lower_ir::lower_to_canonical_ir(&parsed).unwrap();
        let nodes_by_id = ir
            .entities
            .iter()
            .map(|node| (node.id, node))
            .collect::<HashMap<_, _>>();
        let entity_names = ir
            .entities
            .iter()
            .map(|node| (node.id, backend_local_name(&node.name)))
            .collect::<HashMap<_, _>>();
        let body = ir
            .entities
            .iter()
            .find(|node| node.name == "model.body")
            .expect("body node");
        let moved = ir
            .entities
            .iter()
            .find(|node| node.name == "model.moved")
            .expect("moved node");
        let mut output = Vec::new();
        emit_missing_entity_dependencies(moved, &mut output, &entity_names, &nodes_by_id, &ir)
            .unwrap();
        let [Statement::Let(helper)] = output.as_slice() else {
            panic!("expected one generated helper binding: {output:?}");
        };
        assert_eq!(helper.name.span, node_source_span(body));
        assert_eq!(helper.span, node_source_span(body));
    }

    #[test]
    fn materialize_expands_control_flow_from_canonical_entity_order() {
        let source = r#"
            geometry model {
                for i in 0..2 {
                    let body = Box(1.0, 1.0, 1.0)
                }
                if true {
                    let cap = Sphere(r: 0.25)
                }
            }
        "#;
        let parsed = meshlang_parser::parse(source, "materialize-control-flow.mlg").unwrap();
        let ir = crate::lower_ir::lower_to_canonical_ir(&parsed).unwrap();
        let materialized = materialize_geometry(&parsed, &ir).unwrap();
        let Declaration::Geometry(geometry) = &materialized.declarations[0] else {
            panic!("expected materialized geometry declaration");
        };

        assert!(geometry
            .body
            .statements
            .iter()
            .all(|statement| matches!(statement, Statement::Let(_))));
        let names = geometry
            .body
            .statements
            .iter()
            .filter_map(|statement| match statement {
                Statement::Let(binding) => Some(binding.name.name.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>();
        assert_eq!(names, vec!["body_0", "body_1", "cap"]);
        let Statement::Let(first) = &geometry.body.statements[0] else {
            panic!("expected first loop binding");
        };
        let Statement::Let(second) = &geometry.body.statements[1] else {
            panic!("expected second loop binding");
        };
        let Expr::Binary(binary) = &second.value else {
            panic!("expected loop volumes to be chained through a union");
        };
        let Expr::Ident(previous) = binary.left.as_ref() else {
            panic!("expected chained union left operand to be an identifier");
        };
        assert_eq!(previous.name, "body_0");
        assert_eq!(previous.span, first.span);
        assert_eq!(binary.span, second.span);
    }

    #[test]
    fn materialize_preserves_dynamic_geometry_control_flow() {
        let source = r#"
            geometry model {
                if runtime_flag {
                    let body = Box(1.0, 1.0, 1.0)
                }
            }
        "#;
        let parsed = meshlang_parser::parse(source, "materialize-dynamic.mlg").unwrap();
        let ir = CanonicalGeometryIr::new(parsed.metadata.clone());
        let materialized = materialize_geometry(&parsed, &ir).unwrap();
        let Declaration::Geometry(geometry) = &materialized.declarations[0] else {
            panic!("expected geometry declaration");
        };
        assert!(matches!(geometry.body.statements[0], Statement::If(_)));
    }

    #[test]
    fn materialize_preserves_dynamic_else_if_after_a_constant_false_branch() {
        let source = r#"
            geometry model {
                if false {
                    let discarded = Box(1.0, 1.0, 1.0)
                } else if runtime_flag {
                    let body = Box(1.0, 1.0, 1.0)
                }
            }
        "#;
        let parsed = meshlang_parser::parse(source, "materialize-dynamic-else-if.mlg").unwrap();
        let ir = CanonicalGeometryIr::new(parsed.metadata.clone());
        let materialized = materialize_geometry(&parsed, &ir).unwrap();
        let Declaration::Geometry(geometry) = &materialized.declarations[0] else {
            panic!("expected geometry declaration");
        };
        let Statement::If(statement) = &geometry.body.statements[0] else {
            panic!("expected dynamic else-if statement");
        };
        assert!(matches!(statement.condition, Expr::Ident(ref id) if id.name == "runtime_flag"));
    }

    #[test]
    fn loop_substitution_rewrites_if_expression_branches() {
        let source = r#"
            geometry model {
                let value = if i == 1 { 10 } else { 20 }
            }
        "#;
        let parsed = meshlang_parser::parse(source, "materialize-if-expression.mlg").unwrap();
        let Declaration::Geometry(geometry) = &parsed.declarations[0] else {
            panic!("expected geometry declaration");
        };
        let Statement::Let(binding) = &geometry.body.statements[0] else {
            panic!("expected scalar binding");
        };
        let mut values = HashMap::new();
        values.insert("i".to_owned(), Expr::literal(Literal::Int(1), Span::DUMMY));
        let Expr::If(statement) = substitute_loop_expr(&binding.value, &values) else {
            panic!("expected if expression");
        };
        assert!(matches!(
            &statement.condition,
            Expr::Binary(BinaryExpr {
                left,
                op: BinaryOp::Eq,
                ..
            }) if matches!(left.as_ref(), Expr::Literal(LiteralExpr { value: Literal::Int(1), .. }))
        ));
        assert!(matches!(
            statement.then_block.statements.as_slice(),
            [Statement::Expr(Expr::Literal(LiteralExpr {
                value: Literal::Int(10),
                ..
            }))]
        ));
        assert!(matches!(
            statement.else_block.as_ref(),
            Some(ElseBranch::Else(Block {
                ref statements,
                ..
            })) if matches!(
                statements.as_slice(),
                [Statement::Expr(Expr::Literal(LiteralExpr {
                    value: Literal::Int(20),
                    ..
                }))]
            )
        ));
    }

    #[test]
    fn constant_condition_resolution_is_bounded_for_recursive_aliases() {
        let mut values = HashMap::new();
        values.insert(
            "a".to_owned(),
            Expr::Ident(Ident {
                name: "a".into(),
                span: Span::DUMMY,
            }),
        );
        assert_eq!(
            syntactic_const_bool(
                &Expr::Ident(Ident {
                    name: "a".into(),
                    span: Span::DUMMY,
                }),
                &values
            ),
            None
        );
    }

    #[test]
    fn backend_capability_gate_rejects_hessian_metrics_before_emission() {
        let source = r#"
            geometry model {
                let body = Box(1.0, 1.0, 1.0)
            }
            mesh for model {
                metric model.body {
                    kind: hessian
                    field: 1.0
                    minimum_size: 0.01
                    maximum_size: 0.1
                }
            }
        "#;
        let parsed = meshlang_parser::parse(source, "hessian.mlg").unwrap();
        let ir = crate::lower_ir::lower_to_canonical_ir(&parsed).unwrap();
        let error = validate_backend_capabilities(&ir).unwrap_err();
        assert!(matches!(error, CompileError::Unsupported(message) if message.contains("Hessian")));
    }

    #[test]
    fn backend_capability_gate_admits_project_but_rejects_compound_operations() {
        let source = r#"
            geometry model {
                let body = Box(1.0, 1.0, 1.0)
                let projected = body |> project_iso_surface(tolerance: 0.01)
            }
        "#;
        let parsed = meshlang_parser::parse(source, "project-capability.mlg").unwrap();
        let ir = crate::lower_ir::lower_to_canonical_ir(&parsed).unwrap();
        assert!(validate_backend_capabilities(&ir).is_ok());

        let mut compound_ir = ir;
        let body = compound_ir
            .entities
            .iter_mut()
            .find(|node| node.name == "model.body")
            .expect("body node");
        body.operation = GeometryOperation::Compound;
        let error = validate_backend_capabilities(&compound_ir).unwrap_err();
        assert!(matches!(
            error,
            CompileError::Unsupported(message) if message.contains("compound geometry")
        ));
    }

    #[test]
    fn materializer_preserves_dynamic_loop_bounds_for_backend_emission() {
        let source = r#"
            param count = 2
            geometry model {
                for i in 0..count {
                    let body = Box(1.0, 1.0, 1.0)
                }
            }
        "#;
        let parsed = meshlang_parser::parse(source, "dynamic-loop-bound.mlg").unwrap();
        let ir = crate::lower_ir::lower_to_canonical_ir(&parsed).unwrap();
        assert_eq!(ir.loops.len(), 1);
        let materialized = materialize_geometry(&parsed, &ir).unwrap();
        let geometry = materialized
            .declarations
            .iter()
            .find_map(|declaration| match declaration {
                Declaration::Geometry(geometry) if geometry.name.name == "model" => Some(geometry),
                _ => None,
            })
            .expect("model geometry");
        assert!(geometry
            .body
            .statements
            .iter()
            .any(|statement| matches!(statement, Statement::For(_))));
    }

    #[test]
    fn materializer_keeps_source_expression_for_unknown_builtin_symbol_ids() {
        let source = r#"
            geometry model {
                let body = Box(1.0, 1.0, 1.0)
            }
        "#;
        let parsed = meshlang_parser::parse(source, "builtin-symbol-fallback.mlg").unwrap();
        let mut ir = crate::lower_ir::lower_to_canonical_ir(&parsed).unwrap();
        ir.entities[0].arguments[0].value = IrValue::Parameter(SymbolId(999));
        let materialized = materialize_geometry(&parsed, &ir).unwrap();
        let binding = geometry_binding(&materialized, "body");
        let rendered = format!("{:?}", binding.value);
        assert!(!rendered.contains("__symbol_"), "{rendered}");
    }
}

fn transform_name(operation: TransformOperation) -> &'static str {
    match operation {
        TransformOperation::Translate => "translate",
        TransformOperation::Rotate => "rotate",
        TransformOperation::Scale => "scale",
        TransformOperation::Mirror => "mirror",
        TransformOperation::Affine => "affine",
    }
}

fn sweep_name(operation: SweepOperation) -> &'static str {
    match operation {
        SweepOperation::Extrude => "extrude",
        SweepOperation::Revolve => "revolve",
        SweepOperation::Pipe => "sweep",
    }
}

fn projection_name(method: ProjectionMethod) -> &'static str {
    match method {
        ProjectionMethod::IsoSurface => "project_iso_surface",
        ProjectionMethod::SignedDistance => "project_signed_distance",
        ProjectionMethod::ClosestPoint => "project_closest_point",
    }
}

fn operation_name(operation: &GeometryOperation) -> &'static str {
    match operation {
        GeometryOperation::Shell => "shell",
        GeometryOperation::Fillet => "fillet",
        GeometryOperation::Chamfer => "chamfer",
        GeometryOperation::Twist => "twist",
        GeometryOperation::Copy => "copy",
        GeometryOperation::Fragment => "fragment",
        GeometryOperation::Coherence => "coherence",
        _ => "operation",
    }
}
