//! Typed lowering from the resolved MeshLang AST to backend-neutral geometry IR.

use crate::error::CompileError;
use meshlang_ast::types::TypeAnnotation;
use meshlang_ast::*;
use std::collections::{BTreeSet, HashMap};
use std::sync::Arc;

const FIRST_INTERNAL_ENTITY_ID: u64 = 1_u64 << 32;

/// Lower a flattened, type-checked source file into canonical geometry IR.
pub fn lower_to_canonical_ir(file: &SourceFile) -> Result<CanonicalGeometryIr, CompileError> {
    let mut lowerer = IrLowerer::new(file);
    lowerer.lower()?;
    lowerer.ir.validate().map_err(|errors| {
        CompileError::Internal(
            errors
                .into_iter()
                .map(|error| error.to_string())
                .collect::<Vec<_>>()
                .join("; "),
        )
    })?;
    Ok(lowerer.ir)
}

/// Structural-hash cache for repeated geometry variants in long-lived tools.
#[derive(Default)]
pub struct CanonicalIrCache {
    entries: HashMap<StructuralHash, Arc<CanonicalGeometryIr>>,
    hits: u64,
    misses: u64,
}

impl CanonicalIrCache {
    pub fn lower(&mut self, file: &SourceFile) -> Result<Arc<CanonicalGeometryIr>, CompileError> {
        let key = file.structural_hash().map_err(|error| {
            CompileError::Internal(format!("failed to hash frontend AST: {error}"))
        })?;
        if let Some(ir) = self.entries.get(&key) {
            self.hits += 1;
            return Ok(Arc::clone(ir));
        }
        let ir = Arc::new(lower_to_canonical_ir(file)?);
        self.entries.insert(key, Arc::clone(&ir));
        self.misses += 1;
        Ok(ir)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn hits(&self) -> u64 {
        self.hits
    }

    pub fn misses(&self) -> u64 {
        self.misses
    }

    /// Remove all cached IR variants while retaining cumulative statistics.
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Reset hit/miss counters without discarding cached variants.
    pub fn reset_stats(&mut self) {
        self.hits = 0;
        self.misses = 0;
    }
}

struct IrLowerer<'a> {
    file: &'a SourceFile,
    ir: CanonicalGeometryIr,
    functions: HashMap<String, &'a FnDecl>,
    templates: HashMap<String, &'a GeometryBlock>,
    symbols: HashMap<String, SymbolId>,
    values: HashMap<String, IrValue>,
    entities: HashMap<String, EntityId>,
    topology_entities: HashMap<String, EntityId>,
    next_entity: [u64; 4],
    next_internal: [u64; 4],
    next_group: u64,
    current_geometry: Option<String>,
    generated_name_counter: u64,
    loop_generated_entities: BTreeSet<String>,
    /// Active compile-time loop indices, including enclosing loops.
    ///
    /// Keeping the full path here makes names produced through nested
    /// `for`/`if` lowering deterministic and prevents an inner loop from
    /// reusing the same entity name for every outer iteration.
    loop_indices: Vec<i64>,
    /// Dynamic control-flow branch path used to keep same-spelled bindings in
    /// mutually exclusive branches collision-free and deterministic.
    branch_path: Vec<(usize, usize)>,
    active_controls: Vec<usize>,
    active_loops: Vec<usize>,
}

impl<'a> IrLowerer<'a> {
    fn new(file: &'a SourceFile) -> Self {
        let mut ir = CanonicalGeometryIr::new(file.metadata.clone());
        for identity in &file.source_identities {
            if !ir.source_identities.iter().any(|known| known == identity) {
                ir.source_identities.push(identity.clone());
            }
        }
        Self {
            file,
            ir,
            functions: HashMap::new(),
            templates: HashMap::new(),
            symbols: HashMap::new(),
            values: HashMap::new(),
            entities: HashMap::new(),
            topology_entities: HashMap::new(),
            next_entity: [1; 4],
            next_internal: [FIRST_INTERNAL_ENTITY_ID; 4],
            next_group: 1,
            current_geometry: None,
            generated_name_counter: 0,
            loop_generated_entities: BTreeSet::new(),
            loop_indices: Vec::new(),
            branch_path: Vec::new(),
            active_controls: Vec::new(),
            active_loops: Vec::new(),
        }
    }

    fn lower(&mut self) -> Result<(), CompileError> {
        for declaration in &self.file.declarations {
            if let Declaration::Function(function) = declaration {
                self.functions.insert(function.name.name.clone(), function);
                self.intern_symbol(&format!("function:{}", function.name.name));
            } else if let Declaration::Geometry(geometry) = declaration {
                if !geometry.params.is_empty() || !geometry.ports.is_empty() {
                    self.templates.insert(geometry.name.name.clone(), geometry);
                }
            }
        }

        // Values precede modelling metadata so forward-independent parameter
        // references have stable symbol IDs before manifests are built.
        for declaration in &self.file.declarations {
            match declaration {
                Declaration::Param(parameter) => self.lower_parameter(
                    &parameter.name.name,
                    &parameter.value,
                    parameter.unit.as_ref(),
                    parameter,
                )?,
                Declaration::Const(constant) => self.lower_parameter(
                    &constant.name.name,
                    &constant.value,
                    constant.unit.as_ref(),
                    constant,
                )?,
                Declaration::DesignVariable(variable) => self.lower_design_variable(variable)?,
                _ => {}
            }
        }

        for declaration in &self.file.declarations {
            if let Declaration::Geometry(geometry) = declaration {
                // Parameterized geometry blocks are declarations/templates;
                // their entities are lowered only at an explicit call site.
                // Keeping the declaration body out of canonical output avoids
                // treating an uninstantiated template as a physical entity.
                if geometry.params.is_empty() && geometry.ports.is_empty() {
                    self.lower_geometry(geometry)?;
                }
            }
        }

        for declaration in &self.file.declarations {
            self.lower_metadata_declaration(declaration)?;
        }
        self.build_sensitivity_manifest();
        Ok(())
    }

    fn lower_parameter<N: AstNode>(
        &mut self,
        name: &str,
        expression: &Expr,
        unit: Option<&units::Unit>,
        node: &N,
    ) -> Result<(), CompileError> {
        let id = self.intern_symbol(&format!("parameter:{name}"));
        self.symbols.insert(name.to_owned(), id);
        let mut value = self.lower_value(expression)?;
        let dimension = unit.map_or_else(|| value_dimension(&value), |unit| unit.dimension);
        if let Some(unit) = unit {
            value = apply_unit_scale(value, unit.to_si_factor, unit.dimension)?;
        }
        self.values.insert(name.to_owned(), IrValue::Parameter(id));
        self.ir.parameters.push(IrParameter {
            id,
            name: name.to_owned(),
            value,
            dimension,
            provenance: self.provenance(node),
        });
        Ok(())
    }

    fn lower_design_variable(&mut self, variable: &DesignVariableDecl) -> Result<(), CompileError> {
        let id = self.intern_symbol(&format!("design-variable:{}", variable.name.name));
        self.symbols.insert(variable.name.name.clone(), id);
        self.values
            .insert(variable.name.name.clone(), IrValue::Parameter(id));
        let initial = self.lower_value(&variable.initial)?;
        let bounds = variable
            .bounds
            .as_ref()
            .map(|bounds| {
                Ok(IrBounds {
                    lower: self.lower_value(&bounds.lower)?,
                    upper: self.lower_value(&bounds.upper)?,
                    lower_inclusive: bounds.lower_inclusive,
                    upper_inclusive: bounds.upper_inclusive,
                })
            })
            .transpose()?;
        let distribution = variable
            .distribution
            .as_ref()
            .map(|distribution| self.lower_distribution(distribution))
            .transpose()?;
        let kind = match variable.variable_type {
            DesignVariableType::Continuous => IrDesignVariableKind::Continuous,
            DesignVariableType::Integer => IrDesignVariableKind::Integer,
            DesignVariableType::Boolean => IrDesignVariableKind::Boolean,
            DesignVariableType::Categorical { .. } => IrDesignVariableKind::Categorical,
            DesignVariableType::Shape => IrDesignVariableKind::Shape,
            DesignVariableType::Topology => IrDesignVariableKind::Topology,
            DesignVariableType::LevelSet => IrDesignVariableKind::LevelSet,
        };
        self.ir.optimization.variables.push(IrDesignVariable {
            id,
            name: variable.name.name.clone(),
            kind,
            initial,
            bounds,
            distribution,
            provenance: self.provenance(variable),
        });
        Ok(())
    }

    fn lower_geometry(&mut self, geometry: &GeometryBlock) -> Result<(), CompileError> {
        self.current_geometry = Some(geometry.name.name.clone());
        self.lower_block(&geometry.body, &mut HashMap::new())?;
        self.current_geometry = None;
        Ok(())
    }

    fn lower_block(
        &mut self,
        block: &Block,
        loop_values: &mut HashMap<String, i64>,
    ) -> Result<(), CompileError> {
        self.lower_block_with_result(block, None, loop_values)?;
        Ok(())
    }

    fn lower_block_with_result(
        &mut self,
        block: &Block,
        result_name: Option<String>,
        loop_values: &mut HashMap<String, i64>,
    ) -> Result<Option<EntityId>, CompileError> {
        let mut shadowed_values = Vec::new();
        let mut result = None;
        let last_statement = block.statements.len().checked_sub(1);

        for (index, statement) in block.statements.iter().enumerate() {
            let is_last = last_statement == Some(index);
            match statement {
                Statement::Let(binding) => {
                    if self.expression_produces_entity(&binding.value) {
                        let name = if is_last {
                            result_name
                                .clone()
                                .unwrap_or_else(|| self.loop_binding_name(&binding.name.name))
                        } else {
                            self.loop_binding_name(&binding.name.name)
                        };
                        let id = self.lower_entity_expr(&binding.value, Some(name), loop_values)?;
                        if is_last {
                            result = Some(id);
                        }
                    } else {
                        let value = self.lower_value_with_loop(&binding.value, loop_values)?;
                        let previous = self.values.insert(binding.name.name.clone(), value);
                        shadowed_values.push((binding.name.name.clone(), previous));
                    }
                }
                Statement::Expr(expression) if self.expression_produces_entity(expression) => {
                    let id = self.lower_entity_expr(
                        expression,
                        if is_last { result_name.clone() } else { None },
                        loop_values,
                    )?;
                    if is_last {
                        result = Some(id);
                    }
                }
                Statement::Expr(_) | Statement::KeyValue(_) | Statement::Group(_) => {}
                Statement::For(statement) => self.lower_for(statement, loop_values)?,
                Statement::If(statement) if is_last && result_name.is_some() => {
                    result =
                        Some(self.lower_if_result(statement, result_name.clone(), loop_values)?);
                }
                Statement::If(statement) => self.lower_if(statement, loop_values)?,
            }
        }
        for (name, previous) in shadowed_values.into_iter().rev() {
            if let Some(previous) = previous {
                self.values.insert(name, previous);
            } else {
                self.values.remove(&name);
            }
        }
        Ok(result)
    }

    fn lower_for(
        &mut self,
        statement: &ForStatement,
        loop_values: &mut HashMap<String, i64>,
    ) -> Result<(), CompileError> {
        let mut ranges = Vec::with_capacity(statement.clauses.len());
        let mut has_dynamic_bound = false;
        for clause in &statement.clauses {
            let Some(start) = self.try_const_int(&clause.range.start, loop_values)? else {
                has_dynamic_bound = true;
                continue;
            };
            let Some(mut end) = self.try_const_int(&clause.range.end, loop_values)? else {
                has_dynamic_bound = true;
                continue;
            };
            if clause.range.inclusive {
                end = end.checked_add(1).ok_or_else(|| {
                    CompileError::Unsupported("inclusive loop range overflows i64".into())
                })?;
            }
            ranges.push((clause.var.name.clone(), start, end));
        }
        if has_dynamic_bound {
            return self.lower_dynamic_for(statement, loop_values);
        }

        self.lower_for_product(statement, &ranges, 0, loop_values)
    }

    fn lower_dynamic_for(
        &mut self,
        statement: &ForStatement,
        loop_values: &mut HashMap<String, i64>,
    ) -> Result<(), CompileError> {
        if statement.clauses.is_empty() {
            return Err(CompileError::TypeError {
                message: "dynamic loop requires at least one range clause".into(),
            });
        }

        let loop_index = self.ir.loops.len();
        let parent = self.active_loops.last().copied();
        let mut clauses = Vec::with_capacity(statement.clauses.len());
        let mut shadowed_values = Vec::with_capacity(statement.clauses.len());

        for clause in &statement.clauses {
            let start = self.lower_value_with_loop(&clause.range.start, loop_values)?;
            let end = self.lower_value_with_loop(&clause.range.end, loop_values)?;
            let variable = self.intern_symbol(&format!("loop:{loop_index}:{}", clause.var.name));
            clauses.push(IrLoopClause {
                variable,
                name: clause.var.name.clone(),
                start,
                end,
                inclusive: clause.range.inclusive,
                provenance: self.provenance(clause),
            });
            shadowed_values.push((
                clause.var.name.clone(),
                self.values
                    .insert(clause.var.name.clone(), IrValue::Parameter(variable)),
            ));
        }

        let parent_control = self.active_controls.last().copied();
        self.ir.loops.push(IrLoopControl {
            parent,
            parent_control,
            clauses,
            entities: Vec::new(),
            provenance: self.provenance(statement),
        });
        self.active_loops.push(loop_index);
        let entity_start = self.ir.entities.len();
        let result = self.lower_block(&statement.body, &mut HashMap::new());
        self.active_loops.pop();

        for (name, previous) in shadowed_values.into_iter().rev() {
            if let Some(previous) = previous {
                self.values.insert(name, previous);
            } else {
                self.values.remove(&name);
            }
        }

        result?;
        self.ir.loops[loop_index].entities = self.ir.entities[entity_start..]
            .iter()
            .map(|node| node.id)
            .collect();
        Ok(())
    }

    fn lower_for_product(
        &mut self,
        statement: &ForStatement,
        ranges: &[(String, i64, i64)],
        depth: usize,
        loop_values: &mut HashMap<String, i64>,
    ) -> Result<(), CompileError> {
        if depth == ranges.len() {
            let mut shadowed_values = Vec::new();
            for body_statement in &statement.body.statements {
                match body_statement {
                    Statement::Let(binding) => {
                        if self.expression_produces_entity(&binding.value) {
                            let name = self.loop_binding_name(&binding.name.name);
                            self.loop_generated_entities.insert(name.clone());
                            self.lower_entity_expr(&binding.value, Some(name), loop_values)?;
                        } else {
                            let value = self.lower_value_with_loop(&binding.value, loop_values)?;
                            let previous = self.values.insert(binding.name.name.clone(), value);
                            shadowed_values.push((binding.name.name.clone(), previous));
                        }
                    }
                    Statement::Expr(expression) if self.expression_produces_entity(expression) => {
                        self.lower_entity_expr(expression, None, loop_values)?;
                    }
                    Statement::For(nested) => self.lower_for(nested, loop_values)?,
                    Statement::If(nested) => self.lower_if(nested, loop_values)?,
                    _ => {}
                }
            }
            for (name, previous) in shadowed_values.into_iter().rev() {
                if let Some(previous) = previous {
                    self.values.insert(name, previous);
                } else {
                    self.values.remove(&name);
                }
            }
            return Ok(());
        }
        let (name, start, end) = &ranges[depth];
        for value in *start..*end {
            let previous = loop_values.insert(name.clone(), value);
            self.loop_indices.push(value);
            let result = self.lower_for_product(statement, ranges, depth + 1, loop_values);
            self.loop_indices.pop();
            if let Some(previous) = previous {
                loop_values.insert(name.clone(), previous);
            } else {
                loop_values.remove(name);
            }
            result?;
        }
        Ok(())
    }

    fn lower_if(
        &mut self,
        statement: &IfStatement,
        loop_values: &mut HashMap<String, i64>,
    ) -> Result<(), CompileError> {
        match self.try_const_bool(&statement.condition, loop_values)? {
            Some(true) => self.lower_block(&statement.then_block, loop_values),
            Some(false) => match &statement.else_block {
                Some(ElseBranch::Else(block)) => self.lower_block(block, loop_values),
                Some(ElseBranch::ElseIf(statement)) => self.lower_if(statement, loop_values),
                None => Ok(()),
            },
            None => self.lower_dynamic_if(statement, loop_values),
        }
    }

    fn lower_dynamic_if(
        &mut self,
        statement: &IfStatement,
        loop_values: &mut HashMap<String, i64>,
    ) -> Result<(), CompileError> {
        let control_id = self.ir.controls.len();
        let parent = self.active_controls.last().copied();
        let parent_loop = self.active_loops.last().copied();
        self.ir.controls.push(IrConditionalControl {
            parent,
            parent_loop,
            branches: Vec::new(),
            provenance: self.provenance(statement),
        });
        self.active_controls.push(control_id);

        let mut branches = Vec::new();
        self.lower_dynamic_if_chain(statement, loop_values, control_id, 0, &mut branches)?;

        self.active_controls.pop();
        self.ir.controls[control_id].branches = branches;
        Ok(())
    }

    fn lower_dynamic_if_chain(
        &mut self,
        statement: &IfStatement,
        loop_values: &mut HashMap<String, i64>,
        control_id: usize,
        branch_index: usize,
        branches: &mut Vec<IrConditionalBranch>,
    ) -> Result<(), CompileError> {
        let condition = self.lower_value_with_loop(&statement.condition, loop_values)?;
        let entity_start = self.ir.entities.len();
        self.branch_path.push((control_id, branch_index));
        let result = self.lower_block(&statement.then_block, loop_values);
        self.branch_path.pop();
        result?;
        branches.push(IrConditionalBranch {
            condition: Some(condition),
            entities: self.ir.entities[entity_start..]
                .iter()
                .map(|node| node.id)
                .collect(),
            provenance: self.provenance(statement),
        });

        match &statement.else_block {
            Some(ElseBranch::Else(block)) => {
                let entity_start = self.ir.entities.len();
                self.branch_path.push((control_id, branch_index + 1));
                let result = self.lower_block(block, loop_values);
                self.branch_path.pop();
                result?;
                branches.push(IrConditionalBranch {
                    condition: None,
                    entities: self.ir.entities[entity_start..]
                        .iter()
                        .map(|node| node.id)
                        .collect(),
                    provenance: self.provenance(block),
                });
            }
            Some(ElseBranch::ElseIf(next)) => self.lower_dynamic_if_chain(
                next,
                loop_values,
                control_id,
                branch_index + 1,
                branches,
            )?,
            None => {}
        }
        Ok(())
    }

    fn lower_entity_expr(
        &mut self,
        expression: &Expr,
        target_name: Option<String>,
        loop_values: &HashMap<String, i64>,
    ) -> Result<EntityId, CompileError> {
        match expression {
            Expr::Ident(identifier) => {
                let source = self.resolve_entity_name(&identifier.name)?;
                let id = self.entities[&source];
                if let Some(name) = target_name {
                    self.emit_entity(
                        id.dimension,
                        name,
                        GeometryOperation::Copy,
                        vec![id],
                        Vec::new(),
                        expression,
                        false,
                    )
                } else {
                    Ok(id)
                }
            }
            Expr::FieldAccess(_) => {
                let id = self.lower_single_entity(expression)?;
                if let Some(name) = target_name {
                    self.emit_entity(
                        id.dimension,
                        name,
                        GeometryOperation::Copy,
                        vec![id],
                        Vec::new(),
                        expression,
                        false,
                    )
                } else {
                    Ok(id)
                }
            }
            Expr::Constructor(constructor) => {
                self.lower_constructor(constructor, target_name, loop_values)
            }
            Expr::Binary(binary) if is_entity_csg_operator(binary.op) => {
                self.lower_boolean(binary, target_name, loop_values)
            }
            Expr::Pipe(pipe) => self.lower_pipe(pipe, target_name, loop_values),
            Expr::Call(call) if self.functions.contains_key(&call.function.name) => {
                let inlined = self.inline_call(call)?;
                self.lower_entity_expr(&inlined, target_name, loop_values)
            }
            Expr::Paren(paren) => self.lower_entity_expr(&paren.value, target_name, loop_values),
            Expr::Array(array) => {
                let mut inputs = Vec::with_capacity(array.elements.len());
                for element in &array.elements {
                    inputs.push(self.lower_entity_expr(element, None, loop_values)?);
                }
                let dimension = common_dimension(&inputs)?;
                let internal = target_name.is_none();
                let name = target_name.unwrap_or_else(|| self.internal_name("array"));
                self.emit_entity(
                    dimension,
                    name,
                    GeometryOperation::Array,
                    inputs,
                    Vec::new(),
                    expression,
                    internal,
                )
            }
            Expr::If(statement) => {
                if self.const_bool(&statement.condition, loop_values)? {
                    self.lower_block_result(&statement.then_block, target_name, loop_values)
                } else {
                    match &statement.else_block {
                        Some(ElseBranch::Else(block)) => {
                            self.lower_block_result(block, target_name, loop_values)
                        }
                        Some(ElseBranch::ElseIf(statement)) => {
                            self.lower_if_result(statement, target_name, loop_values)
                        }
                        None => Err(CompileError::TypeError {
                            message: "geometry if-expression requires an else branch".into(),
                        }),
                    }
                }
            }
            _ => Err(CompileError::TypeError {
                message: "expression does not produce a canonical geometry entity".into(),
            }),
        }
    }

    fn lower_block_result(
        &mut self,
        block: &Block,
        target_name: Option<String>,
        loop_values: &HashMap<String, i64>,
    ) -> Result<EntityId, CompileError> {
        if block.statements.len() == 1 {
            if let Statement::If(statement) = &block.statements[0] {
                return self.lower_if_result(statement, target_name, loop_values);
            }
        }
        let mut scoped_loop_values = loop_values.clone();
        self.lower_block_with_result(block, target_name, &mut scoped_loop_values)?
            .ok_or_else(|| CompileError::TypeError {
                message: "geometry if-expression branch must return an entity".into(),
            })
    }

    fn lower_if_result(
        &mut self,
        statement: &IfStatement,
        target_name: Option<String>,
        loop_values: &HashMap<String, i64>,
    ) -> Result<EntityId, CompileError> {
        if self.const_bool(&statement.condition, loop_values)? {
            self.lower_block_result(&statement.then_block, target_name, loop_values)
        } else {
            match &statement.else_block {
                Some(ElseBranch::Else(block)) => {
                    self.lower_block_result(block, target_name, loop_values)
                }
                Some(ElseBranch::ElseIf(statement)) => {
                    self.lower_if_result(statement, target_name, loop_values)
                }
                None => Err(CompileError::TypeError {
                    message: "geometry if-expression requires an else branch".into(),
                }),
            }
        }
    }

    fn lower_constructor(
        &mut self,
        constructor: &ConstructorExpr,
        target_name: Option<String>,
        loop_values: &HashMap<String, i64>,
    ) -> Result<EntityId, CompileError> {
        if let Some(template) = self.templates.get(&constructor.type_name.name).copied() {
            return self.lower_template_instance(template, constructor, target_name, loop_values);
        }
        let (kind, dimension) = primitive_kind(&constructor.type_name.name)?;
        let mut inputs = Vec::new();
        let mut arguments = Vec::new();
        for (index, argument) in constructor.args.iter().enumerate() {
            if expression_can_be_entity(&argument.value) {
                if let Ok(entity) = self.lower_entity_expr(&argument.value, None, loop_values) {
                    inputs.push(entity);
                    continue;
                }
            }
            arguments.push(IrArgument {
                name: constructor_argument_name(&constructor.type_name.name, argument, index),
                value: self.lower_value_with_loop(&argument.value, loop_values)?,
            });
        }
        let internal = target_name.is_none();
        let name = target_name.unwrap_or_else(|| self.internal_name("primitive"));
        self.emit_entity(
            dimension,
            name,
            GeometryOperation::Primitive { kind },
            inputs,
            arguments,
            constructor,
            internal,
        )
    }

    /// Instantiate a parameterized geometry template into the caller's
    /// canonical namespace. Templates without declared ports retain the v1
    /// single-result contract; ported templates expose each declared binding
    /// below the instance namespace (`instance.port`).
    fn lower_template_instance(
        &mut self,
        template: &GeometryBlock,
        constructor: &ConstructorExpr,
        target_name: Option<String>,
        loop_values: &HashMap<String, i64>,
    ) -> Result<EntityId, CompileError> {
        if !template.ports.is_empty() {
            return self.lower_ported_template_instance(
                template,
                constructor,
                target_name,
                loop_values,
            );
        }
        if template.body.statements.is_empty() {
            return Err(CompileError::TypeError {
                message: format!("geometry template '{}' body is empty", template.name.name),
            });
        }

        let bound = self.bind_template_arguments(template, constructor, loop_values)?;
        let previous_geometry = self.current_geometry.clone();
        let instance_geometry = target_name
            .clone()
            .unwrap_or_else(|| self.internal_name("template"));
        self.current_geometry = Some(instance_geometry);
        let entity_start = self.ir.entities.len();
        let mut template_loop_values = loop_values.clone();
        let result =
            self.lower_block_with_result(&template.body, target_name, &mut template_loop_values);
        self.current_geometry = previous_geometry;

        for (name, previous) in bound.into_iter().rev() {
            if let Some(previous) = previous {
                self.values.insert(name, previous);
            } else {
                self.values.remove(&name);
            }
        }

        let id = result?.ok_or_else(|| CompileError::TypeError {
            message: format!(
                "geometry template '{}' must end with an entity expression or binding",
                template.name.name
            ),
        })?;
        let template_provenance = self.provenance(template);
        for node in self.ir.entities.iter_mut().skip(entity_start) {
            for reference in &template_provenance {
                if !node.provenance.contains(reference) {
                    node.provenance.push(reference.clone());
                }
            }
        }
        if let Some(node) = self.ir.entities.iter_mut().find(|node| node.id == id) {
            for reference in &template_provenance {
                if !node.provenance.contains(reference) {
                    node.provenance.push(reference.clone());
                }
            }
        }
        Ok(id)
    }

    fn lower_ported_template_instance(
        &mut self,
        template: &GeometryBlock,
        constructor: &ConstructorExpr,
        target_name: Option<String>,
        loop_values: &HashMap<String, i64>,
    ) -> Result<EntityId, CompileError> {
        let instance = target_name.ok_or_else(|| CompileError::TypeError {
            message: format!(
                "ported geometry template '{}' must be bound to an instance",
                template.name.name
            ),
        })?;
        let mut seen = BTreeSet::new();
        for port in &template.ports {
            if port.default.is_some() {
                return Err(CompileError::TypeError {
                    message: format!(
                        "geometry template port '{}.{}' cannot have a default value",
                        template.name.name, port.name.name
                    ),
                });
            }
            if !seen.insert(port.name.name.as_str()) {
                return Err(CompileError::TypeError {
                    message: format!(
                        "geometry template '{}' declares duplicate port '{}'",
                        template.name.name, port.name.name
                    ),
                });
            }
            template_port_dimension(port)?;
        }

        let bound = self.bind_template_arguments(template, constructor, loop_values)?;
        let previous_geometry = self.current_geometry.replace(instance.clone());
        // The caller's compile-time loop/branch namespace is already encoded
        // in `instance`. Template-local bindings must therefore start from a
        // clean suffix/prefix state; otherwise a port declared as `body`
        // inside `model.inner_0` would be lowered as `body_0` (or receive the
        // caller's branch path twice) and could not be exposed as
        // `model.inner_0.body`.
        let previous_loop_indices = std::mem::take(&mut self.loop_indices);
        let previous_branch_path = std::mem::take(&mut self.branch_path);
        let mut template_loop_values = loop_values.clone();
        let lowered = self.lower_block(&template.body, &mut template_loop_values);
        self.branch_path = previous_branch_path;
        self.loop_indices = previous_loop_indices;
        self.current_geometry = previous_geometry;
        self.restore_template_arguments(bound);
        lowered?;

        let template_provenance = self.provenance(template);
        let mut ports = Vec::with_capacity(template.ports.len());
        for port in &template.ports {
            let name = format!("{instance}.{}", port.name.name);
            let expected = template_port_dimension(port)?;
            let id = if let Some(id) = self.entities.get(&name).copied() {
                id
            } else {
                // A loop can produce several deterministic bindings for one
                // public port ('body_0', 'body_1', ...). Treat those bindings
                // as the port's canonical union instead of leaving callers
                // with an arbitrary iteration or an unbound port.
                let prefix = format!("{name}_");
                let mut loop_entities = self
                    .entities
                    .iter()
                    .filter_map(|(candidate, id)| {
                        if !self.loop_generated_entities.contains(candidate) {
                            return None;
                        }
                        let suffix = candidate.strip_prefix(&prefix)?;
                        let indices = suffix
                            .split('_')
                            .map(|part| part.parse::<i64>().ok())
                            .collect::<Option<Vec<_>>>()?;
                        (id.dimension == expected).then(|| (indices, candidate.clone(), *id))
                    })
                    .collect::<Vec<_>>();
                loop_entities
                    .sort_by(|left, right| left.0.cmp(&right.0).then_with(|| left.1.cmp(&right.1)));
                if loop_entities.is_empty() {
                    return Err(CompileError::TypeError {
                        message: format!(
                            "geometry template '{}' declares port '{}' but does not bind it",
                            template.name.name, port.name.name
                        ),
                    });
                }
                let inputs = loop_entities
                    .into_iter()
                    .map(|(_, _, id)| id)
                    .collect::<Vec<_>>();
                let child_provenance = inputs
                    .iter()
                    .filter_map(|input| self.ir.entities.iter().find(|node| node.id == *input))
                    .flat_map(|node| node.provenance.iter().cloned())
                    .collect::<Vec<_>>();
                let aggregate = self.emit_entity(
                    expected,
                    name.clone(),
                    GeometryOperation::Boolean {
                        kind: BooleanOperation::Union,
                    },
                    inputs,
                    Vec::new(),
                    template,
                    false,
                )?;
                if let Some(node) = self
                    .ir
                    .entities
                    .iter_mut()
                    .find(|node| node.id == aggregate)
                {
                    for reference in child_provenance {
                        if !node.provenance.contains(&reference) {
                            node.provenance.push(reference);
                        }
                    }
                }
                aggregate
            };
            if id.dimension != expected {
                return Err(CompileError::TypeError {
                    message: format!(
                        "geometry template port '{}.{}' expects a {} entity but produced {}",
                        template.name.name,
                        port.name.name,
                        expected.topological_dimension(),
                        id.dimension.topological_dimension()
                    ),
                });
            }
            if let Some(node) = self.ir.entities.iter_mut().find(|node| node.id == id) {
                for reference in &template_provenance {
                    if !node.provenance.contains(reference) {
                        node.provenance.push(reference.clone());
                    }
                }
            }
            ports.push(id);
        }

        // The first declared port is the template's primary entity, preserving
        // the expression contract for `let frame = Frame(...)`; all ports are
        // additionally addressable as `frame.<port>` without backend aliases.
        let primary = ports[0];
        self.entities.insert(instance, primary);
        Ok(primary)
    }

    fn bind_template_arguments(
        &mut self,
        template: &GeometryBlock,
        constructor: &ConstructorExpr,
        loop_values: &HashMap<String, i64>,
    ) -> Result<Vec<(String, Option<IrValue>)>, CompileError> {
        let has_named_arguments = constructor
            .args
            .iter()
            .any(|argument| argument.name.is_some());
        if has_named_arguments
            && constructor
                .args
                .iter()
                .any(|argument| argument.name.is_none())
        {
            return Err(CompileError::TypeError {
                message: format!(
                    "geometry template '{}' mixes named and positional arguments",
                    template.name.name
                ),
            });
        }
        if !has_named_arguments && constructor.args.len() > template.params.len() {
            return Err(CompileError::TypeError {
                message: format!(
                    "geometry template '{}' expects at most {} arguments but got {}",
                    template.name.name,
                    template.params.len(),
                    constructor.args.len()
                ),
            });
        }
        if has_named_arguments {
            let mut seen = BTreeSet::new();
            for argument in &constructor.args {
                let Some(argument_name) = argument.name.as_ref() else {
                    return Err(CompileError::TypeError {
                        message: format!(
                            "geometry template '{}' requires named arguments",
                            template.name.name
                        ),
                    });
                };
                let name = argument_name.name.as_str();
                if !template
                    .params
                    .iter()
                    .any(|parameter| parameter.name.name == name)
                {
                    return Err(CompileError::TypeError {
                        message: format!(
                            "geometry template '{}' has unknown argument '{}'",
                            template.name.name, name
                        ),
                    });
                }
                if !seen.insert(name) {
                    return Err(CompileError::TypeError {
                        message: format!(
                            "geometry template '{}' has duplicate argument '{}'",
                            template.name.name, name
                        ),
                    });
                }
            }
        }

        let mut bound = Vec::with_capacity(template.params.len());
        for (index, parameter) in template.params.iter().enumerate() {
            let argument = constructor
                .args
                .iter()
                .find(|argument| {
                    argument
                        .name
                        .as_ref()
                        .is_some_and(|name| name.name == parameter.name.name)
                })
                .or_else(|| {
                    if constructor
                        .args
                        .iter()
                        .any(|argument| argument.name.is_some())
                    {
                        None
                    } else {
                        constructor.args.get(index)
                    }
                })
                .map(|argument| &argument.value)
                .or(parameter.default.as_ref())
                .ok_or_else(|| CompileError::TypeError {
                    message: format!(
                        "missing argument '{}' in geometry template '{}'",
                        parameter.name.name, template.name.name
                    ),
                })?;
            let value = self.lower_value_with_loop(argument, loop_values)?;
            let previous = self.values.insert(parameter.name.name.clone(), value);
            bound.push((parameter.name.name.clone(), previous));
        }
        Ok(bound)
    }

    fn restore_template_arguments(&mut self, bound: Vec<(String, Option<IrValue>)>) {
        for (name, previous) in bound.into_iter().rev() {
            if let Some(previous) = previous {
                self.values.insert(name, previous);
            } else {
                self.values.remove(&name);
            }
        }
    }

    fn lower_boolean(
        &mut self,
        binary: &BinaryExpr,
        target_name: Option<String>,
        loop_values: &HashMap<String, i64>,
    ) -> Result<EntityId, CompileError> {
        let mut operands = Vec::new();
        flatten_csg_operands(binary, binary.op, &mut operands);
        let mut inputs = Vec::with_capacity(operands.len());
        for operand in operands {
            inputs.push(self.lower_entity_expr(operand, None, loop_values)?);
        }
        let dimension = common_dimension(&inputs)?;
        let operation = match binary.op {
            BinaryOp::CsgUnion => BooleanOperation::Union,
            BinaryOp::Sub | BinaryOp::CsgDifference => BooleanOperation::Difference,
            BinaryOp::CsgIntersect => BooleanOperation::Intersection,
            _ => unreachable!(),
        };
        let internal = target_name.is_none();
        let name = target_name.unwrap_or_else(|| self.internal_name("boolean"));
        self.emit_entity(
            dimension,
            name,
            GeometryOperation::Boolean { kind: operation },
            inputs,
            Vec::new(),
            binary,
            internal,
        )
    }

    fn lower_pipe(
        &mut self,
        pipe: &PipeExpr,
        target_name: Option<String>,
        loop_values: &HashMap<String, i64>,
    ) -> Result<EntityId, CompileError> {
        let input = self.lower_entity_expr(&pipe.value, None, loop_values)?;
        let mut inputs = vec![input];
        if pipe.function.name == "fragment" || pipe.function.name == "sweep" {
            for argument in &pipe.args {
                if expression_can_be_entity(&argument.value) {
                    if let Ok(entity) = self.lower_entity_expr(&argument.value, None, loop_values) {
                        inputs.push(entity);
                    }
                }
            }
        }
        let (operation, dimension) = pipe_operation(&pipe.function.name, input.dimension)?;
        let arguments = self.lower_pipe_arguments(pipe, loop_values)?;
        let internal = target_name.is_none();
        let name = target_name.unwrap_or_else(|| self.internal_name("operation"));
        self.emit_entity(
            dimension, name, operation, inputs, arguments, pipe, internal,
        )
    }

    fn lower_pipe_arguments(
        &mut self,
        pipe: &PipeExpr,
        loop_values: &HashMap<String, i64>,
    ) -> Result<Vec<IrArgument>, CompileError> {
        let mut arguments = Vec::new();
        for (index, argument) in pipe.args.iter().enumerate() {
            if matches!(pipe.function.name.as_str(), "fragment" | "sweep")
                && expression_can_be_entity(&argument.value)
            {
                continue;
            }
            arguments.push(IrArgument {
                name: pipe_argument_name(&pipe.function.name, argument, index),
                value: self.lower_value_with_loop(&argument.value, loop_values)?,
            });
        }
        Ok(arguments)
    }

    // Keeping the complete node construction contract in one call makes it
    // difficult to accidentally omit provenance or stable-ID classification.
    #[allow(clippy::too_many_arguments)]
    fn emit_entity<N: AstNode>(
        &mut self,
        dimension: EntityDimension,
        name: String,
        operation: GeometryOperation,
        inputs: Vec<EntityId>,
        arguments: Vec<IrArgument>,
        node: &N,
        internal: bool,
    ) -> Result<EntityId, CompileError> {
        if self.entities.contains_key(&name) {
            return Err(CompileError::DuplicateDefinition {
                name: name.rsplit('.').next().unwrap_or(&name).to_owned(),
                span: node.span(),
            });
        }
        let index = dimension.topological_dimension() as usize;
        let entity_index = if internal {
            let value = self.next_internal[index];
            self.next_internal[index] += 1;
            value
        } else {
            let value = self.next_entity[index];
            self.next_entity[index] += 1;
            value
        };
        let id = EntityId::new(dimension, entity_index);
        self.entities.insert(name.clone(), id);
        // Keep the operation's own source reference first, then carry the
        // complete provenance chain of every input entity.  Backend sidecars
        // expose this list, so derived entities remain traceable through
        // transforms, copies, and CSG operations without changing the IR
        // schema or backend-specific naming.
        let mut provenance = self.provenance(node);
        for input in &inputs {
            if let Some(parent) = self
                .ir
                .entities
                .iter()
                .find(|candidate| candidate.id == *input)
            {
                for reference in &parent.provenance {
                    if !provenance.contains(reference) {
                        provenance.push(reference.clone());
                    }
                }
            }
        }
        self.ir.entities.push(GeometryNode {
            id,
            name,
            operation,
            inputs,
            arguments,
            provenance,
        });
        Ok(id)
    }

    fn lower_metadata_declaration(
        &mut self,
        declaration: &Declaration,
    ) -> Result<(), CompileError> {
        match declaration {
            Declaration::PhysicalRegion(region) => {
                let entities = self.lower_entity_list(&region.entities)?;
                ensure_dimension(&entities, region.dimension, &region.name.name)?;
                let material = region
                    .material
                    .as_ref()
                    .map(|material| self.intern_symbol(&format!("material:{}", material.name)));
                let id = PhysicalGroupId(self.next_group);
                self.next_group += 1;
                self.ir.physical_groups.push(PhysicalGroup {
                    id,
                    name: region.name.name.clone(),
                    dimension: region.dimension,
                    entities,
                    material,
                    provenance: self.provenance(region),
                });
            }
            Declaration::Boundary(boundary) => {
                let entities = self.lower_entity_list(&boundary.entities)?;
                ensure_dimension(&entities, boundary.dimension, &boundary.name.name)?;
                let id = PhysicalGroupId(self.next_group);
                self.next_group += 1;
                self.ir.physical_groups.push(PhysicalGroup {
                    id,
                    name: boundary.name.name.clone(),
                    dimension: boundary.dimension,
                    entities: entities.clone(),
                    material: None,
                    provenance: self.provenance(boundary),
                });
                let kind = match &boundary.kind {
                    BoundaryKind::Dirichlet => IrBoundaryKind::Dirichlet,
                    BoundaryKind::Neumann => IrBoundaryKind::Neumann,
                    BoundaryKind::Robin => IrBoundaryKind::Robin,
                    BoundaryKind::Impedance => IrBoundaryKind::Impedance,
                    BoundaryKind::Symmetry => IrBoundaryKind::Symmetry,
                    BoundaryKind::Periodic => IrBoundaryKind::Periodic,
                    BoundaryKind::Named(name) => IrBoundaryKind::Named(
                        self.intern_symbol(&format!("boundary-kind:{}", name.name)),
                    ),
                };
                let settings = boundary
                    .settings
                    .iter()
                    .map(|setting| {
                        Ok(IrProperty {
                            name: setting.key.name.clone(),
                            value: self.lower_value(&setting.value)?,
                        })
                    })
                    .collect::<Result<Vec<_>, CompileError>>()?;
                self.ir.boundaries.push(IrBoundary {
                    name: boundary.name.name.clone(),
                    dimension: boundary.dimension,
                    entities,
                    kind,
                    settings,
                    provenance: self.provenance(boundary),
                });
            }
            Declaration::Interface(interface) => {
                let primary = self.lower_single_entity(&interface.primary)?;
                let secondary = self.lower_single_entity(&interface.secondary)?;
                if primary.dimension != interface.dimension
                    || secondary.dimension != interface.dimension
                {
                    return Err(CompileError::TypeError {
                        message: format!(
                            "interface '{}' has dimension-wrong entities",
                            interface.name.name
                        ),
                    });
                }
                let kind = match &interface.kind {
                    InterfaceKind::Conforming => IrInterfaceKind::Conforming,
                    InterfaceKind::Contact => IrInterfaceKind::Contact,
                    InterfaceKind::Mortar => IrInterfaceKind::Mortar,
                    InterfaceKind::FluidStructure => IrInterfaceKind::FluidStructure,
                    InterfaceKind::Named(name) => IrInterfaceKind::Named(
                        self.intern_symbol(&format!("interface-kind:{}", name.name)),
                    ),
                };
                self.ir.interfaces.push(IrInterface {
                    name: interface.name.name.clone(),
                    primary,
                    secondary,
                    kind,
                    provenance: self.provenance(interface),
                });
            }
            Declaration::MaterialRegion(region) => {
                let entities = self.lower_entity_list(&region.region)?;
                let material = self.intern_symbol(&format!("material:{}", region.material.name));
                let properties = region
                    .properties
                    .iter()
                    .map(|property| {
                        Ok(IrProperty {
                            name: property.key.name.clone(),
                            value: self.lower_value(&property.value)?,
                        })
                    })
                    .collect::<Result<Vec<_>, CompileError>>()?;
                self.ir.material_regions.push(IrMaterialRegion {
                    name: region.name.name.clone(),
                    material,
                    entities,
                    properties,
                    provenance: self.provenance(region),
                });
            }
            Declaration::CoordinateSystem(system) => self.lower_coordinate_system(system)?,
            Declaration::Objective(objective) => {
                let weight = objective
                    .weight
                    .as_ref()
                    .map(|value| self.lower_f64(value))
                    .transpose()?
                    .unwrap_or(1.0);
                let value = self.lower_value(&objective.expression)?;
                let provenance = self.provenance(objective);
                self.ir.optimization.objectives.push(IrObjective {
                    name: objective.name.as_ref().map(|name| name.name.clone()),
                    sense: objective.sense,
                    value,
                    weight,
                    provenance,
                });
            }
            Declaration::Constraint(constraint) => {
                if let Some(lowered) = self.lower_constraint(&constraint.kind)? {
                    self.ir.optimization.constraints.push(IrConstraintRecord {
                        name: constraint.name.as_ref().map(|name| name.name.clone()),
                        constraint: lowered,
                        provenance: self.provenance(constraint),
                    });
                }
            }
            Declaration::Uncertainty(uncertainty) => {
                let id = self.intern_symbol(&format!("uncertainty:{}", uncertainty.name.name));
                let distribution = self.lower_distribution(&uncertainty.distribution)?;
                self.ir.uncertainties.push(IrUncertainty {
                    id,
                    name: uncertainty.name.name.clone(),
                    distribution,
                    provenance: self.provenance(uncertainty),
                });
            }
            Declaration::GeometryField(field) => self.lower_geometry_field(field)?,
            Declaration::DesignRegion(region) => self.lower_design_region(region)?,
            Declaration::Mesh(mesh) => self.lower_mesh(mesh)?,
            Declaration::Physics(physics) => {
                for statement in &physics.statements {
                    if let PhysicsStatement::Group(group) = statement {
                        let entities = self.lower_entity_list(&group.value)?;
                        let dimension = match group.bc_type.name.as_str() {
                            "volume" | "domain" => EntityDimension::Volume,
                            "curve" | "edge" => EntityDimension::Curve,
                            "point" | "vertex" => EntityDimension::Point,
                            _ => entities
                                .first()
                                .map(|entity| entity.dimension)
                                .unwrap_or(EntityDimension::Surface),
                        };
                        ensure_dimension(&entities, dimension, &group.name.name)?;
                        let id = PhysicalGroupId(self.next_group);
                        self.next_group += 1;
                        self.ir.physical_groups.push(PhysicalGroup {
                            id,
                            name: group.name.name.clone(),
                            dimension,
                            entities,
                            material: None,
                            provenance: self.provenance(group),
                        });
                    }
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn lower_coordinate_system(
        &mut self,
        system: &CoordinateSystemDecl,
    ) -> Result<(), CompileError> {
        let (origin, basis) = match &system.kind {
            CoordinateSystemKind::Cartesian { origin, axes } => {
                if axes.len() != 3 {
                    return Err(CompileError::TypeError {
                        message: format!(
                            "coordinate system '{}' requires three axes",
                            system.name.name
                        ),
                    });
                }
                let x = normalize(self.lower_vec3(&axes[0])?)?;
                let y = normalize(self.lower_vec3(&axes[1])?)?;
                let z = normalize(self.lower_vec3(&axes[2])?)?;
                ensure_orthogonal(x, y, &system.name.name)?;
                ensure_orthogonal(x, z, &system.name.name)?;
                ensure_orthogonal(y, z, &system.name.name)?;
                let basis = [x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]];
                (self.lower_vec3(origin)?, basis)
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
                let z = normalize(self.lower_vec3(axis)?)?;
                let reference = self.lower_vec3(reference)?;
                let y = normalize(cross(z, reference)).map_err(|_| CompileError::TypeError {
                    message: format!(
                        "coordinate system '{}' axis and reference must not be parallel",
                        system.name.name
                    ),
                })?;
                let x = normalize(cross(y, z))?;
                (
                    self.lower_vec3(origin)?,
                    [x[0], x[1], x[2], y[0], y[1], y[2], z[0], z[1], z[2]],
                )
            }
        };
        let id = self.intern_symbol(&format!("coordinate-system:{}", system.name.name));
        self.ir.coordinate_systems.push(IrCoordinateSystem {
            id,
            name: system.name.name.clone(),
            origin,
            basis,
            provenance: self.provenance(system),
        });
        Ok(())
    }

    fn lower_constraint(
        &mut self,
        constraint: &ConstraintKind,
    ) -> Result<Option<IrConstraint>, CompileError> {
        Ok(Some(match constraint {
            ConstraintKind::Equality {
                left,
                right,
                tolerance,
            } => IrConstraint::Equality {
                left: self.lower_value(left)?,
                right: self.lower_value(right)?,
                tolerance: tolerance
                    .as_ref()
                    .map(|value| self.lower_f64(value))
                    .transpose()?
                    .unwrap_or(0.0),
            },
            ConstraintKind::Inequality {
                left,
                relation,
                right,
            } => IrConstraint::Inequality {
                left: self.lower_value(left)?,
                relation: *relation,
                right: self.lower_value(right)?,
            },
            ConstraintKind::Range { expression, bounds } => IrConstraint::Range {
                value: self.lower_value(expression)?,
                bounds: IrBounds {
                    lower: self.lower_value(&bounds.lower)?,
                    upper: self.lower_value(&bounds.upper)?,
                    lower_inclusive: bounds.lower_inclusive,
                    upper_inclusive: bounds.upper_inclusive,
                },
            },
            ConstraintKind::VolumeFraction { region, maximum } => IrConstraint::VolumeFraction {
                region: self.lower_entity_list(region)?,
                maximum: self.lower_value(maximum)?,
            },
            ConstraintKind::MinimumFeature { region, radius } => IrConstraint::MinimumFeature {
                region: self.lower_entity_list(region)?,
                radius: self.lower_value(radius)?,
            },
        }))
    }

    fn lower_geometry_field(&mut self, field: &GeometryFieldDecl) -> Result<(), CompileError> {
        let id = self.intern_symbol(&format!("geometry-field:{}", field.name.name));
        let kind = match &field.kind {
            GeometryFieldKind::Topology {
                region,
                initial_density,
                filter_radius,
            } => IrGeometryFieldKind::Topology {
                region: self.lower_entity_list(region)?,
                initial_density: self.lower_value(initial_density)?,
                filter_radius: filter_radius
                    .as_ref()
                    .map(|value| self.lower_value(value))
                    .transpose()?,
            },
            GeometryFieldKind::LevelSet { field, iso_value } => IrGeometryFieldKind::LevelSet {
                field: self.lower_value(field)?,
                iso_value: self.lower_value(iso_value)?,
            },
            GeometryFieldKind::Projected {
                geometry,
                provenance,
            } => IrGeometryFieldKind::Projected {
                geometry: self.lower_entity_list(geometry)?,
                source_field: self
                    .intern_symbol(&format!("geometry-field:{}", provenance.source_field.name)),
                method: provenance.method,
                tolerance: provenance
                    .tolerance
                    .as_ref()
                    .map(|value| self.lower_value(value))
                    .transpose()?,
            },
        };
        self.ir.fields.push(IrGeometryField {
            id,
            name: field.name.name.clone(),
            kind,
            provenance: self.provenance(field),
        });
        Ok(())
    }

    fn lower_design_region(&mut self, region: &DesignRegionDecl) -> Result<(), CompileError> {
        let kind = match region.kind {
            DesignRegionKind::Design => IrDesignRegionKind::Design,
            DesignRegionKind::NonDesign => IrDesignRegionKind::NonDesign,
            DesignRegionKind::KeepOut => IrDesignRegionKind::KeepOut,
        };
        let mut annotations = Vec::new();
        for annotation in &region.annotations {
            annotations.push(match annotation {
                ManufacturingAnnotation::MinimumThickness { value, unit } => {
                    IrManufacturingAnnotation::MinimumThickness(
                        self.lower_measure(value, unit.as_ref())?,
                    )
                }
                ManufacturingAnnotation::MinimumRadius { value, unit } => {
                    IrManufacturingAnnotation::MinimumRadius(
                        self.lower_measure(value, unit.as_ref())?,
                    )
                }
                ManufacturingAnnotation::Overhang {
                    maximum_angle,
                    direction,
                } => IrManufacturingAnnotation::Overhang {
                    maximum_angle: self.lower_f64(maximum_angle)?,
                    direction: self.lower_vec3(direction)?,
                },
                ManufacturingAnnotation::Symmetry { kind } => match kind {
                    SymmetryKind::Plane { origin, normal } => {
                        IrManufacturingAnnotation::SymmetryPlane {
                            origin: self.lower_vec3(origin)?,
                            normal: self.lower_vec3(normal)?,
                        }
                    }
                    SymmetryKind::Axis { origin, direction } => {
                        IrManufacturingAnnotation::SymmetryAxis {
                            origin: self.lower_vec3(origin)?,
                            direction: self.lower_vec3(direction)?,
                        }
                    }
                    SymmetryKind::Cyclic { axis, sectors } => {
                        let axis = self.lower_vec3(axis)?;
                        let axis_norm = axis
                            .iter()
                            .map(|component| component * component)
                            .sum::<f64>()
                            .sqrt();
                        if !axis_norm.is_finite() || axis_norm <= f64::EPSILON {
                            return Err(CompileError::TypeError {
                                message: "cyclic symmetry axis must be finite and non-zero".into(),
                            });
                        }
                        let sectors = self.lower_f64(sectors)?;
                        if !sectors.is_finite()
                            || sectors.fract() != 0.0
                            || !(2.0..=f64::from(u32::MAX)).contains(&sectors)
                        {
                            return Err(CompileError::TypeError {
                                message: "cyclic symmetry sectors must be an integer >= 2".into(),
                            });
                        }
                        IrManufacturingAnnotation::SymmetryCyclic {
                            axis,
                            sectors: sectors as u32,
                        }
                    }
                },
                ManufacturingAnnotation::DrawDirection { direction } => {
                    IrManufacturingAnnotation::DrawDirection(self.lower_vec3(direction)?)
                }
                ManufacturingAnnotation::Frozen => IrManufacturingAnnotation::Frozen,
            });
        }
        let entities = self.lower_entity_list(&region.region)?;
        let provenance = self.provenance(region);
        self.ir.optimization.regions.push(IrDesignRegion {
            name: region.name.name.clone(),
            kind,
            entities,
            annotations,
            provenance,
        });
        Ok(())
    }

    fn lower_mesh(&mut self, mesh: &MeshBlock) -> Result<(), CompileError> {
        if let Some(target) = &mesh.target {
            self.ir.mesh.targets.push(target.name.clone());
        }
        for statement in &mesh.statements {
            match statement {
                MeshStatement::KeyValue(setting) => {
                    let property = self.lower_mesh_property(setting)?;
                    self.ir.mesh.options.push(property);
                }
                MeshStatement::Metric(metric) => {
                    let tensor = match &metric.metric {
                        MeshMetricKind::Isotropic { size } => {
                            IrMetricTensor::Isotropic(self.lower_f64(size)?)
                        }
                        MeshMetricKind::Anisotropic { components } => {
                            if components.len() != 6 {
                                return Err(CompileError::TypeError {
                                    message: "anisotropic metric requires six components".into(),
                                });
                            }
                            let mut values = [0.0; 6];
                            for (slot, expression) in values.iter_mut().zip(components) {
                                *slot = self.lower_f64(expression)?;
                            }
                            IrMetricTensor::Anisotropic(values)
                        }
                        MeshMetricKind::Hessian {
                            field,
                            minimum_size,
                            maximum_size,
                        } => IrMetricTensor::Hessian {
                            field: self.lower_value(field)?,
                            minimum_size: self.lower_f64(minimum_size)?,
                            maximum_size: self.lower_f64(maximum_size)?,
                        },
                    };
                    let coordinate_system = metric.coordinate_system.as_ref().map(|system| {
                        self.intern_symbol(&format!("coordinate-system:{}", system.name))
                    });
                    let target = self.lower_mesh_entity_list(&metric.target)?;
                    let provenance = self.provenance(metric);
                    self.ir.mesh.metrics.push(IrMeshMetric {
                        target,
                        tensor,
                        coordinate_system,
                        provenance,
                    });
                }
                MeshStatement::Refine(refine) => {
                    let kind = match &refine.kind {
                        RefineKind::Near { entity, args } => IrRefineKind::Near {
                            entity: self.lower_mesh_entity(entity)?,
                            args: self.lower_mesh_properties(args)?,
                        },
                        RefineKind::On { entity } => IrRefineKind::On {
                            entity: self.lower_mesh_entity(entity)?,
                        },
                        RefineKind::In { region } => IrRefineKind::In {
                            region: self.lower_mesh_entity(region)?,
                        },
                        RefineKind::Field { args } => IrRefineKind::Field {
                            args: self.lower_mesh_properties(args)?,
                        },
                    };
                    let body = self.lower_mesh_properties(&refine.body)?;
                    self.ir.mesh.refinements.push(IrRefineControl {
                        kind,
                        body,
                        provenance: self.provenance(refine),
                    });
                }
                MeshStatement::BoundaryLayer(layer) => {
                    let values = layer
                        .body
                        .iter()
                        .map(|value| (value.key.name.as_str(), value))
                        .collect::<HashMap<_, _>>();
                    let number_of_layers = values
                        .get("layers")
                        .map(|value| self.lower_layer_count(value))
                        .transpose()?;
                    let first_layer_height = self.lower_key_value_f64(required_setting(
                        &values,
                        &["hwall_n", "first_layer_height", "first_height"],
                    )?)?;
                    let growth_ratio = values
                        .get("ratio")
                        .or_else(|| values.get("growth_ratio"))
                        .or_else(|| values.get("growth_rate"))
                        .map(|value| self.lower_key_value_f64(value))
                        .transpose()?
                        .unwrap_or(1.1);
                    let total_thickness = values
                        .get("thickness")
                        .or_else(|| values.get("total_thickness"))
                        .map(|value| self.lower_key_value_f64(value))
                        .transpose()?
                        .unwrap_or_else(|| {
                            let layers = number_of_layers.unwrap_or(1) as f64;
                            if (growth_ratio - 1.0).abs() <= f64::EPSILON {
                                first_layer_height * layers
                            } else {
                                first_layer_height * (growth_ratio.powf(layers) - 1.0)
                                    / (growth_ratio - 1.0)
                            }
                        });
                    if first_layer_height <= 0.0
                        || growth_ratio <= 0.0
                        || total_thickness <= 0.0
                        || !first_layer_height.is_finite()
                        || !growth_ratio.is_finite()
                        || !total_thickness.is_finite()
                    {
                        return Err(CompileError::TypeError {
                            message: "boundary-layer heights, thickness, and growth ratio must be finite and positive".into(),
                        });
                    }
                    let target = self.lower_mesh_entity_list(&layer.target)?;
                    let provenance = self.provenance(layer);
                    self.ir.mesh.boundary_layers.push(IrBoundaryLayer {
                        target,
                        first_layer_height,
                        total_thickness,
                        growth_ratio,
                        number_of_layers,
                        provenance,
                    });
                }
                MeshStatement::Transfinite(transfinite) => {
                    let kind = match &transfinite.kind {
                        TransfiniteKind::Curve { entity, settings } => IrTransfiniteKind::Curve {
                            entity: self.lower_mesh_entity(entity)?,
                            settings: self.lower_mesh_properties(settings)?,
                        },
                        TransfiniteKind::Surface { entity, settings } => {
                            IrTransfiniteKind::Surface {
                                entity: self.lower_mesh_entity(entity)?,
                                settings: self.lower_mesh_properties(settings)?,
                            }
                        }
                        TransfiniteKind::Volume { entity, settings } => IrTransfiniteKind::Volume {
                            entity: self.lower_mesh_entity(entity)?,
                            settings: self.lower_mesh_properties(settings)?,
                        },
                    };
                    self.ir.mesh.transfinite.push(IrTransfiniteControl {
                        kind,
                        provenance: self.provenance(transfinite),
                    });
                }
                MeshStatement::Embed(embed) => {
                    let source = self.lower_mesh_entity(&embed.source)?;
                    let target = self.lower_mesh_entity(&embed.target)?;
                    let source_dimension = embed_dim_to_entity_dimension(embed.embed_dim);
                    let target_dimension = embed_dim_to_entity_dimension(embed.target_dim);
                    if source.dimension != source_dimension || target.dimension != target_dimension
                    {
                        return Err(CompileError::TypeError {
                            message: "embed source or target has the wrong dimension".into(),
                        });
                    }
                    self.ir.mesh.embeds.push(IrEmbedControl {
                        source,
                        target,
                        source_dimension,
                        target_dimension,
                        provenance: self.provenance(embed),
                    });
                }
                MeshStatement::Periodic(periodic) => {
                    let source = self.lower_mesh_entity(&periodic.source)?;
                    let target = self.lower_mesh_entity(&periodic.target)?;
                    let coefficients = self.lower_transform(&periodic.transform)?;
                    self.ir.periodic_pairs.push(IrPeriodicPair {
                        source,
                        target,
                        transform: IrTransform { coefficients },
                        provenance: self.provenance(periodic),
                    });
                }
                MeshStatement::Compound(compound) => {
                    let mut entities = Vec::new();
                    for expression in &compound.entities {
                        entities.extend(self.lower_mesh_entity_list(expression)?);
                    }
                    if entities.is_empty() {
                        return Err(CompileError::TypeError {
                            message: "compound mesh control requires at least one entity".into(),
                        });
                    }
                    self.ir.mesh.compounds.push(IrCompoundControl {
                        entities,
                        provenance: self.provenance(compound),
                    });
                }
            }
        }
        Ok(())
    }

    fn lower_mesh_properties(
        &mut self,
        properties: &[KeyValue],
    ) -> Result<Vec<IrProperty>, CompileError> {
        properties
            .iter()
            .map(|property| self.lower_mesh_property(property))
            .collect()
    }

    fn lower_mesh_property(&mut self, property: &KeyValue) -> Result<IrProperty, CompileError> {
        let value = self.lower_value(&property.value)?;
        let value = if let Some(unit) = &property.unit {
            apply_unit_scale(value, unit.to_si_factor, unit.dimension)?
        } else {
            value
        };
        Ok(IrProperty {
            name: property.key.name.clone(),
            value,
        })
    }

    fn lower_mesh_entity(&mut self, expression: &Expr) -> Result<EntityId, CompileError> {
        match expression {
            Expr::Ident(_) | Expr::FieldAccess(_) => self.lower_single_entity(expression),
            Expr::Paren(paren) => self.lower_mesh_entity(&paren.value),
            _ => self.lower_entity_expr(expression, None, &HashMap::new()),
        }
    }

    fn lower_mesh_entity_list(&mut self, expression: &Expr) -> Result<Vec<EntityId>, CompileError> {
        match expression {
            Expr::Array(array) => array
                .elements
                .iter()
                .map(|value| self.lower_mesh_entity(value))
                .collect(),
            _ => Ok(vec![self.lower_mesh_entity(expression)?]),
        }
    }

    fn lower_transform(&mut self, expression: &Expr) -> Result<[f64; 16], CompileError> {
        let values = self.lower_numeric_vector(expression)?;
        match values.len() {
            3 => Ok([
                1.0, 0.0, 0.0, values[0], 0.0, 1.0, 0.0, values[1], 0.0, 0.0, 1.0, values[2], 0.0,
                0.0, 0.0, 1.0,
            ]),
            16 => values
                .try_into()
                .map_err(|_| CompileError::Internal("invalid affine transform".into())),
            _ => Err(CompileError::TypeError {
                message: "periodic transform must contain 3 or 16 scalars".into(),
            }),
        }
    }

    fn lower_entity_list(&mut self, expression: &Expr) -> Result<Vec<EntityId>, CompileError> {
        match expression {
            Expr::Array(array) => array
                .elements
                .iter()
                .map(|value| self.lower_single_entity(value))
                .collect(),
            Expr::Binary(binary) if is_entity_csg_operator(binary.op) => {
                Ok(vec![self.lower_entity_expr(
                    expression,
                    None,
                    &HashMap::new(),
                )?])
            }
            _ => Ok(vec![self.lower_single_entity(expression)?]),
        }
    }

    fn lower_single_entity(&mut self, expression: &Expr) -> Result<EntityId, CompileError> {
        let name = match expression {
            Expr::Ident(identifier) => self.resolve_entity_name(&identifier.name)?,
            Expr::FieldAccess(_) => {
                let path = entity_path(expression).ok_or_else(|| CompileError::TypeError {
                    message: "expected an entity name".into(),
                })?;
                self.resolve_entity_path(&path)
            }
            Expr::Paren(paren) => return self.lower_single_entity(&paren.value),
            _ => {
                return Err(CompileError::TypeError {
                    message: "metadata entity references must be named geometry bindings".into(),
                });
            }
        };
        if let Some(entity) = self.entities.get(&name).copied() {
            return Ok(entity);
        }
        // Topology selectors such as `solid.faces` are evaluated by the
        // backend after OCC synchronization. Their sensitivity/provenance
        // owner in canonical IR is the named parent entity.
        if let Some(entity) = self.topology_entities.get(&name).copied() {
            return Ok(entity);
        }
        if let Some((owner, target, selector)) = topology_selector(&name) {
            if let Some(parent) = self.entities.get(owner).copied() {
                let generated_name = self.internal_name("boundary");
                let id = self.emit_entity(
                    target,
                    generated_name,
                    GeometryOperation::Boundary { target, selector },
                    vec![parent],
                    Vec::new(),
                    expression,
                    true,
                )?;
                self.topology_entities.insert(name, id);
                return Ok(id);
            }
        }
        Err(CompileError::UndefinedReference { name })
    }

    fn lower_value(&mut self, expression: &Expr) -> Result<IrValue, CompileError> {
        self.lower_value_with_loop(expression, &HashMap::new())
    }

    fn lower_value_with_loop(
        &mut self,
        expression: &Expr,
        loop_values: &HashMap<String, i64>,
    ) -> Result<IrValue, CompileError> {
        Ok(match expression {
            Expr::Literal(literal) => match &literal.value {
                Literal::Int(value) => IrValue::Integer(*value),
                Literal::Float(value) => IrValue::Scalar {
                    value: *value,
                    dimension: units::Dimension::DIMENSIONLESS,
                },
                Literal::String(value) => IrValue::Text(value.clone()),
                Literal::Bool(value) => IrValue::Boolean(*value),
            },
            Expr::Ident(identifier) => {
                if let Some(value) = loop_values.get(&identifier.name) {
                    IrValue::Integer(*value)
                } else if let Ok(name) = self.resolve_entity_name(&identifier.name) {
                    IrValue::Entity(self.entities[&name])
                } else if let Some(value) = self.values.get(&identifier.name) {
                    value.clone()
                } else {
                    let id = self.intern_symbol(&format!("value:{}", identifier.name));
                    self.symbols.insert(identifier.name.clone(), id);
                    IrValue::Parameter(id)
                }
            }
            Expr::Unary(unary) => IrValue::Unary {
                op: match unary.op {
                    UnaryOp::Neg => IrUnaryOperator::Negate,
                    UnaryOp::Not => IrUnaryOperator::LogicalNot,
                },
                operand: Box::new(self.lower_value_with_loop(&unary.operand, loop_values)?),
            },
            Expr::Binary(binary) if !is_explicit_csg_operator(binary.op) => IrValue::Binary {
                op: lower_binary_operator(binary.op)?,
                left: Box::new(self.lower_value_with_loop(&binary.left, loop_values)?),
                right: Box::new(self.lower_value_with_loop(&binary.right, loop_values)?),
            },
            Expr::Paren(paren) => self.lower_value_with_loop(&paren.value, loop_values)?,
            Expr::Array(array) => IrValue::Vector(
                array
                    .elements
                    .iter()
                    .map(|value| self.lower_value_with_loop(value, loop_values))
                    .collect::<Result<_, _>>()?,
            ),
            Expr::Index(index) => IrValue::Index {
                collection: Box::new(self.lower_value_with_loop(&index.object, loop_values)?),
                index: Box::new(self.lower_value_with_loop(&index.index, loop_values)?),
            },
            Expr::FieldAccess(field) => IrValue::Field {
                object: Box::new(self.lower_value_with_loop(&field.object, loop_values)?),
                field: self.intern_symbol(&format!("field:{}", field.field.name)),
            },
            Expr::Call(call) => IrValue::Call {
                function: self.intern_symbol(&format!("function:{}", call.function.name)),
                arguments: call
                    .args
                    .iter()
                    .map(|argument| self.lower_value_with_loop(&argument.value, loop_values))
                    .collect::<Result<_, _>>()?,
            },
            Expr::StructInit(value) => IrValue::Call {
                function: self
                    .intern_symbol(&format!("struct-constructor:{}", value.type_name.name)),
                arguments: value
                    .fields
                    .iter()
                    .map(|field| self.lower_value_with_loop(&field.value, loop_values))
                    .collect::<Result<_, _>>()?,
            },
            _ => {
                return Err(CompileError::TypeError {
                    message: "expression cannot be represented as a canonical IR value".into(),
                });
            }
        })
    }

    fn lower_distribution(
        &mut self,
        distribution: &Distribution,
    ) -> Result<IrDistribution, CompileError> {
        Ok(match distribution {
            Distribution::Deterministic { value } => {
                IrDistribution::Deterministic(self.lower_value(value)?)
            }
            Distribution::Uniform { lower, upper } => IrDistribution::Uniform {
                lower: self.lower_value(lower)?,
                upper: self.lower_value(upper)?,
            },
            Distribution::Normal {
                mean,
                standard_deviation,
            } => IrDistribution::Normal {
                mean: self.lower_value(mean)?,
                standard_deviation: self.lower_value(standard_deviation)?,
            },
            Distribution::LogNormal {
                mean,
                standard_deviation,
            } => IrDistribution::LogNormal {
                mean: self.lower_value(mean)?,
                standard_deviation: self.lower_value(standard_deviation)?,
            },
            Distribution::Discrete { values, weights } => IrDistribution::Discrete {
                values: values
                    .iter()
                    .map(|value| self.lower_value(value))
                    .collect::<Result<_, _>>()?,
                weights: weights
                    .iter()
                    .map(|weight| self.lower_f64(weight))
                    .collect::<Result<_, _>>()?,
            },
        })
    }

    fn lower_f64(&mut self, expression: &Expr) -> Result<f64, CompileError> {
        match self.lower_value(expression)? {
            IrValue::Integer(value) => Ok(value as f64),
            IrValue::Scalar { value, .. } => Ok(value),
            _ => Err(CompileError::TypeError {
                message: "expected a compile-time scalar".into(),
            }),
        }
    }

    fn lower_measure(
        &mut self,
        expression: &Expr,
        unit: Option<&units::Unit>,
    ) -> Result<f64, CompileError> {
        let value = self.lower_f64(expression)?;
        let scaled = unit.map_or(value, |unit| value * unit.to_si_factor);
        if scaled.is_finite() {
            Ok(scaled)
        } else {
            Err(CompileError::TypeError {
                message: "scaled measurement must be finite".into(),
            })
        }
    }

    fn lower_key_value_f64(&mut self, value: &KeyValue) -> Result<f64, CompileError> {
        self.lower_measure(&value.value, value.unit.as_ref())
    }

    fn lower_layer_count(&mut self, value: &KeyValue) -> Result<u32, CompileError> {
        if value.unit.is_some() {
            return Err(CompileError::TypeError {
                message: "boundary-layer count must be dimensionless".into(),
            });
        }
        let count = self.lower_f64(&value.value)?;
        if !count.is_finite() || count < 1.0 || count.fract() != 0.0 || count > f64::from(u32::MAX)
        {
            return Err(CompileError::TypeError {
                message: "boundary-layer count must be a positive 32-bit integer".into(),
            });
        }
        Ok(count as u32)
    }

    fn lower_numeric_vector(&mut self, expression: &Expr) -> Result<Vec<f64>, CompileError> {
        match expression {
            Expr::Array(array) => array
                .elements
                .iter()
                .map(|value| self.lower_f64(value))
                .collect(),
            Expr::Paren(paren) => self.lower_numeric_vector(&paren.value),
            _ => Err(CompileError::TypeError {
                message: "expected a numeric vector".into(),
            }),
        }
    }

    fn lower_vec3(&mut self, expression: &Expr) -> Result<[f64; 3], CompileError> {
        self.lower_numeric_vector(expression)?
            .try_into()
            .map_err(|_| CompileError::TypeError {
                message: "expected a three-component vector".into(),
            })
    }

    fn const_int(
        &self,
        expression: &Expr,
        loop_values: &HashMap<String, i64>,
    ) -> Result<i64, CompileError> {
        const_int_expression(expression, loop_values, &self.values)
    }

    fn try_const_int(
        &self,
        expression: &Expr,
        loop_values: &HashMap<String, i64>,
    ) -> Result<Option<i64>, CompileError> {
        match self.const_int(expression, loop_values) {
            Ok(value) => Ok(Some(value)),
            Err(CompileError::Unsupported(message)) if message.contains("loop range must") => {
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    fn const_bool(
        &mut self,
        expression: &Expr,
        loop_values: &HashMap<String, i64>,
    ) -> Result<bool, CompileError> {
        self.try_const_bool(expression, loop_values)?
            .ok_or_else(|| {
                CompileError::Unsupported(
                    "geometry condition must be a compile-time boolean".into(),
                )
            })
    }

    fn try_const_bool(
        &mut self,
        expression: &Expr,
        loop_values: &HashMap<String, i64>,
    ) -> Result<Option<bool>, CompileError> {
        let value = self.lower_value_with_loop(expression, loop_values)?;
        Ok(const_bool_value(&value))
    }

    fn inline_call(&mut self, call: &CallExpr) -> Result<Expr, CompileError> {
        let function = self.functions[&call.function.name];
        let function_name = function.name.name.clone();
        let params = function.params.clone();
        let body = function.body.clone();
        let mut substitutions = HashMap::new();
        for (index, parameter) in params.iter().enumerate() {
            let argument = call
                .args
                .iter()
                .find(|argument| {
                    argument
                        .name
                        .as_ref()
                        .is_some_and(|name| name.name == parameter.name.name)
                })
                .or_else(|| call.args.get(index));
            let value = argument
                .map(|argument| argument.value.clone())
                .or_else(|| parameter.default.clone())
                .ok_or_else(|| CompileError::TypeError {
                    message: format!(
                        "missing argument '{}' in call to '{}'",
                        parameter.name.name, function_name
                    ),
                })?;
            substitutions.insert(parameter.name.name.clone(), value);
        }

        self.inline_block_value(&body, &substitutions, &function_name)
    }

    /// Inline a function body into one expression while preserving compile-time
    /// control flow for the canonical lowerer.  Branch blocks remain blocks so
    /// `lower_block_result` can apply the same loop/scope rules as top-level
    /// geometry; sequential local bindings are substituted eagerly.
    fn inline_block_value(
        &mut self,
        block: &Block,
        parent_substitutions: &HashMap<String, Expr>,
        function_name: &str,
    ) -> Result<Expr, CompileError> {
        let mut substitutions = parent_substitutions.clone();
        let Some((last_index, _)) = block.statements.iter().enumerate().next_back() else {
            return Err(CompileError::Unsupported(format!(
                "function '{function_name}' has no final expression"
            )));
        };

        for (index, statement) in block.statements.iter().enumerate() {
            if index == last_index {
                return match statement {
                    Statement::Let(binding) => Ok(substitute_expr(&binding.value, &substitutions)),
                    Statement::Expr(expression) => Ok(substitute_expr(expression, &substitutions)),
                    Statement::If(statement) => Ok(Expr::If(Box::new(substitute_inline_if(
                        statement,
                        &substitutions,
                    )))),
                    Statement::For(statement) => {
                        self.inline_for_value(statement, &substitutions, function_name)
                    }
                    Statement::KeyValue(_) | Statement::Group(_) => {
                        Err(CompileError::Unsupported(format!(
                            "function '{function_name}' must end in a value-producing statement"
                        )))
                    }
                };
            }

            match statement {
                Statement::Let(binding) => {
                    let value = substitute_expr(&binding.value, &substitutions);
                    substitutions.insert(binding.name.name.clone(), value);
                }
                Statement::If(statement) => {
                    self.apply_inline_if_bindings(statement, &mut substitutions, function_name)?;
                }
                Statement::For(statement) => {
                    self.apply_inline_for_bindings(statement, &mut substitutions, function_name)?;
                }
                Statement::Expr(_) | Statement::KeyValue(_) | Statement::Group(_) => {
                    return Err(CompileError::Unsupported(format!(
                        "function '{function_name}' has a non-final expression"
                    )));
                }
            }
        }

        unreachable!("non-empty block always returns its final statement")
    }

    /// Apply the bindings from a compile-time inline `if` branch to the
    /// surrounding substitution environment. A non-final branch in an
    /// inline function can only contain bindings; its condition must be
    /// decidable while compiling the call.
    fn apply_inline_if_bindings(
        &mut self,
        statement: &IfStatement,
        substitutions: &mut HashMap<String, Expr>,
        function_name: &str,
    ) -> Result<(), CompileError> {
        let condition = substitute_expr(&statement.condition, substitutions);
        let Some(take_then) = inline_const_bool(&condition) else {
            return Err(CompileError::Unsupported(format!(
                "inline function '{function_name}' branch condition must be a compile-time boolean"
            )));
        };

        if take_then {
            self.apply_inline_binding_block(&statement.then_block, substitutions, function_name)
        } else {
            match statement.else_block.as_ref() {
                Some(ElseBranch::Else(block)) => {
                    self.apply_inline_binding_block(block, substitutions, function_name)
                }
                Some(ElseBranch::ElseIf(next)) => {
                    self.apply_inline_if_bindings(next.as_ref(), substitutions, function_name)
                }
                None => Ok(()),
            }
        }
    }

    fn apply_inline_binding_block(
        &mut self,
        block: &Block,
        substitutions: &mut HashMap<String, Expr>,
        function_name: &str,
    ) -> Result<(), CompileError> {
        for statement in &block.statements {
            match statement {
                Statement::Let(binding) => {
                    let value = substitute_expr(&binding.value, substitutions);
                    substitutions.insert(binding.name.name.clone(), value);
                }
                Statement::If(statement) => {
                    self.apply_inline_if_bindings(statement, substitutions, function_name)?;
                }
                Statement::For(statement) => {
                    self.apply_inline_for_bindings(statement, substitutions, function_name)?;
                }
                Statement::Expr(_) | Statement::KeyValue(_) | Statement::Group(_) => {
                    return Err(CompileError::Unsupported(format!(
                        "inline function '{function_name}' branch may only contain bindings"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Expand a non-final inline loop into aggregate entity bindings.
    ///
    /// Loop-local scalar bindings are available while evaluating each body,
    /// but only entity-valued bindings escape the loop.  Repeated entity
    /// bindings are folded in iteration order with the same canonical CSG
    /// union used by a value-producing final loop.
    fn apply_inline_for_bindings(
        &mut self,
        statement: &ForStatement,
        substitutions: &mut HashMap<String, Expr>,
        function_name: &str,
    ) -> Result<(), CompileError> {
        const MAX_INLINE_ITERATIONS: u64 = 1_000_000;
        let mut products = vec![HashMap::<String, Expr>::new()];

        for clause in &statement.clauses {
            let mut next = Vec::new();
            for values in products {
                let mut range_substitutions = substitutions.clone();
                range_substitutions.extend(values.clone());
                let start_expr = substitute_expr(&clause.range.start, &range_substitutions);
                let end_expr = substitute_expr(&clause.range.end, &range_substitutions);
                let start = self.const_int(&start_expr, &HashMap::new())?;
                let mut end = self.const_int(&end_expr, &HashMap::new())?;
                if clause.range.inclusive {
                    end = end.checked_add(1).ok_or_else(|| {
                        CompileError::Unsupported(
                            "inclusive inline-function range exceeds i64".into(),
                        )
                    })?;
                }
                let count = end.saturating_sub(start) as u64;
                if count > MAX_INLINE_ITERATIONS {
                    return Err(CompileError::Unsupported(format!(
                        "inline function '{function_name}' loop exceeds {MAX_INLINE_ITERATIONS} iterations"
                    )));
                }
                for value in start..end {
                    let mut expanded = values.clone();
                    expanded.insert(
                        clause.var.name.clone(),
                        Expr::literal(Literal::Int(value), clause.var.span),
                    );
                    next.push(expanded);
                    if next.len() as u64 > MAX_INLINE_ITERATIONS {
                        return Err(CompileError::Unsupported(format!(
                            "inline function '{function_name}' loop product exceeds {MAX_INLINE_ITERATIONS} iterations"
                        )));
                    }
                }
            }
            products = next;
        }

        let mut binding_names = BTreeSet::new();
        Self::collect_inline_binding_names(&statement.body, &mut binding_names);
        let mut aggregate_values: HashMap<String, Vec<Expr>> = HashMap::new();

        for loop_values in products {
            let mut local = substitutions.clone();
            for (name, value) in loop_values {
                local.insert(name, value);
            }
            self.apply_inline_binding_block(&statement.body, &mut local, function_name)?;
            for name in &binding_names {
                let Some(value) = local.get(name) else {
                    continue;
                };
                if self.expression_produces_entity(value) {
                    aggregate_values
                        .entry(name.clone())
                        .or_default()
                        .push(value.clone());
                }
            }
        }

        if aggregate_values.is_empty() {
            return Err(CompileError::Unsupported(format!(
                "inline function '{function_name}' loop must produce an entity binding"
            )));
        }

        for (name, values) in aggregate_values {
            let mut values = values.into_iter();
            let Some(first) = values.next() else {
                continue;
            };
            let aggregate = values.fold(first, |left, right| {
                Expr::Binary(BinaryExpr {
                    left: Box::new(left),
                    op: BinaryOp::CsgUnion,
                    right: Box::new(right),
                    span: statement.span,
                })
            });
            substitutions.insert(name, aggregate);
        }
        Ok(())
    }

    fn collect_inline_binding_names(block: &Block, names: &mut BTreeSet<String>) {
        for statement in &block.statements {
            match statement {
                Statement::Let(binding) => {
                    names.insert(binding.name.name.clone());
                }
                Statement::If(statement) => {
                    Self::collect_inline_binding_names(&statement.then_block, names);
                    match statement.else_block.as_ref() {
                        Some(ElseBranch::Else(block)) => {
                            Self::collect_inline_binding_names(block, names);
                        }
                        Some(ElseBranch::ElseIf(next)) => {
                            Self::collect_inline_binding_names_from_if(next, names);
                        }
                        None => {}
                    }
                }
                Statement::For(statement) => {
                    Self::collect_inline_binding_names(&statement.body, names);
                }
                Statement::Expr(_) | Statement::KeyValue(_) | Statement::Group(_) => {}
            }
        }
    }

    fn collect_inline_binding_names_from_if(statement: &IfStatement, names: &mut BTreeSet<String>) {
        Self::collect_inline_binding_names(&statement.then_block, names);
        match statement.else_block.as_ref() {
            Some(ElseBranch::Else(block)) => Self::collect_inline_binding_names(block, names),
            Some(ElseBranch::ElseIf(next)) => {
                Self::collect_inline_binding_names_from_if(next, names)
            }
            None => {}
        }
    }

    fn inline_for_value(
        &mut self,
        statement: &ForStatement,
        substitutions: &HashMap<String, Expr>,
        function_name: &str,
    ) -> Result<Expr, CompileError> {
        const MAX_INLINE_ITERATIONS: u64 = 1_000_000;
        let mut products = vec![HashMap::<String, Expr>::new()];

        for clause in &statement.clauses {
            let mut next = Vec::new();
            for values in products {
                let mut range_substitutions = substitutions.clone();
                range_substitutions.extend(values.clone());
                let start_expr = substitute_expr(&clause.range.start, &range_substitutions);
                let end_expr = substitute_expr(&clause.range.end, &range_substitutions);
                let start = self.const_int(&start_expr, &HashMap::new())?;
                let mut end = self.const_int(&end_expr, &HashMap::new())?;
                if clause.range.inclusive {
                    end = end.checked_add(1).ok_or_else(|| {
                        CompileError::Unsupported(
                            "inclusive inline-function range exceeds i64".into(),
                        )
                    })?;
                }
                let count = end.saturating_sub(start) as u64;
                if count > MAX_INLINE_ITERATIONS {
                    return Err(CompileError::Unsupported(format!(
                        "inline function '{function_name}' loop exceeds {MAX_INLINE_ITERATIONS} iterations"
                    )));
                }
                for value in start..end {
                    let mut expanded = values.clone();
                    expanded.insert(
                        clause.var.name.clone(),
                        Expr::literal(Literal::Int(value), clause.var.span),
                    );
                    next.push(expanded);
                    if next.len() as u64 > MAX_INLINE_ITERATIONS {
                        return Err(CompileError::Unsupported(format!(
                            "inline function '{function_name}' loop product exceeds {MAX_INLINE_ITERATIONS} iterations"
                        )));
                    }
                }
            }
            products = next;
        }

        let mut values = Vec::with_capacity(products.len());
        for loop_values in products {
            let mut local = substitutions.clone();
            for (name, value) in loop_values {
                local.insert(name, value);
            }
            values.push(self.inline_block_value(&statement.body, &local, function_name)?);
        }

        let mut values = values.into_iter();
        let Some(first) = values.next() else {
            return Err(CompileError::Unsupported(format!(
                "inline function '{function_name}' loop has no iterations"
            )));
        };
        values.try_fold(first, |left, right| {
            Ok(Expr::Binary(BinaryExpr {
                left: Box::new(left),
                op: BinaryOp::CsgUnion,
                right: Box::new(right),
                span: statement.span,
            }))
        })
    }

    fn expression_produces_entity(&mut self, expression: &Expr) -> bool {
        match expression {
            Expr::Ident(identifier) => self.resolve_entity_name(&identifier.name).is_ok(),
            Expr::FieldAccess(_) => entity_path(expression)
                .is_some_and(|name| self.entities.contains_key(&self.resolve_entity_path(&name))),
            Expr::Constructor(_) | Expr::Pipe(_) | Expr::If(_) => true,
            Expr::Binary(binary) if is_explicit_csg_operator(binary.op) => true,
            Expr::Binary(binary) if binary.op == BinaryOp::Sub => {
                self.expression_produces_entity(&binary.left)
                    || self.expression_produces_entity(&binary.right)
            }
            Expr::Call(call) => {
                let Some(function) = self.functions.get(&call.function.name) else {
                    return false;
                };
                if let Some(return_type) = function.return_type.as_ref() {
                    matches!(
                        return_type,
                        TypeAnnotation::Named(named)
                            if matches!(
                                named.name.as_str(),
                                "Point" | "Curve" | "Surface" | "Solid" | "FaceSet" | "EdgeSet"
                            )
                    )
                } else {
                    self.inline_call(call)
                        .is_ok_and(|inlined| self.expression_produces_entity(&inlined))
                }
            }
            Expr::Paren(paren) => self.expression_produces_entity(&paren.value),
            Expr::Array(array) => {
                !array.elements.is_empty()
                    && array
                        .elements
                        .iter()
                        .all(|element| self.expression_produces_entity(element))
            }
            _ => false,
        }
    }

    fn build_sensitivity_manifest(&mut self) {
        self.ir.sensitivity.entries = self
            .ir
            .optimization
            .variables
            .iter()
            .map(|variable| {
                // Internal lowering nodes are part of the dependency graph,
                // even though they are not exposed in the public manifest.
                // A parameter is often attached to an internal primitive
                // which is subsequently consumed by a public CSG/transform
                // result.  Skipping those nodes loses the dependency before
                // it can reach the public result.
                let mut affected_entities = BTreeSet::new();
                let mut parameters = BTreeSet::new();

                collect_manifest_metadata_dependencies(
                    &self.ir,
                    variable.id,
                    &mut parameters,
                    &mut affected_entities,
                );

                for node in &self.ir.entities {
                    let mut node_parameters = BTreeSet::new();
                    for argument in &node.arguments {
                        collect_ir_value_parameters(&argument.value, &mut node_parameters);
                    }
                    if node_parameters.contains(&variable.id) {
                        affected_entities.insert(node.id);
                        parameters.extend(node_parameters);
                    }
                }

                // Dynamic branch predicates are dependencies of every entity
                // in their branch, even when the branch geometry itself has
                // only constant arguments.
                for control in &self.ir.controls {
                    for branch in &control.branches {
                        let mut branch_parameters = BTreeSet::new();
                        if let Some(condition) = &branch.condition {
                            collect_ir_value_parameters(condition, &mut branch_parameters);
                        }
                        if branch_parameters.contains(&variable.id) {
                            affected_entities.extend(branch.entities.iter().copied());
                            parameters.extend(branch_parameters);
                        }
                    }
                }

                // Geometry operations inherit sensitivity through their input
                // entities, even when their own argument list is constant.
                let mut changed = true;
                while changed {
                    changed = false;
                    for node in &self.ir.entities {
                        if affected_entities.contains(&node.id) {
                            continue;
                        }
                        if node
                            .inputs
                            .iter()
                            .any(|input| affected_entities.contains(input))
                        {
                            affected_entities.insert(node.id);
                            changed = true;
                            for argument in &node.arguments {
                                let mut node_parameters = BTreeSet::new();
                                collect_ir_value_parameters(&argument.value, &mut node_parameters);
                                parameters.extend(node_parameters);
                            }
                        }
                    }
                }

                // Keep the dependency traversal internal, but expose only
                // stable public geometry bindings to downstream consumers.
                let public_entities = self
                    .ir
                    .entities
                    .iter()
                    .filter(|node| {
                        affected_entities.contains(&node.id)
                            && !node
                                .name
                                .split('.')
                                .any(|component| component.starts_with("__ir_"))
                    })
                    .map(|node| node.id)
                    .collect::<BTreeSet<_>>();
                let physical_groups = self
                    .ir
                    .physical_groups
                    .iter()
                    .filter(|group| {
                        group
                            .entities
                            .iter()
                            .any(|entity| affected_entities.contains(entity))
                    })
                    .map(|group| group.id)
                    .collect::<Vec<_>>();
                let topology_dependent = matches!(
                    variable.kind,
                    IrDesignVariableKind::Topology | IrDesignVariableKind::LevelSet
                );
                SensitivityEntry {
                    variable: variable.id,
                    parameters: parameters.into_iter().collect(),
                    entities: public_entities.into_iter().collect(),
                    physical_groups,
                    remesh: if topology_dependent {
                        RemeshResponsibility::TopologyDependent
                    } else {
                        RemeshResponsibility::Global
                    },
                    state_transfer: if topology_dependent {
                        StateTransferResponsibility::Conservative
                    } else {
                        StateTransferResponsibility::Interpolate
                    },
                }
            })
            .collect();
    }

    fn provenance<N: AstNode>(&self, node: &N) -> Vec<SourceReference> {
        let node_id = node.node_id(&self.file.metadata.identity);
        let mut references = Vec::new();
        for record in self.file.provenance.for_node(node_id) {
            references.extend(record.sources.iter().cloned());
        }
        if references.is_empty() {
            references.push(SourceReference {
                source: self.file.metadata.identity.id,
                node: node_id,
                span: node.span(),
            });
        }
        references
    }

    fn intern_symbol(&mut self, name: &str) -> SymbolId {
        if let Some(id) = self.symbols.get(name) {
            return *id;
        }
        let hash = StructuralHash::from_bytes(name.as_bytes());
        let mut bytes = [0; 8];
        bytes.copy_from_slice(&hash.0[..8]);
        let id = SymbolId(u64::from_le_bytes(bytes));
        self.symbols.insert(name.to_owned(), id);
        id
    }

    fn qualified_name(&self, local: &str) -> String {
        self.current_geometry.as_ref().map_or_else(
            || local.to_owned(),
            |geometry| {
                if self.branch_path.is_empty() {
                    return format!("{geometry}.{local}");
                }
                let branch_prefix = self
                    .branch_path
                    .iter()
                    .map(|(control, branch)| format!("__if_{control}_{branch}"))
                    .collect::<Vec<_>>()
                    .join(".");
                format!("{geometry}.{branch_prefix}.{local}")
            },
        )
    }

    fn loop_binding_name(&self, local: &str) -> String {
        if self.loop_indices.is_empty() {
            return self.qualified_name(local);
        }
        let suffix = self
            .loop_indices
            .iter()
            .map(i64::to_string)
            .collect::<Vec<_>>()
            .join("_");
        self.qualified_name(&format!("{local}_{suffix}"))
    }

    fn resolve_entity_name(&self, name: &str) -> Result<String, CompileError> {
        let qualified = self.qualified_name(name);
        if self.entities.contains_key(&qualified) {
            Ok(qualified)
        } else if self.entities.contains_key(name) {
            Ok(name.to_owned())
        } else {
            Err(CompileError::UndefinedReference {
                name: name.to_owned(),
            })
        }
    }

    fn resolve_entity_path(&self, path: &str) -> String {
        if self.entities.contains_key(path) || self.topology_entities.contains_key(path) {
            return path.to_owned();
        }
        let qualified = self.qualified_name(path);
        if self.entities.contains_key(&qualified) || self.topology_entities.contains_key(&qualified)
        {
            qualified
        } else {
            path.to_owned()
        }
    }

    fn internal_name(&mut self, operation: &str) -> String {
        self.generated_name_counter += 1;
        self.qualified_name(&format!("__ir_{operation}_{}", self.generated_name_counter))
    }
}

fn collect_ir_value_parameters(value: &IrValue, parameters: &mut BTreeSet<SymbolId>) {
    match value {
        IrValue::Parameter(id) => {
            parameters.insert(*id);
        }
        IrValue::Unary { operand, .. } => collect_ir_value_parameters(operand, parameters),
        IrValue::Binary { left, right, .. } => {
            collect_ir_value_parameters(left, parameters);
            collect_ir_value_parameters(right, parameters);
        }
        IrValue::Vector(values)
        | IrValue::Call {
            arguments: values, ..
        } => {
            for value in values {
                collect_ir_value_parameters(value, parameters);
            }
        }
        IrValue::Index { collection, index } => {
            collect_ir_value_parameters(collection, parameters);
            collect_ir_value_parameters(index, parameters);
        }
        IrValue::Field { object, .. } => collect_ir_value_parameters(object, parameters),
        IrValue::Integer(_)
        | IrValue::Scalar { .. }
        | IrValue::Boolean(_)
        | IrValue::Text(_)
        | IrValue::Entity(_) => {}
    }
}

fn collect_manifest_metadata_dependencies(
    ir: &CanonicalGeometryIr,
    variable: SymbolId,
    parameters: &mut BTreeSet<SymbolId>,
    affected_entities: &mut BTreeSet<EntityId>,
) {
    let public_entities = ir
        .entities
        .iter()
        .filter(|node| {
            !node
                .name
                .split('.')
                .any(|component| component.starts_with("__ir_"))
        })
        .map(|node| node.id)
        .collect::<Vec<_>>();

    for option in &ir.mesh.options {
        collect_manifest_value_dependency(
            &option.value,
            variable,
            &public_entities,
            parameters,
            affected_entities,
        );
    }
    for metric in &ir.mesh.metrics {
        let entities = &metric.target;
        match &metric.tensor {
            IrMetricTensor::Isotropic(_) | IrMetricTensor::Anisotropic(_) => {}
            IrMetricTensor::Hessian { field, .. } => {
                collect_manifest_value_dependency(
                    field,
                    variable,
                    entities,
                    parameters,
                    affected_entities,
                );
            }
        }
    }
    for refinement in &ir.mesh.refinements {
        let entities = match &refinement.kind {
            IrRefineKind::Near { entity, .. }
            | IrRefineKind::On { entity }
            | IrRefineKind::In { region: entity } => std::slice::from_ref(entity),
            IrRefineKind::Field { .. } => public_entities.as_slice(),
        };
        match &refinement.kind {
            IrRefineKind::Near { args, .. } | IrRefineKind::Field { args } => {
                collect_manifest_property_dependencies(
                    args,
                    variable,
                    entities,
                    parameters,
                    affected_entities,
                );
            }
            IrRefineKind::On { .. } | IrRefineKind::In { .. } => {}
        }
        collect_manifest_property_dependencies(
            &refinement.body,
            variable,
            entities,
            parameters,
            affected_entities,
        );
    }
    for transfinite in &ir.mesh.transfinite {
        match &transfinite.kind {
            IrTransfiniteKind::Curve { entity, settings }
            | IrTransfiniteKind::Surface { entity, settings }
            | IrTransfiniteKind::Volume { entity, settings } => {
                collect_manifest_property_dependencies(
                    settings,
                    variable,
                    std::slice::from_ref(entity),
                    parameters,
                    affected_entities,
                );
            }
        }
    }
    for boundary in &ir.boundaries {
        collect_manifest_property_dependencies(
            &boundary.settings,
            variable,
            &boundary.entities,
            parameters,
            affected_entities,
        );
    }
    for region in &ir.material_regions {
        collect_manifest_property_dependencies(
            &region.properties,
            variable,
            &region.entities,
            parameters,
            affected_entities,
        );
    }
    for field in &ir.fields {
        match &field.kind {
            IrGeometryFieldKind::Topology {
                region,
                initial_density,
                filter_radius,
            } => {
                collect_manifest_value_dependency(
                    initial_density,
                    variable,
                    region,
                    parameters,
                    affected_entities,
                );
                if let Some(filter_radius) = filter_radius {
                    collect_manifest_value_dependency(
                        filter_radius,
                        variable,
                        region,
                        parameters,
                        affected_entities,
                    );
                }
            }
            IrGeometryFieldKind::LevelSet { field, iso_value } => {
                collect_manifest_value_dependency(
                    field,
                    variable,
                    &public_entities,
                    parameters,
                    affected_entities,
                );
                collect_manifest_value_dependency(
                    iso_value,
                    variable,
                    &public_entities,
                    parameters,
                    affected_entities,
                );
            }
            IrGeometryFieldKind::Projected {
                geometry,
                tolerance,
                ..
            } => {
                if let Some(tolerance) = tolerance {
                    collect_manifest_value_dependency(
                        tolerance,
                        variable,
                        geometry,
                        parameters,
                        affected_entities,
                    );
                }
            }
        }
    }
    for objective in &ir.optimization.objectives {
        collect_manifest_value_dependency(
            &objective.value,
            variable,
            &public_entities,
            parameters,
            affected_entities,
        );
    }
}

fn collect_manifest_property_dependencies(
    properties: &[IrProperty],
    variable: SymbolId,
    entities: &[EntityId],
    parameters: &mut BTreeSet<SymbolId>,
    affected_entities: &mut BTreeSet<EntityId>,
) {
    for property in properties {
        collect_manifest_value_dependency(
            &property.value,
            variable,
            entities,
            parameters,
            affected_entities,
        );
    }
}

fn collect_manifest_value_dependency(
    value: &IrValue,
    variable: SymbolId,
    entities: &[EntityId],
    parameters: &mut BTreeSet<SymbolId>,
    affected_entities: &mut BTreeSet<EntityId>,
) {
    let mut dependencies = BTreeSet::new();
    collect_ir_value_parameters(value, &mut dependencies);
    if dependencies.contains(&variable) {
        parameters.extend(dependencies);
        affected_entities.extend(entities.iter().copied());
    }
}

fn primitive_kind(name: &str) -> Result<(PrimitiveKind, EntityDimension), CompileError> {
    Ok(match name {
        "Point" => (PrimitiveKind::Point, EntityDimension::Point),
        "Line" => (PrimitiveKind::Line, EntityDimension::Curve),
        "Arc" => (PrimitiveKind::Arc, EntityDimension::Curve),
        "Spline" => (PrimitiveKind::Spline, EntityDimension::Curve),
        "BSpline" => (PrimitiveKind::BSpline, EntityDimension::Curve),
        "Polygon" => (PrimitiveKind::Polygon, EntityDimension::Surface),
        "Rectangle" => (PrimitiveKind::Rectangle, EntityDimension::Surface),
        "Disk" | "Circle" => (PrimitiveKind::Disk, EntityDimension::Surface),
        "Box" => (PrimitiveKind::Box, EntityDimension::Volume),
        "Sphere" => (PrimitiveKind::Sphere, EntityDimension::Volume),
        "Cylinder" => (PrimitiveKind::Cylinder, EntityDimension::Volume),
        "Cone" => (PrimitiveKind::Cone, EntityDimension::Volume),
        "Torus" => (PrimitiveKind::Torus, EntityDimension::Volume),
        "Wedge" => (PrimitiveKind::Wedge, EntityDimension::Volume),
        "Loft" => (PrimitiveKind::Loft, EntityDimension::Volume),
        other => {
            return Err(CompileError::Unsupported(format!(
                "shape constructor '{other}' in canonical IR"
            )));
        }
    })
}

fn template_port_dimension(port: &ParamDef) -> Result<EntityDimension, CompileError> {
    let meshlang_ast::types::TypeAnnotation::Named(annotation) = &port.type_ann else {
        return Err(CompileError::TypeError {
            message: format!(
                "geometry template port '{}' must use a concrete geometry type",
                port.name.name
            ),
        });
    };
    match annotation.name.as_str() {
        "Point" => Ok(EntityDimension::Point),
        "Curve" | "EdgeSet" => Ok(EntityDimension::Curve),
        "Surface" | "FaceSet" => Ok(EntityDimension::Surface),
        "Solid" => Ok(EntityDimension::Volume),
        other => Err(CompileError::TypeError {
            message: format!(
                "geometry template port '{}' has non-geometry type '{}'",
                port.name.name, other
            ),
        }),
    }
}

fn pipe_operation(
    name: &str,
    input: EntityDimension,
) -> Result<(GeometryOperation, EntityDimension), CompileError> {
    Ok(match name {
        "translate" => (
            GeometryOperation::Transform {
                kind: TransformOperation::Translate,
            },
            input,
        ),
        "rotate" => (
            GeometryOperation::Transform {
                kind: TransformOperation::Rotate,
            },
            input,
        ),
        "scale" => (
            GeometryOperation::Transform {
                kind: TransformOperation::Scale,
            },
            input,
        ),
        "mirror" => (
            GeometryOperation::Transform {
                kind: TransformOperation::Mirror,
            },
            input,
        ),
        "affine" => (
            GeometryOperation::Transform {
                kind: TransformOperation::Affine,
            },
            input,
        ),
        "extrude" => (
            GeometryOperation::Sweep {
                kind: SweepOperation::Extrude,
            },
            EntityDimension::Volume,
        ),
        "revolve" => (
            GeometryOperation::Sweep {
                kind: SweepOperation::Revolve,
            },
            EntityDimension::Volume,
        ),
        "sweep" => (
            GeometryOperation::Sweep {
                kind: SweepOperation::Pipe,
            },
            EntityDimension::Volume,
        ),
        "copy" => (GeometryOperation::Copy, input),
        "fragment" => (GeometryOperation::Fragment, input),
        "shell" => (GeometryOperation::Shell, input),
        "twist" => (GeometryOperation::Twist, EntityDimension::Volume),
        "coherence" => (GeometryOperation::Coherence, input),
        "fillet" => (GeometryOperation::Fillet, input),
        "chamfer" => (GeometryOperation::Chamfer, input),
        "array" => (GeometryOperation::Array, input),
        "project_iso_surface" => (
            GeometryOperation::Project {
                method: ProjectionMethod::IsoSurface,
            },
            input,
        ),
        "project_signed_distance" => (
            GeometryOperation::Project {
                method: ProjectionMethod::SignedDistance,
            },
            input,
        ),
        "project_closest_point" => (
            GeometryOperation::Project {
                method: ProjectionMethod::ClosestPoint,
            },
            input,
        ),
        other => {
            return Err(CompileError::Unsupported(format!(
                "geometry operation '{other}' in canonical IR"
            )));
        }
    })
}

fn constructor_argument_name(
    constructor: &str,
    argument: &Argument,
    index: usize,
) -> IrArgumentName {
    let name = argument.name.as_ref().map(|name| name.name.as_str());
    match name {
        Some("origin") => IrArgumentName::Origin,
        Some("from" | "start") => IrArgumentName::Start,
        Some("to" | "end") => IrArgumentName::End,
        Some("center") => IrArgumentName::Center,
        Some("points") => IrArgumentName::Points,
        Some("sections") => IrArgumentName::Sections,
        Some("ruled") => IrArgumentName::Ruled,
        Some("r") => IrArgumentName::Radius,
        Some("r1") => IrArgumentName::RadiusStart,
        Some("r2") => IrArgumentName::RadiusEnd,
        Some("r_major") => IrArgumentName::MajorRadius,
        Some("r_minor") => IrArgumentName::MinorRadius,
        Some("h" | "height") => IrArgumentName::Height,
        _ => match constructor {
            "Point" => IrArgumentName::Origin,
            "Line" => [IrArgumentName::Start, IrArgumentName::End][index.min(1)],
            "Arc" => [
                IrArgumentName::Start,
                IrArgumentName::Center,
                IrArgumentName::End,
            ][index.min(2)],
            "Spline" | "BSpline" | "Polygon" => IrArgumentName::Points,
            "Loft" => IrArgumentName::Sections,
            "Sphere" | "Disk" | "Circle" => IrArgumentName::Radius,
            "Cylinder" => [IrArgumentName::Radius, IrArgumentName::Height][index.min(1)],
            "Cone" => [
                IrArgumentName::RadiusStart,
                IrArgumentName::RadiusEnd,
                IrArgumentName::Height,
            ][index.min(2)],
            "Torus" => [IrArgumentName::MajorRadius, IrArgumentName::MinorRadius][index.min(1)],
            _ => IrArgumentName::Size,
        },
    }
}

fn pipe_argument_name(function: &str, argument: &Argument, index: usize) -> IrArgumentName {
    let name = argument.name.as_ref().map(|name| name.name.as_str());
    match name {
        Some("angle") => IrArgumentName::Angle,
        Some("axis") => IrArgumentName::Axis,
        Some("r") => IrArgumentName::Radius,
        Some("factor" | "sx" | "sy" | "sz") => IrArgumentName::Scale,
        Some("count" | "n") => IrArgumentName::Count,
        Some("offset") => IrArgumentName::Offset,
        Some("matrix") => IrArgumentName::Matrix,
        Some("layers") => IrArgumentName::Layers,
        Some("recombine") => IrArgumentName::Recombine,
        Some("tolerance") => IrArgumentName::Tolerance,
        Some("thickness") => IrArgumentName::Thickness,
        Some("d" | "distance") => IrArgumentName::Distance,
        Some("dx" | "dy" | "dz" | "direction") => IrArgumentName::Direction,
        _ => match function {
            "translate" | "extrude" | "twist" => IrArgumentName::Direction,
            "rotate" | "revolve" => {
                if index == 0 {
                    IrArgumentName::Angle
                } else {
                    IrArgumentName::Axis
                }
            }
            "scale" => IrArgumentName::Scale,
            "mirror" => IrArgumentName::Plane,
            "fillet" => IrArgumentName::Radius,
            "chamfer" => IrArgumentName::Distance,
            "shell" => IrArgumentName::Thickness,
            "array" => {
                if index == 0 {
                    IrArgumentName::Count
                } else {
                    IrArgumentName::Offset
                }
            }
            _ => IrArgumentName::Tolerance,
        },
    }
}

fn lower_binary_operator(operator: BinaryOp) -> Result<IrBinaryOperator, CompileError> {
    Ok(match operator {
        BinaryOp::Add => IrBinaryOperator::Add,
        BinaryOp::Sub => IrBinaryOperator::Subtract,
        BinaryOp::Mul => IrBinaryOperator::Multiply,
        BinaryOp::Div => IrBinaryOperator::Divide,
        BinaryOp::Mod => IrBinaryOperator::Modulo,
        BinaryOp::Pow => IrBinaryOperator::Power,
        BinaryOp::Eq => IrBinaryOperator::Equal,
        BinaryOp::Ne => IrBinaryOperator::NotEqual,
        BinaryOp::Lt => IrBinaryOperator::Less,
        BinaryOp::Le => IrBinaryOperator::LessEqual,
        BinaryOp::Gt => IrBinaryOperator::Greater,
        BinaryOp::Ge => IrBinaryOperator::GreaterEqual,
        BinaryOp::And => IrBinaryOperator::LogicalAnd,
        BinaryOp::Or => IrBinaryOperator::LogicalOr,
        _ => {
            return Err(CompileError::TypeError {
                message: "CSG operator cannot be lowered as a scalar value".into(),
            });
        }
    })
}

fn is_explicit_csg_operator(operator: BinaryOp) -> bool {
    matches!(
        operator,
        BinaryOp::CsgUnion | BinaryOp::CsgDifference | BinaryOp::CsgIntersect
    )
}

fn is_entity_csg_operator(operator: BinaryOp) -> bool {
    operator == BinaryOp::Sub || is_explicit_csg_operator(operator)
}

fn flatten_csg_operands<'a>(
    binary: &'a BinaryExpr,
    operation: BinaryOp,
    operands: &mut Vec<&'a Expr>,
) {
    if let Expr::Binary(left) = binary.left.as_ref() {
        if left.op == operation && is_entity_csg_operator(left.op) {
            flatten_csg_operands(left, operation, operands);
        } else {
            operands.push(&binary.left);
        }
    } else {
        operands.push(&binary.left);
    }
    operands.push(&binary.right);
}

fn expression_can_be_entity(expression: &Expr) -> bool {
    matches!(
        expression,
        Expr::Ident(_)
            | Expr::FieldAccess(_)
            | Expr::Constructor(_)
            | Expr::Pipe(_)
            | Expr::Array(_)
            | Expr::If(_)
            | Expr::Paren(_)
    ) || matches!(expression, Expr::Binary(binary) if is_entity_csg_operator(binary.op))
        || matches!(expression, Expr::Call(_))
}

fn entity_path(expression: &Expr) -> Option<String> {
    match expression {
        Expr::Ident(identifier) => Some(identifier.name.clone()),
        Expr::FieldAccess(field) => Some(format!(
            "{}.{}",
            entity_path(&field.object)?,
            field.field.name
        )),
        Expr::Paren(paren) => entity_path(&paren.value),
        _ => None,
    }
}

fn topology_selector(path: &str) -> Option<(&str, EntityDimension, Option<String>)> {
    for (marker, dimension) in [
        (".faces", EntityDimension::Surface),
        (".edges", EntityDimension::Curve),
        (".vertices", EntityDimension::Point),
    ] {
        if let Some(offset) = path.find(marker) {
            let owner = &path[..offset];
            let remainder = &path[offset + marker.len()..];
            if remainder.is_empty() || remainder.starts_with('.') {
                return Some((
                    owner,
                    dimension,
                    remainder.strip_prefix('.').map(str::to_owned),
                ));
            }
        }
    }
    None
}

fn common_dimension(inputs: &[EntityId]) -> Result<EntityDimension, CompileError> {
    let dimension = inputs
        .first()
        .map(|entity| entity.dimension)
        .ok_or_else(|| CompileError::TypeError {
            message: "geometry operation requires at least one input".into(),
        })?;
    if inputs.iter().any(|entity| entity.dimension != dimension) {
        return Err(CompileError::TypeError {
            message: "geometry operation inputs must have identical dimensions".into(),
        });
    }
    Ok(dimension)
}

fn ensure_dimension(
    inputs: &[EntityId],
    expected: EntityDimension,
    name: &str,
) -> Result<(), CompileError> {
    if inputs.iter().any(|entity| entity.dimension != expected) {
        return Err(CompileError::TypeError {
            message: format!("physical group '{name}' contains an entity with the wrong dimension"),
        });
    }
    Ok(())
}

fn value_dimension(value: &IrValue) -> units::Dimension {
    match value {
        IrValue::Scalar { dimension, .. } => *dimension,
        _ => units::Dimension::DIMENSIONLESS,
    }
}

fn apply_unit_scale(
    value: IrValue,
    scale: f64,
    dimension: units::Dimension,
) -> Result<IrValue, CompileError> {
    match value {
        IrValue::Integer(value) => Ok(IrValue::Scalar {
            value: value as f64 * scale,
            dimension,
        }),
        IrValue::Scalar { value, .. } => Ok(IrValue::Scalar {
            value: value * scale,
            dimension,
        }),
        _ => Err(CompileError::TypeError {
            message: "units can only be attached to scalar parameters".into(),
        }),
    }
}

fn required_setting<'a>(
    settings: &'a HashMap<&str, &'a KeyValue>,
    names: &[&str],
) -> Result<&'a KeyValue, CompileError> {
    names
        .iter()
        .find_map(|name| settings.get(name).copied())
        .ok_or_else(|| CompileError::TypeError {
            message: format!("missing required mesh setting '{}'", names[0]),
        })
}

fn embed_dim_to_entity_dimension(dimension: EmbedDim) -> EntityDimension {
    match dimension {
        EmbedDim::Point => EntityDimension::Point,
        EmbedDim::Curve => EntityDimension::Curve,
        EmbedDim::Surface => EntityDimension::Surface,
        EmbedDim::Volume => EntityDimension::Volume,
    }
}

fn const_int_expression(
    expression: &Expr,
    loop_values: &HashMap<String, i64>,
    values: &HashMap<String, IrValue>,
) -> Result<i64, CompileError> {
    let overflow = || CompileError::Unsupported("compile-time integer overflow".into());
    match expression {
        Expr::Literal(LiteralExpr {
            value: Literal::Int(value),
            ..
        }) => Ok(*value),
        Expr::Ident(identifier) => loop_values
            .get(&identifier.name)
            .copied()
            .or_else(|| values.get(&identifier.name).and_then(const_ir_int))
            .ok_or_else(|| {
                CompileError::Unsupported("loop range must be a compile-time integer".into())
            }),
        Expr::Paren(paren) => const_int_expression(&paren.value, loop_values, values),
        Expr::Unary(unary) if unary.op == UnaryOp::Neg => {
            const_int_expression(&unary.operand, loop_values, values)?
                .checked_neg()
                .ok_or_else(overflow)
        }
        Expr::Binary(binary) => {
            let left = const_int_expression(&binary.left, loop_values, values)?;
            let right = const_int_expression(&binary.right, loop_values, values)?;
            match binary.op {
                BinaryOp::Add => left.checked_add(right).ok_or_else(overflow),
                BinaryOp::Sub => left.checked_sub(right).ok_or_else(overflow),
                BinaryOp::Mul => left.checked_mul(right).ok_or_else(overflow),
                BinaryOp::Div => left.checked_div(right).ok_or_else(overflow),
                BinaryOp::Mod => left.checked_rem(right).ok_or_else(overflow),
                BinaryOp::Pow => {
                    let exponent = u32::try_from(right).map_err(|_| {
                        CompileError::Unsupported(
                            "compile-time integer exponent must be non-negative".into(),
                        )
                    })?;
                    left.checked_pow(exponent).ok_or_else(overflow)
                }
                _ => Err(CompileError::Unsupported(
                    "loop range must be a compile-time integer".into(),
                )),
            }
        }
        _ => Err(CompileError::Unsupported(
            "loop range must be a compile-time integer".into(),
        )),
    }
}

fn const_ir_int(value: &IrValue) -> Option<i64> {
    match value {
        IrValue::Integer(value) => Some(*value),
        IrValue::Unary {
            op: IrUnaryOperator::Negate,
            operand,
        } => const_ir_int(operand)?.checked_neg(),
        IrValue::Binary { op, left, right } => {
            let left = const_ir_int(left)?;
            let right = const_ir_int(right)?;
            match op {
                IrBinaryOperator::Add => left.checked_add(right),
                IrBinaryOperator::Subtract => left.checked_sub(right),
                IrBinaryOperator::Multiply => left.checked_mul(right),
                IrBinaryOperator::Divide => left.checked_div(right),
                IrBinaryOperator::Modulo => left.checked_rem(right),
                IrBinaryOperator::Power => {
                    let exponent = u32::try_from(right).ok()?;
                    left.checked_pow(exponent)
                }
                _ => None,
            }
        }
        _ => None,
    }
}
/// Evaluate a scalar expression that is independent of runtime geometry and
/// parameters. This is deliberately small: it is only used to select the
/// branch of an inline function before the body is lowered to one expression.
fn inline_const_scalar(expression: &Expr) -> Option<f64> {
    match expression {
        Expr::Literal(LiteralExpr {
            value: Literal::Int(value),
            ..
        }) => Some(*value as f64),
        Expr::Literal(LiteralExpr {
            value: Literal::Float(value),
            ..
        }) => Some(*value),
        Expr::Paren(paren) => inline_const_scalar(&paren.value),
        Expr::Unary(unary) if unary.op == UnaryOp::Neg => {
            Some(-inline_const_scalar(&unary.operand)?)
        }
        Expr::Binary(binary) => {
            let left = inline_const_scalar(&binary.left)?;
            let right = inline_const_scalar(&binary.right)?;
            let value = match binary.op {
                BinaryOp::Add => left + right,
                BinaryOp::Sub => left - right,
                BinaryOp::Mul => left * right,
                BinaryOp::Div if right != 0.0 => left / right,
                BinaryOp::Mod if right != 0.0 => left % right,
                BinaryOp::Pow => left.powf(right),
                _ => return None,
            };
            value.is_finite().then_some(value)
        }
        _ => None,
    }
}

fn inline_const_bool(expression: &Expr) -> Option<bool> {
    match expression {
        Expr::Literal(LiteralExpr {
            value: Literal::Bool(value),
            ..
        }) => Some(*value),
        Expr::Paren(paren) => inline_const_bool(&paren.value),
        Expr::Unary(unary) if unary.op == UnaryOp::Not => Some(!inline_const_bool(&unary.operand)?),
        Expr::Binary(binary) => match binary.op {
            BinaryOp::And => {
                Some(inline_const_bool(&binary.left)? && inline_const_bool(&binary.right)?)
            }
            BinaryOp::Or => {
                Some(inline_const_bool(&binary.left)? || inline_const_bool(&binary.right)?)
            }
            BinaryOp::Eq | BinaryOp::Ne => {
                if let (Some(left), Some(right)) = (
                    inline_const_bool(&binary.left),
                    inline_const_bool(&binary.right),
                ) {
                    Some(if binary.op == BinaryOp::Eq {
                        left == right
                    } else {
                        left != right
                    })
                } else {
                    let left = inline_const_scalar(&binary.left)?;
                    let right = inline_const_scalar(&binary.right)?;
                    Some(if binary.op == BinaryOp::Eq {
                        left == right
                    } else {
                        left != right
                    })
                }
            }
            BinaryOp::Lt | BinaryOp::Gt | BinaryOp::Le | BinaryOp::Ge => {
                let left = inline_const_scalar(&binary.left)?;
                let right = inline_const_scalar(&binary.right)?;
                Some(match binary.op {
                    BinaryOp::Lt => left < right,
                    BinaryOp::Gt => left > right,
                    BinaryOp::Le => left <= right,
                    BinaryOp::Ge => left >= right,
                    _ => unreachable!("comparison operator matched above"),
                })
            }
            _ => None,
        },
        _ => None,
    }
}

fn const_bool_value(value: &IrValue) -> Option<bool> {
    match value {
        IrValue::Boolean(value) => Some(*value),
        IrValue::Unary {
            op: IrUnaryOperator::LogicalNot,
            operand,
        } => Some(!const_bool_value(operand)?),
        IrValue::Binary { op, left, right } => match op {
            IrBinaryOperator::LogicalAnd => {
                Some(const_bool_value(left)? && const_bool_value(right)?)
            }
            IrBinaryOperator::LogicalOr => {
                Some(const_bool_value(left)? || const_bool_value(right)?)
            }
            IrBinaryOperator::Equal
            | IrBinaryOperator::NotEqual
            | IrBinaryOperator::Less
            | IrBinaryOperator::LessEqual
            | IrBinaryOperator::Greater
            | IrBinaryOperator::GreaterEqual => {
                let left = const_scalar_value(left)?;
                let right = const_scalar_value(right)?;
                Some(match op {
                    IrBinaryOperator::Equal => left == right,
                    IrBinaryOperator::NotEqual => left != right,
                    IrBinaryOperator::Less => left < right,
                    IrBinaryOperator::LessEqual => left <= right,
                    IrBinaryOperator::Greater => left > right,
                    IrBinaryOperator::GreaterEqual => left >= right,
                    _ => unreachable!("comparison operator matched above"),
                })
            }
            _ => None,
        },
        _ => None,
    }
}

fn const_scalar_value(value: &IrValue) -> Option<f64> {
    match value {
        IrValue::Integer(value) => Some(*value as f64),
        IrValue::Scalar { value, .. } => Some(*value),
        IrValue::Unary {
            op: IrUnaryOperator::Negate,
            operand,
        } => Some(-const_scalar_value(operand)?),
        IrValue::Binary { op, left, right } => {
            let left = const_scalar_value(left)?;
            let right = const_scalar_value(right)?;
            match op {
                IrBinaryOperator::Add => Some(left + right),
                IrBinaryOperator::Subtract => Some(left - right),
                IrBinaryOperator::Multiply => Some(left * right),
                IrBinaryOperator::Divide if right != 0.0 => Some(left / right),
                IrBinaryOperator::Modulo if right != 0.0 => Some(left % right),
                IrBinaryOperator::Power => Some(left.powf(right)),
                _ => None,
            }
        }
        _ => None,
    }
}

fn normalize(vector: [f64; 3]) -> Result<[f64; 3], CompileError> {
    let norm = vector.iter().map(|value| value * value).sum::<f64>().sqrt();
    if norm <= f64::EPSILON {
        return Err(CompileError::TypeError {
            message: "coordinate-system axis must be non-zero".into(),
        });
    }
    Ok([vector[0] / norm, vector[1] / norm, vector[2] / norm])
}

fn cross(left: [f64; 3], right: [f64; 3]) -> [f64; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [f64; 3], right: [f64; 3]) -> f64 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn ensure_orthogonal(first: [f64; 3], second: [f64; 3], system: &str) -> Result<(), CompileError> {
    if dot(first, second).abs() > 1e-10 {
        return Err(CompileError::TypeError {
            message: format!("coordinate system '{system}' axes must be orthogonal"),
        });
    }
    Ok(())
}

fn substitute_inline_if(
    statement: &IfStatement,
    substitutions: &HashMap<String, Expr>,
) -> IfStatement {
    IfStatement {
        condition: substitute_expr(&statement.condition, substitutions),
        then_block: substitute_inline_block(&statement.then_block, substitutions),
        else_block: statement.else_block.as_ref().map(|branch| match branch {
            ElseBranch::Else(block) => {
                ElseBranch::Else(substitute_inline_block(block, substitutions))
            }
            ElseBranch::ElseIf(statement) => {
                ElseBranch::ElseIf(Box::new(substitute_inline_if(statement, substitutions)))
            }
        }),
        span: statement.span,
    }
}

fn substitute_inline_block(block: &Block, substitutions: &HashMap<String, Expr>) -> Block {
    Block {
        statements: block
            .statements
            .iter()
            .map(|statement| match statement {
                Statement::Let(binding) => Statement::Let(LetStatement {
                    name: binding.name.clone(),
                    type_ann: binding.type_ann.clone(),
                    value: substitute_expr(&binding.value, substitutions),
                    span: binding.span,
                }),
                Statement::Expr(expression) => {
                    Statement::Expr(substitute_expr(expression, substitutions))
                }
                Statement::For(statement) => Statement::For(ForStatement {
                    clauses: statement
                        .clauses
                        .iter()
                        .map(|clause| ForClause {
                            var: clause.var.clone(),
                            range: RangeExpr {
                                start: Box::new(substitute_expr(
                                    &clause.range.start,
                                    substitutions,
                                )),
                                end: Box::new(substitute_expr(&clause.range.end, substitutions)),
                                inclusive: clause.range.inclusive,
                                span: clause.range.span,
                            },
                            span: clause.span,
                        })
                        .collect(),
                    body: substitute_inline_block(&statement.body, substitutions),
                    span: statement.span,
                }),
                Statement::If(statement) => {
                    Statement::If(substitute_inline_if(statement, substitutions))
                }
                Statement::KeyValue(value) => Statement::KeyValue(KeyValue {
                    key: value.key.clone(),
                    value: substitute_expr(&value.value, substitutions),
                    unit: value.unit.clone(),
                    span: value.span,
                }),
                Statement::Group(group) => Statement::Group(GroupDecl {
                    name: group.name.clone(),
                    bc_type: group.bc_type.clone(),
                    value: substitute_expr(&group.value, substitutions),
                    span: group.span,
                }),
            })
            .collect(),
        span: block.span,
    }
}

fn substitute_expr(expression: &Expr, substitutions: &HashMap<String, Expr>) -> Expr {
    match expression {
        Expr::Ident(identifier) => substitutions
            .get(&identifier.name)
            .cloned()
            .unwrap_or_else(|| expression.clone()),
        Expr::Literal(_) => expression.clone(),
        Expr::Binary(binary) => Expr::Binary(BinaryExpr {
            left: Box::new(substitute_expr(&binary.left, substitutions)),
            op: binary.op,
            right: Box::new(substitute_expr(&binary.right, substitutions)),
            span: binary.span,
        }),
        Expr::Unary(unary) => Expr::Unary(UnaryExpr {
            op: unary.op,
            operand: Box::new(substitute_expr(&unary.operand, substitutions)),
            span: unary.span,
        }),
        Expr::Pipe(pipe) => Expr::Pipe(PipeExpr {
            value: Box::new(substitute_expr(&pipe.value, substitutions)),
            function: pipe.function.clone(),
            args: pipe
                .args
                .iter()
                .map(|argument| substitute_argument(argument, substitutions))
                .collect(),
            span: pipe.span,
        }),
        Expr::FieldAccess(field) => Expr::FieldAccess(FieldAccessExpr {
            object: Box::new(substitute_expr(&field.object, substitutions)),
            field: field.field.clone(),
            span: field.span,
        }),
        Expr::Index(index) => Expr::Index(IndexExpr {
            object: Box::new(substitute_expr(&index.object, substitutions)),
            index: Box::new(substitute_expr(&index.index, substitutions)),
            span: index.span,
        }),
        Expr::Constructor(constructor) => Expr::Constructor(ConstructorExpr {
            type_name: constructor.type_name.clone(),
            args: constructor
                .args
                .iter()
                .map(|argument| substitute_argument(argument, substitutions))
                .collect(),
            span: constructor.span,
        }),
        Expr::Call(call) => Expr::Call(CallExpr {
            function: call.function.clone(),
            args: call
                .args
                .iter()
                .map(|argument| substitute_argument(argument, substitutions))
                .collect(),
            span: call.span,
        }),
        Expr::Paren(paren) => Expr::Paren(ParenExpr::new(
            substitute_expr(&paren.value, substitutions),
            paren.span,
        )),
        Expr::Array(array) => Expr::Array(ArrayExpr {
            elements: array
                .elements
                .iter()
                .map(|element| substitute_expr(element, substitutions))
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
                    value: substitute_expr(&field.value, substitutions),
                    span: field.span,
                })
                .collect(),
            span: value.span,
        }),
        Expr::If(_) => expression.clone(),
    }
}

fn substitute_argument(argument: &Argument, substitutions: &HashMap<String, Expr>) -> Argument {
    Argument {
        name: argument.name.clone(),
        value: substitute_expr(&argument.value, substitutions),
        span: argument.span,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lowers_named_dimension_qualified_entities_and_boolean_chain() {
        let source = r#"
            geometry Main {
                let panel = Rectangle(2.0, 1.0)
                let hole = Disk(r: 0.2)
                let trimmed = panel - hole
            }
        "#;
        let ast = meshlang_parser::parse(source, "canonical.mlg").unwrap();
        let ir = lower_to_canonical_ir(&ast).unwrap();
        assert_eq!(ir.entities.last().unwrap().name, "Main.trimmed");
        assert_eq!(
            ir.entities.last().unwrap().id.dimension,
            EntityDimension::Surface
        );
        assert!(matches!(
            ir.entities.last().unwrap().operation,
            GeometryOperation::Boolean {
                kind: BooleanOperation::Difference
            }
        ));
        assert!(ir.validate().is_ok());
    }

    #[test]
    fn canonical_ids_and_hash_are_deterministic() {
        let source = "geometry Main { let solid = Box(1.0, 2.0, 3.0) }";
        let ast = meshlang_parser::parse(source, "deterministic.mlg").unwrap();
        let first = lower_to_canonical_ir(&ast).unwrap();
        let second = lower_to_canonical_ir(&ast).unwrap();
        assert_eq!(
            first.entities[0].id,
            EntityId::new(EntityDimension::Volume, 1)
        );
        assert_eq!(
            first.structural_hash().unwrap(),
            second.structural_hash().unwrap()
        );
    }

    #[test]
    fn geometry_scalar_locals_are_values_not_entities_and_do_not_leak() {
        let source = r#"
            geometry First {
                let width = 3.0 - 1.0
                let solid = Box(width, 1.0, 1.0)
            }
            geometry Second {
                let width = 4.0
                let solid = Box(width, 2.0, 2.0)
            }
        "#;
        let ast = meshlang_parser::parse(source, "locals.mlg").unwrap();
        let ir = lower_to_canonical_ir(&ast).unwrap();
        assert_eq!(ir.entities.len(), 2);
        assert_eq!(ir.entities[0].name, "First.solid");
        assert_eq!(ir.entities[1].name, "Second.solid");
    }

    #[test]
    fn canonical_cache_reuses_structurally_identical_ast_and_invalidates_changes() {
        let first = meshlang_parser::parse(
            "geometry Main { let solid = Box(1.0, 1.0, 1.0) }",
            "cached.mlg",
        )
        .unwrap();
        let changed = meshlang_parser::parse(
            "geometry Main { let solid = Box(2.0, 1.0, 1.0) }",
            "cached.mlg",
        )
        .unwrap();
        let mut cache = CanonicalIrCache::default();
        let initial = cache.lower(&first).unwrap();
        let repeated = cache.lower(&first).unwrap();
        let invalidated = cache.lower(&changed).unwrap();

        assert!(Arc::ptr_eq(&initial, &repeated));
        assert!(!Arc::ptr_eq(&initial, &invalidated));
        assert_eq!((cache.len(), cache.hits(), cache.misses()), (2, 1, 2));
    }

    #[test]
    fn canonical_metadata_scales_units_and_builds_orthonormal_coordinates() {
        let source = r#"
            geometry Main { let solid = Box(1.0, 1.0, 1.0) }
            design_region shell {
                kind: design
                region: Main.solid
                annotations: [minimum_thickness(value: 2.0, unit: "mm")]
            }
            coordinate_system local {
                kind: cylindrical
                origin: [0.0, 0.0, 0.0]
                axis: [0.0, 0.0, 2.0]
                reference: [2.0, 0.0, 1.0]
            }
            mesh for Main {
                boundary_layer on(Main.solid.faces) {
                    first_height: 1.0 mm
                    thickness: 5.0 mm
                    layers: 3
                    growth_rate: 1.2
                }
            }
        "#;
        let ast = meshlang_parser::parse(source, "scaled-metadata.mlg").unwrap();
        let ir = lower_to_canonical_ir(&ast).unwrap();
        assert!(matches!(
            ir.optimization.regions[0].annotations[0],
            IrManufacturingAnnotation::MinimumThickness(value)
                if (value - 0.002).abs() < 1e-12
        ));
        let layer = &ir.mesh.boundary_layers[0];
        assert!((layer.first_layer_height - 0.001).abs() < 1e-12);
        assert!((layer.total_thickness - 0.005).abs() < 1e-12);
        assert_eq!(layer.number_of_layers, Some(3));
        let basis = ir.coordinate_systems[0].basis;
        let x = [basis[0], basis[1], basis[2]];
        let y = [basis[3], basis[4], basis[5]];
        let z = [basis[6], basis[7], basis[8]];
        assert!(dot(x, y).abs() < 1e-12);
        assert!(dot(x, z).abs() < 1e-12);
        assert!(dot(y, z).abs() < 1e-12);
    }

    #[test]
    fn canonical_metadata_rejects_nonorthogonal_axes_and_fractional_layers() {
        let nonorthogonal = meshlang_parser::parse(
            r#"coordinate_system bad {
                kind: cartesian
                origin: [0.0, 0.0, 0.0]
                axes: [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            }"#,
            "nonorthogonal.mlg",
        )
        .unwrap();
        assert!(lower_to_canonical_ir(&nonorthogonal)
            .unwrap_err()
            .to_string()
            .contains("axes must be orthogonal"));

        let fractional_layers = meshlang_parser::parse(
            r#"
                geometry Main { let solid = Box(1.0, 1.0, 1.0) }
                mesh for Main {
                    boundary_layer on(Main.solid.faces) {
                        first_height: 0.001
                        layers: 2.5
                    }
                }
            "#,
            "fractional-layers.mlg",
        )
        .unwrap();
        assert!(lower_to_canonical_ir(&fractional_layers)
            .unwrap_err()
            .to_string()
            .contains("positive 32-bit integer"));
    }

    #[test]
    fn canonical_metadata_lowers_and_validates_cyclic_symmetry() {
        let source = r#"
            geometry Main { let solid = Box(1.0, 1.0, 1.0) }
            design_region rotor {
                kind: design
                region: Main.solid
                annotations: [symmetry(kind: cyclic(axis: [0.0, 0.0, 2.0], sectors: 8))]
            }
        "#;
        let ast = meshlang_parser::parse(source, "cyclic.mlg").unwrap();
        let ir = lower_to_canonical_ir(&ast).unwrap();
        assert!(matches!(
            ir.optimization.regions[0].annotations[0],
            IrManufacturingAnnotation::SymmetryCyclic { axis, sectors }
                if axis == [0.0, 0.0, 2.0] && sectors == 8
        ));

        let invalid_axis = meshlang_parser::parse(
            &source.replace("[0.0, 0.0, 2.0]", "[0.0, 0.0, 0.0]"),
            "cyclic-zero-axis.mlg",
        )
        .unwrap();
        assert!(lower_to_canonical_ir(&invalid_axis)
            .unwrap_err()
            .to_string()
            .contains("axis must be finite and non-zero"));

        let invalid_sectors = meshlang_parser::parse(
            &source.replace("sectors: 8", "sectors: 1"),
            "cyclic-one-sector.mlg",
        )
        .unwrap();
        assert!(lower_to_canonical_ir(&invalid_sectors)
            .unwrap_err()
            .to_string()
            .contains("sectors must be an integer >= 2"));
    }

    #[test]
    fn multi_statement_functions_are_canonical_ir_not_backend_only() {
        for (final_statement, fixture) in [
            ("base - ball", "function-final-expression.mlg"),
            ("let result = base - ball", "function-final-binding.mlg"),
        ] {
            let source = format!(
                r#"
fn cutout(base: Solid, r: Float) -> Solid {{
    let ball = Sphere(r: r)
    {final_statement}
}}
geometry Main {{
    let base = Box(1.0, 1.0, 1.0)
    let result = cutout(base, 0.25)
}}
"#
            );
            let ast = meshlang_parser::parse(&source, fixture).unwrap();
            let ir = lower_to_canonical_ir(&ast).unwrap();
            let result = ir
                .entities
                .iter()
                .find(|entity| entity.name == "Main.result")
                .expect("function result must be in canonical IR");
            assert!(matches!(
                result.operation,
                GeometryOperation::Boolean {
                    kind: BooleanOperation::Difference
                }
            ));
            assert_eq!(result.inputs.len(), 2);
            assert!(ir.validate().is_ok());
        }
    }

    #[test]
    fn geometry_if_expression_without_else_is_rejected_by_lowering() {
        let source = r#"
            geometry Main {
                let body = if false {
                    Box(1.0, 1.0, 1.0)
                }
            }
        "#;
        let ast = meshlang_parser::parse(source, "if-expression-no-else.mlg").unwrap();
        let error = lower_to_canonical_ir(&ast).unwrap_err();
        assert!(matches!(
            error,
            CompileError::TypeError { message }
                if message.contains("if-expression requires an else branch")
        ));
    }
}
