//! Code generator: AST → Gmsh Python API script.
//!
//! Generates a standalone Python script using `import gmsh` that can be run
//! as `python output.py` to produce the mesh.

use crate::error::CompileError;
use crate::name_resolution::{self, Scope};
use crate::tag_allocator::GmshDim;
use crate::topology::{CsgOp, TopologyQuery, TopologyTracker};
use meshlang_ast::*;
use std::collections::HashMap;

/// Runtime state for one dynamic `for` loop. Lists hold all dim-tags emitted
/// by the loop, so downstream bindings cannot collapse to the last iteration.
#[derive(Debug, Default)]
struct RuntimeLoopContext {
    id: usize,
    lists: Vec<(GmshDim, String)>,
    source_names: Vec<String>,
}

pub struct PythonCodegen {
    buf: String,
    indent: usize,
    scope: Option<Scope>,
    topo: TopologyTracker,
    /// Current geometry block name (for qualified paths).
    current_geometry: Option<String>,
    /// Maps qualified entity name → Python variable name.
    name_to_var: HashMap<String, String>,
    /// Maps Python variable name → Gmsh entity dimension.
    var_to_dim: HashMap<String, GmshDim>,
    /// Dimensions resolved once by canonical IR, keyed by qualified name.
    canonical_dimensions: HashMap<String, GmshDim>,
    /// Maps a `Polygon` surface variable name → the Python variable holding
    /// its underlying `addCurveLoop` wire tag. OCC `addThruSections` requires
    /// wire tags (not surface tags), so `Loft` sections produced by `Polygon`
    /// must be looked up here before falling back to the surface var.
    polygon_wire_var: HashMap<String, String>,
    /// Counter for unique variable names.
    var_counter: usize,
    /// User-defined functions available for inlining.
    functions: HashMap<String, FnDecl>,
    /// Field variable names created by refine blocks, to compose into a Min background field.
    refine_field_vars: Vec<String>,
    /// Most recently installed background field, so canonical controls can
    /// extend rather than replace source refine fields.
    background_field_var: Option<String>,
    runtime_loops: Vec<RuntimeLoopContext>,
    runtime_loop_counter: usize,
}

impl Default for PythonCodegen {
    fn default() -> Self {
        Self::new()
    }
}

impl PythonCodegen {
    pub fn new() -> Self {
        Self {
            buf: String::with_capacity(4096),
            indent: 0,
            scope: None,
            topo: TopologyTracker::new(),
            current_geometry: None,
            name_to_var: HashMap::new(),
            var_to_dim: HashMap::new(),
            canonical_dimensions: HashMap::new(),
            polygon_wire_var: HashMap::new(),
            var_counter: 0,
            functions: HashMap::new(),
            refine_field_vars: Vec::new(),
            background_field_var: None,
            runtime_loops: Vec::new(),
            runtime_loop_counter: 0,
        }
    }

    pub fn output(self) -> String {
        self.buf
    }

    /// Verify that every public canonical entity was materialized by the
    /// Python backend with the dimension selected by the canonical IR.
    pub fn validate_canonical_output(&self, ir: &CanonicalGeometryIr) -> Result<(), CompileError> {
        for node in &ir.entities {
            if node
                .name
                .split('.')
                .any(|component| component.starts_with("__ir_"))
            {
                continue;
            }
            let expected = gmsh_dim_from_entity_dimension(node.id.dimension);
            let Some(actual) = self.canonical_dimensions.get(&node.name) else {
                return Err(CompileError::Internal(format!(
                    "canonical entity '{}' has no Python dimension plan",
                    node.name
                )));
            };
            if *actual != expected || !self.name_to_var.contains_key(&node.name) {
                return Err(CompileError::Internal(format!(
                    "canonical entity '{}' was not materialized with dimension {:?}",
                    node.name, expected
                )));
            }
        }
        Ok(())
    }

    /// Emit manufacturing controls that are representable in the Gmsh Python
    /// backend. Cyclic design regions are materialized as rotated OCC copies;
    /// the source variables remain the canonical sector entities.
    pub fn emit_canonical_controls(
        &mut self,
        ir: &CanonicalGeometryIr,
    ) -> Result<(), CompileError> {
        let nodes = ir
            .entities
            .iter()
            .map(|node| (node.id, node))
            .collect::<HashMap<_, _>>();
        let mut controls = String::new();
        let mut control_index = 0usize;

        for region in &ir.optimization.regions {
            for annotation in &region.annotations {
                if let IrManufacturingAnnotation::SymmetryCyclic { axis, sectors } = annotation {
                    if *sectors < 2 {
                        return Err(CompileError::Internal(format!(
                            "cyclic symmetry region '{}' has invalid sector count {}",
                            region.name, sectors
                        )));
                    }

                    let entities = region
                        .entities
                        .iter()
                        .map(|id| {
                            let node = nodes.get(id).ok_or_else(|| {
                                CompileError::Internal(format!(
                                    "cyclic symmetry region '{}' references unknown entity {:?}",
                                    region.name, id
                                ))
                            })?;
                            let variable = self.name_to_var.get(&node.name).ok_or_else(|| {
                                CompileError::Internal(format!(
                                    "cyclic symmetry entity '{}' was not materialized by Python backend",
                                    node.name
                                ))
                            })?;
                            let dimension = self
                                .canonical_dimensions
                                .get(&node.name)
                                .copied()
                                .ok_or_else(|| {
                                    CompileError::Internal(format!(
                                        "cyclic symmetry entity '{}' has no canonical dimension",
                                        node.name
                                    ))
                                })?;
                            Ok((dimension as i32, variable.clone()))
                        })
                        .collect::<Result<Vec<_>, CompileError>>()?;
                    if entities.is_empty() {
                        continue;
                    }

                    let index = control_index;
                    control_index += 1;
                    controls.push_str(&format!(
                        "# MeshLang cyclic symmetry region={} sectors={} axis={:?}\n",
                        region.name, sectors, axis
                    ));
                    let dimtags = entities
                        .iter()
                        .map(|(dimension, variable)| format!("({}, {})", dimension, variable))
                        .collect::<Vec<_>>()
                        .join(", ");
                    controls.push_str(&format!("_ml_cyclic_entities_{index} = [{dimtags}]\n"));
                    controls.push_str(&format!(
                        "for _ml_cyclic_sector_{index} in range(1, {sectors}):\n"
                    ));
                    controls.push_str(&format!(
                        "    _ml_cyclic_angle_{index} = 2.0 * math.pi * _ml_cyclic_sector_{index} / {sectors}.0\n"
                    ));
                    controls.push_str(&format!(
                        "    _ml_cyclic_copies_{index} = occ.copy(_ml_cyclic_entities_{index})\n"
                    ));
                    controls.push_str(&format!(
                        "    occ.rotate(_ml_cyclic_copies_{index}, 0.0, 0.0, 0.0, {}, {}, {}, _ml_cyclic_angle_{index})\n",
                        format_float_py(axis[0]),
                        format_float_py(axis[1]),
                        format_float_py(axis[2])
                    ));
                }
            }
        }

        let transfinite_entity_names = ir
            .entities
            .iter()
            .map(|node| {
                let name = self
                    .name_to_var
                    .get(&node.name)
                    .cloned()
                    .unwrap_or_else(|| node.name.replace('.', "__"));
                (node.id, name)
            })
            .collect::<HashMap<_, _>>();
        for control in &ir.mesh.transfinite {
            let statement = crate::canonical_ast::materialize_transfinite_control(
                control,
                ir,
                &transfinite_entity_names,
            )?;
            let start = self.buf.len();
            self.emit_transfinite_py(&statement)?;
            controls.push_str(&self.buf.split_off(start));
        }

        let refine_entity_names = ir
            .entities
            .iter()
            .map(|node| {
                let name = self
                    .name_to_var
                    .get(&node.name)
                    .cloned()
                    .unwrap_or_else(|| node.name.replace('.', "__"));
                (node.id, name)
            })
            .collect::<HashMap<_, _>>();
        for control in &ir.mesh.refinements {
            let statement = crate::canonical_ast::materialize_refine_control(
                control,
                ir,
                &refine_entity_names,
            )?;
            let start = self.buf.len();
            self.emit_refine_py(&statement)?;
            controls.push_str(&self.buf.split_off(start));
        }
        for metric in &ir.mesh.metrics {
            if matches!(metric.tensor, IrMetricTensor::Hessian { .. }) {
                continue;
            }
            let statement = crate::canonical_ast::materialize_mesh_metric_control(
                metric,
                ir,
                &refine_entity_names,
            )?;
            let start = self.buf.len();
            self.emit_mesh_metric_py(&statement)?;
            controls.push_str(&self.buf.split_off(start));
        }
        for control in &ir.mesh.boundary_layers {
            let statement = crate::canonical_ast::materialize_boundary_layer_control(
                control,
                ir,
                &transfinite_entity_names,
            )?;
            let start = self.buf.len();
            self.emit_boundary_layer_py(&statement)?;
            controls.push_str(&self.buf.split_off(start));
        }

        for pair in &ir.periodic_pairs {
            let source = nodes.get(&pair.source).ok_or_else(|| {
                CompileError::Internal(format!(
                    "canonical periodic control references unknown source entity {:?}",
                    pair.source
                ))
            })?;
            let target = nodes.get(&pair.target).ok_or_else(|| {
                CompileError::Internal(format!(
                    "canonical periodic control references unknown target entity {:?}",
                    pair.target
                ))
            })?;
            if source.id.dimension != target.id.dimension {
                return Err(CompileError::Internal(
                    "canonical periodic control source/target dimensions drifted after validation"
                        .into(),
                ));
            }
            let dimension = match source.id.dimension {
                EntityDimension::Curve => 1,
                EntityDimension::Surface => 2,
                other => {
                    return Err(CompileError::Unsupported(format!(
                        "canonical periodic control does not support {:?} entities",
                        other
                    )));
                }
            };
            let source_var = self.name_to_var.get(&source.name).ok_or_else(|| {
                CompileError::Internal(format!(
                    "canonical periodic source '{}' not materialized by Python backend",
                    source.name
                ))
            })?;
            let target_var = self.name_to_var.get(&target.name).ok_or_else(|| {
                CompileError::Internal(format!(
                    "canonical periodic target '{}' not materialized by Python backend",
                    target.name
                ))
            })?;
            let coefficients = pair
                .transform
                .coefficients
                .iter()
                .map(|value| format_float_py(*value))
                .collect::<Vec<_>>()
                .join(", ");
            controls.push_str(&format!(
                "gmsh.model.mesh.setPeriodic({}, [{}], [{}], [{}])\n",
                dimension, source_var, target_var, coefficients
            ));
        }

        for embed in &ir.mesh.embeds {
            let source = nodes.get(&embed.source).ok_or_else(|| {
                CompileError::Internal(format!(
                    "canonical mesh embed references unknown source entity {:?}",
                    embed.source
                ))
            })?;
            let target = nodes.get(&embed.target).ok_or_else(|| {
                CompileError::Internal(format!(
                    "canonical mesh embed references unknown target entity {:?}",
                    embed.target
                ))
            })?;
            if source.id.dimension != embed.source_dimension
                || target.id.dimension != embed.target_dimension
            {
                return Err(CompileError::Internal(
                    "canonical mesh embed dimension drifted after validation".into(),
                ));
            }
            let source_dim = gmsh_dim_from_entity_dimension(embed.source_dimension) as u8;
            let target_dim = gmsh_dim_from_entity_dimension(embed.target_dimension) as u8;
            let source_var = self.name_to_var.get(&source.name).ok_or_else(|| {
                CompileError::Internal(format!(
                    "canonical mesh embed source '{}' not materialized by Python backend",
                    source.name
                ))
            })?;
            let target_var = self.name_to_var.get(&target.name).ok_or_else(|| {
                CompileError::Internal(format!(
                    "canonical mesh embed target '{}' not materialized by Python backend",
                    target.name
                ))
            })?;
            controls.push_str(&format!(
                "gmsh.model.mesh.embed({}, [{}], {}, {})\n",
                source_dim, source_var, target_dim, target_var
            ));
        }

        for compound in &ir.mesh.compounds {
            let first = compound.entities.first().ok_or_else(|| {
                CompileError::Internal("canonical mesh compound has no entities".into())
            })?;
            let first_node = nodes.get(first).ok_or_else(|| {
                CompileError::Internal(format!(
                    "canonical mesh compound references unknown entity {:?}",
                    first
                ))
            })?;
            let dimension = gmsh_dim_from_entity_dimension(first.dimension) as u8;
            let variables = compound
                .entities
                .iter()
                .map(|id| {
                    let node = nodes.get(id).ok_or_else(|| {
                        CompileError::Internal(format!(
                            "canonical mesh compound references unknown entity {:?}",
                            id
                        ))
                    })?;
                    if node.id.dimension != first_node.id.dimension {
                        return Err(CompileError::Internal(
                            "canonical mesh compound dimension drifted after validation".into(),
                        ));
                    }
                    self.name_to_var.get(&node.name).cloned().ok_or_else(|| {
                        CompileError::Internal(format!(
                        "canonical mesh compound entity '{}' not materialized by Python backend",
                        node.name
                    ))
                    })
                })
                .collect::<Result<Vec<_>, CompileError>>()?;
            controls.push_str(&format!(
                "gmsh.model.mesh.setCompound({}, [{}])\n",
                dimension,
                variables.join(", ")
            ));
        }

        // Boundary-layer controls are captured after the source mesh has
        // finalized its refine fields. Install a canonical Min background
        // field for any fields emitted above.
        if !self.refine_field_vars.is_empty() {
            let min_var = self.fresh_var("bg");
            let mut field_vars = self.refine_field_vars.clone();
            if let Some(previous) = &self.background_field_var {
                field_vars.insert(0, previous.clone());
            }
            controls.push_str(&format!("{} = gmsh.model.mesh.field.add('Min')\n", min_var));
            controls.push_str(&format!(
                "gmsh.model.mesh.field.setNumbers({}, 'FieldsList', [{}])\n",
                min_var,
                field_vars.join(", ")
            ));
            controls.push_str(&format!(
                "gmsh.model.mesh.field.setAsBackgroundMesh({})\n",
                min_var
            ));
            self.background_field_var = Some(min_var);
            self.refine_field_vars.clear();
        }

        if controls.is_empty() {
            return Ok(());
        }
        let marker = "occ.synchronize()\n";
        let marker_index = self.buf.rfind(marker).ok_or_else(|| {
            CompileError::Internal(
                "Python cyclic symmetry controls require a final occ.synchronize()".into(),
            )
        })?;
        self.buf.insert_str(marker_index, &controls);
        Ok(())
    }

    /// Install the backend-neutral entity plan. Emission still follows the
    /// source AST for portable text generation, but every named entity's
    /// dimension comes from the validated canonical IR.
    pub fn apply_canonical_plan(&mut self, ir: &CanonicalGeometryIr) -> Result<(), CompileError> {
        ir.validate().map_err(|errors| {
            CompileError::Internal(
                errors
                    .into_iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<_>>()
                    .join("; "),
            )
        })?;
        self.canonical_dimensions = ir
            .entities
            .iter()
            .filter(|node| !node.name.contains(".__ir_"))
            .map(|node| {
                (
                    node.name.clone(),
                    gmsh_dim_from_entity_dimension(node.id.dimension),
                )
            })
            .collect();
        Ok(())
    }

    fn fresh_var(&mut self, prefix: &str) -> String {
        self.var_counter += 1;
        format!("{}_{}", prefix, self.var_counter)
    }

    pub fn emit_source_file(&mut self, file: &SourceFile) -> Result<(), CompileError> {
        self.scope = Some(name_resolution::resolve(file)?);

        // Collect user-defined functions for inlining
        for decl in &file.declarations {
            if let Declaration::Function(f) = decl {
                self.functions.insert(f.name.name.clone(), f.clone());
            }
        }

        // Preamble
        self.line("#!/usr/bin/env python3");
        self.line(&format!(
            "# Generated by MeshLang compiler v{}",
            env!("CARGO_PKG_VERSION")
        ));
        self.line("# Do not edit — regenerate from .mlg source");
        self.line("");
        self.line("import gmsh");
        self.line("import sys");
        self.line("import math");
        self.line("");
        self.line("def _ml_tags(value):");
        self.indent += 1;
        self.line("return list(value) if isinstance(value, (list, tuple)) else [value]");
        self.indent -= 1;
        self.line("");
        self.line("def _ml_dimtags(dim, value):");
        self.indent += 1;
        self.line("if isinstance(value, tuple) and len(value) == 2 and all(isinstance(item, int) for item in value):");
        self.indent += 1;
        self.line("return [value]");
        self.indent -= 1;
        self.line("if isinstance(value, list) and all(isinstance(item, (list, tuple)) and len(item) == 2 for item in value):");
        self.indent += 1;
        self.line("return [tuple(item) for item in value]");
        self.indent -= 1;
        self.line("return [(dim, tag) for tag in _ml_tags(value)]");
        self.indent -= 1;
        self.line("");
        self.line("def _ml_boundary(dimtags, combined=False, oriented=False, recursive=False):");
        self.indent += 1;
        self.line("live = set(gmsh.model.getEntities())");
        self.line("filtered = [(dim, tag) for dim, tag in dimtags if (dim, tag) in live]");
        self.line("if not filtered:");
        self.indent += 1;
        self.line("return []");
        self.indent -= 1;
        self.line("return gmsh.model.getBoundary(filtered, combined=combined, oriented=oriented, recursive=recursive)");
        self.indent -= 1;
        self.line("");
        self.line("def _ml_only(dim, dimtags):");
        self.indent += 1;
        self.line("return [tag for entity_dim, tag in dimtags if entity_dim == dim]");
        self.indent -= 1;
        self.line("");
        self.line("def _ml_single(value, operation):");
        self.indent += 1;
        self.line("tags = _ml_tags(value)");
        self.line("if len(tags) != 1:");
        self.indent += 1;
        self.line("raise RuntimeError(f'{operation} requires one entity, got {len(tags)}')");
        self.indent -= 1;
        self.line("return tags[0]");
        self.indent -= 1;
        self.line("");
        self.line("gmsh.initialize()");
        self.line("gmsh.model.add(\"meshlang_model\")");
        self.line("occ = gmsh.model.occ");
        self.line("");
        self.line("# --- MeshLang topology helpers ---");
        self.buf.push_str(TopologyQuery::python_helpers());
        self.line("");

        for decl in &file.declarations {
            self.emit_declaration(decl)?;
        }

        // Finalize
        self.line("occ.synchronize()");

        // Emit topology helper functions if any physics groups use deferred resolution
        self.line("");
        self.line("gmsh.finalize()");

        Ok(())
    }

    fn emit_declaration(&mut self, decl: &Declaration) -> Result<(), CompileError> {
        // OCC-generated entity tags are not queryable until the current
        // geometry has been synchronized.  Metadata declarations can appear
        // after geometry blocks and use getBoundary/addPhysicalGroup below.
        if matches!(
            decl,
            Declaration::Physics(_)
                | Declaration::PhysicalRegion(_)
                | Declaration::Boundary(_)
                | Declaration::Interface(_)
                | Declaration::MaterialRegion(_)
        ) {
            self.line("occ.synchronize()");
        }

        match decl {
            Declaration::Param(p) => {
                let val = self.emit_expr(&p.value)?;
                self.line(&format!("{} = {}", p.name.name, val));
            }
            Declaration::Const(c) => {
                let val = self.emit_expr(&c.value)?;
                self.line(&format!("{} = {}", c.name.name, val));
            }
            Declaration::Geometry(g) if g.params.is_empty() && g.ports.is_empty() => {
                self.emit_geometry(g)?
            }
            Declaration::Geometry(g) => {
                self.line(&format!("# geometry template: {}", g.name.name));
            }
            Declaration::Mesh(m) => self.emit_mesh(m)?,
            Declaration::Physics(p) => self.emit_physics(p)?,
            Declaration::Export(e) => self.emit_export(e)?,
            Declaration::Import(imp) => {
                let ext = imp.path.rsplit('.').next().unwrap_or("");
                match ext.to_lowercase().as_str() {
                    "step" | "stp" | "iges" | "igs" | "brep" | "stl" | "obj" => {
                        self.line(&format!("occ.importShapes('{}')", imp.path));
                    }
                    "mlg" => {} // resolved during flatten pass
                    _ => {
                        self.line(&format!("# import: {}", imp.path));
                    }
                }
            }
            Declaration::DesignVariable(variable) => {
                let value = self.emit_expr(&variable.initial)?;
                self.line(&format!("{} = {}", variable.name.name, value));
            }
            Declaration::Uncertainty(value) => {
                self.line(&format!(
                    "# uncertainty: {} (see provenance manifest)",
                    value.name.name
                ));
            }
            Declaration::Function(_) | Declaration::Struct(_) => {}
            Declaration::PhysicalRegion(region) => self.emit_named_physical_group_py(
                &region.name.name,
                region.dimension,
                &region.entities,
            )?,
            Declaration::Boundary(boundary) => self.emit_named_physical_group_py(
                &boundary.name.name,
                boundary.dimension,
                &boundary.entities,
            )?,
            Declaration::Interface(interface) => {
                self.emit_named_physical_group_py(
                    &format!("{}_primary", interface.name.name),
                    interface.dimension,
                    &interface.primary,
                )?;
                self.emit_named_physical_group_py(
                    &format!("{}_secondary", interface.name.name),
                    interface.dimension,
                    &interface.secondary,
                )?;
            }
            Declaration::MaterialRegion(region) => {
                let dimension = self
                    .expr_gmsh_dim(&region.region)
                    .unwrap_or(GmshDim::Volume);
                self.emit_named_physical_group_py(
                    &region.name.name,
                    entity_dimension_from_gmsh(dimension),
                    &region.region,
                )?;
            }
            Declaration::Objective(_)
            | Declaration::Constraint(_)
            | Declaration::DesignRegion(_)
            | Declaration::CoordinateSystem(_)
            | Declaration::GeometryField(_) => {
                self.line("# modelling metadata: see provenance manifest");
            }
        }
        Ok(())
    }

    fn emit_named_physical_group_py(
        &mut self,
        name: &str,
        dimension: EntityDimension,
        entities: &Expr,
    ) -> Result<(), CompileError> {
        let dim = dimension.topological_dimension();
        let tuples = self.metadata_entity_tuples_py(entities, dimension)?;
        let variable = self.fresh_var("physical_entities");
        self.line(&format!("{} = {}", variable, tuples));
        self.line(&format!(
            "gmsh.model.addPhysicalGroup({}, [abs(tag) for entity_dim, tag in {} if entity_dim == {}], name='{}')",
            dim, variable, dim, name
        ));
        Ok(())
    }

    fn metadata_entity_tuples_py(
        &self,
        entities: &Expr,
        dimension: EntityDimension,
    ) -> Result<String, CompileError> {
        if let Expr::Array(array) = entities {
            let values = array
                .elements
                .iter()
                .map(|entity| self.metadata_entity_tuples_py(entity, dimension))
                .collect::<Result<Vec<_>, _>>()?;
            return Ok(format!("({})", values.join(" + ")));
        }
        let path = entity_path(entities).ok_or_else(|| CompileError::TypeError {
            message: "physical-group entities must be named geometry bindings".into(),
        })?;
        if path.ends_with(".faces") {
            let owner = path.trim_end_matches(".faces");
            let variable = self
                .name_to_var
                .get(owner)
                .cloned()
                .unwrap_or_else(|| owner.split('.').next_back().unwrap_or(owner).to_owned());
            return Ok(format!(
                "_ml_boundary(_ml_dimtags(3, {}), oriented=False)",
                variable
            ));
        }
        let variable = self
            .name_to_var
            .get(&path)
            .cloned()
            .unwrap_or_else(|| path.split('.').next_back().unwrap_or(&path).to_owned());
        Ok(format!(
            "_ml_dimtags({}, {})",
            dimension.topological_dimension(),
            variable
        ))
    }

    fn emit_geometry(&mut self, geom: &GeometryBlock) -> Result<(), CompileError> {
        self.line(&format!("# geometry: {}", geom.name.name));
        self.current_geometry = Some(geom.name.name.clone());

        for stmt in &geom.body.statements {
            match stmt {
                Statement::Let(l) => self.emit_geometry_let(l, &geom.name.name)?,
                Statement::For(f) => self.emit_geometry_for(f, &geom.name.name)?,
                Statement::If(i) => self.emit_geometry_if(i, &geom.name.name)?,
                _ => {}
            }
        }

        self.current_geometry = None;
        self.line("");
        Ok(())
    }

    fn emit_geometry_let(&mut self, l: &LetStatement, geo_name: &str) -> Result<(), CompileError> {
        let qname = format!("{}.{}", geo_name, l.name.name);
        let variable_name = l.name.name.replace('.', "__");
        // Track qualified name → Python variable name
        self.name_to_var
            .insert(qname.clone(), variable_name.clone());

        match &l.value {
            Expr::Constructor(c) if c.type_name.name == "Polygon" => {
                self.emit_polygon_py(&variable_name, c)?;
                self.var_to_dim
                    .insert(variable_name.clone(), GmshDim::Surface);
                self.topo.register_unknown(&qname, 0);
            }
            Expr::Constructor(c) => {
                let call = self.emit_constructor_call(c)?;
                self.line(&format!("{} = {}", variable_name, call));
                self.var_to_dim
                    .insert(variable_name.clone(), constructor_dim(c));
                match c.type_name.name.as_str() {
                    "Box" => self.topo.register_box(&qname, 0),
                    "Cylinder" => self.topo.register_cylinder(&qname, 0),
                    "Sphere" => self.topo.register_sphere(&qname, 0),
                    _ => self.topo.register_unknown(&qname, 0),
                }
            }
            Expr::Pipe(p) => {
                let dim = self.emit_pipe_chain_py(&variable_name, p)?;
                self.var_to_dim.insert(variable_name.clone(), dim);
                self.register_pipe_base_topo(&qname, p);
            }
            Expr::Binary(b) if self.is_csg_binary(b) => {
                self.emit_csg_py(&variable_name, b)?;
                let left_name = self.expr_name_best_effort(&b.left, geo_name);
                let right_name = self.expr_name_best_effort(&b.right, geo_name);
                let op = match b.op {
                    BinaryOp::Sub | BinaryOp::CsgDifference => CsgOp::Difference,
                    BinaryOp::CsgUnion => CsgOp::Union,
                    BinaryOp::CsgIntersect => CsgOp::Intersection,
                    _ => CsgOp::Difference,
                };
                self.topo
                    .register_csg_result(&qname, 0, &left_name, &right_name, op);
            }
            Expr::Call(c) if self.functions.contains_key(&c.function.name) => {
                let inlined = self.try_inline_call(c)?;
                self.emit_geometry_let_value(&variable_name, geo_name, &qname, &inlined)?;
            }
            Expr::StructInit(si) => {
                // Emit as a Python dict: name = {'field': value, ...}
                let mut entries = Vec::new();
                for field in &si.fields {
                    let val = self.emit_expr(&field.value)?;
                    entries.push(format!("'{}': {}", field.name.name, val));
                }
                self.line(&format!("{} = {{{}}}", variable_name, entries.join(", ")));
            }
            Expr::FieldAccess(_) => {
                let value = match self.try_resolve_topology_expr_py(&l.value)? {
                    Some(value) => value,
                    None => self.emit_expr(&l.value)?,
                };
                self.line(&format!("{} = {}", variable_name, value));
            }
            _ => {
                let val = self.emit_expr(&l.value)?;
                self.line(&format!("{} = {}", variable_name, val));
            }
        }
        if let Some(dimension) = self.canonical_dimensions.get(&qname).copied() {
            self.var_to_dim.insert(variable_name, dimension);
        }
        Ok(())
    }

    /// Emit a geometry let binding from an inlined expression.
    fn emit_geometry_let_value(
        &mut self,
        var_name: &str,
        geo_name: &str,
        qname: &str,
        expr: &Expr,
    ) -> Result<(), CompileError> {
        match expr {
            Expr::Constructor(c) => {
                let call = self.emit_constructor_call(c)?;
                self.line(&format!("{} = {}", var_name, call));
                self.var_to_dim
                    .insert(var_name.to_string(), constructor_dim(c));
                match c.type_name.name.as_str() {
                    "Box" => self.topo.register_box(qname, 0),
                    "Cylinder" => self.topo.register_cylinder(qname, 0),
                    "Sphere" => self.topo.register_sphere(qname, 0),
                    _ => self.topo.register_unknown(qname, 0),
                }
            }
            Expr::Pipe(p) => {
                let dim = self.emit_pipe_chain_py(var_name, p)?;
                self.var_to_dim.insert(var_name.to_string(), dim);
                self.register_pipe_base_topo(qname, p);
            }
            Expr::Binary(b) if self.is_csg_binary(b) => {
                self.emit_csg_py(var_name, b)?;
                let left_name = self.expr_name_best_effort(&b.left, geo_name);
                let right_name = self.expr_name_best_effort(&b.right, geo_name);
                let op = match b.op {
                    BinaryOp::Sub | BinaryOp::CsgDifference => CsgOp::Difference,
                    BinaryOp::CsgUnion => CsgOp::Union,
                    BinaryOp::CsgIntersect => CsgOp::Intersection,
                    _ => CsgOp::Difference,
                };
                self.topo
                    .register_csg_result(qname, 0, &left_name, &right_name, op);
            }
            Expr::FieldAccess(_) => {
                let value = match self.try_resolve_topology_expr_py(expr)? {
                    Some(value) => value,
                    None => self.emit_expr(expr)?,
                };
                self.line(&format!("{} = {}", var_name, value));
            }
            _ => {
                let val = self.emit_expr(expr)?;
                self.line(&format!("{} = {}", var_name, val));
            }
        }
        if let Some(dimension) = self.canonical_dimensions.get(qname).copied() {
            self.var_to_dim.insert(var_name.to_string(), dimension);
        }
        Ok(())
    }

    /// Geometry topology paths use Gmsh dim-tag queries, while ordinary
    /// struct fields retain Python's dictionary access convention.
    fn try_resolve_topology_expr_py(
        &mut self,
        expr: &Expr,
    ) -> Result<Option<String>, CompileError> {
        let Some(path) = entity_path(expr) else {
            return Ok(None);
        };
        let parts = path.split('.').collect::<Vec<_>>();
        let known_entity = (1..=parts.len()).rev().any(|length| {
            let candidate = parts[..length].join(".");
            self.topo.lookup(&candidate).is_some()
                || self.current_geometry.as_ref().is_some_and(|geometry| {
                    self.topo
                        .lookup(&format!("{geometry}.{candidate}"))
                        .is_some()
                })
        });
        if !known_entity {
            return Ok(None);
        }
        self.resolve_physics_expr_py(expr)
            .map(Some)
            .map_err(CompileError::Unsupported)
    }

    /// Inline a single-expression function call by substituting arguments.
    fn try_inline_call(&self, call: &CallExpr) -> Result<Expr, CompileError> {
        let func = self.functions.get(&call.function.name).ok_or_else(|| {
            CompileError::UndefinedReference {
                name: call.function.name.clone(),
            }
        })?;

        let mut subst: HashMap<String, Expr> = HashMap::new();
        for (i, param) in func.params.iter().enumerate() {
            let arg_expr = if let Some(arg) = call
                .args
                .iter()
                .find(|a| a.name.as_ref().is_some_and(|n| n.name == param.name.name))
            {
                arg.value.clone()
            } else if let Some(arg) = call.args.get(i) {
                arg.value.clone()
            } else if let Some(ref default) = param.default {
                default.clone()
            } else {
                return Err(CompileError::Internal(format!(
                    "missing argument '{}' in call to '{}'",
                    param.name.name, call.function.name
                )));
            };
            subst.insert(param.name.name.clone(), arg_expr);
        }

        if func.body.statements.is_empty() {
            return Err(CompileError::Internal(format!(
                "function '{}' has an empty body",
                call.function.name
            )));
        }

        for statement in &func.body.statements[..func.body.statements.len() - 1] {
            let Statement::Let(local) = statement else {
                if matches!(statement, Statement::If(_)) {
                    return Err(CompileError::Unsupported(format!(
                        "inline function '{}' branch condition must be a compile-time boolean",
                        call.function.name
                    )));
                }
                return Err(CompileError::Unsupported(format!(
                    "function '{}' contains a non-expression statement before its result",
                    call.function.name
                )));
            };
            let value = substitute_expr(&local.value, &subst);
            subst.insert(local.name.name.clone(), value);
        }

        let last_expr = func.body.statements.last().ok_or_else(|| {
            CompileError::Internal(format!("function '{}' has empty body", call.function.name))
        })?;

        match last_expr {
            Statement::Expr(e) => Ok(substitute_expr(e, &subst)),
            Statement::Let(l) => Ok(substitute_expr(&l.value, &subst)),
            _ => Err(CompileError::Internal(format!(
                "function '{}' body must end with an expression",
                call.function.name
            ))),
        }
    }

    /// Compile-time unroll a for loop in a geometry body (Python backend).
    fn emit_geometry_for(&mut self, f: &ForStatement, geo_name: &str) -> Result<(), CompileError> {
        let ranges = match eval_for_ranges(&f.clauses) {
            Ok(ranges) => ranges,
            Err(CompileError::Unsupported(message)) if message.contains("compile-time integer") => {
                return self.emit_dynamic_geometry_for(f, geo_name);
            }
            Err(error) => return Err(error),
        };

        // Skip if any range is empty
        if ranges.iter().any(|(start, end)| start >= end) {
            return Ok(());
        }

        let mut iteration_vars: Vec<String> = Vec::new();

        let mut indices: Vec<i64> = vec![0; ranges.len()];
        let starts: Vec<i64> = ranges.iter().map(|r| r.0).collect();
        let ends: Vec<i64> = ranges.iter().map(|r| r.1).collect();
        let var_names: Vec<String> = f.clauses.iter().map(|c| c.var.name.clone()).collect();

        for (i, s) in starts.iter().enumerate() {
            indices[i] = *s;
        }

        loop {
            let mut subst: HashMap<String, Expr> = HashMap::new();
            for (i, name) in var_names.iter().enumerate() {
                subst.insert(
                    name.clone(),
                    Expr::literal(Literal::Int(indices[i]), f.span),
                );
            }

            for stmt in &f.body.statements {
                match stmt {
                    Statement::Let(l) => {
                        let idx_str: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
                        let iter_suffix = idx_str.join("_");
                        let iter_name = format!("{}_{}", l.name.name, iter_suffix);
                        let qname = format!("{}.{}", geo_name, iter_name);
                        self.name_to_var.insert(qname.clone(), iter_name.clone());
                        let subst_value = substitute_expr(&l.value, &subst);
                        self.emit_geometry_let_value(&iter_name, geo_name, &qname, &subst_value)?;
                        iteration_vars.push(iter_name);
                    }
                    Statement::Expr(e) => {
                        let subst_expr = substitute_expr(e, &subst);
                        let var = self.ensure_entity_py(&subst_expr)?;
                        iteration_vars.push(var);
                    }
                    _ => {}
                }
            }

            let mut carry = true;
            for i in (0..indices.len()).rev() {
                if carry {
                    indices[i] += 1;
                    if indices[i] >= ends[i] {
                        indices[i] = starts[i];
                    } else {
                        carry = false;
                    }
                }
            }
            if carry {
                break;
            }
        }

        // BooleanUnion (occ.fuse) all iteration results
        if iteration_vars.len() > 1 {
            let mut acc = iteration_vars[0].clone();
            for var in &iteration_vars[1..] {
                let result_var = self.fresh_var("for_union");
                self.line(&format!(
                    "{}_result = occ.fuse(_ml_dimtags(3, {}), _ml_dimtags(3, {}))",
                    result_var, acc, var
                ));
                self.line(&format!(
                    "{} = _ml_only(3, {}_result[0])",
                    result_var, result_var
                ));
                acc = result_var;
            }
        }

        Ok(())
    }

    /// Constant-fold an if statement in a geometry body (Python backend).
    fn runtime_loop_list(&mut self, dim: GmshDim) -> Result<String, CompileError> {
        let Some(index) = self.runtime_loops.len().checked_sub(1) else {
            return Err(CompileError::Internal(
                "runtime loop entity list requested outside a loop".into(),
            ));
        };
        if let Some((_, name)) = self.runtime_loops[index]
            .lists
            .iter()
            .find(|(existing, _)| *existing == dim)
        {
            return Ok(name.clone());
        }

        let name = format!(
            "__ml_loop_{}_{}",
            self.runtime_loops[index].id,
            self.runtime_loops[index].lists.len()
        );
        self.runtime_loops[index].lists.push((dim, name.clone()));
        self.line(&format!("{} = []", name));
        Ok(name)
    }

    fn runtime_loop_note_source(&mut self, name: String) {
        if let Some(context) = self.runtime_loops.last_mut() {
            if !context.source_names.contains(&name) {
                context.source_names.push(name);
            }
        }
    }

    fn runtime_loop_record_entity(
        &mut self,
        dim: GmshDim,
        variable: &str,
    ) -> Result<(), CompileError> {
        let list = self.runtime_loop_list(dim)?;
        self.line(&format!(
            "{}.extend(_ml_dimtags({}, {}))",
            list, dim as usize, variable
        ));
        Ok(())
    }

    fn runtime_loop_record_list(
        &mut self,
        dim: GmshDim,
        child_list: &str,
    ) -> Result<(), CompileError> {
        let list = self.runtime_loop_list(dim)?;
        self.line(&format!("{}.extend({})", list, child_list));
        Ok(())
    }

    fn runtime_loop_collect_body_entities(
        &mut self,
        statements: &[Statement],
    ) -> Result<(), CompileError> {
        for statement in statements {
            match statement {
                Statement::Let(binding) => {
                    let qualified = format!(
                        "{geo}.{}",
                        binding.name.name,
                        geo = self.current_geometry.as_deref().unwrap_or("")
                    );
                    let Some(variable) = self.name_to_var.get(&qualified).cloned() else {
                        continue;
                    };
                    let Some(dim) = self.var_to_dim.get(&variable).copied() else {
                        continue;
                    };
                    self.runtime_loop_note_source(qualified);
                    self.runtime_loop_record_entity(dim, &variable)?;
                }
                Statement::If(_) | Statement::For(_) | Statement::Expr(_) => {}
                _ => {}
            }
        }
        Ok(())
    }

    fn finish_runtime_loop(&mut self) -> Result<(), CompileError> {
        let context = self
            .runtime_loops
            .pop()
            .ok_or_else(|| CompileError::Internal("runtime loop context stack underflow".into()))?;
        let lists = context.lists.clone();
        let source_names = context.source_names.clone();
        let has_parent = !self.runtime_loops.is_empty();

        for (dim, list) in lists {
            for source in &source_names {
                let Some(variable) = self.name_to_var.get(source).cloned() else {
                    continue;
                };
                if self.var_to_dim.get(&variable).copied() == Some(dim) {
                    self.name_to_var.insert(source.clone(), list.clone());
                    self.var_to_dim.insert(list.clone(), dim);
                }
            }
            if has_parent {
                self.runtime_loop_record_list(dim, &list)?;
            } else {
                self.line(&format!("# MeshLang runtime loop aggregate: {}", list));
            }
        }

        for source in source_names {
            self.runtime_loop_note_source(source);
        }
        Ok(())
    }

    fn emit_dynamic_geometry_for(
        &mut self,
        f: &ForStatement,
        geo_name: &str,
    ) -> Result<(), CompileError> {
        self.runtime_loop_counter += 1;
        self.runtime_loops.push(RuntimeLoopContext {
            id: self.runtime_loop_counter,
            ..RuntimeLoopContext::default()
        });
        let result = self.emit_dynamic_geometry_for_clause(f, geo_name, 0);
        match result {
            Ok(()) => self.finish_runtime_loop(),
            Err(error) => {
                self.runtime_loops.pop();
                Err(error)
            }
        }
    }

    fn emit_dynamic_geometry_for_clause(
        &mut self,
        f: &ForStatement,
        geo_name: &str,
        clause_index: usize,
    ) -> Result<(), CompileError> {
        let clause = f.clauses.get(clause_index).ok_or_else(|| {
            CompileError::Internal("dynamic loop clause index out of bounds".into())
        })?;
        let start = self.emit_expr(&clause.range.start)?;
        let end = self.emit_expr(&clause.range.end)?;
        let stop = if clause.range.inclusive {
            format!("({end}) + 1")
        } else {
            end
        };
        self.line(&format!(
            "for {} in range({}, {}):",
            clause.var.name, start, stop
        ));
        self.indent += 1;
        if clause_index + 1 < f.clauses.len() {
            self.emit_dynamic_geometry_for_clause(f, geo_name, clause_index + 1)?;
        } else {
            for statement in &f.body.statements {
                match statement {
                    Statement::Let(binding) => self.emit_geometry_let(binding, geo_name)?,
                    Statement::For(nested) => self.emit_geometry_for(nested, geo_name)?,
                    Statement::If(nested) => self.emit_geometry_if(nested, geo_name)?,
                    _ => {}
                }
            }
            self.runtime_loop_collect_body_entities(&f.body.statements)?;
        }
        self.indent -= 1;
        Ok(())
    }

    fn emit_geometry_if(&mut self, i: &IfStatement, geo_name: &str) -> Result<(), CompileError> {
        if let Some(cond_val) = try_eval_const_bool(&i.condition) {
            if cond_val {
                for stmt in &i.then_block.statements {
                    match stmt {
                        Statement::Let(l) => self.emit_geometry_let(l, geo_name)?,
                        Statement::For(f) => self.emit_geometry_for(f, geo_name)?,
                        Statement::If(nested) => self.emit_geometry_if(nested, geo_name)?,
                        _ => {}
                    }
                }
                if !self.runtime_loops.is_empty() {
                    self.runtime_loop_collect_body_entities(&i.then_block.statements)?;
                }
            } else if let Some(ref else_branch) = i.else_block {
                match else_branch {
                    ElseBranch::Else(block) => {
                        for stmt in &block.statements {
                            match stmt {
                                Statement::Let(l) => self.emit_geometry_let(l, geo_name)?,
                                Statement::For(f) => self.emit_geometry_for(f, geo_name)?,
                                Statement::If(nested) => self.emit_geometry_if(nested, geo_name)?,
                                _ => {}
                            }
                        }
                        if !self.runtime_loops.is_empty() {
                            self.runtime_loop_collect_body_entities(&block.statements)?;
                        }
                    }
                    ElseBranch::ElseIf(elif) => self.emit_geometry_if(elif, geo_name)?,
                }
            }
            Ok(())
        } else {
            // Python can handle runtime if — emit as Python if/else
            let cond_str = self.emit_expr(&i.condition)?;
            self.line(&format!("if {}:", cond_str));
            self.indent += 1;
            for stmt in &i.then_block.statements {
                match stmt {
                    Statement::Let(l) => self.emit_geometry_let(l, geo_name)?,
                    Statement::For(f) => self.emit_geometry_for(f, geo_name)?,
                    Statement::If(nested) => self.emit_geometry_if(nested, geo_name)?,
                    _ => {}
                }
            }
            if !self.runtime_loops.is_empty() {
                self.runtime_loop_collect_body_entities(&i.then_block.statements)?;
            }
            self.indent -= 1;
            if let Some(ref else_branch) = i.else_block {
                match else_branch {
                    ElseBranch::Else(block) => {
                        self.line("else:");
                        self.indent += 1;
                        for stmt in &block.statements {
                            match stmt {
                                Statement::Let(l) => self.emit_geometry_let(l, geo_name)?,
                                Statement::For(f) => self.emit_geometry_for(f, geo_name)?,
                                Statement::If(nested) => self.emit_geometry_if(nested, geo_name)?,
                                _ => {}
                            }
                        }
                        if !self.runtime_loops.is_empty() {
                            self.runtime_loop_collect_body_entities(&block.statements)?;
                        }
                        self.indent -= 1;
                    }
                    ElseBranch::ElseIf(elif) => {
                        self.emit_geometry_elif(elif, geo_name)?;
                    }
                }
            }
            Ok(())
        }
    }

    fn emit_geometry_elif(&mut self, i: &IfStatement, geo_name: &str) -> Result<(), CompileError> {
        let cond_str = self.emit_expr(&i.condition)?;
        self.line(&format!("elif {}:", cond_str));
        self.indent += 1;
        for stmt in &i.then_block.statements {
            match stmt {
                Statement::Let(l) => self.emit_geometry_let(l, geo_name)?,
                Statement::For(f) => self.emit_geometry_for(f, geo_name)?,
                Statement::If(nested) => self.emit_geometry_if(nested, geo_name)?,
                _ => {}
            }
        }
        if !self.runtime_loops.is_empty() {
            self.runtime_loop_collect_body_entities(&i.then_block.statements)?;
        }
        self.indent -= 1;

        if let Some(ref else_branch) = i.else_block {
            match else_branch {
                ElseBranch::Else(block) => {
                    self.line("else:");
                    self.indent += 1;
                    for stmt in &block.statements {
                        match stmt {
                            Statement::Let(l) => self.emit_geometry_let(l, geo_name)?,
                            Statement::For(f) => self.emit_geometry_for(f, geo_name)?,
                            Statement::If(nested) => self.emit_geometry_if(nested, geo_name)?,
                            _ => {}
                        }
                    }
                    if !self.runtime_loops.is_empty() {
                        self.runtime_loop_collect_body_entities(&block.statements)?;
                    }
                    self.indent -= 1;
                }
                ElseBranch::ElseIf(elif) => self.emit_geometry_elif(elif, geo_name)?,
            }
        }

        Ok(())
    }

    fn emit_polygon_py(&mut self, name: &str, c: &ConstructorExpr) -> Result<(), CompileError> {
        if let Some(Expr::Array(arr)) = self.find_named_arg_expr(&c.args, "points") {
            if arr.elements.len() < 3 {
                return Err(CompileError::Internal(
                    "Polygon requires at least 3 points".into(),
                ));
            }
            {
                let mut pt_vars = Vec::new();
                for pt in &arr.elements {
                    if let Expr::Array(coords) = pt {
                        let vals: Vec<String> = coords
                            .elements
                            .iter()
                            .map(|e| self.emit_expr(e))
                            .collect::<Result<_, _>>()?;
                        let pv = self.fresh_var("pt");
                        let (x, y, z) = match vals.len() {
                            2 => (vals[0].clone(), vals[1].clone(), "0".to_string()),
                            3 => (vals[0].clone(), vals[1].clone(), vals[2].clone()),
                            _ => {
                                return Err(CompileError::Internal(
                                    "point must have 2 or 3 coords".into(),
                                ));
                            }
                        };
                        self.line(&format!("{} = occ.addPoint({}, {}, {})", pv, x, y, z));
                        pt_vars.push(pv);
                    }
                }
                let mut line_vars = Vec::new();
                for i in 0..pt_vars.len() {
                    let next = (i + 1) % pt_vars.len();
                    let lv = self.fresh_var("ln");
                    self.line(&format!(
                        "{} = occ.addLine({}, {})",
                        lv, pt_vars[i], pt_vars[next]
                    ));
                    line_vars.push(lv);
                }
                let cl_var = self.fresh_var("cl");
                self.line(&format!(
                    "{} = occ.addCurveLoop([{}])",
                    cl_var,
                    line_vars.join(", ")
                ));
                self.line(&format!("{} = occ.addPlaneSurface([{}])", name, cl_var));
                self.polygon_wire_var.insert(name.to_string(), cl_var);
                return Ok(());
            }
        }
        Err(CompileError::Internal(
            "Polygon requires 'points' array".into(),
        ))
    }

    fn register_pipe_base_topo(&mut self, qname: &str, pipe: &PipeExpr) {
        match pipe.value.as_ref() {
            Expr::Constructor(c) => match c.type_name.name.as_str() {
                "Box" => self.topo.register_box(qname, 0),
                "Cylinder" => self.topo.register_cylinder(qname, 0),
                "Sphere" => self.topo.register_sphere(qname, 0),
                _ => self.topo.register_unknown(qname, 0),
            },
            Expr::Pipe(inner) => self.register_pipe_base_topo(qname, inner),
            _ => self.topo.register_unknown(qname, 0),
        }
    }

    fn expr_name_best_effort(&self, expr: &Expr, geo_name: &str) -> String {
        match expr {
            Expr::Ident(id) => format!("{}.{}", geo_name, id.name),
            Expr::FieldAccess(fa) => {
                let base = self.expr_name_best_effort(&fa.object, geo_name);
                format!("{}.{}", base, fa.field.name)
            }
            _ => String::new(),
        }
    }

    fn emit_constructor_call(&self, c: &ConstructorExpr) -> Result<String, CompileError> {
        match c.type_name.name.as_str() {
            "Box" => {
                let args = self.collect_args(&c.args, 3)?;
                Ok(format!(
                    "occ.addBox(0, 0, 0, {}, {}, {})",
                    args[0], args[1], args[2]
                ))
            }
            "Sphere" => {
                let r = self.find_named_arg(&c.args, "r")?;
                Ok(format!("occ.addSphere(0, 0, 0, {})", r))
            }
            "Cylinder" => {
                let r = self.find_named_arg(&c.args, "r")?;
                let h = self.find_named_arg(&c.args, "h")?;
                Ok(format!("occ.addCylinder(0, 0, 0, 0, 0, {}, {})", h, r))
            }
            "Cone" => {
                let r1 = self.find_named_arg(&c.args, "r1")?;
                let r2 = self.find_named_arg(&c.args, "r2")?;
                let h = self.find_named_arg(&c.args, "h")?;
                Ok(format!("occ.addCone(0, 0, 0, 0, 0, {}, {}, {})", h, r1, r2))
            }
            "Torus" => {
                let r_major = self.find_named_arg(&c.args, "r_major")?;
                let r_minor = self.find_named_arg(&c.args, "r_minor")?;
                Ok(format!("occ.addTorus(0, 0, 0, {}, {})", r_major, r_minor))
            }
            "Wedge" => {
                let args = self.collect_args(&c.args, 4)?;
                Ok(format!(
                    "occ.addWedge(0, 0, 0, {}, {}, {}, ltx={})",
                    args[0], args[1], args[2], args[3]
                ))
            }
            "Rectangle" => {
                let args = self.collect_args(&c.args, 2)?;
                Ok(format!(
                    "occ.addRectangle(0, 0, 0, {}, {})",
                    args[0], args[1]
                ))
            }
            "Disk" => {
                let r = self.find_named_arg(&c.args, "r")?;
                Ok(format!("occ.addDisk(0, 0, 0, {r}, {r})"))
            }
            "Polygon" => {
                // Polygon is handled separately in emit_polygon_py (needs &mut self)
                Err(CompileError::Internal(
                    "Polygon must be emitted via emit_polygon_py".into(),
                ))
            }
            "Point" => {
                // Point(x, y, z) — standalone 3D point
                let args = self.collect_args(&c.args, 3)?;
                Ok(format!(
                    "occ.addPoint({}, {}, {})",
                    args[0], args[1], args[2]
                ))
            }
            "Line" => {
                let from = self.find_named_arg(&c.args, "from")?;
                let to = self.find_named_arg(&c.args, "to")?;
                Ok(format!("occ.addLine({}, {})", from, to))
            }
            "Arc" => {
                let args = self.collect_args(&c.args, 3)?;
                Ok(format!(
                    "occ.addCircleArc({}, {}, {})",
                    args[0], args[1], args[2]
                ))
            }
            "Spline" => {
                let args = self.collect_args(&c.args, c.args.len())?;
                Ok(format!("occ.addSpline([{}])", args.join(", ")))
            }
            "BSpline" => {
                let args = self.collect_args(&c.args, c.args.len())?;
                Ok(format!("occ.addBSpline([{}])", args.join(", ")))
            }
            "Circle" => {
                // Circle(r) — alias for Disk
                let r = self.find_named_arg(&c.args, "r")?;
                Ok(format!("occ.addDisk(0, 0, 0, {r}, {r})"))
            }
            "Loft" => {
                // Loft(sections: [s1, s2, ...], ruled: bool) → occ.addThruSections
                // `ruled: true` requests linear interpolation between
                // sections (no spline overshoot through corner trajectories).
                //
                // OCC `addThruSections` expects WIRE tags. `Polygon` sections
                // are emitted as `addPlaneSurface` whose underlying wire is
                // recorded in `polygon_wire_var`; `resolve_section_tag_py`
                // prefers that wire over the surface var so the call lands in
                // a well-defined OCC operation.
                let ruled = self.find_named_arg_bool(&c.args, "ruled").unwrap_or(false);
                let suffix = if ruled {
                    ", makeSolid=True, makeRuled=True"
                } else {
                    ""
                };
                if let Some(sections_expr) = self.find_named_arg_expr(&c.args, "sections") {
                    if let Expr::Array(arr) = sections_expr {
                        let section_tags: Vec<String> = arr
                            .elements
                            .iter()
                            .map(|e| self.resolve_section_tag_py(e))
                            .collect::<Result<_, _>>()?;
                        Ok(format!(
                            "occ.addThruSections([{}]{})",
                            section_tags.join(", "),
                            suffix
                        ))
                    } else {
                        Err(CompileError::Internal(
                            "Loft 'sections' must be an array".into(),
                        ))
                    }
                } else {
                    let section_tags: Vec<String> = c
                        .args
                        .iter()
                        .map(|a| self.resolve_section_tag_py(&a.value))
                        .collect::<Result<_, _>>()?;
                    Ok(format!(
                        "occ.addThruSections([{}]{})",
                        section_tags.join(", "),
                        suffix
                    ))
                }
            }
            other => Err(CompileError::Unsupported(format!(
                "constructor '{}'",
                other
            ))),
        }
    }

    fn emit_pipe_chain_py(&mut self, name: &str, pipe: &PipeExpr) -> Result<GmshDim, CompileError> {
        // Emit the base expression
        let mut dim = match pipe.value.as_ref() {
            Expr::Constructor(c) => {
                let call = self.emit_constructor_call(c)?;
                self.line(&format!("{} = {}", name, call));
                constructor_dim(c)
            }
            Expr::Pipe(inner) => self.emit_pipe_chain_py(name, inner)?,
            Expr::Ident(id) => {
                let resolved = self.resolve_scoped_entity_var_py(&id.name);
                let val = if resolved != id.name {
                    resolved
                } else {
                    self.emit_expr(&pipe.value)?
                };
                self.line(&format!("{} = {}", name, val));
                self.var_to_dim
                    .get(&id.name)
                    .copied()
                    .unwrap_or(GmshDim::Volume)
            }
            _ => {
                let val = if let Expr::FieldAccess(field) = pipe.value.as_ref() {
                    let object = self.emit_expr(&field.object)?;
                    let path = format!("{object}.{}", field.field.name);
                    let resolved = self.resolve_scoped_entity_var_py(&path);
                    if resolved != path {
                        resolved
                    } else {
                        self.emit_expr(&pipe.value)?
                    }
                } else if let Some(path) = entity_path(&pipe.value) {
                    let resolved = self.resolve_scoped_entity_var_py(&path);
                    if resolved != path {
                        resolved
                    } else {
                        self.emit_expr(&pipe.value)?
                    }
                } else {
                    self.emit_expr(&pipe.value)?
                };
                self.line(&format!("{} = {}", name, val));
                GmshDim::Volume
            }
        };

        // Apply the transform
        match pipe.function.name.as_str() {
            "translate" => {
                let args = self.collect_args(&pipe.args, 3)?;
                self.line(&format!(
                    "occ.translate(_ml_dimtags({}, {}), {}, {}, {})",
                    dim as i32, name, args[0], args[1], args[2]
                ));
            }
            "rotate" => {
                let angle = self.angle_arg_degrees_to_radians(&pipe.args)?;
                let axis_str = self.find_named_arg_str(&pipe.args, "axis")?;
                let (ax, ay, az) = match axis_str.as_str() {
                    "X" => ("1".to_string(), "0".to_string(), "0".to_string()),
                    "Y" => ("0".to_string(), "1".to_string(), "0".to_string()),
                    "Z" => ("0".to_string(), "0".to_string(), "1".to_string()),
                    _ => {
                        let axis_expr = self.find_named_arg_expr(&pipe.args, "axis");
                        match axis_expr {
                            Some(Expr::Array(arr)) if arr.elements.len() == 3 => {
                                let ax = self.emit_expr(&arr.elements[0])?;
                                let ay = self.emit_expr(&arr.elements[1])?;
                                let az = self.emit_expr(&arr.elements[2])?;
                                (ax, ay, az)
                            }
                            _ => ("0".to_string(), "0".to_string(), "1".to_string()),
                        }
                    }
                };
                self.line(&format!(
                    "occ.rotate(_ml_dimtags({}, {}), 0, 0, 0, {}, {}, {}, {})",
                    dim as i32, name, ax, ay, az, angle
                ));
            }
            "scale" => {
                // scale(factor) for uniform, scale(sx, sy, sz) for non-uniform
                let (sx, sy, sz) = if let Ok(args) = self.collect_args(&pipe.args, 3) {
                    (args[0].clone(), args[1].clone(), args[2].clone())
                } else {
                    let args = self.collect_args(&pipe.args, 1)?;
                    (args[0].clone(), args[0].clone(), args[0].clone())
                };
                self.line(&format!(
                    "occ.dilate(_ml_dimtags({}, {}), 0, 0, 0, {}, {}, {})",
                    dim as i32, name, sx, sy, sz
                ));
            }
            "mirror" => {
                // mirror(nx, ny, nz) — reflect through plane nx*x + ny*y + nz*z = 0
                // occ.mirror(dimTags, a, b, c, d) for plane ax+by+cz+d=0
                let args = self.collect_args(&pipe.args, 3)?;
                self.line(&format!(
                    "occ.mirror(_ml_dimtags({}, {}), {}, {}, {}, 0)",
                    dim as i32, name, args[0], args[1], args[2]
                ));
            }
            "affine" => {
                let matrix = self.affine_matrix_args(&pipe.args)?;
                self.line(&format!(
                    "occ.affineTransform(_ml_dimtags({}, {}), [{}])",
                    dim as i32,
                    name,
                    matrix.join(", ")
                ));
            }
            "extrude" => {
                let args = self.collect_args(&pipe.args, 3)?;
                let result_var = self.fresh_var("extrude");
                self.line(&format!(
                    "{} = occ.extrude(_ml_dimtags(2, {}), {}, {}, {})",
                    result_var, name, args[0], args[1], args[2]
                ));
                self.line(&format!("{} = _ml_only(3, {})", name, result_var));
                dim = GmshDim::Volume;
            }
            "revolve" => {
                let angle = self.angle_arg_degrees_to_radians(&pipe.args)?;
                let axis_str = self.find_named_arg_str(&pipe.args, "axis")?;
                let (ax, ay, az) = match axis_str.as_str() {
                    "X" => ("1", "0", "0"),
                    "Y" => ("0", "1", "0"),
                    "Z" => ("0", "0", "1"),
                    _ => ("0", "0", "1"),
                };
                let result_var = self.fresh_var("revolve");
                self.line(&format!(
                    "{} = occ.revolve(_ml_dimtags(2, {}), 0, 0, 0, {}, {}, {}, {})",
                    result_var, name, ax, ay, az, angle
                ));
                self.line(&format!("{} = _ml_only(3, {})", name, result_var));
                dim = GmshDim::Volume;
            }
            "sweep" => {
                // sweep(path: curve_entity) — pipe extrude surface along a wire path
                let path_str = self.find_named_arg_str(&pipe.args, "path")?;
                let wire_var = self.fresh_var("wire");
                let result_var = self.fresh_var("pipe");
                self.line(&format!(
                    "{} = occ.addWire(_ml_tags({}))",
                    wire_var, path_str
                ));
                self.line(&format!(
                    "{} = occ.addPipe(_ml_dimtags(2, {}), {})",
                    result_var, name, wire_var
                ));
                self.line(&format!("{} = _ml_only(3, {})", name, result_var));
                dim = GmshDim::Volume;
            }
            "copy" => {
                // copy() — duplicate the entity
                let copy_var = self.fresh_var("copy");
                self.line(&format!(
                    "{} = occ.copy(_ml_dimtags({}, {}))",
                    copy_var, dim as i32, name
                ));
                self.line(&format!(
                    "{} = _ml_only({}, {})",
                    name, dim as i32, copy_var
                ));
            }
            "fillet" => {
                // fillet(r) — fillet all edges of the volume
                let r = self.find_named_arg(&pipe.args, "r")?;
                self.line("occ.synchronize()");
                let edges_var = self.fresh_var("edges");
                self.line(&format!(
                    "{} = gmsh.model.getBoundary(_ml_boundary(_ml_dimtags(3, {}), combined=False, oriented=False), combined=False, oriented=False)",
                    edges_var, name
                ));
                let edge_tags_var = self.fresh_var("edge_tags");
                self.line(&format!(
                    "{} = [abs(e[1]) for e in {}]",
                    edge_tags_var, edges_var
                ));
                let result_var = self.fresh_var("fillet");
                self.line(&format!(
                    "{} = occ.fillet(_ml_dimtags(3, {}), {}, [{}])",
                    result_var, name, edge_tags_var, r
                ));
                self.line(&format!("{} = _ml_only(3, {})", name, result_var));
            }
            "chamfer" => {
                // chamfer(d) — chamfer all edges of the volume
                let d = self.find_named_arg(&pipe.args, "d")?;
                self.line("occ.synchronize()");
                let edges_var = self.fresh_var("edges");
                self.line(&format!(
                    "{} = gmsh.model.getBoundary(_ml_boundary(_ml_dimtags(3, {}), combined=False, oriented=False), combined=False, oriented=False)",
                    edges_var, name
                ));
                let edge_tags_var = self.fresh_var("edge_tags");
                self.line(&format!(
                    "{} = [abs(e[1]) for e in {}]",
                    edge_tags_var, edges_var
                ));
                // chamfer needs curves, surfaces, distances — use first adjacent surface for each edge
                let faces_var = self.fresh_var("faces");
                self.line(&format!(
                    "{} = [abs(f[1]) for f in _ml_boundary(_ml_dimtags(3, {}), combined=False, oriented=False)]",
                    faces_var, name
                ));
                let result_var = self.fresh_var("chamfer");
                self.line(&format!(
                    "{} = occ.chamfer(_ml_dimtags(3, {}), {}, {}[:len({})], [{}])",
                    result_var, name, edge_tags_var, faces_var, edge_tags_var, d
                ));
                self.line(&format!("{} = _ml_only(3, {})", name, result_var));
            }
            "shell" => {
                // shell(thickness) — hollow a solid by offsetting inward
                let thickness = self.find_named_arg(&pipe.args, "thickness")?;
                // addThickSolid(volumeTag, excludeSurfaceTags, offset)
                // With no excluded faces, hollows the entire solid
                let result_var = self.fresh_var("shell");
                self.line(&format!(
                    "{} = occ.addThickSolid(_ml_single({}, 'shell'), [], {})",
                    result_var, name, thickness
                ));
                self.line(&format!("{} = _ml_only(3, {})", name, result_var));
            }
            "fragment" => {
                // fragment(tool) or fragment([t1, t2, ...]) — BooleanFragments
                let tool_arg = &pipe.args[0].value;
                let tool_dimtags: String = match tool_arg {
                    Expr::Array(arr) => {
                        let names: Vec<String> = arr
                            .elements
                            .iter()
                            .map(|e| self.ensure_entity_py(e))
                            .collect::<Result<Vec<_>, _>>()?;
                        names
                            .iter()
                            .map(|n| format!("_ml_dimtags(3, {})", n))
                            .collect::<Vec<_>>()
                            .join(" + ")
                    }
                    _ => {
                        let tool_name = self.ensure_entity_py(tool_arg)?;
                        format!("_ml_dimtags(3, {})", tool_name)
                    }
                };
                let result_var = self.fresh_var("frag");
                self.line(&format!(
                    "{}_result = occ.fragment(_ml_dimtags(3, {}), {})",
                    result_var, name, tool_dimtags
                ));
                self.line(&format!("{} = _ml_only(3, {}_result[0])", name, result_var));
            }
            "twist" => {
                // twist(dx, dy, dz, axis, angle) — combined translate + rotate extrusion
                let dx = self
                    .find_named_arg(&pipe.args, "dx")
                    .unwrap_or_else(|_| "0".into());
                let dy = self
                    .find_named_arg(&pipe.args, "dy")
                    .unwrap_or_else(|_| "0".into());
                let dz = self
                    .find_named_arg(&pipe.args, "dz")
                    .unwrap_or_else(|_| "0".into());
                let angle = self.angle_arg_degrees_to_radians(&pipe.args)?;
                let axis_str = self
                    .find_named_arg_str(&pipe.args, "axis")
                    .unwrap_or_else(|_| "Z".into());
                let (ax, ay, az) = match axis_str.as_str() {
                    "X" => ("1", "0", "0"),
                    "Y" => ("0", "1", "0"),
                    "Z" => ("0", "0", "1"),
                    _ => ("0", "0", "1"),
                };
                let result_var = self.fresh_var("twist");
                self.line(&format!(
                    "{} = occ.twist(_ml_dimtags(2, {}), 0, 0, 0, {}, {}, {}, {}, {}, {}, {})",
                    result_var, name, dx, dy, dz, ax, ay, az, angle
                ));
                self.line(&format!("{} = _ml_only(3, {})", name, result_var));
                dim = GmshDim::Volume;
            }
            "coherence" => {
                self.line("occ.removeAllDuplicates()");
            }
            "project_iso_surface" | "project_signed_distance" | "project_closest_point" => {
                // Python/Gmsh has no portable field-evaluation contract.
                // Keep the source binding and emit only an explicit metadata
                // marker; the canonical sidecar carries the consumer request.
                self.line(&format!(
                    "# MeshLang projection {name}: metadata-only {}",
                    pipe.function.name
                ));
            }
            other => {
                return Err(CompileError::Internal(format!(
                    "unknown transform '{}'",
                    other
                )));
            }
        }

        Ok(dim)
    }

    fn emit_csg_py(&mut self, name: &str, b: &BinaryExpr) -> Result<(), CompileError> {
        let same_op = |lhs: BinaryOp, rhs: BinaryOp| {
            matches!(
                (lhs, rhs),
                (
                    BinaryOp::Sub | BinaryOp::CsgDifference,
                    BinaryOp::Sub | BinaryOp::CsgDifference
                ) | (BinaryOp::CsgUnion, BinaryOp::CsgUnion)
                    | (BinaryOp::CsgIntersect, BinaryOp::CsgIntersect)
            )
        };

        let op = b.op;
        let mut tool_exprs = vec![&*b.right];
        let mut base_expr = &*b.left;
        while let Expr::Binary(inner) = base_expr {
            if same_op(inner.op, op) && self.is_csg_binary(inner) {
                tool_exprs.push(&*inner.right);
                base_expr = &*inner.left;
            } else {
                break;
            }
        }
        tool_exprs.reverse();

        let left_dim = self.expr_gmsh_dim(base_expr).ok_or_else(|| {
            CompileError::Internal("cannot determine Python CSG operand dimension".into())
        })?;
        let left_var = self.ensure_entity_py(base_expr)?;
        let mut tool_tags = Vec::with_capacity(tool_exprs.len());
        for tool_expr in tool_exprs {
            let tool_dim = self.expr_gmsh_dim(tool_expr).ok_or_else(|| {
                CompileError::Internal("cannot determine Python CSG tool dimension".into())
            })?;
            if left_dim != tool_dim {
                return Err(CompileError::TypeError {
                    message: format!(
                        "CSG operands must same dimension ({} and {})",
                        left_dim as u8, tool_dim as u8
                    ),
                });
            }
            let tool_var = self.ensure_entity_py(tool_expr)?;
            tool_tags.push(format!("_ml_dimtags({}, {})", tool_dim as u8, tool_var));
        }

        let method = match op {
            BinaryOp::Sub | BinaryOp::CsgDifference => "cut",
            BinaryOp::CsgUnion => "fuse",
            BinaryOp::CsgIntersect => "intersect",
            _ => return Err(CompileError::Internal("non-CSG op".into())),
        };
        let result_var = self.fresh_var("csg");
        self.line(&format!(
            "{}_result = occ.{}(_ml_dimtags({}, {}), {})",
            result_var,
            method,
            left_dim as u8,
            left_var,
            tool_tags.join(" + ")
        ));
        self.line(&format!(
            "{} = _ml_only({}, {}_result[0])",
            name, left_dim as u8, result_var
        ));
        self.var_to_dim.insert(name.to_string(), left_dim);
        Ok(())
    }

    fn expr_gmsh_dim(&self, expr: &Expr) -> Option<GmshDim> {
        match expr {
            Expr::Ident(id) => self
                .var_to_dim
                .get(&id.name)
                .copied()
                .or_else(|| self.var_to_dim.get(&id.name.replace('.', "__")).copied()),
            Expr::FieldAccess(field) => self
                .resolve_entity_var_py(expr)
                .ok()
                .and_then(|var| self.var_to_dim.get(&var).copied())
                .or(match field.field.name.as_str() {
                    "faces" | "top" | "bottom" | "front" | "back" | "left" | "right"
                    | "lateral" | "outer" | "inner" => Some(GmshDim::Surface),
                    "edges" => Some(GmshDim::Curve),
                    "vertices" => Some(GmshDim::Point),
                    _ => None,
                }),
            Expr::Constructor(constructor) => Some(constructor_dim(constructor)),
            Expr::Pipe(pipe) => match pipe.function.name.as_str() {
                "extrude" | "revolve" | "sweep" => Some(GmshDim::Volume),
                _ => self.expr_gmsh_dim(&pipe.value),
            },
            Expr::Binary(binary) if self.is_csg_binary(binary) => self.expr_gmsh_dim(&binary.left),
            Expr::Call(call) if self.functions.contains_key(&call.function.name) => self
                .try_inline_call(call)
                .ok()
                .and_then(|inlined| self.expr_gmsh_dim(&inlined)),
            Expr::Paren(inner) => self.expr_gmsh_dim(&inner.value),
            _ => None,
        }
    }

    /// Ensure an expression is available as a Python variable and return its name.
    /// For identifiers, returns the variable name directly.
    /// For nested CSG/pipe/constructors, emits intermediate code and returns a temp var.
    fn ensure_entity_py(&mut self, expr: &Expr) -> Result<String, CompileError> {
        match expr {
            Expr::Ident(id) => Ok(id.name.clone()),
            Expr::Constructor(c) => {
                let tmp = self.fresh_var("shape");
                let call = self.emit_constructor_call(c)?;
                self.line(&format!("{} = {}", tmp, call));
                self.var_to_dim.insert(tmp.clone(), constructor_dim(c));
                Ok(tmp)
            }
            Expr::Pipe(p) => {
                let tmp = self.fresh_var("pipe");
                let dim = self.emit_pipe_chain_py(&tmp, p)?;
                self.var_to_dim.insert(tmp.clone(), dim);
                Ok(tmp)
            }
            Expr::Binary(b) if self.is_csg_binary(b) => {
                let tmp = self.fresh_var("csg_tmp");
                self.emit_csg_py(&tmp, b)?;
                Ok(tmp)
            }
            Expr::Call(c) if self.functions.contains_key(&c.function.name) => {
                let inlined = self.try_inline_call(c)?;
                self.ensure_entity_py(&inlined)
            }
            _ => self.emit_expr(expr),
        }
    }

    fn is_csg_binary(&self, b: &BinaryExpr) -> bool {
        match b.op {
            BinaryOp::CsgUnion | BinaryOp::CsgDifference | BinaryOp::CsgIntersect => true,
            BinaryOp::Sub => {
                self.is_csg_operand_expr(&b.left) || self.is_csg_operand_expr(&b.right)
            }
            _ => false,
        }
    }

    fn is_csg_operand_expr(&self, expr: &Expr) -> bool {
        match expr {
            Expr::Ident(id) => {
                let qname = if let Some(ref geo) = self.current_geometry {
                    format!("{}.{}", geo, id.name)
                } else {
                    id.name.clone()
                };
                self.topo.lookup(&qname).is_some()
            }
            Expr::FieldAccess(_) => self
                .current_geometry
                .as_deref()
                .map(|geo| self.expr_name_best_effort(expr, geo))
                .is_some_and(|name| self.topo.lookup(&name).is_some()),
            Expr::Constructor(c) => matches!(
                c.type_name.name.as_str(),
                "Box"
                    | "Sphere"
                    | "Cylinder"
                    | "Cone"
                    | "Torus"
                    | "Wedge"
                    | "Rectangle"
                    | "Disk"
                    | "Circle"
                    | "Polygon"
            ),
            Expr::Pipe(_) => true,
            Expr::Binary(b) => self.is_csg_binary(b),
            Expr::Call(c) if self.functions.contains_key(&c.function.name) => self
                .try_inline_call(c)
                .is_ok_and(|inlined| self.is_csg_operand_expr(&inlined)),
            Expr::Paren(inner) => self.is_csg_operand_expr(&inner.value),
            _ => false,
        }
    }

    fn emit_mesh(&mut self, mesh: &MeshBlock) -> Result<(), CompileError> {
        self.line("# mesh control");
        self.line("occ.synchronize()");

        for stmt in &mesh.statements {
            match stmt {
                MeshStatement::KeyValue(kv) => {
                    let val = self.emit_expr(&kv.value)?;
                    match kv.key.name.as_str() {
                        "default_size" => {
                            self.line(&format!(
                                "gmsh.option.setNumber('Mesh.CharacteristicLengthMax', {})",
                                val
                            ));
                        }
                        "min_size" => {
                            self.line(&format!(
                                "gmsh.option.setNumber('Mesh.CharacteristicLengthMin', {})",
                                val
                            ));
                        }
                        "order" => {
                            self.line(&format!(
                                "gmsh.option.setNumber('Mesh.ElementOrder', {})",
                                val
                            ));
                        }
                        "smoothing_steps" => {
                            self.line(&format!("gmsh.option.setNumber('Mesh.Smoothing', {})", val));
                        }
                        "algorithm_2d" => {
                            let algo = self.mesh_algo_2d_py(&kv.value)?;
                            self.line(&format!(
                                "gmsh.option.setNumber('Mesh.Algorithm', {})",
                                algo
                            ));
                        }
                        "algorithm_3d" => {
                            let algo = self.mesh_algo_3d_py(&kv.value)?;
                            self.line(&format!(
                                "gmsh.option.setNumber('Mesh.Algorithm3D', {})",
                                algo
                            ));
                        }
                        "optimize" => {
                            self.line("gmsh.option.setNumber('Mesh.Optimize', 1)");
                        }
                        "recombine" => {
                            if val == "True" || val == "1" {
                                self.line("gmsh.option.setNumber('Mesh.RecombineAll', 1)");
                            }
                        }
                        "optimize_threshold" => {
                            self.line(&format!(
                                "gmsh.option.setNumber('Mesh.OptimizeThreshold', {})",
                                val
                            ));
                        }
                        "optimize_netgen" => {
                            self.line("gmsh.option.setNumber('Mesh.OptimizeNetgen', 1)");
                        }
                        "max_size" => {
                            self.line(&format!(
                                "gmsh.option.setNumber('Mesh.CharacteristicLengthMax', {})",
                                val
                            ));
                        }
                        "size_from_curvature" => {
                            self.line(&format!(
                                "gmsh.option.setNumber('Mesh.MeshSizeFromCurvature', {})",
                                val
                            ));
                        }
                        "size_factor" => {
                            self.line(&format!(
                                "gmsh.option.setNumber('Mesh.MeshSizeFactor', {})",
                                val
                            ));
                        }
                        "second_order_incomplete" => {
                            self.line("gmsh.option.setNumber('Mesh.SecondOrderIncomplete', 1)");
                        }
                        "subdivision" => {
                            self.line(&format!(
                                "gmsh.option.setNumber('Mesh.SubdivisionAlgorithm', {})",
                                val
                            ));
                        }
                        "aniso_max" => {
                            self.line(&format!("gmsh.option.setNumber('Mesh.AnisoMax', {})", val));
                        }
                        "recombine_algorithm" => {
                            let algo = match val.as_str() {
                                "simple" => "0",
                                "blossom" => "1",
                                "simple_full_quad" => "2",
                                "blossom_full_quad" => "3",
                                _ => &val,
                            };
                            self.line(&format!(
                                "gmsh.option.setNumber('Mesh.RecombinationAlgorithm', {})",
                                algo
                            ));
                        }
                        "binary" => {
                            self.line("gmsh.option.setNumber('Mesh.Binary', 1)");
                        }
                        "save_all" => {
                            self.line("gmsh.option.setNumber('Mesh.SaveAll', 1)");
                        }
                        "target_frequency" | "medium_speed" | "elements_per_wavelength" => {
                            // Handled below in frequency-aware sizing
                        }
                        other => {
                            return Err(CompileError::Unsupported(format!(
                                "mesh option '{}'",
                                other
                            )));
                        }
                    }
                }
                MeshStatement::Refine(r) => {
                    self.emit_refine_py(r)?;
                }
                MeshStatement::BoundaryLayer(bl) => {
                    self.emit_boundary_layer_py(bl)?;
                }
                MeshStatement::Transfinite(t) => {
                    self.emit_transfinite_py(t)?;
                }
                MeshStatement::Embed(e) => {
                    self.emit_embed_py(e)?;
                }
                MeshStatement::Periodic(p) => {
                    self.emit_periodic_py(p)?;
                }
                MeshStatement::Compound(c) => {
                    self.emit_compound_py(c)?;
                }
                MeshStatement::Metric(metric) => self.emit_mesh_metric_py(metric)?,
            }
        }

        // Frequency-aware mesh sizing
        let mut target_freq: Option<String> = None;
        let mut medium_speed: Option<String> = None;
        let mut epw: Option<String> = None;
        let mut has_default_size = false;

        for stmt in &mesh.statements {
            if let MeshStatement::KeyValue(kv) = stmt {
                match kv.key.name.as_str() {
                    "target_frequency" => target_freq = Some(self.emit_expr(&kv.value)?),
                    "medium_speed" => medium_speed = Some(self.emit_expr(&kv.value)?),
                    "elements_per_wavelength" => epw = Some(self.emit_expr(&kv.value)?),
                    "default_size" => has_default_size = true,
                    _ => {}
                }
            }
        }

        if let (Some(freq), Some(speed)) = (target_freq, medium_speed) {
            if !has_default_size {
                let epw_val = epw.unwrap_or_else(|| "6".to_string());
                self.line(&format!(
                    "# frequency-aware sizing: {} / {} / {}",
                    speed, freq, epw_val
                ));
                self.line(&format!(
                    "gmsh.option.setNumber('Mesh.CharacteristicLengthMax', {} / {} / {})",
                    speed, freq, epw_val
                ));
            }
        }

        // Compose all refine fields into a Min field and set as background
        if !self.refine_field_vars.is_empty() {
            let min_var = self.fresh_var("bg");
            let mut field_vars = self.refine_field_vars.clone();
            if let Some(previous) = &self.background_field_var {
                field_vars.insert(0, previous.clone());
            }
            self.line(&format!("{} = gmsh.model.mesh.field.add('Min')", min_var));
            self.line(&format!(
                "gmsh.model.mesh.field.setNumbers({}, 'FieldsList', [{}])",
                min_var,
                field_vars.join(", ")
            ));
            self.line(&format!(
                "gmsh.model.mesh.field.setAsBackgroundMesh({})",
                min_var
            ));
            self.background_field_var = Some(min_var);
            self.refine_field_vars.clear();
        }

        self.line("");
        Ok(())
    }

    fn emit_refine_py(&mut self, refine: &RefineStatement) -> Result<(), CompileError> {
        match &refine.kind {
            RefineKind::Near { entity, args } => {
                let entity_str = self.emit_expr(entity)?;
                self.line(&format!("# refine near {}", entity_str));

                let field_var = self.fresh_var("field");
                self.line(&format!(
                    "{}_dist = gmsh.model.mesh.field.add('Distance')",
                    field_var
                ));

                // Attach entity to the Distance field
                self.line(&format!(
                "gmsh.model.mesh.field.setNumbers({}_dist, 'SurfacesList', [abs(tag) for _, tag in _ml_boundary(_ml_dimtags(3, {}), combined=False, oriented=False)])",
                    field_var, entity_str
                ));

                let mut size_val_str: Option<String> = None;
                for kv in &refine.body {
                    if kv.key.name == "size" {
                        size_val_str = Some(self.emit_expr(&kv.value)?);
                    }
                }
                if let Some(ref size_val) = size_val_str {
                    self.line(&format!(
                        "{}_thresh = gmsh.model.mesh.field.add('Threshold')",
                        field_var
                    ));
                    self.line(&format!(
                        "gmsh.model.mesh.field.setNumber({0}_thresh, 'InField', {0}_dist)",
                        field_var
                    ));
                    self.line(&format!(
                        "gmsh.model.mesh.field.setNumber({}_thresh, 'SizeMin', {})",
                        field_var, size_val
                    ));

                    for arg in args {
                        if arg.key.name == "distance" {
                            let dist = self.emit_expr(&arg.value)?;
                            self.line(&format!(
                                "gmsh.model.mesh.field.setNumber({}_thresh, 'DistMax', {})",
                                field_var, dist
                            ));
                        }
                    }

                    // SizeMax from growth_rate or default
                    let mut has_growth_rate = false;
                    for kv in &refine.body {
                        if kv.key.name == "growth_rate" {
                            let gr = self.emit_expr(&kv.value)?;
                            self.line(&format!(
                                "gmsh.model.mesh.field.setNumber({}_thresh, 'SizeMax', {} * {})",
                                field_var, size_val, gr
                            ));
                            has_growth_rate = true;
                        }
                    }
                    if !has_growth_rate {
                        self.line(&format!(
                            "gmsh.model.mesh.field.setNumber({}_thresh, 'SizeMax', {} * 10)",
                            field_var, size_val
                        ));
                    }

                    let thresh_var = format!("{}_thresh", field_var);
                    self.refine_field_vars.push(thresh_var);
                }
            }
            RefineKind::On { entity } => {
                let entity_str = self.emit_expr(entity)?;
                self.line(&format!("# refine on {}", entity_str));
                for kv in &refine.body {
                    match kv.key.name.as_str() {
                        "size" => {
                            let size_val = self.emit_expr(&kv.value)?;
                            // Use gmsh.model.mesh.setSize on entity boundary points
                            self.line(&format!(
                    "gmsh.model.mesh.setSize(_ml_boundary(_ml_dimtags(2, {}), combined=False, oriented=False, recursive=True), {})",
                                entity_str, size_val
                            ));
                        }
                        "order" => {
                            let order_val = self.emit_expr(&kv.value)?;
                            // Per-surface mesh order: set element order for this surface
                            self.line(&format!(
                                "gmsh.option.setNumber('Mesh.ElementOrder', {})  # surface {}",
                                order_val, entity_str
                            ));
                        }
                        _ => {}
                    }
                }
            }
            RefineKind::In { region } => {
                let bounds = self.box_region_bounds_py(region)?;
                self.line("# refine in translated Box region");

                let field_var = self.fresh_var("field");
                self.line(&format!(
                    "{}_box = gmsh.model.mesh.field.add('Box')",
                    field_var
                ));
                for (name, value) in [
                    ("XMin", &bounds[0]),
                    ("XMax", &bounds[1]),
                    ("YMin", &bounds[2]),
                    ("YMax", &bounds[3]),
                    ("ZMin", &bounds[4]),
                    ("ZMax", &bounds[5]),
                ] {
                    self.line(&format!(
                        "gmsh.model.mesh.field.setNumber({}_box, '{}', {})",
                        field_var, name, value
                    ));
                }

                for kv in &refine.body {
                    match kv.key.name.as_str() {
                        "size" => {
                            let size_val = self.emit_expr(&kv.value)?;
                            self.line(&format!(
                                "gmsh.model.mesh.field.setNumber({}_box, 'VIn', {})",
                                field_var, size_val
                            ));
                        }
                        "size_out" => {
                            let size_val = self.emit_expr(&kv.value)?;
                            self.line(&format!(
                                "gmsh.model.mesh.field.setNumber({}_box, 'VOut', {})",
                                field_var, size_val
                            ));
                        }
                        _ => {}
                    }
                }
                let box_var = format!("{}_box", field_var);
                self.refine_field_vars.push(box_var);
            }
            RefineKind::Field { args } => {
                self.line("# refine field");
                let field_var = self.fresh_var("field");
                self.line(&format!(
                    "{}_dist = gmsh.model.mesh.field.add('Distance')",
                    field_var
                ));

                // Set distance_from entity if specified
                for kv in args {
                    if kv.key.name == "distance_from" {
                        let entity_str = self.emit_expr(&kv.value)?;
                        self.line(&format!(
                            "gmsh.model.mesh.field.setNumbers({}_dist, 'SurfacesList', [{}])",
                            field_var, entity_str
                        ));
                    }
                }

                // Threshold field
                self.line(&format!(
                    "{}_thresh = gmsh.model.mesh.field.add('Threshold')",
                    field_var
                ));
                self.line(&format!(
                    "gmsh.model.mesh.field.setNumber({0}_thresh, 'InField', {0}_dist)",
                    field_var
                ));

                for kv in &refine.body {
                    match kv.key.name.as_str() {
                        "size" | "size_min" => {
                            let val = self.emit_expr(&kv.value)?;
                            self.line(&format!(
                                "gmsh.model.mesh.field.setNumber({}_thresh, 'SizeMin', {})",
                                field_var, val
                            ));
                        }
                        "size_max" => {
                            let val = self.emit_expr(&kv.value)?;
                            self.line(&format!(
                                "gmsh.model.mesh.field.setNumber({}_thresh, 'SizeMax', {})",
                                field_var, val
                            ));
                        }
                        "distance_min" => {
                            let val = self.emit_expr(&kv.value)?;
                            self.line(&format!(
                                "gmsh.model.mesh.field.setNumber({}_thresh, 'DistMin', {})",
                                field_var, val
                            ));
                        }
                        "distance_max" => {
                            let val = self.emit_expr(&kv.value)?;
                            self.line(&format!(
                                "gmsh.model.mesh.field.setNumber({}_thresh, 'DistMax', {})",
                                field_var, val
                            ));
                        }
                        _ => {}
                    }
                }
                let thresh_var = format!("{}_thresh", field_var);
                self.refine_field_vars.push(thresh_var);
            }
        }
        Ok(())
    }

    fn box_region_bounds_py(&self, expr: &Expr) -> Result<[String; 6], CompileError> {
        match expr {
            Expr::Constructor(c) if c.type_name.name == "Box" => {
                let sizes = self.collect_args(&c.args, 3)?;
                Ok([
                    "0".into(),
                    sizes[0].clone(),
                    "0".into(),
                    sizes[1].clone(),
                    "0".into(),
                    sizes[2].clone(),
                ])
            }
            Expr::Pipe(pipe) if pipe.function.name == "translate" => {
                let mut bounds = self.box_region_bounds_py(&pipe.value)?;
                let offsets = self.collect_args(&pipe.args, 3)?;
                for axis in 0..3 {
                    let offset = &offsets[axis];
                    bounds[axis * 2] = format!("({} + {})", bounds[axis * 2], offset);
                    bounds[axis * 2 + 1] = format!("({} + {})", bounds[axis * 2 + 1], offset);
                }
                Ok(bounds)
            }
            _ => Err(CompileError::Unsupported(
                "Python refinement regions must be Box optionally followed by translate".into(),
            )),
        }
    }

    fn emit_mesh_metric_py(&mut self, metric: &MeshMetricStatement) -> Result<(), CompileError> {
        let field = self.fresh_var("metric");
        match &metric.metric {
            MeshMetricKind::Isotropic { size } => {
                let size = self.emit_expr(size)?;
                self.line(&format!(
                    "{} = gmsh.model.mesh.field.add('MathEval')",
                    field
                ));
                self.line(&format!(
                    "gmsh.model.mesh.field.setString({}, 'F', str({}))",
                    field, size
                ));
            }
            MeshMetricKind::Anisotropic { components } => {
                if components.len() != 6 {
                    return Err(CompileError::TypeError {
                        message: "anisotropic metric requires six components".into(),
                    });
                }
                self.line(&format!(
                    "{} = gmsh.model.mesh.field.add('MathEvalAniso')",
                    field
                ));
                for (name, value) in ["M11", "M22", "M33", "M12", "M13", "M23"]
                    .into_iter()
                    .zip(components)
                {
                    let value = self.emit_expr(value)?;
                    self.line(&format!(
                        "gmsh.model.mesh.field.setString({}, '{}', str({}))",
                        field, name, value
                    ));
                }
            }
            MeshMetricKind::Hessian { .. } => {
                return Err(CompileError::Unsupported(
                    "Hessian metric evaluation requires an in-process field provider".into(),
                ));
            }
        }
        let restrict = self.fresh_var("metric_restrict");
        self.line(&format!(
            "{} = gmsh.model.mesh.field.add('Restrict')",
            restrict
        ));
        self.line(&format!(
            "gmsh.model.mesh.field.setNumber({}, 'InField', {})",
            restrict, field
        ));
        let target_dimension = entity_path(&metric.target)
            .filter(|path| path.ends_with(".faces"))
            .map(|_| EntityDimension::Surface)
            .unwrap_or_else(|| {
                entity_dimension_from_gmsh(
                    self.expr_gmsh_dim(&metric.target)
                        .unwrap_or(GmshDim::Volume),
                )
            });
        let tuples = self.metadata_entity_tuples_py(&metric.target, target_dimension)?;
        let target = self.fresh_var("metric_target");
        self.line(&format!("{} = {}", target, tuples));
        for (dimension, property) in [
            (0, "PointsList"),
            (1, "CurvesList"),
            (2, "SurfacesList"),
            (3, "VolumesList"),
        ] {
            self.line(&format!(
                "gmsh.model.mesh.field.setNumbers({}, '{}', [abs(tag) for entity_dim, tag in {} if entity_dim == {}])",
                restrict, property, target, dimension
            ));
        }
        self.refine_field_vars.push(restrict);
        Ok(())
    }

    fn emit_boundary_layer_py(&mut self, bl: &BoundaryLayerStatement) -> Result<(), CompileError> {
        let target_str = self.emit_expr(&bl.target)?;
        self.line(&format!("# boundary layer on {}", target_str));

        let field_var = self.fresh_var("bl_field");
        self.line(&format!(
            "{} = gmsh.model.mesh.field.add('BoundaryLayer')",
            field_var
        ));

        // Attach target entity surfaces
        self.line(&format!(
                "gmsh.model.mesh.field.setNumbers({}, 'SurfacesList', [abs(tag) for _, tag in _ml_boundary(_ml_dimtags(3, {}), combined=False, oriented=False)])",
            field_var, target_str
        ));

        for kv in &bl.body {
            match kv.key.name.as_str() {
                "first_height" => {
                    let val = self.emit_expr(&kv.value)?;
                    self.line(&format!(
                        "gmsh.model.mesh.field.setNumber({}, 'Size', {})",
                        field_var, val
                    ));
                }
                "thickness" | "total_thickness" => {
                    let val = self.emit_expr(&kv.value)?;
                    self.line(&format!(
                        "gmsh.model.mesh.field.setNumber({}, 'Thickness', {})",
                        field_var, val
                    ));
                }
                "layers" => {
                    let val = self.emit_expr(&kv.value)?;
                    self.line(&format!(
                        "gmsh.model.mesh.field.setNumber({}, 'NbLayers', {})",
                        field_var, val
                    ));
                }
                "growth_rate" => {
                    let val = self.emit_expr(&kv.value)?;
                    self.line(&format!(
                        "gmsh.model.mesh.field.setNumber({}, 'Ratio', {})",
                        field_var, val
                    ));
                }
                _ => {}
            }
        }

        // BoundaryLayer fields must also be in the background
        self.refine_field_vars.push(field_var.clone());

        Ok(())
    }

    fn emit_transfinite_py(&mut self, t: &TransfiniteStatement) -> Result<(), CompileError> {
        match &t.kind {
            TransfiniteKind::Curve { entity, settings } => {
                let entity_str = self.emit_expr(entity)?;
                let mut nodes = "10".to_string();
                let mut mesh_type = String::new();
                let mut coef = String::new();
                for kv in settings {
                    match kv.key.name.as_str() {
                        "nodes" => nodes = self.emit_expr(&kv.value)?,
                        "type" => {
                            if let Expr::Literal(LiteralExpr {
                                value: Literal::String(s),
                                ..
                            }) = &kv.value
                            {
                                mesh_type = s.clone();
                            }
                        }
                        "coef" => coef = self.emit_expr(&kv.value)?,
                        _ => {}
                    }
                }
                let mut cmd = format!(
                    "gmsh.model.mesh.setTransfiniteCurve({}, {}",
                    entity_str, nodes
                );
                if !mesh_type.is_empty() {
                    cmd.push_str(&format!(", '{}'", mesh_type));
                    if !coef.is_empty() {
                        cmd.push_str(&format!(", {}", coef));
                    }
                }
                cmd.push(')');
                self.line(&cmd);
            }
            TransfiniteKind::Surface { entity, settings } => {
                let entity_str = self.emit_expr(entity)?;
                let mut arrangement = String::new();
                let mut corners = String::new();
                for kv in settings {
                    match kv.key.name.as_str() {
                        "arrangement" => {
                            if let Expr::Literal(LiteralExpr {
                                value: Literal::String(s),
                                ..
                            }) = &kv.value
                            {
                                arrangement = s.clone();
                            } else if let Expr::Ident(id) = &kv.value {
                                arrangement = id.name.clone();
                            }
                        }
                        "corners" => {
                            if let Expr::Array(arr) = &kv.value {
                                let tags: Vec<String> = arr
                                    .elements
                                    .iter()
                                    .map(|e| self.emit_expr(e))
                                    .collect::<Result<_, _>>()?;
                                corners = tags.join(", ");
                            }
                        }
                        _ => {}
                    }
                }
                let mut cmd = format!("gmsh.model.mesh.setTransfiniteSurface({}", entity_str);
                if !arrangement.is_empty() {
                    cmd.push_str(&format!(", '{}'", arrangement));
                }
                if !corners.is_empty() {
                    cmd.push_str(&format!(", [{}]", corners));
                }
                cmd.push(')');
                self.line(&cmd);
            }
            TransfiniteKind::Volume { entity, settings } => {
                let entity_str = self.emit_expr(entity)?;
                let mut corners = String::new();
                for kv in settings {
                    if kv.key.name == "corners" {
                        if let Expr::Array(arr) = &kv.value {
                            let tags: Vec<String> = arr
                                .elements
                                .iter()
                                .map(|e| self.emit_expr(e))
                                .collect::<Result<_, _>>()?;
                            corners = tags.join(", ");
                        }
                    }
                }
                let mut cmd = format!("gmsh.model.mesh.setTransfiniteVolume({}", entity_str);
                if !corners.is_empty() {
                    cmd.push_str(&format!(", [{}]", corners));
                }
                cmd.push(')');
                self.line(&cmd);
            }
        }
        Ok(())
    }

    fn emit_embed_py(&mut self, e: &EmbedStatement) -> Result<(), CompileError> {
        let source_str = self.emit_expr(&e.source)?;
        let target_str = self.emit_expr(&e.target)?;
        let embed_dim = match e.embed_dim {
            EmbedDim::Point => 0,
            EmbedDim::Curve => 1,
            EmbedDim::Surface => 2,
            EmbedDim::Volume => 3,
        };
        let target_dim = match e.target_dim {
            EmbedDim::Point => 0,
            EmbedDim::Curve => 1,
            EmbedDim::Surface => 2,
            EmbedDim::Volume => 3,
        };
        self.line(&format!(
            "gmsh.model.mesh.embed({}, [{}], {}, {})",
            embed_dim, source_str, target_dim, target_str
        ));
        Ok(())
    }

    fn emit_periodic_py(&mut self, p: &PeriodicStatement) -> Result<(), CompileError> {
        let source_str = self.emit_expr(&p.source)?;
        let target_str = self.emit_expr(&p.target)?;
        let transform_str = self.emit_expr(&p.transform)?;
        let dim = match p.dim {
            PeriodicDim::Curve => 1,
            PeriodicDim::Surface => 2,
        };
        self.line(&format!(
            "gmsh.model.mesh.setPeriodic({}, [{}], [{}], {})",
            dim, source_str, target_str, transform_str
        ));
        Ok(())
    }

    fn emit_compound_py(&mut self, c: &CompoundStatement) -> Result<(), CompileError> {
        let first = c.entities.first().ok_or_else(|| CompileError::TypeError {
            message: "compound requires at least one entity".into(),
        })?;
        let dim = self
            .expr_gmsh_dim(first)
            .ok_or_else(|| CompileError::TypeError {
                message: "cannot infer compound entity dimension".into(),
            })?;

        let tags: Vec<String> = c
            .entities
            .iter()
            .map(|entity| {
                let entity_dim =
                    self.expr_gmsh_dim(entity)
                        .ok_or_else(|| CompileError::TypeError {
                            message: "cannot infer compound entity dimension".into(),
                        })?;
                if entity_dim != dim {
                    return Err(CompileError::TypeError {
                        message: "compound entities must have the same dimension".into(),
                    });
                }
                self.resolve_entity_var_py(entity)
            })
            .collect::<Result<_, _>>()?;
        self.line(&format!(
            "gmsh.model.mesh.setCompound({}, [{}])",
            dim as u8,
            tags.join(", ")
        ));
        Ok(())
    }

    fn resolve_entity_var_py(&self, expr: &Expr) -> Result<String, CompileError> {
        match expr {
            Expr::Ident(id) => Ok(self.resolve_scoped_entity_var_py(&id.name)),
            Expr::FieldAccess(_) => {
                if let Some(path) = entity_path(expr) {
                    let resolved = self.resolve_scoped_entity_var_py(&path);
                    if resolved != path {
                        return Ok(resolved);
                    }
                }
                self.emit_expr(expr)
            }
            _ => self.emit_expr(expr),
        }
    }

    fn resolve_scoped_entity_var_py(&self, path: &str) -> String {
        self.name_to_var
            .get(path)
            .cloned()
            .or_else(|| {
                self.current_geometry.as_ref().and_then(|geometry| {
                    self.name_to_var.get(&format!("{geometry}.{path}")).cloned()
                })
            })
            .or_else(|| {
                self.current_geometry.as_ref().and_then(|geometry| {
                    let flattened = path.replace('.', "__");
                    self.name_to_var
                        .get(&format!("{geometry}.{flattened}"))
                        .cloned()
                })
            })
            .unwrap_or_else(|| path.to_owned())
    }

    fn emit_physics(&mut self, physics: &PhysicsBlock) -> Result<(), CompileError> {
        self.line("# physics groups");
        for stmt in &physics.statements {
            if let PhysicsStatement::Group(g) = stmt {
                // Resolve the expression to a topology path and emit Python code
                // that uses variable names instead of integer tags
                let py_code = self.resolve_physics_expr_py(&g.value);
                match py_code {
                    Ok(py_expr) => {
                        let var_name = format!("pg_{}", g.name.name);
                        self.line(&format!("{} = {}", var_name, py_expr));

                        let dim = match g.bc_type.name.as_str() {
                            "volume" => 3,
                            _ => 2,
                        };
                        self.line(&format!(
                            "gmsh.model.addPhysicalGroup({dim}, [abs(t) for _, t in {var}], name='{name}')",
                            dim = dim,
                            var = var_name,
                            name = g.name.name
                        ));
                    }
                    Err(msg) => {
                        return Err(CompileError::Unsupported(format!(
                            "physics group '{}': {}",
                            g.name.name, msg
                        )));
                    }
                }
            }
        }
        self.line("");
        Ok(())
    }

    /// Resolve a physics group expression to Python code using variable names.
    fn resolve_physics_expr_py(&mut self, expr: &Expr) -> Result<String, String> {
        if let Expr::Binary(b) = expr {
            return match b.op {
                BinaryOp::Sub | BinaryOp::CsgDifference => {
                    let left = self.resolve_physics_expr_py(&b.left)?;
                    let right = self.resolve_physics_expr_py(&b.right)?;
                    Ok(format!("mlg_face_diff({}, {})", left, right))
                }
                BinaryOp::CsgUnion => {
                    let left = self.resolve_physics_expr_py(&b.left)?;
                    let right = self.resolve_physics_expr_py(&b.right)?;
                    Ok(format!("({} + {})", left, right))
                }
                _ => Err("expected topology set operation".into()),
            };
        }

        let parts = self.collect_dot_path_py(expr)?;

        // Find the entity part and topology access part
        // e.g., ["cabinet", "solid", "faces"] → entity="cabinet.solid", access=["faces"]
        // e.g., ["cabinet", "port", "faces", "top"] → entity="cabinet.port", access=["faces","top"]

        // Try progressively longer prefixes to find registered entity
        let mut entity_qname = None;
        let mut topo_parts = &parts[..];

        for i in (1..=parts.len()).rev() {
            let candidate = parts[..i].join(".");
            let scoped_candidate = self
                .current_geometry
                .as_ref()
                .map(|geometry| format!("{geometry}.{candidate}"));
            if self.topo.lookup(&candidate).is_some()
                || scoped_candidate
                    .as_deref()
                    .is_some_and(|candidate| self.topo.lookup(candidate).is_some())
            {
                entity_qname = Some(candidate);
                topo_parts = &parts[i..];
                break;
            }
        }

        let entity_qname =
            entity_qname.ok_or_else(|| format!("unknown entity '{}'", parts.join(".")))?;
        let entity_qname = if self.topo.lookup(&entity_qname).is_some() {
            entity_qname
        } else if let Some(geometry) = &self.current_geometry {
            format!("{geometry}.{entity_qname}")
        } else {
            entity_qname
        };

        // Get the Python variable name for this entity
        let py_var = self
            .name_to_var
            .get(&entity_qname)
            .cloned()
            .unwrap_or_else(|| {
                entity_qname
                    .split('.')
                    .next_back()
                    .unwrap_or("unknown")
                    .to_string()
            });

        let shape = self.topo.lookup(&entity_qname).cloned();

        let topo_strs: Vec<&str> = topo_parts.iter().map(|s| s.as_str()).collect();

        match topo_strs.as_slice() {
            [] => Ok(format!("_ml_dimtags(3, {})", py_var)),
            ["faces"] => Ok(format!(
                "_ml_boundary(_ml_dimtags(3, {}), oriented=False)",
                py_var
            )),
            ["faces", face_name] => {
                if let Some(ref shape) = shape {
                    match shape {
                        crate::topology::ShapeTopology::Box { .. } => {
                            let (coord_idx, sign) = match *face_name {
                                "bottom" => (2, "min"),
                                "top" => (2, "max"),
                                "back" => (1, "min"),
                                "front" => (1, "max"),
                                "left" => (0, "min"),
                                "right" => (0, "max"),
                                _ => return Err(format!("unknown box face '{}'", face_name)),
                            };
                            Ok(format!(
                                "mlg_find_box_face(_ml_single({}, 'box face selector'), {}, '{}')",
                                py_var, coord_idx, sign
                            ))
                        }
                        crate::topology::ShapeTopology::Cylinder { .. } => Ok(format!(
                            "mlg_find_cyl_face(_ml_single({}, 'cylinder face selector'), '{}')",
                            py_var, face_name
                        )),
                        crate::topology::ShapeTopology::Sphere { .. } => Ok(format!(
                            "_ml_boundary(_ml_dimtags(3, {}), oriented=False)",
                            py_var
                        )),
                        _ => Ok(format!(
                            "_ml_boundary(_ml_dimtags(3, {}), oriented=False)  # .{}",
                            py_var, face_name
                        )),
                    }
                } else {
                    Err(format!(
                        "unknown shape for face access on '{}'",
                        entity_qname
                    ))
                }
            }
            ["edges"] => Ok(format!(
                "gmsh.model.getBoundary(_ml_boundary(_ml_dimtags(3, {}), oriented=False), oriented=False)",
                py_var
            )),
            _ => Err(format!(
                "unsupported topology access '.{}'",
                topo_parts
                    .iter()
                    .map(|s| s.as_str())
                    .collect::<Vec<_>>()
                    .join(".")
            )),
        }
    }

    fn collect_dot_path_py(&self, expr: &Expr) -> Result<Vec<String>, String> {
        match expr {
            Expr::Ident(id) => Ok(vec![id.name.clone()]),
            Expr::FieldAccess(fa) => {
                let mut parts = self.collect_dot_path_py(&fa.object)?;
                parts.push(fa.field.name.clone());
                Ok(parts)
            }
            Expr::Binary(b) if matches!(b.op, BinaryOp::Sub | BinaryOp::CsgDifference) => {
                let left = self.resolve_physics_expr_py_immut(&b.left)?;
                let right = self.resolve_physics_expr_py_immut(&b.right)?;
                Err(format!("mlg_face_diff({}, {})", left, right))
            }
            _ => Err("expected dot-access path".into()),
        }
    }

    /// Immutable version for use inside collect_dot_path_py error paths.
    fn resolve_physics_expr_py_immut(&self, expr: &Expr) -> Result<String, String> {
        match expr {
            Expr::Ident(id) => Ok(id.name.clone()),
            Expr::FieldAccess(fa) => {
                let mut parts = vec![];
                self.collect_dot_parts(&fa.object, &mut parts);
                parts.push(fa.field.name.clone());
                Ok(parts.join("."))
            }
            _ => Err("complex expr".into()),
        }
    }

    fn collect_dot_parts(&self, expr: &Expr, parts: &mut Vec<String>) {
        match expr {
            Expr::Ident(id) => parts.push(id.name.clone()),
            Expr::FieldAccess(fa) => {
                self.collect_dot_parts(&fa.object, parts);
                parts.push(fa.field.name.clone());
            }
            _ => parts.push("?".into()),
        }
    }

    fn emit_export(&mut self, export: &ExportBlock) -> Result<(), CompileError> {
        self.line("# export");

        let mut format = "msh4".to_string();
        let mut filename = "output.msh".to_string();
        let mut order = 1;
        let mut binary = false;
        let mut save_all = false;

        for kv in &export.settings {
            match kv.key.name.as_str() {
                "format" => {
                    if let Expr::Ident(id) = &kv.value {
                        format = id.name.clone();
                    }
                }
                "filename" => {
                    if let Expr::Literal(LiteralExpr {
                        value: Literal::String(s),
                        ..
                    }) = &kv.value
                    {
                        filename = s.clone();
                    }
                }
                "order" => {
                    if let Expr::Literal(LiteralExpr {
                        value: Literal::Int(n),
                        ..
                    }) = &kv.value
                    {
                        order = *n;
                    }
                }
                "binary" => {
                    binary = matches!(
                        &kv.value,
                        Expr::Literal(LiteralExpr {
                            value: Literal::Bool(true),
                            ..
                        })
                    );
                }
                "save_all" => {
                    save_all = matches!(
                        &kv.value,
                        Expr::Literal(LiteralExpr {
                            value: Literal::Bool(true),
                            ..
                        })
                    );
                }
                _ => {}
            }
        }

        if order > 1 {
            self.line(&format!(
                "gmsh.option.setNumber('Mesh.ElementOrder', {})",
                order
            ));
            self.line("gmsh.option.setNumber('Mesh.HighOrderOptimize', 1)");
        }

        if binary {
            self.line("gmsh.option.setNumber('Mesh.Binary', 1)");
        }

        if save_all {
            self.line("gmsh.option.setNumber('Mesh.SaveAll', 1)");
        }

        // Set format version
        match format.as_str() {
            "msh2" => self.line("gmsh.option.setNumber('Mesh.MshFileVersion', 2.2)"),
            "msh4" => self.line("gmsh.option.setNumber('Mesh.MshFileVersion', 4.1)"),
            _ => {} // other formats determined by file extension
        }

        self.line("gmsh.model.mesh.generate(3)");

        if order > 1 {
            self.line("gmsh.model.mesh.optimize('HighOrder')");
        }

        self.line(&format!("gmsh.write('{}')", filename));

        Ok(())
    }

    // ── Expression emission ──────────────────────────────────────

    fn emit_expr(&self, expr: &Expr) -> Result<String, CompileError> {
        match expr {
            Expr::Literal(LiteralExpr {
                value: Literal::Int(v),
                ..
            }) => Ok(v.to_string()),
            Expr::Literal(LiteralExpr {
                value: Literal::Float(v),
                ..
            }) => Ok(format_float_py(*v)),
            Expr::Literal(LiteralExpr {
                value: Literal::Bool(b),
                ..
            }) => Ok(if *b { "True" } else { "False" }.to_string()),
            Expr::Literal(LiteralExpr {
                value: Literal::String(s),
                ..
            }) => Ok(format!("'{}'", escape_python_string(s))),
            Expr::Ident(id) => {
                // Map built-in constants to Python equivalents
                match id.name.as_str() {
                    "PI" | "pi" => Ok("math.pi".to_string()),
                    _ => Ok(id.name.clone()),
                }
            }
            Expr::Binary(b) => {
                let l = self.emit_expr(&b.left)?;
                let r = self.emit_expr(&b.right)?;
                let op = match b.op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub | BinaryOp::CsgDifference => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Mod => "%",
                    BinaryOp::Pow => "**",
                    BinaryOp::Eq => "==",
                    BinaryOp::Ne => "!=",
                    BinaryOp::Lt => "<",
                    BinaryOp::Gt => ">",
                    BinaryOp::Le => "<=",
                    BinaryOp::Ge => ">=",
                    BinaryOp::And => "and",
                    BinaryOp::Or => "or",
                    _ => "?",
                };
                Ok(format!("({} {} {})", l, op, r))
            }
            Expr::Unary(u) => {
                let v = self.emit_expr(&u.operand)?;
                match u.op {
                    UnaryOp::Neg => Ok(format!("(-{})", v)),
                    UnaryOp::Not => Ok(format!("(not {})", v)),
                }
            }
            Expr::FieldAccess(fa) => {
                // Ported geometry instances are flattened into Python bindings
                // such as `frame__inlet`; source-level `frame.inlet` would
                // otherwise reference a Python object that is never emitted.
                if let Some(path) = entity_path(expr) {
                    let resolved = self.resolve_scoped_entity_var_py(&path);
                    if resolved != path {
                        return Ok(resolved);
                    }
                }

                let obj = self.emit_expr(&fa.object)?;
                // For struct field access, use dict key access
                // For geometry topology, use dot notation
                Ok(format!("{}['{}']", obj, fa.field.name))
            }
            Expr::StructInit(si) => {
                let mut entries = Vec::new();
                for field in &si.fields {
                    let val = self.emit_expr(&field.value)?;
                    entries.push(format!("'{}': {}", field.name.name, val));
                }
                Ok(format!("{{{}}}", entries.join(", ")))
            }
            Expr::Call(c) => {
                // Inline user-defined functions in scalar contexts (Polygon
                // point coordinates, etc.) so the emitted Python is plain
                // arithmetic rather than an undefined function call.
                if self.functions.contains_key(&c.function.name) {
                    let inlined = self.try_inline_call(c)?;
                    return self.emit_expr(&inlined);
                }
                let args: Vec<String> = c
                    .args
                    .iter()
                    .map(|a| self.emit_expr(&a.value))
                    .collect::<Result<_, _>>()?;
                // Map built-in math functions to Python math module
                let py_name = match c.function.name.as_str() {
                    "cos" => "math.cos",
                    "sin" => "math.sin",
                    "tan" => "math.tan",
                    "acos" => "math.acos",
                    "asin" => "math.asin",
                    "atan" => "math.atan",
                    "atan2" => "math.atan2",
                    "sqrt" => "math.sqrt",
                    "exp" => "math.exp",
                    "log" => "math.log",
                    "log10" => "math.log10",
                    "abs" => "abs",
                    "floor" => "math.floor",
                    "ceil" => "math.ceil",
                    "fmod" => "math.fmod",
                    "hypot" => "math.hypot",
                    other => other,
                };
                Ok(format!("{}({})", py_name, args.join(", ")))
            }
            Expr::Array(array) => {
                let elements = array
                    .elements
                    .iter()
                    .map(|element| self.emit_expr(element))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(format!("[{}]", elements.join(", ")))
            }
            Expr::Index(index) => {
                let object = self.emit_expr(&index.object)?;
                let index_value = self.emit_expr(&index.index)?;
                Ok(format!("{object}[{index_value}]"))
            }
            Expr::Paren(inner) => {
                let v = self.emit_expr(&inner.value)?;
                Ok(format!("({})", v))
            }
            _ => Err(CompileError::Unsupported(
                "Python backend cannot emit this expression in a scalar context".into(),
            )),
        }
    }

    fn collect_args(
        &self,
        args: &[Argument],
        expected: usize,
    ) -> Result<Vec<String>, CompileError> {
        if args.len() != expected {
            return Err(CompileError::TypeError {
                message: format!(
                    "expected {} positional arguments, got {}",
                    expected,
                    args.len()
                ),
            });
        }
        if args.iter().any(|arg| arg.name.is_some()) {
            return Err(CompileError::TypeError {
                message: format!(
                    "expected {} positional arguments, got named arguments",
                    expected,
                ),
            });
        }

        args.iter().map(|a| self.emit_expr(&a.value)).collect()
    }

    fn find_named_arg(&self, args: &[Argument], name: &str) -> Result<String, CompileError> {
        for arg in args {
            if let Some(ref n) = arg.name {
                if n.name == name {
                    return self.emit_expr(&arg.value);
                }
            }
        }
        Err(CompileError::Internal(format!(
            "missing argument '{}'",
            name
        )))
    }

    fn angle_arg_degrees_to_radians(&self, args: &[Argument]) -> Result<String, CompileError> {
        let angle = self.find_named_arg(args, "angle")?;
        Ok(format!("({} * math.pi / 180.0)", angle))
    }

    fn find_named_arg_expr<'a>(&self, args: &'a [Argument], name: &str) -> Option<&'a Expr> {
        args.iter()
            .find(|a| a.name.as_ref().is_some_and(|n| n.name == name))
            .map(|a| &a.value)
    }

    fn affine_matrix_args(&self, args: &[Argument]) -> Result<Vec<String>, CompileError> {
        let matrix = self
            .find_named_arg_expr(args, "matrix")
            .or_else(|| args.first().map(|argument| &argument.value))
            .ok_or_else(|| CompileError::TypeError {
                message: "affine expects matrix: [16 scalars]".into(),
            })?;
        let Expr::Array(array) = matrix else {
            return Err(CompileError::TypeError {
                message: "affine matrix must be a 16-element array".into(),
            });
        };
        if array.elements.len() != 16 {
            return Err(CompileError::TypeError {
                message: format!(
                    "affine matrix must contain 16 scalars, got {}",
                    array.elements.len()
                ),
            });
        }
        array
            .elements
            .iter()
            .map(|element| self.emit_expr(element))
            .collect()
    }

    fn find_named_arg_bool(&self, args: &[Argument], name: &str) -> Option<bool> {
        match self.find_named_arg_expr(args, name)? {
            Expr::Literal(LiteralExpr {
                value: Literal::Bool(b),
                ..
            }) => Some(*b),
            _ => None,
        }
    }

    /// Resolve a Loft section to the Python variable holding its OCC tag.
    /// `Polygon` sections store an `addCurveLoop` wire var alongside the
    /// surface var; OCC `addThruSections` requires the wire tag, so we
    /// prefer it whenever available. Other shapes (Disk, Circle) fall back
    /// to the surface var emitted by `emit_expr`.
    fn resolve_section_tag_py(&self, expr: &Expr) -> Result<String, CompileError> {
        let surface_var = self.emit_expr(expr)?;
        if let Some(wire_var) = self.polygon_wire_var.get(&surface_var) {
            return Ok(wire_var.clone());
        }
        Ok(surface_var)
    }

    fn find_named_arg_str(&self, args: &[Argument], name: &str) -> Result<String, CompileError> {
        for arg in args {
            if let Some(ref n) = arg.name {
                if n.name == name {
                    if let Expr::Ident(id) = &arg.value {
                        return Ok(id.name.clone());
                    }
                }
            }
        }
        Err(CompileError::Internal(format!(
            "missing argument '{}'",
            name
        )))
    }

    fn mesh_algo_2d_py(&self, expr: &Expr) -> Result<&'static str, CompileError> {
        if let Expr::Ident(id) = expr {
            Ok(match id.name.as_str() {
                "meshadapt" => "1",
                "automatic" => "2",
                "initial_only" => "3",
                "delaunay" => "5",
                "frontal" => "6",
                "bamg" => "7",
                "frontal_quads" => "8",
                "pack_parallelograms" => "9",
                "quasi_structured" => "11",
                _ => "5",
            })
        } else {
            Ok("5")
        }
    }

    fn mesh_algo_3d_py(&self, expr: &Expr) -> Result<&'static str, CompileError> {
        if let Expr::Ident(id) = expr {
            Ok(match id.name.as_str() {
                "delaunay" => "1",
                "initial_only" => "3",
                "frontal" => "4",
                "mmg3d" => "7",
                "hxt" => "10",
                _ => "1",
            })
        } else {
            Ok("1")
        }
    }

    fn line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.buf.push_str("    ");
        }
        self.buf.push_str(text);
        self.buf.push('\n');
    }
}

fn format_float_py(v: f64) -> String {
    if v == v.floor() && v.abs() < 1e15 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

fn escape_python_string(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('\'', "\\'")
        .replace('\n', "\\n")
        .replace('\r', "\\r")
}

/// Evaluate for-loop clause ranges to (start, end_exclusive) integer pairs.
fn eval_for_ranges(clauses: &[ForClause]) -> Result<Vec<(i64, i64)>, CompileError> {
    clauses
        .iter()
        .map(|c| {
            let start = try_eval_const_int(&c.range.start)?.ok_or_else(|| {
                CompileError::Unsupported(
                    "for loop range start must be a compile-time integer".into(),
                )
            })?;
            let end = try_eval_const_int(&c.range.end)?.ok_or_else(|| {
                CompileError::Unsupported(
                    "for loop range end must be a compile-time integer".into(),
                )
            })?;
            let end = if c.range.inclusive {
                end.checked_add(1).ok_or_else(const_int_overflow)?
            } else {
                end
            };
            Ok((start, end))
        })
        .collect()
}

fn try_eval_const_int(expr: &Expr) -> Result<Option<i64>, CompileError> {
    match expr {
        Expr::Literal(LiteralExpr {
            value: Literal::Int(v),
            ..
        }) => Ok(Some(*v)),
        Expr::Literal(LiteralExpr {
            value: Literal::Float(v),
            ..
        }) => Ok(Some(*v as i64)),
        Expr::Unary(u) if u.op == UnaryOp::Neg => {
            let v = try_eval_const_int(&u.operand)?;
            match v {
                Some(v) => Ok(Some(v.checked_neg().ok_or_else(const_int_overflow)?)),
                None => Ok(None),
            }
        }
        Expr::Binary(b) => {
            let Some(l) = try_eval_const_int(&b.left)? else {
                return Ok(None);
            };
            let Some(r) = try_eval_const_int(&b.right)? else {
                return Ok(None);
            };
            let value = match b.op {
                BinaryOp::Add => l.checked_add(r),
                BinaryOp::Sub => l.checked_sub(r),
                BinaryOp::Mul => l.checked_mul(r),
                BinaryOp::Div if r != 0 => l.checked_div(r),
                _ => return Ok(None),
            }
            .ok_or_else(const_int_overflow)?;
            Ok(Some(value))
        }
        Expr::Paren(inner) => try_eval_const_int(&inner.value),
        _ => Ok(None),
    }
}

fn const_int_overflow() -> CompileError {
    CompileError::Unsupported("integer overflow in compile-time constant expression".into())
}

fn constructor_dim(c: &ConstructorExpr) -> GmshDim {
    match c.type_name.name.as_str() {
        "Point" => GmshDim::Point,
        "Line" | "Arc" | "Spline" | "BSpline" => GmshDim::Curve,
        "Rectangle" | "Disk" | "Circle" | "Polygon" => GmshDim::Surface,
        _ => GmshDim::Volume,
    }
}

fn gmsh_dim_from_entity_dimension(dimension: EntityDimension) -> GmshDim {
    match dimension {
        EntityDimension::Point => GmshDim::Point,
        EntityDimension::Curve => GmshDim::Curve,
        EntityDimension::Surface => GmshDim::Surface,
        EntityDimension::Volume => GmshDim::Volume,
    }
}

fn entity_dimension_from_gmsh(dimension: GmshDim) -> EntityDimension {
    match dimension {
        GmshDim::Point => EntityDimension::Point,
        GmshDim::Curve => EntityDimension::Curve,
        GmshDim::Surface => EntityDimension::Surface,
        GmshDim::Volume => EntityDimension::Volume,
    }
}

fn entity_path(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Ident(id) => Some(id.name.clone()),
        Expr::FieldAccess(field) => {
            let base = entity_path(&field.object)?;
            Some(format!("{base}.{}", field.field.name))
        }
        Expr::Paren(inner) => entity_path(&inner.value),
        _ => None,
    }
}

fn try_eval_const_bool(expr: &Expr) -> Option<bool> {
    match expr {
        Expr::Literal(LiteralExpr {
            value: Literal::Bool(b),
            ..
        }) => Some(*b),
        Expr::Binary(b) => {
            if let (Ok(Some(l)), Ok(Some(r))) =
                (try_eval_const_int(&b.left), try_eval_const_int(&b.right))
            {
                match b.op {
                    BinaryOp::Eq => Some(l == r),
                    BinaryOp::Ne => Some(l != r),
                    BinaryOp::Lt => Some(l < r),
                    BinaryOp::Gt => Some(l > r),
                    BinaryOp::Le => Some(l <= r),
                    BinaryOp::Ge => Some(l >= r),
                    _ => None,
                }
            } else {
                let lb = try_eval_const_bool(&b.left);
                let rb = try_eval_const_bool(&b.right);
                match b.op {
                    BinaryOp::And => Some(lb? && rb?),
                    BinaryOp::Or => Some(lb? || rb?),
                    _ => None,
                }
            }
        }
        Expr::Unary(u) if u.op == UnaryOp::Not => try_eval_const_bool(&u.operand).map(|v| !v),
        Expr::Paren(inner) => try_eval_const_bool(&inner.value),
        _ => None,
    }
}

/// Substitute identifiers in an expression using a name→expr map.
fn substitute_expr(expr: &Expr, subst: &HashMap<String, Expr>) -> Expr {
    match expr {
        Expr::Ident(id) => {
            if let Some(replacement) = subst.get(&id.name) {
                replacement.clone()
            } else {
                expr.clone()
            }
        }
        Expr::Binary(b) => Expr::Binary(BinaryExpr {
            left: Box::new(substitute_expr(&b.left, subst)),
            op: b.op,
            right: Box::new(substitute_expr(&b.right, subst)),
            span: b.span,
        }),
        Expr::Unary(u) => Expr::Unary(UnaryExpr {
            op: u.op,
            operand: Box::new(substitute_expr(&u.operand, subst)),
            span: u.span,
        }),
        Expr::Pipe(p) => Expr::Pipe(PipeExpr {
            value: Box::new(substitute_expr(&p.value, subst)),
            function: p.function.clone(),
            args: p
                .args
                .iter()
                .map(|a| Argument {
                    name: a.name.clone(),
                    value: substitute_expr(&a.value, subst),
                    span: a.span,
                })
                .collect(),
            span: p.span,
        }),
        Expr::Constructor(c) => Expr::Constructor(ConstructorExpr {
            type_name: c.type_name.clone(),
            args: c
                .args
                .iter()
                .map(|a| Argument {
                    name: a.name.clone(),
                    value: substitute_expr(&a.value, subst),
                    span: a.span,
                })
                .collect(),
            span: c.span,
        }),
        Expr::Call(c) => Expr::Call(CallExpr {
            function: c.function.clone(),
            args: c
                .args
                .iter()
                .map(|a| Argument {
                    name: a.name.clone(),
                    value: substitute_expr(&a.value, subst),
                    span: a.span,
                })
                .collect(),
            span: c.span,
        }),
        Expr::Paren(inner) => Expr::Paren(ParenExpr::new(
            substitute_expr(&inner.value, subst),
            inner.span,
        )),
        Expr::FieldAccess(fa) => Expr::FieldAccess(FieldAccessExpr {
            object: Box::new(substitute_expr(&fa.object, subst)),
            field: fa.field.clone(),
            span: fa.span,
        }),
        _ => expr.clone(),
    }
}

#[cfg(test)]
mod expression_tests {
    use super::*;
    use meshlang_ast::{span::Span, Declaration, Expr};

    fn parse_value(source: &str) -> Expr {
        let file = meshlang_parser::parse(&format!("param value = {source}"), "expr.mlg")
            .expect("expression fixture must parse");
        let Declaration::Param(parameter) = &file.declarations[0] else {
            panic!("expected parameter declaration");
        };
        parameter.value.clone()
    }

    #[test]
    fn scalar_emission_maps_constants_math_and_power() {
        let expression = parse_value("PI + sin(1.0) ** 2");
        let emitted = PythonCodegen::new().emit_expr(&expression).unwrap();
        assert_eq!(emitted, "(math.pi + (math.sin(1.0) ** 2))");
    }

    #[test]
    fn scalar_emission_uses_python_string_and_field_conventions() {
        let codegen = PythonCodegen::new();
        let string = parse_value(r#""line\nvalue""#);
        assert_eq!(codegen.emit_expr(&string).unwrap(), r#"'line\nvalue'"#);
        let field = parse_value("material.density");
        assert_eq!(codegen.emit_expr(&field).unwrap(), "material['density']");
    }

    #[test]
    fn port_access_uses_flattened_python_binding() {
        let mut codegen = PythonCodegen::new();
        codegen.current_geometry = Some("model".to_owned());
        codegen
            .name_to_var
            .insert("model.frame.inlet".to_owned(), "frame__inlet".to_owned());

        let field = parse_value("frame.inlet");
        assert_eq!(codegen.emit_expr(&field).unwrap(), "frame__inlet");
    }

    #[test]
    fn expression_emission_covers_parentheses_arrays_and_indices() {
        let codegen = PythonCodegen::new();
        assert_eq!(
            codegen.emit_expr(&parse_value("(1 + 2)")).unwrap(),
            "((1 + 2))"
        );
        assert_eq!(
            codegen.emit_expr(&parse_value("[1, 2, 3]")).unwrap(),
            "[1, 2, 3]"
        );
        assert_eq!(
            codegen.emit_expr(&parse_value("[1, 2, 3][1]")).unwrap(),
            "[1, 2, 3][1]"
        );
    }

    #[test]
    fn shape_and_pipe_emission_are_tested_at_the_codegen_boundary() {
        let Expr::Constructor(constructor) = parse_value("Box(1.0, 2.0, 3.0)") else {
            panic!("expected Box constructor");
        };
        assert_eq!(
            PythonCodegen::new()
                .emit_constructor_call(&constructor)
                .unwrap(),
            "occ.addBox(0, 0, 0, 1.0, 2.0, 3.0)"
        );

        let Expr::Pipe(pipe) = parse_value("Box(1.0, 2.0, 3.0) |> translate(1.0, 0.0, 0.0)") else {
            panic!("expected transform pipe");
        };
        let mut codegen = PythonCodegen::new();
        codegen.emit_pipe_chain_py("test_body", &pipe).unwrap();
        assert!(codegen.output().contains("occ.translate"));
    }

    #[test]
    fn loop_substitution_reaches_expression_emission() {
        let expression = parse_value("i + 1");
        let mut substitutions = HashMap::new();
        substitutions.insert("i".to_owned(), Expr::literal(Literal::Int(3), Span::DUMMY));
        let substituted = substitute_expr(&expression, &substitutions);
        assert_eq!(
            PythonCodegen::new().emit_expr(&substituted).unwrap(),
            "(3 + 1)"
        );
    }

    #[test]
    fn geometry_topology_binding_uses_gmsh_query_not_dict_access() {
        let file = meshlang_parser::parse(
            "geometry test { let body = Box(1.0, 2.0, 3.0) let faces = body.faces }",
            "topology-binding.mlg",
        )
        .expect("topology fixture must parse");
        let Declaration::Geometry(geometry) = &file.declarations[0] else {
            panic!("expected geometry declaration");
        };

        let mut codegen = PythonCodegen::new();
        codegen
            .emit_geometry(geometry)
            .expect("topology binding should emit");
        let output = codegen.output();
        assert!(output.contains("faces = _ml_boundary(_ml_dimtags(3, body), oriented=False)"));
        assert!(!output.contains("faces = body['faces']"));
    }
}

#[cfg(test)]
mod csg_parity_tests {
    use super::*;
    use meshlang_ast::Declaration;

    #[test]
    fn python_flattens_same_operator_csg_chains() {
        let file = meshlang_parser::parse(
            "geometry test { let a = Box(1.0, 2.0, 3.0) let b = Box(0.5, 0.5, 0.5) let c = Box(0.25, 0.25, 0.25) let result = a - b - c }",
            "csg-chain.mlg",
        )
        .expect("CSG fixture must parse");
        let Declaration::Geometry(geometry) = &file.declarations[0] else {
            panic!("expected geometry declaration");
        };

        let mut codegen = PythonCodegen::new();
        codegen.emit_geometry(geometry).expect("CSG must emit");
        let output = codegen.output();
        assert_eq!(output.matches("occ.cut(").count(), 1);
        assert!(output.contains(" + _ml_dimtags("));
    }
}
