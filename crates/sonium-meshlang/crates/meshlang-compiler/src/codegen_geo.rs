//! Code generator: AST → Gmsh `.geo` file (OpenCASCADE kernel).
//!
//! The .geo output uses SetFactory("OpenCASCADE") for full CSG support.
//! Entity tags are managed by the TagAllocator. A `.mlg.map` sidecar
//! is generated to map MeshLang names ↔ Gmsh integer tags.

use crate::error::CompileError;
use crate::name_resolution::{self, Scope};
use crate::tag_allocator::{GmshDim, TagAllocator};
use meshlang_ast::*;
use std::collections::{HashMap, HashSet};

use crate::topology::{BoxFace, CsgOp, CylinderFace, TopologyTracker};

/// Runtime state for one dynamic `for` loop.
///
/// Compile-time loops can immediately union their products. Runtime loops
/// must retain every entity produced by each iteration, including products
/// emitted by nested loops, so later physical-group/topology expressions do
/// not silently address only the final iteration.
#[derive(Debug, Default)]
struct RuntimeLoopContext {
    id: usize,
    lists: Vec<(GmshDim, String)>,
    source_names: Vec<String>,
}

pub struct GeoCodegen {
    buf: String,
    tags: TagAllocator,
    topo: TopologyTracker,
    indent: usize,
    scope: Option<Scope>,
    /// Current geometry block name (for qualified paths).
    current_geometry: Option<String>,
    /// Counter for temporary intermediate CSG results.
    temp_counter: usize,
    /// User-defined functions available for inlining.
    functions: HashMap<String, FnDecl>,
    /// Field IDs created by refine blocks, to compose into a Min background field.
    refine_field_ids: Vec<i32>,
    /// Most recently installed background field, so canonical controls can
    /// extend rather than replace source refine fields.
    background_field_id: Option<i32>,
    /// Named entities that must remain addressable after a later CSG use.
    preserved_entities: HashSet<String>,
    runtime_loops: Vec<RuntimeLoopContext>,
    runtime_loop_counter: usize,
    /// Physical-group names actually emitted as `Physical …("name")`
    /// statements. The canonical-output check reads this registry instead
    /// of scanning the output text, where a name could also appear in a
    /// comment, a string, or as a substring of another name.
    emitted_physical_groups: HashSet<String>,
}

impl Default for GeoCodegen {
    fn default() -> Self {
        Self::new()
    }
}

impl GeoCodegen {
    pub fn new() -> Self {
        Self {
            buf: String::with_capacity(4096),
            tags: TagAllocator::new(),
            topo: TopologyTracker::new(),
            indent: 0,
            scope: None,
            current_geometry: None,
            temp_counter: 0,
            functions: HashMap::new(),
            refine_field_ids: Vec::new(),
            background_field_id: None,
            preserved_entities: HashSet::new(),
            runtime_loops: Vec::new(),
            runtime_loop_counter: 0,
            emitted_physical_groups: HashSet::new(),
        }
    }

    /// Install canonical, dimension-qualified IDs before emitting backend
    /// temporaries. Internal lowering nodes are deliberately excluded from
    /// the public `.mlg.map` sidecar.
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
        for node in &ir.entities {
            if node
                .name
                .split('.')
                .any(|component| component.starts_with("__ir_"))
            {
                continue;
            }
            let dim = gmsh_dim_from_entity_dimension(node.id.dimension);
            let tag = i32::try_from(node.id.index).map_err(|_| {
                CompileError::Internal(format!(
                    "canonical entity '{}' ID {} exceeds the Gmsh tag range",
                    node.name, node.id.index
                ))
            })?;
            self.tags
                .reserve(dim, &node.name, tag)
                .map_err(CompileError::Internal)?;
        }
        Ok(())
    }

    fn fresh_temp(&mut self, prefix: &str) -> String {
        self.temp_counter += 1;
        if let Some(ref geo) = self.current_geometry {
            format!("{}.__tmp_{}_{}", geo, prefix, self.temp_counter)
        } else {
            format!("__tmp_{}_{}", prefix, self.temp_counter)
        }
    }

    pub fn output(self) -> String {
        self.buf
    }

    /// Verify that AST emission materialized every public canonical entity
    /// and did not introduce a dimension mismatch.
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
            let Some(actual) = self.tags.lookup(&node.name) else {
                return Err(CompileError::Internal(format!(
                    "canonical entity '{}' was not materialized by GEO backend",
                    node.name
                )));
            };
            if actual.dim != expected {
                return Err(CompileError::Internal(format!(
                    "canonical entity '{}' dimension drifted from {:?} to {:?}",
                    node.name, expected, actual.dim
                )));
            }
        }
        for group in &ir.physical_groups {
            if !self.emitted_physical_groups.contains(&group.name) {
                return Err(CompileError::Internal(format!(
                    "canonical physical group '{}' was not materialized by GEO backend",
                    group.name
                )));
            }
        }
        Ok(())
    }

    /// Emit manufacturing controls that are representable in the Gmsh backend.
    /// Cyclic design regions are materialized as rotated OCC copies. The
    /// canonical entities remain the source sector; the copies intentionally
    /// use Gmsh's anonymous runtime tags and are consumed by downstream mesh
    /// tooling as additional sectors.
    pub fn emit_canonical_controls(
        &mut self,
        ir: &CanonicalGeometryIr,
    ) -> Result<(), CompileError> {
        let nodes = ir
            .entities
            .iter()
            .map(|node| (node.id, node))
            .collect::<HashMap<_, _>>();
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
                            let entity = self.tags.lookup(&node.name).ok_or_else(|| {
                                CompileError::Internal(format!(
                                    "cyclic symmetry entity '{}' was not materialized by GEO backend",
                                    node.name
                                ))
                            })?;
                            Ok((node, entity))
                        })
                        .collect::<Result<Vec<_>, CompileError>>()?;
                    if entities.is_empty() {
                        continue;
                    }

                    let index = control_index;
                    control_index += 1;
                    self.line(&format!(
                        "// MeshLang cyclic symmetry region={} sectors={} axis={:?}",
                        region.name, sectors, axis
                    ));
                    self.line(&format!(
                        "For __ml_cyclic_sector_{index} In {{1:{}}}",
                        sectors - 1
                    ));
                    self.indent += 1;
                    self.line(&format!(
                        "__ml_cyclic_angle_{index} = 2*Pi*__ml_cyclic_sector_{index}/{};",
                        sectors
                    ));
                    for (node, entity) in entities {
                        let axis = format!(
                            "{{{}, {}, {}}}",
                            format_float(axis[0]),
                            format_float(axis[1]),
                            format_float(axis[2])
                        );
                        let entity_ref = format!(
                            "{}{{{}}};",
                            entity.dim.keyword(),
                            self.entity_geo_ref(&node.name)
                        );
                        self.line(&format!(
                            "Rotate {{ {axis}, {{0, 0, 0}}, __ml_cyclic_angle_{index} }} {{ Duplicata {{ {entity_ref} }} }}"
                        ));
                    }
                    self.indent -= 1;
                    self.line("EndFor");
                }
            }
        }

        let transfinite_entity_names = ir
            .entities
            .iter()
            .map(|node| (node.id, self.entity_geo_ref(&node.name)))
            .collect::<HashMap<_, _>>();
        for control in &ir.mesh.transfinite {
            let statement = crate::canonical_ast::materialize_transfinite_control(
                control,
                ir,
                &transfinite_entity_names,
            )?;
            self.emit_transfinite(&statement)?;
        }

        let refine_entity_names = ir
            .entities
            .iter()
            // Keep the canonical qualified spelling here. GEO's expression
            // resolver uses field access (`model.panel`) to recover the
            // planned dimension; converting to a flattened tag too early
            // makes metric/refine dimension inference fall back to Volume.
            .map(|node| (node.id, node.name.clone()))
            .collect::<HashMap<_, _>>();
        for control in &ir.mesh.refinements {
            let statement = crate::canonical_ast::materialize_refine_control(
                control,
                ir,
                &refine_entity_names,
            )?;
            self.emit_refine(&statement)?;
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
            self.emit_mesh_metric(&statement)?;
        }
        for control in &ir.mesh.boundary_layers {
            self.emit_canonical_boundary_layer(control, ir)?;
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
                EntityDimension::Curve => GmshDim::Curve,
                EntityDimension::Surface => GmshDim::Surface,
                other => {
                    return Err(CompileError::Unsupported(format!(
                        "canonical periodic control does not support {:?} entities",
                        other
                    )));
                }
            };
            if self.tags.lookup(&source.name).is_none() {
                return Err(CompileError::Internal(format!(
                    "canonical periodic source '{}' not materialized by GEO backend",
                    source.name
                )));
            }
            if self.tags.lookup(&target.name).is_none() {
                return Err(CompileError::Internal(format!(
                    "canonical periodic target '{}' not materialized by GEO backend",
                    target.name
                )));
            }
            let coefficients = pair
                .transform
                .coefficients
                .iter()
                .map(|value| format_float(*value))
                .collect::<Vec<_>>()
                .join(", ");
            self.line(&format!(
                "Periodic {} {{{}}} = {{{}}} Translate {{{}}};",
                dimension.keyword(),
                self.entity_geo_ref(&source.name),
                self.entity_geo_ref(&target.name),
                coefficients
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
            let source_dim = gmsh_dim_from_entity_dimension(embed.source_dimension);
            let target_dim = gmsh_dim_from_entity_dimension(embed.target_dimension);
            if self.tags.lookup(&source.name).is_none() {
                return Err(CompileError::Internal(format!(
                    "canonical mesh embed source '{}' not materialized by GEO backend",
                    source.name
                )));
            }
            if self.tags.lookup(&target.name).is_none() {
                return Err(CompileError::Internal(format!(
                    "canonical mesh embed target '{}' not materialized by GEO backend",
                    target.name
                )));
            }
            self.line(&format!(
                "{}{{{}}} In {}{{{}}};",
                source_dim.keyword(),
                self.entity_geo_ref(&source.name),
                target_dim.keyword(),
                self.entity_geo_ref(&target.name)
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
            let dimension = gmsh_dim_from_entity_dimension(first.dimension);
            let tags = compound
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
                    if self.tags.lookup(&node.name).is_none() {
                        return Err(CompileError::Internal(format!(
                            "canonical mesh compound entity '{}' not materialized by GEO backend",
                            node.name
                        )));
                    }
                    Ok(self.entity_geo_ref(&node.name))
                })
                .collect::<Result<Vec<_>, CompileError>>()?;
            self.line(&format!(
                "Compound {} {{{}}};",
                dimension.keyword(),
                tags.join(", ")
            ));
        }

        // Canonical boundary-layer controls are emitted after the source mesh
        // declaration, so compose their fields here rather than relying on
        // the source declaration's refine-field finalization pass.
        if !self.refine_field_ids.is_empty() {
            let min_id = self.tags.alloc_anonymous(GmshDim::Point).tag;
            self.line(&format!("Field[{}] = Min;", min_id));
            let mut ids = self
                .refine_field_ids
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>();
            if let Some(previous) = self.background_field_id {
                ids.insert(0, previous.to_string());
            }
            self.line(&format!(
                "Field[{}].FieldsList = {{{}}};",
                min_id,
                ids.join(", ")
            ));
            self.line(&format!("Background Field = {};", min_id));
            self.background_field_id = Some(min_id);
            self.refine_field_ids.clear();
        }

        Ok(())
    }

    pub fn tag_allocator(&self) -> &TagAllocator {
        &self.tags
    }

    pub fn emit_source_file(&mut self, file: &SourceFile) -> Result<(), CompileError> {
        // Resolve names first
        self.scope = Some(name_resolution::resolve(file)?);
        self.preserved_entities = collect_preserved_entities(file);

        // Collect user-defined functions for inlining
        for decl in &file.declarations {
            if let Declaration::Function(f) = decl {
                self.functions.insert(f.name.name.clone(), f.clone());
            }
        }

        // Header
        self.line(&format!(
            "// Generated by MeshLang compiler v{}",
            env!("CARGO_PKG_VERSION")
        ));
        self.line("// Do not edit — regenerate from .mlg source");
        self.line("");
        self.line("SetFactory(\"OpenCASCADE\");");
        self.line("Geometry.OldNewReg = 0;");
        self.line("");

        // Emit declarations in order
        for decl in &file.declarations {
            self.emit_declaration(decl)?;
        }

        // Emit physical groups
        for pg in self.tags.physical_groups().to_vec() {
            let entities_str: Vec<String> = pg.entities.iter().map(|t| t.to_string()).collect();
            let dim_str = match pg.dim {
                GmshDim::Point => "Point",
                GmshDim::Curve => "Curve",
                GmshDim::Surface => "Surface",
                GmshDim::Volume => "Volume",
            };
            self.line(&format!(
                "Physical {}(\"{}\", {}) = {{{}}};",
                dim_str,
                pg.name,
                pg.tag,
                entities_str.join(", ")
            ));
            self.emitted_physical_groups.insert(pg.name.clone());
        }

        Ok(())
    }

    fn emit_declaration(&mut self, decl: &Declaration) -> Result<(), CompileError> {
        match decl {
            Declaration::Param(p) => self.emit_param(p),
            Declaration::Const(c) => self.emit_const(c),
            Declaration::Geometry(g) if g.params.is_empty() && g.ports.is_empty() => {
                self.emit_geometry(g)
            }
            Declaration::Geometry(g) => {
                self.line(&format!("// geometry template: {}", g.name.name));
                Ok(())
            }
            Declaration::Mesh(m) => self.emit_mesh(m),
            Declaration::Physics(p) => self.emit_physics(p),
            Declaration::Export(e) => self.emit_export(e),
            Declaration::Import(imp) => {
                // CAD imports (STEP, IGES, BREP, STL, OBJ) are merged into the model
                let ext = imp.path.rsplit('.').next().unwrap_or("");
                match ext.to_lowercase().as_str() {
                    "step" | "stp" | "iges" | "igs" | "brep" | "stl" | "obj" => {
                        self.line(&format!("Merge \"{}\";", imp.path));
                    }
                    "mlg" => {
                        // .mlg imports are resolved during the flatten pass
                    }
                    _ => {
                        self.line(&format!("// import: {}", imp.path));
                    }
                }
                Ok(())
            }
            Declaration::Function(_) => {
                // Functions are inlined at call sites
                Ok(())
            }
            Declaration::Struct(_) => {
                // Structs are type-only, no codegen
                Ok(())
            }
            Declaration::DesignVariable(variable) => {
                let value = self.emit_expr_str(&variable.initial)?;
                self.line(&format!("{} = {};", variable.name.name, value));
                Ok(())
            }
            Declaration::Uncertainty(value) => {
                self.line(&format!(
                    "// uncertainty: {} (see provenance manifest)",
                    value.name.name
                ));
                Ok(())
            }
            Declaration::PhysicalRegion(region) => self.emit_named_physical_group(
                &region.name.name,
                region.dimension,
                &region.entities,
            ),
            Declaration::Boundary(boundary) => self.emit_named_physical_group(
                &boundary.name.name,
                boundary.dimension,
                &boundary.entities,
            ),
            Declaration::Interface(interface) => {
                self.emit_named_physical_group(
                    &format!("{}_primary", interface.name.name),
                    interface.dimension,
                    &interface.primary,
                )?;
                self.emit_named_physical_group(
                    &format!("{}_secondary", interface.name.name),
                    interface.dimension,
                    &interface.secondary,
                )
            }
            Declaration::MaterialRegion(region) => {
                let dimension = self
                    .expr_gmsh_dim(&region.region)
                    .unwrap_or(GmshDim::Volume);
                self.emit_named_physical_group(
                    &region.name.name,
                    entity_dimension_from_gmsh(dimension),
                    &region.region,
                )
            }
            Declaration::Objective(_)
            | Declaration::Constraint(_)
            | Declaration::DesignRegion(_)
            | Declaration::CoordinateSystem(_)
            | Declaration::GeometryField(_) => {
                self.line("// modelling metadata: see provenance manifest");
                Ok(())
            }
        }
    }

    fn emit_named_physical_group(
        &mut self,
        name: &str,
        dimension: EntityDimension,
        entities: &Expr,
    ) -> Result<(), CompileError> {
        let keyword = gmsh_dim_from_entity_dimension(dimension).keyword();
        if let Some(path) = entity_path(entities) {
            if path.ends_with(".faces") {
                let owner = path.trim_end_matches(".faces");
                let reference = self.entity_geo_ref(owner);
                self.line(&format!(
                    "Physical Surface(\"{}\") = Boundary{{ Volume{{{}}}; }};",
                    name, reference
                ));
                self.emitted_physical_groups.insert(name.to_string());
                return Ok(());
            }
        }
        let references = match entities {
            Expr::Array(array) => array
                .elements
                .iter()
                .map(|entity| self.resolve_entity_tag(entity))
                .collect::<Result<Vec<_>, _>>()?,
            _ => vec![self.resolve_entity_tag(entities)?],
        };
        self.line(&format!(
            "Physical {}(\"{}\") = {{{}}};",
            keyword,
            name,
            references.join(", ")
        ));
        self.emitted_physical_groups.insert(name.to_string());
        Ok(())
    }

    fn emit_param(&mut self, param: &ParamDecl) -> Result<(), CompileError> {
        let val = self.emit_expr_str(&param.value)?;
        self.line(&format!("{} = {};", param.name.name, val));
        Ok(())
    }

    fn emit_const(&mut self, c: &ConstDecl) -> Result<(), CompileError> {
        let val = self.emit_expr_str(&c.value)?;
        self.line(&format!("{} = {};", c.name.name, val));
        Ok(())
    }

    fn emit_geometry(&mut self, geom: &GeometryBlock) -> Result<(), CompileError> {
        self.line(&format!("// geometry: {}", geom.name.name));
        self.current_geometry = Some(geom.name.name.clone());

        for stmt in &geom.body.statements {
            match stmt {
                Statement::Let(l) => self.emit_geometry_let(l)?,
                Statement::Expr(e) => {
                    let _ = self.emit_expr_str(e)?;
                }
                Statement::For(f) => self.emit_geometry_for(f)?,
                Statement::If(i) => self.emit_geometry_if(i)?,
                _ => {}
            }
        }

        self.current_geometry = None;
        self.line("");
        Ok(())
    }

    fn emit_geometry_let(&mut self, l: &LetStatement) -> Result<(), CompileError> {
        let qname = if let Some(ref geo) = self.current_geometry {
            format!("{}.{}", geo, l.name.name)
        } else {
            l.name.name.clone()
        };

        match &l.value {
            Expr::Constructor(c) => {
                self.emit_shape_constructor(&qname, c)?;
            }
            Expr::Pipe(p) => {
                // Evaluate the pipe chain
                self.emit_pipe_chain(&qname, p)?;
            }
            Expr::Binary(b) if self.is_csg_binary(b) => {
                self.emit_csg(&qname, b)?;
            }
            Expr::Call(c) if self.functions.contains_key(&c.function.name) => {
                let inlined = self.try_inline_call(c)?;
                self.emit_geometry_let_value(&qname, &l.name.name, &inlined)?;
            }
            Expr::StructInit(si) => {
                // Emit struct fields as prefixed variables: name_field = value;
                for field in &si.fields {
                    let val = self.emit_expr_str(&field.value)?;
                    self.line(&format!("{}_{} = {};", l.name.name, field.name.name, val));
                }
            }
            _ => {
                let val = self.emit_expr_str(&l.value)?;
                self.line(&format!("{} = {};", l.name.name, val));
            }
        }
        Ok(())
    }

    /// Emit a geometry let binding from an already-resolved value expression.
    fn emit_geometry_let_value(
        &mut self,
        qname: &str,
        _var_name: &str,
        expr: &Expr,
    ) -> Result<(), CompileError> {
        match expr {
            Expr::Constructor(c) => self.emit_shape_constructor(qname, c),
            Expr::Pipe(p) => self.emit_pipe_chain(qname, p),
            Expr::Binary(b) if self.is_csg_binary(b) => self.emit_csg(qname, b),
            _ => {
                let val = self.emit_expr_str(expr)?;
                self.line(&format!(
                    "{} = {};",
                    qname.split('.').next_back().unwrap_or(qname),
                    val
                ));
                Ok(())
            }
        }
    }

    /// Compile-time unroll a for loop in a geometry body.
    /// Each iteration produces entities; all are BooleanUnion'd at the end.
    fn emit_geometry_for(&mut self, f: &ForStatement) -> Result<(), CompileError> {
        let ranges = match eval_for_ranges(&f.clauses) {
            Ok(ranges) => ranges,
            Err(CompileError::Unsupported(message)) if message.contains("compile-time integer") => {
                return self.emit_dynamic_geometry_for(f);
            }
            Err(error) => return Err(error),
        };

        // Skip if any range is empty
        if ranges.iter().any(|(start, end)| start >= end) {
            return Ok(());
        }

        let mut iteration_volumes: Vec<String> = Vec::new();

        // Cartesian product of all clause ranges
        let mut indices: Vec<i64> = vec![0; ranges.len()];
        let starts: Vec<i64> = ranges.iter().map(|r| r.0).collect();
        let ends: Vec<i64> = ranges.iter().map(|r| r.1).collect();
        let var_names: Vec<String> = f.clauses.iter().map(|c| c.var.name.clone()).collect();

        // Initialize
        for (i, s) in starts.iter().enumerate() {
            indices[i] = *s;
        }

        loop {
            // Build substitution map for this iteration
            let mut subst: HashMap<String, Expr> = HashMap::new();
            for (i, name) in var_names.iter().enumerate() {
                subst.insert(
                    name.clone(),
                    Expr::literal(Literal::Int(indices[i]), f.span),
                );
            }

            // Emit each statement in the body with substituted loop vars
            for stmt in &f.body.statements {
                match stmt {
                    Statement::Let(l) => {
                        let idx_str: Vec<String> = indices.iter().map(|i| i.to_string()).collect();
                        let iter_suffix = idx_str.join("_");
                        let iter_name = format!("{}_{}", l.name.name, iter_suffix);
                        let qname = if let Some(ref geo) = self.current_geometry {
                            format!("{}.{}", geo, iter_name)
                        } else {
                            iter_name.clone()
                        };
                        let subst_value = substitute_expr(&l.value, &subst);
                        self.emit_geometry_let_value(&qname, &iter_name, &subst_value)?;
                        if self
                            .tags
                            .lookup(&qname)
                            .is_some_and(|entity| entity.dim == GmshDim::Volume)
                        {
                            iteration_volumes.push(self.entity_geo_ref(&qname));
                        }
                    }
                    Statement::Expr(e) => {
                        let subst_expr = substitute_expr(e, &subst);
                        let dimension = self.expr_gmsh_dim(&subst_expr);
                        let entity_ref = self.ensure_entity_ref(&subst_expr)?;
                        if dimension == Some(GmshDim::Volume) {
                            iteration_volumes.push(entity_ref);
                        }
                    }
                    _ => {}
                }
            }

            // Advance indices (rightmost first)
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
                break; // all indices wrapped around
            }
        }

        // BooleanUnion all iteration results if more than one volume
        if iteration_volumes.len() > 1 {
            let mut acc_ref = iteration_volumes[0].clone();
            for tag_ref in &iteration_volumes[1..] {
                let result_name = self.fresh_temp("for_union");
                let gv = self.alloc_runtime_entity_list(GmshDim::Volume, &result_name);
                self.line(&format!(
                    "{}[] = BooleanUnion {{ Volume{{{}}}; Delete; }}{{ Volume{{{}}}; Delete; }};",
                    gv, acc_ref, tag_ref
                ));
                acc_ref = format!("{}[]", gv);
            }
        }

        Ok(())
    }

    /// Constant-fold an if statement in a geometry body.
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
        self.line(&format!("{}[] = {{}};", name));
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
        entity_ref: &str,
    ) -> Result<(), CompileError> {
        let list = self.runtime_loop_list(dim)?;
        self.line(&format!("{}[] += {{{}}};", list, entity_ref));
        Ok(())
    }

    fn runtime_loop_record_list(
        &mut self,
        dim: GmshDim,
        child_list: &str,
    ) -> Result<(), CompileError> {
        let list = self.runtime_loop_list(dim)?;
        self.line(&format!("{}[] += {}[];", list, child_list));
        Ok(())
    }

    fn runtime_loop_collect_body_entities(
        &mut self,
        statements: &[Statement],
    ) -> Result<(), CompileError> {
        for statement in statements {
            match statement {
                Statement::Let(binding) => {
                    let Some(dim) = self.expr_gmsh_dim(&binding.value) else {
                        continue;
                    };
                    let qualified = self
                        .current_geometry
                        .as_ref()
                        .map(|geometry| format!("{geometry}.{}", binding.name.name))
                        .unwrap_or_else(|| binding.name.name.clone());
                    if self.tags.lookup(&qualified).is_none() {
                        continue;
                    }
                    self.runtime_loop_note_source(qualified.clone());
                    let entity_ref = self.entity_geo_ref(&qualified);
                    self.runtime_loop_record_entity(dim, &entity_ref)?;
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
            let list_ref = format!("{}[]", list);
            for source in &source_names {
                if self
                    .tags
                    .lookup(source)
                    .is_some_and(|entity| entity.dim == dim)
                {
                    self.tags.reassign_geo_var(source, &list_ref, dim);
                }
            }
            if has_parent {
                self.runtime_loop_record_list(dim, &list)?;
            } else {
                self.line(&format!(
                    "// MeshLang runtime loop aggregate {:?}: {}",
                    dim, list_ref
                ));
            }
        }

        for source in source_names {
            self.runtime_loop_note_source(source);
        }
        Ok(())
    }

    /// Emit a runtime loop while retaining all geometry entities it creates.
    fn emit_dynamic_geometry_for(&mut self, f: &ForStatement) -> Result<(), CompileError> {
        self.runtime_loop_counter += 1;
        self.runtime_loops.push(RuntimeLoopContext {
            id: self.runtime_loop_counter,
            ..RuntimeLoopContext::default()
        });
        let result = self.emit_dynamic_geometry_for_clause(f, 0);
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
        clause_index: usize,
    ) -> Result<(), CompileError> {
        let clause = f.clauses.get(clause_index).ok_or_else(|| {
            CompileError::Internal("dynamic loop clause index out of bounds".into())
        })?;
        let start = self.emit_expr_str(&clause.range.start)?;
        let end = self.emit_expr_str(&clause.range.end)?;
        let end = if clause.range.inclusive {
            end
        } else {
            format!("({end}) - 1")
        };
        self.line(&format!("For {} In {{{start}:{end}}}", clause.var.name));
        self.indent += 1;
        if clause_index + 1 < f.clauses.len() {
            self.emit_dynamic_geometry_for_clause(f, clause_index + 1)?;
        } else {
            self.emit_geometry_statement_list(&f.body.statements)?;
            self.runtime_loop_collect_body_entities(&f.body.statements)?;
        }
        self.indent -= 1;
        self.line("EndFor");
        Ok(())
    }

    fn emit_geometry_statement_list(
        &mut self,
        statements: &[Statement],
    ) -> Result<(), CompileError> {
        for statement in statements {
            match statement {
                Statement::Let(binding) => self.emit_geometry_let(binding)?,
                Statement::Expr(expression) => {
                    let _ = self.emit_expr_str(expression)?;
                }
                Statement::For(for_statement) => self.emit_geometry_for(for_statement)?,
                Statement::If(if_statement) => self.emit_geometry_if(if_statement)?,
                _ => {}
            }
        }
        Ok(())
    }

    fn emit_geometry_elseif(&mut self, i: &IfStatement) -> Result<(), CompileError> {
        let condition = self.emit_expr_str(&i.condition)?;
        self.line(&format!("ElseIf ({condition})"));
        self.indent += 1;
        self.emit_geometry_statement_list(&i.then_block.statements)?;
        if !self.runtime_loops.is_empty() {
            self.runtime_loop_collect_body_entities(&i.then_block.statements)?;
        }
        self.indent -= 1;
        if let Some(else_branch) = &i.else_block {
            match else_branch {
                ElseBranch::Else(block) => {
                    self.line("Else");
                    self.indent += 1;
                    self.emit_geometry_statement_list(&block.statements)?;
                    if !self.runtime_loops.is_empty() {
                        self.runtime_loop_collect_body_entities(&block.statements)?;
                    }
                    self.indent -= 1;
                }
                ElseBranch::ElseIf(elif) => self.emit_geometry_elseif(elif)?,
            }
        }
        Ok(())
    }

    fn emit_geometry_if(&mut self, i: &IfStatement) -> Result<(), CompileError> {
        if let Some(cond_val) = try_eval_const_bool(&i.condition) {
            if cond_val {
                for stmt in &i.then_block.statements {
                    match stmt {
                        Statement::Let(l) => self.emit_geometry_let(l)?,
                        Statement::Expr(e) => {
                            let _ = self.emit_expr_str(e)?;
                        }
                        Statement::For(f) => self.emit_geometry_for(f)?,
                        Statement::If(nested) => self.emit_geometry_if(nested)?,
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
                                Statement::Let(l) => self.emit_geometry_let(l)?,
                                Statement::Expr(e) => {
                                    let _ = self.emit_expr_str(e)?;
                                }
                                Statement::For(f) => self.emit_geometry_for(f)?,
                                Statement::If(nested) => self.emit_geometry_if(nested)?,
                                _ => {}
                            }
                        }
                    }
                    ElseBranch::ElseIf(elif) => self.emit_geometry_if(elif)?,
                }
            }
        } else {
            let condition = self.emit_expr_str(&i.condition)?;
            self.line(&format!("If ({condition})"));
            self.indent += 1;
            self.emit_geometry_statement_list(&i.then_block.statements)?;
            if !self.runtime_loops.is_empty() {
                self.runtime_loop_collect_body_entities(&i.then_block.statements)?;
            }
            self.indent -= 1;
            if let Some(else_branch) = &i.else_block {
                match else_branch {
                    ElseBranch::Else(block) => {
                        self.line("Else");
                        self.indent += 1;
                        self.emit_geometry_statement_list(&block.statements)?;
                        if !self.runtime_loops.is_empty() {
                            self.runtime_loop_collect_body_entities(&block.statements)?;
                        }
                        self.indent -= 1;
                    }
                    ElseBranch::ElseIf(elif) => self.emit_geometry_elseif(elif)?,
                }
            }
            self.line("EndIf");
        }
        Ok(())
    }

    /// Inline a single-expression function call by substituting arguments.
    fn try_inline_call(&self, call: &CallExpr) -> Result<Expr, CompileError> {
        let func = self.functions.get(&call.function.name).ok_or_else(|| {
            CompileError::UndefinedReference {
                name: call.function.name.clone(),
            }
        })?;

        // Build parameter → argument mapping
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

        // Find the return expression: last statement in the body
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

    /// Allocate a runtime tag and emit `var = newdim;`. Returns the .geo variable name.
    fn alloc_runtime_tag(&mut self, dim: GmshDim, name: &str) -> String {
        self.tags.alloc(dim, name);
        let geo_var = self
            .tags
            .geo_var(name)
            .expect("alloc always registers a geo_var")
            .to_string();
        self.line(&format!("{} = {};", geo_var, dim.new_tag_fn()));
        geo_var
    }

    /// Register a conceptual entity whose Gmsh operation can return one or
    /// more concrete entities. The tag map retains one stable compiler ID,
    /// while `.geo` references use the complete runtime array.
    fn alloc_runtime_entity_list(&mut self, dim: GmshDim, name: &str) -> String {
        self.tags.alloc(dim, name);
        let geo_var = self
            .tags
            .geo_var(name)
            .expect("alloc always registers a geo_var")
            .to_string();
        self.tags
            .reassign_geo_var(name, &format!("{}[]", geo_var), dim);
        geo_var
    }

    fn emit_shape_constructor(
        &mut self,
        name: &str,
        c: &ConstructorExpr,
    ) -> Result<(), CompileError> {
        // 2D primitives allocate as Surface, not Volume
        match c.type_name.name.as_str() {
            "Rectangle" => {
                let gv = self.alloc_runtime_tag(GmshDim::Surface, name);
                let args = self.eval_positional_args(&c.args, &["x", "y"])?;
                self.line(&format!(
                    "Rectangle({}) = {{0, 0, 0, {}, {}}};",
                    gv, args[0], args[1]
                ));
                return Ok(());
            }
            "Disk" => {
                let gv = self.alloc_runtime_tag(GmshDim::Surface, name);
                let r = self.eval_named_arg(&c.args, "r")?;
                self.line(&format!("Disk({}) = {{0, 0, 0, {}}};", gv, r));
                return Ok(());
            }
            "Circle" => {
                let gv = self.alloc_runtime_tag(GmshDim::Surface, name);
                let r = self.eval_named_arg(&c.args, "r")?;
                self.line(&format!("Disk({}) = {{0, 0, 0, {}}};", gv, r));
                return Ok(());
            }
            "Polygon" => {
                if let Some(Expr::Array(arr)) = self.find_named_arg_expr(&c.args, "points") {
                    if arr.elements.len() < 3 {
                        return Err(CompileError::Internal(
                            "Polygon requires at least 3 points".into(),
                        ));
                    }
                    let mut point_vars = Vec::new();
                    for (idx, pt) in arr.elements.iter().enumerate() {
                        if let Expr::Array(coords) = pt {
                            let pt_name = format!("{}__pt{}", name, idx);
                            let pv = self.alloc_runtime_tag(GmshDim::Point, &pt_name);
                            let vals: Vec<String> = coords
                                .elements
                                .iter()
                                .map(|e| self.emit_expr_str(e))
                                .collect::<Result<_, _>>()?;
                            let (x, y, z) = match vals.len() {
                                2 => (vals[0].clone(), vals[1].clone(), "0".to_string()),
                                3 => (vals[0].clone(), vals[1].clone(), vals[2].clone()),
                                _ => {
                                    return Err(CompileError::Internal(
                                        "point must have 2 or 3 coordinates".into(),
                                    ));
                                }
                            };
                            self.line(&format!("Point({}) = {{{}, {}, {}}};", pv, x, y, z));
                            point_vars.push(pv);
                        }
                    }
                    let mut line_vars = Vec::new();
                    for i in 0..point_vars.len() {
                        let next = (i + 1) % point_vars.len();
                        let ln_name = format!("{}__ln{}", name, i);
                        let lv = self.alloc_runtime_tag(GmshDim::Curve, &ln_name);
                        self.line(&format!(
                            "Line({}) = {{{}, {}}};",
                            lv, point_vars[i], point_vars[next]
                        ));
                        line_vars.push(lv);
                    }
                    let cl_name = format!("{}__cl", name);
                    let cl_var = self.alloc_runtime_tag(GmshDim::Curve, &cl_name);
                    self.line(&format!(
                        "Curve Loop({}) = {{{}}};",
                        cl_var,
                        line_vars.join(", ")
                    ));
                    let gv = self.alloc_runtime_tag(GmshDim::Surface, name);
                    self.line(&format!("Plane Surface({}) = {{{}}};", gv, cl_var));
                    return Ok(());
                }
                return Err(CompileError::Internal(
                    "Polygon requires 'points' array argument".into(),
                ));
            }
            "Point" => {
                // Point(x, y, z) — standalone 3D point
                let args = self.eval_positional_args(&c.args, &["x", "y", "z"])?;
                let gv = self.alloc_runtime_tag(GmshDim::Point, name);
                self.line(&format!(
                    "Point({}) = {{{}, {}, {}}};",
                    gv, args[0], args[1], args[2]
                ));
                return Ok(());
            }
            "Line" => {
                let from = self.eval_named_arg(&c.args, "from")?;
                let to = self.eval_named_arg(&c.args, "to")?;
                let gv = self.alloc_runtime_tag(GmshDim::Curve, name);
                self.line(&format!("Line({}) = {{{}, {}}};", gv, from, to));
                return Ok(());
            }
            "Arc" => {
                let args = self.eval_positional_args(&c.args, &["start", "center", "end"])?;
                let gv = self.alloc_runtime_tag(GmshDim::Curve, name);
                self.line(&format!(
                    "Circle({}) = {{{}, {}, {}}};",
                    gv, args[0], args[1], args[2]
                ));
                return Ok(());
            }
            "Spline" => {
                // Spline(p0, p1, p2, ...) — accepts point identifiers or tags
                let pt_tags: Vec<String> = c
                    .args
                    .iter()
                    .map(|a| self.resolve_point_tag(&a.value))
                    .collect::<Result<_, _>>()?;
                let gv = self.alloc_runtime_tag(GmshDim::Curve, name);
                self.line(&format!("Spline({}) = {{{}}};", gv, pt_tags.join(", ")));
                return Ok(());
            }
            "BSpline" => {
                let pt_tags: Vec<String> = c
                    .args
                    .iter()
                    .map(|a| self.resolve_point_tag(&a.value))
                    .collect::<Result<_, _>>()?;
                let gv = self.alloc_runtime_tag(GmshDim::Curve, name);
                self.line(&format!("BSpline({}) = {{{}}};", gv, pt_tags.join(", ")));
                return Ok(());
            }
            _ => {}
        }

        // 3D primitives — use runtime tag allocation
        let gv = self.alloc_runtime_tag(GmshDim::Volume, name);

        match c.type_name.name.as_str() {
            "Box" => {
                let args = self.eval_positional_args(&c.args, &["x", "y", "z"])?;
                self.line(&format!(
                    "Box({}) = {{0, 0, 0, {}, {}, {}}};",
                    gv, args[0], args[1], args[2]
                ));
                self.topo.register_box(name, 0);
            }
            "Sphere" => {
                let r = self.eval_named_arg(&c.args, "r")?;
                self.line(&format!("Sphere({}) = {{0, 0, 0, {}}};", gv, r));
                self.topo.register_sphere(name, 0);
            }
            "Cylinder" => {
                let r = self.eval_named_arg(&c.args, "r")?;
                let h = self.eval_named_arg(&c.args, "h")?;
                self.line(&format!(
                    "Cylinder({}) = {{0, 0, 0, 0, 0, {}, {}}};",
                    gv, h, r
                ));
                self.topo.register_cylinder(name, 0);
            }
            "Cone" => {
                let r1 = self.eval_named_arg(&c.args, "r1")?;
                let r2 = self.eval_named_arg(&c.args, "r2")?;
                let h = self.eval_named_arg(&c.args, "h")?;
                self.line(&format!(
                    "Cone({}) = {{0, 0, 0, 0, 0, {}, {}, {}}};",
                    gv, h, r1, r2
                ));
            }
            "Torus" => {
                let r_major = self.eval_named_arg(&c.args, "r_major")?;
                let r_minor = self.eval_named_arg(&c.args, "r_minor")?;
                self.line(&format!(
                    "Torus({}) = {{0, 0, 0, {}, {}}};",
                    gv, r_major, r_minor
                ));
            }
            "Wedge" => {
                let args = self.eval_positional_args(&c.args, &["x", "y", "z", "ltx"])?;
                self.line(&format!(
                    "Wedge({}) = {{0, 0, 0, {}, {}, {}, {}}};",
                    gv, args[0], args[1], args[2], args[3]
                ));
            }
            "Loft" => {
                // Loft(sections: [s1, s2, ...], ruled: bool) → ThruSections
                // ThruSections takes curve loop / wire tags (integers). The
                // optional `ruled` flag selects linear interpolation between
                // sections (gmsh `Ruled ThruSections`) instead of the default
                // smooth cubic B-spline. Linear interpolation is bounded by
                // the convex hull of the sections — no spline overshoot,
                // which matters when sections have sharp corners (rectangles).
                let section_exprs =
                    if let Some(sections_expr) = self.find_named_arg_expr(&c.args, "sections") {
                        if let Expr::Array(arr) = sections_expr {
                            arr.elements.clone()
                        } else {
                            return Err(CompileError::Internal(
                                "Loft 'sections' must be an array".into(),
                            ));
                        }
                    } else {
                        c.args.iter().map(|a| a.value.clone()).collect()
                    };
                let section_tags: Vec<String> = section_exprs
                    .iter()
                    .map(|e| self.resolve_loft_section_tag(e))
                    .collect::<Result<_, _>>()?;
                let ruled = self.find_named_arg_bool(&c.args, "ruled").unwrap_or(false);
                let prefix = if ruled {
                    "Ruled ThruSections"
                } else {
                    "ThruSections"
                };
                self.line(&format!(
                    "{}({}) = {{{}}};",
                    prefix,
                    gv,
                    section_tags.join(", ")
                ));
            }
            other => {
                return Err(CompileError::Unsupported(format!(
                    "shape constructor '{}'",
                    other
                )));
            }
        }

        if self.topo.lookup(name).is_none() {
            self.topo.register_unknown(name, 0);
        }

        Ok(())
    }

    fn emit_pipe_chain(&mut self, name: &str, pipe: &PipeExpr) -> Result<(), CompileError> {
        // First, emit the base value
        match pipe.value.as_ref() {
            Expr::Constructor(c) => {
                self.emit_shape_constructor(name, c)?;
            }
            Expr::Pipe(inner) => {
                self.emit_pipe_chain(name, inner)?;
            }
            Expr::Ident(id) => {
                // Reference to an existing entity. The two MeshLang names get
                // distinct compiler IDs while sharing the runtime .geo tag
                // variable; later in-place transforms therefore remain
                // traceable without breaking the reverse-map bijection.
                let source_name = self.expr_entity_name(&pipe.value)?;
                if let Some(entity) = self.tags.lookup(&source_name) {
                    let gv = self
                        .tags
                        .geo_var(&source_name)
                        .map(str::to_owned)
                        .unwrap_or_else(|| entity.tag.to_string());
                    self.tags.reassign_geo_var(name, &gv, entity.dim);
                    self.topo.register_alias(name, &source_name);
                } else {
                    self.line(&format!("// pipe base: {}", id.name));
                }
            }
            _ => {
                let val = self.emit_expr_str(&pipe.value)?;
                self.line(&format!("// pipe base: {}", val));
            }
        }

        // Then apply the transform
        let entity = self
            .tags
            .lookup(name)
            .ok_or_else(|| CompileError::UndefinedReference {
                name: name.to_string(),
            })?;

        // Use the entity's actual dimension and .geo variable reference
        let dim_kw = entity.dim.keyword();
        let eref = self.entity_geo_ref(name);

        match pipe.function.name.as_str() {
            "translate" => {
                let args = self.eval_positional_args(&pipe.args, &["dx", "dy", "dz"])?;
                self.line(&format!(
                    "Translate {{{}, {}, {}}} {{ {}{{{}}}; }}",
                    args[0], args[1], args[2], dim_kw, eref
                ));
            }
            "rotate" => {
                let angle = self.angle_arg_degrees_to_radians(&pipe.args)?;
                let axis_str = self.eval_named_arg_str(&pipe.args, "axis")?;
                let (ax, ay, az) = match axis_str.as_str() {
                    "X" => ("1".to_string(), "0".to_string(), "0".to_string()),
                    "Y" => ("0".to_string(), "1".to_string(), "0".to_string()),
                    "Z" => ("0".to_string(), "0".to_string(), "1".to_string()),
                    _ => {
                        // Try to parse as array expression [ax, ay, az]
                        let axis_expr = self.find_named_arg_expr(&pipe.args, "axis");
                        match axis_expr {
                            Some(Expr::Array(arr)) if arr.elements.len() == 3 => {
                                let ax = self.emit_expr_str(&arr.elements[0])?;
                                let ay = self.emit_expr_str(&arr.elements[1])?;
                                let az = self.emit_expr_str(&arr.elements[2])?;
                                (ax, ay, az)
                            }
                            _ => ("0".to_string(), "0".to_string(), "1".to_string()),
                        }
                    }
                };
                // Gmsh syntax: Rotate { {axis-direction}, {point-on-axis}, angle }
                self.line(&format!(
                    "Rotate {{{{{}, {}, {}}}, {{0, 0, 0}}, {}}} {{ {}{{{}}}; }}",
                    ax, ay, az, angle, dim_kw, eref
                ));
            }
            "scale" => {
                // scale(factor) for uniform, scale(sx, sy, sz) for non-uniform
                let args = if let Ok(a) = self.eval_positional_args(&pipe.args, &["sx", "sy", "sz"])
                {
                    a
                } else {
                    let f = self.eval_positional_args(&pipe.args, &["factor"])?;
                    vec![f[0].clone(), f[0].clone(), f[0].clone()]
                };
                self.line(&format!(
                    "Dilate {{{{0, 0, 0}}, {{{}, {}, {}}}}} {{ {}{{{}}}; }}",
                    args[0], args[1], args[2], dim_kw, eref
                ));
            }
            "mirror" => {
                // mirror(nx, ny, nz) — reflect through plane nx*x + ny*y + nz*z = 0
                // Gmsh Symmetry takes {a, b, c, d} for plane ax+by+cz+d=0
                let args = self.eval_positional_args(&pipe.args, &["nx", "ny", "nz"])?;
                self.line(&format!(
                    "Symmetry {{{}, {}, {}, 0}} {{ {}{{{}}}; }}",
                    args[0], args[1], args[2], dim_kw, eref
                ));
            }
            "affine" => {
                let matrix = self.affine_matrix_args(&pipe.args)?;
                self.line(&format!(
                    "Affine {{{}}} {{ {}{{{}}}; }}",
                    matrix.join(", "),
                    dim_kw,
                    eref
                ));
            }
            "extrude" => {
                let args = self.eval_positional_args(&pipe.args, &["dx", "dy", "dz"])?;
                let out_var = format!("__extrude_{}", self.temp_counter);
                self.temp_counter += 1;
                self.line(&format!(
                    "{}[] = Extrude {{{}, {}, {}}} {{ Surface{{{}}}; }}",
                    out_var, args[0], args[1], args[2], eref
                ));
                let result_var = name.replace('.', "__");
                self.line(&format!("{} = {}[1];", result_var, out_var));
                self.tags
                    .reassign_geo_var(name, &result_var, GmshDim::Volume);
            }
            "revolve" => {
                let angle = self.angle_arg_degrees_to_radians(&pipe.args)?;
                let axis_str = self.eval_named_arg_str(&pipe.args, "axis")?;
                let (ax, ay, az) = match axis_str.as_str() {
                    "X" => ("1", "0", "0"),
                    "Y" => ("0", "1", "0"),
                    "Z" => ("0", "0", "1"),
                    _ => ("0", "0", "1"),
                };
                // Gmsh syntax: Extrude { {axis-direction}, {point-on-axis}, angle }
                let out_var = format!("__revolve_{}", self.temp_counter);
                self.temp_counter += 1;
                self.line(&format!(
                    "{}[] = Extrude {{{{{}, {}, {}}}, {{0, 0, 0}}, {}}} {{ Surface{{{}}}; }}",
                    out_var, ax, ay, az, angle, eref
                ));
                let result_var = name.replace('.', "__");
                self.line(&format!("{} = {}[1];", result_var, out_var));
                self.tags
                    .reassign_geo_var(name, &result_var, GmshDim::Volume);
            }
            "sweep" => {
                // sweep(path: curve_entity) — extrude surface along a wire path
                let path_str = self.eval_named_arg_str(&pipe.args, "path")?;
                let path_ref = if let Some(ref geo) = self.current_geometry {
                    let qualified = format!("{}.{}", geo, path_str);
                    self.entity_geo_ref(&qualified)
                } else {
                    self.entity_geo_ref(&path_str)
                };
                let wire_var = format!("__wire_{}", self.temp_counter);
                self.temp_counter += 1;
                self.line(&format!("{} = newc;", wire_var));
                self.line(&format!("Wire({}) = {{{}}};", wire_var, path_ref));
                let out_var = format!("__sweep_{}", self.temp_counter);
                self.temp_counter += 1;
                self.line(&format!(
                    "{}[] = Extrude {{ Surface{{{}}}; }} Using Wire {{{}}};",
                    out_var, eref, wire_var
                ));
                let result_var = name.replace('.', "__");
                // `Extrude ... Using Wire` returns the swept entity itself as
                // the first (and, on Gmsh 4.8, only) result.
                self.line(&format!("{} = {}[0];", result_var, out_var));
                self.tags
                    .reassign_geo_var(name, &result_var, GmshDim::Volume);
            }
            "copy" => {
                let dim_str = entity.dim.keyword();
                self.line(&format!(
                    "__copy_result[] = Duplicata {{ {}{{{}}}; }};",
                    dim_str, eref
                ));
                let copy_var = format!("__copy_{}", self.temp_counter);
                self.temp_counter += 1;
                self.line(&format!("{} = __copy_result[0];", copy_var));
                self.tags.reassign_geo_var(name, &copy_var, entity.dim);
            }
            "fragment" => {
                // fragment(tool) or fragment([t1, t2, ...]) — BooleanFragments
                let tool_arg = &pipe.args[0].value;
                let tool_refs: Vec<String> = match tool_arg {
                    Expr::Array(arr) => arr
                        .elements
                        .iter()
                        .map(|e| self.resolve_entity_tag(e))
                        .collect::<Result<Vec<_>, _>>()?,
                    _ => vec![self.resolve_entity_tag(tool_arg)?],
                };
                let tool_refs_str = tool_refs.join(", ");
                let frag_var = format!("__frag_{}", self.temp_counter);
                let result_var = name.replace('.', "__");
                self.temp_counter += 1;
                self.line(&format!(
                    "{}() = BooleanFragments{{ Volume{{{}}}; Delete; }}{{ Volume{{{}}}; Delete; }};",
                    frag_var, eref, tool_refs_str
                ));
                self.line(&format!("{} = {}(0);", result_var, frag_var));
                self.tags
                    .reassign_geo_var(name, &result_var, GmshDim::Volume);
            }
            "shell" => {
                // shell(thickness) — hollow a solid via OCC ThickSolid
                let thickness = self.eval_named_arg(&pipe.args, "thickness")?;
                let result_var = format!("__shell_{}", self.temp_counter);
                self.temp_counter += 1;
                self.line(&format!("{} = newv;", result_var));
                self.line(&format!(
                    "ThickSolid({}) = {{{}, {}}};",
                    result_var, eref, thickness
                ));
                self.tags
                    .reassign_geo_var(name, &result_var, GmshDim::Volume);
            }
            "twist" => {
                let dx = self
                    .eval_named_arg(&pipe.args, "dx")
                    .unwrap_or_else(|_| "0".into());
                let dy = self
                    .eval_named_arg(&pipe.args, "dy")
                    .unwrap_or_else(|_| "0".into());
                let dz = self
                    .eval_named_arg(&pipe.args, "dz")
                    .unwrap_or_else(|_| "0".into());
                let angle = self.angle_arg_degrees_to_radians(&pipe.args)?;
                let axis_str = self
                    .eval_named_arg_str(&pipe.args, "axis")
                    .unwrap_or_else(|_| "Z".into());
                let (ax, ay, az) = match axis_str.as_str() {
                    "X" => ("1", "0", "0"),
                    "Y" => ("0", "1", "0"),
                    "Z" => ("0", "0", "1"),
                    _ => ("0", "0", "1"),
                };
                // Gmsh syntax: Extrude { {translation}, {axis-direction}, {point-on-axis}, angle }
                let out_var = format!("__twist_{}", self.temp_counter);
                self.temp_counter += 1;
                self.line(&format!(
                    "{}[] = Extrude {{{{{}, {}, {}}}, {{{}, {}, {}}}, {{0, 0, 0}}, {}}} {{ Surface{{{}}}; }}",
                    out_var,
                    dx, dy, dz, ax, ay, az, angle, eref
                ));
                let result_var = name.replace('.', "__");
                self.line(&format!("{} = {}[1];", result_var, out_var));
                self.tags
                    .reassign_geo_var(name, &result_var, GmshDim::Volume);
            }
            "coherence" => {
                self.line("Coherence;");
            }
            "fillet" => {
                let r = self.eval_named_arg(&pipe.args, "r")?;
                let faces_var = format!("__fillet_faces_{}", self.temp_counter);
                let edges_var = format!("__fillet_edges_{}", self.temp_counter);
                self.temp_counter += 1;
                self.line(&format!(
                    "{}() = Boundary{{ Volume{{{}}}; }};",
                    faces_var, eref
                ));
                self.line(&format!(
                    "{}() = Unique(Abs(Boundary{{ Surface{{{}()}}; }}));",
                    edges_var, faces_var
                ));
                self.line(&format!("Fillet{{{}}}{{{}()}}{{{}}}", eref, edges_var, r));
            }
            "chamfer" => {
                let d = self.eval_named_arg(&pipe.args, "d")?;
                let faces_var = format!("__chamfer_faces_{}", self.temp_counter);
                let edges_var = format!("__chamfer_edges_{}", self.temp_counter);
                self.temp_counter += 1;
                self.line(&format!(
                    "{}() = Boundary{{ Volume{{{}}}; }};",
                    faces_var, eref
                ));
                self.line(&format!(
                    "{}() = Unique(Abs(Boundary{{ Surface{{{}()}}; }}));",
                    edges_var, faces_var
                ));
                self.line(&format!(
                    "Chamfer{{{}}}{{{}()}}{{{}()}}{{{}}}",
                    eref, edges_var, faces_var, d
                ));
            }
            "project_iso_surface" | "project_signed_distance" | "project_closest_point" => {
                // Native GEO has no portable field-evaluation/iso-surface
                // contract. Preserve the canonical alias and descriptor in
                // sidecars without fabricating a Gmsh geometry operation.
                self.line(&format!(
                    "// MeshLang projection {name}: metadata-only {}",
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

        if self
            .tags
            .lookup(name)
            .is_some_and(|entity| entity.dim == GmshDim::Volume)
            && self.topo.lookup(name).is_none()
        {
            self.topo.register_unknown(name, 0);
        }

        Ok(())
    }

    /// Flatten a chain of same-op CSG expressions: `a - b - c - d`
    /// into (base=a, tools=[b, c, d]).  Only flattens the same operator.
    fn flatten_csg_chain<'a>(&self, expr: &'a BinaryExpr) -> (&'a Expr, Vec<&'a Expr>, BinaryOp) {
        let op = expr.op;
        let mut tools = vec![&*expr.right];
        let mut current = &*expr.left;
        while let Expr::Binary(inner) = current {
            if same_csg_op(inner.op, op) && self.is_csg_binary(inner) {
                tools.push(&*inner.right);
                current = &*inner.left;
            } else {
                break;
            }
        }
        tools.reverse();
        (current, tools, op)
    }

    fn emit_csg(&mut self, name: &str, b: &BinaryExpr) -> Result<(), CompileError> {
        let (op_name, csg_op) = match b.op {
            BinaryOp::Sub | BinaryOp::CsgDifference => ("BooleanDifference", CsgOp::Difference),
            BinaryOp::CsgUnion => ("BooleanUnion", CsgOp::Union),
            BinaryOp::CsgIntersect => ("BooleanIntersection", CsgOp::Intersection),
            _ => {
                return Err(CompileError::Internal(format!(
                    "non-CSG op in emit_csg: {:?}",
                    b.op
                )));
            }
        };

        // Flatten chains of the same operator: a - b - c → BoolDiff(a, {b, c})
        let (base_expr, tool_exprs, _) = self.flatten_csg_chain(b);

        let left_name = self.expr_entity_name(base_expr).unwrap_or_default();
        let right_name = tool_exprs
            .first()
            .and_then(|e| self.expr_entity_name(e).ok())
            .unwrap_or_default();

        let left_ref = self.ensure_entity_ref(base_expr)?;
        let tool_refs: Vec<String> = tool_exprs
            .iter()
            .map(|e| self.ensure_entity_ref(e))
            .collect::<Result<Vec<_>, _>>()?;

        let dim = self.expr_gmsh_dim(base_expr).ok_or_else(|| {
            CompileError::Internal("cannot determine CSG operand dimension".into())
        })?;
        for tool_expr in &tool_exprs {
            let tool_dim = self.expr_gmsh_dim(tool_expr).ok_or_else(|| {
                CompileError::Internal("cannot determine CSG tool dimension".into())
            })?;
            if tool_dim != dim {
                return Err(CompileError::TypeError {
                    message: format!(
                        "CSG operands must have the same dimension ({} and {})",
                        dim as u8, tool_dim as u8
                    ),
                });
            }
        }

        let gv = self.alloc_runtime_entity_list(dim, name);
        let keyword = dim.keyword();
        let base_name = self.expr_entity_name(base_expr).ok();
        let base_delete = base_name
            .as_ref()
            .is_none_or(|base_name| !self.preserved_entities.contains(base_name));
        let base_block = format!(
            "{keyword}{{{left_ref}}};{}",
            if base_delete { " Delete;" } else { "" }
        );
        let delete_tools = tool_exprs.iter().all(|tool_expr| {
            self.expr_entity_name(tool_expr)
                .ok()
                .is_none_or(|tool_name| !self.preserved_entities.contains(&tool_name))
        });
        let tool_blocks = format!(
            "{}{}",
            tool_refs
                .iter()
                .map(|tool_ref| format!("{keyword}{{{tool_ref}}};"))
                .collect::<Vec<_>>()
                .join(" "),
            if delete_tools { " Delete;" } else { "" }
        );

        self.line(&format!(
            "{}[] = {} {{ {} }}{{ {} }};",
            gv, op_name, base_block, tool_blocks
        ));

        self.topo
            .register_csg_result(name, 0, &left_name, &right_name, csg_op);

        Ok(())
    }

    fn expr_gmsh_dim(&self, expr: &Expr) -> Option<GmshDim> {
        match expr {
            Expr::Ident(id) => {
                let qname = if let Some(ref geo) = self.current_geometry {
                    format!("{}.{}", geo, id.name)
                } else {
                    id.name.clone()
                };
                self.tags.lookup(&qname).map(|entity| entity.dim)
            }
            Expr::FieldAccess(_) => self
                .expr_entity_name(expr)
                .ok()
                .and_then(|name| self.tags.lookup(&name))
                .map(|entity| entity.dim),
            Expr::Constructor(c) => Some(constructor_gmsh_dim(c)),
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

    /// Ensure an expression has been emitted and return its .geo variable reference.
    fn ensure_entity_ref(&mut self, expr: &Expr) -> Result<String, CompileError> {
        match expr {
            Expr::Ident(id) => {
                let qname = if let Some(ref geo) = self.current_geometry {
                    format!("{}.{}", geo, id.name)
                } else {
                    id.name.clone()
                };
                Ok(self.entity_geo_ref(&qname))
            }
            Expr::FieldAccess(_) => {
                let qname = self.expr_entity_name(expr)?;
                Ok(self.entity_geo_ref(&qname))
            }
            Expr::Constructor(c) => {
                let tmp = self.fresh_temp("shape");
                self.emit_shape_constructor(&tmp, c)?;
                Ok(self.entity_geo_ref(&tmp))
            }
            Expr::Pipe(p) => {
                let tmp = self.fresh_temp("pipe");
                self.emit_pipe_chain(&tmp, p)?;
                Ok(self.entity_geo_ref(&tmp))
            }
            Expr::Binary(b) if self.is_csg_binary(b) => {
                let tmp = self.fresh_temp("csg");
                self.emit_csg(&tmp, b)?;
                Ok(self.entity_geo_ref(&tmp))
            }
            Expr::Call(c) if self.functions.contains_key(&c.function.name) => {
                let inlined = self.try_inline_call(c)?;
                self.ensure_entity_ref(&inlined)
            }
            _ => Err(CompileError::Internal(
                "expression cannot be used as CSG operand".into(),
            )),
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
                self.tags
                    .lookup(&qname)
                    .is_some_and(|entity| matches!(entity.dim, GmshDim::Surface | GmshDim::Volume))
            }
            Expr::FieldAccess(_) => self
                .expr_entity_name(expr)
                .ok()
                .and_then(|name| self.tags.lookup(&name))
                .is_some_and(|entity| matches!(entity.dim, GmshDim::Surface | GmshDim::Volume)),
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
        self.line("// mesh control");

        for stmt in &mesh.statements {
            match stmt {
                MeshStatement::KeyValue(kv) => {
                    match kv.key.name.as_str() {
                        "default_size" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Mesh.CharacteristicLengthMax = {};", val));
                            // Also set MeshSizeFromPoints for initial sizing
                            self.line("Mesh.MeshSizeFromPoints = 0;");
                            self.line("Mesh.MeshSizeExtendFromBoundary = 0;");
                        }
                        "min_size" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Mesh.CharacteristicLengthMin = {};", val));
                        }
                        "max_size" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Mesh.CharacteristicLengthMax = {};", val));
                        }
                        "order" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Mesh.ElementOrder = {};", val));
                        }
                        "algorithm_2d" => {
                            let val = self.mesh_algo_2d(&kv.value)?;
                            self.line(&format!("Mesh.Algorithm = {};", val));
                        }
                        "algorithm_3d" => {
                            let val = self.mesh_algo_3d(&kv.value)?;
                            self.line(&format!("Mesh.Algorithm3D = {};", val));
                        }
                        "optimize" => {
                            self.line("Mesh.Optimize = 1;");
                        }
                        "smoothing_steps" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Mesh.Smoothing = {};", val));
                        }
                        "recombine" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            if val == "1" || val == "true" {
                                self.line("Mesh.RecombineAll = 1;");
                            }
                        }
                        "optimize_threshold" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Mesh.OptimizeThreshold = {};", val));
                        }
                        "optimize_netgen" => {
                            self.line("Mesh.OptimizeNetgen = 1;");
                        }
                        "size_from_curvature" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Mesh.MeshSizeFromCurvature = {};", val));
                        }
                        "size_factor" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Mesh.MeshSizeFactor = {};", val));
                        }
                        "second_order_incomplete" => {
                            self.line("Mesh.SecondOrderIncomplete = 1;");
                        }
                        "subdivision" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Mesh.SubdivisionAlgorithm = {};", val));
                        }
                        "aniso_max" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Mesh.AnisoMax = {};", val));
                        }
                        "recombine_algorithm" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            let algo = match val.as_str() {
                                "simple" => "0",
                                "blossom" => "1",
                                "simple_full_quad" => "2",
                                "blossom_full_quad" => "3",
                                _ => &val,
                            };
                            self.line(&format!("Mesh.RecombinationAlgorithm = {};", algo));
                        }
                        "binary" => {
                            self.line("Mesh.Binary = 1;");
                        }
                        "save_all" => {
                            self.line("Mesh.SaveAll = 1;");
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
                    self.emit_refine(r)?;
                }
                MeshStatement::BoundaryLayer(bl) => {
                    self.emit_boundary_layer(bl)?;
                }
                MeshStatement::Transfinite(t) => {
                    self.emit_transfinite(t)?;
                }
                MeshStatement::Embed(e) => {
                    self.emit_embed(e)?;
                }
                MeshStatement::Periodic(p) => {
                    self.emit_periodic(p)?;
                }
                MeshStatement::Compound(c) => {
                    self.emit_compound(c)?;
                }
                MeshStatement::Metric(metric) => self.emit_mesh_metric(metric)?,
            }
        }

        // Frequency-aware mesh sizing: compute element size from frequency parameters
        let mut target_freq: Option<String> = None;
        let mut medium_speed: Option<String> = None;
        let mut epw: Option<String> = None;
        let mut has_default_size = false;

        for stmt in &mesh.statements {
            if let MeshStatement::KeyValue(kv) = stmt {
                match kv.key.name.as_str() {
                    "target_frequency" => target_freq = Some(self.emit_expr_str(&kv.value)?),
                    "medium_speed" => medium_speed = Some(self.emit_expr_str(&kv.value)?),
                    "elements_per_wavelength" => epw = Some(self.emit_expr_str(&kv.value)?),
                    "default_size" => has_default_size = true,
                    _ => {}
                }
            }
        }

        if let (Some(freq), Some(speed)) = (target_freq, medium_speed) {
            if !has_default_size {
                let epw_val = epw.unwrap_or_else(|| "6".to_string());
                self.line(&format!(
                    "// frequency-aware sizing: {} / {} / {}",
                    speed, freq, epw_val
                ));
                self.line(&format!(
                    "Mesh.CharacteristicLengthMax = {} / {} / {};",
                    speed, freq, epw_val
                ));
            }
        }

        // Compose all refine fields into a Min field and set as background
        if !self.refine_field_ids.is_empty() {
            let min_id = self.tags.alloc_anonymous(GmshDim::Point).tag;
            self.line(&format!("Field[{}] = Min;", min_id));
            let mut ids: Vec<String> = self
                .refine_field_ids
                .iter()
                .map(|id| id.to_string())
                .collect();
            if let Some(previous) = self.background_field_id {
                ids.insert(0, previous.to_string());
            }
            self.line(&format!(
                "Field[{}].FieldsList = {{{}}};",
                min_id,
                ids.join(", ")
            ));
            self.line(&format!("Background Field = {};", min_id));
            self.background_field_id = Some(min_id);
            self.refine_field_ids.clear();
        }

        self.line("");
        Ok(())
    }

    fn emit_refine(&mut self, refine: &RefineStatement) -> Result<(), CompileError> {
        // Emit as a Gmsh mesh size field
        match &refine.kind {
            RefineKind::Near { entity, args } => {
                let entity_str = self.emit_expr_str(entity)?;
                self.line(&format!("// refine near {} ", entity_str));
                // Build a Distance + Threshold field pair
                let field_id = self.tags.alloc_anonymous(GmshDim::Point).tag;
                self.line(&format!("Field[{}] = Distance;", field_id));

                // Resolve entity tag for the Distance field's entity list
                let topo_query = self.resolve_expr_to_topology(entity);
                match &topo_query {
                    Ok(crate::topology::TopologyQuery::Surface { tag }) => {
                        self.line(&format!("Field[{}].SurfacesList = {{{}}};", field_id, tag));
                    }
                    Ok(crate::topology::TopologyQuery::AllFaces { volume_tag }) => {
                        self.line(&format!(
                            "Field[{}].SurfacesList = {{Boundary{{ Volume{{{}}}; }}}};",
                            field_id, volume_tag
                        ));
                    }
                    _ => {}
                }

                // Extract size and distance from body
                let mut size_val_str: Option<String> = None;
                for kv in &refine.body {
                    if kv.key.name == "size" {
                        size_val_str = Some(self.emit_expr_str(&kv.value)?);
                    }
                }
                if let Some(ref size_val) = size_val_str {
                    let thresh_id = self.tags.alloc_anonymous(GmshDim::Point).tag;
                    self.line(&format!("Field[{}] = Threshold;", thresh_id));
                    self.line(&format!("Field[{}].InField = {};", thresh_id, field_id));
                    self.line(&format!("Field[{}].SizeMin = {};", thresh_id, size_val));

                    for arg in args {
                        if arg.key.name == "distance" {
                            let dist = self.emit_expr_str(&arg.value)?;
                            self.line(&format!("Field[{}].DistMax = {};", thresh_id, dist));
                        }
                    }

                    // SizeMax from growth_rate or fallback to a large multiple
                    let mut has_growth_rate = false;
                    for kv in &refine.body {
                        if kv.key.name == "growth_rate" {
                            // Approximate: SizeMax ≈ SizeMin * growth_rate^(distance/size)
                            // Simplified: just set SizeMax to a reasonable multiple
                            let gr = self.emit_expr_str(&kv.value)?;
                            self.line(&format!(
                                "Field[{}].SizeMax = {} * {};",
                                thresh_id, size_val, gr
                            ));
                            has_growth_rate = true;
                        }
                    }
                    if !has_growth_rate {
                        // Default SizeMax: 10x SizeMin to allow transition
                        self.line(&format!(
                            "Field[{}].SizeMax = {} * 10;",
                            thresh_id, size_val
                        ));
                    }

                    self.refine_field_ids.push(thresh_id);
                }
            }
            RefineKind::On { entity } => {
                let entity_str = self.emit_expr_str(entity)?;
                self.line(&format!("// refine on {}", entity_str));
                // Resolve entity to topology query for surface tags
                let topo_query = self.resolve_expr_to_topology(entity);
                for kv in &refine.body {
                    match kv.key.name.as_str() {
                        "size" => {
                            let size_val = self.emit_expr_str(&kv.value)?;
                            match &topo_query {
                                Ok(crate::topology::TopologyQuery::Surface { tag }) => {
                                    self.line(&format!(
                                        "MeshSize{{ Surface{{{}}}; }} = {};",
                                        tag, size_val
                                    ));
                                }
                                Ok(crate::topology::TopologyQuery::AllFaces { volume_tag }) => {
                                    // Apply to all faces of the volume
                                    self.line(&format!(
                                        "MeshSize{{ Boundary{{ Volume{{{}}}; }}; }} = {};",
                                        volume_tag, size_val
                                    ));
                                }
                                _ => {
                                    // Fallback: use entity name as identifier
                                    self.line(&format!(
                                        "MeshSize{{ Surface{{{}}}; }} = {};",
                                        entity_str, size_val
                                    ));
                                }
                            }
                        }
                        "order" => {
                            let order_val = self.emit_expr_str(&kv.value)?;
                            // Per-surface mesh order: set element order for this surface's mesh
                            // Gmsh applies ElementOrder globally, so we set it before meshing
                            // and note the target surface for documentation
                            self.line(&format!(
                                "// per-surface order {} for {}",
                                order_val, entity_str
                            ));
                            match &topo_query {
                                Ok(crate::topology::TopologyQuery::Surface { tag }) => {
                                    self.line(&format!(
                                        "Mesh.ElementOrder = {};  // Surface {{{}}}",
                                        order_val, tag
                                    ));
                                }
                                Ok(crate::topology::TopologyQuery::AllFaces { volume_tag }) => {
                                    self.line(&format!(
                                        "Mesh.ElementOrder = {};  // Boundary of Volume {{{}}}",
                                        order_val, volume_tag
                                    ));
                                }
                                _ => {
                                    self.line(&format!(
                                        "Mesh.ElementOrder = {};  // {}",
                                        order_val, entity_str
                                    ));
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
            RefineKind::In { region } => {
                let bounds = self.box_region_bounds(region)?;
                self.line("// refine in translated Box region");

                let box_field_id = self.tags.alloc_anonymous(GmshDim::Point).tag;
                self.line(&format!("Field[{}] = Box;", box_field_id));
                self.line(&format!("Field[{}].XMin = {};", box_field_id, bounds[0]));
                self.line(&format!("Field[{}].XMax = {};", box_field_id, bounds[1]));
                self.line(&format!("Field[{}].YMin = {};", box_field_id, bounds[2]));
                self.line(&format!("Field[{}].YMax = {};", box_field_id, bounds[3]));
                self.line(&format!("Field[{}].ZMin = {};", box_field_id, bounds[4]));
                self.line(&format!("Field[{}].ZMax = {};", box_field_id, bounds[5]));

                for kv in &refine.body {
                    match kv.key.name.as_str() {
                        "size" => {
                            let size_val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Field[{}].VIn = {};", box_field_id, size_val));
                        }
                        "size_out" => {
                            let size_val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Field[{}].VOut = {};", box_field_id, size_val));
                        }
                        _ => {}
                    }
                }
                self.refine_field_ids.push(box_field_id);
            }
            RefineKind::Field { args } => {
                self.line("// refine field");

                // Distance field
                let dist_id = self.tags.alloc_anonymous(GmshDim::Point).tag;
                self.line(&format!("Field[{}] = Distance;", dist_id));

                for kv in args {
                    if kv.key.name == "distance_from" {
                        let entity_str = self.emit_expr_str(&kv.value)?;
                        self.line(&format!(
                            "Field[{}].SurfacesList = {{{}}};",
                            dist_id, entity_str
                        ));
                    }
                }

                // Threshold field
                let thresh_id = self.tags.alloc_anonymous(GmshDim::Point).tag;
                self.line(&format!("Field[{}] = Threshold;", thresh_id));
                self.line(&format!("Field[{}].InField = {};", thresh_id, dist_id));

                for kv in &refine.body {
                    match kv.key.name.as_str() {
                        "size" | "size_min" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Field[{}].SizeMin = {};", thresh_id, val));
                        }
                        "size_max" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Field[{}].SizeMax = {};", thresh_id, val));
                        }
                        "distance_min" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Field[{}].DistMin = {};", thresh_id, val));
                        }
                        "distance_max" => {
                            let val = self.emit_expr_str(&kv.value)?;
                            self.line(&format!("Field[{}].DistMax = {};", thresh_id, val));
                        }
                        _ => {}
                    }
                }
                self.refine_field_ids.push(thresh_id);
            }
        }
        Ok(())
    }

    fn emit_mesh_metric(&mut self, metric: &MeshMetricStatement) -> Result<(), CompileError> {
        let metric_id = self.tags.alloc_anonymous(GmshDim::Point).tag;
        match &metric.metric {
            MeshMetricKind::Isotropic { size } => {
                let size = self.emit_expr_str(size)?;
                self.line(&format!("Field[{}] = MathEval;", metric_id));
                self.line(&format!("Field[{}].F = \"{}\";", metric_id, size));
            }
            MeshMetricKind::Anisotropic { components } => {
                if components.len() != 6 {
                    return Err(CompileError::TypeError {
                        message: "anisotropic metric requires six components".into(),
                    });
                }
                self.line(&format!("Field[{}] = MathEvalAniso;", metric_id));
                for (name, value) in ["M11", "M22", "M33", "M12", "M13", "M23"]
                    .into_iter()
                    .zip(components)
                {
                    let value = self.emit_expr_str(value)?;
                    self.line(&format!("Field[{}].{} = \"{}\";", metric_id, name, value));
                }
            }
            MeshMetricKind::Hessian { .. } => {
                return Err(CompileError::Unsupported(
                    "Hessian metric evaluation requires an in-process field provider".into(),
                ));
            }
        }

        let restrict_id = self.tags.alloc_anonymous(GmshDim::Point).tag;
        self.line(&format!("Field[{}] = Restrict;", restrict_id));
        self.line(&format!("Field[{}].InField = {};", restrict_id, metric_id));
        if let Some(path) = entity_path(&metric.target) {
            if path.ends_with(".faces") {
                let owner = path.trim_end_matches(".faces");
                let reference = self.entity_geo_ref(owner);
                self.line(&format!(
                    "Field[{}].SurfacesList = {{Boundary{{ Volume{{{}}}; }}}};",
                    restrict_id, reference
                ));
            } else {
                let reference = self.resolve_entity_tag(&metric.target)?;
                let dimension = self
                    .expr_gmsh_dim(&metric.target)
                    .unwrap_or(GmshDim::Volume);
                self.line(&format!(
                    "Field[{}].{}sList = {{{}}};",
                    restrict_id,
                    dimension.keyword(),
                    reference
                ));
            }
        } else {
            return Err(CompileError::TypeError {
                message: "mesh metric target must be a named entity".into(),
            });
        }
        self.refine_field_ids.push(restrict_id);
        Ok(())
    }

    fn emit_boundary_layer(&mut self, bl: &BoundaryLayerStatement) -> Result<(), CompileError> {
        let target_str = self.emit_expr_str(&bl.target)?;
        self.line(&format!("// boundary layer on {}", target_str));

        let field_id = self.tags.alloc_anonymous(GmshDim::Point).tag;
        self.line(&format!("Field[{}] = BoundaryLayer;", field_id));

        // Attach target entity surfaces
        let topo_query = self.resolve_expr_to_topology(&bl.target);
        match &topo_query {
            Ok(crate::topology::TopologyQuery::Surface { tag }) => {
                self.line(&format!("Field[{}].SurfacesList = {{{}}};", field_id, tag));
            }
            Ok(crate::topology::TopologyQuery::AllFaces { volume_tag }) => {
                self.line(&format!(
                    "Field[{}].SurfacesList = {{Boundary{{ Volume{{{}}}; }}}};",
                    field_id, volume_tag
                ));
            }
            _ => {
                self.line(&format!(
                    "Field[{}].SurfacesList = {{{}}}; // fallback",
                    field_id, target_str
                ));
            }
        }

        for kv in &bl.body {
            match kv.key.name.as_str() {
                "first_height" => {
                    let val = self.emit_expr_str(&kv.value)?;
                    self.line(&format!("Field[{}].Size = {};", field_id, val));
                }
                "thickness" | "total_thickness" => {
                    let val = self.emit_expr_str(&kv.value)?;
                    self.line(&format!("Field[{}].Thickness = {};", field_id, val));
                }
                "layers" => {
                    let val = self.emit_expr_str(&kv.value)?;
                    self.line(&format!("Field[{}].NbLayers = {};", field_id, val));
                }
                "growth_rate" => {
                    let val = self.emit_expr_str(&kv.value)?;
                    self.line(&format!("Field[{}].Ratio = {};", field_id, val));
                }
                _ => {}
            }
        }

        // BoundaryLayer fields must also be in the background
        self.refine_field_ids.push(field_id);

        Ok(())
    }

    fn emit_canonical_boundary_layer(
        &mut self,
        bl: &IrBoundaryLayer,
        ir: &CanonicalGeometryIr,
    ) -> Result<(), CompileError> {
        let nodes = ir
            .entities
            .iter()
            .map(|node| (node.id, node))
            .collect::<HashMap<_, _>>();
        if bl.target.is_empty() {
            return Err(CompileError::Internal(
                "canonical boundary-layer control has no target entities".into(),
            ));
        }
        let mut surface_refs = Vec::new();
        let mut volume_refs = Vec::new();
        for entity in &bl.target {
            let node = nodes.get(entity).ok_or_else(|| {
                CompileError::Internal(format!(
                    "canonical boundary-layer control references unknown entity {:?}",
                    entity
                ))
            })?;
            let reference = self.entity_geo_ref(&node.name);
            match node.id.dimension {
                EntityDimension::Surface => surface_refs.push(reference),
                EntityDimension::Volume => volume_refs.push(reference),
                other => {
                    return Err(CompileError::Unsupported(format!(
                        "canonical boundary-layer control does not support {:?} targets",
                        other
                    )));
                }
            }
        }
        if !surface_refs.is_empty() && !volume_refs.is_empty() {
            return Err(CompileError::Unsupported(
                "canonical boundary-layer targets cannot mix surfaces and volumes in GEO".into(),
            ));
        }

        self.line("// canonical boundary layer");
        let field_id = self.tags.alloc_anonymous(GmshDim::Point).tag;
        self.line(&format!("Field[{}] = BoundaryLayer;", field_id));
        if !surface_refs.is_empty() {
            self.line(&format!(
                "Field[{}].SurfacesList = {{{}}};",
                field_id,
                surface_refs.join(", ")
            ));
        } else {
            self.line(&format!(
                "Field[{}].SurfacesList = {{Boundary{{ Volume{{{}}}; }}}};",
                field_id,
                volume_refs.join(", ")
            ));
        }
        self.line(&format!(
            "Field[{}].Size = {};",
            field_id,
            format_float(bl.first_layer_height)
        ));
        self.line(&format!(
            "Field[{}].Thickness = {};",
            field_id,
            format_float(bl.total_thickness)
        ));
        if let Some(layers) = bl.number_of_layers {
            self.line(&format!("Field[{}].NbLayers = {};", field_id, layers));
        }
        self.line(&format!(
            "Field[{}].Ratio = {};",
            field_id,
            format_float(bl.growth_ratio)
        ));
        self.refine_field_ids.push(field_id);
        Ok(())
    }

    fn emit_transfinite(&mut self, t: &TransfiniteStatement) -> Result<(), CompileError> {
        match &t.kind {
            TransfiniteKind::Curve { entity, settings } => {
                let entity_str = self.emit_expr_str(entity)?;
                let mut nodes = "10".to_string();
                let mut mesh_type = String::new();
                let mut coef = String::new();
                for kv in settings {
                    match kv.key.name.as_str() {
                        "nodes" => nodes = self.emit_expr_str(&kv.value)?,
                        "type" => {
                            if let Expr::Literal(LiteralExpr {
                                value: Literal::String(s),
                                ..
                            }) = &kv.value
                            {
                                mesh_type = s.clone();
                            }
                        }
                        "coef" => coef = self.emit_expr_str(&kv.value)?,
                        _ => {}
                    }
                }
                let mut cmd = format!("Transfinite Curve {{{}}} = {}", entity_str, nodes);
                if !mesh_type.is_empty() {
                    cmd.push_str(&format!(" Using {}", mesh_type));
                    if !coef.is_empty() {
                        cmd.push_str(&format!(" {}", coef));
                    }
                }
                cmd.push(';');
                self.line(&cmd);
            }
            TransfiniteKind::Surface { entity, settings } => {
                let entity_str = self.emit_expr_str(entity)?;
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
                                    .map(|e| self.emit_expr_str(e))
                                    .collect::<Result<_, _>>()?;
                                corners = tags.join(", ");
                            }
                        }
                        _ => {}
                    }
                }
                let mut cmd = format!("Transfinite Surface {{{}}}", entity_str);
                if !corners.is_empty() {
                    cmd.push_str(&format!(" = {{{}}}", corners));
                }
                if !arrangement.is_empty() {
                    cmd.push_str(&format!(" {}", arrangement));
                }
                cmd.push(';');
                self.line(&cmd);
            }
            TransfiniteKind::Volume { entity, settings } => {
                let entity_str = self.emit_expr_str(entity)?;
                let mut corners = String::new();
                for kv in settings {
                    if kv.key.name == "corners" {
                        if let Expr::Array(arr) = &kv.value {
                            let tags: Vec<String> = arr
                                .elements
                                .iter()
                                .map(|e| self.emit_expr_str(e))
                                .collect::<Result<_, _>>()?;
                            corners = tags.join(", ");
                        }
                    }
                }
                let mut cmd = format!("Transfinite Volume {{{}}}", entity_str);
                if !corners.is_empty() {
                    cmd.push_str(&format!(" = {{{}}}", corners));
                }
                cmd.push(';');
                self.line(&cmd);
            }
        }
        Ok(())
    }

    fn emit_embed(&mut self, e: &EmbedStatement) -> Result<(), CompileError> {
        let source_str = self.emit_expr_str(&e.source)?;
        let target_str = self.emit_expr_str(&e.target)?;
        let embed_dim = match e.embed_dim {
            EmbedDim::Point => 0,
            EmbedDim::Curve => 1,
            EmbedDim::Surface => 2,
            EmbedDim::Volume => 3,
        };
        let embed_type = match e.embed_dim {
            EmbedDim::Point => "Point",
            EmbedDim::Curve => "Curve",
            EmbedDim::Surface => "Surface",
            EmbedDim::Volume => "Volume",
        };
        let target_type = match e.target_dim {
            EmbedDim::Point => "Point",
            EmbedDim::Curve => "Curve",
            EmbedDim::Surface => "Surface",
            EmbedDim::Volume => "Volume",
        };
        // .geo uses: Point{tag} In Surface{tag};
        // or the general: <dim> {tag} In <dim> {tag};
        let _ = embed_dim; // used only for Python
        self.line(&format!(
            "{}{{{}}} In {}{{{}}};",
            embed_type, source_str, target_type, target_str
        ));
        Ok(())
    }

    fn emit_periodic(&mut self, p: &PeriodicStatement) -> Result<(), CompileError> {
        let source_str = self.emit_expr_str(&p.source)?;
        let target_str = self.emit_expr_str(&p.target)?;
        let transform_str = self.emit_expr_str(&p.transform)?;

        let dim_str = match p.dim {
            PeriodicDim::Curve => "Curve",
            PeriodicDim::Surface => "Surface",
        };
        self.line(&format!(
            "Periodic {} {{{}}} = {{{}}} Translate {{{}}};",
            dim_str, source_str, target_str, transform_str
        ));
        Ok(())
    }

    fn emit_compound(&mut self, c: &CompoundStatement) -> Result<(), CompileError> {
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
                self.resolve_entity_tag(entity)
            })
            .collect::<Result<_, _>>()?;
        self.line(&format!(
            "Compound {} {{{}}};",
            dim.keyword(),
            tags.join(", ")
        ));
        Ok(())
    }

    fn emit_physics(&mut self, physics: &PhysicsBlock) -> Result<(), CompileError> {
        self.line("// physics groups");
        for stmt in &physics.statements {
            match stmt {
                PhysicsStatement::Group(g) => {
                    // Resolve the expression to a topology query
                    let topo_query = self.resolve_expr_to_topology(&g.value);
                    match topo_query {
                        Ok(query) => {
                            let geo_expr = query.to_geo();
                            self.line(&format!(
                                "// group {}: {} = {}",
                                g.name.name, g.bc_type.name, geo_expr
                            ));
                            // Emit as a physical group using Boundary queries
                            // In .geo, we need the actual entity tags, which requires
                            // Gmsh to be running. Emit as a post-synchronize block.
                            // Resolve the base entity name to its .geo variable reference.
                            // For face queries like "cheese.swiss.faces", the volume
                            // entity is "cheese.swiss" (strip the trailing .faces/.edges etc.)
                            let parts = self.collect_dot_path(&g.value).unwrap_or_default();
                            let base_entity_name = {
                                // Find the longest prefix that is a registered entity
                                let mut name = parts.join(".");
                                for i in (1..=parts.len()).rev() {
                                    let candidate = parts[..i].join(".");
                                    if self.tags.lookup(&candidate).is_some()
                                        || self.tags.geo_var(&candidate).is_some()
                                    {
                                        name = candidate;
                                        break;
                                    }
                                }
                                name
                            };
                            let eref = self.entity_geo_ref(&base_entity_name);

                            match &query {
                                crate::topology::TopologyQuery::AllFaces { .. } => {
                                    self.line(&format!(
                                        "Physical Surface(\"{}\") = Boundary{{ Volume{{{}}}; }};",
                                        g.name.name, eref
                                    ));
                                    self.emitted_physical_groups.insert(g.name.name.clone());
                                }
                                crate::topology::TopologyQuery::Volume { .. } => {
                                    self.line(&format!(
                                        "Physical Volume(\"{}\") = {{{}}};",
                                        g.name.name, eref
                                    ));
                                    self.emitted_physical_groups.insert(g.name.name.clone());
                                }
                                crate::topology::TopologyQuery::Surface { .. } => {
                                    self.line(&format!(
                                        "Physical Surface(\"{}\") = {{{}}};",
                                        g.name.name, eref
                                    ));
                                    self.emitted_physical_groups.insert(g.name.name.clone());
                                }
                                crate::topology::TopologyQuery::BoxFaceDeferred {
                                    face, ..
                                } => {
                                    let (axis, maximum) = match face {
                                        BoxFace::Left => (0, false),
                                        BoxFace::Right => (0, true),
                                        BoxFace::Back => (1, false),
                                        BoxFace::Front => (1, true),
                                        BoxFace::Bottom => (2, false),
                                        BoxFace::Top => (2, true),
                                    };
                                    self.emit_axis_face_group(&g.name.name, &eref, axis, maximum);
                                }
                                crate::topology::TopologyQuery::CylinderFaceDeferred {
                                    face,
                                    ..
                                } => match face {
                                    CylinderFace::Bottom => {
                                        self.emit_axis_face_group(&g.name.name, &eref, 2, false)
                                    }
                                    CylinderFace::Top => {
                                        self.emit_axis_face_group(&g.name.name, &eref, 2, true)
                                    }
                                    CylinderFace::Lateral => {
                                        return Err(CompileError::Unsupported(format!(
                                            "GEO lateral-face selector for physical group '{}'",
                                            g.name.name
                                        )));
                                    }
                                },
                                _ => {
                                    return Err(CompileError::Unsupported(format!(
                                        "GEO topology query for physical group '{}': {}",
                                        g.name.name, geo_expr
                                    )));
                                }
                            }
                        }
                        Err(msg) => {
                            if msg.contains("face set operations require Python backend") {
                                return Err(CompileError::Unsupported(format!(
                                    "physics group '{}': {}",
                                    g.name.name, msg
                                )));
                            }
                            return Err(CompileError::TypeError {
                                message: format!(
                                    "physics group '{}' has invalid topology: {}",
                                    g.name.name, msg
                                ),
                            });
                        }
                    }
                }
                PhysicsStatement::Let(l) => {
                    self.line(&format!("// let {} = ...", l.name.name));
                }
            }
        }
        self.line("");
        Ok(())
    }

    fn emit_axis_face_group(
        &mut self,
        group_name: &str,
        volume_ref: &str,
        axis: usize,
        maximum: bool,
    ) {
        let id = self.temp_counter;
        self.temp_counter += 1;
        let bbox = format!("__ml_face_bbox_{id}");
        let tolerance = format!("__ml_face_tol_{id}");
        let surfaces = format!("__ml_face_surfaces_{id}");
        self.line(&format!(
            "{}() = BoundingBox Volume {{{}}};",
            bbox, volume_ref
        ));
        self.line(&format!(
            "{} = 1e-6 * (({}(3) - {}(0)) + ({}(4) - {}(1)) + ({}(5) - {}(2)) + 1);",
            tolerance, bbox, bbox, bbox, bbox, bbox, bbox
        ));

        let mut lower = Vec::with_capacity(3);
        let mut upper = Vec::with_capacity(3);
        for coordinate in 0..3 {
            if coordinate == axis {
                let index = if maximum { coordinate + 3 } else { coordinate };
                lower.push(format!("{}({}) - {}", bbox, index, tolerance));
                upper.push(format!("{}({}) + {}", bbox, index, tolerance));
            } else {
                lower.push(format!("{}({}) - {}", bbox, coordinate, tolerance));
                upper.push(format!("{}({}) + {}", bbox, coordinate + 3, tolerance));
            }
        }
        self.line(&format!(
            "{}() = Surface In BoundingBox {{{}, {}, {}, {}, {}, {}}};",
            surfaces, lower[0], lower[1], lower[2], upper[0], upper[1], upper[2]
        ));
        self.line(&format!(
            "Physical Surface(\"{}\") = {{{}()}};",
            group_name, surfaces
        ));
        self.emitted_physical_groups.insert(group_name.to_string());
    }

    /// Resolve an expression to a TopologyQuery by walking dot-access chains.
    fn resolve_expr_to_topology(
        &self,
        expr: &Expr,
    ) -> Result<crate::topology::TopologyQuery, String> {
        let parts = self.collect_dot_path(expr)?;
        let parts_ref: Vec<&str> = parts.iter().map(|s| s.as_str()).collect();
        self.topo.resolve_path(&parts_ref)
    }

    /// Collect a dot-access chain into a list of identifiers.
    fn collect_dot_path(&self, expr: &Expr) -> Result<Vec<String>, String> {
        match expr {
            Expr::Ident(id) => Ok(vec![id.name.clone()]),
            Expr::FieldAccess(fa) => {
                let mut parts = self.collect_dot_path(&fa.object)?;
                parts.push(fa.field.name.clone());
                Ok(parts)
            }
            Expr::Binary(b) if matches!(b.op, BinaryOp::Sub | BinaryOp::CsgDifference) => {
                // face set difference: expr - expr
                // Return an error so the caller handles it as a set operation
                Err("face set operations require Python backend".into())
            }
            _ => Err("expected dot-access path for topology reference".into()),
        }
    }

    fn emit_export(&mut self, export: &ExportBlock) -> Result<(), CompileError> {
        self.line("// export");
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
                    } else {
                        format = self.emit_expr_str(&kv.value)?;
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
            self.line(&format!("Mesh.ElementOrder = {};", order));
            self.line("Mesh.HighOrderOptimize = 1;");
        }

        if binary {
            self.line("Mesh.Binary = 1;");
        }

        if save_all {
            self.line("Mesh.SaveAll = 1;");
        }

        // Set format version
        match format.as_str() {
            "msh2" => self.line("Mesh.MshFileVersion = 2.2;"),
            "msh4" => self.line("Mesh.MshFileVersion = 4.1;"),
            "unv" => self.line("Mesh.Format = 2;"),
            "med" => self.line("Mesh.Format = 33;"),
            "cgns" => self.line("Mesh.Format = 32;"),
            "inp" | "abaqus" => self.line("Mesh.Format = 39;"),
            "bdf" | "nastran" => self.line("Mesh.Format = 31;"),
            "su2" => self.line("Mesh.Format = 42;"),
            "stl" | "vtk" => {} // format determined by extension
            _ => {}
        }

        // Do NOT emit Mesh/Save commands — they cause the Gmsh GUI to
        // mesh automatically on file load.  Users mesh via `gmsh -3` or
        // the GUI interactively.  The target filename is preserved as a
        // comment for documentation.
        if order > 1 {
            self.line("// OptimizeMesh \"HighOrder\";");
        }
        self.line(&format!("// Output: {}", filename));

        Ok(())
    }

    /// Resolve a Loft section. Sections built from `Polygon` are emitted as
    /// `Plane Surface` + a backing `Curve Loop` (registered as `{name}__cl`).
    /// Gmsh's OpenCASCADE `ThruSections` requires WIRE/curve-loop tags — passing
    /// the surface tag yields "Unknown OpenCASCADE wire or curve loop". When
    /// such a curve-loop tag exists, prefer it; otherwise fall back to the
    /// entity tag (Disk/Circle sections are surfaces whose boundary wire OCC
    /// extracts implicitly).
    fn resolve_loft_section_tag(&self, expr: &Expr) -> Result<String, CompileError> {
        if let Expr::Ident(id) = expr {
            let qname = if let Some(ref geo) = self.current_geometry {
                format!("{}.{}", geo, id.name)
            } else {
                id.name.clone()
            };
            let cl_qname = format!("{}__cl", qname);
            if let Some(gv) = self.tags.geo_var(&cl_qname) {
                return Ok(gv.to_string());
            }
            if let Some(entity) = self.tags.lookup(&cl_qname) {
                return Ok(entity.tag.to_string());
            }
        }
        self.resolve_entity_tag(expr)
    }

    /// Resolve an expression to a Gmsh entity reference (variable name or integer).
    /// For identifiers, looks up the .geo variable name in the tag allocator.
    /// Falls back to emit_expr_str for literal integers / expressions.
    fn resolve_entity_tag(&self, expr: &Expr) -> Result<String, CompileError> {
        if matches!(expr, Expr::Ident(_) | Expr::FieldAccess(_)) {
            let qname = self.expr_entity_name(expr)?;
            if let Some(gv) = self.tags.geo_var(&qname) {
                return Ok(gv.to_string());
            }
            if let Some(entity) = self.tags.lookup(&qname) {
                return Ok(entity.tag.to_string());
            }
        }
        self.emit_expr_str(expr)
    }

    /// Alias for resolve_entity_tag.
    fn resolve_point_tag(&self, expr: &Expr) -> Result<String, CompileError> {
        self.resolve_entity_tag(expr)
    }

    /// Get the .geo variable name for a named entity, falling back to the integer tag.
    fn entity_geo_ref(&self, name: &str) -> String {
        if let Some(gv) = self.tags.geo_var(name) {
            gv.to_string()
        } else if let Some(entity) = self.tags.lookup(name) {
            entity.tag.to_string()
        } else {
            name.replace('.', "__")
        }
    }

    // ── Expression stringification ───────────────────────────────

    fn emit_expr_str(&self, expr: &Expr) -> Result<String, CompileError> {
        match expr {
            Expr::Literal(LiteralExpr {
                value: Literal::Int(v),
                ..
            }) => Ok(v.to_string()),
            Expr::Literal(LiteralExpr {
                value: Literal::Float(v),
                ..
            }) => Ok(format_float(*v)),
            Expr::Literal(LiteralExpr {
                value: Literal::Bool(b),
                ..
            }) => Ok(if *b { "1" } else { "0" }.to_string()),
            Expr::Literal(LiteralExpr {
                value: Literal::String(s),
                ..
            }) => Ok(format!("\"{}\"", escape_geo_string(s))),
            Expr::Ident(id) => {
                // Map built-in constants to Gmsh equivalents
                match id.name.as_str() {
                    "PI" | "pi" => Ok("Pi".to_string()),
                    _ => Ok(id.name.clone()),
                }
            }
            Expr::Binary(b) => {
                let l = self.emit_expr_str(&b.left)?;
                let r = self.emit_expr_str(&b.right)?;
                let op = match b.op {
                    BinaryOp::Add => "+",
                    BinaryOp::Sub | BinaryOp::CsgDifference => "-",
                    BinaryOp::Mul => "*",
                    BinaryOp::Div => "/",
                    BinaryOp::Mod => "%",
                    BinaryOp::Pow => "^",
                    _ => "?",
                };
                Ok(format!("({} {} {})", l, op, r))
            }
            Expr::Unary(u) => {
                let v = self.emit_expr_str(&u.operand)?;
                match u.op {
                    UnaryOp::Neg => Ok(format!("(-{})", v)),
                    UnaryOp::Not => Ok(format!("(!{})", v)),
                }
            }
            Expr::FieldAccess(fa) => {
                let obj = self.emit_expr_str(&fa.object)?;
                Ok(format!("{}.{}", obj, fa.field.name))
            }
            Expr::Call(c) => {
                // User-defined functions are inlined: substitute args into the
                // body and re-emit. This is the only path that survives into
                // scalar contexts like Polygon point coordinates — gmsh's .geo
                // grammar has no user functions, so an unresolved `foo(x)`
                // would be emitted verbatim and fail at parse time.
                if self.functions.contains_key(&c.function.name) {
                    let inlined = self.try_inline_call(c)?;
                    return self.emit_expr_str(&inlined);
                }
                let args: Vec<String> = c
                    .args
                    .iter()
                    .map(|a| self.emit_expr_str(&a.value))
                    .collect::<Result<_, _>>()?;
                // Map built-in math functions to Gmsh equivalents (capitalized)
                let geo_name = match c.function.name.as_str() {
                    "cos" => "Cos",
                    "sin" => "Sin",
                    "tan" => "Tan",
                    "acos" => "Acos",
                    "asin" => "Asin",
                    "atan" => "Atan",
                    "atan2" => "Atan2",
                    "sqrt" => "Sqrt",
                    "exp" => "Exp",
                    "log" => "Log",
                    "log10" => "Log10",
                    "abs" => "Fabs",
                    "floor" => "Floor",
                    "ceil" => "Ceil",
                    "fmod" => "Fmod",
                    "hypot" => "Hypot",
                    other => other,
                };
                Ok(format!("{}({})", geo_name, args.join(", ")))
            }
            Expr::Array(array) => {
                let elements = array
                    .elements
                    .iter()
                    .map(|element| self.emit_expr_str(element))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(format!("{{{}}}", elements.join(", ")))
            }
            Expr::Index(index) => {
                let object = self.emit_expr_str(&index.object)?;
                let index_value = self.emit_expr_str(&index.index)?;
                Ok(format!("{object}[{index_value}]"))
            }
            Expr::Paren(inner) => {
                let v = self.emit_expr_str(&inner.value)?;
                Ok(format!("({})", v))
            }
            Expr::StructInit(si) => {
                // Struct init in .geo: emit as a series of variable assignments
                // Return the prefix used for the fields
                Ok(format!("/* struct {} */", si.type_name.name))
            }
            _ => Err(CompileError::Unsupported(format!(
                "GEO backend cannot emit expression {:?} in a scalar context",
                expr
            ))),
        }
    }

    fn expr_entity_name(&self, expr: &Expr) -> Result<String, CompileError> {
        match expr {
            Expr::Ident(id) => {
                if let Some(ref geo) = self.current_geometry {
                    Ok(format!("{}.{}", geo, id.name))
                } else {
                    Ok(id.name.clone())
                }
            }
            Expr::FieldAccess(fa) => {
                let base = self.expr_entity_name(&fa.object)?;
                Ok(format!("{}.{}", base, fa.field.name))
            }
            _ => Err(CompileError::Internal("cannot extract entity name".into())),
        }
    }

    fn eval_positional_args(
        &self,
        args: &[Argument],
        names: &[&str],
    ) -> Result<Vec<String>, CompileError> {
        if args.len() != names.len() {
            return Err(CompileError::TypeError {
                message: format!(
                    "expected {} positional arguments ({}) but got {}",
                    names.len(),
                    names.join(", "),
                    args.len()
                ),
            });
        }
        if args.iter().any(|arg| arg.name.is_some()) {
            return Err(CompileError::TypeError {
                message: format!("expected positional arguments ({})", names.join(", ")),
            });
        }

        args.iter().map(|a| self.emit_expr_str(&a.value)).collect()
    }

    fn box_region_bounds(&self, expr: &Expr) -> Result<[String; 6], CompileError> {
        match expr {
            Expr::Constructor(c) if c.type_name.name == "Box" => {
                let sizes = self.eval_positional_args(&c.args, &["x", "y", "z"])?;
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
                let mut bounds = self.box_region_bounds(&pipe.value)?;
                let offsets = self.eval_positional_args(&pipe.args, &["dx", "dy", "dz"])?;
                for axis in 0..3 {
                    let offset = &offsets[axis];
                    bounds[axis * 2] = format!("({} + {})", bounds[axis * 2], offset);
                    bounds[axis * 2 + 1] = format!("({} + {})", bounds[axis * 2 + 1], offset);
                }
                Ok(bounds)
            }
            _ => Err(CompileError::Unsupported(
                "GEO refinement regions must be Box optionally followed by translate".into(),
            )),
        }
    }

    fn eval_named_arg(&self, args: &[Argument], name: &str) -> Result<String, CompileError> {
        for arg in args {
            if let Some(ref n) = arg.name {
                if n.name == name {
                    return self.emit_expr_str(&arg.value);
                }
            }
        }
        // Try positional fallback
        Err(CompileError::Internal(format!(
            "missing required argument '{}'",
            name
        )))
    }

    fn angle_arg_degrees_to_radians(&self, args: &[Argument]) -> Result<String, CompileError> {
        let angle = self.eval_named_arg(args, "angle")?;
        Ok(format!("({} * Pi / 180.0)", angle))
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
            .map(|element| self.emit_expr_str(element))
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

    fn eval_named_arg_str(&self, args: &[Argument], name: &str) -> Result<String, CompileError> {
        for arg in args {
            if let Some(ref n) = arg.name {
                if n.name == name {
                    if let Expr::Ident(id) = &arg.value {
                        return Ok(id.name.clone());
                    }
                    return self.emit_expr_str(&arg.value);
                }
            }
        }
        Err(CompileError::Internal(format!(
            "missing required argument '{}'",
            name
        )))
    }

    fn mesh_algo_2d(&self, expr: &Expr) -> Result<&'static str, CompileError> {
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

    fn mesh_algo_3d(&self, expr: &Expr) -> Result<&'static str, CompileError> {
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

    // ── Output helpers ───────────────────────────────────────────

    fn line(&mut self, text: &str) {
        for _ in 0..self.indent {
            self.buf.push_str("    ");
        }
        self.buf.push_str(text);
        self.buf.push('\n');
    }
}

fn collect_preserved_entities(file: &SourceFile) -> HashSet<String> {
    let mut preserved = HashSet::new();
    for declaration in &file.declarations {
        match declaration {
            Declaration::Physics(physics) => {
                for statement in &physics.statements {
                    if let PhysicsStatement::Group(group) = statement {
                        preserve_entity_expr(&group.value, &mut preserved);
                    }
                }
            }
            Declaration::PhysicalRegion(region) => {
                preserve_entity_expr(&region.entities, &mut preserved)
            }
            Declaration::Boundary(boundary) => {
                preserve_entity_expr(&boundary.entities, &mut preserved)
            }
            Declaration::Interface(interface) => {
                preserve_entity_expr(&interface.primary, &mut preserved);
                preserve_entity_expr(&interface.secondary, &mut preserved);
            }
            Declaration::MaterialRegion(region) => {
                preserve_entity_expr(&region.region, &mut preserved)
            }
            Declaration::Mesh(mesh) => {
                for statement in &mesh.statements {
                    match statement {
                        MeshStatement::Refine(refine) => match &refine.kind {
                            RefineKind::Near { entity, .. } | RefineKind::On { entity } => {
                                preserve_entity_expr(entity, &mut preserved)
                            }
                            RefineKind::In { region } => {
                                preserve_entity_expr(region, &mut preserved)
                            }
                            RefineKind::Field { .. } => {}
                        },
                        MeshStatement::BoundaryLayer(layer) => {
                            preserve_entity_expr(&layer.target, &mut preserved)
                        }
                        MeshStatement::Transfinite(transfinite) => {
                            let entity = match &transfinite.kind {
                                TransfiniteKind::Curve { entity, .. }
                                | TransfiniteKind::Surface { entity, .. }
                                | TransfiniteKind::Volume { entity, .. } => entity,
                            };
                            preserve_entity_expr(entity, &mut preserved);
                        }
                        MeshStatement::Embed(embed) => {
                            preserve_entity_expr(&embed.source, &mut preserved);
                            preserve_entity_expr(&embed.target, &mut preserved);
                        }
                        MeshStatement::Periodic(periodic) => {
                            preserve_entity_expr(&periodic.source, &mut preserved);
                            preserve_entity_expr(&periodic.target, &mut preserved);
                        }
                        MeshStatement::Compound(compound) => {
                            for entity in &compound.entities {
                                preserve_entity_expr(entity, &mut preserved);
                            }
                        }
                        MeshStatement::Metric(metric) => {
                            preserve_entity_expr(&metric.target, &mut preserved)
                        }
                        MeshStatement::KeyValue(_) => {}
                    }
                }
            }
            _ => {}
        }
    }
    preserved
}

fn preserve_entity_expr(expression: &Expr, preserved: &mut HashSet<String>) {
    if let Expr::Array(array) = expression {
        for element in &array.elements {
            preserve_entity_expr(element, preserved);
        }
        return;
    }
    let Some(path) = entity_path(expression) else {
        return;
    };
    let mut parts: Vec<&str> = path.split('.').collect();
    if let Some(selector) = parts.iter().position(|part| {
        matches!(
            *part,
            "faces"
                | "edges"
                | "vertices"
                | "top"
                | "bottom"
                | "front"
                | "back"
                | "left"
                | "right"
                | "lateral"
                | "outer"
                | "inner"
                | "surface"
        )
    }) {
        parts.truncate(selector);
    }
    if !parts.is_empty() {
        preserved.insert(parts.join("."));
    }
}

/// Check if two ops map to the same Gmsh boolean operation.
/// Sub and CsgDifference both → BooleanDifference.
fn same_csg_op(a: BinaryOp, b: BinaryOp) -> bool {
    fn normalize(op: BinaryOp) -> u8 {
        match op {
            BinaryOp::Sub | BinaryOp::CsgDifference => 0,
            BinaryOp::CsgUnion => 1,
            BinaryOp::CsgIntersect => 2,
            _ => 255,
        }
    }
    normalize(a) == normalize(b)
}

fn constructor_gmsh_dim(constructor: &ConstructorExpr) -> GmshDim {
    match constructor.type_name.name.as_str() {
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

fn format_float(v: f64) -> String {
    if v == v.floor() && v.abs() < 1e15 {
        format!("{:.1}", v)
    } else {
        format!("{}", v)
    }
}

fn escape_geo_string(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
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

/// Try to evaluate an expression as a compile-time integer.
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

/// Try to evaluate a boolean condition at compile time.
fn try_eval_const_bool(expr: &Expr) -> Option<bool> {
    match expr {
        Expr::Literal(LiteralExpr {
            value: Literal::Bool(b),
            ..
        }) => Some(*b),
        Expr::Binary(b) => {
            // Try integer comparison
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
                // Try boolean logic
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
        let emitted = GeoCodegen::new().emit_expr_str(&expression).unwrap();
        assert_eq!(emitted, "(Pi + (Sin(1.0) ^ 2))");
    }

    #[test]
    fn scalar_emission_escapes_strings_and_preserves_field_paths() {
        let codegen = GeoCodegen::new();
        let string = parse_value(r#""line\nvalue""#);
        assert_eq!(codegen.emit_expr_str(&string).unwrap(), r#""line\nvalue""#);
        let field = parse_value("material.density");
        assert_eq!(codegen.emit_expr_str(&field).unwrap(), "material.density");
    }

    #[test]
    fn expression_emission_covers_parentheses_arrays_and_indices() {
        let codegen = GeoCodegen::new();
        assert_eq!(
            codegen.emit_expr_str(&parse_value("(1 + 2)")).unwrap(),
            "((1 + 2))"
        );
        assert_eq!(
            codegen.emit_expr_str(&parse_value("[1, 2, 3]")).unwrap(),
            "{1, 2, 3}"
        );
        assert_eq!(
            codegen.emit_expr_str(&parse_value("[1, 2, 3][1]")).unwrap(),
            "{1, 2, 3}[1]"
        );
    }

    #[test]
    fn shape_and_pipe_emission_are_tested_at_the_codegen_boundary() {
        let Expr::Constructor(constructor) = parse_value("Box(1.0, 2.0, 3.0)") else {
            panic!("expected Box constructor");
        };
        let mut codegen = GeoCodegen::new();
        codegen
            .emit_shape_constructor("test.body", &constructor)
            .unwrap();
        assert!(codegen.output().contains("Box("));

        let Expr::Pipe(pipe) = parse_value("Box(1.0, 2.0, 3.0) |> translate(1.0, 0.0, 0.0)") else {
            panic!("expected transform pipe");
        };
        let mut codegen = GeoCodegen::new();
        codegen.emit_pipe_chain("test.body", &pipe).unwrap();
        assert!(codegen.output().contains("Translate {1.0, 0.0, 0.0}"));
    }

    #[test]
    fn loop_substitution_reaches_expression_emission() {
        let expression = parse_value("i + 1");
        let mut substitutions = HashMap::new();
        substitutions.insert("i".to_owned(), Expr::literal(Literal::Int(3), Span::DUMMY));
        let substituted = substitute_expr(&expression, &substitutions);
        assert_eq!(
            GeoCodegen::new().emit_expr_str(&substituted).unwrap(),
            "(3 + 1)"
        );
    }
}
