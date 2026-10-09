//! Name resolution: resolve identifiers to their declarations.
//!
//! This pass walks the AST and builds a scope chain, ensuring all references
//! are valid before codegen. It also collects parameter values for constant folding.

use crate::error::CompileError;
use meshlang_ast::*;
use std::collections::{HashMap, HashSet};

/// A resolved symbol in the scope.
#[derive(Debug, Clone)]
pub enum Symbol {
    /// A parameter or constant with a compile-time value.
    Param { name: String, value: Option<f64> },
    /// A geometry block.
    Geometry { name: String },
    /// A let-bound variable (geometry entity, intermediate value).
    Let {
        name: String,
        /// Dotted path prefix for topology access: "cabinet.shell"
        qualified_path: String,
    },
    /// A function.
    Function { name: String },
    /// A physical group.
    PhysicalGroup { name: String, bc_type: String },
}

/// Scope for name resolution.
#[derive(Debug, Default)]
pub struct Scope {
    symbols: HashMap<String, Symbol>,
    parent: Option<Box<Scope>>,
}

impl Scope {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_parent(parent: Scope) -> Self {
        Self {
            symbols: HashMap::new(),
            parent: Some(Box::new(parent)),
        }
    }

    pub fn define(&mut self, name: &str, symbol: Symbol) {
        self.symbols.insert(name.to_string(), symbol);
    }

    pub fn lookup(&self, name: &str) -> Option<&Symbol> {
        self.symbols
            .get(name)
            .or_else(|| self.parent.as_ref().and_then(|p| p.lookup(name)))
    }
}

fn define_unique(
    scope: &mut Scope,
    identifier: &Ident,
    symbol: Symbol,
) -> Result<(), CompileError> {
    if scope.symbols.contains_key(&identifier.name) {
        return Err(CompileError::DuplicateDefinition {
            name: identifier.name.clone(),
            span: identifier.span,
        });
    }
    scope.define(&identifier.name, symbol);
    Ok(())
}

fn require_unique<'a>(
    identifiers: impl IntoIterator<Item = &'a Ident>,
) -> Result<(), CompileError> {
    let mut names = HashSet::new();
    for identifier in identifiers {
        if !names.insert(identifier.name.as_str()) {
            return Err(CompileError::DuplicateDefinition {
                name: identifier.name.clone(),
                span: identifier.span,
            });
        }
    }
    Ok(())
}

/// Resolve all names in a source file.
pub fn resolve(file: &SourceFile) -> Result<Scope, CompileError> {
    let mut scope = Scope::new();

    // Built-in transforms
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
        scope.define(
            name,
            Symbol::Function {
                name: name.to_string(),
            },
        );
    }

    // Built-in axis constants
    for name in &["X", "Y", "Z", "XY", "XZ", "YZ"] {
        scope.define(
            name,
            Symbol::Param {
                name: name.to_string(),
                value: None,
            },
        );
    }

    let mut physical_group_names = HashSet::new();
    for decl in &file.declarations {
        match decl {
            Declaration::Param(p) => {
                let value = try_eval_const(&p.value);
                define_unique(
                    &mut scope,
                    &p.name,
                    Symbol::Param {
                        name: p.name.name.clone(),
                        value,
                    },
                )?;
            }
            Declaration::Const(c) => {
                let value = try_eval_const(&c.value);
                define_unique(
                    &mut scope,
                    &c.name,
                    Symbol::Param {
                        name: c.name.name.clone(),
                        value,
                    },
                )?;
            }
            Declaration::Geometry(g) => {
                require_unique(g.params.iter().map(|param| &param.name).chain(
                    g.body.statements.iter().filter_map(|statement| {
                        if let Statement::Let(binding) = statement {
                            Some(&binding.name)
                        } else {
                            None
                        }
                    }),
                ))?;
                define_unique(
                    &mut scope,
                    &g.name,
                    Symbol::Geometry {
                        name: g.name.name.clone(),
                    },
                )?;
                // Geometry parameters and local bindings intentionally do not
                // enter the returned module scope. Backends qualify geometry
                // entities as `<geometry>.<binding>` and the type checker owns
                // the lexical environment while visiting the body.
            }
            Declaration::Function(f) => {
                require_unique(f.params.iter().map(|param| &param.name))?;
                define_unique(
                    &mut scope,
                    &f.name,
                    Symbol::Function {
                        name: f.name.name.clone(),
                    },
                )?;
            }
            Declaration::Physics(p) => {
                require_unique(p.statements.iter().map(|statement| match statement {
                    PhysicsStatement::Group(group) => &group.name,
                    PhysicsStatement::Let(binding) => &binding.name,
                }))?;
                for stmt in &p.statements {
                    if let PhysicsStatement::Group(g) = stmt {
                        if !physical_group_names.insert(g.name.name.as_str()) {
                            return Err(CompileError::DuplicateDefinition {
                                name: g.name.name.clone(),
                                span: g.name.span,
                            });
                        }
                        scope.define(
                            &g.name.name,
                            Symbol::PhysicalGroup {
                                name: g.name.name.clone(),
                                bc_type: g.bc_type.name.clone(),
                            },
                        );
                    }
                }
            }
            Declaration::Struct(structure) => {
                require_unique(structure.fields.iter().map(|field| &field.name))?;
            }
            Declaration::DesignVariable(variable) => {
                define_unique(
                    &mut scope,
                    &variable.name,
                    Symbol::Param {
                        name: variable.name.name.clone(),
                        value: try_eval_const(&variable.initial),
                    },
                )?;
            }
            Declaration::Uncertainty(uncertainty) => {
                define_unique(
                    &mut scope,
                    &uncertainty.name,
                    Symbol::Param {
                        name: uncertainty.name.name.clone(),
                        value: None,
                    },
                )?;
            }
            Declaration::DesignRegion(region) => define_metadata_symbol(&mut scope, &region.name)?,
            Declaration::PhysicalRegion(region) => {
                define_metadata_symbol(&mut scope, &region.name)?
            }
            Declaration::Boundary(boundary) => define_metadata_symbol(&mut scope, &boundary.name)?,
            Declaration::Interface(interface) => {
                define_metadata_symbol(&mut scope, &interface.name)?
            }
            Declaration::CoordinateSystem(system) => {
                define_metadata_symbol(&mut scope, &system.name)?
            }
            Declaration::MaterialRegion(region) => {
                define_metadata_symbol(&mut scope, &region.name)?
            }
            Declaration::GeometryField(field) => define_metadata_symbol(&mut scope, &field.name)?,
            Declaration::Objective(objective) => {
                if let Some(name) = &objective.name {
                    define_metadata_symbol(&mut scope, name)?;
                }
            }
            Declaration::Constraint(constraint) => {
                if let Some(name) = &constraint.name {
                    define_metadata_symbol(&mut scope, name)?;
                }
            }
            _ => {}
        }
    }

    Ok(scope)
}

fn define_metadata_symbol(scope: &mut Scope, identifier: &Ident) -> Result<(), CompileError> {
    define_unique(
        scope,
        identifier,
        Symbol::Param {
            name: identifier.name.clone(),
            value: None,
        },
    )
}

/// Try to evaluate a constant expression to a float.
/// Returns None for anything non-trivial.
fn try_eval_const(expr: &Expr) -> Option<f64> {
    match expr {
        Expr::Literal(LiteralExpr {
            value: Literal::Float(v),
            ..
        }) => Some(*v),
        Expr::Literal(LiteralExpr {
            value: Literal::Int(v),
            ..
        }) => Some(*v as f64),
        Expr::Binary(b) => {
            let l = try_eval_const(&b.left)?;
            let r = try_eval_const(&b.right)?;
            match b.op {
                BinaryOp::Add => Some(l + r),
                BinaryOp::Sub => Some(l - r),
                BinaryOp::Mul => Some(l * r),
                BinaryOp::Div => {
                    if r == 0.0 {
                        None
                    } else {
                        Some(l / r)
                    }
                }
                BinaryOp::Pow => Some(l.powf(r)),
                _ => None,
            }
        }
        Expr::Unary(u) => {
            let v = try_eval_const(&u.operand)?;
            match u.op {
                UnaryOp::Neg => Some(-v),
                _ => None,
            }
        }
        Expr::Paren(inner) => try_eval_const(&inner.value),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn geometry_bindings_do_not_leak_into_module_scope() {
        let source = r#"
            geometry first {
                let shared = Box(1.0, 1.0, 1.0)
            }
            geometry second {
                let shared = Sphere(r: 0.5)
            }
        "#;
        let ast = meshlang_parser::parse(source, "scope.mlg").unwrap();

        let scope = resolve(&ast).unwrap();

        assert!(matches!(
            scope.lookup("first"),
            Some(Symbol::Geometry { .. })
        ));
        assert!(matches!(
            scope.lookup("second"),
            Some(Symbol::Geometry { .. })
        ));
        assert!(scope.lookup("shared").is_none());
    }
}
