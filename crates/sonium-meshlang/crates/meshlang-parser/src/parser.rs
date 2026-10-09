//! Recursive-descent parser: token stream → AST.

use std::sync::Arc;

use meshlang_ast::span::Span;
use meshlang_ast::types::{NamedType, TypeAnnotation};
use meshlang_ast::units::Unit;
use meshlang_ast::*;
use serde::Serialize;

use crate::error::{DiagnosticCode, ParseError};
use crate::lexer::{Token, TokenKind};

pub const LANGUAGE_VERSION: u32 = 1;
pub const GRAMMAR_VERSION: &str = "1.0";
pub const DEFAULT_MAX_NESTING: usize = 128;

#[derive(Debug, Clone, Serialize)]
pub struct ParseReport {
    pub source_file: SourceFile,
    pub diagnostics: Vec<ParseError>,
}

impl ParseReport {
    pub fn is_success(&self) -> bool {
        self.diagnostics.is_empty()
    }
}

pub struct Parser {
    tokens: Vec<Token>,
    pos: usize,
    file_id: u32,
    source_name: Option<String>,
    nesting_depth: usize,
    max_nesting: usize,
}

type PResult<T> = Result<T, ParseError>;

impl Parser {
    pub fn new(tokens: Vec<Token>, file_id: u32) -> Self {
        Self::with_max_nesting(tokens, file_id, DEFAULT_MAX_NESTING)
    }

    pub fn with_max_nesting(mut tokens: Vec<Token>, file_id: u32, max_nesting: usize) -> Self {
        if tokens.last().map(|token| token.kind) != Some(TokenKind::Eof) {
            let end = tokens.last().map_or(0, |token| token.span.end);
            tokens.push(Token {
                kind: TokenKind::Eof,
                span: Span::with_file(end, end, file_id),
                text: Arc::from(""),
                error: None,
            });
        }
        Self {
            tokens,
            pos: 0,
            file_id,
            source_name: None,
            nesting_depth: 0,
            max_nesting: max_nesting.max(1),
        }
    }

    pub fn set_source_name(&mut self, source_name: impl Into<String>) {
        self.source_name = Some(source_name.into());
    }

    // ── Top Level ────────────────────────────────────────────────

    pub fn parse_source_file(&mut self) -> PResult<SourceFile> {
        let start = self.pos_span();
        let mut declarations = Vec::new();

        if self.check(TokenKind::Version) {
            self.parse_version_marker()?;
        }

        while !self.at_eof() {
            declarations.push(self.parse_declaration()?);
        }

        Ok(SourceFile::new(declarations, self.source_span(start)))
    }

    /// Parse as much of a file as possible while collecting a bounded set of
    /// diagnostics. Recovery only moves forward and therefore remains linear
    /// in the token count even for adversarial input.
    pub fn parse_source_file_recovering(&mut self, max_diagnostics: usize) -> ParseReport {
        // A zero limit would record no diagnostics at all and report
        // `is_success() == true` on erroneous input. Clamp to one so a
        // failing parse always surfaces at least the truncation error.
        let max_diagnostics = max_diagnostics.max(1);
        let start = self.pos_span();
        let mut declarations = Vec::new();
        let mut diagnostics = Vec::new();

        if self.check(TokenKind::Version)
            && let Err(error) = self.parse_version_marker()
            && diagnostics.len() < max_diagnostics
        {
            diagnostics.push(error);
        }

        while !self.at_eof() && diagnostics.len() < max_diagnostics {
            let failed_at = self.pos;
            match self.parse_declaration() {
                Ok(declaration) => declarations.push(declaration),
                Err(error) => {
                    if diagnostics.len() < max_diagnostics {
                        diagnostics.push(error);
                    }
                    self.synchronize_declaration(failed_at);
                }
            }
        }

        if !self.at_eof() {
            let limit_error = self.error_code(
                DiagnosticCode::TooManyDiagnostics,
                "diagnostic limit reached; remaining input was not parsed",
            );
            if diagnostics.len() == max_diagnostics {
                diagnostics[max_diagnostics - 1] = limit_error;
            } else {
                diagnostics.push(limit_error);
            }
        }

        ParseReport {
            source_file: SourceFile::new(declarations, self.source_span(start)),
            diagnostics,
        }
    }

    fn parse_version_marker(&mut self) -> PResult<()> {
        self.expect(TokenKind::Version)?;
        let version = self.current().clone();
        self.expect(TokenKind::Integer)?;
        self.eat(TokenKind::Semicolon);

        let parsed = version.text.parse::<u32>().map_err(|_| {
            self.error_at(
                DiagnosticCode::UnsupportedVersion,
                version.span,
                "language version must be a positive integer",
            )
        })?;
        if parsed != LANGUAGE_VERSION {
            return Err(self.error_at(
                DiagnosticCode::UnsupportedVersion,
                version.span,
                &format!("unsupported MeshLang version {parsed}; expected {LANGUAGE_VERSION}"),
            ));
        }
        Ok(())
    }

    fn parse_declaration(&mut self) -> PResult<Declaration> {
        if self.check(TokenKind::Error) {
            return Err(self.lexical_error());
        }
        match self.current().kind {
            TokenKind::Param => Ok(Declaration::Param(self.parse_param_decl()?)),
            TokenKind::Const => Ok(Declaration::Const(self.parse_const_decl()?)),
            TokenKind::Import => Ok(Declaration::Import(self.parse_import_decl()?)),
            TokenKind::Geometry => Ok(Declaration::Geometry(self.parse_geometry_block(false)?)),
            TokenKind::Mesh => Ok(Declaration::Mesh(self.parse_mesh_block()?)),
            TokenKind::Physics => Ok(Declaration::Physics(self.parse_physics_block()?)),
            TokenKind::Export => Ok(Declaration::Export(self.parse_export_block()?)),
            TokenKind::Struct => Ok(Declaration::Struct(self.parse_struct_decl(false)?)),
            TokenKind::Fn => Ok(Declaration::Function(self.parse_fn_decl(false)?)),
            TokenKind::Design => Ok(Declaration::DesignVariable(
                self.parse_design_variable_decl()?,
            )),
            TokenKind::Uncertainty => Ok(Declaration::Uncertainty(self.parse_uncertainty_decl()?)),
            TokenKind::Objective => Ok(Declaration::Objective(self.parse_objective_decl()?)),
            TokenKind::Constraint => Ok(Declaration::Constraint(self.parse_constraint_decl()?)),
            TokenKind::DesignRegion => {
                Ok(Declaration::DesignRegion(self.parse_design_region_decl()?))
            }
            TokenKind::PhysicalRegion => Ok(Declaration::PhysicalRegion(
                self.parse_physical_region_decl()?,
            )),
            TokenKind::Boundary => Ok(Declaration::Boundary(self.parse_boundary_decl()?)),
            TokenKind::Interface => Ok(Declaration::Interface(self.parse_interface_decl()?)),
            TokenKind::CoordinateSystem => Ok(Declaration::CoordinateSystem(
                self.parse_coordinate_system_decl()?,
            )),
            TokenKind::MaterialRegion => Ok(Declaration::MaterialRegion(
                self.parse_material_region_decl()?,
            )),
            TokenKind::Field => Ok(Declaration::GeometryField(
                self.parse_geometry_field_decl()?,
            )),
            TokenKind::Pub => {
                self.advance();
                match self.current().kind {
                    TokenKind::Param => Ok(Declaration::Param(self.parse_param_decl()?)),
                    TokenKind::Const => Ok(Declaration::Const(self.parse_const_decl()?)),
                    TokenKind::Geometry => {
                        Ok(Declaration::Geometry(self.parse_geometry_block(true)?))
                    }
                    TokenKind::Struct => Ok(Declaration::Struct(self.parse_struct_decl(true)?)),
                    TokenKind::Fn => Ok(Declaration::Function(self.parse_fn_decl(true)?)),
                    _ => Err(self.error(
                        "expected 'param', 'const', 'geometry', 'struct', or 'fn' after 'pub'",
                    )),
                }
            }
            _ => Err(self.error(&format!(
                "expected declaration, found '{}'",
                self.current().text
            ))),
        }
    }

    // ── Param / Const ────────────────────────────────────────────

    fn parse_param_decl(&mut self) -> PResult<ParamDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::Param)?;
        let name = self.parse_ident()?;
        let type_ann = if self.eat(TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(TokenKind::Assign)?;
        let value = self.parse_expr()?;
        let unit = self.try_parse_unit()?;

        Ok(ParamDecl {
            name,
            type_ann,
            value,
            unit,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_const_decl(&mut self) -> PResult<ConstDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::Const)?;
        let name = self.parse_ident()?;
        let type_ann = if self.eat(TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(TokenKind::Assign)?;
        let value = self.parse_expr()?;
        let unit = self.try_parse_unit()?;

        Ok(ConstDecl {
            name,
            type_ann,
            value,
            unit,
            span: start.merge(self.prev_span()),
        })
    }

    // ── Optimization and model metadata ───────────────────────────

    fn parse_design_variable_decl(&mut self) -> PResult<DesignVariableDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::Design)?;
        let name = self.parse_ident()?;
        let mut fields = self.parse_metadata_fields()?;
        let type_field = self.required_field(&mut fields, "type")?;
        let values = Self::take_field(&mut fields, "values");
        let variable_type = self.parse_design_variable_type(type_field.value, values)?;
        let initial = self.required_field(&mut fields, "initial")?.value;
        let lower = Self::take_field(&mut fields, "lower");
        let upper = Self::take_field(&mut fields, "upper");
        let bounds = match (lower, upper) {
            (None, None) => None,
            (Some(lower), Some(upper)) => Some(DesignBounds {
                span: lower.span.merge(upper.span),
                lower: lower.value,
                upper: upper.value,
                lower_inclusive: true,
                upper_inclusive: true,
            }),
            _ => return Err(self.error("design bounds require both 'lower' and 'upper'")),
        };
        let unit = Self::take_field(&mut fields, "unit")
            .map(|field| self.parse_unit_field(field))
            .transpose()?;
        let distribution = Self::take_field(&mut fields, "distribution")
            .map(|field| self.parse_distribution(field.value))
            .transpose()?;
        self.reject_unknown_fields(&fields)?;
        Ok(DesignVariableDecl {
            name,
            variable_type,
            initial,
            bounds,
            unit,
            distribution,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_uncertainty_decl(&mut self) -> PResult<UncertaintyDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::Uncertainty)?;
        let name = self.parse_ident()?;
        let mut fields = self.parse_metadata_fields()?;
        let distribution =
            self.parse_distribution(self.required_field(&mut fields, "distribution")?.value)?;
        self.reject_unknown_fields(&fields)?;
        Ok(UncertaintyDecl {
            name,
            distribution,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_objective_decl(&mut self) -> PResult<ObjectiveDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::Objective)?;
        let name = if self.check(TokenKind::LBrace) {
            None
        } else {
            Some(self.parse_ident()?)
        };
        let mut fields = self.parse_metadata_fields()?;
        let sense_field = self.required_field(&mut fields, "sense")?;
        let sense_ident = self.expr_into_ident(sense_field.value, sense_field.span)?;
        let sense = match sense_ident.name.as_str() {
            "minimize" => ObjectiveSense::Minimize,
            "maximize" => ObjectiveSense::Maximize,
            _ => {
                return Err(self.error_at(
                    DiagnosticCode::UnexpectedToken,
                    sense_ident.span,
                    "objective sense must be 'minimize' or 'maximize'",
                ));
            }
        };
        let expression = self.required_field(&mut fields, "expression")?.value;
        let weight = Self::take_field(&mut fields, "weight").map(|field| field.value);
        self.reject_unknown_fields(&fields)?;
        Ok(ObjectiveDecl {
            name,
            sense,
            expression,
            weight,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_constraint_decl(&mut self) -> PResult<ConstraintDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::Constraint)?;
        let name = if self.check(TokenKind::LBrace) {
            None
        } else {
            Some(self.parse_ident()?)
        };
        let mut fields = self.parse_metadata_fields()?;
        let kind_field = self.required_field(&mut fields, "kind")?;
        let kind_ident = self.expr_into_ident(kind_field.value, kind_field.span)?;
        let kind = match kind_ident.name.as_str() {
            "equality" => ConstraintKind::Equality {
                left: self.required_field(&mut fields, "left")?.value,
                right: self.required_field(&mut fields, "right")?.value,
                tolerance: Self::take_field(&mut fields, "tolerance").map(|field| field.value),
            },
            "inequality" => {
                let relation_field = self.required_field(&mut fields, "relation")?;
                let relation_ident =
                    self.expr_into_ident(relation_field.value, relation_field.span)?;
                let relation = match relation_ident.name.as_str() {
                    "less" => InequalityRelation::Less,
                    "less_equal" => InequalityRelation::LessEqual,
                    "greater" => InequalityRelation::Greater,
                    "greater_equal" => InequalityRelation::GreaterEqual,
                    _ => {
                        return Err(self.error_at(
                            DiagnosticCode::UnexpectedToken,
                            relation_ident.span,
                            "invalid inequality relation",
                        ));
                    }
                };
                ConstraintKind::Inequality {
                    left: self.required_field(&mut fields, "left")?.value,
                    relation,
                    right: self.required_field(&mut fields, "right")?.value,
                }
            }
            "range" => {
                let expression = self.required_field(&mut fields, "expression")?.value;
                let lower = self.required_field(&mut fields, "lower")?;
                let upper = self.required_field(&mut fields, "upper")?;
                ConstraintKind::Range {
                    expression,
                    bounds: DesignBounds {
                        span: lower.span.merge(upper.span),
                        lower: lower.value,
                        upper: upper.value,
                        lower_inclusive: true,
                        upper_inclusive: true,
                    },
                }
            }
            "volume_fraction" => ConstraintKind::VolumeFraction {
                region: self.required_field(&mut fields, "region")?.value,
                maximum: self.required_field(&mut fields, "maximum")?.value,
            },
            "minimum_feature" => ConstraintKind::MinimumFeature {
                region: self.required_field(&mut fields, "region")?.value,
                radius: self.required_field(&mut fields, "radius")?.value,
            },
            _ => {
                return Err(self.error_at(
                    DiagnosticCode::UnexpectedToken,
                    kind_ident.span,
                    "unknown constraint kind",
                ));
            }
        };
        self.reject_unknown_fields(&fields)?;
        Ok(ConstraintDecl {
            name,
            kind,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_design_region_decl(&mut self) -> PResult<DesignRegionDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::DesignRegion)?;
        let name = self.parse_ident()?;
        let mut fields = self.parse_metadata_fields()?;
        let kind_field = self.required_field(&mut fields, "kind")?;
        let kind_ident = self.expr_into_ident(kind_field.value, kind_field.span)?;
        let kind = match kind_ident.name.as_str() {
            "design" => DesignRegionKind::Design,
            "non_design" => DesignRegionKind::NonDesign,
            "keep_out" => DesignRegionKind::KeepOut,
            _ => {
                return Err(self.error_at(
                    DiagnosticCode::UnexpectedToken,
                    kind_ident.span,
                    "invalid design-region kind",
                ));
            }
        };
        let region = self.required_field(&mut fields, "region")?.value;
        let annotations = Self::take_field(&mut fields, "annotations")
            .map(|field| self.parse_annotations(field.value))
            .transpose()?
            .unwrap_or_default();
        self.reject_unknown_fields(&fields)?;
        Ok(DesignRegionDecl {
            name,
            region,
            kind,
            annotations,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_physical_region_decl(&mut self) -> PResult<PhysicalRegionDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::PhysicalRegion)?;
        let dimension = self.parse_entity_dimension()?;
        let name = self.parse_ident()?;
        let mut fields = self.parse_metadata_fields()?;
        let entities = self.required_field(&mut fields, "entities")?.value;
        let material = Self::take_field(&mut fields, "material")
            .map(|field| self.expr_into_ident(field.value, field.span))
            .transpose()?;
        self.reject_unknown_fields(&fields)?;
        Ok(PhysicalRegionDecl {
            name,
            dimension,
            entities,
            material,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_boundary_decl(&mut self) -> PResult<BoundaryDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::Boundary)?;
        let dimension = self.parse_entity_dimension()?;
        let name = self.parse_ident()?;
        let mut fields = self.parse_metadata_fields()?;
        let entities = self.required_field(&mut fields, "entities")?.value;
        let kind_field = self.required_field(&mut fields, "kind")?;
        let kind_ident = self.expr_into_ident(kind_field.value, kind_field.span)?;
        let kind = match kind_ident.name.as_str() {
            "dirichlet" => BoundaryKind::Dirichlet,
            "neumann" => BoundaryKind::Neumann,
            "robin" => BoundaryKind::Robin,
            "impedance" => BoundaryKind::Impedance,
            "symmetry" => BoundaryKind::Symmetry,
            "periodic" => BoundaryKind::Periodic,
            _ => BoundaryKind::Named(kind_ident),
        };
        Ok(BoundaryDecl {
            name,
            dimension,
            entities,
            kind,
            settings: fields,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_interface_decl(&mut self) -> PResult<InterfaceDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::Interface)?;
        let dimension = self.parse_entity_dimension()?;
        let name = self.parse_ident()?;
        let mut fields = self.parse_metadata_fields()?;
        let primary = self.required_field(&mut fields, "primary")?.value;
        let secondary = self.required_field(&mut fields, "secondary")?.value;
        let kind_field = self.required_field(&mut fields, "kind")?;
        let kind_ident = self.expr_into_ident(kind_field.value, kind_field.span)?;
        let kind = match kind_ident.name.as_str() {
            "conforming" => InterfaceKind::Conforming,
            "contact" => InterfaceKind::Contact,
            "mortar" => InterfaceKind::Mortar,
            "fluid_structure" => InterfaceKind::FluidStructure,
            _ => InterfaceKind::Named(kind_ident),
        };
        Ok(InterfaceDecl {
            name,
            dimension,
            primary,
            secondary,
            kind,
            settings: fields,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_coordinate_system_decl(&mut self) -> PResult<CoordinateSystemDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::CoordinateSystem)?;
        let name = self.parse_ident()?;
        let mut fields = self.parse_metadata_fields()?;
        let kind_field = self.required_field(&mut fields, "kind")?;
        let kind_ident = self.expr_into_ident(kind_field.value, kind_field.span)?;
        let kind = match kind_ident.name.as_str() {
            "cartesian" => CoordinateSystemKind::Cartesian {
                origin: self.required_field(&mut fields, "origin")?.value,
                axes: self.expr_into_array(self.required_field(&mut fields, "axes")?.value)?,
            },
            "cylindrical" => CoordinateSystemKind::Cylindrical {
                origin: self.required_field(&mut fields, "origin")?.value,
                axis: self.required_field(&mut fields, "axis")?.value,
                reference: self.required_field(&mut fields, "reference")?.value,
            },
            "spherical" => CoordinateSystemKind::Spherical {
                origin: self.required_field(&mut fields, "origin")?.value,
                polar_axis: self.required_field(&mut fields, "polar_axis")?.value,
                reference: self.required_field(&mut fields, "reference")?.value,
            },
            _ => {
                return Err(self.error_at(
                    DiagnosticCode::UnexpectedToken,
                    kind_ident.span,
                    "unknown coordinate-system kind",
                ));
            }
        };
        self.reject_unknown_fields(&fields)?;
        Ok(CoordinateSystemDecl {
            name,
            kind,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_material_region_decl(&mut self) -> PResult<MaterialRegionDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::MaterialRegion)?;
        let name = self.parse_ident()?;
        let mut fields = self.parse_metadata_fields()?;
        let region = self.required_field(&mut fields, "region")?.value;
        let material_field = self.required_field(&mut fields, "material")?;
        let material = self.expr_into_ident(material_field.value, material_field.span)?;
        Ok(MaterialRegionDecl {
            name,
            region,
            material,
            properties: fields,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_geometry_field_decl(&mut self) -> PResult<GeometryFieldDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::Field)?;
        let name = self.parse_ident()?;
        let mut fields = self.parse_metadata_fields()?;
        let kind_field = self.required_field(&mut fields, "kind")?;
        let kind_ident = self.expr_into_ident(kind_field.value, kind_field.span)?;
        let kind = match kind_ident.name.as_str() {
            "topology" => GeometryFieldKind::Topology {
                region: self.required_field(&mut fields, "region")?.value,
                initial_density: self.required_field(&mut fields, "initial_density")?.value,
                filter_radius: Self::take_field(&mut fields, "filter_radius")
                    .map(|field| field.value),
            },
            "level_set" => GeometryFieldKind::LevelSet {
                field: self.required_field(&mut fields, "field")?.value,
                iso_value: self.required_field(&mut fields, "iso_value")?.value,
            },
            "projected" => {
                let source_field = self.required_field(&mut fields, "source_field")?;
                let source_field = self.expr_into_ident(source_field.value, source_field.span)?;
                let method_field = self.required_field(&mut fields, "method")?;
                let method_ident = self.expr_into_ident(method_field.value, method_field.span)?;
                let method = match method_ident.name.as_str() {
                    "iso_surface" => ProjectionMethod::IsoSurface,
                    "signed_distance" => ProjectionMethod::SignedDistance,
                    "closest_point" => ProjectionMethod::ClosestPoint,
                    _ => {
                        return Err(self.error_at(
                            DiagnosticCode::UnexpectedToken,
                            method_ident.span,
                            "unknown projection method",
                        ));
                    }
                };
                GeometryFieldKind::Projected {
                    geometry: self.required_field(&mut fields, "geometry")?.value,
                    provenance: ProjectedGeometryProvenance {
                        source_field,
                        method,
                        tolerance: Self::take_field(&mut fields, "tolerance")
                            .map(|field| field.value),
                    },
                }
            }
            _ => {
                return Err(self.error_at(
                    DiagnosticCode::UnexpectedToken,
                    kind_ident.span,
                    "unknown geometry-field kind",
                ));
            }
        };
        self.reject_unknown_fields(&fields)?;
        Ok(GeometryFieldDecl {
            name,
            kind,
            span: start.merge(self.prev_span()),
        })
    }

    // ── Import ───────────────────────────────────────────────────

    fn parse_import_decl(&mut self) -> PResult<ImportDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::Import)?;

        // import { A, B } from "path"
        if self.check(TokenKind::LBrace) {
            self.advance();
            let mut names = vec![self.parse_ident()?];
            while self.eat(TokenKind::Comma) {
                if self.check(TokenKind::RBrace) {
                    break;
                }
                names.push(self.parse_ident()?);
            }
            self.expect(TokenKind::RBrace)?;
            self.expect(TokenKind::From)?;
            let path = self.parse_string()?;
            return Ok(ImportDecl {
                path,
                kind: ImportKind::Selective(names),
                span: start.merge(self.prev_span()),
            });
        }

        // import "path" (as alias)?
        let path = self.parse_string()?;
        let kind = if self.eat(TokenKind::As) {
            ImportKind::Aliased(self.parse_ident()?)
        } else {
            ImportKind::Whole
        };

        Ok(ImportDecl {
            path,
            kind,
            span: start.merge(self.prev_span()),
        })
    }

    // ── Functions ────────────────────────────────────────────────

    fn parse_fn_decl(&mut self, is_pub: bool) -> PResult<FnDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::Fn)?;
        let name = self.parse_ident()?;
        self.expect(TokenKind::LParen)?;
        let params = self.parse_param_list()?;
        self.expect(TokenKind::RParen)?;

        let return_type = if self.eat(TokenKind::Arrow) {
            Some(self.parse_type()?)
        } else {
            None
        };

        let body = self.parse_block()?;

        Ok(FnDecl {
            is_pub,
            name,
            params,
            return_type,
            body,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_param_list(&mut self) -> PResult<Vec<ParamDef>> {
        let mut params = Vec::new();
        if self.check(TokenKind::RParen) {
            return Ok(params);
        }

        params.push(self.parse_param_def()?);
        while self.eat(TokenKind::Comma) {
            if self.check(TokenKind::RParen) {
                break; // trailing comma
            }
            params.push(self.parse_param_def()?);
        }
        Ok(params)
    }

    fn parse_param_def(&mut self) -> PResult<ParamDef> {
        let start = self.pos_span();
        let name = self.parse_ident()?;
        self.expect(TokenKind::Colon)?;
        let type_ann = self.parse_type()?;
        let default = if self.eat(TokenKind::Assign) {
            Some(self.parse_expr()?)
        } else {
            None
        };

        Ok(ParamDef {
            name,
            type_ann,
            default,
            span: start.merge(self.prev_span()),
        })
    }

    // ── Geometry Block ───────────────────────────────────────────

    fn parse_geometry_block(&mut self, is_pub: bool) -> PResult<GeometryBlock> {
        let start = self.pos_span();
        self.expect(TokenKind::Geometry)?;
        let name = self.parse_ident()?;

        let params = if self.eat(TokenKind::LParen) {
            let p = self.parse_param_list()?;
            self.expect(TokenKind::RParen)?;
            p
        } else {
            vec![]
        };

        // A parameterized geometry can explicitly expose several named
        // entity ports: `geometry Frame(...) -> (body: Solid, inlet: Solid)`.
        // The parenthesized form keeps ports unambiguous with the following
        // geometry block and leaves the existing single-result syntax intact.
        let ports = if self.eat(TokenKind::Arrow) {
            self.expect(TokenKind::LParen)?;
            let ports = self.parse_param_list()?;
            self.expect(TokenKind::RParen)?;
            ports
        } else {
            vec![]
        };

        let body = self.parse_block()?;

        Ok(GeometryBlock {
            is_pub,
            name,
            params,
            ports,
            body,
            span: start.merge(self.prev_span()),
        })
    }

    // ── Mesh Block ───────────────────────────────────────────────

    fn parse_mesh_block(&mut self) -> PResult<MeshBlock> {
        let start = self.pos_span();
        self.expect(TokenKind::Mesh)?;

        // `mesh for model { ... }` is the explicit spelling.  Accept the
        // documented/legacy `mesh model { ... }` form as well so existing
        // product examples and downstream files remain source-compatible.
        let target = if self.eat(TokenKind::For) || self.check(TokenKind::Ident) {
            Some(self.parse_ident()?)
        } else {
            None
        };

        self.expect(TokenKind::LBrace)?;

        let mut statements = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_eof() {
            if self.check(TokenKind::Refine) {
                statements.push(MeshStatement::Refine(self.parse_refine_statement()?));
            } else if self.check(TokenKind::BoundaryLayer) {
                statements.push(MeshStatement::BoundaryLayer(
                    self.parse_boundary_layer_statement()?,
                ));
            } else if self.check(TokenKind::Transfinite) {
                statements.push(MeshStatement::Transfinite(
                    self.parse_transfinite_statement()?,
                ));
            } else if self.check(TokenKind::Embed) {
                statements.push(MeshStatement::Embed(self.parse_embed_statement()?));
            } else if self.check(TokenKind::Periodic) {
                statements.push(MeshStatement::Periodic(self.parse_periodic_statement()?));
            } else if self.check(TokenKind::Compound) {
                statements.push(MeshStatement::Compound(self.parse_compound_statement()?));
            } else if self.check(TokenKind::Metric) {
                statements.push(MeshStatement::Metric(self.parse_mesh_metric_statement()?));
            } else {
                statements.push(MeshStatement::KeyValue(self.parse_kv_pair()?));
            }
        }

        self.expect(TokenKind::RBrace)?;

        Ok(MeshBlock {
            target,
            statements,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_refine_statement(&mut self) -> PResult<RefineStatement> {
        let start = self.pos_span();
        self.expect(TokenKind::Refine)?;

        let kind = if self.eat(TokenKind::Near) {
            self.expect(TokenKind::LParen)?;
            let entity = self.parse_expr()?;
            self.expect(TokenKind::Comma)?;
            let args = self.parse_kv_args()?;
            self.expect(TokenKind::RParen)?;
            RefineKind::Near { entity, args }
        } else if self.eat(TokenKind::On) {
            self.expect(TokenKind::LParen)?;
            let entity = self.parse_expr()?;
            self.expect(TokenKind::RParen)?;
            RefineKind::On { entity }
        } else if self.check(TokenKind::In) {
            self.advance();
            self.expect(TokenKind::LParen)?;
            let region = self.parse_expr()?;
            self.expect(TokenKind::RParen)?;
            RefineKind::In { region }
        } else {
            // field(...)
            self.expect_ident("field")?;
            self.expect(TokenKind::LParen)?;
            let args = self.parse_kv_args()?;
            self.expect(TokenKind::RParen)?;
            RefineKind::Field { args }
        };

        self.expect(TokenKind::LBrace)?;
        let body = self.parse_kv_pairs_until_rbrace()?;
        self.expect(TokenKind::RBrace)?;

        Ok(RefineStatement {
            kind,
            body,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_boundary_layer_statement(&mut self) -> PResult<BoundaryLayerStatement> {
        let start = self.pos_span();
        self.expect(TokenKind::BoundaryLayer)?;
        self.expect(TokenKind::On)?;
        self.expect(TokenKind::LParen)?;
        let target = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;

        self.expect(TokenKind::LBrace)?;
        let body = self.parse_kv_pairs_until_rbrace()?;
        self.expect(TokenKind::RBrace)?;

        Ok(BoundaryLayerStatement {
            target,
            body,
            span: start.merge(self.prev_span()),
        })
    }

    // ── Transfinite / Embed / Periodic / Compound ────────────────

    /// Parse `transfinite curve(entity, nodes: N, ...) | surface(entity, ...) | volume(entity, ...)`
    fn parse_transfinite_statement(&mut self) -> PResult<TransfiniteStatement> {
        let start = self.pos_span();
        self.expect(TokenKind::Transfinite)?;

        // Next token determines the kind: "curve", "surface", or "volume"
        let kind_tok = self.current().clone();
        let kind = match kind_tok.text.as_ref() {
            "curve" => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                let entity = self.parse_expr()?;
                let mut settings = Vec::new();
                while self.eat(TokenKind::Comma) {
                    if self.check(TokenKind::RParen) {
                        break;
                    }
                    settings.push(self.parse_kv_pair()?);
                }
                self.expect(TokenKind::RParen)?;
                TransfiniteKind::Curve { entity, settings }
            }
            "surface" => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                let entity = self.parse_expr()?;
                let mut settings = Vec::new();
                while self.eat(TokenKind::Comma) {
                    if self.check(TokenKind::RParen) {
                        break;
                    }
                    settings.push(self.parse_kv_pair()?);
                }
                self.expect(TokenKind::RParen)?;
                TransfiniteKind::Surface { entity, settings }
            }
            "volume" => {
                self.advance();
                self.expect(TokenKind::LParen)?;
                let entity = self.parse_expr()?;
                let mut settings = Vec::new();
                while self.eat(TokenKind::Comma) {
                    if self.check(TokenKind::RParen) {
                        break;
                    }
                    settings.push(self.parse_kv_pair()?);
                }
                self.expect(TokenKind::RParen)?;
                TransfiniteKind::Volume { entity, settings }
            }
            _ => {
                return Err(
                    self.error("expected 'curve', 'surface', or 'volume' after 'transfinite'")
                );
            }
        };

        Ok(TransfiniteStatement {
            kind,
            span: start.merge(self.prev_span()),
        })
    }

    /// Parse `embed point(source) in volume(target)` or `embed curve(source) in surface(target)`
    fn parse_embed_statement(&mut self) -> PResult<EmbedStatement> {
        let start = self.pos_span();
        self.expect(TokenKind::Embed)?;

        // Parse embed dimension + source
        let embed_dim_tok = self.current().clone();
        let embed_dim = match embed_dim_tok.text.as_ref() {
            "point" => EmbedDim::Point,
            "curve" => EmbedDim::Curve,
            "surface" => EmbedDim::Surface,
            _ => return Err(self.error("expected 'point', 'curve', or 'surface' after 'embed'")),
        };
        self.advance();
        self.expect(TokenKind::LParen)?;
        let source = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;

        // Expect 'in'
        self.expect(TokenKind::In)?;

        // Parse target dimension + target
        let target_dim_tok = self.current().clone();
        let target_dim = match target_dim_tok.text.as_ref() {
            "surface" => EmbedDim::Surface,
            "volume" => EmbedDim::Volume,
            _ => return Err(self.error("expected 'surface' or 'volume' after 'in'")),
        };
        self.advance();
        self.expect(TokenKind::LParen)?;
        let target = self.parse_expr()?;
        self.expect(TokenKind::RParen)?;

        Ok(EmbedStatement {
            embed_dim,
            source,
            target,
            target_dim,
            span: start.merge(self.prev_span()),
        })
    }

    /// Parse `periodic surface(source, target, transform: [...])`
    fn parse_periodic_statement(&mut self) -> PResult<PeriodicStatement> {
        let start = self.pos_span();
        self.expect(TokenKind::Periodic)?;

        // Parse required dimension word
        let dim_tok = self.current().clone();
        let dim = match dim_tok.text.as_ref() {
            "surface" => PeriodicDim::Surface,
            "curve" => PeriodicDim::Curve,
            _ => return Err(self.error("expected 'surface' or 'curve' after 'periodic'")),
        };
        self.advance();

        self.expect(TokenKind::LParen)?;
        let source = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;
        let target = self.parse_expr()?;
        self.expect(TokenKind::Comma)?;

        // Expect `transform:` or `translation:` key
        let _kv = self.parse_kv_pair()?;
        let transform = _kv.value;

        self.expect(TokenKind::RParen)?;

        Ok(PeriodicStatement {
            dim,
            source,
            target,
            transform,
            span: start.merge(self.prev_span()),
        })
    }

    /// Parse `compound surface(a, b, c)` or `compound volume(a, b)`
    fn parse_compound_statement(&mut self) -> PResult<CompoundStatement> {
        let start = self.pos_span();
        self.expect(TokenKind::Compound)?;

        // Require dimension word
        let dim_tok = self.current().clone();
        match dim_tok.text.as_ref() {
            "surface" | "curve" | "volume" => self.advance(),
            _ => {
                return Err(self.error("expected 'surface', 'curve', or 'volume' after 'compound'"));
            }
        }

        self.expect(TokenKind::LParen)?;
        let mut entities = vec![self.parse_expr()?];
        while self.eat(TokenKind::Comma) {
            if self.check(TokenKind::RParen) {
                break;
            }
            entities.push(self.parse_expr()?);
        }
        self.expect(TokenKind::RParen)?;

        Ok(CompoundStatement {
            entities,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_mesh_metric_statement(&mut self) -> PResult<MeshMetricStatement> {
        let start = self.pos_span();
        self.expect(TokenKind::Metric)?;
        let target = self.parse_expr()?;
        let mut fields = self.parse_metadata_fields()?;
        let kind_field = self.required_field(&mut fields, "kind")?;
        let kind_ident = self.expr_into_ident(kind_field.value, kind_field.span)?;
        let metric = match kind_ident.name.as_str() {
            "isotropic" => MeshMetricKind::Isotropic {
                size: self.required_field(&mut fields, "size")?.value,
            },
            "anisotropic" => MeshMetricKind::Anisotropic {
                components: self
                    .expr_into_array(self.required_field(&mut fields, "components")?.value)?,
            },
            "hessian" => MeshMetricKind::Hessian {
                field: self.required_field(&mut fields, "field")?.value,
                minimum_size: self.required_field(&mut fields, "minimum_size")?.value,
                maximum_size: self.required_field(&mut fields, "maximum_size")?.value,
            },
            _ => {
                return Err(self.error_at(
                    DiagnosticCode::UnexpectedToken,
                    kind_ident.span,
                    "unknown mesh metric kind",
                ));
            }
        };
        let coordinate_system = Self::take_field(&mut fields, "coordinate_system")
            .map(|field| self.expr_into_ident(field.value, field.span))
            .transpose()?;
        self.reject_unknown_fields(&fields)?;
        Ok(MeshMetricStatement {
            target,
            metric,
            coordinate_system,
            span: start.merge(self.prev_span()),
        })
    }

    // ── Physics Block ────────────────────────────────────────────

    fn parse_physics_block(&mut self) -> PResult<PhysicsBlock> {
        let start = self.pos_span();
        self.expect(TokenKind::Physics)?;
        self.expect(TokenKind::LBrace)?;

        let mut statements = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_eof() {
            if self.check(TokenKind::Group) {
                statements.push(PhysicsStatement::Group(self.parse_group_decl()?));
            } else if self.check(TokenKind::Let) {
                statements.push(PhysicsStatement::Let(self.parse_let_statement()?));
            } else {
                return Err(self.error("expected 'group' or 'let' in physics block"));
            }
        }

        self.expect(TokenKind::RBrace)?;

        Ok(PhysicsBlock {
            statements,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_group_decl(&mut self) -> PResult<GroupDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::Group)?;
        let name = self.parse_ident()?;
        self.expect(TokenKind::Colon)?;
        let bc_type = self.parse_ident()?;
        self.expect(TokenKind::Assign)?;
        let value = self.parse_expr()?;

        Ok(GroupDecl {
            name,
            bc_type,
            value,
            span: start.merge(self.prev_span()),
        })
    }

    // ── Export Block ─────────────────────────────────────────────

    fn parse_export_block(&mut self) -> PResult<ExportBlock> {
        let start = self.pos_span();
        self.expect(TokenKind::Export)?;

        let name = if self.eat(TokenKind::As) {
            Some(self.parse_string()?)
        } else {
            None
        };

        self.expect(TokenKind::LBrace)?;
        let settings = self.parse_kv_pairs_until_rbrace()?;
        self.expect(TokenKind::RBrace)?;

        Ok(ExportBlock {
            name,
            settings,
            span: start.merge(self.prev_span()),
        })
    }

    // ── Struct ───────────────────────────────────────────────────

    fn parse_struct_decl(&mut self, is_pub: bool) -> PResult<StructDecl> {
        let start = self.pos_span();
        self.expect(TokenKind::Struct)?;
        let name = self.parse_ident()?;
        self.expect(TokenKind::LBrace)?;

        let mut fields = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_eof() {
            let fstart = self.pos_span();
            let fname = self.parse_ident()?;
            self.expect(TokenKind::Colon)?;
            let ftype = self.parse_type()?;
            self.eat(TokenKind::Comma);

            fields.push(StructField {
                name: fname,
                type_ann: ftype,
                span: fstart.merge(self.prev_span()),
            });
        }

        self.expect(TokenKind::RBrace)?;

        Ok(StructDecl {
            is_pub,
            name,
            fields,
            span: start.merge(self.prev_span()),
        })
    }

    // ── Statements ───────────────────────────────────────────────

    fn parse_block(&mut self) -> PResult<Block> {
        self.enter_nesting()?;
        let result = self.parse_block_inner();
        self.leave_nesting();
        result
    }

    fn parse_block_inner(&mut self) -> PResult<Block> {
        let start = self.pos_span();
        self.expect(TokenKind::LBrace)?;

        let mut statements = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_eof() {
            statements.push(self.parse_statement()?);
        }

        self.expect(TokenKind::RBrace)?;

        Ok(Block {
            statements,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_statement(&mut self) -> PResult<Statement> {
        match self.current().kind {
            TokenKind::Let => Ok(Statement::Let(self.parse_let_statement()?)),
            TokenKind::For => Ok(Statement::For(self.parse_for_statement()?)),
            TokenKind::If => Ok(Statement::If(self.parse_if_statement()?)),
            TokenKind::Group => Ok(Statement::Group(self.parse_group_decl()?)),
            // Key-value pairs (ident: expr) vs expressions
            _ => {
                // Peek ahead: if ident followed by colon, it's a kv pair
                if Self::is_value_identifier_kind(self.current().kind)
                    && self.peek_kind(1) == Some(TokenKind::Colon)
                {
                    Ok(Statement::KeyValue(self.parse_kv_pair()?))
                } else {
                    Ok(Statement::Expr(self.parse_expr()?))
                }
            }
        }
    }

    fn parse_let_statement(&mut self) -> PResult<LetStatement> {
        let start = self.pos_span();
        self.expect(TokenKind::Let)?;
        let name = self.parse_ident()?;
        let type_ann = if self.eat(TokenKind::Colon) {
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(TokenKind::Assign)?;
        let value = self.parse_expr()?;

        Ok(LetStatement {
            name,
            type_ann,
            value,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_for_statement(&mut self) -> PResult<ForStatement> {
        let start = self.pos_span();
        self.expect(TokenKind::For)?;

        let mut clauses = vec![self.parse_for_clause()?];
        while self.eat(TokenKind::Comma) {
            clauses.push(self.parse_for_clause()?);
        }

        let body = self.parse_block()?;

        Ok(ForStatement {
            clauses,
            body,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_for_clause(&mut self) -> PResult<ForClause> {
        let start = self.pos_span();
        let var = self.parse_ident()?;
        self.expect(TokenKind::In)?;
        let range = self.parse_range_expr()?;

        Ok(ForClause {
            var,
            range,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_range_expr(&mut self) -> PResult<RangeExpr> {
        let start = self.pos_span();
        let lo = self.parse_additive()?;

        let inclusive = if self.eat(TokenKind::DotDotEq) {
            true
        } else {
            self.expect(TokenKind::DotDot)?;
            false
        };

        let hi = self.parse_additive()?;

        Ok(RangeExpr {
            start: Box::new(lo),
            end: Box::new(hi),
            inclusive,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_if_statement(&mut self) -> PResult<IfStatement> {
        let start = self.pos_span();
        self.expect(TokenKind::If)?;
        let condition = self.parse_expr()?;
        let then_block = self.parse_block()?;

        let else_block = if self.eat(TokenKind::Else) {
            if self.check(TokenKind::If) {
                Some(ElseBranch::ElseIf(Box::new(self.parse_if_statement()?)))
            } else {
                Some(ElseBranch::Else(self.parse_block()?))
            }
        } else {
            None
        };

        Ok(IfStatement {
            condition,
            then_block,
            else_block,
            span: start.merge(self.prev_span()),
        })
    }

    // ── Expressions (Pratt-style precedence climbing) ────────────

    pub fn parse_expr(&mut self) -> PResult<Expr> {
        self.enter_nesting()?;
        let result = self.parse_pipe();
        self.leave_nesting();
        result
    }

    fn parse_pipe(&mut self) -> PResult<Expr> {
        let mut expr = self.parse_csg()?;

        while self.eat(TokenKind::PipeArrow) {
            let start = expr.span();
            let function = self.parse_ident()?;
            self.expect(TokenKind::LParen)?;
            let args = self.parse_argument_list()?;
            self.expect(TokenKind::RParen)?;

            expr = Expr::Pipe(PipeExpr {
                value: Box::new(expr),
                function,
                args,
                span: start.merge(self.prev_span()),
            });
        }

        Ok(expr)
    }

    fn parse_csg(&mut self) -> PResult<Expr> {
        let mut expr = self.parse_or()?;

        loop {
            let start = expr.span();
            let op = match self.current().kind {
                TokenKind::Pipe => {
                    self.advance();
                    BinaryOp::CsgUnion
                }
                TokenKind::Ampersand if self.peek_kind(1) != Some(TokenKind::Ampersand) => {
                    self.advance();
                    BinaryOp::CsgIntersect
                }
                // Note: `-` for CSG difference is handled in parse_additive
                // and disambiguated during type checking. At parse time,
                // `-` is always parsed as arithmetic Sub.
                _ => break,
            };

            let right = self.parse_or()?;
            expr = Expr::Binary(BinaryExpr {
                left: Box::new(expr),
                op,
                right: Box::new(right),
                span: start.merge(self.prev_span()),
            });
        }

        Ok(expr)
    }

    fn parse_or(&mut self) -> PResult<Expr> {
        let mut expr = self.parse_and()?;
        while self.eat(TokenKind::PipePipe) {
            let start = expr.span();
            let right = self.parse_and()?;
            expr = Expr::Binary(BinaryExpr {
                left: Box::new(expr),
                op: BinaryOp::Or,
                right: Box::new(right),
                span: start.merge(self.prev_span()),
            });
        }
        Ok(expr)
    }

    fn parse_and(&mut self) -> PResult<Expr> {
        let mut expr = self.parse_comparison()?;
        while self.eat(TokenKind::AmpAmp) {
            let start = expr.span();
            let right = self.parse_comparison()?;
            expr = Expr::Binary(BinaryExpr {
                left: Box::new(expr),
                op: BinaryOp::And,
                right: Box::new(right),
                span: start.merge(self.prev_span()),
            });
        }
        Ok(expr)
    }

    fn parse_comparison(&mut self) -> PResult<Expr> {
        let expr = self.parse_additive()?;

        let op = match self.current().kind {
            TokenKind::Eq => BinaryOp::Eq,
            TokenKind::Ne => BinaryOp::Ne,
            TokenKind::Lt => BinaryOp::Lt,
            TokenKind::Gt => BinaryOp::Gt,
            TokenKind::Le => BinaryOp::Le,
            TokenKind::Ge => BinaryOp::Ge,
            _ => return Ok(expr),
        };

        let start = expr.span();
        self.advance();
        let right = self.parse_additive()?;

        Ok(Expr::Binary(BinaryExpr {
            left: Box::new(expr),
            op,
            right: Box::new(right),
            span: start.merge(self.prev_span()),
        }))
    }

    fn parse_additive(&mut self) -> PResult<Expr> {
        let mut expr = self.parse_multiplicative()?;

        loop {
            let start = expr.span();
            let op = match self.current().kind {
                TokenKind::Plus => BinaryOp::Add,
                TokenKind::Minus => BinaryOp::Sub, // Disambiguated as CsgDifference during type check
                _ => break,
            };
            self.advance();
            let right = self.parse_multiplicative()?;
            expr = Expr::Binary(BinaryExpr {
                left: Box::new(expr),
                op,
                right: Box::new(right),
                span: start.merge(self.prev_span()),
            });
        }

        Ok(expr)
    }

    fn parse_multiplicative(&mut self) -> PResult<Expr> {
        let mut expr = self.parse_unary()?;

        loop {
            let start = expr.span();
            let op = match self.current().kind {
                TokenKind::Star => BinaryOp::Mul,
                TokenKind::Slash => BinaryOp::Div,
                TokenKind::Percent
                    if self
                        .peek_kind(1)
                        .is_some_and(Self::is_expression_start_kind) =>
                {
                    BinaryOp::Mod
                }
                _ => break,
            };
            self.advance();
            let right = self.parse_unary()?;
            expr = Expr::Binary(BinaryExpr {
                left: Box::new(expr),
                op,
                right: Box::new(right),
                span: start.merge(self.prev_span()),
            });
        }

        Ok(expr)
    }

    fn parse_unary(&mut self) -> PResult<Expr> {
        self.enter_nesting()?;
        let result = self.parse_unary_inner();
        self.leave_nesting();
        result
    }

    fn parse_unary_inner(&mut self) -> PResult<Expr> {
        match self.current().kind {
            TokenKind::Minus => {
                let start = self.pos_span();
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expr::Unary(UnaryExpr {
                    op: UnaryOp::Neg,
                    operand: Box::new(operand),
                    span: start.merge(self.prev_span()),
                }))
            }
            TokenKind::Bang => {
                let start = self.pos_span();
                self.advance();
                let operand = self.parse_unary()?;
                Ok(Expr::Unary(UnaryExpr {
                    op: UnaryOp::Not,
                    operand: Box::new(operand),
                    span: start.merge(self.prev_span()),
                }))
            }
            _ => self.parse_power(),
        }
    }

    fn parse_power(&mut self) -> PResult<Expr> {
        let expr = self.parse_postfix()?;

        if self.eat(TokenKind::StarStar) {
            let start = expr.span();
            let right = self.parse_unary()?; // right-associative
            return Ok(Expr::Binary(BinaryExpr {
                left: Box::new(expr),
                op: BinaryOp::Pow,
                right: Box::new(right),
                span: start.merge(self.prev_span()),
            }));
        }

        Ok(expr)
    }

    fn parse_postfix(&mut self) -> PResult<Expr> {
        let mut expr = self.parse_primary()?;

        loop {
            match self.current().kind {
                TokenKind::Dot => {
                    let start = expr.span();
                    self.advance();
                    let field = self.parse_ident()?;
                    expr = Expr::FieldAccess(FieldAccessExpr {
                        object: Box::new(expr),
                        field,
                        span: start.merge(self.prev_span()),
                    });
                }
                TokenKind::LBracket => {
                    let start = expr.span();
                    self.advance();
                    let index = self.parse_expr()?;
                    self.expect(TokenKind::RBracket)?;
                    expr = Expr::Index(IndexExpr {
                        object: Box::new(expr),
                        index: Box::new(index),
                        span: start.merge(self.prev_span()),
                    });
                }
                _ => break,
            }
        }

        Ok(expr)
    }

    fn parse_primary(&mut self) -> PResult<Expr> {
        match self.current().kind {
            TokenKind::Integer => {
                let token = self.current().clone();
                self.advance();
                let val: i64 = token.text.parse().map_err(|_| {
                    self.error_at(
                        DiagnosticCode::InvalidNumber,
                        token.span,
                        "integer literal is outside the signed 64-bit range",
                    )
                })?;
                Ok(Expr::literal(Literal::Int(val), token.span))
            }
            TokenKind::Float => {
                let token = self.current().clone();
                self.advance();
                let val: f64 = token.text.parse().map_err(|_| {
                    self.error_at(
                        DiagnosticCode::InvalidNumber,
                        token.span,
                        "invalid floating-point literal",
                    )
                })?;
                if !val.is_finite() {
                    return Err(self.error_at(
                        DiagnosticCode::InvalidNumber,
                        token.span,
                        "floating-point literal overflows the finite 64-bit range",
                    ));
                }
                Ok(Expr::literal(Literal::Float(val), token.span))
            }
            TokenKind::StringLit => {
                let span = self.pos_span();
                let s = self.parse_string()?;
                Ok(Expr::literal(Literal::String(s), span))
            }
            TokenKind::True => {
                let span = self.pos_span();
                self.advance();
                Ok(Expr::literal(Literal::Bool(true), span))
            }
            TokenKind::False => {
                let span = self.pos_span();
                self.advance();
                Ok(Expr::literal(Literal::Bool(false), span))
            }
            TokenKind::TypeIdent => {
                let start = self.pos_span();
                let name = Ident {
                    name: self.current().text.to_string(),
                    span: self.pos_span(),
                };
                self.advance();

                if self.eat(TokenKind::LParen) {
                    // Constructor: TypeName(args)
                    let args = self.parse_argument_list()?;
                    self.expect(TokenKind::RParen)?;
                    Ok(Expr::Constructor(ConstructorExpr {
                        type_name: name,
                        args,
                        span: start.merge(self.prev_span()),
                    }))
                } else if self.eat(TokenKind::LBrace) {
                    // Struct init: TypeName { field: value, ... }
                    let fields = self.parse_struct_init_fields()?;
                    self.expect(TokenKind::RBrace)?;
                    Ok(Expr::StructInit(StructInitExpr {
                        type_name: name,
                        fields,
                        span: start.merge(self.prev_span()),
                    }))
                } else {
                    // Bare type identifier used as value (e.g., axis: Z)
                    Ok(Expr::Ident(name))
                }
            }
            kind if Self::is_value_identifier_kind(kind) => {
                let start = self.pos_span();
                let ident = self.parse_ident()?;
                // If followed by `(`, parse as a function call
                if self.eat(TokenKind::LParen) {
                    let args = self.parse_argument_list()?;
                    self.expect(TokenKind::RParen)?;
                    Ok(Expr::Call(CallExpr {
                        function: ident,
                        args,
                        span: start.merge(self.prev_span()),
                    }))
                } else {
                    Ok(Expr::Ident(ident))
                }
            }
            TokenKind::LParen => {
                let opening = self.pos_span();
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(TokenKind::RParen)?;
                let closing = self.prev_span();
                let span = opening.try_merge(closing).map_err(|_| {
                    self.error_at(
                        DiagnosticCode::UnexpectedToken,
                        opening,
                        "parenthesized expression crosses source files",
                    )
                })?;
                Ok(Expr::Paren(ParenExpr::new(expr, span)))
            }
            TokenKind::LBracket => {
                let start = self.pos_span();
                self.advance();
                let mut elements = Vec::new();
                if !self.check(TokenKind::RBracket) {
                    elements.push(self.parse_expr()?);
                    while self.eat(TokenKind::Comma) {
                        if self.check(TokenKind::RBracket) {
                            break;
                        }
                        elements.push(self.parse_expr()?);
                    }
                }
                self.expect(TokenKind::RBracket)?;
                Ok(Expr::Array(ArrayExpr {
                    elements,
                    span: start.merge(self.prev_span()),
                }))
            }
            TokenKind::If => {
                let stmt = self.parse_if_statement()?;
                Ok(Expr::If(Box::new(stmt)))
            }
            TokenKind::Error => Err(self.lexical_error()),
            _ => Err(self.error(&format!(
                "expected expression, found '{}'",
                self.current().text
            ))),
        }
    }

    fn parse_argument_list(&mut self) -> PResult<Vec<Argument>> {
        let mut args = Vec::new();
        if self.check(TokenKind::RParen) {
            return Ok(args);
        }

        args.push(self.parse_argument()?);
        while self.eat(TokenKind::Comma) {
            if self.check(TokenKind::RParen) {
                break;
            }
            args.push(self.parse_argument()?);
        }
        Ok(args)
    }

    fn parse_argument(&mut self) -> PResult<Argument> {
        let start = self.pos_span();

        // Try named argument: ident ":" expr
        if Self::is_value_identifier_kind(self.current().kind)
            && self.peek_kind(1) == Some(TokenKind::Colon)
        {
            let name = self.parse_ident()?;
            self.expect(TokenKind::Colon)?;
            let value = self.parse_expr()?;
            return Ok(Argument {
                name: Some(name),
                value,
                span: start.merge(self.prev_span()),
            });
        }

        // Positional argument
        let value = self.parse_expr()?;
        Ok(Argument {
            name: None,
            value,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_struct_init_fields(&mut self) -> PResult<Vec<StructInitField>> {
        let mut fields = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_eof() {
            let start = self.pos_span();
            let name = self.parse_ident()?;
            self.expect(TokenKind::Colon)?;
            let value = self.parse_expr()?;
            fields.push(StructInitField {
                name,
                value,
                span: start.merge(self.prev_span()),
            });
            if !self.eat(TokenKind::Comma) {
                break;
            }
        }
        Ok(fields)
    }

    // ── Types ────────────────────────────────────────────────────

    fn parse_type(&mut self) -> PResult<TypeAnnotation> {
        self.enter_nesting()?;
        let result = self.parse_type_inner();
        self.leave_nesting();
        result
    }

    fn parse_type_inner(&mut self) -> PResult<TypeAnnotation> {
        if self.check(TokenKind::LBracket) {
            return self.parse_array_type();
        }

        let start = self.pos_span();
        let name = if self.check(TokenKind::TypeIdent) {
            let n = self.current().text.to_string();
            self.advance();
            n
        } else if self.check(TokenKind::Ident) {
            // Allow lowercase type names: float, int, bool, etc.
            let n = self.current().text.to_string();
            self.advance();
            n
        } else {
            return Err(self.error("expected type name"));
        };

        // Generic: Type<Params>
        if self.eat(TokenKind::Lt) {
            let mut params = vec![self.parse_type()?];
            while self.eat(TokenKind::Comma) {
                params.push(self.parse_type()?);
            }
            self.expect(TokenKind::Gt)?;
            return Ok(TypeAnnotation::Generic {
                name,
                params,
                span: start.merge(self.prev_span()),
            });
        }

        Ok(TypeAnnotation::Named(NamedType {
            name,
            span: start.merge(self.prev_span()),
        }))
    }

    fn parse_array_type(&mut self) -> PResult<TypeAnnotation> {
        let start = self.pos_span();
        self.expect(TokenKind::LBracket)?;
        let element = self.parse_type()?;

        if self.eat(TokenKind::Semicolon) {
            // Fixed-size: [Type; N]
            let size_token = self.current().clone();
            self.expect(TokenKind::Integer)?;
            let size: u64 = size_token.text.parse().map_err(|_| {
                self.error_at(
                    DiagnosticCode::InvalidNumber,
                    size_token.span,
                    "array size is outside the unsigned 64-bit range",
                )
            })?;
            self.expect(TokenKind::RBracket)?;
            Ok(TypeAnnotation::FixedArray {
                element: Box::new(element),
                size,
                span: start.merge(self.prev_span()),
            })
        } else {
            // Dynamic: [Type]
            self.expect(TokenKind::RBracket)?;
            Ok(TypeAnnotation::DynArray {
                element: Box::new(element),
                span: start.merge(self.prev_span()),
            })
        }
    }

    // ── Helpers ──────────────────────────────────────────────────

    fn parse_ident(&mut self) -> PResult<Ident> {
        let tok = self.current().clone();
        match tok.kind {
            kind if kind == TokenKind::TypeIdent || Self::is_value_identifier_kind(kind) => {
                self.advance();
                Ok(Ident {
                    name: tok.text.to_string(),
                    span: tok.span,
                })
            }
            _ => Err(self.error(&format!("expected identifier, found '{}'", tok.text))),
        }
    }

    fn expect_ident(&mut self, expected: &str) -> PResult<()> {
        let tok = self.current();
        if Self::is_value_identifier_kind(tok.kind) && tok.text.as_ref() == expected {
            self.advance();
            Ok(())
        } else {
            Err(self.error(&format!("expected '{}', found '{}'", expected, tok.text)))
        }
    }

    fn parse_string(&mut self) -> PResult<String> {
        let tok = self.current().clone();
        self.expect(TokenKind::StringLit)?;
        let Some(inner) = tok
            .text
            .strip_prefix('"')
            .and_then(|text| text.strip_suffix('"'))
        else {
            return Err(self.error_at(
                DiagnosticCode::Lexical,
                tok.span,
                "malformed string literal",
            ));
        };

        let mut value = String::with_capacity(inner.len());
        let mut chars = inner.chars();
        while let Some(ch) = chars.next() {
            if ch != '\\' {
                value.push(ch);
                continue;
            }
            let Some(escaped) = chars.next() else {
                return Err(self.error_at(
                    DiagnosticCode::Lexical,
                    tok.span,
                    "unterminated string escape",
                ));
            };
            value.push(match escaped {
                '"' => '"',
                '\\' => '\\',
                'n' => '\n',
                'r' => '\r',
                't' => '\t',
                '0' => '\0',
                _ => {
                    return Err(self.error_at(
                        DiagnosticCode::Lexical,
                        tok.span,
                        "invalid string escape",
                    ));
                }
            });
        }
        Ok(value)
    }

    fn try_parse_unit(&mut self) -> PResult<Option<Unit>> {
        // A following `name:` starts the next key/value entry, not a unit.
        if self.peek_kind(1) == Some(TokenKind::Colon) {
            return Ok(None);
        }

        let starts_unit = matches!(
            self.current().kind,
            TokenKind::Ident | TokenKind::TypeIdent | TokenKind::Percent
        ) || (self.check(TokenKind::Integer)
            && self.current().text.as_ref() == "1");
        if !starts_unit {
            return Ok(None);
        }

        let start = self.current().span;
        let mut end = self.pos + 1;
        let mut last_end = self.current().span.end;
        let mut unit_text = self.current().text.to_string();
        let mut paren_depth = 0usize;
        let mut previous = self.current().kind;

        while let Some(token) = self.tokens.get(end) {
            // Compound units are lexical suffixes. Requiring adjacency avoids
            // swallowing a following expression or closing delimiter.
            if token.span.start != last_end {
                break;
            }
            let allowed = match token.kind {
                TokenKind::Ident
                | TokenKind::TypeIdent
                | TokenKind::Star
                | TokenKind::Slash
                | TokenKind::Percent
                | TokenKind::Caret
                | TokenKind::Integer => true,
                TokenKind::Plus | TokenKind::Minus => previous == TokenKind::Caret,
                TokenKind::LParen => {
                    paren_depth += 1;
                    true
                }
                TokenKind::RParen if paren_depth > 0 => {
                    paren_depth -= 1;
                    true
                }
                _ => false,
            };
            if !allowed {
                break;
            }
            unit_text.push_str(token.text.as_ref());
            last_end = token.span.end;
            previous = token.kind;
            end += 1;
        }

        match Unit::parse(&unit_text) {
            Ok(unit) => {
                self.pos = end;
                Ok(Some(unit))
            }
            Err(error) => {
                let span = start.merge(
                    self.tokens
                        .get(end.saturating_sub(1))
                        .map_or(start, |token| token.span),
                );
                Err(self.error_at(
                    DiagnosticCode::InvalidUnit,
                    span,
                    &format!("invalid unit '{unit_text}': {error}"),
                ))
            }
        }
    }

    fn parse_kv_pair(&mut self) -> PResult<KeyValue> {
        let start = self.pos_span();
        let key = self.parse_ident()?;
        self.expect(TokenKind::Colon)?;
        let value = self.parse_expr()?;
        let unit = self.try_parse_unit()?;

        Ok(KeyValue {
            key,
            value,
            unit,
            span: start.merge(self.prev_span()),
        })
    }

    fn parse_kv_args(&mut self) -> PResult<Vec<KeyValue>> {
        let mut args = vec![self.parse_kv_pair()?];
        while self.eat(TokenKind::Comma) {
            if self.check(TokenKind::RParen) {
                break;
            }
            args.push(self.parse_kv_pair()?);
        }
        Ok(args)
    }

    fn parse_kv_pairs_until_rbrace(&mut self) -> PResult<Vec<KeyValue>> {
        let mut pairs = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_eof() {
            pairs.push(self.parse_kv_pair()?);
        }
        Ok(pairs)
    }

    // ── Token manipulation ───────────────────────────────────────

    fn parse_metadata_fields(&mut self) -> PResult<Vec<KeyValue>> {
        self.expect(TokenKind::LBrace)?;
        let mut fields = Vec::new();
        while !self.check(TokenKind::RBrace) && !self.at_eof() {
            fields.push(self.parse_kv_pair()?);
            self.eat(TokenKind::Comma);
        }
        self.expect(TokenKind::RBrace)?;
        Ok(fields)
    }

    fn take_field(fields: &mut Vec<KeyValue>, name: &str) -> Option<KeyValue> {
        fields
            .iter()
            .position(|field| field.key.name == name)
            .map(|position| fields.remove(position))
    }

    fn required_field(&self, fields: &mut Vec<KeyValue>, name: &str) -> PResult<KeyValue> {
        Self::take_field(fields, name)
            .ok_or_else(|| self.error(&format!("missing required field '{name}'")))
    }

    fn reject_unknown_fields(&self, fields: &[KeyValue]) -> PResult<()> {
        if let Some(field) = fields.first() {
            return Err(self.error_at(
                DiagnosticCode::UnexpectedToken,
                field.span,
                &format!("unknown or duplicate field '{}'", field.key.name),
            ));
        }
        Ok(())
    }

    fn expr_into_ident(&self, expression: Expr, span: Span) -> PResult<Ident> {
        match expression {
            Expr::Ident(ident) => Ok(ident),
            Expr::Literal(literal) => match literal.value {
                Literal::String(name) => Ok(Ident { name, span }),
                _ => Err(self.error_at(
                    DiagnosticCode::UnexpectedToken,
                    literal.span,
                    "expected identifier or string",
                )),
            },
            _ => Err(self.error_at(
                DiagnosticCode::UnexpectedToken,
                span,
                "expected identifier or string",
            )),
        }
    }

    fn expr_into_array(&self, expression: Expr) -> PResult<Vec<Expr>> {
        let span = expression.span();
        match expression {
            Expr::Array(array) => Ok(array.elements),
            _ => Err(self.error_at(
                DiagnosticCode::UnexpectedToken,
                span,
                "expected array expression",
            )),
        }
    }

    fn parse_unit_field(&self, field: KeyValue) -> PResult<Unit> {
        if let Some(unit) = field.unit {
            return Ok(unit);
        }
        let text = match field.value {
            Expr::Ident(ident) => ident.name,
            Expr::Literal(literal) => match literal.value {
                Literal::String(text) => text,
                _ => {
                    return Err(self.error_at(
                        DiagnosticCode::InvalidUnit,
                        literal.span,
                        "unit must be an identifier or string",
                    ));
                }
            },
            _ => {
                return Err(self.error_at(
                    DiagnosticCode::InvalidUnit,
                    field.span,
                    "unit must be an identifier or string",
                ));
            }
        };
        Unit::parse(&text).map_err(|error| {
            self.error_at(
                DiagnosticCode::InvalidUnit,
                field.span,
                &format!("invalid unit '{text}': {error}"),
            )
        })
    }

    fn parse_entity_dimension(&mut self) -> PResult<EntityDimension> {
        let dimension = self.parse_ident()?;
        match dimension.name.as_str() {
            "point" => Ok(EntityDimension::Point),
            "curve" => Ok(EntityDimension::Curve),
            "surface" => Ok(EntityDimension::Surface),
            "volume" => Ok(EntityDimension::Volume),
            _ => Err(self.error_at(
                DiagnosticCode::UnexpectedToken,
                dimension.span,
                "dimension must be point, curve, surface, or volume",
            )),
        }
    }

    fn parse_design_variable_type(
        &self,
        expression: Expr,
        values_field: Option<KeyValue>,
    ) -> PResult<DesignVariableType> {
        let span = expression.span();
        if let Expr::Call(mut call) = expression {
            if call.function.name == "categorical" {
                let values = self.required_argument(&mut call.args, "values", call.span)?;
                return Ok(DesignVariableType::Categorical {
                    values: self.expr_into_array(values)?,
                });
            }
            return Err(self.error_at(
                DiagnosticCode::UnexpectedToken,
                call.span,
                "unknown design-variable type",
            ));
        }
        let ident = self.expr_into_ident(expression, span)?;
        match ident.name.as_str() {
            "continuous" => Ok(DesignVariableType::Continuous),
            "integer" => Ok(DesignVariableType::Integer),
            "boolean" => Ok(DesignVariableType::Boolean),
            "shape" => Ok(DesignVariableType::Shape),
            "topology" => Ok(DesignVariableType::Topology),
            "level_set" => Ok(DesignVariableType::LevelSet),
            "categorical" => {
                let values = values_field.ok_or_else(|| {
                    self.error_at(
                        DiagnosticCode::UnexpectedToken,
                        ident.span,
                        "categorical design variables require a 'values' array",
                    )
                })?;
                Ok(DesignVariableType::Categorical {
                    values: self.expr_into_array(values.value)?,
                })
            }
            _ => Err(self.error_at(
                DiagnosticCode::UnexpectedToken,
                ident.span,
                "unknown design-variable type",
            )),
        }
    }

    fn parse_distribution(&self, expression: Expr) -> PResult<Distribution> {
        let span = expression.span();
        let Expr::Call(mut call) = expression else {
            return Err(self.error_at(
                DiagnosticCode::UnexpectedToken,
                span,
                "distribution must be a constructor call",
            ));
        };
        let distribution = match call.function.name.as_str() {
            "deterministic" => Distribution::Deterministic {
                value: self.required_argument(&mut call.args, "value", call.span)?,
            },
            "uniform" => Distribution::Uniform {
                lower: self.required_argument(&mut call.args, "lower", call.span)?,
                upper: self.required_argument(&mut call.args, "upper", call.span)?,
            },
            "normal" => Distribution::Normal {
                mean: self.required_argument(&mut call.args, "mean", call.span)?,
                standard_deviation: self.required_argument(
                    &mut call.args,
                    "standard_deviation",
                    call.span,
                )?,
            },
            "log_normal" => Distribution::LogNormal {
                mean: self.required_argument(&mut call.args, "mean", call.span)?,
                standard_deviation: self.required_argument(
                    &mut call.args,
                    "standard_deviation",
                    call.span,
                )?,
            },
            "discrete" => Distribution::Discrete {
                values: self.expr_into_array(self.required_argument(
                    &mut call.args,
                    "values",
                    call.span,
                )?)?,
                weights: self.expr_into_array(self.required_argument(
                    &mut call.args,
                    "weights",
                    call.span,
                )?)?,
            },
            _ => {
                return Err(self.error_at(
                    DiagnosticCode::UnexpectedToken,
                    call.span,
                    "unknown distribution",
                ));
            }
        };
        if let Some(extra) = call.args.first() {
            return Err(self.error_at(
                DiagnosticCode::UnexpectedToken,
                extra.span,
                "unexpected distribution argument",
            ));
        }
        Ok(distribution)
    }

    fn take_argument(arguments: &mut Vec<Argument>, name: &str) -> Option<Expr> {
        if let Some(position) = arguments.iter().position(|argument| {
            argument
                .name
                .as_ref()
                .is_some_and(|ident| ident.name == name)
        }) {
            return Some(arguments.remove(position).value);
        }
        arguments
            .iter()
            .position(|argument| argument.name.is_none())
            .map(|position| arguments.remove(position).value)
    }

    fn required_argument(
        &self,
        arguments: &mut Vec<Argument>,
        name: &str,
        span: Span,
    ) -> PResult<Expr> {
        Self::take_argument(arguments, name).ok_or_else(|| {
            self.error_at(
                DiagnosticCode::UnexpectedToken,
                span,
                &format!("missing required argument '{name}'"),
            )
        })
    }

    fn parse_annotations(&self, expression: Expr) -> PResult<Vec<ManufacturingAnnotation>> {
        self.expr_into_array(expression)?
            .into_iter()
            .map(|annotation| self.parse_annotation(annotation))
            .collect()
    }

    fn parse_annotation(&self, expression: Expr) -> PResult<ManufacturingAnnotation> {
        let span = expression.span();
        if let Expr::Ident(ident) = expression {
            return if ident.name == "frozen" {
                Ok(ManufacturingAnnotation::Frozen)
            } else {
                Err(self.error_at(
                    DiagnosticCode::UnexpectedToken,
                    ident.span,
                    "unknown manufacturing annotation",
                ))
            };
        }
        let Expr::Call(mut call) = expression else {
            return Err(self.error_at(
                DiagnosticCode::UnexpectedToken,
                span,
                "manufacturing annotation must be a call or 'frozen'",
            ));
        };
        let annotation = match call.function.name.as_str() {
            "minimum_thickness" => {
                let value = self.required_argument(&mut call.args, "value", call.span)?;
                let unit = Self::take_argument(&mut call.args, "unit")
                    .map(|expr| self.parse_unit_expression(expr))
                    .transpose()?;
                ManufacturingAnnotation::MinimumThickness { value, unit }
            }
            "minimum_radius" => {
                let value = self.required_argument(&mut call.args, "value", call.span)?;
                let unit = Self::take_argument(&mut call.args, "unit")
                    .map(|expr| self.parse_unit_expression(expr))
                    .transpose()?;
                ManufacturingAnnotation::MinimumRadius { value, unit }
            }
            "overhang" => ManufacturingAnnotation::Overhang {
                maximum_angle: self.required_argument(
                    &mut call.args,
                    "maximum_angle",
                    call.span,
                )?,
                direction: self.required_argument(&mut call.args, "direction", call.span)?,
            },
            "draw_direction" => ManufacturingAnnotation::DrawDirection {
                direction: self.required_argument(&mut call.args, "direction", call.span)?,
            },
            "symmetry" => ManufacturingAnnotation::Symmetry {
                kind: self.parse_symmetry_kind(self.required_argument(
                    &mut call.args,
                    "kind",
                    call.span,
                )?)?,
            },
            _ => {
                return Err(self.error_at(
                    DiagnosticCode::UnexpectedToken,
                    call.span,
                    "unknown manufacturing annotation",
                ));
            }
        };
        if let Some(extra) = call.args.first() {
            return Err(self.error_at(
                DiagnosticCode::UnexpectedToken,
                extra.span,
                "unexpected manufacturing-annotation argument",
            ));
        }
        Ok(annotation)
    }

    fn parse_symmetry_kind(&self, expression: Expr) -> PResult<SymmetryKind> {
        let span = expression.span();
        let Expr::Call(mut call) = expression else {
            return Err(self.error_at(
                DiagnosticCode::UnexpectedToken,
                span,
                "symmetry kind must be a constructor call",
            ));
        };
        let kind = match call.function.name.as_str() {
            "plane" => SymmetryKind::Plane {
                origin: self.required_argument(&mut call.args, "origin", call.span)?,
                normal: self.required_argument(&mut call.args, "normal", call.span)?,
            },
            "axis" => SymmetryKind::Axis {
                origin: self.required_argument(&mut call.args, "origin", call.span)?,
                direction: self.required_argument(&mut call.args, "direction", call.span)?,
            },
            "cyclic" => SymmetryKind::Cyclic {
                axis: self.required_argument(&mut call.args, "axis", call.span)?,
                sectors: self.required_argument(&mut call.args, "sectors", call.span)?,
            },
            _ => {
                return Err(self.error_at(
                    DiagnosticCode::UnexpectedToken,
                    call.span,
                    "unknown symmetry kind",
                ));
            }
        };
        if let Some(extra) = call.args.first() {
            return Err(self.error_at(
                DiagnosticCode::UnexpectedToken,
                extra.span,
                "unexpected symmetry argument",
            ));
        }
        Ok(kind)
    }

    fn parse_unit_expression(&self, expression: Expr) -> PResult<Unit> {
        let span = expression.span();
        let text = match expression {
            Expr::Ident(ident) => ident.name,
            Expr::Literal(literal) => match literal.value {
                Literal::String(text) => text,
                _ => {
                    return Err(self.error_at(
                        DiagnosticCode::InvalidUnit,
                        literal.span,
                        "unit argument must be an identifier or string",
                    ));
                }
            },
            _ => {
                return Err(self.error_at(
                    DiagnosticCode::InvalidUnit,
                    span,
                    "unit argument must be an identifier or string",
                ));
            }
        };
        Unit::parse(&text).map_err(|error| {
            self.error_at(
                DiagnosticCode::InvalidUnit,
                span,
                &format!("invalid unit '{text}': {error}"),
            )
        })
    }

    fn is_value_identifier_kind(kind: TokenKind) -> bool {
        matches!(
            kind,
            TokenKind::Ident
                | TokenKind::From
                | TokenKind::Near
                | TokenKind::On
                | TokenKind::Default
                | TokenKind::Design
                | TokenKind::Uncertainty
                | TokenKind::Objective
                | TokenKind::Constraint
                | TokenKind::DesignRegion
                | TokenKind::PhysicalRegion
                | TokenKind::Boundary
                | TokenKind::Interface
                | TokenKind::CoordinateSystem
                | TokenKind::MaterialRegion
                | TokenKind::Geometry
                | TokenKind::Field
                | TokenKind::Metric
        )
    }

    fn is_expression_start_kind(kind: TokenKind) -> bool {
        matches!(
            kind,
            TokenKind::Integer
                | TokenKind::Float
                | TokenKind::StringLit
                | TokenKind::True
                | TokenKind::False
                | TokenKind::TypeIdent
                | TokenKind::LParen
                | TokenKind::LBracket
                | TokenKind::If
                | TokenKind::Minus
                | TokenKind::Bang
                | TokenKind::Error
        ) || Self::is_value_identifier_kind(kind)
    }

    fn current(&self) -> &Token {
        &self.tokens[self.pos.min(self.tokens.len() - 1)]
    }

    fn advance(&mut self) {
        if self.pos + 1 < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn check(&self, kind: TokenKind) -> bool {
        self.current().kind == kind
    }

    fn eat(&mut self, kind: TokenKind) -> bool {
        if self.check(kind) {
            self.advance();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, kind: TokenKind) -> PResult<()> {
        if self.check(kind) {
            self.advance();
            Ok(())
        } else {
            Err(self.error(&format!(
                "expected {:?}, found '{}'",
                kind,
                self.current().text
            )))
        }
    }

    fn at_eof(&self) -> bool {
        self.current().kind == TokenKind::Eof
    }

    fn peek_kind(&self, offset: usize) -> Option<TokenKind> {
        self.pos
            .checked_add(offset)
            .and_then(|position| self.tokens.get(position))
            .map(|token| token.kind)
    }

    fn pos_span(&self) -> Span {
        self.current().span
    }

    fn prev_span(&self) -> Span {
        if self.pos > 0 {
            self.tokens[self.pos - 1].span
        } else {
            Span::DUMMY
        }
    }

    fn source_span(&self, start: Span) -> Span {
        if self.pos == 0 {
            start
        } else {
            start.merge(self.prev_span())
        }
    }

    fn enter_nesting(&mut self) -> PResult<()> {
        if self.nesting_depth >= self.max_nesting {
            return Err(self.error_code(
                DiagnosticCode::NestingLimit,
                &format!("maximum parser nesting of {} exceeded", self.max_nesting),
            ));
        }
        self.nesting_depth += 1;
        Ok(())
    }

    fn leave_nesting(&mut self) {
        self.nesting_depth = self.nesting_depth.saturating_sub(1);
    }

    fn synchronize_declaration(&mut self, failed_at: usize) {
        if self.pos <= failed_at && !self.at_eof() {
            self.advance();
        }

        let mut brace_depth = self.tokens[failed_at..self.pos.min(self.tokens.len())]
            .iter()
            .fold(0_i32, |depth, token| match token.kind {
                TokenKind::LBrace => depth + 1,
                TokenKind::RBrace => (depth - 1).max(0),
                _ => depth,
            });

        while !self.at_eof() {
            if brace_depth == 0 && self.is_declaration_start() {
                break;
            }
            match self.current().kind {
                TokenKind::LBrace => brace_depth += 1,
                TokenKind::RBrace => brace_depth = (brace_depth - 1).max(0),
                _ => {}
            }
            self.advance();
        }
    }

    fn is_declaration_start(&self) -> bool {
        matches!(
            self.current().kind,
            TokenKind::Param
                | TokenKind::Const
                | TokenKind::Import
                | TokenKind::Geometry
                | TokenKind::Mesh
                | TokenKind::Physics
                | TokenKind::Export
                | TokenKind::Struct
                | TokenKind::Fn
                | TokenKind::Pub
                | TokenKind::Design
                | TokenKind::Uncertainty
                | TokenKind::Objective
                | TokenKind::Constraint
                | TokenKind::DesignRegion
                | TokenKind::PhysicalRegion
                | TokenKind::Boundary
                | TokenKind::Interface
                | TokenKind::CoordinateSystem
                | TokenKind::MaterialRegion
                | TokenKind::Field
        )
    }

    fn lexical_error(&self) -> ParseError {
        let token = self.current();
        let message = token.error.map_or("invalid token", |error| error.message());
        self.error_at(DiagnosticCode::Lexical, token.span, message)
    }

    fn error(&self, msg: &str) -> ParseError {
        self.error_code(DiagnosticCode::UnexpectedToken, msg)
    }

    fn error_code(&self, code: DiagnosticCode, msg: &str) -> ParseError {
        self.error_at(code, self.current().span, msg)
    }

    fn error_at(&self, code: DiagnosticCode, span: Span, msg: &str) -> ParseError {
        let error = ParseError::new(code, msg, span);
        if let Some(source_name) = &self.source_name {
            error.with_source_name(source_name.clone())
        } else {
            error
        }
    }

    pub fn file_id(&self) -> u32 {
        self.file_id
    }
}

#[cfg(test)]
mod tests {
    use super::{ParseReport, Parser};
    use crate::lexer::{Lexer, TokenKind};
    use meshlang_ast::span::Span;
    use meshlang_ast::{Expr, SourceFile};

    fn parser_for(source: &str, file_id: u32) -> Parser {
        let mut lexer = Lexer::new(source, file_id);
        Parser::new(lexer.tokenize(), file_id)
    }

    #[test]
    fn parse_argument_accepts_named_contextual_and_positional_forms() {
        let mut named = parser_for("from: 1", 7);
        let argument = named.parse_argument().expect("named argument");
        assert_eq!(
            argument.name.as_ref().map(|name| name.name.as_str()),
            Some("from")
        );
        assert!(matches!(argument.value, Expr::Literal(_)));

        let mut positional = parser_for("1", 7);
        let argument = positional.parse_argument().expect("positional argument");
        assert!(argument.name.is_none());
        assert!(matches!(argument.value, Expr::Literal(_)));
    }

    #[test]
    fn expression_start_contract_and_parser_identity_are_explicit() {
        assert!(Parser::is_expression_start_kind(TokenKind::Integer));
        assert!(Parser::is_expression_start_kind(TokenKind::From));
        assert!(!Parser::is_expression_start_kind(TokenKind::Comma));

        let parser = parser_for("1", 41);
        assert_eq!(parser.file_id(), 41);
    }

    #[test]
    fn parse_report_success_is_empty_diagnostic_contract() {
        let report = ParseReport {
            source_file: SourceFile::new(Vec::new(), Span::with_file(0, 0, 3)),
            diagnostics: Vec::new(),
        };
        assert!(report.is_success());
    }
}
