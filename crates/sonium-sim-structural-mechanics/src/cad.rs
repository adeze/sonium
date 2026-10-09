//! Parameterized CAD/MeshLang entry points for structural studies.
//!
//! The design loop remains the small, body-fitted FEM workflow in
//! [`crate::optimization`]. This module gives that loop a versioned geometry
//! source: each design can be emitted as MeshLang, compiled to Gmsh `.geo`,
//! and (when Gmsh is installed) meshed into Sonium's [`Mesh`]. Keeping the
//! source generation deterministic makes the geometry part of a result's
//! provenance instead of an implicit collection of coordinates.

use serde::{Deserialize, Serialize};

use crate::{
    BracketShapeDesign, BracketShapeOptimizationResult, BracketShapeOptimizationStudy,
    BracketStudy, StructuralMechanicsError,
};

/// SI-valued parameters for the rectangular structural bracket CAD model.
///
/// The MeshLang source uses a three-dimensional box so it can be handed to
/// Gmsh for a solid mesh. The existing bracket optimizer is a plane-stress
/// study; its body-fitted FEM reanalysis remains the fast validation loop,
/// while this source is the portable CAD/meshing representation of each
/// design.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct MeshlangBracketParameters {
    pub length_m: f64,
    pub height_m: f64,
    pub thickness_m: f64,
    pub mesh_size_m: f64,
}

impl Default for MeshlangBracketParameters {
    fn default() -> Self {
        Self {
            length_m: 0.18,
            height_m: 0.036,
            thickness_m: 0.006,
            mesh_size_m: 0.006,
        }
    }
}

impl MeshlangBracketParameters {
    /// Create parameters after checking that all dimensions are usable in a
    /// positive-volume CAD model.
    pub fn new(
        length_m: f64,
        height_m: f64,
        thickness_m: f64,
        mesh_size_m: f64,
    ) -> Result<Self, StructuralMechanicsError> {
        let parameters = Self {
            length_m,
            height_m,
            thickness_m,
            mesh_size_m,
        };
        parameters.validate()?;
        Ok(parameters)
    }

    fn validate(self) -> Result<(), StructuralMechanicsError> {
        for (field, value) in [
            ("meshlang.length_m", self.length_m),
            ("meshlang.height_m", self.height_m),
            ("meshlang.thickness_m", self.thickness_m),
            ("meshlang.mesh_size_m", self.mesh_size_m),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.mesh_size_m > self.length_m
            || self.mesh_size_m > self.height_m
            || self.mesh_size_m > self.thickness_m
        {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "meshlang.mesh_size_m",
                value: "mesh size must not exceed the smallest bracket dimension".to_string(),
            });
        }
        Ok(())
    }

    /// Emit deterministic MeshLang source for this parameter point.
    pub fn source(self) -> Result<String, StructuralMechanicsError> {
        self.validate()?;
        Ok(format!(
            "// Sonium structural bracket CAD source; all values are SI metres.\n\
             param length_m = {:.17e}\n\
             param height_m = {:.17e}\n\
             param thickness_m = {:.17e}\n\
             param mesh_size_m = {:.17e}\n\
             geometry bracket {{\n\
                 let body = Box(length_m, height_m, thickness_m)\n\
                     |> translate(0.0, -0.5 * height_m, -0.5 * thickness_m)\n\
             }}\n\
             mesh for bracket {{\n\
                 min_size: mesh_size_m\n\
                 max_size: mesh_size_m\n\
                 order: 1\n\
             }}\n\
             physics {{\n\
                 group solid: volume = bracket.body\n\
                 group fixed: dirichlet = bracket.body.faces.left\n\
                 group loaded: neumann = bracket.body.faces.right\n\
             }}\n\
             export {{\n\
                 format: msh4\n\
                 order: 1\n\
                 filename: \"structural_bracket.msh\"\n\
             }}\n",
            self.length_m, self.height_m, self.thickness_m, self.mesh_size_m
        ))
    }

    /// Construct the corresponding introductory plane-stress study inputs.
    pub fn bracket_study(
        self,
        base: &BracketStudy,
    ) -> Result<BracketStudy, StructuralMechanicsError> {
        self.validate()?;
        let mut study = base.clone();
        study.geometry.length_m = self.length_m;
        study.geometry.height_m = self.height_m;
        study.geometry.thickness_m = self.thickness_m;
        Ok(study)
    }
}

/// Compiled geometry and machine-readable manifests returned by MeshLang.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MeshlangBracketCompileArtifact {
    pub source: String,
    pub geo_code: String,
    pub canonical_ir_json: String,
    pub sensitivity_manifest_json: String,
    pub provenance_manifest_json: String,
    pub backend_metadata_json: String,
}

/// A MeshLang-backed wrapper around the existing body-fitted bracket design
/// optimization. The optimization itself still reports its adjoint and
/// transactional [`sonium_fem::mesh::MeshMotion`] evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshlangBracketShapeOptimizationStudy {
    pub optimization: BracketShapeOptimizationStudy,
    pub mesh_size_m: f64,
}

impl MeshlangBracketShapeOptimizationStudy {
    pub fn introductory(base: BracketStudy) -> Self {
        let mesh_size_m = (base.geometry.thickness_m * 0.75)
            .min(base.geometry.height_m * 0.5)
            .max(1.0e-6);
        Self {
            optimization: BracketShapeOptimizationStudy::introductory(base),
            mesh_size_m,
        }
    }

    fn parameters_for_design(
        &self,
        design: BracketShapeDesign,
    ) -> Result<MeshlangBracketParameters, StructuralMechanicsError> {
        MeshlangBracketParameters::new(
            design.length_m,
            self.optimization.base.geometry.height_m,
            design.thickness_m,
            self.mesh_size_m.min(design.thickness_m),
        )
    }
}

/// FEM optimization result with initial and final CAD artifacts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MeshlangBracketShapeOptimizationResult {
    pub optimization: BracketShapeOptimizationResult,
    pub initial_geometry: MeshlangBracketCompileArtifact,
    pub final_geometry: MeshlangBracketCompileArtifact,
}

/// Run the body-fitted FEM design loop and compile its initial/final designs
/// into MeshLang artifacts. The function does not invoke Gmsh.
pub fn run_meshlang_bracket_shape_optimization(
    study: &MeshlangBracketShapeOptimizationStudy,
) -> Result<MeshlangBracketShapeOptimizationResult, StructuralMechanicsError> {
    if !study.mesh_size_m.is_finite() || study.mesh_size_m <= 0.0 {
        return Err(StructuralMechanicsError::InvalidConfiguration {
            field: "meshlang.mesh_size_m",
            value: study.mesh_size_m.to_string(),
        });
    }

    let initial_design = BracketShapeDesign {
        length_m: study.optimization.base.geometry.length_m,
        thickness_m: study.optimization.base.geometry.thickness_m,
    };
    let initial_geometry = compile_design(study, initial_design)?;
    let optimization = crate::run_bracket_shape_optimization(&study.optimization)?;
    let final_geometry = compile_design(study, optimization.final_design)?;

    Ok(MeshlangBracketShapeOptimizationResult {
        optimization,
        initial_geometry,
        final_geometry,
    })
}

fn compile_design(
    study: &MeshlangBracketShapeOptimizationStudy,
    design: BracketShapeDesign,
) -> Result<MeshlangBracketCompileArtifact, StructuralMechanicsError> {
    let parameters = study.parameters_for_design(design)?;
    compile_meshlang_bracket(parameters)
}

/// Compile one parameter point without invoking Gmsh.
pub fn compile_meshlang_bracket(
    parameters: MeshlangBracketParameters,
) -> Result<MeshlangBracketCompileArtifact, StructuralMechanicsError> {
    let source = parameters.source()?;
    let result = meshlang_compiler::compile_to_geo_full(&source, "structural_bracket.mlg")
        .map_err(|error| StructuralMechanicsError::SolverFailure {
            stage: "meshlang structural CAD compilation",
            message: error.to_string(),
        })?;
    Ok(MeshlangBracketCompileArtifact {
        source,
        geo_code: result.code,
        canonical_ir_json: result.canonical_ir_json,
        sensitivity_manifest_json: result.sensitivity_manifest_json,
        provenance_manifest_json: result.provenance_manifest_json,
        backend_metadata_json: result.backend_metadata_json,
    })
}

/// Compile and mesh one parameter point through the optional Gmsh bridge.
///
/// This is intentionally separate from [`run_meshlang_bracket_shape_optimization`]:
/// the optimization's body-fitted reanalysis is deterministic and does not
/// require an external process, while this method is the CAD-to-volume-mesh
/// hand-off for users who have Gmsh installed.
pub fn mesh_meshlang_bracket(
    parameters: MeshlangBracketParameters,
) -> Result<sonium_fem::mesh::Mesh, StructuralMechanicsError> {
    let source = parameters.source()?;
    sonium_mesher::mesh::mesh_from_mlg_str(&source, "structural_bracket.mlg").map_err(|error| {
        StructuralMechanicsError::InvalidMesh {
            reason: format!("MeshLang/Gmsh structural bridge failed: {error}"),
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn source_contains_parameterized_geometry_and_boundary_groups() {
        let parameters = MeshlangBracketParameters::default();
        let source = parameters.source().unwrap();
        assert!(source.contains("param length_m"));
        assert!(source.contains("Box(length_m, height_m, thickness_m)"));
        assert!(source.contains("group fixed"));
        assert!(source.contains("group loaded"));
    }

    #[test]
    fn invalid_mesh_size_is_rejected_before_compilation() {
        let error = MeshlangBracketParameters::new(0.18, 0.036, 0.006, 0.007).unwrap_err();
        assert!(matches!(
            error,
            StructuralMechanicsError::InvalidConfiguration {
                field: "meshlang.mesh_size_m",
                ..
            }
        ));
    }

    #[test]
    fn meshlang_compilation_emits_provenance_artifacts() {
        let mut base = BracketStudy::introductory_bracket();
        base.geometry.nx = 8;
        base.geometry.ny = 2;
        let mut study = MeshlangBracketShapeOptimizationStudy::introductory(base);
        study.optimization.iterations = 1;

        let result = run_meshlang_bracket_shape_optimization(&study).unwrap();
        assert!(!result.initial_geometry.geo_code.is_empty());
        assert!(!result.initial_geometry.canonical_ir_json.is_empty());
        assert!(!result.initial_geometry.provenance_manifest_json.is_empty());
        assert!(!result.final_geometry.geo_code.is_empty());
        assert!(result.optimization.body_fitted_reanalysis);
        assert!(result.optimization.mesh_motion_moved_nodes > 0);
    }

    #[test]
    fn meshlang_mesh_bridge_loads_a_volume_mesh_when_gmsh_is_configured() {
        if std::env::var_os("GMSH_PATH").is_none() && std::env::var_os("MESHLANG_GMSH").is_none() {
            return;
        }
        let mesh = mesh_meshlang_bracket(MeshlangBracketParameters::default()).unwrap();
        assert_eq!(mesh.dimension, 3);
        assert!(!mesh.nodes.is_empty());
        assert!(!mesh.elements.is_empty());
        assert!(mesh.physical_groups.by_name(3, "solid").is_some());
        assert!(mesh.physical_groups.by_name(2, "fixed").is_some());
        assert!(mesh.physical_groups.by_name(2, "loaded").is_some());
    }
}
