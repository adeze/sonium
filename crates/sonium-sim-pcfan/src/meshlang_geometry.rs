//! Alternative PC fan geometry via MeshLang DSL.
//!
//! This module provides a declarative geometry description using MeshLang,
//! an alternative to the procedural geometry in `geometry.rs`.

use meshlang_compiler::{CompileError, compile_to_geo_full};

const PCFAN_MESHLANG_SOURCE: &str = include_str!("../shapes/pcfan.mlg");

pub struct MeshlangPcfanGeometry {
    pub code: String,
    pub tag_map: Option<serde_json::Value>,
    pub diagnostics: Vec<meshlang_compiler::Diagnostic>,
}

impl MeshlangPcfanGeometry {
    pub fn compile() -> Result<Self, CompileError> {
        let result = compile_to_geo_full(PCFAN_MESHLANG_SOURCE, "pcfan.mlg")?;
        Ok(Self {
            code: result.code,
            tag_map: result
                .tag_map_json
                .and_then(|j| serde_json::from_str(&j).ok()),
            diagnostics: result.diagnostics,
        })
    }

    pub fn compile_to_geo() -> Result<String, CompileError> {
        meshlang_compiler::compile_to_geo(PCFAN_MESHLANG_SOURCE, "pcfan.mlg")
    }
}

impl std::fmt::Debug for MeshlangPcfanGeometry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshlangPcfanGeometry")
            .field("code_length", &self.code.len())
            .field("tag_map", &self.tag_map.is_some())
            .field("diagnostics_count", &self.diagnostics.len())
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meshlang_compiles_successfully() {
        let result = MeshlangPcfanGeometry::compile();
        assert!(
            result.is_ok(),
            "MeshLang compilation failed: {:?}",
            result.err()
        );
        let geom = result.unwrap();
        assert!(!geom.code.is_empty(), "Generated code should not be empty");
        assert!(
            geom.code.contains("SetFactory"),
            "Should contain Gmsh OpenCASCADE factory"
        );
        assert!(
            geom.code.contains("pcfan"),
            "Should reference pcfan geometry"
        );
    }

    #[test]
    fn meshlang_compile_to_geo_returns_string() {
        let result = MeshlangPcfanGeometry::compile_to_geo();
        assert!(result.is_ok());
        let code = result.unwrap();
        assert!(!code.is_empty());
        assert!(code.contains("Box(") || code.contains("Cylinder("));
    }

    #[test]
    fn meshlang_produces_valid_geo_syntax() {
        let code = MeshlangPcfanGeometry::compile_to_geo().unwrap();
        assert!(code.contains("SetFactory(\"OpenCASCADE\")"));
        assert!(code.contains("Geometry.OldNewReg = 0"));
    }

    #[test]
    fn meshlang_contains_basic_components() {
        let code = MeshlangPcfanGeometry::compile_to_geo().unwrap();
        assert!(code.contains("pcfan"), "Should contain pcfan geometry");
        assert!(
            code.contains("Box(") || code.contains("box1"),
            "Should contain Box"
        );
        assert!(
            code.contains("Cylinder(") || code.contains("cyl1"),
            "Should contain Cylinder"
        );
    }
}
