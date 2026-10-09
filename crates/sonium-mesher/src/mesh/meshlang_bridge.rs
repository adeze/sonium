//! MeshLang integration: compile `.mlg` sources and invoke Gmsh to produce a [`Mesh`].
//!
//! The pipeline is:
//!   `.mlg` source → [`meshlang_compiler`] → `.geo` → Gmsh subprocess → `.msh` → [`Mesh`]
//!
//! Gmsh must be installed and available on `PATH` (or set via `GMSH_PATH` or
//! `MESHLANG_GMSH`).

use std::ffi::OsStr;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

use super::Mesh;

/// Structured failure from the MeshLang → Gmsh → [`Mesh`] pipeline.
///
/// The checked APIs preserve the stage that rejected the input so applications
/// can distinguish a language diagnostic from an unavailable meshing backend
/// or an invalid generated mesh. Legacy string-returning helpers remain for
/// compatibility.
#[derive(Debug)]
pub enum MeshLangMeshingError {
    /// The source file could not be read.
    SourceRead {
        /// Requested source path.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },
    /// The MeshLang frontend rejected the source.
    Compile(meshlang_compiler::CompileError),
    /// The isolated artifact directory could not be created.
    TemporaryDirectory(std::io::Error),
    /// The generated GEO artifact could not be written.
    WriteGeo {
        /// GEO output path.
        path: PathBuf,
        /// Underlying filesystem error.
        source: std::io::Error,
    },
    /// Gmsh could not be launched.
    GmshLaunch {
        /// Configured Gmsh executable.
        command: String,
        /// Underlying process-launch error.
        source: std::io::Error,
    },
    /// Gmsh ran but reported a non-success exit status.
    GmshExit {
        /// Configured Gmsh executable.
        command: String,
        /// Platform exit code, if available.
        code: Option<i32>,
    },
    /// The generated MSH artifact did not pass Sonium's importer.
    MeshImport {
        /// MSH artifact path.
        path: PathBuf,
        /// Importer diagnostic.
        message: String,
    },
}

impl fmt::Display for MeshLangMeshingError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::SourceRead { path, source } => {
                write!(
                    f,
                    "failed to read MeshLang source {}: {source}",
                    path.display()
                )
            }
            Self::Compile(error) => write!(f, "MeshLang compile error: {error}"),
            Self::TemporaryDirectory(error) => {
                write!(f, "failed to create MeshLang meshing directory: {error}")
            }
            Self::WriteGeo { path, source } => {
                write!(
                    f,
                    "failed to write generated GEO {}: {source}",
                    path.display()
                )
            }
            Self::GmshLaunch { command, source } => {
                write!(f, "failed to run Gmsh executable {command:?}: {source}")
            }
            Self::GmshExit { command, code } => {
                write!(f, "Gmsh executable {command:?} exited with code {code:?}")
            }
            Self::MeshImport { path, message } => {
                write!(
                    f,
                    "failed to import generated MSH {}: {message}",
                    path.display()
                )
            }
        }
    }
}

impl std::error::Error for MeshLangMeshingError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::SourceRead { source, .. }
            | Self::WriteGeo { source, .. }
            | Self::GmshLaunch { source, .. } => Some(source),
            Self::Compile(error) => Some(error),
            Self::TemporaryDirectory(error) => Some(error),
            Self::GmshExit { .. } | Self::MeshImport { .. } => None,
        }
    }
}

/// Compile a MeshLang source string and mesh it with Gmsh.
///
/// `filename` is used only for error messages.
/// Requires Gmsh on `PATH` (or the `GMSH_PATH`/`MESHLANG_GMSH` env vars).
pub fn mesh_from_mlg_str(source: &str, filename: &str) -> Result<Mesh, String> {
    try_mesh_from_mlg_str(source, filename).map_err(|error| error.to_string())
}

/// Checked [`mesh_from_mlg_str`] pipeline with stage-preserving errors.
pub fn try_mesh_from_mlg_str(source: &str, filename: &str) -> Result<Mesh, MeshLangMeshingError> {
    let geo_code = meshlang_compiler::compile_to_geo(source, filename)
        .map_err(MeshLangMeshingError::Compile)?;

    let dir = tempfile::tempdir().map_err(MeshLangMeshingError::TemporaryDirectory)?;
    let geo_path = dir.path().join("mesh.geo");
    let msh_path = dir.path().join("mesh.msh");

    std::fs::write(&geo_path, &geo_code).map_err(|source| MeshLangMeshingError::WriteGeo {
        path: geo_path.clone(),
        source,
    })?;

    run_gmsh(&geo_path, &msh_path)?;

    super::gmsh::load_gmsh(&msh_path).map_err(|message| MeshLangMeshingError::MeshImport {
        path: msh_path,
        message,
    })
}

/// Compile a `.mlg` file and mesh it with Gmsh.
pub fn mesh_from_mlg_file(path: impl AsRef<Path>) -> Result<Mesh, String> {
    try_mesh_from_mlg_file(path).map_err(|error| error.to_string())
}

/// Checked [`mesh_from_mlg_file`] pipeline with stage-preserving errors.
pub fn try_mesh_from_mlg_file(path: impl AsRef<Path>) -> Result<Mesh, MeshLangMeshingError> {
    let path = path.as_ref();
    let source =
        std::fs::read_to_string(path).map_err(|source| MeshLangMeshingError::SourceRead {
            path: path.to_path_buf(),
            source,
        })?;
    try_mesh_from_mlg_str(&source, &path.to_string_lossy())
}

fn run_gmsh(geo: &Path, msh: &Path) -> Result<(), MeshLangMeshingError> {
    let gmsh = std::env::var("GMSH_PATH")
        .or_else(|_| std::env::var("MESHLANG_GMSH"))
        .unwrap_or_else(|_| "gmsh".to_string());
    run_gmsh_with_command(geo, msh, OsStr::new(&gmsh), &gmsh)
}

fn run_gmsh_with_command(
    geo: &Path,
    msh: &Path,
    command: &OsStr,
    command_display: &str,
) -> Result<(), MeshLangMeshingError> {
    let status = Command::new(command)
        .arg("-3")
        .arg(geo)
        .arg("-o")
        .arg(msh)
        .arg("-format")
        .arg("msh4")
        .status()
        .map_err(|source| MeshLangMeshingError::GmshLaunch {
            command: command_display.to_owned(),
            source,
        })?;

    if status.success() {
        Ok(())
    } else {
        Err(MeshLangMeshingError::GmshExit {
            command: command_display.to_owned(),
            code: status.code(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checked_meshlang_pipeline_preserves_frontend_file_and_gmsh_stages() {
        assert!(matches!(
            try_mesh_from_mlg_str("geometry {", "broken.mlg"),
            Err(MeshLangMeshingError::Compile(_))
        ));

        let directory = tempfile::tempdir().unwrap();
        let missing = directory.path().join("missing-input.mlg");
        assert!(matches!(
            try_mesh_from_mlg_file(&missing),
            Err(MeshLangMeshingError::SourceRead { path, .. }) if path == missing
        ));

        assert!(matches!(
            run_gmsh_with_command(
                Path::new("input.geo"),
                Path::new("output.msh"),
                OsStr::new("sonium-gmsh-definitely-not-installed"),
                "sonium-gmsh-definitely-not-installed",
            ),
            Err(MeshLangMeshingError::GmshLaunch { .. })
        ));
    }

    #[test]
    fn compiled_meshlang_mesh_loads_into_the_consumer_when_gmsh_is_configured() {
        if std::env::var_os("GMSH_PATH").is_none() && std::env::var_os("MESHLANG_GMSH").is_none() {
            return;
        }

        let source = r#"
            geometry smoke {
                let domain = Box(1.0, 1.0, 1.0)
            }
            mesh for smoke {
                min_size: 0.2
                max_size: 0.4
                order: 1
            }
            physics {
                group fluid: volume = smoke.domain
                group walls: neumann = smoke.domain.faces
            }
            export { format: msh4 order: 1 filename: "smoke.msh" }
        "#;

        let mesh = try_mesh_from_mlg_str(source, "consumer-smoke.mlg")
            .expect("MeshLang -> Gmsh -> sonium-mesher smoke pipeline failed");
        assert_eq!(mesh.dimension, 3);
        assert!(!mesh.nodes.is_empty());
        assert!(!mesh.elements.is_empty());
        assert!(mesh.physical_groups.by_name(3, "fluid").is_some());
        assert!(mesh.physical_groups.by_name(2, "walls").is_some());
        mesh.validate_physical_groups().unwrap();
    }
}
