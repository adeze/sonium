//! Bounded imports and versioned project files; saving never overwrites an existing file.

use crate::{DesignerError, Project};
use serde::{Deserialize, Serialize};
use std::{
    fmt::Display,
    fs::File,
    io::{Read, Write},
    path::Path,
};

const MAX_BYTES: u64 = 16 * 1024 * 1024;

#[derive(Serialize, Deserialize)]
struct Document {
    version: u32,
    project: Project,
}

fn error(path: &Path, error: impl Display) -> DesignerError {
    DesignerError::Invalid(format!("{}: {error}", path.display()))
}

pub fn read_text(path: &Path) -> Result<String, DesignerError> {
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|e| error(path, e))?
        .take(MAX_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| error(path, e))?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(error(path, "file exceeds the 16 MiB import limit"));
    }
    String::from_utf8(bytes).map_err(|e| error(path, e))
}

pub fn load_project(path: &Path) -> Result<Project, DesignerError> {
    let document: Document = serde_json::from_str(&read_text(path)?).map_err(|e| error(path, e))?;
    if document.version != 1 {
        return Err(error(path, "unsupported speaker-project version"));
    }
    document.project.validate_draft()?;
    Ok(document.project)
}

pub fn save_project_new(path: &Path, project: &Project) -> Result<(), DesignerError> {
    project.validate_draft()?;
    let bytes = serde_json::to_vec_pretty(&Document {
        version: 1,
        project: project.clone(),
    })
    .map_err(|e| error(path, e))?;
    if bytes.len() as u64 > MAX_BYTES {
        return Err(error(path, "project exceeds the 16 MiB import limit"));
    }
    let mut file = File::options()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| error(path, format!("{e}; choose a new filename")))?;
    if let Err(e) = file.write_all(&bytes).and_then(|_| file.sync_all()) {
        drop(file);
        let _ = std::fs::remove_file(path);
        return Err(error(path, e));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn directory() -> std::path::PathBuf {
        let id = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let path = std::env::temp_dir().join(format!("sonium-project-{}-{id}", std::process::id()));
        std::fs::create_dir(&path).unwrap();
        path
    }

    #[test]
    fn project_roundtrip_preserves_drafts_and_refuses_overwrite() {
        let dir = directory();
        let path = dir.join("project.json");
        save_project_new(&path, &Project::default()).unwrap();
        assert!(load_project(&path).unwrap().woofer.response.is_none());
        let original = std::fs::read(&path).unwrap();
        assert!(save_project_new(&path, &Project::demo()).is_err());
        assert_eq!(std::fs::read(&path).unwrap(), original);
        let demo = dir.join("demo.json");
        save_project_new(&demo, &Project::demo()).unwrap();
        assert!(load_project(&demo).unwrap().synthetic);
        std::fs::remove_dir_all(dir).unwrap();
    }

    #[test]
    fn project_load_rejects_bad_versions_invalid_values_and_oversized_input() {
        let dir = directory();
        let path = dir.join("bad.json");
        let mut document = Document {
            version: 2,
            project: Project::default(),
        };
        std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        assert!(load_project(&path).is_err());
        document.version = 1;
        document.project.crossover.woofer_capacitance_uf = -1.0;
        std::fs::write(&path, serde_json::to_vec(&document).unwrap()).unwrap();
        assert!(load_project(&path).is_err());
        File::options()
            .write(true)
            .truncate(true)
            .open(&path)
            .unwrap()
            .set_len(MAX_BYTES + 1)
            .unwrap();
        assert!(read_text(&path).unwrap_err().to_string().contains("16 MiB"));
        std::fs::remove_dir_all(dir).unwrap();
    }
}
