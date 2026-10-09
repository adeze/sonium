//! ParaView collection (`.pvd`) writer for time / frequency series.
//!
//! A `.pvd` is a small XML index pointing at one `.vtu` per timestep,
//! letting ParaView animate or scrub through a sweep.

use std::io::Write;
use std::path::{Path, PathBuf};

/// Parse and validate the small XML collection emitted by [`write_pvd`].
///
/// PVD is deliberately kept dependency-free here. The writer emits one
/// `DataSet` element per line, so this validates the required collection
/// envelope and each generated dataset record without accepting an empty or
/// unrelated XML document as a valid output artifact.
pub fn validate_pvd_file(path: impl AsRef<Path>) -> std::io::Result<usize> {
    let contents = std::fs::read_to_string(path)?;
    if !contents.contains("<VTKFile type=\"Collection\"")
        || !contents.contains("<Collection>")
        || !contents.contains("</Collection>")
        || !contents.contains("</VTKFile>")
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "PVD file has an invalid collection envelope",
        ));
    }

    let datasets = contents
        .lines()
        .filter(|line| line.trim_start().starts_with("<DataSet "))
        .collect::<Vec<_>>();
    if datasets.is_empty()
        || datasets
            .iter()
            .any(|line| !line.contains("timestep=\"") || !line.contains("file=\""))
    {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            "PVD file has no valid dataset records",
        ));
    }
    Ok(datasets.len())
}

/// Write a `.pvd` collection from `(timestep, vtu_filename)` pairs.
///
/// Filenames must be non-empty relative paths below the directory containing
/// the `.pvd`; timesteps must be finite and strictly increasing. XML-sensitive
/// filename characters are escaped before writing.
pub fn write_pvd(path: impl AsRef<Path>, timesteps: &[(f64, &str)]) -> std::io::Result<()> {
    validate_entries(timesteps)?;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "<?xml version=\"1.0\"?>")?;
    writeln!(
        f,
        "<VTKFile type=\"Collection\" version=\"0.1\" byte_order=\"LittleEndian\">"
    )?;
    writeln!(f, "  <Collection>")?;
    for (t, file) in timesteps {
        writeln!(
            f,
            "    <DataSet timestep=\"{t}\" file=\"{}\"/>",
            escape_xml_attribute(file)
        )?;
    }
    writeln!(f, "  </Collection>")?;
    writeln!(f, "</VTKFile>")?;
    Ok(())
}

fn validate_entries(timesteps: &[(f64, &str)]) -> std::io::Result<()> {
    if timesteps.is_empty() {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "PVD collection must contain at least one dataset",
        ));
    }
    let mut previous = None;
    for &(timestep, file) in timesteps {
        if !timestep.is_finite() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "PVD timestep must be finite",
            ));
        }
        if previous.is_some_and(|value| timestep <= value) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "PVD timesteps must be strictly increasing",
            ));
        }
        previous = Some(timestep);
        let path = Path::new(file);
        if file.is_empty()
            || file.contains('\0')
            || path.is_absolute()
            || path
                .components()
                .any(|component| matches!(component, std::path::Component::ParentDir))
        {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "PVD dataset filenames must be non-empty relative paths",
            ));
        }
    }
    Ok(())
}

fn escape_xml_attribute(value: &str) -> String {
    value
        .replace('&', "&amp;")
        .replace('"', "&quot;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('\'', "&apos;")
}

/// Builder that accumulates `(timestep, file)` entries and writes the `.pvd`
/// on [`finish`](PvdCollection::finish). Filenames are stored relative; the
/// caller is responsible for placing the matching `.vtu` next to the `.pvd`.
pub struct PvdCollection {
    path: PathBuf,
    entries: Vec<(f64, String)>,
}

impl PvdCollection {
    pub fn new(path: impl Into<PathBuf>) -> Self {
        Self {
            path: path.into(),
            entries: Vec::new(),
        }
    }

    /// Record a dataset entry. `file` should be the basename of the `.vtu`
    /// next to the `.pvd`.
    pub fn push(&mut self, timestep: f64, file: impl Into<String>) {
        self.entries.push((timestep, file.into()));
    }

    /// Checked variant of [`Self::push`] for callers that need immediate
    /// admission errors rather than waiting for [`Self::finish`].
    pub fn try_push(&mut self, timestep: f64, file: impl Into<String>) -> std::io::Result<()> {
        let file = file.into();
        let previous = self.entries.last().map(|(value, _)| *value);
        validate_entries(&[(timestep, file.as_str())])?;
        if previous.is_some_and(|value| timestep <= value) {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "PVD timesteps must be strictly increasing",
            ));
        }
        self.entries.push((timestep, file));
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn finish(self) -> std::io::Result<()> {
        let pairs: Vec<(f64, &str)> = self.entries.iter().map(|(t, f)| (*t, f.as_str())).collect();
        write_pvd(&self.path, &pairs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pvd_collection_writes_xml() {
        let tmp = tempfile::tempdir().unwrap();
        let pvd = tmp.path().join("series.pvd");
        let mut c = PvdCollection::new(&pvd);
        c.push(100.0, "frame_0.vtu");
        c.push(200.0, "frame_1.vtu");
        c.finish().unwrap();
        let s = std::fs::read_to_string(&pvd).unwrap();
        assert!(s.contains("Collection"));
        assert!(s.contains("frame_0.vtu"));
        assert!(s.contains("frame_1.vtu"));
        assert!(s.contains("timestep=\"100\""));
    }

    #[test]
    fn pvd_rejects_nonfinite_nonmonotone_and_traversal_entries() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("series.pvd");
        assert!(write_pvd(&path, &[(f64::NAN, "frame.vtu")]).is_err());
        assert!(write_pvd(&path, &[(2.0, "a.vtu"), (1.0, "b.vtu")]).is_err());
        assert!(write_pvd(&path, &[(1.0, "../outside.vtu")]).is_err());
        assert!(write_pvd(&path, &[(1.0, "/absolute.vtu")]).is_err());
        assert!(write_pvd(&path, &[]).is_err());
    }

    #[test]
    fn pvd_escapes_xml_filename_attributes_and_try_push_checks_order() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("series.pvd");
        let mut collection = PvdCollection::new(&path);
        collection.try_push(1.0, "nested/a&b.vtu").unwrap();
        assert!(collection.try_push(1.0, "same-time.vtu").is_err());
        collection.try_push(2.0, "nested/c\"d.vtu").unwrap();
        collection.finish().unwrap();
        let xml = std::fs::read_to_string(path).unwrap();
        assert!(xml.contains("a&amp;b.vtu"));
        assert!(xml.contains("c&quot;d.vtu"));
    }
}
