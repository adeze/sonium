//! Output directory resolution for sonium binaries.
//!
//! Layout: `<root>/<crate>/<binary>/`. The root is `$SONIUM_OUTPUT_DIR` if
//! set, otherwise `ProjectDirs::cache_dir().join("out")`.

use directories::ProjectDirs;
use std::io;
use std::path::PathBuf;

/// Resolve the root output directory for sonium artifacts.
///
/// Panics when neither `SONIUM_OUTPUT_DIR` nor the platform project
/// directories resolve; prefer [`try_sonium_output_root`] in library code.
pub fn sonium_output_root() -> PathBuf {
    try_sonium_output_root().expect("ProjectDirs unavailable; set SONIUM_OUTPUT_DIR")
}

/// Fallible [`sonium_output_root`] for library callers: a missing
/// `ProjectDirs` entry becomes an `io::Error`, never a process abort.
pub fn try_sonium_output_root() -> io::Result<PathBuf> {
    if let Ok(custom) = std::env::var("SONIUM_OUTPUT_DIR") {
        return Ok(PathBuf::from(custom));
    }
    let proj = ProjectDirs::from("org", "sonium", "sonium").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            "ProjectDirs unavailable; set SONIUM_OUTPUT_DIR",
        )
    })?;
    Ok(proj.cache_dir().join("out"))
}

/// Per-binary output directory: `<root>/<crate>/<binary>/`.
///
/// The directory is created (recursively) before being returned. Panics on
/// I/O failure; prefer [`try_binary_output_dir`] in library code.
pub fn binary_output_dir(crate_name: &str, binary_name: &str) -> PathBuf {
    try_binary_output_dir(crate_name, binary_name).expect("failed to create binary output dir")
}

/// Fallible [`binary_output_dir`] for library callers.
pub fn try_binary_output_dir(crate_name: &str, binary_name: &str) -> io::Result<PathBuf> {
    let dir = try_sonium_output_root()?.join(crate_name).join(binary_name);
    std::fs::create_dir_all(&dir)?;
    Ok(dir)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn env_override_takes_precedence() {
        let tmp = tempfile::tempdir().unwrap();
        // SAFETY: single-threaded test process for env mutation.
        unsafe {
            std::env::set_var("SONIUM_OUTPUT_DIR", tmp.path());
        }
        let root = sonium_output_root();
        unsafe {
            std::env::remove_var("SONIUM_OUTPUT_DIR");
        }
        assert_eq!(root, tmp.path());
    }

    #[test]
    fn binary_dir_is_created() {
        let tmp = tempfile::tempdir().unwrap();
        unsafe {
            std::env::set_var("SONIUM_OUTPUT_DIR", tmp.path());
        }
        let dir = binary_output_dir("sonium-xem", "test_binary");
        unsafe {
            std::env::remove_var("SONIUM_OUTPUT_DIR");
        }
        assert!(dir.is_dir());
        assert!(dir.ends_with("sonium-xem/test_binary"));
    }
}
