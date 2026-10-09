use crate::results::RunResult;
use std::path::Path;
use std::process::Command;

/// Map a sonium preconditioner name to the corresponding PETSc name.
fn petsc_precond_name(precond: &str) -> &str {
    match precond {
        "none" => "none",
        "jacobi" => "jacobi",
        "ilu" => "ilu",
        "ilut" => "ilu",
        "amg" => "amg",
        other => panic!("Unknown preconditioner for PETSc mapping: {other}"),
    }
}

/// Run a PETSc benchmark binary and parse its JSON output into a `RunResult`.
///
/// The PETSc binary is expected to:
/// 1. Accept command-line args: `<mtx_path> <solver> <precond> <tol> <max_iter>`
/// 2. Print a single JSON object to stdout with its timing/convergence fields
///    (`solver`, `precond`, `n`, `nnz`, `setup_ms`, `solve_ms`, `total_ms`,
///    `iterations`, `residual`, `converged`).
///
/// `matrix`/`scalar` identify the benchmark case (the C binary cannot know
/// them); they are injected alongside `engine: "petsc"` before the object is
/// deserialized into the wider `RunResult`.
///
/// Returns `None` if the binary is not found or fails to execute.
pub fn run_petsc(
    petsc_bin: &str,
    mtx_path: &Path,
    solver: &str,
    precond: &str,
    tolerance: f64,
    max_iter: usize,
    matrix: &str,
    scalar: &str,
) -> Option<RunResult> {
    let petsc_precond = petsc_precond_name(precond);

    let output = Command::new(petsc_bin)
        .arg(mtx_path.as_os_str())
        .arg(solver)
        .arg(petsc_precond)
        .arg(format!("{:.e}", tolerance))
        .arg(max_iter.to_string())
        .output();

    let output = match output {
        Ok(o) => o,
        Err(e) => {
            log::warn!("Failed to execute PETSc binary {petsc_bin}: {e}");
            return None;
        }
    };

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        log::warn!(
            "PETSc binary exited with status {}: {}",
            output.status,
            stderr.trim()
        );
        return None;
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    let mut value: serde_json::Value = match serde_json::from_str(stdout.trim()) {
        Ok(value) => value,
        Err(e) => {
            log::warn!("Failed to parse PETSc JSON output: {e}");
            log::debug!("PETSc stdout was: {stdout}");
            return None;
        }
    };
    let object = match value.as_object_mut() {
        Some(object) => object,
        None => {
            log::warn!("PETSc output is not a JSON object");
            return None;
        }
    };
    // Adapt the C binary's narrow schema to RunResult: rename `precond`,
    // inject the case context the binary cannot know.
    if let Some(precond_value) = object.remove("precond") {
        object.insert("preconditioner".to_string(), precond_value);
    }
    object.insert(
        "engine".to_string(),
        serde_json::Value::String("petsc".to_string()),
    );
    object.insert(
        "matrix".to_string(),
        serde_json::Value::String(matrix.to_string()),
    );
    object.insert(
        "scalar".to_string(),
        serde_json::Value::String(scalar.to_string()),
    );
    match serde_json::from_value::<RunResult>(value) {
        Ok(result) => Some(result),
        Err(e) => {
            log::warn!("Failed to adapt PETSc JSON to RunResult: {e}");
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_petsc_precond_mapping() {
        assert_eq!(petsc_precond_name("none"), "none");
        assert_eq!(petsc_precond_name("jacobi"), "jacobi");
        assert_eq!(petsc_precond_name("ilu"), "ilu");
        assert_eq!(petsc_precond_name("ilut"), "ilu");
        assert_eq!(petsc_precond_name("amg"), "amg");
    }

    #[test]
    #[should_panic(expected = "Unknown preconditioner")]
    fn test_petsc_precond_unknown() {
        petsc_precond_name("bogus");
    }

    #[test]
    fn test_run_petsc_missing_binary() {
        let result = run_petsc(
            "/nonexistent/petsc_bench",
            Path::new("/tmp/test.mtx"),
            "gmres",
            "none",
            1e-10,
            1000,
            "poisson_2d",
            "f64",
        );
        assert!(result.is_none());
    }
}
