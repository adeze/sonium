use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkReport {
    pub schema_version: u32,
    pub timestamp: String,
    pub platform: PlatformInfo,
    pub sonium_version: String,
    pub git_sha: Option<String>,
    pub petsc_available: bool,
    pub runs: Vec<RunResult>,
    pub validation: ValidationSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlatformInfo {
    pub name: String,
    pub os: String,
    pub arch: String,
    pub cpu_model: String,
    pub total_cores: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunResult {
    /// "sonium" or "petsc"
    pub engine: String,
    pub solver: String,
    pub preconditioner: String,
    pub matrix: String,
    /// "f64" or "complex64"
    pub scalar: String,
    pub n: usize,
    pub nnz: usize,
    #[serde(default = "default_rhs_count")]
    pub rhs_count: usize,
    pub setup_ms: f64,
    pub solve_ms: f64,
    pub total_ms: f64,
    pub iterations: usize,
    pub residual: f64,
    pub converged: bool,
    #[serde(default)]
    pub bytes_per_spmv: Option<u64>,
    #[serde(default)]
    pub bytes_per_nnz: Option<f64>,
    #[serde(default)]
    pub estimated_spmv_gb_s: Option<f64>,
    #[serde(default)]
    pub setup_fraction: Option<f64>,
    #[serde(default)]
    pub time_limited_by: Option<String>,
    #[serde(default)]
    pub mean_solve_ms_per_rhs: Option<f64>,
    #[serde(default)]
    pub mean_iterations_per_rhs: Option<f64>,
    #[serde(default)]
    pub setup_reuse_scope: Option<String>,
    #[serde(default)]
    pub setup_reuse_break_even_rhs: Option<f64>,
    #[serde(default)]
    pub amortized_total_ms_per_rhs_4: Option<f64>,
    #[serde(default)]
    pub amortized_total_ms_per_rhs_16: Option<f64>,
    #[serde(default)]
    pub krylov_spmv_count: Option<usize>,
    #[serde(default)]
    pub krylov_preconditioner_count: Option<usize>,
    #[serde(default)]
    pub krylov_dot_reductions: Option<usize>,
    #[serde(default)]
    pub krylov_norm_count: Option<usize>,
    #[serde(default)]
    pub krylov_axpy_passes: Option<usize>,
    #[serde(default)]
    pub krylov_basis_bytes_touched: Option<u64>,
    #[serde(default)]
    pub krylov_synchronization_points: Option<usize>,
    #[serde(default)]
    pub krylov_exact_residual_checks: Option<usize>,
}

fn default_rhs_count() -> usize {
    1
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationSummary {
    pub total_comparisons: usize,
    pub passed: usize,
    pub failed: usize,
    pub warnings: usize,
    pub failures: Vec<ValidationFailure>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationFailure {
    pub solver: String,
    pub preconditioner: String,
    pub matrix: String,
    pub scalar: String,
    pub n: usize,
    pub reason: String,
    pub sonium_residual: f64,
    pub petsc_residual: f64,
    pub sonium_iterations: usize,
    pub petsc_iterations: usize,
}
