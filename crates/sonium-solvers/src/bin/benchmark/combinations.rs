use std::fmt;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SolverKind {
    Gmres,
    Fgmres,
    BiCgstab,
    Cg,
    Idrs,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PrecondKind {
    None,
    Jacobi,
    Ilu,
    Ilut,
    Amg,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatrixKind {
    Poisson2D,
    AnisotropicDiffusion2D,
    SaddlePoint2D,
    Helmholtz2D,
    Laplacian3D,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScalarKind {
    F64,
    Complex64,
}

impl fmt::Display for SolverKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SolverKind::Gmres => write!(f, "gmres"),
            SolverKind::Fgmres => write!(f, "fgmres"),
            SolverKind::BiCgstab => write!(f, "bicgstab"),
            SolverKind::Cg => write!(f, "cg"),
            SolverKind::Idrs => write!(f, "idrs"),
        }
    }
}

impl fmt::Display for PrecondKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrecondKind::None => write!(f, "none"),
            PrecondKind::Jacobi => write!(f, "jacobi"),
            PrecondKind::Ilu => write!(f, "ilu"),
            PrecondKind::Ilut => write!(f, "ilut"),
            PrecondKind::Amg => write!(f, "amg"),
        }
    }
}

impl fmt::Display for MatrixKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixKind::Poisson2D => write!(f, "poisson_2d"),
            MatrixKind::AnisotropicDiffusion2D => write!(f, "anisotropic_diffusion_2d"),
            MatrixKind::SaddlePoint2D => write!(f, "saddle_point_2d"),
            MatrixKind::Helmholtz2D => write!(f, "helmholtz_2d"),
            MatrixKind::Laplacian3D => write!(f, "laplacian_3d"),
        }
    }
}

impl fmt::Display for ScalarKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ScalarKind::F64 => write!(f, "f64"),
            ScalarKind::Complex64 => write!(f, "complex64"),
        }
    }
}

#[derive(Debug, Clone)]
pub struct BenchmarkCase {
    pub solver: SolverKind,
    pub preconditioner: PrecondKind,
    pub matrix: MatrixKind,
    pub scalar: ScalarKind,
    pub sizes: Vec<usize>,
}

/// Return the full test matrix of ~60 benchmark combinations.
pub fn all_cases() -> Vec<BenchmarkCase> {
    let sizes_2d_full = vec![1_000, 10_000, 100_000, 1_000_000];
    let sizes_2d_small = vec![1_000, 10_000];
    let sizes_2d = vec![1_000, 10_000, 100_000];
    let sizes_3d = vec![1_000, 10_000, 100_000];

    let mut cases = Vec::new();

    // GMRES with none/jacobi on Poisson2D f64 — capped at 10k (too slow above)
    for precond in [PrecondKind::None, PrecondKind::Jacobi] {
        cases.push(BenchmarkCase {
            solver: SolverKind::Gmres,
            preconditioner: precond,
            matrix: MatrixKind::Poisson2D,
            scalar: ScalarKind::F64,
            sizes: sizes_2d_small.clone(),
        });
    }

    // GMRES with ilu/ilut/amg on Poisson2D f64
    for precond in [PrecondKind::Ilu, PrecondKind::Ilut, PrecondKind::Amg] {
        cases.push(BenchmarkCase {
            solver: SolverKind::Gmres,
            preconditioner: precond,
            matrix: MatrixKind::Poisson2D,
            scalar: ScalarKind::F64,
            sizes: sizes_2d_full.clone(),
        });
    }

    // GMRES with none/jacobi on Helmholtz2D Complex64 — capped at 10k
    for precond in [PrecondKind::None, PrecondKind::Jacobi] {
        cases.push(BenchmarkCase {
            solver: SolverKind::Gmres,
            preconditioner: precond,
            matrix: MatrixKind::Helmholtz2D,
            scalar: ScalarKind::Complex64,
            sizes: sizes_2d_small.clone(),
        });
    }

    // GMRES with ilu on Helmholtz2D Complex64
    cases.push(BenchmarkCase {
        solver: SolverKind::Gmres,
        preconditioner: PrecondKind::Ilu,
        matrix: MatrixKind::Helmholtz2D,
        scalar: ScalarKind::Complex64,
        sizes: sizes_2d.clone(),
    });

    // GMRES with ilu on a regularized saddle-point system
    cases.push(BenchmarkCase {
        solver: SolverKind::Gmres,
        preconditioner: PrecondKind::Ilu,
        matrix: MatrixKind::SaddlePoint2D,
        scalar: ScalarKind::F64,
        sizes: sizes_2d.clone(),
    });

    // FGMRES with amg on Poisson2D f64
    cases.push(BenchmarkCase {
        solver: SolverKind::Fgmres,
        preconditioner: PrecondKind::Amg,
        matrix: MatrixKind::Poisson2D,
        scalar: ScalarKind::F64,
        sizes: sizes_2d_full.clone(),
    });

    // BiCGSTAB with none on Poisson2D f64
    cases.push(BenchmarkCase {
        solver: SolverKind::BiCgstab,
        preconditioner: PrecondKind::None,
        matrix: MatrixKind::Poisson2D,
        scalar: ScalarKind::F64,
        sizes: sizes_2d.clone(),
    });

    // BiCGSTAB with none on Helmholtz2D Complex64
    cases.push(BenchmarkCase {
        solver: SolverKind::BiCgstab,
        preconditioner: PrecondKind::None,
        matrix: MatrixKind::Helmholtz2D,
        scalar: ScalarKind::Complex64,
        sizes: sizes_2d.clone(),
    });

    // CG with none/jacobi/amg on Poisson2D f64
    for precond in [PrecondKind::None, PrecondKind::Jacobi, PrecondKind::Amg] {
        cases.push(BenchmarkCase {
            solver: SolverKind::Cg,
            preconditioner: precond,
            matrix: MatrixKind::Poisson2D,
            scalar: ScalarKind::F64,
            sizes: sizes_2d_full.clone(),
        });
    }

    // CG with jacobi/amg on anisotropic diffusion f64
    for precond in [PrecondKind::Jacobi, PrecondKind::Amg] {
        cases.push(BenchmarkCase {
            solver: SolverKind::Cg,
            preconditioner: precond,
            matrix: MatrixKind::AnisotropicDiffusion2D,
            scalar: ScalarKind::F64,
            sizes: sizes_2d.clone(),
        });
    }

    // CG with none/jacobi/amg on Laplacian3D f64
    for precond in [PrecondKind::None, PrecondKind::Jacobi, PrecondKind::Amg] {
        cases.push(BenchmarkCase {
            solver: SolverKind::Cg,
            preconditioner: precond,
            matrix: MatrixKind::Laplacian3D,
            scalar: ScalarKind::F64,
            sizes: sizes_3d.clone(),
        });
    }

    // IDR(s) with none/jacobi on Poisson2D f64
    for precond in [PrecondKind::None, PrecondKind::Jacobi] {
        cases.push(BenchmarkCase {
            solver: SolverKind::Idrs,
            preconditioner: precond,
            matrix: MatrixKind::Poisson2D,
            scalar: ScalarKind::F64,
            sizes: sizes_2d.clone(),
        });
    }

    cases
}

/// Filter cases by a pattern matching "solver/precond/matrix".
///
/// Each segment is matched as a substring against the Display form of the
/// corresponding enum. An empty segment or "*" matches everything.
/// Examples: "gmres//", "cg/amg/poisson", "/jacobi/", "*/*/laplacian_3d"
pub fn filter_cases(cases: &[BenchmarkCase], filter: &str) -> Vec<BenchmarkCase> {
    let parts: Vec<&str> = filter.splitn(3, '/').collect();
    let solver_pat = parts.first().copied().unwrap_or("");
    let precond_pat = parts.get(1).copied().unwrap_or("");
    let matrix_pat = parts.get(2).copied().unwrap_or("");

    cases
        .iter()
        .filter(|c| {
            matches_segment(solver_pat, &c.solver.to_string())
                && matches_segment(precond_pat, &c.preconditioner.to_string())
                && matches_segment(matrix_pat, &c.matrix.to_string())
        })
        .cloned()
        .collect()
}

fn matches_segment(pattern: &str, value: &str) -> bool {
    pattern.is_empty() || pattern == "*" || value.contains(pattern)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_cases_count() {
        let cases = all_cases();
        // GMRES none/jacobi capped at 10k (2 sizes), rest unchanged
        // 2*2+3*4 + 2*2+1*3 + 1*4 + 1*3+1*3 + 3*4+3*3 + 2*3 = 4+12+4+3+4+3+3+12+9+6 = 60
        let total: usize = cases.iter().map(|c| c.sizes.len()).sum();
        assert!(total >= 60, "Expected at least 60 combos, got {total}");
    }

    #[test]
    fn test_filter_by_solver() {
        let cases = all_cases();
        // "gmres" should match Gmres and Fgmres (both contain "gmres")
        let filtered = filter_cases(&cases, "gmres//");
        assert!(
            filtered
                .iter()
                .all(|c| c.solver == SolverKind::Gmres || c.solver == SolverKind::Fgmres)
        );
        assert!(!filtered.is_empty());

        // "bicgstab" should only match BiCgstab
        let filtered = filter_cases(&cases, "bicgstab//");
        assert!(filtered.iter().all(|c| c.solver == SolverKind::BiCgstab));
        assert!(!filtered.is_empty());
    }

    #[test]
    fn test_filter_by_precond() {
        let cases = all_cases();
        let filtered = filter_cases(&cases, "*/amg/");
        assert!(
            filtered
                .iter()
                .all(|c| c.preconditioner == PrecondKind::Amg)
        );
        assert!(!filtered.is_empty());

        let filtered = filter_cases(&cases, "gmres/ilu/saddle");
        assert!(
            filtered
                .iter()
                .all(|c| c.matrix == MatrixKind::SaddlePoint2D)
        );
        assert!(!filtered.is_empty());
    }

    #[test]
    fn test_filter_by_matrix() {
        let cases = all_cases();
        let filtered = filter_cases(&cases, "*/*/laplacian_3d");
        assert!(filtered.iter().all(|c| c.matrix == MatrixKind::Laplacian3D));
        assert!(!filtered.is_empty());

        let filtered = filter_cases(&cases, "*/*/anisotropic");
        assert!(
            filtered
                .iter()
                .all(|c| c.matrix == MatrixKind::AnisotropicDiffusion2D)
        );
        assert!(!filtered.is_empty());
    }

    #[test]
    fn test_display_impls() {
        assert_eq!(SolverKind::Gmres.to_string(), "gmres");
        assert_eq!(SolverKind::Fgmres.to_string(), "fgmres");
        assert_eq!(
            MatrixKind::AnisotropicDiffusion2D.to_string(),
            "anisotropic_diffusion_2d"
        );
        assert_eq!(MatrixKind::SaddlePoint2D.to_string(), "saddle_point_2d");
        assert_eq!(SolverKind::BiCgstab.to_string(), "bicgstab");
        assert_eq!(SolverKind::Cg.to_string(), "cg");
        assert_eq!(SolverKind::Idrs.to_string(), "idrs");
        assert_eq!(PrecondKind::None.to_string(), "none");
        assert_eq!(PrecondKind::Jacobi.to_string(), "jacobi");
        assert_eq!(PrecondKind::Ilu.to_string(), "ilu");
        assert_eq!(PrecondKind::Ilut.to_string(), "ilut");
        assert_eq!(PrecondKind::Amg.to_string(), "amg");
        assert_eq!(MatrixKind::Poisson2D.to_string(), "poisson_2d");
        assert_eq!(MatrixKind::Helmholtz2D.to_string(), "helmholtz_2d");
        assert_eq!(MatrixKind::Laplacian3D.to_string(), "laplacian_3d");
        assert_eq!(ScalarKind::F64.to_string(), "f64");
        assert_eq!(ScalarKind::Complex64.to_string(), "complex64");
    }
}
