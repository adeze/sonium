//! Reference iteration-count checks against external solver literature.
//!
//! These tests encode well-established iteration counts from the published
//! literature and from PETSc/hypre documentation for three standard problem
//! classes, then assert that `sonium-solvers` converges within a documented
//! factor of those reference counts on an equivalent setup (same problem
//! class, comparable size bracket, same relative tolerance).
//!
//! PETSc and hypre are *not* required to run these tests: the reference
//! numbers are recorded as data with citations. The live PETSc comparison
//! harness lives in `src/bin/benchmark/petsc.rs`; this file is the hermetic
//! counterpart that can run everywhere.
//!
//! References
//! ----------
//! [1] V. E. Henson and U. M. Yang, "BoomerAMG: A parallel algebraic
//!     multigrid solver and preconditioner", Applied Numerical Mathematics
//!     41(1):155-177, 2002. doi:10.1016/S0168-9274(01)00115-5
//!     hypre BoomerAMG docs: https://hypre.readthedocs.io/en/latest/solvers-boomeramg.html
//!     PETSc PCHYPRE docs: https://petsc.org/release/docs/manualpages/PC/PCHYPRE.html
//!     Reported behaviour: CG + BoomerAMG on 3D Poisson (7-point stencil)
//!     converges in ~10-20 iterations to relative tolerance 1e-6,
//!     essentially independent of problem size.
//!
//! [2] H. C. Elman, D. J. Silvester, A. J. Wathen, "Finite Elements and
//!     Fast Iterative Solvers", 2nd ed., Oxford University Press, 2014.
//!     doi:10.1093/acprof:oso/9780199678792.001.0001
//!     GMRES + ILU(0) on 2D convection-diffusion (mildly nonsymmetric,
//!     central differences) converges in tens of iterations to 1e-6;
//!     ~20-60 iterations is the typical bracket for grids in the 1k-10k
//!     DOF range at moderate Peclet number.
//!
//! [3] Y. A. Erlangga, C. Vuik, C. W. Oosterlee, "On a class of
//!     preconditioners for solving the Helmholtz equation", Applied
//!     Numerical Mathematics 50(3-4):409-425, 2004.
//!     doi:10.1016/j.apnum.2004.01.009
//!     and Y. A. Erlangga, C. W. Oosterlee, C. Vuik, "A novel multigrid
//!     based preconditioner for heterogeneous Helmholtz problems", SIAM
//!     J. Sci. Comput. 27(4):1471-1492, 2006. doi:10.1137/040615195
//!     Shifted-Laplacian preconditioned GMRES on 2D Helmholtz converges in
//!     ~10-30 iterations to 1e-6 for small-to-moderate wavenumbers
//!     (k^2 h << 1 regime), growing roughly linearly with k.

use num_complex::Complex64;
use sonium_solvers_iterative::{CgConfig, GmresConfig, gmres, gmres_preconditioned, pcg};
use sonium_solvers_preconditioners::{AmgConfig, AmgPreconditioner, IluPreconditioner};
use sonium_solvers_sparse::CsrMatrix;

/// Factor by which sonium-solvers may exceed the top of a reference
/// iteration bracket before we call it a genuine gap. 2x accounts for
/// differences in coarsening details, smoothing counts, and grid size
/// within the bracket; anything beyond that warrants investigation
/// (see the module docs and each test's comments).
const REFERENCE_SLACK: f64 = 2.0;

/// Assert `iters` is at most `REFERENCE_SLACK` times `reference_hi`, with a
/// diagnostic message naming the case and the reference bracket.
fn assert_within_reference(case: &str, iters: usize, reference_hi: usize) {
    let bound = (REFERENCE_SLACK * reference_hi as f64).ceil() as usize;
    eprintln!("[reference] {case}: {iters} iters (bracket top {reference_hi}, bound {bound})");
    assert!(
        iters <= bound,
        "{case}: {iters} iterations exceeds {REFERENCE_SLACK}x the reference \
         bracket top ({reference_hi}) -> bound {bound}. Investigate whether \
         this is a solver gap or a test-construction artifact before \
         loosening the bound."
    );
}

// ---------------------------------------------------------------------------
// Matrix generators (kept local: the benchmark-binary generators in
// src/bin/benchmark/matrices.rs are not part of the library API).
// ---------------------------------------------------------------------------

/// 3D Poisson 7-point stencil on an n^3 grid (SPD). Mirrors
/// `benchmark/matrices.rs::laplacian_3d`.
fn poisson_3d(n: usize) -> CsrMatrix<f64> {
    let nn = n * n * n;
    let mut triplets = Vec::with_capacity(7 * nn);
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                let row = i * n * n + j * n + k;
                triplets.push((row, row, 6.0));
                if i > 0 {
                    triplets.push((row, (i - 1) * n * n + j * n + k, -1.0));
                }
                if i + 1 < n {
                    triplets.push((row, (i + 1) * n * n + j * n + k, -1.0));
                }
                if j > 0 {
                    triplets.push((row, i * n * n + (j - 1) * n + k, -1.0));
                }
                if j + 1 < n {
                    triplets.push((row, i * n * n + (j + 1) * n + k, -1.0));
                }
                if k > 0 {
                    triplets.push((row, i * n * n + j * n + k - 1, -1.0));
                }
                if k + 1 < n {
                    triplets.push((row, i * n * n + j * n + k + 1, -1.0));
                }
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, triplets)
}

/// 2D convection-diffusion on an n*n grid, central differences:
/// `-epsilon * Delta u + b . grad u = f` with b = (1, 0). Nonsymmetric for
/// `wind / epsilon > 0`; mildly so at the default parameters.
fn convection_diffusion_2d(n: usize, epsilon: f64, wind: f64) -> CsrMatrix<f64> {
    let nn = n * n;
    // Mesh spacing on the unit square; central differences for both terms.
    let h = 1.0 / (n as f64 + 1.0);
    let diff = epsilon / (h * h);
    let conv = wind / (2.0 * h);
    let mut triplets = Vec::with_capacity(5 * nn);
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            triplets.push((row, row, 4.0 * diff));
            if i > 0 {
                triplets.push((row, (i - 1) * n + j, -diff));
            }
            if i + 1 < n {
                triplets.push((row, (i + 1) * n + j, -diff));
            }
            // x-direction convection, central: -conv * u_{j-1} + conv * u_{j+1}
            if j > 0 {
                triplets.push((row, i * n + j - 1, -diff - conv));
            }
            if j + 1 < n {
                triplets.push((row, i * n + j + 1, -diff + conv));
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, triplets)
}

/// 2D Helmholtz `-Delta u - k^2 u` on an n*n grid (indefinite, complex-typed
/// so it can be paired with a complex shifted-Laplacian preconditioner).
/// Mirrors `benchmark/matrices.rs::helmholtz_2d` but without the imaginary
/// stabilization term — the shifted-Laplacian preconditioner supplies the
/// damping instead.
fn helmholtz_2d(n: usize, k: f64) -> CsrMatrix<Complex64> {
    let nn = n * n;
    let h = 1.0 / (n as f64 + 1.0);
    let k2h2 = k * k * h * h;
    let mut triplets: Vec<(usize, usize, Complex64)> = Vec::with_capacity(5 * nn);
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            triplets.push((row, row, Complex64::new(4.0 - k2h2, 0.0)));
            if i > 0 {
                triplets.push((row, (i - 1) * n + j, Complex64::new(-1.0, 0.0)));
            }
            if i + 1 < n {
                triplets.push((row, (i + 1) * n + j, Complex64::new(-1.0, 0.0)));
            }
            if j > 0 {
                triplets.push((row, i * n + j - 1, Complex64::new(-1.0, 0.0)));
            }
            if j + 1 < n {
                triplets.push((row, i * n + j + 1, Complex64::new(-1.0, 0.0)));
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, triplets)
}

/// Erlangga shifted Laplacian `M = -Delta u - (1 + i*beta) k^2 u`, the
/// classic wave-ray / shifted-Laplacian Helmholtz preconditioner [3].
/// beta = 0.5 is the standard choice from [3] (2004 paper, Table 1).
fn shifted_laplacian_2d(n: usize, k: f64, beta: f64) -> CsrMatrix<Complex64> {
    let nn = n * n;
    let h = 1.0 / (n as f64 + 1.0);
    let k2h2 = k * k * h * h;
    let mut triplets: Vec<(usize, usize, Complex64)> = Vec::with_capacity(5 * nn);
    for i in 0..n {
        for j in 0..n {
            let row = i * n + j;
            triplets.push((row, row, Complex64::new(4.0 - k2h2, -beta * k2h2)));
            if i > 0 {
                triplets.push((row, (i - 1) * n + j, Complex64::new(-1.0, 0.0)));
            }
            if i + 1 < n {
                triplets.push((row, (i + 1) * n + j, Complex64::new(-1.0, 0.0)));
            }
            if j > 0 {
                triplets.push((row, i * n + j - 1, Complex64::new(-1.0, 0.0)));
            }
            if j + 1 < n {
                triplets.push((row, i * n + j + 1, Complex64::new(-1.0, 0.0)));
            }
        }
    }
    CsrMatrix::from_triplets(nn, nn, triplets)
}

// ---------------------------------------------------------------------------
// Reference checks
// ---------------------------------------------------------------------------

/// CG + AMG on 3D Poisson 7-pt vs hypre BoomerAMG reference [1].
///
/// Reference bracket: 10-20 CG iterations to rtol 1e-6 for CG+BoomerAMG on
/// 3D 7-point Poisson, weakly size-independent. Grid: 12^3 = 1728 DOFs,
/// inside the size bracket where the reference was measured (1k-100k DOFs).
#[test]
fn cg_amg_poisson_3d_matches_boomeramg_reference() {
    let n = 12;
    let matrix = poisson_3d(n);
    let b = vec![1.0; n * n * n];

    let amg = AmgPreconditioner::from_csr(&matrix, AmgConfig::default());
    let config = CgConfig {
        max_iterations: 500,
        tolerance: 1e-6,
        print_interval: 0,
    };
    let sol = pcg(&matrix, &amg, &b, &config);

    assert!(sol.converged, "CG+AMG did not converge on 3D Poisson");
    assert_within_reference("CG+AMG on 3D Poisson 7-pt", sol.iterations, 20);
}

/// CG + AMG with the BoomerAMG-mirror configuration
/// (`AmgConfig::for_hypre_boomeramg_comparison`: Falgout coarsening,
/// modified-classical interpolation, CF-ordered SGS) on the same 3D
/// Poisson problem vs the same hypre reference [1].
///
/// Same bracket (10-20 iterations) and bound convention as the default
/// config test above; guards the vendor-comparison configuration against
/// regressions in coarsening, interpolation normalization, and smoother
/// ordering.
#[test]
fn cg_amg_hypre_mirror_poisson_3d_matches_reference() {
    let n = 12;
    let matrix = poisson_3d(n);
    let b = vec![1.0; n * n * n];

    let amg = AmgPreconditioner::from_csr(&matrix, AmgConfig::for_hypre_boomeramg_comparison());
    assert!(amg.complexity_within_targets());
    let config = CgConfig {
        max_iterations: 500,
        tolerance: 1e-6,
        print_interval: 0,
    };
    let sol = pcg(&matrix, &amg, &b, &config);

    assert!(
        sol.converged,
        "CG+AMG (mirror) did not converge on 3D Poisson"
    );
    assert_within_reference("CG+AMG mirror on 3D Poisson 7-pt", sol.iterations, 20);
}

/// GMRES + ILU(0) on 2D convection-diffusion vs reference [2].
///
/// Reference bracket: 20-60 GMRES iterations to rtol 1e-6 for ILU(0)-
/// preconditioned GMRES on moderately nonsymmetric 2D convection-diffusion
/// in the 1k-10k DOF bracket. Grid: 48x48 = 2304 DOFs, grid Peclet number
/// (wind * h / epsilon = 2) kept moderate so the reference bracket applies.
#[test]
fn gmres_ilu0_convection_diffusion_matches_reference() {
    let n = 48;
    // wind * h / epsilon = 1 * (1/49) / 0.01 ~= 2: convection-dominated but
    // still in the regime where ILU(0)+GMRES behaves per [2].
    let matrix = convection_diffusion_2d(n, 0.01, 1.0);
    let b = vec![1.0; n * n];

    let ilu = IluPreconditioner::from_csr(&matrix);
    let config = GmresConfig {
        max_iterations: 20,
        restart: 100,
        tolerance: 1e-6,
        ..GmresConfig::default()
    };
    let sol = gmres_preconditioned(&matrix, &ilu, &b, &config);

    assert!(
        sol.converged,
        "GMRES+ILU(0) did not converge on 2D convection-diffusion"
    );
    assert_within_reference(
        "GMRES+ILU(0) on 2D convection-diffusion",
        sol.iterations,
        60,
    );
}

/// GMRES on 2D Helmholtz with shifted-Laplacian AMG preconditioner vs
/// reference [3].
///
/// Reference bracket: 10-30 GMRES iterations to rtol 1e-6 for
/// shifted-Laplacian-preconditioned GMRES at small-to-moderate k. Grid:
/// 32x32 = 1024 DOFs, k chosen so k^2 h^2 ~ 0.05 (small-k regime where the
/// bracket holds; counts grow ~linearly in k beyond it).
#[test]
fn gmres_shifted_laplacian_helmholtz_matches_reference() {
    let n = 32;
    let h = 1.0 / (n as f64 + 1.0);
    // k^2 h^2 = 0.05 -> k ~ 7.4: a few wavelengths across the domain.
    let k = (0.05f64).sqrt() / h;
    let helmholtz = helmholtz_2d(n, k);
    // Standard Erlangga shift beta = 0.5 [3].
    let shifted = shifted_laplacian_2d(n, k, 0.5);
    let b = vec![Complex64::new(1.0, 0.0); n * n];

    let amg = AmgPreconditioner::from_csr(&shifted, AmgConfig::for_helmholtz_shifted_laplacian());
    let config = GmresConfig {
        max_iterations: 20,
        restart: 100,
        tolerance: 1e-6,
        ..GmresConfig::default()
    };
    let sol = gmres_preconditioned(&helmholtz, &amg, &b, &config);

    assert!(
        sol.converged,
        "GMRES with shifted-Laplacian AMG did not converge on 2D Helmholtz \
         (k = {k:.2}, {} DOFs)",
        n * n
    );
    assert_within_reference(
        "GMRES + shifted-Laplacian AMG on 2D Helmholtz",
        sol.iterations,
        30,
    );
}

/// Sanity anchor: unpreconditioned GMRES on the same Helmholtz problem must
/// be dramatically worse than the preconditioned one, confirming the
/// preconditioner (not problem size) is what keeps counts in the reference
/// bracket. Not a reference check — a test-construction guard.
#[test]
fn helmholtz_unpreconditioned_is_much_worse() {
    let n = 32;
    let h = 1.0 / (n as f64 + 1.0);
    let k = (0.05f64).sqrt() / h;
    let helmholtz = helmholtz_2d(n, k);
    let shifted = shifted_laplacian_2d(n, k, 0.5);
    let b = vec![Complex64::new(1.0, 0.0); n * n];

    let config = GmresConfig {
        max_iterations: 20,
        restart: 100,
        tolerance: 1e-6,
        ..GmresConfig::default()
    };
    let bare = gmres(&helmholtz, &b, &config);
    let amg = AmgPreconditioner::from_csr(&shifted, AmgConfig::for_helmholtz_shifted_laplacian());
    let prec = gmres_preconditioned(&helmholtz, &amg, &b, &config);

    assert!(prec.converged);
    assert!(
        !bare.converged || bare.iterations > 3 * prec.iterations.max(1),
        "preconditioning had no effect (bare {} vs preconditioned {} iters) — \
         the Helmholtz test above would not be exercising the preconditioner",
        bare.iterations,
        prec.iterations
    );
}
