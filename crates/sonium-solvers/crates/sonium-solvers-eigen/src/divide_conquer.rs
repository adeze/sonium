//! Divide-and-conquer symmetric eigensolver (Cuppen's method, the LAPACK
//! `dstedc` route).
//!
//! `symmetric_eigen` in [`crate::lobpcg`] dispatches here for `n >=
//! DC_THRESHOLD`. Below that size the scalar Householder + implicit-shift QL
//! path is already faster than vendor LAPACK, so it stays untouched.
//!
//! # Algorithm
//!
//! 1. The dispatcher tridiagonalizes with the shared Householder kernel, then
//!    [`tridiag_dc_eigen`] solves the tridiagonal problem and the dispatcher
//!    back-transforms with one tuned GEMM.
//! 2. `tridiag_dc_eigen` recurses: split `T` at `m = n/2`, tear the coupling
//!    `β = f[m-1]` into a rank-1 update `T = diag(T1', T2') + β u uᵀ` with `u`
//!    supported on the two interface positions, and solve the halves
//!    recursively (leaves of `DC_LEAF` or fewer go through the existing
//!    implicit-shift QL directly on their tridiagonal segment).
//! 3. The merge diagonalizes the halves (`Q1 D1 Q1ᵀ`, `Q2 D2 Q2ᵀ`), forms the
//!    rank-1 problem `D + ρ z zᵀ` with `ρ = β` and `z` from the interface
//!    rows of `Q1`/`Q2`, sorts, deflates (Gu–Eisenstat: tiny `z` components
//!    and close eigenvalue pairs via Givens rotations), solves the reduced
//!    secular equation with a Newton/bisection-safe hybrid, and rebuilds the
//!    node eigenvectors with the recomputed-`ρ̂` trick (LAPACK `dlaed3`) so
//!    the computed vectors stay orthogonal to machine precision.
//!
//! The merge is the delicate part; the unit tests below check orthogonality,
//! residuals, and agreement with the QL path on random, clustered, repeated,
//! and Wilkinson-type spectra.

use rayon::prelude::*;
use sonium_solvers_core::dense::DenseMatrix;
use sonium_solvers_core::traits::{SolverError, SolverStatus};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

/// Secular-evaluation census for the vendor-parity work: while
/// [`symmetric_eigen_dc`] profiles (`SONIUM_EIGEN_PROFILE`), every secular
/// function evaluation bumps [`SEC_EVALS`] and each merge node reports its
/// share. Gated on one atomic load per call plus a predictable branch per
/// evaluation — unmeasurable when off. Since the halve-join overlaps
/// subtrees, concurrently merging nodes share these counters: per-node
/// eval/histogram attribution is approximate under concurrency (the
/// wall-time phase splits stay exact); totals across a solve stay exact.
static SEC_PROFILE: AtomicBool = AtomicBool::new(false);
static SEC_EVALS: AtomicU64 = AtomicU64::new(0);

/// One secular sweep (O(k) work unit, same as one legacy `secular(λ)`
/// evaluation) for the census above.
pub(crate) fn secular_sweep_bump() {
    if SEC_PROFILE.load(Ordering::Relaxed) {
        SEC_EVALS.fetch_add(1, Ordering::Relaxed);
        SEC_TL.with(|c| c.set(c.get() + 1));
    }
}

/// Per-root evaluation histogram (value solves vs vector-refinement solves)
/// plus the offset-path trigger count. Rounds through a thread-local
/// accumulator — rayon never migrates a running task, so the task's end
/// delta is exactly its own evaluations even under parallel merges — then
/// folded into the shared bins. Iteration counts are bitwise-deterministic
/// for a fixed matrix, so this census is exact under any host load.
use std::cell::Cell;
thread_local! {
    static SEC_TL: Cell<u64> = const { Cell::new(0) };
}
/// Bins 0..=127 plus a 128 overflow bin.
static SEC_HIST_SOLVE: [AtomicU64; 129] = [const { AtomicU64::new(0) }; 129];
static SEC_HIST_VEC: [AtomicU64; 129] = [const { AtomicU64::new(0) }; 129];
static SEC_TRIG: AtomicU64 = AtomicU64::new(0);

/// Leaves of this size or smaller go straight through implicit-shift QL.
/// 96 (up from 64): at n = 300 the four serial 75-merges cost more than
/// fatter QL leaves (18.1 vs 18.3–18.9 ms medians, tests green); the
/// trade was re-confirmed after the rational-interpolation secular
/// solver landed (cheaper merges) — 96 still wins on arithmetic
/// (4 × QL-75 ≈ 1.3 ms vs 8 × QL-37 + 4 × 75-merge ≈ 1.5–1.9 ms).
const DC_LEAF: usize = 96;

/// Dispatcher cutover: `symmetric_eigen` uses D&C at or above this size.
/// 128 was tried: slower below n=150 (the scalar QL path is unblocked)
/// and it broke D&C coverage down to 64, so 64 stands.
pub(crate) const DC_THRESHOLD: usize = 64;

/// Machine epsilon for f64 deflation tolerances.
const EPS: f64 = f64::EPSILON;

/// Threading cutover: secular per-root solves spread over rayon at or above
/// this reduced size. Per-node census (SONIUM_EIGEN_PROFILE) shows small
/// merges anti-scale — k=75 runs ~1.7x faster serial (pool dispatch and
/// contention exceed the per-root work) — while the k=300 root needs
/// parallel (7.2ms serial vs ~1ms). 128 keeps the clear winners on each
/// side; the ordered collect is bitwise-identical either way, so this
/// changes no computed value, only the schedule.
const PAR_ROOT_MIN: usize = 128;

/// Grain size (minimum items per rayon task) for the per-root collects
/// above. One task per root drowns in dispatch + work-stealing churn
/// when the pool is wide (measured 4–5× merge-node slowdown at 128
/// threads vs 16 for k=150, where each root is ~10 µs of div-bound
/// work). `k / (4·pool)` keeps ~4× oversubscription for load balance
/// with a 4-item floor amortizing task overhead; large problems still
/// fan out (grain grows sublinearly). Ordered collects stay
/// bitwise-identical, so this changes only the schedule.
fn par_grain(k: usize) -> usize {
    let pool = rayon::current_num_threads().max(1);
    (k / (4 * pool)).max(4)
}

/// Capped worker pool for the D&C solve below. Millisecond-scale kernels
/// (per-root secular solves, small GEMM panels, DSYMV columns) drown in
/// dispatch + work-stealing churn on wide pools: n=300 measured 19.3 ms
/// on a 128-thread pool vs 15.5 ms at 16 threads (same binary, only
/// `RAYON_NUM_THREADS` changed). The cap is `min(16, hardware, global
/// pool)`: never more than the host/user chose globally, never more
/// than hardware, never more than the measured saturation knee. Large
/// problems (`n > EIGEN_POOL_MAX_N`) stay on the caller's pool so
/// compute-bound giant GEMMs keep all cores; that regime is unverified
/// and keeps the status quo ante.
const EIGEN_POOL_MAX_THREADS: usize = 16;
const EIGEN_POOL_MAX_N: usize = 2048;
static EIGEN_POOL: std::sync::OnceLock<rayon::ThreadPool> = std::sync::OnceLock::new();

fn with_eigen_pool<R: Send>(n: usize, f: impl FnOnce() -> R + Send) -> R {
    if n > EIGEN_POOL_MAX_N {
        return f();
    }
    let avail = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(8);
    let cap = EIGEN_POOL_MAX_THREADS
        .min(avail)
        .min(rayon::current_num_threads().max(1));
    if cap >= rayon::current_num_threads() {
        // Already narrow enough (laptop pools, explicit user caps).
        return f();
    }
    let pool = EIGEN_POOL.get_or_init(|| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(cap)
            .thread_name(|i| format!("sonium-eigen-{i}"))
            .build()
            .expect("eigen thread pool builds")
    });
    // The pool is fixed at first init; if the process pool changed since
    // (tests fiddling with threads), a stale-wider pool only costs the
    // churn this was built to avoid — still correct either way.
    pool.install(f)
}

/// Entry point used by [`crate::lobpcg::symmetric_eigen`]: solve the dense
/// symmetric problem via tridiagonal D&C plus one GEMM back-transform.
///
/// `a` must be square `n × n` and finite (checked by the caller).
pub(crate) fn symmetric_eigen_dc(
    a: &DenseMatrix<f64>,
    n: usize,
) -> Result<(Vec<f64>, DenseMatrix<f64>), SolverError> {
    with_eigen_pool(n, || symmetric_eigen_dc_inner(a, n))
}

fn symmetric_eigen_dc_inner(
    a: &DenseMatrix<f64>,
    n: usize,
) -> Result<(Vec<f64>, DenseMatrix<f64>), SolverError> {
    debug_assert_eq!(a.nrows(), n);
    debug_assert_eq!(a.ncols(), n);
    // Phase profiler for the vendor-parity work: set SONIUM_EIGEN_PROFILE
    // to dump per-phase milliseconds to stderr. One env lookup per solve;
    // zero measurable overhead when unset.
    let profile = std::env::var_os("SONIUM_EIGEN_PROFILE").is_some();
    // Tridiagonalize with the shared kernel: on exit `t` holds Q, `d` the
    // diagonal, `e` the subdiagonal with `e[0] = 0`. Above one panel width
    // the blocked (GEMM) reduction wins over the scalar kernel.
    let mut t = a.as_slice().to_vec();
    let mut d = vec![0.0; n];
    let mut e = vec![0.0; n];
    let t_ph = std::time::Instant::now();
    if n > crate::lobpcg::HH_PANEL {
        crate::lobpcg::householder_tridiag_blocked(&mut t, n, &mut d, &mut e);
    } else {
        crate::lobpcg::householder_tridiag(&mut t, n, &mut d, &mut e);
    }
    let ms_tridiag = t_ph.elapsed().as_secs_f64() * 1000.0;
    let f: Vec<f64> = (0..n.saturating_sub(1)).map(|i| e[i + 1]).collect();
    let q = DenseMatrix::from_row_major_vec(n, n, t);
    if profile {
        SEC_EVALS.store(0, Ordering::Relaxed);
        SEC_TRIG.store(0, Ordering::Relaxed);
        for h in SEC_HIST_SOLVE.iter().chain(SEC_HIST_VEC.iter()) {
            h.store(0, Ordering::Relaxed);
        }
        SEC_PROFILE.store(true, Ordering::Relaxed);
    }
    let t_ph = std::time::Instant::now();
    let mut bufs = MergeBufs::default();
    let (evals, zt) = tridiag_dc_eigen(d, f, &mut bufs)?;
    let ms_dc = t_ph.elapsed().as_secs_f64() * 1000.0;
    if profile {
        SEC_PROFILE.store(false, Ordering::Relaxed);
    }
    // Back-transform Z = Q · Z_T with the tuned GEMM.
    let t_ph = std::time::Instant::now();
    let z = q.matmul(&zt);
    let ms_back = t_ph.elapsed().as_secs_f64() * 1000.0;
    if profile {
        eprintln!(
            "[sonium-eigen-profile] n={n} tridiag={ms_tridiag:.3}ms dc={ms_dc:.3}ms backtransform={ms_back:.3}ms"
        );
    }
    Ok((evals, z))
}

/// Per-subtree scratch for the merge nodes below: the transient k×k
/// `Q̃` and `W` buffers, recycled across one subtree's post-order merges
/// via take/resize/wrap/take-back (the halve-join hands each side its
/// own instance, so concurrent merges never share these). Returned
/// matrices (node eigenvectors, child blocks) stay owned — the parent
/// merge reads its children while reusing these buffers, so backing a
/// return value here would alias. Same values in the same order either
/// way: bitwise identical to per-node allocation.
#[derive(Default)]
struct MergeBufs {
    qt: Vec<f64>,
    w: Vec<f64>,
}

/// Eigendecomposition of a symmetric tridiagonal matrix given by diagonal
/// `d` (len `n`) and off-diagonal `f` (len `n - 1`, `f[i]` couples rows
/// `i` and `i + 1`). Returns eigenvalues (unordered) with eigenvectors as
/// the columns of the returned matrix.
fn tridiag_dc_eigen(
    d: Vec<f64>,
    f: Vec<f64>,
    bufs: &mut MergeBufs,
) -> Result<(Vec<f64>, DenseMatrix<f64>), SolverError> {
    let n = d.len();
    debug_assert!(f.len() + 1 == n || n == 0);
    if n == 0 {
        return Ok((Vec::new(), DenseMatrix::zeros(0, 0)));
    }
    if n == 1 {
        return Ok((d, DenseMatrix::from_row_major_vec(1, 1, vec![1.0])));
    }
    if n <= DC_LEAF {
        return tridiag_ql_leaf(d, f);
    }
    // Pre-size merge scratch for this subtree's top merge: capacity only
    // grows down the right spine (reserve never shrinks, and the left arm
    // re-enters here with its own size), so no merge ever realloc-copies
    // stale buffers mid-solve. Allocation-only: bitwise-neutral.
    bufs.qt.reserve(n * n);
    bufs.w.reserve(n * n);
    let m = n / 2;
    let beta = f[m - 1];
    // Tear the coupling into a rank-1 update: T1'[m-1,m-1] -= β,
    // T2'[0,0] -= β, so T = diag(T1', T2') + β u uᵀ with u supported on
    // the interface positions (m-1, m).
    let mut d1 = d[..m].to_vec();
    let f1 = f[..m - 1].to_vec();
    d1[m - 1] -= beta;
    let mut d2 = d[m..].to_vec();
    let f2 = f[m..].to_vec();
    d2[0] -= beta;

    // The halves are independent subproblems (disjoint inputs, disjoint
    // outputs), so they run as a rayon join: at n = 300 the two 150-wide
    // subtrees (2 QL-75 leaves + k=150 merge each) overlap instead of
    // serialising (~1 ms wall at RAYON=16). Each side gets its own
    // MergeBufs — the passed buffer stays with the right subtree and the
    // parent merge below, so no transient is shared across threads.
    // Bitwise-identical under any schedule: join returns (left, right)
    // in order, every combine step below consumes them positionally,
    // and all inner parallel collects are ordered.
    let (left, right) = rayon::join(
        || tridiag_dc_eigen(d1, f1, &mut MergeBufs::default()),
        || tridiag_dc_eigen(d2, f2, &mut *bufs),
    );
    let (e1, q1) = left?;
    let (e2, q2) = right?;

    if beta == 0.0 {
        // Already decoupled: merge the sorted halves with block-diagonal
        // vectors, no secular equation involved.
        return merge_decoupled(e1, &q1, e2, &q2);
    }

    // z = Q̃ᵀu: last row of Q1 followed by first row of Q2
    // (eigenvectors are stored as columns).
    let mut ds = Vec::with_capacity(n);
    ds.extend_from_slice(&e1);
    ds.extend_from_slice(&e2);
    let mut zs = Vec::with_capacity(n);
    for j in 0..m {
        zs.push(q1[[m - 1, j]]);
    }
    for j in 0..(n - m) {
        zs.push(q2[[0, j]]);
    }
    rank1_merge(ds, zs, beta, &q1, &q2, bufs)
}

/// Leaf solver: run the existing implicit-shift QL directly on a small
/// tridiagonal segment, accumulating into an identity matrix.
fn tridiag_ql_leaf(d: Vec<f64>, f: Vec<f64>) -> Result<(Vec<f64>, DenseMatrix<f64>), SolverError> {
    let n = d.len();
    let prof = SEC_PROFILE.load(Ordering::Relaxed);
    let t_leaf = std::time::Instant::now();
    let mut dd = d;
    // QL input convention (same as Householder output): `ee[0]` is spare,
    // `ee[1..n]` holds the subdiagonals; the routine shifts them into place.
    let mut ee = vec![0.0; n];
    ee[1..].copy_from_slice(&f);
    let mut z = vec![0.0; n * n];
    for i in 0..n {
        z[i * n + i] = 1.0;
    }
    crate::lobpcg::implicit_shift_ql(&mut dd, &mut ee, n, &mut z, 100 * n + 100)?;
    if prof {
        eprintln!(
            "[sonium-eigen-leaf] n={n} ms={:.3}",
            t_leaf.elapsed().as_secs_f64() * 1000.0
        );
    }
    Ok((dd, DenseMatrix::from_row_major_vec(n, n, z)))
}

/// Merge two decoupled halves (`β == 0`): sort the combined spectrum and
/// stack the eigenvector blocks diagonally.
fn merge_decoupled(
    e1: Vec<f64>,
    q1: &DenseMatrix<f64>,
    e2: Vec<f64>,
    q2: &DenseMatrix<f64>,
) -> Result<(Vec<f64>, DenseMatrix<f64>), SolverError> {
    let m = e1.len();
    let n = m + e2.len();
    let mut order: Vec<usize> = (0..n).collect();
    let all: Vec<f64> = e1.iter().chain(e2.iter()).copied().collect();
    order.sort_by(|&a, &b| all[a].total_cmp(&all[b]));
    let evals: Vec<f64> = order.iter().map(|&i| all[i]).collect();
    let mut q = DenseMatrix::zeros(n, n);
    for (new_j, &old) in order.iter().enumerate() {
        if old < m {
            for i in 0..m {
                q[[i, new_j]] = q1[[i, old]];
            }
        } else {
            for i in 0..(n - m) {
                q[[m + i, new_j]] = q2[[i, old - m]];
            }
        }
    }
    Ok((evals, q))
}

/// Merge step for `D + ρ z zᵀ` with diagonal `d` (len `k`), update vector
/// `z`, and the halves' eigenvector blocks `q1` (`m × m`), `q2`
/// (`(k-m) × (k-m)`). Returns node eigenvalues with eigenvectors as columns.
fn rank1_merge(
    d: Vec<f64>,
    z: Vec<f64>,
    rho: f64,
    q1: &DenseMatrix<f64>,
    q2: &DenseMatrix<f64>,
    bufs: &mut MergeBufs,
) -> Result<(Vec<f64>, DenseMatrix<f64>), SolverError> {
    let k = d.len();
    debug_assert_eq!(z.len(), k);
    let m = q1.nrows();
    debug_assert_eq!(m + q2.nrows(), k);
    // Profile-gated node census: wall time plus secular-evaluation share.
    let prof = SEC_PROFILE.load(Ordering::Relaxed);
    let t_node = std::time::Instant::now();
    let e0 = if prof {
        SEC_EVALS.load(Ordering::Relaxed)
    } else {
        0
    };

    // Sort ascending, carrying `z` and the Q̃ = diag(Q1, Q2) columns along.
    let mut order: Vec<usize> = (0..k).collect();
    order.sort_by(|&a, &b| d[a].total_cmp(&d[b]));
    let mut ds: Vec<f64> = order.iter().map(|&i| d[i]).collect();
    let mut zs: Vec<f64> = order.iter().map(|&i| z[i]).collect();
    // Q̃ with permuted columns: column `order[new]` of diag(Q1, Q2).
    // NOTE: splitting Q_node = Q̃·U into two half-GEMMs over the source
    // blocks was tried (half the flops, and the skipped ±0 terms are
    // exactly neutral): triplicate runs showed +0.4ms, not a win — the
    // gather copies, extra allocations, and doubled pool dispatches exceed
    // the flop savings. The full GEMM stands.
    // Q̃ from recycled scratch with an explicit zero fill: the permute
    // below writes only the block-diagonal columns (the off-diagonal
    // blocks must read zero — the old zeros() provided that, stale reuse
    // values do not). Same fill, minus the alloc. Only the stale prefix
    // needs clearing: `resize` already zeroed the grown tail.
    let stale = bufs.qt.len().min(k * k);
    bufs.qt.resize(k * k, 0.0);
    let mut qt = DenseMatrix::from_row_major_vec(k, k, std::mem::take(&mut bufs.qt));
    qt.as_mut_slice()[..stale].fill(0.0);
    // Row-outer permute through the inverse index: source rows stream
    // left-to-right (sequential loads) and each destination row's
    // writes stay within one destination row (L1-resident after
    // first touch). The old column-outer form strided both sides by
    // k (~90k L1 misses at k = 300, ~0.2 ms). Every element is
    // written exactly once with the same value: bitwise-identical.
    {
        let qts = qt.as_mut_slice();
        let q1s = q1.as_slice();
        let q2s = q2.as_slice();
        let m1 = q1.ncols();
        let m2 = q2.ncols();
        let mut inv = vec![0usize; k];
        for (new_j, &old) in order.iter().enumerate() {
            inv[old] = new_j;
        }
        for i in 0..m {
            let dst = &mut qts[i * k..(i + 1) * k];
            let src = &q1s[i * m1..(i + 1) * m1];
            for old in 0..m {
                dst[inv[old]] = src[old];
            }
        }
        for i in 0..(k - m) {
            let dst = &mut qts[(m + i) * k..(m + i + 1) * k];
            let src = &q2s[i * m2..(i + 1) * m2];
            for old in m..k {
                dst[inv[old]] = src[old - m];
            }
        }
    }

    // Gu–Eisenstat deflation.
    let maxd = ds.iter().map(|v| v.abs()).fold(0.0_f64, f64::max);
    let tol = 8.0 * EPS * maxd;
    // Pass 2 records its Givens rotations (a, j, c, s) instead of
    // accumulating a k×k G: U = G·W then applies as one row-rotation
    // pair per deflation ([`givens_apply_left`], 8·k flops each) rather
    // than a 2k³ GEMM plus the G alloc (~19 deflations at the n=300
    // root: 46k flops replace 54M). An empty list skips the application
    // (I·W is exactly W for finite inputs).
    let mut rots: Vec<(usize, usize, f64, f64)> = Vec::new();
    // `live[j]` = still in the reduced problem; deflated slots record
    // (sorted position, accepted eigenvalue).
    let mut live = vec![true; k];
    let mut deflated: Vec<(usize, f64)> = Vec::new();

    // Pass 1: negligible z components — accept d[i] outright. Near-pole
    // roots that this pass keeps are handled downstream by offset (τ)
    // refinement, so no representability criterion is needed here.
    for i in 0..k {
        if (rho * zs[i]).abs() <= tol {
            live[i] = false;
            deflated.push((i, ds[i]));
        }
    }
    // Pass 2: close eigenvalue pairs — rotate z_j away, accept the rotated
    // diagonal, keep the other. The dropped D' off-diagonal is O(tol).
    // Criterion (LAPACK dlaed2): |Δd·C·S| ≤ tol with C = z_j/τ, S = −z_a/τ,
    // τ = hypot — fires on genuinely close pairs and on imbalanced pairs
    // where one z-component dominates.
    let mut prev: Option<usize> = None;
    for j in 0..k {
        if !live[j] {
            continue;
        }
        if let Some(a) = prev {
            let tau = (zs[a] * zs[a] + zs[j] * zs[j]).sqrt();
            if tau > 0.0 {
                let cc = zs[j] / tau;
                let ss = -zs[a] / tau;
                // |Δd·C·S| is exactly the dropped D' off-diagonal
                // |cs·(d_b − d_a)| under the rotation below.
                if ((ds[j] - ds[a]) * cc * ss).abs() <= tol {
                    let (c, s) = (zs[a] / tau, zs[j] / tau);
                    let dda = ds[a];
                    let ddb = ds[j];
                    ds[a] = c * c * dda + s * s * ddb;
                    let def_val = s * s * dda + c * c * ddb;
                    zs[a] = tau;
                    // Record R(a, j) for [`givens_apply_left`] (replaces
                    // the old G ← G·R accumulation loop).
                    rots.push((a, j, c, s));
                    live[j] = false;
                    deflated.push((j, def_val));
                    continue;
                }
            }
        }
        prev = Some(j);
    }

    let kept: Vec<usize> = (0..k).filter(|&i| live[i]).collect();
    let nk = kept.len();
    let d_red: Vec<f64> = kept.iter().map(|&i| ds[i]).collect();
    let z_red: Vec<f64> = kept.iter().map(|&i| zs[i]).collect();

    // Reduced secular problem (empty / singleton need no iteration).
    let ms_setup = if prof {
        Some(t_node.elapsed().as_secs_f64() * 1000.0)
    } else {
        None
    };
    let t_sec = std::time::Instant::now();
    let (lambdas, v_red): (Vec<f64>, Vec<Vec<f64>>) = if nk == 0 {
        (Vec::new(), Vec::new())
    } else if nk == 1 {
        (vec![d_red[0]], vec![vec![1.0]])
    } else {
        secular_eigen(&d_red, &z_red, rho)?
    };
    let ms_sec = if prof {
        Some(t_sec.elapsed().as_secs_f64() * 1000.0)
    } else {
        None
    };

    // Assemble U = G · W, where W scatters deflated basis vectors and the
    // reduced eigenvectors back to sorted positions. Built as one dense
    // matrix and multiplied through the tuned GEMM: the old per-column
    // scalar loops are O(n·k²) double-indexed work the kernel does ~10x
    // faster (same arithmetic up to summation order).
    let mut evals: Vec<f64> = Vec::with_capacity(k);
    // W from recycled scratch with an explicit zero fill (the scatter
    // below is sparse, so stale reuse values must go — same fill the
    // old zeros() did, minus the alloc). Only the stale prefix needs
    // clearing: `resize` already zeroed the grown tail.
    let wstale = bufs.w.len().min(k * k);
    bufs.w.resize(k * k, 0.0);
    let mut wmat = DenseMatrix::from_row_major_vec(k, k, std::mem::take(&mut bufs.w));
    wmat.as_mut_slice()[..wstale].fill(0.0);
    // Flat slice writes: the [[..]] indexing form bounds-checks every
    // element (~0.2 ms/node at k=300); manual row-major offsets are
    // check-free and identical. The reduced scatter runs row-outer —
    // one sequential run per kept row — instead of striding columns
    // by k. Same elements, same values: bitwise-identical.
    let ws = wmat.as_mut_slice();
    let mut col = 0;
    for &(pos, val) in &deflated {
        evals.push(val);
        ws[pos * k + col] = 1.0;
        col += 1;
    }
    for &lam in &lambdas {
        evals.push(lam);
    }
    let nd = col;
    // Root-outer (column-outer) scatter: each root vector reads
    // contiguously while W writes stride by rows. The old row-outer form
    // (sequential writes, strided reads across scattered per-root Vecs)
    // paid L2/L3 load latency per element (~0.4ms at the k=300 root);
    // stores retire through the write buffers far cheaper than the
    // latency-bound scattered loads cost. Same elements, same values.
    for (j, vj) in v_red.iter().enumerate() {
        let vjs = vj.as_slice();
        for (t, &slot) in kept.iter().enumerate() {
            ws[slot * k + nd + j] = vjs[t];
        }
    }
    col += v_red.len();
    debug_assert_eq!(col, k);
    let ms_pre_gemm = t_node.elapsed().as_secs_f64() * 1000.0;
    let t_gemm = std::time::Instant::now();
    // U = G·W by direct rotation application (no G matrix, no GEMM).
    let mut u = wmat;
    if !rots.is_empty() {
        givens_apply_left(u.as_mut_slice(), k, &rots);
    }

    // Node eigenvectors Q_node = Q̃ · U through the tuned GEMM.
    let q_node = qt.matmul(&u);
    // Hand the transient buffers back (Q_node stays owned: the parent
    // merge reads it while reusing these).
    bufs.qt = qt.take_data();
    bufs.w = u.take_data();
    let ms_gemm = t_gemm.elapsed().as_secs_f64() * 1000.0;
    if prof {
        // Drain nonzero histogram bins (counts are deterministic per
        // matrix, hence exact under any host load), then zero for the
        // next node.
        let mut hs = String::new();
        for (b, h) in SEC_HIST_SOLVE.iter().enumerate() {
            let c = h.load(Ordering::Relaxed);
            if c > 0 {
                hs += &format!("{b}:{c} ");
                h.store(0, Ordering::Relaxed);
            }
        }
        let mut hv = String::new();
        for (b, h) in SEC_HIST_VEC.iter().enumerate() {
            let c = h.load(Ordering::Relaxed);
            if c > 0 {
                hv += &format!("{b}:{c} ");
                h.store(0, Ordering::Relaxed);
            }
        }
        let trig = SEC_TRIG.load(Ordering::Relaxed);
        SEC_TRIG.store(0, Ordering::Relaxed);
        eprintln!(
            "[sonium-eigen-node] k={k} evals={} ms={:.3} setup={:.3} sec={:.3} asm={:.3} gemm={:.3} trig={trig} hist_solve=[{hs}] hist_vec=[{hv}]",
            SEC_EVALS.load(Ordering::Relaxed) - e0,
            t_node.elapsed().as_secs_f64() * 1000.0,
            ms_setup.unwrap_or(0.0),
            ms_sec.unwrap_or(0.0),
            ms_pre_gemm - ms_setup.unwrap_or(0.0) - ms_sec.unwrap_or(0.0),
            ms_gemm
        );
    }
    Ok((evals, q_node))
}

/// Apply recorded deflation rotations to the rows of `u` (row-major
/// k×k): `U = G·W` with `G = R_1···R_m` in record order.
///
/// The old code accumulated `G` by right-multiplication (`G ← G·R`) and
/// formed `U` with a 2k³ GEMM. The same product applies left-to-W in
/// reverse record order (innermost factor first): with
/// `R = [[c,−s],[s,c]]` at rows `(a, j)` — the matrix form of the
/// recorded right-multiplication update — each rotation is
/// `new_row_a = c·row_a − s·row_b`,
/// `new_row_b = s·row_a + c·row_b` (8·k flops). Summation order differs
/// from the GEMM by ulps (pinned by `givens_apply_matches_explicit_g`);
/// the merge tests' 1e-12 bars and the orthogonality gate govern.
fn givens_apply_left(u: &mut [f64], k: usize, rots: &[(usize, usize, f64, f64)]) {
    debug_assert_eq!(u.len(), k * k);
    for &(a, j, c, s) in rots.iter().rev() {
        debug_assert!(a < k && j < k);
        for t in 0..k {
            let x = u[a * k + t];
            let y = u[j * k + t];
            u[a * k + t] = c * x - s * y;
            u[j * k + t] = s * x + c * y;
        }
    }
}

/// Solve the reduced secular eigenproblem for diagonal `d` (sorted
/// ascending, len `k >= 2`), update vector `z` (no negligible components,
/// pairwise-separated values), and nonzero `rho`.
///
/// Escape hatch for A/B and debugging: `SONIUM_EIGEN_SECULAR=legacy`
/// forces the τ-first bracket solver everywhere; any other value (or
/// unset) uses the rational-interpolation fast path with legacy
/// fallback per node.
fn secular_force_legacy() -> bool {
    static CACHED: AtomicU64 = AtomicU64::new(u64::MAX);
    let v = CACHED.load(Ordering::Relaxed);
    if v != u64::MAX {
        return v != 0;
    }
    let legacy = std::env::var("SONIUM_EIGEN_SECULAR")
        .map(|s| s.eq_ignore_ascii_case("legacy"))
        .unwrap_or(false);
    CACHED.store(legacy as u64, Ordering::Relaxed);
    legacy
}

/// Returns eigenvalues with matching eigenvectors (each normalized).
/// Fast path first: LAPACK-style rational interpolation
/// ([`secular_eigen_laed`], 3–5 sweeps/root, no refinement pass); the
/// legacy bracket solver below runs only when the fast path declines.
/// Same contract either way: ascending values, normalized vectors.
fn secular_eigen(d: &[f64], z: &[f64], rho: f64) -> Result<(Vec<f64>, Vec<Vec<f64>>), SolverError> {
    if !secular_force_legacy()
        && let Some((lambdas, mut vecs)) = secular_eigen_laed(d, z, rho)
    {
        mgs_close_pairs(&lambdas, &mut vecs);
        return Ok((lambdas, vecs));
    }
    secular_eigen_legacy(d, z, rho)
}

/// LAPACK-style secular solve: DLAED4/DLAED5 values plus the DLAED3
/// recomputed-`ẑ` Löwner vectors (see [`crate::laed`]). Returns `None`
/// whenever anything fails to converge or looks non-finite — the
/// caller then runs the legacy bracket solver for the whole node, so
/// this path never introduces a new failure mode.
fn secular_eigen_laed(d: &[f64], z: &[f64], rho: f64) -> Option<(Vec<f64>, Vec<Vec<f64>>)> {
    use crate::laed::{laed4_root, laed5_pair, recompute_w};
    use std::borrow::Cow;
    let k = d.len();
    let norm_sq: f64 = z.iter().map(|v| v * v).sum();
    if !norm_sq.is_finite() || norm_sq <= 0.0 {
        return None;
    }
    let nrm = norm_sq.sqrt();
    // dlaed2-style normalization: ρ′ = ρ‖z‖², ẑ = z/‖z‖.
    let rho_p = rho * norm_sq;
    // Solve in ρ > 0 orientation; ρ < 0 negates-and-reverses the problem
    // (eigenvalues negate, order reverses) and maps back afterwards.
    let flip = rho_p < 0.0;
    let rp = rho_p.abs();
    let (dp, zp): (Cow<'_, [f64]>, Cow<'_, [f64]>) = if flip {
        let mut dp = vec![0.0; k];
        let mut zp = vec![0.0; k];
        for i in 0..k {
            dp[i] = -d[k - 1 - i];
            zp[i] = z[k - 1 - i] / nrm;
        }
        (Cow::Owned(dp), Cow::Owned(zp))
    } else {
        (
            Cow::Borrowed(d),
            Cow::Owned(z.iter().map(|v| v / nrm).collect()),
        )
    };
    let counting = SEC_PROFILE.load(Ordering::Relaxed);
    if k == 2 {
        // DLAED5 returns both roots with normalized Löwner vectors used
        // directly (DLAED3's K == 2 path skips the ŵ recomputation).
        let (l0, e0) = laed5_pair(dp[0], dp[1], zp[0], zp[1], rp, false);
        let (l1, e1) = laed5_pair(dp[0], dp[1], zp[0], zp[1], rp, true);
        if !(l0.is_finite() && l1.is_finite() && l0 < l1) {
            return None;
        }
        if flip {
            return Some((vec![-l1, -l0], vec![vec![e1[1], e1[0]], vec![e0[1], e0[0]]]));
        }
        return Some((vec![l0, l1], vec![e0.to_vec(), e1.to_vec()]));
    }
    let solve_one = |j: usize| -> Option<(f64, Vec<f64>)> {
        // Thread-local census identical to the legacy path: the task never
        // migrates, so the end delta is exactly this root's sweep count.
        let tl0 = if counting {
            SEC_TL.with(|c| c.get())
        } else {
            0
        };
        let mut delta = vec![0.0; k];
        let lam = laed4_root(&dp, &zp, rp, j, &mut delta, counting)?;
        if counting {
            let d = SEC_TL.with(|c| c.get()) - tl0;
            SEC_HIST_SOLVE[(d.min(128)) as usize].fetch_add(1, Ordering::Relaxed);
        }
        Some((lam, delta))
    };
    // Same ordered parallel collect as the legacy path: bitwise-identical
    // results under either schedule.
    let out: Vec<Option<(f64, Vec<f64>)>> = if k >= PAR_ROOT_MIN {
        (0..k)
            .into_par_iter()
            .with_min_len(par_grain(k))
            .map(solve_one)
            .collect()
    } else {
        (0..k).map(solve_one).collect()
    };
    let mut lams = Vec::with_capacity(k);
    let mut deltas = Vec::with_capacity(k);
    for o in out {
        let (lam, dl) = o?;
        lams.push(lam);
        deltas.push(dl);
    }
    // Bracket order is value order; anything else means a root escaped
    // (must not happen — steps are clamped) → decline to legacy.
    for j in 0..k {
        if !lams[j].is_finite() || (j > 0 && lams[j] <= lams[j - 1]) {
            return None;
        }
    }
    let w = recompute_w(&dp, &zp, &deltas)?;
    let mut vecs = Vec::with_capacity(k);
    for dl in &deltas {
        let mut v = Vec::with_capacity(k);
        let mut n = 0.0;
        for i in 0..k {
            let c = w[i] / dl[i];
            if !c.is_finite() {
                return None;
            }
            v.push(c);
            n += c * c;
        }
        n = n.sqrt();
        if !n.is_finite() || n <= 0.0 {
            return None;
        }
        for c in &mut v {
            *c /= n;
        }
        vecs.push(v);
    }
    if flip {
        lams = (0..k).map(|j| -lams[k - 1 - j]).collect();
        let mut mv = Vec::with_capacity(k);
        for j in 0..k {
            mv.push((0..k).map(|i| vecs[k - 1 - j][k - 1 - i]).collect());
        }
        vecs = mv;
    }
    Some((lams, vecs))
}

/// Legacy τ-first bracket solver: values from safeguarded brackets,
/// vectors as plain Löwner `v_i = z_i / (d_i − λ_j)` except where the
/// predicted residual `|f(λ̂)|/‖v̂‖` exceeds [`TAU_RESID_TOL`] — there
/// the offset τ = λ − d_m is re-solved directly
/// ([`lowener_near_pole`]). Kept as the fallback when the
/// rational-interpolation path declines.
fn secular_eigen_legacy(
    d: &[f64],
    z: &[f64],
    rho: f64,
) -> Result<(Vec<f64>, Vec<Vec<f64>>), SolverError> {
    let k = d.len();
    debug_assert!(k >= 2);
    debug_assert_eq!(z.len(), k);
    debug_assert!(rho != 0.0);
    let norm_sq: f64 = z.iter().map(|v| v * v).sum();

    // Secular function and derivative. One division per element: with
    // inv = 1/denom the fp term reuses t·inv (a multiply) instead of a
    // second division — divisions dominate this loop's cost.
    let counting = SEC_PROFILE.load(Ordering::Relaxed);
    let secular = |lam: f64| -> (f64, f64) {
        if counting {
            SEC_EVALS.fetch_add(1, Ordering::Relaxed);
            SEC_TL.with(|c| c.set(c.get() + 1));
        }
        let mut f = 1.0;
        let mut fp = 0.0;
        for i in 0..k {
            let denom = d[i] - lam;
            let inv = 1.0 / denom;
            let term = rho * z[i] * z[i] * inv;
            f += term;
            fp += term * inv;
        }
        (f, fp)
    };

    // Per-root solves are independent (disjoint outputs, read-only shared
    // inputs) so large problems spread them over rayon threads; the ordered
    // parallel collect keeps results bitwise-identical to the serial loop.
    let solve_one = |j: usize| -> Result<f64, SolverError> {
        // Thread-local baseline: this task never migrates mid-execution,
        // so the end delta is exactly this root's evaluations.
        let tl0 = if counting {
            SEC_TL.with(|c| c.get())
        } else {
            0
        };
        // Interlacing brackets (d sorted ascending). Bracket ends at poles
        // are never *evaluated* — evaluating exactly at a pole yields
        // z²/0 infinities — their signs come from interlacing theory:
        // rho > 0, interior (d[j], d[j+1]): f → −∞ from the left pole,
        // +∞ from the right one. Outer/negative-rho cases analogously.
        // Only the analytic outer bounds (never poles) are evaluated.
        let (lo, hi, flo, fhi) = if rho > 0.0 {
            if j < k - 1 {
                (d[j], d[j + 1], f64::NEG_INFINITY, f64::INFINITY)
            } else {
                let mut hi = d[k - 1] + rho * norm_sq + 1e-300;
                let mut fhi = secular(hi).0;
                for _ in 0..64 {
                    if fhi > 0.0 {
                        break;
                    }
                    hi = d[k - 1] + 2.0 * (hi - d[k - 1]) + 1e-300;
                    fhi = secular(hi).0;
                }
                (d[k - 1], hi, f64::NEG_INFINITY, fhi)
            }
        } else if j == 0 {
            // rho < 0: root below d[0]; f decreases from 1 down through the
            // root to −∞ at the pole, so the straddle is (flo > 0, fhi < 0)
            // (reversed orientation, handled by the solver's initial swap).
            let mut lo = d[0] + rho * norm_sq - 1e-300;
            let mut flo = secular(lo).0;
            for _ in 0..64 {
                if flo > 0.0 {
                    break;
                }
                lo = d[0] + 2.0 * (lo - d[0]) - 1e-300;
                flo = secular(lo).0;
            }
            (lo, d[0], flo, f64::NEG_INFINITY)
        } else {
            (d[j - 1], d[j], f64::INFINITY, f64::NEG_INFINITY)
        };
        // Guard against roundoff-collapsed brackets (deflation promises
        // separation, but stay total).
        let (lo, hi) = if lo < hi {
            (lo, hi)
        } else {
            let span = (hi.abs().max(lo.abs()).max(1.0)) * 16.0 * EPS;
            (lo - span, hi + span)
        };
        // τ-first root solve (nearest-pole equation converges in a handful
        // of steps where raw secular Newton crawls), revert-guarded Newton
        // polish, raw bracket fallback. Values separate to ~1 ulp of |λ|;
        // steep roots whose Löwner residual would suffer take the offset
        // path in the vector stage.
        let out = secular_root_tau_first(&secular, d, z, rho, j, lo, hi, flo, fhi);
        if counting {
            let delta = SEC_TL.with(|c| c.get()) - tl0;
            SEC_HIST_SOLVE[(delta.min(128)) as usize].fetch_add(1, Ordering::Relaxed);
        }
        out
    };
    let lambdas: Vec<f64> = if k >= PAR_ROOT_MIN {
        (0..k)
            .into_par_iter()
            .with_min_len(par_grain(k))
            .map(&solve_one)
            .collect::<Result<Vec<f64>, _>>()?
    } else {
        (0..k).map(solve_one).collect::<Result<Vec<f64>, _>>()?
    };

    // Eigenvectors via Löwner vectors v_i = z_i/(d_i − λ), normalized.
    // The bracket solver above only separates values to ~1 ulp of |λ|, so
    // for a root next to a pole the secular residual |f(λ̂)| can be large
    // (steep f) and the plain Löwner residual |f(λ̂)|·‖z‖/‖v̂‖ with it. Those
    // columns use offset (τ) refinement instead: with τ = λ − d_m solved
    // directly, the dominant denominator −τ is accurate to ~1e-14 relative
    // no matter how close the root sits to the pole, so the direction — and
    // the residual — stay accurate. Trigger on the predicted plain-Löwner
    // residual (exact identity, fused into the same pass at no extra cost).
    // Phase 1: plain Löwner pass per column (fused f(λ̂) and ‖v̂‖ at no
    // extra cost), attempting the offset path where the predicted residual
    // |f(λ̂)|/‖v̂‖ exceeds tolerance.
    struct Cand {
        plain: Vec<f64>,
        pred: f64,
        tau: Option<(Vec<f64>, usize, bool)>,
    }
    // Adjacent-in-value columns (bracket order is value order: each root
    // sits strictly inside its own interlacing interval) whose relative gap
    // is small enough that independent ~1 ulp bracket slop in the two
    // values would show up as a pairwise dot above the tolerance. Both
    // columns attempt the offset path then (accurate τ̂ on each side keeps
    // the shared-pole denominators accurate and the pair orthogonal).
    let close: Vec<bool> = (0..k)
        .map(|j| {
            let scale = lambdas[j].abs().max(1e-300);
            (j > 0 && (lambdas[j] - lambdas[j - 1]).abs() <= CLOSE_PAIR_TOL * scale)
                || (j + 1 < k && (lambdas[j + 1] - lambdas[j]).abs() <= CLOSE_PAIR_TOL * scale)
        })
        .collect();
    // Cand columns are independent like the values above: same ordered
    // parallel collect, same bitwise-identical results.
    let cand_one = |j: usize| -> Cand {
        let tl0 = if counting {
            SEC_TL.with(|c| c.get())
        } else {
            0
        };
        let lam = lambdas[j];
        // Nearest pole.
        let mut m = 0;
        for i in 1..k {
            if (d[i] - lam).abs() < (d[m] - lam).abs() {
                m = i;
            }
        }
        let mut v: Vec<f64> = Vec::with_capacity(k);
        let mut f = 1.0;
        let mut norm_sq = 0.0;
        let mut finite = true;
        for i in 0..k {
            let denom = d[i] - lam;
            let vi = z[i] / denom;
            if !vi.is_finite() {
                finite = false;
                break;
            }
            v.push(vi);
            f += rho * z[i] * vi;
            norm_sq += vi * vi;
        }
        let norm = norm_sq.sqrt().max(1e-300);
        let pred = if finite {
            f.abs() / norm
        } else {
            f64::INFINITY
        };
        // Normalized plain vector (τ→0 limit e_m on an exact pole hit).
        let plain = if finite {
            for x in &mut v {
                *x /= norm;
            }
            v
        } else {
            let mut e = vec![0.0; k];
            e[m] = 1.0;
            e
        };
        // Predicted plain residual is |f(λ̂)|·‖z‖/‖v̂‖ with ‖z‖ = O(1)
        // folded into TAU_RESID_TOL's margin.
        // Direction-error trigger: the dominant denominator (d_m − λ̂) is
        // formed by Sterbenz-exact subtraction but still carries absolute
        // error ~eps·|d_m|, i.e. relative error eps·|d_m|/gap_m. Scaled by
        // the component's weight |v̂_m|/‖v̂‖ this is the vector's direction
        // error from cancellation — invisible to the residual-based trigger
        // above (a small residual can coexist with a wrong direction), yet
        // it surfaces directly as pairwise dots with columns sharing the
        // pole. All quantities are already in hand.
        let gap_m = (d[m] - lam).abs();
        let dir_err = if finite {
            EPS * d[m].abs() * z[m].abs() / (gap_m * gap_m * norm).max(1e-300)
        } else {
            f64::INFINITY
        };
        let tau = if pred <= TAU_RESID_TOL && !close[j] && dir_err <= TAU_DIR_TOL {
            None
        } else {
            lowener_near_pole(d, z, rho, m, lam)
                .ok()
                .map(|(tv, side)| (tv, m, side))
        };
        if counting {
            let delta = SEC_TL.with(|c| c.get()) - tl0;
            SEC_HIST_VEC[(delta.min(128)) as usize].fetch_add(1, Ordering::Relaxed);
            if tau.is_some() {
                SEC_TRIG.fetch_add(1, Ordering::Relaxed);
            }
        }
        Cand { plain, pred, tau }
    };
    let cands: Vec<Cand> = if k >= PAR_ROOT_MIN {
        (0..k)
            .into_par_iter()
            .with_min_len(par_grain(k))
            .map(&cand_one)
            .collect()
    } else {
        (0..k).map(cand_one).collect()
    };
    // Phase 2: one root per pole side. If several columns claim the same
    // (pole, side) — at most one can own its root; the others' solves found
    // a neighbour's artifact — keep the offset vector for the column that
    // needs it most (largest plain predicted residual) and revert the rest
    // to plain. Order-independent.
    use std::collections::HashMap;
    let mut winner: HashMap<(usize, bool), usize> = HashMap::new();
    for (idx, c) in cands.iter().enumerate() {
        if let Some((_, m, side)) = &c.tau {
            let key = (*m, *side);
            match winner.get(&key) {
                Some(&w) if cands[w].pred >= c.pred => {}
                _ => {
                    winner.insert(key, idx);
                }
            }
        }
    }
    let mut vecs = Vec::with_capacity(k);
    for (idx, c) in cands.into_iter().enumerate() {
        let v = match c.tau {
            Some((tv, m, side)) if winner.get(&(m, side)) == Some(&idx) => tv,
            _ => c.plain,
        };
        vecs.push(v);
    }
    mgs_close_pairs(&lambdas, &mut vecs);
    Ok((lambdas, vecs))
}

/// Close-pair re-orthogonalization shared by both secular paths:
/// adjacent columns within CLOSE_PAIR_TOL can carry independent ~1 ulp
/// value slop that surfaces as a pairwise dot (re-solving cannot
/// correlate the two errors). One MGS step per close pair zeroes the
/// dot; the residual cost is ~dot·gap/‖T‖ (the correction lies along a
/// neighbour eigenvector), negligible next to the 1e-13 vector budget.
/// Sequential in j order, hence deterministic.
fn mgs_close_pairs(lambdas: &[f64], vecs: &mut [Vec<f64>]) {
    let k = lambdas.len();
    for j in 0..k.saturating_sub(1) {
        let scale = lambdas[j].abs().max(lambdas[j + 1].abs()).max(1e-300);
        if (lambdas[j + 1] - lambdas[j]).abs() <= CLOSE_PAIR_TOL * scale {
            let dot: f64 = vecs[j]
                .iter()
                .zip(vecs[j + 1].iter())
                .map(|(a, b)| a * b)
                .sum();
            if dot != 0.0 {
                let (lo, hi) = vecs.split_at_mut(j + 1);
                for (t, x) in hi[0].iter_mut().enumerate() {
                    *x -= dot * lo[j][t];
                }
                let nrm: f64 = vecs[j + 1]
                    .iter()
                    .map(|x| x * x)
                    .sum::<f64>()
                    .sqrt()
                    .max(1e-300);
                for x in &mut vecs[j + 1] {
                    *x /= nrm;
                }
            }
        }
    }
}

/// Root solve for interlacing bracket `j`: nearest-pole τ-equation first.
///
/// The raw secular Newton crawls near poles (pole stiffness rejects most
/// Newton steps: ~44 O(k) evals/root measured), while the one-sided
/// τ-hybrid ([`tau_solve_oneside`]) converges in a handful. Candidates are
/// the bracket ends' pole sides (interior roots) or the forced exterior
/// side (outermost roots, by interlacing); each side's extent is pole-free
/// by construction (half-gaps, analytic outer bounds). A short
/// revert-guarded Newton polish on the raw equation then recovers ~1 ulp
/// values for well-conditioned roots (near-pole roots keep the relatively
/// accurate τ value, which is what the vectors need). Falls back to the raw
/// safeguarded bracket solve ([`secular_newton_safe`]) when neither side
/// straddles.
fn secular_root_tau_first(
    secular: &dyn Fn(f64) -> (f64, f64),
    d: &[f64],
    z: &[f64],
    rho: f64,
    j: usize,
    lo: f64,
    hi: f64,
    flo: f64,
    fhi: f64,
) -> Result<f64, SolverError> {
    let k = d.len();
    debug_assert_eq!(z.len(), k);
    // (pole, positive-side, far) attempts in order; at most two.
    let mut cands = [(0usize, true, 0.0f64); 2];
    let mut nc = 0;
    if rho > 0.0 {
        if j < k - 1 {
            let gap = d[j + 1] - d[j];
            if gap > 0.0 {
                let far = 0.5 * gap;
                cands[0] = (j, true, far);
                cands[1] = (j + 1, false, far);
                nc = 2;
            }
        } else {
            let far = hi - d[k - 1];
            if far.is_finite() && far > 0.0 {
                cands[0] = (k - 1, true, far);
                nc = 1;
            }
        }
    } else if j == 0 {
        let far = d[0] - lo;
        if far.is_finite() && far > 0.0 {
            cands[0] = (0, false, far);
            nc = 1;
        }
    } else {
        let gap = d[j] - d[j - 1];
        if gap > 0.0 {
            let far = 0.5 * gap;
            cands[0] = (j - 1, true, far);
            cands[1] = (j, false, far);
            nc = 2;
        }
    }
    let counting = SEC_PROFILE.load(Ordering::Relaxed);
    for &(m, side, far) in cands.iter().take(nc) {
        // τ-equation shifted to pole m. One division per element (see the
        // `secular` closure): fp reuses t·inv.
        // NOTE: routing the denominators through an explicit AVX2
        // reciprocal kernel (bitwise-identical vdivpd, verified by a
        // temporary to_bits test) was A/B-tested here: no measurable gain
        // — LLVM already emits vdivpd for the plain loop, so the kernel
        // only added store/load passes. The fused scalar loop stands.
        let ft = |tau: f64| -> (f64, f64, f64) {
            if counting {
                SEC_EVALS.fetch_add(1, Ordering::Relaxed);
                SEC_TL.with(|c| c.set(c.get() + 1));
            }
            let dm = -tau;
            let invm = 1.0 / dm;
            let tm = rho * z[m] * z[m] * invm;
            let mut f = 1.0 + tm;
            let mut fp = tm * invm;
            let mut scale = 1.0 + tm.abs();
            for i in 0..k {
                if i != m {
                    let den = (d[i] - d[m]) - tau;
                    let inv = 1.0 / den;
                    let t = rho * z[i] * z[i] * inv;
                    f += t;
                    fp += t * inv;
                    scale += t.abs();
                }
            }
            (f, fp, scale)
        };
        // R0 first-order offset τ* ≈ ρz_m²/(1 + R0) as the hybrid's start:
        // near-pole roots start at the answer instead of mid-bracket.
        // Same reliability gate as the side-order estimate in
        // [`lowener_near_pole`]; the hybrid reverts to mid-bracket when the
        // estimate is off-side or outside.
        let mut r0 = 0.0;
        for i in 0..k {
            if i != m {
                r0 += rho * z[i] * z[i] / (d[i] - d[m]);
            }
        }
        let hint = if (1.0 + r0).abs() >= 0.01 {
            let est = rho * z[m] * z[m] / (1.0 + r0);
            if est.is_finite() {
                let end = if side { far } else { -far };
                let inside = if side {
                    est > 0.0 && est < end
                } else {
                    est < 0.0 && est > end
                };
                if inside { Some(est) } else { None }
            } else {
                None
            }
        } else {
            None
        };
        if let Some(tau) = tau_solve_oneside(&ft, rho, side, far, hint) {
            // Revert-guarded Newton polish on the raw equation.
            let mut lam = d[m] + tau;
            let (f0, _) = secular(lam);
            let f0a = f0.abs();
            for _ in 0..3 {
                let (f, fp) = secular(lam);
                if !f.is_finite() || !fp.is_finite() || fp == 0.0 {
                    break;
                }
                let nl = lam - f / fp;
                if !(nl > lo && nl < hi) {
                    break;
                }
                lam = nl;
            }
            let (f1, _) = secular(lam);
            if f1.is_finite() && f1.abs() <= f0a {
                return Ok(lam);
            }
            return Ok(d[m] + tau);
        }
    }
    // Raw bracket fallback (existing path).
    secular_newton_safe(secular, lo, hi, flo, fhi)
}

/// Offset-path trigger: columns whose predicted plain-Löwner residual
/// |f(λ̂)|/‖v̂‖ (exact identity, ‖z‖ = O(1) folded into the margin) exceeds
/// this take the τ-refinement path. 1e-13 holds the merge tests' 1e-12 bars
/// with a decade of margin; loosening to 1e-12 was tried and reverted (no
/// measurable gain, ~6x worse residuals at n=300 — unlike the direction
/// trigger, these solves buy real residual accuracy).
const TAU_RESID_TOL: f64 = 1e-13;

/// Pair trigger (relative eigenvalue gap): adjacent columns at or below
/// this gap both attempt the offset path, since independent bracket slop in
/// the two values otherwise surfaces as a pairwise dot above tolerance.
const CLOSE_PAIR_TOL: f64 = 1e-3;

/// Direction trigger: columns whose dominant Löwner denominator carries
/// cancellation error eps·|d_m|/gap_m above this (weighted by the
/// component) take the offset path, whose re-solved τ̂ is relatively
/// accurate no matter how small the gap. Loosening to 1e-12 was tried and
/// reverted: run-to-run spread (±0.3ms) covers the apparent gain, so the
/// tighter gate stands (more offset solves, zero measured cost, better
/// orthogonality margin).
const TAU_DIR_TOL: f64 = 1e-13;

/// One-sided τ-equation solve for the root on one side of pole `m`.
///
/// Solves `1 + ρz_m²/(−τ) + Σ_{j≠m} ρz_j²/((d_j − d_m) − τ) = 0` on the
/// one-sided bracket `(0, far)` (`want_positive`) or `(−far, 0)`, where
/// `far` is a healthy extent (no pole within it). Theory signs at the pole
/// end (m-term dominates: `f(0⁺)` has the sign of `−ρ·∞`, `f(0⁻)` the
/// opposite); the far end is evaluated and must straddle, else `None` (no
/// root on this side). Newton-safe hybrid (Newton for speed, bisection
/// whenever Newton would leave the bracket or stall), stopping on small
/// `|f|` relative to the live term scale — the bracket width alone cannot
/// resolve roots below ~1 ulp of the pole, but the function value can.
/// Returns the offset τ (root `λ = d[m] + τ`). `hint` optionally seeds the
/// hybrid's start (must lie strictly inside the bracket; the safeguard
/// recomputes from the midpoint otherwise) — a good estimate such as the
/// R0 first-order offset skips the bisection crawl across empty bracket.
///
/// Measured trip census (GOE probe, all sizes): mode 5, mean ~14.6, long
/// bisection tail, ~0.3% hitting the 100-cap. A dlaed4-style rational step
/// would address the tail (~0.5ms prize) but is not implemented: the gain
/// is small next to the accuracy risk in this stabilized solver.
fn tau_solve_oneside(
    ftau: &dyn Fn(f64) -> (f64, f64, f64),
    rho: f64,
    want_positive: bool,
    far: f64,
    hint: Option<f64>,
) -> Option<f64> {
    if !far.is_finite() || far <= 0.0 {
        return None;
    }
    let s_pos = if rho > 0.0 { -1.0 } else { 1.0 };
    let (blo0, bhi0, pole_neg) = if want_positive {
        (0.0, far, s_pos < 0.0)
    } else {
        (-far, 0.0, s_pos > 0.0)
    };
    // pole_neg = (f at pole end < 0).
    let (ffar, _, _) = ftau(if want_positive { far } else { -far });
    if !ffar.is_finite() {
        return None;
    }
    // Straddle requires opposite signs at the two ends.
    if (ffar < 0.0) == pole_neg {
        return None;
    }
    let mut blo = blo0;
    let mut bhi = bhi0;
    // Sign at the pole end (theory); far end has the opposite sign by the
    // straddle check above. The solver preserves the invariant. The pole
    // end is blo on the positive side but bhi on the negative side, so the
    // "negative slot" flips: without this the update below inverts the
    // bracket for (ρ < 0, negative side) and converges to the far endpoint
    // instead of the root.
    let neg_is_blo = if want_positive { pole_neg } else { !pole_neg };
    let mut tau = match hint {
        Some(h) if h > blo && h < bhi => h,
        _ => 0.5 * (blo + bhi),
    };
    let (mut f, mut fp, _) = ftau(tau);
    let mut dx_old = (bhi - blo).abs();
    let mut dx = dx_old;
    for _ in 0..100 {
        if ((tau - bhi) * fp - f) * ((tau - blo) * fp - f) > 0.0
            || (2.0 * f).abs() > dx_old * fp.abs()
            || !fp.is_finite()
        {
            dx_old = dx;
            dx = 0.5 * (bhi - blo);
            tau = blo + dx;
        } else {
            dx_old = dx;
            dx = f / fp;
            tau -= dx;
        }
        // Interior guard: tau must stay strictly inside (blo, bhi) —
        // never touch the pole end (0) or the far end.
        if !(tau > blo && tau < bhi) {
            dx_old = (bhi - blo).abs();
            dx = 0.5 * (bhi - blo);
            tau = blo + dx;
        }
        let (fmid, fpmid, scale) = ftau(tau);
        if fmid == 0.0 || fmid.abs() <= 64.0 * EPS * scale {
            blo = tau;
            bhi = tau;
            break;
        }
        if !fmid.is_finite() {
            // Another pole inside (shouldn't happen post-deflation):
            // discard the step and bisect from the current bracket.
            dx_old = (bhi - blo).abs();
            dx = 0.5 * (bhi - blo);
            tau = blo + dx;
            let (fb, fpb, _) = ftau(tau);
            f = fb;
            fp = fpb;
            if (f < 0.0) == neg_is_blo {
                blo = tau;
            } else {
                bhi = tau;
            }
            continue;
        }
        f = fmid;
        fp = fpmid;
        if (f < 0.0) == neg_is_blo {
            blo = tau;
        } else {
            bhi = tau;
        }
        if (bhi - blo).abs() <= 8.0 * EPS * blo.abs().max(bhi.abs()).max(1e-300) {
            break;
        }
    }
    Some(0.5 * (blo + bhi))
}

/// Löwner vector for a root `lam` near pole `m`, via offset refinement.
///
/// Solves the τ-equation `1 + ρz_m²/(−τ) + Σ_{j≠m} ρz_j²/(d_j − d_m − τ) = 0`
/// for τ = λ − d_m in two stages. Stage 1 runs Newton from τ_exp = λ̂ − d_m
/// inside a relative trust box: the main solver places τ_exp within bracket
/// slop of the true root, so this converges to the column's OWN root and
/// can never wander to a neighbouring pole's artifact. Stage 2 (only when
/// τ_exp hugs the pole, i.e. the dip regime with unrepresentable offsets)
/// runs a Newton-safe hybrid on the one-sided bracket that straddles (pole
/// end from theory, far end evaluated; at most one root per side by
/// interlacing). Offsets far below 1 ulp of d_m are perfectly good doubles,
/// and the gap denominators `(d_j − d_m) − τ` are exact by Sterbenz, so the
/// vector direction stays accurate no matter how close the true root sits
/// to the pole. Both stages stop on small |f| against the live term scale.
/// Returns the vector with the side used, so the caller can enforce one
/// root per pole side across columns.
fn lowener_near_pole(
    d: &[f64],
    z: &[f64],
    rho: f64,
    m: usize,
    lam: f64,
) -> Result<(Vec<f64>, bool), SolverError> {
    let k = d.len();
    // One-sided search extents: half the distance to the neighbouring pole
    // on EACH side (d is sorted ascending). Every other pole stays strictly
    // outside either bracket, so at most one root per side by interlacing —
    // while each bracket reaches far enough to contain that side's genuine
    // root (a shared symmetric `half` would cut off roots sitting past the
    // nearer neighbour). Sides without a neighbour inherit the other side's
    // extent (pole-free there, so any finite extent is safe).
    let gap_neg = if m > 0 { d[m] - d[m - 1] } else { -1.0 };
    let gap_pos = if m + 1 < k { d[m + 1] - d[m] } else { -1.0 };
    let fallback = lam.abs().max(1e-300);
    let far_neg = if gap_neg > 0.0 {
        0.5 * gap_neg
    } else if gap_pos > 0.0 {
        0.5 * gap_pos
    } else {
        fallback
    };
    let far_pos = if gap_pos > 0.0 {
        0.5 * gap_pos
    } else if gap_neg > 0.0 {
        0.5 * gap_neg
    } else {
        fallback
    };
    // τ-equation value, derivative, and a scale (Σ|terms|) for the
    // function-value stop. One fused pass with one division per element:
    // fp reuses t·inv (see the `secular` closure).
    let counting = SEC_PROFILE.load(Ordering::Relaxed);
    let ftau = |tau: f64| -> (f64, f64, f64) {
        if counting {
            SEC_EVALS.fetch_add(1, Ordering::Relaxed);
            SEC_TL.with(|c| c.set(c.get() + 1));
        }
        let dm = -tau;
        let invm = 1.0 / dm;
        let tm = rho * z[m] * z[m] * invm;
        let mut f = 1.0 + tm;
        let mut fp = tm * invm;
        let mut scale = 1.0 + tm.abs();
        for j in 0..k {
            if j != m {
                let den = (d[j] - d[m]) - tau;
                let inv = 1.0 / den;
                let t = rho * z[j] * z[j] * inv;
                f += t;
                fp += t * inv;
                scale += t.abs();
            }
        }
        (f, fp, scale)
    };
    // Offset vector assembly shared by both stages below.
    let build_vec = |tau: f64| -> Result<Vec<f64>, SolverError> {
        let mut v = vec![0.0; k];
        v[m] = z[m] / (-tau);
        for j in 0..k {
            if j != m {
                v[j] = z[j] / ((d[j] - d[m]) - tau);
            }
        }
        let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
        if !norm.is_finite() || norm == 0.0 {
            return Err(SolverError::ConvergenceError {
                status: SolverStatus::MaxIterationsReached,
                iterations: 200,
                residual: f64::INFINITY,
            });
        }
        for x in &mut v {
            *x /= norm;
        }
        Ok(v)
    };
    let tau_exp = lam - d[m];
    // Stage 1: Newton from τ_exp with a relative trust box. The main solver
    // places τ_exp within bracket slop (~1 ulp of |λ̂|) of the true root, so
    // this converges to the column's OWN root — never a neighbouring pole's
    // artifact (artifacts sit at relative distance ~1 from τ_exp for
    // representable roots, outside the trust box). The box preserves τ_exp's
    // sign, so the pole is never touched, and m being the nearest pole keeps
    // every other pole outside the box.
    if tau_exp != 0.0 {
        let (blo, bhi) = if tau_exp > 0.0 {
            (0.5 * tau_exp, 1.5 * tau_exp)
        } else {
            (1.5 * tau_exp, 0.5 * tau_exp)
        };
        let mut tau = tau_exp;
        for _ in 0..30 {
            let (f, fp, scale) = ftau(tau);
            if !f.is_finite() || !fp.is_finite() {
                break;
            }
            if f == 0.0 || f.abs() <= 64.0 * EPS * scale {
                return Ok((build_vec(tau)?, tau > 0.0));
            }
            if fp == 0.0 {
                break;
            }
            let nt = (tau - f / fp).clamp(blo, bhi);
            if (nt - tau).abs() <= 8.0 * EPS * tau.abs().max(1e-300) {
                // Stalled inside the box with |f| still large: no root
                // reachable here (extremum, or the root lies outside).
                break;
            }
            tau = nt;
        }
    }
    // Stage 2: pole-artifact search for the dip regime (tiny z_m, whose
    // true offset sits below representability and which stage 1 therefore
    // cannot reach). Gated on τ_exp hugging the pole: representable roots
    // are handled above, and anything farther out belongs to plain Löwner
    // (the caller falls back). Side order from the R0 estimate below.
    if tau_exp.abs() > 64.0 * EPS * d[m].abs().max(lam.abs()).max(1e-300) {
        return Err(SolverError::ConvergenceError {
            status: SolverStatus::MaxIterationsReached,
            iterations: 0,
            residual: f64::INFINITY,
        });
    }
    // Far-end extents below (pole-free by construction) feed the one-sided
    // τ solves; pole-end theory signs live in [`tau_solve_oneside`].
    // Side order from the R0 estimate τ* ≈ ρz_m²/(1 + R0); fall back to
    // positive-first when it is unreliable.
    let mut r0 = 0.0;
    for j in 0..k {
        if j != m {
            r0 += rho * z[j] * z[j] / (d[j] - d[m]);
        }
    }
    let est = rho * z[m] * z[m] / (1.0 + r0);
    let positive_first = if est.is_finite() && (1.0 + r0).abs() >= 0.01 {
        est > 0.0
    } else {
        true
    };
    // Each side runs the shared one-sided τ hybrid; the first straddling
    // side wins, in R0 order.
    let mut tau_opt: Option<(f64, bool)> = None;
    for side in 0..2 {
        let want_positive = if side == 0 {
            positive_first
        } else {
            !positive_first
        };
        let far = if want_positive { far_pos } else { far_neg };
        if let Some(tau) = tau_solve_oneside(&ftau, rho, want_positive, far, None) {
            tau_opt = Some((tau, want_positive));
            break;
        }
    }
    let (tau, used_positive) = tau_opt.ok_or(SolverError::ConvergenceError {
        status: SolverStatus::MaxIterationsReached,
        iterations: 200,
        residual: f64::INFINITY,
    })?;
    Ok((build_vec(tau)?, used_positive))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lobpcg::{
        householder_tridiag, householder_tridiag_blocked, implicit_shift_ql, symmetric_eigen,
    };

    #[test]
    fn blocked_householder_matches_unblocked() {
        // Blocked (GEMM, ascending LAPACK-style) tridiagonalisation against
        // the scalar descending-NR kernel: each Q independently reconstructs
        // A (orthogonality + Q·T·Qᵀ), and both reductions yield the same
        // eigenvalues. (The T diagonals themselves legitimately differ: the
        // tridiagonal reduction is unique only up to off-diagonal signs and
        // reduction order, so no diagonal-match assertion here.)
        for &n in &[2usize, 3, 17, 33, 64, 100] {
            let a0 = random_symmetric(n, 1000 + n);
            let an = a0.as_slice().iter().map(|x| x * x).sum::<f64>().sqrt();
            let mut au = a0.as_slice().to_vec();
            let mut ab = a0.as_slice().to_vec();
            let (mut du, mut eu) = (vec![0.0; n], vec![0.0; n]);
            let (mut db, mut eb) = (vec![0.0; n], vec![0.0; n]);
            householder_tridiag(&mut au, n, &mut du, &mut eu);
            householder_tridiag_blocked(&mut ab, n, &mut db, &mut eb);
            // Eigenvalue agreement through the shared QL iteration.
            let mut zu = vec![0.0; n * n];
            let mut zb = vec![0.0; n * n];
            for i in 0..n {
                zu[i * n + i] = 1.0;
                zb[i * n + i] = 1.0;
            }
            let (mut lu, mut lb) = (du.clone(), db.clone());
            let (mut mhu, mut mhb) = (eu.clone(), eb.clone());
            implicit_shift_ql(&mut lu, &mut mhu, n, &mut zu, 100 * n + 100)
                .expect("unblocked QL converges");
            implicit_shift_ql(&mut lb, &mut mhb, n, &mut zb, 100 * n + 100)
                .expect("blocked QL converges");
            lu.sort_by(|x, y| x.partial_cmp(y).unwrap());
            lb.sort_by(|x, y| x.partial_cmp(y).unwrap());
            let ed = lu
                .iter()
                .zip(lb.iter())
                .map(|(x, y)| (x - y).abs())
                .fold(0.0, f64::max);
            eprintln!("DIAG hh n={n} eigdiff={ed:.3e}");
            assert!(ed < 1e-10 * an, "n={n} eig diff {ed:.3e}");
            for (tag, qq, dd_, ee_) in [("unblocked", &au, &du, &eu), ("blocked", &ab, &db, &eb)] {
                // Q orthogonality.
                let mut orth = 0.0f64;
                for i in 0..n {
                    for j in (i + 1)..n {
                        let mut dot = 0.0;
                        for t in 0..n {
                            dot += qq[t * n + i] * qq[t * n + j];
                        }
                        orth = orth.max(dot.abs());
                    }
                }
                // Reconstruction Q·T·Qᵀ = A.
                let mut rec = 0.0f64;
                for i in 0..n {
                    for j in 0..n {
                        let mut s = 0.0;
                        for t in 0..n {
                            let mut zt = 0.0;
                            zt += dd_[t] * qq[i * n + t];
                            if t > 0 {
                                zt += ee_[t] * qq[i * n + t - 1];
                            }
                            if t + 1 < n {
                                zt += ee_[t + 1] * qq[i * n + t + 1];
                            }
                            s += zt * qq[j * n + t];
                        }
                        rec = rec.max((s - a0[[i, j]]).abs());
                    }
                }
                eprintln!("DIAG hh n={n} {tag} orth={orth:.3e} rec={:.3e}", rec / an);
                assert!(orth < 1e-13, "n={n} {tag} orth {orth:.3e}");
                assert!(rec / an < 1e-13, "n={n} {tag} rec {:.3e}", rec / an);
            }
        }
    }

    fn prng(k: usize) -> f64 {
        let h = (k as u64)
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407)
            >> 11;
        (h % 2000) as f64 / 1000.0 - 1.0
    }

    fn random_symmetric(n: usize, offset: usize) -> DenseMatrix<f64> {
        let mut m = DenseMatrix::from_fn(n, n, |i, j| prng(offset + i * n + j));
        for i in 0..n {
            for j in (i + 1)..n {
                let v = 0.5 * (m[[i, j]] + m[[j, i]]);
                m[[i, j]] = v;
                m[[j, i]] = v;
            }
        }
        m
    }

    /// Max relative residual ‖A v − λ v‖ / ‖A‖_F and orthonormality
    /// ‖VᵀV − I‖_∞.
    fn decomp_errors(a: &DenseMatrix<f64>, evals: &[f64], vecs: &DenseMatrix<f64>) -> (f64, f64) {
        let n = a.nrows();
        let anorm = a
            .as_slice()
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt()
            .max(1e-300);
        let mut resid: f64 = 0.0;
        for j in 0..n {
            let mut num = 0.0;
            for i in 0..n {
                let mut av = 0.0;
                for k in 0..n {
                    av += a[[i, k]] * vecs[[k, j]];
                }
                let r = av - evals[j] * vecs[[i, j]];
                num += r * r;
            }
            resid = resid.max(num.sqrt() / anorm);
        }
        let mut orth: f64 = 0.0;
        for i in 0..n {
            for j in 0..n {
                let mut dot = 0.0;
                for k in 0..n {
                    dot += vecs[[k, i]] * vecs[[k, j]];
                }
                let target = if i == j { 1.0 } else { 0.0 };
                orth = orth.max((dot - target).abs());
            }
        }
        (resid, orth)
    }

    #[test]
    fn secular_solver_standalone() {
        // Synthetic well-separated problem, both rho signs.
        for &rho in &[2.5, -1.75] {
            let k = 20;
            let d: Vec<f64> = (0..k).map(|i| i as f64 * 1.7 + 0.3).collect();
            let z: Vec<f64> = (0..k).map(|i| prng(999 + i) + 1.5).collect();
            let (lams, vecs) = secular_eigen(&d, &z, rho).expect("secular solve");
            assert_eq!(lams.len(), k);
            // Roots satisfy f(λ) = 0 and interlace.
            let mut sorted = lams.clone();
            sorted.sort_by(|x, y| x.total_cmp(y));
            for (j, &lam) in sorted.iter().enumerate() {
                let mut f = 1.0;
                for i in 0..k {
                    f += rho * z[i] * z[i] / (d[i] - lam);
                }
                assert!(f.abs() < 1e-9, "rho={rho} f(λ{j}) = {f}");
                if rho > 0.0 {
                    if j < k - 1 {
                        assert!(lam > d[j] && lam < d[j + 1], "rho>0 λ{j}={lam}");
                    } else {
                        assert!(lam > d[k - 1], "rho>0 λmax={lam}");
                    }
                } else if j == 0 {
                    assert!(lam < d[0], "rho<0 λ0={lam}");
                } else {
                    assert!(lam > d[j - 1] && lam < d[j], "rho<0 λ{j}={lam}");
                }
            }
            // Eigenvectors orthonormal and exact: (D + ρzzᵀ)v = λv, with a
            // RELATIVE residual bar (the absolute floor scales with ‖A‖,
            // same convention as LAPACK's n·ε·‖A‖ test ratio).
            let scale: f64 = d
                .iter()
                .map(|v| v.abs())
                .fold(0.0_f64, f64::max)
                .max(rho.abs() * z.iter().map(|v| v * v).sum::<f64>())
                .max(1e-300);
            for (j, &lam) in lams.iter().enumerate() {
                let mut num = 0.0;
                for i in 0..k {
                    let mut av = d[i] * vecs[j][i];
                    let mut zdv = 0.0;
                    for t in 0..k {
                        zdv += z[t] * vecs[j][t];
                    }
                    av += rho * z[i] * zdv;
                    let r = av - lam * vecs[j][i];
                    num += r * r;
                }
                let rres = num.sqrt() / scale;
                assert!(rres < 1e-13, "rho={rho} rel-resid {rres:.3e}");
            }
            for a in 0..k {
                for b in 0..k {
                    let dot: f64 = (0..k).map(|i| vecs[a][i] * vecs[b][i]).sum();
                    let want = if a == b { 1.0 } else { 0.0 };
                    assert!((dot - want).abs() < 1e-12, "orth {a},{b}");
                }
            }
        }
    }

    #[test]
    fn rank1_merge_with_identity_blocks() {
        // Direct merge test: D diagonal, Q1 = Q2 = I. Then Q_node must equal
        // the assembled U, and (evals, Q_node) must diagonalize T = D+ρzzᵀ.
        let m1 = 16;
        let m2 = 17;
        let k = m1 + m2;
        let d: Vec<f64> = (0..k).map(|i| i as f64 * 0.9 + 0.2).collect();
        let z: Vec<f64> = (0..k).map(|i| prng(4242 + i) * 0.5 + 0.6).collect();
        let rho = 0.7;
        let mut q1 = DenseMatrix::zeros(m1, m1);
        for i in 0..m1 {
            q1[[i, i]] = 1.0;
        }
        let mut q2 = DenseMatrix::zeros(m2, m2);
        for i in 0..m2 {
            q2[[i, i]] = 1.0;
        }
        let mut bufs = MergeBufs::default();
        let (evals, qn) =
            rank1_merge(d.clone(), z.clone(), rho, &q1, &q2, &mut bufs).expect("merge");
        assert_eq!(evals.len(), k);
        // Residual vs explicit T = D + ρzzᵀ.
        let scale = d
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-300);
        let mut resid: f64 = 0.0;
        for j in 0..k {
            let mut num = 0.0;
            for i in 0..k {
                let mut av = d[i] * qn[[i, j]];
                let mut zdv = 0.0;
                for t in 0..k {
                    zdv += z[t] * qn[[t, j]];
                }
                av += rho * z[i] * zdv;
                let r = av - evals[j] * qn[[i, j]];
                num += r * r;
            }
            resid = resid.max(num.sqrt() / scale);
        }
        let mut orth: f64 = 0.0;
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = (0..k).map(|t| qn[[t, i]] * qn[[t, j]]).sum();
                orth = orth.max((dot - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        assert!(resid < 1e-12, "merge resid {resid:.3e}");
        assert!(orth < 1e-12, "merge orth {orth:.3e}");
    }

    #[test]
    fn rank1_merge_with_unsorted_d() {
        // Same as the identity-block test but with UNSORTED diagonal, so the
        // sort/permutation path (qt column moves) is exercised.
        let m1 = 16;
        let m2 = 17;
        let k = m1 + m2;
        let mut d: Vec<f64> = (0..k).map(|i| i as f64 * 0.9 + 0.2).collect();
        // Deterministic shuffle: reverse each half (keeps halves intact).
        d[..m1].reverse();
        d[m1..].reverse();
        let z: Vec<f64> = (0..k).map(|i| prng(777 + i) * 0.5 + 0.6).collect();
        let rho = -0.9;
        let mut q1 = DenseMatrix::zeros(m1, m1);
        for i in 0..m1 {
            q1[[i, i]] = 1.0;
        }
        let mut q2 = DenseMatrix::zeros(m2, m2);
        for i in 0..m2 {
            q2[[i, i]] = 1.0;
        }
        let mut bufs = MergeBufs::default();
        let (evals, qn) =
            rank1_merge(d.clone(), z.clone(), rho, &q1, &q2, &mut bufs).expect("merge");
        assert_eq!(evals.len(), k);
        let scale = d
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-300);
        let mut resid: f64 = 0.0;
        for j in 0..k {
            let mut num = 0.0;
            for i in 0..k {
                let mut av = d[i] * qn[[i, j]];
                let mut zdv = 0.0;
                for t in 0..k {
                    zdv += z[t] * qn[[t, j]];
                }
                av += rho * z[i] * zdv;
                let r = av - evals[j] * qn[[i, j]];
                num += r * r;
            }
            resid = resid.max(num.sqrt() / scale);
        }
        let mut orth: f64 = 0.0;
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = (0..k).map(|t| qn[[t, i]] * qn[[t, j]]).sum();
                orth = orth.max((dot - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        eprintln!("DIAG unsorted merge resid={resid:.3e} orth={orth:.3e}");
        assert!(resid < 1e-12, "unsorted merge resid {resid:.3e}");
        assert!(orth < 1e-12, "unsorted merge orth {orth:.3e}");
    }

    #[test]
    fn leaf_outputs_are_orthonormal() {
        // Direct check of the one assumption the merge relies on.
        let n = 17;
        let d: Vec<f64> = (0..n).map(|i| prng(555 + i) * 4.0).collect();
        let f: Vec<f64> = (0..n - 1).map(|i| prng(666 + i)).collect();
        let (evals, q) = tridiag_ql_leaf(d.clone(), f.clone()).expect("leaf");
        assert_eq!(evals.len(), n);
        let mut orth: f64 = 0.0;
        for i in 0..n {
            for j in 0..n {
                let dot: f64 = (0..n).map(|t| q[[t, i]] * q[[t, j]]).sum();
                orth = orth.max((dot - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        // Leaf residual vs explicit tridiagonal.
        let mut tri = DenseMatrix::zeros(n, n);
        for i in 0..n {
            tri[[i, i]] = d[i];
            if i + 1 < n {
                tri[[i, i + 1]] = f[i];
                tri[[i + 1, i]] = f[i];
            }
        }
        let (resid, _) = decomp_errors(&tri, &evals, &q);
        eprintln!("DIAG leaf resid={resid:.3e} orth={orth:.3e}");
        assert!(orth < 1e-13, "leaf orth {orth:.3e}");
        assert!(resid < 1e-12, "leaf resid {resid:.3e}");
    }

    /// Offset-path vector for a tiny-z pole: the returned vector must be
    /// pole-dominated with a small Rayleigh-quotient residual against the
    /// explicit rank-1 update (self-consistent: no secular eigenvalue needed).
    fn check_tau_vector(d: &[f64], z: &[f64], rho: f64, m: usize) {
        let k = d.len();
        let lam_hint = d[m];
        let (v, _) = lowener_near_pole(d, z, rho, m, lam_hint).expect("tau vec");
        assert!(v[m].abs() > 0.999, "pole not dominant: {:?}", v);
        // Rayleigh quotient + residual vs explicit T = D + ρzzᵀ.
        let mut zdv = 0.0;
        for t in 0..k {
            zdv += z[t] * v[t];
        }
        let mut rq = 0.0;
        for i in 0..k {
            rq += v[i] * (d[i] * v[i] + rho * z[i] * zdv);
        }
        let mut num = 0.0;
        for i in 0..k {
            let r = d[i] * v[i] + rho * z[i] * zdv - rq * v[i];
            num += r * r;
        }
        assert!(
            num.sqrt() < 1e-12,
            "rho={rho} m={m} Rayleigh resid {:.3e}",
            num.sqrt()
        );
    }

    #[test]
    fn tau_path_positive_side() {
        // ρ > 0, tiny z_0: near-pole root just above d_0 (positive τ side).
        let d = vec![0.0, 1.0, 2.0];
        let z = vec![1e-13, 0.5, 0.5];
        check_tau_vector(&d, &z, 0.7, 0);
    }

    #[test]
    fn tau_path_negative_side() {
        // ρ < 0, tiny z_0: near-pole root just below d_0 (negative τ side).
        // Regression: the bisection update once assumed the pole end was
        // always blo, inverting the bracket on this side and returning the
        // far endpoint (a non-root) instead.
        let d = vec![0.0, 1.0, 2.0];
        let z = vec![1e-13, 0.5, 0.5];
        check_tau_vector(&d, &z, -0.7, 0);
    }

    #[test]
    fn rank1_merge_with_close_pair() {
        // Force pass-2 deflation: plant an exact-tie pair with healthy z.
        // Vectors must stay orthonormal with small residuals.
        let m1 = 16;
        let m2 = 17;
        let k = m1 + m2;
        let mut d: Vec<f64> = (0..k).map(|i| i as f64 * 0.9 + 0.2).collect();
        d[5] = d[6]; // exact tie → pass-2 rotation must fire
        let z: Vec<f64> = (0..k).map(|i| prng(31337 + i) * 0.5 + 0.6).collect();
        let rho = 0.7;
        let mut q1 = DenseMatrix::zeros(m1, m1);
        for i in 0..m1 {
            q1[[i, i]] = 1.0;
        }
        let mut q2 = DenseMatrix::zeros(m2, m2);
        for i in 0..m2 {
            q2[[i, i]] = 1.0;
        }
        let mut bufs = MergeBufs::default();
        let (evals, qn) =
            rank1_merge(d.clone(), z.clone(), rho, &q1, &q2, &mut bufs).expect("merge");
        assert_eq!(evals.len(), k);
        let scale = d
            .iter()
            .map(|v| v.abs())
            .fold(0.0_f64, f64::max)
            .max(1e-300);
        let mut resid: f64 = 0.0;
        for j in 0..k {
            let mut num = 0.0;
            for i in 0..k {
                let mut av = d[i] * qn[[i, j]];
                let mut zdv = 0.0;
                for t in 0..k {
                    zdv += z[t] * qn[[t, j]];
                }
                av += rho * z[i] * zdv;
                let r = av - evals[j] * qn[[i, j]];
                num += r * r;
            }
            resid = resid.max(num.sqrt() / scale);
        }
        let mut orth: f64 = 0.0;
        for i in 0..k {
            for j in 0..k {
                let dot: f64 = (0..k).map(|t| qn[[t, i]] * qn[[t, j]]).sum();
                orth = orth.max((dot - if i == j { 1.0 } else { 0.0 }).abs());
            }
        }
        eprintln!("DIAG closepair resid={resid:.3e} orth={orth:.3e}");
        assert!(resid < 1e-12, "closepair resid {resid:.3e}");
        assert!(orth < 1e-12, "closepair orth {orth:.3e}");
    }

    #[test]
    fn givens_apply_matches_explicit_g() {
        // Direct rotation application must equal the old
        // accumulate-G-then-GEMM form (U = G·W) up to summation-order
        // ulps. Reference G uses the retired accumulation loop verbatim.
        let k = 33;
        let mut rots = Vec::new();
        for i in 0..17 {
            let a = (i * 7 + 3) % k;
            let mut j = (i * 13 + 5) % k;
            if j == a {
                j = (j + 1) % k;
            }
            let (s, c) = ((i as f64) * 0.37).sin_cos();
            rots.push((a, j, c, s));
        }
        let w: Vec<f64> = (0..k * k).map(|t| prng(9000 + t)).collect();
        let mut g = DenseMatrix::zeros(k, k);
        for i in 0..k {
            g[[i, i]] = 1.0;
        }
        for &(a, j, c, s) in &rots {
            for i in 0..k {
                let gia = g[[i, a]];
                let gib = g[[i, j]];
                g[[i, a]] = c * gia + s * gib;
                g[[i, j]] = -s * gia + c * gib;
            }
        }
        let wm = DenseMatrix::from_row_major_vec(k, k, w.clone());
        let want = g.matmul(&wm);
        let mut got = w;
        givens_apply_left(&mut got, k, &rots);
        let scale = want
            .as_slice()
            .iter()
            .map(|v| v.abs())
            .fold(0.0f64, f64::max)
            .max(1e-300);
        let diff = want
            .as_slice()
            .iter()
            .zip(got.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0, f64::max);
        eprintln!("DIAG givens-apply diff={diff:.3e} scale={scale:.3e}");
        assert!(diff < 1e-13 * scale, "givens apply drift {diff:.3e}");
    }

    #[test]
    fn tridiag_single_merge_vs_ql() {
        // n=33: exactly ONE merge (16/17 leaves). Compares D&C against QL on
        // the identical tridiagonal, values and vectors.
        let n = 33;
        let d: Vec<f64> = (0..n).map(|i| prng(31337 + i) * 4.0).collect();
        let f: Vec<f64> = (0..n - 1).map(|i| prng(271828 + i)).collect();
        let mut bufs = MergeBufs::default();
        let (dc_vals, dc_vecs) = tridiag_dc_eigen(d.clone(), f.clone(), &mut bufs).expect("dc");
        // QL reference.
        let mut dd = d.clone();
        let mut ee = vec![0.0; n];
        ee[1..].copy_from_slice(&f);
        let mut zq = vec![0.0; n * n];
        for i in 0..n {
            zq[i * n + i] = 1.0;
        }
        implicit_shift_ql(&mut dd, &mut ee, n, &mut zq, 100 * n + 100).expect("ql");
        let mut dv = dc_vals.clone();
        dv.sort_by(|x, y| x.total_cmp(y));
        dd.sort_by(|x, y| x.total_cmp(y));
        for (j, (&got, &want)) in dv.iter().zip(dd.iter()).enumerate() {
            assert!(
                (got - want).abs() <= 1e-9 * want.abs().max(1.0),
                "λ{j}: {got} vs {want}"
            );
        }
        // Vector check vs explicit tridiagonal.
        let mut tri = DenseMatrix::zeros(n, n);
        for i in 0..n {
            tri[[i, i]] = d[i];
            if i + 1 < n {
                tri[[i, i + 1]] = f[i];
                tri[[i + 1, i]] = f[i];
            }
        }
        let (resid, orth) = decomp_errors(&tri, &dc_vals, &dc_vecs);
        eprintln!("DIAG single-merge resid={resid:.3e} orth={orth:.3e}");
        assert!(resid < 1e-12, "single-merge resid {resid:.3e}");
        assert!(orth < 1e-12, "single-merge orth {orth:.3e}");
    }

    #[test]
    fn dc_multi_merge_reuses_scratch_correctly() {
        // n=200: two k=100 merges plus the k=200 root share the MergeBufs
        // scratch — the off-diagonal blocks of Q̃ and the sparse W scatter
        // must not inherit stale reuse values (regression test: stale Q̃
        // blocks broke every multi-merge solve while single-merge sizes
        // stayed green).
        let n = 200;
        assert!(n > 2 * DC_LEAF); // two merge levels, three nodes
        let d: Vec<f64> = (0..n).map(|i| prng(424242 + i) * 4.0).collect();
        let f: Vec<f64> = (0..n - 1).map(|i| prng(565656 + i)).collect();
        let mut bufs = MergeBufs::default();
        let (evals, vecs) = tridiag_dc_eigen(d.clone(), f.clone(), &mut bufs).expect("dc");
        let mut tri = DenseMatrix::zeros(n, n);
        for i in 0..n {
            tri[[i, i]] = d[i];
            if i + 1 < n {
                tri[[i, i + 1]] = f[i];
                tri[[i + 1, i]] = f[i];
            }
        }
        let (resid, orth) = decomp_errors(&tri, &evals, &vecs);
        eprintln!("DIAG multi-merge resid={resid:.3e} orth={orth:.3e}");
        assert!(resid < 1e-12, "multi-merge resid {resid:.3e}");
        assert!(orth < 1e-12, "multi-merge orth {orth:.3e}");
    }

    #[test]
    fn dc_parallel_recursion_is_bitwise_deterministic() {
        // The halve-join recursion overlaps subtrees on the rayon pool:
        // results must not depend on the schedule. n=200 exercises two
        // merge levels (ordered inner collects + positional combine).
        let n = 200;
        let d: Vec<f64> = (0..n).map(|i| prng(424242 + i) * 4.0).collect();
        let f: Vec<f64> = (0..n - 1).map(|i| prng(565656 + i)).collect();
        let run = || {
            let mut bufs = MergeBufs::default();
            tridiag_dc_eigen(d.clone(), f.clone(), &mut bufs).expect("dc")
        };
        let (v0, q0) = run();
        for _ in 0..3 {
            let (v1, q1) = run();
            assert_eq!(v0, v1, "eigenvalues differ across runs");
            assert_eq!(
                q0.as_slice(),
                q1.as_slice(),
                "eigenvectors differ across runs"
            );
        }
    }

    #[test]
    fn manual_merge_identity_check() {
        // Replicate the n=33 split by hand and verify the merge identity:
        // with Q̃ = diag(Q1, Q2), D the child eigenvalues, z the interface
        // vector and ρ = β, Q̃(D + ρzzᵀ)Q̃ᵀ must equal the explicit T.
        let n = 33;
        let m = n / 2;
        let d: Vec<f64> = (0..n).map(|i| prng(31337 + i) * 4.0).collect();
        let f: Vec<f64> = (0..n - 1).map(|i| prng(271828 + i)).collect();
        let beta = f[m - 1];
        let mut d1 = d[..m].to_vec();
        let f1 = f[..m - 1].to_vec();
        d1[m - 1] -= beta;
        let mut d2 = d[m..].to_vec();
        let f2 = f[m..].to_vec();
        d2[0] -= beta;
        let (e1, q1) = tridiag_ql_leaf(d1, f1).expect("leaf1");
        let (e2, q2) = tridiag_ql_leaf(d2, f2).expect("leaf2");
        // Explicit T.
        let mut t = DenseMatrix::zeros(n, n);
        for i in 0..n {
            t[[i, i]] = d[i];
            if i + 1 < n {
                t[[i, i + 1]] = f[i];
                t[[i + 1, i]] = f[i];
            }
        }
        // Rebuild M = Q̃(D + ρzzᵀ)Q̃ᵀ and compare with T.
        let mut z = vec![0.0; n];
        for j in 0..m {
            z[j] = q1[[m - 1, j]];
        }
        for j in 0..(n - m) {
            z[m + j] = q2[[0, j]];
        }
        let dall: Vec<f64> = e1.iter().chain(e2.iter()).copied().collect();
        // M[i,j] = Σ_t Qt[i,t] (D[t] Qt[j,t] + ρ z[t] (z·Qt[j,:]))
        // with Qt = diag(Q1, Q2).
        let mut worst: f64 = 0.0;
        for i in 0..n {
            for j in 0..n {
                // Row i of Q̃ (only the owning block is nonzero).
                let mut mij = 0.0;
                for t in 0..n {
                    let qti = if i < m {
                        if t < m { q1[[i, t]] } else { 0.0 }
                    } else if t >= m {
                        q2[[i - m, t - m]]
                    } else {
                        0.0
                    };
                    let qtj = if j < m {
                        if t < m { q1[[j, t]] } else { 0.0 }
                    } else if t >= m {
                        q2[[j - m, t - m]]
                    } else {
                        0.0
                    };
                    mij += qti * dall[t] * qtj;
                }
                // ρ (Q̃z)_i (Q̃z)_j ; Q̃z = u = e_{m-1} + e_m exactly.
                let uiz = if i == m - 1 || i == m { 1.0 } else { 0.0 };
                let ujz = if j == m - 1 || j == m { 1.0 } else { 0.0 };
                mij += beta * uiz * ujz;
                worst = worst.max((mij - t[[i, j]]).abs());
            }
        }
        eprintln!("DIAG merge-identity max|M - T| = {worst:.3e}");
        assert!(worst < 1e-11, "merge identity violated: {worst:.3e}");

        // Now run the REAL rank1_merge on this exact triple and check its
        // output against the explicit T.
        let mut z2 = vec![0.0; n];
        for j in 0..m {
            z2[j] = q1[[m - 1, j]];
        }
        for j in 0..(n - m) {
            z2[m + j] = q2[[0, j]];
        }
        let dall2: Vec<f64> = e1.iter().chain(e2.iter()).copied().collect();
        let mut bufs = MergeBufs::default();
        let (mevals, mvecs) =
            rank1_merge(dall2, z2, beta, &q1, &q2, &mut bufs).expect("real merge");
        let (mresid, morth) = decomp_errors(&t, &mevals, &mvecs);
        eprintln!("DIAG real-merge resid={mresid:.3e} orth={morth:.3e}");
        // Per-column residuals + norms to localize the bad columns.
        {
            let an = t
                .as_slice()
                .iter()
                .map(|x| x * x)
                .sum::<f64>()
                .sqrt()
                .max(1e-300);
            for j in 0..n {
                let mut num = 0.0;
                let mut nrm = 0.0;
                for i in 0..n {
                    let mut av = 0.0;
                    for kk in 0..n {
                        av += t[[i, kk]] * mvecs[[kk, j]];
                    }
                    let r = av - mevals[j] * mvecs[[i, j]];
                    num += r * r;
                    nrm += mvecs[[i, j]] * mvecs[[i, j]];
                }
                let rr = (num.sqrt() / an) as f64;
                if rr > 1e-10 {
                    eprintln!(
                        "DIAG   col {j}: eval={:.6e} resid={:.3e} norm={:.6e}",
                        mevals[j],
                        rr,
                        nrm.sqrt()
                    );
                }
            }
        }
        assert!(mresid < 1e-12, "real-merge resid {mresid:.3e}");
        assert!(morth < 1e-12, "real-merge orth {morth:.3e}");
    }

    #[test]
    fn dc_random_matrices_are_accurate() {
        for &n in &[64usize, 70, 100, 150, 200] {
            let a = random_symmetric(n, n * 7919);
            assert!(n >= DC_THRESHOLD);
            let (evals, vecs) = symmetric_eigen(&a, n).expect("D&C eigensolve");
            assert_eq!(evals.len(), n);
            let (resid, orth) = decomp_errors(&a, &evals, &vecs);
            assert!(resid < 1e-12, "n={n} residual {resid:.3e}");
            assert!(orth < 1e-12, "n={n} orthonormality {orth:.3e}");
        }
    }

    #[test]
    fn dc_agrees_with_ql_on_same_tridiagonal() {
        // Gold-standard check: identical Householder tridiagonal in, D&C vs
        // full-size QL out. Exercises every merge level (n=130 → recursion
        // depth 3 with mixed deflation).
        let n = 130;
        let a = random_symmetric(n, 12345);
        let mut t = a.as_slice().to_vec();
        let mut d = vec![0.0; n];
        let mut e = vec![0.0; n];
        householder_tridiag(&mut t, n, &mut d, &mut e);
        let f: Vec<f64> = (0..n - 1).map(|i| e[i + 1]).collect();
        let mut bufs = MergeBufs::default();
        let (dc_vals, _) = tridiag_dc_eigen(d.clone(), f, &mut bufs).expect("D&C tridiagonal");
        // QL reference on the identical tridiagonal: the raw Householder
        // output already matches the QL input convention (spare first slot).
        let mut dd: Vec<f64>;
        let mut ee2: Vec<f64>;
        {
            let mut tt = a.as_slice().to_vec();
            let mut dq = vec![0.0; n];
            let mut eq = vec![0.0; n];
            householder_tridiag(&mut tt, n, &mut dq, &mut eq);
            dd = dq;
            ee2 = eq;
        }
        let mut zq = vec![0.0; n * n];
        for i in 0..n {
            zq[i * n + i] = 1.0;
        }
        implicit_shift_ql(&mut dd, &mut ee2, n, &mut zq, 100 * n + 100).expect("QL reference");
        let mut dc_sorted = dc_vals;
        dc_sorted.sort_by(|x, y| x.total_cmp(y));
        dd.sort_by(|x, y| x.total_cmp(y));
        for (j, (&got, &want)) in dc_sorted.iter().zip(dd.iter()).enumerate() {
            let scale = want.abs().max(1.0);
            assert!(
                (got - want).abs() <= 1e-9 * scale,
                "λ{j}: D&C {got} vs QL {want}"
            );
        }
    }

    #[test]
    fn dc_backtransform_diagnostic() {
        // Same n=130 setup, but check the BACK-TRANSFORMED vectors:
        // Z_full = Q_h · Z_dc must diagonalize the original dense A.
        let n = 130;
        let a = random_symmetric(n, 12345);
        let mut t = a.as_slice().to_vec();
        let mut d = vec![0.0; n];
        let mut e = vec![0.0; n];
        householder_tridiag(&mut t, n, &mut d, &mut e);
        let f: Vec<f64> = (0..n - 1).map(|i| e[i + 1]).collect();
        let mut bufs = MergeBufs::default();
        let (dc_vals, z_dc) =
            tridiag_dc_eigen(d.clone(), f.clone(), &mut bufs).expect("D&C tridiagonal");
        // Reference tridiagonal for the Z_dc check below.
        let mut tri = DenseMatrix::zeros(n, n);
        for i in 0..n {
            tri[[i, i]] = d[i];
            if i + 1 < n {
                tri[[i, i + 1]] = f[i];
                tri[[i + 1, i]] = f[i];
            }
        }
        let (tresid, torth) = decomp_errors(&tri, &dc_vals, &z_dc);
        eprintln!("DIAG tridiag-stage resid={tresid:.3e} orth={torth:.3e}");
        let qh = DenseMatrix::from_row_major_vec(n, n, t);
        let z_full = qh.matmul(&z_dc);
        let (resid, orth) = decomp_errors(&a, &dc_vals, &z_full);
        eprintln!("DIAG backtransform resid={resid:.3e} orth={orth:.3e}");
        assert!(resid < 1e-12, "backtransform resid {resid:.3e}");
        assert!(orth < 1e-12, "backtransform orth {orth:.3e}");
    }

    #[test]
    fn dc_diagonal_input_is_exact() {
        let n = 80;
        let mut a = DenseMatrix::zeros(n, n);
        for i in 0..n {
            a[[i, i]] = (i as f64) - 40.0;
        }
        let (mut evals, vecs) = symmetric_eigen(&a, n).expect("D&C diagonal");
        evals.sort_by(|x, y| x.total_cmp(y));
        for (j, &v) in evals.iter().enumerate() {
            assert!((v - (j as f64 - 40.0)).abs() < 1e-9, "λ{j} = {v}");
        }
        let (_, orth) = decomp_errors(&a, &evals, &vecs);
        assert!(orth < 1e-12, "orth {orth:.3e}");
    }

    #[test]
    fn dc_repeated_eigenvalues_stay_orthonormal() {
        // Identity: every vector is an eigenvector; the merge must deflate
        // everything and still return an orthonormal basis.
        let n = 96;
        let mut a = DenseMatrix::zeros(n, n);
        for i in 0..n {
            a[[i, i]] = 1.0;
        }
        let (evals, vecs) = symmetric_eigen(&a, n).expect("D&C identity");
        for &v in &evals {
            assert!((v - 1.0).abs() < 1e-9, "λ = {v}");
        }
        let (resid, orth) = decomp_errors(&a, &evals, &vecs);
        assert!(resid < 1e-12, "resid {resid:.3e}");
        assert!(orth < 1e-12, "orth {orth:.3e}");
    }

    #[test]
    fn dc_wilkinson_matrix_with_close_pairs() {
        // Wilkinson W_21^+: diag(10..1, 0?) has nearly-equal eigenvalue
        // pairs — the classic deflation stress test.
        let n = 70;
        let mut a = DenseMatrix::zeros(n, n);
        for i in 0..n {
            a[[i, i]] = (n - i) as f64;
            if i + 1 < n {
                a[[i, i + 1]] = 1.0;
                a[[i + 1, i]] = 1.0;
            }
        }
        let (evals, vecs) = symmetric_eigen(&a, n).expect("D&C Wilkinson");
        assert_eq!(evals.len(), n);
        let (resid, orth) = decomp_errors(&a, &evals, &vecs);
        assert!(resid < 1e-12, "resid {resid:.3e}");
        assert!(orth < 1e-12, "orth {orth:.3e}");
        // Trace preserved.
        let trace: f64 = evals.iter().sum();
        let want: f64 = (1..=n).map(|i| i as f64).sum();
        assert!(
            (trace - want).abs() < 1e-8 * want,
            "trace {trace} vs {want}"
        );
    }
}

/// Newton/bisection-safe scalar root finder (classic `rtsafe`): Newton steps
/// that would leave the straddling bracket or converge too slowly fall back
/// to bisection. Converges for any continuous `f` with `f(lo)`, `f(hi)` of
/// opposite sign; the secular function is smooth inside pole brackets.
fn secular_newton_safe(
    secular: &dyn Fn(f64) -> (f64, f64),
    x1: f64,
    x2: f64,
    f1: f64,
    f2: f64,
) -> Result<f64, SolverError> {
    // Bracket must straddle; if roundoff broke it, fail loudly rather than
    // returning a pole-adjacent value.
    if !((f1 < 0.0 && f2 > 0.0) || (f1 > 0.0 && f2 < 0.0)) {
        return Err(SolverError::ConvergenceError {
            status: SolverStatus::MaxIterationsReached,
            iterations: 0,
            residual: (f2 - f1).abs(),
        });
    }
    // Orient so f(xl) < 0 < f(xh).
    let (mut xl, mut xh, mut fl, mut fh) = if f1 < 0.0 {
        (x1, x2, f1, f2)
    } else {
        (x2, x1, f2, f1)
    };
    let mut x = 0.5 * (xl + xh);
    let mut dx_old = (xh - xl).abs();
    let mut dx = dx_old;
    let (mut f, mut df) = secular(x);
    for _ in 0..100 {
        // Bisect when Newton would jump out of brackets or stall.
        if ((x - xh) * df - f) * ((x - xl) * df - f) > 0.0
            || (2.0 * f).abs() > dx_old * df.abs()
            || !df.is_finite()
        {
            dx_old = dx;
            dx = 0.5 * (xh - xl);
            x = xl + dx;
        } else {
            dx_old = dx;
            dx = f / df;
            x -= dx;
        }
        // Converged when the BRACKET is tight (step size lies: near a pole
        // Newton takes tiny steps far from the root). The midpoint of a
        // tight bracket is an accurate root regardless of path taken.
        if (xh - xl).abs() <= 8.0 * EPS * xl.abs().max(xh.abs()).max(1e-300) {
            return Ok(0.5 * (xl + xh));
        }
        let (fx, dfx) = secular(x);
        if !fx.is_finite() {
            // A Newton step landed on (or next to) a pole: discard it and
            // bisect from the current bracket instead.
            dx_old = (xh - xl).abs();
            dx = 0.5 * (xh - xl);
            x = xl + dx;
            let (fx2, dfx2) = secular(x);
            f = fx2;
            df = dfx2;
            if f < 0.0 {
                xl = x;
                fl = f;
            } else {
                xh = x;
                fh = f;
            }
            continue;
        }
        f = fx;
        df = dfx;
        if f == 0.0 {
            // Exact root (bisection landing on it to the last bit): the
            // strict straddle below would trip on fh == 0.0, so return now.
            return Ok(x);
        }
        if f < 0.0 {
            xl = x;
            fl = f;
        } else {
            xh = x;
            fh = f;
        }
        debug_assert!(fl < 0.0 && fh > 0.0);
    }
    // Total on exhaustion: 100 safeguarded steps always collapse the bracket
    // to ~1e-30 of its width, so the midpoint is within rounding of the root
    // even when the width stop inside never fires (root below ~1 ulp of a
    // bracket end, e.g. an endpoint exactly at zero). Erroring here would
    // fail the whole decomposition — and every downstream caller such as
    // LOBPCG's Ritz solves — for input the values are actually fine on. The
    // vector stage re-derives steep directions from the offset equation, so
    // it does not inherit this bracket slop.
    Ok(0.5 * (xl + xh))
}
