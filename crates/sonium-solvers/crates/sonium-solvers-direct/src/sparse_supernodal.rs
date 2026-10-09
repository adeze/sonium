//! SuperLU-style sparse panel factorization.
//!
//! The numeric algorithm is left-looking by panels:
//! 1. Scatter a bounded column panel of `A*Q` into dense workspace.
//! 2. Apply every completed panel as `W_rows -= L_rows,panel * U_panel,W`
//!    through `ComplexField::mat_mul_nn`.
//! 3. Factor the active panel with threshold partial pivoting.
//! 4. Compress its columns into the same CSC L/U representation used by the
//!    simplicial backend.
//!
//! Relaxed symbolic supernodes are etree parent chains. Numeric panels may
//! combine adjacent short supernodes up to `PANEL_WIDTH`, mirroring SuperLU's
//! distinction between symbolic supernodes and panel scheduling.

use super::sparse_lu::{SparseLuError, colamd, compute_col_etree};
use num_traits::{Float, FromPrimitive, One, Zero};
use rayon::prelude::*;
use sonium_solvers_core::traits::ComplexField;
use sonium_solvers_sparse::CsrMatrix;
use std::sync::Mutex;

/// Target panel width (also caps strict-supernode chains). Measured optimum
/// on the 2D-Laplacian family at n=16384 (5-rep means, same binary):
/// width 8 → 135ms, 12 → 130ms, **16 → 118ms**, 24 → 125ms, 32 → 134ms,
/// 64 → 189ms. Narrower panels shrink the per-column pivot-search active
/// sets and the in-panel rank-update rectangles; below 16 the per-panel and
/// per-application fixed costs (reach walk, sorts, GEMM dispatch) dominate.
const PANEL_WIDTH: usize = 16;

// NOTE (chain-panel experiment, reverted 2026-09-25): merging unbranched
// etree runs into 64-wide panels cut contributor applications 12x on the
// AMD-chained grid (349K -> 29K) but DOUBLED numeric time there
// (243 -> 472ms) and hurt healthy trees too (ND 86 -> 132ms). The
// per-pair arithmetic scales superlinearly with width (panel-wide TRSM,
// wider dense tails, wider reach marking MORE chunks: n_gemm 21.5K ->
// 32.6K), eating the pair-count win. Chain links also fire constantly on
// ND trees (root paths are chains), over-merging to 278 panels from 1038.
// Fragmentation needs per-apply fixed-cost reduction, not wider panels.

/// Reusable symbolic analysis for matrices with an identical CSR pattern.
#[derive(Debug, Clone)]
pub struct SparseLuSymbolic {
    n: usize,
    col_perm: Vec<usize>,
    etree: Vec<usize>,
    supernodes: Vec<(usize, usize)>,
    panels: Vec<(usize, usize)>,
    /// Per panel, the constituent strict-supernode ranges (panel boundaries
    /// always align with supernode boundaries since panels merge whole
    /// supernodes). Contributor updates apply per chunk rather than per
    /// panel: columns inside one strict supernode have nested row
    /// structures, so each chunk's update rectangle is tight, while a whole
    /// panel's union over divergent structures is mostly zeros (measured
    /// ~7x rectangle-work reduction at n=16384).
    panel_chunks: Vec<Vec<(usize, usize)>>,
    /// FNV hash of the pattern, kept only as a fast negative check; exact
    /// equality is confirmed against `pattern_row_ptrs`/`pattern_col_indices`.
    pattern_signature: u64,
    /// Exact CSR sparsity pattern the symbolic analysis is valid for.
    pattern_row_ptrs: Vec<usize>,
    pattern_col_indices: Vec<usize>,
    /// Per permuted column, `(original_row, CSR value index)` scatter entries.
    /// Exact-pattern validation makes these indices reusable across numeric
    /// factorizations with different values.
    column_scatter: Vec<Vec<(usize, usize)>>,
    /// Static numeric wave per panel (see [`panel_numeric_waves`]): panels in
    /// the same wave have pairwise-disjoint static touched-original-row sets
    /// and no contributor relation, so they may factor in any order with
    /// bit-identical results. Pattern-only, hence safe under symbolic reuse.
    /// Empty when the census was skipped (serial factorize path); the
    /// driver then runs one ascending wave.
    panel_levels: Vec<usize>,
}

impl SparseLuSymbolic {
    pub fn dimension(&self) -> usize {
        self.n
    }

    pub fn supernode_count(&self) -> usize {
        self.supernodes.len()
    }

    pub fn panel_count(&self) -> usize {
        self.panels.len()
    }

    pub fn elimination_tree(&self) -> &[usize] {
        &self.etree
    }

    /// Static numeric wave per panel (see [`panel_numeric_waves`]).
    pub fn numeric_waves(&self) -> &[usize] {
        &self.panel_levels
    }
}

/// Panel execution order for the numeric phase.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum PanelOrder {
    /// Legacy ascending panel index order.
    Ascending,
    /// Static wave order: ascending wave, ascending panel index within a wave.
    Waves,
}

/// Runtime panel-order override (diagnostic/A-B + kill-switch:
/// `SONIUM_SPARSE_NUMERIC_ORDER=ascending` restores the legacy order).
/// Read once per factorization, like `SONIUM_SPARSE_PAR_MIN_NNZ`.
fn panel_order_from_env() -> PanelOrder {
    match std::env::var("SONIUM_SPARSE_NUMERIC_ORDER")
        .unwrap_or_default()
        .to_ascii_lowercase()
        .as_str()
    {
        "ascending" | "serial" | "0" => PanelOrder::Ascending,
        _ => PanelOrder::Waves,
    }
}

/// Within-wave parallelism control.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum ParallelWaves {
    /// Parallel multi-panel waves at or above the size gate.
    Auto,
    /// Parallel multi-panel waves regardless of the size gate (tests).
    Always,
    /// Serial waves (tests + the ascending-order legacy path).
    Never,
}

/// Minimum `n` for parallel numeric waves (diagnostic/A-B + kill-switch:
/// `SONIUM_SPARSE_NUMERIC_PAR_MIN_N=huge` forces serial waves).
/// Read once per factorization.
///
/// The default floor (65536) sits above every size measured so far: at
/// n <= 16384 strict-partial-pivoting swap interference violates 34-88%
/// of parallel waves, and the attempt-plus-redo waste outweighs the
/// wave parallelism (parallel is ~1.1x slower than serial at grid128).
/// The machinery stays available above the floor and under `Always`
/// (tests) plus the env override; re-measure before lowering.
fn numeric_par_min_n() -> usize {
    std::env::var("SONIUM_SPARSE_NUMERIC_PAR_MIN_N")
        .ok()
        .and_then(|value| value.parse().ok())
        .unwrap_or(65536)
}

/// Wave diagnostic logging (`SONIUM_SPARSE_NUMERIC_DEBUG=1` prints one
/// stderr line per factorization: wave count, parallel waves attempted,
/// serial redos). Off by default; zero cost when off.
fn numeric_debug_enabled() -> bool {
    matches!(
        std::env::var("SONIUM_SPARSE_NUMERIC_DEBUG")
            .unwrap_or_default()
            .to_ascii_lowercase()
            .as_str(),
        "1" | "true" | "yes"
    )
}

/// Static touched-row sets and numeric wave schedule (pattern-only).
///
/// For each column `c`, the static touched set `ST(c)` is the union of the
/// scatter original rows of `c` and of every reach column `j < c` (same
/// etree ancestor walk the numeric driver uses to flag contributors). The
/// column etree of `A^T*A` bounds `L`'s structure for any pivot sequence,
/// so `ST` also bounds the below-diagonal rows of every contributor update
/// the panel will ever apply, plus every row its pivot search can inspect:
/// every original row the panel reads or swaps is in `ST`, regardless of
/// numeric cancellation (cancellation only removes nonzeros).
///
/// A panel joins wave `1 + max` level over its contributor panels and over
/// ascendingly-earlier panels with an overlapping `ST` set, floored at the
/// previous panel's wave so levels are nondecreasing in panel index.
/// Same-wave panels are therefore pairwise `ST`-disjoint
/// ascending-consecutive runs with no contributor relation. Contributor
/// edges are ordered explicitly (an exact replay of the driver's
/// needed-flag walk): `ST` overlap does not subsume them, because the
/// shared scatter row sits in the starter column, which need not belong to
/// the contributor's own panel.
///
/// The nondecreasing floor is load-bearing for soundness, not just
/// scheduling: a pivot swap also moves the diagonal-side original, which
/// sits wherever prior swaps left it and is therefore outside every static
/// touched set. Out-of-ascending-order execution would let such a move
/// invalidate another panel's scatter positions (observed as a false
/// `SingularMatrix` on conv32 with unconstrained waves). With the floor,
/// wave execution order is exactly ascending panel order, so serial
/// execution is byte-identical by construction; parallelism (in a later
/// step) is valid only within a wave, where exact recorded swap/touched
/// sets gate the commit. Panels in one wave factor in any order with
/// identical results: disjoint touched originals mean disjoint row swaps
/// (which commute) and unaffected scatter positions and pivot decisions.
///
/// Cost is two driver-equivalent reach censuses (no bitsets: scatter sets
/// are tiny, so sets union through generation-stamped row marks).
fn panel_numeric_waves(
    n: usize,
    panels: &[(usize, usize)],
    panel_chunks: &[Vec<(usize, usize)>],
    etree: &[usize],
    column_scatter: &[Vec<(usize, usize)>],
) -> Vec<usize> {
    if panels.is_empty() {
        return Vec::new();
    }
    let mut row_to_cols: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (column, entries) in column_scatter.iter().enumerate() {
        for &(original_row, _) in entries {
            row_to_cols[original_row].push(column);
        }
    }
    // Chunk -> owning panel map for the contributor census below.
    let mut chunk_of_column = vec![0usize; n];
    let mut chunk_panel: Vec<usize> = Vec::new();
    for (panel_index, chunks) in panel_chunks.iter().enumerate() {
        for &(chunk_start, chunk_end) in chunks {
            let chunk = chunk_panel.len();
            chunk_panel.push(panel_index);
            chunk_of_column[chunk_start..chunk_end].fill(chunk);
        }
    }
    // Per-column walk generations (the walk's stop condition varies per
    // column, so marks cannot be shared across columns of a panel), one
    // row-stamp generation per panel, and the max wave+1 per touched row.
    let mut col_marks = vec![0usize; n];
    let mut row_marks = vec![0usize; n];
    let mut row_level = vec![0usize; n];
    let mut contributor_marks = vec![0usize; n];
    let mut contributor_generation = 0usize;
    let mut generation = 0usize;
    let mut st_rows: Vec<usize> = Vec::new();
    let mut levels = vec![0usize; panels.len()];
    for (panel_index, &(panel_start, panel_end)) in panels.iter().enumerate() {
        generation += 1;
        let row_generation = generation;
        st_rows.clear();
        for column in panel_start..panel_end {
            for &(original_row, _) in &column_scatter[column] {
                if row_marks[original_row] != row_generation {
                    row_marks[original_row] = row_generation;
                    st_rows.push(original_row);
                }
            }
            generation += 1;
            let col_generation = generation;
            for &(original_row, _) in &column_scatter[column] {
                for &start in &row_to_cols[original_row] {
                    if start >= column {
                        continue;
                    }
                    let mut current = start;
                    while col_marks[current] != col_generation {
                        col_marks[current] = col_generation;
                        if current < column {
                            for &(reach_row, _) in &column_scatter[current] {
                                if row_marks[reach_row] != row_generation {
                                    row_marks[reach_row] = row_generation;
                                    st_rows.push(reach_row);
                                }
                            }
                        }
                        let parent = etree[current];
                        if parent == usize::MAX || parent > column {
                            break;
                        }
                        current = parent;
                    }
                }
            }
        }
        let mut level = 0usize;
        for &row in &st_rows {
            level = level.max(row_level[row]);
        }
        // Contributor census: exact replay of the driver's needed-flag
        // reach walk for this panel. Contributors land strictly earlier.
        contributor_generation += 1;
        for entries in &column_scatter[panel_start..panel_end] {
            for &(original_row, _) in entries {
                for &start in &row_to_cols[original_row] {
                    if start >= panel_start {
                        continue;
                    }
                    let mut current = start;
                    while contributor_marks[current] != contributor_generation {
                        contributor_marks[current] = contributor_generation;
                        if current < panel_start {
                            let source = chunk_panel[chunk_of_column[current]];
                            if source != panel_index {
                                level = level.max(levels[source] + 1);
                            }
                        }
                        let parent = etree[current];
                        if parent == usize::MAX || parent >= panel_end {
                            break;
                        }
                        current = parent;
                    }
                }
            }
        }
        // Nondecreasing floor (see the soundness note above): wave
        // execution order stays exactly ascending panel order.
        if panel_index > 0 {
            level = level.max(levels[panel_index - 1]);
        }
        levels[panel_index] = level;
        for &row in &st_rows {
            row_level[row] = level + 1;
        }
    }
    levels
}

pub(super) struct NumericFactors<T: ComplexField> {
    pub l_col_ptrs: Vec<usize>,
    pub l_row_indices: Vec<usize>,
    pub l_values: Vec<T>,
    pub u_col_ptrs: Vec<usize>,
    pub u_row_indices: Vec<usize>,
    pub u_values: Vec<T>,
    pub row_perm: Vec<usize>,
    pub col_perm: Vec<usize>,
    pub supernode_count: usize,
}

/// One strict-supernode slice of a completed panel: columns with nested row
/// structures, so the (`below × width × chunk-width`) update rectangle is
/// tight. See `SparseLuSymbolic::panel_chunks`.
///
/// The TRSM still runs once per whole panel (the inter-chunk `L_21` coupling
/// inside the source panel must propagate through the solve — solving each
/// chunk against only its own diagonal block drops it and breaks exactness);
/// only the GEMM/scatter splits per chunk.
struct CompletedPanelChunk<T: ComplexField> {
    start: usize,
    end: usize,
    below_original_rows: Vec<usize>,
    lower: Vec<T>,
}

struct CompletedPanel<T: ComplexField> {
    start: usize,
    end: usize,
    /// Full panel unit-lower-triangular block (strictly-lower part packed;
    /// diagonal implicit unit). Shared by the single panel-wide TRSM.
    diagonal_l: Vec<T>,
    chunks: Vec<CompletedPanelChunk<T>>,
}

/// One panel's L/U columns before the ascending merge: wave execution may
/// complete panels out of ascending order, and CSC column blocks must be
/// contiguous in ascending column order, so each panel stages its columns
/// locally and the driver concatenates panels ascendingly after all waves.
struct PanelFactors<T: ComplexField> {
    l_ptr: Vec<usize>,
    l_rows: Vec<usize>,
    l_vals: Vec<T>,
    u_ptr: Vec<usize>,
    u_rows: Vec<usize>,
    u_vals: Vec<T>,
}

pub(super) fn factorize<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    pivot_threshold: T::Real,
) -> Result<NumericFactors<T>, SparseLuError> {
    // The wave-level census (~14ms at n=16384) only serves the parallel
    // path: serial wave execution is exactly ascending panel order (the
    // nondecreasing floor), so skip the census unless parallel waves may
    // engage (same gate the driver uses below).
    let need_levels = panel_order_from_env() == PanelOrder::Waves
        && matrix.num_rows >= numeric_par_min_n()
        && rayon::current_num_threads() > 1;
    let symbolic = analyze_flagged(matrix, need_levels)?;
    factorize_with_symbolic(matrix, &symbolic, pivot_threshold)
}

pub(super) fn analyze<T: ComplexField>(
    matrix: &CsrMatrix<T>,
) -> Result<SparseLuSymbolic, SparseLuError> {
    analyze_with_column_perm(matrix, colamd(matrix))
}

pub(super) fn analyze_flagged<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    need_levels: bool,
) -> Result<SparseLuSymbolic, SparseLuError> {
    analyze_with_column_perm_flagged(matrix, colamd(matrix), need_levels)
}

/// Symbolic analysis with a caller-supplied column permutation
/// (`col_perm[k]` is the original column eliminated at step `k`).
/// Used for orderings other than the default COLAMD (e.g. AMD on
/// symmetric-structure problems); the numeric phase is ordering-agnostic.
pub(super) fn analyze_with_column_perm<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    col_perm: Vec<usize>,
) -> Result<SparseLuSymbolic, SparseLuError> {
    analyze_with_column_perm_flagged(matrix, col_perm, true)
}

/// Flagged analysis: `need_levels = false` skips the wave-level census
/// and stores empty levels (serial execution only; the driver runs one
/// ascending wave, byte-identical to serial wave execution).
pub(super) fn analyze_with_column_perm_flagged<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    col_perm: Vec<usize>,
    need_levels: bool,
) -> Result<SparseLuSymbolic, SparseLuError> {
    let n = matrix.num_rows;
    if n != matrix.num_cols {
        return Err(SparseLuError::NotSquare {
            rows: matrix.num_rows,
            cols: matrix.num_cols,
        });
    }
    if col_perm.len() != n {
        return Err(SparseLuError::DimensionMismatch {
            expected: n,
            actual: col_perm.len(),
        });
    }
    // Reject non-permutations before any indexing (caller-supplied).
    let mut seen = vec![false; n];
    for &column in &col_perm {
        if column >= n || seen[column] {
            return Err(SparseLuError::InvalidColumnPermutation { column });
        }
        seen[column] = true;
    }

    let mut col_perm_inv = vec![0usize; n];
    for (position, &column) in col_perm.iter().enumerate() {
        col_perm_inv[column] = position;
    }
    let mut a_cols: Vec<Vec<(usize, T)>> = vec![Vec::new(); n];
    let mut column_scatter: Vec<Vec<(usize, usize)>> = vec![Vec::new(); n];
    for row in 0..n {
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let permuted_column = col_perm_inv[matrix.col_indices[index]];
            a_cols[permuted_column].push((row, matrix.values[index]));
            column_scatter[permuted_column].push((row, index));
        }
    }
    let etree = compute_col_etree(&a_cols, n);
    // NOTE (amalgamation probe, reverted 2026-09-25): relaxed width
    // 16->8/4/2/1 cut GEMM flops 161.6->117.8/100.3/94.0/81.5M (density
    // 0.52->0.59/0.62/0.62/0.69, finer chunks tighten below-sets) but time
    // ROSE 94->96/104/120/147ms: per-chunk dispatch+scatter (~0.5µs) eats
    // the flop win. w16 optimal on time. Exact-pattern needs fused-chunk
    // (not finer-chunk) dispatch to beat ~0.5µs/chunk barrier.
    let supernodes = relaxed_supernodes(&etree, PANEL_WIDTH);
    // NOTE (split-panels experiment, reverted): panels = strict supernodes
    // did not raise update density (0.46 -> 0.49) and cost ~30ms in panel
    // overhead (111 -> 140ms). The rectangle padding is element-scattered
    // within reached (row, col) pairs, not driven by divergent target
    // columns — only exact-pattern updates could skip it.
    let panels = numeric_panels(&supernodes, PANEL_WIDTH);
    // Chunk split per panel: the strict supernodes covered by each panel
    // (binary search on the sorted supernode starts; panels merge whole
    // supernodes so every panel is an exact union of consecutive entries).
    // NOTE (chunk-coarsening sweep, 2026-09-28): merging consecutive
    // supernodes within a panel up to width 4/8/16 (fewer, bigger-k GEMMs)
    // LOST 105.9 -> 108.8/119.1/140.7ms numeric at grid128: the looser
    // below-sets (zero flops) plus lost per-chunk value skips outweigh
    // the dispatch savings. Every-supernode-its-own-chunk stays optimal.
    let mut panel_chunks: Vec<Vec<(usize, usize)>> = Vec::with_capacity(panels.len());
    for &(start, end) in &panels {
        let first = supernodes.partition_point(|&(s, _)| s < start);
        let mut chunks = Vec::new();
        let mut cursor = first;
        while cursor < supernodes.len() && supernodes[cursor].0 < end {
            debug_assert!(supernodes[cursor].0 >= start && supernodes[cursor].1 <= end);
            chunks.push(supernodes[cursor]);
            cursor += 1;
        }
        debug_assert!(chunks.iter().map(|&(_, e)| e).max().unwrap_or(start) == end);
        panel_chunks.push(chunks);
    }
    let panel_levels = if need_levels {
        panel_numeric_waves(n, &panels, &panel_chunks, &etree, &column_scatter)
    } else {
        Vec::new()
    };
    Ok(SparseLuSymbolic {
        n,
        col_perm,
        etree,
        supernodes,
        panels,
        panel_chunks,
        pattern_signature: pattern_signature(matrix),
        pattern_row_ptrs: matrix.row_ptrs.clone(),
        pattern_col_indices: matrix.col_indices.clone(),
        column_scatter,
        panel_levels,
    })
}

pub(super) fn factorize_with_symbolic<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    symbolic: &SparseLuSymbolic,
    pivot_threshold: T::Real,
) -> Result<NumericFactors<T>, SparseLuError> {
    factorize_with_symbolic_ordered(
        matrix,
        symbolic,
        pivot_threshold,
        panel_order_from_env(),
        ParallelWaves::Auto,
    )
}

/// Current-to-original row permutation pair threaded through panel
/// factorization. Serial execution mutates one global instance; parallel
/// waves give each thread a private wave-entry clone.
struct RowMapping {
    perm: Vec<usize>,
    inv: Vec<usize>,
}

/// Per-panel numeric scratch. Serial execution reuses one instance across
/// panels (it self-restores its invariants at each panel end); parallel
/// waves give each thread its own instance.
struct PanelWorkspace<T: ComplexField> {
    work: Vec<T>,
    active: Vec<bool>,
    active_rows: Vec<usize>,
    needed: Vec<bool>,
    source_seen: Vec<usize>,
    live_sources: Vec<usize>,
    set_list: Vec<usize>,
    nz_blocks: Vec<usize>,
    update_buf: Vec<T>,
    touched: Vec<usize>,
    row_slot: Vec<usize>,
    below_mark: Vec<u32>,
    below_gen: u32,
    dense_block: Vec<T>,
    dense_map: Vec<usize>,
    dense_pos: Vec<usize>,
    reach_marks: Vec<usize>,
    generation: usize,
}

impl<T: ComplexField> PanelWorkspace<T> {
    fn new(n: usize, total_chunks: usize, num_panels: usize, max_panel_width: usize) -> Self {
        Self {
            // Dense `n × max_panel_width` staging area (each panel strides
            // it by its own width); the panel-end clear restores all-zero.
            work: vec![T::zero(); n * max_panel_width],
            active: vec![false; n],
            active_rows: Vec::new(),
            needed: vec![false; total_chunks],
            source_seen: vec![0usize; num_panels],
            live_sources: Vec::new(),
            set_list: Vec::new(),
            nz_blocks: Vec::new(),
            update_buf: Vec::new(),
            touched: Vec::new(),
            row_slot: vec![usize::MAX; n],
            below_mark: vec![0u32; n],
            below_gen: 0,
            dense_block: vec![T::zero(); n * max_panel_width],
            dense_map: Vec::new(),
            dense_pos: vec![usize::MAX; n],
            reach_marks: vec![0usize; n],
            generation: 0,
        }
    }
}

/// Exact interference record for wave validation: every original row whose
/// mapping this panel read (scatter originals plus applied contributor
/// below-rows) and every recorded swap pair (both originals, including the
/// diagonal-side original that sits outside every static touched set).
/// Empty cost when unused (serial path passes `None`).
#[derive(Default)]
struct PanelRecord {
    touched: Vec<usize>,
    swaps: Vec<(usize, usize)>,
}

/// One parallel-wave panel outcome: panel index plus staged factors,
/// packed update source, and interference record (or the panel's error).
type WaveAttempt<T> = (
    usize,
    Result<(PanelFactors<T>, CompletedPanel<T>, PanelRecord), SparseLuError>,
);

/// Factor one panel against completed contributors: scatter, needed-flag
/// reach walk, ascending contributor application, dense gather, serial
/// in-panel threshold-partial-pivot factorization, and packing. Reads the
/// shared wave-entry mapping state through `mapping` (global in serial
/// execution, a private wave-entry clone in parallel waves) and applies
/// this panel's swaps to it. Returns the staged factors plus the packed
/// completed panel; the driver owns completion bookkeeping and the merge.
#[allow(clippy::too_many_arguments)]
fn factor_panel<T: ComplexField>(
    panel_index: usize,
    matrix: &CsrMatrix<T>,
    symbolic: &SparseLuSymbolic,
    completed: &[Option<CompletedPanel<T>>],
    chunk_source: &[usize],
    chunk_offsets: &[usize],
    chunk_of_column: &[usize],
    row_to_cols: &[Vec<usize>],
    mapping: &mut RowMapping,
    ws: &mut PanelWorkspace<T>,
    pivot_threshold: T::Real,
    pivot_tol: T::Real,
    mut record: Option<&mut PanelRecord>,
) -> Result<(PanelFactors<T>, CompletedPanel<T>), SparseLuError> {
    let panels = &symbolic.panels;
    let (panel_start, panel_end) = panels[panel_index];
    let panel_width = panel_end - panel_start;
    let PanelWorkspace {
        work,
        active,
        active_rows,
        needed,
        source_seen,
        live_sources,
        set_list,
        nz_blocks,
        update_buf,
        touched,
        row_slot,
        below_mark,
        below_gen,
        dense_block,
        dense_map,
        dense_pos,
        reach_marks,
        generation,
    } = ws;
    let row_perm = &mut mapping.perm;
    let row_perm_inv = &mut mapping.inv;
    // Panel-local L/U staging (merged ascendingly after all waves).
    let mut local_l_ptr = vec![0usize; panel_width + 1];
    let mut local_l_rows: Vec<usize> = Vec::new();
    let mut local_l_vals: Vec<T> = Vec::new();
    let mut local_u_ptr = vec![0usize; panel_width + 1];
    let mut local_u_rows: Vec<usize> = Vec::new();
    let mut local_u_vals: Vec<T> = Vec::new();

    for local_column in 0..panel_width {
        let column = panel_start + local_column;
        for &(original_row, value_index) in &symbolic.column_scatter[column] {
            let row = row_perm_inv[original_row];
            work[row * panel_width + local_column] += matrix.values[value_index];
            mark_active(row, active, active_rows);
            if let Some(rec) = record.as_mut() {
                rec.touched.push(original_row);
            }
        }
    }

    // Mark the completed chunks on an etree ancestor chain of any column
    // sharing an original row with this panel.
    *generation += 1;
    // Stamp-clear only the flags set by the previous panel (tracked in
    // `set_list`): replaces the O(total_chunks) clear+resize per panel.
    for &chunk in set_list.iter() {
        needed[chunk] = false;
    }
    set_list.clear();
    live_sources.clear();
    for column in panel_start..panel_end {
        for &(original_row, _) in &symbolic.column_scatter[column] {
            for &j in &row_to_cols[original_row] {
                // Skip starts at/after this panel: etree parents always
                // have higher indices than children, so a chain starting
                // at `j >= panel_start` only visits columns that fail the
                // `col < panel_start` flag condition below — it can set no
                // needed flag, and its generation-local marks die with
                // this panel. Flag-identical; only dead marks are skipped.
                if j >= panel_start {
                    continue;
                }
                let mut col = j;
                while reach_marks[col] != *generation {
                    reach_marks[col] = *generation;
                    if col < panel_start {
                        let chunk = chunk_of_column[col];
                        // Wave order: chunks of not-yet-completed panels
                        // are skipped (their source index is still the
                        // sentinel). True contributors always completed
                        // in strictly earlier waves: contributor edges
                        // are ordered explicitly at analyze time (see
                        // `panel_numeric_waves`). The walk itself
                        // continues regardless.
                        if chunk_source[chunk] != usize::MAX && !needed[chunk] {
                            needed[chunk] = true;
                            set_list.push(chunk);
                            let source = chunk_source[chunk];
                            // Collect each live source once (first stamp):
                            // the contributor loop below iterates this
                            // sorted list instead of scanning all panels.
                            if source_seen[source] != *generation {
                                source_seen[source] = *generation;
                                live_sources.push(source);
                            }
                        }
                    }
                    let parent = symbolic.etree[col];
                    if parent == usize::MAX || parent >= panel_end {
                        break;
                    }
                    col = parent;
                }
            }
        }
    }

    // NOTE (parallel-apply attempt, reverted): contributors form a
    // left-looking dependence chain — a later contributor's TRSM reads
    // rows that earlier contributors' scatters update — so the compute
    // phase cannot fan out over a pre-panel snapshot (it reads stale
    // values; caught by `supernodal_parallel_apply_matches_residual`
    // with residual ~2.9). Real parallelism here needs etree-guided
    // level scheduling, not a flat fan-out. Serial order below.
    // Live sources apply in ascending index order (sorted collection
    // from the reach walk above): the same order — and every
    // floating-point result — as the old full-panel scan with its
    // liveness skip, at O(live) instead of O(panels) iterations.
    live_sources.sort_unstable();
    for &source_index in live_sources.iter() {
        debug_assert_eq!(source_seen[source_index], *generation);
        // Needed flags are only set for completed panels (see the
        // completion guard above), so the source is always present;
        // the error arm is unreachable-by-construction diagnostics.
        let Some(source) = completed[source_index].as_ref() else {
            return Err(SparseLuError::NumericalBreakdown {
                stage: "supernodal wave contributor incomplete",
                index: source_index,
            });
        };
        let chunk_base = chunk_offsets[source_index];
        let chunk_end = chunk_offsets[source_index + 1];
        apply_completed_panel(
            source,
            panel_width,
            row_perm_inv,
            work,
            active,
            active_rows,
            update_buf,
            touched,
            &needed[chunk_base..chunk_end],
            record.as_mut().map(|rec| &mut rec.touched),
        );
    }
    // Pre-mark every panel row so each has a dense row below even when
    // its diagonal was never scattered (those rows are inert zeros).
    // This also freezes the active set: every later mark in the column
    // loop (swaps, rank updates) over-marks an already-active row, so
    // the dense mappings built below stay complete for the whole panel.
    for row in panel_start..panel_end {
        mark_active(row, active, active_rows);
    }
    // Sort for a deterministic ascending gather layout (`active_rows`
    // is already unique via `mark_active`'s bool guard). The lazy
    // `active_sorted` machinery below stays as a safety net though it
    // should never fire now that the set is frozen.
    active_rows.sort_unstable();
    active_rows.dedup();
    let mut active_sorted = true;
    // Gather the panel's active rows into the dense block (every used
    // cell is overwritten, so no zeroing is needed) and stamp the
    // global-to-block positions.
    let num_panel_rows = active_rows.len();
    dense_map.clear();
    dense_map.extend_from_slice(active_rows);
    {
        let block = &mut dense_block[..num_panel_rows * panel_width];
        for (position, &row) in active_rows.iter().enumerate() {
            dense_pos[row] = position;
            block[position * panel_width..(position + 1) * panel_width]
                .copy_from_slice(&work[row * panel_width..(row + 1) * panel_width]);
        }
    }
    for local_column in 0..panel_width {
        let column = panel_start + local_column;
        // Global `column` always has a dense row (pre-marked at gather);
        // its value is zero when never scattered.
        let column_block = dense_pos[column];
        debug_assert!(column_block != usize::MAX);
        let diagonal_abs = dense_block[column_block * panel_width + local_column].norm();
        if !diagonal_abs.is_finite() {
            return Err(SparseLuError::NumericalBreakdown {
                stage: "supernodal pivot search",
                index: column,
            });
        }
        let mut max_abs = diagonal_abs;
        let mut max_row = column;
        // Dense pivot search in block order rather than ascending global
        // rows: the maximum VALUE is order-independent, so the singular
        // verdict below matches the serial scan exactly — only
        // exact-tie pivot choices may differ (same residual class).
        if !active_sorted {
            active_rows.sort_unstable();
            active_sorted = true;
        }
        // Fused pivot/partition scan: a single contiguous pass over the
        // block records every nonzero block (in block order) while
        // tracking the pivot maximum over rows at/below the diagonal.
        // This replaces the old two-pass structure (pivot search, then a
        // second full partition scan); classification of the recorded
        // blocks happens after the swap below. Maximum selection is
        // untouched (same values, same order, same pre-swap rows — zero
        // values can never exceed a norm maximum, and norm(0) is finite,
        // so skipping their norms changes nothing), and classification
        // resolves post-swap rows exactly as the old partition did:
        // same pairs, same order, bit-identical factors.
        nz_blocks.clear();
        for block in 0..num_panel_rows {
            let value = dense_block[block * panel_width + local_column];
            if value == T::zero() {
                continue;
            }
            nz_blocks.push(block);
            let row = dense_map[block];
            if row < column {
                continue;
            }
            let candidate = value.norm();
            if !candidate.is_finite() {
                return Err(SparseLuError::NumericalBreakdown {
                    stage: "supernodal pivot search",
                    index: row * panel_width + local_column,
                });
            }
            if candidate > max_abs {
                max_abs = candidate;
                max_row = row;
            }
        }
        if max_abs <= pivot_tol {
            return Err(SparseLuError::SingularMatrix(column));
        }
        let pivot_row = if diagonal_abs >= pivot_threshold * max_abs {
            column
        } else {
            max_row
        };
        if pivot_row != column {
            // Mirror the interchange by relabeling positions to slots
            // (mapping + stamps only — slot contents stay put), which
            // is exactly the serial segment swap seen through the
            // position indirection: position `column` now resolves to
            // the pivot's values and vice versa. Moving contents AND
            // relabeling would cancel out and silently skip the swap.
            // The strided workspace is untouched from here to the
            // panel-end clear. Both rows are in the dense set (frozen
            // active set).
            let pivot_block = dense_pos[pivot_row];
            debug_assert!(pivot_block != usize::MAX);
            debug_assert!(column_block != pivot_block);
            dense_map.swap(column_block, pivot_block);
            dense_pos[column] = pivot_block;
            dense_pos[pivot_row] = column_block;
            let original_column_row = row_perm[column];
            let original_pivot_row = row_perm[pivot_row];
            row_perm.swap(column, pivot_row);
            row_perm_inv[original_column_row] = pivot_row;
            row_perm_inv[original_pivot_row] = column;
            if let Some(rec) = record.as_mut() {
                rec.swaps.push((original_column_row, original_pivot_row));
                // Both swap originals also count as touched: validation
                // must see diagonal-side originals (which sit outside
                // every static touched set) to keep swap replay safe.
                rec.touched.push(original_column_row);
                rec.touched.push(original_pivot_row);
            }
            active_sorted &= !mark_active(column, active, active_rows);
            active_sorted &= !mark_active(pivot_row, active, active_rows);
        }

        // Single-pass column tail over the fused scan's recorded nonzero
        // blocks (block order) with post-swap rows: U-push, multiplier
        // formation + L-push, and rank update happen inline per block
        // instead of replaying three pair lists over three passes.
        // Order- and value-identical to the old
        // partition/upush/mult/rank sequence: U entries (sentinel first
        // when the diagonal is missing, then block order), L pushes and
        // rank updates in block order, same FMAs in the same order, same
        // error stages and indices. Column-local values of unprocessed
        // blocks are intact when read (each block is visited once; mult
        // overwrites only its own slot, rank only trailing segments of
        // its own row, and the pivot row is never written in the loop).
        let diagonal_block = dense_pos[column];
        let diagonal = dense_block[diagonal_block * panel_width + local_column];
        // No zero check here: a zero diagonal fails the inversion check
        // below exactly as before (after the sentinel push, whose
        // entries are discarded with the vectors on the error path).
        if diagonal == T::zero() {
            // Missing diagonal pushes a zero, exactly like the serial
            // path reading the untouched (zero) workspace cell.
            // Sentinel first, preserving the old push order.
            local_u_rows.push(column);
            local_u_vals.push(T::zero());
        }
        let diagonal_inv = diagonal.inv();
        if !diagonal_inv.norm().is_finite() {
            return Err(SparseLuError::NumericalBreakdown {
                stage: "supernodal pivot inversion",
                index: column,
            });
        }
        // NOTE (axpy attempt, reverted): one `T::vec_axpy` per row
        // measured ~1ms SLOWER (9.3 vs 8.6ms t_rank, same residual
        // class). The segments are short (≤15) so dispatch/call overhead
        // dominates, and the loop is memory-bound (L2-streaming row
        // traffic) where SIMD adds no rate — the scalar loop already
        // touches each updated element exactly once.
        let pivot_block = diagonal_block;
        // Hoisted pivot-row trailing pattern: the pivot row is never
        // written in the block loop below, so its trailing nonzeros are
        // loop-invariant. Scanning them once per column (<=15 cells)
        // replaces the per-(block x trailing) zero-skip branch with a
        // branch-free FMA list while executing the same FMAs in the
        // same order (ascending trailing index, block order) —
        // bit-identical factors with fewer pair iterations.
        let mut trail: [(usize, T); 64] = std::array::from_fn(|_| (0usize, T::zero()));
        let mut trail_len = 0usize;
        debug_assert!(panel_width - local_column - 1 < 64);
        for next_local in (local_column + 1)..panel_width {
            let upper = dense_block[pivot_block * panel_width + next_local];
            if upper != T::zero() {
                trail[trail_len] = (next_local, upper);
                trail_len += 1;
            }
        }
        for &block in nz_blocks.iter() {
            let row = dense_map[block];
            if row <= column {
                local_u_rows.push(row);
                local_u_vals.push(dense_block[block * panel_width + local_column]);
                continue;
            }
            let multiplier = dense_block[block * panel_width + local_column] * diagonal_inv;
            if !multiplier.norm().is_finite() {
                return Err(SparseLuError::NumericalBreakdown {
                    stage: "supernodal multiplier formation",
                    index: row * panel_width + local_column,
                });
            }
            dense_block[block * panel_width + local_column] = multiplier;
            local_l_rows.push(row_perm[row]);
            local_l_vals.push(multiplier);
            let base = block * panel_width;
            for &(next_local, upper) in trail[..trail_len].iter() {
                let slot = &mut dense_block[base + next_local];
                *slot -= multiplier * upper;
                if !slot.norm().is_finite() {
                    return Err(SparseLuError::NumericalBreakdown {
                        stage: "supernodal panel update",
                        index: row * panel_width + next_local,
                    });
                }
            }
            active_sorted &= !mark_active(row, active, active_rows);
        }
        local_u_ptr[local_column + 1] = local_u_rows.len();
        local_l_ptr[local_column + 1] = local_l_rows.len();
    }
    let completed = pack_completed_panel(
        panel_start,
        panel_end,
        &symbolic.panel_chunks[panel_index],
        &local_l_ptr,
        &local_l_rows,
        &local_l_vals,
        row_perm_inv,
        row_slot,
        below_mark,
        below_gen,
    );
    let factors = PanelFactors {
        l_ptr: local_l_ptr,
        l_rows: local_l_rows,
        l_vals: local_l_vals,
        u_ptr: local_u_ptr,
        u_rows: local_u_rows,
        u_vals: local_u_vals,
    };
    // Restore the all-zero workspace invariant for the next panel by
    // clearing exactly the touched rows (same stride as the writes
    // above). `active_rows` holds each touched row once via `active`.
    for &row in active_rows.iter() {
        work[row * panel_width..(row + 1) * panel_width].fill(T::zero());
        active[row] = false;
    }
    active_rows.clear();
    Ok((factors, completed))
}

/// Record one completed panel: assign its global-chunk source range and
/// stage its factors plus packed update source under its panel index.
fn commit_panel<T: ComplexField>(
    panel_index: usize,
    factors: PanelFactors<T>,
    completed: CompletedPanel<T>,
    chunk_source: &mut [usize],
    chunk_offsets: &[usize],
    completed_panels: &mut [Option<CompletedPanel<T>>],
    panel_factors: &mut [Option<PanelFactors<T>>],
) {
    let chunk_base = chunk_offsets[panel_index];
    chunk_source[chunk_base..chunk_base + completed.chunks.len()].fill(panel_index);
    completed_panels[panel_index] = Some(completed);
    panel_factors[panel_index] = Some(factors);
}

/// Factor one wave's panels ascendingly against the shared evolving
/// mapping. Used below the parallel gate, for the ascending-order legacy
/// path, and as the deterministic serial redo of a failed parallel wave
/// (which yields serial semantics, including first-error precedence).
#[allow(clippy::too_many_arguments)]
fn factor_wave_serial<T: ComplexField>(
    wave: &[usize],
    matrix: &CsrMatrix<T>,
    symbolic: &SparseLuSymbolic,
    completed_panels: &mut [Option<CompletedPanel<T>>],
    chunk_source: &mut [usize],
    chunk_offsets: &[usize],
    chunk_of_column: &[usize],
    row_to_cols: &[Vec<usize>],
    panel_factors: &mut [Option<PanelFactors<T>>],
    mapping: &mut RowMapping,
    workspace: &mut PanelWorkspace<T>,
    pivot_threshold: T::Real,
    pivot_tol: T::Real,
) -> Result<(), SparseLuError> {
    for &panel_index in wave {
        let (factors, completed) = factor_panel(
            panel_index,
            matrix,
            symbolic,
            completed_panels,
            chunk_source,
            chunk_offsets,
            chunk_of_column,
            row_to_cols,
            mapping,
            workspace,
            pivot_threshold,
            pivot_tol,
            None,
        )?;
        commit_panel(
            panel_index,
            factors,
            completed,
            chunk_source,
            chunk_offsets,
            completed_panels,
            panel_factors,
        );
    }
    Ok(())
}

/// Check a scratch workspace out of the wave pool (lazily grown, so serial
/// execution pays nothing).
fn checkout_workspace<T: ComplexField>(
    pool: &Mutex<Vec<PanelWorkspace<T>>>,
    n: usize,
    total_chunks: usize,
    num_panels: usize,
    max_panel_width: usize,
) -> PanelWorkspace<T> {
    pool.lock()
        .ok()
        .and_then(|mut guard| guard.pop())
        .unwrap_or_else(|| PanelWorkspace::new(n, total_chunks, num_panels, max_panel_width))
}

/// Return a workspace to the wave pool (poisoned locks degrade to dropping
/// the workspace instead of blocking factorization).
fn checkin_workspace<T: ComplexField>(
    pool: &Mutex<Vec<PanelWorkspace<T>>>,
    workspace: PanelWorkspace<T>,
) {
    if let Ok(mut guard) = pool.lock() {
        guard.push(workspace);
    }
}

pub(super) fn factorize_with_symbolic_ordered<T: ComplexField>(
    matrix: &CsrMatrix<T>,
    symbolic: &SparseLuSymbolic,
    pivot_threshold: T::Real,
    order: PanelOrder,
    parallel: ParallelWaves,
) -> Result<NumericFactors<T>, SparseLuError> {
    let n = matrix.num_rows;
    if n != symbolic.n || matrix.num_cols != symbolic.n || !pattern_matches(matrix, symbolic) {
        return Err(SparseLuError::PatternMismatch);
    }
    if !pivot_threshold.is_finite() {
        return Err(SparseLuError::InvalidPivotThreshold);
    }
    let col_perm = symbolic.col_perm.clone();
    let panels = &symbolic.panels;

    let pivot_threshold = pivot_threshold.max(T::Real::zero()).min(T::Real::one());
    let matrix_scale = matrix
        .values
        .iter()
        .fold(T::Real::zero(), |scale, value| scale.max(value.norm()));
    let dimension = T::Real::from_usize(n.max(1)).ok_or(SparseLuError::NumericalBreakdown {
        stage: "supernodal pivot tolerance dimension conversion",
        index: n,
    })?;
    let pivot_tol =
        T::Real::epsilon() * dimension * matrix_scale.max(T::Real::min_positive_value());
    if !pivot_tol.is_finite() {
        return Err(SparseLuError::NumericalBreakdown {
            stage: "supernodal pivot tolerance formation",
            index: n,
        });
    }

    // Execution order: ascending panel index (legacy) or static waves
    // (ascending wave, ascending panel index within a wave). Wave panels
    // are ST-disjoint, so any wave-respecting order yields identical
    // factors; the ascending-within-wave tie-break keeps the order total
    // and deterministic. Empty levels (census skipped on the serial
    // factorize path) mean one ascending wave: serial wave execution is
    // exactly ascending panel order via the nondecreasing floor.
    let execution_order: Vec<usize> = match order {
        PanelOrder::Ascending => (0..panels.len()).collect(),
        PanelOrder::Waves if symbolic.panel_levels.is_empty() => (0..panels.len()).collect(),
        PanelOrder::Waves => {
            let mut wave_order: Vec<usize> = (0..panels.len()).collect();
            wave_order.sort_by_key(|&panel| (symbolic.panel_levels[panel], panel));
            wave_order
        }
    };
    // Per-panel staged factors (merged ascendingly after all waves) and
    // panel-indexed completed panels: wave execution completes panels out
    // of ascending order, so completion-indexed storage would misalign.
    let mut panel_factors: Vec<Option<PanelFactors<T>>> = (0..panels.len()).map(|_| None).collect();
    let mut completed_panels: Vec<Option<CompletedPanel<T>>> =
        (0..panels.len()).map(|_| None).collect();

    // Etree-pruned chunk application setup: a completed chunk can only
    // contribute a nonzero update to the current panel if one of its columns
    // lies on the Gilbert-Peierls reach of a current column, and every such
    // reach column sits on an etree ancestor chain starting from a column
    // that shares an original row with the current panel (same argument as
    // the simplicial backend's reach computation). Chunks off every chain
    // are skipped exactly — their update would be entirely zero. Flags are
    // per chunk (not per panel) so divergent structures inside one panel
    // prune independently.
    let mut row_to_cols: Vec<Vec<usize>> = vec![Vec::new(); n];
    for (column, entries) in symbolic.column_scatter.iter().enumerate() {
        for &(original_row, _) in entries {
            row_to_cols[original_row].push(column);
        }
    }
    // Flat chunk ids: `chunk_offsets[p]` is the first global chunk id of
    // panel `p`; `chunk_of_column` maps every column to its chunk.
    let mut chunk_offsets: Vec<usize> = Vec::with_capacity(panels.len() + 1);
    let mut chunk_of_column = vec![0usize; n];
    let mut next_chunk = 0usize;
    for (panel_index, panel_range) in panels.iter().enumerate() {
        chunk_offsets.push(next_chunk);
        for &(chunk_start, chunk_end) in &symbolic.panel_chunks[panel_index] {
            debug_assert!(chunk_start >= panel_range.0 && chunk_end <= panel_range.1);
            chunk_of_column[chunk_start..chunk_end].fill(next_chunk);
            next_chunk += 1;
        }
    }
    chunk_offsets.push(next_chunk);
    let total_chunks = next_chunk;
    // Global-chunk -> completed-panel source index, assigned per panel on
    // completion (runtime chunks match symbolic chunks 1:1 in order, so the
    // global runtime id equals the symbolic id). Pre-sized with a sentinel:
    // wave execution completes panels out of ascending order, so appending
    // would misalign against the panel-indexed `chunk_offsets`. Backs the
    // per-panel contributor-liveness stamps below.
    let mut chunk_source: Vec<usize> = vec![usize::MAX; total_chunks];

    // Panel workspace dimensions (the workspace itself now lives behind
    // `PanelWorkspace`, reused across panels in serial execution and
    // per-thread in parallel waves).
    let max_panel_width = panels
        .iter()
        .map(|&(start, end)| end - start)
        .max()
        .unwrap_or(0);

    let mut workspace = PanelWorkspace::new(n, total_chunks, panels.len(), max_panel_width);
    let mut mapping = RowMapping {
        perm: (0..n).collect(),
        inv: (0..n).collect(),
    };
    // Group the execution order into waves (levels are nondecreasing in
    // both orders, so groups are ascending-consecutive runs). Empty levels
    // run everything as one ascending wave (serial path only).
    let mut waves: Vec<Vec<usize>> = Vec::new();
    if symbolic.panel_levels.is_empty() {
        waves.push(execution_order.clone());
    } else {
        for &panel in &execution_order {
            match waves.last_mut() {
                Some(wave) if symbolic.panel_levels[wave[0]] == symbolic.panel_levels[panel] => {
                    wave.push(panel)
                }
                _ => waves.push(vec![panel]),
            }
        }
    }
    let workspace_pool: Mutex<Vec<PanelWorkspace<T>>> = Mutex::new(Vec::new());
    let mut validation_marks = vec![0usize; n];
    let mut validation_gen = 0usize;
    let threads_available = rayon::current_num_threads() > 1;
    let par_above_floor = n >= numeric_par_min_n();
    let debug_waves = numeric_debug_enabled();
    let mut par_wave_count = 0usize;
    let mut redo_count = 0usize;
    let mut redo_panel_count = 0usize;
    let mut violation_count = 0usize;
    for wave in &waves {
        // The ascending-order legacy path stays purely serial; parallel
        // waves additionally need multiple panels, the size gate (unless
        // forced), and more than one rayon thread.
        // `Always` bypasses every gate (including the thread gate, so
        // tests exercise the parallel machinery deterministically even
        // single-threaded); `Auto` needs all of them.
        let go_parallel = order == PanelOrder::Waves
            && parallel != ParallelWaves::Never
            && wave.len() > 1
            && (parallel == ParallelWaves::Always || (threads_available && par_above_floor));
        if !go_parallel {
            factor_wave_serial(
                wave,
                matrix,
                symbolic,
                &mut completed_panels,
                &mut chunk_source,
                &chunk_offsets,
                &chunk_of_column,
                &row_to_cols,
                &mut panel_factors,
                &mut mapping,
                &mut workspace,
                pivot_threshold,
                pivot_tol,
            )?;
            continue;
        }
        // Parallel attempt: each panel factors against a private wave-entry
        // mapping clone with a pooled thread workspace, recording exact
        // touched/swap sets. The attempt is side-effect free on shared
        // state (results are collected, never committed in place).
        par_wave_count += 1;
        let entry_perm = mapping.perm.clone();
        let entry_inv = mapping.inv.clone();
        let mut attempts: Vec<WaveAttempt<T>> = wave
            .par_iter()
            .map(|&panel_index| {
                let mut thread_workspace = checkout_workspace(
                    &workspace_pool,
                    n,
                    total_chunks,
                    panels.len(),
                    max_panel_width,
                );
                let mut thread_mapping = RowMapping {
                    perm: entry_perm.clone(),
                    inv: entry_inv.clone(),
                };
                let mut record = PanelRecord::default();
                let outcome = factor_panel(
                    panel_index,
                    matrix,
                    symbolic,
                    &completed_panels,
                    &chunk_source,
                    &chunk_offsets,
                    &chunk_of_column,
                    &row_to_cols,
                    &mut thread_mapping,
                    &mut thread_workspace,
                    pivot_threshold,
                    pivot_tol,
                    Some(&mut record),
                );
                match outcome {
                    Ok((factors, completed)) => {
                        // Only success-path workspaces return to the pool:
                        // `factor_panel` restores its all-zero invariant at
                        // panel end, but the early-error returns skip that
                        // clear. A dirty workspace must never be reused —
                        // stale `active` flags suppress re-registration of
                        // scattered rows (so their values are never cleared
                        // either) and corrupt every later panel checked out
                        // from it. Errors are rare (the wave redoes
                        // serially), so dropping regrows the pool cheaply.
                        checkin_workspace(&workspace_pool, thread_workspace);
                        (panel_index, Ok((factors, completed, record)))
                    }
                    Err(error) => (panel_index, Err(error)),
                }
            })
            .collect();
        // Deterministic ascending commit order regardless of rayon
        // scheduling (collect preserves order, but sort defensively).
        attempts.sort_by_key(|&(panel_index, _)| panel_index);
        // Any error redoes the whole wave serially (yielding serial
        // semantics, including first-error precedence). Otherwise exact
        // validation ascendingly: each panel must not read any original
        // swapped by an ascendingly-earlier same-wave panel (it factored
        // without those swaps; serial-wave order would have included
        // them). The clean ascending prefix commits; only the suffix from
        // the first violation redoes serially (same-wave panels share no
        // contributor relation, so redone panels depend solely on the
        // committed prefix state, which matches serial-wave order).
        let any_error = attempts.iter().any(|(_, result)| result.is_err());
        let mut first_violation: Option<usize> = None;
        if !any_error {
            validation_gen += 1;
            for (position, (_, result)) in attempts.iter().enumerate() {
                let Ok((_, _, record)) = result.as_ref() else {
                    first_violation = Some(position);
                    break;
                };
                let mut violated = false;
                for &row in &record.touched {
                    if validation_marks[row] == validation_gen {
                        violated = true;
                        break;
                    }
                }
                if violated {
                    if first_violation.is_none() {
                        first_violation = Some(position);
                    }
                    violation_count += 1;
                    continue;
                }
                for &(first, second) in &record.swaps {
                    validation_marks[first] = validation_gen;
                    validation_marks[second] = validation_gen;
                }
            }
        }
        if any_error {
            factor_wave_serial(
                wave,
                matrix,
                symbolic,
                &mut completed_panels,
                &mut chunk_source,
                &chunk_offsets,
                &chunk_of_column,
                &row_to_cols,
                &mut panel_factors,
                &mut mapping,
                &mut workspace,
                pivot_threshold,
                pivot_tol,
            )?;
            continue;
        }
        // Split at the first violation (whole wave when clean): commit the
        // clean prefix ascendingly (replaying recorded swaps in
        // application order — validation proved them disjoint from every
        // later committed panel's reads, so each replay matches that
        // panel's private application), then redo the suffix serially.
        let split = first_violation.unwrap_or(attempts.len());
        if split < attempts.len() {
            redo_count += 1;
            redo_panel_count += attempts.len() - split;
        }
        let mut suffix = Vec::with_capacity(attempts.len() - split);
        for (position, (panel_index, result)) in attempts.into_iter().enumerate() {
            if position < split {
                let Ok((factors, completed, record)) = result else {
                    unreachable!("clean prefix validated without errors");
                };
                for &(first, second) in &record.swaps {
                    let position_first = mapping.inv[first];
                    let position_second = mapping.inv[second];
                    mapping.perm.swap(position_first, position_second);
                    mapping.inv[first] = position_second;
                    mapping.inv[second] = position_first;
                }
                commit_panel(
                    panel_index,
                    factors,
                    completed,
                    &mut chunk_source,
                    &chunk_offsets,
                    &mut completed_panels,
                    &mut panel_factors,
                );
            } else {
                suffix.push(panel_index);
            }
        }
        if !suffix.is_empty() {
            factor_wave_serial(
                &suffix,
                matrix,
                symbolic,
                &mut completed_panels,
                &mut chunk_source,
                &chunk_offsets,
                &chunk_of_column,
                &row_to_cols,
                &mut panel_factors,
                &mut mapping,
                &mut workspace,
                pivot_threshold,
                pivot_tol,
            )?;
        }
    }
    if debug_waves {
        eprintln!(
            "[supernodal] waves={} par_waves={} redos={} redo_panels={} violations={}",
            waves.len(),
            par_wave_count,
            redo_count,
            redo_panel_count,
            violation_count
        );
    }
    // Ascending merge of the per-panel staged factors into global CSC
    // storage: each column's block stays contiguous in ascending column
    // order regardless of execution order. Counting assembly: one pass
    // over staged column pointers yields exact per-column counts, the
    // prefix sum sizes every vector once, and one scatter pass copies
    // each block to its final offset — no Vec growth copies and no
    // per-column extend dispatch. Same columns, same order, same values:
    // bit-identical to the extend loop.
    let mut l_col_ptrs = vec![0usize; n + 1];
    let mut u_col_ptrs = vec![0usize; n + 1];
    for panel_index in 0..panels.len() {
        // Every panel runs exactly once (the execution order is a
        // permutation); the error arm is unreachable-by-construction.
        let Some(staged) = panel_factors[panel_index].as_ref() else {
            return Err(SparseLuError::NumericalBreakdown {
                stage: "supernodal wave merge incomplete",
                index: panel_index,
            });
        };
        let (panel_start, panel_end) = panels[panel_index];
        for local_column in 0..(panel_end - panel_start) {
            let column = panel_start + local_column;
            l_col_ptrs[column + 1] = staged.l_ptr[local_column + 1] - staged.l_ptr[local_column];
            u_col_ptrs[column + 1] = staged.u_ptr[local_column + 1] - staged.u_ptr[local_column];
        }
    }
    for column in 0..n {
        l_col_ptrs[column + 1] += l_col_ptrs[column];
        u_col_ptrs[column + 1] += u_col_ptrs[column];
    }
    let l_total = l_col_ptrs[n];
    let u_total = u_col_ptrs[n];
    let mut l_row_indices = vec![0usize; l_total];
    let mut l_values = vec![T::zero(); l_total];
    let mut u_row_indices = vec![0usize; u_total];
    let mut u_values = vec![T::zero(); u_total];
    for panel_index in 0..panels.len() {
        let Some(staged) = panel_factors[panel_index].take() else {
            return Err(SparseLuError::NumericalBreakdown {
                stage: "supernodal wave merge incomplete",
                index: panel_index,
            });
        };
        // One copy per panel per array: staged column blocks are
        // contiguous across the panel's columns and the destination span
        // covers the panel's consecutive columns, so the whole panel moves
        // in four copies instead of four per column (identical bytes).
        let (panel_start, panel_end) = panels[panel_index];
        let panel_width = panel_end - panel_start;
        let l_len = staged.l_ptr[panel_width];
        let u_len = staged.u_ptr[panel_width];
        let dest_l = l_col_ptrs[panel_start]..l_col_ptrs[panel_end];
        let dest_u = u_col_ptrs[panel_start]..u_col_ptrs[panel_end];
        debug_assert_eq!(dest_l.len(), l_len);
        debug_assert_eq!(dest_u.len(), u_len);
        l_row_indices[dest_l.clone()].copy_from_slice(&staged.l_rows[..l_len]);
        l_values[dest_l].copy_from_slice(&staged.l_vals[..l_len]);
        u_row_indices[dest_u.clone()].copy_from_slice(&staged.u_rows[..u_len]);
        u_values[dest_u].copy_from_slice(&staged.u_vals[..u_len]);
    }
    Ok(NumericFactors {
        l_col_ptrs,
        l_row_indices,
        l_values,
        u_col_ptrs,
        u_row_indices,
        u_values,
        row_perm: mapping.perm,
        col_perm,
        supernode_count: symbolic.supernodes.len(),
    })
}

fn pattern_signature<T: ComplexField>(matrix: &CsrMatrix<T>) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for value in matrix
        .row_ptrs
        .iter()
        .chain(matrix.col_indices.iter())
        .copied()
    {
        hash ^= value as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    hash
}

/// Exact CSR pattern validation for symbolic reuse. The FNV hash is checked
/// first as a fast negative filter, but equality is always confirmed by
/// comparing `row_ptrs` and `col_indices` element by element — a hash match
/// alone never validates a pattern.
fn pattern_matches<T: ComplexField>(matrix: &CsrMatrix<T>, symbolic: &SparseLuSymbolic) -> bool {
    matrix.row_ptrs.len() == symbolic.pattern_row_ptrs.len()
        && matrix.col_indices.len() == symbolic.pattern_col_indices.len()
        && pattern_signature(matrix) == symbolic.pattern_signature
        && matrix.row_ptrs == symbolic.pattern_row_ptrs
        && matrix.col_indices == symbolic.pattern_col_indices
}

/// In-place panel TRSM without touching the shared target rows: solves
/// `U_source,target = L^-1 * W` directly in the contributor's own (disjoint)
/// workspace rows.
///
/// In-place is exact: with unit-diagonal `L`, solved row `row` reads only
/// already-solved rows `previous < row`, so no private copy is needed — the
/// same arithmetic runs in the same order as the former copy-solve-writeback
/// sequence, bit-identical, minus two `source_width x target_width` passes.
///
/// The solve runs over the WHOLE panel even though the GEMM/scatter below
/// splits per chunk: the inter-chunk `L_21` coupling inside the source panel
/// must propagate through the solve (solving each chunk against only its own
/// diagonal block drops it and breaks exactness — caught by
/// `supernodal_matches_simplicial_on_nonsymmetric_laplacian` with
/// reconstruction error ~0.19).
///
/// `work_slice` covers exactly rows `base_row..base_row + source_width`
/// with `target_width` stride. `touched` is cleared on entry and collects
/// the nonzero solved rows, so the caller can merge them in order.
/// Returns false when the contributor is vacuous (all-zero slice).
fn compute_panel_update<T: ComplexField>(
    source: &CompletedPanel<T>,
    target_width: usize,
    work_slice: &mut [T],
    base_row: usize,
    touched: &mut Vec<usize>,
) -> bool {
    let source_width = source.end - source.start;
    debug_assert_eq!(base_row, source.start);
    debug_assert_eq!(work_slice.len(), source_width * target_width);
    touched.clear();
    if work_slice.iter().all(|value| *value == T::zero()) {
        return false;
    }

    // U_source,target = L_source,source^-1 * W_source,target, one vectorized
    // AXPY per (row, previous) pair: `row -= multiplier * previous` runs
    // through `T::vec_axpy` (SIMD on f32/f64/complex), replacing the scalar
    // inner loop. Bit-identical: one rounding per multiply-add in the same
    // order (negation is exact). Measured ~7ms faster than the scalar loop
    // at n=16384 (interleaved same-binary A/B, non-overlapping 5-rep means).
    //
    // NOTE (inline attempt, reverted): an inline scalar loop over the
    // <=16-wide segments measured neutral in a same-binary interleaved A/B
    // (111.7 vs 111.5ms means, within noise) — the TRSM streams target rows
    // from L2/L3, so kernel choice is irrelevant; keep the FMA kernel for
    // its rounding.
    for row in 0..source_width {
        let (done, todo) = work_slice.split_at_mut(row * target_width);
        let row_part = &mut todo[..target_width];
        for previous in 0..row {
            let multiplier = source.diagonal_l[row * source_width + previous];
            if multiplier == T::zero() {
                continue;
            }
            let prev_part = &done[previous * target_width..(previous + 1) * target_width];
            T::vec_axpy(-multiplier, prev_part, row_part);
        }
        if row_part.iter().any(|value| *value != T::zero()) {
            touched.push(source.start + row);
        }
    }
    true
}

/// One chunk's GEMM from the solved panel block: `update = L_chunk *
/// U_chunk_rows`, where `U_chunk_rows` are the chunk's rows of the
/// panel-wide solved `work_slice` (solved by `compute_panel_update`).
/// `update_buf` is driver-owned scratch reused across applications (grown on
/// demand; every entry read below is fully overwritten first).
/// Returns the length of the valid prefix of `update_buf` (`None` when the
/// chunk contributes nothing: empty below-set or all-zero solved rows).
///
/// NOTE (live-row gather attempt, reverted): solved-block zeros are
/// element-scattered, not row-structured — per-row liveness caught almost
/// nothing (chunk flops 134M -> 130M) while gather traffic plus smaller,
/// less-efficient GEMMs cost ~9ms net (111 -> 120ms). Only exact-pattern
/// (Gustavson-style) updates could beat the 46%-dense rectangles, which is
/// a different algorithm, not a kernel tweak.
///
/// NOTE (exact-pattern Gustavson attempt, reverted 2026-09-25): TWO fused
/// variants were tried — k-major axpy (per nonzero solved entry, axpy the L
/// column into permuted `work` rows: 314.5 ms) and slot-major dots (per
/// (below-row, target) pair, dot the dense L segment against nonzero solved
/// entries, one `work` RMW per pair: 236.5 ms) vs the dense GEMM+scatter
/// baseline (~109 ms). Both kept factor_nnz identical and the residual in
/// the same class (3.62e-14 vs 3.63e-14), so the pattern math was right —
/// the density regime is wrong: at 46%-dense rectangles the software
/// zero-skip branch costs more per element than the vectorized FMA it
/// saves, and the fused permuted writes lose the GEMM's contiguous streams
/// plus the buffer-write amortization. Exact-pattern apply only wins at low
/// density (<10-20%); here the rectangles must get FEWER (better fill /
/// frontal structure), not cheaper. Keep the dense path.
fn compute_chunk_update<T: ComplexField>(
    chunk: &CompletedPanelChunk<T>,
    target_width: usize,
    solved: &[T],
    panel_start: usize,
    update_buf: &mut Vec<T>,
) -> Option<usize> {
    let chunk_width = chunk.end - chunk.start;
    if chunk.below_original_rows.is_empty() {
        return None;
    }
    // Value-level skip: a zero solved block gives a zero update. (The
    // pattern-level skip via `needed` flags happens in the driver.)
    let chunk_base = (chunk.start - panel_start) * target_width;
    if solved[chunk_base..chunk_base + chunk_width * target_width]
        .iter()
        .all(|value| *value == T::zero())
    {
        return None;
    }
    // The chunk's solved rows are consecutive in the panel slice, so the
    // GEMM reads them in place with a row offset (no gather needed).
    let update_len = chunk.below_original_rows.len() * target_width;
    // Growth-only reuse: `mat_mul_nn` overwrites all of `C` by contract, so
    // only the buffer length matters — stale contents are never read (the
    // GEMM writes exactly `..update_len` and the scatter consumes exactly
    // that prefix). Skips the per-dispatch re-zeroing; bit-identical.
    if update_buf.len() < update_len {
        update_buf.resize(update_len, T::zero());
    }
    T::mat_mul_nn(
        chunk.below_original_rows.len(),
        target_width,
        chunk_width,
        &chunk.lower,
        &solved[chunk_base..chunk_base + chunk_width * target_width],
        &mut update_buf[..update_len],
    );
    Some(update_len)
}

/// Serial scatter of one contributor's update buffer into the shared
/// target rows, in caller-provided order.
///
/// NOTE (scatter ordering, reverted): visiting rows in current-row
/// ascending order via a per-application slot sort was tried to make the
/// `work` read-modify-write stream prefetch sequentially. It measured ~7%
/// slower overall at n=16384: unlike the once-per-panel `active_rows`
/// sort, the sort runs per contributor application, and its O(b log b)
/// cost plus the order-buffer traffic exceeds the prefetch gain on these
/// cache-resident row segments. Keep original order.
fn scatter_panel_update<T: ComplexField>(
    chunk: &CompletedPanelChunk<T>,
    update: &[T],
    target_width: usize,
    row_perm_inv: &[usize],
    work: &mut [T],
    active: &mut [bool],
    active_rows: &mut Vec<usize>,
    mut record_touched: Option<&mut Vec<usize>>,
) {
    for (row_slot_index, &original_row) in chunk.below_original_rows.iter().enumerate() {
        let row = row_perm_inv[original_row];
        for target in 0..target_width {
            work[row * target_width + target] -= update[row_slot_index * target_width + target];
        }
        mark_active(row, active, active_rows);
        if let Some(touched) = record_touched.as_mut() {
            touched.push(original_row);
        }
    }
}

/// Apply one contributing panel: a single panel-wide TRSM (exact, see
/// `compute_panel_update`), then one tight GEMM/scatter per chunk whose
/// `needed` flag is set. `needed` covers exactly this panel's chunks.
/// `record_touched` (when set) collects the scattered below-original-rows
/// for wave validation.
#[allow(clippy::too_many_arguments)]
fn apply_completed_panel<T: ComplexField>(
    source: &CompletedPanel<T>,
    target_width: usize,
    row_perm_inv: &[usize],
    work: &mut [T],
    active: &mut [bool],
    active_rows: &mut Vec<usize>,
    update_buf: &mut Vec<T>,
    touched: &mut Vec<usize>,
    needed: &[bool],
    mut record_touched: Option<&mut Vec<usize>>,
) {
    // The driver calls only liveness-stamped sources (proven exact:
    // TEMP census held zero vacuous entries), so at least one flag is set.
    debug_assert!(needed.iter().any(|&flag| flag));
    let source_width = source.end - source.start;
    let base = source.start * target_width;
    // NOTE (pre-TRSM vacuity skip, reverted 2026-09-25): scanning the work
    // block before the TRSM to skip vacuous applies LOST by ~3.6ms in a
    // same-binary interleaved A/B (98.9 vs 95.3ms means). The full-slice
    // pre-scan costs 11.5ms (no early exit on all-zero blocks) while the
    // TRSMs it saves were ~free (L1-hot diagonal skips, 0.3ms); the old
    // in-TRSM early-out already skipped the chunk loop for empties, so
    // there was almost nothing left to save. Reach tightening (fewer
    // empty applies), not cheaper detection, is the lever.
    let live = compute_panel_update(
        source,
        target_width,
        &mut work[base..base + source_width * target_width],
        source.start,
        touched,
    );
    if !live {
        return;
    }
    for &row in touched.iter() {
        mark_active(row, active, active_rows);
    }
    // Each chunk GEMM reborrows the solved block immutably (its rows are
    // consecutive in the panel slice — no gather); each scatter reborrows
    // `work` mutably. The borrows are sequential, never overlapping: chunk
    // scatters only touch below-panel rows, disjoint from the solved rows.
    for (chunk, &flag) in source.chunks.iter().zip(needed.iter()) {
        if !flag {
            continue;
        }
        let update_len = {
            let solved = &work[base..base + source_width * target_width];
            compute_chunk_update(chunk, target_width, solved, source.start, update_buf)
        };
        if let Some(len) = update_len {
            scatter_panel_update(
                chunk,
                &update_buf[..len],
                target_width,
                row_perm_inv,
                work,
                active,
                active_rows,
                record_touched.as_deref_mut(),
            );
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn pack_completed_panel<T: ComplexField>(
    start: usize,
    end: usize,
    chunks: &[(usize, usize)],
    // Panel-relative L storage: `panel_l_ptr[column - start]` brackets the
    // panel's own column (the driver stages each panel locally and merges
    // ascendingly after all waves).
    panel_l_ptr: &[usize],
    l_row_indices: &[usize],
    l_values: &[T],
    row_perm_inv: &[usize],
    // Driver-owned scratch reused across panels (avoids an O(n) allocation
    // per panel). Stale entries are never read: each chunk's first loop
    // writes `row_slot` for exactly the original rows its second loop looks
    // up (same chunk columns, same `row >= end` condition), so every read
    // hits a freshly written slot.
    row_slot: &mut [usize],
    // Generation-stamped dedup marks for the below-row collection (one
    // `u32` per original row, zeroed once by the driver; `below_gen`
    // advances per chunk and never wraps in practice).
    below_mark: &mut [u32],
    below_gen: &mut u32,
) -> CompletedPanel<T> {
    debug_assert_eq!(row_slot.len(), row_perm_inv.len());
    // Full panel triangular block for the single panel-wide TRSM (see
    // `compute_panel_update`): the strictly-lower part including the
    // inter-chunk coupling; the diagonal stays zero for unit L.
    let panel_width = end - start;
    let mut diagonal_l = vec![T::zero(); panel_width * panel_width];
    for column in start..end {
        for index in panel_l_ptr[column - start]..panel_l_ptr[column - start + 1] {
            let original_row = l_row_indices[index];
            let row = row_perm_inv[original_row];
            if row < end {
                diagonal_l[(row - start) * panel_width + (column - start)] = l_values[index];
            }
        }
    }
    let mut packed = Vec::with_capacity(chunks.len());
    for &(chunk_start, chunk_end) in chunks {
        let width = chunk_end - chunk_start;
        // Stamp-dedup collection in first-seen (column-major) order: no
        // consumer requires sorted below rows — the GEMM and the scatter
        // both follow `below_original_rows` consistently, each target row is
        // scattered exactly once per chunk, and chunk order is unchanged —
        // so values are bit-identical to the old sort+dedup path while the
        // O(b log b) sort and the separate slot-assignment pass disappear.
        // `row_slot` is assigned inline during the same pass.
        *below_gen = below_gen.wrapping_add(1);
        if *below_gen == 0 {
            below_mark.fill(0);
            *below_gen = 1;
        }
        let stamp = *below_gen;
        let mut below_original_rows = Vec::new();
        for column in chunk_start..chunk_end {
            for &original_row in
                &l_row_indices[panel_l_ptr[column - start]..panel_l_ptr[column - start + 1]]
            {
                // Below-panel rows only; in-panel rows (including later
                // chunks' rows) were consumed by the in-panel updates.
                if row_perm_inv[original_row] >= end && below_mark[original_row] != stamp {
                    below_mark[original_row] = stamp;
                    row_slot[original_row] = below_original_rows.len();
                    below_original_rows.push(original_row);
                }
            }
        }
        let mut lower = vec![T::zero(); below_original_rows.len() * width];
        for column in chunk_start..chunk_end {
            for index in panel_l_ptr[column - start]..panel_l_ptr[column - start + 1] {
                let original_row = l_row_indices[index];
                let row = row_perm_inv[original_row];
                if row >= end {
                    lower[row_slot[original_row] * width + (column - chunk_start)] =
                        l_values[index];
                }
            }
        }
        packed.push(CompletedPanelChunk {
            start: chunk_start,
            end: chunk_end,
            below_original_rows,
            lower,
        });
    }
    CompletedPanel {
        start,
        end,
        diagonal_l,
        chunks: packed,
    }
}

/// Mark `row` touched, returning whether it was newly added (an append
/// breaks the ascending order of `active_rows`; the column loop tracks this
/// to re-sort lazily — see `active_sorted`).
fn mark_active(row: usize, active: &mut [bool], active_rows: &mut Vec<usize>) -> bool {
    if !active[row] {
        active[row] = true;
        active_rows.push(row);
        return true;
    }
    false
}

pub(super) fn relaxed_supernodes(etree: &[usize], max_width: usize) -> Vec<(usize, usize)> {
    let n = etree.len();
    let mut nodes = Vec::new();
    let mut start = 0;
    while start < n {
        let mut end = start + 1;
        while end < n && end - start < max_width && etree[end - 1] == end {
            end += 1;
        }
        nodes.push((start, end));
        start = end;
    }
    nodes
}

fn numeric_panels(supernodes: &[(usize, usize)], max_width: usize) -> Vec<(usize, usize)> {
    let mut panels = Vec::new();
    let mut cursor = 0;
    while cursor < supernodes.len() {
        let start = supernodes[cursor].0;
        let mut end = supernodes[cursor].1;
        cursor += 1;
        while cursor < supernodes.len() && supernodes[cursor].1 - start <= max_width {
            end = supernodes[cursor].1;
            cursor += 1;
        }
        panels.push((start, end));
    }
    panels
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn relaxed_supernodes_follow_parent_chains() {
        let etree = vec![1, 2, 3, usize::MAX, 5, usize::MAX];
        assert_eq!(relaxed_supernodes(&etree, 3), vec![(0, 3), (3, 4), (4, 6)]);
    }

    #[test]
    fn numeric_panels_combine_short_supernodes() {
        let supernodes = vec![(0, 2), (2, 3), (3, 6), (6, 10)];
        assert_eq!(
            numeric_panels(&supernodes, 5),
            vec![(0, 3), (3, 6), (6, 10)]
        );
    }

    fn fnv_values_hash(values: &[f64]) -> u64 {
        let mut hash = 0xcbf2_9ce4_8422_2325_u64;
        for &value in values {
            hash ^= value.to_bits();
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        hash
    }

    fn convection_diffusion_grid(grid: usize) -> CsrMatrix<f64> {
        let n = grid * grid;
        let mut row_ptrs = Vec::with_capacity(n + 1);
        let mut column_indices = Vec::with_capacity(5 * n);
        let mut values = Vec::with_capacity(5 * n);
        row_ptrs.push(0);
        for row in 0..grid {
            for column in 0..grid {
                let index = row * grid + column;
                if row > 0 {
                    column_indices.push(index - grid);
                    values.push(-1.1);
                }
                if column > 0 {
                    column_indices.push(index - 1);
                    values.push(-1.2);
                }
                column_indices.push(index);
                values.push(4.01);
                if column + 1 < grid {
                    column_indices.push(index + 1);
                    values.push(-0.8);
                }
                if row + 1 < grid {
                    column_indices.push(index + grid);
                    values.push(-0.9);
                }
                row_ptrs.push(column_indices.len());
            }
        }
        CsrMatrix::from_raw_parts(n, n, row_ptrs, column_indices, values)
    }

    #[test]
    fn chunk_update_buffer_reuse_preserves_factor_bytes() {
        // The growth-only update buffer relies on `mat_mul_nn` overwriting
        // `C` (trait contract): stale contents must never leak into factors.
        // Pins (nnz, value bits) on a grid with multi-chunk panels, i.e.
        // varying update lengths through the reused buffer. Golden re-pinned
        // 2026-09-28 for the merged BLAS/LAPACK kernels (last-ulp
        // reassociation; nnz/pivot/residual gates unchanged).
        let matrix = convection_diffusion_grid(12);
        let symbolic = analyze(&matrix).unwrap();
        assert!(
            symbolic.panel_chunks.iter().any(|chunks| chunks.len() > 1),
            "test matrix must exercise multi-chunk panels"
        );
        let numeric = factorize_with_symbolic(&matrix, &symbolic, 1.0).unwrap();
        assert_eq!(numeric.l_values.len(), 1155);
        assert_eq!(numeric.u_values.len(), 1299);
        assert_eq!(
            fnv_values_hash(&numeric.l_values) ^ fnv_values_hash(&numeric.u_values),
            0x8f39_62fe_24ef_fff7
        );
    }

    #[test]
    fn pattern_validation_rejects_single_entry_change() {
        // Two patterns with identical row_ptrs and nnz, differing in exactly
        // one column index. A hash-only check could in principle collide;
        // exact comparison must always reject.
        let first = CsrMatrix::from_raw_parts(
            4,
            4,
            vec![0, 2, 4, 6, 8],
            vec![0, 1, 0, 1, 1, 2, 2, 3],
            vec![2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        );
        let symbolic = analyze(&first).unwrap();

        let mut changed_cols = vec![0, 1, 0, 1, 1, 2, 2, 3];
        changed_cols[5] = 3; // row 2 now touches column 3 instead of column 2
        let second = CsrMatrix::from_raw_parts(
            4,
            4,
            vec![0, 2, 4, 6, 8],
            changed_cols,
            vec![2.0, 1.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        );
        assert!(matches!(
            factorize_with_symbolic(&second, &symbolic, 1.0),
            Err(SparseLuError::PatternMismatch)
        ));

        // Same pattern, different values: must validate and factorize.
        let third = CsrMatrix::from_raw_parts(
            4,
            4,
            vec![0, 2, 4, 6, 8],
            vec![0, 1, 0, 1, 1, 2, 2, 3],
            vec![4.0, 0.5, 0.5, 4.0, 0.5, 4.0, 0.5, 4.0],
        );
        assert!(factorize_with_symbolic(&third, &symbolic, 1.0).is_ok());
    }

    /// Contributor panels per panel, recomputed independently of wave
    /// construction by replaying the driver's needed-flag reach walk over
    /// the pattern. A panel is a contributor when any of its chunks is
    /// flagged for the target.
    fn panel_contributors(symbolic: &SparseLuSymbolic) -> Vec<Vec<usize>> {
        let n = symbolic.n;
        let panels = &symbolic.panels;
        let mut row_to_cols: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (column, entries) in symbolic.column_scatter.iter().enumerate() {
            for &(original_row, _) in entries {
                row_to_cols[original_row].push(column);
            }
        }
        let mut chunk_of_column = vec![0usize; n];
        let mut chunk_panel = Vec::new();
        for (panel_index, chunks) in symbolic.panel_chunks.iter().enumerate() {
            for &(chunk_start, chunk_end) in chunks {
                let chunk = chunk_panel.len();
                chunk_panel.push(panel_index);
                chunk_of_column[chunk_start..chunk_end].fill(chunk);
            }
        }
        let mut reach_marks = vec![0usize; n];
        let mut generation = 0usize;
        let mut contributors = vec![Vec::new(); panels.len()];
        for (panel_index, &(panel_start, panel_end)) in panels.iter().enumerate() {
            generation += 1;
            let mut seen = vec![false; panels.len()];
            for column in panel_start..panel_end {
                for &(original_row, _) in &symbolic.column_scatter[column] {
                    for &start in &row_to_cols[original_row] {
                        if start >= panel_start {
                            continue;
                        }
                        let mut current = start;
                        while reach_marks[current] != generation {
                            reach_marks[current] = generation;
                            if current < panel_start {
                                let source = chunk_panel[chunk_of_column[current]];
                                if source != panel_index && !seen[source] {
                                    seen[source] = true;
                                    contributors[panel_index].push(source);
                                }
                            }
                            let parent = symbolic.etree[current];
                            if parent == usize::MAX || parent >= panel_end {
                                break;
                            }
                            current = parent;
                        }
                    }
                }
            }
            contributors[panel_index].sort_unstable();
        }
        contributors
    }

    #[test]
    fn numeric_waves_respect_contributors_and_are_deterministic() {
        for grid in [8, 12, 16] {
            let matrix = convection_diffusion_grid(grid);
            let first = analyze(&matrix).unwrap();
            let second = analyze(&matrix).unwrap();
            assert_eq!(
                first.panel_levels, second.panel_levels,
                "wave schedule must be a pure function of the pattern"
            );
            assert_eq!(first.panel_levels.len(), first.panels.len());
            for window in first.panel_levels.windows(2) {
                assert!(
                    window[0] <= window[1],
                    "grid {grid}: levels must be nondecreasing (serial wave order == ascending)"
                );
            }
            let contributors = panel_contributors(&first);
            for (panel, sources) in contributors.iter().enumerate() {
                for &source in sources {
                    assert!(
                        first.panel_levels[source] < first.panel_levels[panel],
                        "grid {grid}: contributor {source} must precede panel {panel}"
                    );
                }
            }
        }
    }

    // Compact first-divergence report instead of full dumps.
    fn assert_numeric_factors_equal_diag<T: ComplexField + PartialEq + std::fmt::Debug>(
        left: &NumericFactors<T>,
        right: &NumericFactors<T>,
        tag: &str,
    ) {
        for (name, l, r) in [
            ("l_col_ptrs", &left.l_col_ptrs, &right.l_col_ptrs),
            ("u_col_ptrs", &left.u_col_ptrs, &right.u_col_ptrs),
            ("row_perm", &left.row_perm, &right.row_perm),
        ] {
            if l != r {
                let pos = l.iter().zip(r.iter()).position(|(a, b)| a != b);
                panic!("{tag}: {name} first diff at {pos:?}");
            }
        }
        if left.l_row_indices != right.l_row_indices {
            let pos = left
                .l_row_indices
                .iter()
                .zip(right.l_row_indices.iter())
                .position(|(a, b)| a != b);
            panic!("{tag}: l_row_indices first diff at {pos:?}");
        }
        if left.u_row_indices != right.u_row_indices {
            let pos = left
                .u_row_indices
                .iter()
                .zip(right.u_row_indices.iter())
                .position(|(a, b)| a != b);
            panic!("{tag}: u_row_indices first diff at {pos:?}");
        }
        if left.l_values != right.l_values {
            let pos = left
                .l_values
                .iter()
                .zip(right.l_values.iter())
                .position(|(a, b)| a != b)
                .unwrap();
            panic!(
                "{tag}: l_values first diff at {pos}: {:?} vs {:?}",
                left.l_values[pos], right.l_values[pos]
            );
        }
        if left.u_values != right.u_values {
            let pos = left
                .u_values
                .iter()
                .zip(right.u_values.iter())
                .position(|(a, b)| a != b)
                .unwrap();
            panic!(
                "{tag}: u_values first diff at {pos}: {:?} vs {:?}",
                left.u_values[pos], right.u_values[pos]
            );
        }
    }

    fn assert_numeric_factors_equal<T: ComplexField + PartialEq>(
        left: &NumericFactors<T>,
        right: &NumericFactors<T>,
    ) {
        assert_eq!(left.l_col_ptrs, right.l_col_ptrs);
        assert_eq!(left.l_row_indices, right.l_row_indices);
        assert_eq!(left.l_values, right.l_values);
        assert_eq!(left.u_col_ptrs, right.u_col_ptrs);
        assert_eq!(left.u_row_indices, right.u_row_indices);
        assert_eq!(left.u_values, right.u_values);
        assert_eq!(left.row_perm, right.row_perm);
        assert_eq!(left.col_perm, right.col_perm);
        assert_eq!(left.supernode_count, right.supernode_count);
    }

    #[test]
    fn numeric_waves_expose_parallelism_at_scale() {
        // Guards the wave census against a regression that serializes
        // everything: grid32 must already contain a multi-panel wave
        // (measured widths on convdiff: 16 -> max 2, 32 -> max 9,
        // 40 -> max 14, 64 -> max 38, 128 -> max 163).
        let matrix = convection_diffusion_grid(32);
        let symbolic = analyze(&matrix).unwrap();
        let mut widths: std::collections::BTreeMap<usize, usize> =
            std::collections::BTreeMap::new();
        for &level in &symbolic.panel_levels {
            *widths.entry(level).or_insert(0) += 1;
        }
        let max_width = widths.values().copied().max().unwrap_or(0);
        assert!(
            max_width > 1,
            "grid32 must expose a multi-panel wave, got widths {widths:?}"
        );
    }

    #[test]
    fn wave_order_matches_ascending_factors_bitwise() {
        // Wave execution reorders panels across ascending index order; the
        // ST-disjointness schedule must make that reorder result-neutral.
        // f64 corpus plus a complex spot check, at full and threshold (0.1)
        // partial pivoting (the reuse path uses 0.1).
        let corpus: Vec<CsrMatrix<f64>> = vec![
            convection_diffusion_grid(8),
            convection_diffusion_grid(12),
            convection_diffusion_grid(16),
            CsrMatrix::from_triplets(
                4,
                4,
                vec![
                    (0, 1, 2.0),
                    (0, 3, 1.0),
                    (1, 0, -2.0),
                    (1, 1, 1.0),
                    (1, 2, 1.0),
                    (2, 1, 1.0),
                    (2, 2, 3.0),
                    (2, 3, -2.0),
                    (3, 0, 1.0),
                    (3, 3, 2.0),
                ],
            ),
        ];
        for matrix in &corpus {
            let symbolic = analyze(matrix).unwrap();
            for threshold in [1.0, 0.1] {
                let ascending = factorize_with_symbolic_ordered(
                    matrix,
                    &symbolic,
                    threshold,
                    PanelOrder::Ascending,
                    ParallelWaves::Never,
                )
                .unwrap();
                let waves = factorize_with_symbolic_ordered(
                    matrix,
                    &symbolic,
                    threshold,
                    PanelOrder::Waves,
                    ParallelWaves::Never,
                )
                .unwrap();
                assert_numeric_factors_equal(&ascending, &waves);
            }
        }
        let real = convection_diffusion_grid(8);
        let mut triplets = Vec::with_capacity(real.values.len());
        for row in 0..real.num_rows {
            for index in real.row_ptrs[row]..real.row_ptrs[row + 1] {
                let value = real.values[index];
                triplets.push((
                    row,
                    real.col_indices[index],
                    num_complex::Complex64::new(value, 0.25 * value),
                ));
            }
        }
        let complex = CsrMatrix::from_triplets(real.num_rows, real.num_cols, triplets);
        let symbolic = analyze(&complex).unwrap();
        let ascending = factorize_with_symbolic_ordered(
            &complex,
            &symbolic,
            1.0,
            PanelOrder::Ascending,
            ParallelWaves::Never,
        )
        .unwrap();
        let waves = factorize_with_symbolic_ordered(
            &complex,
            &symbolic,
            1.0,
            PanelOrder::Waves,
            ParallelWaves::Never,
        )
        .unwrap();
        assert_numeric_factors_equal(&ascending, &waves);
    }

    #[test]
    fn parallel_waves_match_serial_factors_bitwise() {
        // Forced-parallel waves (gate bypassed) must produce field-identical
        // factors to the serial legacy path on the corpus, at both full
        // and threshold partial pivoting. Exact validation plus the serial
        // redo make any interference fall back instead of diverging.
        let corpus: Vec<CsrMatrix<f64>> = vec![
            convection_diffusion_grid(8),
            convection_diffusion_grid(12),
            convection_diffusion_grid(16),
            convection_diffusion_grid(24),
            // grid80 exercises the pooled-workspace error path: an errored
            // panel must not poison later checkouts (regression cover for
            // the drop-on-error fix).
            convection_diffusion_grid(80),
        ];
        for (matrix_index, matrix) in corpus.iter().enumerate() {
            let symbolic = analyze(matrix).unwrap();
            for threshold in [1.0, 0.1] {
                let tag = format!("matrix{matrix_index} threshold{threshold}");
                let serial = factorize_with_symbolic_ordered(
                    matrix,
                    &symbolic,
                    threshold,
                    PanelOrder::Ascending,
                    ParallelWaves::Never,
                )
                .unwrap();
                let parallel = factorize_with_symbolic_ordered(
                    matrix,
                    &symbolic,
                    threshold,
                    PanelOrder::Waves,
                    ParallelWaves::Always,
                )
                .unwrap();
                assert_numeric_factors_equal_diag(&serial, &parallel, &tag);
            }
        }
    }
}
