//! Iterative block solves for harmonic electroacoustic systems.
//!
//! The production path keeps the coupled operator in fluid/structure/electrical
//! blocks, applies homogeneous essential constraints without flattening it, and
//! uses right-preconditioned FGMRES.  By default the preconditioner factors
//! each physics block separately: the fluid factor is a complex
//! shifted-Laplacian while the structure and circuit factors use their dynamic
//! diagonal blocks.  [`CoupledPreconditioner::Monolithic`] instead factors the
//! fully assembled constrained system (coupling blocks included) and is the
//! fallback for near-resonant frequencies where the block factors stagnate.
//! A sweep object carries the previous solution and may reuse ILU factors over
//! a bounded frequency interval.

use std::collections::HashSet;

use num_complex::Complex64;
use sonium_solvers::iterative::{FgmresConfig, fgmres_preconditioned};
use sonium_solvers::preconditioners::{IluPreconditioner, IlutConfig, IlutPreconditioner};
use sonium_solvers::sparse::CsrMatrix;
use sonium_solvers::traits::{LinearOperator, Preconditioner};

use super::{CoupledSystem, CoupledSystemError, HarmonicCoupledPoint};
use crate::assembly::TripletMatrix;

/// Selects the ILU factorization strategy for the coupled preconditioner.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CoupledPreconditioner {
    /// One ILU per physics block (fluid/structure/electrical); cheapest, but
    /// ignores the coupling blocks. Near an undamped fluid/structure
    /// resonance the block approximation can stall above tight tolerances.
    BlockDiagonal,
    /// Single equilibrated ILU of the fully assembled constrained system.
    /// Includes the coupling blocks, so it stays effective at near-resonant
    /// frequencies where [`CoupledPreconditioner::BlockDiagonal`] stagnates.
    Monolithic,
}

/// Controls the production electroacoustic block solve.
#[derive(Debug, Clone, PartialEq)]
pub struct CoupledIterativeConfig {
    /// Number of restarted FGMRES cycles.
    pub max_restarts: usize,
    /// Krylov basis size per restart.
    pub restart: usize,
    /// Exact relative residual target. The coupled operator mixes pressure,
    /// stiffness, and circuit scales up to ~1e10 against a scalar-voltage
    /// right-hand side, so the attainable f64 floor of a compensated exact
    /// residual sits near 1e-9 (see the `correction_fgmres` documentation):
    /// a tighter target makes acceptance a rounding-noise coin flip. The
    /// default therefore sits a decade above the floor, in line with every
    /// consumer's QA gate (1e-8 or looser); the defect-correction loop still
    /// polishes accepted solutions towards `tolerance/10` whenever the floor
    /// allows, so typical residuals remain far below this target.
    pub relative_tolerance: f64,
    /// Real shifted-Laplacian mass coefficient, scaled by `k²`.
    pub acoustic_shift_real: f64,
    /// Imaginary shifted-Laplacian mass coefficient, scaled by `k²`.
    pub acoustic_shift_imag: f64,
    /// Largest relative frequency change for reusing existing ILU factors.
    pub preconditioner_reuse_band: f64,
    /// ILU factorization strategy.
    pub preconditioner: CoupledPreconditioner,
}

impl Default for CoupledIterativeConfig {
    fn default() -> Self {
        Self {
            max_restarts: 20,
            restart: 40,
            relative_tolerance: 1.0e-8,
            acoustic_shift_real: 0.5,
            acoustic_shift_imag: 0.5,
            preconditioner_reuse_band: 0.2,
            preconditioner: CoupledPreconditioner::BlockDiagonal,
        }
    }
}

impl CoupledIterativeConfig {
    pub fn validate(&self) -> Result<(), CoupledIterativeError> {
        if self.max_restarts == 0 || self.restart == 0 {
            return Err(CoupledIterativeError::InvalidConfiguration(
                "FGMRES restart counts must be non-zero",
            ));
        }
        if !self.relative_tolerance.is_finite()
            || self.relative_tolerance <= 0.0
            || !self.acoustic_shift_real.is_finite()
            || self.acoustic_shift_real < 0.0
            || !self.acoustic_shift_imag.is_finite()
            || self.acoustic_shift_imag <= 0.0
            || !self.preconditioner_reuse_band.is_finite()
            || !(0.0..=1.0).contains(&self.preconditioner_reuse_band)
        {
            return Err(CoupledIterativeError::InvalidConfiguration(
                "solver tolerances, shifts, and reuse band are invalid",
            ));
        }
        Ok(())
    }

    fn fgmres(&self) -> FgmresConfig<f64> {
        FgmresConfig {
            max_iterations: self.max_restarts,
            restart: self.restart,
            tolerance: self.relative_tolerance,
            print_interval: 0,
            convergence_check_interval: 1,
            // Never accept only the projected Krylov residual in this
            // multiphysics path: verify every apparent convergence.
            exact_residual_check_interval: 1,
        }
    }

    /// Configuration for the defect-correction solves.  A correction only
    /// needs a few digits — the outer refinement loop supplies the rest — but
    /// the projected Krylov residual is unreliable on this operator, so the
    /// tolerance must sit well above the true attainable floor (~1e-9
    /// relative) and keep the full restart budget: a tight tolerance or a
    /// small budget makes the solve thrash restarts without progress.
    fn correction_fgmres(&self) -> FgmresConfig<f64> {
        FgmresConfig {
            tolerance: 1.0e-6,
            ..self.fgmres()
        }
    }
}

/// Per-frequency convergence and reuse evidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CoupledSolveDiagnostics {
    pub iterations: usize,
    pub relative_residual: f64,
    pub converged: bool,
    pub initial_guess_reused: bool,
    pub preconditioner_rebuilt: bool,
    /// Number of defect-correction (iterative refinement) steps applied after
    /// the Krylov solve.
    pub defect_corrections: usize,
    /// Exact relative residual before any defect correction; equals
    /// `relative_residual` when no correction step ran.
    pub residual_before_correction: f64,
}

/// Maximum number of defect-correction steps after the Krylov solve, matching
/// the room FEM runner's refinement budget.
const MAX_DEFECT_CORRECTIONS: usize = 2;

/// Solution returned by [`CoupledFrequencyContinuation`].
#[derive(Debug, Clone, PartialEq)]
pub struct CoupledIterativeSolution {
    pub values: Vec<Complex64>,
    pub diagnostics: CoupledSolveDiagnostics,
}

/// Stateful frequency-continuation solver for one fixed coupled discretisation.
pub struct CoupledFrequencyContinuation<'a> {
    system: &'a CoupledSystem,
    config: CoupledIterativeConfig,
    previous_solution: Option<Vec<Complex64>>,
    preconditioner_frequency_hz: Option<f64>,
    preconditioner_constraints: Vec<usize>,
    preconditioner: Option<CoupledIluPreconditioner>,
}

impl<'a> CoupledFrequencyContinuation<'a> {
    pub fn new(
        system: &'a CoupledSystem,
        config: CoupledIterativeConfig,
    ) -> Result<Self, CoupledIterativeError> {
        system.validate()?;
        config.validate()?;
        Ok(Self {
            system,
            config,
            previous_solution: None,
            preconditioner_frequency_hz: None,
            preconditioner_constraints: Vec::new(),
            preconditioner: None,
        })
    }

    pub fn reset_continuation(&mut self) {
        self.previous_solution = None;
    }

    pub fn reset_preconditioner(&mut self) {
        self.preconditioner_frequency_hz = None;
        self.preconditioner_constraints.clear();
        self.preconditioner = None;
    }

    /// Solve one voltage- or force-driven point with homogeneous essential DOFs.
    pub fn solve(
        &mut self,
        point: HarmonicCoupledPoint,
        rhs: &[Complex64],
        homogeneous_constraints: &[usize],
    ) -> Result<CoupledIterativeSolution, CoupledIterativeError> {
        self.solve_with_load(
            point,
            rhs,
            homogeneous_constraints,
            Complex64::new(0.0, 0.0),
        )
    }

    /// Solve with a condensed external interface load applied to the solid
    /// block (see [`CoupledSystem::assemble_harmonic_with_load`]).
    ///
    /// ILU reuse policy: factor rebuilds are driven only by the configured
    /// frequency band and the constraint set, exactly as in [`Self::solve`];
    /// a changed `interface_load_scale` at the same frequency reuses the
    /// existing (lagged) factors. The load is a small damping-like boundary
    /// correction, so the lagged factors stay effective, and correctness is
    /// unaffected because acceptance is gated on the exact compensated
    /// residual of the freshly assembled (load-inclusive) system, with
    /// defect correction against that same system.
    pub fn solve_with_load(
        &mut self,
        point: HarmonicCoupledPoint,
        rhs: &[Complex64],
        homogeneous_constraints: &[usize],
        interface_load_scale: Complex64,
    ) -> Result<CoupledIterativeSolution, CoupledIterativeError> {
        point.validate()?;
        let n = self.system.total_dofs();
        if rhs.len() != n {
            return Err(CoupledIterativeError::DimensionMismatch {
                expected: n,
                actual: rhs.len(),
            });
        }
        let constraints = canonical_constraints(homogeneous_constraints, n)?;
        let mut constrained_rhs = rhs.to_vec();
        for &dof in &constraints {
            constrained_rhs[dof] = Complex64::new(0.0, 0.0);
        }

        let operator = self
            .system
            .assemble_harmonic_block_operator_with_load(point, interface_load_scale)?;
        let constrained_operator = HomogeneousConstraintOperator::new(&operator, n, &constraints);
        // Assembled CSR for the compensated residual/defect evaluation and,
        // in monolithic mode, the preconditioner factor.
        let matrix = self
            .system
            .assemble_harmonic_with_load(point, interface_load_scale)?
            .to_csr();
        let mut constrained = vec![false; n];
        for &dof in &constraints {
            constrained[dof] = true;
        }
        let frequency_hz = point.omega_rad_per_s / (2.0 * std::f64::consts::PI);
        let rebuild = self.should_rebuild_preconditioner(frequency_hz, &constraints);
        if rebuild {
            self.preconditioner = Some(match self.config.preconditioner {
                CoupledPreconditioner::BlockDiagonal => {
                    CoupledIluPreconditioner::Block(BlockIluPreconditioner::build(
                        self.system,
                        point,
                        &constraints,
                        &self.config,
                        interface_load_scale,
                    ))
                }
                CoupledPreconditioner::Monolithic => CoupledIluPreconditioner::Monolithic(
                    EquilibratedIlu::new_ilut(&constrain_csr(&matrix, &constraints)),
                ),
            });
            self.preconditioner_frequency_hz = Some(frequency_hz);
            self.preconditioner_constraints.clone_from(&constraints);
        }

        let initial_guess = self
            .previous_solution
            .as_deref()
            .filter(|guess| guess.len() == n);
        let initial_guess_reused = initial_guess.is_some();
        let preconditioner = self
            .preconditioner
            .as_ref()
            .expect("preconditioner is built before FGMRES");
        let result = fgmres_preconditioned(
            &constrained_operator,
            preconditioner,
            &constrained_rhs,
            initial_guess,
            &self.config.fgmres(),
        );
        let mut x = result.x;
        let mut iterations = result.iterations;
        let (mut defect, mut exact_residual) =
            compensated_constrained_residual(&matrix, &x, &constrained_rhs, &constrained);
        let residual_before_correction = exact_residual;
        // Defect correction (iterative refinement): Krylov recurrences can
        // stagnate at an attainable-accuracy floor just above the tolerance,
        // close enough that parallel-assembly reduction noise flips the gate
        // run to run.  Solving A·δ = b − A·x with the already-factored
        // preconditioner targets exactly the remaining error and is far more
        // reliable than restarting the original solve.  The defect must come
        // from the compensated evaluation: at this floor a plain f64 matvec
        // residual is mostly rounding noise, which is why refinement against
        // it cannot improve the iterate.  Refine whenever the residual is
        // above tolerance/10 so accepted solutions sit well below the gate
        // instead of on top of it.
        let correction_config = self.config.correction_fgmres();
        let compensated_operator = CompensatedCsrOperator { matrix: &matrix };
        let constrained_compensated =
            HomogeneousConstraintOperator::new(&compensated_operator, n, &constraints);
        let mut defect_corrections = 0;
        while defect_corrections < MAX_DEFECT_CORRECTIONS
            && exact_residual.is_finite()
            && exact_residual * 10.0 > self.config.relative_tolerance
        {
            // The correction solve runs against the compensated operator:
            // its defect right-hand side is of the same order as the plain
            // matvec rounding floor, so a Krylov method steered by plain
            // residuals would thrash against that floor instead of
            // converging.
            let correction = fgmres_preconditioned(
                &constrained_compensated,
                preconditioner,
                &defect,
                None,
                &correction_config,
            );
            iterations += correction.iterations;
            let mut refined = x.clone();
            for (value, delta) in refined.iter_mut().zip(correction.x) {
                *value += delta;
            }
            let (refined_defect, refined_residual) =
                compensated_constrained_residual(&matrix, &refined, &constrained_rhs, &constrained);
            // Never let a correction degrade the iterate.
            if !refined_residual.is_finite() || refined_residual >= exact_residual {
                break;
            }
            x = refined;
            defect = refined_defect;
            exact_residual = refined_residual;
            defect_corrections += 1;
        }
        if exact_residual > self.config.relative_tolerance || !exact_residual.is_finite() {
            return Err(CoupledIterativeError::ConvergenceFailure {
                iterations,
                relative_residual: exact_residual,
            });
        }

        self.previous_solution = Some(x.clone());
        Ok(CoupledIterativeSolution {
            values: x,
            diagnostics: CoupledSolveDiagnostics {
                iterations,
                relative_residual: exact_residual,
                converged: true,
                initial_guess_reused,
                preconditioner_rebuilt: rebuild,
                defect_corrections,
                residual_before_correction,
            },
        })
    }

    fn should_rebuild_preconditioner(&self, frequency_hz: f64, constraints: &[usize]) -> bool {
        let Some(reference) = self.preconditioner_frequency_hz else {
            return true;
        };
        if self.preconditioner.is_none() || self.preconditioner_constraints != constraints {
            return true;
        }
        let relative_change = (frequency_hz - reference).abs() / frequency_hz.max(reference);
        relative_change > self.config.preconditioner_reuse_band
    }
}

/// Relative complex-symmetry defect after applying the physical row scaling.
///
/// With RMS phasors and `e^(+iωt)`, scaling fluid rows by `1/(ρω²)` and
/// electrical rows by `i/ω` makes reciprocal acoustic/structural/motor
/// couplings complex symmetric.  This is a useful assembly QA invariant.
pub fn harmonic_reciprocity_defect(
    system: &CoupledSystem,
    point: HarmonicCoupledPoint,
) -> Result<f64, CoupledSystemError> {
    let matrix = system.assemble_harmonic(point)?.to_csr();
    let nf = system.n_fluid;
    let ns = system.n_solid;
    let omega = point.omega_rad_per_s;
    let fluid_scale = Complex64::new(1.0 / (point.fluid_density_kg_per_m3 * omega * omega), 0.0);
    let electrical_scale = Complex64::new(0.0, 1.0 / omega);
    let mut entries = std::collections::HashMap::new();
    for row in 0..matrix.num_rows {
        let scale = if row < nf {
            fluid_scale
        } else if row < nf + ns {
            Complex64::new(1.0, 0.0)
        } else {
            electrical_scale
        };
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            entries.insert(
                (row, matrix.col_indices[index]),
                scale * matrix.values[index],
            );
        }
    }
    let mut defect_squared = 0.0;
    let mut norm_squared = 0.0;
    let block = |dof: usize| {
        if dof < nf {
            0
        } else if dof < nf + ns {
            1
        } else {
            2
        }
    };
    for (&(row, col), &value) in &entries {
        if block(row) == block(col) {
            continue;
        }
        let transpose = entries
            .get(&(col, row))
            .copied()
            .unwrap_or_else(|| Complex64::new(0.0, 0.0));
        defect_squared += (value - transpose).norm_sqr();
        norm_squared += value.norm_sqr();
    }
    if norm_squared == 0.0 {
        return Ok(0.0);
    }
    Ok(defect_squared.sqrt() / norm_squared.sqrt())
}

struct HomogeneousConstraintOperator<'a, A: LinearOperator<Complex64>> {
    inner: &'a A,
    constrained: Vec<bool>,
}

impl<'a, A: LinearOperator<Complex64>> HomogeneousConstraintOperator<'a, A> {
    fn new(inner: &'a A, n: usize, constraints: &[usize]) -> Self {
        let mut constrained = vec![false; n];
        for &dof in constraints {
            constrained[dof] = true;
        }
        Self { inner, constrained }
    }
}

impl<A: LinearOperator<Complex64>> LinearOperator<Complex64>
    for HomogeneousConstraintOperator<'_, A>
{
    fn num_rows(&self) -> usize {
        self.inner.num_rows()
    }

    fn num_cols(&self) -> usize {
        self.inner.num_cols()
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        let mut constrained_x = x.to_vec();
        for (value, &is_constrained) in constrained_x.iter_mut().zip(&self.constrained) {
            if is_constrained {
                *value = Complex64::new(0.0, 0.0);
            }
        }
        self.inner.apply_into(&constrained_x, y);
        for ((output, &input), &is_constrained) in y.iter_mut().zip(x).zip(&self.constrained) {
            if is_constrained {
                *output = input;
            }
        }
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        let mut constrained_x = x.to_vec();
        for (value, &is_constrained) in constrained_x.iter_mut().zip(&self.constrained) {
            if is_constrained {
                *value = Complex64::new(0.0, 0.0);
            }
        }
        self.inner.apply_transpose_into(&constrained_x, y);
        for ((output, &input), &is_constrained) in y.iter_mut().zip(x).zip(&self.constrained) {
            if is_constrained {
                *output = input;
            }
        }
    }
}

/// Preconditioner built for the configured [`CoupledPreconditioner`] strategy.
enum CoupledIluPreconditioner {
    Block(BlockIluPreconditioner),
    Monolithic(EquilibratedIlu),
}

impl Preconditioner<Complex64> for CoupledIluPreconditioner {
    fn apply(&self, residual: &[Complex64]) -> Vec<Complex64> {
        let mut output = vec![Complex64::new(0.0, 0.0); residual.len()];
        self.apply_into(residual, &mut output);
        output
    }

    fn apply_into(&self, residual: &[Complex64], output: &mut [Complex64]) {
        match self {
            Self::Block(block) => block.apply_into(residual, output),
            Self::Monolithic(monolithic) => monolithic.apply_into(residual, output),
        }
    }
}

struct BlockIluPreconditioner {
    ranges: Vec<std::ops::Range<usize>>,
    blocks: Vec<EquilibratedIlu>,
}

impl BlockIluPreconditioner {
    fn build(
        system: &CoupledSystem,
        point: HarmonicCoupledPoint,
        constraints: &[usize],
        config: &CoupledIterativeConfig,
        interface_load_scale: Complex64,
    ) -> Self {
        let nf = system.n_fluid;
        let ns = system.n_solid;
        let mut ranges = vec![0..nf, nf..nf + ns];
        if let Some(electrical) = &system.electrical {
            ranges.push(nf + ns..nf + ns + electrical.n_dofs);
        }
        let omega = point.omega_rad_per_s;
        let k_squared = point.wavenumber_squared();
        let acoustic_shift = Complex64::new(
            config.acoustic_shift_real * k_squared,
            config.acoustic_shift_imag * k_squared,
        );
        let mut matrices = vec![
            combine_real_triplets(
                nf,
                &[
                    (&system.fluid_stiffness, Complex64::new(1.0, 0.0)),
                    (&system.fluid_mass, acoustic_shift),
                    (&system.fluid_damping, Complex64::new(0.0, omega)),
                ],
            ),
            system.harmonic_solid_block_with_load(omega, interface_load_scale),
        ];
        if let Some(electrical) = &system.electrical {
            matrices.push(combine_real_triplets(
                electrical.n_dofs,
                &[
                    (&electrical.resistance, Complex64::new(1.0, 0.0)),
                    (&electrical.inductance, Complex64::new(0.0, omega)),
                ],
            ));
        }
        for (range, matrix) in ranges.iter().zip(&mut matrices) {
            let local_constraints: Vec<_> = constraints
                .iter()
                .copied()
                .filter(|dof| range.contains(dof))
                .map(|dof| dof - range.start)
                .collect();
            *matrix = constrain_csr(matrix, &local_constraints);
        }
        let blocks = matrices.iter().map(EquilibratedIlu::new).collect();
        Self { ranges, blocks }
    }
}

/// Row equilibration is essential for mixed-unit block systems. Factoring
/// `D P` and applying `(D P)^-1 D r` is algebraically identical to `P^-1 r`
/// while keeping pressure, stiffness, and circuit scales away from ILU pivots.
struct EquilibratedIlu {
    factor: Box<dyn Preconditioner<Complex64> + Send + Sync>,
    row_scale: Vec<f64>,
}

impl EquilibratedIlu {
    fn new(matrix: &CsrMatrix<Complex64>) -> Self {
        Self::with_factor(matrix, |scaled| {
            Box::new(IluPreconditioner::from_csr(scaled))
        })
    }

    /// Factor with a tight ILUT (small drop tolerance, generous fill) instead
    /// of ILU(0).  Near an undamped resonance the block systems are hard
    /// enough that ILU(0)'s discarded fill dominates the Krylov cost.
    fn new_ilut(matrix: &CsrMatrix<Complex64>) -> Self {
        Self::with_factor(matrix, |scaled| {
            Box::new(IlutPreconditioner::from_csr_with_config(
                scaled,
                &IlutConfig::new(1.0e-6, 20, 1.0e-12),
            ))
        })
    }

    fn with_factor(
        matrix: &CsrMatrix<Complex64>,
        factor: impl FnOnce(&CsrMatrix<Complex64>) -> Box<dyn Preconditioner<Complex64> + Send + Sync>,
    ) -> Self {
        let row_scale: Vec<_> = (0..matrix.num_rows)
            .map(|row| {
                let norm = matrix.values[matrix.row_ptrs[row]..matrix.row_ptrs[row + 1]]
                    .iter()
                    .map(|value| value.norm_sqr())
                    .sum::<f64>()
                    .sqrt();
                1.0 / norm.max(f64::MIN_POSITIVE)
            })
            .collect();
        let mut scaled = matrix.clone();
        for (row, &scale) in row_scale.iter().enumerate() {
            for index in scaled.row_ptrs[row]..scaled.row_ptrs[row + 1] {
                scaled.values[index] *= scale;
            }
        }
        Self {
            factor: factor(&scaled),
            row_scale,
        }
    }

    fn apply_into(&self, residual: &[Complex64], output: &mut [Complex64]) {
        let scaled: Vec<_> = residual
            .iter()
            .zip(&self.row_scale)
            .map(|(&value, &scale)| value * scale)
            .collect();
        self.factor.apply_into(&scaled, output);
    }
}

impl Preconditioner<Complex64> for BlockIluPreconditioner {
    fn apply(&self, residual: &[Complex64]) -> Vec<Complex64> {
        let mut output = vec![Complex64::new(0.0, 0.0); residual.len()];
        self.apply_into(residual, &mut output);
        output
    }

    fn apply_into(&self, residual: &[Complex64], output: &mut [Complex64]) {
        assert_eq!(residual.len(), output.len());
        for (range, block) in self.ranges.iter().zip(&self.blocks) {
            block.apply_into(&residual[range.clone()], &mut output[range.clone()]);
        }
    }
}

fn combine_real_triplets(
    dimension: usize,
    terms: &[(&TripletMatrix<f64>, Complex64)],
) -> CsrMatrix<Complex64> {
    let capacity = terms.iter().map(|(matrix, _)| matrix.nnz()).sum();
    let mut triplets = Vec::with_capacity(capacity);
    for &(matrix, scale) in terms {
        for index in 0..matrix.nnz() {
            triplets.push((
                matrix.rows[index],
                matrix.cols[index],
                scale * matrix.values[index],
            ));
        }
    }
    CsrMatrix::from_triplets(dimension, dimension, triplets)
}

fn constrain_csr(matrix: &CsrMatrix<Complex64>, constraints: &[usize]) -> CsrMatrix<Complex64> {
    if constraints.is_empty() {
        return matrix.clone();
    }
    let constrained: HashSet<_> = constraints.iter().copied().collect();
    let mut triplets = Vec::with_capacity(matrix.values.len() + constraints.len());
    for row in 0..matrix.num_rows {
        if constrained.contains(&row) {
            triplets.push((row, row, Complex64::new(1.0, 0.0)));
            continue;
        }
        for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let col = matrix.col_indices[index];
            if !constrained.contains(&col) {
                triplets.push((row, col, matrix.values[index]));
            }
        }
    }
    CsrMatrix::from_triplets(matrix.num_rows, matrix.num_cols, triplets)
}

fn canonical_constraints(
    constraints: &[usize],
    dimension: usize,
) -> Result<Vec<usize>, CoupledIterativeError> {
    let mut result = constraints.to_vec();
    result.sort_unstable();
    result.dedup();
    if let Some(&dof) = result.iter().find(|&&dof| dof >= dimension) {
        return Err(CoupledIterativeError::ConstraintOutOfRange { dof, dimension });
    }
    Ok(result)
}

/// Neumaier compensated accumulation of `term` (with its own rounding error
/// `term_error`) into `sum`/`compensation`.
fn neumaier_add(sum: &mut f64, compensation: &mut f64, term: f64, term_error: f64) {
    let new_sum = *sum + term;
    *compensation += if sum.abs() >= term.abs() {
        (*sum - new_sum) + term
    } else {
        (term - new_sum) + *sum
    } + term_error;
    *sum = new_sum;
}

/// Accumulate the complex product `a·x` into compensated real/imaginary sums,
/// using TwoProd (FMA) error terms for each real multiplication.
#[allow(clippy::too_many_arguments)]
fn neumaier_product_add(
    sum_re: &mut f64,
    comp_re: &mut f64,
    sum_im: &mut f64,
    comp_im: &mut f64,
    a: Complex64,
    x: Complex64,
) {
    // TwoProd: product + error == exact product, so a negated term carries
    // the negated error.
    let product = a.re * x.re;
    neumaier_add(sum_re, comp_re, product, a.re.mul_add(x.re, -product));
    let product = a.im * x.im;
    neumaier_add(sum_re, comp_re, -product, -a.im.mul_add(x.im, -product));
    let product = a.re * x.im;
    neumaier_add(sum_im, comp_im, product, a.re.mul_add(x.im, -product));
    let product = a.im * x.re;
    neumaier_add(sum_im, comp_im, product, a.im.mul_add(x.re, -product));
}

/// Compensated (Neumaier + TwoProd) dot product of one matrix row with `x`.
fn compensated_row_dot(matrix: &CsrMatrix<Complex64>, row: usize, x: &[Complex64]) -> Complex64 {
    let (mut sum_re, mut comp_re) = (0.0, 0.0);
    let (mut sum_im, mut comp_im) = (0.0, 0.0);
    for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
        let a = matrix.values[index];
        neumaier_product_add(
            &mut sum_re,
            &mut comp_re,
            &mut sum_im,
            &mut comp_im,
            a,
            x[matrix.col_indices[index]],
        );
    }
    Complex64::new(sum_re + comp_re, sum_im + comp_im)
}

/// Matrix-free operator that applies a CSR matrix with compensated row dot
/// products.  The coupled operator mixes pressure, stiffness, and circuit
/// scales up to ~1e10, so a plain f64 matvec floors any residual measured
/// through it near `eps·‖A‖·‖x‖ ≈ 1e-9` — exactly the acceptance gate — and
/// parallel reduction order turns the check into a coin flip.  Compensated
/// summation drops the floor towards `eps²·‖A‖·‖x‖`; it is used for the
/// defect-correction solves and residual verification, not the hot Krylov
/// path, where the ~4x matvec cost does not pay.
struct CompensatedCsrOperator<'a> {
    matrix: &'a CsrMatrix<Complex64>,
}

impl LinearOperator<Complex64> for CompensatedCsrOperator<'_> {
    fn num_rows(&self) -> usize {
        self.matrix.num_rows
    }

    fn num_cols(&self) -> usize {
        self.matrix.num_cols
    }

    fn apply_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        for (row, output) in y.iter_mut().enumerate() {
            *output = compensated_row_dot(self.matrix, row, x);
        }
    }

    fn apply_transpose_into(&self, x: &[Complex64], y: &mut [Complex64]) {
        // No consumer on this path needs an accurate transpose; delegate.
        self.matrix.apply_transpose_into(x, y);
    }
}

/// Residual of the constrained system, evaluated row-by-row from the
/// assembled matrix with compensated dot products.  Mirrors
/// [`HomogeneousConstraintOperator`]: constrained columns are zeroed and
/// constrained rows reduce to `r_i = b_i - x_i`.
fn compensated_constrained_residual(
    matrix: &CsrMatrix<Complex64>,
    solution: &[Complex64],
    rhs: &[Complex64],
    constrained: &[bool],
) -> (Vec<Complex64>, f64) {
    let mut zeroed = solution.to_vec();
    for (value, &is_constrained) in zeroed.iter_mut().zip(constrained) {
        if is_constrained {
            *value = Complex64::new(0.0, 0.0);
        }
    }
    let mut residual = vec![Complex64::new(0.0, 0.0); matrix.num_rows];
    let mut residual_norm_squared = 0.0;
    for row in 0..matrix.num_rows {
        let value = if constrained[row] {
            rhs[row] - solution[row]
        } else {
            let (mut sum_re, mut comp_re) = (0.0, 0.0);
            let (mut sum_im, mut comp_im) = (0.0, 0.0);
            neumaier_add(&mut sum_re, &mut comp_re, rhs[row].re, 0.0);
            neumaier_add(&mut sum_im, &mut comp_im, rhs[row].im, 0.0);
            for index in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
                let a = matrix.values[index];
                neumaier_product_add(
                    &mut sum_re,
                    &mut comp_re,
                    &mut sum_im,
                    &mut comp_im,
                    -a,
                    zeroed[matrix.col_indices[index]],
                );
            }
            Complex64::new(sum_re + comp_re, sum_im + comp_im)
        };
        residual_norm_squared += value.norm_sqr();
        residual[row] = value;
    }
    let rhs_norm = rhs.iter().map(|value| value.norm_sqr()).sum::<f64>().sqrt();
    (
        residual,
        residual_norm_squared.sqrt() / rhs_norm.max(f64::MIN_POSITIVE),
    )
}

#[derive(Debug, thiserror::Error)]
pub enum CoupledIterativeError {
    #[error("invalid iterative coupled-solver configuration: {0}")]
    InvalidConfiguration(&'static str),
    #[error("coupled system dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
    #[error("essential constraint DOF {dof} is outside dimension {dimension}")]
    ConstraintOutOfRange { dof: usize, dimension: usize },
    #[error(
        "coupled FGMRES failed after {iterations} iterations (relative residual {relative_residual:e})"
    )]
    ConvergenceFailure {
        iterations: usize,
        relative_residual: f64,
    },
    #[error(transparent)]
    Coupled(#[from] CoupledSystemError),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::acoustics::electroacoustics::{ElectricalSubsystem, HarmonicCoupledPoint};

    fn diagonal(size: usize, value: f64) -> TripletMatrix<f64> {
        let mut matrix = TripletMatrix::new(size, size);
        for index in 0..size {
            matrix.push(index, index, value);
        }
        matrix
    }

    fn test_system() -> CoupledSystem {
        let mut coupling = TripletMatrix::new(2, 2);
        coupling.push(0, 0, 0.1);
        coupling.push(1, 1, 0.15);
        let mut motor = TripletMatrix::new(1, 2);
        motor.push(0, 1, 0.4);
        CoupledSystem {
            n_fluid: 2,
            n_solid: 2,
            fluid_stiffness: diagonal(2, 5.0),
            fluid_mass: diagonal(2, 0.002),
            fluid_damping: diagonal(2, 1.0e-5),
            solid_stiffness: diagonal(2, 2.0e5),
            solid_mass: diagonal(2, 0.02),
            solid_damping: diagonal(2, 0.4),
            coupling,
            interface_load: None,
            electrical: Some(ElectricalSubsystem {
                n_dofs: 1,
                resistance: diagonal(1, 6.0),
                inductance: diagonal(1, 4.0e-4),
                coupling: motor,
            }),
        }
    }

    fn point(frequency_hz: f64) -> HarmonicCoupledPoint {
        HarmonicCoupledPoint {
            omega_rad_per_s: 2.0 * std::f64::consts::PI * frequency_hz,
            sound_speed_m_per_s: 343.0,
            fluid_density_kg_per_m3: 1.204,
        }
    }

    #[test]
    fn compensated_operator_matches_plain_csr_apply() {
        let system = test_system();
        let point = point(100.0);
        let matrix = system.assemble_harmonic(point).unwrap().to_csr();
        let operator = CompensatedCsrOperator { matrix: &matrix };
        let x: Vec<Complex64> = (0..system.total_dofs())
            .map(|i| Complex64::new(0.3 + i as f64, -0.1 * i as f64))
            .collect();
        let plain = matrix.apply(&x);
        let compensated = operator.apply(&x);
        for row in 0..matrix.num_rows {
            let diff = (plain[row] - compensated[row]).norm();
            assert!(
                diff < 1.0e-12 * plain[row].norm().max(1.0),
                "row {row}: plain={:?} compensated={:?}",
                plain[row],
                compensated[row]
            );
        }
    }

    #[test]
    fn block_fgmres_converges_with_exact_residual_and_constraint() {
        let system = test_system();
        let mut solver =
            CoupledFrequencyContinuation::new(&system, CoupledIterativeConfig::default()).unwrap();
        let mut rhs = vec![Complex64::new(0.0, 0.0); system.total_dofs()];
        rhs[4] = Complex64::new(2.83, 0.0);
        let solution = solver.solve(point(100.0), &rhs, &[2]).unwrap();
        assert!(solution.diagnostics.converged);
        assert!(solution.diagnostics.relative_residual < 1.0e-9);
        assert!(solution.values[2].norm() < 1.0e-12);
        assert!(solution.diagnostics.preconditioner_rebuilt);
        assert!(!solution.diagnostics.initial_guess_reused);
    }

    #[test]
    fn continuation_reuses_guess_and_nearby_block_factors() {
        let system = test_system();
        let mut solver =
            CoupledFrequencyContinuation::new(&system, CoupledIterativeConfig::default()).unwrap();
        let mut rhs = vec![Complex64::new(0.0, 0.0); system.total_dofs()];
        rhs[4] = Complex64::new(1.0, 0.0);
        solver.solve(point(100.0), &rhs, &[]).unwrap();
        let nearby = solver.solve(point(110.0), &rhs, &[]).unwrap();
        assert!(nearby.diagnostics.initial_guess_reused);
        assert!(!nearby.diagnostics.preconditioner_rebuilt);
        let distant = solver.solve(point(200.0), &rhs, &[]).unwrap();
        assert!(distant.diagnostics.initial_guess_reused);
        assert!(distant.diagnostics.preconditioner_rebuilt);
    }

    #[test]
    fn monolithic_preconditioner_converges_with_exact_residual_and_constraint() {
        let system = test_system();
        let config = CoupledIterativeConfig {
            preconditioner: CoupledPreconditioner::Monolithic,
            ..CoupledIterativeConfig::default()
        };
        let mut solver = CoupledFrequencyContinuation::new(&system, config).unwrap();
        let mut rhs = vec![Complex64::new(0.0, 0.0); system.total_dofs()];
        rhs[4] = Complex64::new(2.83, 0.0);
        let solution = solver.solve(point(100.0), &rhs, &[2]).unwrap();
        assert!(solution.diagnostics.converged);
        assert!(solution.diagnostics.relative_residual < 1.0e-9);
        assert!(solution.values[2].norm() < 1.0e-12);
        assert!(solution.diagnostics.preconditioner_rebuilt);
        assert!(!solution.diagnostics.initial_guess_reused);
        // The monolithic factors are reused over the same band as the blocks.
        let nearby = solver.solve(point(110.0), &rhs, &[2]).unwrap();
        assert!(nearby.diagnostics.converged);
        assert!(!nearby.diagnostics.preconditioner_rebuilt);
    }

    #[test]
    fn physical_row_scaling_recovers_complex_reciprocity() {
        let defect = harmonic_reciprocity_defect(&test_system(), point(200.0)).unwrap();
        assert!(defect < 1.0e-14, "reciprocity defect {defect:e}");
    }
}
