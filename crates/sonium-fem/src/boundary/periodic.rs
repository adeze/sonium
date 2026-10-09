//! Periodic and cyclic sector boundary contracts.

use crate::assembly::TripletMatrix;
use crate::boundary::generic::{FemSystem, FemSystemError};
use crate::mesh::{Face, Mesh, topology::MeshTopology};
use num_complex::Complex64;
use sonium_solvers::ComplexField;
use sonium_solvers::sparse::CsrMatrix;
use std::collections::{BTreeMap, HashMap, HashSet};

/// Complex phase for a sector-periodic circumferential order.
///
/// A field with order `m` satisfies `u(theta + alpha) = exp(i*m*alpha) u(theta)`.
pub fn sector_phase(circumferential_order: i32, sector_angle_rad: f64) -> Complex64 {
    let phase = circumferential_order as f64 * sector_angle_rad;
    Complex64::new(phase.cos(), phase.sin())
}

/// One linear periodic constraint: `u_slave = phase * u_master`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodicDofConstraint {
    pub master: usize,
    pub slave: usize,
    pub phase: Complex64,
}

impl PeriodicDofConstraint {
    pub fn residual(&self, solution: &[Complex64]) -> Complex64 {
        solution[self.slave] - self.phase * solution[self.master]
    }

    /// Checked periodic residual evaluation for solver-facing vectors.
    pub fn residual_checked(&self, solution: &[Complex64]) -> Result<Complex64, MpcError> {
        let expected = self.master.max(self.slave).saturating_add(1);
        if solution.len() < expected {
            return Err(MpcError::VectorDimension {
                actual: solution.len(),
                expected,
            });
        }
        if !self.phase.re.is_finite() || !self.phase.im.is_finite() {
            return Err(MpcError::NonFiniteCoefficient {
                slave: self.slave,
                master: self.master,
            });
        }
        if !solution[self.master].re.is_finite()
            || !solution[self.master].im.is_finite()
            || !solution[self.slave].re.is_finite()
            || !solution[self.slave].im.is_finite()
        {
            return Err(MpcError::NonFiniteVector {
                dof: if !solution[self.master].re.is_finite()
                    || !solution[self.master].im.is_finite()
                {
                    self.master
                } else {
                    self.slave
                },
            });
        }
        Ok(self.residual(solution))
    }
}

/// Expand paired sector nodes to per-DOF cyclic constraints.
pub fn expand_node_periodic_constraints(
    node_pairs: &[(usize, usize)],
    dofs_per_node: usize,
    phase: Complex64,
) -> Vec<PeriodicDofConstraint> {
    let mut constraints = Vec::with_capacity(node_pairs.len() * dofs_per_node);
    for &(master_node, slave_node) in node_pairs {
        for component in 0..dofs_per_node {
            constraints.push(PeriodicDofConstraint {
                master: master_node * dofs_per_node + component,
                slave: slave_node * dofs_per_node + component,
                phase,
            });
        }
    }
    constraints
}

/// Return the residual of every periodic constraint for diagnostics/QA.
pub fn periodic_residuals(
    solution: &[Complex64],
    constraints: &[PeriodicDofConstraint],
) -> Vec<Complex64> {
    constraints
        .iter()
        .map(|constraint| constraint.residual(solution))
        .collect()
}

/// Pair two already-sorted boundary node lists, validating equal length.
pub fn pair_sorted_sector_nodes(
    masters: &[usize],
    slaves: &[usize],
) -> Result<Vec<(usize, usize)>, String> {
    if masters.len() != slaves.len() {
        return Err(format!(
            "sector periodic boundaries have different node counts: {} masters, {} slaves",
            masters.len(),
            slaves.len()
        ));
    }
    Ok(masters
        .iter()
        .zip(slaves.iter())
        .map(|(&master, &slave)| (master, slave))
        .collect())
}

/// One weighted master term in a general multipoint constraint.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MpcTerm {
    pub dof: usize,
    pub coefficient: Complex64,
}

/// General linear relation `u_slave = sum(coefficient * u_master)`.
#[derive(Debug, Clone, PartialEq)]
pub struct MultiPointConstraint {
    pub slave: usize,
    pub masters: Vec<MpcTerm>,
}

impl From<PeriodicDofConstraint> for MultiPointConstraint {
    fn from(constraint: PeriodicDofConstraint) -> Self {
        Self {
            slave: constraint.slave,
            masters: vec![MpcTerm {
                dof: constraint.master,
                coefficient: constraint.phase,
            }],
        }
    }
}

/// Sparse prolongation contract `u_full = P u_reduced`.
#[derive(Debug, Clone, PartialEq)]
pub struct MpcSystem {
    num_full_dofs: usize,
    reduced_full_dofs: Vec<usize>,
    /// Per full DOF, `(reduced_dof, coefficient)` entries in one row of P.
    expansion: Vec<Vec<(usize, Complex64)>>,
}

impl MpcSystem {
    pub fn new(
        num_full_dofs: usize,
        constraints: &[MultiPointConstraint],
    ) -> Result<Self, MpcError> {
        let mut by_slave = HashMap::with_capacity(constraints.len());
        for constraint in constraints {
            if constraint.slave >= num_full_dofs {
                return Err(MpcError::DofOutOfRange {
                    dof: constraint.slave,
                    num_dofs: num_full_dofs,
                });
            }
            if constraint.masters.is_empty() {
                return Err(MpcError::EmptyMasters {
                    slave: constraint.slave,
                });
            }
            if by_slave.insert(constraint.slave, constraint).is_some() {
                return Err(MpcError::DuplicateSlave(constraint.slave));
            }
            for term in &constraint.masters {
                if term.dof >= num_full_dofs {
                    return Err(MpcError::DofOutOfRange {
                        dof: term.dof,
                        num_dofs: num_full_dofs,
                    });
                }
                if !term.coefficient.re.is_finite() || !term.coefficient.im.is_finite() {
                    return Err(MpcError::NonFiniteCoefficient {
                        slave: constraint.slave,
                        master: term.dof,
                    });
                }
            }
        }

        let reduced_full_dofs: Vec<_> = (0..num_full_dofs)
            .filter(|dof| !by_slave.contains_key(dof))
            .collect();
        let reduced_index: HashMap<_, _> = reduced_full_dofs
            .iter()
            .enumerate()
            .map(|(index, &dof)| (dof, index))
            .collect();
        let mut memo: Vec<Option<BTreeMap<usize, Complex64>>> = vec![None; num_full_dofs];
        let mut visiting = HashSet::new();
        let mut expansion = Vec::with_capacity(num_full_dofs);
        for dof in 0..num_full_dofs {
            let full_coefficients = resolve_mpc_dof(dof, &by_slave, &mut memo, &mut visiting)?;
            let mut row = Vec::with_capacity(full_coefficients.len());
            for (full_master, coefficient) in full_coefficients {
                let index = reduced_index[&full_master];
                if coefficient.norm() > 0.0 {
                    row.push((index, coefficient));
                }
            }
            expansion.push(row);
        }
        Ok(Self {
            num_full_dofs,
            reduced_full_dofs,
            expansion,
        })
    }

    pub fn from_periodic(
        num_full_dofs: usize,
        constraints: &[PeriodicDofConstraint],
    ) -> Result<Self, MpcError> {
        let constraints: Vec<_> = constraints.iter().copied().map(Into::into).collect();
        Self::new(num_full_dofs, &constraints)
    }

    pub fn num_full_dofs(&self) -> usize {
        self.num_full_dofs
    }

    pub fn num_reduced_dofs(&self) -> usize {
        self.reduced_full_dofs.len()
    }

    pub fn reduced_full_dofs(&self) -> &[usize] {
        &self.reduced_full_dofs
    }

    pub fn prolongation_row(&self, full_dof: usize) -> &[(usize, Complex64)] {
        &self.expansion[full_dof]
    }

    /// Checked access to one row of the prolongation operator.
    pub fn try_prolongation_row(&self, full_dof: usize) -> Result<&[(usize, Complex64)], MpcError> {
        self.expansion
            .get(full_dof)
            .map(Vec::as_slice)
            .ok_or(MpcError::DofOutOfRange {
                dof: full_dof,
                num_dofs: self.num_full_dofs,
            })
    }

    /// Compute `P^H A P` without materializing P.
    pub fn condense_matrix(
        &self,
        matrix: &CsrMatrix<Complex64>,
    ) -> Result<CsrMatrix<Complex64>, MpcError> {
        if matrix.num_rows != self.num_full_dofs || matrix.num_cols != self.num_full_dofs {
            return Err(MpcError::MatrixDimension {
                rows: matrix.num_rows,
                cols: matrix.num_cols,
                expected: self.num_full_dofs,
            });
        }
        let mut triplets = Vec::new();
        for full_row in 0..matrix.num_rows {
            for (full_column, value) in matrix.row_entries(full_row) {
                for &(reduced_row, row_coefficient) in &self.expansion[full_row] {
                    for &(reduced_column, column_coefficient) in &self.expansion[full_column] {
                        let contribution = row_coefficient.conj() * value * column_coefficient;
                        if !contribution.re.is_finite() || !contribution.im.is_finite() {
                            return Err(MpcError::NonFiniteCondensedValue {
                                row: reduced_row,
                                column: reduced_column,
                            });
                        }
                        triplets.push((reduced_row, reduced_column, contribution));
                    }
                }
            }
        }
        Ok(CsrMatrix::from_triplets(
            self.num_reduced_dofs(),
            self.num_reduced_dofs(),
            triplets,
        ))
    }

    /// Compute P^T A P for a real matrix and real-valued constraints.
    ///
    /// This accepts periodic and anti-periodic phases (+1 and -1), plus
    /// arbitrary real multi-point coefficients. A genuinely complex Bloch
    /// phase is rejected instead of silently discarding its imaginary part.
    pub fn condense_real_matrix(
        &self,
        matrix: &CsrMatrix<f64>,
    ) -> Result<CsrMatrix<f64>, MpcError> {
        if matrix.num_rows != self.num_full_dofs || matrix.num_cols != self.num_full_dofs {
            return Err(MpcError::MatrixDimension {
                rows: matrix.num_rows,
                cols: matrix.num_cols,
                expected: self.num_full_dofs,
            });
        }
        for (full_dof, row) in self.expansion.iter().enumerate() {
            if row.iter().any(|(_, coefficient)| coefficient.im != 0.0) {
                return Err(MpcError::ComplexCoefficientInRealSystem { full_dof });
            }
        }
        let mut triplets = Vec::new();
        for full_row in 0..matrix.num_rows {
            for (full_column, value) in matrix.row_entries(full_row) {
                for &(reduced_row, row_coefficient) in &self.expansion[full_row] {
                    for &(reduced_column, column_coefficient) in &self.expansion[full_column] {
                        let contribution = row_coefficient.re * value * column_coefficient.re;
                        if !contribution.is_finite() {
                            return Err(MpcError::NonFiniteCondensedValue {
                                row: reduced_row,
                                column: reduced_column,
                            });
                        }
                        triplets.push((reduced_row, reduced_column, contribution));
                    }
                }
            }
        }
        Ok(CsrMatrix::from_triplets(
            self.num_reduced_dofs(),
            self.num_reduced_dofs(),
            triplets,
        ))
    }

    /// Condense a checked complex FEM system with `u_full = P u_reduced`.
    ///
    /// Dirichlet elimination must happen after this operation. Keeping the
    /// ordering explicit prevents a constrained identity row from being
    /// interpreted as a physical periodic equation. The returned system has
    /// reduced dimensions; use [`Self::recover`] with its solution vector to
    /// reconstruct the full field.
    pub fn condense_fem_system_complex(
        &self,
        system: &FemSystem<Complex64>,
    ) -> Result<FemSystem<Complex64>, MpcError> {
        self.validate_fem_system(system, |value| value.re.is_finite() && value.im.is_finite())?;
        let condensed = self.condense_matrix(&system.matrix.to_csr())?;
        let rhs = self.condense_rhs(&system.rhs)?;
        Ok(FemSystem::new(csr_to_triplet(condensed), rhs))
    }

    /// Condense a checked real FEM system with real-valued constraints.
    ///
    /// Real systems accept arbitrary real multipoint coefficients and
    /// anti-periodic phases. A genuinely complex phase is rejected by
    /// [`Self::condense_real_matrix`] rather than being silently truncated.
    /// As with the complex variant, Dirichlet elimination must follow MPC
    /// condensation.
    pub fn condense_fem_system_real(
        &self,
        system: &FemSystem<f64>,
    ) -> Result<FemSystem<f64>, MpcError> {
        self.validate_fem_system(system, f64::is_finite)?;
        let condensed = self.condense_real_matrix(&system.matrix.to_csr())?;
        let rhs = self.condense_real_rhs(&system.rhs)?;
        Ok(FemSystem::new(csr_to_triplet(condensed), rhs))
    }

    fn validate_fem_system<T>(
        &self,
        system: &FemSystem<T>,
        is_finite: impl Fn(T) -> bool,
    ) -> Result<(), MpcError>
    where
        T: Copy + Default + std::ops::AddAssign + Send + Sync,
    {
        system.validate().map_err(MpcError::InvalidFemSystem)?;
        if !system.dirichlet_dofs.is_empty() {
            return Err(MpcError::DirichletConstraintsPresent);
        }
        if system.matrix.num_rows != self.num_full_dofs
            || system.matrix.num_cols != self.num_full_dofs
        {
            return Err(MpcError::MatrixDimension {
                rows: system.matrix.num_rows,
                cols: system.matrix.num_cols,
                expected: self.num_full_dofs,
            });
        }
        for (triplet, &value) in system.matrix.values.iter().enumerate() {
            if !is_finite(value) {
                return Err(MpcError::NonFiniteMatrixValue { triplet });
            }
        }
        for (index, &value) in system.rhs.iter().enumerate() {
            if !is_finite(value) {
                return Err(MpcError::NonFiniteRhsEntry { index });
            }
        }
        Ok(())
    }

    /// Compute `P^H b`.
    pub fn condense_rhs(&self, rhs: &[Complex64]) -> Result<Vec<Complex64>, MpcError> {
        if rhs.len() != self.num_full_dofs {
            return Err(MpcError::VectorDimension {
                actual: rhs.len(),
                expected: self.num_full_dofs,
            });
        }
        let mut reduced = vec![Complex64::new(0.0, 0.0); self.num_reduced_dofs()];
        for (full_dof, &value) in rhs.iter().enumerate() {
            for &(reduced_dof, coefficient) in &self.expansion[full_dof] {
                reduced[reduced_dof] += coefficient.conj() * value;
                if !reduced[reduced_dof].re.is_finite() || !reduced[reduced_dof].im.is_finite() {
                    return Err(MpcError::NonFiniteCondensedRhs { reduced_dof });
                }
            }
        }
        Ok(reduced)
    }

    /// Compute `P^T b` for a real system and real-valued constraints.
    pub fn condense_real_rhs(&self, rhs: &[f64]) -> Result<Vec<f64>, MpcError> {
        if rhs.len() != self.num_full_dofs {
            return Err(MpcError::VectorDimension {
                actual: rhs.len(),
                expected: self.num_full_dofs,
            });
        }
        let mut reduced = vec![0.0; self.num_reduced_dofs()];
        for (full_dof, &value) in rhs.iter().enumerate() {
            for &(reduced_dof, coefficient) in &self.expansion[full_dof] {
                if coefficient.im != 0.0 {
                    return Err(MpcError::ComplexCoefficientInRealSystem { full_dof });
                }
                reduced[reduced_dof] += coefficient.re * value;
                if !reduced[reduced_dof].is_finite() {
                    return Err(MpcError::NonFiniteCondensedRhs { reduced_dof });
                }
            }
        }
        Ok(reduced)
    }

    /// Recover `u_full = P u_reduced`.
    pub fn recover(&self, reduced: &[Complex64]) -> Result<Vec<Complex64>, MpcError> {
        if reduced.len() != self.num_reduced_dofs() {
            return Err(MpcError::VectorDimension {
                actual: reduced.len(),
                expected: self.num_reduced_dofs(),
            });
        }
        Ok(self
            .expansion
            .iter()
            .map(|row| {
                row.iter()
                    .map(|&(dof, coefficient)| coefficient * reduced[dof])
                    .sum()
            })
            .collect())
    }

    /// Recover a finite real full state from a reduced state.
    ///
    /// Real FEM transfer must not silently discard a complex periodic phase;
    /// such a system is rejected before any adaptive interpolation occurs.
    pub fn recover_real(&self, reduced: &[f64]) -> Result<Vec<f64>, MpcError> {
        if reduced.len() != self.num_reduced_dofs() {
            return Err(MpcError::VectorDimension {
                actual: reduced.len(),
                expected: self.num_reduced_dofs(),
            });
        }
        if let Some((index, _)) = reduced
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(MpcError::NonFiniteVector { dof: index });
        }
        let mut full = Vec::with_capacity(self.num_full_dofs);
        for (full_dof, row) in self.expansion.iter().enumerate() {
            let mut value = 0.0;
            for &(reduced_dof, coefficient) in row {
                if coefficient.im != 0.0 {
                    return Err(MpcError::ComplexCoefficientInRealSystem { full_dof });
                }
                value += coefficient.re * reduced[reduced_dof];
            }
            if !value.is_finite() {
                return Err(MpcError::NonFiniteVector { dof: full_dof });
            }
            full.push(value);
        }
        Ok(full)
    }

    /// Restrict a full real state to reduced coordinates after checking that
    /// it satisfies every MPC row within a relative/absolute tolerance.
    pub fn restrict_real_checked(
        &self,
        full: &[f64],
        tolerance: f64,
    ) -> Result<Vec<f64>, MpcError> {
        if full.len() != self.num_full_dofs {
            return Err(MpcError::VectorDimension {
                actual: full.len(),
                expected: self.num_full_dofs,
            });
        }
        if !tolerance.is_finite() || tolerance < 0.0 {
            return Err(MpcError::InvalidTolerance(tolerance));
        }
        if let Some((index, _)) = full
            .iter()
            .enumerate()
            .find(|(_, value)| !value.is_finite())
        {
            return Err(MpcError::NonFiniteVector { dof: index });
        }
        let reduced: Vec<f64> = self
            .reduced_full_dofs
            .iter()
            .map(|&full_dof| full[full_dof])
            .collect();
        let reconstructed = self.recover_real(&reduced)?;
        for (full_dof, (&actual, &expected)) in full.iter().zip(&reconstructed).enumerate() {
            let scale = actual.abs().max(expected.abs()).max(1.0);
            let residual = (actual - expected).abs();
            if residual > tolerance * scale {
                return Err(MpcError::StateConstraintResidual {
                    full_dof,
                    residual,
                    tolerance,
                });
            }
        }
        Ok(reduced)
    }
}

fn csr_to_triplet<T>(matrix: CsrMatrix<T>) -> TripletMatrix<T>
where
    T: ComplexField + Copy + Default + std::ops::AddAssign + Send + Sync,
{
    let mut triplet =
        TripletMatrix::with_capacity(matrix.num_rows, matrix.num_cols, matrix.values.len());
    for row in 0..matrix.num_rows {
        for (column, value) in matrix.row_entries(row) {
            triplet.push(row, column, value);
        }
    }
    triplet
}

fn resolve_mpc_dof(
    dof: usize,
    constraints: &HashMap<usize, &MultiPointConstraint>,
    memo: &mut [Option<BTreeMap<usize, Complex64>>],
    visiting: &mut HashSet<usize>,
) -> Result<BTreeMap<usize, Complex64>, MpcError> {
    if let Some(resolved) = &memo[dof] {
        return Ok(resolved.clone());
    }
    if !visiting.insert(dof) {
        return Err(MpcError::ConstraintCycle(dof));
    }
    let mut resolved = BTreeMap::new();
    if let Some(constraint) = constraints.get(&dof) {
        for term in &constraint.masters {
            for (master, coefficient) in resolve_mpc_dof(term.dof, constraints, memo, visiting)? {
                *resolved.entry(master).or_insert(Complex64::new(0.0, 0.0)) +=
                    term.coefficient * coefficient;
            }
        }
    } else {
        resolved.insert(dof, Complex64::new(1.0, 0.0));
    }
    visiting.remove(&dof);
    memo[dof] = Some(resolved.clone());
    Ok(resolved)
}

/// Rigid transformation used to map master-face nodes onto a slave face.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeriodicTransform {
    pub rotation: [[f64; 3]; 3],
    pub translation: [f64; 3],
}

impl PeriodicTransform {
    pub const IDENTITY: Self = Self {
        rotation: [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        translation: [0.0; 3],
    };

    pub fn rotation_about_z(angle_rad: f64) -> Self {
        let (sin, cos) = angle_rad.sin_cos();
        Self {
            rotation: [[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]],
            translation: [0.0; 3],
        }
    }

    pub fn apply(&self, point: [f64; 3]) -> [f64; 3] {
        std::array::from_fn(|row| {
            self.translation[row]
                + (0..3)
                    .map(|column| self.rotation[row][column] * point[column])
                    .sum::<f64>()
        })
    }
}

/// Pair master/slave nodes geometrically, validating tolerance and bijection.
pub fn pair_periodic_nodes_geometric(
    points: &[[f64; 3]],
    masters: &[usize],
    slaves: &[usize],
    transform: PeriodicTransform,
    tolerance: f64,
) -> Result<Vec<(usize, usize)>, MpcError> {
    if !tolerance.is_finite() || tolerance <= 0.0 {
        return Err(MpcError::InvalidTolerance(tolerance));
    }
    if masters.len() != slaves.len() {
        return Err(MpcError::PairCountMismatch {
            masters: masters.len(),
            slaves: slaves.len(),
        });
    }
    for &node in masters.iter().chain(slaves) {
        if node >= points.len() {
            return Err(MpcError::NodeOutOfRange(node));
        }
    }
    let mut used = HashSet::new();
    let mut pairs = Vec::with_capacity(masters.len());
    for &master in masters {
        let target = transform.apply(points[master]);
        let mut candidates: Vec<_> = slaves
            .iter()
            .copied()
            .filter(|slave| !used.contains(slave))
            .map(|slave| {
                let distance = points[slave]
                    .iter()
                    .zip(target)
                    .map(|(actual, expected)| (actual - expected).powi(2))
                    .sum::<f64>()
                    .sqrt();
                (slave, distance)
            })
            .filter(|(_, distance)| *distance <= tolerance)
            .collect();
        candidates.sort_by(|a, b| a.1.total_cmp(&b.1));
        match candidates.as_slice() {
            [] => return Err(MpcError::NoGeometricMatch { master }),
            [best, second, ..] if (second.1 - best.1).abs() <= tolerance * 1.0e-6 => {
                return Err(MpcError::AmbiguousGeometricMatch { master });
            }
            [(slave, _), ..] => {
                used.insert(*slave);
                pairs.push((master, *slave));
            }
        }
    }
    Ok(pairs)
}

/// Pair oriented edge DOFs from an already validated node pairing.
///
/// Each edge is `(global_dof, [start_node, end_node])`. The returned sign is
/// `+1` when endpoint order agrees and `-1` when it reverses, matching the
/// orientation convention of first-order H(curl) basis functions.
pub fn pair_periodic_edges_oriented(
    master_edges: &[(usize, [usize; 2])],
    slave_edges: &[(usize, [usize; 2])],
    node_pairs: &[(usize, usize)],
) -> Result<Vec<(usize, usize, i8)>, MpcError> {
    if master_edges.len() != slave_edges.len() {
        return Err(MpcError::PairCountMismatch {
            masters: master_edges.len(),
            slaves: slave_edges.len(),
        });
    }
    let node_map: HashMap<_, _> = node_pairs.iter().copied().collect();
    let mut used_slave_dofs = HashSet::new();
    let mut pairs = Vec::with_capacity(master_edges.len());
    for &(master_dof, [master_start, master_end]) in master_edges {
        let mapped = [
            *node_map
                .get(&master_start)
                .ok_or(MpcError::UnpairedEdgeNode(master_start))?,
            *node_map
                .get(&master_end)
                .ok_or(MpcError::UnpairedEdgeNode(master_end))?,
        ];
        let candidates: Vec<_> = slave_edges
            .iter()
            .filter(|(slave_dof, _)| !used_slave_dofs.contains(slave_dof))
            .filter_map(|&(slave_dof, slave_nodes)| {
                if slave_nodes == mapped {
                    Some((slave_dof, 1))
                } else if slave_nodes == [mapped[1], mapped[0]] {
                    Some((slave_dof, -1))
                } else {
                    None
                }
            })
            .collect();
        match candidates.as_slice() {
            [] => return Err(MpcError::NoPeriodicEdgeMatch { master_dof }),
            [pair] => {
                used_slave_dofs.insert(pair.0);
                pairs.push((master_dof, pair.0, pair.1));
            }
            _ => return Err(MpcError::AmbiguousPeriodicEdgeMatch { master_dof }),
        }
    }
    Ok(pairs)
}

/// Pair oriented triangular or quadrilateral face DOFs after node pairing.
///
/// The returned sign is the orientation of the mapped master face relative
/// to the slave face. This is the sign required by lowest-order H(div)
/// normal-flux moments when both face DOFs use the mesh-global orientation.
pub fn pair_periodic_faces_oriented(
    master_faces: &[(usize, Face)],
    slave_faces: &[(usize, Face)],
    node_pairs: &[(usize, usize)],
) -> Result<Vec<(usize, usize, i8)>, MpcError> {
    if master_faces.len() != slave_faces.len() {
        return Err(MpcError::PairCountMismatch {
            masters: master_faces.len(),
            slaves: slave_faces.len(),
        });
    }
    let node_map: HashMap<_, _> = node_pairs.iter().copied().collect();
    let mut used_slave_dofs = HashSet::new();
    let mut pairs = Vec::with_capacity(master_faces.len());

    for &(master_dof, ref master_face) in master_faces {
        let mapped_nodes: Vec<_> = master_face
            .nodes()
            .into_iter()
            .map(|node| {
                node_map
                    .get(&node)
                    .copied()
                    .ok_or(MpcError::UnpairedFaceNode(node))
            })
            .collect::<Result<_, _>>()?;
        let mut candidates = Vec::new();
        for &(slave_dof, ref slave_face) in slave_faces {
            if used_slave_dofs.contains(&slave_dof) {
                continue;
            }
            let slave_nodes = slave_face.nodes();
            let same_nodes = if mapped_nodes.len() == slave_nodes.len() {
                let mut mapped = mapped_nodes.clone();
                let mut target = slave_nodes.clone();
                mapped.sort_unstable();
                target.sort_unstable();
                mapped == target
            } else {
                false
            };
            if same_nodes
                && let Some(orientation) = face_orientation_sign(&mapped_nodes, &slave_nodes)
            {
                candidates.push((slave_dof, orientation));
            }
        }
        match candidates.as_slice() {
            [] => return Err(MpcError::NoPeriodicFaceMatch { master_dof }),
            [(slave_dof, orientation)] => {
                used_slave_dofs.insert(*slave_dof);
                pairs.push((master_dof, *slave_dof, *orientation));
            }
            [..] => return Err(MpcError::AmbiguousPeriodicFaceMatch { master_dof }),
        }
    }
    Ok(pairs)
}

/// Pair triangular periodic faces and retain the vertex permutation required
/// by RT1 face-mode transfer.
pub fn pair_periodic_hdiv_rt1_faces(
    master_faces: &[(usize, Face)],
    slave_faces: &[(usize, Face)],
    node_pairs: &[(usize, usize)],
) -> Result<Vec<HdivRt1PeriodicFacePair>, MpcError> {
    if master_faces.len() != slave_faces.len() {
        return Err(MpcError::PairCountMismatch {
            masters: master_faces.len(),
            slaves: slave_faces.len(),
        });
    }
    let node_map: HashMap<_, _> = node_pairs.iter().copied().collect();
    let mut used_slave_dofs = HashSet::new();
    let mut pairs = Vec::with_capacity(master_faces.len());
    for &(master_dof, ref master_face) in master_faces {
        let Face::Triangle(master_nodes) = master_face else {
            return Err(MpcError::InvalidFaceVertexMap {
                master_face: master_dof,
                slave_face: master_dof,
            });
        };
        let mapped_nodes = master_nodes
            .iter()
            .map(|&node| {
                node_map
                    .get(&node)
                    .copied()
                    .ok_or(MpcError::UnpairedFaceNode(node))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut candidates = Vec::new();
        for &(slave_dof, ref slave_face) in slave_faces {
            if used_slave_dofs.contains(&slave_dof) {
                continue;
            }
            let Face::Triangle(slave_nodes) = slave_face else {
                continue;
            };
            let mut permutation = [0usize; 3];
            let mut valid = true;
            for (master_index, &mapped_node) in mapped_nodes.iter().enumerate() {
                let Some(slave_index) = slave_nodes.iter().position(|&node| node == mapped_node)
                else {
                    valid = false;
                    break;
                };
                permutation[master_index] = slave_index;
            }
            if valid && rt1_face_mode_change_matrix(permutation).is_some() {
                candidates.push((slave_dof, permutation));
            }
        }
        match candidates.as_slice() {
            [] => return Err(MpcError::NoPeriodicFaceMatch { master_dof }),
            [(slave_dof, vertex_permutation)] => {
                used_slave_dofs.insert(*slave_dof);
                pairs.push(HdivRt1PeriodicFacePair {
                    master_face: master_dof,
                    slave_face: *slave_dof,
                    vertex_permutation: *vertex_permutation,
                });
            }
            [..] => return Err(MpcError::AmbiguousPeriodicFaceMatch { master_dof }),
        }
    }
    Ok(pairs)
}

fn face_orientation_sign(mapped_nodes: &[usize], slave_nodes: &[usize]) -> Option<i8> {
    if !matches!(mapped_nodes.len(), 3 | 4) || mapped_nodes.len() != slave_nodes.len() {
        return None;
    }
    let count = mapped_nodes.len();
    for shift in 0..count {
        if (0..count).all(|index| mapped_nodes[index] == slave_nodes[(shift + index) % count]) {
            return Some(1);
        }
        if (0..count)
            .all(|index| mapped_nodes[index] == slave_nodes[(shift + count - index) % count])
        {
            return Some(-1);
        }
    }
    None
}

/// Expand node pairs for a Cartesian vector satisfying `u_s = phase R u_m`.
pub fn expand_rotated_vector_constraints(
    node_pairs: &[(usize, usize)],
    dofs_per_node: usize,
    rotation: [[f64; 3]; 3],
    phase: Complex64,
) -> Result<Vec<MultiPointConstraint>, MpcError> {
    if !(1..=3).contains(&dofs_per_node) {
        return Err(MpcError::InvalidComponents(dofs_per_node));
    }
    let mut constraints = Vec::with_capacity(node_pairs.len() * dofs_per_node);
    for &(master_node, slave_node) in node_pairs {
        for (slave_component, rotation_row) in rotation.iter().enumerate().take(dofs_per_node) {
            let masters = (0..dofs_per_node)
                .filter_map(|master_component| {
                    let coefficient = phase * rotation_row[master_component];
                    (coefficient.norm() > 0.0).then_some(MpcTerm {
                        dof: master_node * dofs_per_node + master_component,
                        coefficient,
                    })
                })
                .collect();
            constraints.push(MultiPointConstraint {
                slave: slave_node * dofs_per_node + slave_component,
                masters,
            });
        }
    }
    Ok(constraints)
}

/// Expand oriented edge/face DOF pairs, including H(curl) sign changes.
pub fn expand_oriented_dof_constraints(
    pairs: &[(usize, usize, i8)],
    phase: Complex64,
) -> Result<Vec<PeriodicDofConstraint>, MpcError> {
    pairs
        .iter()
        .map(|&(master, slave, orientation)| {
            if orientation != -1 && orientation != 1 {
                return Err(MpcError::InvalidOrientation(orientation));
            }
            Ok(PeriodicDofConstraint {
                master,
                slave,
                phase: phase * orientation as f64,
            })
        })
        .collect()
}

/// Expand periodic edge pairs for the order-2 Nédélec edge modes.
///
/// The first edge mode is odd under reversal while the second mode is even.
/// An oriented pair therefore produces one signed and one unsigned
/// constraint. `edge_pairs` contains global mesh-edge indices, not expanded
/// DOF indices.
pub fn expand_order2_oriented_hcurl_constraints(
    edge_pairs: &[(usize, usize, i8)],
    phase: Complex64,
) -> Result<Vec<PeriodicDofConstraint>, MpcError> {
    if !phase.re.is_finite() || !phase.im.is_finite() {
        return Err(MpcError::NonFiniteCoefficient {
            master: usize::MAX,
            slave: usize::MAX,
        });
    }

    let mut constraints = Vec::with_capacity(edge_pairs.len().saturating_mul(2));
    for &(master_edge, slave_edge, orientation) in edge_pairs {
        if !matches!(orientation, -1 | 1) {
            return Err(MpcError::InvalidOrientation(orientation));
        }

        let master_base = master_edge
            .checked_mul(2)
            .ok_or(MpcError::DofIndexOverflow(master_edge))?;
        let slave_base = slave_edge
            .checked_mul(2)
            .ok_or(MpcError::DofIndexOverflow(slave_edge))?;
        let odd_phase = phase * f64::from(orientation);

        constraints.push(PeriodicDofConstraint {
            master: master_base,
            slave: slave_base,
            phase: odd_phase,
        });
        constraints.push(PeriodicDofConstraint {
            master: master_base + 1,
            slave: slave_base + 1,
            phase,
        });
    }
    Ok(constraints)
}

/// One order-2 tetrahedral H(curl) periodic face pair.
///
/// The vertex arrays describe the same physical face correspondence: entry
/// `i` in `master_vertices` maps to entry `i` in `slave_vertices`. The arrays
/// need not be sorted; the implementation canonicalizes both faces by their
/// global vertex IDs before deriving the two-mode face permutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HcurlOrder2PeriodicFacePair {
    pub master_face: usize,
    pub slave_face: usize,
    pub master_vertices: [usize; 3],
    pub slave_vertices: [usize; 3],
}

/// Expand order-2 tetrahedral H(curl) face-bubble periodic constraints.
///
/// The two canonical face modes are `lambda_a W_bc` and `lambda_b W_ca`.
/// A vertex permutation is converted to its exact 2x2 change-of-basis matrix;
/// the resulting MPC rows therefore handle cyclic and reversed face orderings
/// instead of reducing all face orientation to a scalar sign.
pub fn expand_order2_hcurl_face_constraints(
    face_pairs: &[HcurlOrder2PeriodicFacePair],
    edge_dof_count: usize,
    phase: Complex64,
) -> Result<Vec<MultiPointConstraint>, MpcError> {
    if !phase.re.is_finite() || !phase.im.is_finite() {
        return Err(MpcError::NonFiniteCoefficient {
            master: usize::MAX,
            slave: usize::MAX,
        });
    }

    let mut constraints = Vec::with_capacity(face_pairs.len().saturating_mul(2));
    for pair in face_pairs {
        let matrix = order2_face_mode_change_matrix(pair.master_vertices, pair.slave_vertices)
            .ok_or(MpcError::InvalidFaceVertexMap {
                master_face: pair.master_face,
                slave_face: pair.slave_face,
            })?;
        let master_base = edge_dof_count
            .checked_add(
                pair.master_face
                    .checked_mul(2)
                    .ok_or(MpcError::DofIndexOverflow(pair.master_face))?,
            )
            .ok_or(MpcError::DofIndexOverflow(pair.master_face))?;
        let slave_base = edge_dof_count
            .checked_add(
                pair.slave_face
                    .checked_mul(2)
                    .ok_or(MpcError::DofIndexOverflow(pair.slave_face))?,
            )
            .ok_or(MpcError::DofIndexOverflow(pair.slave_face))?;
        let master_mode1 = master_base
            .checked_add(1)
            .ok_or(MpcError::DofIndexOverflow(pair.master_face))?;
        let slave_mode1 = slave_base
            .checked_add(1)
            .ok_or(MpcError::DofIndexOverflow(pair.slave_face))?;
        let master_dofs = [master_base, master_mode1];
        let slave_dofs = [slave_base, slave_mode1];

        for (row, &slave_dof) in slave_dofs.iter().enumerate() {
            let masters = [0usize, 1usize]
                .into_iter()
                .filter_map(|column| {
                    let coefficient = phase * matrix[row][column];
                    (coefficient.norm() > 0.0).then_some(MpcTerm {
                        dof: master_dofs[column],
                        coefficient,
                    })
                })
                .collect();
            constraints.push(MultiPointConstraint {
                slave: slave_dof,
                masters,
            });
        }
    }
    Ok(constraints)
}

fn order2_face_mode_change_matrix(
    master_vertices: [usize; 3],
    slave_vertices: [usize; 3],
) -> Option<[[f64; 2]; 2]> {
    if has_duplicate_face_vertex(master_vertices) || has_duplicate_face_vertex(slave_vertices) {
        return None;
    }

    let mut master_sorted = master_vertices;
    let mut slave_sorted = slave_vertices;
    master_sorted.sort_unstable();
    slave_sorted.sort_unstable();

    let mut permutation = [0usize; 3];
    for (master_position, &master_vertex) in master_sorted.iter().enumerate() {
        let correspondence_index = master_vertices
            .iter()
            .position(|&vertex| vertex == master_vertex)?;
        let slave_vertex = slave_vertices[correspondence_index];
        let slave_position = slave_sorted
            .iter()
            .position(|&vertex| vertex == slave_vertex)?;
        // The change-of-basis maps slave rows back to master columns, so it
        // needs the inverse of the master-to-slave vertex correspondence.
        permutation[slave_position] = master_position;
    }

    if permutation == [0, 1, 2] {
        return Some([[1.0, 0.0], [0.0, 1.0]]);
    }
    if permutation == [2, 0, 1] {
        return Some([[-1.0, -1.0], [1.0, 0.0]]);
    }
    if permutation == [0, 2, 1] {
        return Some([[-1.0, 0.0], [1.0, 1.0]]);
    }
    let parity = permutation_parity(permutation);
    let mut matrix = [[0.0; 2]; 2];
    add_face_mode_column(&mut matrix, 0, permutation[0], parity);
    add_face_mode_column(&mut matrix, 1, permutation[1], parity);
    Some(matrix)
}

fn has_duplicate_face_vertex(vertices: [usize; 3]) -> bool {
    vertices[0] == vertices[1] || vertices[0] == vertices[2] || vertices[1] == vertices[2]
}

fn permutation_parity(permutation: [usize; 3]) -> f64 {
    let inversions = (0..3)
        .flat_map(|left| ((left + 1)..3).map(move |right| (left, right)))
        .filter(|&(left, right)| permutation[left] > permutation[right])
        .count();
    if inversions % 2 == 0 { 1.0 } else { -1.0 }
}

fn add_face_mode_column(
    matrix: &mut [[f64; 2]; 2],
    column: usize,
    first_vertex: usize,
    parity: f64,
) {
    match first_vertex {
        0 => matrix[0][column] += parity,
        1 => matrix[1][column] += parity,
        2 => {
            matrix[0][column] -= parity;
            matrix[1][column] -= parity;
        }
        _ => unreachable!("face permutation has exactly three positions"),
    }
}

/// A triangular RT1 H(div) periodic face pair.
///
/// `vertex_permutation[i]` gives the local index on the slave face that
/// corresponds to local master-face vertex `i`. The three face moments use
/// the polynomial basis `[1, a, b]` in each ordered triangular face frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdivRt1PeriodicFacePair {
    pub master_face: usize,
    pub slave_face: usize,
    pub vertex_permutation: [usize; 3],
}

/// Expand RT1 triangular-face periodic constraints.
///
/// The change of basis is derived from the barycentric coordinate
/// permutation, so cyclic and reversed face orderings are handled without a
/// caller-supplied dense matrix. This bounded contract is for tetrahedral
/// triangular faces whose three global face modes are contiguous.
pub fn expand_hdiv_rt1_periodic_face_constraints(
    face_pairs: &[HdivRt1PeriodicFacePair],
    phase: Complex64,
) -> Result<Vec<MultiPointConstraint>, MpcError> {
    if !phase.re.is_finite() || !phase.im.is_finite() {
        return Err(MpcError::NonFiniteCoefficient {
            master: usize::MAX,
            slave: usize::MAX,
        });
    }
    let mut constraints = Vec::with_capacity(face_pairs.len().saturating_mul(3));
    for pair in face_pairs {
        let matrix = rt1_face_mode_change_matrix(pair.vertex_permutation).ok_or(
            MpcError::InvalidFaceVertexMap {
                master_face: pair.master_face,
                slave_face: pair.slave_face,
            },
        )?;
        let master_base = pair
            .master_face
            .checked_mul(3)
            .ok_or(MpcError::DofIndexOverflow(pair.master_face))?;
        let slave_base = pair
            .slave_face
            .checked_mul(3)
            .ok_or(MpcError::DofIndexOverflow(pair.slave_face))?;
        let orientation = permutation_parity(pair.vertex_permutation);
        for (row, matrix_row) in matrix.iter().enumerate() {
            let mut masters = Vec::with_capacity(3);
            for (column, &entry) in matrix_row.iter().enumerate() {
                let coefficient = phase * orientation * entry;
                if coefficient.norm() > 0.0 {
                    masters.push(MpcTerm {
                        dof: master_base
                            .checked_add(column)
                            .ok_or(MpcError::DofIndexOverflow(pair.master_face))?,
                        coefficient,
                    });
                }
            }
            constraints.push(MultiPointConstraint {
                slave: slave_base
                    .checked_add(row)
                    .ok_or(MpcError::DofIndexOverflow(pair.slave_face))?,
                masters,
            });
        }
    }
    Ok(constraints)
}

fn rt1_face_mode_change_matrix(permutation: [usize; 3]) -> Option<[[f64; 3]; 3]> {
    if permutation.iter().any(|&index| index >= 3)
        || (0..3).any(|left| (left + 1..3).any(|right| permutation[left] == permutation[right]))
    {
        return None;
    }
    let mut inverse = [0usize; 3];
    for (master, &slave) in permutation.iter().enumerate() {
        inverse[slave] = master;
    }
    let barycentric = |index: usize| match index {
        0 => [1.0, -1.0, -1.0],
        1 => [0.0, 1.0, 0.0],
        2 => [0.0, 0.0, 1.0],
        _ => unreachable!(),
    };
    Some([
        [1.0, 0.0, 0.0],
        barycentric(inverse[1]),
        barycentric(inverse[2]),
    ])
}

#[allow(dead_code)]
fn permutation_parity_i8(permutation: [usize; 3]) -> i8 {
    let inversions = (0..3)
        .flat_map(|left| {
            (left + 1..3).map(move |right| (permutation[left] > permutation[right]) as i8)
        })
        .sum::<i8>();
    if inversions % 2 == 0 { 1 } else { -1 }
}

/// Complete cyclic mapping extracted from two marked faces of a 3D sector.
///
/// Scalar H1 fields use the node pairing directly. Cartesian nodal vectors
/// additionally use `transform.rotation`. Lowest-order H(curl) fields use the
/// oriented edge pairing: rotation of the physical vector and edge tangent is
/// then represented exactly by the transformed line integral and orientation
/// sign.
#[derive(Debug, Clone, PartialEq)]
pub struct SectorPeriodicMap {
    pub node_pairs: Vec<(usize, usize)>,
    pub edge_pairs: Vec<(usize, usize, i8)>,
    pub face_pairs: Vec<(usize, usize, i8)>,
    pub transform: PeriodicTransform,
    pub phase: Complex64,
}

impl SectorPeriodicMap {
    #[allow(clippy::too_many_arguments)]
    pub fn from_marked_faces(
        mesh: &Mesh,
        master_marker: i32,
        slave_marker: i32,
        sector_angle_rad: f64,
        circumferential_order: i32,
        tolerance: f64,
    ) -> Result<Self, MpcError> {
        if !sector_angle_rad.is_finite() || sector_angle_rad <= 0.0 {
            return Err(MpcError::InvalidSectorAngle(sector_angle_rad));
        }
        let transform = PeriodicTransform::rotation_about_z(sector_angle_rad);
        let mut masters = marked_boundary_nodes(mesh, master_marker);
        let mut slaves = marked_boundary_nodes(mesh, slave_marker);
        masters.sort_unstable();
        slaves.sort_unstable();
        let points: Vec<_> = mesh
            .nodes
            .iter()
            .map(|point| [point.x, point.y, point.z])
            .collect();
        let node_pairs =
            pair_periodic_nodes_geometric(&points, &masters, &slaves, transform, tolerance)?;

        let topology = MeshTopology::build(mesh);
        let master_edges = marked_boundary_edges(&topology, master_marker);
        let slave_edges = marked_boundary_edges(&topology, slave_marker);
        let edge_pairs = pair_periodic_edges_oriented(&master_edges, &slave_edges, &node_pairs)?;
        let master_faces = marked_boundary_faces(&topology, master_marker);
        let slave_faces = marked_boundary_faces(&topology, slave_marker);
        let face_pairs = pair_periodic_faces_oriented(&master_faces, &slave_faces, &node_pairs)?;
        Ok(Self {
            node_pairs,
            edge_pairs,
            face_pairs,
            transform,
            phase: sector_phase(circumferential_order, sector_angle_rad),
        })
    }

    /// Lowest-order H(curl) line-integral constraints.
    pub fn hcurl_constraints(&self) -> Result<Vec<PeriodicDofConstraint>, MpcError> {
        expand_oriented_dof_constraints(&self.edge_pairs, self.phase)
    }

    /// Lowest-order 3-D H(div) normal-flux constraints on periodic faces.
    pub fn hdiv_rt0_face_constraints_3d(&self) -> Result<Vec<PeriodicDofConstraint>, MpcError> {
        expand_oriented_dof_constraints(&self.face_pairs, self.phase)
    }

    /// First-order 3-D H(div) triangular-face mode constraints.
    pub fn hdiv_rt1_face_constraints_3d(
        &self,
        face_pairs: &[HdivRt1PeriodicFacePair],
    ) -> Result<Vec<MultiPointConstraint>, MpcError> {
        expand_hdiv_rt1_periodic_face_constraints(face_pairs, self.phase)
    }

    /// Pair marked triangular faces through the map's node correspondence
    /// and expand their RT1 periodic mode constraints.
    pub fn hdiv_rt1_face_constraints_from_faces(
        &self,
        master_faces: &[(usize, Face)],
        slave_faces: &[(usize, Face)],
    ) -> Result<Vec<MultiPointConstraint>, MpcError> {
        let pairs = pair_periodic_hdiv_rt1_faces(master_faces, slave_faces, &self.node_pairs)?;
        self.hdiv_rt1_face_constraints_3d(&pairs)
    }

    /// Higher-order H(curl) edge constraints for the order-2 Nédélec space.
    ///
    /// This expands each mesh-edge pair into its two global edge modes. The
    /// first mode carries the edge orientation sign; the second mode is even
    /// under reversal.
    pub fn hcurl_order2_constraints(&self) -> Result<Vec<PeriodicDofConstraint>, MpcError> {
        expand_order2_oriented_hcurl_constraints(&self.edge_pairs, self.phase)
    }

    /// Expand order-2 H(curl) face-mode constraints using this map's phase.
    pub fn hcurl_order2_face_constraints(
        &self,
        face_pairs: &[HcurlOrder2PeriodicFacePair],
        edge_dof_count: usize,
    ) -> Result<Vec<MultiPointConstraint>, MpcError> {
        expand_order2_hcurl_face_constraints(face_pairs, edge_dof_count, self.phase)
    }

    /// Build one checked MPC system containing this map's order-2 H(curl)
    /// edge and face constraints.
    pub fn hcurl_order2_mpc_system(
        &self,
        num_full_dofs: usize,
        face_pairs: &[HcurlOrder2PeriodicFacePair],
        edge_dof_count: usize,
    ) -> Result<MpcSystem, MpcError> {
        let edge_constraints = self.hcurl_order2_constraints()?.into_iter().map(Into::into);
        let face_constraints = self
            .hcurl_order2_face_constraints(face_pairs, edge_dof_count)?
            .into_iter();
        let constraints = edge_constraints.chain(face_constraints).collect::<Vec<_>>();
        MpcSystem::new(num_full_dofs, &constraints)
    }

    /// Condense a complex FEM system using this map's order-2 H(curl) rows.
    ///
    /// This preserves the checked MPC-before-Dirichlet ordering and returns a
    /// solver-ready reduced `FemSystem` whose full field can be reconstructed
    /// with the same checked MPC system.
    pub fn condense_hcurl_order2_fem_system_complex(
        &self,
        system: &FemSystem<Complex64>,
        face_pairs: &[HcurlOrder2PeriodicFacePair],
        edge_dof_count: usize,
    ) -> Result<FemSystem<Complex64>, MpcError> {
        let mpc =
            self.hcurl_order2_mpc_system(system.matrix.num_rows, face_pairs, edge_dof_count)?;
        mpc.condense_fem_system_complex(system)
    }

    /// Scalar nodal constraints with a global DOF offset.
    ///
    /// If `active_nodes` is supplied, pairs that are inactive on both faces are
    /// omitted. A pair active on only one side indicates a non-conforming
    /// material interface and is rejected.
    pub fn scalar_constraints(
        &self,
        dof_offset: usize,
        active_nodes: Option<&HashSet<usize>>,
    ) -> Result<Vec<MultiPointConstraint>, MpcError> {
        let mut constraints = Vec::new();
        for &(master, slave) in &self.node_pairs {
            if let Some(active) = active_nodes {
                match (active.contains(&master), active.contains(&slave)) {
                    (false, false) => continue,
                    (true, true) => {}
                    _ => {
                        return Err(MpcError::PeriodicActivityMismatch { master, slave });
                    }
                }
            }
            constraints.push(MultiPointConstraint {
                slave: dof_offset + slave,
                masters: vec![MpcTerm {
                    dof: dof_offset + master,
                    coefficient: self.phase,
                }],
            });
        }
        Ok(constraints)
    }

    /// Cartesian nodal vector constraints `u_s = phase R u_m`.
    pub fn rotated_vector_constraints(
        &self,
        dofs_per_node: usize,
    ) -> Result<Vec<MultiPointConstraint>, MpcError> {
        expand_rotated_vector_constraints(
            &self.node_pairs,
            dofs_per_node,
            self.transform.rotation,
            self.phase,
        )
    }
}

fn marked_boundary_nodes(mesh: &Mesh, marker: i32) -> Vec<usize> {
    let mut nodes: Vec<_> = mesh
        .boundaries
        .iter()
        .filter(|face| face.marker == marker)
        .flat_map(|face| face.nodes.iter().copied())
        .collect();
    nodes.sort_unstable();
    nodes.dedup();
    nodes
}

/// Pair nodes on two marked boundaries using a geometric transform.
///
/// The transform maps master coordinates to their expected slave
/// coordinates. The result can be converted into scalar periodic constraints
/// with expand_node_periodic_constraints.
pub fn pair_marked_boundary_nodes(
    mesh: &Mesh,
    master_marker: i32,
    slave_marker: i32,
    transform: PeriodicTransform,
    tolerance: f64,
) -> Result<Vec<(usize, usize)>, MpcError> {
    let masters = marked_boundary_nodes(mesh, master_marker);
    let slaves = marked_boundary_nodes(mesh, slave_marker);
    let points: Vec<_> = mesh
        .nodes
        .iter()
        .map(|node| [node.x, node.y, node.z])
        .collect();
    pair_periodic_nodes_geometric(&points, &masters, &slaves, transform, tolerance)
}

fn marked_boundary_edges(topology: &MeshTopology, marker: i32) -> Vec<(usize, [usize; 2])> {
    topology
        .boundary_edges
        .get(&marker)
        .into_iter()
        .flatten()
        .map(|&edge| {
            let (start, end) = topology.edges[edge];
            (edge, [start, end])
        })
        .collect()
}

fn marked_boundary_faces(topology: &MeshTopology, marker: i32) -> Vec<(usize, Face)> {
    topology
        .boundary_faces
        .get(&marker)
        .into_iter()
        .flatten()
        .map(|&face| (face, topology.faces[face].clone()))
        .collect()
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum MpcError {
    #[error("DOF {dof} is outside 0..{num_dofs}")]
    DofOutOfRange { dof: usize, num_dofs: usize },
    #[error("order-2 H(curl) edge index {0} overflows its two-mode DOF pair")]
    DofIndexOverflow(usize),
    #[error("periodic H(curl) face vertex map is invalid for faces {master_face} -> {slave_face}")]
    InvalidFaceVertexMap {
        master_face: usize,
        slave_face: usize,
    },
    #[error("constraint for slave DOF {slave} has no masters")]
    EmptyMasters { slave: usize },
    #[error("slave DOF {0} has more than one constraint")]
    DuplicateSlave(usize),
    #[error("constraint cycle contains DOF {0}")]
    ConstraintCycle(usize),
    #[error("constraint coefficient from master {master} to slave {slave} is non-finite")]
    NonFiniteCoefficient { slave: usize, master: usize },
    #[error("constraint expansion at full DOF {full_dof} is complex in a real system")]
    ComplexCoefficientInRealSystem { full_dof: usize },
    #[error("matrix is {rows}x{cols}; expected {expected}x{expected}")]
    MatrixDimension {
        rows: usize,
        cols: usize,
        expected: usize,
    },
    #[error("vector has length {actual}; expected {expected}")]
    VectorDimension { actual: usize, expected: usize },
    #[error("periodic solution value at DOF {dof} is non-finite")]
    NonFiniteVector { dof: usize },
    #[error("FEM system matrix triplet {triplet} is non-finite")]
    NonFiniteMatrixValue { triplet: usize },
    #[error("FEM system RHS entry {index} is non-finite")]
    NonFiniteRhsEntry { index: usize },
    #[error("condensed matrix entry ({row}, {column}) is non-finite")]
    NonFiniteCondensedValue { row: usize, column: usize },
    #[error("condensed RHS entry {reduced_dof} is non-finite")]
    NonFiniteCondensedRhs { reduced_dof: usize },
    #[error("FEM system contains Dirichlet constraints; condense MPCs first")]
    DirichletConstraintsPresent,
    #[error("invalid FEM system: {0}")]
    InvalidFemSystem(FemSystemError),
    #[error("periodic tolerance {0} must be finite and positive")]
    InvalidTolerance(f64),
    #[error(
        "full state at DOF {full_dof} violates the MPC reconstruction by {residual} (tolerance {tolerance})"
    )]
    StateConstraintResidual {
        full_dof: usize,
        residual: f64,
        tolerance: f64,
    },
    #[error("periodic faces have {masters} master nodes and {slaves} slave nodes")]
    PairCountMismatch { masters: usize, slaves: usize },
    #[error("node index {0} is outside the coordinate array")]
    NodeOutOfRange(usize),
    #[error("edge endpoint node {0} is absent from the periodic node pairing")]
    UnpairedEdgeNode(usize),
    #[error("master node {master} has no slave within tolerance")]
    NoGeometricMatch { master: usize },
    #[error("master node {master} has multiple equally close slave candidates")]
    AmbiguousGeometricMatch { master: usize },
    #[error("master edge DOF {master_dof} has no matching slave edge")]
    NoPeriodicEdgeMatch { master_dof: usize },
    #[error("master edge DOF {master_dof} has multiple matching slave edges")]
    AmbiguousPeriodicEdgeMatch { master_dof: usize },
    #[error("face node {0} absent periodic node pairing")]
    UnpairedFaceNode(usize),
    #[error("master face DOF {master_dof} has no matching slave face")]
    NoPeriodicFaceMatch { master_dof: usize },
    #[error("master face DOF {master_dof} has multiple matching slave faces")]
    AmbiguousPeriodicFaceMatch { master_dof: usize },
    #[error("vector periodic constraints support one to three components, not {0}")]
    InvalidComponents(usize),
    #[error("oriented periodic DOF sign must be -1 or +1, not {0}")]
    InvalidOrientation(i8),
    #[error("sector angle {0} must be finite and positive")]
    InvalidSectorAngle(f64),
    #[error("periodic material activity differs for paired nodes {master} and {slave}")]
    PeriodicActivityMismatch { master: usize, slave: usize },
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::FRAC_PI_2;

    #[test]
    fn checked_periodic_residual_rejects_short_and_nonfinite_vectors() {
        let constraint = PeriodicDofConstraint {
            master: 2,
            slave: 5,
            phase: Complex64::new(1.0, 0.0),
        };
        assert_eq!(
            constraint.residual_checked(&vec![Complex64::new(0.0, 0.0); 5]),
            Err(MpcError::VectorDimension {
                actual: 5,
                expected: 6,
            })
        );

        let mut solution = vec![Complex64::new(0.0, 0.0); 6];
        solution[2].re = f64::NAN;
        assert_eq!(
            constraint.residual_checked(&solution),
            Err(MpcError::NonFiniteVector { dof: 2 })
        );

        let nonfinite_phase = PeriodicDofConstraint {
            phase: Complex64::new(f64::INFINITY, 0.0),
            ..constraint
        };
        assert_eq!(
            nonfinite_phase.residual_checked(&vec![Complex64::new(0.0, 0.0); 6]),
            Err(MpcError::NonFiniteCoefficient {
                slave: 5,
                master: 2,
            })
        );
    }

    #[test]
    fn checked_prolongation_row_rejects_out_of_range_dof() {
        let system = MpcSystem::from_periodic(
            3,
            &[PeriodicDofConstraint {
                master: 0,
                slave: 2,
                phase: Complex64::new(1.0, 0.0),
            }],
        )
        .expect("valid periodic constraint");
        assert!(matches!(
            system.try_prolongation_row(3),
            Err(MpcError::DofOutOfRange {
                dof: 3,
                num_dofs: 3
            })
        ));
    }

    #[test]
    fn sector_phase_matches_quarter_wave() {
        let phase = sector_phase(1, FRAC_PI_2);
        assert!(phase.re.abs() < 1.0e-14);
        assert!((phase.im - 1.0).abs() < 1.0e-14);
    }

    #[test]
    fn node_pairs_expand_to_component_constraints() {
        let constraints =
            expand_node_periodic_constraints(&[(2, 5), (3, 6)], 3, Complex64::new(-1.0, 0.0));
        assert_eq!(constraints.len(), 6);
        assert_eq!(constraints[0].master, 6);
        assert_eq!(constraints[0].slave, 15);
        assert_eq!(constraints[5].master, 11);
        assert_eq!(constraints[5].slave, 20);
    }

    #[test]
    fn residual_uses_slave_minus_phase_master() {
        let constraint = PeriodicDofConstraint {
            master: 0,
            slave: 1,
            phase: Complex64::new(0.0, 1.0),
        };
        let solution = [Complex64::new(2.0, 0.0), Complex64::new(0.0, 2.0)];
        assert_eq!(constraint.residual(&solution), Complex64::new(0.0, 0.0));
    }

    #[test]
    fn mpc_condensation_matches_p_hermitian_a_p_and_recovers_periodicity() {
        let matrix = CsrMatrix::from_triplets(
            3,
            3,
            vec![
                (0, 0, Complex64::new(4.0, 0.0)),
                (0, 1, Complex64::new(1.0, 1.0)),
                (1, 0, Complex64::new(1.0, -1.0)),
                (1, 1, Complex64::new(3.0, 0.0)),
                (1, 2, Complex64::new(2.0, 0.0)),
                (2, 1, Complex64::new(2.0, 0.0)),
                (2, 2, Complex64::new(5.0, 0.0)),
            ],
        );
        let periodic = PeriodicDofConstraint {
            master: 0,
            slave: 2,
            phase: Complex64::new(0.0, 1.0),
        };
        let mpc = MpcSystem::from_periodic(3, &[periodic]).unwrap();
        let condensed = mpc.condense_matrix(&matrix).unwrap();

        let value = |row, column| {
            condensed
                .row_entries(row)
                .find(|&(entry_column, _)| entry_column == column)
                .map(|(_, value)| value)
                .unwrap_or_default()
        };
        assert!((value(0, 0) - Complex64::new(9.0, 0.0)).norm() < 1.0e-12);
        assert!((value(0, 1) - Complex64::new(1.0, -1.0)).norm() < 1.0e-12);
        assert!((value(1, 0) - Complex64::new(1.0, 1.0)).norm() < 1.0e-12);
        assert!((value(1, 1) - Complex64::new(3.0, 0.0)).norm() < 1.0e-12);

        let reduced = [Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.5)];
        let recovered = mpc.recover(&reduced).unwrap();
        assert!(periodic.residual(&recovered).norm() < 1.0e-14);
        let rhs = mpc
            .condense_rhs(&[
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 3.0),
            ])
            .unwrap();
        assert_eq!(
            rhs,
            vec![Complex64::new(4.0, 0.0), Complex64::new(2.0, 0.0)]
        );
    }

    #[test]
    fn real_mpc_condensation_accepts_antiperiodic_and_rejects_complex_phase() {
        let matrix = CsrMatrix::from_triplets(
            2,
            2,
            vec![(0, 0, 2.0), (0, 1, -1.0), (1, 0, -1.0), (1, 1, 2.0)],
        );
        let anti = MpcSystem::from_periodic(
            2,
            &[PeriodicDofConstraint {
                master: 0,
                slave: 1,
                phase: Complex64::new(-1.0, 0.0),
            }],
        )
        .unwrap();
        let condensed = anti.condense_real_matrix(&matrix).unwrap();
        assert_eq!(condensed.num_rows, 1);
        assert!((condensed.values[0] - 6.0).abs() < 1.0e-14);

        let complex = MpcSystem::from_periodic(
            2,
            &[PeriodicDofConstraint {
                master: 0,
                slave: 1,
                phase: Complex64::new(0.0, 1.0),
            }],
        )
        .unwrap();
        assert!(matches!(
            complex.condense_real_matrix(&matrix),
            Err(MpcError::ComplexCoefficientInRealSystem { .. })
        ));
    }

    #[test]
    fn chained_constraints_resolve_and_cycles_are_rejected() {
        let constraints = [
            MultiPointConstraint {
                slave: 1,
                masters: vec![MpcTerm {
                    dof: 0,
                    coefficient: Complex64::new(2.0, 0.0),
                }],
            },
            MultiPointConstraint {
                slave: 2,
                masters: vec![MpcTerm {
                    dof: 1,
                    coefficient: Complex64::new(3.0, 0.0),
                }],
            },
        ];
        let mpc = MpcSystem::new(3, &constraints).unwrap();
        assert_eq!(
            mpc.recover(&[Complex64::new(1.0, 0.0)]).unwrap(),
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(6.0, 0.0)
            ]
        );

        let cycle = [
            MultiPointConstraint {
                slave: 0,
                masters: vec![MpcTerm {
                    dof: 1,
                    coefficient: Complex64::new(1.0, 0.0),
                }],
            },
            MultiPointConstraint {
                slave: 1,
                masters: vec![MpcTerm {
                    dof: 0,
                    coefficient: Complex64::new(1.0, 0.0),
                }],
            },
        ];
        assert!(matches!(
            MpcSystem::new(2, &cycle),
            Err(MpcError::ConstraintCycle(_))
        ));
    }

    #[test]
    fn condense_complex_fem_system_returns_solver_ready_reduced_system() {
        let mut matrix = TripletMatrix::new(3, 3);
        matrix.push(0, 0, Complex64::new(4.0, 0.0));
        matrix.push(0, 1, Complex64::new(1.0, 1.0));
        matrix.push(1, 0, Complex64::new(1.0, -1.0));
        matrix.push(1, 1, Complex64::new(3.0, 0.0));
        matrix.push(1, 2, Complex64::new(2.0, 0.0));
        matrix.push(2, 1, Complex64::new(2.0, 0.0));
        matrix.push(2, 2, Complex64::new(5.0, 0.0));
        let system = FemSystem::new(
            matrix,
            vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(0.0, 3.0),
            ],
        );
        let periodic = MpcSystem::from_periodic(
            3,
            &[PeriodicDofConstraint {
                master: 0,
                slave: 2,
                phase: Complex64::new(0.0, 1.0),
            }],
        )
        .unwrap();

        let condensed = periodic
            .condense_fem_system_complex(&system)
            .expect("valid complex FEM system should condense");
        assert_eq!(condensed.matrix.num_rows, 2);
        assert_eq!(condensed.matrix.num_cols, 2);
        assert_eq!(
            condensed.rhs,
            vec![Complex64::new(4.0, 0.0), Complex64::new(2.0, 0.0)]
        );
        let csr = condensed.matrix.to_csr();
        assert!((csr.get(0, 0) - Complex64::new(9.0, 0.0)).norm() < 1.0e-12);
        assert!((csr.get(0, 1) - Complex64::new(1.0, -1.0)).norm() < 1.0e-12);
        assert!((csr.get(1, 0) - Complex64::new(1.0, 1.0)).norm() < 1.0e-12);
        assert!((csr.get(1, 1) - Complex64::new(3.0, 0.0)).norm() < 1.0e-12);

        let recovered = periodic
            .recover(&[Complex64::new(2.0, 0.0), Complex64::new(-1.0, 0.5)])
            .unwrap();
        assert!(
            PeriodicDofConstraint {
                master: 0,
                slave: 2,
                phase: Complex64::new(0.0, 1.0),
            }
            .residual(&recovered)
            .norm()
                < 1.0e-14
        );
    }

    #[test]
    fn condense_real_fem_system_supports_antiperiodic_constraints() {
        let mut matrix = TripletMatrix::new(2, 2);
        matrix.push(0, 0, 2.0);
        matrix.push(0, 1, -1.0);
        matrix.push(1, 0, -1.0);
        matrix.push(1, 1, 2.0);
        let system = FemSystem::new(matrix, vec![3.0, 4.0]);
        let periodic = MpcSystem::from_periodic(
            2,
            &[PeriodicDofConstraint {
                master: 0,
                slave: 1,
                phase: Complex64::new(-1.0, 0.0),
            }],
        )
        .unwrap();

        let condensed = periodic
            .condense_fem_system_real(&system)
            .expect("real anti-periodic FEM system should condense");
        assert_eq!(condensed.rhs, vec![-1.0]);
        let csr = condensed.matrix.to_csr();
        assert!((csr.get(0, 0) - 6.0).abs() < 1.0e-12);
    }

    #[test]
    fn fem_system_condensation_rejects_nonfinite_invalid_and_dirichlet_systems() {
        let periodic = MpcSystem::from_periodic(
            2,
            &[PeriodicDofConstraint {
                master: 0,
                slave: 1,
                phase: Complex64::new(1.0, 0.0),
            }],
        )
        .unwrap();

        let mut nonfinite = TripletMatrix::new(2, 2);
        nonfinite.push(0, 0, f64::NAN);
        let system = FemSystem::new(nonfinite, vec![0.0, 0.0]);
        assert!(matches!(
            periodic.condense_fem_system_real(&system),
            Err(MpcError::NonFiniteMatrixValue { triplet: 0 })
        ));

        let huge_mpc = MpcSystem::new(
            2,
            &[MultiPointConstraint {
                slave: 1,
                masters: vec![MpcTerm {
                    dof: 0,
                    coefficient: Complex64::new(f64::MAX, 0.0),
                }],
            }],
        )
        .unwrap();
        let mut huge_matrix = TripletMatrix::new(2, 2);
        huge_matrix.push(1, 1, f64::MAX);
        let huge_system = FemSystem::new(huge_matrix, vec![0.0, 0.0]);
        assert!(matches!(
            huge_mpc.condense_fem_system_real(&huge_system),
            Err(MpcError::NonFiniteCondensedValue { .. })
        ));

        let rhs_overflow_mpc = MpcSystem::new(
            2,
            &[MultiPointConstraint {
                slave: 1,
                masters: vec![MpcTerm {
                    dof: 0,
                    coefficient: Complex64::new(2.0, 0.0),
                }],
            }],
        )
        .unwrap();
        let rhs_overflow = FemSystem::new(TripletMatrix::new(2, 2), vec![0.0, f64::MAX]);
        assert!(matches!(
            rhs_overflow_mpc.condense_fem_system_real(&rhs_overflow),
            Err(MpcError::NonFiniteCondensedRhs { .. })
        ));

        let mut malformed = TripletMatrix::new(2, 2);
        malformed.push(2, 0, 1.0);
        let system = FemSystem::new(malformed, vec![0.0, 0.0]);
        assert!(matches!(
            periodic.condense_fem_system_real(&system),
            Err(MpcError::InvalidFemSystem(
                FemSystemError::MatrixIndexOutOfBounds { .. }
            ))
        ));

        let matrix = TripletMatrix::new(2, 2);
        let mut system = FemSystem::new(matrix, vec![0.0, 0.0]);
        crate::boundary::generic::try_apply_homogeneous_dirichlet_dofs(&mut system, &[0])
            .expect("valid Dirichlet fixture");
        assert!(matches!(
            periodic.condense_fem_system_real(&system),
            Err(MpcError::DirichletConstraintsPresent)
        ));
    }

    #[test]
    fn geometric_pairing_and_vector_rotation_match_sector_faces() {
        let points = [
            [1.0, 0.0, 0.0],
            [2.0, 0.0, 1.0],
            [0.0, 2.0, 1.0],
            [0.0, 1.0, 0.0],
        ];
        let transform = PeriodicTransform::rotation_about_z(FRAC_PI_2);
        let pairs =
            pair_periodic_nodes_geometric(&points, &[0, 1], &[2, 3], transform, 1.0e-12).unwrap();
        assert_eq!(pairs, vec![(0, 3), (1, 2)]);

        let constraints = expand_rotated_vector_constraints(
            &[(0, 1)],
            3,
            transform.rotation,
            Complex64::new(1.0, 0.0),
        )
        .unwrap();
        let mpc = MpcSystem::new(6, &constraints).unwrap();
        let full = mpc
            .recover(&[
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ])
            .unwrap();
        assert!((full[3] - Complex64::new(-3.0, 0.0)).norm() < 1.0e-14);
        assert!((full[4] - Complex64::new(2.0, 0.0)).norm() < 1.0e-14);
        assert!((full[5] - Complex64::new(4.0, 0.0)).norm() < 1.0e-14);
    }

    #[test]
    fn oriented_hcurl_pair_applies_edge_sign_and_floquet_phase() {
        let phase = Complex64::new(0.0, 1.0);
        let pairs = pair_periodic_edges_oriented(&[(2, [0, 1])], &[(7, [3, 2])], &[(0, 2), (1, 3)])
            .unwrap();
        assert_eq!(pairs, vec![(2, 7, -1)]);
        let constraints = expand_oriented_dof_constraints(&pairs, phase).unwrap();
        assert_eq!(constraints[0].phase, Complex64::new(0.0, -1.0));
        assert!(matches!(
            expand_oriented_dof_constraints(&[(0, 1, 0)], phase),
            Err(MpcError::InvalidOrientation(0))
        ));
    }

    #[test]
    fn order2_hcurl_periodic_edges_preserve_mode_parity_and_phase() {
        let phase = Complex64::new(0.6, 0.8);
        let map = SectorPeriodicMap {
            node_pairs: vec![(10, 20)],
            edge_pairs: vec![(3, 8, -1), (4, 9, 1)],
            face_pairs: vec![],
            transform: PeriodicTransform::IDENTITY,
            phase,
        };

        let constraints = map.hcurl_order2_constraints().unwrap();
        assert_eq!(constraints.len(), 4);
        assert_eq!(
            constraints[0],
            PeriodicDofConstraint {
                master: 6,
                slave: 16,
                phase: -phase,
            }
        );
        assert_eq!(
            constraints[1],
            PeriodicDofConstraint {
                master: 7,
                slave: 17,
                phase,
            }
        );
        assert_eq!(constraints[2].phase, phase);
        assert_eq!(constraints[3].phase, phase);

        let mpcs: Vec<_> = constraints.iter().copied().map(Into::into).collect();
        let system = MpcSystem::new(20, &mpcs).unwrap();
        let mut reduced = vec![Complex64::new(0.0, 0.0); system.num_reduced_dofs()];
        for (reduced_dof, &full_dof) in system.reduced_full_dofs().iter().enumerate() {
            reduced[reduced_dof] = Complex64::new(full_dof as f64 + 1.0, -0.25);
        }
        let full = system.recover(&reduced).unwrap();
        for constraint in &constraints {
            assert!(constraint.residual_checked(&full).unwrap().norm() < 1.0e-12);
        }
    }

    #[test]
    fn order2_hcurl_periodic_edges_reject_bad_phase_orientation_and_overflow() {
        assert!(matches!(
            expand_order2_oriented_hcurl_constraints(&[(0, 1, 1)], Complex64::new(f64::NAN, 0.0),),
            Err(MpcError::NonFiniteCoefficient { .. })
        ));
        assert!(matches!(
            expand_order2_oriented_hcurl_constraints(&[(0, 1, 0)], Complex64::new(1.0, 0.0),),
            Err(MpcError::InvalidOrientation(0))
        ));
        assert!(matches!(
            expand_order2_oriented_hcurl_constraints(
                &[(usize::MAX, 1, 1)],
                Complex64::new(1.0, 0.0),
            ),
            Err(MpcError::DofIndexOverflow(usize::MAX))
        ));
    }

    #[test]
    fn real_mpc_recovery_and_restriction_are_checked() {
        let system = MpcSystem::new(
            3,
            &[MultiPointConstraint {
                slave: 2,
                masters: vec![MpcTerm {
                    dof: 0,
                    coefficient: Complex64::new(2.0, 0.0),
                }],
            }],
        )
        .unwrap();
        assert_eq!(
            system.recover_real(&[3.0, 4.0]).unwrap(),
            vec![3.0, 4.0, 6.0]
        );
        assert_eq!(
            system
                .restrict_real_checked(&[3.0, 4.0, 6.0], 1.0e-12)
                .unwrap(),
            vec![3.0, 4.0]
        );
        assert!(matches!(
            system.restrict_real_checked(&[3.0, 4.0, 6.5], 1.0e-12),
            Err(MpcError::StateConstraintResidual { full_dof: 2, .. })
        ));
        assert!(matches!(
            system.recover_real(&[f64::NAN, 4.0]),
            Err(MpcError::NonFiniteVector { dof: 0 })
        ));
    }
}
/// First-order H(div) periodic quadrilateral face pair.
///
/// The four RT1 face modes are ordered as `[1, u, v, u * v]` in the
/// topology-owned square face frame. `vertex_permutation[i]` gives the
/// slave-face vertex corresponding to master-face vertex `i`. Only the
/// eight symmetries of a square are accepted; arbitrary permutations do not
/// define a valid quadrilateral face coordinate map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HdivRt1PeriodicQuadFacePair {
    pub master_face: usize,
    pub slave_face: usize,
    pub vertex_permutation: [usize; 4],
}

const RT1_QUAD_FACE_COORDINATES: [[i8; 2]; 4] = [[-1, -1], [1, -1], [1, 1], [-1, 1]];

const RT1_QUAD_FACE_SYMMETRIES: [[[i8; 2]; 2]; 8] = [
    [[1, 0], [0, 1]],
    [[0, -1], [1, 0]],
    [[-1, 0], [0, -1]],
    [[0, 1], [-1, 0]],
    [[-1, 0], [0, 1]],
    [[0, 1], [1, 0]],
    [[1, 0], [0, -1]],
    [[0, -1], [-1, 0]],
];

fn rt1_quad_face_mode_change_matrix(permutation: [usize; 4]) -> Option<([[f64; 4]; 4], f64)> {
    if permutation.iter().any(|&index| index >= 4)
        || (0..4).any(|left| (left + 1..4).any(|right| permutation[left] == permutation[right]))
    {
        return None;
    }

    let symmetry = RT1_QUAD_FACE_SYMMETRIES.iter().copied().find(|candidate| {
        (0..4).all(|master_vertex| {
            let master = RT1_QUAD_FACE_COORDINATES[master_vertex];
            let slave = RT1_QUAD_FACE_COORDINATES[permutation[master_vertex]];
            let mapped = [
                candidate[0][0] * master[0] + candidate[0][1] * master[1],
                candidate[1][0] * master[0] + candidate[1][1] * master[1],
            ];
            mapped == slave
        })
    })?;

    let a = f64::from(symmetry[0][0]);
    let b = f64::from(symmetry[0][1]);
    let c = f64::from(symmetry[1][0]);
    let d = f64::from(symmetry[1][1]);
    let orientation = a * d - b * c;
    Some((
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, a, b, 0.0],
            [0.0, c, d, 0.0],
            [0.0, 0.0, 0.0, a * d + b * c],
        ],
        orientation,
    ))
}

/// Expand first-order H(div) periodic constraints on quadrilateral faces.
///
/// This bounded contract uses four contiguous global modes per quadrilateral
/// face, as provided by the affine RT1 hexahedral numbering. The mode-change
/// matrix is derived from the square symmetry represented by the vertex map;
/// its determinant supplies the face orientation sign.
pub fn expand_hdiv_rt1_periodic_quad_face_constraints(
    face_pairs: &[HdivRt1PeriodicQuadFacePair],
    phase: Complex64,
) -> Result<Vec<MultiPointConstraint>, MpcError> {
    if !phase.re.is_finite() || !phase.im.is_finite() {
        return Err(MpcError::NonFiniteCoefficient {
            master: usize::MAX,
            slave: usize::MAX,
        });
    }

    let mut constraints = Vec::with_capacity(face_pairs.len().saturating_mul(4));
    for pair in face_pairs {
        let (matrix, orientation) = rt1_quad_face_mode_change_matrix(pair.vertex_permutation)
            .ok_or(MpcError::InvalidFaceVertexMap {
                master_face: pair.master_face,
                slave_face: pair.slave_face,
            })?;
        let master_base = pair
            .master_face
            .checked_mul(4)
            .ok_or(MpcError::DofIndexOverflow(pair.master_face))?;
        let slave_base = pair
            .slave_face
            .checked_mul(4)
            .ok_or(MpcError::DofIndexOverflow(pair.slave_face))?;

        for (row, matrix_row) in matrix.iter().enumerate() {
            let mut masters = Vec::with_capacity(4);
            for (column, &entry) in matrix_row.iter().enumerate() {
                let coefficient = phase * (orientation * entry);
                if coefficient.norm() > 0.0 {
                    masters.push(MpcTerm {
                        dof: master_base
                            .checked_add(column)
                            .ok_or(MpcError::DofIndexOverflow(pair.master_face))?,
                        coefficient,
                    });
                }
            }
            constraints.push(MultiPointConstraint {
                slave: slave_base
                    .checked_add(row)
                    .ok_or(MpcError::DofIndexOverflow(pair.slave_face))?,
                masters,
            });
        }
    }
    Ok(constraints)
}

/// Pair periodic quadrilateral faces and retain their square-coordinate map.
pub fn pair_periodic_hdiv_rt1_quadrilateral_faces(
    master_faces: &[(usize, Face)],
    slave_faces: &[(usize, Face)],
    node_pairs: &[(usize, usize)],
) -> Result<Vec<HdivRt1PeriodicQuadFacePair>, MpcError> {
    if master_faces.len() != slave_faces.len() {
        return Err(MpcError::PairCountMismatch {
            masters: master_faces.len(),
            slaves: slave_faces.len(),
        });
    }

    let node_map: HashMap<_, _> = node_pairs.iter().copied().collect();
    let mut used_slave_faces = HashSet::new();
    let mut pairs = Vec::with_capacity(master_faces.len());

    for &(master_face, ref master) in master_faces {
        let Face::Quad(master_nodes) = master else {
            return Err(MpcError::InvalidFaceVertexMap {
                master_face,
                slave_face: master_face,
            });
        };
        let mapped_nodes = master_nodes.map(|node| {
            node_map
                .get(&node)
                .copied()
                .ok_or(MpcError::UnpairedFaceNode(node))
        });
        let mapped_nodes = mapped_nodes.into_iter().collect::<Result<Vec<_>, _>>()?;
        if mapped_nodes.len() != 4
            || (0..4)
                .any(|left| (left + 1..4).any(|right| mapped_nodes[left] == mapped_nodes[right]))
        {
            return Err(MpcError::InvalidFaceVertexMap {
                master_face,
                slave_face: master_face,
            });
        }

        let mut mapped_sorted = mapped_nodes.clone();
        mapped_sorted.sort_unstable();
        let mut candidates = Vec::new();
        let mut saw_same_nodes = false;
        for &(slave_face, ref slave) in slave_faces {
            if used_slave_faces.contains(&slave_face) {
                continue;
            }
            let Face::Quad(slave_nodes) = slave else {
                continue;
            };
            let mut slave_sorted = slave_nodes.to_vec();
            slave_sorted.sort_unstable();
            if slave_sorted != mapped_sorted {
                continue;
            }
            saw_same_nodes = true;
            let mut permutation = [0usize; 4];
            for (master_index, &mapped_node) in mapped_nodes.iter().enumerate() {
                permutation[master_index] = slave_nodes
                    .iter()
                    .position(|&slave_node| slave_node == mapped_node)
                    .ok_or(MpcError::InvalidFaceVertexMap {
                        master_face,
                        slave_face,
                    })?;
            }
            if rt1_quad_face_mode_change_matrix(permutation).is_some() {
                candidates.push((slave_face, permutation));
            }
        }

        match candidates.as_slice() {
            [] if saw_same_nodes => {
                return Err(MpcError::InvalidFaceVertexMap {
                    master_face,
                    slave_face: master_face,
                });
            }
            [] => {
                return Err(MpcError::NoPeriodicFaceMatch {
                    master_dof: master_face,
                });
            }
            [(slave_face, vertex_permutation)] => {
                used_slave_faces.insert(*slave_face);
                pairs.push(HdivRt1PeriodicQuadFacePair {
                    master_face,
                    slave_face: *slave_face,
                    vertex_permutation: *vertex_permutation,
                });
            }
            _ => {
                return Err(MpcError::AmbiguousPeriodicFaceMatch {
                    master_dof: master_face,
                });
            }
        }
    }
    Ok(pairs)
}

impl SectorPeriodicMap {
    /// Build RT1 periodic constraints for marked quadrilateral faces.
    pub fn hdiv_rt1_quadrilateral_face_constraints_from_faces(
        &self,
        master_faces: &[(usize, Face)],
        slave_faces: &[(usize, Face)],
    ) -> Result<Vec<MultiPointConstraint>, MpcError> {
        let pairs = pair_periodic_hdiv_rt1_quadrilateral_faces(
            master_faces,
            slave_faces,
            &self.node_pairs,
        )?;
        expand_hdiv_rt1_periodic_quad_face_constraints(&pairs, self.phase)
    }
}
impl SectorPeriodicMap {
    /// Build a checked MPC system for periodic RT1 quadrilateral faces.
    pub fn hdiv_rt1_quadrilateral_face_mpc_system(
        &self,
        num_full_dofs: usize,
        master_faces: &[(usize, Face)],
        slave_faces: &[(usize, Face)],
    ) -> Result<MpcSystem, MpcError> {
        let constraints =
            self.hdiv_rt1_quadrilateral_face_constraints_from_faces(master_faces, slave_faces)?;
        MpcSystem::new(num_full_dofs, &constraints)
    }

    /// Condense a complex FEM system with periodic RT1 quadrilateral faces.
    ///
    /// The returned system is solver-ready in reduced coordinates; recover a
    /// full field with the same checked MPC system constructed from the face
    /// metadata. Dirichlet elimination remains a subsequent operation.
    pub fn condense_hdiv_rt1_quadrilateral_fem_system_complex(
        &self,
        system: &FemSystem<Complex64>,
        master_faces: &[(usize, Face)],
        slave_faces: &[(usize, Face)],
    ) -> Result<FemSystem<Complex64>, MpcError> {
        let mpc = self.hdiv_rt1_quadrilateral_face_mpc_system(
            system.matrix.num_rows,
            master_faces,
            slave_faces,
        )?;
        mpc.condense_fem_system_complex(system)
    }

    /// Build mixed RT1 constraints from topology-owned 3-D face offsets.
    ///
    /// This overload is for periodic faces belonging to one validated mesh;
    /// it prevents callers from accidentally reconstructing mixed
    /// triangular/quadrilateral offsets with a uniform mode count.
    pub fn hdiv_rt1_mixed_face_constraints_from_layout(
        &self,
        master_faces: &[(usize, Face)],
        slave_faces: &[(usize, Face)],
        layout: &crate::spaces::HdivRt1DofLayout3d,
    ) -> Result<Vec<MultiPointConstraint>, MpcError> {
        self.hdiv_rt1_mixed_face_constraints_from_faces(
            master_faces,
            slave_faces,
            &layout.face_offsets,
            &layout.face_offsets,
        )
    }

    /// Build a checked mixed RT1 MPC system from a topology-owned layout.
    pub fn hdiv_rt1_mixed_face_mpc_system_from_layout(
        &self,
        num_full_dofs: usize,
        master_faces: &[(usize, Face)],
        slave_faces: &[(usize, Face)],
        layout: &crate::spaces::HdivRt1DofLayout3d,
    ) -> Result<MpcSystem, MpcError> {
        let constraints =
            self.hdiv_rt1_mixed_face_constraints_from_layout(master_faces, slave_faces, layout)?;
        MpcSystem::new(num_full_dofs, &constraints)
    }

    /// Condense a complex FEM system using topology-owned mixed RT1 offsets.
    pub fn condense_hdiv_rt1_mixed_fem_system_complex_from_layout(
        &self,
        system: &FemSystem<Complex64>,
        master_faces: &[(usize, Face)],
        slave_faces: &[(usize, Face)],
        layout: &crate::spaces::HdivRt1DofLayout3d,
    ) -> Result<FemSystem<Complex64>, MpcError> {
        let mpc = self.hdiv_rt1_mixed_face_mpc_system_from_layout(
            system.matrix.num_rows,
            master_faces,
            slave_faces,
            layout,
        )?;
        mpc.condense_fem_system_complex(system)
    }
}
/// Expand RT1 quadrilateral constraints using explicit global face-mode
/// offsets. This is the mixed-face-safe form; `master_face` and `slave_face`
/// index the corresponding entries in the two offset arrays.
pub fn expand_hdiv_rt1_periodic_quad_face_constraints_with_offsets(
    face_pairs: &[HdivRt1PeriodicQuadFacePair],
    master_face_dof_offsets: &[usize],
    slave_face_dof_offsets: &[usize],
    phase: Complex64,
) -> Result<Vec<MultiPointConstraint>, MpcError> {
    if !phase.re.is_finite() || !phase.im.is_finite() {
        return Err(MpcError::NonFiniteCoefficient {
            master: usize::MAX,
            slave: usize::MAX,
        });
    }
    let mut constraints = Vec::with_capacity(face_pairs.len().saturating_mul(4));
    for pair in face_pairs {
        let master_base =
            *master_face_dof_offsets
                .get(pair.master_face)
                .ok_or(MpcError::DofOutOfRange {
                    dof: pair.master_face,
                    num_dofs: master_face_dof_offsets.len(),
                })?;
        let slave_base =
            *slave_face_dof_offsets
                .get(pair.slave_face)
                .ok_or(MpcError::DofOutOfRange {
                    dof: pair.slave_face,
                    num_dofs: slave_face_dof_offsets.len(),
                })?;
        let (matrix, orientation) = rt1_quad_face_mode_change_matrix(pair.vertex_permutation)
            .ok_or(MpcError::InvalidFaceVertexMap {
                master_face: pair.master_face,
                slave_face: pair.slave_face,
            })?;
        for (row, matrix_row) in matrix.iter().enumerate() {
            let mut masters = Vec::with_capacity(4);
            for (column, &entry) in matrix_row.iter().enumerate() {
                let coefficient = phase * (orientation * entry);
                if coefficient.norm() > 0.0 {
                    masters.push(MpcTerm {
                        dof: master_base
                            .checked_add(column)
                            .ok_or(MpcError::DofIndexOverflow(master_base))?,
                        coefficient,
                    });
                }
            }
            constraints.push(MultiPointConstraint {
                slave: slave_base
                    .checked_add(row)
                    .ok_or(MpcError::DofIndexOverflow(slave_base))?,
                masters,
            });
        }
    }
    Ok(constraints)
}
impl SectorPeriodicMap {
    /// Build quadrilateral RT1 periodic constraints with mixed-face offsets.
    pub fn hdiv_rt1_quadrilateral_face_constraints_from_faces_with_offsets(
        &self,
        master_faces: &[(usize, Face)],
        slave_faces: &[(usize, Face)],
        master_face_dof_offsets: &[usize],
        slave_face_dof_offsets: &[usize],
    ) -> Result<Vec<MultiPointConstraint>, MpcError> {
        let pairs = pair_periodic_hdiv_rt1_quadrilateral_faces(
            master_faces,
            slave_faces,
            &self.node_pairs,
        )?;
        expand_hdiv_rt1_periodic_quad_face_constraints_with_offsets(
            &pairs,
            master_face_dof_offsets,
            slave_face_dof_offsets,
            self.phase,
        )
    }
}
/// RT1 periodic face pair for a mixed triangular/quadrilateral 3-D surface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HdivRt1PeriodicMixedFacePair {
    Triangle(HdivRt1PeriodicFacePair),
    Quadrilateral(HdivRt1PeriodicQuadFacePair),
}

/// Pair marked RT1 faces while preserving each face's topology-specific mode
/// transform. Triangular faces carry three modes and quadrilateral faces four.
pub fn pair_periodic_hdiv_rt1_mixed_faces(
    master_faces: &[(usize, Face)],
    slave_faces: &[(usize, Face)],
    node_pairs: &[(usize, usize)],
) -> Result<Vec<HdivRt1PeriodicMixedFacePair>, MpcError> {
    if master_faces.len() != slave_faces.len() {
        return Err(MpcError::PairCountMismatch {
            masters: master_faces.len(),
            slaves: slave_faces.len(),
        });
    }

    let triangle_masters = master_faces
        .iter()
        .filter_map(|(face, topology)| {
            matches!(topology, Face::Triangle(_)).then_some((*face, topology.clone()))
        })
        .collect::<Vec<_>>();
    let triangle_slaves = slave_faces
        .iter()
        .filter_map(|(face, topology)| {
            matches!(topology, Face::Triangle(_)).then_some((*face, topology.clone()))
        })
        .collect::<Vec<_>>();
    let quadrilateral_masters = master_faces
        .iter()
        .filter_map(|(face, topology)| {
            matches!(topology, Face::Quad(_)).then_some((*face, topology.clone()))
        })
        .collect::<Vec<_>>();
    let quadrilateral_slaves = slave_faces
        .iter()
        .filter_map(|(face, topology)| {
            matches!(topology, Face::Quad(_)).then_some((*face, topology.clone()))
        })
        .collect::<Vec<_>>();

    let triangles = pair_periodic_hdiv_rt1_faces(&triangle_masters, &triangle_slaves, node_pairs)?;
    let quadrilaterals = pair_periodic_hdiv_rt1_quadrilateral_faces(
        &quadrilateral_masters,
        &quadrilateral_slaves,
        node_pairs,
    )?;
    let mut pairs = Vec::with_capacity(master_faces.len());
    pairs.extend(
        triangles
            .into_iter()
            .map(HdivRt1PeriodicMixedFacePair::Triangle),
    );
    pairs.extend(
        quadrilaterals
            .into_iter()
            .map(HdivRt1PeriodicMixedFacePair::Quadrilateral),
    );
    Ok(pairs)
}

fn expand_hdiv_rt1_periodic_triangle_pair_with_offsets(
    pair: HdivRt1PeriodicFacePair,
    master_base: usize,
    slave_base: usize,
    phase: Complex64,
) -> Result<Vec<MultiPointConstraint>, MpcError> {
    let matrix = rt1_face_mode_change_matrix(pair.vertex_permutation).ok_or(
        MpcError::InvalidFaceVertexMap {
            master_face: pair.master_face,
            slave_face: pair.slave_face,
        },
    )?;
    let orientation = permutation_parity(pair.vertex_permutation);
    let mut constraints = Vec::with_capacity(3);
    for (row, matrix_row) in matrix.iter().enumerate() {
        let mut masters = Vec::with_capacity(3);
        for (column, &entry) in matrix_row.iter().enumerate() {
            let coefficient = phase * (orientation * entry);
            if coefficient.norm() > 0.0 {
                masters.push(MpcTerm {
                    dof: master_base
                        .checked_add(column)
                        .ok_or(MpcError::DofIndexOverflow(master_base))?,
                    coefficient,
                });
            }
        }
        constraints.push(MultiPointConstraint {
            slave: slave_base
                .checked_add(row)
                .ok_or(MpcError::DofIndexOverflow(slave_base))?,
            masters,
        });
    }
    Ok(constraints)
}

/// Expand mixed RT1 triangular/quadrilateral periodic face constraints using
/// topology-derived global offsets.
pub fn expand_hdiv_rt1_periodic_mixed_face_constraints(
    face_pairs: &[HdivRt1PeriodicMixedFacePair],
    master_face_dof_offsets: &[usize],
    slave_face_dof_offsets: &[usize],
    phase: Complex64,
) -> Result<Vec<MultiPointConstraint>, MpcError> {
    if !phase.re.is_finite() || !phase.im.is_finite() {
        return Err(MpcError::NonFiniteCoefficient {
            master: usize::MAX,
            slave: usize::MAX,
        });
    }
    let mut constraints = Vec::new();
    for pair in face_pairs {
        match *pair {
            HdivRt1PeriodicMixedFacePair::Triangle(pair) => {
                let master_base = *master_face_dof_offsets.get(pair.master_face).ok_or(
                    MpcError::DofOutOfRange {
                        dof: pair.master_face,
                        num_dofs: master_face_dof_offsets.len(),
                    },
                )?;
                let slave_base = *slave_face_dof_offsets.get(pair.slave_face).ok_or(
                    MpcError::DofOutOfRange {
                        dof: pair.slave_face,
                        num_dofs: slave_face_dof_offsets.len(),
                    },
                )?;
                constraints.extend(expand_hdiv_rt1_periodic_triangle_pair_with_offsets(
                    pair,
                    master_base,
                    slave_base,
                    phase,
                )?);
            }
            HdivRt1PeriodicMixedFacePair::Quadrilateral(pair) => {
                let expanded = expand_hdiv_rt1_periodic_quad_face_constraints_with_offsets(
                    &[pair],
                    master_face_dof_offsets,
                    slave_face_dof_offsets,
                    phase,
                )?;
                constraints.extend(expanded);
            }
        }
    }
    Ok(constraints)
}

impl SectorPeriodicMap {
    /// Build mixed RT1 triangular/quadrilateral periodic face constraints.
    pub fn hdiv_rt1_mixed_face_constraints_from_faces(
        &self,
        master_faces: &[(usize, Face)],
        slave_faces: &[(usize, Face)],
        master_face_dof_offsets: &[usize],
        slave_face_dof_offsets: &[usize],
    ) -> Result<Vec<MultiPointConstraint>, MpcError> {
        let pairs =
            pair_periodic_hdiv_rt1_mixed_faces(master_faces, slave_faces, &self.node_pairs)?;
        expand_hdiv_rt1_periodic_mixed_face_constraints(
            &pairs,
            master_face_dof_offsets,
            slave_face_dof_offsets,
            self.phase,
        )
    }

    /// Build a checked MPC system for mixed RT1 periodic faces.
    pub fn hdiv_rt1_mixed_face_mpc_system(
        &self,
        num_full_dofs: usize,
        master_faces: &[(usize, Face)],
        slave_faces: &[(usize, Face)],
        master_face_dof_offsets: &[usize],
        slave_face_dof_offsets: &[usize],
    ) -> Result<MpcSystem, MpcError> {
        let constraints = self.hdiv_rt1_mixed_face_constraints_from_faces(
            master_faces,
            slave_faces,
            master_face_dof_offsets,
            slave_face_dof_offsets,
        )?;
        MpcSystem::new(num_full_dofs, &constraints)
    }

    /// Condense a complex FEM system with mixed RT1 periodic faces.
    pub fn condense_hdiv_rt1_mixed_fem_system_complex(
        &self,
        system: &FemSystem<Complex64>,
        master_faces: &[(usize, Face)],
        slave_faces: &[(usize, Face)],
        master_face_dof_offsets: &[usize],
        slave_face_dof_offsets: &[usize],
    ) -> Result<FemSystem<Complex64>, MpcError> {
        let mpc = self.hdiv_rt1_mixed_face_mpc_system(
            system.matrix.num_rows,
            master_faces,
            slave_faces,
            master_face_dof_offsets,
            slave_face_dof_offsets,
        )?;
        mpc.condense_fem_system_complex(system)
    }
}
