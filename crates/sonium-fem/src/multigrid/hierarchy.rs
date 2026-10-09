//! Mesh hierarchy for geometric multigrid
//!
//! Manages a sequence of meshes from fine to coarse for multigrid methods.

use crate::assembly::{
    HelmholtzMatrix, MassMatrix, StiffnessMatrix, TripletMatrix, assemble_mass, assemble_stiffness,
};
use crate::basis::PolynomialDegree;
use crate::mesh::Mesh;
use num_complex::Complex64;
use std::collections::HashMap;
use std::fmt;

/// A level in the multigrid hierarchy
#[derive(Debug)]
pub struct MultigridLevel {
    /// Mesh at this level
    pub mesh: Mesh,
    /// Number of degrees of freedom
    pub n_dofs: usize,
    /// Stiffness matrix (optional, assembled on demand)
    pub stiffness: Option<StiffnessMatrix>,
    /// Mass matrix (optional)
    pub mass: Option<MassMatrix>,
    /// System matrix in COO format (legacy Helmholtz or generic TripletMatrix)
    pub system: Option<HelmholtzMatrix>,
    /// Generic system matrix (for physics-agnostic multigrid)
    pub generic_system: Option<TripletMatrix<Complex64>>,
    /// Prolongation operator to next finer level (None for finest)
    pub prolongation: Option<TransferMatrix>,
    /// Restriction operator to next coarser level (None for coarsest)
    pub restriction: Option<TransferMatrix>,
}

/// Transfer matrix (prolongation or restriction)
#[derive(Debug, Clone)]
pub struct TransferMatrix {
    /// Row indices
    pub rows: Vec<usize>,
    /// Column indices
    pub cols: Vec<usize>,
    /// Values
    pub values: Vec<f64>,
    /// Number of rows (fine DOFs for prolongation, coarse for restriction)
    pub n_rows: usize,
    /// Number of columns
    pub n_cols: usize,
}

/// Failure while validating or applying a multigrid transfer operator.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferError {
    /// The COO storage arrays do not describe the same number of entries.
    StorageLengthMismatch {
        rows: usize,
        cols: usize,
        values: usize,
    },
    /// A COO entry addresses a row or column outside the declared shape.
    EntryOutOfBounds {
        entry: usize,
        row: usize,
        col: usize,
        n_rows: usize,
        n_cols: usize,
    },
    /// A transfer weight is not finite.
    NonFiniteValue { entry: usize },
    /// A vector does not match the operator dimension.
    VectorLengthMismatch {
        operation: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A structured transfer input is not valid.
    InvalidInput {
        operation: &'static str,
        reason: &'static str,
    },
}

impl fmt::Display for TransferError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::StorageLengthMismatch { rows, cols, values } => write!(
                f,
                "transfer COO lengths differ: rows={rows}, cols={cols}, values={values}"
            ),
            Self::EntryOutOfBounds {
                entry,
                row,
                col,
                n_rows,
                n_cols,
            } => write!(
                f,
                "transfer entry {entry} ({row}, {col}) exceeds shape {n_rows}x{n_cols}"
            ),
            Self::NonFiniteValue { entry } => {
                write!(f, "transfer entry {entry} has a non-finite value")
            }
            Self::VectorLengthMismatch {
                operation,
                expected,
                actual,
            } => write!(
                f,
                "{operation} input has length {actual}, expected {expected}"
            ),
            Self::InvalidInput { operation, reason } => {
                write!(f, "invalid {operation} input: {reason}")
            }
        }
    }
}

impl std::error::Error for TransferError {}

/// Failure while accessing or validating a multigrid hierarchy.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HierarchyError {
    /// No levels are available for a multigrid operation.
    EmptyHierarchy,
    /// A level index is outside the hierarchy.
    LevelOutOfRange { level: usize, num_levels: usize },
    /// The level has no assembled system operator.
    SystemNotAssembled { level: usize },
    /// A system operator does not match the level's declared DOF count.
    SystemDimensionMismatch {
        level: usize,
        expected: usize,
        actual: usize,
    },
    /// A system operator's COO storage is malformed.
    InvalidSystem { level: usize, reason: &'static str },
    /// A transfer operator is malformed or has the wrong shape.
    InvalidTransfer {
        level: usize,
        kind: &'static str,
        error: TransferError,
    },
    /// A transfer operator has the right storage but the wrong dimensions.
    TransferDimensionMismatch {
        level: usize,
        kind: &'static str,
        expected_rows: usize,
        expected_cols: usize,
        actual_rows: usize,
        actual_cols: usize,
    },
    /// The hierarchy wavenumber is not finite.
    NonFiniteWavenumber,
}

impl fmt::Display for HierarchyError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyHierarchy => write!(f, "multigrid hierarchy has no levels"),
            Self::LevelOutOfRange { level, num_levels } => {
                write!(f, "multigrid level {level} is outside {num_levels} levels")
            }
            Self::SystemNotAssembled { level } => {
                write!(f, "multigrid system at level {level} is not assembled")
            }
            Self::SystemDimensionMismatch {
                level,
                expected,
                actual,
            } => write!(
                f,
                "multigrid system at level {level} has dimension {actual}, expected {expected}"
            ),
            Self::InvalidSystem { level, reason } => {
                write!(f, "invalid multigrid system at level {level}: {reason}")
            }
            Self::InvalidTransfer { level, kind, error } => {
                write!(f, "invalid {kind} transfer at level {level}: {error}")
            }
            Self::TransferDimensionMismatch {
                level,
                kind,
                expected_rows,
                expected_cols,
                actual_rows,
                actual_cols,
            } => write!(
                f,
                "{kind} transfer at level {level} has shape {actual_rows}x{actual_cols}, expected {expected_rows}x{expected_cols}"
            ),
            Self::NonFiniteWavenumber => write!(f, "multigrid wavenumber must be finite"),
        }
    }
}

impl std::error::Error for HierarchyError {}

impl TransferMatrix {
    pub fn new(n_rows: usize, n_cols: usize) -> Self {
        Self {
            rows: Vec::new(),
            cols: Vec::new(),
            values: Vec::new(),
            n_rows,
            n_cols,
        }
    }

    /// Add entry (i, j, v)
    pub fn add(&mut self, i: usize, j: usize, v: f64) {
        self.rows.push(i);
        self.cols.push(j);
        self.values.push(v);
    }

    /// Add an entry after checking its coordinates and value.
    pub fn try_add(&mut self, i: usize, j: usize, v: f64) -> Result<(), TransferError> {
        self.validate()?;
        if i >= self.n_rows || j >= self.n_cols {
            return Err(TransferError::EntryOutOfBounds {
                entry: self.rows.len(),
                row: i,
                col: j,
                n_rows: self.n_rows,
                n_cols: self.n_cols,
            });
        }
        if !v.is_finite() {
            return Err(TransferError::NonFiniteValue {
                entry: self.rows.len(),
            });
        }
        self.add(i, j, v);
        Ok(())
    }

    /// Validate the public COO representation before applying it.
    pub fn validate(&self) -> Result<(), TransferError> {
        if self.rows.len() != self.cols.len() || self.rows.len() != self.values.len() {
            return Err(TransferError::StorageLengthMismatch {
                rows: self.rows.len(),
                cols: self.cols.len(),
                values: self.values.len(),
            });
        }
        for (entry, ((&row, &col), &value)) in self
            .rows
            .iter()
            .zip(&self.cols)
            .zip(&self.values)
            .enumerate()
        {
            if row >= self.n_rows || col >= self.n_cols {
                return Err(TransferError::EntryOutOfBounds {
                    entry,
                    row,
                    col,
                    n_rows: self.n_rows,
                    n_cols: self.n_cols,
                });
            }
            if !value.is_finite() {
                return Err(TransferError::NonFiniteValue { entry });
            }
        }
        Ok(())
    }

    /// Apply y = A * x (matrix-vector product)
    pub fn apply(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.n_rows];
        for k in 0..self.rows.len() {
            y[self.rows[k]] += Complex64::new(self.values[k], 0.0) * x[self.cols[k]];
        }
        y
    }

    /// Checked matrix-vector product that rejects malformed storage and dimensions.
    pub fn apply_checked(&self, x: &[Complex64]) -> Result<Vec<Complex64>, TransferError> {
        self.validate()?;
        if x.len() != self.n_cols {
            return Err(TransferError::VectorLengthMismatch {
                operation: "transfer application",
                expected: self.n_cols,
                actual: x.len(),
            });
        }
        Ok(self.apply(x))
    }

    /// Apply y = A^T * x (transpose)
    pub fn apply_transpose(&self, x: &[Complex64]) -> Vec<Complex64> {
        let mut y = vec![Complex64::new(0.0, 0.0); self.n_cols];
        for k in 0..self.rows.len() {
            y[self.cols[k]] += Complex64::new(self.values[k], 0.0) * x[self.rows[k]];
        }
        y
    }

    /// Checked transpose product that rejects malformed storage and dimensions.
    pub fn apply_transpose_checked(
        &self,
        x: &[Complex64],
    ) -> Result<Vec<Complex64>, TransferError> {
        self.validate()?;
        if x.len() != self.n_rows {
            return Err(TransferError::VectorLengthMismatch {
                operation: "transpose transfer application",
                expected: self.n_rows,
                actual: x.len(),
            });
        }
        Ok(self.apply_transpose(x))
    }
}

/// Multigrid mesh hierarchy
pub struct MultigridHierarchy {
    /// Levels from finest (0) to coarsest (n-1)
    pub levels: Vec<MultigridLevel>,
    /// Polynomial degree for FEM
    pub degree: PolynomialDegree,
    /// Wavenumber for Helmholtz
    pub wavenumber: Complex64,
}

impl MultigridHierarchy {
    /// Create hierarchy from a sequence of meshes (finest first)
    pub fn from_meshes(meshes: Vec<Mesh>, degree: PolynomialDegree, wavenumber: Complex64) -> Self {
        let n_levels = meshes.len();
        let mut levels = Vec::with_capacity(n_levels);

        for mesh in meshes {
            let n_dofs = mesh.num_nodes();
            levels.push(MultigridLevel {
                mesh,
                n_dofs,
                stiffness: None,
                mass: None,
                system: None,
                generic_system: None,
                prolongation: None,
                restriction: None,
            });
        }

        // Build transfer operators between levels
        for i in 0..n_levels.saturating_sub(1) {
            let (p, r) = build_transfer_operators(&levels[i].mesh, &levels[i + 1].mesh);
            levels[i].restriction = Some(r);
            levels[i + 1].prolongation = Some(p);
        }

        Self {
            levels,
            degree,
            wavenumber,
        }
    }

    /// Create hierarchy by successive coarsening
    pub fn from_fine_mesh(
        fine_mesh: Mesh,
        n_levels: usize,
        degree: PolynomialDegree,
        wavenumber: Complex64,
    ) -> Self {
        let mut meshes = vec![fine_mesh];

        // Generate coarser meshes by selecting every other node
        // This is a simplified coarsening - real implementations use graph coarsening
        for _ in 1..n_levels {
            let coarse = coarsen_mesh(meshes.last().unwrap());
            if coarse.num_nodes() < 4 {
                break; // Too coarse
            }
            meshes.push(coarse);
        }

        Self::from_meshes(meshes, degree, wavenumber)
    }

    /// Number of levels
    pub fn num_levels(&self) -> usize {
        self.levels.len()
    }

    /// Get finest level
    pub fn finest(&self) -> &MultigridLevel {
        &self.levels[0]
    }

    /// Get coarsest level
    pub fn coarsest(&self) -> &MultigridLevel {
        &self.levels[self.levels.len() - 1]
    }

    /// Return the finest level, reporting an error for an empty hierarchy.
    pub fn try_finest(&self) -> Result<&MultigridLevel, HierarchyError> {
        self.levels.first().ok_or(HierarchyError::EmptyHierarchy)
    }

    /// Return the coarsest level, reporting an error for an empty hierarchy.
    pub fn try_coarsest(&self) -> Result<&MultigridLevel, HierarchyError> {
        self.levels.last().ok_or(HierarchyError::EmptyHierarchy)
    }

    /// Return the assembled system at a level without indexing or unwrapping.
    pub fn try_system_at(
        &self,
        level_idx: usize,
    ) -> Result<&dyn super::CooOperator, HierarchyError> {
        let level = self
            .levels
            .get(level_idx)
            .ok_or(HierarchyError::LevelOutOfRange {
                level: level_idx,
                num_levels: self.levels.len(),
            })?;
        if let Some(ref sys) = level.generic_system {
            return Ok(sys);
        }
        level
            .system
            .as_ref()
            .map(|system| system as &dyn super::CooOperator)
            .ok_or(HierarchyError::SystemNotAssembled { level: level_idx })
    }

    /// Validate every matrix and transfer used by a multigrid cycle.
    pub fn validate(&self) -> Result<(), HierarchyError> {
        if self.levels.is_empty() {
            return Err(HierarchyError::EmptyHierarchy);
        }
        if !self.wavenumber.re.is_finite() || !self.wavenumber.im.is_finite() {
            return Err(HierarchyError::NonFiniteWavenumber);
        }

        for (level_idx, level) in self.levels.iter().enumerate() {
            let system = self.try_system_at(level_idx)?;
            if system.dim() != level.n_dofs {
                return Err(HierarchyError::SystemDimensionMismatch {
                    level: level_idx,
                    expected: level.n_dofs,
                    actual: system.dim(),
                });
            }
            if system.rows().len() != system.cols().len()
                || system.rows().len() != system.values().len()
            {
                return Err(HierarchyError::InvalidSystem {
                    level: level_idx,
                    reason: "COO storage lengths differ",
                });
            }
            for ((&row, &col), value) in
                system.rows().iter().zip(system.cols()).zip(system.values())
            {
                if row >= system.dim() || col >= system.dim() {
                    return Err(HierarchyError::InvalidSystem {
                        level: level_idx,
                        reason: "COO entry is out of bounds",
                    });
                }
                if !value.re.is_finite() || !value.im.is_finite() {
                    return Err(HierarchyError::InvalidSystem {
                        level: level_idx,
                        reason: "COO value is non-finite",
                    });
                }
            }

            let Some(next) = self.levels.get(level_idx + 1) else {
                continue;
            };
            let restriction =
                level
                    .restriction
                    .as_ref()
                    .ok_or(HierarchyError::InvalidTransfer {
                        level: level_idx,
                        kind: "restriction",
                        error: TransferError::InvalidInput {
                            operation: "restriction",
                            reason: "operator is missing",
                        },
                    })?;
            validate_transfer_shape(
                restriction,
                level_idx,
                "restriction",
                next.n_dofs,
                level.n_dofs,
            )?;

            let prolongation =
                next.prolongation
                    .as_ref()
                    .ok_or_else(|| HierarchyError::InvalidTransfer {
                        level: level_idx + 1,
                        kind: "prolongation",
                        error: TransferError::InvalidInput {
                            operation: "prolongation",
                            reason: "operator is missing",
                        },
                    })?;
            validate_transfer_shape(
                prolongation,
                level_idx + 1,
                "prolongation",
                level.n_dofs,
                next.n_dofs,
            )?;
        }
        Ok(())
    }

    /// Assemble matrices at all levels (Helmholtz-specific).
    #[deprecated(note = "Use assemble_all_with for physics-agnostic multigrid")]
    pub fn assemble_all(&mut self) {
        for level in &mut self.levels {
            let stiffness = assemble_stiffness(&level.mesh, self.degree);
            let mass = assemble_mass(&level.mesh, self.degree);
            let system = HelmholtzMatrix::new(&stiffness, &mass, self.wavenumber);

            level.stiffness = Some(stiffness);
            level.mass = Some(mass);
            level.system = Some(system);
        }
    }

    /// Assemble matrices at all levels using a generic assembler closure.
    ///
    /// The closure takes a mesh and polynomial degree and returns a
    /// `TripletMatrix<Complex64>` representing the system matrix at that level.
    /// This is physics-agnostic: the caller defines how to assemble.
    ///
    /// # Example
    /// ```ignore
    /// hierarchy.assemble_all_with(|mesh, degree| {
    ///     let asm = build_acoustics_assembler(mesh, degree);
    ///     let csr = asm.assemble_helmholtz(k, &boundary_coeffs);
    ///     TripletMatrix::from_csr(&csr) // convert back to triplet
    /// });
    /// ```
    pub fn assemble_all_with<F>(&mut self, assembler: F)
    where
        F: Fn(&Mesh, PolynomialDegree) -> TripletMatrix<Complex64>,
    {
        for level in &mut self.levels {
            level.generic_system = Some(assembler(&level.mesh, self.degree));
        }
    }

    /// Assemble matrix at specific level (Helmholtz-specific).
    #[deprecated(note = "Use assemble_all_with for physics-agnostic multigrid")]
    pub fn assemble_level(&mut self, level_idx: usize) {
        let level = &mut self.levels[level_idx];
        let stiffness = assemble_stiffness(&level.mesh, self.degree);
        let mass = assemble_mass(&level.mesh, self.degree);
        let system = HelmholtzMatrix::new(&stiffness, &mass, self.wavenumber);

        level.stiffness = Some(stiffness);
        level.mass = Some(mass);
        level.system = Some(system);
    }

    /// Get the system operator at a given level (generic, supports both paths).
    pub fn system_at(&self, level_idx: usize) -> &dyn super::CooOperator {
        let level = &self.levels[level_idx];
        if let Some(ref sys) = level.generic_system {
            return sys;
        }
        level
            .system
            .as_ref()
            .expect("System matrix not assembled at this level")
    }
}

fn validate_transfer_shape(
    transfer: &TransferMatrix,
    level: usize,
    kind: &'static str,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<(), HierarchyError> {
    transfer
        .validate()
        .map_err(|error| HierarchyError::InvalidTransfer { level, kind, error })?;
    if transfer.n_rows != expected_rows || transfer.n_cols != expected_cols {
        return Err(HierarchyError::TransferDimensionMismatch {
            level,
            kind,
            expected_rows,
            expected_cols,
            actual_rows: transfer.n_rows,
            actual_cols: transfer.n_cols,
        });
    }
    Ok(())
}

/// Build prolongation and restriction operators between two mesh levels
fn build_transfer_operators(
    fine_mesh: &Mesh,
    coarse_mesh: &Mesh,
) -> (TransferMatrix, TransferMatrix) {
    let n_fine = fine_mesh.num_nodes();
    let n_coarse = coarse_mesh.num_nodes();

    let mut prolongation = TransferMatrix::new(n_fine, n_coarse);
    let mut restriction = TransferMatrix::new(n_coarse, n_fine);

    // Build node mapping from coarse to fine
    // For geometric multigrid, coarse nodes are a subset of fine nodes
    let mut coarse_to_fine: HashMap<usize, usize> = HashMap::new();

    // Find matching nodes by position
    let tol = 1e-10;
    for (ci, cp) in coarse_mesh.nodes.iter().enumerate() {
        for (fi, fp) in fine_mesh.nodes.iter().enumerate() {
            let dist =
                ((cp.x - fp.x).powi(2) + (cp.y - fp.y).powi(2) + (cp.z - fp.z).powi(2)).sqrt();
            if dist < tol {
                coarse_to_fine.insert(ci, fi);
                break;
            }
        }
    }

    // Prolongation: inject coarse values to fine, interpolate others
    // Simplified: only inject for matching nodes
    for (ci, &fi) in &coarse_to_fine {
        prolongation.add(fi, *ci, 1.0);
    }

    // For non-matching fine nodes, find nearest coarse nodes and interpolate
    // This is a simplified linear interpolation
    for fi in 0..n_fine {
        let is_coarse = coarse_to_fine.values().any(|&v| v == fi);
        if !is_coarse {
            // Find nearest coarse nodes
            let fp = &fine_mesh.nodes[fi];
            let mut distances: Vec<(usize, f64)> = coarse_mesh
                .nodes
                .iter()
                .enumerate()
                .map(|(ci, cp)| {
                    let d = ((cp.x - fp.x).powi(2) + (cp.y - fp.y).powi(2) + (cp.z - fp.z).powi(2))
                        .sqrt();
                    (ci, d)
                })
                .collect();
            distances.sort_by(|a, b| a.1.total_cmp(&b.1));

            // Use inverse distance weighting for nearest neighbors
            let n_neighbors = distances.len().min(3);
            let total_inv_dist: f64 = distances[..n_neighbors]
                .iter()
                .map(|(_, d)| if *d > 1e-15 { 1.0 / d } else { 1e15 })
                .sum();

            for (ci, d) in distances.iter().take(n_neighbors) {
                let weight = if *d > 1e-15 { 1.0 / d } else { 1e15 };
                prolongation.add(fi, *ci, weight / total_inv_dist);
            }
        }
    }

    // Restriction: transpose of prolongation
    // Uses injection for coarse nodes and inverse-distance weights for non-coarse fine nodes.
    // No additional scaling factor is applied; weights already sum to 1 per row.
    for k in 0..prolongation.rows.len() {
        restriction.add(
            prolongation.cols[k],
            prolongation.rows[k],
            prolongation.values[k],
        );
    }

    (prolongation, restriction)
}

/// Create a coarser mesh by skipping nodes
/// This is a very simplified coarsening for demonstration
fn coarsen_mesh(fine_mesh: &Mesh) -> Mesh {
    use crate::mesh::{unit_cube_tetrahedra, unit_square_triangles};

    // For structured meshes, create a coarser version
    // This is a placeholder - real implementations use graph coarsening
    let n_fine = fine_mesh.num_nodes();
    let n_coarse = ((n_fine as f64).sqrt() / 2.0).ceil() as usize;

    if fine_mesh.dimension == 2 {
        unit_square_triangles(n_coarse.max(1))
    } else {
        unit_cube_tetrahedra(n_coarse.max(1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;

    #[test]
    fn test_hierarchy_creation() {
        let mesh = unit_square_triangles(4);
        let k = Complex64::new(1.0, 0.0);

        let hierarchy = MultigridHierarchy::from_fine_mesh(mesh, 3, PolynomialDegree::P1, k);

        assert!(hierarchy.num_levels() >= 2);
        assert!(hierarchy.finest().n_dofs > hierarchy.coarsest().n_dofs);
    }

    #[test]
    fn test_transfer_matrix_apply() {
        let mut p = TransferMatrix::new(4, 2);
        p.add(0, 0, 1.0);
        p.add(1, 0, 0.5);
        p.add(1, 1, 0.5);
        p.add(2, 1, 1.0);
        p.add(3, 0, 0.5);
        p.add(3, 1, 0.5);

        let x = vec![Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        let y = p.apply(&x);

        assert_eq!(y.len(), 4);
        assert!((y[0].re - 1.0).abs() < 1e-10); // 1*1
        assert!((y[1].re - 1.5).abs() < 1e-10); // 0.5*1 + 0.5*2
        assert!((y[2].re - 2.0).abs() < 1e-10); // 1*2
    }

    #[test]
    fn checked_transfer_rejects_bad_storage_and_dimensions() {
        let mut bad_entry = TransferMatrix::new(2, 1);
        assert!(matches!(
            bad_entry.try_add(2, 0, 1.0),
            Err(TransferError::EntryOutOfBounds { entry: 0, .. })
        ));
        bad_entry.add(2, 0, 1.0);
        assert!(matches!(
            bad_entry.validate(),
            Err(TransferError::EntryOutOfBounds { entry: 0, .. })
        ));

        let mut bad_value = TransferMatrix::new(1, 1);
        bad_value.add(0, 0, f64::NAN);
        assert_eq!(
            bad_value.validate(),
            Err(TransferError::NonFiniteValue { entry: 0 })
        );

        let valid = TransferMatrix::new(2, 1);
        let input = [Complex64::new(1.0, 0.0), Complex64::new(2.0, 0.0)];
        assert_eq!(
            valid.apply_checked(&input),
            Err(TransferError::VectorLengthMismatch {
                operation: "transfer application",
                expected: 1,
                actual: 2,
            })
        );
    }

    #[test]
    fn checked_hierarchy_access_reports_empty_and_unassembled_levels() {
        let empty = MultigridHierarchy::from_meshes(
            Vec::new(),
            PolynomialDegree::P1,
            Complex64::new(0.0, 0.0),
        );
        assert!(matches!(
            empty.try_finest(),
            Err(HierarchyError::EmptyHierarchy)
        ));
        assert!(matches!(
            empty.try_system_at(0),
            Err(HierarchyError::LevelOutOfRange {
                level: 0,
                num_levels: 0
            })
        ));

        let hierarchy = MultigridHierarchy::from_meshes(
            vec![unit_square_triangles(2)],
            PolynomialDegree::P1,
            Complex64::new(0.0, 0.0),
        );
        assert!(matches!(
            hierarchy.try_system_at(0),
            Err(HierarchyError::SystemNotAssembled { level: 0 })
        ));
        assert_eq!(
            hierarchy.validate(),
            Err(HierarchyError::SystemNotAssembled { level: 0 })
        );
    }
}
