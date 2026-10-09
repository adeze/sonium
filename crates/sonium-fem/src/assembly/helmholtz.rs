//! Helmholtz equation assembly (legacy)
//!
//! **Deprecated**: Use the generic assembly infrastructure instead:
//! - Forms: [`physics::acoustics::pressure::HelmholtzForm`](crate::physics::acoustics::pressure::HelmholtzForm)
//! - Assembly: [`assembly::assemble_bilinear`](crate::assembly::assemble_bilinear) + [`PrebakedAssembler`](crate::assembly::PrebakedAssembler)
//! - Boundary: [`assembly::assemble_boundary_mass_generic`](crate::assembly::assemble_boundary_mass_generic)
//!
//! The types and functions in this module are retained for backward compatibility.

use super::mass::{MassMatrix, assemble_mass};
use super::stiffness::{StiffnessMatrix, assemble_stiffness};
use crate::basis::PolynomialDegree;
use crate::conventions::{ACOUSTIC_CONVENTION, FemConvention, FemProblemError};
use crate::forms::{LinearForm, LocalVector};
use crate::mesh::Mesh;
use crate::parallelism::{ELEMENTS_PER_THREAD, collect_indexed};
use crate::spaces::{BasisTabulation, H1Space};
use num_complex::Complex64;
use sonium_solvers::CsrMatrix;
use std::collections::HashSet;

/// Helmholtz system matrix in triplet format
#[derive(Debug, Clone)]
pub struct HelmholtzMatrix {
    /// Row indices
    pub rows: Vec<usize>,
    /// Column indices
    pub cols: Vec<usize>,
    /// Values (A = K - k²M)
    pub values: Vec<Complex64>,
    /// Matrix dimension
    pub dim: usize,
    /// Wavenumber k
    pub wavenumber: Complex64,
}

/// Structural failure while combining legacy stiffness and mass triplets.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HelmholtzMatrixError {
    /// The public FEM mesh failed structural validation before assembly.
    InvalidMesh { reason: String },
    /// The assembled right-hand side contains a non-finite contribution.
    NonFiniteRhs { dof: usize },
    /// The component matrices do not describe the same square system.
    DimensionMismatch {
        stiffness_dim: usize,
        mass_dim: usize,
    },
    /// A component has inconsistent COO storage lengths.
    TripletLengthMismatch {
        matrix: &'static str,
        rows: usize,
        cols: usize,
        values: usize,
    },
    /// A component triplet falls outside its declared dimension.
    TripletIndexOutOfBounds {
        matrix: &'static str,
        entry: usize,
        row: usize,
        col: usize,
        dim: usize,
    },
    /// A component value or wavenumber is non-finite.
    NonFiniteValue { matrix: &'static str, entry: usize },
    /// The supplied complex wavenumber contains a non-finite component.
    NonFiniteWavenumber,
}

impl std::fmt::Display for HelmholtzMatrixError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidMesh { reason } => write!(f, "invalid Helmholtz mesh: {reason}"),
            Self::NonFiniteRhs { dof } => write!(f, "Helmholtz RHS DOF {dof} is non-finite"),
            Self::DimensionMismatch {
                stiffness_dim,
                mass_dim,
            } => write!(
                f,
                "stiffness dimension {stiffness_dim} differs from mass dimension {mass_dim}"
            ),
            Self::TripletLengthMismatch {
                matrix,
                rows,
                cols,
                values,
            } => write!(
                f,
                "{matrix} COO lengths differ: rows={rows}, cols={cols}, values={values}"
            ),
            Self::TripletIndexOutOfBounds {
                matrix,
                entry,
                row,
                col,
                dim,
            } => write!(
                f,
                "{matrix} triplet {entry} ({row}, {col}) outside {dim}x{dim}"
            ),
            Self::NonFiniteValue { matrix, entry } => {
                write!(f, "{matrix} triplet {entry} has a non-finite value")
            }
            Self::NonFiniteWavenumber => write!(f, "wavenumber must be finite"),
        }
    }
}

impl std::error::Error for HelmholtzMatrixError {}

impl HelmholtzMatrix {
    /// Create Helmholtz matrix A = K - k²M
    pub fn new(stiffness: &StiffnessMatrix, mass: &MassMatrix, wavenumber: Complex64) -> Self {
        Self::try_new(stiffness, mass, wavenumber)
            .unwrap_or_else(|error| panic!("invalid Helmholtz components: {error}"))
    }

    /// Checked variant of [`Self::new`] that rejects malformed components
    /// before combining their triplets.
    pub fn try_new(
        stiffness: &StiffnessMatrix,
        mass: &MassMatrix,
        wavenumber: Complex64,
    ) -> Result<Self, HelmholtzMatrixError> {
        if stiffness.dim != mass.dim {
            return Err(HelmholtzMatrixError::DimensionMismatch {
                stiffness_dim: stiffness.dim,
                mass_dim: mass.dim,
            });
        }
        validate_real_component(
            "stiffness",
            stiffness.dim,
            &stiffness.rows,
            &stiffness.cols,
            &stiffness.values,
        )?;
        validate_real_component("mass", mass.dim, &mass.rows, &mass.cols, &mass.values)?;
        if !wavenumber.re.is_finite() || !wavenumber.im.is_finite() {
            return Err(HelmholtzMatrixError::NonFiniteWavenumber);
        }
        let dim = stiffness.dim;
        let k_sq = wavenumber * wavenumber;

        // Combine entries
        let mut combined_nnz = stiffness.nnz() + mass.nnz();
        let mut rows = Vec::with_capacity(combined_nnz);
        let mut cols = Vec::with_capacity(combined_nnz);
        let mut values = Vec::with_capacity(combined_nnz);

        // Add stiffness entries (K)
        for i in 0..stiffness.nnz() {
            rows.push(stiffness.rows[i]);
            cols.push(stiffness.cols[i]);
            values.push(Complex64::new(stiffness.values[i], 0.0));
        }

        // Subtract k²M entries (-k²M)
        for i in 0..mass.nnz() {
            rows.push(mass.rows[i]);
            cols.push(mass.cols[i]);
            values.push(-k_sq * Complex64::new(mass.values[i], 0.0));
        }

        combined_nnz = values.len();
        let _ = combined_nnz; // suppress unused warning

        Ok(Self {
            rows,
            cols,
            values,
            dim,
            wavenumber,
        })
    }

    /// Number of non-zeros (may include duplicates)
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Validate the public COO storage before conversion or solver use.
    pub fn validate(&self) -> Result<(), HelmholtzMatrixError> {
        if !self.wavenumber.re.is_finite() || !self.wavenumber.im.is_finite() {
            return Err(HelmholtzMatrixError::NonFiniteWavenumber);
        }
        validate_complex_component("helmholtz", self.dim, &self.rows, &self.cols, &self.values)
    }

    /// Convert to compressed (summing duplicate entries)
    pub fn to_compressed(&self) -> HelmholtzMatrix {
        self.try_to_compressed()
            .unwrap_or_else(|error| panic!("invalid Helmholtz matrix: {error}"))
    }

    /// Checked variant of [`Self::to_compressed`].
    pub fn try_to_compressed(&self) -> Result<HelmholtzMatrix, HelmholtzMatrixError> {
        self.validate()?;
        use std::collections::HashMap;

        let mut entries: HashMap<(usize, usize), Complex64> = HashMap::new();

        for k in 0..self.nnz() {
            let key = (self.rows[k], self.cols[k]);
            *entries.entry(key).or_insert(Complex64::new(0.0, 0.0)) += self.values[k];
        }

        let mut rows = Vec::with_capacity(entries.len());
        let mut cols = Vec::with_capacity(entries.len());
        let mut values = Vec::with_capacity(entries.len());

        for ((r, c), v) in entries {
            if v.norm() > 1e-15 {
                rows.push(r);
                cols.push(c);
                values.push(v);
            }
        }

        Ok(HelmholtzMatrix {
            rows,
            cols,
            values,
            dim: self.dim,
            wavenumber: self.wavenumber,
        })
    }

    /// Convert to CSR sparse matrix format
    ///
    /// The triplets are converted to CSR format using `CsrMatrix::from_triplets`,
    /// which handles duplicate entries by summing them.
    pub fn to_csr(&self) -> CsrMatrix<Complex64> {
        self.try_to_csr()
            .unwrap_or_else(|error| panic!("invalid Helmholtz matrix: {error}"))
    }

    /// Checked variant of [`Self::to_csr`].
    pub fn try_to_csr(&self) -> Result<CsrMatrix<Complex64>, HelmholtzMatrixError> {
        self.validate()?;
        let triplets: Vec<(usize, usize, Complex64)> = self
            .rows
            .iter()
            .zip(self.cols.iter())
            .zip(self.values.iter())
            .map(|((&r, &c), &v)| (r, c, v))
            .collect();

        Ok(CsrMatrix::from_triplets(self.dim, self.dim, triplets))
    }
}

fn validate_real_component(
    matrix: &'static str,
    dim: usize,
    rows: &[usize],
    cols: &[usize],
    values: &[f64],
) -> Result<(), HelmholtzMatrixError> {
    if rows.len() != cols.len() || rows.len() != values.len() {
        return Err(HelmholtzMatrixError::TripletLengthMismatch {
            matrix,
            rows: rows.len(),
            cols: cols.len(),
            values: values.len(),
        });
    }
    for (entry, ((&row, &col), &value)) in rows.iter().zip(cols).zip(values).enumerate() {
        if row >= dim || col >= dim {
            return Err(HelmholtzMatrixError::TripletIndexOutOfBounds {
                matrix,
                entry,
                row,
                col,
                dim,
            });
        }
        if !value.is_finite() {
            return Err(HelmholtzMatrixError::NonFiniteValue { matrix, entry });
        }
    }
    Ok(())
}

fn validate_complex_component(
    matrix: &'static str,
    dim: usize,
    rows: &[usize],
    cols: &[usize],
    values: &[Complex64],
) -> Result<(), HelmholtzMatrixError> {
    if rows.len() != cols.len() || rows.len() != values.len() {
        return Err(HelmholtzMatrixError::TripletLengthMismatch {
            matrix,
            rows: rows.len(),
            cols: cols.len(),
            values: values.len(),
        });
    }
    for (entry, ((&row, &col), &value)) in rows.iter().zip(cols).zip(values).enumerate() {
        if row >= dim || col >= dim {
            return Err(HelmholtzMatrixError::TripletIndexOutOfBounds {
                matrix,
                entry,
                row,
                col,
                dim,
            });
        }
        if !value.re.is_finite() || !value.im.is_finite() {
            return Err(HelmholtzMatrixError::NonFiniteValue { matrix, entry });
        }
    }
    Ok(())
}

/// Assembled Helmholtz problem including RHS
#[derive(Debug, Clone)]
pub struct HelmholtzProblem {
    /// System matrix A = K - k²M
    pub matrix: HelmholtzMatrix,
    /// Right-hand side vector
    pub rhs: Vec<Complex64>,
    /// Stiffness matrix (for reference)
    pub stiffness: StiffnessMatrix,
    /// Mass matrix (for reference)
    pub mass: MassMatrix,
    /// Polynomial degree used for the scalar H1 space and its global DOF map.
    ///
    /// The legacy boundary adapters need this metadata to constrain
    /// topology-derived edge/face DOFs for P2/P3 problems.
    pub degree: PolynomialDegree,
    /// Nodes where Dirichlet BCs have been applied (populated by apply_dirichlet())
    pub dirichlet_nodes: HashSet<usize>,
    /// Convention admitted with this problem. Legacy constructors use the
    /// canonical acoustic convention; XEM admission stores the validated
    /// exchange convention here for downstream boundary/result consumers.
    pub convention: FemConvention,
}

impl HelmholtzProblem {
    /// Assemble Helmholtz problem from mesh
    ///
    /// # Arguments
    /// * `mesh` - The finite element mesh
    /// * `degree` - Polynomial degree for basis functions
    /// * `wavenumber` - Complex wavenumber k (can be complex for damping)
    /// * `source` - Source function f(x, y, z) for RHS
    pub fn assemble<F>(
        mesh: &Mesh,
        degree: PolynomialDegree,
        wavenumber: Complex64,
        source: F,
    ) -> Self
    where
        F: Fn(f64, f64, f64) -> Complex64 + Sync,
    {
        Self::try_assemble(mesh, degree, wavenumber, source)
            .unwrap_or_else(|error| panic!("invalid Helmholtz problem: {error}"))
    }

    /// Checked variant of [`Self::assemble`] used by XEM admission and other
    /// fallible boundaries.  Legacy callers retain the infallible API above.
    pub fn try_assemble<F>(
        mesh: &Mesh,
        degree: PolynomialDegree,
        wavenumber: Complex64,
        source: F,
    ) -> Result<Self, HelmholtzMatrixError>
    where
        F: Fn(f64, f64, f64) -> Complex64 + Sync,
    {
        if mesh.nodes.is_empty() || mesh.elements.is_empty() {
            return Err(HelmholtzMatrixError::InvalidMesh {
                reason: "mesh requires at least one node and element".to_string(),
            });
        }
        mesh.validate_connectivity()
            .map_err(|errors| HelmholtzMatrixError::InvalidMesh {
                reason: errors
                    .into_iter()
                    .map(|error| error.to_string())
                    .collect::<Vec<_>>()
                    .join("; "),
            })?;
        let stiffness = assemble_stiffness(mesh, degree);
        let mass = assemble_mass(mesh, degree);
        let matrix = HelmholtzMatrix::try_new(&stiffness, &mass, wavenumber)?;

        // Assemble RHS: b_i = ∫ f * φ_i dΩ
        let rhs = assemble_rhs(mesh, degree, &source);
        if let Some(dof) = rhs.iter().enumerate().find_map(|(dof, value)| {
            (!value.re.is_finite() || !value.im.is_finite()).then_some(dof)
        }) {
            return Err(HelmholtzMatrixError::NonFiniteRhs { dof });
        }

        Ok(Self {
            matrix,
            rhs,
            stiffness,
            mass,
            degree,
            dirichlet_nodes: HashSet::new(),
            convention: ACOUSTIC_CONVENTION,
        })
    }

    /// Assemble a legacy Helmholtz system after admitting its XEM metadata.
    ///
    /// The mesh, source closure, and wavenumber are backend-owned inputs; XEM
    /// supplies the versioned acoustics domain, pressure field layout, and
    /// harmonic/orientation convention.  Field cardinality is checked against
    /// the assembled global DOF count before the result is returned.
    pub fn assemble_from_xem<F>(
        exchange: &sonium_xem::MultiphysicsProblem,
        pressure_field_id: &str,
        mesh: &Mesh,
        degree: PolynomialDegree,
        wavenumber: Complex64,
        source: F,
    ) -> Result<Self, FemProblemError>
    where
        F: Fn(f64, f64, f64) -> Complex64 + Sync,
    {
        // The Helmholtz adapter is a partitioned FEM entry point. Construct
        // the shared graph here so malformed field links and cyclic
        // partitioned dependencies cannot be admitted as an isolated solve.
        let coupling_graph = sonium_xem::CouplingGraph::from_problem(exchange)?;
        let _partitioned_order = coupling_graph.partitioned_order()?;
        let convention = FemConvention::from_problem(exchange)?;
        let field = FemConvention::pressure_field(exchange, pressure_field_id)?;
        validate_xem_mesh(mesh)?;
        let assembled = Self::try_assemble(mesh, degree, wavenumber, source)
            .map_err(|error| FemProblemError::Backend(error.to_string()))?;
        let expected = field.layout.element_count()?;
        if expected != assembled.num_dofs() {
            return Err(FemProblemError::Field {
                field_id: pressure_field_id.to_owned(),
                reason: format!(
                    "layout declares {expected} values but the assembled FEM system has {} DOFs",
                    assembled.num_dofs()
                ),
            });
        }
        Ok(Self {
            convention,
            ..assembled
        })
    }

    /// Number of degrees of freedom
    pub fn num_dofs(&self) -> usize {
        self.matrix.dim
    }
}

/// Reject mesh states that would otherwise panic inside legacy basis and
/// quadrature code. This is intentionally a structural admission check; the
/// element kernels remain responsible for their numerical Jacobian checks.
fn validate_xem_mesh(mesh: &Mesh) -> Result<(), FemProblemError> {
    if mesh.nodes.is_empty() || mesh.elements.is_empty() {
        return Err(FemProblemError::Backend(
            "Helmholtz mesh requires at least one node and element".into(),
        ));
    }
    if mesh.dimension != 2 && mesh.dimension != 3 {
        return Err(FemProblemError::Backend(format!(
            "Helmholtz mesh dimension must be 2 or 3, got {}",
            mesh.dimension
        )));
    }
    if mesh
        .nodes
        .iter()
        .any(|point| !point.x.is_finite() || !point.y.is_finite() || !point.z.is_finite())
    {
        return Err(FemProblemError::Backend(
            "Helmholtz mesh contains a non-finite node".into(),
        ));
    }
    for (element_index, element) in mesh.elements.iter().enumerate() {
        if element.element_type.dimension() != mesh.dimension {
            return Err(FemProblemError::Backend(format!(
                "element {element_index} has dimension {}, mesh is {}D",
                element.element_type.dimension(),
                mesh.dimension
            )));
        }
        let valid_node_count = element.nodes.len() == element.element_type.num_nodes_p1()
            || element.nodes.len() == element.element_type.num_nodes_p2();
        if !valid_node_count {
            return Err(FemProblemError::Backend(format!(
                "element {element_index} has unsupported node count {}",
                element.nodes.len()
            )));
        }
        if let Some(&node) = element.nodes.iter().find(|&&node| node >= mesh.nodes.len()) {
            return Err(FemProblemError::Backend(format!(
                "element {element_index} references node {node} outside the mesh"
            )));
        }
    }
    Ok(())
}

/// Compute element RHS contributions (returns (node_index, value) pairs)
fn compute_element_rhs<F>(
    mesh: &Mesh,
    elem_idx: usize,
    degree: PolynomialDegree,
    source: &F,
) -> Vec<(usize, Complex64)>
where
    F: Fn(f64, f64, f64) -> Complex64,
{
    use crate::basis::{Jacobian, evaluate_shape};
    use crate::mesh::ElementType;
    use crate::quadrature::for_mass;

    let elem = &mesh.elements[elem_idx];
    let elem_type = elem.element_type;
    let vertices = elem.vertices();
    let n_nodes = vertices.len();

    let quad = for_mass(elem_type, degree.degree());

    // Get coordinates based on dimension
    let is_3d = matches!(
        elem_type,
        ElementType::Tetrahedron | ElementType::Hexahedron
    );

    let mut f_local = vec![Complex64::new(0.0, 0.0); n_nodes];

    for qp in quad.iter() {
        let shape = evaluate_shape(elem_type, degree, qp.xi(), qp.eta(), qp.zeta());

        // Map reference coordinates to physical coordinates
        let (x, y, z, det_j) = if is_3d {
            let coords: Vec<[f64; 3]> = vertices
                .iter()
                .map(|&v| [mesh.nodes[v].x, mesh.nodes[v].y, mesh.nodes[v].z])
                .collect();
            let jac = Jacobian::from_3d(&shape.gradients, shape.grad_dim, &coords);

            let x: f64 = shape
                .values
                .iter()
                .zip(&coords)
                .map(|(n, c)| n * c[0])
                .sum();
            let y: f64 = shape
                .values
                .iter()
                .zip(&coords)
                .map(|(n, c)| n * c[1])
                .sum();
            let z: f64 = shape
                .values
                .iter()
                .zip(&coords)
                .map(|(n, c)| n * c[2])
                .sum();

            (x, y, z, jac.det.abs())
        } else {
            let coords: Vec<[f64; 2]> = vertices
                .iter()
                .map(|&v| [mesh.nodes[v].x, mesh.nodes[v].y])
                .collect();
            let jac = Jacobian::from_2d(&shape.gradients, shape.grad_dim, &coords);

            let x: f64 = shape
                .values
                .iter()
                .zip(&coords)
                .map(|(n, c)| n * c[0])
                .sum();
            let y: f64 = shape
                .values
                .iter()
                .zip(&coords)
                .map(|(n, c)| n * c[1])
                .sum();

            (x, y, 0.0, jac.det.abs())
        };

        let f_val = source(x, y, z);

        for (f, &s) in f_local.iter_mut().zip(shape.values.iter()) {
            *f += f_val * Complex64::new(s * det_j * qp.weight, 0.0);
        }
    }

    // Return (global_node_index, value) pairs
    vertices
        .iter()
        .enumerate()
        .map(|(i, &gi)| (gi, f_local[i]))
        .collect()
}

/// Complex-valued source form used by the legacy Helmholtz RHS adapter for
/// topology-numbered high-order H1 spaces.  The public generic `SourceForm`
/// is intentionally real-valued; Helmholtz sources are complex and therefore
/// need this small local form to reuse the checked production assembler.
struct ComplexSourceForm<'a, F> {
    source: &'a F,
}

impl<F> LinearForm<Complex64> for ComplexSourceForm<'_, F>
where
    F: Fn(f64, f64, f64) -> Complex64 + Sync,
{
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<Complex64>,
    ) {
        let [x, y, z] = test.physical_coords;
        let source = (self.source)(x, y, z);
        let weighted_source = source * (weight * test.det_j);
        for i in 0..test.n_dofs {
            local_vector.add(i, weighted_source * test.value(i));
        }
    }
}

/// Assemble right-hand side vector
fn assemble_rhs<F>(mesh: &Mesh, degree: PolynomialDegree, source: &F) -> Vec<Complex64>
where
    F: Fn(f64, f64, f64) -> Complex64 + Sync,
{
    // The legacy RHS kernel scatters only vertex nodes.  For P2/P3 the H1
    // space has topology-derived edge/face DOFs, so use the same checked
    // generic path as the high-order stiffness and mass adapters above.
    if degree != PolynomialDegree::P1 {
        let space = H1Space::new(degree);
        return super::generic::assemble_linear_checked(
            &ComplexSourceForm { source },
            &space,
            mesh,
            match degree {
                PolynomialDegree::P2 => 7,
                PolynomialDegree::P3 => 8,
                PolynomialDegree::P1 => unreachable!(),
            },
        )
        .unwrap_or_else(|error| panic!("invalid high-order Helmholtz RHS assembly: {error}"));
    }

    let n_dofs = mesh.num_nodes();
    let n_elems = mesh.num_elements();

    // Compute all element contributions (rayon above the machine-adaptive
    // gate, serial for small meshes)
    let all_contribs: Vec<Vec<(usize, Complex64)>> =
        collect_indexed(n_elems, ELEMENTS_PER_THREAD, |elem_idx| {
            compute_element_rhs(mesh, elem_idx, degree, source)
        });

    // Merge contributions into final RHS vector
    let mut rhs = vec![Complex64::new(0.0, 0.0); n_dofs];
    for contribs in all_contribs {
        for (gi, val) in contribs {
            rhs[gi] += val;
        }
    }

    rhs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::unit_square_triangles;
    use sonium_solvers::direct::lu_solve;

    #[test]
    fn checked_helmholtz_constructor_rejects_malformed_components() {
        let stiffness = StiffnessMatrix::new(2);
        let mass = MassMatrix::new(3);
        assert!(matches!(
            HelmholtzMatrix::try_new(&stiffness, &mass, Complex64::new(1.0, 0.0)),
            Err(HelmholtzMatrixError::DimensionMismatch {
                stiffness_dim: 2,
                mass_dim: 3,
            })
        ));

        let mut bad_stiffness = StiffnessMatrix::new(2);
        bad_stiffness.add(0, 2, 1.0);
        let valid_mass = MassMatrix::new(2);
        assert!(matches!(
            HelmholtzMatrix::try_new(&bad_stiffness, &valid_mass, Complex64::new(1.0, 0.0)),
            Err(HelmholtzMatrixError::TripletIndexOutOfBounds { .. })
        ));

        assert!(matches!(
            HelmholtzMatrix::try_new(
                &StiffnessMatrix::new(2),
                &MassMatrix::new(2),
                Complex64::new(f64::NAN, 0.0),
            ),
            Err(HelmholtzMatrixError::NonFiniteWavenumber)
        ));

        let mut malformed = HelmholtzMatrix::new(
            &StiffnessMatrix::new(2),
            &MassMatrix::new(2),
            Complex64::new(1.0, 0.0),
        );
        malformed.rows.push(0);
        assert!(matches!(
            malformed.try_to_csr(),
            Err(HelmholtzMatrixError::TripletLengthMismatch {
                matrix: "helmholtz",
                ..
            })
        ));
    }

    #[test]
    fn checked_assembly_rejects_malformed_mesh_before_element_kernels() {
        let mut mesh = unit_square_triangles(1);
        mesh.elements[0].nodes[0] = mesh.nodes.len();

        let error = HelmholtzProblem::try_assemble(
            &mesh,
            PolynomialDegree::P1,
            Complex64::new(1.0, 0.0),
            |_, _, _| Complex64::new(1.0, 0.0),
        )
        .expect_err("checked assembly must reject invalid connectivity");
        assert!(matches!(error, HelmholtzMatrixError::InvalidMesh { .. }));
    }

    #[test]
    fn checked_assembly_rejects_non_finite_source_contributions() {
        let mesh = unit_square_triangles(1);
        let error = HelmholtzProblem::try_assemble(
            &mesh,
            PolynomialDegree::P1,
            Complex64::new(1.0, 0.0),
            |_, _, _| Complex64::new(f64::NAN, 0.0),
        )
        .expect_err("checked assembly must reject a non-finite RHS");
        assert!(matches!(error, HelmholtzMatrixError::NonFiniteRhs { .. }));
    }

    #[test]
    fn test_helmholtz_assembly() {
        let mesh = unit_square_triangles(4);
        let k = Complex64::new(1.0, 0.0);

        let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |_, _, _| {
            Complex64::new(1.0, 0.0)
        });

        assert_eq!(problem.num_dofs(), mesh.num_nodes());
        assert!(problem.matrix.nnz() > 0);
    }

    #[test]
    fn xem_admission_propagates_convention_and_checks_pressure_layout() {
        let mesh = unit_square_triangles(1);
        let mut exchange = sonium_xem::MultiphysicsProblem {
            id: "fem-admission".into(),
            domains: vec![sonium_xem::Domain {
                id: "acoustics".into(),
                label: "Acoustics".into(),
                kind: sonium_xem::DomainKind::Acoustics,
            }],
            fields: vec![sonium_xem::FieldSpec {
                id: "pressure".into(),
                name: "nodal pressure".into(),
                domain_id: "acoustics".into(),
                kind: sonium_xem::FieldKind::ComplexScalar,
                units: Some("Pa".into()),
                layout: sonium_xem::TensorLayout {
                    scalar_type: sonium_xem::ScalarType::ComplexF64,
                    shape: vec![mesh.num_nodes()],
                    strides: vec![1],
                    ..Default::default()
                },
            }],
            ..Default::default()
        };
        exchange.conventions.harmonic_time = sonium_xem::HarmonicConvention::ExpPlusIwt;
        let admitted = HelmholtzProblem::assemble_from_xem(
            &exchange,
            "pressure",
            &mesh,
            PolynomialDegree::P1,
            Complex64::new(1.0, 0.0),
            |_, _, _| Complex64::new(1.0, 0.0),
        )
        .expect("valid XEM metadata and mesh should be admitted");
        assert_eq!(admitted.convention.spatial_sign, -1);
        assert_eq!(admitted.num_dofs(), mesh.num_nodes());

        exchange.fields[0].units = Some("m/s".into());
        assert!(matches!(
            HelmholtzProblem::assemble_from_xem(
                &exchange,
                "pressure",
                &mesh,
                PolynomialDegree::P1,
                Complex64::new(1.0, 0.0),
                |_, _, _| Complex64::new(1.0, 0.0),
            ),
            Err(crate::conventions::FemProblemError::Field { .. })
        ));

        exchange.fields[0].units = Some("Pa".into());
        exchange.fields[0].layout.shape = vec![mesh.num_nodes() + 1];
        assert!(matches!(
            HelmholtzProblem::assemble_from_xem(
                &exchange,
                "pressure",
                &mesh,
                PolynomialDegree::P1,
                Complex64::new(1.0, 0.0),
                |_, _, _| Complex64::new(1.0, 0.0),
            ),
            Err(crate::conventions::FemProblemError::Field { .. })
        ));

        let mut invalid_mesh = mesh.clone();
        invalid_mesh.elements.clear();
        exchange.fields[0].layout.shape = vec![mesh.num_nodes()];
        assert!(matches!(
            HelmholtzProblem::assemble_from_xem(
                &exchange,
                "pressure",
                &invalid_mesh,
                PolynomialDegree::P1,
                Complex64::new(1.0, 0.0),
                |_, _, _| Complex64::new(1.0, 0.0),
            ),
            Err(crate::conventions::FemProblemError::Backend(_))
        ));

        exchange.fields.push(sonium_xem::FieldSpec {
            id: "temperature".into(),
            name: "temperature".into(),
            domain_id: "acoustics".into(),
            kind: sonium_xem::FieldKind::Scalar,
            units: Some("K".into()),
            layout: sonium_xem::TensorLayout {
                shape: vec![mesh.num_nodes()],
                strides: vec![1],
                ..Default::default()
            },
        });
        exchange.couplings = vec![
            sonium_xem::Coupling {
                id: "pressure-temperature".into(),
                source_field_id: "pressure".into(),
                target_field_id: "temperature".into(),
                kind: sonium_xem::CouplingKind::Partitioned,
                conservation: Some("energy".into()),
            },
            sonium_xem::Coupling {
                id: "temperature-pressure".into(),
                source_field_id: "temperature".into(),
                target_field_id: "pressure".into(),
                kind: sonium_xem::CouplingKind::Partitioned,
                conservation: Some("energy".into()),
            },
        ];
        assert!(matches!(
            HelmholtzProblem::assemble_from_xem(
                &exchange,
                "pressure",
                &mesh,
                PolynomialDegree::P1,
                Complex64::new(1.0, 0.0),
                |_, _, _| Complex64::new(1.0, 0.0),
            ),
            Err(crate::conventions::FemProblemError::CouplingGraph(
                sonium_xem::CouplingGraphError::Cycle(_)
            ))
        ));
    }

    #[test]
    fn test_helmholtz_k_zero_equals_laplacian() {
        // When k=0, Helmholtz reduces to Laplace: -∇²u = f
        use std::collections::HashMap;

        let mesh = unit_square_triangles(4);
        let k = Complex64::new(0.0, 0.0);

        let stiffness = assemble_stiffness(&mesh, PolynomialDegree::P1);
        let mass = assemble_mass(&mesh, PolynomialDegree::P1);
        let helmholtz = HelmholtzMatrix::new(&stiffness, &mass, k).to_compressed();

        // Compress stiffness for fair comparison
        let mut stiff_map: HashMap<(usize, usize), Complex64> = HashMap::new();
        for i in 0..stiffness.nnz() {
            *stiff_map
                .entry((stiffness.rows[i], stiffness.cols[i]))
                .or_insert(Complex64::new(0.0, 0.0)) += Complex64::new(stiffness.values[i], 0.0);
        }

        // Check that Helmholtz values match stiffness values
        for i in 0..helmholtz.nnz() {
            let key = (helmholtz.rows[i], helmholtz.cols[i]);
            let stiff_val = stiff_map
                .get(&key)
                .copied()
                .unwrap_or(Complex64::new(0.0, 0.0));
            let diff = (helmholtz.values[i] - stiff_val).norm();
            assert!(
                diff < 1e-10,
                "Mismatch at {:?}: {} vs {}",
                key,
                helmholtz.values[i],
                stiff_val
            );
        }
    }

    #[test]
    fn test_helmholtz_rhs() {
        let mesh = unit_square_triangles(4);
        let k = Complex64::new(1.0, 0.0);

        // Constant source f=1 should give non-zero RHS
        let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |_, _, _| {
            Complex64::new(1.0, 0.0)
        });

        // Sum of RHS should be approximately 1 (integral of 1 over unit square)
        let rhs_sum: Complex64 = problem.rhs.iter().sum();
        assert!(
            (rhs_sum.re - 1.0).abs() < 1e-10,
            "RHS sum {} should be 1.0",
            rhs_sum
        );
    }

    #[test]
    fn test_helmholtz_complex_wavenumber() {
        // Complex wavenumber for damped Helmholtz
        let mesh = unit_square_triangles(2);
        let k = Complex64::new(1.0, 0.1); // Slightly damped

        let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |_, _, _| {
            Complex64::new(1.0, 0.0)
        });

        // Matrix should have complex entries
        let has_imag = problem.matrix.values.iter().any(|v| v.im.abs() > 1e-15);
        assert!(
            has_imag,
            "Complex wavenumber should produce complex matrix entries"
        );
    }

    #[test]
    fn test_poisson_mms_convergence_gate_on_refined_triangles() {
        // u = x(1-x)y(1-y), with -Δu = 2(x(1-x) + y(1-y)). The exact
        // manufactured solution vanishes on the square boundary, so this
        // exercises the assembled stiffness/RHS and Dirichlet elimination
        // rather than only checking matrix symmetry.
        let mut errors = Vec::new();
        for subdivisions in [2, 4, 8] {
            let mesh = unit_square_triangles(subdivisions);
            let problem = HelmholtzProblem::assemble(
                &mesh,
                PolynomialDegree::P1,
                Complex64::new(0.0, 0.0),
                |x, y, _| Complex64::new(2.0 * (x * (1.0 - x) + y * (1.0 - y)), 0.0),
            );
            let mut matrix = problem.matrix.to_csr().to_dense();
            let mut rhs = problem.rhs;
            let boundary: Vec<usize> = mesh
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(index, point)| {
                    (point.x.abs() < 1e-12
                        || point.y.abs() < 1e-12
                        || (point.x - 1.0).abs() < 1e-12
                        || (point.y - 1.0).abs() < 1e-12)
                        .then_some(index)
                })
                .collect();
            for &node in &boundary {
                for column in 0..matrix.ncols() {
                    matrix[[node, column]] = Complex64::new(0.0, 0.0);
                }
                for row in 0..matrix.nrows() {
                    matrix[[row, node]] = Complex64::new(0.0, 0.0);
                }
                matrix[[node, node]] = Complex64::new(1.0, 0.0);
                rhs[node] = Complex64::new(0.0, 0.0);
            }
            let solution = lu_solve(&matrix, &rhs).expect("MMS Dirichlet system should solve");
            let max_error = mesh
                .nodes
                .iter()
                .enumerate()
                .map(|(index, point)| {
                    let exact = point.x * (1.0 - point.x) * point.y * (1.0 - point.y);
                    (solution[index].re - exact).abs()
                })
                .fold(0.0, f64::max);
            errors.push(max_error);
        }

        let observed_order = (errors[0] / errors[2]).ln() / 4.0_f64.ln();
        assert!(
            observed_order > 1.7,
            "P1 manufactured-solution convergence order {observed_order:.3} is too low: {errors:?}"
        );
    }

    #[test]
    fn test_helmholtz_mms_convergence_gate_at_nonzero_wavenumber() {
        // u = sin(πx) sin(πy) solves
        // -Δu - k²u = (2π² - k²) sin(πx) sin(πy), with homogeneous
        // Dirichlet data.  This gate exercises the mass term as well as the
        // stiffness/RHS path and catches sign errors hidden by k = 0 tests.
        let k = Complex64::new(2.3, 0.0);
        let mut errors = Vec::new();
        for subdivisions in [4, 8, 16] {
            let mesh = unit_square_triangles(subdivisions);
            let coefficient = 2.0 * std::f64::consts::PI.powi(2) - k.re * k.re;
            let problem = HelmholtzProblem::assemble(&mesh, PolynomialDegree::P1, k, |x, y, _| {
                Complex64::new(
                    coefficient
                        * (std::f64::consts::PI * x).sin()
                        * (std::f64::consts::PI * y).sin(),
                    0.0,
                )
            });
            let mut matrix = problem.matrix.to_csr().to_dense();
            let mut rhs = problem.rhs;
            let boundary: Vec<usize> = mesh
                .nodes
                .iter()
                .enumerate()
                .filter_map(|(index, point)| {
                    (point.x.abs() < 1e-12
                        || point.y.abs() < 1e-12
                        || (point.x - 1.0).abs() < 1e-12
                        || (point.y - 1.0).abs() < 1e-12)
                        .then_some(index)
                })
                .collect();
            for &node in &boundary {
                for column in 0..matrix.ncols() {
                    matrix[[node, column]] = Complex64::new(0.0, 0.0);
                }
                for row in 0..matrix.nrows() {
                    matrix[[row, node]] = Complex64::new(0.0, 0.0);
                }
                matrix[[node, node]] = Complex64::new(1.0, 0.0);
                rhs[node] = Complex64::new(0.0, 0.0);
            }
            let solution = lu_solve(&matrix, &rhs).expect("Helmholtz MMS system should solve");
            let max_error = mesh
                .nodes
                .iter()
                .enumerate()
                .map(|(index, point)| {
                    let exact = (std::f64::consts::PI * point.x).sin()
                        * (std::f64::consts::PI * point.y).sin();
                    (solution[index].re - exact).abs()
                })
                .fold(0.0, f64::max);
            errors.push(max_error);
        }

        let observed_order = (errors[0] / errors[2]).ln() / 4.0_f64.ln();
        assert!(
            observed_order > 1.8,
            "nonzero-k Helmholtz convergence order {observed_order:.3} is too low: {errors:?}"
        );
    }
}
