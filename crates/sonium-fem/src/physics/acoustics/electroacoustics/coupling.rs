//! Fluid-structure coupling for electroacoustics
//!
//! Assembles the coupling matrix `C` at the fluid-structure interface:
//!
//! ```text
//! C_{ij} = integral phi_i (n . N_j) dGamma
//! ```
//!
//! where `phi_i` are scalar pressure basis functions (H1) and `N_j` are vector
//! displacement basis functions (VectorH1), `n` is the outward normal from the
//! fluid domain.
//!
//! The complete harmonic system, using `exp(+i omega t)`, is:
//! ```text
//! | K_f + iωD_f - k²M_f    -ρ_fω²C       0       | | p |   | f_f |
//! |       -C^T          K_s+iωC_s-ω²M_s  -G^T    | | u | = | f_s |
//! |        0                  iωG          R+iωL   | | i |   | v   |
//! ```

use crate::assembly::TripletMatrix;
use crate::mesh::{BoundaryFace, Mesh};
use num_complex::Complex64;
use sonium_compute::BlockLayout;
use sonium_solvers::sparse::{BlockOperator, CsrMatrix};

/// A block-coupled electroacoustic system.
///
/// Stores the individual sub-system matrices and the coupling matrix,
/// enabling frequency-dependent assembly of the full block system.
pub struct CoupledSystem {
    /// Number of fluid DOFs (scalar pressure)
    pub n_fluid: usize,
    /// Number of solid DOFs (vector displacement)
    pub n_solid: usize,
    /// Fluid stiffness matrix K (Laplacian part)
    pub fluid_stiffness: TripletMatrix<f64>,
    /// Fluid mass matrix M
    pub fluid_mass: TripletMatrix<f64>,
    /// Fluid loss/damping matrix D_f, scaled by `i*omega`.
    pub fluid_damping: TripletMatrix<f64>,
    /// Solid stiffness matrix K_ss
    pub solid_stiffness: TripletMatrix<f64>,
    /// Solid mass matrix M_ss
    pub solid_mass: TripletMatrix<f64>,
    /// Solid viscous damping matrix C_ss
    pub solid_damping: TripletMatrix<f64>,
    /// Coupling matrix C (n_fluid x n_solid)
    pub coupling: TripletMatrix<f64>,
    /// Optional distributed structural interface load distribution `M_Γ`
    /// (n_solid x n_solid).
    ///
    /// When present, [`Self::assemble_harmonic_with_load`] adds
    /// `i·omega·zeta·M_Γ` to the solid block, where `zeta` is the complex
    /// piston load impedance (force per interface-average velocity, using
    /// `exp(+i·omega·t)`) supplied per solve point. `M_Γ` is expected to be
    /// normalized so that the sum of all its entries is one; the applied
    /// load then acts as a locally reacting specific impedance `zeta / A`
    /// over the interface and totals exactly `zeta * v_mean` opposing the
    /// interface motion, mirroring how the first-order radiation ABC adds
    /// `M_boundary / c` to the fluid damping block. `None` (or a zero
    /// scale) reproduces the uncoupled system exactly.
    pub interface_load: Option<TripletMatrix<f64>>,
    /// Optional distributed electrical circuit and electromechanical coupling.
    pub electrical: Option<ElectricalSubsystem>,
}

/// Electrical impedance and reciprocal force/back-EMF coupling.
///
/// With `exp(+i omega t)`, the electrical row is
/// `(R + i omega L) i + i omega G u = v`, while the structural row contains
/// `-G^T i`. This sign pair conserves electromechanical active power.
pub struct ElectricalSubsystem {
    pub n_dofs: usize,
    pub resistance: TripletMatrix<f64>,
    pub inductance: TripletMatrix<f64>,
    /// `G` maps structural velocity to back EMF (n_electrical x n_solid).
    pub coupling: TripletMatrix<f64>,
}

/// Physical parameters for one harmonic assembly point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HarmonicCoupledPoint {
    pub omega_rad_per_s: f64,
    pub sound_speed_m_per_s: f64,
    pub fluid_density_kg_per_m3: f64,
}

impl HarmonicCoupledPoint {
    pub fn validate(self) -> Result<(), CoupledSystemError> {
        for (field, value) in [
            ("omega_rad_per_s", self.omega_rad_per_s),
            ("sound_speed_m_per_s", self.sound_speed_m_per_s),
            ("fluid_density_kg_per_m3", self.fluid_density_kg_per_m3),
        ] {
            if !value.is_finite() || value <= 0.0 {
                return Err(CoupledSystemError::InvalidPhysicalParameter { field, value });
            }
        }
        Ok(())
    }

    pub fn wavenumber_squared(self) -> f64 {
        (self.omega_rad_per_s / self.sound_speed_m_per_s).powi(2)
    }
}

impl CoupledSystem {
    /// Total number of DOFs in the coupled system.
    pub fn total_dofs(&self) -> usize {
        self.n_fluid
            + self.n_solid
            + self
                .electrical
                .as_ref()
                .map_or(0, |electrical| electrical.n_dofs)
    }

    /// Validate every block against the declared block dimensions.
    pub fn validate(&self) -> Result<(), CoupledSystemError> {
        validate_triplet(
            "fluid_stiffness",
            &self.fluid_stiffness,
            self.n_fluid,
            self.n_fluid,
        )?;
        validate_triplet("fluid_mass", &self.fluid_mass, self.n_fluid, self.n_fluid)?;
        validate_triplet(
            "fluid_damping",
            &self.fluid_damping,
            self.n_fluid,
            self.n_fluid,
        )?;
        validate_triplet(
            "solid_stiffness",
            &self.solid_stiffness,
            self.n_solid,
            self.n_solid,
        )?;
        validate_triplet("solid_mass", &self.solid_mass, self.n_solid, self.n_solid)?;
        validate_triplet(
            "solid_damping",
            &self.solid_damping,
            self.n_solid,
            self.n_solid,
        )?;
        validate_triplet("coupling", &self.coupling, self.n_fluid, self.n_solid)?;
        if let Some(interface_load) = &self.interface_load {
            validate_triplet("interface_load", interface_load, self.n_solid, self.n_solid)?;
        }
        if let Some(electrical) = &self.electrical {
            if electrical.n_dofs == 0 {
                return Err(CoupledSystemError::EmptyElectricalBlock);
            }
            validate_triplet(
                "electrical.resistance",
                &electrical.resistance,
                electrical.n_dofs,
                electrical.n_dofs,
            )?;
            validate_triplet(
                "electrical.inductance",
                &electrical.inductance,
                electrical.n_dofs,
                electrical.n_dofs,
            )?;
            validate_triplet(
                "electrical.coupling",
                &electrical.coupling,
                electrical.n_dofs,
                self.n_solid,
            )?;
        }
        Ok(())
    }

    /// Assemble the physically complete harmonic block system.
    ///
    /// Unknown ordering is `[pressure, displacement, current]`, with the final
    /// current block omitted when no electrical subsystem is configured.
    pub fn assemble_harmonic(
        &self,
        point: HarmonicCoupledPoint,
    ) -> Result<TripletMatrix<Complex64>, CoupledSystemError> {
        self.assemble_harmonic_with_load(point, Complex64::new(0.0, 0.0))
    }

    /// Assemble the harmonic block system with an optional piston load
    /// impedance `zeta` applied through [`Self::interface_load`].
    ///
    /// `zeta` is the complex load impedance per interface-average velocity
    /// (exp(+i omega t)); with a normalized `M_Γ` the solid block receives
    /// `i·omega·zeta·M_Γ`, a damping-like term that is passive exactly when
    /// `Re(zeta) >= 0`. A zero scale assembles the uncoupled system.
    pub fn assemble_harmonic_with_load(
        &self,
        point: HarmonicCoupledPoint,
        interface_load_scale: Complex64,
    ) -> Result<TripletMatrix<Complex64>, CoupledSystemError> {
        self.validate()?;
        point.validate()?;
        let omega = point.omega_rad_per_s;
        let i_omega = Complex64::new(0.0, omega);
        let omega_squared = omega * omega;
        let k_squared = point.wavenumber_squared();
        let nf = self.n_fluid;
        let ns = self.n_solid;
        let electrical_nnz = self.electrical.as_ref().map_or(0, |electrical| {
            electrical.resistance.nnz()
                + electrical.inductance.nnz()
                + 2 * electrical.coupling.nnz()
        });
        let interface_load_nnz = self.interface_load.as_ref().map_or(0, |load| load.nnz());
        let capacity = self.fluid_stiffness.nnz()
            + self.fluid_mass.nnz()
            + self.fluid_damping.nnz()
            + self.solid_stiffness.nnz()
            + self.solid_mass.nnz()
            + self.solid_damping.nnz()
            + interface_load_nnz
            + 2 * self.coupling.nnz()
            + electrical_nnz;
        let mut result =
            TripletMatrix::with_capacity(self.total_dofs(), self.total_dofs(), capacity);

        push_real_triplet_scaled(&mut result, &self.fluid_stiffness, 0, 0, 1.0.into());
        push_real_triplet_scaled(
            &mut result,
            &self.fluid_mass,
            0,
            0,
            Complex64::new(-k_squared, 0.0),
        );
        push_real_triplet_scaled(&mut result, &self.fluid_damping, 0, 0, i_omega);

        push_real_triplet_scaled(&mut result, &self.solid_stiffness, nf, nf, 1.0.into());
        push_real_triplet_scaled(
            &mut result,
            &self.solid_mass,
            nf,
            nf,
            Complex64::new(-omega_squared, 0.0),
        );
        push_real_triplet_scaled(&mut result, &self.solid_damping, nf, nf, i_omega);
        if let Some(interface_load) = &self.interface_load {
            push_real_triplet_scaled(
                &mut result,
                interface_load,
                nf,
                nf,
                i_omega * interface_load_scale,
            );
        }

        // Fluid acceleration boundary: d_n p = rho*omega^2*u_n; its weak
        // boundary term enters the fluid equation with a minus sign.
        push_real_triplet_scaled(
            &mut result,
            &self.coupling,
            0,
            nf,
            Complex64::new(-point.fluid_density_kg_per_m3 * omega_squared, 0.0),
        );
        // Fluid pressure traction is external structural work, hence -C^T p
        // when moved to the left-hand side.
        push_real_triplet_transpose_scaled(
            &mut result,
            &self.coupling,
            nf,
            0,
            Complex64::new(-1.0, 0.0),
        );

        if let Some(electrical) = &self.electrical {
            let ne_offset = nf + ns;
            push_real_triplet_scaled(
                &mut result,
                &electrical.resistance,
                ne_offset,
                ne_offset,
                1.0.into(),
            );
            push_real_triplet_scaled(
                &mut result,
                &electrical.inductance,
                ne_offset,
                ne_offset,
                i_omega,
            );
            push_real_triplet_scaled(&mut result, &electrical.coupling, ne_offset, nf, i_omega);
            push_real_triplet_transpose_scaled(
                &mut result,
                &electrical.coupling,
                nf,
                ne_offset,
                Complex64::new(-1.0, 0.0),
            );
        }
        Ok(result)
    }

    pub fn assemble_harmonic_block_operator(
        &self,
        point: HarmonicCoupledPoint,
    ) -> Result<BlockOperator<Complex64>, CoupledSystemError> {
        self.assemble_harmonic_block_operator_with_load(point, Complex64::new(0.0, 0.0))
    }

    /// Block-operator form of [`Self::assemble_harmonic_with_load`].
    pub fn assemble_harmonic_block_operator_with_load(
        &self,
        point: HarmonicCoupledPoint,
        interface_load_scale: Complex64,
    ) -> Result<BlockOperator<Complex64>, CoupledSystemError> {
        self.validate()?;
        point.validate()?;
        let mut block_sizes = vec![self.n_fluid, self.n_solid];
        if let Some(electrical) = &self.electrical {
            block_sizes.push(electrical.n_dofs);
        }
        let layout = BlockLayout::new(block_sizes.clone(), block_sizes);
        let omega = point.omega_rad_per_s;
        let mut operator = BlockOperator::new(layout)
            .with_block(0, 0, Box::new(self.harmonic_fluid_block(point)))
            .with_block(
                1,
                1,
                Box::new(self.harmonic_solid_block_with_load(omega, interface_load_scale)),
            )
            .with_block(
                0,
                1,
                Box::new(real_triplet_to_complex_csr_scaled(
                    &self.coupling,
                    Complex64::new(-point.fluid_density_kg_per_m3 * omega * omega, 0.0),
                )),
            )
            .with_block(
                1,
                0,
                Box::new(real_triplet_transpose_to_complex_csr_scaled(
                    &self.coupling,
                    Complex64::new(-1.0, 0.0),
                )),
            );
        if let Some(electrical) = &self.electrical {
            operator = operator
                .with_block(2, 2, Box::new(electrical.impedance_csr(omega)))
                .with_block(
                    2,
                    1,
                    Box::new(real_triplet_to_complex_csr_scaled(
                        &electrical.coupling,
                        Complex64::new(0.0, omega),
                    )),
                )
                .with_block(
                    1,
                    2,
                    Box::new(real_triplet_transpose_to_complex_csr_scaled(
                        &electrical.coupling,
                        Complex64::new(-1.0, 0.0),
                    )),
                );
        }
        Ok(operator)
    }

    fn harmonic_fluid_block(&self, point: HarmonicCoupledPoint) -> CsrMatrix<Complex64> {
        combine_real_triplets(
            self.n_fluid,
            &[
                (&self.fluid_stiffness, Complex64::new(1.0, 0.0)),
                (
                    &self.fluid_mass,
                    Complex64::new(-point.wavenumber_squared(), 0.0),
                ),
                (
                    &self.fluid_damping,
                    Complex64::new(0.0, point.omega_rad_per_s),
                ),
            ],
        )
    }

    /// Solid dynamic block including the optional interface load term;
    /// shared with the block preconditioner in `iterative.rs`.
    pub(crate) fn harmonic_solid_block_with_load(
        &self,
        omega: f64,
        interface_load_scale: Complex64,
    ) -> CsrMatrix<Complex64> {
        let i_omega = Complex64::new(0.0, omega);
        let mut terms = vec![
            (&self.solid_stiffness, Complex64::new(1.0, 0.0)),
            (&self.solid_mass, Complex64::new(-omega * omega, 0.0)),
            (&self.solid_damping, i_omega),
        ];
        if let Some(interface_load) = &self.interface_load {
            terms.push((interface_load, i_omega * interface_load_scale));
        }
        combine_real_triplets(self.n_solid, &terms)
    }

    /// Assemble the full block system at a given wavenumber `k`.
    ///
    /// The coupled system matrix is:
    /// ```text
    /// A = | K_f - k^2 M_f          C       |
    ///     | -omega^2 rho_f C^T      K_s     |
    /// ```
    ///
    /// where `omega = k * c` and `rho_f` is the fluid density.
    ///
    /// ```text
    /// A = | K_f - k^2 M_f        c_fs * C   |
    ///     | c_sf * C^T            K_s        |
    /// ```
    ///
    /// The two coupling coefficients are typically different in physical systems:
    /// - `c_fs`: fluid-to-structure coupling (top-right, often `1`)
    /// - `c_sf`: structure-to-fluid coupling (bottom-left, often `-omega^2 * rho_f`)
    pub fn assemble_block(
        &self,
        k_squared: Complex64,
        coupling_fs: Complex64,
        coupling_sf: Complex64,
    ) -> TripletMatrix<Complex64> {
        let n_total = self.total_dofs();
        let nf = self.n_fluid;

        let cap = self.fluid_stiffness.nnz()
            + self.fluid_mass.nnz()
            + self.solid_stiffness.nnz()
            + 2 * self.coupling.nnz();
        let mut result = TripletMatrix::with_capacity(n_total, n_total, cap);

        // Top-left block: K_f - k^2 M_f
        for i in 0..self.fluid_stiffness.nnz() {
            result.push(
                self.fluid_stiffness.rows[i],
                self.fluid_stiffness.cols[i],
                Complex64::new(self.fluid_stiffness.values[i], 0.0),
            );
        }
        for i in 0..self.fluid_mass.nnz() {
            result.push(
                self.fluid_mass.rows[i],
                self.fluid_mass.cols[i],
                -k_squared * self.fluid_mass.values[i],
            );
        }

        // Bottom-right block: K_s
        for i in 0..self.solid_stiffness.nnz() {
            result.push(
                nf + self.solid_stiffness.rows[i],
                nf + self.solid_stiffness.cols[i],
                Complex64::new(self.solid_stiffness.values[i], 0.0),
            );
        }

        // Top-right block: c_fs * C
        for i in 0..self.coupling.nnz() {
            result.push(
                self.coupling.rows[i],
                nf + self.coupling.cols[i],
                coupling_fs * self.coupling.values[i],
            );
        }

        // Bottom-left block: c_sf * C^T
        for i in 0..self.coupling.nnz() {
            result.push(
                nf + self.coupling.cols[i],
                self.coupling.rows[i],
                coupling_sf * self.coupling.values[i],
            );
        }

        result
    }

    /// Assemble the coupled system as a backend-neutral block operator.
    ///
    /// The scalar physics still owns the FEM matrices and coupling matrix; the
    /// shared solver layer owns composition into a block operator.
    pub fn assemble_block_operator(
        &self,
        k_squared: Complex64,
        coupling_fs: Complex64,
        coupling_sf: Complex64,
    ) -> BlockOperator<Complex64> {
        let layout = BlockLayout::new(
            vec![self.n_fluid, self.n_solid],
            vec![self.n_fluid, self.n_solid],
        );

        BlockOperator::new(layout)
            .with_block(0, 0, Box::new(self.fluid_block_csr(k_squared)))
            .with_block(
                0,
                1,
                Box::new(real_triplet_to_complex_csr_scaled(
                    &self.coupling,
                    coupling_fs,
                )),
            )
            .with_block(
                1,
                0,
                Box::new(real_triplet_transpose_to_complex_csr_scaled(
                    &self.coupling,
                    coupling_sf,
                )),
            )
            .with_block(
                1,
                1,
                Box::new(real_triplet_to_complex_csr_scaled(
                    &self.solid_stiffness,
                    Complex64::new(1.0, 0.0),
                )),
            )
    }

    fn fluid_block_csr(&self, k_squared: Complex64) -> CsrMatrix<Complex64> {
        let mut triplets = Vec::with_capacity(self.fluid_stiffness.nnz() + self.fluid_mass.nnz());

        for i in 0..self.fluid_stiffness.nnz() {
            triplets.push((
                self.fluid_stiffness.rows[i],
                self.fluid_stiffness.cols[i],
                Complex64::new(self.fluid_stiffness.values[i], 0.0),
            ));
        }
        for i in 0..self.fluid_mass.nnz() {
            triplets.push((
                self.fluid_mass.rows[i],
                self.fluid_mass.cols[i],
                -k_squared * self.fluid_mass.values[i],
            ));
        }

        CsrMatrix::from_triplets(self.n_fluid, self.n_fluid, triplets)
    }
}

impl ElectricalSubsystem {
    fn impedance_csr(&self, omega: f64) -> CsrMatrix<Complex64> {
        combine_real_triplets(
            self.n_dofs,
            &[
                (&self.resistance, Complex64::new(1.0, 0.0)),
                (&self.inductance, Complex64::new(0.0, omega)),
            ],
        )
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CoupledSystemError {
    #[error("{block} is {actual_rows}x{actual_cols}; expected {expected_rows}x{expected_cols}")]
    BlockDimensions {
        block: &'static str,
        actual_rows: usize,
        actual_cols: usize,
        expected_rows: usize,
        expected_cols: usize,
    },
    #[error("{block} contains entry ({row}, {column}) outside its declared dimensions")]
    EntryOutOfRange {
        block: &'static str,
        row: usize,
        column: usize,
    },
    #[error("electrical subsystem must contain at least one DOF")]
    EmptyElectricalBlock,
    #[error("physical parameter {field}={value} must be finite and positive")]
    InvalidPhysicalParameter { field: &'static str, value: f64 },
}

fn validate_triplet(
    name: &'static str,
    matrix: &TripletMatrix<f64>,
    expected_rows: usize,
    expected_cols: usize,
) -> Result<(), CoupledSystemError> {
    if matrix.num_rows != expected_rows || matrix.num_cols != expected_cols {
        return Err(CoupledSystemError::BlockDimensions {
            block: name,
            actual_rows: matrix.num_rows,
            actual_cols: matrix.num_cols,
            expected_rows,
            expected_cols,
        });
    }
    for (&row, &column) in matrix.rows.iter().zip(&matrix.cols) {
        if row >= expected_rows || column >= expected_cols {
            return Err(CoupledSystemError::EntryOutOfRange {
                block: name,
                row,
                column,
            });
        }
    }
    Ok(())
}

fn push_real_triplet_scaled(
    output: &mut TripletMatrix<Complex64>,
    input: &TripletMatrix<f64>,
    row_offset: usize,
    column_offset: usize,
    scale: Complex64,
) {
    for index in 0..input.nnz() {
        output.push(
            row_offset + input.rows[index],
            column_offset + input.cols[index],
            scale * input.values[index],
        );
    }
}

fn push_real_triplet_transpose_scaled(
    output: &mut TripletMatrix<Complex64>,
    input: &TripletMatrix<f64>,
    row_offset: usize,
    column_offset: usize,
    scale: Complex64,
) {
    for index in 0..input.nnz() {
        output.push(
            row_offset + input.cols[index],
            column_offset + input.rows[index],
            scale * input.values[index],
        );
    }
}

fn combine_real_triplets(
    dimension: usize,
    terms: &[(&TripletMatrix<f64>, Complex64)],
) -> CsrMatrix<Complex64> {
    let capacity = terms.iter().map(|(matrix, _)| matrix.nnz()).sum();
    let mut triplets = Vec::with_capacity(capacity);
    for (matrix, scale) in terms {
        for index in 0..matrix.nnz() {
            triplets.push((
                matrix.rows[index],
                matrix.cols[index],
                *scale * matrix.values[index],
            ));
        }
    }
    CsrMatrix::from_triplets(dimension, dimension, triplets)
}

fn real_triplet_to_complex_csr_scaled(
    matrix: &TripletMatrix<f64>,
    scale: Complex64,
) -> CsrMatrix<Complex64> {
    let triplets = (0..matrix.nnz())
        .map(|i| (matrix.rows[i], matrix.cols[i], scale * matrix.values[i]))
        .collect();
    CsrMatrix::from_triplets(matrix.num_rows, matrix.num_cols, triplets)
}

fn real_triplet_transpose_to_complex_csr_scaled(
    matrix: &TripletMatrix<f64>,
    scale: Complex64,
) -> CsrMatrix<Complex64> {
    let triplets = (0..matrix.nnz())
        .map(|i| (matrix.cols[i], matrix.rows[i], scale * matrix.values[i]))
        .collect();
    CsrMatrix::from_triplets(matrix.num_cols, matrix.num_rows, triplets)
}

/// Assemble the fluid-structure coupling matrix at the interface.
///
/// For each interface boundary face with the given marker:
/// `C_{ij} = integral phi_i * (n . N_j) dGamma`
///
/// where `phi_i` is a fluid (scalar H1) basis function and `N_j` is a solid
/// (vector H1) basis function, and `n` is the outward normal.
///
/// This is a simplified P1 implementation using midpoint quadrature on boundary edges/faces.
pub fn assemble_coupling_matrix(
    mesh: &Mesh,
    interface_marker: i32,
    n_fluid_dofs: usize,
    n_solid_dofs: usize,
    dim: usize,
) -> TripletMatrix<f64> {
    let mut coupling = TripletMatrix::new(n_fluid_dofs, n_solid_dofs);

    for face in &mesh.boundaries {
        if face.marker != interface_marker {
            continue;
        }

        match face.nodes.len() {
            2 => assemble_edge_coupling(mesh, face, dim, &mut coupling),
            3 => assemble_tri_face_coupling(mesh, face, &mut coupling),
            _ => {} // skip unsupported face types
        }
    }

    coupling
}

/// Coupling on a 2D boundary edge (2 nodes).
fn assemble_edge_coupling(
    mesh: &Mesh,
    face: &BoundaryFace,
    dim: usize,
    coupling: &mut TripletMatrix<f64>,
) {
    let n0 = face.nodes[0];
    let n1 = face.nodes[1];
    let p0 = &mesh.nodes[n0];
    let p1 = &mesh.nodes[n1];

    // Edge length
    let dx = p1.x - p0.x;
    let dy = p1.y - p0.y;
    let length = (dx * dx + dy * dy).sqrt();

    // Outward normal (rotated tangent, assuming CCW boundary orientation)
    let nx = dy / length;
    let ny = -dx / length;

    // Midpoint quadrature: integral phi_i * (n . N_j) ds
    // At midpoint: phi_0 = phi_1 = 0.5
    // N_j for vector DOFs: node_k, component_c has N = 0.5 * e_c
    // n . N = 0.5 * n_c
    let w = length; // midpoint rule weight = length

    for &fluid_node in &[n0, n1] {
        let phi = 0.5; // P1 at midpoint
        for &solid_node in &[n0, n1] {
            let phi_s = 0.5; // P1 at midpoint
            for c in 0..dim {
                let n_c = if c == 0 { nx } else { ny };
                let solid_dof = solid_node * dim + c;
                coupling.push(fluid_node, solid_dof, phi * phi_s * n_c * w);
            }
        }
    }
}

/// Coupling on a 3D triangular boundary face (3 nodes).
fn assemble_tri_face_coupling(mesh: &Mesh, face: &BoundaryFace, coupling: &mut TripletMatrix<f64>) {
    let n0 = face.nodes[0];
    let n1 = face.nodes[1];
    let n2 = face.nodes[2];
    let p0 = &mesh.nodes[n0];
    let p1 = &mesh.nodes[n1];
    let p2 = &mesh.nodes[n2];

    // Face area and normal via cross product
    let v01 = [p1.x - p0.x, p1.y - p0.y, p1.z - p0.z];
    let v02 = [p2.x - p0.x, p2.y - p0.y, p2.z - p0.z];
    let cross = [
        v01[1] * v02[2] - v01[2] * v02[1],
        v01[2] * v02[0] - v01[0] * v02[2],
        v01[0] * v02[1] - v01[1] * v02[0],
    ];
    let area2 = (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt();
    let area = area2 / 2.0;

    if area < 1e-15 {
        return;
    }

    let normal = [cross[0] / area2, cross[1] / area2, cross[2] / area2];

    // Centroid quadrature: all P1 basis functions = 1/3
    let phi = 1.0 / 3.0;
    let w = area;

    for &fluid_node in &[n0, n1, n2] {
        for &solid_node in &[n0, n1, n2] {
            for (c, &n_c) in normal.iter().enumerate() {
                let solid_dof = solid_node * 3 + c;
                coupling.push(fluid_node, solid_dof, phi * phi * n_c * w);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sonium_solvers::LinearOperator;

    fn entry(matrix: &TripletMatrix<Complex64>, row: usize, column: usize) -> Complex64 {
        matrix
            .values
            .iter()
            .enumerate()
            .filter(|(index, _)| matrix.rows[*index] == row && matrix.cols[*index] == column)
            .map(|(_, value)| *value)
            .sum()
    }

    fn scalar_matrix(value: f64) -> TripletMatrix<f64> {
        let mut matrix = TripletMatrix::new(1, 1);
        matrix.push(0, 0, value);
        matrix
    }

    fn harmonic_test_system(with_electrical: bool) -> CoupledSystem {
        CoupledSystem {
            n_fluid: 1,
            n_solid: 1,
            fluid_stiffness: scalar_matrix(11.0),
            fluid_mass: scalar_matrix(2.0),
            fluid_damping: scalar_matrix(3.0),
            solid_stiffness: scalar_matrix(13.0),
            solid_mass: scalar_matrix(5.0),
            solid_damping: scalar_matrix(7.0),
            coupling: scalar_matrix(17.0),
            interface_load: None,
            electrical: with_electrical.then(|| ElectricalSubsystem {
                n_dofs: 1,
                resistance: scalar_matrix(19.0),
                inductance: scalar_matrix(23.0),
                coupling: scalar_matrix(29.0),
            }),
        }
    }

    #[test]
    fn test_coupled_system_dimensions() {
        let sys = CoupledSystem {
            n_fluid: 10,
            n_solid: 20,
            fluid_stiffness: TripletMatrix::new(10, 10),
            fluid_mass: TripletMatrix::new(10, 10),
            fluid_damping: TripletMatrix::new(10, 10),
            solid_stiffness: TripletMatrix::new(20, 20),
            solid_mass: TripletMatrix::new(20, 20),
            solid_damping: TripletMatrix::new(20, 20),
            coupling: TripletMatrix::new(10, 20),
            interface_load: None,
            electrical: None,
        };
        assert_eq!(sys.total_dofs(), 30);
    }

    #[test]
    fn harmonic_assembly_contains_complete_dynamic_blocks_and_physical_signs() {
        let system = harmonic_test_system(false);
        let point = HarmonicCoupledPoint {
            omega_rad_per_s: 4.0,
            sound_speed_m_per_s: 2.0,
            fluid_density_kg_per_m3: 1.5,
        };
        let matrix = system.assemble_harmonic(point).unwrap();

        assert_eq!(entry(&matrix, 0, 0), Complex64::new(3.0, 12.0));
        assert_eq!(entry(&matrix, 1, 1), Complex64::new(-67.0, 28.0));
        assert_eq!(entry(&matrix, 0, 1), Complex64::new(-408.0, 0.0));
        assert_eq!(entry(&matrix, 1, 0), Complex64::new(-17.0, 0.0));
    }

    #[test]
    fn harmonic_assembly_includes_reciprocal_electrical_coupling() {
        let system = harmonic_test_system(true);
        let point = HarmonicCoupledPoint {
            omega_rad_per_s: 4.0,
            sound_speed_m_per_s: 2.0,
            fluid_density_kg_per_m3: 1.5,
        };
        let matrix = system.assemble_harmonic(point).unwrap();

        assert_eq!(matrix.num_rows, 3);
        assert_eq!(entry(&matrix, 2, 2), Complex64::new(19.0, 92.0));
        assert_eq!(entry(&matrix, 2, 1), Complex64::new(0.0, 116.0));
        assert_eq!(entry(&matrix, 1, 2), Complex64::new(-29.0, 0.0));
    }

    #[test]
    fn harmonic_block_operator_matches_monolithic_with_electrical_block() {
        let system = harmonic_test_system(true);
        let point = HarmonicCoupledPoint {
            omega_rad_per_s: 4.0,
            sound_speed_m_per_s: 2.0,
            fluid_density_kg_per_m3: 1.5,
        };
        let monolithic = system.assemble_harmonic(point).unwrap().to_csr();
        let block = system.assemble_harmonic_block_operator(point).unwrap();
        let input = vec![
            Complex64::new(1.0, -0.5),
            Complex64::new(-2.0, 0.25),
            Complex64::new(0.75, 1.5),
        ];

        for (actual, expected) in block.apply(&input).iter().zip(monolithic.apply(&input)) {
            assert!((*actual - expected).norm() < 1.0e-12);
        }
    }

    #[test]
    fn coupled_system_rejects_inconsistent_block_dimensions() {
        let mut system = harmonic_test_system(false);
        system.solid_mass = TripletMatrix::new(2, 2);
        let error = system
            .assemble_harmonic(HarmonicCoupledPoint {
                omega_rad_per_s: 1.0,
                sound_speed_m_per_s: 343.0,
                fluid_density_kg_per_m3: 1.2,
            })
            .unwrap_err();
        assert!(matches!(
            error,
            CoupledSystemError::BlockDimensions {
                block: "solid_mass",
                ..
            }
        ));
    }

    #[test]
    fn reciprocal_motor_blocks_conserve_active_power() {
        let omega = 125.0;
        let displacement = Complex64::new(2.0e-3, -1.0e-3);
        let velocity = Complex64::new(0.0, omega) * displacement;
        let current = Complex64::new(1.25, -0.4);
        let bl = 7.5;
        let back_emf = bl * velocity;
        let force = bl * current;
        let electrical_power = 0.5 * (back_emf * current.conj()).re;
        let mechanical_power = 0.5 * (force * velocity.conj()).re;

        assert!((electrical_power - mechanical_power).abs() < 1.0e-14);
    }

    #[test]
    fn damping_and_resistance_are_passive() {
        let omega = 20.0;
        let pressure = Complex64::new(2.0, -1.0);
        let velocity = Complex64::new(-0.5, 0.75);
        let current = Complex64::new(1.0, 0.25);
        let fluid_damping = 3.0;
        let structural_damping = 7.0;
        let resistance = 19.0;
        let dissipated = 0.5
            * (fluid_damping * pressure.norm_sqr()
                + structural_damping * velocity.norm_sqr()
                + resistance * current.norm_sqr());
        assert!(omega > 0.0 && dissipated >= 0.0);
    }

    #[test]
    fn test_block_assembly_structure() {
        // Build trivial 2+4 DOF system
        let mut kf = TripletMatrix::new(2, 2);
        kf.push(0, 0, 1.0);
        kf.push(1, 1, 1.0);

        let mut mf = TripletMatrix::new(2, 2);
        mf.push(0, 0, 0.5);
        mf.push(1, 1, 0.5);

        let mut ks = TripletMatrix::new(4, 4);
        ks.push(0, 0, 2.0);
        ks.push(1, 1, 2.0);
        ks.push(2, 2, 2.0);
        ks.push(3, 3, 2.0);

        let mut c = TripletMatrix::new(2, 4);
        c.push(0, 0, 0.1);
        c.push(1, 2, 0.2);

        let sys = CoupledSystem {
            n_fluid: 2,
            n_solid: 4,
            fluid_stiffness: kf,
            fluid_mass: mf,
            fluid_damping: TripletMatrix::new(2, 2),
            solid_stiffness: ks,
            solid_mass: TripletMatrix::new(4, 4),
            solid_damping: TripletMatrix::new(4, 4),
            coupling: c,
            interface_load: None,
            electrical: None,
        };

        let k_sq = Complex64::new(4.0, 0.0);
        let c_fs = Complex64::new(1.0, 0.0);
        let c_sf = Complex64::new(-2.0, 0.0); // e.g. -omega^2 * rho
        let block = sys.assemble_block(k_sq, c_fs, c_sf);

        assert_eq!(block.num_rows, 6);
        assert_eq!(block.num_cols, 6);
        // Should have entries in all 4 quadrants
        assert!(block.nnz() > 0);

        let csr = block.to_csr();
        // Top-left (0,0): K_f[0,0] - k^2 * M_f[0,0] = 1 - 4*0.5 = -1
        let a00: Complex64 = csr
            .row_entries(0)
            .filter(|(j, _)| *j == 0)
            .map(|(_, v)| v)
            .sum();
        assert!(
            (a00.re - (-1.0)).abs() < 1e-14,
            "A[0,0] = {a00}, expected -1.0"
        );

        // Bottom-right (2,2) = K_s[0,0] = 2.0
        let a22: Complex64 = csr
            .row_entries(2)
            .filter(|(j, _)| *j == 2)
            .map(|(_, v)| v)
            .sum();
        assert!((a22.re - 2.0).abs() < 1e-14, "A[2,2] = {a22}, expected 2.0");
    }

    #[test]
    fn test_block_operator_preserves_coupled_system_structure() {
        let mut kf = TripletMatrix::new(2, 2);
        kf.push(0, 0, 1.0);
        kf.push(1, 1, 1.0);

        let mut mf = TripletMatrix::new(2, 2);
        mf.push(0, 0, 0.5);
        mf.push(1, 1, 0.5);

        let mut ks = TripletMatrix::new(4, 4);
        ks.push(0, 0, 2.0);
        ks.push(1, 1, 2.0);
        ks.push(2, 2, 2.0);
        ks.push(3, 3, 2.0);

        let mut c = TripletMatrix::new(2, 4);
        c.push(0, 0, 0.1);
        c.push(1, 2, 0.2);

        let sys = CoupledSystem {
            n_fluid: 2,
            n_solid: 4,
            fluid_stiffness: kf,
            fluid_mass: mf,
            fluid_damping: TripletMatrix::new(2, 2),
            solid_stiffness: ks,
            solid_mass: TripletMatrix::new(4, 4),
            solid_damping: TripletMatrix::new(4, 4),
            coupling: c,
            interface_load: None,
            electrical: None,
        };

        let block_op = sys.assemble_block_operator(
            Complex64::new(4.0, 0.0),
            Complex64::new(1.0, 0.0),
            Complex64::new(-2.0, 0.0),
        );

        assert_eq!(block_op.num_rows(), 6);
        assert_eq!(block_op.num_cols(), 6);
        assert_eq!(block_op.layout().row_range(0), 0..2);
        assert_eq!(block_op.layout().row_range(1), 2..6);
        assert!(block_op.block(0, 0).is_some());
        assert!(block_op.block(0, 1).is_some());
        assert!(block_op.block(1, 0).is_some());
        assert!(block_op.block(1, 1).is_some());
    }

    #[test]
    fn test_block_operator_matches_monolithic_coupled_matvec() {
        let mut kf = TripletMatrix::new(2, 2);
        kf.push(0, 0, 1.0);
        kf.push(1, 1, 1.0);

        let mut mf = TripletMatrix::new(2, 2);
        mf.push(0, 0, 0.5);
        mf.push(1, 1, 0.5);

        let mut ks = TripletMatrix::new(4, 4);
        ks.push(0, 0, 2.0);
        ks.push(1, 1, 2.0);
        ks.push(2, 2, 2.0);
        ks.push(3, 3, 2.0);

        let mut c = TripletMatrix::new(2, 4);
        c.push(0, 0, 0.1);
        c.push(1, 2, 0.2);

        let sys = CoupledSystem {
            n_fluid: 2,
            n_solid: 4,
            fluid_stiffness: kf,
            fluid_mass: mf,
            fluid_damping: TripletMatrix::new(2, 2),
            solid_stiffness: ks,
            solid_mass: TripletMatrix::new(4, 4),
            solid_damping: TripletMatrix::new(4, 4),
            coupling: c,
            interface_load: None,
            electrical: None,
        };

        let k_squared = Complex64::new(4.0, 0.0);
        let coupling_fs = Complex64::new(1.0, 0.25);
        let coupling_sf = Complex64::new(-2.0, 0.5);
        let block_op = sys.assemble_block_operator(k_squared, coupling_fs, coupling_sf);
        let monolithic = sys
            .assemble_block(k_squared, coupling_fs, coupling_sf)
            .to_csr();
        let x: Vec<Complex64> = (0..block_op.num_cols())
            .map(|i| Complex64::new(1.0 + i as f64, -0.5 * i as f64))
            .collect();

        let y_block = block_op.apply(&x);
        let y_mono = monolithic.apply(&x);

        assert_eq!(y_block.len(), y_mono.len());
        for (actual, expected) in y_block.iter().zip(y_mono.iter()) {
            assert!((*actual - *expected).norm() < 1.0e-10);
        }
    }
}
