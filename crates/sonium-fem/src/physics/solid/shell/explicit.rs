//! Central-difference explicit dynamics for T3 shells.
//!
//! Time-integrates the existing triangle shell kernels with the
//! second-order central-difference scheme on a lumped diagonal mass:
//! `a = M^-1 (F_ext - F_int(u))`, `v += dt a`, `u += dt v`. The force
//! path reuses [`ShellElement::stiffness_global`] for linear elasticity
//! and [`ElastoplasticShellForm`] (corotational CST membrane plus DKT
//! bending with J2 through-thickness points) for geometric and material
//! nonlinearity, committing plastic state every step like the crash
//! kernels. Masses reuse the lumped triangle distribution (nodal
//! `rho A t / 3` on translations, `m t^2 / 12` on rotations), matching
//! [`ShellElement::mass_global`] diagonals and the crash
//! [`T3CrashGeom::lumped_masses`](super::crash::T3CrashGeom::lumped_masses).
//!
//! The stable step comes from the membrane wave bound `Lc / c` per
//! element ([`ShellExplicitModel::wave_dt`]) or, for elastic models,
//! from power iteration on `M^-1 K`
//! ([`ShellExplicitModel::elastic_critical_dt`]), which also covers
//! rotary and drilling modes. [`DynamicRelaxation`] adds kinetic
//! damping (velocities reset at kinetic-energy peaks) so the same
//! integrator settles quasi-static loads without tuned viscosity.
//! [`ShellExplicitModel::static_solve`] is a dense verification aid
//! for cross-checking relaxed answers, not a production solver.
//!
//! Only homogeneous (pinned) boundary conditions are modeled: callers
//! mark fixed DOFs with [`ShellExplicitModel::constrain_dof`], and the
//! stepper holds their displacement while zeroing their velocity.

// Rust guideline compliant 2026-02-21

use super::elastoplastic::ElastoplasticShellForm;
use super::element::ShellElement;
use crate::forms::NonlinearElementForm;
use crate::physics::solid::plasticity::J2Material;

/// Error of explicit shell dynamics.
#[derive(Debug, Clone)]
pub struct ShellExplicitError {
    /// Human-readable failure description.
    pub message: String,
}

impl ShellExplicitError {
    /// Builds an error from a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for ShellExplicitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "explicit shell error: {}", self.message)
    }
}

impl std::error::Error for ShellExplicitError {}

/// Lumped diagonal masses for T3 shells.
///
/// Each element contributes nodal `rho A t / 3` to the three
/// translational DOFs of its nodes and `m t^2 / 12` to the three
/// rotational DOFs. The distribution is isotropic per node, so global
/// element orientation never changes the diagonal. Constrained DOFs
/// keep their mass; the stepper simply never integrates them.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::mesh::Point;
/// use sonium_fem::physics::solid::shell::ShellSection;
/// use sonium_fem::physics::solid::shell::{ShellElement, explicit::lumped_masses};
///
/// let section = ShellSection::new(200.0e9, 0.3, 0.002, 7800.0);
/// let element = ShellElement::from_points(
///     [0, 1, 2],
///     &Point::new_3d(0.0, 0.0, 0.0),
///     &Point::new_3d(1.0, 0.0, 0.0),
///     &Point::new_3d(0.0, 1.0, 0.0),
///     section,
/// );
/// let masses = lumped_masses(&[element], 18).unwrap();
/// let translations: f64 = [0, 1, 2, 6, 7, 8, 12, 13, 14].iter().map(|i| masses[*i]).sum();
/// assert!((translations - 3.0 * 7800.0 * 0.5 * 0.002).abs() < 1e-12);
/// ```
///
/// # Errors
///
/// Returns [`ShellExplicitError`] for an empty element list, a node
/// index outside `n_dofs`, or a non-positive element mass.
pub fn lumped_masses(
    elements: &[ShellElement],
    n_dofs: usize,
) -> Result<Vec<f64>, ShellExplicitError> {
    if elements.is_empty() {
        return Err(ShellExplicitError::new(
            "explicit shell needs at least one element",
        ));
    }
    let mut masses = vec![0.0; n_dofs];
    for element in elements {
        let nodal = element.section.mass_per_area() * element.area / 3.0;
        if !nodal.is_finite() || nodal <= 0.0 {
            return Err(ShellExplicitError::new(format!(
                "element has non-positive lumped mass {nodal:.6e}"
            )));
        }
        let rotary = nodal * element.section.thickness.powi(2) / 12.0;
        for node in element.nodes {
            if node * 6 + 5 >= n_dofs {
                return Err(ShellExplicitError::new(format!(
                    "node {node} exceeds {n_dofs} DOFs"
                )));
            }
            for component in 0..3 {
                masses[node * 6 + component] += nodal;
                masses[node * 6 + 3 + component] += rotary;
            }
        }
    }
    Ok(masses)
}

/// Characteristic length of a T3 element in m.
///
/// `Lc = A / L_max` with the longest edge from the local coordinates,
/// the same definition the crash kernels use for the `Lc / c` step.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::mesh::Point;
/// use sonium_fem::physics::solid::shell::ShellSection;
/// use sonium_fem::physics::solid::shell::{ShellElement, explicit::tri_char_length};
///
/// let section = ShellSection::new(200.0e9, 0.3, 0.002, 7800.0);
/// let element = ShellElement::from_points(
///     [0, 1, 2],
///     &Point::new_3d(0.0, 0.0, 0.0),
///     &Point::new_3d(1.0, 0.0, 0.0),
///     &Point::new_3d(0.0, 1.0, 0.0),
///     section,
/// );
/// assert!((tri_char_length(&element) - 0.5 / 2.0f64.sqrt()).abs() < 1e-14);
/// ```
#[must_use]
pub fn tri_char_length(element: &ShellElement) -> f64 {
    let mut longest: f64 = 0.0;
    for i in 0..3 {
        let a = element.local_coords[i];
        let b = element.local_coords[(i + 1) % 3];
        let edge = ((b.0 - a.0).powi(2) + (b.1 - a.1).powi(2)).sqrt();
        longest = longest.max(edge);
    }
    element.area / longest
}

/// Force kernel behind a shell explicit model.
enum ForceKernel {
    /// Linear elasticity with cached element stiffnesses.
    Elastic {
        /// Global stiffness per element (18x18 row-major).
        stiffness: Vec<Vec<f64>>,
        /// Global DOFs per element.
        dofs: Vec<[usize; 18]>,
    },
    /// Corotational J2 elastoplasticity with committed state.
    Elastoplastic(ElastoplasticShellForm),
}

impl std::fmt::Debug for ForceKernel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Elastic { dofs, .. } => f
                .debug_struct("Elastic")
                .field("elements", &dofs.len())
                .finish(),
            Self::Elastoplastic(form) => f
                .debug_struct("Elastoplastic")
                .field("elements", &form.num_elements())
                .finish(),
        }
    }
}

/// Central-difference explicit model for T3 shells.
///
/// Owns the force kernel, the lumped diagonal mass, and the pinned
/// DOF set. Elastic models cache element stiffnesses; elastoplastic
/// models carry committed J2 state advanced by every force evaluation.
#[derive(Debug)]
pub struct ShellExplicitModel {
    force: ForceKernel,
    masses: Vec<f64>,
    constrained: Vec<bool>,
    n_dofs: usize,
    wave_dt: f64,
}

impl ShellExplicitModel {
    /// Builds an elastic model with cached stiffnesses.
    ///
    /// # Errors
    ///
    /// Returns [`ShellExplicitError`] for an empty element list, a
    /// node index outside `n_dofs`, a non-positive element mass, or a
    /// degenerate element length or wave speed.
    pub fn elastic(elements: Vec<ShellElement>, n_dofs: usize) -> Result<Self, ShellExplicitError> {
        let masses = lumped_masses(&elements, n_dofs)?;
        let wave_dt = wave_step(&elements)?;
        let dofs = elements.iter().map(ShellElement::global_dofs).collect();
        let stiffness = elements
            .iter()
            .map(ShellElement::stiffness_global)
            .collect();
        Ok(Self {
            force: ForceKernel::Elastic { stiffness, dofs },
            masses,
            constrained: vec![false; n_dofs],
            n_dofs,
            wave_dt,
        })
    }

    /// Builds a corotational J2 elastoplastic model.
    ///
    /// Every [`internal_forces`](Self::internal_forces) call evaluates
    /// the element residuals at `u` and then commits the plastic state
    /// there, which is the explicit constitutive update.
    ///
    /// # Errors
    ///
    /// Returns [`ShellExplicitError`] under the same conditions as
    /// [`elastic`](Self::elastic).
    pub fn elastoplastic(
        elements: Vec<ShellElement>,
        n_dofs: usize,
        material: J2Material,
    ) -> Result<Self, ShellExplicitError> {
        let masses = lumped_masses(&elements, n_dofs)?;
        let wave_dt = wave_step(&elements)?;
        Ok(Self {
            force: ForceKernel::Elastoplastic(ElastoplasticShellForm::new(
                elements, n_dofs, material,
            )),
            masses,
            constrained: vec![false; n_dofs],
            n_dofs,
            wave_dt,
        })
    }

    /// Total number of global DOFs.
    #[must_use]
    pub fn n_dofs(&self) -> usize {
        self.n_dofs
    }

    /// Lumped diagonal masses.
    #[must_use]
    pub fn masses(&self) -> &[f64] {
        &self.masses
    }

    /// Reports whether a DOF is pinned.
    ///
    /// # Panics
    ///
    /// Panics for a DOF outside the model.
    #[must_use]
    pub fn is_constrained(&self, dof: usize) -> bool {
        self.constrained[dof]
    }

    /// Pins one DOF at its current displacement.
    ///
    /// # Panics
    ///
    /// Panics for a DOF outside the model.
    pub fn constrain_dof(&mut self, dof: usize) {
        self.constrained[dof] = true;
    }

    /// Pins all six DOFs of a node.
    ///
    /// # Panics
    ///
    /// Panics for a node outside the model.
    pub fn fix_node(&mut self, node: usize) {
        for component in 0..6 {
            self.constrain_dof(node * 6 + component);
        }
    }

    /// Membrane wave step `min(Lc / c)` over elements in s.
    ///
    /// This is the raw CFL bound without a safety margin; callers
    /// scale it (0.5-0.9 is customary) or use
    /// [`elastic_critical_dt`](Self::elastic_critical_dt) for elastic
    /// models, which additionally covers rotary and drilling modes.
    #[must_use]
    pub fn wave_dt(&self) -> f64 {
        self.wave_dt
    }

    /// Critical step `2 / omega_max` of an elastic model in s.
    ///
    /// Estimates the top eigenfrequency of `M^-1 K` on the free DOFs
    /// by deterministic power iteration (a white-noise-seeded start,
    /// Rayleigh quotient checked every 10 passes), without any safety
    /// margin. Plastic tangent only softens below the elastic
    /// stiffness, so the elastic bound stays conservative after yield.
    ///
    /// # Errors
    ///
    /// Returns [`ShellExplicitError`] for elastoplastic models (use
    /// [`wave_dt`](Self::wave_dt) there), a zero iteration cap, or a
    /// non-positive estimate.
    pub fn elastic_critical_dt(&self, iterations: usize) -> Result<f64, ShellExplicitError> {
        if iterations == 0 {
            return Err(ShellExplicitError::new(
                "power iteration needs a positive cap",
            ));
        }
        let ForceKernel::Elastic { .. } = &self.force else {
            return Err(ShellExplicitError::new(
                "power iteration needs the elastic kernel; elastoplastic models use wave_dt",
            ));
        };
        // Deterministic white start on the free DOFs: plate top modes
        // are oscillatory, and a smooth start carries almost none of
        // them, stalling convergence for thousands of passes. The
        // constants are Knuth's MMIX linear generator; any fixed seed
        // works, and the fixed stream keeps the estimate repeatable.
        let mut seed: u64 = 0x243F_6A88_85A3_08D3;
        let mut z: Vec<f64> = Vec::with_capacity(self.n_dofs);
        for i in 0..self.n_dofs {
            if self.constrained[i] {
                z.push(0.0);
                continue;
            }
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            z.push((seed >> 33) as f64 / 1073741824.0 - 1.0);
        }
        normalize_free(&self.masses, &self.constrained, &mut z);
        // Rayleigh tolerance 1e-6 pins dt to 5e-7 relative, far inside
        // any caller margin; the quotient approaches from below, so an
        // early stop stays on the conservative side.
        let mut lambda_prev = 0.0;
        for pass in 0..iterations {
            let kz = self.elastic_matvec(&z);
            for i in 0..self.n_dofs {
                z[i] = if self.constrained[i] {
                    0.0
                } else {
                    kz[i] / self.masses[i]
                };
            }
            normalize_free(&self.masses, &self.constrained, &mut z);
            if pass % 10 == 9 {
                let lambda = rayleigh_quotient(&self.elastic_matvec(&z), &z, &self.masses);
                if (lambda - lambda_prev).abs() < 1e-6 * lambda.abs().max(1.0e-300) {
                    break;
                }
                lambda_prev = lambda;
            }
        }
        let lambda = rayleigh_quotient(&self.elastic_matvec(&z), &z, &self.masses);
        if !lambda.is_finite() || lambda <= 0.0 {
            return Err(ShellExplicitError::new(format!(
                "power iteration gave non-positive eigenvalue {lambda:.6e}"
            )));
        }
        Ok(2.0 / lambda.sqrt())
    }

    /// Internal forces `F_int(u)` over the global DOFs.
    ///
    /// Elastic models return `K u`; elastoplastic models return the
    /// corotational residuals and commit the J2 state at `u`.
    ///
    /// # Errors
    ///
    /// Returns [`ShellExplicitError`] for a displacement of the wrong
    /// length or with non-finite entries.
    pub fn internal_forces(&mut self, u: &[f64]) -> Result<Vec<f64>, ShellExplicitError> {
        check_vector(u, self.n_dofs, "displacement")?;
        match &mut self.force {
            ForceKernel::Elastic { .. } => Ok(self.elastic_matvec(u)),
            ForceKernel::Elastoplastic(form) => {
                let mut forces = vec![0.0; self.n_dofs];
                for elem in 0..form.num_elements() {
                    let residual = form.element_residual(elem, u);
                    for (local, global) in form.local_to_global(elem).iter().enumerate() {
                        forces[*global] += residual[local];
                    }
                }
                form.commit_state(u);
                Ok(forces)
            }
        }
    }

    /// Force residual `F_ext - F_int(u)` over the global DOFs.
    ///
    /// # Errors
    ///
    /// Returns [`ShellExplicitError`] for inputs of the wrong length
    /// or with non-finite entries.
    pub fn force_residual(
        &mut self,
        u: &[f64],
        f_ext: &[f64],
    ) -> Result<Vec<f64>, ShellExplicitError> {
        check_vector(f_ext, self.n_dofs, "external force")?;
        let internal = self.internal_forces(u)?;
        Ok(internal
            .iter()
            .zip(f_ext.iter())
            .map(|(f_int, f)| f - f_int)
            .collect())
    }

    /// Advances one central-difference step.
    ///
    /// `v += dt M^-1 (F_ext - F_int(u))` then `u += dt v` on the free
    /// DOFs; pinned DOFs keep their displacement with zero velocity.
    ///
    /// # Errors
    ///
    /// Returns [`ShellExplicitError`] for a non-positive step, inputs
    /// of the wrong length, or non-finite input entries.
    pub fn step(
        &mut self,
        state: &mut ShellExplicitState,
        f_ext: &[f64],
        dt: f64,
    ) -> Result<(), ShellExplicitError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ShellExplicitError::new(format!(
                "explicit step dt must be positive finite, got {dt:.6e}"
            )));
        }
        if state.u.len() != self.n_dofs || state.v.len() != self.n_dofs {
            return Err(ShellExplicitError::new("state vectors must match n_dofs"));
        }
        check_vector(f_ext, self.n_dofs, "external force")?;
        let internal = self.internal_forces(&state.u.clone())?;
        for i in 0..self.n_dofs {
            if self.constrained[i] {
                state.v[i] = 0.0;
                continue;
            }
            let acceleration = (f_ext[i] - internal[i]) / self.masses[i];
            state.v[i] += dt * acceleration;
        }
        for i in 0..self.n_dofs {
            if !self.constrained[i] {
                state.u[i] += dt * state.v[i];
            }
        }
        state.time += dt;
        Ok(())
    }

    /// Kinetic energy `v^T M v / 2` in J.
    ///
    /// # Panics
    ///
    /// Panics for a velocity of the wrong length.
    #[must_use]
    pub fn kinetic_energy(&self, v: &[f64]) -> f64 {
        assert_eq!(v.len(), self.n_dofs, "velocity must match n_dofs");
        0.5 * v
            .iter()
            .zip(self.masses.iter())
            .map(|(v, m)| m * v * v)
            .sum::<f64>()
    }

    /// Elastic strain energy `u^T K u / 2` in J.
    ///
    /// # Errors
    ///
    /// Returns [`ShellExplicitError`] for elastoplastic models or a
    /// displacement of the wrong length.
    pub fn elastic_strain_energy(&self, u: &[f64]) -> Result<f64, ShellExplicitError> {
        if !matches!(self.force, ForceKernel::Elastic { .. }) {
            return Err(ShellExplicitError::new(
                "strain energy needs the elastic kernel",
            ));
        }
        check_vector_len(u, self.n_dofs, "displacement")?;
        let ku = self.elastic_matvec(u);
        Ok(0.5 * u.iter().zip(ku.iter()).map(|(u, ku)| u * ku).sum::<f64>())
    }

    /// Dense static solve on the free DOFs.
    ///
    /// Assembles the free-free stiffness and runs Gaussian elimination
    /// with partial pivoting; pinned DOFs solve to zero. This is a
    /// verification aid for cross-checking relaxed answers, sized for
    /// leg meshes (hundreds of free DOFs), not a production solver.
    ///
    /// # Errors
    ///
    /// Returns [`ShellExplicitError`] for elastoplastic models, a load
    /// of the wrong length, or a (numerically) singular stiffness.
    pub fn static_solve(&self, f_ext: &[f64]) -> Result<Vec<f64>, ShellExplicitError> {
        let ForceKernel::Elastic { stiffness, dofs } = &self.force else {
            return Err(ShellExplicitError::new(
                "static solve needs the elastic kernel",
            ));
        };
        check_vector(f_ext, self.n_dofs, "external force")?;
        let mut free_of = vec![usize::MAX; self.n_dofs];
        let mut free = Vec::new();
        for i in 0..self.n_dofs {
            if !self.constrained[i] {
                free_of[i] = free.len();
                free.push(i);
            }
        }
        let n = free.len();
        let mut matrix = vec![0.0; n * n];
        let mut rhs = vec![0.0; n];
        for dof in &free {
            rhs[free_of[*dof]] = f_ext[*dof];
        }
        for (ke, de) in stiffness.iter().zip(dofs.iter()) {
            for (li, gi) in de.iter().enumerate() {
                if self.constrained[*gi] {
                    continue;
                }
                for (lj, gj) in de.iter().enumerate() {
                    if self.constrained[*gj] {
                        continue;
                    }
                    matrix[free_of[*gi] * n + free_of[*gj]] += ke[li * 18 + lj];
                }
            }
        }
        let solution = gauss_solve(&mut matrix, &mut rhs, n)?;
        let mut u = vec![0.0; self.n_dofs];
        for (k, dof) in free.iter().enumerate() {
            u[*dof] = solution[k];
        }
        Ok(u)
    }

    /// Applies the cached elastic stiffness to a vector.
    fn elastic_matvec(&self, u: &[f64]) -> Vec<f64> {
        let ForceKernel::Elastic { stiffness, dofs } = &self.force else {
            unreachable!("elastic matvec needs the elastic kernel");
        };
        let mut out = vec![0.0; self.n_dofs];
        for (ke, de) in stiffness.iter().zip(dofs.iter()) {
            for (li, gi) in de.iter().enumerate() {
                let mut row = 0.0;
                for (lj, gj) in de.iter().enumerate() {
                    row += ke[li * 18 + lj] * u[*gj];
                }
                out[*gi] += row;
            }
        }
        out
    }
}

/// Minimum `Lc / c` over elements in s.
fn wave_step(elements: &[ShellElement]) -> Result<f64, ShellExplicitError> {
    let mut step = f64::INFINITY;
    for element in elements {
        let length = tri_char_length(element);
        let speed = element.section.plate_sound_speed();
        let dt = length / speed;
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ShellExplicitError::new(format!(
                "element has non-positive wave step {dt:.6e}"
            )));
        }
        step = step.min(dt);
    }
    Ok(step)
}

/// Rayleigh quotient `z^T Kz / z^T Mz` of a power iterate.
fn rayleigh_quotient(kz: &[f64], z: &[f64], masses: &[f64]) -> f64 {
    let mut num = 0.0;
    let mut den = 0.0;
    for ((kz, z), m) in kz.iter().zip(z.iter()).zip(masses.iter()) {
        num += z * kz;
        den += m * z * z;
    }
    num / den
}

/// Normalizes a vector in the mass norm over free DOFs.
fn normalize_free(masses: &[f64], constrained: &[bool], z: &mut [f64]) {
    let norm = z
        .iter()
        .zip(masses.iter())
        .zip(constrained.iter())
        .map(|((z, m), c)| if *c { 0.0 } else { m * z * z })
        .sum::<f64>()
        .sqrt();
    if norm > 0.0 {
        for value in z.iter_mut() {
            *value /= norm;
        }
    }
}

/// Validates a global vector for length and finiteness.
fn check_vector(vector: &[f64], n_dofs: usize, name: &str) -> Result<(), ShellExplicitError> {
    check_vector_len(vector, n_dofs, name)?;
    if !vector.iter().all(|x| x.is_finite()) {
        return Err(ShellExplicitError::new(format!("{name} must be finite")));
    }
    Ok(())
}

/// Validates a global vector for length only.
fn check_vector_len(vector: &[f64], n_dofs: usize, name: &str) -> Result<(), ShellExplicitError> {
    if vector.len() != n_dofs {
        return Err(ShellExplicitError::new(format!(
            "{name} has length {}, need {n_dofs}",
            vector.len()
        )));
    }
    Ok(())
}

/// Gaussian elimination with partial pivoting, in place.
fn gauss_solve(
    matrix: &mut [f64],
    rhs: &mut [f64],
    n: usize,
) -> Result<Vec<f64>, ShellExplicitError> {
    for col in 0..n {
        let mut pivot = col;
        let mut largest = matrix[col * n + col].abs();
        for row in (col + 1)..n {
            let candidate = matrix[row * n + col].abs();
            if candidate > largest {
                largest = candidate;
                pivot = row;
            }
        }
        if !largest.is_finite() || largest <= 1e-14 {
            return Err(ShellExplicitError::new(format!(
                "stiffness is singular at column {col} (pivot {largest:.3e})"
            )));
        }
        if pivot != col {
            for k in col..n {
                matrix.swap(col * n + k, pivot * n + k);
            }
            rhs.swap(col, pivot);
        }
        let diagonal = matrix[col * n + col];
        for row in (col + 1)..n {
            let factor = matrix[row * n + col] / diagonal;
            matrix[row * n + col] = 0.0;
            for k in (col + 1)..n {
                matrix[row * n + k] -= factor * matrix[col * n + k];
            }
            rhs[row] -= factor * rhs[col];
        }
    }
    let mut solution = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = rhs[row];
        for k in (row + 1)..n {
            sum -= matrix[row * n + k] * solution[k];
        }
        solution[row] = sum / matrix[row * n + row];
    }
    Ok(solution)
}

/// Displacement/velocity state of an explicit shell run.
#[derive(Debug, Clone)]
pub struct ShellExplicitState {
    /// Global displacements in m and rad.
    pub u: Vec<f64>,
    /// Global velocities in m/s and rad/s.
    pub v: Vec<f64>,
    /// Simulated time in s.
    pub time: f64,
}

impl ShellExplicitState {
    /// Builds a zero state for `n_dofs` DOFs.
    #[must_use]
    pub fn new(n_dofs: usize) -> Self {
        Self {
            u: vec![0.0; n_dofs],
            v: vec![0.0; n_dofs],
            time: 0.0,
        }
    }
}

/// Outcome of a dynamic-relaxation run.
#[derive(Debug, Clone)]
pub struct RelaxationOut {
    /// Settled global displacements in m and rad.
    pub displacement: Vec<f64>,
    /// Steps taken.
    pub steps: usize,
    /// Final free-DOF residual norm in N (N m on rotations).
    pub residual_norm: f64,
    /// Free-DOF load norm in N (N m on rotations).
    pub load_norm: f64,
    /// Whether the residual met the tolerance.
    pub converged: bool,
}

/// Dynamic relaxation with kinetic damping.
///
/// Runs central-difference steps under a constant load and resets all
/// velocities whenever kinetic energy peaks (the structure passed
/// through static equilibrium), which settles quasi-static answers
/// without a tuned viscosity. Convergence checks the free-DOF force
/// residual `|F_ext - F_int| / |F_ext|` every `check_every` steps.
///
/// Resets key on kinetic energy smoothed over 8 steps (about two
/// top-mode periods near the stability limit): raw step-to-step
/// comparisons let top-mode ripple trigger a reset every other step,
/// and each reset re-baselines near the oscillation peak, ratcheting
/// energy upward instead of settling. Even so, relaxation wants
/// margin below the free-vibration bound: use at most half the
/// critical step (top-mode overshoot per restart stays small there),
/// not the 0.8-0.9 customary for pure time marching.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::mesh::Point;
/// use sonium_fem::physics::solid::shell::ShellSection;
/// use sonium_fem::physics::solid::shell::{
///     ShellElement, assemble_shell_load,
///     explicit::{DynamicRelaxation, ShellExplicitModel},
/// };
///
/// let section = ShellSection::new(200.0e9, 0.3, 0.01, 7800.0);
/// let corner = |x: f64, y: f64| Point::new_3d(x, y, 0.0);
/// let elements = vec![
///     ShellElement::from_points([0, 1, 2], &corner(0.0, 0.0), &corner(1.0, 0.0), &corner(1.0, 1.0), section),
///     ShellElement::from_points([0, 2, 3], &corner(0.0, 0.0), &corner(1.0, 1.0), &corner(0.0, 1.0), section),
/// ];
/// let mut model = ShellExplicitModel::elastic(elements.clone(), 24).unwrap();
/// for node in 0..4 {
///     if node != 2 {
///         model.fix_node(node);
///     }
/// }
/// let load = assemble_shell_load(&elements, 24, |_| -100.0);
/// let driver = DynamicRelaxation::new(0.5 * model.wave_dt(), 1e-6, 20_000).unwrap();
/// let out = driver.relax(&mut model, &load).unwrap();
/// assert!(out.converged);
/// ```
#[derive(Debug, Clone)]
pub struct DynamicRelaxation {
    /// Fixed step in s (must sit below the critical step).
    pub dt: f64,
    /// Relative residual tolerance for convergence.
    pub tolerance: f64,
    /// Step cap; hitting it returns `converged: false`.
    pub max_steps: usize,
    /// Residual-check cadence in steps.
    pub check_every: usize,
}

impl DynamicRelaxation {
    /// Builds a driver with checks every 10 steps.
    ///
    /// # Errors
    ///
    /// Returns [`ShellExplicitError`] for a non-positive step,
    /// tolerance, or step cap.
    pub fn new(dt: f64, tolerance: f64, max_steps: usize) -> Result<Self, ShellExplicitError> {
        if !dt.is_finite() || dt <= 0.0 {
            return Err(ShellExplicitError::new(
                "relaxation dt must be positive finite",
            ));
        }
        if !tolerance.is_finite() || tolerance <= 0.0 {
            return Err(ShellExplicitError::new(
                "relaxation tolerance must be positive finite",
            ));
        }
        if max_steps == 0 {
            return Err(ShellExplicitError::new(
                "relaxation needs a positive step cap",
            ));
        }
        Ok(Self {
            dt,
            tolerance,
            max_steps,
            check_every: 10,
        })
    }

    /// Relaxes `model` from rest under a constant load.
    ///
    /// # Errors
    ///
    /// Returns [`ShellExplicitError`] for a load of the wrong length,
    /// a zero load norm (no equilibrium to settle), non-finite state,
    /// or a stepper failure.
    pub fn relax(
        &self,
        model: &mut ShellExplicitModel,
        f_ext: &[f64],
    ) -> Result<RelaxationOut, ShellExplicitError> {
        check_vector(f_ext, model.n_dofs, "external force")?;
        let load_norm = free_norm(f_ext, &model.constrained);
        if !load_norm.is_finite() || load_norm <= 0.0 {
            return Err(ShellExplicitError::new(
                "relaxation needs a load with positive free-DOF norm",
            ));
        }
        let mut state = ShellExplicitState::new(model.n_dofs);
        // Smoothed kinetic energy over a short window: comparing raw
        // step values lets top-mode ripple (period ~4 steps near the
        // limit) trigger reset avalanches that pump energy upward.
        // Eight steps span about two top-mode periods there, so the
        // ripple averages out while low-mode trends (hundreds of
        // steps) pass through. The ring starts filled with the first
        // sample to avoid a startup transient.
        const SMOOTH_WINDOW: usize = 8;
        let mut ring = [f64::NAN; SMOOTH_WINDOW];
        let mut ring_next = 0usize;
        let mut smoothed_prev = 0.0;
        let mut residual_norm: f64;
        // A zero cadence (public field) falls back to every step
        // instead of panicking on `step % 0`.
        let cadence = self.check_every.max(1);
        for step in 1..=self.max_steps {
            model.step(&mut state, f_ext, self.dt)?;
            if !state.u.iter().chain(state.v.iter()).all(|x| x.is_finite()) {
                return Err(ShellExplicitError::new(format!(
                    "relaxation went non-finite at step {step} (dt above the critical step?)"
                )));
            }
            // Kinetic damping: a falling (smoothed) kinetic energy
            // means the structure just passed equilibrium, so restart
            // from rest.
            let kinetic = model.kinetic_energy(&state.v);
            if ring[0].is_nan() {
                ring = [kinetic; SMOOTH_WINDOW];
            } else {
                ring[ring_next] = kinetic;
                ring_next = (ring_next + 1) % SMOOTH_WINDOW;
            }
            let smoothed = ring.iter().sum::<f64>() / SMOOTH_WINDOW as f64;
            if smoothed < smoothed_prev {
                state.v.fill(0.0);
                ring = [0.0; SMOOTH_WINDOW];
                smoothed_prev = 0.0;
            } else {
                smoothed_prev = smoothed;
            }
            if step % cadence == 0 {
                residual_norm =
                    free_norm(&model.force_residual(&state.u, f_ext)?, &model.constrained);
                if residual_norm / load_norm < self.tolerance {
                    return Ok(RelaxationOut {
                        displacement: state.u,
                        steps: step,
                        residual_norm,
                        load_norm,
                        converged: true,
                    });
                }
            }
        }
        residual_norm = free_norm(&model.force_residual(&state.u, f_ext)?, &model.constrained);
        Ok(RelaxationOut {
            displacement: state.u,
            steps: self.max_steps,
            residual_norm,
            load_norm,
            converged: residual_norm / load_norm < self.tolerance,
        })
    }
}

/// Euclidean norm over the free DOFs.
fn free_norm(vector: &[f64], constrained: &[bool]) -> f64 {
    vector
        .iter()
        .zip(constrained.iter())
        .map(|(x, c)| if *c { 0.0 } else { x * x })
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mesh::Point;
    use crate::physics::solid::plasticity::HardeningLaw;
    use crate::physics::solid::shell::{ShellSection, assemble_shell_load};

    fn steel() -> ShellSection {
        ShellSection::new(200.0e9, 0.3, 0.01, 7800.0)
    }

    fn unit_triangle(section: ShellSection) -> ShellElement {
        ShellElement::from_points(
            [0, 1, 2],
            &Point::new_3d(0.0, 0.0, 0.0),
            &Point::new_3d(1.0, 0.0, 0.0),
            &Point::new_3d(0.0, 1.0, 0.0),
            section,
        )
    }

    /// Two-triangle unit square: nodes 0..3, triangles (0,1,2)+(0,2,3).
    fn square_pair(section: ShellSection) -> (Vec<ShellElement>, usize) {
        let corner = |x: f64, y: f64| Point::new_3d(x, y, 0.0);
        let elements = vec![
            ShellElement::from_points(
                [0, 1, 2],
                &corner(0.0, 0.0),
                &corner(1.0, 0.0),
                &corner(1.0, 1.0),
                section,
            ),
            ShellElement::from_points(
                [0, 2, 3],
                &corner(0.0, 0.0),
                &corner(1.0, 1.0),
                &corner(0.0, 1.0),
                section,
            ),
        ];
        (elements, 24)
    }

    #[test]
    fn lumped_mass_conserves_total() {
        let element = unit_triangle(steel());
        let masses = lumped_masses(&[element], 18).unwrap();
        let translations: f64 = [0, 1, 2, 6, 7, 8, 12, 13, 14]
            .iter()
            .map(|i| masses[*i])
            .sum();
        assert!((translations - 3.0 * 7800.0 * 0.5 * 0.01).abs() < 1e-9);
        // Rotary: nodal m t^2 / 12 on each of the 9 rotational slots.
        let nodal = 7800.0 * 0.5 * 0.01 / 3.0;
        let rotary: f64 = [3, 4, 5, 9, 10, 11, 15, 16, 17]
            .iter()
            .map(|i| masses[*i])
            .sum();
        assert!((rotary - 9.0 * nodal * 0.01f64.powi(2) / 12.0).abs() < 1e-18);
        assert!(lumped_masses(&[], 18).is_err());
    }

    #[test]
    fn rigid_translation_gives_zero_forces() {
        let (elements, n_dofs) = square_pair(steel());
        let mut model = ShellExplicitModel::elastic(elements, n_dofs).unwrap();
        // Uniform translation is strain-free for CST/DKT: forces vanish
        // relative to the stiffness scale.
        let mut u = vec![0.0; n_dofs];
        for node in 0..4 {
            u[node * 6] = 1.0;
            u[node * 6 + 1] = 2.0;
            u[node * 6 + 2] = 3.0;
        }
        let forces = model.internal_forces(&u).unwrap();
        let norm: f64 = forces.iter().map(|f| f * f).sum::<f64>().sqrt();
        assert!(norm < 1e-3, "rigid forces {norm:.3e} exceed noise");
    }

    #[test]
    fn wave_dt_matches_closed_form() {
        let element = unit_triangle(steel());
        let expected = tri_char_length(&element) / steel().plate_sound_speed();
        let model = ShellExplicitModel::elastic(vec![element], 18).unwrap();
        assert!((model.wave_dt() - expected).abs() < 1e-18);
        assert!(model.wave_dt() > 0.0);
    }

    #[test]
    fn power_bound_brackets_stability_limit() {
        let (elements, n_dofs) = square_pair(steel());
        let mut model = ShellExplicitModel::elastic(elements, n_dofs).unwrap();
        for node in [0, 1, 3] {
            model.fix_node(node);
        }
        let critical = model.elastic_critical_dt(200).unwrap();
        assert!(critical > 0.0);
        let f_ext = vec![0.0; n_dofs];
        // Below the bound the undamped pluck stays bounded: symplectic
        // energy oscillates (large swings are normal near the limit)
        // but never blows up over 20_000 steps.
        let mut state = ShellExplicitState::new(n_dofs);
        state.u[2 * 6 + 2] = 0.01;
        let energy0 = model.elastic_strain_energy(&state.u).unwrap();
        for _ in 0..20_000 {
            model.step(&mut state, &f_ext, 0.8 * critical).unwrap();
        }
        let energy1 =
            model.kinetic_energy(&state.v) + model.elastic_strain_energy(&state.u).unwrap();
        assert!(
            energy1 < 10.0 * energy0,
            "run below the bound blew up: {energy1:.3e} vs {energy0:.3e}"
        );
        // Above the bound the top mode is linearly unstable: an
        // all-DOF pluck must blow up fast, proving the bound is not
        // wildly conservative. The stepper's finiteness guard may end
        // the run with an error, which counts as blowup.
        let mut hot = ShellExplicitState::new(n_dofs);
        for dof in 12..18 {
            hot.u[dof] = 0.01;
        }
        let hot0 = model.elastic_strain_energy(&hot.u).unwrap();
        let mut blew_up = false;
        for _ in 0..3000 {
            if model.step(&mut hot, &f_ext, 1.5 * critical).is_err() {
                blew_up = true;
                break;
            }
            let energy = model.kinetic_energy(&hot.v)
                + model.elastic_strain_energy(&hot.u).unwrap_or(f64::INFINITY);
            if !energy.is_finite() || energy > 1.0e6 * hot0 {
                blew_up = true;
                break;
            }
        }
        assert!(blew_up, "run 50% above the bound stayed bounded");
    }

    #[test]
    fn step_pins_constrained_dofs() {
        let (elements, n_dofs) = square_pair(steel());
        let mut model = ShellExplicitModel::elastic(elements, n_dofs).unwrap();
        model.fix_node(0);
        model.fix_node(1);
        let mut state = ShellExplicitState::new(n_dofs);
        let f_ext = vec![1.0; n_dofs];
        for _ in 0..50 {
            model
                .step(&mut state, &f_ext, 0.1 * model.wave_dt())
                .unwrap();
        }
        for dof in 0..12 {
            assert_eq!(state.u[dof], 0.0);
            assert_eq!(state.v[dof], 0.0);
        }
        assert!(state.u[12..].iter().any(|u| u.abs() > 0.0));
        assert!(model.step(&mut state, &f_ext, 0.0).is_err());
        assert!(model.step(&mut state, &[0.0; 3], 1e-6).is_err());
    }

    #[test]
    fn relaxation_recovers_static_answer() {
        let (elements, n_dofs) = square_pair(steel());
        let mut model = ShellExplicitModel::elastic(elements.clone(), n_dofs).unwrap();
        for node in [0, 1, 3] {
            model.fix_node(node);
        }
        let load = assemble_shell_load(&elements, n_dofs, |_| -100.0);
        let driver = DynamicRelaxation::new(0.5 * model.wave_dt(), 1e-6, 20_000).unwrap();
        let out = driver.relax(&mut model, &load).unwrap();
        assert!(
            out.converged,
            "residual {:.3e} after {} steps",
            out.residual_norm, out.steps
        );
        let answer = model.static_solve(&load).unwrap();
        let diff: f64 = out
            .displacement
            .iter()
            .zip(answer.iter())
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f64>()
            .sqrt();
        let scale: f64 = answer.iter().map(|a| a * a).sum::<f64>().sqrt();
        assert!(
            diff / scale < 0.01,
            "relax/static mismatch {:.3e}",
            diff / scale
        );
        assert!(model.static_solve(&[0.0; 3]).is_err());
    }

    #[test]
    fn elastoplastic_step_yields_and_advances() {
        let (elements, n_dofs) = square_pair(steel());
        // Very low yield: any appreciable load plastifies.
        let soft = J2Material::new(200.0e9, 0.3, 1.0e5, HardeningLaw::Perfect);
        let mut model = ShellExplicitModel::elastoplastic(elements, n_dofs, soft).unwrap();
        for node in [0, 1, 3] {
            model.fix_node(node);
        }
        // A large transverse push on the free node yields outer fibers.
        let mut u = vec![0.0; n_dofs];
        u[2 * 6 + 2] = 0.05;
        let forces = model.internal_forces(&u).unwrap();
        assert!(forces.iter().all(|f| f.is_finite()));
        let ForceKernel::Elastoplastic(form) = &model.force else {
            panic!("elastoplastic model lost its kernel");
        };
        assert!(form.yielding_count() > 0);
        assert!(form.max_equiv_plastic_strain() > 0.0);
        // Elastic-only APIs refuse the plastic model loudly.
        assert!(model.elastic_critical_dt(10).is_err());
        assert!(model.elastic_strain_energy(&u).is_err());
        assert!(model.static_solve(&vec![0.0; n_dofs]).is_err());
        // The stepper runs on the plastic path without NaN.
        let mut state = ShellExplicitState::new(n_dofs);
        let f_ext = vec![0.0; n_dofs];
        model
            .step(&mut state, &f_ext, 0.1 * model.wave_dt())
            .unwrap();
        assert!(state.u.iter().chain(state.v.iter()).all(|x| x.is_finite()));
    }

    #[test]
    fn bad_inputs_fail_loudly() {
        let (elements, n_dofs) = square_pair(steel());
        assert!(ShellExplicitModel::elastic(Vec::new(), n_dofs).is_err());
        // Node 99 overruns the 24-DOF model.
        let bad = ShellElement::from_points(
            [0, 1, 99],
            &Point::new_3d(0.0, 0.0, 0.0),
            &Point::new_3d(1.0, 0.0, 0.0),
            &Point::new_3d(0.0, 1.0, 0.0),
            steel(),
        );
        assert!(ShellExplicitModel::elastic(vec![bad], n_dofs).is_err());
        let mut model = ShellExplicitModel::elastic(elements, n_dofs).unwrap();
        assert!(model.internal_forces(&[0.0; 3]).is_err());
        assert!(model.internal_forces(&vec![f64::NAN; n_dofs]).is_err());
        assert!(DynamicRelaxation::new(0.0, 1e-6, 100).is_err());
        assert!(DynamicRelaxation::new(1e-6, 1e-6, 0).is_err());
        let driver = DynamicRelaxation::new(1e-6, 1e-6, 100).unwrap();
        assert!(driver.relax(&mut model, &vec![0.0; n_dofs]).is_err());
    }
}
