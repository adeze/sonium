//! Central-difference explicit dynamics engine.
//!
//! Integrates crash and impact transients with the conditionally stable
//! central-difference scheme on a lumped (diagonal) mass matrix: per-step
//! cost is one force evaluation plus vector updates, with no linear
//! solves. Covers the CFL step estimate from the dilatational wave speed,
//! density-based mass scaling, Flanagan-Belytschko viscous hourglass
//! control for reduced-integration bricks, and a one-dimensional elastic
//! bar with its d'Alembert reference for QA. Contact forces enter through
//! the caller-supplied force vector (see the explicit contact path).

// Rust guideline compliant 2026-02-21

/// Dilatational wave speed `sqrt((K + 4G/3) / rho)` in m/s.
///
/// Takes Young's modulus in Pa, Poisson ratio, and density in kg/m^3.
/// Sets the CFL limit of every 3D explicit element.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::explicit_dynamics::dilatational_wave_speed;
///
/// let c = dilatational_wave_speed(200.0e9, 0.3, 7800.0);
/// assert!(c > 5000.0 && c < 6500.0);
/// ```
pub fn dilatational_wave_speed(youngs: f64, poisson: f64, density: f64) -> f64 {
    let k = youngs / (3.0 * (1.0 - 2.0 * poisson));
    let g = youngs / (2.0 * (1.0 + poisson));
    ((k + 4.0 * g / 3.0) / density).sqrt()
}

/// Bar wave speed `sqrt(E / rho)` in m/s for slender members.
pub fn bar_wave_speed(youngs: f64, density: f64) -> f64 {
    (youngs / density).sqrt()
}

/// Stable explicit step from the CFL condition in s.
///
/// Returns `safety * min_length / wave_speed`; safety below one absorbs
/// mesh distortion and nonlinear stiffening (0.9 standard, 0.67 for
/// severe impact).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::explicit_dynamics::critical_timestep;
///
/// let dt = critical_timestep(0.01, 5000.0, 0.9);
/// assert!((dt - 1.8e-6).abs() < 1e-12);
/// ```
pub fn critical_timestep(min_length: f64, wave_speed: f64, safety: f64) -> f64 {
    safety * min_length / wave_speed.max(1e-30)
}

/// Density multiplier that raises the critical step to a target.
///
/// Mass scaling multiplies selected element densities by
/// `(dt_target / dt_critical)^2`, slowing their wave speed exactly to
/// the target step. Returns 1 when no scaling is needed.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::explicit_dynamics::mass_scaling_factor;
///
/// assert!((mass_scaling_factor(2.0e-6, 1.0e-6) - 4.0).abs() < 1e-12);
/// assert!((mass_scaling_factor(0.5e-6, 1.0e-6) - 1.0).abs() < 1e-12);
/// ```
pub fn mass_scaling_factor(dt_target: f64, dt_critical: f64) -> f64 {
    let ratio = dt_target / dt_critical.max(1e-30);
    if ratio > 1.0 { ratio * ratio } else { 1.0 }
}

/// Configuration of the central-difference integrator.
#[derive(Debug, Clone, Copy)]
pub struct ExplicitConfig {
    /// CFL safety factor in `(0, 1]`.
    pub cfl_safety: f64,
    /// Viscous hourglass coefficient (0.05 to 0.15 typical).
    pub hourglass_coefficient: f64,
}

impl Default for ExplicitConfig {
    fn default() -> Self {
        Self {
            cfl_safety: 0.9,
            hourglass_coefficient: 0.1,
        }
    }
}

/// Lumped-mass central-difference integrator state.
///
/// Holds displacements at whole steps and velocities at half steps:
/// `v += M^-1 F dt`, then `u += v dt`. Zero inverse mass pins a degree
/// of freedom (infinite mass) without renumbering.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::explicit_dynamics::ExplicitState;
///
/// let mut state = ExplicitState::new(&[2.0, 2.0]);
/// state.set_velocity(&[1.0, 0.0]);
/// state.step(&[0.0, 0.0], 0.1);
/// assert!((state.displacement()[0] - 0.1).abs() < 1e-12);
/// ```
#[derive(Debug, Clone)]
pub struct ExplicitState {
    /// Displacements at the current whole step.
    displacement: Vec<f64>,
    /// Velocities at the current half step.
    velocity_half: Vec<f64>,
    /// Inverse lumped masses (zero pins the DOF).
    mass_inv: Vec<f64>,
    /// Accumulated simulated time in s.
    time: f64,
}

impl ExplicitState {
    /// Creates a zeroed state from per-DOF lumped masses in kg.
    ///
    /// # Panics
    ///
    /// Panics when a mass is negative or the list is empty.
    pub fn new(masses: &[f64]) -> Self {
        assert!(!masses.is_empty(), "explicit state needs degrees of freedom");
        assert!(
            masses.iter().all(|m| *m >= 0.0),
            "lumped masses must be non-negative"
        );
        let num = masses.len();
        Self {
            displacement: vec![0.0; num],
            velocity_half: vec![0.0; num],
            mass_inv: masses
                .iter()
                .map(|m| if *m > 0.0 { 1.0 / m } else { 0.0 })
                .collect(),
            time: 0.0,
        }
    }

    /// Creates a state with uniform per-DOF mass in kg.
    pub fn uniform(num_dofs: usize, mass: f64) -> Self {
        Self::new(&vec![mass; num_dofs])
    }

    /// Counts the degrees of freedom.
    pub fn num_dofs(&self) -> usize {
        self.displacement.len()
    }

    /// Returns the simulated time in s.
    pub fn time(&self) -> f64 {
        self.time
    }

    /// Borrows the whole-step displacements.
    pub fn displacement(&self) -> &[f64] {
        &self.displacement
    }

    /// Borrows the half-step velocities.
    pub fn velocity(&self) -> &[f64] {
        &self.velocity_half
    }

    /// Sets displacements, discarding velocity history.
    ///
    /// # Panics
    ///
    /// Panics when the length mismatches the DOF count.
    pub fn set_displacement(&mut self, u: &[f64]) {
        assert_eq!(u.len(), self.num_dofs(), "displacement length mismatch");
        self.displacement.copy_from_slice(u);
    }

    /// Sets the half-step velocities directly (first-order start).
    ///
    /// Pinned DOFs (zero inverse mass) always receive zero velocity.
    ///
    /// # Panics
    ///
    /// Panics when the length mismatches the DOF count.
    pub fn set_velocity(&mut self, v: &[f64]) {
        assert_eq!(v.len(), self.num_dofs(), "velocity length mismatch");
        for (slot, (mi, value)) in self
            .velocity_half
            .iter_mut()
            .zip(self.mass_inv.iter().zip(v.iter()))
        {
            *slot = if *mi > 0.0 { *value } else { 0.0 };
        }
    }

    /// Starts half-step velocities from whole-step data exactly.
    ///
    /// Sets `v_{-1/2} = v_0 - M^-1 F_0 dt / 2` so the first step is
    /// second-order accurate. Takes whole-step velocities, initial
    /// forces, and the step in s.
    ///
    /// # Panics
    ///
    /// Panics on length mismatch or non-positive `dt`.
    pub fn start_half_step(&mut self, v0: &[f64], f0: &[f64], dt: f64) {
        assert!(dt > 0.0, "explicit step needs positive dt");
        assert_eq!(v0.len(), self.num_dofs(), "velocity length mismatch");
        assert_eq!(f0.len(), self.num_dofs(), "force length mismatch");
        for i in 0..self.num_dofs() {
            self.velocity_half[i] = if self.mass_inv[i] > 0.0 {
                v0[i] - 0.5 * dt * self.mass_inv[i] * f0[i]
            } else {
                0.0
            };
        }
    }

    /// Advances one central-difference step of `dt` seconds.
    ///
    /// Takes the total force vector (internal plus contact plus external)
    /// at the current whole step. Pinned DOFs (zero inverse mass) keep
    /// zero velocity and displacement.
    ///
    /// # Panics
    ///
    /// Panics on force-length mismatch or non-positive `dt`.
    pub fn step(&mut self, forces: &[f64], dt: f64) {
        assert!(dt > 0.0, "explicit step needs positive dt");
        assert_eq!(forces.len(), self.num_dofs(), "force length mismatch");
        for (i, force) in forces.iter().enumerate() {
            if self.mass_inv[i] == 0.0 {
                self.velocity_half[i] = 0.0;
                continue;
            }
            self.velocity_half[i] += dt * self.mass_inv[i] * force;
            self.displacement[i] += dt * self.velocity_half[i];
        }
        self.time += dt;
    }

    /// Kinetic energy in J at the half step.
    pub fn kinetic_energy(&self) -> f64 {
        self.velocity_half
            .iter()
            .zip(self.mass_inv.iter())
            .map(|(v, mi)| if *mi > 0.0 { 0.5 * v * v / mi } else { 0.0 })
            .sum()
    }

    /// Maximum nodal speed in m/s, for CFL monitoring.
    pub fn max_speed(&self) -> f64 {
        self.velocity_half.iter().fold(0.0, |m, v| m.max(v.abs()))
    }
}

/// Natural coordinates of the HEX8 nodes (xi, eta, zeta).
pub const HEX8_NATURAL: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

/// Hourglass base vectors (xi eta, eta zeta, zeta xi, xi eta zeta).
fn hourglass_base() -> [[f64; 8]; 4] {
    let mut h = [[0.0; 8]; 4];
    for (i, n) in HEX8_NATURAL.iter().enumerate() {
        h[0][i] = n[0] * n[1];
        h[1][i] = n[1] * n[2];
        h[2][i] = n[2] * n[0];
        h[3][i] = n[0] * n[1] * n[2];
    }
    h
}

/// Flanagan-Belytschko hourglass projection of one brick.
///
/// Returns the four gamma vectors orthogonalized against linear fields,
/// the element volume, and the characteristic length `V^(1/3)`. Takes
/// the eight nodal coordinates in m.
///
/// # Panics
///
/// Panics when the centroid Jacobian is not positive (inverted element).
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::explicit_dynamics::hex8_hourglass_gammas;
///
/// let coords = [
///     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
/// ];
/// let (_, volume, length) = hex8_hourglass_gammas(&coords);
/// assert!((volume - 1.0).abs() < 1e-12);
/// assert!((length - 1.0).abs() < 1e-12);
/// ```
pub fn hex8_hourglass_gammas(coords: &[[f64; 3]; 8]) -> ([[f64; 8]; 4], f64, f64) {
    // Centroid Jacobian from dN/dxi = natural / 8.
    let mut jac = [0.0; 9];
    for (node, x) in coords.iter().enumerate() {
        for j in 0..3 {
            for k in 0..3 {
                jac[j * 3 + k] += x[j] * HEX8_NATURAL[node][k] / 8.0;
            }
        }
    }
    let det = jac[0] * (jac[4] * jac[8] - jac[5] * jac[7])
        - jac[1] * (jac[3] * jac[8] - jac[5] * jac[6])
        + jac[2] * (jac[3] * jac[7] - jac[4] * jac[6]);
    assert!(det > 0.0, "hourglass needs a positive centroid Jacobian");
    let inv_det = 1.0 / det;
    let inv = [
        (jac[4] * jac[8] - jac[5] * jac[7]) * inv_det,
        (jac[2] * jac[7] - jac[1] * jac[8]) * inv_det,
        (jac[1] * jac[5] - jac[2] * jac[4]) * inv_det,
        (jac[5] * jac[6] - jac[3] * jac[8]) * inv_det,
        (jac[0] * jac[8] - jac[2] * jac[6]) * inv_det,
        (jac[2] * jac[3] - jac[0] * jac[5]) * inv_det,
        (jac[3] * jac[7] - jac[4] * jac[6]) * inv_det,
        (jac[1] * jac[6] - jac[0] * jac[7]) * inv_det,
        (jac[0] * jac[4] - jac[1] * jac[3]) * inv_det,
    ];
    // Centroid B-matrix rows b[j][I] = dN_I/dx_j.
    let mut b = [[0.0; 8]; 3];
    for j in 0..3 {
        for (node, n) in HEX8_NATURAL.iter().enumerate() {
            b[j][node] = (inv[j * 3] * n[0] + inv[j * 3 + 1] * n[1] + inv[j * 3 + 2] * n[2]) / 8.0;
        }
    }
    // Gamma projection: h - sum_j (h . x_j) b_j.
    let base = hourglass_base();
    let mut gamma = [[0.0; 8]; 4];
    for a in 0..4 {
        let mut hx = [0.0; 3];
        for j in 0..3 {
            hx[j] = (0..8).map(|i| base[a][i] * coords[i][j]).sum();
        }
        for i in 0..8 {
            gamma[a][i] = base[a][i] - (hx[0] * b[0][i] + hx[1] * b[1][i] + hx[2] * b[2][i]);
        }
    }
    let volume = 8.0 * det;
    (gamma, volume, volume.cbrt())
}

/// Viscous hourglass nodal forces of one reduced-integration brick.
///
/// Resists the four hourglass velocity modes with the generalized damping
/// `kappa rho c V^(2/3)`: takes nodal coordinates and velocities plus
/// density, wave speed, and the dimensionless coefficient, and returns
/// eight nodal force vectors. Orthogonal to rigid motion and uniform
/// strain by the gamma projection.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::explicit_dynamics::hex8_hourglass_forces;
///
/// let coords = [
///     [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0],
/// ];
/// let rigid = [[1.0, 0.0, 0.0]; 8];
/// let forces = hex8_hourglass_forces(&coords, &rigid, 7800.0, 5000.0, 0.1);
/// for f in forces {
///     assert!(f[0].abs() + f[1].abs() + f[2].abs() < 1e-6);
/// }
/// ```
pub fn hex8_hourglass_forces(
    coords: &[[f64; 3]; 8],
    velocities: &[[f64; 3]; 8],
    density: f64,
    wave_speed: f64,
    kappa: f64,
) -> [[f64; 3]; 8] {
    let (gamma, volume, _) = hex8_hourglass_gammas(coords);
    // Modal hourglass velocities q_a = gamma_a . v per spatial axis.
    let mut modal = [[0.0; 3]; 4];
    for a in 0..4 {
        for j in 0..3 {
            modal[a][j] = (0..8).map(|i| gamma[a][i] * velocities[i][j]).sum();
        }
    }
    // Generalized damping scales with face area V^(2/3): rho c A q has
    // units of force, matching a viscous traction on the element faces.
    let damping = kappa * density * wave_speed * volume.powf(2.0 / 3.0);
    let mut forces = [[0.0; 3]; 8];
    for a in 0..4 {
        for j in 0..3 {
            let generalized = -damping * modal[a][j];
            for i in 0..8 {
                forces[i][j] += gamma[a][i] * generalized;
            }
        }
    }
    forces
}

/// d'Alembert solution of a Gaussian pulse on an infinite bar.
///
/// Returns the axial displacement at position `x` and time `t` for the
/// initial displacement `A exp(-((x - x0) / w)^2)` with zero velocity:
/// two half-amplitude pulses traveling at `±c`. Units are SI.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::explicit_dynamics::dalembert_gaussian;
///
/// let u0 = dalembert_gaussian(5.0, 0.0, 5000.0, 5.0, 0.5, 1.0e-3);
/// assert!((u0 - 1.0e-3).abs() < 1e-15);
/// ```
pub fn dalembert_gaussian(
    x: f64,
    t: f64,
    wave_speed: f64,
    center: f64,
    width: f64,
    amplitude: f64,
) -> f64 {
    let right = (x - center - wave_speed * t) / width;
    let left = (x - center + wave_speed * t) / width;
    0.5 * amplitude * ((-right * right).exp() + (-left * left).exp())
}

/// Strain energy of a Gaussian displacement pulse in J.
///
/// Closed form `E A A^2 sqrt(pi/2) / (2 w)` for the initial pulse
/// `A exp(-((x - x0)/w)^2)` on a bar of modulus `E` and area `A`:
/// half the pulse energy starts as strain energy (the rest converts to
/// kinetic as the pulse splits). Feeds the energy C2 golden.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::explicit_dynamics::gaussian_pulse_strain_energy;
///
/// let e = gaussian_pulse_strain_energy(200.0e9, 0.01, 0.4, 1.0e-3);
/// assert!((e - 3133.0).abs() < 1.0);
/// ```
pub fn gaussian_pulse_strain_energy(youngs: f64, area: f64, width: f64, amplitude: f64) -> f64 {
    youngs * area * amplitude * amplitude * (std::f64::consts::PI / 2.0).sqrt() / (2.0 * width)
}

/// One-dimensional elastic bar for explicit QA.
///
/// Axial chain of linear elements with lumped mass and exact element
/// stiffness, integrated by [`ExplicitState`]. The left end pins when
/// `fixed_left` is set (zero inverse mass). Drives the d'Alembert pulse
/// comparison and the energy-conservation QA.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::explicit_dynamics::ElasticBar1D;
///
/// let bar = ElasticBar1D::new(100, 10.0, 0.01, 200.0e9, 7800.0, false);
/// assert!(bar.stable_step(0.9) > 0.0);
/// ```
#[derive(Debug, Clone)]
pub struct ElasticBar1D {
    /// Element count.
    num_elements: usize,
    /// Bar length in m.
    length: f64,
    /// Cross-section area in m^2.
    area: f64,
    /// Young's modulus in Pa.
    youngs: f64,
    /// Density in kg/m^3.
    density: f64,
    /// Whether node 0 is pinned.
    fixed_left: bool,
    /// Integrator state over the nodal displacements.
    state: ExplicitState,
}

impl ElasticBar1D {
    /// Creates a bar with lumped mass and zeroed state.
    ///
    /// Takes element count, length, area, modulus, density, and the left
    /// pin flag. End nodes carry half element masses.
    ///
    /// # Panics
    ///
    /// Panics when the element count is zero or data is not positive.
    pub fn new(
        num_elements: usize,
        length: f64,
        area: f64,
        youngs: f64,
        density: f64,
        fixed_left: bool,
    ) -> Self {
        assert!(num_elements > 0, "bar needs elements");
        assert!(
            length > 0.0 && area > 0.0 && youngs > 0.0 && density > 0.0,
            "bar data must be positive"
        );
        let le = length / num_elements as f64;
        let middle = density * area * le;
        let mut masses = vec![middle; num_elements + 1];
        masses[0] = 0.5 * middle;
        masses[num_elements] = 0.5 * middle;
        if fixed_left {
            masses[0] = 0.0;
        }
        Self {
            num_elements,
            length,
            area,
            youngs,
            density,
            fixed_left,
            state: ExplicitState::new(&masses),
        }
    }

    /// Element length in m.
    pub fn element_length(&self) -> f64 {
        self.length / self.num_elements as f64
    }

    /// Bar wave speed in m/s.
    pub fn wave_speed(&self) -> f64 {
        bar_wave_speed(self.youngs, self.density)
    }

    /// Stable step in s at a CFL safety factor.
    pub fn stable_step(&self, safety: f64) -> f64 {
        critical_timestep(self.element_length(), self.wave_speed(), safety)
    }

    /// Nodal coordinates in m.
    pub fn nodes(&self) -> Vec<f64> {
        (0..=self.num_elements)
            .map(|i| self.element_length() * i as f64)
            .collect()
    }

    /// Borrows the integrator state.
    pub fn state(&self) -> &ExplicitState {
        &self.state
    }

    /// Imposes a Gaussian displacement pulse with zero velocity.
    ///
    /// Takes pulse center, width, and amplitude in SI; exact-starts the
    /// half-step velocities from the pulse forces at `dt`.
    pub fn impose_gaussian_pulse(&mut self, center: f64, width: f64, amplitude: f64, dt: f64) {
        let nodes = self.nodes();
        let u: Vec<f64> = nodes
            .iter()
            .map(|x| {
                let z = (x - center) / width;
                amplitude * (-z * z).exp()
            })
            .collect();
        self.state.set_displacement(&u);
        let forces = self.internal_forces(&u);
        let zeros = vec![0.0; u.len()];
        self.state.start_half_step(&zeros, &forces, dt);
    }

    /// Internal elastic forces in N at displacements `u`.
    pub fn internal_forces(&self, u: &[f64]) -> Vec<f64> {
        let le = self.element_length();
        let k = self.youngs * self.area / le;
        let mut f = vec![0.0; self.num_elements + 1];
        for e in 0..self.num_elements {
            // Tension-positive axial force pulls both nodes inward.
            let axial = k * (u[e + 1] - u[e]);
            f[e] += axial;
            f[e + 1] -= axial;
        }
        if self.fixed_left {
            f[0] = 0.0;
        }
        f
    }

    /// Advances one step of `dt` seconds under internal forces.
    ///
    /// # Panics
    ///
    /// Panics when `dt` is not positive.
    pub fn step(&mut self, dt: f64) {
        let u = self.state.displacement().to_vec();
        let f = self.internal_forces(&u);
        self.state.step(&f, dt);
    }

    /// Advances one step with additional external nodal forces in N.
    ///
    /// # Panics
    ///
    /// Panics on length mismatch or non-positive `dt`.
    pub fn step_with_forces(&mut self, dt: f64, external: &[f64]) {
        assert_eq!(external.len(), self.num_elements + 1, "force length mismatch");
        let u = self.state.displacement().to_vec();
        let mut f = self.internal_forces(&u);
        for (total, extra) in f.iter_mut().zip(external.iter()) {
            *total += extra;
        }
        self.state.step(&f, dt);
    }

    /// Strain energy in J at the current displacements.
    pub fn strain_energy(&self) -> f64 {
        let u = self.state.displacement();
        let le = self.element_length();
        let k = self.youngs * self.area / le;
        (0..self.num_elements)
            .map(|e| 0.5 * k * (u[e + 1] - u[e]).powi(2))
            .sum()
    }

    /// Total mechanical energy (kinetic plus strain) in J.
    pub fn total_energy(&self) -> f64 {
        self.state.kinetic_energy() + self.strain_energy()
    }
}

/// Elastic impact force plateau of a bar on a rigid wall.
///
/// Returns `rho c A V0`, the contact force while the compressive wave
/// travels to the free end and back (`0 < t < 2L/c`). Feeds the Taylor
/// bar C3 probe in its elastic prove-out form.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::explicit_dynamics::elastic_bar_impact_force;
///
/// let f = elastic_bar_impact_force(100.0, 7800.0, 5000.0, 1.0e-4);
/// assert!((f - 390.0e3).abs() < 1.0);
/// ```
pub fn elastic_bar_impact_force(
    velocity: f64,
    density: f64,
    wave_speed: f64,
    area: f64,
) -> f64 {
    density * wave_speed * area * velocity
}

/// Taylor bar impact setup (cylinder on a rigid wall).
///
/// Describes the classic Taylor test geometry and loading: a cylinder of
/// `length` and `radius` striking a rigid wall at `velocity` along `-z`.
/// The mesh builder clips a Cartesian grid to the circular section and
/// projects boundary nodes radially, so every brick keeps positive
/// volume. Runs couple this setup with the explicit contact wall.
///
/// # Examples
///
/// ```rust
/// use sonium_fem::solver::explicit_dynamics::TaylorBarSetup;
///
/// let setup = TaylorBarSetup::copper_bar();
/// let (nodes, bricks) = setup.mesh(6, 6);
/// assert!(!bricks.is_empty() && !nodes.is_empty());
/// ```
#[derive(Debug, Clone, Copy)]
pub struct TaylorBarSetup {
    /// Cylinder length in m.
    pub length: f64,
    /// Cylinder radius in m.
    pub radius: f64,
    /// Impact velocity in m/s along -z.
    pub velocity: f64,
    /// Density in kg/m^3.
    pub density: f64,
    /// Young's modulus in Pa.
    pub youngs: f64,
    /// Poisson ratio, dimensionless.
    pub poisson_ratio: f64,
}

impl TaylorBarSetup {
    /// Creates a Taylor setup from geometry, velocity, and material.
    pub fn new(
        length: f64,
        radius: f64,
        velocity: f64,
        density: f64,
        youngs: f64,
        poisson_ratio: f64,
    ) -> Self {
        Self {
            length,
            radius,
            velocity,
            density,
            youngs,
            poisson_ratio,
        }
    }

    /// OFHC copper bar after Taylor (1948), 227 m/s.
    pub fn copper_bar() -> Self {
        Self::new(0.0254, 0.003175, 227.0, 8930.0, 117.0e9, 0.35)
    }

    /// Cross-section area in m^2.
    pub fn area(&self) -> f64 {
        std::f64::consts::PI * self.radius * self.radius
    }

    /// Elastic impact force plateau in N (pre-yield probe).
    pub fn elastic_plateau_force(&self) -> f64 {
        elastic_bar_impact_force(
            self.velocity,
            self.density,
            bar_wave_speed(self.youngs, self.density),
            self.area(),
        )
    }

    /// Bricks the cylinder with `nz` axial by `n` cross-section cells.
    ///
    /// Returns nodes in m (impact face at `z = 0`) and 0-based HEX8
    /// connectivity. Cross-section cells fully inside the circle are kept;
    /// region-boundary nodes project radially onto the surface.
    ///
    /// # Panics
    ///
    /// Panics when a resolution is zero.
    pub fn mesh(&self, nz: usize, n: usize) -> (Vec<[f64; 3]>, Vec<[usize; 8]>) {
        assert!(nz > 0 && n > 0, "taylor mesh needs positive resolution");
        let step = 2.0 * self.radius / n as f64;
        let corner = |i: usize| -self.radius + step * i as f64;
        let inside = |i: usize, j: usize| {
            let (x, y) = (corner(i), corner(j));
            x * x + y * y <= self.radius * self.radius
        };
        let mut kept = vec![false; n * n];
        for i in 0..n {
            for j in 0..n {
                kept[i * n + j] = (0..=1).all(|di| (0..=1).all(|dj| inside(i + di, j + dj)));
            }
        }
        // One layer of node ids shared by all axial stations.
        let mut layer_id = vec![0usize; (n + 1) * (n + 1)];
        let mut layer_nodes: Vec<[f64; 2]> = Vec::new();
        for i in 0..=n {
            for j in 0..=n {
                let mut used = false;
                let mut boundary = false;
                for di in 0..=1 {
                    for dj in 0..=1 {
                        let (ci, cj) = (i + 1 - di, j + 1 - dj);
                        if ci == 0 || cj == 0 || ci > n || cj > n {
                            continue;
                        }
                        if kept[(ci - 1) * n + (cj - 1)] {
                            used = true;
                        } else {
                            boundary = true;
                        }
                    }
                }
                if i == 0 || j == 0 || i == n || j == n {
                    boundary = true;
                }
                if !used {
                    continue;
                }
                let (mut x, mut y) = (corner(i), corner(j));
                if boundary {
                    let r = (x * x + y * y).sqrt();
                    if r > 1e-12 {
                        x *= self.radius / r;
                        y *= self.radius / r;
                    }
                }
                layer_id[i * (n + 1) + j] = layer_nodes.len();
                layer_nodes.push([x, y]);
            }
        }
        let per_layer = layer_nodes.len();
        let mut nodes = Vec::with_capacity(per_layer * (nz + 1));
        for k in 0..=nz {
            let z = self.length * k as f64 / nz as f64;
            for xy in &layer_nodes {
                nodes.push([xy[0], xy[1], z]);
            }
        }
        let mut bricks = Vec::new();
        for k in 0..nz {
            for i in 0..n {
                for j in 0..n {
                    if !kept[i * n + j] {
                        continue;
                    }
                    let at = |di: usize, dj: usize| layer_id[(i + di) * (n + 1) + (j + dj)];
                    let base = k * per_layer;
                    bricks.push([
                        base + at(0, 0),
                        base + at(1, 0),
                        base + at(1, 1),
                        base + at(0, 1),
                        base + per_layer + at(0, 0),
                        base + per_layer + at(1, 0),
                        base + per_layer + at(1, 1),
                        base + per_layer + at(0, 1),
                    ]);
                }
            }
        }
        (nodes, bricks)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_speeds_match_closed_forms() {
        let c = dilatational_wave_speed(200.0e9, 0.3, 7800.0);
        let k = 200.0e9 / (3.0 * 0.4f64);
        let g = 200.0e9 / 2.6f64;
        let expected: f64 = ((k + 4.0 * g / 3.0) / 7800.0).sqrt();
        assert!((c - expected).abs() < 1e-9);
        assert!((bar_wave_speed(200.0e9, 7800.0) - (200.0e9f64 / 7800.0).sqrt()).abs() < 1e-9);
    }

    #[test]
    fn free_flight_advances_exactly() {
        let mut state = ExplicitState::uniform(3, 2.0);
        state.set_velocity(&[1.0, -2.0, 0.5]);
        state.step(&[0.0, 0.0, 0.0], 0.1);
        assert!((state.displacement()[0] - 0.1).abs() < 1e-12);
        assert!((state.displacement()[1] + 0.2).abs() < 1e-12);
        // Kinetic energy of the drift.
        assert!((state.kinetic_energy() - 0.5 * 2.0 * (1.0 + 4.0 + 0.25)).abs() < 1e-12);
    }

    #[test]
    fn pinned_dof_stays_put() {
        let mut state = ExplicitState::new(&[0.0, 1.0]);
        state.set_velocity(&[5.0, 1.0]);
        state.step(&[1.0e9, 0.0], 0.01);
        assert_eq!(state.displacement()[0], 0.0);
        assert!((state.displacement()[1] - 0.01).abs() < 1e-12);
    }

    #[test]
    fn bar_pulse_matches_dalembert() {
        // Long bar, pulse well away from the ends at the sample time.
        let mut bar = ElasticBar1D::new(400, 20.0, 0.01, 200.0e9, 7800.0, false);
        let c = bar.wave_speed();
        let dt = bar.stable_step(0.5);
        bar.impose_gaussian_pulse(10.0, 0.4, 1.0e-3, dt);
        let steps = (0.0008 / dt) as usize;
        for _ in 0..steps {
            bar.step(dt);
        }
        let t = steps as f64 * dt;
        let nodes = bar.nodes();
        let u = bar.state().displacement().to_vec();
        let mut max_err = 0.0f64;
        for (i, x) in nodes.iter().enumerate() {
            let expected = dalembert_gaussian(*x, t, c, 10.0, 0.4, 1.0e-3);
            max_err = max_err.max((u[i] - expected).abs());
        }
        assert!(max_err < 5.0e-5, "max d'Alembert error {max_err:.3e}");
    }

    #[test]
    fn pulse_strain_energy_matches_quadrature() {
        let (e, a, w, amp) = (200.0e9, 0.01, 0.4, 1.0e-3);
        // Trapezoidal strain energy of the analytic pulse slope.
        let n = 20000;
        let (mut energy, mut prev) = (0.0, 0.0);
        for i in 0..=n {
            let x = 10.0 * i as f64 / n as f64;
            let z = (x - 5.0) / w;
            let slope = amp * (-2.0 * z / w) * (-z * z).exp();
            let density = 0.5 * e * a * slope * slope;
            if i > 0 {
                energy += 0.5 * (prev + density) * (10.0 / n as f64);
            }
            prev = density;
        }
        let closed = gaussian_pulse_strain_energy(e, a, w, amp);
        assert!((energy - closed).abs() / closed < 1e-6, "quad {energy}, closed {closed}");
    }

    #[test]
    fn fixed_bar_conserves_energy() {
        let mut bar = ElasticBar1D::new(50, 5.0, 0.01, 200.0e9, 7800.0, true);
        // Half-CFL step: the measured-energy oscillation scales with dt^2,
        // so this separates integrator drift from sampling phase.
        let dt = bar.stable_step(0.5);
        // Pluck: triangular displacement, zero velocity.
        let nodes = bar.nodes();
        let u: Vec<f64> = nodes
            .iter()
            .map(|x| 1.0e-3 * (1.0 - x / 5.0) * (x / 5.0) * 4.0)
            .collect();
        bar.state.set_displacement(&u);
        let f = bar.internal_forces(&u);
        bar.state.start_half_step(&vec![0.0; u.len()], &f, dt);
        let e0 = bar.total_energy();
        for _ in 0..4000 {
            bar.step(dt);
        }
        let drift = (bar.total_energy() - e0).abs() / e0;
        assert!(drift < 0.01, "energy drift {drift:.3e}");
    }

    #[test]
    fn hourglass_ignores_linear_fields() {
        let coords = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 3.0],
            [2.0, 0.0, 3.0],
            [2.0, 1.0, 3.0],
            [0.0, 1.0, 3.0],
        ];
        let (gamma, volume, _) = hex8_hourglass_gammas(&coords);
        assert!((volume - 6.0).abs() < 1e-9);
        // Gamma rows are orthogonal to constants and linear fields.
        for a in 0..4 {
            let sum: f64 = gamma[a].iter().sum();
            assert!(sum.abs() < 1e-12, "mode {a} sees translation");
            for j in 0..3 {
                let dot: f64 = (0..8).map(|i| gamma[a][i] * coords[i][j]).sum();
                assert!(dot.abs() < 1e-9, "mode {a} sees linear field {j}");
            }
        }
        // Linear velocity field draws no hourglass force.
        let velocities: [[f64; 3]; 8] = std::array::from_fn(|i| {
            [coords[i][0] + 1.0, 2.0 * coords[i][1], -coords[i][2]]
        });
        let forces = hex8_hourglass_forces(&coords, &velocities, 7800.0, 5000.0, 0.1);
        for f in forces {
            assert!(f[0].abs() + f[1].abs() + f[2].abs() < 1e-6);
        }
    }

    #[test]
    fn hourglass_resists_hourglass_mode() {
        let coords = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let (gamma, _, _) = hex8_hourglass_gammas(&coords);
        // Velocity along the first hourglass pattern in x.
        let velocities: [[f64; 3]; 8] = std::array::from_fn(|i| [gamma[0][i], 0.0, 0.0]);
        let forces = hex8_hourglass_forces(&coords, &velocities, 7800.0, 5000.0, 0.1);
        // Power f . v must be negative (dissipative).
        let power: f64 = (0..8).map(|i| forces[i][0] * velocities[i][0]).sum();
        assert!(power < 0.0, "hourglass power {power} must dissipate");
    }

    #[test]
    fn taylor_mesh_has_positive_volumes() {
        let setup = TaylorBarSetup::copper_bar();
        let (nodes, bricks) = setup.mesh(6, 8);
        assert!(!bricks.is_empty());
        for brick in &bricks {
            let p = |i: usize| nodes[brick[i]];
            // Five-tet decomposition of the hex.
            let tet = |a: [f64; 3], b: [f64; 3], c: [f64; 3], d: [f64; 3]| {
                let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
                let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
                let ad = [d[0] - a[0], d[1] - a[1], d[2] - a[2]];
                ((ab[1] * ac[2] - ab[2] * ac[1]) * ad[0]
                    + (ab[2] * ac[0] - ab[0] * ac[2]) * ad[1]
                    + (ab[0] * ac[1] - ab[1] * ac[0]) * ad[2])
                    / 6.0
            };
            let n: Vec<[f64; 3]> = (0..8).map(p).collect();
            let vol = tet(n[0], n[1], n[3], n[4]).abs()
                + tet(n[1], n[2], n[3], n[6]).abs()
                + tet(n[1], n[3], n[4], n[6]).abs()
                + tet(n[3], n[4], n[6], n[7]).abs()
                + tet(n[1], n[4], n[5], n[6]).abs();
            assert!(vol > 1e-15, "inverted taylor brick {brick:?}");
        }
        // Impact face sits at z = 0.
        assert!(nodes.iter().any(|n| n[2] == 0.0));
    }

    #[test]
    fn elastic_impact_plateau_matches_rho_c_a_v() {
        let setup = TaylorBarSetup::copper_bar();
        let expected =
            setup.density * bar_wave_speed(setup.youngs, setup.density) * setup.area() * setup.velocity;
        assert!((setup.elastic_plateau_force() - expected).abs() < 1.0);
    }
}
