//! Explicit H8 hexahedron with hourglass control and C1 materials.
//!
//! Under-integrated 8-node brick (Theory Manual "Solid Hexahedron
//! Elements", p. 80; `/DEF_SOLID Isolid = 1`, Reference Guide p. 59):
//! one centroid material point carries the uniform-strain response
//! while the Phase-1 Flanagan-Belytschko kernels
//! ([`hexa8_gamma`](sonium_compute::explicit_hourglass::hexa8_gamma),
//! [`hexa8_hourglass_forces`](sonium_compute::explicit_hourglass::hexa8_hourglass_forces))
//! supply orthogonal hourglass stabilization ("Flanagan-Belytschko
//! Formulation", pp. 86-89, Eqs 326-331), and
//! [`bulk_viscosity_q`](sonium_compute::explicit_hourglass::bulk_viscosity_q)
//! adds shock viscosity ("Shock Waves", pp. 91-92, Eq 340). Stress
//! comes from any frozen C1 [`StressUpdateModel`]; rate-dependent
//! materials evaluate at the estimated step rate. Node ordering and
//! centroid-gradient conventions reuse [`HexMesh`](super::hex::HexMesh)
//! and the [`hex`](super::hex) shape-function machinery.
//!
//! Nodal vectors are interleaved `[ux, uy, uz]` per node in
//! [`HEX_NODES`](super::hex::HEX_NODES) order, matching the compute
//! kernel ordering.

// Rust guideline compliant 2026-02-21

use super::super::plasticity::update::{StressUpdateModel, voigt_equivalent_increment_c1};
use super::hex::{HexMesh, hex_jacobian, hex_shape_derivatives, physical_gradients};
use sonium_compute::explicit_hourglass::{
    BULK_VISCOSITY_LINEAR_DEFAULT, BULK_VISCOSITY_QUADRATIC_DEFAULT, bulk_viscosity_q, hexa8_gamma,
    hexa8_hourglass_forces,
};

/// Error of the H8 explicit path.
#[derive(Debug, Clone)]
pub struct HexExplicitError {
    /// Human-readable failure description.
    pub message: String,
}

impl HexExplicitError {
    /// Builds an error from a message.
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for HexExplicitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "hex explicit error: {}", self.message)
    }
}

impl std::error::Error for HexExplicitError {}

/// Explicit-step parameters of the H8 path.
#[derive(Debug, Clone, Copy)]
pub struct HexExplicitParams {
    /// Mass density in kg/m^3.
    pub density: f64,
    /// Sound speed in m/s.
    pub sound_speed: f64,
    /// Viscous hourglass scale, dimensionless.
    ///
    /// Multiplies `rho c L^2` (N s/m) into the Flanagan-Belytschko
    /// viscous coefficient. The default 0.05 sits in the literature
    /// 0.01-0.10 band: large enough to damp spurious modes within a
    /// few steps, small enough to leave physical waves alone.
    pub hg_scale: f64,
    /// Quadratic shock-viscosity coefficient (default 1.10).
    pub bulk_c0: f64,
    /// Linear shock-viscosity coefficient (default 0.05).
    pub bulk_c1: f64,
}

impl HexExplicitParams {
    /// Builds parameters from density and sound speed with defaults.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::nonlinear_solid::HexExplicitParams;
    ///
    /// let params = HexExplicitParams::new(7800.0, 4500.0);
    /// assert!((params.hg_scale - 0.05).abs() < 1e-15);
    /// ```
    ///
    /// # Panics
    ///
    /// Panics for negative density, sound speed, hourglass scale, or
    /// viscosity coefficients.
    #[must_use]
    pub fn new(density: f64, sound_speed: f64) -> Self {
        assert!(
            density.is_finite() && density >= 0.0,
            "density must be finite non-negative"
        );
        assert!(
            sound_speed.is_finite() && sound_speed >= 0.0,
            "sound speed must be finite non-negative"
        );
        Self {
            density,
            sound_speed,
            hg_scale: 0.05,
            bulk_c0: BULK_VISCOSITY_QUADRATIC_DEFAULT,
            bulk_c1: BULK_VISCOSITY_LINEAR_DEFAULT,
        }
    }

    /// Overrides the viscous hourglass scale (dimensionless).
    #[must_use]
    pub fn with_hg_scale(mut self, hg_scale: f64) -> Self {
        assert!(
            hg_scale.is_finite() && hg_scale >= 0.0,
            "hourglass scale must be finite non-negative"
        );
        self.hg_scale = hg_scale;
        self
    }
}

/// Under-integrated H8 geometry: centroid operator plus hourglass vectors.
#[derive(Debug, Clone)]
pub struct HexExplicitGeom {
    /// Node coordinates in [`HEX_NODES`](super::hex::HEX_NODES) order.
    pub coords: [[f64; 3]; 8],
    /// Element volume in m^3 (8x centroid Jacobian determinant).
    pub volume: f64,
    /// Centroid physical gradients `dN/dx`, 8 nodes by 3 directions.
    pub grad_b: [[f64; 3]; 8],
    /// Flanagan-Belytschko Gamma vectors, 4 mode-major rows of 8.
    pub gamma: [f64; 32],
    /// Characteristic length `V / A_max` in m.
    pub char_length: f64,
}

impl HexExplicitGeom {
    /// Builds H8 geometry from eight node coordinates.
    ///
    /// The centroid Jacobian maps the reference cube (volume 8), so the
    /// physical volume is `8 det J(0)`.
    ///
    /// # Examples
    ///
    /// ```rust
    /// use sonium_fem::physics::solid::nonlinear_solid::HexExplicitGeom;
    ///
    /// let coords = [
    ///     [0.0, 0.0, 0.0],
    ///     [1.0, 0.0, 0.0],
    ///     [1.0, 1.0, 0.0],
    ///     [0.0, 1.0, 0.0],
    ///     [0.0, 0.0, 1.0],
    ///     [1.0, 0.0, 1.0],
    ///     [1.0, 1.0, 1.0],
    ///     [0.0, 1.0, 1.0],
    /// ];
    /// let geom = HexExplicitGeom::new(coords).unwrap();
    /// assert!((geom.volume - 1.0).abs() < 1e-14);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`HexExplicitError`] for non-finite coordinates or a
    /// degenerate (non-positive centroid Jacobian) element.
    pub fn new(coords: [[f64; 3]; 8]) -> Result<Self, HexExplicitError> {
        if !coords.iter().flatten().all(|v| v.is_finite()) {
            return Err(HexExplicitError::new("hex coordinates must be finite"));
        }
        let center = [0.0, 0.0, 0.0];
        let deriv = hex_shape_derivatives(&center);
        let (_, det, inv) = hex_jacobian(&coords, &deriv);
        if !det.is_finite() || det <= 0.0 {
            return Err(HexExplicitError::new(
                "hex needs a positive centroid Jacobian",
            ));
        }
        let grad_b = physical_gradients(&inv, &deriv);
        let mut flat = [0.0; 24];
        for (node, point) in coords.iter().enumerate() {
            flat[node * 3] = point[0];
            flat[node * 3 + 1] = point[1];
            flat[node * 3 + 2] = point[2];
        }
        let mut gamma = [0.0; 32];
        hexa8_gamma(&flat, &mut gamma);
        // Face areas: 6 quads from the HEX_NODES ordering, split in two.
        let faces = [
            [0, 1, 2, 3],
            [4, 5, 6, 7],
            [0, 1, 5, 4],
            [1, 2, 6, 5],
            [2, 3, 7, 6],
            [3, 0, 4, 7],
        ];
        let mut max_area: f64 = 0.0;
        for face in faces {
            let area = quad_area(
                &coords[face[0]],
                &coords[face[1]],
                &coords[face[2]],
                &coords[face[3]],
            );
            max_area = max_area.max(area);
        }
        Ok(Self {
            coords,
            volume: 8.0 * det,
            grad_b,
            gamma,
            char_length: 8.0 * det / max_area,
        })
    }

    /// Builds H8 geometry for one [`HexMesh`] element.
    ///
    /// # Errors
    ///
    /// Returns [`HexExplicitError`] when the element index is out of
    /// range or the element is degenerate.
    pub fn from_mesh(mesh: &HexMesh, elem_idx: usize) -> Result<Self, HexExplicitError> {
        if elem_idx >= mesh.conn.len() {
            return Err(HexExplicitError::new("hex element index out of range"));
        }
        Self::new(mesh.element_coords(elem_idx))
    }

    /// Lumped nodal masses in kg, one eighth of the total each.
    #[must_use]
    pub fn lumped_masses(&self, density: f64) -> [f64; 8] {
        let nodal = density * self.volume / 8.0;
        [nodal; 8]
    }

    /// Critical explicit step `L / c` in s.
    #[must_use]
    pub fn critical_step(&self, sound_speed: f64) -> f64 {
        self.char_length / sound_speed
    }

    /// Viscous hourglass coefficient in N s/m at the parameters.
    #[must_use]
    pub fn hg_coefficient(&self, params: &HexExplicitParams) -> f64 {
        params.hg_scale * params.density * params.sound_speed * self.char_length.powi(2)
    }
}

/// History state of one H8 element (single centroid point).
#[derive(Debug, Clone)]
pub struct HexExplicitState<S> {
    /// Material state at the centroid point.
    pub centroid: S,
    /// Accumulated hourglass energy in J.
    pub hg_energy: f64,
}

impl<S> HexExplicitState<S> {
    /// Builds H8 state from a centroid material-point state.
    #[must_use]
    pub fn new(centroid: S) -> Self {
        Self {
            centroid,
            hg_energy: 0.0,
        }
    }
}

/// Internal-force evaluation of one H8 element.
#[derive(Debug, Clone)]
pub struct HexForceOut {
    /// Nodal internal forces in N, interleaved per node.
    pub forces: [f64; 24],
    /// Viscous hourglass power in W (non-negative).
    pub hg_power: f64,
    /// Shock viscosity pressure in Pa.
    pub bulk_q: f64,
    /// Equivalent plastic strain increment of the step.
    pub dep: f64,
    /// True when the centroid point yielded.
    pub yielding: bool,
    /// Committed centroid Cauchy stress in C1 Voigt order (Pa).
    ///
    /// Carries the material response only; the shock pressure `bulk_q`
    /// enters the forces but not this tensor. Damage drivers read the
    /// triaxiality off this field.
    pub stress: [f64; 6],
    /// Equivalent total-strain-rate estimate in 1/s.
    ///
    /// Rate-dependent materials evaluated this step at this rate
    /// (`voigt_equivalent_increment_c1(d_eps) / dt`); damage drivers
    /// reuse it so flow and fracture see one consistent rate.
    pub strain_rate: f64,
}

/// Explicit internal forces of one H8 element.
///
/// Advances the centroid C1 material by the uniform-strain increment
/// from the nodal velocities over `dt`, commits it, and returns `V B^T
/// (sigma - q I)` plus viscous hourglass forces. Rate-dependent
/// materials evaluate at the estimated step rate, reported back in
/// [`HexForceOut::strain_rate`] alongside the committed
/// [`HexForceOut::stress`] for damage drivers. Hourglass energy
/// accumulates into the state with explicit Euler (`P dt`).
///
/// # Errors
///
/// Returns [`HexExplicitError`] for non-positive `dt` or non-finite input.
pub fn hex_explicit_forces<M>(
    geom: &HexExplicitGeom,
    state: &mut HexExplicitState<M::State>,
    material: &M,
    velocities: &[f64; 24],
    dt: f64,
    params: &HexExplicitParams,
) -> Result<HexForceOut, HexExplicitError>
where
    M: StressUpdateModel<StrainIncrement = [f64; 6]>,
{
    if dt.is_nan() || dt <= 0.0 {
        return Err(HexExplicitError::new("explicit step dt must be positive"));
    }
    if !velocities.iter().all(|v| v.is_finite()) {
        return Err(HexExplicitError::new("hex velocities must be finite"));
    }
    let mut d_eps = [0.0; 6];
    for (node, grad) in geom.grad_b.iter().enumerate() {
        let v = [
            velocities[node * 3] * dt,
            velocities[node * 3 + 1] * dt,
            velocities[node * 3 + 2] * dt,
        ];
        d_eps[0] += grad[0] * v[0];
        d_eps[1] += grad[1] * v[1];
        d_eps[2] += grad[2] * v[2];
        d_eps[3] += grad[1] * v[0] + grad[0] * v[1];
        d_eps[4] += grad[2] * v[1] + grad[1] * v[2];
        d_eps[5] += grad[0] * v[2] + grad[2] * v[0];
    }
    let vol_rate = (d_eps[0] + d_eps[1] + d_eps[2]) / dt;
    // The increment is C1 ordered ([xx, yy, zz, xy, yz, zx] with
    // engineering shears, matching update.rs), so the shared helper
    // applies directly and agrees exactly with Law2::update_with_rate.
    let strain_rate = voigt_equivalent_increment_c1(&d_eps) / dt;
    let bulk_q = bulk_viscosity_q(
        params.density,
        params.sound_speed,
        geom.char_length,
        params.bulk_c0,
        params.bulk_c1,
        vol_rate,
    );
    let update = if material.is_rate_dependent() {
        material.update_with_rate(&state.centroid, &d_eps, dt)
    } else {
        material.update(&state.centroid, &d_eps)
    };
    material.commit(&mut state.centroid, &update);
    let mut forces = [0.0; 24];
    for (node, grad) in geom.grad_b.iter().enumerate() {
        let s = update.stress;
        // Positive shock pressure is compressive; material Cauchy stress is tensile-positive.
        let n = [s[0] - bulk_q, s[1] - bulk_q, s[2] - bulk_q];
        forces[node * 3] = (n[0] * grad[0] + s[3] * grad[1] + s[5] * grad[2]) * geom.volume;
        forces[node * 3 + 1] = (s[3] * grad[0] + n[1] * grad[1] + s[4] * grad[2]) * geom.volume;
        forces[node * 3 + 2] = (s[5] * grad[0] + s[4] * grad[1] + n[2] * grad[2]) * geom.volume;
    }
    let mut hg_forces = [0.0; 24];
    hexa8_hourglass_forces(
        &geom.gamma,
        velocities,
        geom.hg_coefficient(params),
        &mut hg_forces,
    );
    let mut hg_power = 0.0;
    for i in 0..24 {
        forces[i] += hg_forces[i];
        hg_power += hg_forces[i] * velocities[i];
    }
    state.hg_energy += hg_power * dt;
    Ok(HexForceOut {
        forces,
        hg_power,
        bulk_q,
        dep: update.delta_equiv_plastic_strain,
        yielding: update.yielding,
        stress: update.stress,
        strain_rate,
    })
}

/// Area of a planar quad from four corners, split into triangles.
fn quad_area(p0: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3], p3: &[f64; 3]) -> f64 {
    tri_area(p0, p1, p2) + tri_area(p0, p2, p3)
}

/// Area of a triangle from three points.
fn tri_area(p0: &[f64; 3], p1: &[f64; 3], p2: &[f64; 3]) -> f64 {
    let u = [p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]];
    let v = [p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]];
    let cross = [
        u[1] * v[2] - u[2] * v[1],
        u[2] * v[0] - u[0] * v[2],
        u[0] * v[1] - u[1] * v[0],
    ];
    0.5 * (cross[0] * cross[0] + cross[1] * cross[1] + cross[2] * cross[2]).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::solid::plasticity::{HardeningLaw, J2Material, PlasticState};
    use sonium_compute::explicit_hourglass::HEXA8_HOURGLASS_BASE;

    fn elastic_j2() -> J2Material {
        J2Material::new(200.0e9, 0.3, 1.0e12, HardeningLaw::Perfect)
    }

    fn unit_cube() -> HexExplicitGeom {
        HexExplicitGeom::from_mesh(&HexMesh::unit_cube(1.0), 0).unwrap()
    }

    #[test]
    fn artificial_bulk_pressure_dissipates_compressive_mechanical_power() {
        let geom = unit_cube();
        let mesh = HexMesh::unit_cube(1.0);
        for (quadratic, linear) in [(1.1, 0.0), (0.0, 0.05)] {
            for rate in [-0.01, 0.01] {
                let velocity = std::array::from_fn(|i| mesh.coords[i / 3][i % 3] * rate);
                let mut base_params = HexExplicitParams::new(7800.0, 4500.0);
                base_params.bulk_c0 = 0.0;
                base_params.bulk_c1 = 0.0;
                let mut viscous = base_params.clone();
                viscous.bulk_c0 = quadratic;
                viscous.bulk_c1 = linear;
                let baseline = hex_explicit_forces(
                    &geom,
                    &mut HexExplicitState::new(PlasticState::new_3d()),
                    &elastic_j2(),
                    &velocity,
                    1e-6,
                    &base_params,
                )
                .unwrap();
                let output = hex_explicit_forces(
                    &geom,
                    &mut HexExplicitState::new(PlasticState::new_3d()),
                    &elastic_j2(),
                    &velocity,
                    1e-6,
                    &viscous,
                )
                .unwrap();
                let power = -output
                    .forces
                    .iter()
                    .zip(&baseline.forces)
                    .zip(velocity)
                    .map(|((f, base), v)| (f - base) * v)
                    .sum::<f64>();
                let expected = output.bulk_q * 3.0 * rate * geom.volume;
                assert!((power - expected).abs() < 1e-10 * expected.abs().max(1.0));
                if rate < 0.0 {
                    assert!(power < 0.0);
                } else {
                    assert_eq!(power, 0.0);
                }
                assert_eq!(baseline.stress, output.stress);
            }
        }
    }

    #[test]
    fn cube_gamma_matches_base_modes() {
        let geom = unit_cube();
        assert!((geom.volume - 1.0).abs() < 1e-14);
        for (a, base_row) in HEXA8_HOURGLASS_BASE.iter().enumerate() {
            for (node, base) in base_row.iter().enumerate() {
                assert_eq!(geom.gamma[a * 8 + node], *base);
            }
        }
        // Centroid gradients of the unit cube are +/- 1/4.
        for grad in geom.grad_b {
            for g in grad {
                assert!((g.abs() - 0.25).abs() < 1e-15);
            }
        }
    }

    #[test]
    fn patch_gives_closed_form_stress() {
        let geom = unit_cube();
        let material = elastic_j2();
        let mut state = HexExplicitState::new(PlasticState::new_3d());
        let params = HexExplicitParams::new(7800.0, 4500.0);
        // Uniaxial stretch at 1/s over dt = 1e-3: hourglass-free field.
        let mesh = HexMesh::unit_cube(1.0);
        let mut velocities = [0.0; 24];
        for node in 0..8 {
            velocities[node * 3] = mesh.coords[node][0];
        }
        let out = hex_explicit_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params)
            .unwrap();
        assert!(!out.yielding);
        // Linear fields carry no hourglass content: power is ~zero.
        assert!(out.hg_power.abs() < 1e-9);
        let g = 200.0e9 / 2.6;
        let k = 200.0e9 / 1.2;
        let expected = (k + 4.0 * g / 3.0) * 1.0e-3;
        assert!((state.centroid.stress[0] - expected).abs() < 1e-3 * expected);
        // Self-equilibrated forces.
        let mut force = [0.0; 3];
        for node in 0..8 {
            for (slot, inc) in force.iter_mut().zip(&out.forces[node * 3..node * 3 + 3]) {
                *slot += *inc;
            }
        }
        let scale: f64 = out.forces.iter().map(|f| f.abs()).sum::<f64>().max(1.0e-30);
        for component in force.iter() {
            assert!(component.abs() / scale < 1.0e-12);
        }
    }

    #[test]
    fn hourglass_dissipates_on_spurious_mode() {
        let geom = unit_cube();
        let material = elastic_j2();
        let mut state = HexExplicitState::new(PlasticState::new_3d());
        let params = HexExplicitParams::new(7800.0, 4500.0);
        // Pure hourglass velocity pattern (mode 0) in x.
        let mut velocities = [0.0; 24];
        for node in 0..8 {
            velocities[node * 3] = HEXA8_HOURGLASS_BASE[0][node];
        }
        let out = hex_explicit_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params)
            .unwrap();
        // Uniform strain of a pure hourglass pattern vanishes: only the
        // viscous forces resist, with positive power.
        assert!(out.hg_power > 0.0);
        assert!(state.hg_energy > 0.0);
        // Viscous coefficient check: P = c (gamma . v)^2 summed.
        let modal: f64 = HEXA8_HOURGLASS_BASE[0].iter().map(|h| h * h).sum();
        let expected = geom.hg_coefficient(&params) * modal * modal;
        assert!((out.hg_power - expected).abs() / expected < 1e-12);
    }

    #[test]
    fn rigid_translation_gives_zero_forces() {
        let geom = unit_cube();
        let material = elastic_j2();
        let mut state = HexExplicitState::new(PlasticState::new_3d());
        let params = HexExplicitParams::new(7800.0, 4500.0);
        let velocities = [1.0; 24];
        let out = hex_explicit_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params)
            .unwrap();
        assert!(out.forces.iter().all(|f| f.abs() < 1e-9));
        assert_eq!(out.hg_power, 0.0);
    }

    #[test]
    fn hg_scale_zero_disables_stabilization() {
        let geom = unit_cube();
        let material = elastic_j2();
        let params = HexExplicitParams::new(7800.0, 4500.0).with_hg_scale(0.0);
        assert_eq!(geom.hg_coefficient(&params), 0.0);
        let mut state = HexExplicitState::new(PlasticState::new_3d());
        let mut velocities = [0.0; 24];
        for node in 0..8 {
            velocities[node * 3] = HEXA8_HOURGLASS_BASE[0][node];
        }
        let out = hex_explicit_forces(&geom, &mut state, &material, &velocities, 1.0e-3, &params)
            .unwrap();
        assert!(out.forces.iter().all(|f| f.abs() < 1e-9));
    }

    #[test]
    fn mass_and_step_match_closed_form() {
        let geom = HexExplicitGeom::from_mesh(&HexMesh::unit_cube(2.0), 0).unwrap();
        assert!((geom.volume - 8.0).abs() < 1e-13);
        let masses = geom.lumped_masses(7800.0);
        assert!((masses.iter().sum::<f64>() - 8.0 * 7800.0).abs() < 1e-9);
        // V / A_max = 8 / 4 = 2 m.
        assert!((geom.char_length - 2.0).abs() < 1e-14);
        assert!((geom.critical_step(4500.0) - 2.0 / 4500.0).abs() < 1e-18);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(HexExplicitGeom::from_mesh(&HexMesh::unit_cube(1.0), 7).is_err());
        let geom = unit_cube();
        let material = elastic_j2();
        let mut state = HexExplicitState::new(PlasticState::new_3d());
        let params = HexExplicitParams::new(7800.0, 4500.0);
        assert!(
            hex_explicit_forces(&geom, &mut state, &material, &[0.0; 24], 0.0, &params).is_err()
        );
        let mut bad = [0.0; 24];
        bad[3] = f64::NAN;
        assert!(hex_explicit_forces(&geom, &mut state, &material, &bad, 1.0e-3, &params).is_err());
    }
}
