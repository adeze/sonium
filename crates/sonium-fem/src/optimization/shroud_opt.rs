//! Propeller shroud shape optimization
//!
//! Optimizes the axisymmetric waveguide profile around a drone propeller
//! to minimize radiated noise subject to geometric constraints.
//!
//! # Design variables
//!
//! 5 spline control points define the wall radius `r(z)`. Endpoints (inlet
//! and outlet) are fixed; the 3 interior radii are free.
//!
//! # Objective
//!
//! Composite objective combining noise, drag, and smoothness:
//!
//! ```text
//! J(x) = SWL_A(x) + w_drag · drag(x) + w_smooth · roughness(x)
//! ```
//!
//! - `SWL_A`: A-weighted sound power level at outlet [dB]
//! - `drag`: analytical pressure drag penalty ∝ ∫ (dr/dz)² dz
//! - `roughness`: curvature penalty ∝ ∫ (d²r/dz²)² dz (promotes smooth profiles)
//!
//! # Constraints
//!
//! - Box bounds on each control point radius: `r_min ≤ r_i ≤ r_max`
//! - Volume constraint folded into objective as a penalty (avoids
//!   augmented-Lagrangian complexity for 3 variables)
//!
//! # Gradient
//!
//! Central finite differences: `∂J/∂x_i ≈ (J(x+he_i) - J(x-he_i)) / (2h)`.
//! Cost per gradient = 2n = 6 Helmholtz solves (n=3 free variables).

use num_complex::Complex64;
use std::collections::HashMap;
use std::f64::consts::PI;

use crate::assembly::assemble_axi_boundary_mass;
use crate::basis::PolynomialDegree;
use crate::mesh::duct_profile_mesh;
use crate::physics::acoustics::aeroacoustics::propeller_source::PropellerSource;
use crate::physics::acoustics::pressure::AcousticsAssemblerExt;
use crate::physics::acoustics::pressure::axisymmetric::build_axi_acoustics_assembler_with_boundaries;
use crate::physics::acoustics::pressure::sound_metrics::{
    a_weighting, sound_power_from_boundary, sound_power_level,
};
use sonium_solvers::{FgmresConfig, IluPreconditioner, fgmres_preconditioned};

// ---------------------------------------------------------------------------
// Profile
// ---------------------------------------------------------------------------

/// Shroud profile defined by spline control points.
///
/// Wall radius `r(z)` is linearly interpolated between control points.
/// First and last points are fixed (inlet/outlet); interior points are free.
#[derive(Debug, Clone)]
pub struct ShroudProfile {
    pub z_positions: Vec<f64>,
    pub r_values: Vec<f64>,
    pub free_indices: Vec<usize>,
}

impl ShroudProfile {
    /// Straight-duct baseline with all control points free.
    pub fn straight(radius: f64, z_min: f64, z_max: f64, n_control_points: usize) -> Self {
        assert!(n_control_points >= 2);
        let dz = (z_max - z_min) / (n_control_points - 1) as f64;
        let z_positions: Vec<f64> = (0..n_control_points)
            .map(|i| z_min + i as f64 * dz)
            .collect();
        let r_values = vec![radius; n_control_points];
        let free_indices = (0..n_control_points).collect();
        Self {
            z_positions,
            r_values,
            free_indices,
        }
    }

    pub fn n_free(&self) -> usize {
        self.free_indices.len()
    }

    pub fn pack(&self) -> Vec<f64> {
        self.free_indices
            .iter()
            .map(|&i| self.r_values[i])
            .collect()
    }

    pub fn unpack(&self, x: &[f64]) -> Self {
        let mut p = self.clone();
        for (k, &i) in self.free_indices.iter().enumerate() {
            p.r_values[i] = x[k];
        }
        p
    }

    /// Evaluate wall radius at axial position z using a natural cubic spline
    /// through the control points. Clamped to endpoint values outside the range.
    pub fn wall_radius(&self, z: f64) -> f64 {
        let (xs, ys) = (&self.z_positions, &self.r_values);
        let n = xs.len();
        if n < 2 {
            return ys[0];
        }
        if z <= xs[0] {
            return ys[0];
        }
        if z >= xs[n - 1] {
            return ys[n - 1];
        }

        // Compute second derivatives (natural cubic spline, m[0]=m[n-1]=0)
        let m = Self::spline_second_derivatives(xs, ys);

        // Find interval
        let idx = xs
            .partition_point(|&xi| xi < z)
            .saturating_sub(1)
            .min(n - 2);
        let h = xs[idx + 1] - xs[idx];
        let a = (xs[idx + 1] - z) / h;
        let b = (z - xs[idx]) / h;

        a * ys[idx]
            + b * ys[idx + 1]
            + (a * a * a - a) * h * h / 6.0 * m[idx]
            + (b * b * b - b) * h * h / 6.0 * m[idx + 1]
    }

    /// Natural cubic spline second derivatives via Thomas algorithm.
    fn spline_second_derivatives(xs: &[f64], ys: &[f64]) -> Vec<f64> {
        let n = xs.len();
        if n <= 2 {
            return vec![0.0; n];
        }

        let h: Vec<f64> = xs.windows(2).map(|w| w[1] - w[0]).collect();
        let mut rhs = vec![0.0; n];
        for i in 1..n - 1 {
            rhs[i] = 6.0 * ((ys[i + 1] - ys[i]) / h[i] - (ys[i] - ys[i - 1]) / h[i - 1]);
        }

        // Forward sweep
        let mut cp = vec![0.0; n];
        let mut dp = vec![0.0; n];
        for i in 1..n - 1 {
            let diag = 2.0 * (h[i - 1] + h[i]);
            let sub = if i > 1 { h[i - 1] } else { 0.0 };
            let denom = diag - sub * cp[i - 1];
            cp[i] = h[i] / denom;
            dp[i] = (rhs[i] - sub * dp[i - 1]) / denom;
        }

        // Back substitution
        let mut m = vec![0.0; n];
        for i in (1..n - 1).rev() {
            m[i] = dp[i] - cp[i] * m[i + 1];
        }
        m
    }

    /// Enclosed volume of the axisymmetric duct: V = π ∫ r(z)² dz.
    /// Uses trapezoidal rule on control points.
    pub fn volume(&self) -> f64 {
        let mut v = 0.0;
        for i in 0..self.z_positions.len() - 1 {
            let dz = self.z_positions[i + 1] - self.z_positions[i];
            let r0 = self.r_values[i];
            let r1 = self.r_values[i + 1];
            v += dz * (r0 * r0 + r0 * r1 + r1 * r1) / 3.0; // exact for linear r(z)
        }
        PI * v
    }

    /// Drag penalty: ∫ (dr/dz)² dz — penalizes rapid area changes.
    /// Evaluated from the spline at many stations for accuracy.
    pub fn drag_penalty(&self) -> f64 {
        let z0 = self.z_positions[0];
        let z1 = *self.z_positions.last().unwrap();
        let n_eval = 200;
        let dz = (z1 - z0) / n_eval as f64;
        let eps = dz * 0.01;
        let mut d = 0.0;
        for i in 0..n_eval {
            let z = z0 + (i as f64 + 0.5) * dz;
            let dr_dz = (self.wall_radius(z + eps) - self.wall_radius(z - eps)) / (2.0 * eps);
            d += dr_dz * dr_dz * dz;
        }
        d
    }

    /// Curvature penalty: ∫ (d²r/dz²)² dz — penalizes sharp bends.
    /// Evaluated from the spline second derivative at many stations.
    pub fn curvature_penalty(&self) -> f64 {
        let z0 = self.z_positions[0];
        let z1 = *self.z_positions.last().unwrap();
        let n_eval = 200;
        let dz = (z1 - z0) / n_eval as f64;
        let eps = dz * 0.1;
        let mut c = 0.0;
        for i in 0..n_eval {
            let z = z0 + (i as f64 + 0.5) * dz;
            let r_m = self.wall_radius(z - eps);
            let r_0 = self.wall_radius(z);
            let r_p = self.wall_radius(z + eps);
            let d2r = (r_p - 2.0 * r_0 + r_m) / (eps * eps);
            c += d2r * d2r * dz;
        }
        c
    }
}

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

/// Configuration for shroud optimization.
#[derive(Debug, Clone)]
pub struct ShroudOptConfig {
    pub propeller: PropellerSource,
    pub n_harmonics: usize,
    pub nz: usize,
    pub nr: usize,
    pub r_min: f64,
    pub r_max: f64,
    /// Per-variable bound overrides: `(index_in_free, lower, upper)`.
    /// These override `r_min`/`r_max` for specific control points.
    pub per_point_bounds: Vec<(usize, f64, f64)>,
    pub max_iterations: usize,
    pub fd_step: f64,
    /// Weight for drag penalty in composite objective [dB / (m)]
    pub w_drag: f64,
    /// Weight for curvature penalty [dB / (1/m³)]
    pub w_curvature: f64,
    /// Maximum allowed volume [m³]. Exceeding this adds a quadratic penalty.
    pub v_max: f64,
    /// Volume penalty weight [dB / m⁶]
    pub w_volume: f64,
}

impl Default for ShroudOptConfig {
    fn default() -> Self {
        let prop = PropellerSource::drone_5inch(8000.0);
        let r_pipe = 0.070;
        let l = 0.12;
        // Default V_max: 120% of straight duct volume
        let v_straight = PI * r_pipe * r_pipe * l;
        Self {
            propeller: prop,
            n_harmonics: 4,
            nz: 40,
            nr: 8,
            r_min: 0.065,
            r_max: 0.110,
            per_point_bounds: Vec::new(),
            max_iterations: 30,
            fd_step: 2e-4,
            w_drag: 200.0,
            w_curvature: 0.5,
            v_max: v_straight * 1.2,
            w_volume: 1e6,
        }
    }
}

/// Optimization result.
#[derive(Debug, Clone)]
pub struct ShroudOptResult {
    pub profile: ShroudProfile,
    pub objective_history: Vec<f64>,
    pub initial_obj: f64,
    pub final_obj: f64,
    pub initial_swl: f64,
    pub final_swl: f64,
    pub iterations: usize,
}

// ---------------------------------------------------------------------------
// Acoustic objective (noise only)
// ---------------------------------------------------------------------------

fn incident_pressure_neumann_flux(wavenumber: f64, peak_pressure_pa: f64) -> Complex64 {
    Complex64::new(0.0, wavenumber * peak_pressure_pa)
}

/// A-weighted sound power level at the duct outlet.
pub fn evaluate_noise_objective(profile: &ShroudProfile, config: &ShroudOptConfig) -> f64 {
    let c = config.propeller.speed_of_sound;
    let rho = config.propeller.density;

    let z_min = profile.z_positions[0];
    let z_max = *profile.z_positions.last().unwrap();
    let inlet_area_m2 = PI * profile.wall_radius(z_min).powi(2);
    let amps = (1..=config.n_harmonics)
        .map(|h| {
            config
                .propeller
                .inlet_pressure_amplitude_for_area(h, inlet_area_m2)
        })
        .collect::<Vec<_>>();
    let mesh = duct_profile_mesh(
        z_min,
        z_max,
        |_| 0.0,
        |z| profile.wall_radius(z),
        config.nz,
        config.nr,
    );

    let assembler =
        build_axi_acoustics_assembler_with_boundaries(&mesh, PolynomialDegree::P1, &[1, 2]);
    let b_inlet = assemble_axi_boundary_mass(&mesh, PolynomialDegree::P1, 1);
    let b_outlet = assemble_axi_boundary_mass(&mesh, PolynomialDegree::P1, 2);

    let solver = FgmresConfig {
        max_iterations: 300,
        restart: 40,
        tolerance: 1e-7,
        print_interval: 0,
        exact_residual_check_interval: 0,
        convergence_check_interval: 1,
    };

    let mut total_a_power = 0.0;

    for h in 1..=config.n_harmonics {
        let freq = config.propeller.harmonic_frequency(h);
        let omega = 2.0 * PI * freq;
        let k_val = omega / c;
        let k = Complex64::new(k_val, 0.0);

        let mut bc = HashMap::new();
        bc.insert(2, Complex64::new(0.0, k_val));

        let system = assembler.assemble_helmholtz(k, &bc);
        let precond_mat = assembler.assemble_shifted_laplacian(1.0, -0.5 * k_val * k_val);
        let precond = IluPreconditioner::from_csr(&precond_mat);

        // `amps` is a peak pressure phasor. For an incident progressive
        // wave, u_n = p/(rho*c), so the Helmholtz Neumann datum is
        // i*omega*rho*u_n = i*k*p (Pa/m), not i*omega*rho*p.
        let flux = incident_pressure_neumann_flux(k_val, amps[h - 1]);
        let n = mesh.num_nodes();
        let mut rhs = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..b_inlet.nnz() {
            rhs[b_inlet.rows[i]] += flux * b_inlet.values[i];
        }

        let result = fgmres_preconditioned(&system, &precond, &rhs, None, &solver);

        if result.converged {
            let mut p2 = 0.0;
            for i in 0..b_outlet.nnz() {
                p2 += (result.x[b_outlet.rows[i]].conj()
                    * b_outlet.values[i]
                    * result.x[b_outlet.cols[i]])
                    .re;
            }
            let power = sound_power_from_boundary(p2, rho, c);
            let aw = a_weighting(freq);
            total_a_power += power * aw * aw;
        } else {
            total_a_power += 1.0; // convergence failure penalty
        }
    }

    sound_power_level(total_a_power)
}

// ---------------------------------------------------------------------------
// Composite objective (noise + drag + smoothness + volume penalty)
// ---------------------------------------------------------------------------

/// Composite objective for optimization.
///
/// `J = SWL_A + w_drag·drag + w_curv·curvature + w_vol·max(0, V-V_max)²`
pub fn evaluate_composite_objective(
    profile: &ShroudProfile,
    config: &ShroudOptConfig,
) -> (f64, f64, f64, f64, f64) {
    let swl = evaluate_noise_objective(profile, config);
    let drag = profile.drag_penalty();
    let curv = profile.curvature_penalty();
    let vol = profile.volume();
    let vol_excess = (vol - config.v_max).max(0.0);

    let obj = swl
        + config.w_drag * drag
        + config.w_curvature * curv
        + config.w_volume * vol_excess * vol_excess;

    (obj, swl, drag, curv, vol)
}

// ---------------------------------------------------------------------------
// Optimization driver
// ---------------------------------------------------------------------------

/// Run shroud shape optimization with central finite-difference gradients.
pub fn run_shroud_optimization(
    initial_profile: &ShroudProfile,
    config: &ShroudOptConfig,
) -> ShroudOptResult {
    use sonium_solvers::optimization::{Bound, LbfgsbConfig, lbfgsb};

    let n = initial_profile.n_free();
    let x0 = initial_profile.pack();

    let mut bounds: Vec<Bound<f64>> = (0..n)
        .map(|_| Bound::both(config.r_min, config.r_max))
        .collect();
    for &(idx, lo, hi) in &config.per_point_bounds {
        if idx < n {
            bounds[idx] = Bound::both(lo, hi);
        }
    }

    let lbfgsb_config = LbfgsbConfig {
        max_iterations: config.max_iterations,
        grad_tolerance: 1e-4,
        fun_tolerance: 1e-8,
        memory: 5,
        bounds,
    };

    let profile_ref = initial_profile;
    let config_ref = config;

    let objective = |x: &[f64]| -> f64 {
        let p = profile_ref.unpack(x);
        evaluate_composite_objective(&p, config_ref).0
    };

    // Central finite differences: (f(x+h) - f(x-h)) / 2h
    let gradient = |x: &[f64]| -> Vec<f64> {
        let h = config_ref.fd_step;
        let mut grad = vec![0.0; n];
        for i in 0..n {
            let mut xp = x.to_vec();
            let mut xm = x.to_vec();
            xp[i] += h;
            xm[i] -= h;
            grad[i] = (objective(&xp) - objective(&xm)) / (2.0 * h);
        }
        grad
    };

    let (initial_obj, initial_swl, _, _, _) =
        evaluate_composite_objective(&initial_profile.unpack(x0.as_slice()), config);

    let result = lbfgsb(&objective, &gradient, x0.as_slice(), &lbfgsb_config);

    match result {
        Ok(opt) => {
            let final_profile = initial_profile.unpack(&opt.x);
            let (final_obj, final_swl, _, _, _) =
                evaluate_composite_objective(&final_profile, config);

            ShroudOptResult {
                profile: final_profile,
                objective_history: Vec::new(),
                initial_obj,
                final_obj,
                initial_swl,
                final_swl,
                iterations: opt.iterations,
            }
        }
        Err(_) => ShroudOptResult {
            profile: initial_profile.clone(),
            objective_history: Vec::new(),
            initial_obj,
            final_obj: initial_obj,
            initial_swl,
            final_swl: initial_swl,
            iterations: 0,
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shroud_profile_pack_unpack() {
        let profile = ShroudProfile::straight(0.07, 0.0, 0.2, 5);
        assert_eq!(profile.n_free(), 5); // all points free

        let packed = profile.pack();
        assert_eq!(packed.len(), 5);

        let mut modified = packed.clone();
        modified[0] = 0.08; // modify inlet
        modified[2] = 0.10; // modify middle
        modified[4] = 0.06; // modify outlet
        let unpacked = profile.unpack(modified.as_slice());
        assert!((unpacked.r_values[0] - 0.08).abs() < 1e-10);
        assert!((unpacked.r_values[2] - 0.10).abs() < 1e-10);
        assert!((unpacked.r_values[4] - 0.06).abs() < 1e-10);
    }

    #[test]
    fn test_wall_radius_interpolation() {
        let mut profile = ShroudProfile::straight(0.05, 0.0, 1.0, 3);
        profile.r_values = vec![0.05, 0.10, 0.05];

        // Spline passes through control points exactly
        assert!((profile.wall_radius(0.0) - 0.05).abs() < 1e-10);
        assert!((profile.wall_radius(0.5) - 0.10).abs() < 1e-10);
        assert!((profile.wall_radius(1.0) - 0.05).abs() < 1e-10);
        // Midpoint between 0 and 0.5: spline is smooth, not exactly 0.075
        let r_quarter = profile.wall_radius(0.25);
        assert!(
            r_quarter > 0.05 && r_quarter < 0.10,
            "r(0.25) = {r_quarter}"
        );
    }

    #[test]
    fn test_volume_straight_duct() {
        let profile = ShroudProfile::straight(0.07, 0.0, 0.12, 5);
        let v = profile.volume();
        let v_exact = PI * 0.07 * 0.07 * 0.12;
        assert!((v - v_exact).abs() < 1e-10, "V={v}, expected {v_exact}");
    }

    #[test]
    fn test_drag_penalty_straight_zero() {
        let profile = ShroudProfile::straight(0.07, 0.0, 0.12, 5);
        assert!(
            profile.drag_penalty() < 1e-20,
            "Straight duct should have zero drag penalty"
        );
    }

    #[test]
    fn test_drag_penalty_expansion_nonzero() {
        let mut profile = ShroudProfile::straight(0.07, 0.0, 0.12, 5);
        profile.r_values[2] = 0.10; // expand the middle
        assert!(
            profile.drag_penalty() > 0.0,
            "Expansion should have nonzero drag penalty"
        );
    }

    #[test]
    fn test_curvature_penalty_straight_zero() {
        let profile = ShroudProfile::straight(0.07, 0.0, 0.12, 5);
        // Spline through constant values has zero curvature (up to FD noise)
        assert!(
            profile.curvature_penalty() < 1e-6,
            "Straight duct curvature penalty should be ~0, got {}",
            profile.curvature_penalty()
        );
    }

    #[test]
    fn test_evaluate_noise_objective_runs() {
        let profile = ShroudProfile::straight(0.07, 0.0, 0.15, 5);
        let config = ShroudOptConfig {
            n_harmonics: 2,
            nz: 20,
            nr: 5,
            ..Default::default()
        };
        let swl = evaluate_noise_objective(&profile, &config);
        assert!(swl.is_finite(), "SWL should be finite, got {swl}");
    }

    #[test]
    fn incident_pressure_is_converted_to_neumann_gradient() {
        let flux = incident_pressure_neumann_flux(12.0, 0.25);
        assert_eq!(flux, Complex64::new(0.0, 3.0));
    }
}
