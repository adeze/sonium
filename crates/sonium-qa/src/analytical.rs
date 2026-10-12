//! Analytical solutions for FEM QA validation
//!
//! Provides exact closed-form solutions for classical benchmark problems
//! across multiple physics domains. All functions are f64-only.
//!
//! The [`rad`] submodule (OpenCourant comparison harness) is declared here so
//! the crate root stays untouched; the crate-level glob re-export surfaces
//! it as `sonium_qa::rad`.

use num_complex::Complex64;
use std::f64::consts::PI;

#[path = "rad.rs"]
pub mod rad;

pub const EPSILON_0: f64 = 8.854_187_817e-12;

// =====================================================================
// Spherical special functions for Helmholtz scattering
// =====================================================================

/// Spherical Bessel function of the first kind, `j_n(x)`.
pub fn spherical_bessel_j(n: usize, x: f64) -> f64 {
    sonium_wave::analytical::spherical_bessel_j(n, x)
}

/// Spherical Bessel function of the second kind, `y_n(x)`.
pub fn spherical_bessel_y(n: usize, x: f64) -> f64 {
    sonium_wave::analytical::spherical_bessel_y(n, x)
}

/// Spherical Hankel function of the first kind, `h_n^(1)(x) = j_n(x) + i y_n(x)`.
pub fn spherical_hankel_h1(n: usize, x: f64) -> Complex64 {
    Complex64::new(spherical_bessel_j(n, x), spherical_bessel_y(n, x))
}

/// Derivative of `j_n(x)` using the recurrence `j_n' = j_{n-1} - (n+1)j_n/x`.
///
/// `j_n` is regular at the origin, so the derivative has a finite limit there
/// (`j_0'(0) = 0`, `j_1'(0) = 1/3`, `j_n'(0) = 0` for `n >= 2`) instead of a
/// panic: reference helpers must not abort an automated QA binary on a
/// degenerate probe.
pub fn spherical_bessel_j_derivative(n: usize, x: f64) -> f64 {
    if n == 0 {
        return -spherical_bessel_j(1, x);
    }
    if x == 0.0 {
        return if n == 1 { 1.0 / 3.0 } else { 0.0 };
    }
    spherical_bessel_j(n - 1, x) - (n as f64 + 1.0) * spherical_bessel_j(n, x) / x
}

/// Derivative of `y_n(x)` using the recurrence `y_n' = y_{n-1} - (n+1)y_n/x`.
///
/// `y_n` is genuinely singular at the origin, so this returns NaN for `x == 0`
/// (which fails QA error gates closed via non-finite propagation) instead of
/// panicking and aborting the whole binary.
pub fn spherical_bessel_y_derivative(n: usize, x: f64) -> f64 {
    if n == 0 {
        return -spherical_bessel_y(1, x);
    }
    if x == 0.0 {
        return f64::NAN;
    }
    spherical_bessel_y(n - 1, x) - (n as f64 + 1.0) * spherical_bessel_y(n, x) / x
}

/// Derivative of `h_n^(1)(x)`.
pub fn spherical_hankel_h1_derivative(n: usize, x: f64) -> Complex64 {
    Complex64::new(
        spherical_bessel_j_derivative(n, x),
        spherical_bessel_y_derivative(n, x),
    )
}

// =====================================================================
// Thermal
// =====================================================================

/// 1D steady-state conduction with Dirichlet-Dirichlet BCs.
/// T(0) = t1, T(L) = t2.
pub fn thermal_1d_dirichlet(x: f64, l: f64, t1: f64, t2: f64) -> f64 {
    t1 + (t2 - t1) * x / l
}

/// 1D steady-state conduction with uniform volumetric heat generation.
/// Dirichlet at both ends: T(0) = T(L) = t_wall.
/// q_dot [W/m³], k [W/(m·K)].
pub fn thermal_1d_heat_generation(x: f64, l: f64, k: f64, q_dot: f64, t_wall: f64) -> f64 {
    t_wall + (q_dot / (2.0 * k)) * x * (l - x)
}

/// 2D steady-state conduction on a rectangular domain [0,a]×[0,b].
/// T=0 on bottom (y=0), left (x=0), right (x=a).
/// T=T0*sin(πx/a) on top (y=b).
/// Single-term Fourier solution, smooth at all corners.
pub fn thermal_2d_rectangle_fourier(x: f64, y: f64, a: f64, b: f64, t0: f64) -> f64 {
    let sin_term = (PI * x / a).sin();
    let sinh_term = (PI * y / a).sinh();
    let sinh_b = (PI * b / a).sinh();
    t0 * sin_term * sinh_term / sinh_b
}

/// 2D steady-state conduction on a circular disk.
/// Inner radius r0 at T0, outer radius r1 at T1.
///
/// A solid disk (`r0 <= 0`) with no heat generation and a Dirichlet outer wall
/// is isothermal: regularity at the origin forces the logarithmic term to
/// vanish, so `T(r) = T1` everywhere (`t0` is then unconstrained by the steady
/// problem and ignored). Returning `t0` only at the center would be a
/// single-point discontinuity, not a solution.
pub fn thermal_2d_disk(r: f64, r0: f64, r1: f64, t0: f64, t1: f64) -> f64 {
    if r0 <= 0.0 {
        // Solid disk steady state is uniform at the outer-wall temperature.
        let _ = (r, t0);
        t1
    } else {
        // Annulus
        t0 + (t1 - t0) * (r / r0).ln() / (r1 / r0).ln()
    }
}

/// 1D transient conduction in a semi-infinite solid.
/// Initial temperature ti, surface temperature ts at t>0.
/// alpha = k/(rho*cp) [m²/s].
///
/// Reference error budget: `erf` is the Cephes implementation
/// (`spec_math::cephes64::erf`, relative error ~1e-16), so the reference is
/// exact to machine precision and any gate tolerance >> 1e-15 tests the solver,
/// not this reference.
pub fn thermal_1d_transient(x: f64, time: f64, alpha: f64, ti: f64, ts: f64) -> f64 {
    if time <= 0.0 {
        return ti;
    }
    let eta = x / (2.0 * (alpha * time).sqrt());
    ti + (ts - ti) * (1.0 - spec_math::cephes64::erf(eta))
}

// =====================================================================
// Elasticity
// =====================================================================

/// 1D bar under uniform axial load.
/// u(x) = F*x/(E*A), fixed at x=0.
pub fn elasticity_1d_bar(x: f64, f: f64, e: f64, a: f64) -> f64 {
    f * x / (e * a)
}

/// Thick-walled cylinder (Lamé solution) under internal pressure.
/// Returns (sigma_r, sigma_theta) at radius r.
/// a = inner radius, b = outer radius, p = internal pressure.
pub fn elasticity_lame_cylinder(r: f64, a: f64, b: f64, p: f64) -> (f64, f64) {
    let a2 = a * a;
    let b2 = b * b;
    let r2 = r * r;
    let denom = b2 - a2;

    let sigma_r = p * a2 / denom * (1.0 - b2 / r2);
    let sigma_theta = p * a2 / denom * (1.0 + b2 / r2);
    (sigma_r, sigma_theta)
}

/// Lamé radial displacement for thick-walled cylinder under internal pressure.
/// Plane strain: u_r = (1+nu)/E * [(1-2*nu)*A*r - B/r]
/// where A = p*a²/(b²-a²), B = p*a²*b²/(b²-a²).
pub fn elasticity_lame_displacement(r: f64, a: f64, b: f64, p: f64, e: f64, nu: f64) -> f64 {
    let a2 = a * a;
    let b2 = b * b;
    let denom = b2 - a2;
    let big_a = p * a2 / denom;
    let big_b = p * a2 * b2 / denom;
    (1.0 + nu) / e * ((1.0 - 2.0 * nu) * big_a * r - big_b / r)
}

/// Kirsch solution: infinite plate with circular hole under uniform tension.
/// Returns (sigma_r, sigma_theta, tau_rtheta) at polar coordinates (r, theta).
/// a = hole radius, sigma0 = applied tension at infinity.
pub fn elasticity_kirsch(r: f64, theta: f64, a: f64, sigma0: f64) -> (f64, f64, f64) {
    let a2 = a * a;
    let r2 = r * r;
    let a2_over_r2 = a2 / r2;
    let a4_over_r4 = a2_over_r2 * a2_over_r2;

    let cos2t = (2.0 * theta).cos();
    let sin2t = (2.0 * theta).sin();

    let sigma_r = sigma0 / 2.0 * (1.0 - a2_over_r2)
        + sigma0 / 2.0 * (1.0 - 4.0 * a2_over_r2 + 3.0 * a4_over_r4) * cos2t;
    let sigma_theta =
        sigma0 / 2.0 * (1.0 + a2_over_r2) - sigma0 / 2.0 * (1.0 + 3.0 * a4_over_r4) * cos2t;
    let tau_rtheta = -sigma0 / 2.0 * (1.0 + 2.0 * a2_over_r2 - 3.0 * a4_over_r4) * sin2t;

    (sigma_r, sigma_theta, tau_rtheta)
}

// =====================================================================
// Beam (Euler-Bernoulli)
// =====================================================================

/// Cantilever beam with end point load.
/// w(x) = F*x²*(3L - x)/(6*E*I)
pub fn beam_euler_cantilever_point(x: f64, l: f64, e: f64, i: f64, f: f64) -> f64 {
    f * x * x * (3.0 * l - x) / (6.0 * e * i)
}

/// Cantilever beam with uniformly distributed load.
/// w(x) = q*x²*(6L² - 4L*x + x²)/(24*E*I)
pub fn beam_euler_cantilever_uniform(x: f64, l: f64, e: f64, i: f64, q: f64) -> f64 {
    q * x * x * (6.0 * l * l - 4.0 * l * x + x * x) / (24.0 * e * i)
}

/// Simply supported beam with uniform load.
/// w(x) = q*x*(L³ - 2L*x² + x³)/(24*E*I)
pub fn beam_euler_ss_uniform(x: f64, l: f64, e: f64, i: f64, q: f64) -> f64 {
    q * x * (l * l * l - 2.0 * l * x * x + x * x * x) / (24.0 * e * i)
}

/// Simply supported beam with central point load.
/// For x <= L/2: w(x) = F*x*(3L² - 4x²)/(48*E*I)
pub fn beam_euler_ss_central(x: f64, l: f64, e: f64, i: f64, f: f64) -> f64 {
    let half_l = l / 2.0;
    if x <= half_l {
        f * x * (3.0 * l * l - 4.0 * x * x) / (48.0 * e * i)
    } else {
        f * (l - x) * (3.0 * l * l - 4.0 * (l - x) * (l - x)) / (48.0 * e * i)
    }
}

/// Propped cantilever (clamped at x=0, pinned at x=L) with uniform load.
/// Reaction at prop: R = 3qL/8.
/// w(x) = q*x²*(L - x)*(3L - 2x)/(48*E*I)
pub fn beam_euler_propped_uniform(x: f64, l: f64, e: f64, i: f64, q: f64) -> f64 {
    q * x * x * (l - x) * (3.0 * l - 2.0 * x) / (48.0 * e * i)
}

// =====================================================================
// Stokes
// =====================================================================

/// Plane Couette flow: u(y) = U*y/H.
pub fn stokes_couette(y: f64, h: f64, u_wall: f64) -> f64 {
    u_wall * y / h
}

/// Plane Poiseuille flow: u(y) = 4*U_max*y*(H-y)/H² = -(dp/dx)/(2μ)*y*(H-y).
pub fn stokes_plane_poiseuille(y: f64, h: f64, dpdx: f64, mu: f64) -> f64 {
    -dpdx * y * (h - y) / (2.0 * mu)
}

/// Plane Poiseuille-Couette flow: superposition.
pub fn stokes_poiseuille_couette(y: f64, h: f64, dpdx: f64, mu: f64, u_wall: f64) -> f64 {
    stokes_plane_poiseuille(y, h, dpdx, mu) + stokes_couette(y, h, u_wall)
}

/// Circular Poiseuille (Hagen-Poiseuille): u(r) = Δp*(R²-r²)/(4μL).
pub fn stokes_circular_poiseuille(r: f64, r_pipe: f64, dpdx: f64, mu: f64) -> f64 {
    -dpdx * (r_pipe * r_pipe - r * r) / (4.0 * mu)
}

/// Taylor-Couette flow: u_θ(r) = A*r + B/r.
/// omega1 at r1, omega2 at r2. Requires `0 < r1 < r2`; degenerate or inverted
/// radii return NaN (fail-closed in error gates) instead of dividing by
/// `r2² - r1² == 0` or silently flipping the sign of `a`/`b`.
pub fn stokes_taylor_couette(r: f64, r1: f64, r2: f64, omega1: f64, omega2: f64) -> f64 {
    if !r1.is_finite() || !r2.is_finite() || r1 <= 0.0 || r1 >= r2 {
        return f64::NAN;
    }
    let a = (omega2 * r2 * r2 - omega1 * r1 * r1) / (r2 * r2 - r1 * r1);
    let b = (omega1 - omega2) * r1 * r1 * r2 * r2 / (r2 * r2 - r1 * r1);
    a * r + b / r
}

// =====================================================================
// Acoustics (Helmholtz)
// =====================================================================

/// 1D standing wave in a rigid-wall duct of length L.
/// p(x) = A*cos(k*x) + B*sin(k*x).
/// For rigid-rigid: dp/dx = 0 at both ends => p(x) = cos(n*π*x/L).
pub fn acoustics_1d_duct_standing(x: f64, l: f64, n: usize) -> f64 {
    (n as f64 * PI * x / l).cos()
}

/// 2D rectangular cavity mode (rigid walls).
/// p_mn(x,y) = cos(m*π*x/Lx)*cos(n*π*y/Ly).
pub fn acoustics_2d_cavity_mode(x: f64, y: f64, lx: f64, ly: f64, m: usize, n: usize) -> f64 {
    (m as f64 * PI * x / lx).cos() * (n as f64 * PI * y / ly).cos()
}

/// 2D manufactured solution for Helmholtz on [0,Lx]×[0,Ly] with Dirichlet BCs.
/// p(x,y) = sin(kx*x)*sin(ky*y) where kx = m*π/Lx, ky = n*π/Ly.
/// Satisfies ∇²p + k²p = 0 with k² = kx² + ky².
pub fn acoustics_2d_dirichlet_standing(
    x: f64,
    y: f64,
    lx: f64,
    ly: f64,
    m: usize,
    n: usize,
) -> f64 {
    (m as f64 * PI * x / lx).sin() * (n as f64 * PI * y / ly).sin()
}

/// Eigenfrequency of a 2D rectangular cavity.
/// f_mn = (c/2)*sqrt((m/Lx)² + (n/Ly)²).
pub fn acoustics_2d_cavity_frequency(lx: f64, ly: f64, m: usize, n: usize, c: f64) -> f64 {
    let k = ((m as f64 / lx).powi(2) + (n as f64 / ly).powi(2)).sqrt() * PI;
    k * c / (2.0 * PI)
}

// =====================================================================
// Maxwell / Waveguide
// =====================================================================

/// TE_mn rectangular waveguide cutoff frequency.
/// f_c = (c/2)*sqrt((m/a)² + (n/b)²).
pub fn maxwell_rectangular_waveguide_cutoff(a: f64, b: f64, m: usize, n: usize, c: f64) -> f64 {
    (c / 2.0) * ((m as f64 / a).powi(2) + (n as f64 / b).powi(2)).sqrt()
}

/// 1D PEC cavity resonant frequency (TE/TM).
/// f_n = n*c/(2*L).
pub fn maxwell_1d_cavity_frequency(l: f64, n: usize, c: f64) -> f64 {
    n as f64 * c / (2.0 * l)
}

// =====================================================================
// Electrostatics
// =====================================================================

/// Parallel plate capacitor: E = V/d.
pub fn electrostatics_parallel_plate_field(v: f64, d: f64) -> f64 {
    v / d
}

/// Coaxial cable electric field at radius r.
/// E(r) = V/(r*ln(b/a)).
pub fn electrostatics_coaxial_field(r: f64, v: f64, a: f64, b: f64) -> f64 {
    v / (r * (b / a).ln())
}

/// Coaxial cable capacitance per unit length.
/// C' = 2πε/ln(b/a).
pub fn electrostatics_coaxial_capacitance(a: f64, b: f64, eps: f64) -> f64 {
    2.0 * PI * eps / (b / a).ln()
}

/// Conducting sphere in uniform external field E0.
/// Surface charge density σ = 3εE0*cos(θ).
pub fn electrostatics_sphere_surface_charge(e0: f64, eps: f64, theta: f64) -> f64 {
    3.0 * eps * e0 * theta.cos()
}

/// Image-method potential for a point charge above a grounded plane.
pub fn electrostatics_point_charge_image_potential(
    rho: f64,
    z: f64,
    charge_height: f64,
    charge: f64,
    eps: f64,
) -> f64 {
    let r1 = (rho * rho + (z - charge_height).powi(2)).sqrt();
    let r2 = (rho * rho + (z + charge_height).powi(2)).sqrt();
    charge / (4.0 * PI * eps) * (1.0 / r1 - 1.0 / r2)
}

/// Capacitance per unit length of two equal parallel cylinders.
pub fn electrostatics_two_cylinder_capacitance(radius: f64, center_distance: f64, eps: f64) -> f64 {
    PI * eps / ((center_distance / (2.0 * radius)).acosh())
}

// =====================================================================
// Maxwell
// =====================================================================

pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;

pub fn maxwell_pec_cavity_1d_frequency(mode: usize, length: f64) -> f64 {
    SPEED_OF_LIGHT * mode as f64 / (2.0 * length)
}

pub fn maxwell_rectangular_waveguide_te_cutoff(m: usize, n: usize, a: f64, b: f64) -> f64 {
    0.5 * SPEED_OF_LIGHT * ((m as f64 / a).powi(2) + (n as f64 / b).powi(2)).sqrt()
}

pub fn maxwell_circular_waveguide_te11_cutoff(radius: f64) -> f64 {
    let j1_prime_root = 1.841_183_781_340_659_3;
    SPEED_OF_LIGHT * j1_prime_root / (2.0 * PI * radius)
}

// =====================================================================
// Plate and shell benchmarks
// =====================================================================

pub fn plate_bending_stiffness(e: f64, nu: f64, h: f64) -> f64 {
    e * h.powi(3) / (12.0 * (1.0 - nu * nu))
}

pub fn biharmonic_ss_rectangular_sinusoidal_center(q0: f64, a: f64, b: f64, d: f64) -> f64 {
    q0 / (d * PI.powi(4) * ((1.0 / a).powi(2) + (1.0 / b).powi(2)).powi(2))
}

pub fn biharmonic_clamped_circular_center(q: f64, radius: f64, d: f64) -> f64 {
    q * radius.powi(4) / (64.0 * d)
}

pub fn biharmonic_ss_circular_point_center(load: f64, radius: f64, d: f64) -> f64 {
    load * radius.powi(2) / (16.0 * PI * d)
}

pub fn biharmonic_levy_rectangle_first_term(q: f64, a: f64, b: f64, d: f64) -> f64 {
    biharmonic_ss_rectangular_sinusoidal_center(16.0 * q / PI.powi(2), a, b, d)
}

pub fn shell_cylindrical_pressure_hoop_stress(pressure: f64, radius: f64, thickness: f64) -> f64 {
    pressure * radius / thickness
}

pub fn shell_spherical_pressure_membrane_stress(pressure: f64, radius: f64, thickness: f64) -> f64 {
    pressure * radius / (2.0 * thickness)
}

pub fn shell_cylindrical_ring_load_decay_length(radius: f64, thickness: f64) -> f64 {
    (radius * thickness).sqrt()
}

// =====================================================================
// Magnetostatics
// =====================================================================

/// Cylindrical conductor with DC current I.
/// B_θ(r) = μ0*I*r/(2π*R²) for r <= R.
pub fn magnetostatics_cylinder_internal(r: f64, i: f64, radius: f64) -> f64 {
    let mu0 = 4.0 * PI * 1e-7;
    mu0 * i * r / (2.0 * PI * radius * radius)
}

/// Cylindrical conductor with DC current I.
/// B_θ(r) = μ0*I/(2π*r) for r >= R.
pub fn magnetostatics_cylinder_external(r: f64, i: f64) -> f64 {
    let mu0 = 4.0 * PI * 1e-7;
    mu0 * i / (2.0 * PI * r)
}

/// Infinite solenoid: B_z = μ0*n*I.
pub fn magnetostatics_solenoid(n_turns_per_m: f64, i: f64) -> f64 {
    let mu0 = 4.0 * PI * 1e-7;
    mu0 * n_turns_per_m * i
}

// =====================================================================
// Manufactured solutions
// =====================================================================

/// MMS for Laplacian on unit square: φ(x,y) = sin(πx) sin(πy).
/// Satisfies φ = 0 on boundary. Source f = -∇²φ = 2π² sin(πx) sin(πy).
pub fn mms_laplacian_sin_sin(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}

/// Source term for MMS Laplacian with sin-sin solution.
pub fn mms_laplacian_sin_sin_source(x: f64, y: f64) -> f64 {
    2.0 * PI * PI * (PI * x).sin() * (PI * y).sin()
}

// =====================================================================
// Electrostatics — sphere in uniform field
// =====================================================================

/// Conducting sphere of radius `a` in uniform field `e0` (pointing in +z).
/// Potential outside the sphere: φ = -E0·r·cos(θ)·(1 - a³/r³).
/// In Cartesian: φ = -E0·z·(1 - a³/r³) where r = sqrt(x²+y²+z²).
pub fn electrostatics_sphere_uniform_field_potential(r: f64, theta: f64, a: f64, e0: f64) -> f64 {
    if r <= a {
        0.0
    } else {
        -e0 * r * theta.cos() * (1.0 - a.powi(3) / r.powi(3))
    }
}

// Batch 3 advanced mechanics helpers
// =====================================================================

/// Eddy-current skin depth `delta = sqrt(2/(omega mu sigma))`.
pub fn eddy_skin_depth(frequency: f64, mu: f64, sigma: f64) -> f64 {
    (2.0 / (2.0 * PI * frequency * mu * sigma)).sqrt()
}

/// Semi-infinite conductor field ratio `exp(-(1+i)x/delta)`.
pub fn eddy_skin_effect_field(depth: f64, skin_depth: f64) -> Complex64 {
    let a = (-depth / skin_depth).exp();
    Complex64::new(
        a * (depth / skin_depth).cos(),
        -a * (depth / skin_depth).sin(),
    )
}

/// Cylindrical AC conductor low-frequency center-to-edge field ratio scale.
pub fn eddy_cylinder_ac_low_ka(ka: f64) -> f64 {
    1.0 - ka.powi(4) / 64.0
}

/// Dodd-Deeds style lift-off envelope for filament/probe coupling.
pub fn eddy_slab_filament_liftoff(liftoff: f64, length_scale: f64) -> f64 {
    (-2.0 * liftoff / length_scale).exp()
}

/// Thin conducting sheet shielding envelope.
pub fn eddy_thin_sheet_transmission(thickness: f64, skin_depth: f64) -> f64 {
    (-thickness / skin_depth).exp()
}

/// Smooth complex MMS field for eddy-current vector potential QA.
pub fn eddy_mms_a_phi(x: f64, y: f64) -> Complex64 {
    Complex64::new(
        (PI * x).sin() * (PI * y).sin(),
        (PI * x).cos() * (PI * y).sin(),
    )
}

/// Compressible Neo-Hookean uniaxial nominal stress for diagonal F.
pub fn neo_hookean_uniaxial_nominal_stress(stretch: f64, mu: f64, lambda_lame: f64) -> f64 {
    let ln_j = stretch.ln();
    mu * stretch + (lambda_lame * ln_j - mu) / stretch
}

/// Simple-shear Neo-Hookean stress and invariants.
pub fn neo_hookean_simple_shear(gamma: f64, mu: f64) -> (f64, f64, f64) {
    let i1 = 3.0 + gamma * gamma;
    let i2 = i1;
    (mu * gamma, i1, i2)
}

/// Thin spherical membrane inflation pressure scale.
pub fn neo_hookean_sphere_inflation_pressure(
    stretch: f64,
    mu: f64,
    thickness: f64,
    radius: f64,
) -> f64 {
    2.0 * mu * thickness / radius * (stretch - stretch.powi(-5))
}

/// Thin cylindrical tube inflation pressure scale.
pub fn neo_hookean_tube_inflation_pressure(
    hoop_stretch: f64,
    mu: f64,
    thickness: f64,
    radius: f64,
) -> f64 {
    mu * thickness / radius * (hoop_stretch - hoop_stretch.powi(-3))
}

/// 1D linear isotropic hardening stress-strain law.
pub fn plasticity_1d_isotropic_hardening_stress(strain: f64, e: f64, sigma_y: f64, h: f64) -> f64 {
    let eps_y = sigma_y / e;
    if strain.abs() <= eps_y {
        e * strain
    } else {
        let sign = strain.signum();
        sign * (sigma_y + h * (strain.abs() - eps_y))
    }
}

/// Elastic-perfectly-plastic thick-cylinder collapse pressure estimate.
pub fn plasticity_thick_cylinder_limit_pressure(a: f64, b: f64, sigma_y: f64) -> f64 {
    sigma_y * (b / a).ln()
}

/// Linear kinematic hardening loop stress.
pub fn plasticity_kinematic_bauschinger_stress(strain: f64, e: f64, sigma_y: f64, c: f64) -> f64 {
    let trial = e * strain;
    if trial.abs() <= sigma_y {
        trial
    } else {
        trial.signum() * sigma_y + c * strain
    }
}

/// Piezoelectric thickness-mode resonance `f_n = n/(2L) sqrt(cD/rho)`.
pub fn piezo_thickness_mode_frequency(mode: usize, length: f64, c_d: f64, rho: f64) -> f64 {
    mode as f64 / (2.0 * length) * (c_d / rho).sqrt()
}

/// 1D piezoelectric charge from axial stress.
pub fn piezo_axial_charge(d_coeff: f64, force: f64) -> f64 {
    d_coeff * force
}

/// Simple bimorph tip-deflection scaling.
pub fn piezo_bimorph_tip_deflection(d31: f64, voltage: f64, length: f64, thickness: f64) -> f64 {
    3.0 * d31 * voltage * length * length / (2.0 * thickness * thickness)
}

/// Radially polarized cylinder capacitance per unit length.
pub fn piezo_radial_cylinder_capacitance(eps: f64, inner_radius: f64, outer_radius: f64) -> f64 {
    2.0 * PI * eps / (outer_radius / inner_radius).ln()
}

/// Hertz sphere-on-half-space contact radius.
pub fn contact_hertz_sphere_radius(force: f64, radius: f64, effective_modulus: f64) -> f64 {
    (3.0 * force * radius / (4.0 * effective_modulus)).powf(1.0 / 3.0)
}

/// Hertz cylinder line-contact half-width.
pub fn contact_hertz_cylinder_half_width(
    force_per_length: f64,
    radius: f64,
    effective_modulus: f64,
) -> f64 {
    (4.0 * force_per_length * radius / (PI * effective_modulus)).sqrt()
}

/// Flat rigid punch average pressure.
pub fn contact_rigid_punch_average_pressure(force: f64, radius: f64) -> f64 {
    force / (PI * radius * radius)
}

/// Winkler layer reaction pressure.
pub fn contact_winkler_layer_pressure(displacement: f64, modulus: f64, thickness: f64) -> f64 {
    modulus * displacement / thickness
}

/// Penalty contact pressure for a prescribed gap.
pub fn contact_penalty_pressure(gap: f64, penalty: f64) -> f64 {
    if gap < 0.0 { -penalty * gap } else { 0.0 }
}

// =====================================================================
// Batch 4 fluids and coupled-physics helpers
// =====================================================================

/// Linearized pressure-amplitude scale for subsonic flow over a sinusoidal wall.
pub fn aero_wavy_wall_pressure_amplitude(
    rho: f64,
    u_inf: f64,
    amplitude: f64,
    wavenumber: f64,
) -> f64 {
    rho * u_inf * u_inf * amplitude * wavenumber
}

/// Downstream and upstream acoustic wavenumbers for a uniform 1D mean flow.
pub fn aero_convected_duct_wavenumbers(omega: f64, c0: f64, u0: f64) -> (f64, f64) {
    (omega / (c0 + u0), omega / (c0 - u0))
}

/// Incompressible potential-flow velocity around a circular cylinder.
pub fn aero_cylinder_potential_velocity(radius: f64, r: f64, theta: f64, u_inf: f64) -> (f64, f64) {
    assert!(
        r >= radius,
        "potential-flow cylinder solution is exterior only"
    );
    let a2_over_r2 = radius * radius / (r * r);
    let u_r = u_inf * (1.0 - a2_over_r2) * theta.cos();
    let u_theta = -u_inf * (1.0 + a2_over_r2) * theta.sin();
    (u_r, u_theta)
}

/// Smooth manufactured pressure field for linearized potential/euler QA.
pub fn aero_lpe_mms_pressure(x: f64, y: f64) -> f64 {
    (PI * x).sin() * (PI * y).sin()
}

/// Laplacian of `aero_lpe_mms_pressure`.
pub fn aero_lpe_mms_laplacian(x: f64, y: f64) -> f64 {
    -2.0 * PI * PI * aero_lpe_mms_pressure(x, y)
}

/// Plane-wave pressure in a rigid tube driven by a piston velocity.
pub fn electroacoustics_piston_tube_pressure(rho: f64, c0: f64, piston_velocity: f64) -> f64 {
    rho * c0 * piston_velocity
}

/// On-axis baffled piston pressure from the Rayleigh integral for uniform velocity.
pub fn electroacoustics_piston_baffle_on_axis(
    rho: f64,
    c0: f64,
    k: f64,
    radius: f64,
    velocity: f64,
    distance: f64,
) -> Complex64 {
    let near = Complex64::from_polar(1.0, -k * distance);
    let edge_distance = (distance * distance + radius * radius).sqrt();
    let edge = Complex64::from_polar(1.0, -k * edge_distance);
    rho * c0 * velocity * (near - edge)
}

/// Surface specific radiation impedance of a pulsating sphere.
pub fn electroacoustics_pulsating_sphere_impedance(
    rho: f64,
    c0: f64,
    k: f64,
    radius: f64,
) -> Complex64 {
    let ka = k * radius;
    let denom = 1.0 + ka * ka;
    rho * c0 * Complex64::new(ka * ka / denom, ka / denom)
}

/// Dipole directivity of an oscillating sphere.
pub fn electroacoustics_oscillating_sphere_directivity(theta: f64) -> f64 {
    theta.cos()
}

/// Harmonic displacement of a mass-spring-dashpot driven by a complex force amplitude.
pub fn electroacoustics_lumped_displacement(
    force: f64,
    mass: f64,
    damping: f64,
    stiffness: f64,
    omega: f64,
) -> Complex64 {
    Complex64::new(force, 0.0) / Complex64::new(stiffness - mass * omega * omega, omega * damping)
}

/// Low-Reynolds-number 2D plane Poiseuille profile parameterized by centerline velocity.
pub fn ns_plane_poiseuille_low_re(y: f64, height: f64, u_max: f64) -> f64 {
    4.0 * u_max * y * (height - y) / (height * height)
}

/// Kovasznay exact Navier-Stokes solution at Reynolds number `re`.
pub fn ns_kovasznay_solution(x: f64, y: f64, re: f64) -> (f64, f64, f64) {
    let lambda = re / 2.0 - (re * re / 4.0 + 4.0 * PI * PI).sqrt();
    let exp_lx = (lambda * x).exp();
    let u = 1.0 - exp_lx * (2.0 * PI * y).cos();
    let v = lambda / (2.0 * PI) * exp_lx * (2.0 * PI * y).sin();
    let p = 0.5 * (1.0 - (2.0 * lambda * x).exp());
    (u, v, p)
}

/// Decaying 2D Taylor-Green vortex.
pub fn ns_taylor_green_vortex(x: f64, y: f64, time: f64, nu: f64) -> (f64, f64, f64) {
    let velocity_decay = (-2.0 * nu * time).exp();
    let pressure_decay = (-4.0 * nu * time).exp();
    let u = -x.cos() * y.sin() * velocity_decay;
    let v = x.sin() * y.cos() * velocity_decay;
    let p = -0.25 * ((2.0 * x).cos() + (2.0 * y).cos()) * pressure_decay;
    (u, v, p)
}

/// Viscous Burgers shock profile connecting `u_left` to `u_right`.
pub fn ns_burgers_viscous_shock(x: f64, time: f64, nu: f64, u_left: f64, u_right: f64) -> f64 {
    let shock_speed = 0.5 * (u_left + u_right);
    let center = x - shock_speed * time;
    0.5 * (u_left + u_right)
        - 0.5 * (u_left - u_right) * ((u_left - u_right) * center / (4.0 * nu)).tanh()
}

/// Divergence-free manufactured velocity field for 2D Navier-Stokes QA.
pub fn ns_mms_velocity(x: f64, y: f64) -> (f64, f64) {
    (
        (PI * x).sin() * (PI * y).cos(),
        -(PI * x).cos() * (PI * y).sin(),
    )
}

/// Smooth manufactured pressure field for 2D Navier-Stokes QA.
pub fn ns_mms_pressure(x: f64, y: f64) -> f64 {
    (PI * x).cos() * (PI * y).cos()
}

/// RC step response across the capacitor.
pub fn circuit_rc_step_voltage(
    time: f64,
    resistance: f64,
    capacitance: f64,
    source_voltage: f64,
) -> f64 {
    source_voltage * (1.0 - (-time / (resistance * capacitance)).exp())
}

/// Undamped series RLC resonance frequency in Hz.
pub fn circuit_rlc_resonance_frequency(inductance: f64, capacitance: f64) -> f64 {
    1.0 / (2.0 * PI * (inductance * capacitance).sqrt())
}

/// Ideal inverting op-amp voltage gain.
pub fn circuit_inverting_op_amp_gain(input_resistance: f64, feedback_resistance: f64) -> f64 {
    -feedback_resistance / input_resistance
}

/// Magnitude of a first-order RC low-pass transfer function.
pub fn circuit_rc_lowpass_magnitude(frequency: f64, resistance: f64, capacitance: f64) -> f64 {
    let omega_rc = 2.0 * PI * frequency * resistance * capacitance;
    1.0 / (1.0 + omega_rc * omega_rc).sqrt()
}

/// Wheatstone bridge output voltage between the two divider midpoints.
pub fn circuit_wheatstone_bridge_output(
    source_voltage: f64,
    r1: f64,
    r2: f64,
    r3: f64,
    r4: f64,
) -> f64 {
    source_voltage * (r2 / (r1 + r2) - r4 / (r3 + r4))
}

// =====================================================================
// Eddy Current
// =====================================================================

/// Skin depth in a conductor: δ = sqrt(2 / (ω μ σ)).
pub fn eddy_current_skin_depth(frequency: f64, mu: f64, sigma: f64) -> f64 {
    let omega = 2.0 * PI * frequency;
    (2.0 / (omega * mu * sigma)).sqrt()
}

/// Current-density amplitude decay in a semi-infinite conductor.
/// |J(x)| / |J(0)| = exp(-x/δ).
pub fn eddy_current_amplitude_decay(x: f64, delta: f64) -> f64 {
    (-x / delta).exp()
}

/// Shielding effectiveness of a thin conductive sheet (dB).
/// SE ≈ 8.686 * t / δ  for a good conductor.
pub fn eddy_current_shielding_db(thickness: f64, delta: f64) -> f64 {
    8.686 * thickness / delta
}

// =====================================================================
// Nonlinear Solid (Neo-Hookean)
// =====================================================================

/// First Piola-Kirchhoff stress P_11 for uniaxial stretch F = diag(λ,1,1).
/// P = μ·λ + (λ_mat·ln λ − μ) / λ.
pub fn neohookean_uniaxial_stress(stretch: f64, mu: f64, lambda_mat: f64) -> f64 {
    mu * stretch + (lambda_mat * stretch.ln() - mu) / stretch
}

/// Shear stress P_12 for simple shear F = [[1,γ,0],[0,1,0],[0,0,1]].
/// For Neo-Hookean (J=1): P_12 = μ·γ.
pub fn neohookean_simple_shear_stress(gamma: f64, mu: f64) -> f64 {
    mu * gamma
}

/// Invariants of the deformation gradient for simple shear (3D).
/// I1 = 3 + γ²,  I2 = 3 + γ²,  J = 1.
pub fn neohookean_simple_shear_invariants(gamma: f64) -> (f64, f64, f64) {
    let i1 = 3.0 + gamma * gamma;
    let i2 = i1; // for simple shear I1 = I2
    let j = 1.0;
    (i1, i2, j)
}

// =====================================================================
// Plasticity (1D)
// =====================================================================

/// 1D uniaxial stress for perfect plasticity.
/// σ = E·ε  if |E·ε| ≤ σ_y,  else σ = σ_y·sign(E·ε).
pub fn plasticity_1d_perfect_stress(strain: f64, e: f64, sigma_y: f64) -> f64 {
    let sigma_trial = e * strain;
    if sigma_trial.abs() <= sigma_y {
        sigma_trial
    } else {
        sigma_y * sigma_trial.signum()
    }
}

/// 1D uniaxial stress for linear isotropic hardening.
/// σ = E·ε  if elastic,  else σ = sign(ε)·(σ_y + H·( |ε| − σ_y/E )).
pub fn plasticity_1d_hardening_stress(strain: f64, e: f64, sigma_y: f64, h: f64) -> f64 {
    let sigma_trial = e * strain;
    if sigma_trial.abs() <= sigma_y {
        sigma_trial
    } else {
        let sign = sigma_trial.signum();
        sign * (sigma_y + h * (strain.abs() - sigma_y / e))
    }
}

// =====================================================================
// Piezoelectric
// =====================================================================

/// Longitudinal bar resonance frequency (thickness mode, n=1).
/// f = n/(2L) · sqrt(cᴱ/ρ).
pub fn piezoelectric_bar_resonance(length: f64, stiffness: f64, density: f64) -> f64 {
    stiffness.sqrt() / (2.0 * length * density.sqrt())
}

/// Piezoelectric strain constant d₃₃ = e₃₃ / cᴱ₃₃.
pub fn piezoelectric_d33(e33: f64, c33: f64) -> f64 {
    e33 / c33
}

/// Free elongation of a d₃₃ rod under voltage V.
/// ΔL = d₃₃ · V · L / t.
pub fn piezoelectric_free_elongation(d33: f64, voltage: f64, length: f64, thickness: f64) -> f64 {
    d33 * voltage * length / thickness
}

/// Blocked stress in a d₃₃ rod under voltage V.
/// T_3 = −e₃₃ · E_3 = −e₃₃ · V / t.
pub fn piezoelectric_blocked_stress(e33: f64, voltage: f64, thickness: f64) -> f64 {
    -e33 * voltage / thickness
}

/// Permittivity at constant stress: εᵀ = εˢ + e² / cᴱ.
pub fn piezoelectric_permittivity_stress(eps_s: f64, e: f64, c: f64) -> f64 {
    eps_s + e * e / c
}

// =====================================================================
// Solids Phase A: beams, torsion, buckling, plates, contact
// =====================================================================

/// Tip deflection of a Timoshenko cantilever under a tip load.
/// w = F*L³/(3EI) + F*L/(kappa*G*A).
pub fn beam_timoshenko_cantilever_tip(
    load: f64,
    l: f64,
    e: f64,
    i: f64,
    g: f64,
    a: f64,
    kappa: f64,
) -> f64 {
    load * l.powi(3) / (3.0 * e * i) + load * l / (kappa * g * a)
}

/// Tip slope of an Euler-Bernoulli cantilever under a tip load.
/// theta = F*L²/(2EI).
pub fn beam_euler_cantilever_tip_slope(l: f64, e: f64, i: f64, f: f64) -> f64 {
    f * l * l / (2.0 * e * i)
}

/// Prop reaction of a propped cantilever (clamped x=0, pinned x=L) with a
/// point load P at station a: R = P*a²*(3L-a)/(2L³).
pub fn beam_euler_propped_point_reaction(p: f64, a: f64, l: f64) -> f64 {
    p * a * a * (3.0 * l - a) / (2.0 * l.powi(3))
}

/// Deflection of a propped cantilever with a point load P at station a.
///
/// Superposes the released-cantilever response with the redundant prop
/// reaction acting at x = L.
pub fn beam_euler_propped_point(x: f64, p: f64, a: f64, l: f64, e: f64, i: f64) -> f64 {
    let released = if x <= a {
        p * x * x * (3.0 * a - x)
    } else {
        p * a * a * (3.0 * x - a)
    } / (6.0 * e * i);
    released - beam_euler_cantilever_point(x, l, e, i, beam_euler_propped_point_reaction(p, a, l))
}

/// Twist of a circular shaft under pure torsion: phi = T*L/(G*J).
pub fn torsion_shaft_twist(torque: f64, l: f64, g: f64, j: f64) -> f64 {
    torque * l / (g * j)
}

/// Torsional constant of a solid circular shaft: J = pi*r⁴/2.
pub fn torsion_shaft_polar_moment(radius: f64) -> f64 {
    PI * radius.powi(4) / 2.0
}

/// Peak shear stress in a circular shaft: tau = T*r/J.
pub fn torsion_shaft_max_shear(torque: f64, radius: f64, j: f64) -> f64 {
    torque * radius / j
}

/// Euler buckling load: P_cr = pi²*E*I/(K*L)².
pub fn column_euler_critical_load(e: f64, i: f64, l: f64, k_factor: f64) -> f64 {
    let effective = k_factor * l;
    PI.powi(2) * e * i / effective.powi(2)
}

/// Complete elliptic integral of the first kind K(k) via the AGM.
///
/// Converges quadratically; the loop exits at machine precision for all
/// `|k| < 1`. Returns NaN for `|k| >= 1` (fail-closed in QA gates).
pub fn elliptic_k_first_kind(k: f64) -> f64 {
    if !k.is_finite() || k.abs() >= 1.0 {
        return f64::NAN;
    }
    let mut a = 1.0;
    let mut b = (1.0 - k * k).sqrt();
    for _ in 0..32 {
        let next_a = 0.5 * (a + b);
        let next_b = (a * b).sqrt();
        a = next_a;
        b = next_b;
        if (a - b).abs() < 1e-16 * a {
            break;
        }
    }
    PI / (2.0 * a)
}

/// Exact pinned-column elastica post-buckling point at modulus k.
///
/// Returns `(P/P_cr, delta/L)` with `P/P_cr = (2K/pi)²` and
/// `delta/L = k/K`, where `K = K(k)` (Timoshenko, Elastic Stability).
pub fn column_elastica_postbuckling(k: f64) -> (f64, f64) {
    let big_k = elliptic_k_first_kind(k);
    (4.0 * big_k * big_k / PI.powi(2), k / big_k)
}

/// Von Mises (two-bar) truss apex load for a downward apex drop w.
///
/// Symmetric bars with horizontal half-span `b`, initial apex height `h`,
/// axial stiffness `E*A`: `P = 2*E*A*(L-l)/L*h'/l` with current height
/// `h' = h - w` and current bar length `l = sqrt(b² + h'²)`. Positive `P`
/// pushes the apex down; compressed bars push it back up.
pub fn truss_von_mises_load(w: f64, b: f64, h: f64, e: f64, a: f64) -> f64 {
    let l0 = (b * b + h * h).sqrt();
    let h_now = h - w;
    let l_now = (b * b + h_now * h_now).sqrt();
    2.0 * e * a * (l0 - l_now) / l0 * h_now / l_now
}

/// Clamped circular plate under uniform pressure: full field.
/// w(r) = q*(a²-r²)²/(64D).
pub fn plate_clamped_circular_deflection(r: f64, q: f64, radius: f64, d: f64) -> f64 {
    q * (radius * radius - r * r).powi(2) / (64.0 * d)
}

/// Simply supported circular plate under uniform pressure: center.
/// w(0) = q*a⁴*(5+nu)/(64*(1+nu)*D).
pub fn plate_ss_circular_center_uniform(q: f64, radius: f64, d: f64, nu: f64) -> f64 {
    q * radius.powi(4) * (5.0 + nu) / (64.0 * (1.0 + nu) * d)
}

/// Simply supported circular plate under a center point load: center.
/// w(0) = P*a²*(3+nu)/(16*pi*(1+nu)*D).
pub fn plate_ss_circular_center_point(load: f64, radius: f64, d: f64, nu: f64) -> f64 {
    load * radius.powi(2) * (3.0 + nu) / (16.0 * PI * (1.0 + nu) * d)
}

/// Thick-walled cylinder under external pressure only.
/// Returns (sigma_r, sigma_theta) at radius r.
pub fn elasticity_thick_cylinder_external(r: f64, a: f64, b: f64, p: f64) -> (f64, f64) {
    let a2 = a * a;
    let b2 = b * b;
    let r2 = r * r;
    let scale = p * b2 / (b2 - a2);
    (-scale * (1.0 - a2 / r2), -scale * (1.0 + a2 / r2))
}

/// Axial stress of a capped thick cylinder.
/// sigma_z = (p_i*a² - p_o*b²)/(b²-a²).
pub fn elasticity_thick_cylinder_axial(a: f64, b: f64, pi: f64, po: f64) -> f64 {
    (pi * a * a - po * b * b) / (b * b - a * a)
}

/// Hertz effective modulus: 1/E* = (1-nu1²)/E1 + (1-nu2²)/E2.
pub fn contact_hertz_effective_modulus(e1: f64, nu1: f64, e2: f64, nu2: f64) -> f64 {
    1.0 / ((1.0 - nu1 * nu1) / e1 + (1.0 - nu2 * nu2) / e2)
}

/// Hertz sphere peak pressure: p0 = 3F/(2*pi*a²).
pub fn contact_hertz_sphere_peak_pressure(force: f64, a: f64) -> f64 {
    3.0 * force / (2.0 * PI * a * a)
}

/// Hertz sphere pressure profile: p(r) = p0*sqrt(1-(r/a)²), zero outside.
pub fn contact_hertz_sphere_pressure(r: f64, p0: f64, a: f64) -> f64 {
    if r.abs() >= a {
        return 0.0;
    }
    let ratio = r / a;
    p0 * (1.0 - ratio * ratio).sqrt()
}

/// Hertz sphere normal approach: delta = a²/R.
pub fn contact_hertz_sphere_approach(a: f64, radius: f64) -> f64 {
    a * a / radius
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn thermal_1d_dirichlet_midpoint() {
        let t = thermal_1d_dirichlet(0.5, 1.0, 100.0, 200.0);
        assert!((t - 150.0).abs() < 1e-12);
    }

    #[test]
    fn thermal_1d_heat_gen_max_at_midpoint() {
        let l = 1.0;
        let k = 1.0;
        let q = 10.0;
        let t_wall = 0.0;
        let t_mid = thermal_1d_heat_generation(l / 2.0, l, k, q, t_wall);
        let expected = q * l * l / (8.0 * k);
        assert!((t_mid - expected).abs() < 1e-12);
    }

    #[test]
    fn beam_cantilever_point_tip() {
        let l = 1.0;
        let e = 200e9;
        let i = 1e-8;
        let f = 1000.0;
        let w_tip = beam_euler_cantilever_point(l, l, e, i, f);
        let expected = f * l * l * l / (3.0 * e * i);
        assert!((w_tip - expected).abs() < 1e-12);
    }

    #[test]
    fn stokes_couette_linear() {
        let h = 1.0;
        let u = 2.0;
        assert!((stokes_couette(0.0, h, u) - 0.0).abs() < 1e-12);
        assert!((stokes_couette(h, h, u) - u).abs() < 1e-12);
        assert!((stokes_couette(h / 2.0, h, u) - u / 2.0).abs() < 1e-12);
    }

    #[test]
    fn stokes_poiseuille_parabolic_max() {
        let h = 1.0;
        let dpdx = -8.0;
        let mu = 1.0;
        let u_max = stokes_plane_poiseuille(h / 2.0, h, dpdx, mu);
        assert!((u_max - 1.0).abs() < 1e-12);
    }

    #[test]
    fn electrostatics_coaxial_capacitance_formula() {
        let a = 1e-3;
        let b = 5e-3;
        let eps = 8.854e-12;
        let c = electrostatics_coaxial_capacitance(a, b, eps);
        let expected = 2.0 * PI * eps / (b / a).ln();
        assert!((c - expected).abs() < 1e-20);
    }

    #[test]
    fn acoustics_cavity_mode_00() {
        let p = acoustics_2d_cavity_mode(0.3, 0.7, 1.0, 1.0, 0, 0);
        assert!((p - 1.0).abs() < 1e-12);
    }

    #[test]
    fn bessel_derivatives_are_defined_at_the_origin() {
        // j_n' limits: j_0'(0) = 0, j_1'(0) = 1/3, j_n'(0) = 0 for n >= 2.
        assert!((spherical_bessel_j_derivative(0, 0.0) - 0.0).abs() < 1e-15);
        assert!((spherical_bessel_j_derivative(1, 0.0) - 1.0 / 3.0).abs() < 1e-15);
        assert!((spherical_bessel_j_derivative(2, 0.0) - 0.0).abs() < 1e-15);
        assert!((spherical_bessel_j_derivative(5, 0.0) - 0.0).abs() < 1e-15);
        // Matches the recurrence just off the origin (continuity check).
        for n in 0..4 {
            let near = spherical_bessel_j_derivative(n, 1e-8);
            let at = spherical_bessel_j_derivative(n, 0.0);
            assert!((near - at).abs() < 1e-6, "n={n}: near={near}, at={at}");
        }
        // y_n' is genuinely singular: NaN (fail-closed), never a panic.
        assert!(spherical_bessel_y_derivative(1, 0.0).is_nan());
        assert!(spherical_bessel_y_derivative(3, 0.0).is_nan());
    }

    #[test]
    fn transient_reference_uses_machine_precision_erf() {
        // erf(1) = 0.84270079294971486934...; the old A&S 7.1.26 fit erred ~1e-7.
        assert!((spec_math::cephes64::erf(1.0) - 0.8427007929497149).abs() < 1e-15);
        // Surface node must equal ts, far field must relax to ti.
        let t_surface = thermal_1d_transient(0.0, 1.0, 1e-5, 100.0, 0.0);
        assert!((t_surface - 0.0).abs() < 1e-12);
        let t_far = thermal_1d_transient(10.0, 1.0, 1e-5, 100.0, 0.0);
        assert!((t_far - 100.0).abs() < 1e-9);
    }

    #[test]
    fn solid_disk_reference_is_isothermal() {
        // Steady solid disk with outer Dirichlet T1: uniform T1, no center spike.
        for r in [0.0, 1e-13, 0.25, 0.9, 1.0] {
            assert!((thermal_2d_disk(r, 0.0, 1.0, 300.0, 100.0) - 100.0).abs() < 1e-15);
        }
        // Annulus branch is unchanged: honors both wall temperatures.
        assert!((thermal_2d_disk(0.5, 0.5, 1.0, 300.0, 100.0) - 300.0).abs() < 1e-12);
        assert!((thermal_2d_disk(1.0, 0.5, 1.0, 300.0, 100.0) - 100.0).abs() < 1e-12);
    }

    #[test]
    fn taylor_couette_rejects_degenerate_radii() {
        assert!(stokes_taylor_couette(1.5, 1.0, 1.0, 1.0, 0.0).is_nan());
        assert!(stokes_taylor_couette(1.5, 2.0, 1.0, 1.0, 0.0).is_nan());
        assert!(stokes_taylor_couette(1.5, 0.0, 1.0, 1.0, 0.0).is_nan());
        assert!(stokes_taylor_couette(1.5, -1.0, 1.0, 1.0, 0.0).is_nan());
        // Rigid-body rotation (omega1 == omega2) recovers u = omega*r.
        let u = stokes_taylor_couette(1.5, 1.0, 2.0, 2.0, 2.0);
        assert!((u - 3.0).abs() < 1e-12);
        // No-slip at both walls for differential rotation.
        assert!((stokes_taylor_couette(1.0, 1.0, 2.0, 1.0, 0.5) - 1.0).abs() < 1e-12);
        assert!((stokes_taylor_couette(2.0, 1.0, 2.0, 1.0, 0.5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn timoshenko_cantilever_adds_shear_to_euler_bernoulli() {
        let (f, l, e, i) = (100.0, 1.0, 200e9, 1e-8);
        let euler = beam_euler_cantilever_point(l, l, e, i, f);
        let timoshenko = beam_timoshenko_cantilever_tip(f, l, e, i, 79e9, 1e-4, 5.0 / 6.0);
        let shear = f * l / ((5.0 / 6.0) * 79e9 * 1e-4);
        assert!((timoshenko - euler - shear).abs() < 1e-15);
        assert!(
            (beam_euler_cantilever_tip_slope(l, e, i, f) - f * l * l / (2.0 * e * i)).abs() < 1e-18
        );
    }

    #[test]
    fn propped_point_load_satisfies_pin_and_clamp() {
        let (p, a, l, e, i) = (800.0, 0.9, 1.5, 200e9, 1e-8);
        let reaction = beam_euler_propped_point_reaction(p, a, l);
        assert!((reaction - p * a * a * (3.0 * l - a) / (2.0 * l.powi(3))).abs() < 1e-9);
        assert!(beam_euler_propped_point(0.0, p, a, l, e, i).abs() < 1e-15);
        assert!(beam_euler_propped_point(l, p, a, l, e, i).abs() < 1e-12);
        // Continuity of deflection and slope at the load station.
        let h = 1e-7;
        let left = beam_euler_propped_point(a - h, p, a, l, e, i);
        let right = beam_euler_propped_point(a + h, p, a, l, e, i);
        assert!((left - right).abs() < 1e-9);
    }

    #[test]
    fn torsion_and_buckling_match_textbook_values() {
        let j = torsion_shaft_polar_moment(0.02);
        assert!((j - PI * 0.02f64.powi(4) / 2.0).abs() < 1e-18);
        assert!(
            (torsion_shaft_twist(500.0, 1.2, 79e9, j) - 500.0 * 1.2 / (79e9 * j)).abs() < 1e-15
        );
        assert!((torsion_shaft_max_shear(500.0, 0.02, j) - 500.0 * 0.02 / j).abs() < 1e-9);
        let pcr = column_euler_critical_load(200e9, 1e-8, 1.0, 1.0);
        assert!((pcr - PI.powi(2) * 200e9 * 1e-8).abs() < 1e-6);
        assert!((column_euler_critical_load(200e9, 1e-8, 1.0, 2.0) - pcr / 4.0).abs() < 1e-6);
        assert!((column_euler_critical_load(200e9, 1e-8, 1.0, 0.5) - 4.0 * pcr).abs() < 1e-6);
    }

    #[test]
    fn elastica_recovers_euler_load_at_zero_deflection() {
        // K(0) = pi/2 exactly; small-k series P/Pcr = 1 + k²/2.
        assert!((elliptic_k_first_kind(0.0) - PI / 2.0).abs() < 1e-15);
        assert!((elliptic_k_first_kind(0.5) - 1.685750354812596).abs() < 1e-12);
        assert!(elliptic_k_first_kind(1.0).is_nan());
        let (ratio, delta) = column_elastica_postbuckling(1e-4);
        assert!((ratio - 1.0).abs() < 1e-7);
        assert!(delta.abs() < 1e-3);
        // Timoshenko Table 2-1 style: k = 0.5 gives P/Pcr = 1.1522...
        let (ratio, delta) = column_elastica_postbuckling(0.5);
        assert!((ratio - 4.0 * 1.685750354812596f64.powi(2) / PI.powi(2)).abs() < 1e-12);
        assert!((delta - 0.5 / 1.685750354812596).abs() < 1e-12);
        assert!(ratio > 1.0);
    }

    #[test]
    fn von_mises_truss_shows_limit_point_and_unloads() {
        // Shallow truss: b = 1, h = 0.1, EA = 1. Symmetric path P(0) = 0,
        // P(2h) = 0 (mirror position), with a limit point in between.
        let (b, h, e, a) = (1.0, 0.1, 1.0, 1.0);
        assert!(truss_von_mises_load(0.0, b, h, e, a).abs() < 1e-15);
        assert!(truss_von_mises_load(2.0 * h, b, h, e, a).abs() < 1e-12);
        let mut peak = 0.0;
        let mut peak_w = 0.0;
        let mut w = 0.0;
        while w <= 2.0 * h {
            let p = truss_von_mises_load(w, b, h, e, a);
            if p > peak {
                peak = p;
                peak_w = w;
            }
            w += 0.0005;
        }
        assert!(peak > 0.0, "must carry load before snap-through");
        assert!(
            peak_w > 0.0 && peak_w < h,
            "limit point precedes the flat position"
        );
        // Past the flat position the truss pulls down (negative load).
        assert!(truss_von_mises_load(1.5 * h, b, h, e, a) < 0.0);
    }

    #[test]
    fn circular_plates_and_vessels_match_textbook_values() {
        let d = plate_bending_stiffness(200e9, 0.3, 0.01);
        let clamped = plate_clamped_circular_deflection(0.0, 1e5, 0.5, d);
        assert!((clamped - 1e5 * 0.5f64.powi(4) / (64.0 * d)).abs() < 1e-15);
        assert!(plate_clamped_circular_deflection(0.5, 1e5, 0.5, d).abs() < 1e-15);
        let ss = plate_ss_circular_center_uniform(1e5, 0.5, d, 0.3);
        assert!((ss / clamped - (5.0 + 0.3) / (1.0 + 0.3)).abs() < 1e-12);
        let ss_p = plate_ss_circular_center_point(1000.0, 0.5, d, 0.3);
        let clamped_p = biharmonic_ss_circular_point_center(1000.0, 0.5, d);
        assert!(ss_p > clamped_p, "SS plate must deflect more than clamped");
        assert!((ss_p / clamped_p - (3.0 + 0.3) / (1.0 + 0.3)).abs() < 1e-12);
    }

    #[test]
    fn thick_cylinder_external_and_hertz_match_textbook_values() {
        let (sr, _) = elasticity_thick_cylinder_external(0.2, 0.1, 0.2, 5e6);
        assert!((sr + 5e6).abs() < 1e-6);
        let (_, st) = elasticity_thick_cylinder_external(0.1, 0.1, 0.2, 5e6);
        assert!((st + 2.0 * 5e6 * 0.2f64.powi(2) / (0.2f64.powi(2) - 0.1f64.powi(2))).abs() < 1e-6);
        assert!(
            (elasticity_thick_cylinder_axial(0.1, 0.2, 10e6, 0.0) - 10e6 * 0.01 / 0.03).abs()
                < 1e-6
        );
        let estar = contact_hertz_effective_modulus(200e9, 0.3, 200e9, 0.3);
        assert!((estar - 200e9 / (2.0 * 0.91)).abs() < 1e3);
        let a = contact_hertz_sphere_radius(1000.0, 0.05, estar);
        let p0 = contact_hertz_sphere_peak_pressure(1000.0, a);
        assert!((contact_hertz_sphere_pressure(0.0, p0, a) - p0).abs() < 1e-9);
        assert!(contact_hertz_sphere_pressure(a, p0, a).abs() < 1e-15);
        assert!((contact_hertz_sphere_approach(a, 0.05) - a * a / 0.05).abs() < 1e-15);
    }
}
