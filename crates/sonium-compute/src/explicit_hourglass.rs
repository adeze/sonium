//! Flanagan-Belytschko hourglass control and shock viscosity.
//!
//! Phase 1 lane 1C kernels for the explicit path: orthogonal hourglass
//! stabilization of the under-integrated HEXA8 brick and the
//! von Neumann-Richtmyer artificial bulk viscosity for shock capturing.
//! All kernels are dependency-free `f64` slice code in SI units; they take
//! plain slices (never sparse matrices or FEM types) per contract P1C2.
//!
//! # Node and mode ordering
//!
//! HEXA8 nodes carry natural coordinates `(xi, eta, zeta)` in `{-1, +1}^3`:
//!
//! ```text
//! node:  0          1          2          3          4          5          6          7
//! xi:   -1         +1         +1         -1         -1         +1         +1         -1
//! eta:  -1         -1         +1         +1         -1         -1         +1         +1
//! zeta: -1         -1         -1         -1         +1         +1         +1         +1
//! ```
//!
//! Coordinates are node-major (`[x0, y0, z0, ...]`, length 24) matching
//! [`ExplicitState`](crate::explicit_state::ExplicitState). The four
//! hourglass modes are `(xi*eta, eta*zeta, zeta*xi, xi*eta*zeta)` and the
//! Gamma output is mode-major (`gamma[a * 8 + i]`, length 32).
//!
//! # Theory references
//!
//! - Hourglass: Altair Radioss 2022 Theory Manual, Element Library,
//!   "Hourglass Modes" / "Flanagan-Belytschko Formulation" (pp. 86-89,
//!   Eqs 327-331) with viscous resisting forces (Eq 326); Flanagan and
//!   Belytschko, "A Uniform Strain Hexahedron and Quadrilateral with
//!   Orthogonal Hourglass Control", IJNME 17, 1981.
//! - Bulk viscosity: same manual, "Shock Waves" (pp. 91-92, Eq 340);
//!   von Neumann and Richtmyer, J. Applied Physics, 1950.
//!
//! # Examples
//!
//! ```rust
//! use sonium_compute::explicit_hourglass::{bulk_viscosity_q, hexa8_gamma};
//!
//! let coords = [
//!     0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
//!     0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
//! ];
//! let mut gamma = [0.0; 32];
//! hexa8_gamma(&coords, &mut gamma);
//! assert_eq!(gamma[0], 1.0);
//!
//! let q = bulk_viscosity_q(7800.0, 4500.0, 0.01, 1.1, 0.05, -1000.0);
//! assert!(q > 0.0);
//! ```

// Rust guideline compliant 2026-02-21

/// HEXA8 natural node coordinates `(xi, eta, zeta)` in lane ordering.
const NATURAL_COORDS: [[f64; 3]; 8] = [
    [-1.0, -1.0, -1.0],
    [1.0, -1.0, -1.0],
    [1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [1.0, -1.0, 1.0],
    [1.0, 1.0, 1.0],
    [-1.0, 1.0, 1.0],
];

/// Flanagan-Belytschko hourglass base modes `(xi*eta, eta*zeta, zeta*xi, xi*eta*zeta)`.
///
/// Exact `+/-1` entries. On any parallelepiped element the orthogonalized
/// [`hexa8_gamma`] output equals these modes bit-for-bit.
pub const HEXA8_HOURGLASS_BASE: [[f64; 8]; 4] = [
    [1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0],
    [1.0, 1.0, -1.0, -1.0, -1.0, -1.0, 1.0, 1.0],
    [1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0],
    [-1.0, 1.0, -1.0, 1.0, 1.0, -1.0, 1.0, -1.0],
];

/// Manual default for the quadratic shock-viscosity coefficient.
///
/// Radioss Theory Manual "Shock Waves": scalar factor on the quadratic
/// viscosity adjusted so the Hugoniot jump conditions hold; default 1.10.
pub const BULK_VISCOSITY_QUADRATIC_DEFAULT: f64 = 1.10;

/// Manual default for the linear shock-viscosity coefficient.
///
/// Radioss Theory Manual "Shock Waves": scalar factor on the linear
/// viscosity damping oscillations behind the shock; default 0.05.
pub const BULK_VISCOSITY_LINEAR_DEFAULT: f64 = 0.05;

/// Orthogonal hourglass vectors of one HEXA8 element.
///
/// Computes the four Flanagan-Belytschko Gamma vectors `Gamma[a, I] =
/// h[a, I] - sum_j (h[a] . x[j]) * b[j, I]`, where `h` are the base
/// modes, `x[j]` the nodal coordinate vectors, and `b[j, I]` the
/// centroid shape-function gradients from the uniform-strain operator.
/// Each Gamma is orthogonal to the constant and linear fields, so the
/// hourglass forces built from it do zero work on rigid-body modes and
/// constant-strain states (Radioss Theory Manual Eqs 327-331).
///
/// `coords` holds 8 node-major positions (length 24, meters); `gamma`
/// receives the 4 mode-major Gamma vectors (length 32).
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_hourglass::hexa8_gamma;
///
/// let coords = [
///     0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0,
///     0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
/// ];
/// let mut gamma = [0.0; 32];
/// hexa8_gamma(&coords, &mut gamma);
/// // Unit cube: Gamma equals the exact base modes.
/// assert_eq!(gamma[0], 1.0);
/// assert_eq!(gamma[1], -1.0);
/// ```
///
/// # Panics
///
/// Panics if either slice has the wrong length, any coordinate is not
/// finite, or the centroid Jacobian determinant is not positive finite
/// (degenerate or inverted element).
pub fn hexa8_gamma(coords: &[f64], gamma: &mut [f64]) {
    assert_eq!(coords.len(), 24, "hexa8_gamma needs 8 node-major coords");
    assert_eq!(gamma.len(), 32, "hexa8_gamma needs 4x8 Gamma output");
    assert!(
        coords.iter().all(|v| v.is_finite()),
        "hexa8_gamma needs finite coords"
    );

    // Centroid Jacobian J[j, m] = (1/8) sum_I nat[I, m] * x[j, I].
    let mut jac = [[0.0; 3]; 3];
    for (node, nat) in NATURAL_COORDS.iter().enumerate() {
        for m in 0..3 {
            for j in 0..3 {
                jac[j][m] += nat[m] * coords[3 * node + j];
            }
        }
    }
    for row in jac.iter_mut() {
        for value in row.iter_mut() {
            *value *= 0.125;
        }
    }

    let det = jac[0][0] * (jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1])
        - jac[0][1] * (jac[1][0] * jac[2][2] - jac[1][2] * jac[2][0])
        + jac[0][2] * (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0]);
    assert!(
        det.is_finite() && det > 0.0,
        "hexa8_gamma needs a non-degenerate element"
    );

    // Adjugate inverse inv[m, j] (cofactor transpose over det).
    let inv = [
        [
            (jac[1][1] * jac[2][2] - jac[1][2] * jac[2][1]) / det,
            (jac[0][2] * jac[2][1] - jac[0][1] * jac[2][2]) / det,
            (jac[0][1] * jac[1][2] - jac[0][2] * jac[1][1]) / det,
        ],
        [
            (jac[1][2] * jac[2][0] - jac[1][0] * jac[2][2]) / det,
            (jac[0][0] * jac[2][2] - jac[0][2] * jac[2][0]) / det,
            (jac[0][2] * jac[1][0] - jac[0][0] * jac[1][2]) / det,
        ],
        [
            (jac[1][0] * jac[2][1] - jac[1][1] * jac[2][0]) / det,
            (jac[0][1] * jac[2][0] - jac[0][0] * jac[2][1]) / det,
            (jac[0][0] * jac[1][1] - jac[0][1] * jac[1][0]) / det,
        ],
    ];

    // Centroid gradients b[j, I] = (1/8) sum_m nat[I, m] * inv[m, j].
    let mut grad = [[0.0; 8]; 3];
    for (node, nat) in NATURAL_COORDS.iter().enumerate() {
        for j in 0..3 {
            let mut sum = 0.0;
            for m in 0..3 {
                sum += nat[m] * inv[m][j];
            }
            grad[j][node] = 0.125 * sum;
        }
    }

    // Projections c[a, j] = sum_J h[a, J] * x[j, J], then orthogonalize.
    for (a, base) in HEXA8_HOURGLASS_BASE.iter().enumerate() {
        let mut proj = [0.0; 3];
        for j in 0..3 {
            let mut sum = 0.0;
            for (node, h) in base.iter().enumerate() {
                sum += h * coords[3 * node + j];
            }
            proj[j] = sum;
        }
        for (node, h) in base.iter().enumerate() {
            gamma[a * 8 + node] =
                h - (proj[0] * grad[0][node] + proj[1] * grad[1][node] + proj[2] * grad[2][node]);
        }
    }
}

/// Hourglass nodal forces from Gamma vectors and modal rates.
///
/// Evaluates `f[i, I] = coeff * sum_a Gamma[a, I] * sum_J Gamma[a, J] *
/// rates[i, J]`: the stiffness form when `rates` holds displacements
/// (coefficient in N/m) or the viscous form when `rates` holds
/// velocities (coefficient in N*s/m, Radioss Theory Manual Eq 326).
/// The viscous power `P = v^T f = coeff * sum(q^2) >= 0` dissipates
/// hourglass energy and vanishes on rigid-body modes by Gamma
/// orthogonality.
///
/// `gamma` holds 4 mode-major vectors (length 32), `rates` 8 node-major
/// rate vectors (length 24); `forces` (length 24, newtons) is
/// overwritten, so callers scatter-add it into the global residual.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_hourglass::{
///     HEXA8_HOURGLASS_BASE, hexa8_hourglass_forces,
/// };
///
/// let gamma: Vec<f64> =
///     HEXA8_HOURGLASS_BASE.iter().flatten().copied().collect();
/// // Rigid translation: hourglass forces vanish exactly.
/// let rates = [2.0; 24];
/// let mut forces = [1.0; 24];
/// hexa8_hourglass_forces(&gamma, &rates, 0.1, &mut forces);
/// assert!(forces.iter().all(|f| *f == 0.0));
/// ```
///
/// # Panics
///
/// Panics if any slice has the wrong length, any Gamma or rate entry
/// is not finite, or the coefficient is negative or not finite.
pub fn hexa8_hourglass_forces(gamma: &[f64], rates: &[f64], coeff: f64, forces: &mut [f64]) {
    assert_eq!(gamma.len(), 32, "hourglass forces need 4x8 Gamma");
    assert_eq!(rates.len(), 24, "hourglass forces need 8 nodal rates");
    assert_eq!(forces.len(), 24, "hourglass forces need 8 nodal forces");
    assert!(
        gamma.iter().all(|v| v.is_finite()),
        "hourglass forces need finite Gamma"
    );
    assert!(
        rates.iter().all(|v| v.is_finite()),
        "hourglass forces need finite rates"
    );
    assert!(
        coeff.is_finite() && coeff >= 0.0,
        "hourglass forces need a non-negative finite coefficient"
    );

    for i in 0..3 {
        let mut modal = [0.0; 4];
        for a in 0..4 {
            let mut sum = 0.0;
            for node in 0..8 {
                sum += gamma[a * 8 + node] * rates[3 * node + i];
            }
            modal[a] = sum;
        }
        for node in 0..8 {
            let mut force = 0.0;
            for a in 0..4 {
                force += gamma[a * 8 + node] * modal[a];
            }
            forces[3 * node + i] = coeff * force;
        }
    }
}

/// Von Neumann-Richtmyer artificial bulk viscosity pressure.
///
/// Returns `q = rho * l * (c0 * l * d^2 - c1 * c * d)` in compression
/// (`d < 0`) and `0` in expansion, where `d` is the volumetric strain
/// rate (1/s), `l` the element characteristic length (m), `c` the sound
/// speed (m/s), `c0` the quadratic coefficient, and `c1` the linear
/// coefficient (both dimensionless). The pressure is added to the
/// Hugoniot pressure to spread steady shocks over a few elements while
/// preserving the Rankine-Hugoniot jump (Radioss Theory Manual Eq 340;
/// defaults [`BULK_VISCOSITY_QUADRATIC_DEFAULT`] and
/// [`BULK_VISCOSITY_LINEAR_DEFAULT`]).
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_hourglass::bulk_viscosity_q;
///
/// let q = bulk_viscosity_q(7800.0, 4500.0, 0.01, 1.1, 0.05, -1000.0);
/// assert!((q - 18_408_000.0).abs() <= 1.0e-9 * 18_408_000.0);
/// // Expansion carries no viscous pressure.
/// assert_eq!(bulk_viscosity_q(7800.0, 4500.0, 0.01, 1.1, 0.05, 500.0), 0.0);
/// ```
///
/// # Panics
///
/// Panics if any argument is not finite or if density, sound speed,
/// length, or either coefficient is negative.
#[must_use]
pub fn bulk_viscosity_q(
    density: f64,
    sound_speed: f64,
    length: f64,
    c0: f64,
    c1: f64,
    vol_strain_rate: f64,
) -> f64 {
    assert!(
        density.is_finite() && density >= 0.0,
        "bulk viscosity needs finite non-negative density"
    );
    assert!(
        sound_speed.is_finite() && sound_speed >= 0.0,
        "bulk viscosity needs finite non-negative sound speed"
    );
    assert!(
        length.is_finite() && length >= 0.0,
        "bulk viscosity needs finite non-negative length"
    );
    assert!(
        c0.is_finite() && c0 >= 0.0,
        "bulk viscosity needs finite non-negative c0"
    );
    assert!(
        c1.is_finite() && c1 >= 0.0,
        "bulk viscosity needs finite non-negative c1"
    );
    assert!(
        vol_strain_rate.is_finite(),
        "bulk viscosity needs a finite strain rate"
    );
    if vol_strain_rate >= 0.0 {
        return 0.0;
    }
    density
        * length
        * (c0 * length * vol_strain_rate * vol_strain_rate - c1 * sound_speed * vol_strain_rate)
}

/// Batched bulk viscosity over element slices.
///
/// Applies [`bulk_viscosity_q`] element-wise with shared `c0`/`c1`
/// coefficients: per-element `densities`, `sound_speeds`, `lengths`,
/// and `vol_strain_rates` map to per-element pressures in `q`, all
/// slices sharing one length.
///
/// # Examples
///
/// ```rust
/// use sonium_compute::explicit_hourglass::bulk_viscosity_q_batch;
///
/// let rho = [7800.0, 7800.0];
/// let c = [4500.0, 4500.0];
/// let len = [0.01, 0.01];
/// let rates = [-100.0, 50.0];
/// let mut q = [0.0; 2];
/// bulk_viscosity_q_batch(&rho, &c, &len, 1.1, 0.05, &rates, &mut q);
/// assert!(q[0] > 0.0);
/// assert_eq!(q[1], 0.0);
/// ```
///
/// # Panics
///
/// Panics if the slices disagree in length or any value violates the
/// [`bulk_viscosity_q`] contract.
pub fn bulk_viscosity_q_batch(
    densities: &[f64],
    sound_speeds: &[f64],
    lengths: &[f64],
    c0: f64,
    c1: f64,
    vol_strain_rates: &[f64],
    q: &mut [f64],
) {
    assert_eq!(densities.len(), q.len(), "viscosity batch needs equal lens");
    assert_eq!(
        sound_speeds.len(),
        q.len(),
        "viscosity batch needs equal lens"
    );
    assert_eq!(lengths.len(), q.len(), "viscosity batch needs equal lens");
    assert_eq!(
        vol_strain_rates.len(),
        q.len(),
        "viscosity batch needs equal lens"
    );
    assert!(
        c0.is_finite() && c0 >= 0.0,
        "bulk viscosity needs finite non-negative c0"
    );
    assert!(
        c1.is_finite() && c1 >= 0.0,
        "bulk viscosity needs finite non-negative c1"
    );
    for (index, out) in q.iter_mut().enumerate() {
        *out = bulk_viscosity_q(
            densities[index],
            sound_speeds[index],
            lengths[index],
            c0,
            c1,
            vol_strain_rates[index],
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Unit-cube HEXA8 coordinates in lane node ordering.
    fn unit_cube() -> [f64; 24] {
        [
            0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0,
            1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0,
        ]
    }

    /// Maximum absolute Gamma residual against the linear fields.
    fn max_linear_residual(gamma: &[f64], coords: &[f64]) -> f64 {
        let mut max = 0.0f64;
        for a in 0..4 {
            let mut dots = [0.0; 4];
            for node in 0..8 {
                let g = gamma[a * 8 + node];
                dots[0] += g;
                dots[1] += g * coords[3 * node];
                dots[2] += g * coords[3 * node + 1];
                dots[3] += g * coords[3 * node + 2];
            }
            for dot in dots {
                max = max.max(dot.abs());
            }
        }
        max
    }

    #[test]
    fn unit_cube_gamma_matches_base_modes() {
        let coords = unit_cube();
        let mut gamma = [0.0; 32];
        hexa8_gamma(&coords, &mut gamma);
        for a in 0..4 {
            for node in 0..8 {
                assert_eq!(
                    gamma[a * 8 + node],
                    HEXA8_HOURGLASS_BASE[a][node],
                    "mode {a} node {node}"
                );
            }
        }
    }

    #[test]
    fn unit_cube_gamma_orthogonality_is_exact() {
        let coords = unit_cube();
        let mut gamma = [0.0; 32];
        hexa8_gamma(&coords, &mut gamma);
        assert_eq!(max_linear_residual(&gamma, &coords), 0.0);
    }

    #[test]
    fn gamma_invariant_on_scaled_translated_cube() {
        // Affine maps preserve Gamma; here coords stay exactly
        // representable so the output is still bit-exact.
        let mut coords = [0.0; 24];
        for (node, nat) in NATURAL_COORDS.iter().enumerate() {
            coords[3 * node] = 2.0 + 3.0 * (nat[0] + 1.0) / 2.0;
            coords[3 * node + 1] = -1.0 + 3.0 * (nat[1] + 1.0) / 2.0;
            coords[3 * node + 2] = 4.0 + 3.0 * (nat[2] + 1.0) / 2.0;
        }
        let mut gamma = [0.0; 32];
        hexa8_gamma(&coords, &mut gamma);
        for a in 0..4 {
            for node in 0..8 {
                assert_eq!(gamma[a * 8 + node], HEXA8_HOURGLASS_BASE[a][node]);
            }
        }
    }

    #[test]
    fn distorted_element_gamma_stays_orthogonal() {
        let mut coords = unit_cube();
        coords[3 * 6 + 2] = 1.5;
        coords[3] = 1.25;
        let mut gamma = [0.0; 32];
        hexa8_gamma(&coords, &mut gamma);
        // Orthogonality holds by construction for any valid shape.
        assert!(max_linear_residual(&gamma, &coords) < 1.0e-12);
        // Distortion really moves Gamma off the base modes.
        let moved = gamma
            .iter()
            .zip(HEXA8_HOURGLASS_BASE.iter().flatten())
            .any(|(got, base)| (got - base).abs() > 1.0e-6);
        assert!(moved, "distorted Gamma should differ from base modes");
    }

    #[test]
    fn hourglass_forces_vanish_on_rigid_translation() {
        let coords = unit_cube();
        let mut gamma = [0.0; 32];
        hexa8_gamma(&coords, &mut gamma);
        let rates = [
            1.5, -2.0, 0.5, 1.5, -2.0, 0.5, 1.5, -2.0, 0.5, 1.5, -2.0, 0.5, 1.5, -2.0, 0.5, 1.5,
            -2.0, 0.5, 1.5, -2.0, 0.5, 1.5, -2.0, 0.5,
        ];
        let mut forces = [7.0; 24];
        hexa8_hourglass_forces(&gamma, &rates, 0.1, &mut forces);
        assert!(forces.iter().all(|f| *f == 0.0));
    }

    #[test]
    fn hourglass_forces_vanish_on_rigid_rotation() {
        let coords = unit_cube();
        let mut gamma = [0.0; 32];
        hexa8_gamma(&coords, &mut gamma);
        // Rigid spin about z with dyadic rate: v = (-y/4, x/4, 0).
        let mut rates = [0.0; 24];
        for node in 0..8 {
            rates[3 * node] = -coords[3 * node + 1] / 4.0;
            rates[3 * node + 1] = coords[3 * node] / 4.0;
        }
        let mut forces = [7.0; 24];
        hexa8_hourglass_forces(&gamma, &rates, 0.1, &mut forces);
        assert!(forces.iter().all(|f| *f == 0.0));
    }

    #[test]
    fn viscous_power_is_nonnegative_and_matches_modal_sum() {
        let coords = unit_cube();
        let mut gamma = [0.0; 32];
        hexa8_gamma(&coords, &mut gamma);
        // Deterministic mixed velocity: hourglass plus rigid parts.
        let mut rates = [0.0; 24];
        for node in 0..8 {
            let h1 = HEXA8_HOURGLASS_BASE[0][node];
            let h2 = HEXA8_HOURGLASS_BASE[1][node];
            rates[3 * node] = 0.5 * h1 + 1.0;
            rates[3 * node + 1] = 0.25 * h2 - 0.5;
            rates[3 * node + 2] = 0.125 * (node as f64) + 0.25;
        }
        let coeff = 0.1;
        let mut forces = [0.0; 24];
        hexa8_hourglass_forces(&gamma, &rates, coeff, &mut forces);
        let power: f64 = rates.iter().zip(forces.iter()).map(|(v, f)| v * f).sum();
        assert!(power >= 0.0, "viscous power must dissipate");
        assert!(power > 0.0, "hourglass content must dissipate");
        // Independent modal identity P = coeff * sum(q^2).
        let mut modal = 0.0;
        for i in 0..3 {
            for a in 0..4 {
                let mut q = 0.0;
                for node in 0..8 {
                    q += gamma[a * 8 + node] * rates[3 * node + i];
                }
                modal += q * q;
            }
        }
        let expected = coeff * modal;
        assert!(
            (power - expected).abs() <= 1.0e-12 * expected.abs().max(1.0),
            "power {power} vs modal {expected}"
        );
    }

    #[test]
    fn stiffness_form_matches_hand_value() {
        // Pure mode-1 displacement in x: q = (8, 0, 0, 0).
        let gamma: Vec<f64> = HEXA8_HOURGLASS_BASE.iter().flatten().copied().collect();
        let mut rates = [0.0; 24];
        for node in 0..8 {
            rates[3 * node] = HEXA8_HOURGLASS_BASE[0][node];
        }
        let mut forces = [0.0; 24];
        hexa8_hourglass_forces(&gamma, &rates, 2.0, &mut forces);
        for node in 0..8 {
            let expected = 16.0 * HEXA8_HOURGLASS_BASE[0][node];
            assert_eq!(forces[3 * node], expected);
            assert_eq!(forces[3 * node + 1], 0.0);
            assert_eq!(forces[3 * node + 2], 0.0);
        }
    }

    #[test]
    fn bulk_viscosity_matches_hand_values() {
        // q = 78 * (1.1*0.01*d^2 + 0.05*4500*|d|) in compression.
        let q = bulk_viscosity_q(7800.0, 4500.0, 0.01, 1.1, 0.05, -1000.0);
        assert!((q - 18_408_000.0).abs() <= 1.0e-12 * 18_408_000.0);
        let q = bulk_viscosity_q(7800.0, 4500.0, 0.01, 1.1, 0.05, -100.0);
        assert!((q - 1_763_580.0).abs() <= 1.0e-12 * 1_763_580.0);
        assert_eq!(bulk_viscosity_q(7800.0, 4500.0, 0.01, 1.1, 0.05, 0.0), 0.0);
        assert_eq!(
            bulk_viscosity_q(7800.0, 4500.0, 0.01, 1.1, 0.05, 500.0),
            0.0
        );
    }

    #[test]
    fn bulk_viscosity_batch_matches_scalar() {
        let rho = [7800.0, 2700.0, 7800.0];
        let c = [4500.0, 5000.0, 4500.0];
        let len = [0.01, 0.02, 0.01];
        let rates = [-1000.0, -50.0, 10.0];
        let mut q = [0.0; 3];
        bulk_viscosity_q_batch(&rho, &c, &len, 1.1, 0.05, &rates, &mut q);
        for index in 0..3 {
            let expected =
                bulk_viscosity_q(rho[index], c[index], len[index], 1.1, 0.05, rates[index]);
            assert_eq!(q[index], expected);
        }
        assert!(q[0] > 0.0 && q[1] > 0.0);
        assert_eq!(q[2], 0.0);
    }

    #[test]
    #[should_panic(expected = "24")]
    fn gamma_rejects_short_coords() {
        let mut gamma = [0.0; 32];
        hexa8_gamma(&[0.0; 12], &mut gamma);
    }

    #[test]
    #[should_panic(expected = "non-degenerate")]
    fn gamma_rejects_degenerate_element() {
        let mut gamma = [0.0; 32];
        hexa8_gamma(&[0.0; 24], &mut gamma);
    }

    #[test]
    #[should_panic(expected = "non-negative finite coefficient")]
    fn forces_reject_negative_coeff() {
        let gamma = [1.0; 32];
        let mut forces = [0.0; 24];
        hexa8_hourglass_forces(&gamma, &[0.0; 24], -0.1, &mut forces);
    }

    #[test]
    #[should_panic(expected = "equal lens")]
    fn viscosity_batch_rejects_mismatched_lens() {
        let mut q = [0.0; 2];
        bulk_viscosity_q_batch(&[7800.0], &[4500.0], &[0.01], 1.1, 0.05, &[-100.0], &mut q);
    }
}
