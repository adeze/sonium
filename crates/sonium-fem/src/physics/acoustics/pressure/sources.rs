//! Acoustic source terms for the Helmholtz and wave equations
//!
//! Provides source types (normal velocity, acceleration, displacement,
//! pressure, monopole) and their conversion to Neumann fluxes or RHS
//! contributions compatible with the FEM assembler.
//!
//! # Boundary sources
//!
//! A boundary source prescribes a kinematic quantity on a surface Γ and
//! contributes to the RHS of the system:
//!
//! ```text
//! RHS_i += flux × ∫_Γ φ_i dΓ
//! ```
//!
//! where `flux` depends on the source type (see [`neumann_flux`]).
//!
//! All phasors use Sonium's `exp(-iωt)` convention.

use num_complex::Complex64;
use std::f64::consts::PI;

use crate::assembly::MassMatrix;

// ---------------------------------------------------------------------------
// Source types
// ---------------------------------------------------------------------------

/// Normal velocity source: v_n = V₀ on Γ.
///
/// Neumann flux: `∂p/∂n = +iωρ₀ V₀`.
#[derive(Debug, Clone, Copy)]
pub struct NormalVelocitySource {
    pub amplitude: Complex64,
    pub omega: f64,
    pub density: f64,
}

/// Normal acceleration source: a_n = A₀ on Γ.
///
/// Neumann flux: `∂p/∂n = -ρ₀ A₀` (velocity = `A₀/(-iω)`).
#[derive(Debug, Clone, Copy)]
pub struct NormalAccelerationSource {
    pub amplitude: Complex64,
    pub density: f64,
}

/// Normal displacement source: d_n = D₀ on Γ.
///
/// Neumann flux: `∂p/∂n = +ω²ρ₀ D₀`.
#[derive(Debug, Clone, Copy)]
pub struct NormalDisplacementSource {
    pub amplitude: Complex64,
    pub omega: f64,
    pub density: f64,
}

/// Monopole (volume velocity) point source.
///
/// Adds `-iωρ₀ Q` to the `K-k²M` RHS at the DOF nearest to `position`.
#[derive(Debug, Clone, Copy)]
pub struct MonopoleSource {
    pub position: [f64; 3],
    pub volume_velocity: Complex64,
    pub omega: f64,
    pub density: f64,
}

// ---------------------------------------------------------------------------
// Neumann flux computation
// ---------------------------------------------------------------------------

/// Trait for sources that produce a Neumann flux ∂p/∂n on a boundary.
pub trait NeumannFlux {
    /// The complex Neumann flux value ∂p/∂n.
    fn neumann_flux(&self) -> Complex64;
}

impl NeumannFlux for NormalVelocitySource {
    fn neumann_flux(&self) -> Complex64 {
        // -iωρv = -∇p, hence ∂p/∂n = +iωρv.
        let i = Complex64::new(0.0, 1.0);
        i * self.omega * self.density * self.amplitude
    }
}

impl NeumannFlux for NormalAccelerationSource {
    fn neumann_flux(&self) -> Complex64 {
        // a = -iωv and ∂p/∂n = iωρv, hence ∂p/∂n = -ρa.
        Complex64::new(-self.density, 0.0) * self.amplitude
    }
}

impl NeumannFlux for NormalDisplacementSource {
    fn neumann_flux(&self) -> Complex64 {
        // v = -iωd and ∂p/∂n = iωρv, hence ∂p/∂n = +ω²ρd.
        Complex64::new(self.omega * self.omega * self.density, 0.0) * self.amplitude
    }
}

impl MonopoleSource {
    /// Source strength for the Helmholtz RHS: −jωρ₀ Q.
    pub fn source_strength(&self) -> Complex64 {
        let j = Complex64::new(0.0, 1.0);
        -j * self.omega * self.density * self.volume_velocity
    }
}

// ---------------------------------------------------------------------------
// Boundary load vector
// ---------------------------------------------------------------------------

/// Compute the boundary load vector b_i = ∫_Γ φ_i dΓ from the boundary
/// mass matrix (row sums, using partition of unity).
///
/// The returned vector has length `boundary_mass.dim`.
pub fn boundary_load_vector(boundary_mass: &MassMatrix) -> Vec<f64> {
    let mut load = vec![0.0; boundary_mass.dim];
    for k in 0..boundary_mass.nnz() {
        load[boundary_mass.rows[k]] += boundary_mass.values[k];
    }
    load
}

/// Apply a Neumann-type boundary source to the RHS vector.
///
/// Adds `flux × ∫_Γ φ_i dΓ` to the RHS for each DOF on the boundary.
/// The `rhs` vector must have length ≥ `boundary_mass.dim`.
/// # Panics
/// Panics if `rhs.len() < boundary_mass.dim`.
pub fn apply_boundary_source(rhs: &mut [Complex64], flux: Complex64, boundary_mass: &MassMatrix) {
    assert!(
        rhs.len() >= boundary_mass.dim,
        "rhs length {} < boundary_mass.dim {}",
        rhs.len(),
        boundary_mass.dim
    );
    let load = boundary_load_vector(boundary_mass);
    for (i, &l) in load.iter().enumerate() {
        if l.abs() > 1e-15 {
            rhs[i] += flux * l;
        }
    }
}

/// Apply a monopole point source to the RHS vector.
///
/// Finds the mesh node nearest to `source.position` and adds the source
/// strength there. The `nodes` slice provides (x, y, z) coordinates.
///
/// Does nothing if `nodes` is empty.
pub fn apply_monopole_source(rhs: &mut [Complex64], source: &MonopoleSource, nodes: &[[f64; 3]]) {
    if nodes.is_empty() {
        return;
    }
    let strength = source.source_strength();
    let mut min_dist = f64::MAX;
    let mut nearest = 0;
    for (i, node) in nodes.iter().enumerate() {
        let dx = node[0] - source.position[0];
        let dy = node[1] - source.position[1];
        let dz = node[2] - source.position[2];
        let dist = dx * dx + dy * dy + dz * dz;
        if dist < min_dist {
            min_dist = dist;
            nearest = i;
        }
    }
    rhs[nearest] += strength;
}

/// Helper: convert frequency in Hz to angular frequency ω = 2πf.
pub fn freq_to_omega(f: f64) -> f64 {
    2.0 * PI * f
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    const RHO: f64 = 1.204;

    #[test]
    fn test_velocity_source_flux() {
        let omega = 2.0 * PI * 1000.0;
        let src = NormalVelocitySource {
            amplitude: Complex64::new(1.0, 0.0), // 1 m/s
            omega,
            density: RHO,
        };
        let flux = src.neumann_flux();
        // Under exp(-iωt), ∂p/∂n = +iωρ V.
        assert!(flux.re.abs() < 1e-10);
        assert!((flux.im - omega * RHO).abs() / (omega * RHO) < 1e-10);
    }

    #[test]
    fn test_acceleration_source_flux_is_real() {
        let src = NormalAccelerationSource {
            amplitude: Complex64::new(9.81, 0.0),
            density: RHO,
        };
        let flux = src.neumann_flux();
        assert!((flux.re + 9.81 * RHO).abs() < 1e-10);
        assert!(flux.im.abs() < 1e-10);
    }

    #[test]
    fn test_displacement_source_equals_minus_omega2_acceleration() {
        let omega = 2.0 * PI * 500.0;
        let d0 = Complex64::new(1e-3, 0.0);
        let disp = NormalDisplacementSource {
            amplitude: d0,
            omega,
            density: RHO,
        };
        let accel = NormalAccelerationSource {
            amplitude: -Complex64::new(omega * omega, 0.0) * d0,
            density: RHO,
        };
        assert!(
            (disp.neumann_flux() - accel.neumann_flux()).norm() < 1e-8,
            "Displacement and acceleration fluxes should match"
        );
        assert!(
            disp.neumann_flux().re > 0.0,
            "positive displacement must produce +ω²ρD flux under exp(-iωt)"
        );
    }

    #[test]
    fn test_boundary_load_vector_sums_to_total() {
        // A simple boundary mass: 2-node edge of length L = 0.5
        // M = (L/6)[2 1; 1 2], row sums = L/2 each, total = L
        let mut bm = MassMatrix::new(2);
        let l = 0.5;
        let f = l / 6.0;
        bm.add(0, 0, 2.0 * f);
        bm.add(0, 1, f);
        bm.add(1, 0, f);
        bm.add(1, 1, 2.0 * f);

        let load = boundary_load_vector(&bm);
        let total: f64 = load.iter().sum();
        assert!(
            (total - l).abs() < 1e-12,
            "Total load = {total}, expected {l}"
        );
    }

    #[test]
    fn test_monopole_nearest_node() {
        let nodes = vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.0, 0.0]];
        let source = MonopoleSource {
            position: [0.6, 0.0, 0.0],
            volume_velocity: Complex64::new(1.0, 0.0),
            omega: 1000.0,
            density: RHO,
        };
        let mut rhs = vec![Complex64::new(0.0, 0.0); 3];
        apply_monopole_source(&mut rhs, &source, &nodes);
        // Nearest to (0.6, 0, 0) is node 2 at (0.5, 0, 0)
        assert!(rhs[2].norm() > 0.0);
        assert!(rhs[0].norm() < 1e-15);
        assert!(rhs[1].norm() < 1e-15);
        assert!(
            rhs[2].im < 0.0,
            "A positive volume velocity has Helmholtz RHS -iωρQ under exp(-iωt)"
        );
    }
}
