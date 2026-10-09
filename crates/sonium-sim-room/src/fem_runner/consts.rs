//! Shared constants and small material-mapping helpers for the FEM room runner.
//!
//! Resident GPU sweep execution now lives in [`super::gpu`]; this module keeps
//! only frequency-independent constants and the absorption back-fit helper.

use num_complex::Complex64;
use sonium_bem::materials::{AmbientMedium, PorousLayer, PorousModel};
#[cfg(test)]
use sonium_xem::Point3D;
#[cfg(test)]
use sonium_xem::Source;

#[cfg(test)]
use super::types::room_complex_isotropic_amplitude;

/// Gaussian sigma (m) used by the `FemSourceModel::Gaussian` reference path
/// and the FreeFEM cross-check. This is NOT the production source model: the
/// production default is the mesh-independent point load, because a fixed
/// 0.1 m Gaussian low-pass filters the monopole by ~exp(-(kσ)²/2).
pub const DEFAULT_SOURCE_WIDTH: f64 = 0.1;

/// Air density (kg/m³) at 20 °C used throughout the project.
pub const AIR_DENSITY: f64 = sonium_xem::constants::AIR_DENSITY_20C;

/// Volume velocity (m³/s) corresponding to `Source::amplitude == 1.0`. Chosen
/// so that an unobstructed monopole at 1 m, 1 kHz radiates ≈ 85 dB SPL — a
/// loud-but-not-painful listening level that puts the synthesised dataset in
/// the same dB range as real room measurements. Derived from the free-field
/// relation `|p| = ω ρ |Q| / (4π r)` with target `|p| = 0.356 Pa` (= 85 dB re
/// 20 µPa) at the reference frequency.
pub const SOURCE_VOLUME_VELOCITY_REFERENCE: f64 =
    crate::source_calibration::SOURCE_VOLUME_VELOCITY_REFERENCE;

/// Reference relative humidity (fraction) for atmospheric absorption. ISO 9613-1
/// pegs reference values at 50 % RH, 20 °C, 1 atm — typical living-room
/// conditions and a defensible default for synthetic data.
pub(super) const AIR_RELATIVE_HUMIDITY: f64 = 0.50;

pub(super) const AIR_TEMPERATURE_K: f64 = 293.15; // 20 °C

pub(super) const AIR_PRESSURE_PA: f64 = 101_325.0;

pub(super) const AIR_REF_PRESSURE_PA: f64 = 101_325.0;

pub(super) const AIR_REF_TEMPERATURE_K: f64 = 293.15;

/// Atmospheric absorption coefficient α (in nepers per metre) at frequency `f`
/// (Hz), per ISO 9613-1:1993. The result is the imaginary part of the complex
/// wavenumber: `k_complex = k_real + i·α`. Values are vanishingly small below
/// ~1 kHz at room scales and grow rapidly above 4 kHz — at 2 kHz, ~0.5 dB / 10 m
/// (≈ 5.8 × 10⁻³ Np/m), at 4 kHz ~2 dB / 10 m.
pub(super) fn iso_9613_air_absorption_np_per_m(frequency_hz: f64) -> f64 {
    let t = AIR_TEMPERATURE_K;
    let p_ratio = AIR_PRESSURE_PA / AIR_REF_PRESSURE_PA;
    let t_ratio = t / AIR_REF_TEMPERATURE_K;

    // Saturation vapor pressure ratio per ISO 9613-1 § 6.
    let p_sat_ref = 10f64.powf(-6.8346 * (273.16 / t).powf(1.261) + 4.6151);
    // ISO 9613-1's relaxation-frequency formulas take `h` as the *mole percent*
    // of water vapor — at 50 % RH, 20 °C, 1 atm this is ~1.155, not ~0.01155.
    let h = 100.0 * AIR_RELATIVE_HUMIDITY * p_sat_ref / p_ratio;

    // Relaxation frequencies (Hz) for oxygen and nitrogen.
    let f_ro = p_ratio * (24.0 + 4.04e4 * h * (0.02 + h) / (0.391 + h));
    let f_rn = p_ratio
        * t_ratio.powf(-0.5)
        * (9.0 + 280.0 * h * (-4.170 * (t_ratio.powf(-1.0 / 3.0) - 1.0)).exp());

    let f = frequency_hz;
    let f2 = f * f;

    // ISO 9613-1 attenuation in dB/m
    let alpha_db_per_m = 8.686
        * f2
        * ((1.84e-11 * (1.0 / p_ratio) * t_ratio.sqrt())
            + t_ratio.powf(-2.5)
                * (0.01275 * (-2239.1 / t).exp() / (f_ro + f2 / f_ro)
                    + 0.1068 * (-3352.0 / t).exp() / (f_rn + f2 / f_rn)));
    // dB/m → Np/m: 1 dB = ln(10)/20 Np
    alpha_db_per_m * std::f64::consts::LN_10 / 20.0
}

/// Numerical imaginary-axis offset for the complex wavenumber, expressed as a
/// fraction of `k_real`. At an exactly sampled real-axis pole of A(k), GMRES
/// sees a near-null eigenvalue; this small floor keeps a rigid-room reference
/// solvable without pretending to supply its physical decay.
///
/// Below ~1 kHz, ISO 9613 atmospheric absorption is vanishingly small
/// (≪ 1e-4 Np/m). Physical low-frequency decay must therefore come from the
/// resolved wall admittance/porous layers. The previous 0.5 % floor masked
/// those losses; 0.05 % is retained only as a pole regularizer. At high
/// frequencies the ISO 9613 term takes over naturally via `max(...)`.
pub const DEFAULT_MODAL_DAMPING_RATIO: f64 = 0.0005;

/// Complex wavenumber with both atmospheric absorption and a minimum modal
/// damping floor. The floor ensures GMRES never sees a near-null eigenvalue
/// at a room mode. Its positive imaginary part makes the outgoing spatial
/// factor `exp(+ikr)` decay under the global `exp(-iωt)` convention and gives
/// `K - k²M` a negative dissipative imaginary part.
pub fn complex_wavenumber(k_real: f64, frequency_hz: f64, modal_damping_ratio: f64) -> Complex64 {
    let atmos = iso_9613_air_absorption_np_per_m(frequency_hz);
    let modal_floor = k_real.abs() * modal_damping_ratio.max(0.0);
    Complex64::new(k_real, atmos.max(modal_floor))
}

/// Target elements per shortest wavelength.
///
/// 3 EPW (the previous default) leaves ≈ 30 % amplitude error and ≈ 10 % modal
/// frequency drift at the band edge for P1 tetrahedra — well below the
/// pollution-error guidance of the FEM-Helmholtz literature. 6 EPW brings
/// the amplitude error under ~10 % and is the calibrated default. The cost
/// is roughly 8× the node count of the old 3-EPW default, which is the price
/// of getting modal amplitudes right; users who want the old performance
/// envelope can set `FEM_ELEMENTS_PER_WAVELENGTH` lower in scenario configs
/// or rebuild with a custom value here.
pub const ELEMENTS_PER_WAVELENGTH: f64 = 6.0;

/// GMRES restart window. This keeps orthogonalization cost bounded while still
/// giving the shifted-Laplacian preconditioner enough Krylov space.
pub(super) const GMRES_RESTART: usize = 50;

/// Cap GMRES restart cycles. With restart=50 this allows up to 10000 Krylov
/// steps per solve before reporting non-convergence.
pub(super) const GMRES_MAX_RESTARTS: usize = 200;

/// Additive Schwarz overlap depth for the CLI-selectable Schwarz preconditioner.
pub(super) const SCHWARZ_OVERLAP: usize = 1;

/// Fixed-point iterations for the CLI-selectable parallel ILU approximation.
pub(super) const ILU_FIXED_POINT_ITERATIONS: usize = 3;

/// Default recycled subspace dimension for CLI data generation.
///
/// GCRO-DR recycling captures harmonic Ritz approximations of the smallest
/// eigenvalues from one frequency and re-uses them at the next, pre-deflating
/// the near-null modes that otherwise stall GMRES at room-mode frequencies.
/// k=30 is large enough to cover the few dominant modes that move slowly with
/// frequency in a frequency sweep, while keeping the coarse-correction cost
/// well below one extra GMRES restart. Empirically, modal-frequency iteration
/// counts drop by ~3–5× versus the un-recycled sweep.
pub const DEFAULT_RECYCLE_SUBSPACE_SIZE: usize = 30;

/// Ignore Gaussian source tails beyond this many standard deviations when
/// assembling the RHS. At 8 sigma the omitted envelope is below 1.3e-14.
pub const SOURCE_TRUNCATION_SIGMAS: f64 = 8.0;

/// Boundary markers
pub(super) const MARKER_FLOOR: i32 = 1;

pub(super) const MARKER_CEILING: i32 = 2;

pub(super) const MARKER_FRONT: i32 = 3;

pub(super) const MARKER_BACK: i32 = 4;

pub(super) const MARKER_LEFT: i32 = 5;

pub(super) const MARKER_RIGHT: i32 = 6;

pub(super) const MARKER_SYM_X: i32 = 7;

pub(super) const MARKER_SYM_Y: i32 = 8;

pub(super) const MARKER_SCHWARZ_INTERFACE_MASS: i32 = 10_001;

pub(super) const MARKER_SCHWARZ_INTERFACE_STIFFNESS: i32 = 10_002;

pub(super) const MAX_DEFECT_CORRECTIONS: usize = 2;

pub(super) const BICGSTAB_FALLBACK_MAX_ITERATIONS: usize = 30_000;

/// Reference frequency at which `SurfaceConfig::Absorption` matches the user's
/// target absorption coefficient. 500 Hz is a typical mid-band reference and
/// roughly corresponds to NRC's anchor band.
pub(super) const ABSORPTION_REFERENCE_FREQ: f64 = 500.0;

/// Thickness assumed for the equivalent Miki layer used to render a
/// frequency-flat absorption coefficient as a frequency-dependent complex
/// admittance. 25 mm matches typical wall-treatment thicknesses (carpet pad,
/// thin glasswool, acoustic panel) and yields a reasonable LF rise.
pub(super) const ABSORPTION_LAYER_THICKNESS: f64 = 0.025;

/// Solve for the flow resistivity (Pa·s/m²) of a 25 mm Miki layer whose
/// normal-incidence absorption coefficient at 500 Hz equals `target_alpha`.
/// Uses bisection in log-σ space — α(σ) is non-monotonic in general (peaks
/// around σ·t/c ≈ 1) but on the rising branch it is monotone increasing for
/// thin layers. Targets above the peak absorption of this 25 mm equivalent
/// layer saturate at the peak instead of inventing a non-physical branch.
pub(super) fn flow_resistivity_for_alpha(
    target_alpha: f64,
    medium: AmbientMedium,
) -> anyhow::Result<f64> {
    medium.validate()?;
    anyhow::ensure!(
        target_alpha.is_finite() && (0.0..=1.0).contains(&target_alpha),
        "absorption coefficient must be finite and between zero and one"
    );
    let target = target_alpha.clamp(0.001, 0.999);

    let alpha_for = |sigma: f64| -> anyhow::Result<f64> {
        let layer = PorousLayer {
            model: PorousModel::Miki,
            thickness: ABSORPTION_LAYER_THICKNESS,
            flow_resistivity: sigma,
            air_gap: 0.0,
        };
        Ok(layer.try_absorption_coefficient(ABSORPTION_REFERENCE_FREQ, medium)?)
    };

    let mut lo_log = 1.0_f64; // σ = 10
    let mut hi_log = 5.5_f64; // σ ≈ 316 000 (well past peak; α drops here)
    // Restrict the upper bound to the rising branch.
    let alpha_lo = alpha_for(10f64.powf(lo_log))?;
    if alpha_lo >= target {
        // Target α is below what even the most porous practical layer can
        // achieve at the reference frequency. Saturate at σ=10 — the resulting
        // admittance is effectively rigid for our purposes.
        return Ok(10f64.powf(lo_log));
    }

    // Walk hi down to the peak so bisection is well-posed on a monotonic
    // segment.
    let mut prev_alpha = alpha_lo;
    let mut probe_log = lo_log;
    let mut peak_log = hi_log;
    while probe_log < hi_log {
        probe_log += 0.1;
        let a = alpha_for(10f64.powf(probe_log))?;
        if a < prev_alpha {
            peak_log = probe_log - 0.1;
            break;
        }
        prev_alpha = a;
    }
    hi_log = peak_log;
    let alpha_hi = alpha_for(10f64.powf(hi_log))?;
    if alpha_hi <= target {
        return Ok(10f64.powf(hi_log));
    }

    for _ in 0..40 {
        let mid_log = 0.5 * (lo_log + hi_log);
        let a = alpha_for(10f64.powf(mid_log))?;
        if a < target {
            lo_log = mid_log;
        } else {
            hi_log = mid_log;
        }
    }
    Ok(10f64.powf(0.5 * (lo_log + hi_log)))
}

#[cfg(test)]
pub(super) fn compute_source_term(
    x: f64,
    y: f64,
    z: f64,
    sources: &[Source],
    source_positions: &[Vec<Point3D>],
    frequency: f64,
    source_width: f64,
) -> Complex64 {
    // Volume-source distribution for the assembled Helmholtz operator
    //   -(∇² + k²) p = q_h .
    //
    // For a point monopole of volume velocity Q at x_s under the e^{-iωt} time
    // convention, q_h(x) = -i ω ρ Q · δ³(x − x_s), and the resulting pressure
    // is -iωρQ times the outgoing Green's function. We replace the Dirac
    // with a *normalized* Gaussian (integrates to 1) so that `amplitude = 1`
    // corresponds to a 1 m³/s volume-velocity monopole independent of σ.
    //
    // Source directivity weights the distributed RHS before propagation so
    // every reflected path inherits the loudspeaker radiation pattern. The
    // production path precomputes the same spatial basis and applies this
    // factor at its nodal entries.
    //
    // For sources within ~3σ of a wall, `source_positions[i]` includes mirror
    // images across each near wall so the effective integrated source strength
    // inside the room equals Q regardless of placement.
    let two_sigma_sq = 2.0 * source_width * source_width;
    let omega = 2.0 * std::f64::consts::PI * frequency;
    // Gaussian normalisation so that ∫ envelope(r) d³r = 1 in ℝ³.
    let norm = 1.0 / ((std::f64::consts::PI * two_sigma_sq).powf(1.5));
    // -i ω ρ · Q_ref · g(x − x_s), with Q_ref converting `amplitude` to a
    // physical volume-velocity scale.
    let coeff = Complex64::new(
        0.0,
        -omega * AIR_DENSITY * SOURCE_VOLUME_VELOCITY_REFERENCE * norm,
    );

    let mut total = Complex64::new(0.0, 0.0);
    let point = Point3D::new(x, y, z);
    for (source, positions) in sources.iter().zip(source_positions.iter()) {
        let q = room_complex_isotropic_amplitude(source, frequency);
        if q.norm() == 0.0 {
            continue;
        }
        let mut env_sum = 0.0;
        for sp in positions {
            let dx = x - sp.x;
            let dy = y - sp.y;
            let dz = z - sp.z;
            let r2 = dx * dx + dy * dy + dz * dz;
            env_sum += (-r2 / two_sigma_sq).exp();
        }
        let directivity = source.complex_directivity_factor(&point, frequency);
        total += coeff * (q * directivity * env_sum);
    }
    total
}
