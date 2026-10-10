//! Annular-duct modal analysis for FEM aeroacoustics.
//!
//! Validated wrappers over the `sonium-wave` duct modes: cut-on/cut-off
//! assessment at operating conditions, lined-wall roots from liner
//! impedance, and rotor-stator interaction screening. The wrappers add
//! mean-flow extraction and error mapping; the modal mathematics lives
//! in `sonium-wave`.

use num_complex::Complex64;
use sonium_wave::analytical::annular_duct_modes::{
    AnnularDuctMode, AxialBranch, lined_circular_root, tyler_sofrin_orders,
};
use sonium_wave::analytical::azimuthal::AzimuthalMode;

use super::boundary::AcousticLiner;
use super::error::{AeroacousticsError, require_positive};
use super::mean_flow::MeanFlow;

/// Cut-on assessment of one annular duct mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModeCutOn {
    /// Whether the mode propagates at the assessment condition.
    pub cut_on: bool,
    /// Cut-on frequency `[Hz]` at the assessment Mach number.
    pub cutoff_frequency_hz: f64,
    /// Downstream axial wavenumber `[rad/m]` (complex if cut off).
    pub axial_downstream: Complex64,
    /// Upstream axial wavenumber `[rad/m]` (complex if cut off).
    pub axial_upstream: Complex64,
    /// Downstream group velocity `[m/s]`, if the mode propagates.
    pub group_velocity_downstream: Option<f64>,
    /// Upstream group velocity `[m/s]`, if the mode propagates.
    pub group_velocity_upstream: Option<f64>,
}

/// Assesses one annular duct mode at operating conditions.
///
/// The radial index follows the `sonium-wave` convention (`(0, 0)` is the
/// plane wave). Axial Mach number is signed along `+z`.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid geometry, frequency, Mach
/// number, or a missed wall root.
#[allow(clippy::too_many_arguments)]
pub fn annular_mode_cuton(
    hub_radius_m: f64,
    tip_radius_m: f64,
    sound_speed_m_per_s: f64,
    azimuthal_order: i32,
    radial_mode: usize,
    frequency_hz: f64,
    mach_axial: f64,
) -> Result<ModeCutOn, AeroacousticsError> {
    require_positive("frequency_hz", frequency_hz)?;
    if mach_axial.abs() >= 1.0 || !mach_axial.is_finite() {
        return Err(AeroacousticsError::InvalidMach {
            magnitude: mach_axial.abs(),
        });
    }
    let mode = AnnularDuctMode::try_new(
        hub_radius_m,
        tip_radius_m,
        sound_speed_m_per_s,
        AzimuthalMode::new(azimuthal_order),
        radial_mode,
    )?;
    let omega = 2.0 * std::f64::consts::PI * frequency_hz;
    let (axial_downstream, axial_upstream) = mode.axial_wavenumbers(omega, mach_axial)?;
    Ok(ModeCutOn {
        cut_on: mode.is_cut_on(omega, mach_axial)?,
        cutoff_frequency_hz: mode.cutoff_frequency_hz(mach_axial)?,
        axial_downstream,
        axial_upstream,
        group_velocity_downstream: mode.group_velocity(
            omega,
            mach_axial,
            AxialBranch::Downstream,
        )?,
        group_velocity_upstream: mode.group_velocity(omega, mach_axial, AxialBranch::Upstream)?,
    })
}

/// Assesses one annular duct mode with a mean-flow state.
///
/// Extracts the sound speed and axial Mach number from `mean_flow`
/// (axial direction `+z`).
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid geometry, frequency, flow,
/// or a missed wall root.
pub fn annular_mode_cuton_with_flow(
    hub_radius_m: f64,
    tip_radius_m: f64,
    azimuthal_order: i32,
    radial_mode: usize,
    frequency_hz: f64,
    mean_flow: &MeanFlow,
) -> Result<ModeCutOn, AeroacousticsError> {
    if !mean_flow.speed_of_sound.is_finite() || mean_flow.speed_of_sound <= 0.0 {
        return Err(AeroacousticsError::NonFiniteField {
            name: "mean_flow.speed_of_sound",
        });
    }
    if mean_flow.velocity.iter().any(|value| !value.is_finite()) {
        return Err(AeroacousticsError::NonFiniteField {
            name: "mean_flow.velocity",
        });
    }
    annular_mode_cuton(
        hub_radius_m,
        tip_radius_m,
        mean_flow.speed_of_sound,
        azimuthal_order,
        radial_mode,
        frequency_hz,
        mean_flow.velocity[2] / mean_flow.speed_of_sound,
    )
}

/// Cut-on screening of one rotor-stator interaction mode.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InteractionCutOn {
    /// Tyler-Sofrin azimuthal order `m = nB − kV`.
    pub azimuthal_order: i32,
    /// Stator-lobe count `k` generating this order.
    pub stator_lobes: usize,
    /// Whether the lowest radial branch propagates.
    pub cut_on: bool,
    /// Cut-on frequency `[Hz]` of the lowest radial branch.
    pub cutoff_frequency_hz: f64,
}

/// Screens rotor-stator interaction orders for duct cut-on.
///
/// Evaluates the Tyler-Sofrin orders `m = nB − kV` for `k` up to
/// `max_stator_lobes` against the lowest radial branch of each order.
/// Cut-off interaction modes decay exponentially and never reach the
/// far field through the duct.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid stage, duct, frequency, or
/// Mach inputs, or a missed wall root.
#[allow(clippy::too_many_arguments)]
pub fn rotor_stator_cuton(
    blades: usize,
    vanes: usize,
    harmonic: usize,
    max_stator_lobes: usize,
    hub_radius_m: f64,
    tip_radius_m: f64,
    sound_speed_m_per_s: f64,
    frequency_hz: f64,
    mach_axial: f64,
) -> Result<Vec<InteractionCutOn>, AeroacousticsError> {
    let orders = tyler_sofrin_orders(blades, vanes, harmonic, max_stator_lobes)?;
    orders
        .into_iter()
        .enumerate()
        .map(|(lobes, order)| {
            let assessment = annular_mode_cuton(
                hub_radius_m,
                tip_radius_m,
                sound_speed_m_per_s,
                order,
                0,
                frequency_hz,
                mach_axial,
            )?;
            Ok(InteractionCutOn {
                azimuthal_order: order,
                stator_lobes: lobes,
                cut_on: assessment.cut_on,
                cutoff_frequency_hz: assessment.cutoff_frequency_hz,
            })
        })
        .collect()
}

/// Lined-wall radial root `κa` for one circular-duct branch.
///
/// Solves the Ingard-Myers dispersion relation with the liner impedance
/// at the operating frequency, starting homotopy from the hard-wall
/// root. Selects the downstream or upstream branch family.
///
/// # Errors
///
/// Returns [`AeroacousticsError`] for invalid parameters or a missed
/// Newton convergence.
#[allow(clippy::too_many_arguments)]
pub fn lined_wall_root(
    azimuthal_order: u32,
    radial_mode: usize,
    frequency_hz: f64,
    density: f64,
    sound_speed_m_per_s: f64,
    axial_velocity_m_per_s: f64,
    duct_radius_m: f64,
    liner: &AcousticLiner,
    downstream: bool,
) -> Result<Complex64, AeroacousticsError> {
    require_positive("frequency_hz", frequency_hz)?;
    let omega = 2.0 * std::f64::consts::PI * frequency_hz;
    let impedance = liner.impedance(omega);
    let branch = if downstream {
        AxialBranch::Downstream
    } else {
        AxialBranch::Upstream
    };
    Ok(lined_circular_root(
        azimuthal_order,
        radial_mode,
        omega,
        density,
        sound_speed_m_per_s,
        axial_velocity_m_per_s,
        duct_radius_m,
        impedance,
        branch,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::physics::acoustics::aeroacoustics::boundary::LinerModel;

    #[test]
    fn rotor_stator_screening_separates_cut_on_from_cut_off() {
        // 9-blade / 10-vane stage in a 0.25 m tip duct: BPF orders are
        // m = 9, −1, −11 with well-separated cutoffs.
        let low = rotor_stator_cuton(9, 10, 1, 2, 0.1, 0.25, 343.0, 500.0, 0.0).unwrap();
        assert_eq!(
            low.iter()
                .map(|mode| mode.azimuthal_order)
                .collect::<Vec<_>>(),
            vec![9, -1, -11]
        );
        assert!(
            !low[0].cut_on,
            "m = 9 cutoff {}",
            low[0].cutoff_frequency_hz
        );
        assert!(
            low[1].cut_on,
            "m = −1 cutoff {}",
            low[1].cutoff_frequency_hz
        );
        assert!(
            !low[2].cut_on,
            "m = −11 cutoff {}",
            low[2].cutoff_frequency_hz
        );
        let high = rotor_stator_cuton(9, 10, 1, 2, 0.1, 0.25, 343.0, 3000.0, 0.0).unwrap();
        assert!(high.iter().all(|mode| mode.cut_on));
    }

    #[test]
    fn cuton_reports_group_velocity_with_flow() {
        let report = annular_mode_cuton_with_flow(
            0.0,
            0.5,
            0,
            1,
            2000.0,
            &MeanFlow::air_standard([0.0, 0.0, 34.3]),
        )
        .unwrap();
        assert!(report.cut_on);
        let downstream = report.group_velocity_downstream.unwrap();
        let upstream = report.group_velocity_upstream.unwrap();
        assert!(downstream > 0.0 && upstream < 0.0);
        assert!(downstream < 343.0 + 34.3 + 1.0);
    }

    #[test]
    fn rigid_liner_matches_hard_wall_cutoff() {
        let liner = AcousticLiner::sdof(1e12, 0.0, 0.0, LinerModel::IngardMyers);
        let root = lined_wall_root(0, 1, 2000.0, 1.2, 343.0, 0.0, 0.5, &liner, true).unwrap();
        assert!((root.re - 3.8317059702075125).abs() < 1e-6);
        assert!(root.im.abs() < 1e-6);
    }

    #[test]
    fn assessment_rejects_sonic_mach() {
        assert!(matches!(
            annular_mode_cuton(0.0, 0.5, 343.0, 0, 1, 2000.0, 1.0),
            Err(AeroacousticsError::InvalidMach { .. })
        ));
    }
}
