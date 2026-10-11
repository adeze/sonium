//! Phase-5 lane 5D G-leg: CAA validation vs FLUIDE_NEW goldens.
//!
//! Offline replay (no engine at gate time): loads the checked-in
//! `FLUIDE_NEW/COMPA1B` and `FLUIDE_NEW/EBCS/INIV` goldens, checks
//! golden-anchored conservation rows on bare-name channels, derives
//! harmonic analogy sources from probe histories through the
//! production snapshot path, and scores radiation limit behavior
//! (far-field agreement, spreading, reciprocity, ring and revolved
//! convergence) with the golden's own relative tolerance as the
//! comparison band. Exit 0 = all probes PASS, exit 1 = any FAIL.
//!
//! Per-probe report lines read `PASS|FAIL <id> measured=<m>
//! bound=<b> <detail>`. Conservation rows band every sample with the
//! total channel's `(abs_tol, rel_tol)` under the schema comparison
//! rule; radiation rows compare dimensionless gaps against the replay
//! channel's `rel_tol`.
//!
//! Harness parameters (documented, not golden facts): the acoustic
//! medium is standard air (`rho0 = 1.2 kg/m^3`, `c0 = 340 m/s`);
//! probe series enter as raw unsteady signals whose `~k` ordinal
//! identities are never assumed (see `goldens/SCHEMA.md`); replay
//! geometry (unit cell/patches, observers at prescribed `kr`) lives
//! in the constants below. Deck time units convert to seconds for
//! the DFT; any other time unit fails closed.

// Rust guideline compliant 2026-02-21

use clap::Parser;
use sonium_fem::physics::acoustics::aeroacoustics::AzimuthalMode;
use sonium_fem::physics::acoustics::aeroacoustics::fwh::{
    FwhSurfacePatch, stationary_fwh_pressure,
};
use sonium_fem::physics::acoustics::aeroacoustics::fwh_2d_axi::{
    FwhRevolvedPatch, RevolvedQuadrature, stationary_fwh_pressure_2d,
};
use sonium_fem::physics::acoustics::aeroacoustics::lighthill::lighthill_far_field_pressure;
use sonium_fem::physics::acoustics::aeroacoustics::replay::{
    detrend, fwh_component_gap, grid_nonuniformity, lighthill_far_field_gap,
    lighthill_reciprocity_gap, mean_step, observer_at_kr, peak_tone, quadrupole_from_history,
    revolved_cross_check, ring_point_gap, spreading_error, within_band,
};
use sonium_fem::physics::acoustics::aeroacoustics::snapshot::{
    dft_peak_phasor, fwh_contour_from_series, fwh_patch_from_series,
};
use sonium_qa::rad::oc_golden::{Golden, GoldenChannel, golden_path, load_golden};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "qa-fem-caa",
    about = "Replay FLUIDE_NEW goldens through CAA analogies."
)]
struct Args {
    /// Goldens root; defaults to the crate `goldens/` dir.
    #[arg(long)]
    goldens: Option<PathBuf>,
}

/// Standard-air density: the acoustic propagation medium.
///
/// The goldens carry unsteady flow signals, not medium properties;
/// the replay propagates in standard air so every probe shares one
/// documented medium. Changing this rescales wavenumbers and moves
/// every `kr`-pinned observer consistently.
const MEDIUM_DENSITY: f64 = 1.2;

/// Standard-air sound speed in m/s (see [`MEDIUM_DENSITY`]).
const MEDIUM_SOUND_SPEED: f64 = 340.0;

/// Unit compact-cell volume in m^3.
///
/// All radiation probes compare dimensionless ratios, so the replay
/// scale is arbitrary; unity keeps phasor magnitudes readable.
const CELL_VOLUME: f64 = 1.0;

/// Far-field agreement range.
///
/// Theory bounds the reactive remainder by ~`2/kr` for the replay
/// tensors, so `kr = 3e4` lands near `7e-5`, an order of magnitude
/// inside the golden `6.6e-4` band.
const KR_FAR: f64 = 3.0e4;

/// Spreading pair for exact `1/r` terms (far-field quadrupole, FW-H
/// thickness): moderate `kr` keeps the check clear of both reactive
/// content and underflow.
const KR_SPREAD_NEAR: f64 = 100.0;
/// Far leg of the exact-spreading pair (see [`KR_SPREAD_NEAR`]).
const KR_SPREAD_FAR: f64 = 200.0;

/// Spreading pair for the asymptotic FW-H loading term.
///
/// Loading carries a `1/r^2` correction, so the pair sits at
/// `kr ~ 1e5` where the asymptotic error `~1/kr` clears the golden
/// band with wide margin.
const KR_LOADING_NEAR: f64 = 1.0e5;
/// Far leg of the loading-spreading pair (see [`KR_LOADING_NEAR`]).
const KR_LOADING_FAR: f64 = 2.0e5;

/// Tone floor as a fraction of the signal peak.
///
/// A phasor below `1e-9` of its signal peak is DFT noise, not a
/// tone; the floor fails such probes closed instead of scoring
/// vacuous sources.
const PHASOR_FLOOR_RATIO: f64 = 1e-9;

/// Revolved/ring compact radius in m.
///
/// At the ~1 kHz replay tone `ka ~ 2e-3`, so the ring-to-point
/// `(ka)^2` truncation sits ~200x inside the golden band.
const COMPACT_RADIUS: f64 = 1.0e-4;

/// Revolved band radius in m (harness geometry).
const REVOLVED_RADIUS: f64 = 0.5;
/// Revolved strip half-height in m.
///
/// `kh ~ 5e-3` keeps the tiling meridional truncation `(kh)^2/24`
/// near `1e-6`, far inside the golden band.
const REVOLVED_HALF_HEIGHT: f64 = 1.5e-4;
/// Hoop tiles for the revolved cross-check (spectrally converged).
const REVOLVED_HOOP_PATCHES: usize = 1440;

/// Reciprocity probe points in m (arbitrary separated pair).
const RECIPROCITY_A: [f64; 3] = [1.3, 0.5, -0.2];
/// Second reciprocity probe point (see [`RECIPROCITY_A`]).
const RECIPROCITY_B: [f64; 3] = [-1.0, 2.0, 0.7];

/// One golden deck replayed by this leg.
struct DeckSpec {
    /// miniqa-relative case dir under `goldens/miniqa`.
    deck: &'static str,
    /// Short tag prefixing probe ids.
    tag: &'static str,
    /// Three node channels forming the velocity history.
    velocity_channels: [&'static str; 3],
    /// Brick channel feeding the replay tone and FW-H pressure.
    pressure_channel: &'static str,
    /// Whether the deck feeds the Lighthill quadrupole rows.
    ///
    /// COMPA1B opts out: every nodal velocity is the same
    /// `+0.2 -> -0.2` step, so every pairwise product is constant and
    /// no stress tone exists at any bin (the floor row would fail
    /// closed). Its live pressure and velocity tones still feed the
    /// FW-H rows.
    lighthill: bool,
}

const DECKS: [DeckSpec; 2] = [
    DeckSpec {
        deck: "FLUIDE_NEW/COMPA1B",
        tag: "compa1b",
        velocity_channels: [
            "SMALLAD00_th_nodes 10 ~4",
            "SMALLAD00_th_nodes 10 ~5",
            "SMALLAD00_th_nodes 10 ~6",
        ],
        pressure_channel: "Bricks 1 ~2",
        lighthill: false,
    },
    DeckSpec {
        deck: "FLUIDE_NEW/EBCS/INIV",
        tag: "iniv",
        velocity_channels: ["thn 5 ~4", "thn 5 ~5", "thn 5 ~6"],
        pressure_channel: "thb 2 ~2",
        lighthill: true,
    },
];

/// A part-to-total identity over bare-name channels.
struct Partition {
    id: &'static str,
    total: &'static str,
    parts: &'static [&'static str],
}

const COMPA1B_PARTITIONS: [Partition; 5] = [
    Partition {
        id: "mass-partition",
        total: "MASS",
        parts: &["Fin MASS", "Frontiere MASS", "Grossier MASS"],
    },
    Partition {
        id: "ymom-partition",
        total: "Y-MOMENTUM",
        parts: &["Fin YMOM", "Frontiere YMOM", "Grossier YMOM"],
    },
    Partition {
        id: "ie-partition",
        total: "INTERNAL ENERGY",
        parts: &["Fin IE", "Frontiere IE", "Grossier IE"],
    },
    Partition {
        id: "ke-partition",
        total: "KINETIC ENERGY",
        parts: &["Fin KE", "Frontiere KE", "Grossier KE"],
    },
    Partition {
        id: "total-energy-partition",
        total: "NO NAME",
        parts: &["INTERNAL ENERGY", "KINETIC ENERGY"],
    },
];

/// One scored probe: id, verdict, measured/bound pair, detail.
struct Row {
    id: String,
    passed: bool,
    measured: String,
    bound: String,
    detail: String,
}

impl Row {
    fn new(id: String, passed: bool, measured: String, bound: String, detail: String) -> Self {
        Self {
            id,
            passed,
            measured,
            bound,
            detail,
        }
    }

    fn fail(id: String, detail: String) -> Self {
        Self::new(id, false, "-".to_string(), "-".to_string(), detail)
    }
}

fn channel<'a>(golden: &'a Golden, name: &str) -> Result<&'a GoldenChannel, String> {
    golden
        .channels
        .get(name)
        .ok_or_else(|| format!("missing channel '{name}'"))
}

fn time_scale_seconds(meta_units: &Option<[String; 3]>) -> Result<f64, String> {
    match meta_units.as_ref().map(|units| units[2].as_str()) {
        Some("ms") => Ok(1e-3),
        Some("s") => Ok(1.0),
        Some("us") => Ok(1e-6),
        other => Err(format!("unsupported deck time unit {other:?}")),
    }
}

fn probe_provenance(golden: &Golden, tag: &str) -> Row {
    Row::new(
        format!("{tag}.normal-termination"),
        golden.meta.normal_termination,
        golden.meta.normal_termination.to_string(),
        "true".to_string(),
        format!("engine {}", golden.meta.engine),
    )
}

fn probe_grid(golden: &Golden, tag: &str) -> Row {
    let id = format!("{tag}.grid-uniformity");
    let step_channel = match channel(golden, "TIME STEP") {
        Ok(found) => found,
        Err(detail) => return Row::fail(id, detail),
    };
    match grid_nonuniformity(&golden.time) {
        Ok(spread) => Row::new(
            id,
            spread <= step_channel.rel_tol,
            format!("{spread:.3e}"),
            format!("{:.3e}", step_channel.rel_tol),
            format!("{} stamps banded by TIME STEP rel_tol", golden.time.len()),
        ),
        Err(error) => Row::fail(id, error.to_string()),
    }
}

fn probe_partition(golden: &Golden, tag: &str, partition: &Partition) -> Row {
    let id = format!("{tag}.{}", partition.id);
    let total = match channel(golden, partition.total) {
        Ok(found) => found,
        Err(detail) => return Row::fail(id, detail),
    };
    let mut parts = Vec::with_capacity(partition.parts.len());
    for name in partition.parts {
        match channel(golden, name) {
            Ok(found) => parts.push(found),
            Err(detail) => return Row::fail(id, detail),
        }
    }
    let mut worst = 0.0f64;
    let mut worst_index = 0;
    let mut failures = 0;
    for (index, expected) in total.values.iter().enumerate() {
        let actual: f64 = parts.iter().map(|part| part.values[index]).sum();
        if !within_band(actual, *expected, total.abs_tol, total.rel_tol) {
            failures += 1;
        }
        let ratio = (actual - expected).abs() / (total.abs_tol + total.rel_tol * expected.abs());
        if ratio > worst {
            worst = ratio;
            worst_index = index;
        }
    }
    Row::new(
        id,
        failures == 0,
        format!("{worst:.3e}@{worst_index}"),
        "1.000e0".to_string(),
        format!(
            "{} samples, {failures} band failures vs {}",
            total.values.len(),
            partition.total
        ),
    )
}

fn probe_iniv_mass(golden: &Golden) -> Row {
    let id = "iniv.mass-conservation".to_string();
    let fetch = |name: &str| channel(golden, name).cloned();
    let (mass, inlet, outlet) = match (fetch("MASS"), fetch("INLET MASS"), fetch("OUTLET MASS")) {
        (Ok(mass), Ok(inlet), Ok(outlet)) => (mass, inlet, outlet),
        _ => return Row::fail(id, "missing MASS/INLET MASS/OUTLET MASS".to_string()),
    };
    let mut worst = 0.0f64;
    let mut worst_index = 0;
    let mut failures = 0;
    for index in 0..mass.values.len() {
        let expected = mass.values[index];
        let actual = mass.values[0] + inlet.values[index] + outlet.values[index];
        if !within_band(actual, expected, mass.abs_tol, mass.rel_tol) {
            failures += 1;
        }
        let ratio = (actual - expected).abs() / (mass.abs_tol + mass.rel_tol * expected.abs());
        if ratio > worst {
            worst = ratio;
            worst_index = index;
        }
    }
    Row::new(
        id,
        failures == 0,
        format!("{worst:.3e}@{worst_index}"),
        "1.000e0".to_string(),
        format!("{} samples vs MASS band", mass.values.len()),
    )
}

/// Strongest fluctuation component of a detrended velocity triple.
fn live_component(fluctuations: &[Vec<f64>]) -> Option<usize> {
    let mut best: Option<usize> = None;
    let mut best_peak = 0.0f64;
    for (index, series) in fluctuations.iter().enumerate() {
        let peak = series.iter().map(|sample| sample.abs()).fold(0.0, f64::max);
        if peak > best_peak {
            best_peak = peak;
            best = Some(index);
        }
    }
    best.filter(|_| best_peak > 0.0)
}

fn probe_radiation(golden: &Golden, spec: &DeckSpec, time_scale: f64, band: f64) -> Vec<Row> {
    let mut rows = Vec::new();
    let pressure = match channel(golden, spec.pressure_channel) {
        Ok(found) => found.clone(),
        Err(detail) => return vec![Row::fail(format!("{}.replay-tone", spec.tag), detail)],
    };
    let mut velocities = Vec::with_capacity(3);
    for name in spec.velocity_channels {
        match channel(golden, name) {
            Ok(found) => velocities.push(found.clone()),
            Err(detail) => return vec![Row::fail(format!("{}.replay-tone", spec.tag), detail)],
        }
    }
    let stamps = golden.time.len();
    if pressure.values.len() != stamps || velocities.iter().any(|v| v.values.len() != stamps) {
        return vec![Row::fail(
            format!("{}.replay-tone", spec.tag),
            "series length mismatch vs time grid".to_string(),
        )];
    }
    let dt_s = match mean_step(&golden.time) {
        Ok(step) => step * time_scale,
        Err(error) => {
            return vec![Row::fail(
                format!("{}.replay-tone", spec.tag),
                error.to_string(),
            )];
        }
    };
    // The replay tone is the pressure signal's spectral peak; every
    // source in this deck replays coherently at that one frequency.
    let pressure_fluct = match detrend(&pressure.values) {
        Ok(split) => split,
        Err(error) => {
            return vec![Row::fail(
                format!("{}.replay-tone", spec.tag),
                error.to_string(),
            )];
        }
    };
    let tone = match peak_tone(&pressure_fluct.fluctuation, dt_s) {
        Ok(found) => found,
        Err(error) => {
            return vec![Row::fail(
                format!("{}.replay-tone", spec.tag),
                error.to_string(),
            )];
        }
    };
    let pressure_peak = pressure_fluct
        .fluctuation
        .iter()
        .map(|sample| sample.abs())
        .fold(0.0, f64::max);
    let pressure_floor = PHASOR_FLOOR_RATIO * pressure_peak;
    rows.push(Row::new(
        format!("{}.pressure-tone-live", spec.tag),
        pressure_peak > 0.0 && tone.phasor.norm() >= pressure_floor,
        format!("{:.3e}", tone.phasor.norm()),
        format!("{pressure_floor:.3e}"),
        format!("bin {} at {:.6e} Hz", tone.bin, tone.frequency_hz),
    ));
    let omega = std::f64::consts::TAU * tone.frequency_hz;
    let wavenumber = omega / MEDIUM_SOUND_SPEED;

    // Velocity history in raw (mean + fluctuation) form: the glue
    // forms rho0 u_i u_j (keeping the mean-fluctuation cross term,
    // the physical shear-noise source), detrends the stress, and only
    // then transforms, so Hann sidelobes never fold the DC operating
    // point into the replay tone. Decks without a stress tone (see
    // DeckSpec::lighthill) skip these rows; the FW-H legs still run.
    let moment = if spec.lighthill {
        let history: Vec<[f64; 3]> = (0..stamps)
            .map(|i| {
                [
                    velocities[0].values[i],
                    velocities[1].values[i],
                    velocities[2].values[i],
                ]
            })
            .collect();
        let signal_peak = history
            .iter()
            .flat_map(|triple| triple.iter())
            .map(|sample| sample.abs())
            .fold(0.0, f64::max);
        let source = match quadrupole_from_history(
            &history,
            MEDIUM_DENSITY,
            dt_s,
            tone.frequency_hz,
            [0.0; 3],
            CELL_VOLUME,
        ) {
            Ok(built) => built,
            Err(error) => {
                rows.push(Row::fail(
                    format!("{}.quadrupole-live", spec.tag),
                    error.to_string(),
                ));
                return rows;
            }
        };
        let moment_peak = source
            .stress_moment
            .iter()
            .flatten()
            .map(|entry| entry.norm())
            .fold(0.0, f64::max);
        let moment_floor =
            PHASOR_FLOOR_RATIO * MEDIUM_DENSITY * signal_peak * signal_peak * CELL_VOLUME;
        rows.push(Row::new(
            format!("{}.quadrupole-live", spec.tag),
            signal_peak > 0.0 && moment_peak >= moment_floor,
            format!("{moment_peak:.3e}"),
            format!("{moment_floor:.3e}"),
            format!("rho0 |u|^2 V floor at {:.6e} Hz", tone.frequency_hz),
        ));

        // Observe near the response maximum: cardinals at KR_FAR, keep
        // the strongest radiating direction for the agreement probe.
        let cardinals = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
        let mut best_direction = cardinals[0];
        let mut best_magnitude = 0.0f64;
        for direction in cardinals {
            let observer = match observer_at_kr(direction, KR_FAR, wavenumber) {
                Ok(point) => point,
                Err(error) => {
                    rows.push(Row::fail(
                        format!("{}.lighthill-far-agreement", spec.tag),
                        error.to_string(),
                    ));
                    return rows;
                }
            };
            match lighthill_far_field_pressure(&source, observer, omega, MEDIUM_SOUND_SPEED) {
                Ok(field) if field.norm() > best_magnitude => {
                    best_magnitude = field.norm();
                    best_direction = direction;
                }
                Ok(_) => {}
                Err(error) => {
                    rows.push(Row::fail(
                        format!("{}.lighthill-far-agreement", spec.tag),
                        error.to_string(),
                    ));
                    return rows;
                }
            }
        }
        if best_magnitude == 0.0 {
            rows.push(Row::fail(
                format!("{}.lighthill-far-agreement", spec.tag),
                "vanishing far field on all cardinals".to_string(),
            ));
            return rows;
        }
        let agreement_observer = match observer_at_kr(best_direction, KR_FAR, wavenumber) {
            Ok(point) => point,
            Err(error) => {
                rows.push(Row::fail(
                    format!("{}.lighthill-far-agreement", spec.tag),
                    error.to_string(),
                ));
                return rows;
            }
        };
        match lighthill_far_field_gap(&source, agreement_observer, omega, MEDIUM_SOUND_SPEED) {
            Ok(gap) => rows.push(Row::new(
                format!("{}.lighthill-far-agreement", spec.tag),
                gap <= band,
                format!("{gap:.3e}"),
                format!("{band:.3e}"),
                format!("kr={KR_FAR:.0e} on {best_direction:?}, theory ~2/kr"),
            )),
            Err(error) => rows.push(Row::fail(
                format!("{}.lighthill-far-agreement", spec.tag),
                error.to_string(),
            )),
        }
        // Exact 1/r spreading of the radiating term at moderate kr.
        let (spread_near, spread_far) = match (
            observer_at_kr(best_direction, KR_SPREAD_NEAR, wavenumber),
            observer_at_kr(best_direction, KR_SPREAD_FAR, wavenumber),
        ) {
            (Ok(near), Ok(far)) => (near, far),
            _ => {
                rows.push(Row::fail(
                    format!("{}.lighthill-spreading", spec.tag),
                    "observer placement failed".to_string(),
                ));
                return rows;
            }
        };
        let spread = (|| {
            let near =
                lighthill_far_field_pressure(&source, spread_near, omega, MEDIUM_SOUND_SPEED)?;
            let far = lighthill_far_field_pressure(&source, spread_far, omega, MEDIUM_SOUND_SPEED)?;
            Ok::<f64, sonium_fem::physics::acoustics::aeroacoustics::AeroacousticsError>(
                spreading_error(
                    near.norm(),
                    KR_SPREAD_NEAR / wavenumber,
                    far.norm(),
                    KR_SPREAD_FAR / wavenumber,
                ),
            )
        })();
        match spread {
            Ok(error) => rows.push(Row::new(
                format!("{}.lighthill-spreading", spec.tag),
                error <= band,
                format!("{error:.3e}"),
                format!("{band:.3e}"),
                format!("kr {KR_SPREAD_NEAR:.0e}->{KR_SPREAD_FAR:.0e}"),
            )),
            Err(error) => rows.push(Row::fail(
                format!("{}.lighthill-spreading", spec.tag),
                error.to_string(),
            )),
        }
        match lighthill_reciprocity_gap(
            source.stress_moment,
            RECIPROCITY_A,
            RECIPROCITY_B,
            omega,
            MEDIUM_SOUND_SPEED,
        ) {
            Ok(gap) => rows.push(Row::new(
                format!("{}.lighthill-reciprocity", spec.tag),
                gap <= band,
                format!("{gap:.3e}"),
                format!("{band:.3e}"),
                format!("{RECIPROCITY_A:?} <-> {RECIPROCITY_B:?}"),
            )),
            Err(error) => rows.push(Row::fail(
                format!("{}.lighthill-reciprocity", spec.tag),
                error.to_string(),
            )),
        }
        Some(source.stress_moment)
    } else {
        None
    };

    // FW-H probes share the replay tone; the velocity leg uses the
    // strongest fluctuation component (documented harness pairing).
    let mut detrended = Vec::with_capacity(3);
    for velocity in &velocities {
        match detrend(&velocity.values) {
            Ok(split) => detrended.push(split.fluctuation),
            Err(error) => {
                rows.push(Row::fail(
                    format!("{}.fwh-component-sum", spec.tag),
                    error.to_string(),
                ));
                return rows;
            }
        }
    }
    let live = match live_component(&detrended) {
        Some(index) => index,
        None => {
            rows.push(Row::fail(
                format!("{}.fwh-component-sum", spec.tag),
                "no live velocity component".to_string(),
            ));
            return rows;
        }
    };
    let zeros = vec![0.0; stamps];
    let patch = |pressure_series: &[f64], velocity_series: &[f64]| {
        fwh_patch_from_series(
            [0.0; 3],
            [0.0, 0.0, 1.0],
            1.0,
            pressure_series,
            velocity_series,
            dt_s,
            tone.frequency_hz,
        )
    };
    let fwh_observer = |kr: f64| observer_at_kr([0.0, 0.0, 1.0], kr, wavenumber);
    let eval_patch = |built: &FwhSurfacePatch, kr: f64| {
        stationary_fwh_pressure(
            std::slice::from_ref(built),
            fwh_observer(kr)?,
            omega,
            MEDIUM_DENSITY,
            MEDIUM_SOUND_SPEED,
        )
    };
    match patch(&pressure_fluct.fluctuation, &detrended[live])
        .map_err(|error| error.to_string())
        .and_then(|built| eval_patch(&built, KR_SPREAD_NEAR).map_err(|error| error.to_string()))
    {
        Ok(field) => {
            let gap = fwh_component_gap(&field);
            rows.push(Row::new(
                format!("{}.fwh-component-sum", spec.tag),
                gap <= band,
                format!("{gap:.3e}"),
                format!("{band:.3e}"),
                format!(
                    "kr={KR_SPREAD_NEAR:.0e}, velocity leg ~{}",
                    spec.velocity_channels[live]
                ),
            ));
        }
        Err(detail) => rows.push(Row::fail(format!("{}.fwh-component-sum", spec.tag), detail)),
    }
    // Thickness spreads exactly; loading spreads asymptotically.
    match patch(&zeros, &detrended[live])
        .map_err(|error| error.to_string())
        .and_then(|built| {
            let near = eval_patch(&built, KR_SPREAD_NEAR).map_err(|error| error.to_string())?;
            let far = eval_patch(&built, KR_SPREAD_FAR).map_err(|error| error.to_string())?;
            Ok(spreading_error(
                near.thickness.norm(),
                KR_SPREAD_NEAR / wavenumber,
                far.thickness.norm(),
                KR_SPREAD_FAR / wavenumber,
            ))
        }) {
        Ok(error) => rows.push(Row::new(
            format!("{}.fwh-thickness-spreading", spec.tag),
            error <= band,
            format!("{error:.3e}"),
            format!("{band:.3e}"),
            format!("kr {KR_SPREAD_NEAR:.0e}->{KR_SPREAD_FAR:.0e}"),
        )),
        Err(detail) => rows.push(Row::fail(
            format!("{}.fwh-thickness-spreading", spec.tag),
            detail,
        )),
    }
    match patch(&pressure_fluct.fluctuation, &zeros)
        .map_err(|error| error.to_string())
        .and_then(|built| {
            let near = eval_patch(&built, KR_LOADING_NEAR).map_err(|error| error.to_string())?;
            let far = eval_patch(&built, KR_LOADING_FAR).map_err(|error| error.to_string())?;
            Ok(spreading_error(
                near.loading.norm(),
                KR_LOADING_NEAR / wavenumber,
                far.loading.norm(),
                KR_LOADING_FAR / wavenumber,
            ))
        }) {
        Ok(error) => rows.push(Row::new(
            format!("{}.fwh-loading-spreading", spec.tag),
            error <= band,
            format!("{error:.3e}"),
            format!("{band:.3e}"),
            format!("kr {KR_LOADING_NEAR:.0e}->{KR_LOADING_FAR:.0e}"),
        )),
        Err(detail) => rows.push(Row::fail(
            format!("{}.fwh-loading-spreading", spec.tag),
            detail,
        )),
    }
    // Planar contour leg on the same paired histories.
    match fwh_contour_from_series(
        [0.0, 0.0],
        [1.0, 0.0],
        1.0,
        &pressure_fluct.fluctuation,
        &detrended[live],
        dt_s,
        tone.frequency_hz,
    )
    .map_err(|error| error.to_string())
    .and_then(|segment| {
        stationary_fwh_pressure_2d(
            std::slice::from_ref(&segment),
            [KR_SPREAD_NEAR / wavenumber, 0.0],
            omega,
            MEDIUM_DENSITY,
            MEDIUM_SOUND_SPEED,
        )
        .map_err(|error| error.to_string())
    }) {
        Ok(field) => {
            let gap = fwh_component_gap(&field);
            rows.push(Row::new(
                format!("{}.fwh2d-component-sum", spec.tag),
                gap <= band,
                format!("{gap:.3e}"),
                format!("{band:.3e}"),
                format!("kr={KR_SPREAD_NEAR:.0e}"),
            ));
        }
        Err(detail) => rows.push(Row::fail(
            format!("{}.fwh2d-component-sum", spec.tag),
            detail,
        )),
    }

    // Axisymmetric legs need rotation-invariant (Tzz-only) stress,
    // which only the axial INIV history provides; the transverse
    // COMPA1B history would converge to its hoop average instead.
    if spec.tag == "iniv" {
        let Some(held) = moment else {
            rows.push(Row::fail(
                format!("{}.ring-compact-limit", spec.tag),
                "missing quadrupole moment".to_string(),
            ));
            return rows;
        };
        let mut per_rad = held;
        for row in &mut per_rad {
            for entry in row.iter_mut() {
                *entry /= std::f64::consts::TAU;
            }
        }
        match ring_point_gap(
            per_rad,
            COMPACT_RADIUS,
            0.0,
            [0.0, 0.0, KR_SPREAD_NEAR / wavenumber],
            omega,
            MEDIUM_SOUND_SPEED,
            1e-12,
        ) {
            Ok(gap) => rows.push(Row::new(
                format!("{}.ring-compact-limit", spec.tag),
                gap <= band,
                format!("{gap:.3e}"),
                format!("{band:.3e}"),
                format!(
                    "a={COMPACT_RADIUS:.0e} m, ka~{:.1e}",
                    wavenumber * COMPACT_RADIUS
                ),
            )),
            Err(error) => rows.push(Row::fail(
                format!("{}.ring-compact-limit", spec.tag),
                error.to_string(),
            )),
        }
        let modal_velocity = match dft_peak_phasor(&detrended[live], dt_s, tone.frequency_hz) {
            Ok(phasor) => phasor,
            Err(error) => {
                rows.push(Row::fail(
                    format!("{}.revolved-cross-check", spec.tag),
                    error.to_string(),
                ));
                return rows;
            }
        };
        match FwhRevolvedPatch::new(
            [REVOLVED_RADIUS, -REVOLVED_HALF_HEIGHT],
            [REVOLVED_RADIUS, REVOLVED_HALF_HEIGHT],
            [1.0, 0.0],
            AzimuthalMode::axisymmetric(),
            tone.phasor,
            modal_velocity,
        )
        .map_err(|error| error.to_string())
        .and_then(|revolved| {
            RevolvedQuadrature::new(1e-10, 4)
                .map_err(|error| error.to_string())
                .and_then(|quadrature| {
                    revolved_cross_check(
                        &revolved,
                        [KR_SPREAD_NEAR / wavenumber, 0.0, 0.0],
                        omega,
                        MEDIUM_DENSITY,
                        MEDIUM_SOUND_SPEED,
                        quadrature,
                        REVOLVED_HOOP_PATCHES,
                    )
                    .map_err(|error| error.to_string())
                })
        }) {
            Ok(gap) => rows.push(Row::new(
                format!("{}.revolved-cross-check", spec.tag),
                gap <= band,
                format!("{gap:.3e}"),
                format!("{band:.3e}"),
                format!(
                    "R={REVOLVED_RADIUS} m, {} hoop tiles",
                    REVOLVED_HOOP_PATCHES
                ),
            )),
            Err(detail) => rows.push(Row::fail(
                format!("{}.revolved-cross-check", spec.tag),
                detail,
            )),
        }
    }
    rows
}

fn replay_deck(root: &std::path::Path, spec: &DeckSpec) -> Vec<Row> {
    let mut rows = Vec::new();
    let path = golden_path(root, spec.deck);
    let golden = match load_golden(&path) {
        Ok(loaded) => loaded,
        Err(error) => {
            return vec![Row::fail(
                format!("{}.load", spec.tag),
                format!("{}: {error}", path.display()),
            )];
        }
    };
    rows.push(probe_provenance(&golden, spec.tag));
    rows.push(probe_grid(&golden, spec.tag));
    if spec.tag == "compa1b" {
        for partition in &COMPA1B_PARTITIONS {
            rows.push(probe_partition(&golden, spec.tag, partition));
        }
    } else {
        rows.push(probe_iniv_mass(&golden));
    }
    let time_scale = match time_scale_seconds(&golden.meta.deck_units) {
        Ok(scale) => scale,
        Err(detail) => {
            rows.push(Row::fail(format!("{}.replay-tone", spec.tag), detail));
            return rows;
        }
    };
    let band = match channel(&golden, spec.pressure_channel) {
        Ok(found) => found.rel_tol,
        Err(detail) => {
            rows.push(Row::fail(format!("{}.replay-tone", spec.tag), detail));
            return rows;
        }
    };
    rows.extend(probe_radiation(&golden, spec, time_scale, band));
    rows
}

fn main() {
    let args = Args::parse();
    let root = args
        .goldens
        .clone()
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens"));
    let mut rows = Vec::new();
    for spec in &DECKS {
        rows.extend(replay_deck(&root, spec));
    }
    let mut failures = 0;
    for row in &rows {
        if row.passed {
            println!(
                "PASS {:<34} measured={:<14} bound={:<10} {}",
                row.id, row.measured, row.bound, row.detail
            );
        } else {
            failures += 1;
            println!(
                "FAIL {:<34} measured={:<14} bound={:<10} {}",
                row.id, row.measured, row.bound, row.detail
            );
        }
    }
    println!(
        "qa-fem-caa: {}/{} probes PASS",
        rows.len() - failures,
        rows.len()
    );
    if failures > 0 {
        println!("FAILED: qa-fem-caa");
        std::process::exit(1);
    }
    println!("SUCCESS: qa-fem-caa");
}
