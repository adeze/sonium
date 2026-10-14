//! PORT-OPTIM Step 11: product handoff for the winning port profile.
//!
//! This module exposes the frozen Step-7 campaign winner — control points plus
//! sampled `r(x)` — through the modeling crate's public API, so `sonium-sim-speaker`
//! (BEM product layer) or a cabinet design can consume the profile without
//! re-implementing the spline. There is deliberately no BEM/CEA-2034 coupling
//! here: the handoff is geometry plus the Tier-1 reproduction recipe, and any
//! product coupling is a separate, explicitly requested step.
//!
//! # Validated envelope
//!
//! Headline claims stay at the 2D/meridian tier (Tier 2, converged 252
//! samples/cycle timestep) cross-checked by the Tier-1 spline linear oracle at
//! the campaign tuning. The Step-9 promotion ladder passes on regenerated
//! validation meshes; the Step-8 3D spot-checks are nightly tier and must not
//! be reported as product proof (the high-drive THD leg disagrees and is
//! labelled `Investigate`). Nothing here upgrades any existing gate's status.
//!
//! # Inward-only limitation
//!
//! The search space is `r(x) <= R0` by construction (normalised variables
//! `u in [0, 1]` map to `r = R0 - u·(R0 - r_min)`), so outward flare — the
//! usual exit-separation fix — is out of scope. The honest campaign outcome is
//! the documented SC2 null result: at equal airflow the winner lowers THD but
//! raises the `U⁸` noise proxy, because any inward-only throat raises bulk
//! velocity. Relaxing to controlled outward flare is a follow-up decision for
//! the user, not a silent scope change.
//!
//! # Equal-airflow reporting rule
//!
//! A narrower port at equal voltage trivially shows less flow-borne noise, so
//! every winner-vs-straight comparison is reported at matched airflow (one
//! linear rescale step plus a verification run measuring the residual
//! honestly) alongside the equal-voltage deltas. Success Criterion 2 judges
//! the equal-airflow levels only.
//!
//! # Consumer path
//!
//! ```rust,no_run
//! use sonium_sim_speaker_modeling::port_flow::{
//!     PortOptimReferenceCase, reproduce_winner_tier1, winner_profile,
//!     winner_tier1_frequency_hz,
//! };
//!
//! let profile = winner_profile()?;
//! let case = PortOptimReferenceCase::default();
//! let frequency_hz = winner_tier1_frequency_hz(&case)?;
//! let tier1 = reproduce_winner_tier1(&case, frequency_hz, 0.01)?;
//! # Ok::<(), sonium_sim_speaker_modeling::port_flow::PortFlowError>(())
//! ```
//!
//! [`WinnerProfile`] is the serializable transfer record (control points plus
//! the sampled meridian); [`winner_profile`] rebuilds the validated spline;
//! [`reproduce_winner_tier1`] replays the Step-9 Tier-1 oracle for the winner
//! from public inputs alone.

use std::path::Path;

use serde::{Deserialize, Serialize};

use super::baseline::PortOptimReferenceCase;
use super::campaign::{
    CampaignArchiverError, campaign_f_target_hz, campaign_profile_options, profile_csv,
};
use super::error::PortFlowError;
use super::geometry3d::{SplineRoundPortGeometry, build_spline_round_port_mesh};
use super::optimize::{ProfileTemplate, campaign_template};
use super::profile::SplinePortProfile;
use super::promotion::{
    STEP9_TIER1_FLARE_LENGTH_M, Tier1TransferPoint, solve_step9_tier1,
};

/// Schema version for [`WinnerProfile`] JSON.
pub const WINNER_PROFILE_SCHEMA_VERSION: u32 = 1;

/// Provenance recorded in every [`WinnerProfile`].
pub const WINNER_PROFILE_PROVENANCE: &str = "PortOptimWinnerProfile[v1,N5InwardOnly]";

/// Control-point count behind the frozen winner (`N - 3 = 2` free radii).
pub const WINNER_N_CONTROL_POINTS: usize = 5;

/// Frozen Step-7 campaign winner: minimum full-`J` feasible non-straight
/// finalist at Vmax Tier 2, `[0.0153777489057228, 0.028578258339773364]`.
///
/// Archived in `fixtures/port_optim_step7/step7_report.json` (`winner_u`) with
/// the same value repeated in the Step-8/Step-9 reports.
pub const WINNER_U: [f64; 2] = [0.0153777489057228, 0.028578258339773364];

/// Sampled-meridian station count carried by [`WinnerProfile`].
///
/// Matches [`STEP7_PROFILE_CSV_STATIONS`](super::campaign::STEP7_PROFILE_CSV_STATIONS)
/// so the record reproduces `fixtures/port_optim_step7/winner_profile.csv`.
pub const WINNER_PROFILE_SAMPLE_STATIONS: usize = 201;

/// Campaign tuning target in Hz for the frozen reference case.
///
/// This is the Tier-0 duct tuning of the straight campaign tube, reproduced
/// here so consumers can cross-check
/// [`winner_tier1_frequency_hz`] without running the duct solve.
pub const WINNER_F_TARGET_HZ: f64 = 92.60634465596921;

/// Frozen Step-9 Tier-1 oracle impedance of the winner `[re, im]` in Pa·s/m³.
///
/// Reference mesh (0.01 m), flare extent 0.01 m, `exp(-i·ω·t)` RMS phasors at
/// [`WINNER_F_TARGET_HZ`]. Archived in
/// `fixtures/port_optim_step9/step9_report.json` (`tier1_winner`).
pub const WINNER_TIER1_IMPEDANCE_PA_S_PER_M3: [f64; 2] =
    [381.15152950342406, -48180.9850406475];

/// Frozen Step-9 Tier-1 oracle box-to-port flow ratio `[re, im]`.
pub const WINNER_TIER1_FLOW_RATIO: [f64; 2] = [63.352079993134424, 69.88844403726944];

/// Frozen Tier-1 true relative residual (solver gate: ≤ 1e-8).
pub const WINNER_TIER1_RELATIVE_RESIDUAL: f64 = 9.919348138116312e-11;

/// Frozen Tier-1 strong/weak exterior-flux relative difference.
pub const WINNER_TIER1_FLUX_RELATIVE_DIFFERENCE: f64 = 0.003063191921426119;

/// Frozen Tier-1 scalar pressure unknowns.
pub const WINNER_TIER1_SCALAR_DOFS: usize = 99666;

/// Frozen Tier-1 mesh node count (deterministic mesh build).
pub const WINNER_TIER1_MESH_NODES: usize = 13088;

/// Frozen Tier-1 mesh element count (deterministic mesh build).
pub const WINNER_TIER1_MESH_ELEMENTS: usize = 71145;

/// Evidence archive paths, relative to the modeling crate directory.
pub const WINNER_EVIDENCE_PATHS: [&str; 8] = [
    "fixtures/port_optim_baseline_v1.json",
    "fixtures/port_optim_step7/step7_report.json",
    "fixtures/port_optim_step7/history_campaign.json",
    "fixtures/port_optim_step7/pareto_front_tier2_coarse.json",
    "fixtures/port_optim_step7/winner_profile.csv",
    "fixtures/port_optim_step8/step8_report.json",
    "fixtures/port_optim_step9/step9_report.json",
    "fixtures/port_optim_step7/quota_profiles.json",
];

/// Frozen winner `u` as an owned vector (scale-free across tiers).
pub fn winner_u() -> Vec<f64> {
    WINNER_U.to_vec()
}

/// Campaign-scale profile template behind the winner (`R0`, `L`,
/// `r_min = 0.6·R0`, plan-default options).
pub fn winner_template(case: &PortOptimReferenceCase) -> ProfileTemplate {
    campaign_template(case, WINNER_N_CONTROL_POINTS, campaign_profile_options())
}

/// Rebuild the frozen winner spline at campaign scale.
///
/// Uses the frozen [`PortOptimReferenceCase`] defaults; prefer
/// [`winner_profile_for_case`] when the case is already at hand.
pub fn winner_profile() -> Result<SplinePortProfile, PortFlowError> {
    winner_profile_for_case(&PortOptimReferenceCase::default())
}

/// Rebuild the frozen winner spline for one reference case.
pub fn winner_profile_for_case(
    case: &PortOptimReferenceCase,
) -> Result<SplinePortProfile, PortFlowError> {
    case.validate()?;
    winner_template(case).unpack(&winner_u())
}

/// Campaign tuning target: Tier-0 duct tuning of the straight campaign tube.
///
/// This is the Tier-1 oracle frequency for the winner reproduction; on the
/// frozen reference case it equals [`WINNER_F_TARGET_HZ`].
pub fn winner_tier1_frequency_hz(case: &PortOptimReferenceCase) -> Result<f64, PortFlowError> {
    case.validate()?;
    let template = winner_template(case);
    campaign_f_target_hz(case, &template)
}

/// Tier-1 envelope geometry meshing the winner profile.
///
/// Campaign chambers from `case`, smoothstep extent
/// [`STEP9_TIER1_FLARE_LENGTH_M`], automatic angular sectors — the exact
/// envelope [`solve_step9_tier1`] solves in.
pub fn winner_tier1_geometry(
    case: &PortOptimReferenceCase,
    mesh_size_m: f64,
) -> Result<SplineRoundPortGeometry, PortFlowError> {
    case.validate()?;
    if !(mesh_size_m.is_finite() && mesh_size_m > 0.0) {
        return Err(PortFlowError::InvalidConfig(format!(
            "winner Tier-1 geometry needs a finite positive mesh size, got {mesh_size_m}"
        )));
    }
    let profile = winner_profile_for_case(case)?;
    Ok(SplineRoundPortGeometry {
        duct_length_m: case.port_length_l_m,
        bore_radius_m: case.port_radius_r0_m,
        chamber_length_m: case.tier1_chamber_length_m,
        chamber_radius_m: case.tier1_chamber_radius_m,
        flare_length_m: STEP9_TIER1_FLARE_LENGTH_M,
        mesh_size_m,
        angular_sectors: None,
        profile,
    })
}

/// Reproduce the winner's Tier-1 transfer from public inputs alone.
///
/// Meshes [`winner_tier1_geometry`] and runs the Step-4 spline linear oracle
/// at `frequency_hz` (normally [`winner_tier1_frequency_hz`]). At the
/// reference mesh
/// ([`STEP9_TIER1_MESH_SIZE_M`](super::promotion::STEP9_TIER1_MESH_SIZE_M))
/// the impedance reproduces
/// [`WINNER_TIER1_IMPEDANCE_PA_S_PER_M3`]; the solve carries its own
/// residual ≤ 1e-8 gate and flux audit.
pub fn reproduce_winner_tier1(
    case: &PortOptimReferenceCase,
    frequency_hz: f64,
    mesh_size_m: f64,
) -> Result<Tier1TransferPoint, PortFlowError> {
    let profile = winner_profile_for_case(case)?;
    solve_step9_tier1(case, &profile, frequency_hz, mesh_size_m)
}

/// Serializable transfer record: control points plus sampled `r(x)`.
///
/// A downstream consumer reads this record (or its JSON) and either uses the
/// sampled meridian directly or rebuilds the validated spline with
/// [`WinnerProfile::profile`] — no spline re-implementation on either path.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WinnerProfile {
    /// Must equal [`WINNER_PROFILE_SCHEMA_VERSION`].
    pub schema_version: u32,
    /// Model lineage; must equal [`WINNER_PROFILE_PROVENANCE`].
    pub provenance: String,
    /// Control-point count `N` (frozen winner: 5).
    pub n_control_points: usize,
    /// Port length `L` in m.
    pub length_m: f64,
    /// Straight-bore radius `R0` in m.
    pub bore_radius_m: f64,
    /// Minimum admissible radius `r_min` in m.
    pub throat_radius_m: f64,
    /// Normalised free radii `u` (knots `2..N-1`).
    pub u: Vec<f64>,
    /// Fixed axial knot positions in m.
    pub knots_m: Vec<f64>,
    /// Absolute radii at the knots in m.
    pub radii_m: Vec<f64>,
    /// Sampled meridian `(x, r)` stations over `[0, L]`.
    pub sampled: Vec<[f64; 2]>,
    /// Campaign tuning target in Hz (Tier-1 oracle frequency).
    pub f_target_hz: f64,
    /// One-line validated-envelope statement carried with the data.
    pub envelope: String,
}

impl WinnerProfile {
    /// Capture the frozen winner record for the frozen reference case.
    pub fn capture() -> Result<Self, PortFlowError> {
        Self::capture_for_case(&PortOptimReferenceCase::default())
    }

    /// Capture the frozen winner record for one reference case.
    pub fn capture_for_case(case: &PortOptimReferenceCase) -> Result<Self, PortFlowError> {
        case.validate()?;
        let profile = winner_profile_for_case(case)?;
        let f_target_hz = winner_tier1_frequency_hz(case)?;
        Ok(Self {
            schema_version: WINNER_PROFILE_SCHEMA_VERSION,
            provenance: WINNER_PROFILE_PROVENANCE.to_string(),
            n_control_points: profile.n_control_points(),
            length_m: profile.length_m(),
            bore_radius_m: profile.bore_radius_m(),
            throat_radius_m: profile.throat_radius_m(),
            // The exact frozen vector, not `profile.normalized()`: the
            // radius round-trip differs in the last ulp, and the record
            // must pin the archived `u` bit-identically.
            u: winner_u(),
            knots_m: profile.knots().to_vec(),
            radii_m: profile.radii().to_vec(),
            sampled: profile.sampled_profile(WINNER_PROFILE_SAMPLE_STATIONS),
            f_target_hz,
            envelope: "2D/meridian Tier 2 + Tier-1 oracle; inward-only r <= R0; \
                 equal-airflow reporting; 3D nightly only"
                .to_string(),
        })
    }

    /// Rebuild the validated winner spline from the stored control points.
    ///
    /// Routes through the campaign template ([`winner_template`]), so the
    /// dense-grid validation (bore bound, straight start, slope, curvature)
    /// re-audits the record before anything can solve on it.
    pub fn profile(&self) -> Result<SplinePortProfile, PortFlowError> {
        SplinePortProfile::from_normalized(
            self.n_control_points,
            self.length_m,
            self.bore_radius_m,
            self.throat_radius_m,
            &self.u,
            campaign_profile_options(),
        )
    }

    /// Sampled `(x, r)` plot data as CSV text with a header row.
    pub fn to_csv(&self) -> Result<String, PortFlowError> {
        Ok(profile_csv(&self.profile()?, self.sampled.len().max(2)))
    }

    /// Serialize the record to pretty JSON (deterministic field order).
    pub fn to_json_string(&self) -> Result<String, CampaignArchiverError> {
        serde_json::to_string_pretty(self)
            .map_err(|error| CampaignArchiverError::Json(error.to_string()))
    }

    /// Write the record to a JSON file.
    pub fn save_json(&self, path: &Path) -> Result<(), CampaignArchiverError> {
        std::fs::write(path, self.to_json_string()?)
            .map_err(|error| CampaignArchiverError::Io(error.to_string()))
    }

    /// Parse record JSON, rejecting schema or provenance drift loudly.
    pub fn from_json_str(raw: &str) -> Result<Self, CampaignArchiverError> {
        let record: WinnerProfile = serde_json::from_str(raw)
            .map_err(|error| CampaignArchiverError::Json(error.to_string()))?;
        if record.schema_version != WINNER_PROFILE_SCHEMA_VERSION {
            return Err(CampaignArchiverError::Json(format!(
                "unsupported winner-profile schema {}, expected {WINNER_PROFILE_SCHEMA_VERSION}",
                record.schema_version
            )));
        }
        if record.provenance != WINNER_PROFILE_PROVENANCE {
            return Err(CampaignArchiverError::Json(format!(
                "winner-profile provenance drift: {}",
                record.provenance
            )));
        }
        Ok(record)
    }

    /// Structural validation: schema, geometry coherence, control-point and
    /// sampled-meridian agreement with a rebuilt spline.
    ///
    /// A structurally valid record still only claims what the archived tier
    /// evidence behind it claims (see the module docs); this rejects
    /// incoherent or tampered records, never invents measurements.
    pub fn validate(&self) -> Result<(), CampaignArchiverError> {
        let bad = |message: String| CampaignArchiverError::Json(message);
        if self.schema_version != WINNER_PROFILE_SCHEMA_VERSION {
            return Err(bad(format!(
                "winner-profile schema {} != {WINNER_PROFILE_SCHEMA_VERSION}",
                self.schema_version
            )));
        }
        if self.provenance != WINNER_PROFILE_PROVENANCE {
            return Err(bad(format!(
                "winner-profile provenance drift: {}",
                self.provenance
            )));
        }
        if self.n_control_points != self.knots_m.len()
            || self.n_control_points != self.radii_m.len()
        {
            return Err(bad(format!(
                "winner-profile control-point count {} disagrees with {} knots / {} radii",
                self.n_control_points,
                self.knots_m.len(),
                self.radii_m.len()
            )));
        }
        if self.u.len() != self.n_control_points.saturating_sub(3) {
            return Err(bad(format!(
                "winner-profile with N = {} needs {} free radii, got {}",
                self.n_control_points,
                self.n_control_points.saturating_sub(3),
                self.u.len()
            )));
        }
        if self.sampled.len() < 2 {
            return Err(bad("winner-profile needs at least 2 sampled stations".to_string()));
        }
        let profile = self
            .profile()
            .map_err(|error| CampaignArchiverError::Json(error.to_string()))?;
        for (index, (knot, radius)) in
            self.knots_m.iter().zip(self.radii_m.iter()).enumerate()
        {
            let want_x = profile.knots()[index];
            let want_r = profile.radii()[index];
            if (knot - want_x).abs() > 1.0e-12 || (radius - want_r).abs() > 1.0e-12 {
                return Err(bad(format!(
                    "winner-profile control point {index} ({knot}, {radius}) disagrees \
                     with the rebuilt spline ({want_x}, {want_r})"
                )));
            }
        }
        for (index, station) in self.sampled.iter().enumerate() {
            let want_r = profile.local_radius_m(station[0]);
            if (station[1] - want_r).abs() > 1.0e-9 {
                return Err(bad(format!(
                    "winner-profile sampled station {index} at x = {} disagrees: {} vs {want_r}",
                    station[0], station[1]
                )));
            }
            if station[1] > self.bore_radius_m + 1.0e-12 {
                return Err(bad(format!(
                    "winner-profile sampled station {index} escapes the bore: {} > {}",
                    station[1], self.bore_radius_m
                )));
            }
        }
        if !(self.f_target_hz.is_finite() && self.f_target_hz > 0.0) {
            return Err(bad(format!(
                "winner-profile f_target must be finite and positive, got {}",
                self.f_target_hz
            )));
        }
        Ok(())
    }
}

/// Build the Tier-1 mesh behind [`winner_tier1_geometry`] (no solve).
///
/// Exposed so a consumer (or a fast test) can check mesh determinism — node
/// and element counts reproduce [`WINNER_TIER1_MESH_NODES`] /
/// [`WINNER_TIER1_MESH_ELEMENTS`] at the reference mesh size — without paying
/// for the reference solve.
pub fn winner_tier1_mesh_node_counts(
    case: &PortOptimReferenceCase,
    mesh_size_m: f64,
) -> Result<(usize, usize), PortFlowError> {
    let geometry = winner_tier1_geometry(case, mesh_size_m)?;
    let mesh = build_spline_round_port_mesh(&geometry)?;
    Ok((mesh.num_nodes(), mesh.elements.len()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn winner_rebuilds_and_validates() {
        let profile = winner_profile().expect("winner must build");
        profile.validate().expect("winner must validate");
        assert_eq!(profile.n_control_points(), WINNER_N_CONTROL_POINTS);
        for (got, want) in profile.normalized().iter().zip(winner_u().iter()) {
            assert!(
                (got - want).abs() < 1.0e-15,
                "winner u round-trip drift: {got} vs {want}"
            );
        }
        assert!(
            profile.slope(0.0).abs() < 1.0e-12,
            "winner must keep the straight start"
        );
        for station in profile.sampled_profile(WINNER_PROFILE_SAMPLE_STATIONS) {
            assert!(
                station[1] <= profile.bore_radius_m() + 1.0e-12,
                "winner must stay inward-only"
            );
            assert!(
                station[1] >= profile.throat_radius_m() - 1.0e-12,
                "winner must respect the throat"
            );
        }
    }

    #[test]
    fn winner_tuning_matches_frozen_target() {
        let case = PortOptimReferenceCase::default();
        let f_target = winner_tier1_frequency_hz(&case).expect("tuning must resolve");
        assert!(
            (f_target - WINNER_F_TARGET_HZ).abs() < 1.0e-9,
            "f_target drift: {f_target} vs {WINNER_F_TARGET_HZ}"
        );
    }

    #[test]
    fn record_captures_control_points_and_sampled_meridian() {
        let record = WinnerProfile::capture().expect("record must capture");
        record.validate().expect("record must validate");
        assert_eq!(record.u, winner_u());
        assert_eq!(record.n_control_points, WINNER_N_CONTROL_POINTS);
        assert_eq!(record.sampled.len(), WINNER_PROFILE_SAMPLE_STATIONS);
        assert!((record.sampled[0][0] - 0.0).abs() < 1.0e-15);
        assert!((record.sampled[WINNER_PROFILE_SAMPLE_STATIONS - 1][0] - record.length_m).abs() < 1.0e-12);
        let rebuilt = record.profile().expect("record must rebuild");
        for (got, want) in rebuilt.normalized().iter().zip(winner_u().iter()) {
            assert!(
                (got - want).abs() < 1.0e-15,
                "rebuilt u round-trip drift: {got} vs {want}"
            );
        }
        let csv = record.to_csv().expect("record must render CSV");
        assert!(csv.starts_with("x_m,r_m\n"));
        assert_eq!(csv.lines().count(), WINNER_PROFILE_SAMPLE_STATIONS + 1);
    }

    #[test]
    fn record_json_round_trips_and_fails_closed() {
        let record = WinnerProfile::capture().expect("record must capture");
        let json = record.to_json_string().expect("record must serialize");
        let parsed = WinnerProfile::from_json_str(&json).expect("round-trip must parse");
        parsed.validate().expect("parsed record must validate");
        assert_eq!(parsed, record);
        let drifted = json.replace(WINNER_PROFILE_PROVENANCE, "PortOptimWinnerProfile[v0,drift]");
        assert!(WinnerProfile::from_json_str(&drifted).is_err());
        let bad_schema = json.replace("\"schema_version\": 1", "\"schema_version\": 99");
        assert!(WinnerProfile::from_json_str(&bad_schema).is_err());
        let mut tampered = parsed.clone();
        tampered.sampled[100][1] += 1.0e-6;
        assert!(
            tampered.validate().is_err(),
            "tampered meridian must fail validation"
        );
        let mut escaped = parsed.clone();
        escaped.sampled[100][1] = escaped.bore_radius_m + 1.0e-6;
        assert!(
            escaped.validate().is_err(),
            "bore escape must fail validation"
        );
    }
}
