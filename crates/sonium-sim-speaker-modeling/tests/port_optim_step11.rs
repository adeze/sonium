//! PORT-OPTIM Step 11: product handoff and documentation.
//!
//! Default (fast, solver-free) tests guard the consumer path: the frozen
//! winner rebuilds and validates, the [`WinnerProfile`] record captures the
//! control points plus the sampled meridian, its JSON round-trips and fails
//! closed, the record agrees with the committed Step-7 archive
//! (`step7_report.json` + `winner_profile.csv`), the frozen Tier-1 constants
//! agree with the committed Step-9 archive, and the Tier-1 mesh behind the
//! reproduction recipe builds deterministically (no solve).
//!
//! The ignored `step11_winner_tier1_reproduction` test runs the real Tier-1
//! oracle replay ([`reproduce_winner_tier1`] at the reference mesh) and
//! asserts the impedance reproduces the frozen Step-9 value. Release-only:
//!
//! ```sh
//! cargo test -p sonium-sim-speaker-modeling --release \
//!   --test port_optim_step11 -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use sonium_sim_speaker_modeling::port_flow::{
    PortOptimReferenceCase, STEP9_TIER1_MESH_SIZE_M, Step7Report, Step9Report, WINNER_F_TARGET_HZ,
    WINNER_N_CONTROL_POINTS, WINNER_PROFILE_PROVENANCE, WINNER_PROFILE_SAMPLE_STATIONS,
    WINNER_TIER1_FLOW_RATIO, WINNER_TIER1_FLUX_RELATIVE_DIFFERENCE,
    WINNER_TIER1_IMPEDANCE_PA_S_PER_M3, WINNER_TIER1_MESH_ELEMENTS, WINNER_TIER1_MESH_NODES,
    WINNER_TIER1_RELATIVE_RESIDUAL, WINNER_TIER1_SCALAR_DOFS, WINNER_U, WinnerProfile,
    reproduce_winner_tier1, winner_profile, winner_tier1_frequency_hz,
    winner_tier1_mesh_node_counts, winner_u,
};

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("fixtures")
        .join(name)
}

fn complex_rel_error(got: [f64; 2], want: [f64; 2]) -> f64 {
    let diff = ((got[0] - want[0]).powi(2) + (got[1] - want[1]).powi(2)).sqrt();
    let scale = (want[0].powi(2) + want[1].powi(2)).sqrt();
    diff / scale
}

#[test]
fn winner_record_captures_frozen_campaign_winner() {
    let record = WinnerProfile::capture().expect("record must capture");
    record.validate().expect("record must validate");
    assert_eq!(record.u, winner_u());
    assert_eq!(record.u, WINNER_U.to_vec());
    assert_eq!(record.n_control_points, WINNER_N_CONTROL_POINTS);
    assert_eq!(record.provenance, WINNER_PROFILE_PROVENANCE);
    assert_eq!(record.sampled.len(), WINNER_PROFILE_SAMPLE_STATIONS);
    assert!(
        (record.f_target_hz - WINNER_F_TARGET_HZ).abs() < 1.0e-9,
        "record tuning must match the frozen target"
    );
    let profile = winner_profile().expect("winner must rebuild");
    profile.validate().expect("winner must validate");
    for (got, want) in profile.normalized().iter().zip(record.u.iter()) {
        assert!(
            (got - want).abs() < 1.0e-15,
            "winner u round-trip drift: {got} vs {want}"
        );
    }
    // Inward-only by construction, on the dense grid.
    for station in profile.sampled_profile(WINNER_PROFILE_SAMPLE_STATIONS) {
        assert!(station[1] <= profile.bore_radius_m() + 1.0e-12);
        assert!(station[1] >= profile.throat_radius_m() - 1.0e-12);
    }
}

#[test]
fn winner_record_agrees_with_step7_archive() {
    let raw = std::fs::read_to_string(fixture_path("port_optim_step7/step7_report.json"))
        .expect("step-7 report fixture must exist");
    let report = Step7Report::from_json_str(&raw).expect("step-7 report must parse");
    report.validate().expect("step-7 report must validate");
    assert_eq!(
        report.winner_u,
        winner_u(),
        "handoff must pin the archived winner"
    );
    assert!(
        (report.f_target_hz - WINNER_F_TARGET_HZ).abs() < 1.0e-9,
        "handoff tuning must match the archived campaign target"
    );
    // Sampled meridian reproduces the committed winner CSV.
    let csv = std::fs::read_to_string(fixture_path("port_optim_step7/winner_profile.csv"))
        .expect("winner CSV fixture must exist");
    let mut lines = csv.lines();
    assert_eq!(lines.next(), Some("x_m,r_m"));
    let profile = winner_profile().expect("winner must rebuild");
    let mut rows = 0;
    for line in lines {
        let (x_text, r_text) = line.split_once(',').expect("CSV row must have two columns");
        let (x, r): (f64, f64) = (
            x_text.parse().expect("x must parse"),
            r_text.parse().expect("r must parse"),
        );
        assert!(
            (profile.local_radius_m(x) - r).abs() < 1.0e-11,
            "sampled r(x) must reproduce the committed CSV at x = {x}"
        );
        rows += 1;
    }
    assert_eq!(rows, WINNER_PROFILE_SAMPLE_STATIONS);
}

#[test]
fn winner_tier1_constants_agree_with_step9_archive() {
    let raw = std::fs::read_to_string(fixture_path("port_optim_step9/step9_report.json"))
        .expect("step-9 report fixture must exist");
    let report = Step9Report::from_json_str(&raw).expect("step-9 report must parse");
    assert_eq!(
        report.winner_u,
        winner_u(),
        "step-9 must archive the same winner"
    );
    let tier1 = &report.tier1_winner;
    assert_eq!(tier1.scalar_dofs, WINNER_TIER1_SCALAR_DOFS);
    assert_eq!(tier1.mesh_nodes, WINNER_TIER1_MESH_NODES);
    assert_eq!(tier1.mesh_elements, WINNER_TIER1_MESH_ELEMENTS);
    assert!(
        complex_rel_error(tier1.port_impedance, WINNER_TIER1_IMPEDANCE_PA_S_PER_M3) < 1.0e-15,
        "frozen impedance must match the archive: {:?}",
        tier1.port_impedance
    );
    assert!(
        complex_rel_error(tier1.flow_ratio, WINNER_TIER1_FLOW_RATIO) < 1.0e-15,
        "frozen flow ratio must match the archive"
    );
    assert!(
        (tier1.relative_residual - WINNER_TIER1_RELATIVE_RESIDUAL).abs()
            / WINNER_TIER1_RELATIVE_RESIDUAL
            < 1.0e-12,
        "frozen residual must match the archive"
    );
    assert!(
        (tier1.flux_relative_difference - WINNER_TIER1_FLUX_RELATIVE_DIFFERENCE).abs()
            / WINNER_TIER1_FLUX_RELATIVE_DIFFERENCE
            < 1.0e-12,
        "frozen flux audit must match the archive"
    );
}

#[test]
fn winner_record_json_round_trips_and_fails_closed() {
    let record = WinnerProfile::capture().expect("record must capture");
    let json = record.to_json_string().expect("record must serialize");
    let parsed = WinnerProfile::from_json_str(&json).expect("round-trip must parse");
    parsed.validate().expect("parsed record must validate");
    assert_eq!(parsed, record);
    let drifted = json.replace(
        WINNER_PROFILE_PROVENANCE,
        "PortOptimWinnerProfile[v0,drift]",
    );
    assert!(WinnerProfile::from_json_str(&drifted).is_err());
    let bad_schema = json.replace("\"schema_version\": 1", "\"schema_version\": 99");
    assert!(WinnerProfile::from_json_str(&bad_schema).is_err());
    // Tampered control data fails validation, never silently shifts the spline.
    let mut bad_u = parsed.clone();
    bad_u.u[0] += 1.0e-6;
    assert!(bad_u.validate().is_err(), "tampered u must fail validation");
    let mut bad_radii = parsed.clone();
    bad_radii.radii_m[2] += 1.0e-6;
    assert!(
        bad_radii.validate().is_err(),
        "tampered radii must fail validation"
    );
}

#[test]
fn winner_tier1_mesh_build_is_deterministic() {
    let case = PortOptimReferenceCase::default();
    let (nodes, elements) = winner_tier1_mesh_node_counts(&case, STEP9_TIER1_MESH_SIZE_M)
        .expect("Tier-1 mesh must build");
    assert_eq!(
        nodes, WINNER_TIER1_MESH_NODES,
        "mesh nodes must reproduce the archive"
    );
    assert_eq!(
        elements, WINNER_TIER1_MESH_ELEMENTS,
        "mesh elements must reproduce the archive"
    );
}

// ---------------------------------------------------------------------------
// Ignored release-only Tier-1 reproduction
// ---------------------------------------------------------------------------

type HandoffError = Box<dyn std::error::Error + Send + Sync>;

/// Full Step-11 consumer proof: load the winner from the public API and replay
/// its Tier-1 impedance at the reference mesh.
#[test]
#[ignore = "Step-11 Tier-1 reproduction; run explicitly in release mode"]
fn step11_winner_tier1_reproduction() -> Result<(), HandoffError> {
    if cfg!(debug_assertions) {
        return Err("run the Step-11 reproduction in release mode (cargo test \
             -p sonium-sim-speaker-modeling --release --test port_optim_step11 \
             -- --ignored --nocapture)"
            .into());
    }
    let case = PortOptimReferenceCase::default();
    let frequency_hz = winner_tier1_frequency_hz(&case).map_err(|error| error.to_string())?;
    assert!(
        (frequency_hz - WINNER_F_TARGET_HZ).abs() < 1.0e-9,
        "oracle frequency must match the frozen target"
    );
    let tier1 = reproduce_winner_tier1(&case, frequency_hz, STEP9_TIER1_MESH_SIZE_M)
        .map_err(|error| error.to_string())?;
    for (got, want) in tier1.profile_normalized.iter().zip(winner_u().iter()) {
        assert!(
            (got - want).abs() < 1.0e-15,
            "replayed u identity drift: {got} vs {want}"
        );
    }
    assert_eq!(tier1.scalar_dofs, WINNER_TIER1_SCALAR_DOFS);
    assert_eq!(tier1.mesh_nodes, WINNER_TIER1_MESH_NODES);
    assert_eq!(tier1.mesh_elements, WINNER_TIER1_MESH_ELEMENTS);
    assert!(
        tier1.relative_residual <= 1.0e-8,
        "replayed solve must pass the residual gate: {}",
        tier1.relative_residual
    );
    let impedance_error =
        complex_rel_error(tier1.port_impedance, WINNER_TIER1_IMPEDANCE_PA_S_PER_M3);
    assert!(
        impedance_error < 1.0e-6,
        "replayed impedance {:?} must reproduce {WINNER_TIER1_IMPEDANCE_PA_S_PER_M3:?} \
         (rel err {impedance_error:.3e})",
        tier1.port_impedance
    );
    println!(
        "step-11 Tier-1 reproduction: Z = {:?} Pa·s/m³ (rel err {impedance_error:.3e}), \
         residual {:.3e}, flux {:.3e}",
        tier1.port_impedance, tier1.relative_residual, tier1.flux_relative_difference
    );
    Ok(())
}
