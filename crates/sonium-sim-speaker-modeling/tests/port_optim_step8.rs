//! PORT-OPTIM Step 8: finalist noise evaluation and nightly 3D spot-check.
//!
//! Default (fast) tests guard the Step-8 machinery without solving: the noise
//! spec matches the production snapshot stream, the campaign winner revolves
//! into a valid 3D spline round-port mesh, and the committed evidence archive
//! (when present) parses and validates.
//!
//! The ignored `step8_finalist_noise_and_3d_spotcheck` test runs the real leg
//! (snapshot Tier-2 noise for winner + straight at three drives, 3D coupled
//! spot-checks at low + high drive) and writes
//! `fixtures/port_optim_step8/step8_report.json`. Release-only:
//!
//! ```sh
//! cargo test -p sonium-sim-speaker-modeling --release \
//!   --test port_optim_step8 -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use sonium_sim_speaker_modeling::PortNoiseMethod;
use sonium_sim_speaker_modeling::port_flow::{
    MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL, PortOptimReferenceCase,
    STEP8_NOISE_HARMONICS, STEP8_REPORT_PROVENANCE, Step7Report, Step8Report,
    build_spline_round_port_mesh, campaign_profile_options, campaign_template, finalist_noise_spec,
    noise_observer_1m_on_axis, noise_snapshot_fidelity, noise_snapshot_stride, spline_3d_geometry,
    spotcheck_template,
};

fn step7_report_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/port_optim_step7/step7_report.json")
}

fn step8_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/port_optim_step8")
}

fn step8_report_path() -> PathBuf {
    step8_dir().join("step8_report.json")
}

fn load_step7_winner() -> Option<(PortOptimReferenceCase, f64, Vec<f64>)> {
    let path = step7_report_path();
    if !path.exists() {
        return None;
    }
    let text = std::fs::read_to_string(&path).ok()?;
    let report = Step7Report::from_json_str(&text).ok()?;
    Some((report.case, report.f_target_hz, report.winner_u))
}

#[test]
fn noise_spec_matches_production_snapshot_stream() {
    let case = PortOptimReferenceCase::default();
    let fidelity = noise_snapshot_fidelity(&case).expect("snapshot fidelity must build");
    assert_eq!(fidelity.snapshot_interval_steps, 21);
    let drive_hz = case
        .tier2_drive_frequency_hz()
        .expect("Tier-2 drive must resolve");
    let dt = 1.0 / (drive_hz * fidelity.samples_per_cycle as f64);
    let snapshot_dt = fidelity.snapshot_interval_steps as f64 * dt;
    let spec = finalist_noise_spec(
        drive_hz,
        snapshot_dt,
        noise_observer_1m_on_axis(&case),
        case.tier2_duct_length_m,
        1.0,
    )
    .expect("noise spec must build");
    assert_eq!(spec.method, PortNoiseMethod::Automatic);
    assert_eq!(spec.frequencies_hz.len(), STEP8_NOISE_HARMONICS);
    spec.validate(snapshot_dt)
        .expect("spec must validate at the snapshot rate");
    let tail_snapshots =
        fidelity.tail_cycles * fidelity.samples_per_cycle / fidelity.snapshot_interval_steps;
    assert_eq!(tail_snapshots, 96);
    println!(
        "step-8 noise: f1={drive_hz:.6} Hz, snapshot_dt={snapshot_dt:.6e} s, \
         freqs={:?}, observer={:?}",
        spec.frequencies_hz, spec.observers[0].position_m,
    );
}

#[test]
fn spline_3d_mesh_revolves_winner_profile() {
    let Some((case, _, winner_u)) = load_step7_winner() else {
        println!("step-7 archive absent (run --ignored campaign first); nothing to revolve");
        return;
    };
    let template = campaign_template(&case, 5, campaign_profile_options());
    let winner = template
        .unpack(&winner_u)
        .expect("winner must unpack at campaign scale");
    let geometry = spline_3d_geometry(&case, &winner, 0.02).expect("3D geometry must build");
    let mesh = build_spline_round_port_mesh(&geometry).expect("3D mesh must build");
    assert_eq!(mesh.dimension, 3);
    assert!(!mesh.elements.is_empty());
    for marker in [MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL] {
        assert!(
            mesh.boundaries.iter().any(|facet| facet.marker == marker),
            "3D mesh lost marker {marker}"
        );
    }
    // Wall nodes on duct stations follow the spline within one cell size.
    let duct_wall: Vec<(f64, f64)> = mesh
        .boundaries
        .iter()
        .filter(|facet| facet.marker == MARKER_WALL)
        .flat_map(|facet| facet.nodes.iter())
        .map(|&node| {
            let point = mesh.nodes[node];
            (point.x, point.y.hypot(point.z))
        })
        .filter(|(x, _)| *x >= 0.0 && *x <= case.port_length_l_m)
        .collect();
    assert!(!duct_wall.is_empty());
    let mut worst = 0.0f64;
    for (x, radius) in &duct_wall {
        let expected = winner.local_radius_m(*x);
        worst = worst.max((radius - expected).abs());
    }
    assert!(
        worst <= 0.02,
        "wall deviates {worst:.3e} m from the spline (mesh 0.02 m)"
    );
    // The dip is real: the winner throat sits below the bore somewhere.
    let throat = winner
        .sampled_profile(201)
        .iter()
        .map(|station| station[1])
        .fold(f64::INFINITY, f64::min);
    assert!(throat < case.port_radius_r0_m);
    println!(
        "step-8 3D mesh: nodes={}, tets={}, facets={}, wall_pts={}, worst_wall_dev={worst:.3e} m",
        mesh.num_nodes(),
        mesh.elements.len(),
        mesh.boundaries.len(),
        duct_wall.len(),
    );
}

#[test]
fn spotcheck_template_matches_campaign_scale() {
    let case = PortOptimReferenceCase::default();
    let template = spotcheck_template(&case);
    let straight = template.straight().expect("straight must build");
    assert!((straight.length_m() - case.port_length_l_m).abs() <= 1.0e-12);
    assert!((straight.bore_radius_m() - case.port_radius_r0_m).abs() <= 1.0e-12);
}

#[test]
fn step8_archive_validates_when_present() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = step8_report_path();
    if !path.exists() {
        println!("step-8 archive absent (run --ignored leg); nothing to validate");
        return Ok(());
    }
    let text = std::fs::read_to_string(&path)?;
    let report = Step8Report::from_json_str(&text).map_err(|error| match error {
        sonium_sim_speaker_modeling::port_flow::CampaignArchiverError::Io(message)
        | sonium_sim_speaker_modeling::port_flow::CampaignArchiverError::Json(message) => message,
    })?;
    assert_eq!(report.provenance, STEP8_REPORT_PROVENANCE);
    report.validate().map_err(|error| match error {
        sonium_sim_speaker_modeling::port_flow::CampaignArchiverError::Io(message)
        | sonium_sim_speaker_modeling::port_flow::CampaignArchiverError::Json(message) => message,
    })?;
    println!("integrated noise floor @1m on-axis (dB re 20uPa):");
    for level in &report.noise {
        println!(
            "  V={:.2} V: straight={:.3} dB winner={:.3} dB delta={:+.3} dB (n={})",
            level.voltage_peak_v,
            level.straight.integrated_noise_floor_db,
            level.winner.integrated_noise_floor_db,
            level.delta_integrated_db,
            level.n_snapshots,
        );
    }
    for level in &report.spot3d {
        println!(
            "  3D[{}] V={:.2} V: q_ratio 2D={:.4} 3D={:.4} (dis {:.2}%), \
             thd_ratio 2D={:.4} 3D={:.4} (dis {:.2}%) -> {:?}",
            level.tier,
            level.voltage_peak_v,
            level.q_ratio_2d,
            level.q_ratio_3d,
            100.0 * level.q_disagreement_rel,
            level.thd_ratio_2d,
            level.thd_ratio_3d,
            100.0 * level.thd_disagreement_rel,
            level.verdict,
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Ignored release-only leg
// ---------------------------------------------------------------------------

type Step8Error = Box<dyn std::error::Error + Send + Sync>;

/// Full Step-8 leg: snapshot noise (winner + straight, three drives) and
/// nightly 3D spot-checks (low + high drive). Writes
/// `fixtures/port_optim_step8/step8_report.json`.
#[test]
#[ignore = "Step-8 finalist noise + 3D spot-check; run explicitly in release mode"]
fn step8_finalist_noise_and_3d_spotcheck() -> Result<(), Step8Error> {
    if cfg!(debug_assertions) {
        return Err(
            "run the Step-8 leg in release mode (cargo test -p sonium-sim-speaker-modeling \
             --release --test port_optim_step8 -- --ignored --nocapture)"
                .into(),
        );
    }

    use rayon::prelude::*;
    use sonium_sim_speaker_modeling::port_flow::{
        FinalistNoiseLevel, PortCandidateMetrics, PortEvalDrive, STEP7_TIER2_DRIVES_V_PEAK,
        STEP8_SPOTCHECK_DRIVES_V_PEAK, SmallSignalReference, SpotCheck3dLevel, SpotCheck3dRun,
        Tier2EvalContext, Tier2EvalOutcome, analyze_tail_noise, campaign_f_target_hz,
        spot_drive_frequency_hz, tail_snapshot_window,
    };
    use sonium_sim_speaker_modeling::port_flow::{SpotCheck3dFidelity, run_3d_spot_check};

    let Some((case, f_target_archive, winner_u)) = load_step7_winner() else {
        return Err("step-7 archive missing; run the Step-7 campaign first".into());
    };
    case.validate().map_err(|error| error.to_string())?;
    let template = campaign_template(&case, 5, campaign_profile_options());
    let f_target = campaign_f_target_hz(&case, &template).map_err(|error| error.to_string())?;
    assert!(
        (f_target - f_target_archive).abs() <= 1.0e-9,
        "campaign tuning drifted: {f_target} vs archived {f_target_archive}"
    );
    let winner_campaign = template
        .unpack(&winner_u)
        .map_err(|error| error.to_string())?;
    let straight_campaign = template.straight().map_err(|error| error.to_string())?;
    let out_dir = step8_dir();
    std::fs::create_dir_all(&out_dir)?;

    // ---- Noise leg: snapshot Tier-2 for straight + winner at three drives. ----
    let context =
        Tier2EvalContext::from_reference_case(case.clone()).map_err(|error| error.to_string())?;
    let drive_hz = case
        .tier2_drive_frequency_hz()
        .map_err(|error| error.to_string())?;
    let options = campaign_profile_options();
    let run_noise_wave = |jobs: &[(Vec<f64>, f64, Option<SmallSignalReference>)]| {
        let results: Result<Vec<(Vec<f64>, Tier2EvalOutcome)>, String> = jobs
            .par_iter()
            .map(|(u, voltage, reference)| {
                let profile = context
                    .candidate_profile(5, u, options)
                    .map_err(|error| error.to_string())?;
                let drive = PortEvalDrive {
                    voltage_peak_v: *voltage,
                    frequency_hz: drive_hz,
                    second_tone: None,
                };
                let mut fidelity =
                    noise_snapshot_fidelity(&case).map_err(|error| error.to_string())?;
                fidelity.compression_reference = *reference;
                let outcome = context
                    .evaluate(&profile, &drive, &fidelity)
                    .map_err(|error| format!("Tier-2 snapshot u={u:?} V={voltage}: {error}"))?;
                Ok((u.clone(), outcome))
            })
            .collect();
        results
    };
    let straight_u = vec![0.0, 0.0];
    let profiles = [straight_u.clone(), winner_u.clone()];
    let wave1_jobs: Vec<(Vec<f64>, f64, Option<SmallSignalReference>)> = profiles
        .iter()
        .map(|u| (u.clone(), STEP7_TIER2_DRIVES_V_PEAK[0], None))
        .collect();
    let wave1 = run_noise_wave(&wave1_jobs)?;
    let reference_of = |u: &[f64]| {
        let outcome = wave1.iter().find(|(w, _)| w == u).map(|(_, o)| o)?;
        Some(SmallSignalReference {
            voltage_peak_v: STEP7_TIER2_DRIVES_V_PEAK[0],
            port_flow_fundamental_amplitude_m3s: outcome
                .metrics
                .port_flow_fundamental_amplitude_m3s,
        })
    };
    let mut wave2_jobs = Vec::new();
    for u in &profiles {
        for voltage in &STEP7_TIER2_DRIVES_V_PEAK[1..] {
            wave2_jobs.push((u.clone(), *voltage, reference_of(u)));
        }
    }
    let wave2 = run_noise_wave(&wave2_jobs)?;
    let outcome_of = |u: &[f64], voltage: f64| {
        wave1
            .iter()
            .chain(wave2.iter())
            .find(|(w, o)| w == u && o.metrics.voltage_peak_v == voltage)
            .map(|(_, outcome)| outcome)
    };
    // Fresh-outcome 2D metrics must replay the Step-7 archive (snapshots on).
    let report7_text = std::fs::read_to_string(step7_report_path())?;
    let report7 = Step7Report::from_json_str(&report7_text).map_err(|error| error.to_string())?;
    for u in &profiles {
        let finalist = report7
            .finalists
            .iter()
            .find(|finalist| finalist.u == *u)
            .ok_or(format!("step-7 finalist {u:?} missing"))?;
        for archived in &finalist.tier2 {
            let outcome = outcome_of(u, archived.voltage_peak_v).ok_or(format!(
                "fresh outcome u={u:?} V={} missing",
                archived.voltage_peak_v
            ))?;
            let rel = (outcome.metrics.port_flow_rms_m3s - archived.port_flow_rms_m3s).abs()
                / archived.port_flow_rms_m3s.abs().max(1.0e-30);
            println!(
                "replay Q_rms u={u:?} @{} V: rel={rel:.3e}",
                archived.voltage_peak_v
            );
            assert!(
                rel <= 1.0e-6,
                "snapshot run diverged from Step-7 archive at u={u:?} V={}: {} vs {}",
                archived.voltage_peak_v,
                outcome.metrics.port_flow_rms_m3s,
                archived.port_flow_rms_m3s,
            );
            assert!(outcome.metrics.health.passed, "snapshot run tripped health");
        }
    }
    // Noise analysis per drive level.
    let observer = noise_observer_1m_on_axis(&case);
    let stride =
        noise_snapshot_stride(case.tier2_samples_per_cycle).map_err(|error| error.to_string())?;
    let mut noise_levels = Vec::new();
    for voltage in STEP7_TIER2_DRIVES_V_PEAK {
        let mut analyzed = Vec::new();
        for u in &profiles {
            let outcome = outcome_of(u, voltage).ok_or("fresh outcome missing")?;
            let metrics: &PortCandidateMetrics = &outcome.metrics;
            let window = tail_snapshot_window(
                &outcome.field_snapshots,
                metrics.tail_samples,
                stride,
                metrics.dt_s,
                metrics.steps_completed,
            )
            .map_err(|error| error.to_string())?;
            let snapshot_dt = window[1].time_s - window[0].time_s;
            let spec = finalist_noise_spec(
                drive_hz,
                snapshot_dt,
                observer,
                case.tier2_duct_length_m,
                1.0,
            )
            .map_err(|error| error.to_string())?;
            let result = analyze_tail_noise(&outcome.mesh, &window, case.medium(), &spec)
                .map_err(|error| format!("noise u={u:?} V={voltage}: {error}"))?;
            println!(
                "noise u={u:?} V={voltage}: n={} floor={:.3} dB methods={:?}",
                result.sample_count,
                result.observers[0].integrated_noise_floor_db,
                result.observers[0]
                    .spectrum
                    .iter()
                    .map(|point| (point.frequency_hz, point.method, point.compactness_ka))
                    .collect::<Vec<_>>(),
            );
            analyzed.push(result);
        }
        noise_levels.push(
            FinalistNoiseLevel::compare(voltage, &analyzed[0], &analyzed[1])
                .map_err(|error| error.to_string())?,
        );
    }

    // ---- 3D leg: nightly spot-checks at low + high drive (parallel). ----
    // `run_3d_spot_check` is a pure function of its inputs, so the four
    // (voltage, profile) jobs run concurrently; assembly below is positional
    // (straight-then-winner per voltage, ascending voltage), never a float
    // lookup on round-tripped `normalized()` values.
    assert!(
        !winner_u.iter().all(|v| *v == 0.0),
        "positional 3D assembly needs a non-straight winner"
    );
    let spot_drive_hz = spot_drive_frequency_hz(f_target).map_err(|error| error.to_string())?;
    let spot_fidelity = SpotCheck3dFidelity::smoke();
    let spot_jobs: Vec<(f64, bool)> = STEP8_SPOTCHECK_DRIVES_V_PEAK
        .iter()
        .flat_map(|voltage| [(*voltage, false), (*voltage, true)])
        .collect();
    let spot_results: Result<Vec<(f64, bool, SpotCheck3dRun)>, String> = spot_jobs
        .par_iter()
        .map(|(voltage, is_winner)| {
            let profile = if *is_winner {
                &winner_campaign
            } else {
                &straight_campaign
            };
            let run = run_3d_spot_check(&case, profile, *voltage, spot_drive_hz, &spot_fidelity)
                .map_err(|error| {
                    format!(
                        "3D spot {} V={voltage}: {error}",
                        if *is_winner { "winner" } else { "straight" }
                    )
                })?;
            Ok((*voltage, *is_winner, run))
        })
        .collect();
    let mut spot_results = spot_results?;
    spot_results.sort_by(|a, b| {
        a.0.partial_cmp(&b.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.1.cmp(&b.1))
    });
    let mut spot_runs: Vec<SpotCheck3dRun> = Vec::new();
    let mut spot_levels = Vec::new();
    for voltage in STEP8_SPOTCHECK_DRIVES_V_PEAK {
        let pair: Vec<&SpotCheck3dRun> = spot_results
            .iter()
            .filter(|(v, _, _)| *v == voltage)
            .map(|(_, _, run)| run)
            .collect();
        assert_eq!(pair.len(), 2, "3D leg must return straight + winner");
        // Sorted straight-first: `false < true`.
        let (r3s, r3w) = (pair[0], pair[1]);
        for (name, run) in [("straight", r3s), ("winner", r3w)] {
            println!(
                "3D[{}] {name} V={voltage}: Q={:.6e} m3/s THD={:.6e} cap={:.3e} \
                 nodes={} tets={} steps={}",
                run.tier,
                run.port_flow_rms_m3s,
                run.thd_fraction,
                run.cap_continuity_relative_error,
                run.mesh_nodes,
                run.mesh_tetrahedra,
                run.steps_completed,
            );
            spot_runs.push(run.clone());
        }
        let q2 = outcome_of(&winner_u, voltage).ok_or("fresh 2D winner missing")?;
        let q2s = outcome_of(&straight_u, voltage).ok_or("fresh 2D straight missing")?;
        let level = SpotCheck3dLevel::compare(
            voltage,
            spot_drive_hz,
            q2.metrics.port_flow_rms_m3s / q2s.metrics.port_flow_rms_m3s,
            r3w.port_flow_rms_m3s / r3s.port_flow_rms_m3s,
            q2.metrics.thd_fraction / q2s.metrics.thd_fraction,
            r3w.thd_fraction / r3s.thd_fraction,
        )
        .map_err(|error| error.to_string())?;
        println!(
            "spot V={voltage}: q_ratio 2D={:.6} 3D={:.6} dis={:.4} | thd_ratio 2D={:.6} 3D={:.6} \
             dis={:.4} -> {:?}",
            level.q_ratio_2d,
            level.q_ratio_3d,
            level.q_disagreement_rel,
            level.thd_ratio_2d,
            level.thd_ratio_3d,
            level.thd_disagreement_rel,
            level.verdict,
        );
        spot_levels.push(level);
    }

    let report = Step8Report {
        schema_version: sonium_sim_speaker_modeling::port_flow::STEP8_REPORT_SCHEMA_VERSION,
        provenance: sonium_sim_speaker_modeling::port_flow::STEP8_REPORT_PROVENANCE.to_string(),
        case: case.clone(),
        f_target_hz: f_target,
        n_control_points: 5,
        winner_u: winner_u.clone(),
        noise: noise_levels,
        spot3d_runs: spot_runs,
        spot3d: spot_levels,
        notes: format!(
            "Step-8 leg: snapshot Tier-2 (252 spc, 40/8 cycles, stride 21, 96 tail snapshots) \
             noise at {:?} V; 3D smoke spot-checks (0.02 m mesh, 32 spc, 2+2 cycles, f={spot_drive_hz:.6} Hz) \
             at {:?} V. Winner u={winner_u:?}.",
            STEP7_TIER2_DRIVES_V_PEAK, STEP8_SPOTCHECK_DRIVES_V_PEAK,
        ),
    };
    report.validate().map_err(|error| error.to_string())?;
    report
        .save_json(&step8_report_path())
        .map_err(|error| error.to_string())?;
    println!(
        "step-8 archive written to {}",
        step8_report_path().display()
    );
    Ok(())
}
