//! PORT-OPTIM Step 9: convergence/promotion evidence for the winner.
//!
//! Default (fast) tests guard the Step-9 machinery without solving: the
//! production ladder plan is schedule-consistent, the regenerated ladder
//! meshes are distinct and healthy, promotion cases match their refinement,
//! and the committed evidence archive (when present) parses and validates.
//!
//! The ignored `step9_coarse_probe` runs one cheap coarse winner solve to
//! smoke-test the new code paths and preview coarse-vs-production drift.
//! The ignored `step9_winner_promotion` test runs the real leg (spatial +
//! temporal ladders at `V_max`, low-drive final-mesh runs, Tier-1 oracle)
//! and writes `fixtures/port_optim_step9/step9_report.json`. Release-only:
//!
//! ```sh
//! cargo test -p sonium-sim-speaker-modeling --release \
//!   --test port_optim_step9 -- --ignored --nocapture
//! ```

use std::path::PathBuf;

use sonium_sim_speaker_modeling::port_flow::{
    MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL, PortOptimReferenceCase,
    STEP9_MAX_FINEST_PAIR_RELATIVE_CHANGE, STEP9_REPORT_PROVENANCE, STEP9_SPATIAL_MESH_SIZES_M,
    STEP9_TEMPORAL_SAMPLES_PER_CYCLE, Step7Report, Step9Report, Tier2EvalContext,
    build_promotion_case, campaign_profile_options, campaign_template, min_profile_area_m2,
    step9_promotion_gate,
};

fn step7_report_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/port_optim_step7/step7_report.json")
}

fn step9_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/port_optim_step9")
}

fn step9_report_path() -> PathBuf {
    step9_dir().join("step9_report.json")
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
fn step9_production_ladder_plan_is_consistent() {
    use sonium_sim_speaker_modeling::port_flow::PromotionLadderConfig;
    let case = PortOptimReferenceCase::default();
    let ladder = PromotionLadderConfig::production(&case).expect("production ladder builds");
    assert_eq!(
        ladder.spatial_mesh_sizes_m,
        STEP9_SPATIAL_MESH_SIZES_M.to_vec()
    );
    assert_eq!(
        ladder.temporal_samples_per_cycle,
        STEP9_TEMPORAL_SAMPLES_PER_CYCLE.to_vec()
    );
    let frequency_hz = case
        .tier2_drive_frequency_hz()
        .expect("Tier-2 drive must resolve");
    let plan = ladder
        .to_runtime_plan(&case, frequency_hz)
        .expect("runtime plan must build");
    // Fine mesh = production mesh; fine dt brackets production from above
    // (production 252 spc is the coarse temporal level).
    assert_eq!(ladder.fine_mesh_size_m(), case.tier2_mesh_size_m);
    assert_eq!(
        ladder.temporal_samples_per_cycle[0],
        case.tier2_samples_per_cycle
    );
    assert_eq!(ladder.fine_samples_per_cycle(), 504);
    let gate = step9_promotion_gate();
    assert_eq!(gate.min_spatial_levels, 3);
    assert_eq!(gate.min_temporal_levels, 2);
    assert_eq!(
        gate.max_spatial_observable_relative_change,
        STEP9_MAX_FINEST_PAIR_RELATIVE_CHANGE
    );
    println!(
        "step-9 ladder: f={frequency_hz:.6} Hz, meshes={:?} m, spc={:?}, cycles={}/{}, V_ladder={} V",
        plan.spatial_mesh_sizes_m,
        ladder.temporal_samples_per_cycle,
        ladder.total_cycles,
        ladder.tail_cycles,
        ladder.voltage_peak_v,
    );
}

#[test]
fn regenerated_ladder_meshes_are_distinct_and_healthy() {
    let Some((case, _, winner_u)) = load_step7_winner() else {
        println!("step-7 archive absent (run --ignored campaign first); nothing to mesh");
        return;
    };
    let context =
        Tier2EvalContext::from_reference_case(case.clone()).expect("Tier-2 context builds");
    let options = campaign_profile_options();
    let winner = context
        .candidate_profile(5, &winner_u, options)
        .expect("winner must build at slot scale");
    let straight = context
        .candidate_profile(5, &[0.0, 0.0], options)
        .expect("straight must build at slot scale");
    let throat = min_profile_area_m2(&winner).expect("winner throat area resolves");
    assert!(throat > 0.0);
    let mut node_counts = Vec::new();
    for mesh_size_m in STEP9_SPATIAL_MESH_SIZES_M {
        let leveled = context
            .clone()
            .with_mesh_size_m(mesh_size_m)
            .expect("mesh-size override builds");
        for (name, profile) in [("winner", &winner), ("straight", &straight)] {
            let mesh = leveled
                .candidate_geometry(profile)
                .build_validation_mesh()
                .expect("validation mesh must regenerate");
            assert_eq!(mesh.dimension, 2);
            assert!(!mesh.elements.is_empty());
            for marker in [MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL] {
                assert!(
                    mesh.boundaries.iter().any(|facet| facet.marker == marker),
                    "{name} mesh at {mesh_size_m} m lost marker {marker}"
                );
            }
            let min_area = sonium_sim_speaker_modeling::port_flow::min_triangle_area_m2(&mesh);
            assert!(
                min_area > 0.0,
                "{name} mesh at {mesh_size_m} m failed positivity: {min_area:.3e}"
            );
            if name == "winner" {
                node_counts.push(mesh.num_nodes());
            }
            println!(
                "step-9 mesh {name} h={mesh_size_m} m: nodes={}, elements={}, min_area={min_area:.3e} m2",
                mesh.num_nodes(),
                mesh.elements.len(),
            );
        }
    }
    // Strictly refining schedule => strictly growing meshes.
    assert!(
        node_counts[0] < node_counts[1] && node_counts[1] < node_counts[2],
        "ladder meshes do not strictly refine: {node_counts:?}"
    );
}

#[test]
fn promotion_case_matches_its_refinement() {
    use sonium_sim_speaker_modeling::port_flow::{PortEvalFidelity, PromotionLadderConfig};
    let Some((case, _, winner_u)) = load_step7_winner() else {
        println!("step-7 archive absent (run --ignored campaign first); nothing to case");
        return;
    };
    let context =
        Tier2EvalContext::from_reference_case(case.clone()).expect("Tier-2 context builds");
    let winner = context
        .candidate_profile(5, &winner_u, campaign_profile_options())
        .expect("winner must build at slot scale");
    let ladder = PromotionLadderConfig::production(&case).expect("ladder builds");
    let frequency_hz = case
        .tier2_drive_frequency_hz()
        .expect("Tier-2 drive must resolve");
    for (mesh_size_m, spc) in [
        (
            ladder.spatial_mesh_sizes_m[0],
            ladder.temporal_samples_per_cycle[0],
        ),
        (ladder.fine_mesh_size_m(), ladder.fine_samples_per_cycle()),
    ] {
        let fidelity = PortEvalFidelity {
            samples_per_cycle: spc,
            total_cycles: ladder.total_cycles,
            tail_cycles: ladder.tail_cycles,
            snapshot_interval_steps: 0,
            compression_reference: None,
        };
        let (runtime_case, stats) = build_promotion_case(
            &context,
            &winner,
            ladder.voltage_peak_v,
            frequency_hz,
            mesh_size_m,
            &fidelity,
        )
        .expect("promotion case must build");
        assert_eq!(runtime_case.mesh.dimension, 2);
        let expected_dt = 1.0 / (frequency_hz * spc as f64);
        assert!(
            (runtime_case.config.bdf.dt - expected_dt).abs() <= 1.0e-15,
            "case dt {} mismatches spc {spc}",
            runtime_case.config.bdf.dt
        );
        assert_eq!(runtime_case.config.bdf.num_steps, ladder.total_cycles * spc);
        assert_eq!(stats.mesh_size_m, mesh_size_m);
        assert!(stats.min_triangle_area_m2 > 0.0);
        println!(
            "step-9 case h={mesh_size_m} m spc={spc}: dt={:.6e} s steps={} nodes={}",
            runtime_case.config.bdf.dt, runtime_case.config.bdf.num_steps, stats.mesh_nodes,
        );
    }
}

#[test]
fn step9_template_matches_campaign_scale() {
    let case = PortOptimReferenceCase::default();
    let template = campaign_template(&case, 5, campaign_profile_options());
    let straight = template.straight().expect("straight must build");
    assert!((straight.length_m() - case.port_length_l_m).abs() <= 1.0e-12);
    assert!((straight.bore_radius_m() - case.port_radius_r0_m).abs() <= 1.0e-12);
}

#[test]
fn step9_archive_validates_when_present() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = step9_report_path();
    if !path.exists() {
        println!("step-9 archive absent (run --ignored leg); nothing to validate");
        return Ok(());
    }
    let text = std::fs::read_to_string(&path)?;
    let report = Step9Report::from_json_str(&text).map_err(|error| match error {
        sonium_sim_speaker_modeling::port_flow::CampaignArchiverError::Io(message)
        | sonium_sim_speaker_modeling::port_flow::CampaignArchiverError::Json(message) => message,
    })?;
    assert_eq!(report.provenance, STEP9_REPORT_PROVENANCE);
    report.validate().map_err(|error| match error {
        sonium_sim_speaker_modeling::port_flow::CampaignArchiverError::Io(message)
        | sonium_sim_speaker_modeling::port_flow::CampaignArchiverError::Json(message) => message,
    })?;
    println!(
        "step-9 verdict: {:?} (winner u={:?}, f_target={:.6} Hz)",
        report.verdict, report.winner_u, report.f_target_hz
    );
    println!(
        "  phasor: spatial={:.4}% temporal={:.4}% cross={:.4}% (bar 2%)",
        100.0 * report.promotion.max_spatial_observable_relative_change,
        100.0 * report.promotion.max_temporal_observable_relative_change,
        100.0 * report.promotion.cross_ladder_observable_relative_change,
    );
    for (name, change) in [
        ("spatial THD", &report.extended.spatial_thd),
        ("spatial proxy", &report.extended.spatial_noise_proxy),
        ("temporal THD", &report.extended.temporal_thd),
        ("temporal proxy", &report.extended.temporal_noise_proxy),
    ] {
        println!(
            "  {name}: coarse={:.6e} fine={:.6e} change={:.4}% (bar 2%){} -> {}",
            change.coarse,
            change.fine,
            100.0 * change.relative_change,
            if change.passed_by_floor {
                " [floor]"
            } else {
                ""
            },
            if change.passed { "pass" } else { "FAIL" },
        );
    }
    println!(
        "  linear: impedance-ratio err={:.4}% (bar 3%) THD={:.3e} Mach={:.3e} -> {}; \
         H-ratio err={:.4}% reported only (Tier-1 cond {:.1}/{:.1})",
        100.0 * report.linear_gate.impedance_ratio_relative_error,
        report.linear_gate.winner_thd,
        report.linear_gate.winner_mach,
        if report.linear_gate.passed {
            "pass"
        } else {
            "FAIL"
        },
        100.0 * report.linear_gate.flow_ratio_relative_error,
        report.tier1_winner.flow_ratio_condition_number,
        report.tier1_straight.flow_ratio_condition_number,
    );
    println!(
        "  mesh independence: straight={:.4}% winner={:.4}% (report-only)",
        100.0 * report.mesh_independence.straight_relative_difference,
        100.0 * report.mesh_independence.winner_relative_difference,
    );
    Ok(())
}

// ---------------------------------------------------------------------------
// Ignored release-only legs
// ---------------------------------------------------------------------------

type Step9Error = Box<dyn std::error::Error + Send + Sync>;

fn release_only(leg: &str) -> Result<(), Step9Error> {
    if cfg!(debug_assertions) {
        return Err(format!(
            "run the Step-9 {leg} in release mode (cargo test -p sonium-sim-speaker-modeling \
             --release --test port_optim_step9 -- --ignored --nocapture)"
        )
        .into());
    }
    Ok(())
}

/// Cheap smoke path: one coarse winner solve at `V_max`.
///
/// Exercises the regenerated-mesh case builder, the coupled solve, and the
/// trace reduction end to end, and previews coarse-vs-production drift
/// against the Step-7 archive before the full ladder runs. The probed
/// resolution defaults to the ladder's coarse corner and can be overridden
/// with `STEP9_PROBE_MESH_M` / `STEP9_PROBE_SPC` (probe-only; the leg's
/// evidence schedule is fixed in code).
#[test]
#[ignore = "Step-9 coarse probe; run explicitly in release mode"]
fn step9_coarse_probe() -> Result<(), Step9Error> {
    release_only("coarse probe")?;
    use sonium_sim_speaker_modeling::port_flow::coupled::run_coupled_port;
    use sonium_sim_speaker_modeling::port_flow::{
        PortEvalFidelity, PromotionLadderConfig, PromotionReductionInputs,
        STEP7_TIER2_DRIVES_V_PEAK, reduce_promotion_trace,
    };

    let Some((case, _, winner_u)) = load_step7_winner() else {
        return Err("step-7 archive missing; run the Step-7 campaign first".into());
    };
    case.validate().map_err(|error| error.to_string())?;
    let context =
        Tier2EvalContext::from_reference_case(case.clone()).map_err(|error| error.to_string())?;
    let drive_hz = case
        .tier2_drive_frequency_hz()
        .map_err(|error| error.to_string())?;
    let winner = context
        .candidate_profile(5, &winner_u, campaign_profile_options())
        .map_err(|error| error.to_string())?;
    let ladder = PromotionLadderConfig::production(&case).map_err(|error| error.to_string())?;
    let mesh_size_m: f64 = std::env::var("STEP9_PROBE_MESH_M")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(ladder.spatial_mesh_sizes_m[0]);
    let spc: usize = std::env::var("STEP9_PROBE_SPC")
        .ok()
        .and_then(|raw| raw.parse().ok())
        .unwrap_or(ladder.temporal_samples_per_cycle[0]);
    let voltage = STEP7_TIER2_DRIVES_V_PEAK[2];
    let fidelity = PortEvalFidelity {
        samples_per_cycle: spc,
        total_cycles: ladder.total_cycles,
        tail_cycles: ladder.tail_cycles,
        snapshot_interval_steps: 0,
        compression_reference: None,
    };
    let started = std::time::Instant::now();
    let (runtime_case, stats) =
        build_promotion_case(&context, &winner, voltage, drive_hz, mesh_size_m, &fidelity)
            .map_err(|error| error.to_string())?;
    let result = run_coupled_port(
        &runtime_case.mesh,
        &runtime_case.velocity_space,
        &runtime_case.pressure_space,
        &runtime_case.config,
    )
    .map_err(|error| error.to_string())?;
    let refinement = sonium_sim_speaker_modeling::port_flow::PortFlowRuntimeRefinement {
        mesh_size_m,
        dt_s: runtime_case.config.bdf.dt,
        frequency_hz: drive_hz,
        samples_per_cycle: spc,
        tail_cycles: ladder.tail_cycles,
        discarded_cycles: ladder.discarded_cycles(),
    };
    let throat_area_m2 = min_profile_area_m2(&winner).map_err(|error| error.to_string())?;
    let diagnostics = reduce_promotion_trace(
        refinement,
        &result,
        PromotionReductionInputs {
            tail_cycles: ladder.tail_cycles,
            discarded_cycles: ladder.discarded_cycles(),
            box_volume_m3: case.box_volume_m3,
            density_kg_per_m3: case.density_kg_per_m3,
            sound_speed_m_per_s: case.sound_speed_m_per_s,
            throat_area_m2,
            duct_length_m: context.duct_length_m(),
        },
        stats,
    )
    .map_err(|error| error.to_string())?;
    let report7_text = std::fs::read_to_string(step7_report_path())?;
    let report7 = Step7Report::from_json_str(&report7_text).map_err(|error| error.to_string())?;
    let archived = report7
        .finalists
        .iter()
        .find(|finalist| finalist.u == winner_u)
        .and_then(|finalist| {
            finalist
                .tier2
                .iter()
                .find(|level| level.voltage_peak_v == voltage)
        })
        .ok_or("Step-7 winner V_max level missing")?;
    let drift = (diagnostics.port_flow_rms_m3s - archived.port_flow_rms_m3s).abs()
        / archived.port_flow_rms_m3s;
    println!(
        "step-9 probe h={mesh_size_m} m spc={spc} V={voltage} V ({} steps, {:.1}s):\n  \
         Q_rms={:.6e} m3/s (Step-7 morphed {:.6e}, drift={:.4}%)\n  \
         THD={:.6e} proxy={:.6e} Mach={:.6e}\n  \
         cap={:.3e} energy_phys={:.3e} lin_res={:.3e} box_cons={:.3e}\n  \
         identity={:.3e} nodes={} elements={}",
        diagnostics.steps_completed,
        started.elapsed().as_secs_f64(),
        diagnostics.port_flow_rms_m3s,
        archived.port_flow_rms_m3s,
        100.0 * drift,
        diagnostics.thd_fraction,
        diagnostics.noise_proxy,
        diagnostics.peak_mach_bulk,
        diagnostics.cap_continuity_relative_error,
        diagnostics.energy_physical_relative_closure,
        diagnostics.max_linear_relative_residual,
        diagnostics.max_box_conservation_relative_residual,
        diagnostics.transfer_identity_relative_error,
        diagnostics.mesh_nodes,
        diagnostics.mesh_elements,
    );
    Ok(())
}

/// Full Step-9 leg: winner ladders at `V_max`, low-drive final-mesh runs,
/// Tier-1 oracle, and the promotion verdict. Writes
/// `fixtures/port_optim_step9/step9_report.json` with either a pass or a
/// demotion verdict; only infrastructure failures (divergence, invalid
/// config) fail the test.
#[test]
#[ignore = "Step-9 winner promotion; run explicitly in release mode"]
fn step9_winner_promotion() -> Result<(), Step9Error> {
    release_only("winner promotion")?;
    use rayon::prelude::*;
    use sonium_sim_speaker_modeling::port_flow::coupled::run_coupled_port;
    use sonium_sim_speaker_modeling::port_flow::{
        MeshIndependenceCheck, PortEvalFidelity, PortFlowRuntimeRefinement, PromotionGateMirror,
        PromotionLadderConfig, PromotionLevelDiagnostics, PromotionReductionInputs,
        PromotionReportMirror, STEP7_TIER2_DRIVES_V_PEAK, STEP9_REPORT_SCHEMA_VERSION,
        STEP9_TIER1_MESH_SIZE_M, Step9LinearGateInputs, Step9Report, Tier1TransferPoint,
        campaign_f_target_hz, check_extended_changes, check_linear_transfer_gate,
        evaluate_step9_verdict, run_step9_ladder, solve_step9_tier1, step9_promotion_gate,
        step9_structural_gate, validate_port_flow_promotion,
    };

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
    let context =
        Tier2EvalContext::from_reference_case(case.clone()).map_err(|error| error.to_string())?;
    let drive_hz = case
        .tier2_drive_frequency_hz()
        .map_err(|error| error.to_string())?;
    let options = campaign_profile_options();
    let winner_slot = context
        .candidate_profile(5, &winner_u, options)
        .map_err(|error| error.to_string())?;
    let winner_campaign = template
        .unpack(&winner_u)
        .map_err(|error| error.to_string())?;
    let straight_campaign = template.straight().map_err(|error| error.to_string())?;
    let ladder = PromotionLadderConfig::production(&case).map_err(|error| error.to_string())?;
    let out_dir = step9_dir();
    std::fs::create_dir_all(&out_dir)?;

    // ---- Ladders at V_max (serial through the shared runner). ----
    let ladder_started = std::time::Instant::now();
    let outcome = run_step9_ladder(&context, &winner_slot, &ladder, drive_hz)
        .map_err(|error| error.to_string())?;
    println!(
        "step-9 ladders done in {:.1}s ({} spatial + {} temporal levels)",
        ladder_started.elapsed().as_secs_f64(),
        outcome.spatial_levels.len(),
        outcome.temporal_levels.len(),
    );
    for level in outcome
        .spatial_levels
        .iter()
        .chain(outcome.temporal_levels.iter())
    {
        println!(
            "  level h={} m spc={}: Q_rms={:.6e} THD={:.3e} proxy={:.3e} \
             cap={:.2e} e_phys={:.2e} nodes={}",
            level.mesh_size_m,
            level.samples_per_cycle,
            level.port_flow_rms_m3s,
            level.thd_fraction,
            level.noise_proxy,
            level.cap_continuity_relative_error,
            level.energy_physical_relative_closure,
            level.mesh_nodes,
        );
    }
    let gate = step9_promotion_gate();
    let strict = validate_port_flow_promotion(&outcome.evidence, gate);
    match &strict {
        Ok(report) => println!(
            "step-9 strict phasor gate: PASS (spatial={:.4}% temporal={:.4}% cross={:.4}%)",
            100.0 * report.max_spatial_observable_relative_change,
            100.0 * report.max_temporal_observable_relative_change,
            100.0 * report.cross_ladder_observable_relative_change,
        ),
        Err(error) => println!("step-9 strict phasor gate: FAIL ({error})"),
    }
    // Evidence-preserving summary: the strict report when it passes, else
    // the same evidence under the structural gate (measured changes kept).
    let promotion_report = match strict {
        Ok(report) => report,
        Err(_) => validate_port_flow_promotion(&outcome.evidence, step9_structural_gate())
            .map_err(|error| error.to_string())?,
    };
    let promotion = PromotionReportMirror::from(&promotion_report);
    let extended = check_extended_changes(
        &outcome.spatial_levels,
        &outcome.temporal_levels,
        gate.max_spatial_observable_relative_change,
    )
    .map_err(|error| error.to_string())?;
    for (name, change) in [
        ("spatial THD", &extended.spatial_thd),
        ("spatial proxy", &extended.spatial_noise_proxy),
        ("temporal THD", &extended.temporal_thd),
        ("temporal proxy", &extended.temporal_noise_proxy),
    ] {
        println!(
            "step-9 {name}: coarse={:.6e} fine={:.6e} change={:.4}%{} -> {}",
            change.coarse,
            change.fine,
            100.0 * change.relative_change,
            if change.passed_by_floor {
                " [floor]"
            } else {
                ""
            },
            if change.passed { "pass" } else { "FAIL" },
        );
    }

    // ---- Low-drive final-mesh runs + Tier-1 oracle (parallel). ----
    // The linear gate validates the production numbers' low-drive limit, so
    // it runs at the production timestep (252 spc) on the final mesh — not
    // at the ladder's bracketing fine dt (504 spc).
    let low_drive_v = STEP7_TIER2_DRIVES_V_PEAK[0];
    let production_spc = case.tier2_samples_per_cycle;
    let fine_fidelity = PortEvalFidelity {
        samples_per_cycle: production_spc,
        total_cycles: ladder.total_cycles,
        tail_cycles: ladder.tail_cycles,
        snapshot_interval_steps: 0,
        compression_reference: None,
    };
    let fine_mesh = ladder.fine_mesh_size_m();
    let run_low_drive = |(name, u): (&str, &[f64])| -> Result<PromotionLevelDiagnostics, String> {
        let profile = context
            .candidate_profile(5, u, options)
            .map_err(|error| error.to_string())?;
        let (runtime_case, stats) = build_promotion_case(
            &context,
            &profile,
            low_drive_v,
            drive_hz,
            fine_mesh,
            &fine_fidelity,
        )
        .map_err(|error| error.to_string())?;
        let result = run_coupled_port(
            &runtime_case.mesh,
            &runtime_case.velocity_space,
            &runtime_case.pressure_space,
            &runtime_case.config,
        )
        .map_err(|error| format!("low-drive {name}: {error}"))?;
        let throat = min_profile_area_m2(&profile).map_err(|error| error.to_string())?;
        let diagnostics = sonium_sim_speaker_modeling::port_flow::reduce_promotion_trace(
            PortFlowRuntimeRefinement {
                mesh_size_m: fine_mesh,
                dt_s: runtime_case.config.bdf.dt,
                frequency_hz: drive_hz,
                samples_per_cycle: production_spc,
                tail_cycles: ladder.tail_cycles,
                discarded_cycles: ladder.discarded_cycles(),
            },
            &result,
            PromotionReductionInputs {
                tail_cycles: ladder.tail_cycles,
                discarded_cycles: ladder.discarded_cycles(),
                box_volume_m3: case.box_volume_m3,
                density_kg_per_m3: case.density_kg_per_m3,
                sound_speed_m_per_s: case.sound_speed_m_per_s,
                throat_area_m2: throat,
                duct_length_m: context.duct_length_m(),
            },
            stats,
        )
        .map_err(|error| format!("low-drive {name} reduction: {error}"))?;
        println!(
            "step-9 low-drive {name} V={low_drive_v} V: Q_rms={:.6e} THD={:.3e} \
             Mach={:.3e} cap={:.2e}",
            diagnostics.port_flow_rms_m3s,
            diagnostics.thd_fraction,
            diagnostics.peak_mach_bulk,
            diagnostics.cap_continuity_relative_error,
        );
        Ok(diagnostics)
    };
    let jobs: Vec<(&str, Vec<f64>)> =
        vec![("winner", winner_u.clone()), ("straight", vec![0.0, 0.0])];
    let low_drive: Result<Vec<PromotionLevelDiagnostics>, String> = jobs
        .par_iter()
        .map(|(n, u)| run_low_drive((*n, u.as_slice())))
        .collect();
    let low_drive = low_drive?;
    let (low_drive_winner, low_drive_straight) = (&low_drive[0], &low_drive[1]);

    let tier1_jobs: Vec<(&str, bool)> = vec![("winner", true), ("straight", false)];
    let tier1: Result<Vec<Tier1TransferPoint>, String> = tier1_jobs
        .par_iter()
        .map(|(name, is_winner)| {
            let profile = if *is_winner {
                &winner_campaign
            } else {
                &straight_campaign
            };
            let point = solve_step9_tier1(&case, profile, f_target, STEP9_TIER1_MESH_SIZE_M)
                .map_err(|error| format!("Tier-1 {name}: {error}"))?;
            println!(
                "step-9 Tier-1 {name}: Z=({:.6e},{:.6e}) H=({:.6e},{:.6e}) cond={:.2} \
                 residual={:.2e} flux={:.2e} dofs={}",
                point.port_impedance[0],
                point.port_impedance[1],
                point.flow_ratio[0],
                point.flow_ratio[1],
                point.flow_ratio_condition_number,
                point.relative_residual,
                point.flux_relative_difference,
                point.scalar_dofs,
            );
            Ok(point)
        })
        .collect();
    let tier1 = tier1?;
    let (tier1_winner, tier1_straight) = (&tier1[0], &tier1[1]);
    let linear_gate = check_linear_transfer_gate(&Step9LinearGateInputs {
        tier2_winner: low_drive_winner,
        tier2_straight: low_drive_straight,
        tier1_winner,
        tier1_straight,
    })
    .map_err(|error| error.to_string())?;
    println!(
        "step-9 linear gate: impedance-ratio err={:.4}% (bar 3%) THD={:.3e} Mach={:.3e} -> {}; \
         H-ratio err={:.4}% reported only (Tier-1 cond {:.1}/{:.1})",
        100.0 * linear_gate.impedance_ratio_relative_error,
        linear_gate.winner_thd,
        linear_gate.winner_mach,
        if linear_gate.passed { "pass" } else { "FAIL" },
        100.0 * linear_gate.flow_ratio_relative_error,
        tier1_winner.flow_ratio_condition_number,
        tier1_straight.flow_ratio_condition_number,
    );
    if !linear_gate.passed {
        for failure in &linear_gate.failures {
            println!("  linear failure: {failure}");
        }
    }

    // ---- Mesh-independence cross-checks (report-only). ----
    let baseline_text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/port_optim_baseline_v1.json"),
    )?;
    let baseline: sonium_sim_speaker_modeling::port_flow::PortOptimBaseline =
        serde_json::from_str(&baseline_text)?;
    let step0_level = baseline
        .tier2
        .iter()
        .find(|level| level.voltage_peak_v == low_drive_v)
        .ok_or("Step-0 low-drive level missing")?;
    let report7_text = std::fs::read_to_string(step7_report_path())?;
    let report7 = Step7Report::from_json_str(&report7_text).map_err(|error| error.to_string())?;
    let step7_level = report7
        .finalists
        .iter()
        .find(|finalist| finalist.u == winner_u)
        .and_then(|finalist| {
            finalist
                .tier2
                .iter()
                .find(|level| level.voltage_peak_v == ladder.voltage_peak_v)
        })
        .ok_or("Step-7 winner V_max level missing")?;
    let winner_fine_q = outcome
        .spatial_levels
        .last()
        .map(|level| level.port_flow_rms_m3s)
        .ok_or("ladder has no fine level")?;
    let mesh_independence = MeshIndependenceCheck::compare(
        low_drive_straight.port_flow_rms_m3s,
        step0_level.port_flow_rms_m3s,
        winner_fine_q,
        step7_level.port_flow_rms_m3s,
    )
    .map_err(|error| error.to_string())?;
    println!(
        "step-9 mesh independence: straight={:.4}% winner={:.4}% (report-only)",
        100.0 * mesh_independence.straight_relative_difference,
        100.0 * mesh_independence.winner_relative_difference,
    );

    // ---- Verdict + archive. ----
    let verdict = evaluate_step9_verdict(
        &promotion,
        &PromotionGateMirror::from(&gate),
        &extended,
        &linear_gate,
        &outcome.spatial_levels,
        &outcome.temporal_levels,
    );
    println!("step-9 verdict: {verdict:?}");
    let thd_path = |change: &sonium_sim_speaker_modeling::port_flow::FinestPairChange| {
        if change.passed_by_floor {
            "floor"
        } else if change.passed {
            "bar"
        } else {
            "FAIL"
        }
    };
    let notes = format!(
        "Step-9 leg: regenerated-validation-mesh ladders (spatial {:?} m, temporal {:?} spc, \
         40/8 cycles) at V_max={} V; low-drive final-mesh runs at {} V; Tier-1 spline oracle \
         (0.01 m reference mesh) at f_target={f_target:.6} Hz. Winner u={winner_u:?}. \
         Measured: temporal proxy {:.4}% vs 2% bar (U^8 of {:.4}% Q change); THD \
         spatial {:.2e}/{:.2e} ({}) temporal {:.2e}/{:.2e} ({}); H-ratio {:.2}% reported \
         only (Tier-1 Helmholtz-pole condition {:.1}, see STEP9_MAX_FLOW_RATIO doc).",
        ladder.spatial_mesh_sizes_m,
        ladder.temporal_samples_per_cycle,
        ladder.voltage_peak_v,
        low_drive_v,
        100.0 * extended.temporal_noise_proxy.relative_change,
        100.0
            * (outcome.temporal_levels[0].port_flow_rms_m3s
                - outcome.temporal_levels[1].port_flow_rms_m3s)
                .abs()
            / outcome.temporal_levels[1].port_flow_rms_m3s,
        extended.spatial_thd.coarse,
        extended.spatial_thd.fine,
        thd_path(&extended.spatial_thd),
        extended.temporal_thd.coarse,
        extended.temporal_thd.fine,
        thd_path(&extended.temporal_thd),
        100.0 * linear_gate.flow_ratio_relative_error,
        tier1_winner.flow_ratio_condition_number,
    );
    let report = Step9Report {
        schema_version: STEP9_REPORT_SCHEMA_VERSION,
        provenance: sonium_sim_speaker_modeling::port_flow::STEP9_REPORT_PROVENANCE.to_string(),
        case: case.clone(),
        f_target_hz: f_target,
        n_control_points: 5,
        winner_u: winner_u.clone(),
        ladder_voltage_peak_v: ladder.voltage_peak_v,
        low_drive_voltage_peak_v: low_drive_v,
        ladder_config: ladder.clone(),
        promotion_evidence: outcome.evidence,
        promotion_gate: PromotionGateMirror::from(&gate),
        promotion,
        spatial_levels: outcome.spatial_levels,
        temporal_levels: outcome.temporal_levels,
        extended,
        linear_gate,
        low_drive_winner: low_drive_winner.clone(),
        low_drive_straight: low_drive_straight.clone(),
        tier1_winner: tier1_winner.clone(),
        tier1_straight: tier1_straight.clone(),
        mesh_independence,
        verdict,
        notes,
    };
    report.validate().map_err(|error| error.to_string())?;
    report
        .save_json(&step9_report_path())
        .map_err(|error| error.to_string())?;
    println!(
        "step-9 archive written to {}",
        step9_report_path().display()
    );
    Ok(())
}
