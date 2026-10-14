//! Certify external refinement of the original complete restrained-warping transient.
//!
//! Optional axial-cell and section-factor arguments run one additional level.
//! A single level records a full reference history without certifying convergence.

// Rust guideline compliant 2026-02-21

#[path = "../bin/fem_opencourant_beam_warping.rs"]
#[expect(
    dead_code,
    reason = "Only the original case factory and mesh refinement are invoked"
)]
mod reference;

use sha2::{Digest, Sha256};
use sonium_qa::rad::{assert_probe_close, engine_exe, run_case, write_engine, write_starter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = engine_exe().ok_or("OpenCourant engine required for the reference study")?;
    let output_path = std::env::var("P2_WARPING_REFERENCE_OUTPUT")?;
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let requested = match args.as_slice() {
        [] => vec![(40, 2), (60, 3), (80, 4)],
        [axial, section] => vec![(axial.parse::<usize>()?, section.parse::<usize>()?)],
        _ => return Err("provide both axial-cell count and section factor, or neither".into()),
    };
    let mut levels = Vec::new();
    let mut values = Vec::new();
    for (axial, section) in requested {
        let case = reference::refined_reference_case(axial, section)?;
        let work = std::env::temp_dir().join(format!(
            "p2_warping_reference_{}_{}_{}",
            std::process::id(),
            axial,
            section
        ));
        std::fs::create_dir_all(&work)?;
        let starter = write_starter(&case);
        let deck = write_engine(&case);
        std::fs::write(work.join("beam_warping_0000.rad"), &starter)?;
        std::fs::write(work.join("beam_warping_0001.rad"), &deck)?;
        println!(
            "external warping reference axial={axial}, section={section}: starting original full 0.5 s history"
        );
        let result = run_case(&work, &engine, &case)?;
        let probe_sample_time = result
            .history
            .times
            .last()
            .copied()
            .ok_or("external probe timestamp unavailable")?;
        let mut probes = Vec::new();
        for probe in &case.probes {
            let (plus, minus) = probe
                .channel
                .split_once('-')
                .ok_or("missing displacement pair")?;
            let value = result.history.final_value(plus)? - result.history.final_value(minus)?;
            assert_probe_close(
                value,
                probe.expected,
                probe.abs_tol,
                probe.rel_tol,
                &probe.label,
            )?;
            println!(
                "external axial={axial}, section={section}, {}={value:.12e}",
                probe.label
            );
            probes.push(value);
        }
        values.push(probes.clone());
        levels.push(
            serde_json::json!({"axial_cells":axial,"section_factor":section,
            "paired_displacements_m":probes,"engine":result.engine,"work_directory":work,
            "probe_sample_time_s":probe_sample_time,
            "physical_stop_time_s":case.stop_time_s,
            "starter_sha256":format!("{:x}",Sha256::digest(starter.as_bytes())),
            "engine_deck_sha256":format!("{:x}",Sha256::digest(deck.as_bytes()))}),
        );
        std::fs::write(
            &output_path,
            serde_json::to_vec_pretty(&serde_json::json!({
            "status":"incomplete","levels":levels,
            "scope":"Original material, full root restraint, exact tip force couple coordinates and 0.5 s ramp/hold history; original midspan and tip DZ pairs; external mesh refinement only"}))?,
        )?;
    }
    if levels.len() == 1 {
        std::fs::write(
            output_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "status":"complete single reference level; refinement and native parity unproven",
                "levels":levels,
                "physical_stop_time_s":0.5,
                "scope":"Original material, root restraint, tip force couple, full ramp/hold duration and original midspan/tip probes. Must be compared with independently completed neighboring mesh levels before convergence admission."
            }))?,
        )?;
        return Ok(());
    }
    let mut refinements = Vec::new();
    let mut passed = true;
    for probe in 0..2 {
        let first = (values[1][probe] - values[0][probe]).abs() / values[1][probe].abs();
        let last = (values[2][probe] - values[1][probe]).abs() / values[2][probe].abs();
        let contraction = last / first.max(1e-15);
        passed &= first <= 0.02 && last <= 0.005 && contraction <= 0.6;
        refinements.push(
            serde_json::json!({"probe":probe,"first_relative_change":first,
            "final_relative_change":last,"contraction":contraction}),
        );
    }
    std::fs::write(
        output_path,
        serde_json::to_vec_pretty(&serde_json::json!({
        "status":if passed {"passed"}else{"failed"},"levels":levels,"refinements":refinements,
        "budgets":{"first_relative_change":0.02,"final_relative_change":0.005,"contraction":0.6},
        "scope":"Full original warping physical transient; external refinement certificate, not native parity or original QA success"}))?,
    )?;
    if !passed {
        return Err("external reference mesh convergence failed".into());
    }
    Ok(())
}
