//! Certify external mesh convergence for the unchanged full curved-arch transient.

// Rust guideline compliant 2026-02-21

#[path = "../bin/fem_opencourant_beam_curved_arch.rs"]
#[expect(
    dead_code,
    reason = "The original case factory supplies the physical fixture; its native gate runner is not invoked"
)]
mod reference;

use sha2::{Digest, Sha256};
use sonium_qa::rad::{assert_probe_close, engine_exe, run_case, write_engine, write_starter};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine =
        engine_exe().ok_or("OpenCourant engine required for the reference convergence study")?;
    let output_path = std::env::var("P2_ARCH_REFERENCE_OUTPUT")?;
    let mut levels = Vec::new();
    let mut values = Vec::new();
    for (arc, section) in [(40, 4), (60, 6), (80, 8)] {
        let case = reference::refined_reference_case(arc, section)?;
        let work = std::env::temp_dir().join(format!(
            "p2_arch_reference_{}_{}_{}",
            std::process::id(),
            arc,
            section
        ));
        std::fs::create_dir_all(&work)?;
        let starter = write_starter(&case);
        let deck = write_engine(&case);
        std::fs::write(work.join("beam_curved_arch_0000.rad"), &starter)?;
        std::fs::write(work.join("beam_curved_arch_0001.rad"), &deck)?;
        println!(
            "external reference arc={arc}, section={section}: starting original full 1 s history"
        );
        let result = run_case(&work, &engine, &case)?;
        let probe = &case.probes[0];
        let value = result.history.final_value(&probe.channel)?;
        assert_probe_close(
            value,
            probe.expected,
            probe.abs_tol,
            probe.rel_tol,
            &probe.label,
        )?;
        println!("external reference arc={arc}, section={section}: final tip DX={value:.12e}");
        values.push(value);
        levels.push(serde_json::json!({"arc_cells":arc,"section_cells":section,"tip_displacement_m":value,
            "engine":result.engine,"work_directory":work,"starter_sha256":format!("{:x}",Sha256::digest(starter.as_bytes())),
            "engine_deck_sha256":format!("{:x}",Sha256::digest(deck.as_bytes()))}));
        std::fs::write(
            &output_path,
            serde_json::to_vec_pretty(&serde_json::json!({
                "status":"incomplete","scope":"Original curved solid, nine original nodal force locations/scales, original full root clamp/material/load history and tip-center probe; external mesh refinement only",
                "levels":levels
            }))?,
        )?;
    }
    let first = (values[1] - values[0]).abs() / values[1].abs();
    let last = (values[2] - values[1]).abs() / values[2].abs();
    let contraction = last / first.max(1e-15);
    let passed = first <= 0.02 && last <= 0.005 && contraction <= 0.6;
    let certificate = serde_json::json!({"status":if passed{"passed"}else{"failed"},"levels":levels,
        "first_relative_change":first,"final_relative_change":last,"contraction":contraction,
        "budgets":{"first_relative_change":0.02,"final_relative_change":0.005,"contraction":0.6},
        "scope":"Original full curved-arch physical transient; mesh-refined OpenCourant reference, not native parity or original QA success"});
    std::fs::write(output_path, serde_json::to_vec_pretty(&certificate)?)?;
    println!(
        "reference convergence: first={first:.6e}, final={last:.6e}, contraction={contraction:.6e}"
    );
    if !passed {
        return Err("external reference mesh convergence failed".into());
    }
    Ok(())
}
