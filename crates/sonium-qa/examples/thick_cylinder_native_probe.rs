//! Measure the complete native plastic-cylinder history using the existing engine.

// Rust guideline compliant 2026-02-21

#[path = "../bin/oc_bench.rs"]
#[expect(
    dead_code,
    reason = "Only the original full native constitutive and integration path is invoked"
)]
mod native;

#[path = "../bin/fem_opencourant_plasticity_thick_cylinder.rs"]
#[expect(
    dead_code,
    reason = "Only the original full external fixture factory is invoked"
)]
mod reference;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(path) = std::env::var("P2_THICK_CYLINDER_REFERENCE_OUTPUT") {
        use sha2::{Digest, Sha256};
        let case = reference::plasticity_thick_cylinder_case();
        let starter = sonium_qa::rad::write_extended_starter(&case);
        let engine_deck = sonium_qa::rad::write_extended_engine(&case);
        let decks = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("decks/plasticity_thick_cylinder");
        if std::fs::read_to_string(decks.join("plasticity_thick_cylinder_0000.rad"))? != starter
            || std::fs::read_to_string(decks.join("plasticity_thick_cylinder_0001.rad"))?
                != engine_deck
        {
            return Err("original native/external cylinder decks differ".into());
        }
        let work = std::env::temp_dir().join(format!(
            "p2_thick_cylinder_original_reference_{}",
            std::process::id()
        ));
        std::fs::create_dir_all(&work)?;
        std::fs::write(work.join("plasticity_thick_cylinder_0000.rad"), &starter)?;
        std::fs::write(
            work.join("plasticity_thick_cylinder_0001.rad"),
            &engine_deck,
        )?;
        let exe = sonium_qa::rad::engine_exe()
            .ok_or("OpenCourant required for original full cylinder measurement")?;
        let output = sonium_qa::rad::run_extended_case(&work, &exe, &case)?;
        let channels = case
            .base
            .probes
            .iter()
            .map(|p| p.channel.clone())
            .collect::<Vec<_>>();
        let values = channels
            .iter()
            .map(|c| output.history.channel(c).map(|v| v.to_vec()))
            .collect::<Result<Vec<_>, _>>()?;
        let final_plastic = [1, 265]
            .iter()
            .map(|id| {
                output
                    .history
                    .final_value(&sonium_qa::rad::bric_channel("PLAS", *id))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let result = serde_json::json!({"status":"external measurement; native parity not established",
            "engine":output.engine,"work_directory":work,"times_s":output.history.times,
            "probe_channels":channels,"probe_values":values,
            "plastic_probe_element_ids":[1,265],"final_element_plastic_strain":final_plastic,
            "starter_sha256":format!("{:x}",Sha256::digest(starter.as_bytes())),
            "engine_deck_sha256":format!("{:x}",Sha256::digest(engine_deck.as_bytes()))});
        std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
        println!(
            "complete original external plastic cylinder measured; diagnostic, no original QA/native parity pass"
        );
        return Ok(());
    }
    let scale = std::env::args().nth(1).map_or(Ok(1.0), |v| v.parse())?;
    let history = native::original_thick_cylinder_history(scale)?;
    std::fs::write(
        std::env::var("P2_THICK_CYLINDER_NATIVE_OUTPUT")?,
        serde_json::to_vec_pretty(&history)?,
    )?;
    println!(
        "complete original native plastic cylinder: stop={:.9e}, steps={}, peak final plastic strain={:.9e}, energy drift={:.9e}; diagnostic, original QA/native parity not established",
        history.final_time_s,
        history.steps,
        history
            .final_element_plastic_strain
            .iter()
            .copied()
            .fold(0.0_f64, f64::max),
        history.relative_energy_drift
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn invalid_timestep_scale_is_rejected_before_execution() {
        for scale in [0.0, -1.0, 2.0, f64::NAN, f64::INFINITY] {
            assert!(native::original_thick_cylinder_history(scale).is_err());
        }
    }
}
