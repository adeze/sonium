//! Diagnose the complete original native open-hole elastic shell history.

// Rust guideline compliant 2026-02-21

#[path = "../bin/fem_opencourant_shell_oht_elastic.rs"]
#[expect(dead_code, reason = "Only the original fixture factory is used")]
mod reference;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    if let Ok(path) = std::env::var("P2_OHT_REFERENCE_OUTPUT") {
        let case = reference::shell_oht_elastic_case();
        let engine =
            sonium_qa::rad::engine_exe().ok_or("OpenCourant required for external measurement")?;
        let work =
            std::env::temp_dir().join(format!("p2_oht_original_reference_{}", std::process::id()));
        std::fs::create_dir_all(&work)?;
        std::fs::write(
            work.join("shell_oht_elastic_0000.rad"),
            sonium_qa::rad::write_extended_starter(&case),
        )?;
        std::fs::write(
            work.join("shell_oht_elastic_0001.rad"),
            sonium_qa::rad::write_extended_engine(&case),
        )?;
        let output = sonium_qa::rad::run_extended_case(&work, &engine, &case)?;
        let values = case
            .base
            .probes
            .iter()
            .map(|probe| output.history.final_value(&probe.channel))
            .collect::<Result<Vec<_>, _>>()?;
        let result = serde_json::json!({"status":"external measurement; native parity not yet established",
            "engine":output.engine,"work_directory":work,"final_time_s":output.history.times.last(),
            "original_probe_values":values,"original_probe_channels":case.base.probes.iter().map(|p|&p.channel).collect::<Vec<_>>()});
        std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }
    let dt = std::env::args().nth(1).map_or(Ok(1e-6), |v| v.parse())?;
    let factor = std::env::args().nth(2).map_or(Ok(1), |v| v.parse())?;
    let result = reference::oht_native::probe(factor, dt)?;
    std::fs::write(
        std::env::var("P2_OHT_NATIVE_OUTPUT")?,
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
