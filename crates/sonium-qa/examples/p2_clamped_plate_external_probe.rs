//! Record the original external plate history without claiming native parity.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    assert_probe_close, engine_exe, run_extended_case, shell_clamped_plate_case,
    write_extended_engine, write_extended_starter,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let case = shell_clamped_plate_case();
    case.validate()?;
    let engine =
        engine_exe().ok_or("OpenCourant engine is required for this reference diagnostic")?;
    let work = std::env::temp_dir().join(format!("p2_plate_external_{}", std::process::id()));
    std::fs::create_dir_all(&work)?;
    std::fs::write(
        work.join("shell_clamped_plate_0000.rad"),
        write_extended_starter(&case),
    )?;
    std::fs::write(
        work.join("shell_clamped_plate_0001.rad"),
        write_extended_engine(&case),
    )?;
    let result = run_extended_case(&work, &engine, &case)?;
    let mut values = Vec::new();
    for probe in &case.base.probes {
        let value = result.history.final_value(&probe.channel)?;
        assert_probe_close(
            value,
            probe.expected,
            probe.abs_tol,
            probe.rel_tol,
            &probe.label,
        )?;
        values.push(
            serde_json::json!({"channel":probe.channel,"value":value,"analytical":probe.expected}),
        );
    }
    let record = serde_json::json!({"scope":"Original full external plate reference health only; not native parity", "engine":result.engine,"work_directory":work,"probes":values});
    std::fs::write(
        std::env::var("P2_PLATE_EXTERNAL_OUTPUT")?,
        serde_json::to_vec_pretty(&record)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&record)?);
    Ok(())
}
