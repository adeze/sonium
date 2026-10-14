//! Diagnose the original complete native roof gravity history.

// Rust guideline compliant 2026-02-21

#[path = "../bin/fem_opencourant_shell_scordelis_lo.rs"]
#[expect(
    dead_code,
    reason = "The original native roof diagnostic is invoked; the external runner is not"
)]
mod reference;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let cells = args.first().map_or(Ok(32), |v| v.parse())?;
    let dt = args.get(1).map_or(Ok(0.002), |v| v.parse())?;
    let result = reference::scordelis_native::probe(cells, dt)?;
    std::fs::write(
        std::env::var("P2_SCORDELIS_NATIVE_OUTPUT")?,
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(())
}
