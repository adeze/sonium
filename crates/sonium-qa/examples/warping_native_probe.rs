//! Measure a complete native warping transient at a requested mesh and timestep.

// Rust guideline compliant 2026-02-21

#[path = "../bin/fem_opencourant_beam_warping.rs"]
#[expect(
    dead_code,
    reason = "Only the original full native transient is invoked"
)]
mod reference;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let factor = args.first().map_or(Ok(5), |v| v.parse())?;
    let timestep = args.get(1).map_or(Ok(0.00025), |v| v.parse())?;
    let result = reference::warping_native::probe_transient(factor, timestep)?;
    std::fs::write(
        std::env::var("P2_WARPING_NATIVE_OUTPUT")?,
        serde_json::to_vec_pretty(&result)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
