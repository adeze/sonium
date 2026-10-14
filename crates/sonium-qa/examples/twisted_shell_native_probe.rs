//! Measure the complete original native twisted-shell transient.

// Rust guideline compliant 2026-02-21

#[path = "../bin/fem_opencourant_shell_twisted_beam.rs"]
#[expect(
    dead_code,
    reason = "The fixture factory and native solver are used; the external runner is not invoked"
)]
mod reference;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    let nx = args.first().map_or(Ok(32), |v| v.parse())?;
    let ny = args.get(1).map_or(Ok(8), |v| v.parse())?;
    let dt = args.get(2).map_or(Ok(0.000025), |v| v.parse())?;
    let result = reference::twisted_native::probe(nx, ny, dt)?;
    std::fs::write(
        std::env::var("P2_TWISTED_NATIVE_OUTPUT")?,
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(())
}
