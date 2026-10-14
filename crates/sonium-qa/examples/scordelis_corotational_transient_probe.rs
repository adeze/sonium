//! Measure the original complete nonlinear roof history with shared QA support.
//!
//! Retains the original full load, physical mass, free rotations and8s duration.
//! A completed diagnostic does not establish original QA native/reference parity.

// Rust guideline compliant 2026-02-21

#[path = "../bin/support/scordelis_corotational.rs"]
mod corotational;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cells: usize = std::env::args().nth(1).map_or(Ok(12), |v| v.parse())?;
    let scale: f64 = std::env::args().nth(2).map_or(Ok(1.0), |v| v.parse())?;
    let result = corotational::probe(cells, scale)?;
    std::fs::write(
        std::env::var("P2_SCORDELIS_COROTATIONAL_TRANSIENT_OUTPUT")?,
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(())
}
