//! Measure the original full nonlinear roof with native energy-consistent MITC4.

// Rust guideline compliant 2026-02-21

#[path = "../bin/support/scordelis_native.rs"]
pub mod native;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cells: usize = std::env::args().nth(1).map_or(Ok(12), |v| v.parse())?;
    let scale: f64 = std::env::args().nth(2).map_or(Ok(1.0), |v| v.parse())?;
    let result = native::quadrilateral::probe(cells, scale)?;
    std::fs::write(
        std::env::var("P2_SCORDELIS_QUAD_OUTPUT")?,
        serde_json::to_vec_pretty(&result)?,
    )?;
    Ok(())
}
