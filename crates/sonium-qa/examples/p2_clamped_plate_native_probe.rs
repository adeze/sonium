//! Record a full native shell transient without claiming external parity.

// Rust guideline compliant 2026-02-21

#[path = "../bin/support/plate_native.rs"]
#[expect(
    dead_code,
    reason = "The standalone solver diagnostic does not invoke the external parity gate"
)]
mod plate_native;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let rings = args.first().map_or(Ok(8), |value| value.parse())?;
    let sectors = args.get(1).map_or(Ok(32), |value| value.parse())?;
    let dt = args.get(2).map_or(Ok(0.00025), |value| value.parse())?;
    let result = match args.get(3).map(String::as_str).unwrap_or("cholesky") {
        "cholesky" => plate_native::solve(rings, sectors, dt)?,
        "scaled" => plate_native::solve_scaled(rings, sectors, dt)?,
        "lu" => plate_native::solve_lu(rings, sectors, dt)?,
        "hht" => plate_native::solve_hht(rings, sectors, dt)?,
        _ => return Err("method must be cholesky, scaled, lu or hht".into()),
    };
    if let Ok(path) = std::env::var("P2_PLATE_OUTPUT") {
        std::fs::write(path, serde_json::to_vec_pretty(&result)?)?;
    }
    Ok(())
}
