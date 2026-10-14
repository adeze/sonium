//! Compare direct factorizations on the complete original I-section transient.

// Rust guideline compliant 2026-02-21

#[path = "../bin/fem_opencourant_beam_warping.rs"]
#[expect(
    dead_code,
    reason = "The reference module supplies the original case and native solver; the external runner is not invoked here"
)]
mod reference;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let factor = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "1".into())
        .parse()?;
    reference::warping_native::compare_factorizations(factor)
}
