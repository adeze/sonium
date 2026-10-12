//! Development CSV export of the native original CRASH box simulation.
// Rust guideline compliant 2026-02-21
#[path = "../bin/support/crash_box.rs"]
mod crash_box;
use serde_json::Value;
use std::path::Path;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut meta: Value = serde_json::from_str(&std::fs::read_to_string(
        root.join("decks/crash_box_original/model.json"),
    )?)?;
    let golden: Value = serde_json::from_str(&std::fs::read_to_string(
        root.join("goldens/miniqa/CRASH/crashsection/box1_42.json"),
    )?)?;
    let mut options = crash_box::Options::original();
    for (name, field) in [
        ("PROJECTION", &mut options.projection),
        ("ROTARY", &mut options.rotary),
        ("LAYERS", &mut options.surface_layers),
        ("FRAME", &mut options.convected),
        ("RATE", &mut options.section_rate),
    ] {
        if let Ok(value) = std::env::var(format!("CRASH_BOX_{name}")) {
            match value.as_str() {
                "1" => *field = true,
                "0" => *field = false,
                _ => return Err(format!("CRASH_BOX_{name} must be 0 or 1").into()),
            }
        }
    }
    if let Ok(value) = std::env::var("CRASH_BOX_HG") {
        options.hourglass = value.parse()?;
    }
    if options.hourglass > 3 {
        return Err("CRASH_BOX_HG must be 0, 1, 2 or 3".into());
    }
    if std::env::var("CRASH_BOX_ELASTIC").as_deref() == Ok("1") {
        // Same elastic counterfactual as a/cap=1e21 and b=0 in literal deck units.
        meta["law2"]["a_pa"] = 1e30.into();
        meta["law2"]["b_pa"] = 0_f64.into();
        meta["law2"]["max_stress_pa"] = 1e30.into();
    }
    let count = std::env::var("CRASH_BOX_STEPS")
        .ok()
        .map(|s| s.parse::<usize>())
        .transpose()?
        .unwrap_or(5000);
    let end = std::env::var("CRASH_BOX_END_S")
        .ok()
        .map(|s| s.parse::<f64>())
        .transpose()?
        .unwrap_or(
            golden["time"]
                .as_array()
                .unwrap()
                .last()
                .unwrap()
                .as_f64()
                .unwrap()
                * 0.001,
        );
    let history = crash_box::simulate(&meta, count, end, options)?;
    println!(
        "time_s,internal_j,plastic_j,hourglass_j,kinetic_j,wall_impulse_ns,node17_dx_m,node17_dy_m,node17_dz_m,node83_dx_m,max_plastic"
    );
    for (i, s) in history.iter().enumerate() {
        if i % (count / 100).max(1) == 0 || i == count {
            print!("{:.12e}", s.time);
            for v in s.values {
                print!(",{v:.12e}");
            }
            println!();
        }
    }
    let initial = history[0].values[3];
    let last = &history.last().unwrap().values;
    eprintln!(
        "initial KE={initial}; final energy residual={} J",
        last[0] + last[2] + last[3] - initial
    );
    Ok(())
}
