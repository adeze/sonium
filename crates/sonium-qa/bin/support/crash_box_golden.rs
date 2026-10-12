//! Full original CRASH impact evolution compared with independently harvested histories.
// Rust guideline compliant 2026-02-21
use super::crash_box::{Options, Snapshot, simulate};
use serde_json::Value;
use sha2::{Digest, Sha256};
use sonium_qa::QaResult;
use std::path::Path;

fn checksum(path: &Path) -> Result<String, String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(std::fs::read(path).map_err(|e| e.to_string())?)
    ))
}
fn at(history: &[Snapshot], time: f64, channel: usize) -> Result<f64, String> {
    if !time.is_finite() || time < history[0].time || time > history.last().unwrap().time {
        return Err("reference stamp outside native integration interval".into());
    }
    let j = history
        .partition_point(|s| s.time <= time)
        .saturating_sub(1);
    if j + 1 == history.len() {
        return Ok(history[j].values[channel]);
    }
    let a = &history[j];
    let b = &history[j + 1];
    let fraction = (time - a.time) / (b.time - a.time);
    Ok(a.values[channel] * (1. - fraction) + b.values[channel] * fraction)
}
fn bounded(mut result: QaResult, name: &str, value: f64, bound: f64) -> Result<QaResult, String> {
    if !value.is_finite() || value > bound {
        return Err(format!("original CRASH {name}={value} exceeds {bound}"));
    }
    result = result.metric(name, value).maximum(name, bound);
    Ok(result)
}
pub(super) fn run() -> Result<QaResult, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let model_path = root.join("decks/crash_box_original/model.json");
    let golden_path = root.join("goldens/miniqa/CRASH/crashsection/box1_42.json");
    let model: Value =
        serde_json::from_str(&std::fs::read_to_string(&model_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let golden: Value =
        serde_json::from_str(&std::fs::read_to_string(&golden_path).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
    let digest = checksum(&golden_path)?;
    if model["golden_sha256"] != digest
        || golden["meta"]["normal_termination"] != true
        || model["starter_sha256"] != golden["meta"]["deck_sha256"]["starter"]
        || model["engine_deck_sha256"] != golden["meta"]["deck_sha256"]["engine"]
        || model["quads"].as_array().map(Vec::len) != Some(140)
        || model["nodes"].as_array().map(Vec::len) != Some(155)
    {
        return Err("original CRASH reconstruction/provenance mismatch".into());
    }
    let times: Vec<f64> = golden["time"]
        .as_array()
        .ok_or("reference times")?
        .iter()
        .map(|v| {
            v.as_f64()
                .map(|t| t * 0.001)
                .ok_or("invalid reference stamp")
        })
        .collect::<Result<_, _>>()?;
    if times.len() < 2 || times.windows(2).any(|t| !t[0].is_finite() || t[1] <= t[0]) {
        return Err("invalid original CRASH reference time grid".into());
    }
    let end = *times.last().unwrap();
    let coarse = simulate(&model, 2500, end, Options::original())?;
    println!("native original CRASH: 2500 explicit steps completed");
    let fine = simulate(&model, 5000, end, Options::original())?;
    println!("native original CRASH: 5000 explicit steps completed");
    // Counterfactuals must independently fail the original response gates.
    let mut elastic_model = model.clone();
    elastic_model["law2"]["a_pa"] = 1e30.into();
    elastic_model["law2"]["b_pa"] = 0_f64.into();
    elastic_model["law2"]["max_stress_pa"] = 1e30.into();
    let elastic = simulate(&elastic_model, 5000, end, Options::original())?;
    let mut free_wall_options = Options::original();
    free_wall_options.tied_wall = false;
    let free_wall = simulate(&model, 5000, end, free_wall_options)?;
    let mut result=QaResult::passed("fem.crash-box-impact-harvest.v1")
        .formulation("native Q4 shell force assembly, committed LAW2 explicit in-plane projection, surface-sampled thickness, convected frame, card hourglass, tied wall and rigid-body dynamics")
        .dimension("original 140 Q4 shells, 154 shell nodes plus master 155; original 100 kg point mass and 140 kg rigid-body mass")
        .reference("independent original CRASH box1_42 TH histories; fixed physical similarity transform L/time=0.001 and stress/density=1e9")
        .reference_checksum(&digest)
        .metric("numerical_solve_executed",1.).minimum("numerical_solve_executed",1.)
        .metric("native_solves",4.).minimum("native_solves",4.)
        .metric("explicit_steps",17500.).minimum("explicit_steps",17500.)
        .metric("compared_reference_samples",times.len() as f64).minimum("compared_reference_samples",100.);
    for (path, description) in [
        (
            &model_path,
            "SHA-checked original mesh/cards with documented physical similarity transform",
        ),
        (
            &golden_path,
            "independently harvested original external-engine TH histories",
        ),
    ] {
        result=result.artifact_with_provenance(path.strip_prefix(root).unwrap().to_string_lossy(),description,
            "scripts/extract_crash_box.py and original miniqa harvest; native evolution reads reference only after solving",
            "OpenCourant engine commit 33e685176cccf0c539a3ce07aa2096985a284e2a",checksum(path)?);
    }
    // One-percent peak-normalized waveform bounds measure native response
    // accuracy. Existing harvested-summary/rounding integrity bounds remain
    // unchanged in the separate integrity result. All reference stamps count.
    for (i, label, scale, name) in [
        (0, "INTERNAL ENERGY", 1., "internal"),
        (1, "PLASTIC WORK", 1., "plastic"),
        (2, "HOURGLASS ENERGY", 1., "hourglass"),
        (3, "KINETIC ENERGY", 1., "kinetic"),
        (4, "Rigid walls 1 rigid_wall_1 ~1", 1., "wall_impulse"),
        (5, "nodes 17 ~1", 0.001, "node17_x"),
        (6, "nodes 17 ~2", 0.001, "node17_y"),
        (7, "nodes 17 ~3", 0.001, "node17_z"),
        (8, "nodes 83 ~1", 0.001, "node83_x"),
    ] {
        let values = golden["channels"][label]["values"]
            .as_array()
            .ok_or("reference channel")?;
        if values.len() != times.len() {
            return Err("reference channel/time length mismatch".into());
        }
        let references = values
            .iter()
            .map(|v| {
                v.as_f64()
                    .map(|v| v * scale)
                    .ok_or("invalid reference value")
            })
            .collect::<Result<Vec<_>, _>>()?;
        let peak = references.iter().map(|v| v.abs()).fold(0_f64, f64::max);
        if !peak.is_finite() || peak <= 0. {
            return Err("invalid reference normalization".into());
        }
        let mut error = 0_f64;
        let mut refinement = 0_f64;
        for (&time, &reference) in times.iter().zip(&references) {
            let fv = at(&fine, time, i)?;
            let cv = at(&coarse, time, i)?;
            if !fv.is_finite() || !cv.is_finite() {
                return Err("nonfinite native history".into());
            }
            error = error.max((fv - reference).abs() / peak);
            refinement = refinement.max((fv - cv).abs() / peak);
        }
        println!(
            "original CRASH {name}: history error={error:.5e}, timestep change={refinement:.5e}"
        );
        result = bounded(result, &format!("{name}_history_error"), error, 0.01)?;
        result = bounded(
            result,
            &format!("{name}_timestep_change"),
            refinement,
            0.003,
        )?;
        if i == 0 {
            let change = (at(&elastic, end, i)? - at(&fine, end, i)?).abs() / peak;
            if change < 0.5 || !change.is_finite() {
                return Err("elastic counterfactual fails to distinguish yielding".into());
            }
            result = result
                .metric("elastic_control_response_change", change)
                .minimum("elastic_control_response_change", 0.5);
        }
        if i == 6 {
            let change = (at(&free_wall, end, i)? - at(&fine, end, i)?).abs() / peak;
            if change < 0.1 || !change.is_finite() {
                return Err("free-wall counterfactual fails to distinguish tied contact".into());
            }
            result = result
                .metric("wall_control_response_change", change)
                .minimum("wall_control_response_change", 0.1);
        }
    }
    let last = &fine.last().unwrap().values;
    result = bounded(
        result,
        "energy_balance_relative",
        (last[0] + last[2] + last[3] - fine[0].values[3]).abs() / fine[0].values[3],
        1e-6,
    )?;
    result = bounded(
        result,
        "elastic_control_plastic_work",
        elastic.last().unwrap().values[1].abs(),
        1e-12,
    )?;
    if last[9] < 0.01 || !last[9].is_finite() {
        return Err("original CRASH has not evolved plastic strain".into());
    }
    Ok(result
        .metric("maximum_equivalent_plastic_strain", last[9])
        .minimum("maximum_equivalent_plastic_strain", 0.01))
}
