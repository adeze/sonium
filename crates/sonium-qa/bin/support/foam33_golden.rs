//! Native plain LAW33 replay against a completed independent engine run.
// Rust guideline compliant 2026-02-21

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sonium_fem::physics::solid::plasticity::foam_plas33::{
    FoamPlas33, FoamPlasIncrement, FoamPlasState,
};
use sonium_qa::QaResult;
use std::path::Path;

fn sha256(path: &Path) -> Result<String, String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(std::fs::read(path).map_err(|e| e.to_string())?)
    ))
}
fn displacement(time: f64) -> f64 {
    if time <= 0.001 {
        -0.5 * time
    } else if time <= 0.0015 {
        -0.0005
    } else if time <= 0.0025 {
        -0.0005 + 0.5 * (time - 0.0015)
    } else {
        0.0
    }
}

pub(super) fn run() -> Result<QaResult, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("decks/foam33_cube");
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("golden.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let model = json!({"young_pa":350e6,"a_pa":-177000.0,"b_pa":677000.0,"c":-3.34,"pressure0_pa":1e5,"porosity":0.1,"gamma0":0.0,"density_kg_m3":433.0,"side_m":0.01,"ka":0,"isolid":18});
    if golden["schema"] != "sonium-foam33-golden-v1"
        || golden["engine_normal_termination"] != true
        || golden["model"] != model
        || golden["path_s_m"]
            != json!([
                [0.0, 0.0],
                [0.001, -0.0005],
                [0.0015, -0.0005],
                [0.0025, 0.0],
                [0.003, 0.0]
            ])
    {
        return Err("LAW33 golden model/path mismatch".into());
    }
    for name in ["foam33_cube_0000.rad", "foam33_cube_0001.rad"] {
        if golden["deck_sha256"][name] != sha256(&root.join(name))? {
            return Err(format!("LAW33 deck checksum mismatch: {name}"));
        }
    }
    if golden["history_sha256"] != sha256(&root.join("history.csv"))?
        || golden["engine_log_sha256"] != sha256(&root.join("engine-output.txt"))?
        || !std::fs::read_to_string(root.join("engine-output.txt"))
            .map_err(|e| e.to_string())?
            .contains("NORMAL TERMINATION")
    {
        return Err("LAW33 harvest checksum/termination mismatch".into());
    }
    let csv = std::fs::read_to_string(root.join("history.csv")).map_err(|e| e.to_string())?;
    let mut lines = csv.lines();
    let header: Vec<_> = lines
        .next()
        .ok_or("empty LAW33 CSV")?
        .split(',')
        .map(|s| s.trim().trim_matches('"'))
        .collect();
    if header.len() != 41
        || !header[25].contains("cube")
        || !header[25].ends_with("var 25")
        || !header[32].ends_with("var 32")
        || !header[35].contains("moving_node")
    {
        return Err("LAW33 CSV variable mapping changed".into());
    }
    let rows: Vec<Vec<f64>> = lines
        .map(|line| {
            line.split(',')
                .map(|s| s.parse::<f64>().map_err(|e| e.to_string()))
                .collect::<Result<_, _>>()
        })
        .collect::<Result<_, _>>()?;
    if rows.len() < 110
        || rows[0][0] != 0.0
        || rows
            .iter()
            .any(|r| r.len() != header.len() || r.iter().any(|v| !v.is_finite()))
        || rows.windows(2).any(|w| w[1][0] <= w[0][0])
        || rows.last().ok_or("no LAW33 rows")?[0] < 0.00295
    {
        return Err("invalid LAW33 replay grid".into());
    }
    let law = FoamPlas33::new(350e6, -177000.0, 677000.0, -3.34, 1e5, 0.1, 0.0)?;
    let mut state = FoamPlasState::default();
    let air_disabled = FoamPlas33::new(350e6, -177000.0, 677000.0, -3.34, 0.0, 0.1, 0.0)?;
    let mut air_disabled_state = FoamPlasState::default();
    let mut air_control_gap = 0.0_f64;
    let mut work = 0.0;
    let mut previous = 0.0;
    let mut substeps = 0;
    let mut stress_error = 0.0_f64;
    let mut density_error = 0.0_f64;
    let mut work_error = 0.0_f64;
    let mut geometry_error = 0.0_f64;
    let mut maximum_plastic = 0.0_f64;
    let energy_scale = rows.iter().map(|r| r[1]).fold(0.0, f64::max);
    if energy_scale <= 0.0 {
        return Err("LAW33 fixture has no stress work".into());
    }
    for (i, row) in rows.iter().enumerate() {
        let time = row[0];
        if i > 0 {
            // Split at each imposed-path corner, independently of TH cadence.
            let mut boundaries = vec![previous];
            for corner in [0.001, 0.0015, 0.0025] {
                if corner > previous && corner < time {
                    boundaries.push(corner);
                }
            }
            boundaries.push(time);
            for interval in boundaries.windows(2) {
                for step in 1..=256 {
                    let left =
                        interval[0] + (interval[1] - interval[0]) * (step - 1) as f64 / 256.0;
                    let right = interval[0] + (interval[1] - interval[0]) * step as f64 / 256.0;
                    let j0 = 1.0 + displacement(left) / 0.01;
                    let j1 = 1.0 + displacement(right) / 0.01;
                    let old = state.stress[0];
                    let update = law.step(
                        &mut state,
                        &FoamPlasIncrement {
                            strain: [(j1 / j0).ln(), 0.0, 0.0, 0.0, 0.0, 0.0],
                            volume_ratio: j1,
                        },
                    )?;
                    air_disabled.step(
                        &mut air_disabled_state,
                        &FoamPlasIncrement {
                            strain: [(j1 / j0).ln(), 0.0, 0.0, 0.0, 0.0, 0.0],
                            volume_ratio: j1,
                        },
                    )?;
                    // Fixed transverse dimensions: sigma A dx = sigma V0 dJ.
                    work += 0.5 * (old + update.stress[0]) * 1e-6 * (j1 - j0);
                    substeps += 1;
                }
            }
        }
        for component in 0..6 {
            stress_error =
                stress_error.max((state.stress[component] - row[25 + component]).abs() / 500000.0);
        }
        air_control_gap =
            air_control_gap.max((air_disabled_state.stress[0] - row[25]).abs() / 500000.0);
        let j = 1.0 + displacement(time) / 0.01;
        density_error = density_error.max((433.0 / j - row[32]).abs() / row[32]);
        geometry_error = geometry_error.max((displacement(time) - row[35]).abs() / 0.0005);
        work_error = work_error.max((work - row[1]).abs() / energy_scale);
        maximum_plastic = maximum_plastic.max(state.equivalent_plastic_strain);
        previous = time;
    }
    if !air_control_gap.is_finite() || air_control_gap < 0.01 {
        return Err("LAW33 trapped-air control is insensitive".into());
    }
    let mut result=QaResult::passed("fem.foam-law33-material-replay.v1")
        .formulation("native plain FOAM_PLAS Ka=0 principal stress cap, trapped-air pressure, independently prescribed logarithmic strain and geometrical volume")
        .dimension("3D material point, constrained transverse dimensions; compression/hold/unload/hold; no spatial solve")
        .reference("independent original OpenCourant single-brick stress/density/displacement/work history")
        .reference_checksum(sha256(&root.join("golden.json"))?)
        .metric("numerical_solve_executed",0.0).maximum("numerical_solve_executed",0.0)
        .metric("native_material_evolution",1.0).minimum("native_material_evolution",1.0)
        .metric("native_stamps",rows.len() as f64).minimum("native_stamps",110.0)
        .metric("native_substeps",substeps as f64).minimum("native_substeps",28000.0)
        .metric("air_disabled_stress_gap",air_control_gap).minimum("air_disabled_stress_gap",0.01)
        .metric("plastic_strain",maximum_plastic).minimum("plastic_strain",0.01);
    for (name, value, bound) in [
        ("stress_history_error", stress_error, 0.005),
        ("work_history_error", work_error, 0.005),
        ("density_history_error", density_error, 1e-5),
        ("imposed_geometry_error", geometry_error, 1e-5),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}={value} exceeds {bound}"));
        }
        result = result.metric(name, value).maximum(name, bound);
    }
    for name in [
        "foam33_cube_0000.rad",
        "foam33_cube_0001.rad",
        "history.csv",
        "engine-output.txt",
        "golden.json",
    ] {
        result = result.artifact_with_provenance(
            format!("decks/foam33_cube/{name}"),
            "original LAW33 engine fixture with plain FOAM_PLAS card",
            "scripts/generate_foam33_deck.py and completed independent engine run",
            "OpenCourant commit 33e685176cccf0c539a3ce07aa2096985a284e2a",
            sha256(&root.join(name))?,
        );
    }
    Ok(result)
}
