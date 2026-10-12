//! Native actual quadratic LAW76 replay against a completed independent engine run.
// Rust guideline compliant 2026-02-21

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sonium_fem::physics::solid::plasticity::funct::Funct1D;
use sonium_fem::physics::solid::plasticity::samp76_quadratic::{
    Samp76Quadratic, Samp76QuadraticState,
};
use sonium_qa::QaResult;
use std::path::Path;

fn sha256(path: &Path) -> Result<String, String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(std::fs::read(path).map_err(|e| e.to_string())?)
    ))
}
fn displacement(time: f64, amplitude: f64) -> f64 {
    if time <= 0.001 {
        amplitude * time / 0.001
    } else if time <= 0.0015 {
        amplitude
    } else if time <= 0.0025 {
        amplitude * (1.0 - (time - 0.0015) / 0.001)
    } else {
        0.0
    }
}

pub(super) fn run(compression: bool) -> Result<QaResult, String> {
    let mode = if compression {
        "compression"
    } else {
        "tension"
    };
    let amplitude: f64 = if compression { -0.00004 } else { 0.00002 };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join(format!("decks/foam76_{mode}"));
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("golden.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    let model = json!({"young_pa":350e6,"poisson":0.0,"tension_pa":500000.0,"compression_pa":900000.0,"shear_pa":420000.0,"hardening_slope":20.0,"nu_p":0.5,"iform":0,"iquad":1,"iconv":1,"density_kg_m3":433.0,"side_m":0.01,"isolid":18});
    if golden["schema"] != "sonium-foam76-golden-v1"
        || golden["engine_normal_termination"] != true
        || golden["model"] != model
        || golden["mode"] != mode
        || golden["path_s_m"]
            != json!([
                [0.0, 0.0],
                [0.001, amplitude],
                [0.0015, amplitude],
                [0.0025, 0.0],
                [0.003, 0.0]
            ])
    {
        return Err("LAW76 golden model/path mismatch".into());
    }
    for name in ["foam76_cube_0000.rad", "foam76_cube_0001.rad"] {
        if golden["deck_sha256"][name] != sha256(&root.join(name))? {
            return Err(format!("LAW76 deck checksum mismatch: {name}"));
        }
    }
    if golden["history_sha256"] != sha256(&root.join("history.csv"))?
        || golden["engine_log_sha256"] != sha256(&root.join("engine-output.txt"))?
        || !std::fs::read_to_string(root.join("engine-output.txt"))
            .map_err(|e| e.to_string())?
            .contains("NORMAL TERMINATION")
    {
        return Err("LAW76 harvest checksum/termination mismatch".into());
    }
    let csv = std::fs::read_to_string(root.join("history.csv")).map_err(|e| e.to_string())?;
    let mut lines = csv.lines();
    let header: Vec<_> = lines
        .next()
        .ok_or("empty LAW76 CSV")?
        .split(',')
        .map(|s| s.trim().trim_matches('"'))
        .collect();
    if header.len() != 41
        || !header[25].contains("cube")
        || !header[25].ends_with("var 25")
        || !header[32].ends_with("var 32")
        || !header[35].contains("moving_node")
        || !header[33].ends_with("var 33")
    {
        return Err("LAW76 CSV variable mapping changed".into());
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
        || rows.last().ok_or("no LAW76 rows")?[0] < 0.00295
    {
        return Err("invalid LAW76 replay grid".into());
    }
    let hardening = Funct1D::new(vec![0.0, 0.5], vec![1.0, 11.0])?;
    let law = Samp76Quadratic::new(350e6, 0.0, 500000.0, 900000.0, 420000.0, hardening.clone())?;
    let mut state = Samp76QuadraticState::default();
    let pressure_blind = Samp76Quadratic::new(
        350e6,
        0.0,
        3.0_f64.sqrt() * 420000.0,
        3.0_f64.sqrt() * 420000.0,
        420000.0,
        hardening,
    )?;
    let mut pressure_blind_state = Samp76QuadraticState::default();
    let mut plastic_error = 0.0_f64;
    let plastic_scale = rows.iter().map(|r| r[33]).fold(0.0, f64::max);
    if plastic_scale < 0.0005 {
        return Err("LAW76 history does not yield".into());
    }
    let mut pressure_control_gap = 0.0_f64;
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
        return Err("LAW76 fixture has no stress work".into());
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
                    let j0 = 1.0 + displacement(left, amplitude) / 0.01;
                    let j1 = 1.0 + displacement(right, amplitude) / 0.01;
                    let old = state.stress[0];
                    let increment = [(j1 / j0).ln(), 0.0, 0.0, 0.0, 0.0, 0.0];
                    let update = law.step(&mut state, &increment)?;
                    pressure_blind.step(&mut pressure_blind_state, &increment)?;
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
        pressure_control_gap =
            pressure_control_gap.max((pressure_blind_state.stress[0] - row[25]).abs() / 500000.0);
        let j = 1.0 + displacement(time, amplitude) / 0.01;
        density_error = density_error.max((433.0 / j - row[32]).abs() / row[32]);
        geometry_error =
            geometry_error.max((displacement(time, amplitude) - row[35]).abs() / amplitude.abs());
        work_error = work_error.max((work - row[1]).abs() / energy_scale);
        plastic_error = plastic_error.max((state.plastic_strain - row[33]).abs() / plastic_scale);
        maximum_plastic = maximum_plastic.max(state.plastic_strain);
        previous = time;
    }
    if !pressure_control_gap.is_finite() || pressure_control_gap < 0.01 {
        return Err("LAW76 pressure-sensitive control is insensitive".into());
    }
    let mut result=QaResult::passed(if compression {"fem.foam-law76-compression-replay.v1"} else {"fem.foam-law76-tension-replay.v1"})
        .formulation("native actual quadratic LAW76 nonassociated isochoric return with tension/compression/shear curves and proportional hardening")
        .dimension("3D material point, constrained transverse dimensions; load/hold/unload/hold; no spatial solve")
        .reference("independent original OpenCourant single-brick stress/plastic-strain/density/displacement/work history")
        .reference_checksum(sha256(&root.join("golden.json"))?)
        .metric("numerical_solve_executed",0.0).maximum("numerical_solve_executed",0.0)
        .metric("native_material_evolution",1.0).minimum("native_material_evolution",1.0)
        .metric("native_stamps",rows.len() as f64).minimum("native_stamps",110.0)
        .metric("native_substeps",substeps as f64).minimum("native_substeps",28000.0)
        .metric("pressure_blind_stress_gap",pressure_control_gap).minimum("pressure_blind_stress_gap",0.01)
        .metric("plastic_strain",maximum_plastic).minimum("plastic_strain",0.0005);
    for (name, value, bound) in [
        ("stress_history_error", stress_error, 0.002),
        ("plastic_history_error", plastic_error, 0.005),
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
        "foam76_cube_0000.rad",
        "foam76_cube_0001.rad",
        "history.csv",
        "engine-output.txt",
        "golden.json",
    ] {
        result = result.artifact_with_provenance(
            format!("decks/foam76_{mode}/{name}"),
            "original actual LAW76 engine fixture with explicit quadratic isochoric card",
            "scripts/generate_foam76_deck.py and completed independent engine run",
            "OpenCourant commit 33e685176cccf0c539a3ce07aa2096985a284e2a",
            sha256(&root.join(name))?,
        );
    }
    Ok(result)
}
