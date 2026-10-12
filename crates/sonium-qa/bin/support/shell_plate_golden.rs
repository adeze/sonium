//! Native elastic shell dynamics compared with a harvested circular plate.
// Rust guideline compliant 2026-02-21

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::shell::explicit::{DynamicRelaxation, ShellExplicitModel};
use sonium_fem::physics::solid::shell::{ShellElement, ShellSection};
use sonium_qa::{QaResult, rad::clamped_plate_shell_mesh};
use std::path::Path;

fn sha256(path: &Path) -> Result<String, String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(std::fs::read(path).map_err(|e| e.to_string())?)
    ))
}

pub(super) fn run() -> Result<QaResult, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("decks/shell_clamped_plate");
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("golden.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if golden["schema"] != "sonium-shell-plate-golden-v1"
        || golden["engine_normal_termination"] != true
        || golden["model"]
            != json!({"radius_m":1.0,"thickness_m":0.02,"youngs_modulus_pa":200e9,"poisson_ratio":0.3,"density_kg_per_m3":7800.0,"pressure_pa":1000.0,"rings":8,"sectors":32})
    {
        return Err("shell reference model/provenance mismatch".into());
    }
    for name in [
        "shell_clamped_plate_0000.rad",
        "shell_clamped_plate_0001.rad",
    ] {
        if golden["deck_sha256"][name] != sha256(&root.join(name))? {
            return Err(format!("deck checksum mismatch: {name}"));
        }
    }
    if golden["history_sha256"] != sha256(&root.join("history.csv"))?
        || golden["engine_log_sha256"] != sha256(&root.join("engine-output.txt"))?
        || !std::fs::read_to_string(root.join("engine-output.txt"))
            .map_err(|e| e.to_string())?
            .contains("NORMAL TERMINATION")
    {
        return Err("shell harvest checksum/termination mismatch".into());
    }
    let csv = std::fs::read_to_string(root.join("history.csv")).map_err(|e| e.to_string())?;
    let mut lines = csv.lines();
    let header: Vec<_> = lines
        .next()
        .ok_or("empty shell history")?
        .split(',')
        .map(|s| s.trim().trim_matches('"'))
        .collect();
    let displacement = header
        .iter()
        .position(|s| {
            s.contains("nodes")
                && s.split_whitespace().any(|w| w == "1")
                && s.trim().ends_with("DZ")
        })
        .ok_or("missing center DZ")?;
    let energy = header
        .iter()
        .position(|s| *s == "INTERNAL ENERGY")
        .ok_or("missing internal energy")?;
    let kinetic = header
        .iter()
        .position(|s| *s == "KINETIC ENERGY")
        .ok_or("missing kinetic energy")?;
    let mut hold = Vec::new();
    let mut previous = -1.0;
    for line in lines {
        let row: Vec<f64> = line
            .split(',')
            .map(|s| s.parse::<f64>().map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        if row.len() != header.len() || row.iter().any(|v| !v.is_finite()) || row[0] <= previous {
            return Err("invalid shell history row".into());
        }
        previous = row[0];
        if row[0] >= 0.9 {
            hold.push([row[displacement], row[energy], row[kinetic]]);
        }
    }
    if hold.len() < 10
        || previous < 0.99
        || hold
            .iter()
            .any(|v| v[0] >= 0.0 || v[1] <= 0.0 || v[2] < 0.0)
    {
        return Err("incomplete/nonphysical shell pressure hold".into());
    }
    let reference_u = hold.iter().map(|v| v[0]).sum::<f64>() / hold.len() as f64;
    let reference_energy = hold.iter().map(|v| v[1]).sum::<f64>() / hold.len() as f64;
    let hold_variation = hold
        .iter()
        .map(|v| (v[0] - reference_u).abs() / reference_u.abs())
        .fold(0.0, f64::max);
    let kinetic_ratio = hold.iter().map(|v| v[2] / v[1]).fold(0.0, f64::max);
    let exact_u = -1000.0 / (64.0 * (200e9 * 0.02_f64.powi(3) / (12.0 * (1.0 - 0.3_f64.powi(2)))));
    let mut runs = Vec::new();
    for rings in [8, 16] {
        let mesh = clamped_plate_shell_mesh(1.0, rings, 4 * rings);
        let section = ShellSection::new(200e9, 0.3, 0.02, 7800.0);
        let points: Vec<_> = mesh
            .nodes
            .iter()
            .map(|p| Point::new_3d(p[0], p[1], p[2]))
            .collect();
        let mut triangles = mesh.tris;
        for q in mesh.quads {
            triangles.extend([[q[0], q[1], q[2]], [q[0], q[2], q[3]]]);
        }
        let elements = triangles
            .into_iter()
            .map(|t| {
                let ids = t.map(|n| n - 1);
                ShellElement::from_points(
                    ids,
                    &points[ids[0]],
                    &points[ids[1]],
                    &points[ids[2]],
                    section,
                )
            })
            .collect();
        let n_dofs = points.len() * 6;
        let mut model = ShellExplicitModel::elastic(elements, n_dofs).map_err(|e| e.to_string())?;
        for node in mesh.rim {
            model.fix_node(node - 1);
        }
        let mut load = vec![0.0; n_dofs];
        for (node, area) in mesh.nodal_area_m2.iter().enumerate() {
            load[node * 6 + 2] = -1000.0 * area;
        }
        let dt = 0.5 * model.elastic_critical_dt(2000).map_err(|e| e.to_string())?;
        let out = DynamicRelaxation::new(dt, 1e-6, 400_000)
            .map_err(|e| e.to_string())?
            .relax(&mut model, &load)
            .map_err(|e| e.to_string())?;
        if !out.converged {
            return Err(format!(
                "shell relaxation did not converge: rings={rings}, residual={}",
                out.residual_norm / out.load_norm
            ));
        }
        let u = out.displacement[2];
        let e = model
            .elastic_strain_energy(&out.displacement)
            .map_err(|e| e.to_string())?;
        let work = 0.5
            * load
                .iter()
                .zip(&out.displacement)
                .map(|(f, u)| f * u)
                .sum::<f64>();
        println!(
            "native plate rings={rings}: center={u:.9e}, energy={e:.9e}, steps={}, residual={:.3e}",
            out.steps,
            out.residual_norm / out.load_norm
        );
        runs.push((
            u,
            e,
            out.residual_norm / out.load_norm,
            (e - work).abs() / e,
            out.steps,
        ));
    }
    let fine = runs[1];
    let mut result = QaResult::passed("fem.shell-plate-harvest.v1")
        .formulation("native CST/DKT shell explicit dynamics with kinetic damping; all six rim DOFs fixed, interior unknowns free")
        .dimension("circular plate; 8x32 and 16x64 polar grids split into triangles")
        .reference("independent OpenCourant QEPH ramp-hold center displacement and internal energy; Kirchhoff circular plate; equilibrium comparison only")
        .reference_checksum(sha256(&root.join("golden.json"))?)
        .metric("numerical_solve_executed",1.0).minimum("numerical_solve_executed",1.0)
        .metric("native_solves",2.0).minimum("native_solves",2.0)
        .metric("explicit_steps",runs.iter().map(|r| r.4 as f64).sum()).minimum("explicit_steps",1.0);
    for name in [
        "shell_clamped_plate_0000.rad",
        "shell_clamped_plate_0001.rad",
        "history.csv",
        "engine-output.txt",
        "golden.json",
    ] {
        result = result.artifact_with_provenance(
            format!("decks/shell_clamped_plate/{name}"),
            "original sonium_qa circular plate deck and completed OpenCourant engine run",
            "scripts/harvest_shell_plate.py; native QA reads archived history independently",
            "OpenCourant engine commit 33e685176cccf0c539a3ce07aa2096985a284e2a",
            sha256(&root.join(name))?,
        );
    }
    for (name, value, bound) in [
        (
            "engine_center_error",
            (fine.0 - reference_u).abs() / reference_u.abs(),
            0.05,
        ),
        (
            "engine_energy_error",
            (fine.1 - reference_energy).abs() / reference_energy,
            0.05,
        ),
        (
            "kirchhoff_center_error",
            (fine.0 - exact_u).abs() / exact_u.abs(),
            0.02,
        ),
        ("reference_hold_variation", hold_variation, 0.01),
        ("reference_kinetic_ratio", kinetic_ratio, 1e-4),
        (
            "free_force_residual",
            runs.iter().map(|r| r.2).fold(0.0, f64::max),
            1e-6,
        ),
        (
            "native_energy_identity",
            runs.iter().map(|r| r.3).fold(0.0, f64::max),
            1e-5,
        ),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}={value} exceeds {bound}"));
        }
        result = result.metric(name, value).maximum(name, bound);
    }
    let rate = (((runs[0].0 - exact_u) / (fine.0 - exact_u)).abs()).log2();
    if !rate.is_finite() || rate < 1.0 {
        return Err(format!("shell spatial rate {rate} below 1"));
    }
    Ok(result
        .metric("center_refinement_rate", rate)
        .minimum("center_refinement_rate", 1.0))
}
