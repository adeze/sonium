//! Native meridional solve against an independently harvested axisymmetric deck.

// Rust guideline compliant 2026-02-21

use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::axisymmetric::{
    AxisymmetricCylinderGeometry, AxisymmetricCylinderMaterial, AxisymmetricCylinderStudy,
    run_axisymmetric_cylinder,
};
use std::path::Path;

fn sha256(path: &Path) -> Result<String, String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    Ok(format!("{:x}", Sha256::digest(bytes)))
}

fn scalar(doc: &Value, key: &str) -> Result<f64, String> {
    doc[key]
        .as_f64()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("invalid golden scalar {key}"))
}

pub(super) fn run() -> Result<QaResult, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("decks/axisym_cylinder");
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("golden.json")).map_err(|e| e.to_string())?,
    )
    .map_err(|e| e.to_string())?;
    if golden["schema"] != "sonium-axisym-golden-v1" || golden["engine_normal_termination"] != true
    {
        return Err("axisymmetric reference lacks normal engine termination".into());
    }
    let expected = json!({"inner_radius_m":0.05,"outer_radius_m":0.10,"length_m":0.05,
        "internal_pressure_pa":1e6,"external_pressure_pa":0.0,"youngs_modulus_pa":200e9,
        "poisson_ratio":0.3,"radial_elements":32,"axial_elements":2,"n2d3d":1,
        "element":"QUAD","plane_strain":true});
    if golden["model"] != expected {
        return Err("axisymmetric deck model differs from native comparison inputs".into());
    }
    for name in ["axisym_cylinder_0000.rad", "axisym_cylinder_0001.rad"] {
        if golden["deck_sha256"][name] != sha256(&root.join(name))? {
            return Err(format!("axisymmetric deck checksum mismatch: {name}"));
        }
    }
    if golden["history_sha256"] != sha256(&root.join("history.csv"))?
        || golden["engine_log_sha256"] != sha256(&root.join("engine-output.txt"))?
        || !std::fs::read_to_string(root.join("engine-output.txt"))
            .map_err(|e| e.to_string())?
            .contains("NORMAL TERMINATION")
    {
        return Err("axisymmetric harvested runtime artifacts do not match provenance".into());
    }
    let csv = std::fs::read_to_string(root.join("history.csv")).map_err(|e| e.to_string())?;
    let mut lines = csv.lines();
    let header: Vec<_> = lines
        .next()
        .ok_or("empty reference CSV")?
        .split(',')
        .map(|s| s.trim().trim_matches('"'))
        .collect();
    let find = |node: &str| -> Result<usize, String> {
        let matches: Vec<_> = header
            .iter()
            .enumerate()
            .filter(|(_, name)| {
                name.split_whitespace().any(|word| word == node) && name.trim().ends_with("DY")
            })
            .map(|(i, _)| i)
            .collect();
        if matches.len() != 1 {
            return Err(format!("ambiguous DY channel for node {node}"));
        }
        Ok(matches[0])
    };
    let columns = [find("34")?, find("66")?];
    let energy_column = header
        .iter()
        .position(|name| *name == "INTERNAL ENERGY")
        .ok_or("missing reference energy")?;
    let kinetic_column = header
        .iter()
        .position(|name| *name == "KINETIC ENERGY")
        .ok_or("missing reference kinetic energy")?;
    let mut samples = Vec::new();
    let mut energies = Vec::new();
    let mut previous_time = f64::NEG_INFINITY;
    for line in lines {
        let row: Vec<f64> = line
            .split(',')
            .map(|s| s.parse::<f64>().map_err(|e| e.to_string()))
            .collect::<Result<_, _>>()?;
        if row.len() != header.len() || row.iter().any(|v| !v.is_finite()) {
            return Err("invalid reference history row".into());
        }
        if row[0] <= previous_time {
            return Err("nonmonotone reference history".into());
        }
        previous_time = row[0];
        if row[0] >= scalar(&golden, "hold_start_s")? {
            samples.push([row[columns[0]], row[columns[1]]]);
            if row[energy_column] <= 0.0 || row[kinetic_column] < 0.0 {
                return Err("invalid reference hold energy".into());
            }
            energies.push([row[energy_column], row[kinetic_column]]);
        }
    }
    if samples.len() < 50 || golden["hold_samples"] != samples.len() {
        return Err("reference pressure hold sample count mismatch".into());
    }
    let means: [f64; 2] =
        std::array::from_fn(|i| samples.iter().map(|s| s[i]).sum::<f64>() / samples.len() as f64);
    let reference_energy = energies.iter().map(|e| e[0]).sum::<f64>() / energies.len() as f64;
    let kinetic_ratio = energies.iter().map(|e| e[1]).fold(0.0, f64::max)
        / energies.iter().map(|e| e[0]).fold(f64::INFINITY, f64::min);
    let mut hold_variation: f64 = 0.0;
    for i in 0..2 {
        if means[i] <= 0.0 {
            return Err("reference displacement is not pressure driven".into());
        }
        for sample in &samples {
            hold_variation = hold_variation.max((sample[i] - means[i]).abs() / means[i]);
        }
    }
    let mut runs = Vec::new();
    for n in [16, 32, 64] {
        let study = AxisymmetricCylinderStudy {
            geometry: AxisymmetricCylinderGeometry {
                inner_radius_m: 0.05,
                outer_radius_m: 0.10,
                length_m: 0.05,
                radial_elements: n,
                axial_elements: n,
            },
            material: AxisymmetricCylinderMaterial {
                youngs_modulus_pa: 200e9,
                poisson_ratio: 0.3,
            },
            internal_pressure_pa: 1e6,
            external_pressure_pa: 0.0,
        };
        runs.push(run_axisymmetric_cylinder(&study).map_err(|e| e.to_string())?);
    }
    let fine = &runs[2];
    let mut result=QaResult::passed("fem.axisymmetric-cylinder-harvest.v1")
        .formulation("native meridional triangular P1 axisymmetric elasticity; natural bore pressure, free radial unknowns and constrained axial plane strain")
        .dimension("r-z cylinder section, 16/32/64 radial and matching axial refinement; engine QUAD/N2D3D=1 fixture")
        .reference("independent OpenCourant quasistatic pressure-hold displacement/energy and Lame closed-form fields; no transient-ramp claim")
        .reference_checksum(sha256(&root.join("golden.json"))?)
        .metric("numerical_solve_executed",1.0).minimum("numerical_solve_executed",1.0)
        .metric("native_solves",3.0).minimum("native_solves",3.0)
        .metric("hold_samples",samples.len() as f64).minimum("hold_samples",50.0);
    for (name, value, bound) in [
        (
            "inner_engine_displacement_error",
            (fine.inner_radial_displacement_m - means[0]).abs() / means[0],
            0.005,
        ),
        (
            "outer_engine_displacement_error",
            (fine.outer_radial_displacement_m - means[1]).abs() / means[1],
            0.005,
        ),
        (
            "engine_energy_error",
            (fine.strain_energy_j - std::f64::consts::TAU * reference_energy).abs()
                / fine.strain_energy_j,
            0.005,
        ),
        ("reference_hold_variation", hold_variation, 0.001),
        ("reference_kinetic_energy_ratio", kinetic_ratio, 1e-6),
        (
            "fine_lame_displacement_error",
            fine.max_radial_displacement_relative_error,
            0.001,
        ),
        (
            "fine_lame_radial_stress_error",
            fine.max_radial_stress_relative_error,
            0.05,
        ),
        (
            "fine_lame_hoop_stress_error",
            fine.max_hoop_stress_relative_error,
            0.05,
        ),
        (
            "free_force_equilibrium",
            runs.iter()
                .map(|r| {
                    r.free_equilibrium_residual_n / (1e6 * std::f64::consts::TAU * 0.05 * 0.05)
                })
                .fold(0.0, f64::max),
            1e-8,
        ),
        (
            "native_energy_identity",
            runs.iter()
                .map(|r| r.energy_balance_relative_error)
                .fold(0.0, f64::max),
            1e-9,
        ),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}={value} exceeds {bound}"));
        }
        result = result.metric(name, value).maximum(name, bound);
    }
    let rate = runs
        .windows(2)
        .map(|p| {
            (p[0].max_radial_displacement_relative_error
                / p[1].max_radial_displacement_relative_error)
                .ln()
                / 2f64.ln()
        })
        .fold(f64::INFINITY, f64::min);
    if !rate.is_finite() || rate < 1.5 {
        return Err(format!("axisymmetric displacement refinement rate={rate}"));
    }
    Ok(result
        .metric("displacement_convergence_rate", rate)
        .minimum("displacement_convergence_rate", 1.5))
}
