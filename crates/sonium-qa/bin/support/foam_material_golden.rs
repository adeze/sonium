//! Native LAW51 replay from independently imposed volume histories.
//!
//! The original PG-degenerated decks use identical gas and JWL equations
//! of state. Their initial energy is per reference volume, not specific
//! energy. All conversions below are from g/mm/ms deck units to SI.
//! This is a material-point gate; no spatial foam solve is claimed.
// Rust guideline compliant 2026-02-21

use serde_json::Value;
use sha2::{Digest, Sha256};
use sonium_fem::physics::solid::plasticity::law_foams::{JwlEos, Multimat51, MultimatState};
use sonium_qa::QaResult;
use std::path::Path;

const DENSITY: f64 = 1.204e6;
const ENERGY: f64 = 250000.0e6 / DENSITY;
const VOLUME: f64 = 1e-6;
const SUBSTEPS: usize = 4096;

fn sha256(path: &Path) -> Result<String, String> {
    Ok(format!(
        "{:x}",
        Sha256::digest(std::fs::read(path).map_err(|e| e.to_string())?)
    ))
}
fn series(doc: &Value, key: &str) -> Result<Vec<f64>, String> {
    doc["channels"][key]["values"]
        .as_array()
        .ok_or_else(|| format!("missing LAW51 channel {key}"))?
        .iter()
        .map(|v| {
            v.as_f64()
                .filter(|v| v.is_finite())
                .ok_or_else(|| format!("nonfinite LAW51 channel {key}"))
        })
        .collect()
}
// Each opposite face follows +/-0.1 t mm on a 10 mm cube.
// This independently prescribed geometry, rather than harvested density
// or pressure, drives native material evolution at every substep.
fn mu(time_ms: f64) -> f64 {
    (1.0 + 0.02 * time_ms).powi(-3) - 1.0
}

pub(super) fn run_law51() -> Result<QaResult, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("goldens/miniqa/LOIS/LOI51/JWL/01-PG_degenerated");
    let mut result=QaResult::passed("fem.foam-law51-material-replay.v1")
        .formulation("native Multimat51/JwlEos prescribed-volume material evolution, adiabatic energy integration; no spatial solve")
        .dimension("two original hydrostatic PG-degenerated brick histories; SI material inputs")
        .reference("independent engine stress/density/energy histories; volume input reconstructed solely from deck imposed displacement")
        .reference_checksum("checked artifact SHA-256 provenance")
        .metric("numerical_solve_executed",0.0).maximum("numerical_solve_executed",0.0)
        .metric("native_material_evolution",1.0).minimum("native_material_evolution",1.0);
    let mut worst_stress = 0.0_f64;
    let mut worst_energy = 0.0_f64;
    let mut worst_density = 0.0_f64;
    let mut exact_energy_error = 0.0_f64;
    let mut golden_energy_band_usage = 0.0_f64;
    let mut engine_exact_energy_band_usage = 0.0_f64;
    let mut frozen_energy_gap = 0.0_f64;
    let mut stamps = 0;
    let mut native_steps = 0;
    for (name, external_pressure) in [("01a_P-E.json", 0.0), ("01b_dP-E.json", 1e11)] {
        let path = root.join(name);
        let doc: Value =
            serde_json::from_str(&std::fs::read_to_string(&path).map_err(|e| e.to_string())?)
                .map_err(|e| e.to_string())?;
        if doc["schema"] != "opencourant-golden-v1"
            || doc["meta"]["normal_termination"] != true
            || doc["meta"]["deck_units"] != serde_json::json!(["g", "mm", "ms"])
        {
            return Err("LAW51 golden provenance mismatch".into());
        }
        let times: Vec<f64> = doc["time"]
            .as_array()
            .ok_or("missing LAW51 times")?
            .iter()
            .map(|v| {
                v.as_f64()
                    .filter(|v| v.is_finite())
                    .ok_or("nonfinite LAW51 time".to_string())
            })
            .collect::<Result<_, _>>()?;
        let stress: Vec<Vec<f64>> = (2..=7)
            .map(|k| series(&doc, &format!("brick 2 brick_2 ~{k}")))
            .collect::<Result<_, _>>()?;
        let density = series(&doc, "brick 2 brick_2 ~9")?;
        let energy = series(&doc, "INTERNAL ENERGY")?;
        if times.len() < 40
            || times[0] != 0.0
            || times.windows(2).any(|w| w[1] <= w[0])
            || stress
                .iter()
                .chain([&density, &energy])
                .any(|s| s.len() != times.len())
        {
            return Err("invalid LAW51 replay grid".into());
        }
        // The two 0.5-volume constituents have the same PG EOS: omega=0.4.
        // Initial E0=250000 MPa is energy/reference-volume, so e0=E0/rho0.
        let eos = JwlEos::new(1e-24, 1e-24, 1.0, 1.0, 0.4, DENSITY, ENERGY)?;
        let law = Multimat51::new(eos, eos, external_pressure, -1e20, 0.0)?;
        let mut state = MultimatState::new(ENERGY, ENERGY);
        let mut previous = 0.0;
        for (i, time) in times.iter().copied().enumerate() {
            if i > 0 {
                for sub in 1..=SUBSTEPS {
                    let left = previous + (time - previous) * (sub - 1) as f64 / SUBSTEPS as f64;
                    let right = previous + (time - previous) * sub as f64 / SUBSTEPS as f64;
                    let increment = -(mu(right) - mu(left)) / 3.0;
                    law.step(
                        &mut state,
                        &[increment, increment, increment, 0.0, 0.0, 0.0],
                        (time - previous) * 1e-3 / SUBSTEPS as f64,
                    );
                    native_steps += 1;
                }
            }
            // Read stress at the accepted energy state; a zero-volume step
            // does not add work and removes the explicit output's one-step lag.
            let update = law.step(&mut state, &[0.0; 6], 0.0);
            for (component, channel) in update.stress.iter().zip(&stress) {
                worst_stress = worst_stress.max((component - channel[i] * 1e6).abs() / 1e11);
            }
            let native_density = DENSITY * (1.0 + state.mu);
            worst_density =
                worst_density.max((native_density - density[i] * 1e6).abs() / (density[i] * 1e6));
            let native_energy = state.energy_r * DENSITY * VOLUME;
            worst_energy =
                worst_energy.max((native_energy - energy[i] * 1e-3).abs() / (energy[i] * 1e-3));
            let exact_energy = ENERGY * DENSITY * VOLUME * (1.0 + mu(time)).powf(0.4);
            exact_energy_error =
                exact_energy_error.max((native_energy - exact_energy).abs() / exact_energy);
            // The archived engine integration has a 0.0565% error against
            // the exact adiabatic solution. Respect its existing, independently
            // recorded 0.066% TH comparison contract, while retaining the
            // tighter 0.02% native integration accuracy bound below.
            if doc["channels"]["INTERNAL ENERGY"]["rel_tol"].as_f64() != Some(0.00066)
                || doc["channels"]["INTERNAL ENERGY"]["abs_tol"].as_f64() != Some(0.00025)
            {
                return Err("LAW51 engine energy comparison contract changed".into());
            }
            let allowance = 0.00025 * 1e-3 + 0.00066 * energy[i] * 1e-3;
            golden_energy_band_usage =
                golden_energy_band_usage.max((native_energy - energy[i] * 1e-3).abs() / allowance);
            engine_exact_energy_band_usage = engine_exact_energy_band_usage
                .max((exact_energy - energy[i] * 1e-3).abs() / allowance);
            if i + 1 == times.len() {
                let frozen = eos.pressure(state.mu, ENERGY) - external_pressure;
                frozen_energy_gap =
                    frozen_energy_gap.max((frozen + stress[0][i] * 1e6).abs() / 1e11);
            }
            previous = time;
            stamps += 1;
        }
        result = result.artifact_with_provenance(
            format!("goldens/miniqa/LOIS/LOI51/JWL/01-PG_degenerated/{name}"),
            "original LAW51 engine deck hydrostatic expansion",
            "harvested TH stresses/density/energy; native replay inputs come from deck geometry",
            "opencourant-golden-v1; engine commit 33e685176cccf0c539a3ce07aa2096985a284e2a",
            sha256(&path)?,
        );
    }
    for (name, value, bound) in [
        ("stress_history_error", worst_stress, 2e-4),
        ("native_exact_energy_error", exact_energy_error, 2e-4),
        ("golden_energy_band_usage", golden_energy_band_usage, 1.0),
        (
            "engine_exact_energy_band_usage",
            engine_exact_energy_band_usage,
            1.0,
        ),
        ("density_history_error", worst_density, 2e-5),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}={value} exceeds {bound}"));
        }
        result = result.metric(name, value).maximum(name, bound);
    }
    if frozen_energy_gap < 0.05 {
        return Err("LAW51 energy-coupling control is insensitive".into());
    }
    Ok(result
        .metric("energy_history_relative_error", worst_energy)
        .metric("frozen_energy_control_gap", frozen_energy_gap)
        .minimum("frozen_energy_control_gap", 0.05)
        .metric("native_stamps", stamps as f64)
        .minimum("native_stamps", 80.0)
        .metric("native_substeps", native_steps as f64)
        .minimum("native_substeps", 300000.0))
}
