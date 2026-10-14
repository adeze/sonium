//! Measure the original full warping transient with objective elastic solids.

// Rust guideline compliant 2026-02-21

#[path = "../bin/fem_opencourant_beam_warping.rs"]
#[expect(dead_code, reason = "Uses the original warping case factory")]
mod reference;

#[path = "../bin/oc_bench.rs"]
#[expect(dead_code, reason = "Uses the objective elastic solid history only")]
mod native;

use sonium_qa::rad::{OpenCourantCase, write_engine, write_starter};

fn observation_probes(case: &OpenCourantCase) -> serde_json::Value {
    let mut channels = std::collections::BTreeSet::new();
    for probe in &case.probes {
        for channel in probe.channel.split('-') {
            channels.insert(channel.to_owned());
        }
    }
    serde_json::json!({
        "scope": "Individual displacement observations only; no reference or QA admission",
        "probes": channels.into_iter().map(|channel| serde_json::json!({
            "label": channel, "channel": channel,
            // This diagnostic records physical traces. Independent parity
            // and convergence are evaluated after completion.
            "expected": 0.0, "abs_tol": f64::MAX, "rel_tol": 0.0,
        })).collect::<Vec<_>>()
    })
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() > 3 {
        return Err("provide axial cells, section factor and CFL scale".into());
    }
    let axial = args.first().map_or(Ok(20), |value| value.parse())?;
    let section = args.get(1).map_or(Ok(1), |value| value.parse())?;
    let cfl = args.get(2).map_or(Ok(1.0), |value| value.parse())?;
    let case = reference::refined_reference_case(axial, section)?;
    let output = std::env::var("SONIUM_WARPING_OBJECTIVE_OUTPUT")?;
    if std::path::Path::new(&output).exists() {
        return Err("existing physical observations must not be overwritten".into());
    }
    let work = std::env::temp_dir().join(format!(
        "warping_objective_{}_{}_{}",
        std::process::id(),
        axial,
        section
    ));
    std::fs::create_dir_all(&work)?;
    std::fs::write(work.join("beam_warping_0000.rad"), write_starter(&case))?;
    std::fs::write(work.join("beam_warping_0001.rad"), write_engine(&case))?;
    std::fs::write(
        work.join("probes.json"),
        serde_json::to_vec_pretty(&observation_probes(&case))?,
    )?;
    let history = native::warping_elastic_history(&work, cfl)?;
    let result = serde_json::json!({
        "status": "completed objective elastic diagnostic; original QA gate unproven",
        "axial_cells": axial, "section_factor": section, "timestep_scale": cfl,
        "physical_stop_time_s": case.stop_time_s, "work_directory": work,
        "native": history,
        "scope": "Original I-section, E/nu/rho, full root restraints, original single-node force couple and complete 0.5 s ramp/hold. Objective eight-point B-bar elastic integration, no plasticity/erosion/mass scaling. Reference parity and h/time convergence remain required."
    });
    std::fs::write(output, serde_json::to_vec_pretty(&result)?)?;
    println!("Completed original full objective warping diagnostic; QA admission unproven");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn observations_cover_both_original_pairs_without_changing_physics() {
        let case = reference::refined_reference_case(40, 2).unwrap();
        let starter = write_starter(&case);
        let engine = write_engine(&case);
        let value = observation_probes(&case);
        let probes = value["probes"].as_array().unwrap();
        assert_eq!(probes.len(), 4);
        for original in &case.probes {
            for channel in original.channel.split('-') {
                assert_eq!(probes.iter().filter(|p| p["channel"] == channel).count(), 1);
            }
        }
        assert_eq!(starter, write_starter(&case));
        assert_eq!(engine, write_engine(&case));
        assert_eq!(case.stop_time_s, 0.5);
        assert_eq!(case.material.youngs_modulus_pa, 200e9);
        assert_eq!(case.material.poisson_ratio, 0.3);
        assert_eq!(case.material.density_kg_per_m3, 7800.0);
        assert_eq!(case.cloads.len(), 2);
        assert_eq!(case.node_sets[case.cloads[0].set].nodes.len(), 1);
        assert_eq!(case.node_sets[case.cloads[1].set].nodes.len(), 1);
    }
}
