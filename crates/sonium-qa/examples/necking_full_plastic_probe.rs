//! Measure full three-dimensional plastic tension without claiming native parity.
//!
//! Preserves the original nominal bar and one-second duration, with explicit
//! Voce hardening,one-percent area imperfection and a pulling tip grip. The
//! original elastic leg is separate. Actual plastic strain and the complete
//! final shape are exported; neither proves a converged necking gate alone.

// Rust guideline compliant 2026-02-21

#[path = "../bin/fem_opencourant_necking_bar.rs"]
#[expect(
    dead_code,
    reason = "Only the original fixture and complete plastic extension are used"
)]
mod original;

#[path = "../bin/oc_bench.rs"]
#[expect(
    dead_code,
    reason = "Uses only the full native plastic tensile history API"
)]
mod native;

use original::necking_full_fixture::{NAME, full_case};
use sha2::{Digest, Sha256};
use sonium_qa::rad::{
    bric_channel, engine_exe, probes_json, run_extended_case, write_extended_engine,
    write_extended_starter,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let factor: usize = std::env::args().nth(1).map_or(Ok(1), |v| v.parse())?;
    let case = full_case(factor)?;
    let native_mode = std::env::var("P2_NECKING_MODE").as_deref() == Ok("native");
    let formulation =
        std::env::var("SONIUM_NECKING_REFERENCE_SOLID").unwrap_or_else(|_| "heph24".to_owned());
    if native_mode && formulation != "heph24" {
        return Err("reference solid formulation cannot select a native integration mode".into());
    }
    let starter = reference_starter(&write_extended_starter(&case), &formulation)?;
    let work =
        std::env::temp_dir().join(format!("p2_necking_full_{factor}_{}", std::process::id()));
    std::fs::create_dir_all(&work)?;
    let engine = write_extended_engine(&case);
    std::fs::write(work.join(format!("{NAME}_0000.rad")), &starter)?;
    std::fs::write(work.join(format!("{NAME}_0001.rad")), &engine)?;
    std::fs::write(
        work.join("probes.json"),
        probes_json(&case.base, "diagnostic only"),
    )?;
    if native_mode {
        let scale: f64 = std::env::args().nth(2).map_or(Ok(1.0), |v| v.parse())?;
        let profile_time = std::env::var("P2_NECKING_PROFILE_TIME_S")
            .ok()
            .map(|v| v.parse())
            .transpose()?;
        let result = native::necking_plastic_history(&work, scale, profile_time)?;
        std::fs::write(
            std::env::var("P2_NECKING_OUTPUT")?,
            serde_json::to_vec_pretty(&result)?,
        )?;
        println!("Complete full3D native plastic tensile diagnostic;original QA gate unproven");
        return Ok(());
    }
    println!(
        "Full3D plastic tensile bar:factor{factor},stop1s,original nominal geometry/E/nu/rho;diagnostic starting"
    );
    let output = run_extended_case(&work, &engine_exe().ok_or("OpenCourant required")?, &case)?;
    // Match profiles using actual adjacent reference observations,without
    // changing physical steps or the complete one-second trajectory.
    let profile_time = std::env::var("P2_NECKING_PROFILE_TIME_S")
        .ok()
        .map(|v| v.parse::<f64>())
        .transpose()?
        .or_else(|| output.history.times.last().copied())
        .ok_or("missing complete tensile reference history")?;
    if !profile_time.is_finite() || profile_time <= 0.0 {
        return Err("finite positive tensile profile time required".into());
    }
    let positions = case
        .base
        .th_nodes
        .iter()
        .map(|&id| {
            let mut p = case.base.mesh.nodes[id - 1];
            for (axis, tag) in ["DX", "DY", "DZ"].iter().enumerate() {
                p[axis] += output
                    .history
                    .value_at(&format!("{tag}/{id}"), profile_time)?;
            }
            Ok(p)
        })
        .collect::<Result<Vec<_>, sonium_qa::rad::RadError>>()?;
    let plastic = (1..=case.base.mesh.bricks.len())
        .map(|id| {
            output
                .history
                .value_at(&bric_channel("PLAS", id), profile_time)
        })
        .collect::<Result<Vec<_>, _>>()?;
    if positions
        .iter()
        .flatten()
        .chain(&plastic)
        .any(|v| !v.is_finite())
    {
        return Err("nonfinite full plastic bar reference profile".into());
    }
    let measured = serde_json::json!({"status":"complete external plastic tension diagnostic;native parity,necking and mesh convergence unproven",
        "refinement_factor":factor,"engine":output.engine,"work_directory":work,
        "reference_solid_formulation":formulation,
        "physical_stop_time_s":case.base.stop_time_s,"profile_sample_time_s":profile_time,
        "history":{"times_s":output.history.times,"channels":output.history.channels},
        "node_ids":case.base.th_nodes,"final_positions_m":positions,"final_element_plastic_strain":plastic,
        "starter_sha256":format!("{:x}",Sha256::digest(starter.as_bytes())),
        "engine_deck_sha256":format!("{:x}",Sha256::digest(engine.as_bytes())),
        "scope":"Original nominalL.1m/b.01m/E200GPa/nu.3/rho7800/rootclamp/full1s;explicit1%area imperfection,250MPa+300MPa*(1-exp(-5ep))Voce curve,tip-only frictionless pulling plane and transverse grip clamp. Gauge section contracts freely. Original subyield elastic20kN leg retained separately. No erosion,mass scaling or measured-alloy claim."
    });
    std::fs::write(
        std::env::var("P2_NECKING_OUTPUT")?,
        serde_json::to_vec_pretty(&measured)?,
    )?;
    println!("Complete full3D plastic tensile reference exported;original native gate unproven");
    Ok(())
}

/// Change only the reference brick integration formulation.
///
/// H8C17 uses fixed eight-point integration, constant-pressure default 1,
/// and non-corotational frame default 1. HEPH24 retains the original writer
/// output. Neither selection admits native parity or mesh convergence.
fn reference_starter(
    starter: &str,
    formulation: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    match formulation {
        "heph24" => Ok(starter.to_owned()),
        "h8c17" => {
            let mut lines = starter.lines().map(str::to_owned).collect::<Vec<_>>();
            let property_rows = lines
                .iter()
                .enumerate()
                .filter(|(_, line)| line.as_str() == "/PROP/SOLID/1/1")
                .map(|(index, _)| index + 2)
                .collect::<Vec<_>>();
            let [row] = property_rows.as_slice() else {
                return Err("exactly one original solid property required".into());
            };
            if lines
                .get(*row)
                .ok_or("original solid property row missing")?
                .split_whitespace()
                .collect::<Vec<_>>()
                != ["24", "2"]
            {
                return Err("original HEPH formulation changed".into());
            }
            // Preserve Ismstr2 and every physical, loading, and output field.
            lines[*row] = format!("{:>10}{:>10}", 17, 2);
            Ok(lines.join("\n") + "\n")
        }
        _ => Err("SONIUM_NECKING_REFERENCE_SOLID must be heph24 or h8c17".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integration_control_preserves_original_physical_deck() {
        for factor in [1, 2] {
            let case = full_case(factor).unwrap();
            let original = write_extended_starter(&case);
            assert_eq!(reference_starter(&original, "heph24").unwrap(), original);
            let alternate = reference_starter(&original, "h8c17").unwrap();
            let before = original.lines().collect::<Vec<_>>();
            let after = alternate.lines().collect::<Vec<_>>();
            assert_eq!(before.len(), after.len());
            let changed = before
                .iter()
                .zip(&after)
                .filter(|(a, b)| a != b)
                .collect::<Vec<_>>();
            assert_eq!(changed.len(), 1);
            assert_eq!(
                changed[0].0.split_whitespace().collect::<Vec<_>>(),
                ["24", "2"]
            );
            assert_eq!(
                changed[0].1.split_whitespace().collect::<Vec<_>>(),
                ["17", "2"]
            );
            assert!(reference_starter(&alternate, "h8c17").is_err());
        }
    }

    #[test]
    fn integration_control_rejects_unknown_or_ambiguous_properties() {
        let starter = write_extended_starter(&full_case(1).unwrap());
        assert!(reference_starter(&starter, "automatic").is_err());
        assert!(reference_starter("/PROP/SOLID/1/1\nTitle\n", "h8c17").is_err());
        assert!(reference_starter(&format!("{starter}{starter}"), "h8c17").is_err());
    }
}
