//! Measure full copper Taylor mushrooming without asserting QA parity.
//!
//! Retains the original geometry, copper elastic properties, impact velocity,
//! free bar, and frictionless moving anvil. Adds isothermal copper Johnson-Cook
//! plasticity and follows deformation beyond repeated release-wave transits.
//! Copper flow parameters: Banerjee, Table 4,
//! https://arxiv.org/html/cond-mat/0512466v1 . This is a matched-model numerical
//! investigation, not a replay of that paper's different experimental geometry
//! or a claim to validate adiabatic heating and thermal softening.

// Rust guideline compliant 2026-02-21

#[path = "../bin/fem_opencourant_taylor_bar.rs"]
#[expect(dead_code, reason = "Only the original physical fixture is used")]
mod original;

use sonium_qa::rad::{bric_channel, engine_exe, run_extended_case, write_extended_engine};
use std::path::PathBuf;

use original::taylor_full_fixture::{NAME, full_case, write_decks};
use original::taylor_native as native;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let work = PathBuf::from(std::env::var("P2_TAYLOR_WORK")?);
    let destination = PathBuf::from(std::env::var("P2_TAYLOR_OUTPUT")?);
    let factor: usize = std::env::var("P2_TAYLOR_MESH_FACTOR").map_or(Ok(1), |v| v.parse())?;
    if !(1..=4).contains(&factor) {
        return Err("Taylor mesh factor must be in 1..=4".into());
    }
    let case = full_case(factor);
    case.validate()?;
    let viscous = match std::env::var("P2_TAYLOR_REFERENCE_FORMULATION").as_deref() {
        Ok("viscous") => true,
        Ok("heph") | Err(_) => false,
        Ok(_) => return Err("P2_TAYLOR_REFERENCE_FORMULATION must be heph or viscous".into()),
    };
    if viscous && std::env::var("P2_TAYLOR_MODE").as_deref() != Ok("external") {
        return Err("viscous reference selector applies only to external diagnostics; native coefficient matching unproven".into());
    }
    write_decks(&case, &work, viscous)?;
    if std::env::var("P2_TAYLOR_MODE").as_deref() == Ok("external") {
        use sha2::{Digest, Sha256};
        let exe = engine_exe().ok_or("OpenCourant required for full plastic reference")?;
        let output = run_extended_case(&work, &exe, &case)?;
        let node_ids = &case.base.th_nodes;
        let final_positions = node_ids
            .iter()
            .map(|&id| {
                let mut p = case.base.mesh.nodes[id - 1];
                for (axis, tag) in ["DX", "DY", "DZ"].iter().enumerate() {
                    p[axis] += output.history.final_value(&format!("{tag}/{id}"))?;
                }
                Ok(p)
            })
            .collect::<Result<Vec<_>, sonium_qa::rad::RadError>>()?;
        let plastic = (1..=case.base.mesh.bricks.len())
            .map(|id| output.history.final_value(&bric_channel("PLAS", id)))
            .collect::<Result<Vec<_>, _>>()?;
        let channels = case
            .base
            .probes
            .iter()
            .map(|p| p.channel.clone())
            .collect::<Vec<_>>();
        let values = channels
            .iter()
            .map(|c| output.history.channel(c).map(|v| v.to_vec()))
            .collect::<Result<Vec<_>, _>>()?;
        let measured = serde_json::json!({
            "status":"full plastic external measurement; native parity unproven",
            "engine":output.engine, "work_directory":work,
            "solid_formulation":if viscous {1} else {24},
            "hourglass_coefficient_match":"unproven; formulation diagnostic only",
            "times_s":output.history.times, "probe_channels":channels, "probe_values":values,
            "node_ids":node_ids, "final_positions_m":final_positions,
            "final_element_plastic_strain":plastic,
            "starter_sha256":format!("{:x}",Sha256::digest(std::fs::read(work.join(format!("{NAME}_0000.rad")))?)),
            "engine_deck_sha256":format!("{:x}",Sha256::digest(write_extended_engine(&case).as_bytes())),
        });
        std::fs::write(destination, serde_json::to_vec_pretty(&measured)?)?;
    } else {
        let scale = std::env::args().nth(1).map_or(Ok(1.0), |v| v.parse())?;
        let penalty: f64 =
            std::env::var("P2_TAYLOR_CONTACT_SCALE").map_or(Ok(1.0), |v| v.parse())?;
        let objective = match std::env::var("P2_TAYLOR_OBJECTIVE").as_deref() {
            Ok("0") => false,
            Ok("1") | Err(_) => true,
            Ok(_) => return Err("P2_TAYLOR_OBJECTIVE must be 0 or 1".into()),
        };
        println!("native Taylor objective stress update enabled: {objective}");
        let profile_time = std::env::var("P2_TAYLOR_PROFILE_TIME_S")
            .ok()
            .map(|v| v.parse())
            .transpose()?;
        let measured =
            native::taylor_plastic_history_sampled(&work, scale, penalty, objective, profile_time)?;
        std::fs::write(destination, serde_json::to_vec_pretty(&measured)?)?;
    }
    println!("Full plastic Taylor measurement completed; diagnostic, QA parity unproven");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use original::taylor_full_fixture::{isothermal_starter, viscous_starter};
    use sonium_qa::rad::{
        RadExtendedCase, RadMaterial, RadPlasticMaterial, write_extended_starter,
    };
    use std::path::Path;

    #[test]
    fn viscous_reference_changes_only_the_element_formulation() {
        let starter = isothermal_starter(&full_case(2)).unwrap();
        let viscous = viscous_starter(&starter).unwrap();
        let original = starter.lines().collect::<Vec<_>>();
        let rendered = viscous.lines().collect::<Vec<_>>();
        assert_eq!(original.len(), rendered.len());
        let changed = original
            .iter()
            .zip(&rendered)
            .enumerate()
            .filter(|(_, (a, b))| a != b)
            .collect::<Vec<_>>();
        assert_eq!(changed.len(), 1);
        let row = changed[0].0;
        assert_eq!(original[row - 2], "/PROP/SOLID/1/1");
        assert_eq!(
            rendered[row].split_whitespace().collect::<Vec<_>>(),
            ["1", "2"]
        );
        assert!(viscous_starter(&viscous).is_err());
    }

    #[test]
    fn full_plastic_fixture_retains_original_impact_physics() {
        let original = original::taylor_bar_case();
        let case = full_case(1);
        case.validate().unwrap();
        assert_eq!(case.base.mesh.nodes, original.mesh.nodes);
        assert_eq!(case.base.mesh.bricks, original.mesh.bricks);
        assert_eq!(case.base.walls[0].normal, original.walls[0].normal);
        assert_eq!(
            case.base.walls[0].velocity_m_per_s,
            original.walls[0].velocity_m_per_s
        );
        assert_eq!(case.base.walls[0].friction, 0.0);
        assert!(case.base.bcs.is_empty());
        assert!(case.base.stop_time_s > 2.0 * 0.0254 / (117e9_f64 / 8930.0).sqrt());
        let RadMaterial::Plastic(RadPlasticMaterial::JohnsonCook(material)) = case.material else {
            panic!("copper plastic material required");
        };
        assert_eq!(material.density_kg_per_m3, 8930.0);
        assert_eq!(material.youngs_modulus_pa, 117e9);
        assert_eq!(material.poisson_ratio, 0.35);
        assert_eq!(material.fail_plastic_strain, 1e20);
        assert!(material.yield_pa < material.youngs_modulus_pa * 0.01);
    }

    #[test]
    fn isothermal_rendering_changes_only_heat_capacity() {
        let case = full_case(1);
        let original = write_extended_starter(&case);
        let isothermal = isothermal_starter(&case).unwrap();
        let original = original.lines().collect::<Vec<_>>();
        let isothermal = isothermal.lines().collect::<Vec<_>>();
        assert_eq!(original.len(), isothermal.len());
        let row = original.iter().position(|s| *s == "/MAT/LAW2/1/1").unwrap() + 6;
        for (index, (a, b)) in original.iter().zip(&isothermal).enumerate() {
            if index == row {
                let a = a.split_whitespace().collect::<Vec<_>>();
                let b = b.split_whitespace().collect::<Vec<_>>();
                assert_eq!(b[2].parse::<f64>().unwrap(), 0.0);
                for index in [0, 1, 3, 4] {
                    assert_eq!(a[index], b[index]);
                }
            } else {
                assert_eq!(a, b);
            }
        }
    }

    #[test]
    fn invalid_profile_time_is_rejected_before_reading_decks() {
        for time in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            assert!(
                native::taylor_plastic_history_sampled(
                    Path::new("missing"),
                    1.0,
                    1.0,
                    true,
                    Some(time)
                )
                .unwrap_err()
                .contains("sample time")
            );
        }
    }

    #[test]
    fn invalid_cfl_scale_is_rejected_before_reading_decks() {
        for scale in [0.0, -1.0, 2.0, f64::NAN, f64::INFINITY] {
            assert!(
                native::taylor_plastic_history_sampled(
                    Path::new("missing"),
                    scale,
                    1.0,
                    true,
                    None
                )
                .is_err()
            );
        }
        for penalty in [0.0, -1.0, 0.5, f64::NAN, f64::INFINITY] {
            assert!(
                native::taylor_plastic_history_sampled(
                    Path::new("missing"),
                    1.0,
                    penalty,
                    true,
                    None
                )
                .is_err()
            );
        }
    }

    #[test]
    fn refinement_preserves_copper_and_contact_setup() {
        let original = full_case(1);
        let fine = full_case(2);
        fine.validate().unwrap();
        assert!(fine.base.mesh.bricks.len() > original.base.mesh.bricks.len());
        assert_eq!(
            fine.base.walls[0].velocity_m_per_s,
            original.base.walls[0].velocity_m_per_s
        );
        assert_eq!(
            fine.base.walls[0].secondary_set,
            original.base.walls[0].secondary_set
        );
        assert_eq!(
            fine.base.functions[0].points,
            original.base.functions[0].points
        );
        assert_eq!(fine.base.stop_time_s, original.base.stop_time_s);
        let z_extent = |case: &RadExtendedCase| {
            let zs = case.base.node_sets[1]
                .nodes
                .iter()
                .map(|&id| case.base.mesh.nodes[id - 1][2]);
            zs.clone().fold(f64::NEG_INFINITY, f64::max) - zs.fold(f64::INFINITY, f64::min)
        };
        assert!((z_extent(&fine) - 0.0254).abs() < 1e-14);
        assert!((z_extent(&original) - z_extent(&fine)).abs() < 1e-14);
        let driver = fine.base.walls[0].driver_node.unwrap();
        assert!(!fine.base.node_sets[1].nodes.contains(&driver));
    }
}
