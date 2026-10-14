//! Measure full original nonlinear external roof mesh convergence.
//!
//! Retains the original QEPH formulation, gravity ramp, physical geometry,
//! mass and free-rotation diaphragm supports. This is independent reference
//! refinement evidence, not native parity or published linear accuracy.
//! An optional `law36` argument tests the same elastic constants through a
//! different material implementation with unreachable yield. Its plastic work
//! must remain zero; this diagnostic does not replace the original LAW1 gate.
//! An optional third argument selects QBAT12 instead of the original QEPH24.
//! Both retain the same physical inputs and full nonlinear loading history.
//! `P2_SCORDELIS_REFERENCE_FULL_PROFILE=1` observes all nodal displacements.

// Rust guideline compliant 2026-02-21

#[path = "../bin/support/scordelis_native.rs"]
pub mod native;

use sha2::{Digest, Sha256};
use sonium_qa::rad::{
    RadExtendedCase, RadMaterial, RadPlasticMaterial, RadTabulatedHardening, engine_exe,
    run_extended_case, write_extended_engine, write_extended_starter,
};

fn reference_case(
    cells: usize,
    law: &str,
    shell: i32,
) -> Result<RadExtendedCase, Box<dyn std::error::Error>> {
    if ![12, 24].contains(&shell) {
        return Err("supported independent shell formulations:QBAT12 or QEPH24".into());
    }
    let mut case = native::refined_case(cells)?;
    case.shell_prop
        .as_mut()
        .ok_or("original shell property missing")?
        .ishell = shell;
    match law {
        "law1" => {}
        "law36" => {
            let m = case.base.material;
            // Far above any elastic roof stress. Runtime plastic work must
            // independently remain zero before this comparison is accepted.
            case.material = RadMaterial::Plastic(RadPlasticMaterial::Tabulated(
                RadTabulatedHardening::perfectly_plastic(
                    m.youngs_modulus_pa,
                    m.poisson_ratio,
                    m.density_kg_per_m3,
                    1e20,
                ),
            ));
        }
        _ => return Err("supported reference material representations:law1 or law36".into()),
    }
    case.validate()?;
    Ok(case)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cells: usize = std::env::args().nth(1).map_or(Ok(64), |v| v.parse())?;
    if !(32..=256).contains(&cells) || !cells.is_multiple_of(2) {
        return Err("even external roof refinement in[32,256] required".into());
    }
    let law = std::env::args().nth(2).unwrap_or_else(|| "law1".into());
    let shell: i32 = std::env::args().nth(3).map_or(Ok(24), |v| v.parse())?;
    let mut case = reference_case(cells, &law, shell)?;
    let probe_node = case.base.th_nodes[0];
    let full_profile = match std::env::var("P2_SCORDELIS_REFERENCE_FULL_PROFILE").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(_) => false,
        Ok(_) => return Err("P2_SCORDELIS_REFERENCE_FULL_PROFILE must be0 or1".into()),
    };
    if full_profile {
        case.base.th_nodes = (1..=case.base.mesh.num_nodes()).collect();
    }
    let original = sonium_qa::rad::shell_scordelis_lo_case();
    let starter = write_extended_starter(&case);
    let engine = write_extended_engine(&case);
    if engine != write_extended_engine(&original) {
        return Err(
            "reference refinement changed original engine duration/loading/output controls".into(),
        );
    }
    let work = std::env::temp_dir().join(format!(
        "p2_scordelis_shell_reference_{}_{}_{}_{}",
        cells,
        shell,
        law,
        std::process::id()
    ));
    std::fs::create_dir_all(&work)?;
    std::fs::write(work.join(format!("{}_0000.rad", case.base.name)), &starter)?;
    std::fs::write(work.join(format!("{}_0001.rad", case.base.name)), &engine)?;
    println!("Nonlinear roof{cells}/shell{shell}/{law}: full8s reference diagnostic starting");
    let output = run_extended_case(&work, &engine_exe().ok_or("OpenCourant required")?, &case)?;
    let values = output.history.channel(&case.base.probes[0].channel)?;
    let hold = output
        .history
        .times
        .iter()
        .zip(values)
        .filter(|(t, _)| **t >= 5.5)
        .map(|(_, &v)| v)
        .collect::<Vec<_>>();
    if hold.is_empty() || hold.iter().any(|v| !v.is_finite()) {
        return Err("complete finite original roof hold observations required".into());
    }
    let mean = hold.iter().sum::<f64>() / hold.len() as f64;
    let plastic_values = output.history.channel("PLASTIC WORK")?;
    if plastic_values.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite elastic material diagnostic plastic-work channel".into());
    }
    let plastic_work = plastic_values
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0, f64::max);
    if !plastic_work.is_finite() || plastic_work > 1e-12 {
        return Err("elastic roof material comparison developed plastic work".into());
    }
    let profile = if full_profile {
        let mut means = Vec::with_capacity(case.base.th_nodes.len());
        let mut positions = Vec::with_capacity(case.base.th_nodes.len());
        for &id in &case.base.th_nodes {
            let mut displacement = [0.0; 3];
            let mut position = case.base.mesh.nodes[id - 1];
            for (axis, tag) in ["DX", "DY", "DZ"].iter().enumerate() {
                let values = output.history.channel(&format!("{tag}/{id}"))?;
                let hold = output
                    .history
                    .times
                    .iter()
                    .zip(values)
                    .filter(|(time, _)| **time >= 5.5)
                    .map(|(_, value)| *value)
                    .collect::<Vec<_>>();
                if hold.is_empty() || values.iter().any(|v| !v.is_finite()) {
                    return Err("complete finite full roof profile channels required".into());
                }
                displacement[axis] = hold.iter().sum::<f64>() / hold.len() as f64;
                position[axis] += values.last().ok_or("missing final roof position")?;
            }
            means.push(displacement);
            positions.push(position);
        }
        serde_json::json!({"node_ids":case.base.th_nodes,
            "mean_hold_displacements_m":means,"final_positions_m":positions,
            "profile_sample_time_s":output.history.times.last(),
            "scope":"Actual complete reference nodal observations;hold mean uses5.5s..8s;final position at actual last reference sample. Native matching-time/full-field parity remains unproven."})
    } else {
        serde_json::Value::Null
    };
    let mut channels = output.history.channels;
    if full_profile {
        let id = probe_node.to_string();
        // Keep the original probe and physical energy histories in the packet;
        // the full spatial measurements are exported once in the profile.
        channels.retain(|name, _| {
            !["DX/", "DY/", "DZ/"]
                .iter()
                .any(|prefix| name.starts_with(prefix))
                || name.rsplit('/').next() == Some(id.as_str())
        });
    }
    let measured = serde_json::json!({
        "status":"complete independent nonlinear reference; native parity and linear accuracy unproven",
        "surface_cells":cells,"engine":output.engine,
        "material_representation":law,"maximum_plastic_work_j":plastic_work,
        "shell_formulation":shell,
        "history":{"times_s":output.history.times,"channels":channels},
        "full_profile_enabled":full_profile,"full_profile":profile,
        "hold_mean_edge_dy_m":mean,"physical_stop_time_s":case.base.stop_time_s,
        "starter_sha256":format!("{:x}",Sha256::digest(starter.as_bytes())),
        "engine_sha256":format!("{:x}",Sha256::digest(engine.as_bytes())),
        "work_directory":work,"scope":"Full nonlinear roof,original physical geometry,E/nu/rho/t,gravity ramp5s/hold to8s,XY diaphragms with free rotations. DefaultQEPH24/LAW1 retains original material/formulation;optionalQBAT12 andLAW36 are explicit numerical formulation/material diagnostics. LAW36 uses unreachable yield and requires zero measured plastic work. No native/reference or published linear accuracy admission."
    });
    std::fs::write(
        std::env::var("P2_SCORDELIS_REFERENCE_MESH_OUTPUT")?,
        serde_json::to_vec_pretty(&measured)?,
    )?;
    println!(
        "Independent full nonlinear roof{cells}/shell{shell}/{law}:hold DY={mean:.12e};diagnostic only"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elastic_material_diagnostic_retains_original_geometry_loading_and_constants() {
        let old = native::refined_case(32).unwrap();
        let original = reference_case(32, "law1", 24).unwrap();
        assert_eq!(
            write_extended_starter(&old),
            write_extended_starter(&original)
        );
        let alternate = reference_case(32, "law36", 24).unwrap();
        assert_eq!(
            write_extended_engine(&old),
            write_extended_engine(&alternate)
        );
        assert_eq!(old.base.mesh.nodes, alternate.base.mesh.nodes);
        assert_eq!(old.shells.quads, alternate.shells.quads);
        let RadMaterial::Plastic(RadPlasticMaterial::Tabulated(m)) = alternate.material else {
            panic!("expected explicit alternate material representation");
        };
        assert_eq!(m.youngs_modulus_pa, old.base.material.youngs_modulus_pa);
        assert_eq!(m.poisson_ratio, old.base.material.poisson_ratio);
        assert_eq!(m.density_kg_per_m3, old.base.material.density_kg_per_m3);
        assert!(m.points.iter().all(|(_, yield_pa)| *yield_pa == 1e20));
        assert_eq!(
            alternate.shell_prop.unwrap().ismstr,
            old.shell_prop.as_ref().unwrap().ismstr
        );
        assert!(reference_case(32, "unknown", 24).is_err());
        let qbat = reference_case(32, "law36", 12).unwrap();
        assert_eq!(qbat.shell_prop.as_ref().unwrap().ishell, 12);
        assert_eq!(write_extended_engine(&qbat), write_extended_engine(&old));
        assert_eq!(qbat.base.mesh.nodes, old.base.mesh.nodes);
        assert!(reference_case(32, "law36", 999).is_err());
    }
}
