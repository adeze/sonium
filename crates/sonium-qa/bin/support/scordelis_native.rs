//! Full native gravity histories of the original Scordelis-Lo roof.

// Rust guideline compliant 2026-02-21

#[path = "shell_transient.rs"]
mod transient;

use sonium_qa::rad::{
    RadExtendedCase, RadMesh, scordelis_lo_shell_mesh, shel_channel, shell_scordelis_lo_case,
};

fn refined_case(cells: usize) -> Result<RadExtendedCase, Box<dyn std::error::Error>> {
    if cells < 2 || !cells.is_multiple_of(2) {
        return Err("even roof refinement >=2 required".into());
    }
    let mut case = shell_scordelis_lo_case();
    let mesh = scordelis_lo_shell_mesh(cells, cells);
    case.base.mesh = RadMesh {
        nodes: mesh.nodes,
        bricks: Vec::new(),
    };
    case.shells.quads = mesh.quads;
    case.base.node_sets[0].nodes = mesh.end_lo;
    case.base.node_sets[1].nodes = mesh.end_hi;
    case.base.th_nodes = vec![mesh.probe_node];
    case.base.probes[0].channel = format!("DY/{}", mesh.probe_node);
    case.base.probes[1].channel = shel_channel("THIC", mesh.probe_shell);
    case.th_shell.elems = vec![mesh.probe_shell];
    case.validate()?;
    Ok(case)
}

/// Measure the original complete roof history without claiming external parity.
///
/// # Errors
///
/// Returns an error for invalid refinement/timestep, failed native solve,
/// or missing finite samples in the original 5.5-8 s hold window.
pub fn probe(cells: usize, dt: f64) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let case = refined_case(cells)?;
    let node = case.base.th_nodes[0];
    let history = transient::run(&case, dt, &[(node - 1) * 6 + 1], false)?;
    let values = history
        .observations
        .iter()
        .map(|v| v[0])
        .collect::<Vec<_>>();
    let samples = history
        .times
        .iter()
        .zip(&values)
        .filter(|(time, _)| **time >= 5.5)
        .map(|(_, v)| *v)
        .collect::<Vec<_>>();
    if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
        return Err("missing finite original roof hold samples".into());
    }
    let mean = samples.iter().sum::<f64>() / samples.len() as f64;
    let result = serde_json::json!({"scope":"Original full roof radius25m, length50m, 80deg span, thickness.25m, original diaphragm supports and mass-proportional gravity ramp/hold to8s; diagnostic only, no external parity or QA success",
        "cells":cells,"dofs":history.dofs,"steps":history.steps,"dt":history.dt,
        "hold_mean_edge_dy_m":mean,"independent_macneal_target_m":-0.3024,
        "relative_macneal_error":(mean+0.3024).abs()/0.3024,
        "external_regression_baseline_m":case.base.probes[0].expected,
        "thickness_m":case.shell_prop.as_ref().unwrap().thickness_m,
        "max_true_dynamic_residual":history.max_true_dynamic_residual,"times_s":history.times,"edge_dy_m":values});
    println!(
        "native original roof {cells}x{cells}: hold mean DY={mean:.12e}, MacNeal error={:.6e}, equation residual={:.6e}",
        (mean + 0.3024).abs() / 0.3024,
        history.max_true_dynamic_residual
    );
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refined_roof_retains_original_supports_material_gravity_and_probe_coordinates() {
        let old = shell_scordelis_lo_case();
        let new = refined_case(48).unwrap();
        assert_eq!(
            sonium_qa::rad::write_extended_engine(&old),
            sonium_qa::rad::write_extended_engine(&new)
        );
        assert_eq!(
            old.base.material.youngs_modulus_pa,
            new.base.material.youngs_modulus_pa
        );
        assert_eq!(
            old.base.material.poisson_ratio,
            new.base.material.poisson_ratio
        );
        assert_eq!(
            old.base.material.density_kg_per_m3,
            new.base.material.density_kg_per_m3
        );
        let a = old.gravity.unwrap();
        let b = new.gravity.unwrap();
        assert_eq!(a.value_m_per_s2, b.value_m_per_s2);
        assert_eq!(a.direction, b.direction);
        assert_eq!(a.function, b.function);
        assert_eq!(old.base.functions[0].points, new.base.functions[0].points);
        for index in 0..2 {
            assert_eq!(old.base.bcs[index].lock, new.base.bcs[index].lock);
        }
        assert_eq!(
            old.base.mesh.nodes[old.base.th_nodes[0] - 1],
            new.base.mesh.nodes[new.base.th_nodes[0] - 1]
        );
        assert!(refined_case(3).is_err());
    }
}
