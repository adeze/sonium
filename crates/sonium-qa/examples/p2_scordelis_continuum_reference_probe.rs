//! Diagnose a full-thickness external continuum reference for the original roof.

// Rust guideline compliant 2026-02-21

use sha2::{Digest, Sha256};
use sonium_qa::rad::{
    RadExtendedCase, RadMesh, engine_exe, run_extended_case, shell_scordelis_lo_case,
    write_extended_engine, write_extended_starter,
};

fn continuum_case(
    cells: usize,
    radial: usize,
) -> Result<RadExtendedCase, Box<dyn std::error::Error>> {
    if cells < 2 || radial < 2 || !cells.is_multiple_of(2) || !radial.is_multiple_of(2) {
        return Err("even roof and thickness cells >=2 required".into());
    }
    let original = shell_scordelis_lo_case();
    let thickness = original.shell_prop.as_ref().unwrap().thickness_m;
    let radius = 25.0;
    let length = 50.0;
    let half_angle = 40.0_f64.to_radians();
    let id = |z: usize, t: usize, r: usize| (z * (cells + 1) + t) * (radial + 1) + r + 1;
    let mut nodes = Vec::new();
    for z in 0..=cells {
        for t in 0..=cells {
            for r in 0..=radial {
                let theta = -half_angle + 2.0 * half_angle * t as f64 / cells as f64;
                let rho = radius + thickness * (r as f64 / radial as f64 - 0.5);
                nodes.push([
                    rho * theta.sin(),
                    rho * theta.cos(),
                    length * z as f64 / cells as f64,
                ]);
            }
        }
    }
    let mut bricks = Vec::new();
    for z in 0..cells {
        for t in 0..cells {
            for r in 0..radial {
                bricks.push([
                    id(z, t, r),
                    id(z + 1, t, r),
                    id(z + 1, t + 1, r),
                    id(z, t + 1, r),
                    id(z, t, r + 1),
                    id(z + 1, t, r + 1),
                    id(z + 1, t + 1, r + 1),
                    id(z, t + 1, r + 1),
                ]);
            }
        }
    }
    let mut base = original.base;
    base.name = "scordelis_lo_continuum".into();
    base.title = "Original Scordelis-Lo full-thickness continuum reference".into();
    base.mesh = RadMesh { nodes, bricks };
    // Fix only midsurface X/Y at the original diaphragms. Outer-layer motion
    // retains shell rotations; clamping all layers would change the benchmark.
    base.node_sets[0].nodes = (0..=cells).map(|t| id(0, t, radial / 2)).collect();
    base.node_sets[1].nodes = (0..=cells).map(|t| id(cells, t, radial / 2)).collect();
    let probe = id(cells / 2, cells, radial / 2);
    base.th_nodes = vec![probe];
    base.probes.truncate(1);
    base.probes[0].channel = format!("DY/{probe}");
    base.probes[0].expected = -0.3024;
    base.probes[0].rel_tol = 0.01;
    let mut case = RadExtendedCase::wrap_elastic(base);
    case.gravity = original.gravity;
    case.validate()?;
    Ok(case)
}

// The standard writer uses reduced-integration HEPH (24). Permit documented
// fully integrated, shear-locking-free formulations for this curved thin roof.
// https://2023.help.altair.com/2023.1/hwsolvers/rad/topics/solvers/rad/prop_type14_solid_starter_r.htm
fn reference_starter(
    case: &RadExtendedCase,
    solid: usize,
) -> Result<String, Box<dyn std::error::Error>> {
    if ![14, 18, 24].contains(&solid) {
        return Err("supported roof solid formulations are14,18,24".into());
    }
    let deck = write_extended_starter(case);
    let original = "/PROP/SOLID/1/1\nelastic brick property\n        24         2\n";
    if deck.matches(original).count() != 1 {
        return Err("standard solid property writer changed".into());
    }
    let replacement = format!(
        "/PROP/SOLID/1/1\nelastic brick property\n{solid:>10}{:>10}\n",
        2
    );
    Ok(deck.replacen(original, &replacement, 1))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = engine_exe().ok_or("OpenCourant required for the continuum reference study")?;
    let output = std::env::var("P2_SCORDELIS_REFERENCE_OUTPUT")?;
    let solid = std::env::var("P2_SCORDELIS_SOLID").map_or(Ok(24), |v| v.parse::<usize>())?;
    let mut levels = Vec::new();
    let mut values = Vec::new();
    for (cells, radial) in [(12, 2), (24, 4), (32, 6)] {
        let case = continuum_case(cells, radial)?;
        let work = std::env::temp_dir().join(format!(
            "p2_scordelis_continuum_{}_{}_{}",
            std::process::id(),
            cells,
            radial
        ));
        std::fs::create_dir_all(&work)?;
        let starter = reference_starter(&case, solid)?;
        let deck = write_extended_engine(&case);
        std::fs::write(work.join("scordelis_lo_continuum_0000.rad"), &starter)?;
        std::fs::write(work.join("scordelis_lo_continuum_0001.rad"), &deck)?;
        println!(
            "external full-thickness roof {cells}x{cells}x{radial}, solid formulation={solid}: original complete 8 s gravity history starting"
        );
        let result = run_extended_case(&work, &engine, &case)?;
        let trace = result.history.channel(&case.base.probes[0].channel)?;
        let hold = result
            .history
            .times
            .iter()
            .zip(trace)
            .filter(|(t, _)| **t >= 5.5)
            .map(|(_, v)| *v)
            .collect::<Vec<_>>();
        if hold.is_empty() || hold.iter().any(|v| !v.is_finite()) {
            return Err("missing finite original roof hold window".into());
        }
        let mean = hold.iter().sum::<f64>() / hold.len() as f64;
        println!(
            "external continuum roof {cells}x{cells}x{radial}: hold DY={mean:.12e}, MacNeal error={:.6e}",
            (mean + 0.3024).abs() / 0.3024
        );
        values.push(mean);
        levels.push(
            serde_json::json!({"surface_cells":cells,"thickness_cells":radial,"solid_formulation":solid,
            "hold_mean_edge_dy_m":mean,"relative_macneal_error":(mean+0.3024).abs()/0.3024,
            "work_directory":work,"engine":result.engine,
            "starter_sha256":format!("{:x}",Sha256::digest(starter.as_bytes())),
            "engine_deck_sha256":format!("{:x}",Sha256::digest(deck.as_bytes()))}),
        );
        std::fs::write(
            &output,
            serde_json::to_vec_pretty(&serde_json::json!({"status":"incomplete","levels":levels,
            "scope":"Original roof geometry/thickness/material/gravity/history and neutral-plane diaphragm translational supports; external continuum mesh refinement, not original QA/native parity success"}))?,
        )?;
    }
    let first = (values[1] - values[0]).abs() / values[1].abs();
    let last = (values[2] - values[1]).abs() / values[2].abs();
    let contraction = last / first.max(1e-15);
    let target = (values[2] + 0.3024).abs() / 0.3024;
    let passed = first <= 0.02 && last <= 0.005 && contraction <= 0.6 && target <= 0.01;
    std::fs::write(
        output,
        serde_json::to_vec_pretty(
            &serde_json::json!({"status":if passed{"passed"}else{"failed"},
        "levels":levels,"first_relative_change":first,"final_relative_change":last,
        "contraction":contraction,"fine_macneal_relative_error":target,
        "budgets":{"first_relative_change":0.02,"final_relative_change":0.005,"contraction":0.6,"fine_macneal_error":0.01},
        "scope":"Actual external full-thickness reference, not original QA success or native parity"}),
        )?,
    )?;
    if !passed {
        return Err("full-thickness roof reference accuracy/convergence failed".into());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn continuum_roof_preserves_physics_volume_and_rotational_support_freedom() {
        let original = shell_scordelis_lo_case();
        let case = continuum_case(12, 2).unwrap();
        assert_eq!(
            original.base.material.youngs_modulus_pa,
            case.base.material.youngs_modulus_pa
        );
        assert_eq!(
            original.base.material.poisson_ratio,
            case.base.material.poisson_ratio
        );
        assert_eq!(
            original.base.material.density_kg_per_m3,
            case.base.material.density_kg_per_m3
        );
        assert_eq!(
            original.base.functions[0].points,
            case.base.functions[0].points
        );
        assert_eq!(original.base.stop_time_s, case.base.stop_time_s);
        assert_eq!(original.base.th_period_s, case.base.th_period_s);
        assert_eq!(
            original.gravity.unwrap().value_m_per_s2,
            case.gravity.unwrap().value_m_per_s2
        );
        for set in &case.base.node_sets {
            assert_eq!(set.nodes.len(), 13);
            assert!(set.nodes.iter().all(|&n| {
                (case.base.mesh.nodes[n - 1][0].hypot(case.base.mesh.nodes[n - 1][1]) - 25.0).abs()
                    < 1e-12
            }));
        }
        assert_eq!(
            original.base.mesh.nodes[original.base.th_nodes[0] - 1],
            case.base.mesh.nodes[case.base.th_nodes[0] - 1]
        );
        let mut volume = 0.0;
        for brick in &case.base.mesh.bricks {
            for local in [
                [0, 1, 2, 6],
                [0, 2, 3, 6],
                [0, 3, 7, 6],
                [0, 7, 4, 6],
                [0, 4, 5, 6],
                [0, 5, 1, 6],
            ] {
                let p = local.map(|i| case.base.mesh.nodes[brick[i] - 1]);
                let a = std::array::from_fn::<_, 3, _>(|i| p[1][i] - p[0][i]);
                let b = std::array::from_fn::<_, 3, _>(|i| p[2][i] - p[0][i]);
                let c = std::array::from_fn::<_, 3, _>(|i| p[3][i] - p[0][i]);
                let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]);
                assert!(det > 0.0);
                volume += det / 6.0;
            }
        }
        let expected = 50.0 * 12.0 * 25.0 * 0.25 * (80.0_f64.to_radians() / 12.0).sin();
        assert!((volume - expected).abs() < 1e-12 * expected);
        assert!(continuum_case(3, 2).is_err());
        assert!(continuum_case(12, 3).is_err());
    }
    #[test]
    fn formulation_variant_changes_only_the_numerical_solid_property() {
        let case = continuum_case(12, 2).unwrap();
        let original = write_extended_starter(&case);
        assert_eq!(reference_starter(&case, 24).unwrap(), original);
        for solid in [14, 18] {
            let changed = reference_starter(&case, solid).unwrap();
            let property = format!("elastic brick property\n{solid:>10}{:>10}\n", 2);
            assert_eq!(
                changed.replace(&property, "elastic brick property\n        24         2\n"),
                original
            );
        }
        assert!(reference_starter(&case, 17).is_err());
    }
}
