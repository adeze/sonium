//! Measure roof formulation and load sensitivity through the full original history.
//!
//! Reduced-load measurements diagnose the linear limit only. They cannot prove
//! the original full-load QA gate. Shell flag meanings follow
//! https://2025.help.altair.com/2025/hwsolvers/rad/topics/solvers/rad/prop_type1_shell_starter_r.htm .

// Rust guideline compliant 2026-02-21

use sha2::{Digest, Sha256};
use sonium_qa::rad::{
    RadExtendedCase, RadMaterial, RadPlasticMaterial, RadTabulatedHardening, engine_exe,
    run_extended_case, shell_scordelis_lo_case, write_extended_engine, write_extended_starter,
};

fn diagnostic_case(
    shell: i32,
    strain: i32,
    load_scale: f64,
) -> Result<RadExtendedCase, Box<dyn std::error::Error>> {
    if ![(24, 2), (24, 1), (2, 3)].contains(&(shell, strain))
        || !load_scale.is_finite()
        || !(0.0 < load_scale && load_scale <= 1.0)
    {
        return Err(
            "supported roof formulations are24/2,24/1,2/3; load scale must be in(0,1]".into(),
        );
    }
    let mut case = shell_scordelis_lo_case();
    let prop = case
        .shell_prop
        .as_mut()
        .ok_or("original shell property missing")?;
    prop.ishell = shell;
    prop.ismstr = strain;
    if shell == 2 {
        // LAW1 forces global N=0 integration, which is incompatible with Q4
        // formulation2. LAW36 retains the original linear elastic response
        // with an unreachable yield stress and five thickness points.
        let material = case.base.material;
        case.material = RadMaterial::Plastic(RadPlasticMaterial::Tabulated(
            RadTabulatedHardening::perfectly_plastic(
                material.youngs_modulus_pa,
                material.poisson_ratio,
                material.density_kg_per_m3,
                1e20,
            ),
        ));
    }
    case.gravity
        .as_mut()
        .ok_or("original gravity missing")?
        .value_m_per_s2 *= load_scale;
    case.validate()?;
    Ok(case)
}

fn diagnostic_engine(
    case: &RadExtendedCase,
    implicit_dt: Option<f64>,
) -> Result<String, Box<dyn std::error::Error>> {
    let mut deck = write_extended_engine(case);
    if let Some(dt) = implicit_dt {
        let prop = case
            .shell_prop
            .as_ref()
            .ok_or("original shell property missing")?;
        if !dt.is_finite()
            || !(1e-4..=0.01).contains(&dt)
            || prop.ishell != 24
            || prop.ismstr != 2
            || case
                .gravity
                .as_ref()
                .ok_or("original gravity missing")?
                .value_m_per_s2
                != -9.81
        {
            return Err(
                "implicit diagnostic requires original full gravity,QEPH24/2,dt in[1e-4,.01]"
                    .into(),
            );
        }
        // Match the native linear benchmark while retaining actual inertia,
        // original full load/ramp/hold and complete8s integration. Newmark's
        // average-acceleration rule has no artificial damping. SMDISP uses
        // the original configuration; Jacobi-preconditioned CG solves each
        // increment with a relative original-matrix residual tolerance.
        // https://help.altair.com/hwsolvers/rad/topics/solvers/rad/impl_nonlin_smdisp_engine_r.htm
        deck.push_str(&format!(
            "/IMPL/NONLIN/1\n        20         2   1.000000000000E-8\n/IMPL/NONLIN/SMDISP\n/IMPL/SOLVER/1\n         2         0         1   1.00000000000E-10\n/IMPL/DYNA/2\n   5.000000000000E-1   2.500000000000E-1\n/IMPL/DTINI\n{dt:>20.12E}\n/IMPL/DT/STOP\n{dt:>20.12E}{dt:>20.12E}\n"
        ));
    } else if case.shell_prop.as_ref().is_some_and(|p| p.ishell == 2) {
        // No minimum timestep, mass scaling or stiffness modification.
        deck.push_str("/DT/NODA\n   9.000000000000E-1    0.000000000000E0\n");
    }
    Ok(deck)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = std::env::args().skip(1).collect::<Vec<_>>();
    if args.len() != 3 {
        return Err("requires shell flag, strain flag, and gravity scale".into());
    }
    let shell: i32 = args[0].parse()?;
    let strain: i32 = args[1].parse()?;
    let scale: f64 = args[2].parse()?;
    let case = diagnostic_case(shell, strain, scale)?;
    let work = std::env::temp_dir().join(format!("p2_scordelis_kinematics_{}", std::process::id()));
    let engine = engine_exe().ok_or("OpenCourant required")?;
    let starter = write_extended_starter(&case);
    let implicit_dt = std::env::var("P2_SCORDELIS_IMPLICIT_DT_S")
        .ok()
        .map(|v| v.parse())
        .transpose()?;
    let deck = diagnostic_engine(&case, implicit_dt)?;
    std::fs::create_dir_all(&work)?;
    std::fs::write(work.join(format!("{}_0000.rad", case.base.name)), &starter)?;
    std::fs::write(work.join(format!("{}_0001.rad", case.base.name)), &deck)?;
    println!(
        "Original32x32 roof, shell={shell}/strain={strain}, gravity scale={scale}: full8s reference starting"
    );
    let output = run_extended_case(&work, &engine, &case)?;
    let displacement = output.history.channel(&case.base.probes[0].channel)?;
    let hold = output
        .history
        .times
        .iter()
        .zip(displacement)
        .filter(|(t, _)| **t >= 5.5)
        .map(|(_, v)| *v)
        .collect::<Vec<_>>();
    if hold.is_empty() || hold.iter().any(|v| !v.is_finite()) {
        return Err("missing finite original hold-window measurements".into());
    }
    let mean = hold.iter().sum::<f64>() / hold.len() as f64;
    let normalized = mean / scale;
    let measured = serde_json::json!({
        "status":"diagnostic measurement; original QA parity unproven",
        "shell_formulation":shell,"strain_flag":strain,"gravity_scale":scale,
        "hold_mean_edge_dy_m":mean,"load_normalized_hold_mean_edge_dy_m":normalized,
        "normalized_macneal_relative_error":(normalized+0.3024).abs()/0.3024,
        "times_s":output.history.times,"edge_dy_m":displacement,
        "work_directory":work,"engine":output.engine,
        "starter_sha256":format!("{:x}",Sha256::digest(starter.as_bytes())),
        "engine_deck_sha256":format!("{:x}",Sha256::digest(deck.as_bytes())),
        "elastic_material_representation":if shell==2{"LAW36; original E/nu/density; unreachable1e20Pa yield; five thickness points"}else{"originalLAW1"},
        "timestep_criterion":if shell==2{"nodal; factor.9; no mass scaling or stiffness modification"}else{"original element criterion"},
        "implicit_small_displacement_dt_s":implicit_dt,
        "scope":"Original geometry,32x32 mesh,E/nu/density/thickness,diaphragms,gravity ramp and full8s stop retained; controlled formulation and gravity-scale diagnostic. Reduced-load results cannot close the original full-load finding."
    });
    std::fs::write(
        std::env::var("P2_SCORDELIS_KINEMATICS_OUTPUT")?,
        serde_json::to_vec_pretty(&measured)?,
    )?;
    println!("Measured hold DY={mean:.12e}; normalized={normalized:.12e}; diagnostic only");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn implicit_controls_preserve_complete_load_history_and_reject_easier_loads() {
        let case = diagnostic_case(24, 2, 1.0).unwrap();
        let original = write_extended_engine(&case);
        let controlled = diagnostic_engine(&case, Some(0.002)).unwrap();
        assert!(controlled.starts_with(&original));
        assert!(controlled.contains("/IMPL/NONLIN/SMDISP\n"));
        assert!(controlled.contains("/IMPL/DYNA/2\n"));
        assert!(!controlled.contains("/IMPL/LINEAR"));
        assert_eq!(diagnostic_engine(&case, None).unwrap(), original);
        for dt in [0.0, f64::NAN, f64::INFINITY, 0.02] {
            assert!(diagnostic_engine(&case, Some(dt)).is_err());
        }
        assert!(diagnostic_engine(&diagnostic_case(24, 2, 0.01).unwrap(), Some(0.002)).is_err());
        assert!(diagnostic_engine(&diagnostic_case(2, 3, 1.0).unwrap(), Some(0.002)).is_err());
    }

    #[test]
    fn controlled_variants_preserve_original_physics_and_complete_history() {
        let original = shell_scordelis_lo_case();
        let original_deck = write_extended_starter(&original);
        assert_eq!(
            write_extended_starter(&diagnostic_case(24, 2, 1.0).unwrap()),
            original_deck
        );
        for (shell, strain, scale) in [(24, 1, 1.0), (2, 3, 1.0), (24, 2, 0.01)] {
            let case = diagnostic_case(shell, strain, scale).unwrap();
            assert_eq!(
                write_extended_engine(&case),
                write_extended_engine(&original)
            );
            let mut restored = case;
            restored.shell_prop = original.shell_prop.clone();
            restored.gravity = original.gravity.clone();
            if shell == 2 {
                let RadMaterial::Plastic(RadPlasticMaterial::Tabulated(material)) =
                    &restored.material
                else {
                    panic!("Q4 requires compatible thickness integration");
                };
                assert_eq!(
                    material.youngs_modulus_pa,
                    original.base.material.youngs_modulus_pa
                );
                assert_eq!(material.poisson_ratio, original.base.material.poisson_ratio);
                assert_eq!(
                    material.density_kg_per_m3,
                    original.base.material.density_kg_per_m3
                );
                assert_eq!(material.points, [(0.0, 1e20), (0.5, 1e20)]);
                restored.material = original.material.clone();
            }
            assert_eq!(write_extended_starter(&restored), original_deck);
        }
        for (shell, strain, scale) in [
            (24, 3, 1.0),
            (1, 1, 1.0),
            (24, 2, 0.0),
            (24, 2, f64::NAN),
            (24, 2, 1.1),
        ] {
            assert!(diagnostic_case(shell, strain, scale).is_err());
        }
    }
}
