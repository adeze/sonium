//! Matching full native histories for the original MacNeal-Harder external fixture.

// Rust guideline compliant 2026-02-21

#[path = "shell_transient.rs"]
mod transient;

use sonium_qa::rad::{RadExtendedCase, RadMesh};

fn refined_case(nx: usize, ny: usize) -> Result<RadExtendedCase, Box<dyn std::error::Error>> {
    if nx < 2 || ny < 2 || !ny.is_multiple_of(2) {
        return Err("positive axial cells and even width cells >=2 required".into());
    }
    let mut case = super::shell_twisted_beam_case();
    let mesh = super::twisted_beam_mesh(nx, ny);
    case.base.mesh = RadMesh {
        nodes: mesh.nodes,
        bricks: Vec::new(),
    };
    case.base.node_sets[0].nodes = mesh.root;
    case.base.node_sets[1].nodes = vec![mesh.tip_center];
    case.base.th_nodes = vec![mesh.tip_center];
    case.base.probes[0].channel = format!("DZ/{}", mesh.tip_center);
    case.shells.quads = mesh.quads;
    case.validate()?;
    Ok(case)
}

pub(super) fn hold_mean(times: &[f64], values: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    if times.len() != values.len() {
        return Err("hold trace lengths differ".into());
    }
    let samples = times
        .iter()
        .zip(values)
        .filter(|(time, _)| **time >= 0.032)
        .map(|(_, v)| *v)
        .collect::<Vec<_>>();
    if samples.is_empty() || samples.iter().any(|v| !v.is_finite()) {
        return Err("missing finite original hold samples".into());
    }
    Ok(samples.iter().sum::<f64>() / samples.len() as f64)
}

/// Measure the full original native twisted-beam transient at a chosen resolution.
///
/// # Errors
///
/// Returns an error for invalid mesh/timestep, failed full shell solve, or missing
/// hold samples. Physical loads, supports, material and averaging window are fixed.
pub fn probe(
    nx: usize,
    ny: usize,
    dt: f64,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let case = refined_case(nx, ny)?;
    let tip = case.base.node_sets[1].nodes[0];
    let history = transient::run(&case, dt, &[(tip - 1) * 6 + 2], false)?;
    let values = history
        .observations
        .iter()
        .map(|v| v[0])
        .collect::<Vec<_>>();
    let mean = hold_mean(&history.times, &values)?;
    let result = serde_json::json!({"scope":"Original complete 90-degree twisted elastic shell, six DOFs per node, original 20 ms ramp/hold to 60 ms; diagnostic, not external parity or QA success",
        "axial_cells":nx,"width_cells":ny,"dofs":history.dofs,"steps":history.steps,"dt":history.dt,
        "hold_mean_tip_dz_m":mean,"macneal_target_m":case.base.probes[0].expected,
        "relative_target_error":(mean-case.base.probes[0].expected).abs()/case.base.probes[0].expected.abs(),
        "thickness_m":case.shell_prop.as_ref().unwrap().thickness_m,
        "max_true_dynamic_residual":history.max_true_dynamic_residual,
        "times_s":history.times,"tip_dz_m":values});
    println!(
        "native twisted-beam {nx}x{ny}: hold mean DZ={mean:.12e}, target error={:.6e}, max equation residual={:.6e}",
        (mean - case.base.probes[0].expected).abs() / case.base.probes[0].expected.abs(),
        history.max_true_dynamic_residual
    );
    Ok(result)
}

fn validate_original(case: &RadExtendedCase) -> Result<(), Box<dyn std::error::Error>> {
    let original = super::shell_twisted_beam_case();
    if sonium_qa::rad::write_extended_starter(case)
        != sonium_qa::rad::write_extended_starter(&original)
        || sonium_qa::rad::write_extended_engine(case)
            != sonium_qa::rad::write_extended_engine(&original)
        || sonium_qa::rad::extended_probes_json(case, "identity")
            != sonium_qa::rad::extended_probes_json(&original, "identity")
    {
        return Err("original twisted-shell fixture identity differs".into());
    }
    Ok(())
}

/// Compare full native twisted-shell histories with both original external probes.
///
/// # Errors
///
/// Returns an error for altered original fixture data, missing external values,
/// native transient failures or missing finite numerical fields.
pub(super) fn parity_result(
    case: &RadExtendedCase,
    case_id: &str,
    engine: &str,
    external: &[f64],
    reference_passed: bool,
) -> Result<sonium_qa::QaResult, Box<dyn std::error::Error>> {
    validate_original(case)?;
    if external.len() != 2 || external.iter().any(|v| !v.is_finite()) {
        return Err("both finite original external twisted-shell probes required".into());
    }
    let runs = [
        probe(32, 8, 0.000025)?,
        probe(48, 12, 0.000025)?,
        probe(64, 16, 0.000025)?,
        probe(64, 16, 0.0000125)?,
    ];
    build_result(case, case_id, engine, external, reference_passed, &runs)
}

fn number(run: &serde_json::Value, key: &str) -> Result<f64, Box<dyn std::error::Error>> {
    run.get(key)
        .and_then(serde_json::Value::as_f64)
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("missing finite twisted-shell field {key}").into())
}

fn build_result(
    case: &RadExtendedCase,
    case_id: &str,
    engine: &str,
    external: &[f64],
    reference_passed: bool,
    runs: &[serde_json::Value; 4],
) -> Result<sonium_qa::QaResult, Box<dyn std::error::Error>> {
    let values = runs
        .iter()
        .map(|r| number(r, "hold_mean_tip_dz_m"))
        .collect::<Result<Vec<_>, _>>()?;
    let first = (values[1] - values[0]).abs() / values[1].abs().max(1e-30);
    let last = (values[2] - values[1]).abs() / values[2].abs().max(1e-30);
    let thickness = number(&runs[2], "thickness_m")?;
    let mut result=sonium_qa::QaResult::passed(case_id)
        .requested_backend("cpu+opencourant").actual_backend("cpu+opencourant")
        .metric("external_reference_executed",1.0)
        .metric("external_reference_health_passed",if reference_passed {1.0}else{0.0})
        .minimum("external_reference_health_passed",1.0)
        .metric("native_parity_executed",1.0).minimum("native_parity_executed",1.0)
        .metric("native_full_shell_histories_executed",4.0).minimum("native_full_shell_histories_executed",4.0)
        .metric("native_hold_mean_tip_dz_m",values[2]).metric("external_hold_mean_tip_dz_m",external[0])
        .metric("native_external_relative_error",(values[2]-external[0]).abs()/external[0].abs().max(1e-30))
        .maximum("native_external_relative_error",0.03)
        .metric("native_macneal_relative_error",(values[2]-case.base.probes[0].expected).abs()/case.base.probes[0].expected.abs())
        .maximum("native_macneal_relative_error",0.01)
        .metric("spatial_first_relative_change",first).maximum("spatial_first_relative_change",0.02)
        .metric("spatial_final_relative_change",last).maximum("spatial_final_relative_change",0.005)
        .metric("spatial_contraction",last/first.max(1e-15)).maximum("spatial_contraction",0.6)
        .metric("temporal_relative_change",(values[2]-values[3]).abs()/values[3].abs().max(1e-30))
        .maximum("temporal_relative_change",0.003)
        .metric("native_thickness_m",thickness)
        .metric("thickness_external_relative_error",(thickness-external[1]).abs()/external[1].abs().max(1e-30))
        .maximum("thickness_external_relative_error",1e-6)
        .metric("thickness_analytical_relative_error",(thickness-case.base.probes[1].expected).abs()/case.base.probes[1].expected.abs())
        .maximum("thickness_analytical_relative_error",case.base.probes[1].rel_tol)
        .reference(engine)
        .dimension("Original 12 in by 1.1 in, .32 in thick 90-degree twisted shell; E=29e6 psi, nu=.22, rho=7800 kg/m3; all six root DOFs clamped; native 32x8,48x12,64x16 meshes")
        .formulation("Native covariant MITC4 shell with full six-DOF stiffness, translational and rotary inertia; undamped compensated checked Newmark; original 1 lbf tip-center +Z force, 20 ms ramp/hold to 60 ms; original 32-60 ms hold mean on external 0.1 ms sample grid; every-step independently recomputed physical force balance")
        .tolerance("native/external displacement <=3%, native MacNeal target <=1%; original external health <=5%; spatial changes <=2% then .5%, contraction <=.6; time change <=.3%; all physical equation residuals <=1e-8; thickness <=1e-6");
    for (index, run) in runs.iter().enumerate() {
        let key = format!("history_{index}_true_dynamic_residual");
        result = result
            .metric(&key, number(run, "max_true_dynamic_residual")?)
            .maximum(key, 1e-8);
    }
    if let Err(error) = result.release_eligible() {
        result.status = sonium_qa::QaStatus::Failed;
        result.message = Some(error);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn refined_case_preserves_original_physics_and_probe_coordinates() {
        let old = super::super::shell_twisted_beam_case();
        let new = refined_case(48, 12).unwrap();
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
        assert_eq!(
            old.shell_prop.as_ref().unwrap().thickness_m,
            new.shell_prop.as_ref().unwrap().thickness_m
        );
        assert_eq!(
            old.base.cloads[0].force_scale_n,
            new.base.cloads[0].force_scale_n
        );
        assert_eq!(old.base.cloads[0].direction, new.base.cloads[0].direction);
        assert_eq!(old.base.functions[0].points, new.base.functions[0].points);
        assert_eq!(old.base.bcs[0].lock, new.base.bcs[0].lock);
        let tip =
            |case: &RadExtendedCase| case.base.mesh.nodes[case.base.node_sets[1].nodes[0] - 1];
        assert_eq!(tip(&old), tip(&new));
        assert!(
            new.base.node_sets[0]
                .nodes
                .iter()
                .all(|&n| new.base.mesh.nodes[n - 1][0] == 0.0)
        );
        assert_eq!(
            new.base.node_sets[0].nodes.len(),
            new.base.mesh.nodes.iter().filter(|p| p[0] == 0.0).count()
        );
        assert!(refined_case(48, 3).is_err());
    }
    #[test]
    fn hold_mean_includes_original_window_and_rejects_invalid_samples() {
        assert_eq!(
            hold_mean(
                &[0.02, 0.0319, 0.032, 0.04, 0.06],
                &[100.0, 100.0, 2.0, 4.0, 6.0]
            )
            .unwrap(),
            4.0
        );
        assert!(hold_mean(&[0.02], &[1.0]).is_err());
        assert!(hold_mean(&[0.04], &[f64::NAN]).is_err());
        assert!(hold_mean(&[0.04], &[]).is_err());
    }
    #[test]
    fn original_support_or_load_changes_are_rejected() {
        let mut case = super::super::shell_twisted_beam_case();
        validate_original(&case).unwrap();
        case.base.bcs[0].lock[4] = false;
        assert!(validate_original(&case).is_err());
        case = super::super::shell_twisted_beam_case();
        case.base.cloads[0].force_scale_n *= 2.0;
        assert!(validate_original(&case).is_err());
    }
    #[test]
    fn matching_displacement_cannot_hide_bad_equations_thickness_or_reference_health() {
        let case = super::super::shell_twisted_beam_case();
        let external = case
            .base
            .probes
            .iter()
            .map(|p| p.expected)
            .collect::<Vec<_>>();
        let mut runs=[0.995,0.999,1.0,1.0001].map(|f|serde_json::json!({
            "hold_mean_tip_dz_m":external[0]*f,"thickness_m":external[1],"max_true_dynamic_residual":1e-10}));
        assert!(
            build_result(&case, "test", "reference", &external, true, &runs)
                .unwrap()
                .release_eligible()
                .is_ok()
        );
        runs[2]["max_true_dynamic_residual"] = serde_json::json!(1e-4);
        assert!(
            build_result(&case, "test", "reference", &external, true, &runs)
                .unwrap()
                .release_eligible()
                .is_err()
        );
        runs[2]["max_true_dynamic_residual"] = serde_json::json!(1e-10);
        runs[2]["thickness_m"] = serde_json::json!(2.0 * external[1]);
        assert!(
            build_result(&case, "test", "reference", &external, true, &runs)
                .unwrap()
                .release_eligible()
                .is_err()
        );
        runs[2]["thickness_m"] = serde_json::json!(external[1]);
        assert!(
            build_result(&case, "test", "reference", &external, false, &runs)
                .unwrap()
                .release_eligible()
                .is_err()
        );
    }
}
