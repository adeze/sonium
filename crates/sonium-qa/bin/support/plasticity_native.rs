//! Matching native plastic loading, unloading, and residuals for the original cylinder.

// Rust guideline compliant 2026-02-21

#[path = "../oc_bench.rs"]
#[expect(
    dead_code,
    reason = "The existing full native constitutive/integration path is shared; benchmark CLI is not invoked"
)]
mod native;

use native::NativePlasticityHistory;
use sha2::{Digest, Sha256};
use sonium_qa::rad::{RadExtendedCase, TimeHistory};

fn elastic_times() -> [f64; 2] {
    let t = super::LOAD_TIME_S * (1.0 - 2.0 * super::ELASTIC_PROBE_PA / super::P_STAR_PA).acos()
        / std::f64::consts::PI;
    let lam = super::E_PA * super::NU / ((1.0 + super::NU) * (1.0 - 2.0 * super::NU));
    let mu = super::E_PA / (2.0 * (1.0 + super::NU));
    [
        t,
        t + (super::B_M - super::A_M) / ((lam + 2.0 * mu) / super::RHO_KG_M3).sqrt(),
    ]
}

fn samples(
    case: &RadExtendedCase,
    history: &TimeHistory,
    external_times: &[f64],
) -> Result<Vec<f64>, Box<dyn std::error::Error>> {
    let times = elastic_times();
    if external_times.len() < super::RESIDUAL_AVG_SAMPLES {
        return Err("complete original residual sample window required".into());
    }
    let tail = &external_times[external_times.len() - super::RESIDUAL_AVG_SAMPLES..];
    if tail
        .iter()
        .any(|t| !t.is_finite() || *t < super::UNLOAD_END_S)
    {
        return Err("original residual samples must lie in the unloaded hold".into());
    }
    case.base
        .probes
        .iter()
        .enumerate()
        .map(|(index, p)| {
            if index < 2 {
                Ok(history.value_at(&p.channel, times[index])?)
            } else {
                let values = tail
                    .iter()
                    .map(|t| history.value_at(&p.channel, *t))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(values.iter().sum::<f64>() / values.len() as f64)
            }
        })
        .collect()
}

fn history(run: &NativePlasticityHistory) -> Result<TimeHistory, Box<dyn std::error::Error>> {
    if run.probe_channels.len() != run.probe_values.len()
        || run.times_s.len() < 2
        || run
            .probe_values
            .iter()
            .any(|v| v.len() != run.times_s.len() || v.iter().any(|x| !x.is_finite()))
        || run.times_s.iter().any(|v| !v.is_finite())
        || run.times_s.windows(2).any(|w| w[1] <= w[0])
    {
        return Err("complete finite increasing native traces required".into());
    }
    if run.times_s[0].abs() > 1e-12
        || (run.times_s.last().ok_or("missing native stop time")? - run.final_time_s).abs() > 1e-12
    {
        return Err("native trace does not span its complete zero-to-stop history".into());
    }
    Ok(TimeHistory {
        times: run.times_s.clone(),
        channels: run
            .probe_channels
            .iter()
            .cloned()
            .zip(run.probe_values.iter().cloned())
            .collect(),
    })
}

fn plastic(run: &NativePlasticityHistory, id: usize) -> Result<f64, Box<dyn std::error::Error>> {
    let index = run
        .element_ids
        .iter()
        .position(|v| *v == id as i64)
        .ok_or("original native plastic probe ID missing")?;
    run.final_element_plastic_strain
        .get(index)
        .copied()
        .filter(|v| v.is_finite())
        .ok_or_else(|| "missing finite actual native plastic strain".into())
}

/// Compare complete native cylinder histories with the original external case.
///
/// # Errors
/// Returns an error for altered original physics, incomplete or nonfinite
/// histories, failed native integration, or unavailable original probe IDs.
pub(super) fn parity_result(
    case: &RadExtendedCase,
    id: &str,
    engine: &str,
    external: &TimeHistory,
    reference_passed: bool,
) -> Result<sonium_qa::QaResult, Box<dyn std::error::Error>> {
    let original = super::plasticity_thick_cylinder_case();
    if sonium_qa::rad::write_extended_starter(case)
        != sonium_qa::rad::write_extended_starter(&original)
        || sonium_qa::rad::write_extended_engine(case)
            != sonium_qa::rad::write_extended_engine(&original)
        || sonium_qa::rad::extended_probes_json(case, "identity")
            != sonium_qa::rad::extended_probes_json(&original, "identity")
    {
        return Err("original full plastic cylinder fixture changed".into());
    }
    let coarse = native::original_thick_cylinder_history(1.0)?;
    let fine = native::original_thick_cylinder_history(0.5)?;
    build_result(
        case,
        id,
        engine,
        external,
        reference_passed,
        &[coarse, fine],
    )
}

fn build_result(
    case: &RadExtendedCase,
    id: &str,
    engine: &str,
    external: &TimeHistory,
    reference_passed: bool,
    runs: &[NativePlasticityHistory; 2],
) -> Result<sonium_qa::QaResult, Box<dyn std::error::Error>> {
    let starter = format!(
        "{:x}",
        Sha256::digest(sonium_qa::rad::write_extended_starter(case).as_bytes())
    );
    let engine_deck = format!(
        "{:x}",
        Sha256::digest(sonium_qa::rad::write_extended_engine(case).as_bytes())
    );
    if runs
        .iter()
        .any(|r| r.starter_sha256 != starter || r.engine_deck_sha256 != engine_deck)
    {
        return Err("native/external original deck bytes differ".into());
    }
    let measured = samples(case, external, &external.times)?;
    let values = runs
        .iter()
        .map(|r| samples(case, &history(r)?, &external.times))
        .collect::<Result<Vec<_>, _>>()?;
    let mut result=sonium_qa::QaResult::passed(id)
        .requested_backend("cpu+opencourant").actual_backend("cpu+opencourant")
        .metric("external_reference_executed",1.0)
        .metric("external_reference_health_passed",if reference_passed{1.0}else{0.0}).minimum("external_reference_health_passed",1.0)
        .metric("native_parity_executed",1.0).minimum("native_parity_executed",1.0)
        .metric("native_complete_plastic_histories",2.0).minimum("native_complete_plastic_histories",2.0)
        .metric("native_timestep_refinement_ratio",runs[1].steps as f64/(runs[0].steps as f64).max(1.0)).minimum("native_timestep_refinement_ratio",1.5)
        .reference(engine)
        .dimension("Original 50/100 mm bore/outer radii, 10 mm height, 12x24x1 3D hex ring; E=200 GPa, nu=.3, rho=7800 kg/m3, 250 MPa perfect-plastic LAW36 yield")
        .formulation("Same original starter/engine bytes; native full 3D finite-strain LAW36 brick dynamics, original z-locked caps and radial nodal pressure forces, 150 MPa raised-cosine loading/unloading and zero hold to 3 ms; native original and half-CFL bounds. Both sides use the original elastic crossing/transit times and exact same last 50 external time stamps for residual stress. Native c1=0, quadratic shock viscosity retained; observed energy balance and bore/outer plastic-zone checks.")
        .tolerance("All four native/external physical probes and bore plastic strain <=3%; timestep changes <=.3%; original analytical displacement/stress budgets retained; bore plastic strain >=5e-4, outer <=1e-9; full stop time <=1e-12 s, no erosion; native relative energy drift <=1%.");
    for (i, p) in case.base.probes.iter().enumerate() {
        for (suffix, value, bound) in [
            (
                "native_external_relative_error",
                (values[1][i] - measured[i]).abs() / measured[i].abs().max(1e-30),
                0.03,
            ),
            (
                "temporal_relative_change",
                (values[0][i] - values[1][i]).abs() / values[1][i].abs().max(1e-30),
                0.003,
            ),
            (
                "native_analytical_normalized_error",
                (values[1][i] - p.expected).abs() / (p.abs_tol + p.rel_tol * p.expected.abs()),
                1.0,
            ),
        ] {
            let key = format!("probe_{i}_{suffix}");
            result = result.metric(&key, value).maximum(key, bound);
        }
        result = result
            .metric(format!("probe_{i}_native_value"), values[1][i])
            .metric(format!("probe_{i}_external_value"), measured[i]);
    }
    for (i, r) in runs.iter().enumerate() {
        for (suffix, value, bound) in [
            (
                "stop_time_error_s",
                (r.final_time_s - case.base.stop_time_s).abs(),
                1e-12,
            ),
            ("initial_time_error_s", history(r)?.times[0].abs(), 1e-12),
            ("energy_drift", r.relative_energy_drift.abs(), 0.01),
            ("eroded_elements", r.eroded_elements as f64, 0.0),
            (
                "outer_plastic_strain",
                plastic(r, super::outer_brick_id())?,
                1e-9,
            ),
        ] {
            let key = format!("history_{i}_{suffix}");
            result = result.metric(&key, value).maximum(key, bound);
        }
        let key = format!("history_{i}_bore_plastic_strain");
        result = result.metric(&key, plastic(r, 1)?).minimum(key, 5e-4);
        let key = format!("history_{i}_outer_plastic_strain_nonnegative");
        result = result
            .metric(&key, plastic(r, super::outer_brick_id())?)
            .minimum(key, 0.0);
        let key = format!("history_{i}_steps");
        result = result.metric(&key, r.steps as f64).minimum(key, 1.0);
    }
    let ext_plastic = external.final_value(&sonium_qa::rad::bric_channel("PLAS", 1))?;
    let ext_outer = external.final_value(&sonium_qa::rad::bric_channel(
        "PLAS",
        super::outer_brick_id(),
    ))?;
    result = result
        .metric("external_bore_plastic_strain", ext_plastic)
        .minimum("external_bore_plastic_strain", 5e-4)
        .metric("external_outer_plastic_strain", ext_outer)
        .maximum("external_outer_plastic_strain", 1e-9)
        .metric("external_outer_plastic_strain_nonnegative", ext_outer)
        .minimum("external_outer_plastic_strain_nonnegative", 0.0);
    let nat_plastic = plastic(&runs[1], 1)?;
    result = result
        .metric(
            "bore_plastic_native_external_relative_error",
            (nat_plastic - ext_plastic).abs() / ext_plastic.abs().max(1e-30),
        )
        .maximum("bore_plastic_native_external_relative_error", 0.03)
        .metric(
            "bore_plastic_temporal_change",
            (plastic(&runs[0], 1)? - nat_plastic).abs() / nat_plastic.abs().max(1e-30),
        )
        .maximum("bore_plastic_temporal_change", 0.003);
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
    fn matching_probes_cannot_hide_absent_plasticity_or_incomplete_history() {
        let case = super::super::plasticity_thick_cylinder_case();
        let times = (0..=1500).map(|i| i as f64 * 2e-6).collect::<Vec<_>>();
        let channels = case
            .base
            .probes
            .iter()
            .map(|p| p.channel.clone())
            .collect::<Vec<_>>();
        let values = case
            .base
            .probes
            .iter()
            .map(|p| vec![p.expected; times.len()])
            .collect::<Vec<_>>();
        let mut external = TimeHistory {
            times: times.clone(),
            channels: channels
                .iter()
                .cloned()
                .zip(values.iter().cloned())
                .collect(),
        };
        external.channels.insert(
            sonium_qa::rad::bric_channel("PLAS", 1),
            vec![0.0007; times.len()],
        );
        external.channels.insert(
            sonium_qa::rad::bric_channel("PLAS", super::super::outer_brick_id()),
            vec![0.0; times.len()],
        );
        let make = |steps| NativePlasticityHistory {
            probe_channels: channels.clone(),
            times_s: times.clone(),
            probe_values: values.clone(),
            element_ids: vec![1, super::super::outer_brick_id() as i64],
            final_element_plastic_strain: vec![0.0007, 0.0],
            final_time_s: 0.003,
            steps,
            relative_energy_drift: 1e-8,
            starter_sha256: format!(
                "{:x}",
                Sha256::digest(sonium_qa::rad::write_extended_starter(&case).as_bytes())
            ),
            engine_deck_sha256: format!(
                "{:x}",
                Sha256::digest(sonium_qa::rad::write_extended_engine(&case).as_bytes())
            ),
            ..NativePlasticityHistory::default()
        };
        let runs = [make(4000), make(8000)];
        assert!(
            build_result(&case, "test", "test", &external, true, &runs)
                .unwrap()
                .release_eligible()
                .is_ok()
        );
        assert!(
            build_result(&case, "test", "test", &external, false, &runs)
                .unwrap()
                .release_eligible()
                .is_err()
        );
        let bad = [
            make(4000),
            NativePlasticityHistory {
                final_element_plastic_strain: vec![0.0, 0.0],
                ..make(8000)
            },
        ];
        assert!(
            build_result(&case, "test", "test", &external, true, &bad)
                .unwrap()
                .release_eligible()
                .is_err()
        );
        let bad = [
            make(4000),
            NativePlasticityHistory {
                eroded_elements: 1,
                ..make(8000)
            },
        ];
        assert!(
            build_result(&case, "test", "test", &external, true, &bad)
                .unwrap()
                .release_eligible()
                .is_err()
        );
        let bad = [
            make(4000),
            NativePlasticityHistory {
                relative_energy_drift: 0.02,
                ..make(8000)
            },
        ];
        assert!(
            build_result(&case, "test", "test", &external, true, &bad)
                .unwrap()
                .release_eligible()
                .is_err()
        );
        let bad = [make(4000), make(4000)];
        assert!(
            build_result(&case, "test", "test", &external, true, &bad)
                .unwrap()
                .release_eligible()
                .is_err()
        );
        let mut bad = [make(4000), make(8000)];
        bad[1].times_s.pop();
        for v in &mut bad[1].probe_values {
            v.pop();
        }
        assert!(build_result(&case, "test", "test", &external, true, &bad).is_err());
        let bad = [
            make(4000),
            NativePlasticityHistory {
                starter_sha256: "changed".into(),
                ..make(8000)
            },
        ];
        assert!(build_result(&case, "test", "test", &external, true, &bad).is_err());
    }
    #[test]
    fn residual_samples_use_actual_original_external_times() {
        let case = super::super::plasticity_thick_cylinder_case();
        let grid = (0..=1500).map(|i| i as f64 * 0.000002).collect::<Vec<_>>();
        let mut h = TimeHistory {
            times: grid.clone(),
            channels: Default::default(),
        };
        for p in &case.base.probes {
            h.channels.insert(
                p.channel.clone(),
                grid.iter().map(|t| 2.0 + 100.0 * t).collect(),
            );
        }
        let got = samples(&case, &h, &grid).unwrap();
        let expected = grid[1451..].iter().map(|t| 2.0 + 100.0 * t).sum::<f64>() / 50.0;
        assert!((got[2] - expected).abs() < 1e-14);
        assert_eq!(got[2], got[3]);
        assert!(samples(&case, &h, &grid[..49]).is_err());
    }
    #[test]
    fn changed_loading_or_geometry_is_rejected_before_native_execution() {
        let mut case = super::super::plasticity_thick_cylinder_case();
        case.base.cloads[0].force_scale_n *= 0.9;
        assert!(parity_result(&case, "test", "test", &TimeHistory::default(), true).is_err());
    }
}
