//! Native parity for the original discrete three-dimensional sphere contact fixture.

// Rust guideline compliant 2026-02-21

#[path = "../oc_bench.rs"]
#[expect(
    dead_code,
    reason = "Only actual native elastic/contact integration is used"
)]
mod engine;

use engine::NativeElasticHistory;
pub(super) use engine::original_hertz_history;
use sha2::{Digest, Sha256};
use sonium_qa::rad::{
    OpenCourantCase, TimeHistory, engine_exe, hertz_sphere_case, probes_json, run_case,
    write_engine, write_starter,
};
use sonium_qa::{QaResult, QaStatus};

/// Render matching full Q1 quadrature without altering the physical fixture.
///
/// # Errors
/// Rejects a missing or altered original solid property row.
pub(super) fn full_gauss_starter(starter: &str) -> Result<String, Box<dyn std::error::Error>> {
    let mut lines = starter.lines().map(str::to_owned).collect::<Vec<_>>();
    let row = lines
        .iter()
        .position(|s| s == "/PROP/SOLID/1/1")
        .ok_or("original solid property missing")?
        + 2;
    if lines
        .get(row)
        .ok_or("original solid property row missing")?
        .split_whitespace()
        .collect::<Vec<_>>()
        != ["24", "2"]
    {
        return Err("original Hertz solid formulation changed".into());
    }
    // Standard H8C: eight Gauss points and full compressible pressure
    // integration. Altair PROP/TYPE14 Isolid17, Ismstr2, Iale0, Icpre3.
    lines[row] = format!("{:>10}{:>10}{:>10}{:>10}", 17, 2, 0, 3);
    Ok(lines.join("\n") + "\n")
}

fn validate(times: &[f64], values: &[f64]) -> Result<(), Box<dyn std::error::Error>> {
    if times.len() < 2
        || times.len() != values.len()
        || times.iter().chain(values).any(|x| !x.is_finite())
        || times.windows(2).any(|w| w[1] <= w[0])
    {
        return Err("complete finite strictly increasing contact history required".into());
    }
    Ok(())
}

fn at(times: &[f64], values: &[f64], time: f64) -> Result<f64, Box<dyn std::error::Error>> {
    if !time.is_finite() || time < times[0] || time > *times.last().ok_or("empty history")? {
        return Err("contact sampling cannot extrapolate".into());
    }
    let i = times.partition_point(|x| *x < time);
    if i == 0 {
        return Ok(values[0]);
    }
    let weight = (time - times[i - 1]) / (times[i] - times[i - 1]);
    Ok(values[i - 1] * (1.0 - weight) + values[i] * weight)
}

fn relative(a: &[f64], b: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    if a.len() != b.len() || a.is_empty() || a.iter().chain(b).any(|x| !x.is_finite()) {
        return Err("aligned finite physical probes required".into());
    }
    let denominator = b.iter().map(|x| x * x).sum::<f64>();
    if denominator <= 0.0 || !denominator.is_finite() {
        return Err("nonzero finite reference contact field required".into());
    }
    Ok((a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / denominator).sqrt())
}

fn compare(
    case: &OpenCourantCase,
    id: &str,
    label: &str,
    reference: &TimeHistory,
    checksum: &str,
    runs: &[NativeElasticHistory; 2],
) -> Result<QaResult, Box<dyn std::error::Error>> {
    let expected = vec![
        case.probes[0].channel.clone(),
        "FNZ/RWALL/1".into(),
        "FNZ/RWALL/2".into(),
    ];
    let starter = format!("{:x}", Sha256::digest(write_starter(case).as_bytes()));
    let deck = format!("{:x}", Sha256::digest(write_engine(case).as_bytes()));
    for run in runs {
        if run.probe_channels != expected
            || run.probe_values.len() != 3
            || run.starter_sha256 != starter
            || run.engine_deck_sha256 != deck
            || run.times_s.first().copied() != Some(0.0)
            || run.steps == 0
            || !run.relative_energy_drift.is_finite()
            || run.relative_energy_drift < 0.0
        {
            return Err("original complete native contact fixture/history required".into());
        }
        for values in &run.probe_values {
            validate(&run.times_s, values)?;
        }
        if (run.times_s.last().ok_or("missing native final time")? - run.final_time_s).abs() > 1e-12
        {
            return Err("native final state and history times differ".into());
        }
    }
    let end = *reference
        .times
        .last()
        .ok_or("missing external contact history")?;
    let mut result=QaResult::passed(id).requested_backend("cpu+opencourant")
        .actual_backend("cpu+opencourant")
        .metric("external_reference_health_passed",1.0).minimum("external_reference_health_passed",1.0)
        .metric("native_parity_executed",1.0).minimum("native_parity_executed",1.0)
        .metric("native_histories_executed",2.0).minimum("native_histories_executed",2.0)
        .metric("full_gauss_reference_executed",1.0).minimum("full_gauss_reference_executed",1.0)
        .metric("external_history_coverage",end/case.stop_time_s).minimum("external_history_coverage",0.98)
        .metric("native_timestep_step_ratio",runs[1].steps as f64/runs[0].steps as f64)
        .minimum("native_timestep_step_ratio",1.5)
        .input_checksum(format!("{starter}:{deck}"))
        .reference_checksum(checksum)
        .reference(format!("{label}; independent fully integrated H8C original physical fixture"))
        .dimension("Original R50mm steel sphere, 10mm mesh, E200GPa/nu.3/rho7800, frictionless planes; top -1e-4m/s, full 0.1s squeeze")
        .formulation("Original HEPH external health retained; matching full-Q1 quadrature external H8C reference and two full native elastic/multiplier-contact histories, sampled at exactly the actual external timestamps. Original coarse discrete fixture; no continuum Hertz-patch force accuracy claim.")
        .tolerance("Three full histories and final same-time probes <=3%; half-CFL changes <=.3%; final wall balance <=1%; numerical energy drift <=1%; exact native stop and complete external history required.");
    let mut final_native = [0.0; 3];
    let mut final_external = [0.0; 3];
    for (j, channel) in expected.iter().enumerate() {
        let external = reference.channel(channel)?;
        validate(&reference.times, external)?;
        let sampled = runs
            .iter()
            .map(|r| {
                reference
                    .times
                    .iter()
                    .map(|t| at(&r.times_s, &r.probe_values[j], *t))
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let name = ["pole_displacement", "bottom_force", "top_force"][j];
        let field = format!("{name}_history_relative_error");
        let temporal = format!("{name}_temporal_relative_change");
        let final_error = format!("{name}_final_relative_error");
        final_native[j] = *sampled[1].last().ok_or("missing final native probe")?;
        final_external[j] = *external.last().ok_or("missing final external probe")?;
        result = result
            .metric(&field, relative(&sampled[1], external)?)
            .maximum(&field, 0.03)
            .metric(&temporal, relative(&sampled[0], &sampled[1])?)
            .maximum(&temporal, 0.003)
            .metric(
                &final_error,
                relative(&[final_native[j]], &[final_external[j]])?,
            )
            .maximum(&final_error, 0.03);
    }
    for (i, run) in runs.iter().enumerate() {
        let stop = format!("native_{i}_stop_error_s");
        let energy = format!("native_{i}_relative_energy_drift");
        result = result
            .metric(&stop, (run.final_time_s - case.stop_time_s).abs())
            .maximum(&stop, 1e-12)
            .metric(&energy, run.relative_energy_drift)
            .maximum(&energy, 0.01);
    }
    for (name, values) in [("native", final_native), ("external", final_external)] {
        let scale = values[1].abs().max(values[2].abs());
        if values[1] >= 0.0 || values[2] <= 0.0 || values[0] >= 0.0 || scale <= 0.0 {
            return Err("physical compressive contact reactions and pole motion required".into());
        }
        let balance = format!("{name}_final_wall_balance");
        let stiffness = scale / (end * 1e-4);
        let low = format!("{name}_stiffness_lower_n_per_m");
        let high = format!("{name}_stiffness_upper_n_per_m");
        result = result
            .metric(&balance, (values[1] + values[2]).abs() / scale)
            .maximum(&balance, 0.01)
            .metric(&low, stiffness)
            .minimum(&low, 1e8)
            .metric(&high, stiffness)
            .maximum(&high, 3e9);
    }
    if let Err(error) = result.release_eligible() {
        result.status = QaStatus::Failed;
        result.message = Some(error);
    }
    Ok(result)
}

/// Execute complete independent-reference and native contact parity validation.
///
/// # Errors
/// Rejects changed original physics, failed reference health, unavailable engine,
/// failed integration, incomplete traces or invalid physical contact fields.
pub(super) fn parity_result(
    case: &OpenCourantCase,
    id: &str,
    label: &str,
    healthy: bool,
) -> Result<QaResult, Box<dyn std::error::Error>> {
    let original = hertz_sphere_case();
    if write_starter(case) != write_starter(&original)
        || write_engine(case) != write_engine(&original)
        || probes_json(case, "identity") != probes_json(&original, "identity")
    {
        return Err("original Hertz physical fixture changed".into());
    }
    if !healthy {
        return Err("original external contact health failed".into());
    }
    let starter = full_gauss_starter(&write_starter(case))?;
    let deck = write_engine(case);
    let checksum = format!(
        "{:x}",
        Sha256::digest(format!("{starter}{deck}").as_bytes())
    );
    let work = std::env::temp_dir().join(format!(
        "p2_hertz_full_gauss_reference_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&work)?;
    std::fs::write(work.join("hertz_sphere_0000.rad"), &starter)?;
    std::fs::write(work.join("hertz_sphere_0001.rad"), &deck)?;
    let exe = engine_exe().ok_or("OpenCourant required for full Gauss contact reference")?;
    let reference = run_case(&work, &exe, case)?;
    let runs = [original_hertz_history(1.0)?, original_hertz_history(0.5)?];
    compare(case, id, label, &reference.history, &checksum, &runs)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn changed_original_physics_is_rejected_before_execution() {
        let mut case = hertz_sphere_case();
        case.material.youngs_modulus_pa *= 0.5;
        assert!(
            parity_result(&case, "test", "none", true)
                .unwrap_err()
                .to_string()
                .contains("physical fixture changed")
        );
    }
    #[test]
    fn interpolation_and_error_checks_reject_false_evidence() {
        assert!(validate(&[0.0, 1.0], &[0.0, f64::NAN]).is_err());
        assert!(validate(&[1.0, 0.0], &[0.0, 1.0]).is_err());
        assert!(relative(&[0.0, 0.0], &[0.0, 0.0]).is_err());
        assert!(relative(&[1.0], &[1.0, 2.0]).is_err());
        assert_eq!(at(&[0.0, 1.0], &[0.0, 2.0], 0.25).unwrap(), 0.5);
        assert!(at(&[0.0, 1.0], &[0.0, 2.0], 1.1).is_err());
    }
}
