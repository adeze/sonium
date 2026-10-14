//! Full native open-hole elastic histories retaining original benchmark physics.

// Rust guideline compliant 2026-02-21

#[path = "shell_transient.rs"]
mod transient;
use sonium_qa::rad::RadExtendedCase;

// Remove only nodes with no element support. Keep every physical coordinate,
// element, force, support and probe, including the original staircase hole.
fn compact(case: &mut RadExtendedCase) -> Result<(), Box<dyn std::error::Error>> {
    let old = case.base.mesh.nodes.clone();
    let mut used = vec![false; old.len()];
    for quad in &case.shells.quads {
        for &node in quad {
            used[node - 1] = true;
        }
    }
    let mut map = vec![0; old.len()];
    case.base.mesh.nodes.clear();
    for (index, point) in old.iter().enumerate() {
        if used[index] {
            case.base.mesh.nodes.push(*point);
            map[index] = case.base.mesh.nodes.len();
        }
    }
    let mapped = |node: usize| -> Result<usize, Box<dyn std::error::Error>> {
        let new = map[node - 1];
        if new == 0 {
            return Err("physical support/load/probe references an unused node".into());
        }
        Ok(new)
    };
    for quad in &mut case.shells.quads {
        for node in quad {
            *node = mapped(*node)?;
        }
    }
    for set in &mut case.base.node_sets {
        for node in &mut set.nodes {
            *node = mapped(*node)?;
        }
    }
    for node in &mut case.base.th_nodes {
        *node = mapped(*node)?;
    }
    for probe in &mut case.base.probes {
        if let Some(node) = probe.channel.strip_prefix("DX/") {
            probe.channel = format!("DX/{}", mapped(node.parse()?)?);
        }
    }
    case.validate()?;
    Ok(())
}

// Center strain of the actual flat rectangular Q4 + Allman membrane.
// The edge bubbles carry (edge length/8)*(theta_z_a-theta_z_b).
// Stress is Pa, while the shell constitutive matrix includes thickness.
fn stress_at(points: &[[f64; 3]; 4], u: &[f64], e: f64, nu: f64, xi: f64, eta: f64) -> f64 {
    let hx = points[1][0] - points[0][0];
    let hy = points[3][1] - points[0][1];
    let ex = (-(1.0 - eta) * u[0] + (1.0 - eta) * u[6] + (1.0 + eta) * u[12] - (1.0 + eta) * u[18])
        / (2.0 * hx)
        - hy * (1.0 - eta * eta) * (u[11] - u[17] + u[23] - u[5]) / (8.0 * hx);
    let ey = (-(1.0 - xi) * u[1] - (1.0 + xi) * u[7] + (1.0 + xi) * u[13] + (1.0 - xi) * u[19])
        / (2.0 * hy)
        - hx * (1.0 - xi * xi) * (u[5] - u[11] + u[17] - u[23]) / (8.0 * hy);
    e / (1.0 - nu * nu) * (ex + nu * ey)
}

#[cfg(test)]
fn center_stress(points: &[[f64; 3]; 4], u: &[f64], e: f64, nu: f64) -> f64 {
    stress_at(points, u, e, nu, 0.0, 0.0)
}

fn subdivided_case(
    factor: usize,
) -> Result<(RadExtendedCase, [f64; 3]), Box<dyn std::error::Error>> {
    if !(1..=8).contains(&factor) {
        return Err("open-hole refinement factor must be 1..=8".into());
    }
    let mut case = super::shell_oht_elastic_case();
    let shell = case.base.probes[1]
        .channel
        .rsplit('/')
        .next()
        .ok_or("original shell probe")?
        .parse::<usize>()?;
    let probe = case.shells.quads[shell - 1]
        .map(|n| case.base.mesh.nodes[n - 1])
        .iter()
        .fold([0.0; 3], |mut a, p| {
            for d in 0..3 {
                a[d] += p[d] / 4.0;
            }
            a
        });
    if factor == 1 {
        return Ok((case, probe));
    }
    let old = case.base.mesh.nodes.clone();
    let nx = 100;
    let ny = 40;
    // Original structured grid coordinates; subdivision retains the actual
    // staircase boundary instead of cutting a different circular hole.
    if old.len() != (nx + 1) * (ny + 1) {
        return Err("original open-hole grid changed".into());
    }
    let xs = (0..=nx).map(|i| old[i * (ny + 1)][0]).collect::<Vec<_>>();
    let ys = (0..=ny).map(|j| old[j][1]).collect::<Vec<_>>();
    let refine_axis = |a: &[f64]| {
        (0..=(a.len() - 1) * factor)
            .map(|i| {
                if i.is_multiple_of(factor) {
                    a[i / factor]
                } else {
                    let k = i / factor;
                    a[k] + (a[k + 1] - a[k]) * (i % factor) as f64 / factor as f64
                }
            })
            .collect::<Vec<_>>()
    };
    let fx = refine_axis(&xs);
    let fy = refine_axis(&ys);
    let id = |i: usize, j: usize| i * (ny * factor + 1) + j + 1;
    let mapped = |n: usize| {
        let k = n - 1;
        id((k / (ny + 1)) * factor, (k % (ny + 1)) * factor)
    };
    let mut quads = Vec::new();
    for q in &case.shells.quads {
        let ix = (q[0] - 1) / (ny + 1);
        let iy = (q[0] - 1) % (ny + 1);
        for i in 0..factor {
            for j in 0..factor {
                let x = ix * factor + i;
                let y = iy * factor + j;
                quads.push([id(x, y), id(x + 1, y), id(x + 1, y + 1), id(x, y + 1)]);
            }
        }
    }
    case.base.mesh.nodes = fx
        .iter()
        .flat_map(|x| fy.iter().map(move |y| [*x, *y, 0.0]))
        .collect();
    case.shells.quads = quads;
    for set in &mut case.base.node_sets {
        for n in &mut set.nodes {
            *n = mapped(*n);
        }
    }
    // Extend the same continuous homogeneous root-edge support. The original
    // 41 loaded grip nodes and per-node forces remain exactly unchanged.
    case.base.node_sets[0].nodes = (0..=ny * factor).map(|j| id(0, j)).collect();
    for n in &mut case.base.th_nodes {
        *n = mapped(*n);
    }
    case.base.probes[0].channel = format!("DX/{}", case.base.th_nodes[0]);
    case.validate()?;
    Ok((case, probe))
}

/// Measure the complete original open-hole history on a subdivided native mesh.
///
/// # Errors
/// Returns an error for invalid refinement, failed full transient equilibrium,
/// or missing original physical stress-probe support. No external parity is claimed.
pub fn probe(factor: usize, dt: f64) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    let (mut case, probe) = subdivided_case(factor)?;
    let original_node_count = case.base.mesh.nodes.len();
    compact(&mut case)?;
    let mut elements = Vec::new();
    let mut observations = vec![(case.base.th_nodes[0] - 1) * 6];
    for q in &case.shells.quads {
        let points = q.map(|n| case.base.mesh.nodes[n - 1]);
        let p0 = points[0];
        let p2 = points[2];
        if probe[0] >= p0[0] - 1e-14
            && probe[0] <= p2[0] + 1e-14
            && probe[1] >= p0[1] - 1e-14
            && probe[1] <= p2[1] + 1e-14
        {
            let xi = 2.0 * (probe[0] - p0[0]) / (p2[0] - p0[0]) - 1.0;
            let eta = 2.0 * (probe[1] - p0[1]) / (p2[1] - p0[1]) - 1.0;
            let offset = observations.len();
            for n in q {
                observations.extend((0..6).map(|d| (n - 1) * 6 + d));
            }
            elements.push((points, xi, eta, offset));
        }
    }
    if elements.is_empty() {
        return Err("original physical stress probe is outside refined mesh".into());
    }
    let history = transient::run(&case, dt, &observations, true)?;
    // The arithmetic trace average at a shared vertex/edge uses all adjacent
    // actual FE gradients evaluated at the original fixed physical point.
    let stresses = history
        .observations
        .iter()
        .map(|row| {
            elements
                .iter()
                .map(|(p, xi, eta, offset)| {
                    stress_at(
                        p,
                        &row[*offset..*offset + 24],
                        case.base.material.youngs_modulus_pa,
                        case.base.material.poisson_ratio,
                        *xi,
                        *eta,
                    )
                })
                .sum::<f64>()
                / elements.len() as f64
        })
        .collect::<Vec<_>>();
    let dx = history
        .observations
        .iter()
        .map(|row| row[0])
        .collect::<Vec<_>>();
    Ok(
        serde_json::json!({"status":"diagnostic; original QA/native parity is not established",
        "refinement_factor":factor,"original_nodes":original_node_count,"active_nodes":case.base.mesh.nodes.len(),
        "quads":case.shells.quads.len(),"dofs":history.dofs,"dt":history.dt,"steps":history.steps,
        "final_time_s":history.times.last(),"native_final_pulled_center_dx_m":dx.last(),
        "native_final_probe_center_sigma_xx_pa":stresses.last(),"times_s":history.times,
        "pulled_center_dx_m":dx,"probe_sigma_xx_pa":stresses,
        "thickness_m":case.shell_prop.as_ref().ok_or("missing shell thickness")?.thickness_m,
        "original_probe_centroid":probe,"probe_adjacent_elements":elements.len(),
        "maximum_true_dynamic_residual":history.max_true_dynamic_residual}),
    )
}
fn sample(
    run: &serde_json::Value,
    key: &str,
    time: f64,
) -> Result<f64, Box<dyn std::error::Error>> {
    let times = run["times_s"]
        .as_array()
        .ok_or("missing native time trace")?;
    let values = run[key].as_array().ok_or("missing native probe trace")?;
    if times.len() != values.len() {
        return Err("native trace lengths differ".into());
    }
    for i in 0..times.len().saturating_sub(1) {
        let a = times[i]
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("invalid native sample time")?;
        let b = times[i + 1]
            .as_f64()
            .filter(|v| v.is_finite())
            .ok_or("invalid native sample time")?;
        if a <= time && time <= b && b > a {
            let x = values[i]
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or("invalid native probe value")?;
            let y = values[i + 1]
                .as_f64()
                .filter(|v| v.is_finite())
                .ok_or("invalid native probe value")?;
            return Ok(x + (y - x) * (time - a) / (b - a));
        }
    }
    Err("external sample time is outside the complete native trace".into())
}

fn number(run: &serde_json::Value, key: &str) -> Result<f64, Box<dyn std::error::Error>> {
    run[key]
        .as_f64()
        .filter(|v| v.is_finite())
        .ok_or_else(|| format!("missing finite native field {key}").into())
}

/// Compare complete matching native and external open-hole elastic histories.
///
/// The original discrete fixture is retained on both sides. Subdivision remains
/// available through `probe` as a separate spatial study; its field cannot be
/// compared to the unchanged coarse external discretization as mesh parity.
///
/// # Errors
/// Rejects changed original physics, incomplete histories, missing channels,
/// failed native integration, and nonfinite physical observations.
pub(super) fn parity_result(
    case: &RadExtendedCase,
    case_id: &str,
    engine: &str,
    external: &sonium_qa::rad::TimeHistory,
    reference_passed: bool,
) -> Result<sonium_qa::QaResult, Box<dyn std::error::Error>> {
    let original = super::shell_oht_elastic_case();
    if sonium_qa::rad::write_extended_starter(case)
        != sonium_qa::rad::write_extended_starter(&original)
        || sonium_qa::rad::write_extended_engine(case)
            != sonium_qa::rad::write_extended_engine(&original)
        || sonium_qa::rad::extended_probes_json(case, "identity")
            != sonium_qa::rad::extended_probes_json(&original, "identity")
    {
        return Err("original open-hole fixture changed".into());
    }
    validate_reference(case, external)?;
    let runs = [probe(1, 1e-6)?, probe(1, 5e-7)?];
    build_result(case, case_id, engine, external, reference_passed, &runs)
}

fn validate_reference(
    case: &RadExtendedCase,
    external: &sonium_qa::rad::TimeHistory,
) -> Result<(), Box<dyn std::error::Error>> {
    let times = &external.times;
    if times.len() < 2
        || times.iter().any(|v| !v.is_finite())
        || times.windows(2).any(|w| w[1] <= w[0])
        || times[0] < 0.0
        || times[0] > case.base.th_period_s
        || times[times.len() - 1] < case.base.stop_time_s - 1.05 * case.base.th_period_s
        || times[times.len() - 1] > case.base.stop_time_s
    {
        return Err("complete finite original external history required".into());
    }
    for probe in &case.base.probes {
        let values = external.channel(&probe.channel)?;
        if values.len() != times.len() || values.iter().any(|v| !v.is_finite()) {
            return Err("complete finite original external channels required".into());
        }
    }
    Ok(())
}

fn relative(a: &[f64], b: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    if a.len() != b.len() || a.is_empty() || a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err("aligned finite native/reference fields required".into());
    }
    let denominator = b.iter().map(|v| v * v).sum::<f64>();
    if denominator <= 0.0 || !denominator.is_finite() {
        return Err("nonzero finite reference field required".into());
    }
    Ok((a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / denominator).sqrt())
}

fn build_result(
    case: &RadExtendedCase,
    case_id: &str,
    engine: &str,
    external: &sonium_qa::rad::TimeHistory,
    reference_passed: bool,
    runs: &[serde_json::Value; 2],
) -> Result<sonium_qa::QaResult, Box<dyn std::error::Error>> {
    use sha2::{Digest, Sha256};
    validate_reference(case, external)?;
    let time = *external.times.last().ok_or("missing external stop")?;
    let starter = sonium_qa::rad::write_extended_starter(case);
    let deck = sonium_qa::rad::write_extended_engine(case);
    let checksum = format!(
        "{:x}",
        Sha256::digest(format!("{starter}{deck}").as_bytes())
    );
    let mut result=sonium_qa::QaResult::passed(case_id)
        .requested_backend("cpu+opencourant").actual_backend("cpu+opencourant")
        .metric("external_reference_executed",1.0).minimum("external_reference_executed",1.0)
        .metric("external_reference_health_passed",f64::from(reference_passed)).minimum("external_reference_health_passed",1.0)
        .metric("native_parity_executed",1.0).minimum("native_parity_executed",1.0)
        .metric("native_complete_shell_histories",2.0).minimum("native_complete_shell_histories",2.0)
        .metric("external_history_coverage",time/case.base.stop_time_s).minimum("external_history_coverage",0.98)
        .metric("comparison_time_s",time).reference(engine)
        .input_checksum(&checksum).reference_checksum(&checksum)
        .dimension("Original 128x32 mm, 1 mm thick homogenized elastic staircase-hole plate; E=52.7 GPa, nu=.24, rho=1600 kg/m3; same original 3900 physical quads on both sides")
        .formulation("Full six-DOF MITC4 with Allman physical stress at the original shell centroid; original 41 grip-node forces, root constraints and complete 1 ms ramp/hold to 2 ms. Two checked Newmark histories at original and half timestep, independent physical equilibrium at every step; all original channels compared at every actual external timestamp. Fixed original discrete elastic fixture; spatial subdivision remains a separate diagnostic, no continuum mesh-convergence claim.")
        .tolerance("Full-history and final native/external displacement and stress <=3%; thickness <=1e-6; original analytical compliance <=10%, staircase-hole Howland stress <=20%; full-history and final timestep changes <=.3%; every-step physical residual <=1e-8; exact full native stop.");
    for (i, run) in runs.iter().enumerate() {
        let times = run["times_s"].as_array().ok_or("missing native times")?;
        if times.len() < 2
            || times
                .iter()
                .any(|v| v.as_f64().is_none_or(|x| !x.is_finite()))
            || times
                .windows(2)
                .any(|w| w[1].as_f64().unwrap() <= w[0].as_f64().unwrap())
            || times[0].as_f64() != Some(0.0)
            || run["refinement_factor"].as_u64() != Some(1)
            || run["quads"].as_u64() != Some(case.shells.quads.len() as u64)
        {
            return Err("complete original native discretization/history required".into());
        }
        for key in ["pulled_center_dx_m", "probe_sigma_xx_pa"] {
            let values = run[key].as_array().ok_or("missing native values")?;
            if values.len() != times.len()
                || values
                    .iter()
                    .any(|v| v.as_f64().is_none_or(|x| !x.is_finite()))
            {
                return Err("complete finite native physical channels required".into());
            }
        }
        let stop = format!("history_{i}_stop_time_error_s");
        let end = number(run, "final_time_s")?;
        let alignment = format!("history_{i}_final_sample_alignment_error_s");
        let residual = format!("history_{i}_true_dynamic_residual");
        result = result
            .metric(&stop, (end - case.base.stop_time_s).abs())
            .maximum(&stop, 1e-12)
            .metric(
                &alignment,
                (times
                    .last()
                    .and_then(|v| v.as_f64())
                    .ok_or("missing final native time")?
                    - end)
                    .abs(),
            )
            .maximum(&alignment, 1e-12)
            .metric(&residual, number(run, "maximum_true_dynamic_residual")?)
            .maximum(&residual, 1e-8);
    }
    let ratio = number(&runs[1], "steps")? / number(&runs[0], "steps")?;
    result = result
        .metric("native_timestep_step_ratio", ratio)
        .minimum("native_timestep_step_ratio", 1.5);
    for (index, key) in ["pulled_center_dx_m", "probe_sigma_xx_pa", "thickness_m"]
        .iter()
        .enumerate()
    {
        let measured = external.channel(&case.base.probes[index].channel)?;
        let sampled = runs
            .iter()
            .map(|r| {
                external
                    .times
                    .iter()
                    .map(|t| {
                        if index == 2 {
                            number(r, key)
                        } else {
                            sample(r, key, *t)
                        }
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .collect::<Result<Vec<_>, _>>()?;
        let fine = *sampled[1].last().ok_or("missing final native probe")?;
        let coarse = *sampled[0].last().ok_or("missing final coarse probe")?;
        let reference = *measured.last().ok_or("missing final reference probe")?;
        result = result
            .metric(format!("native_{key}"), fine)
            .metric(format!("external_{key}"), reference);
        let parity_budget = if index == 2 { 1e-6 } else { 0.03 };
        for (suffix, value, budget) in [
            (
                "history_parity_relative_error",
                relative(&sampled[1], measured)?,
                parity_budget,
            ),
            (
                "parity_relative_error",
                relative(&[fine], &[reference])?,
                parity_budget,
            ),
            (
                "history_temporal_relative_change",
                relative(&sampled[0], &sampled[1])?,
                0.003,
            ),
            ("temporal_change", relative(&[coarse], &[fine])?, 0.003),
            (
                "analytical_relative_error",
                relative(&[fine], &[case.base.probes[index].expected])?,
                case.base.probes[index].rel_tol,
            ),
        ] {
            let metric = format!("{key}_{suffix}");
            result = result.metric(&metric, value).maximum(&metric, budget);
        }
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
    fn result_rejects_false_full_history_evidence() {
        let case = super::super::shell_oht_elastic_case();
        let expected = case
            .base
            .probes
            .iter()
            .map(|p| p.expected)
            .collect::<Vec<_>>();
        let history = sonium_qa::rad::TimeHistory {
            times: vec![0.0, 0.001, 0.002],
            channels: case
                .base
                .probes
                .iter()
                .enumerate()
                .map(|(i, p)| {
                    (
                        p.channel.clone(),
                        if i == 2 {
                            vec![expected[i]; 3]
                        } else {
                            vec![0.0, expected[i] * 0.5, expected[i]]
                        },
                    )
                })
                .collect(),
        };
        let run = serde_json::json!({"times_s":[0.0,0.001,0.002],"pulled_center_dx_m":[0.0,expected[0]*0.5,expected[0]],
            "probe_sigma_xx_pa":[0.0,expected[1]*0.5,expected[1]],"thickness_m":expected[2],"steps":2000,
            "refinement_factor":1,"quads":3900,"maximum_true_dynamic_residual":1e-12,"final_time_s":0.002});
        let mut good = [run.clone(), run.clone()];
        good[1]["steps"] = serde_json::json!(4000);
        assert!(
            build_result(&case, "test", "test", &history, true, &good)
                .unwrap()
                .release_eligible()
                .is_ok()
        );
        assert!(
            build_result(&case, "test", "test", &history, false, &good)
                .unwrap()
                .release_eligible()
                .is_err()
        );
        for (key, value) in [
            ("maximum_true_dynamic_residual", serde_json::json!(1e-3)),
            ("final_time_s", serde_json::json!(0.001)),
            ("thickness_m", serde_json::json!(0.002)),
            ("steps", serde_json::json!(2000)),
        ] {
            let mut bad = good.clone();
            bad[1][key] = value;
            assert!(
                build_result(&case, "test", "test", &history, true, &bad)
                    .unwrap()
                    .release_eligible()
                    .is_err()
            );
        }
        let mut bad = good.clone();
        bad[1]["probe_sigma_xx_pa"] = serde_json::json!([0.0, expected[1] * 2.0, expected[1]]);
        let r = build_result(&case, "test", "test", &history, true, &bad).unwrap();
        assert!(r.metrics["probe_sigma_xx_pa_parity_relative_error"] < 1e-12);
        assert!(
            r.release_eligible().is_err(),
            "Matching final value must not hide a wrong intermediate history"
        );
        for key in ["times_s", "maximum_true_dynamic_residual"] {
            let mut missing = good.clone();
            missing[1].as_object_mut().unwrap().remove(key);
            assert!(build_result(&case, "test", "test", &history, true, &missing).is_err());
        }
        let mut bad = good.clone();
        bad[1]["refinement_factor"] = serde_json::json!(3);
        assert!(build_result(&case, "test", "test", &history, true, &bad).is_err());
        let mut malformed = history.clone();
        malformed.times[1] = 0.002;
        assert!(build_result(&case, "test", "test", &malformed, true, &good).is_err());
        let mut altered = case.clone();
        altered.base.cloads[0].force_scale_n *= 0.5;
        assert!(parity_result(&altered, "test", "test", &history, true).is_err());
        assert!(sample(&run, "pulled_center_dx_m", 0.003).is_err());
    }

    #[test]
    fn subdivision_retains_original_domain_grip_forces_and_stress_point() {
        let (old, original_point) = subdivided_case(1).unwrap();
        let area = |c: &RadExtendedCase| {
            let mut sum = 0.0;
            let mut correction = 0.0;
            for q in &c.shells.quads {
                let a = c.base.mesh.nodes[q[0] - 1];
                let b = c.base.mesh.nodes[q[2] - 1];
                let value = (b[0] - a[0]) * (b[1] - a[1]) - correction;
                let next = sum + value;
                correction = (next - sum) - value;
                sum = next;
            }
            sum
        };
        for factor in [2, 3] {
            let (new, point) = subdivided_case(factor).unwrap();
            assert_eq!(point, original_point);
            assert_eq!(
                new.shells.quads.len(),
                old.shells.quads.len() * factor * factor
            );
            assert!((area(&old) - area(&new)).abs() < 1e-13 * area(&old));
            for set in [1, 2, 3] {
                assert_eq!(
                    old.base.node_sets[set]
                        .nodes
                        .iter()
                        .map(|n| old.base.mesh.nodes[n - 1])
                        .collect::<Vec<_>>(),
                    new.base.node_sets[set]
                        .nodes
                        .iter()
                        .map(|n| new.base.mesh.nodes[n - 1])
                        .collect::<Vec<_>>()
                );
            }
            assert_eq!(new.base.node_sets[0].nodes.len(), 40 * factor + 1);
            assert!(
                new.base.node_sets[0]
                    .nodes
                    .iter()
                    .all(|n| new.base.mesh.nodes[n - 1][0] == 0.0)
            );
            assert_eq!(
                format!("{:?}", old.base.cloads),
                format!("{:?}", new.base.cloads)
            );
            assert_eq!(format!("{:?}", old.base.bcs), format!("{:?}", new.base.bcs));
            assert_eq!(
                format!("{:?}", old.base.functions),
                format!("{:?}", new.base.functions)
            );
            assert_eq!(
                format!("{:?}", old.base.material),
                format!("{:?}", new.base.material)
            );
            assert_eq!(
                format!("{:?}", old.shell_prop),
                format!("{:?}", new.shell_prop)
            );
            assert_eq!(old.base.stop_time_s, new.base.stop_time_s);
            assert_eq!(old.base.th_period_s, new.base.th_period_s);
        }
        assert!(subdivided_case(0).is_err());
    }

    #[test]
    fn compact_preserves_physical_connectivity_loads_supports_and_probe_coordinates() {
        let original = super::super::shell_oht_elastic_case();
        let mut reduced = original.clone();
        compact(&mut reduced).unwrap();
        assert!(reduced.base.mesh.nodes.len() < original.base.mesh.nodes.len());
        for (a, b) in original.shells.quads.iter().zip(&reduced.shells.quads) {
            assert_eq!(
                a.map(|n| original.base.mesh.nodes[n - 1]),
                b.map(|n| reduced.base.mesh.nodes[n - 1])
            );
        }
        for (a, b) in original.base.node_sets.iter().zip(&reduced.base.node_sets) {
            assert_eq!(
                a.nodes
                    .iter()
                    .map(|n| original.base.mesh.nodes[n - 1])
                    .collect::<Vec<_>>(),
                b.nodes
                    .iter()
                    .map(|n| reduced.base.mesh.nodes[n - 1])
                    .collect::<Vec<_>>()
            );
        }
        assert_eq!(
            format!("{:?}", original.base.cloads),
            format!("{:?}", reduced.base.cloads)
        );
        assert_eq!(
            format!("{:?}", original.base.bcs),
            format!("{:?}", reduced.base.bcs)
        );
        assert_eq!(
            format!("{:?}", original.base.functions),
            format!("{:?}", reduced.base.functions)
        );
        assert_eq!(original.base.stop_time_s, reduced.base.stop_time_s);
        assert_eq!(original.base.th_period_s, reduced.base.th_period_s);
        assert_eq!(
            format!("{:?}", original.base.material),
            format!("{:?}", reduced.base.material)
        );
        assert_eq!(
            format!("{:?}", original.shell_prop),
            format!("{:?}", reduced.shell_prop)
        );
        assert_eq!(
            original.base.mesh.nodes[original.base.th_nodes[0] - 1],
            reduced.base.mesh.nodes[reduced.base.th_nodes[0] - 1]
        );
    }

    #[test]
    fn stress_recovers_constant_strain_and_rigid_motion_in_pascal() {
        let points = [
            [0.0, 0.0, 0.0],
            [2.0, 0.0, 0.0],
            [2.0, 3.0, 0.0],
            [0.0, 3.0, 0.0],
        ];
        let mut u = [0.0; 24];
        for (i, p) in points.iter().enumerate() {
            u[6 * i] = 0.001 * p[0] - 0.0003 * p[1] + 0.4;
            u[6 * i + 1] = -0.0002 * p[1] + 0.0003 * p[0] - 0.7;
            u[6 * i + 5] = 0.0003;
        }
        let expected = 200e9 / (1.0 - 0.3 * 0.3) * (0.001 - 0.3 * 0.0002);
        assert!((center_stress(&points, &u, 200e9, 0.3) - expected).abs() < 1e-12 * expected);
        for (xi, eta) in [(-1.0, 1.0), (0.3, -0.7), (1.0, -1.0)] {
            assert!(
                (stress_at(&points, &u, 200e9, 0.3, xi, eta) - expected).abs() < 1e-12 * expected
            );
        }
        // A drilling edge difference produces the independently derived
        // normal bubble derivatives, rather than dropping enriched strains.
        let mut drilling = [0.0; 24];
        drilling[11] = 0.01;
        let ex = -3.0 / (8.0 * 2.0) * 0.01;
        let ey = 2.0 / (8.0 * 3.0) * 0.01;
        let expected = 200e9 / (1.0 - 0.3 * 0.3) * (ex + 0.3 * ey);
        assert!(
            (center_stress(&points, &drilling, 200e9, 0.3) - expected).abs()
                < 1e-12 * expected.abs()
        );
    }
}
