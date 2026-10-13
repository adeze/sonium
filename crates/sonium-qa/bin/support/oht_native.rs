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

/// Compare full native open-hole histories against all original external probes.
///
/// # Errors
/// Returns an error for changed original fixture identity, incomplete external
/// measurements, failed native transients, or absent finite physical observations.
pub(super) fn parity_result(
    case: &RadExtendedCase,
    case_id: &str,
    engine: &str,
    external: &[f64],
    time: f64,
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
    if external.len() != 3
        || external.iter().any(|v| !v.is_finite())
        || !time.is_finite()
        || time < case.base.stop_time_s - 1.05 * case.base.th_period_s
        || time > case.base.stop_time_s
    {
        return Err("complete original external probes and final sample required".into());
    }
    let runs = [
        probe(1, 1e-6)?,
        probe(2, 1e-6)?,
        probe(3, 1e-6)?,
        probe(3, 5e-7)?,
    ];
    build_result(
        case,
        case_id,
        engine,
        external,
        time,
        reference_passed,
        &runs,
    )
}

fn build_result(
    case: &RadExtendedCase,
    case_id: &str,
    engine: &str,
    external: &[f64],
    time: f64,
    reference_passed: bool,
    runs: &[serde_json::Value; 4],
) -> Result<sonium_qa::QaResult, Box<dyn std::error::Error>> {
    let mut result=sonium_qa::QaResult::passed(case_id)
        .requested_backend("cpu+opencourant").actual_backend("cpu+opencourant")
        .metric("external_reference_executed",1.0)
        .metric("external_reference_health_passed",if reference_passed{1.0}else{0.0}).minimum("external_reference_health_passed",1.0)
        .metric("native_parity_executed",1.0).minimum("native_parity_executed",1.0)
        .metric("native_complete_shell_histories",4.0).minimum("native_complete_shell_histories",4.0)
        .metric("comparison_time_s",time).reference(engine)
        .dimension("Original 128x32 mm, 1 mm thick homogenized elastic staircase-hole plate; E=52.7 GPa, nu=.24, rho=1600 kg/m3; 3900/15600/35100 native quads")
        .formulation("Full six-DOF covariant MITC4, Allman-enriched physical stress at original fixed shell centroid; original 41 grip-node forces, root constraints and full 1 ms ramp/hold to 2 ms; checked compensated Newmark with independently recomputed physical force balance at every step. Native probes sampled at actual original external final output time.")
        .tolerance("Native/external displacement and stress <=3%; thickness <=1e-6; original analytical compliance <=10%, staircase-hole Howland stress <=20%; spatial changes <=2% then .5%, contraction <=.6; timestep changes <=.3%; every-step physical residual <=1e-8. Elastic precursor only.");
    for (index, key) in ["pulled_center_dx_m", "probe_sigma_xx_pa"]
        .iter()
        .enumerate()
    {
        let v = runs
            .iter()
            .map(|r| sample(r, key, time))
            .collect::<Result<Vec<_>, _>>()?;
        let first = (v[1] - v[0]).abs() / v[1].abs().max(1e-30);
        let last = (v[2] - v[1]).abs() / v[2].abs().max(1e-30);
        result = result
            .metric(format!("native_{key}"), v[2])
            .metric(format!("external_{key}"), external[index]);
        for (suffix, value, bound) in [
            (
                "parity_relative_error",
                (v[2] - external[index]).abs() / external[index].abs().max(1e-30),
                0.03,
            ),
            (
                "analytical_relative_error",
                (v[2] - case.base.probes[index].expected).abs()
                    / case.base.probes[index].expected.abs(),
                case.base.probes[index].rel_tol,
            ),
            ("spatial_first_change", first, 0.02),
            ("spatial_final_change", last, 0.005),
            ("spatial_contraction", last / first.max(1e-15), 0.6),
            (
                "temporal_change",
                (v[2] - v[3]).abs() / v[3].abs().max(1e-30),
                0.003,
            ),
        ] {
            let metric = format!("{key}_{suffix}");
            result = result.metric(&metric, value).maximum(metric, bound);
        }
    }
    let thickness = number(&runs[2], "thickness_m")?;
    result = result
        .metric("native_thickness_m", thickness)
        .metric(
            "thickness_external_relative_error",
            (thickness - external[2]).abs() / external[2].abs().max(1e-30),
        )
        .maximum("thickness_external_relative_error", 1e-6)
        .metric(
            "thickness_analytical_relative_error",
            (thickness - case.base.probes[2].expected).abs() / case.base.probes[2].expected.abs(),
        )
        .maximum(
            "thickness_analytical_relative_error",
            case.base.probes[2].rel_tol,
        );
    for (i, r) in runs.iter().enumerate() {
        let metric = format!("history_{i}_true_dynamic_residual");
        result = result
            .metric(&metric, number(r, "maximum_true_dynamic_residual")?)
            .maximum(metric, 1e-8);
        let metric = format!("history_{i}_stop_time_error_s");
        result = result
            .metric(
                &metric,
                (number(r, "final_time_s")? - case.base.stop_time_s).abs(),
            )
            .maximum(metric, 1e-12);
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
    fn result_rejects_missing_physics_and_failed_measured_budgets() {
        let case = super::super::shell_oht_elastic_case();
        let external = case
            .base
            .probes
            .iter()
            .map(|p| p.expected)
            .collect::<Vec<_>>();
        let run = serde_json::json!({"times_s":[0.0,0.002],"pulled_center_dx_m":[0.0,external[0]],
            "probe_sigma_xx_pa":[0.0,external[1]],"thickness_m":external[2],
            "maximum_true_dynamic_residual":1e-12,"final_time_s":0.002});
        let good = [run.clone(), run.clone(), run.clone(), run.clone()];
        assert!(
            build_result(&case, "test", "test", &external, 0.002, true, &good)
                .unwrap()
                .release_eligible()
                .is_ok()
        );
        assert!(
            build_result(&case, "test", "test", &external, 0.002, false, &good)
                .unwrap()
                .release_eligible()
                .is_err()
        );
        for (key, value) in [
            ("maximum_true_dynamic_residual", serde_json::json!(1e-3)),
            ("final_time_s", serde_json::json!(0.001)),
            ("thickness_m", serde_json::json!(0.002)),
        ] {
            let mut bad = good.clone();
            bad[2][key] = value;
            assert!(
                build_result(&case, "test", "test", &external, 0.002, true, &bad)
                    .unwrap()
                    .release_eligible()
                    .is_err()
            );
        }
        let mut bad = good.clone();
        bad[2]["probe_sigma_xx_pa"] = serde_json::json!([0.0, 2.0 * external[1]]);
        assert!(
            build_result(&case, "test", "test", &external, 0.002, true, &bad)
                .unwrap()
                .release_eligible()
                .is_err()
        );
        let mut missing = good.clone();
        missing[2]
            .as_object_mut()
            .unwrap()
            .remove("maximum_true_dynamic_residual");
        assert!(build_result(&case, "test", "test", &external, 0.002, true, &missing).is_err());
        assert!(sample(&run, "pulled_center_dx_m", 0.003).is_err());
        let mut altered = case.clone();
        altered.base.cloads[0].force_scale_n *= 0.5;
        assert!(parity_result(&altered, "test", "test", &external, 0.002, true).is_err());
        assert!(parity_result(&case, "test", "test", &external, 0.001, true).is_err());
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
