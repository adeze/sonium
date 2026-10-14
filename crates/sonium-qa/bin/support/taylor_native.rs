//! Complete copper Taylor mushroom-profile and plastic-field parity.

// Rust guideline compliant 2026-02-21

#[path = "../oc_bench.rs"]
#[expect(
    dead_code,
    reason = "Only actual full native plastic integration is used"
)]
mod engine;

use super::taylor_full_fixture::{
    STOP, full_case, isothermal_starter, viscous_starter, write_decks,
};
pub(crate) use engine::{NativePlasticityHistory, taylor_plastic_history_sampled};
use sha2::{Digest, Sha256};
use sonium_qa::rad::{
    OpenCourantCase, RadExtendedCase, TimeHistory, bric_channel, engine_exe, probes_json,
    run_extended_case, write_engine, write_extended_engine, write_starter,
};
use sonium_qa::{QaResult, QaStatus};

struct Profile {
    time_s: f64,
    node_ids: Vec<i64>,
    positions_m: Vec<[f64; 3]>,
    plastic: Vec<f64>,
}

fn reference_profile(
    case: &RadExtendedCase,
    history: &TimeHistory,
) -> Result<Profile, Box<dyn std::error::Error>> {
    let positions_m = case
        .base
        .th_nodes
        .iter()
        .map(|&id| {
            let mut p = case.base.mesh.nodes[id - 1];
            for (j, tag) in ["DX", "DY", "DZ"].iter().enumerate() {
                p[j] += history.final_value(&format!("{tag}/{id}"))?;
            }
            Ok(p)
        })
        .collect::<Result<Vec<_>, sonium_qa::rad::RadError>>()?;
    let plastic = (1..=case.base.mesh.bricks.len())
        .map(|id| history.final_value(&bric_channel("PLAS", id)))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(Profile {
        time_s: *history.times.last().ok_or("missing reference time")?,
        node_ids: case.base.th_nodes.iter().map(|&id| id as i64).collect(),
        positions_m,
        plastic,
    })
}

fn relative(a: &[f64], b: &[f64]) -> Result<f64, Box<dyn std::error::Error>> {
    if a.len() != b.len() || a.is_empty() || a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err("aligned finite full physical fields required".into());
    }
    let denominator = b.iter().map(|v| v * v).sum::<f64>();
    if denominator <= 0.0 || !denominator.is_finite() {
        return Err("nonzero finite independent physical field required".into());
    }
    Ok((a.iter().zip(b).map(|(x, y)| (x - y).powi(2)).sum::<f64>() / denominator).sqrt())
}

// Compare displacement changes rather than the undeformed radius or bar length.
// Cartesian in-plane expansion also detects spurious azimuthal deformation.
fn fields(
    case: &RadExtendedCase,
    profile: &Profile,
) -> Result<[Vec<f64>; 3], Box<dyn std::error::Error>> {
    if profile.node_ids
        != case
            .base
            .th_nodes
            .iter()
            .map(|&id| id as i64)
            .collect::<Vec<_>>()
        || profile.positions_m.len() != profile.node_ids.len()
        || profile.plastic.len() != case.base.mesh.bricks.len()
        || profile
            .positions_m
            .iter()
            .flatten()
            .chain(&profile.plastic)
            .any(|v| !v.is_finite())
        || profile.plastic.iter().any(|v| *v < 0.0)
    {
        return Err(
            "complete original-node mushroom and original-element plastic fields required".into(),
        );
    }
    let mut expansion = Vec::with_capacity(2 * profile.node_ids.len());
    let mut compression = Vec::with_capacity(profile.node_ids.len());
    let speed = case.base.walls[0]
        .velocity_m_per_s
        .ok_or("original impact velocity required")?[2];
    for (&id, p) in profile.node_ids.iter().zip(&profile.positions_m) {
        let original = case.base.mesh.nodes[id as usize - 1];
        expansion.extend([p[0] - original[0], p[1] - original[1]]);
        compression.push(p[2] - speed * profile.time_s - original[2]);
    }
    Ok([expansion, compression, profile.plastic.clone()])
}

fn shape(profile: &Profile) -> [f64; 3] {
    [
        profile
            .positions_m
            .iter()
            .map(|p| p[0].hypot(p[1]))
            .fold(0.0, f64::max),
        profile
            .positions_m
            .iter()
            .map(|p| p[2])
            .fold(f64::NEG_INFINITY, f64::max)
            - profile
                .positions_m
                .iter()
                .map(|p| p[2])
                .fold(f64::INFINITY, f64::min),
        profile.plastic.iter().copied().fold(0.0, f64::max),
    ]
}

fn compare(
    case: &RadExtendedCase,
    id: &str,
    label: &str,
    reference: &Profile,
    healthy: bool,
    runs: &[NativePlasticityHistory; 2],
) -> Result<QaResult, Box<dyn std::error::Error>> {
    let canonical = isothermal_starter(case)?;
    let deck = write_extended_engine(case);
    let input = format!(
        "{:x}",
        Sha256::digest(format!("{canonical}{deck}").as_bytes())
    );
    let rendered = viscous_starter(&canonical)?;
    let ref_checksum = format!(
        "{:x}",
        Sha256::digest(format!("{rendered}{deck}").as_bytes())
    );
    if !reference.time_s.is_finite()
        || reference.time_s < STOP - 1.05 * case.base.th_period_s
        || reference.time_s > STOP
    {
        return Err("complete original-duration external plastic history required".into());
    }
    let external_fields = fields(case, reference)?;
    let external_shape = shape(reference);
    let mut result=QaResult::passed(id).requested_backend("cpu+opencourant").actual_backend("cpu+opencourant")
        .metric("external_reference_executed",1.0).minimum("external_reference_executed",1.0)
        .metric("external_reference_health_passed",f64::from(healthy)).minimum("external_reference_health_passed",1.0)
        .metric("full_plastic_reference_executed",1.0).minimum("full_plastic_reference_executed",1.0)
        .metric("native_parity_executed",1.0).minimum("native_parity_executed",1.0)
        .metric("native_complete_plastic_histories",2.0).minimum("native_complete_plastic_histories",2.0)
        .metric("external_history_coverage",reference.time_s/STOP).minimum("external_history_coverage",0.98)
        .metric("native_timestep_step_ratio",runs[1].steps as f64/runs[0].steps as f64).minimum("native_timestep_step_ratio",1.5)
        .metric("profile_comparison_time_s",reference.time_s)
        .input_checksum(input).reference_checksum(ref_checksum).reference(format!("{label}; independent complete copper plastic impact, viscous Belytschko reference"))
        .dimension("Original L25.4mm/R3.175mm copper bar, E117GPa/nu.35/rho8930,227m/s frictionless moving anvil/free bar; full80us, 24 axial by18 cross-section grid,5184 physical bricks/6225 physical nodes")
        .formulation("Original 4us elastic reference health retained. Full isothermal Johnson-Cook copper (A90MPa/B292MPa/n.31/C.025/ref1s^-1), no erosion, complete native objective Cauchy-stress evolution and full independent external impact. Full Cartesian in-plane expansion, Galilean axial compression and equivalent plastic-strain fields at exactly the actual external timestamp; all original node/element identities retained. Native/reference viscous stabilization formulas differ; quantitative numerical parity measured, no coefficient identity, continuum spatial convergence, experimental replay or thermal-accuracy claim.")
        .tolerance("Full displacement/plastic fields and peak radius/length/plastic strain <=3%; half-CFL full fields and peaks <=.3%; exact80us native stop, complete finite external history, no erosion, numerical energy drift <=1%; actual plastic mushroom expansion >=1.5x initial radius, length <=.75x initial, peak plastic strain >=.5.");
    let mut profiles = Vec::new();
    for (i, run) in runs.iter().enumerate() {
        if run.starter_sha256 != format!("{:x}", Sha256::digest(canonical.as_bytes()))
            || run.engine_deck_sha256 != format!("{:x}", Sha256::digest(deck.as_bytes()))
            || run.node_ids != reference.node_ids
            || run.element_ids
                != (1..=case.base.mesh.bricks.len())
                    .map(|id| id as i64)
                    .collect::<Vec<_>>()
            || run.times_s.len() < 2
            || run.times_s[0] != 0.0
            || run.times_s.iter().any(|t| !t.is_finite())
            || run.times_s.windows(2).any(|w| w[1] <= w[0])
            || run.probe_channels
                != case
                    .base
                    .probes
                    .iter()
                    .map(|p| p.channel.clone())
                    .collect::<Vec<_>>()
            || run.probe_values.len() != case.base.probes.len()
            || run
                .probe_values
                .iter()
                .any(|v| v.len() != run.times_s.len() || v.iter().any(|x| !x.is_finite()))
            || run.profile_sample_time_s != Some(reference.time_s)
            || run.steps == 0
            || !run.relative_energy_drift.is_finite()
            || run.relative_energy_drift < 0.0
        {
            return Err("actual complete original-physics native integration and matching-time profile required".into());
        }
        let profile = Profile {
            time_s: reference.time_s,
            node_ids: run.node_ids.clone(),
            positions_m: run.profile_sample_positions_m.clone(),
            plastic: run.profile_sample_element_plastic_strain.clone(),
        };
        fields(case, &profile)?;
        for (suffix, value, budget) in [
            ("stop_time_error_s", (run.final_time_s - STOP).abs(), 1e-12),
            (
                "final_trace_alignment_error_s",
                (run.times_s.last().ok_or("missing final native trace")? - run.final_time_s).abs(),
                1e-12,
            ),
            ("energy_drift", run.relative_energy_drift, 0.01),
            ("eroded_elements", run.eroded_elements as f64, 0.0),
        ] {
            let metric = format!("native_{i}_{suffix}");
            result = result.metric(&metric, value).maximum(&metric, budget);
        }
        profiles.push(profile);
    }
    let coarse = fields(case, &profiles[0])?;
    let fine = fields(case, &profiles[1])?;
    for (i, name) in ["inplane_expansion", "axial_compression", "plastic_strain"]
        .iter()
        .enumerate()
    {
        let parity = format!("{name}_field_relative_error");
        let temporal = format!("{name}_field_temporal_change");
        result = result
            .metric(&parity, relative(&fine[i], &external_fields[i])?)
            .maximum(&parity, 0.03)
            .metric(&temporal, relative(&coarse[i], &fine[i])?)
            .maximum(&temporal, 0.003);
    }
    let native_shape = shape(&profiles[1]);
    let coarse_shape = shape(&profiles[0]);
    for (i, name) in [
        "mushroom_radius",
        "remaining_length",
        "maximum_plastic_strain",
    ]
    .iter()
    .enumerate()
    {
        let parity = format!("{name}_relative_error");
        let temporal = format!("{name}_temporal_change");
        result = result
            .metric(format!("native_{name}"), native_shape[i])
            .metric(format!("external_{name}"), external_shape[i])
            .metric(&parity, relative(&[native_shape[i]], &[external_shape[i]])?)
            .maximum(&parity, 0.03)
            .metric(&temporal, relative(&[coarse_shape[i]], &[native_shape[i]])?)
            .maximum(&temporal, 0.003);
    }
    for (name, measure) in [("native", native_shape), ("external", external_shape)] {
        let expansion = format!("{name}_radius_expansion_ratio");
        let compression = format!("{name}_remaining_length_ratio");
        let plastic = format!("{name}_plastic_mushroom_peak");
        result = result
            .metric(&expansion, measure[0] / 0.003175)
            .minimum(&expansion, 1.5)
            .metric(&compression, measure[1] / 0.0254)
            .maximum(&compression, 0.75)
            .metric(&plastic, measure[2])
            .minimum(&plastic, 0.5);
    }
    if let Err(error) = result.release_eligible() {
        result.status = QaStatus::Failed;
        result.message = Some(error);
    }
    Ok(result)
}

/// Execute complete matching native and independent plastic Taylor impact.
///
/// # Errors
/// Rejects changed original physics, failed original elastic reference health,
/// incomplete external/native integration and missing finite full physical fields.
pub(crate) fn parity_result(
    original: &OpenCourantCase,
    id: &str,
    label: &str,
    healthy: bool,
) -> Result<QaResult, Box<dyn std::error::Error>> {
    let expected = super::taylor_bar_case();
    if write_starter(original) != write_starter(&expected)
        || write_engine(original) != write_engine(&expected)
        || probes_json(original, "identity") != probes_json(&expected, "identity")
    {
        return Err("original Taylor physical fixture changed".into());
    }
    if !healthy {
        return Err("original elastic Taylor reference health failed".into());
    }
    let case = full_case(3);
    case.validate()?;
    let work = std::env::temp_dir().join(format!(
        "p2_taylor_full_original_gate_{}",
        std::process::id()
    ));
    let external_work = work.join("reference");
    let native_work = work.join("native");
    write_decks(&case, &external_work, true)?;
    write_decks(&case, &native_work, false)?;
    let exe = engine_exe().ok_or("OpenCourant required for complete Taylor reference")?;
    println!("full original copper plastic Taylor: independent80us reference starting");
    let external = run_extended_case(&external_work, &exe, &case)?;
    let profile = reference_profile(&case, &external.history)?;
    let runs = [
        taylor_plastic_history_sampled(&native_work, 1.0, 1.0, true, Some(profile.time_s))?,
        taylor_plastic_history_sampled(&native_work, 0.5, 1.0, true, Some(profile.time_s))?,
    ];
    compare(&case, id, label, &profile, healthy, &runs)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn measured_profile_budgets_reject_false_plastic_and_time_evidence() {
        let case = full_case(1);
        let time = 79.9e-6;
        let ids = case
            .base
            .th_nodes
            .iter()
            .map(|&id| id as i64)
            .collect::<Vec<_>>();
        let positions = ids
            .iter()
            .map(|&id| {
                let p = case.base.mesh.nodes[id as usize - 1];
                [2.0 * p[0], 2.0 * p[1], 0.6 * p[2] + 227.0 * time]
            })
            .collect::<Vec<_>>();
        let ep = vec![0.8; case.base.mesh.bricks.len()];
        let reference = Profile {
            time_s: time,
            node_ids: ids.clone(),
            positions_m: positions.clone(),
            plastic: ep.clone(),
        };
        let canonical = isothermal_starter(&case).unwrap();
        let deck = write_extended_engine(&case);
        let run = NativePlasticityHistory {
            probe_channels: case.base.probes.iter().map(|p| p.channel.clone()).collect(),
            times_s: vec![0.0, STOP],
            probe_values: vec![vec![0.0, 1.0]; case.base.probes.len()],
            element_ids: (1..=case.base.mesh.bricks.len())
                .map(|n| n as i64)
                .collect(),
            steps: 10,
            final_time_s: STOP,
            starter_sha256: format!("{:x}", Sha256::digest(canonical.as_bytes())),
            engine_deck_sha256: format!("{:x}", Sha256::digest(deck.as_bytes())),
            node_ids: ids,
            profile_sample_time_s: Some(time),
            profile_sample_positions_m: positions,
            profile_sample_element_plastic_strain: ep,
            ..NativePlasticityHistory::default()
        };
        let mut good = [run.clone(), run];
        good[1].steps = 20;
        assert!(
            compare(&case, "test", "test", &reference, true, &good)
                .unwrap()
                .release_eligible()
                .is_ok()
        );
        assert!(
            compare(&case, "test", "test", &reference, false, &good)
                .unwrap()
                .release_eligible()
                .is_err()
        );
        for index in 0..4 {
            let mut bad = good.clone();
            match index {
                0 => bad[1].relative_energy_drift = 0.1,
                1 => bad[1].eroded_elements = 1,
                2 => bad[1].final_time_s = STOP / 2.0,
                _ => bad[1].profile_sample_element_plastic_strain.fill(0.0),
            }
            assert!(
                !compare(&case, "test", "test", &reference, true, &bad)
                    .is_ok_and(|result| result.release_eligible().is_ok())
            );
        }
        let mut bad = good.clone();
        bad[1]
            .profile_sample_positions_m
            .iter_mut()
            .for_each(|p| p[0] *= 1.5);
        assert!(
            compare(&case, "test", "test", &reference, true, &bad)
                .unwrap()
                .release_eligible()
                .is_err()
        );
        let mut bad = good.clone();
        bad[1].profile_sample_time_s = Some(time + 1e-7);
        assert!(compare(&case, "test", "test", &reference, true, &bad).is_err());
        let mut bad = good.clone();
        bad[1].profile_sample_element_plastic_strain.pop();
        assert!(compare(&case, "test", "test", &reference, true, &bad).is_err());
    }

    #[test]
    fn changed_original_fixture_rejected_before_execution() {
        let mut case = super::super::taylor_bar_case();
        case.material.youngs_modulus_pa *= 0.5;
        assert!(
            parity_result(&case, "test", "none", true)
                .unwrap_err()
                .to_string()
                .contains("physical fixture changed")
        );
        assert!(parity_result(&super::super::taylor_bar_case(), "test", "none", false).is_err());
    }
    #[test]
    fn full_fields_reject_zero_reference_missing_nodes_and_nonfinite_strain() {
        assert!(relative(&[0.0], &[0.0]).is_err());
        assert!(relative(&[1.0], &[1.0, 2.0]).is_err());
        let case = full_case(1);
        let mut p = Profile {
            time_s: 79.9e-6,
            node_ids: vec![],
            positions_m: vec![],
            plastic: vec![],
        };
        assert!(fields(&case, &p).is_err());
        p.node_ids = case.base.th_nodes.iter().map(|&n| n as i64).collect();
        p.positions_m = p
            .node_ids
            .iter()
            .map(|&n| case.base.mesh.nodes[n as usize - 1])
            .collect();
        p.plastic = vec![0.5; case.base.mesh.bricks.len()];
        assert!(fields(&case, &p).is_ok());
        p.plastic[0] = f64::NAN;
        assert!(fields(&case, &p).is_err());
        p.plastic[0] = -1.0;
        assert!(fields(&case, &p).is_err());
    }
}
