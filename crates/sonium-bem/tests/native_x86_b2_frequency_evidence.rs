//! Native-x86 B2 frequency/ka limitation map for the production MLFMM path.
//!
//! This packet deliberately keeps the fixed production MLFMM expansion visible
//! across frequency.  Each row compares the MLFMM action and checked physical
//! field against an independently assembled dense CPU reference.  Rows above
//! the low-ka budgets are valid limitation observations, not parity claims.

#![cfg(target_arch = "x86_64")]

use num_complex::Complex64;
use serde::Serialize;
use sonium_bem::core::assembly::mlfmm::{
    build_cluster_tree, build_mlfmm_system_stabilized_checked,
};
use sonium_bem::core::assembly::tbem::build_tbem_system_stabilized;
use sonium_bem::core::postprocess::pressure::compute_total_field_checked;
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::core::{BasisOrder, IncidentField, PhysicsParams};

const SCHEMA: &str = "sonium-bem-native-x86-b2-frequency/v1";
const ELEMENT_COUNT: usize = 16;
const PATCH_SEPARATION: f64 = 12.0;
const CHARACTERISTIC_LENGTH: f64 = 0.8;
const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.21;
const TARGET_ELEMENTS_PER_LEAF: usize = 2;
const MLFMM_ACTION_BUDGET: f64 = 1.0e-2;
const MLFMM_FIELD_BUDGET: f64 = 1.0e-3;
const FREQUENCIES_HZ: [f64; 7] = [20.0, 42.0, 60.0, 80.0, 100.0, 140.0, 180.0];

#[derive(Debug, Serialize)]
struct FrequencyArtifact {
    schema: &'static str,
    status: &'static str,
    target_arch: &'static str,
    os: &'static str,
    cpu_model: String,
    microcode: String,
    compiler: String,
    features: String,
    selected_kernel: &'static str,
    runtime_dispatch: &'static str,
    thread_policy: String,
    affinity_policy: String,
    numa_policy: String,
    deterministic_mode: String,
    execution: &'static str,
    cuda_claim: bool,
    avx_claim: bool,
    patch_separation_m: f64,
    characteristic_length_m: f64,
    element_count: usize,
    dof_count: usize,
    field_point_count: usize,
    mlfmm_action_budget: f64,
    mlfmm_field_budget: f64,
    rows: Vec<FrequencyRow>,
    validation: Validation,
}

#[derive(Debug, Serialize)]
struct FrequencyRow {
    frequency_hz: f64,
    wave_number: f64,
    ka: f64,
    mlfmm_levels: usize,
    mlfmm_d_matrices: usize,
    dense_field_energy: f64,
    mlfmm_field_energy: f64,
    mlfmm_action_residual: f64,
    mlfmm_field_residual: f64,
    mlfmm_field_energy_residual: f64,
    status: &'static str,
}

#[derive(Debug, Serialize)]
struct Validation {
    required_frequencies_hz: Vec<f64>,
    low_ka_action_budget: f64,
    low_ka_field_budget: f64,
    bounded_rows: usize,
    exceeded_rows: usize,
    no_transfer_rows: usize,
    status: &'static str,
}

fn metadata(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "unspecified".to_owned())
}

fn cpuinfo(field: &str) -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                (key.trim() == field).then(|| value.trim().to_owned())
            })
        })
        .unwrap_or_else(|| "unknown".to_owned())
}

fn compiler() -> String {
    std::process::Command::new("rustc")
        .arg("-vV")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .and_then(|text| text.lines().next().map(str::to_owned))
        .unwrap_or_else(|| "unknown".to_owned())
}

fn relative_error(actual: &[Complex64], expected: &[Complex64]) -> f64 {
    if actual.len() != expected.len()
        || actual
            .iter()
            .chain(expected)
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return f64::INFINITY;
    }
    let numerator = actual
        .iter()
        .zip(expected)
        .map(|(actual, expected)| (*actual - *expected).norm_sqr())
        .sum::<f64>()
        .sqrt();
    let denominator = expected
        .iter()
        .map(Complex64::norm_sqr)
        .sum::<f64>()
        .sqrt()
        .max(f64::MIN_POSITIVE);
    numerator / denominator
}

fn physics(frequency_hz: f64) -> PhysicsParams {
    PhysicsParams::try_new(frequency_hz, SPEED_OF_SOUND, DENSITY, false)
        .expect("valid native B2 frequency-map physics")
}

fn element(index: usize) -> Element {
    let patch = index / 8;
    let local = index % 8;
    let x = patch as f64 * PATCH_SEPARATION + local as f64 * 1.1;
    let base = index * 3;
    Element {
        connectivity: vec![base, base + 1, base + 2],
        element_type: ElementType::Tri3,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 3],
        center: [
            x + CHARACTERISTIC_LENGTH / 3.0,
            CHARACTERISTIC_LENGTH / 3.0,
            0.0,
        ],
        area: CHARACTERISTIC_LENGTH * CHARACTERISTIC_LENGTH / 2.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::default()]),
        group: 0,
        dof_addresses: vec![index],
    }
}

fn fixture() -> (Vec<Element>, Vec<[f64; 3]>) {
    let elements: Vec<_> = (0..ELEMENT_COUNT).map(element).collect();
    let nodes: Vec<_> = (0..ELEMENT_COUNT)
        .flat_map(|index| {
            let patch = index / 8;
            let local = index % 8;
            let x = patch as f64 * PATCH_SEPARATION + local as f64 * 1.1;
            [
                [x, 0.0, 0.0],
                [x + CHARACTERISTIC_LENGTH, 0.0, 0.0],
                [x, CHARACTERISTIC_LENGTH, 0.0],
            ]
        })
        .collect();
    (elements, nodes)
}

fn probe_vector(length: usize) -> Vec<Complex64> {
    (0..length)
        .map(|index| Complex64::new(0.25 + index as f64 * 0.013, -0.2 + index as f64 * 0.009))
        .collect()
}

fn field_points() -> [[f64; 3]; 4] {
    [
        [0.0, 0.0, 4.0],
        [5.5, 1.5, 3.0],
        [12.0, -1.0, 4.5],
        [20.0, 1.0, 3.5],
    ]
}

fn field_values(
    points: &[[f64; 3]],
    elements: &[Element],
    nodes: &[[f64; 3]],
    surface_pressure: &[Complex64],
    physics: &PhysicsParams,
) -> Vec<Complex64> {
    compute_total_field_checked(
        points,
        elements,
        nodes,
        surface_pressure,
        None,
        &IncidentField::plane_wave_z(),
        physics,
        BasisOrder::Constant,
    )
    .expect("checked B2 frequency-map field evaluation")
    .into_iter()
    .map(|field| field.p_total)
    .collect()
}

fn field_energy(values: &[Complex64]) -> f64 {
    values.iter().map(Complex64::norm_sqr).sum()
}

fn row_status(d_matrices: usize, action: f64, field: f64) -> &'static str {
    if d_matrices == 0 {
        "no-transfer-data"
    } else if action <= MLFMM_ACTION_BUDGET && field <= MLFMM_FIELD_BUDGET {
        "within-low-ka-budget"
    } else {
        "exceeds-low-ka-budget"
    }
}

#[test]
fn native_x86_b2_frequency_limitation_map_v1() {
    // Opt-in evidence producer: skip under the general suite, run only via
    // the dedicated native-x86 script that sets the requirement flag.
    if std::env::var("SONIUM_B2_FREQUENCY_REQUIRE_NATIVE_X86").as_deref() != Ok("1") {
        eprintln!(
            "skipping native B2 frequency evidence: set SONIUM_B2_FREQUENCY_REQUIRE_NATIVE_X86=1 to produce it"
        );
        return;
    }
    assert_eq!(std::env::consts::OS, "linux");

    let (elements, nodes) = fixture();
    let probe = probe_vector(elements.len());
    let points = field_points();
    let mut rows = Vec::with_capacity(FREQUENCIES_HZ.len());
    for &frequency_hz in &FREQUENCIES_HZ {
        let physics = physics(frequency_hz);
        let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
        let levels = build_cluster_tree(&elements, TARGET_ELEMENTS_PER_LEAF, &physics);
        let dense = build_tbem_system_stabilized(&elements, &nodes, &physics, sign, 1.0)
            .matrix
            .matvec(&probe);
        let mlfmm =
            build_mlfmm_system_stabilized_checked(&elements, &nodes, levels, &physics, sign, 1.0)
                .expect("checked B2 frequency-map MLFMM construction");
        let mlfmm_action = mlfmm
            .matvec_checked(&probe)
            .expect("checked B2 frequency-map MLFMM action");
        let mlfmm_action_residual = relative_error(&mlfmm_action, &dense);
        let dense_field = field_values(&points, &elements, &nodes, &dense, &physics);
        let mlfmm_field = field_values(&points, &elements, &nodes, &mlfmm_action, &physics);
        let mlfmm_field_residual = relative_error(&mlfmm_field, &dense_field);
        let dense_field_energy = field_energy(&dense_field);
        let mlfmm_field_energy = field_energy(&mlfmm_field);
        assert!(dense_field_energy.is_finite() && dense_field_energy > 0.0);
        assert!(mlfmm_field_energy.is_finite() && mlfmm_field_energy > 0.0);
        let mlfmm_field_energy_residual =
            (mlfmm_field_energy - dense_field_energy).abs() / dense_field_energy;
        let mlfmm_d_matrices = mlfmm.d_matrices.iter().map(Vec::len).sum::<usize>();
        let status = row_status(
            mlfmm_d_matrices,
            mlfmm_action_residual,
            mlfmm_field_residual,
        );
        rows.push(FrequencyRow {
            frequency_hz,
            wave_number: physics.wave_number,
            ka: physics.wave_number * CHARACTERISTIC_LENGTH,
            mlfmm_levels: mlfmm.num_levels,
            mlfmm_d_matrices,
            dense_field_energy,
            mlfmm_field_energy,
            mlfmm_action_residual,
            mlfmm_field_residual,
            mlfmm_field_energy_residual,
            status,
        });
    }

    assert_eq!(rows.len(), FREQUENCIES_HZ.len());
    assert!(rows.iter().any(|row| row.status == "within-low-ka-budget"));
    assert!(rows.iter().any(|row| row.status == "exceeds-low-ka-budget"));
    assert!(rows.iter().any(|row| row.mlfmm_d_matrices > 0));

    let bounded_rows = rows
        .iter()
        .filter(|row| row.status == "within-low-ka-budget")
        .count();
    let exceeded_rows = rows
        .iter()
        .filter(|row| row.status == "exceeds-low-ka-budget")
        .count();
    let no_transfer_rows = rows
        .iter()
        .filter(|row| row.status == "no-transfer-data")
        .count();
    let artifact = FrequencyArtifact {
        schema: SCHEMA,
        status: "pass",
        target_arch: "x86_64",
        os: "linux-x86_64",
        cpu_model: cpuinfo("model name"),
        microcode: cpuinfo("microcode"),
        compiler: compiler(),
        features: format!(
            "avx2={},fma={},cuda_claim=false,avx_claim=false",
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("fma")
        ),
        selected_kernel: "cpu-reference",
        runtime_dispatch: "selected_kernel=cpu-reference,cuda_claim=false,avx_claim=false",
        thread_policy: metadata("SONIUM_QA_THREAD_POLICY"),
        affinity_policy: metadata("SONIUM_QA_AFFINITY_POLICY"),
        numa_policy: metadata("SONIUM_QA_NUMA_POLICY"),
        deterministic_mode: metadata("SONIUM_QA_DETERMINISTIC_MODE"),
        execution: "cpu-reference-only",
        cuda_claim: false,
        avx_claim: false,
        patch_separation_m: PATCH_SEPARATION,
        characteristic_length_m: CHARACTERISTIC_LENGTH,
        element_count: elements.len(),
        dof_count: probe.len(),
        field_point_count: points.len(),
        mlfmm_action_budget: MLFMM_ACTION_BUDGET,
        mlfmm_field_budget: MLFMM_FIELD_BUDGET,
        rows,
        validation: Validation {
            required_frequencies_hz: FREQUENCIES_HZ.to_vec(),
            low_ka_action_budget: MLFMM_ACTION_BUDGET,
            low_ka_field_budget: MLFMM_FIELD_BUDGET,
            bounded_rows,
            exceeded_rows,
            no_transfer_rows,
            status: "pass",
        },
    };
    eprintln!(
        "sonium_bem_native_x86_b2_frequency={{\"schema\":\"{}\",\"status\":\"{}\",\"rows\":{},\"bounded\":{},\"exceeded\":{},\"no_transfer\":{}}}",
        artifact.schema,
        artifact.status,
        artifact.rows.len(),
        bounded_rows,
        exceeded_rows,
        no_transfer_rows,
    );
    if let Some(directory) = std::env::var_os("SONIUM_OUTPUT_DIR") {
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).expect("create B2 frequency evidence directory");
        let bytes = serde_json::to_vec_pretty(&artifact).expect("serialize B2 frequency evidence");
        std::fs::write(
            directory.join("sonium_bem_native_x86_b2_frequency.json"),
            bytes,
        )
        .expect("write B2 frequency evidence");
    }
}
