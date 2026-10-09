//! Native-x86 B2 physical CPU-reference evidence.
//!
//! This packet keeps the production H-matrix/MLFMM construction on the CPU,
//! then feeds their checked matrix actions into the checked physical field
//! evaluator on a bounded physical Tri3 patch fixture.  It records field energy and centered
//! wavenumber/source probes without claiming CUDA construction or AVX parity.

#![cfg(target_arch = "x86_64")]

use num_complex::Complex64;
use serde::Serialize;
use sonium_bem::core::assembly::hmatrix::{
    AcaVariant, LowRankPrecision, build_hmatrix_stabilized,
    build_hmatrix_stabilized_wavenumber_derivative,
};
use sonium_bem::core::assembly::mlfmm::{
    build_cluster_tree, build_mlfmm_system_stabilized_checked, build_mlfmm_wavenumber_derivative,
};
use sonium_bem::core::assembly::tbem::build_tbem_system_stabilized;
use sonium_bem::core::postprocess::pressure::compute_total_field_checked;
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::core::{BasisOrder, Cluster, IncidentField, PhysicsParams};

const SCHEMA: &str = "sonium-bem-native-x86-b2-physical/v1";
const ELEMENT_COUNT: usize = 16;
const PATCH_SEPARATION: f64 = 12.0;
const FREQUENCY: f64 = 42.0;
// The fixed 19-term production MLFMM expansion is an approximation.  At this
// low-ka physical fixture its independently checked action is bounded below
// one percent; it is not promoted as dense parity.
const MLFMM_ACTION_BUDGET: f64 = 1.0e-2;
const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.21;
const TARGET_ELEMENTS_PER_LEAF: usize = 2;
const ACA_TOLERANCE: f64 = 1.0e-4;
const MAX_RANK: usize = 16;
const WAVENUMBER_STEP: f64 = 1.0e-4;
const ACTION_BUDGET: f64 = 1.0e-7;
const FIELD_BUDGET: f64 = 1.0e-7;
const MLFMM_FIELD_BUDGET: f64 = 1.0e-3;
const DERIVATIVE_BUDGET: f64 = 1.0e-5;
const SOURCE_DERIVATIVE_BUDGET: f64 = 1.0e-8;

#[derive(Debug, Serialize)]
struct PhysicalArtifact {
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
    frequency_hz: f64,
    element_count: usize,
    dof_count: usize,
    field_point_count: usize,
    matrix_action_residual: f64,
    mlfmm_action_residual: f64,
    hmatrix_field_residual: f64,
    mlfmm_field_residual: f64,
    dense_field_energy: f64,
    hmatrix_field_energy: f64,
    mlfmm_field_energy: f64,
    hmatrix_field_energy_residual: f64,
    mlfmm_field_energy_residual: f64,
    hmatrix_wavenumber_derivative_residual: f64,
    mlfmm_wavenumber_derivative_residual: f64,
    source_wavenumber_derivative_residual: f64,
    hmatrix_low_rank_blocks: usize,
    hmatrix_max_rank: usize,
    hmatrix_saturated_blocks: usize,
    mlfmm_levels: usize,
    mlfmm_d_matrices: usize,
    validation: Validation,
}

#[derive(Debug, Serialize)]
struct Validation {
    action_budget: f64,
    field_budget: f64,
    mlfmm_action_budget: f64,
    mlfmm_field_budget: f64,
    derivative_budget: f64,
    source_derivative_budget: f64,
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

fn physics_at_wavenumber(wavenumber: f64) -> PhysicsParams {
    PhysicsParams::try_new(
        wavenumber * SPEED_OF_SOUND / std::f64::consts::TAU,
        SPEED_OF_SOUND,
        DENSITY,
        false,
    )
    .expect("valid physical B2 wavenumber")
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
        center: [x + 0.8 / 3.0, 0.8 / 3.0, 0.0],
        area: 0.32,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::default()]),
        group: 0,
        dof_addresses: vec![index],
    }
}

fn fixture() -> (Vec<Element>, Vec<[f64; 3]>, PhysicsParams) {
    let elements: Vec<_> = (0..ELEMENT_COUNT).map(element).collect();
    let nodes: Vec<_> = (0..ELEMENT_COUNT)
        .flat_map(|index| {
            let patch = index / 8;
            let local = index % 8;
            let x = patch as f64 * PATCH_SEPARATION + local as f64 * 1.1;
            [[x, 0.0, 0.0], [x + 0.8, 0.0, 0.0], [x, 0.8, 0.0]]
        })
        .collect();
    let physics = PhysicsParams::try_new(FREQUENCY, SPEED_OF_SOUND, DENSITY, false)
        .expect("valid physical B2 physics");
    (elements, nodes, physics)
}

fn hmatrix_clusters() -> Vec<Cluster> {
    let mut left = Cluster::new([4.12, 0.27, 0.0]);
    left.element_indices = (0..8).collect();
    left.num_elements = 8;
    left.radius = 4.5;
    left.near_clusters = vec![0];
    left.far_clusters = vec![1];

    let mut right = Cluster::new([16.12, 0.27, 0.0]);
    right.element_indices = (8..ELEMENT_COUNT).collect();
    right.num_elements = 8;
    right.radius = 4.5;
    right.near_clusters = vec![1];
    right.far_clusters = vec![0];
    vec![left, right]
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
    elements: &[sonium_bem::core::types::Element],
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
    .expect("checked physical field evaluation")
    .into_iter()
    .map(|field| field.p_total)
    .collect()
}

fn field_energy(values: &[Complex64]) -> f64 {
    values.iter().map(Complex64::norm_sqr).sum()
}

fn centered_difference(plus: &[Complex64], minus: &[Complex64], step: f64) -> Vec<Complex64> {
    plus.iter()
        .zip(minus)
        .map(|(plus, minus)| (*plus - *minus) / (2.0 * step))
        .collect()
}

#[test]
fn native_x86_b2_physical_hfmm_v1() {
    // Opt-in evidence producer: skip under the general suite, run only via
    // the dedicated native-x86 script that sets the requirement flag.
    if std::env::var("SONIUM_B2_PHYSICAL_REQUIRE_NATIVE_X86").as_deref() != Ok("1") {
        eprintln!(
            "skipping native B2 physical evidence: set SONIUM_B2_PHYSICAL_REQUIRE_NATIVE_X86=1 to produce it"
        );
        return;
    }
    assert_eq!(std::env::consts::OS, "linux");

    let (elements, nodes, physics) = fixture();
    let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
    let levels = build_cluster_tree(&elements, TARGET_ELEMENTS_PER_LEAF, &physics);
    let hmatrix_clusters = hmatrix_clusters();
    let hmatrix = build_hmatrix_stabilized(
        &elements,
        &nodes,
        &hmatrix_clusters,
        &physics,
        ACA_TOLERANCE,
        MAX_RANK,
        sign,
        1.0,
    );
    let dense = build_tbem_system_stabilized(&elements, &nodes, &physics, sign, 1.0).matrix;
    let mlfmm = build_mlfmm_system_stabilized_checked(
        &elements,
        &nodes,
        levels.clone(),
        &physics,
        sign,
        1.0,
    )
    .expect("checked physical B2 MLFMM construction");
    let probe = probe_vector(elements.len());
    let dense_action = dense.matvec(&probe);
    let hmatrix_action = hmatrix
        .matvec_checked(&probe)
        .expect("checked physical B2 H-matrix action");
    let mlfmm_action = mlfmm
        .matvec_checked(&probe)
        .expect("checked physical B2 MLFMM action");
    let matrix_action_residual = relative_error(&hmatrix_action, &dense_action);
    let mlfmm_action_residual = relative_error(&mlfmm_action, &dense_action);
    assert!(
        matrix_action_residual <= ACTION_BUDGET,
        "H-matrix action residual={matrix_action_residual:.3e}"
    );
    assert!(
        mlfmm_action_residual <= MLFMM_ACTION_BUDGET,
        "MLFMM action residual={mlfmm_action_residual:.3e} exceeds {MLFMM_ACTION_BUDGET:.3e}"
    );

    let points = field_points();
    let dense_field = field_values(&points, &elements, &nodes, &dense_action, &physics);
    let hmatrix_field = field_values(&points, &elements, &nodes, &hmatrix_action, &physics);
    let mlfmm_field = field_values(&points, &elements, &nodes, &mlfmm_action, &physics);
    let hmatrix_field_residual = relative_error(&hmatrix_field, &dense_field);
    let mlfmm_field_residual = relative_error(&mlfmm_field, &dense_field);
    assert!(
        hmatrix_field_residual <= FIELD_BUDGET,
        "H-matrix field residual={hmatrix_field_residual:.3e}"
    );
    assert!(
        mlfmm_field_residual <= MLFMM_FIELD_BUDGET,
        "MLFMM field residual={mlfmm_field_residual:.3e} exceeds {MLFMM_FIELD_BUDGET:.3e}"
    );
    let dense_field_energy = field_energy(&dense_field);
    let hmatrix_field_energy = field_energy(&hmatrix_field);
    let mlfmm_field_energy = field_energy(&mlfmm_field);
    assert!(dense_field_energy.is_finite() && dense_field_energy > 0.0);
    let hmatrix_field_energy_residual =
        (hmatrix_field_energy - dense_field_energy).abs() / dense_field_energy;
    let mlfmm_field_energy_residual =
        (mlfmm_field_energy - dense_field_energy).abs() / dense_field_energy;
    assert!(hmatrix_field_energy_residual <= FIELD_BUDGET);
    assert!(mlfmm_field_energy_residual <= MLFMM_FIELD_BUDGET);

    let plus_physics = physics_at_wavenumber(physics.wave_number + WAVENUMBER_STEP);
    let minus_physics = physics_at_wavenumber(physics.wave_number - WAVENUMBER_STEP);
    let plus_dense = build_tbem_system_stabilized(&elements, &nodes, &plus_physics, sign, 1.0)
        .matrix
        .matvec(&probe);
    let minus_dense = build_tbem_system_stabilized(&elements, &nodes, &minus_physics, sign, 1.0)
        .matrix
        .matvec(&probe);
    let centered_dense_derivative = centered_difference(&plus_dense, &minus_dense, WAVENUMBER_STEP);
    let hmatrix_derivative = build_hmatrix_stabilized_wavenumber_derivative(
        &elements,
        &nodes,
        &hmatrix_clusters,
        &physics,
        ACA_TOLERANCE,
        MAX_RANK,
        AcaVariant::Hca,
        LowRankPrecision::F64,
        sign,
        1.0,
        6,
    )
    .expect("physical H-matrix wavenumber derivative construction")
    .matvec_checked(&probe)
    .expect("checked physical H-matrix wavenumber derivative action");
    let hmatrix_wavenumber_derivative_residual =
        relative_error(&hmatrix_derivative, &centered_dense_derivative);
    assert!(hmatrix_wavenumber_derivative_residual <= DERIVATIVE_BUDGET);

    let plus_mlfmm = build_mlfmm_system_stabilized_checked(
        &elements,
        &nodes,
        levels.clone(),
        &plus_physics,
        sign,
        1.0,
    )
    .expect("plus physical MLFMM construction")
    .matvec_checked(&probe)
    .expect("plus physical MLFMM action");
    let minus_mlfmm = build_mlfmm_system_stabilized_checked(
        &elements,
        &nodes,
        levels.clone(),
        &minus_physics,
        sign,
        1.0,
    )
    .expect("minus physical MLFMM construction")
    .matvec_checked(&probe)
    .expect("minus physical MLFMM action");
    let centered_mlfmm_derivative = centered_difference(&plus_mlfmm, &minus_mlfmm, WAVENUMBER_STEP);
    let mlfmm_derivative =
        build_mlfmm_wavenumber_derivative(&elements, &nodes, levels.clone(), &physics, sign, 1.0)
            .expect("physical MLFMM wavenumber derivative construction")
            .apply(&mlfmm, &probe);
    let mlfmm_wavenumber_derivative_residual =
        relative_error(&mlfmm_derivative, &centered_mlfmm_derivative);
    assert!(mlfmm_wavenumber_derivative_residual <= DERIVATIVE_BUDGET);

    let source_points = [[0.07, -0.04, 0.29], [-0.13, 0.08, 0.31]];
    let incident = IncidentField::plane_wave_z();
    let source_derivative =
        incident.evaluate_pressure_wavenumber_derivative(&source_points, &physics);
    let plus_source = incident.evaluate_pressure(&source_points, &plus_physics);
    let minus_source = incident.evaluate_pressure(&source_points, &minus_physics);
    let centered_source_derivative =
        centered_difference(&plus_source, &minus_source, WAVENUMBER_STEP);
    let source_wavenumber_derivative_residual =
        relative_error(&source_derivative, &centered_source_derivative);
    assert!(source_wavenumber_derivative_residual <= SOURCE_DERIVATIVE_BUDGET);

    let stats = hmatrix.rank_stats();
    assert!(
        !hmatrix.low_rank_blocks.is_empty(),
        "physical H/FMM packet must exercise a low-rank H-matrix block (dense={}, low={}, levels={}, d={})",
        hmatrix.dense_blocks.len(),
        hmatrix.low_rank_blocks.len(),
        mlfmm.num_levels,
        mlfmm.d_matrices.iter().map(Vec::len).sum::<usize>(),
    );
    assert!(
        mlfmm.d_matrices.iter().map(Vec::len).sum::<usize>() > 0,
        "physical H/FMM packet must exercise an MLFMM D transfer"
    );
    let artifact = PhysicalArtifact {
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
        frequency_hz: FREQUENCY,
        element_count: elements.len(),
        dof_count: probe.len(),
        field_point_count: points.len(),
        matrix_action_residual,
        mlfmm_action_residual,
        hmatrix_field_residual,
        mlfmm_field_residual,
        dense_field_energy,
        hmatrix_field_energy,
        mlfmm_field_energy,
        hmatrix_field_energy_residual,
        mlfmm_field_energy_residual,
        hmatrix_wavenumber_derivative_residual,
        mlfmm_wavenumber_derivative_residual,
        source_wavenumber_derivative_residual,
        hmatrix_low_rank_blocks: hmatrix.low_rank_blocks.len(),
        hmatrix_max_rank: stats.max_rank,
        hmatrix_saturated_blocks: stats.saturated_count,
        mlfmm_levels: mlfmm.num_levels,
        mlfmm_d_matrices: mlfmm.d_matrices.iter().map(Vec::len).sum(),
        validation: Validation {
            action_budget: ACTION_BUDGET,
            field_budget: FIELD_BUDGET,
            mlfmm_action_budget: MLFMM_ACTION_BUDGET,
            mlfmm_field_budget: MLFMM_FIELD_BUDGET,
            derivative_budget: DERIVATIVE_BUDGET,
            source_derivative_budget: SOURCE_DERIVATIVE_BUDGET,
            status: "pass",
        },
    };
    eprintln!(
        "sonium_bem_native_x86_b2_physical={{\"schema\":\"{}\",\"status\":\"{}\",\"elements\":{},\"action\":{:.3e},\"field\":{:.3e},\"h_drank\":{},\"m_d\":{}}}",
        artifact.schema,
        artifact.status,
        artifact.element_count,
        artifact.matrix_action_residual,
        artifact.mlfmm_field_residual,
        artifact.hmatrix_low_rank_blocks,
        artifact.mlfmm_d_matrices,
    );
    if let Some(directory) = std::env::var_os("SONIUM_OUTPUT_DIR") {
        let directory = std::path::PathBuf::from(directory);
        std::fs::create_dir_all(&directory).expect("create physical B2 evidence directory");
        let bytes = serde_json::to_vec_pretty(&artifact).expect("serialize physical B2 evidence");
        std::fs::write(
            directory.join("sonium_bem_native_x86_b2_physical.json"),
            bytes,
        )
        .expect("write physical B2 evidence");
    }
}
