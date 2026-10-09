//! Native-x86 B2 evidence for the bounded CPU H-matrix/MLFMM setup seam.
//!
//! B2's CUDA action object is exercised by a separate required-device
//! producer.  This packet records the host construction that feeds that
//! object: cluster-tree topology, near/far admission, HCA block metadata, and
//! checked MLFMM transfer construction.  It is deliberately a CPU reference
//! packet and makes no CUDA, AVX, or cross-architecture claim.

#![cfg(target_arch = "x86_64")]

use num_complex::Complex64;
use serde::Serialize;
use sonium_bem::PhysicsParams;
use sonium_bem::core::assembly::mlfmm::{
    MlfmmSystem, build_cluster_tree, build_mlfmm_system_stabilized_checked,
};
use sonium_bem::core::assembly::tbem::build_tbem_system_stabilized;
use sonium_bem::core::assembly::{HMatrix, build_hmatrix};
use sonium_bem::core::types::{BoundaryCondition, Cluster, Element, ElementProperty, ElementType};
use sonium_solvers::dense::DenseMatrix;

const EVIDENCE_SCHEMA: &str = "sonium-bem-native-x86-b2-setup/v1";
const ELEMENT_COUNT: usize = 16;
const TARGET_ELEMENTS_PER_LEAF: usize = 2;
const FREQUENCY: f64 = 42.0;
const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.21;
const H_MATRIX_TOLERANCE: f64 = 1.0e-4;
const H_MATRIX_MAX_RANK: usize = 16;
const ACTION_RESIDUAL_BUDGET: f64 = 1.0e-8;
const DETERMINISM_BUDGET: f64 = 1.0e-12;

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "kebab-case")]
enum B2Execution {
    CpuReferenceOnly,
}

impl B2Execution {
    const fn label(self) -> &'static str {
        "cpu-reference"
    }

    const fn avx_claim(self) -> bool {
        false
    }
}

#[derive(Debug, Serialize)]
struct CapabilityProof {
    operator_family: &'static str,
    precision: &'static str,
    execution: B2Execution,
    accelerated: bool,
    cuda_claim: bool,
    avx_claim: bool,
}

#[derive(Debug, Serialize)]
struct HierarchyPacket {
    level_count: usize,
    clusters_per_level: Vec<usize>,
    root_element_count: usize,
    leaf_cluster_count: usize,
    leaf_element_count: usize,
    near_relation_count: usize,
    far_relation_count: usize,
    expansion_terms: Vec<usize>,
    theta_points: Vec<usize>,
    phi_points: Vec<usize>,
}

#[derive(Debug, Serialize)]
struct HMatrixPacket {
    num_dofs: usize,
    dense_block_count: usize,
    low_rank_block_count: usize,
    memory_bytes: usize,
    compression_ratio: f64,
    min_rank: usize,
    max_rank: usize,
    mean_rank: f64,
    saturated_count: usize,
    saturation_fraction: f64,
    dense_action_residual: f64,
    transpose_action_residual: f64,
    hermitian_action_residual: f64,
}

#[derive(Debug, Serialize)]
struct MlfmmPacket {
    num_dofs: usize,
    num_levels: usize,
    near_block_count: usize,
    t_matrix_count: usize,
    d_matrix_count: usize,
    s_matrix_count: usize,
    sphere_points_per_level: Vec<usize>,
    forward_repeat_residual: f64,
    transpose_repeat_residual: f64,
    hermitian_repeat_residual: f64,
    checked_actions_passed: usize,
}

#[derive(Debug, Serialize)]
struct ValidationPacket {
    status: &'static str,
    action_residual_budget: f64,
    determinism_budget: f64,
    wavenumber_probe: f64,
}

#[derive(Debug, Serialize)]
struct B2SetupArtifact {
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
    capability: CapabilityProof,
    fixture_elements: usize,
    fixture_dofs: usize,
    fixture_frequency_hz: f64,
    fixture_precision: &'static str,
    hierarchy: HierarchyPacket,
    hmatrix: HMatrixPacket,
    mlfmm: MlfmmPacket,
    validation: ValidationPacket,
}

fn cpuinfo_value(field: &str) -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                let (key, value) = line.split_once(':')?;
                (key.trim() == field).then(|| value.trim().to_owned())
            })
        })
        .unwrap_or_else(|| "unknown".into())
}

fn compiler_version() -> String {
    if let Some(version) = option_env!("RUSTC_VERSION") {
        return version.to_owned();
    }
    std::process::Command::new("rustc")
        .arg("-vV")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|output| output.lines().next().unwrap_or("unknown").to_owned())
        .unwrap_or_else(|| "unknown".into())
}

fn metadata_value(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "unspecified".into())
}

fn element(index: usize) -> Element {
    let patch = index / 8;
    let local = index % 8;
    let x = patch as f64 * 12.0 + local as f64 * 1.1;
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
            let x = patch as f64 * 12.0 + local as f64 * 1.1;
            [[x, 0.0, 0.0], [x + 0.8, 0.0, 0.0], [x, 0.8, 0.0]]
        })
        .collect();
    let physics = PhysicsParams::try_new(FREQUENCY, SPEED_OF_SOUND, DENSITY, false)
        .expect("valid native B2 setup physics");
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
    let value = numerator / denominator;
    if value.is_finite() {
        value
    } else {
        f64::INFINITY
    }
}

fn dense_action(
    matrix: &DenseMatrix<Complex64>,
    probe: &[Complex64],
    transpose: bool,
    hermitian: bool,
) -> Vec<Complex64> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| {
                    let mut entry = if transpose {
                        matrix[[column, row]]
                    } else {
                        matrix[[row, column]]
                    };
                    if hermitian {
                        entry = entry.conj();
                    }
                    entry * probe[column]
                })
                .sum()
        })
        .collect()
}

fn finite_action(action: &[Complex64]) -> bool {
    action
        .iter()
        .all(|value| value.re.is_finite() && value.im.is_finite())
}

fn hierarchy_packet(levels: &[sonium_bem::core::types::ClusterLevel]) -> HierarchyPacket {
    let clusters_per_level = levels.iter().map(|level| level.clusters.len()).collect();
    let root_element_count = levels
        .first()
        .and_then(|level| level.clusters.first())
        .map_or(0, |cluster| cluster.element_indices.len());
    let leaf = levels
        .last()
        .map(|level| level.clusters.as_slice())
        .unwrap_or(&[]);
    let near_relation_count = levels
        .iter()
        .flat_map(|level| level.clusters.iter())
        .map(|cluster| cluster.near_clusters.len())
        .sum();
    let far_relation_count = levels
        .iter()
        .flat_map(|level| level.clusters.iter())
        .map(|cluster| cluster.far_clusters.len())
        .sum();
    HierarchyPacket {
        level_count: levels.len(),
        clusters_per_level,
        root_element_count,
        leaf_cluster_count: leaf.len(),
        leaf_element_count: leaf
            .iter()
            .map(|cluster| cluster.element_indices.len())
            .sum(),
        near_relation_count,
        far_relation_count,
        expansion_terms: levels.iter().map(|level| level.expansion_terms).collect(),
        theta_points: levels.iter().map(|level| level.theta_points).collect(),
        phi_points: levels.iter().map(|level| level.phi_points).collect(),
    }
}

fn hmatrix_packet(
    hmatrix: &HMatrix,
    dense: &DenseMatrix<Complex64>,
    probe: &[Complex64],
) -> HMatrixPacket {
    let h_forward = hmatrix
        .matvec_checked(probe)
        .expect("checked H-matrix forward setup action");
    let h_transpose = hmatrix
        .matvec_transpose_checked(probe)
        .expect("checked H-matrix transpose setup action");
    let h_hermitian = hmatrix
        .matvec_hermitian_checked(probe)
        .expect("checked H-matrix Hermitian setup action");
    let stats = hmatrix.rank_stats();
    HMatrixPacket {
        num_dofs: hmatrix.num_dofs,
        dense_block_count: hmatrix.dense_blocks.len(),
        low_rank_block_count: hmatrix.low_rank_blocks.len(),
        memory_bytes: hmatrix.memory_bytes,
        compression_ratio: hmatrix.compression_ratio,
        min_rank: stats.min_rank,
        max_rank: stats.max_rank,
        mean_rank: stats.mean_rank,
        saturated_count: stats.saturated_count,
        saturation_fraction: stats.saturation_fraction,
        dense_action_residual: relative_error(
            &h_forward,
            &dense_action(dense, probe, false, false),
        ),
        transpose_action_residual: relative_error(
            &h_transpose,
            &dense_action(dense, probe, true, false),
        ),
        hermitian_action_residual: relative_error(
            &h_hermitian,
            &dense_action(dense, probe, true, true),
        ),
    }
}

fn mlfmm_packet(mlfmm: &MlfmmSystem, repeat: &MlfmmSystem, probe: &[Complex64]) -> MlfmmPacket {
    let forward = mlfmm
        .matvec_checked(probe)
        .expect("checked MLFMM forward setup action");
    let transpose = mlfmm
        .matvec_transpose_checked(probe)
        .expect("checked MLFMM transpose setup action");
    let hermitian = mlfmm
        .apply_hermitian_checked(probe)
        .expect("checked MLFMM Hermitian setup action");
    let repeat_forward = repeat
        .matvec_checked(probe)
        .expect("repeated checked MLFMM forward setup action");
    let repeat_transpose = repeat
        .matvec_transpose_checked(probe)
        .expect("repeated checked MLFMM transpose setup action");
    let repeat_hermitian = repeat
        .apply_hermitian_checked(probe)
        .expect("repeated checked MLFMM Hermitian setup action");
    assert!(finite_action(&forward));
    assert!(finite_action(&transpose));
    assert!(finite_action(&hermitian));
    MlfmmPacket {
        num_dofs: mlfmm.num_dofs,
        num_levels: mlfmm.num_levels,
        near_block_count: mlfmm.near_matrix.len(),
        t_matrix_count: mlfmm
            .t_matrices
            .iter()
            .map(|level| level.matrices.len())
            .sum(),
        d_matrix_count: mlfmm.d_matrices.iter().map(Vec::len).sum(),
        s_matrix_count: mlfmm
            .s_matrices
            .iter()
            .map(|level| level.matrices.len())
            .sum(),
        sphere_points_per_level: mlfmm.sphere_points_per_level.clone(),
        forward_repeat_residual: relative_error(&forward, &repeat_forward),
        transpose_repeat_residual: relative_error(&transpose, &repeat_transpose),
        hermitian_repeat_residual: relative_error(&hermitian, &repeat_hermitian),
        checked_actions_passed: 3,
    }
}

#[test]
fn native_x86_b2_setup_construction_v1() {
    // Opt-in evidence producer: skip under the general suite, run only via
    // the dedicated native-x86 script that sets the requirement flag.
    if std::env::var("SONIUM_B2_SETUP_REQUIRE_NATIVE_X86").as_deref() != Ok("1") {
        eprintln!(
            "skipping native B2 setup evidence: set SONIUM_B2_SETUP_REQUIRE_NATIVE_X86=1 to produce it"
        );
        return;
    }
    assert_eq!(
        std::env::consts::OS,
        "linux",
        "B2 setup evidence requires Linux"
    );

    let execution = B2Execution::CpuReferenceOnly;
    let (elements, nodes, physics) = fixture();
    let sign = PhysicsParams::optimal_formulation_sign(physics.wave_number);
    let levels = build_cluster_tree(&elements, TARGET_ELEMENTS_PER_LEAF, &physics);
    let repeated_levels = build_cluster_tree(&elements, TARGET_ELEMENTS_PER_LEAF, &physics);
    let hierarchy = hierarchy_packet(&levels);
    assert_eq!(hierarchy.root_element_count, ELEMENT_COUNT);
    assert_eq!(hierarchy.leaf_element_count, ELEMENT_COUNT);
    assert!(hierarchy.level_count >= 2);
    assert!(hierarchy.far_relation_count > 0);

    let leaf_clusters = hmatrix_clusters();
    let hmatrix = build_hmatrix(
        &elements,
        &nodes,
        &leaf_clusters,
        &physics,
        H_MATRIX_TOLERANCE,
        H_MATRIX_MAX_RANK,
    );
    let repeated_hmatrix = build_hmatrix(
        &elements,
        &nodes,
        &leaf_clusters,
        &physics,
        H_MATRIX_TOLERANCE,
        H_MATRIX_MAX_RANK,
    );
    let dense = build_tbem_system_stabilized(&elements, &nodes, &physics, sign, 1.0).matrix;
    let probe: Vec<_> = (0..ELEMENT_COUNT)
        .map(|index| Complex64::new(0.25 + index as f64 * 0.125, -0.5 + index as f64 * 0.0625))
        .collect();
    let hmatrix_data = hmatrix_packet(&hmatrix, &dense, &probe);
    let repeat_action = repeated_hmatrix
        .matvec_checked(&probe)
        .expect("repeated checked H-matrix setup action");
    let hmatrix_action = hmatrix
        .matvec_checked(&probe)
        .expect("checked H-matrix action");
    let hmatrix_repeat_residual = relative_error(&hmatrix_action, &repeat_action);
    assert!(hmatrix_data.dense_action_residual <= ACTION_RESIDUAL_BUDGET);
    assert!(hmatrix_data.transpose_action_residual <= ACTION_RESIDUAL_BUDGET);
    assert!(hmatrix_data.hermitian_action_residual <= ACTION_RESIDUAL_BUDGET);
    assert!(hmatrix_repeat_residual <= DETERMINISM_BUDGET);

    let mlfmm = build_mlfmm_system_stabilized_checked(
        &elements,
        &nodes,
        levels.clone(),
        &physics,
        sign,
        1.0,
    )
    .expect("checked MLFMM setup");
    let repeated_mlfmm = build_mlfmm_system_stabilized_checked(
        &elements,
        &nodes,
        repeated_levels,
        &physics,
        sign,
        1.0,
    )
    .expect("repeated checked MLFMM setup");
    let mlfmm_data = mlfmm_packet(&mlfmm, &repeated_mlfmm, &probe);
    assert_eq!(mlfmm_data.num_dofs, ELEMENT_COUNT);
    assert_eq!(mlfmm_data.checked_actions_passed, 3);
    assert!(mlfmm_data.forward_repeat_residual <= DETERMINISM_BUDGET);
    assert!(mlfmm_data.transpose_repeat_residual <= DETERMINISM_BUDGET);
    assert!(mlfmm_data.hermitian_repeat_residual <= DETERMINISM_BUDGET);

    let artifact = B2SetupArtifact {
        schema: EVIDENCE_SCHEMA,
        status: "pass",
        target_arch: "x86_64",
        os: "linux-x86_64",
        cpu_model: cpuinfo_value("model name"),
        microcode: cpuinfo_value("microcode"),
        compiler: compiler_version(),
        features: format!(
            "avx2={},fma={},cuda_claim=false,avx_claim=false",
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("fma")
        ),
        selected_kernel: execution.label(),
        runtime_dispatch: "selected_kernel=cpu-reference,cuda_claim=false,avx_claim=false",
        thread_policy: metadata_value("SONIUM_QA_THREAD_POLICY"),
        affinity_policy: metadata_value("SONIUM_QA_AFFINITY_POLICY"),
        numa_policy: metadata_value("SONIUM_QA_NUMA_POLICY"),
        deterministic_mode: metadata_value("SONIUM_QA_DETERMINISTIC_MODE"),
        capability: CapabilityProof {
            operator_family: "HMatrix+MLFMM-setup",
            precision: "Complex64",
            execution,
            accelerated: false,
            cuda_claim: false,
            avx_claim: execution.avx_claim(),
        },
        fixture_elements: ELEMENT_COUNT,
        fixture_dofs: ELEMENT_COUNT,
        fixture_frequency_hz: FREQUENCY,
        fixture_precision: "Complex64",
        hierarchy,
        hmatrix: hmatrix_data,
        mlfmm: mlfmm_data,
        validation: ValidationPacket {
            status: "pass",
            action_residual_budget: ACTION_RESIDUAL_BUDGET,
            determinism_budget: DETERMINISM_BUDGET,
            wavenumber_probe: physics.wave_number,
        },
    };
    let bytes = serde_json::to_vec_pretty(&artifact).expect("B2 setup evidence serializes");
    eprintln!(
        "sonium_bem_native_x86_b2_setup={{\"schema\":\"{}\",\"status\":\"{}\",\"selected_kernel\":\"{}\",\"levels\":{},\"h_dense\":{},\"h_low_rank\":{},\"mlfmm_d\":{}}}",
        artifact.schema,
        artifact.status,
        artifact.selected_kernel,
        artifact.hierarchy.level_count,
        artifact.hmatrix.dense_block_count,
        artifact.hmatrix.low_rank_block_count,
        artifact.mlfmm.d_matrix_count,
    );
    if let Some(output_dir) = std::env::var_os("SONIUM_OUTPUT_DIR") {
        let output_dir = std::path::PathBuf::from(output_dir);
        std::fs::create_dir_all(&output_dir).expect("create B2 setup evidence directory");
        std::fs::write(
            output_dir.join("sonium_bem_native_x86_b2_setup.json"),
            bytes,
        )
        .expect("write B2 setup evidence artifact");
    }
}
