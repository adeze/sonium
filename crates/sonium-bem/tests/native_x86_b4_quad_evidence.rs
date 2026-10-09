//! Native-x86 B4 evidence for the production CPU Quad4 Galerkin path.
//!
//! The compute-crate B4 producer can only compare the accelerated regular-pair
//! ABI, which currently has a constant-Tri3 layout.  Quad4 assembly is a
//! production CPU BEM path in `sonium-bem`, but it has no AVX2/FMA entry point.
//! This producer therefore records a deliberately bounded scalar-reference
//! capability.  It must never be interpreted as Quad4 AVX2/FMA parity.

#![cfg(target_arch = "x86_64")]

use num_complex::Complex64;
use serde::Serialize;
use sonium_bem::core::postprocess::compute_scattered_field;
use sonium_bem::core::types::{BoundaryCondition, Element, ElementProperty, ElementType};
use sonium_bem::{
    BasisOrder, PhysicsParams, TransmissionSide, assemble_galerkin_muller_transmission_quad4_pair,
    assemble_galerkin_muller_transmission_quad4_pair_wavenumber_derivative,
    assemble_galerkin_single_layer_quad4_pair, assemble_galerkin_single_layer_with_basis,
    compute_scattered_field_checked,
};

const EVIDENCE_SCHEMA: &str = "sonium-bem-native-x86-b4-quad/v1";
const TAIL_LENGTHS: [usize; 8] = [0, 1, 2, 3, 4, 5, 7, 9];
const PHASES: [(&str, f64); 3] = [("low", 42.0), ("acoustic", 420.0), ("high", 1_000.0)];
const CASE_COUNT: usize = PHASES.len() * 2 * TAIL_LENGTHS.len();
const EXPECTED_PAIR_CALLS: usize = PHASES.len() * 2 * 31;

/// The typed capability state is intentionally scalar-only.  Keeping this as
/// an enum prevents a future producer edit from silently serializing an AVX
/// claim for a path that has no accelerated entry point.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "kebab-case")]
enum Quad4Execution {
    ScalarReferenceOnly,
}

impl Quad4Execution {
    const fn avx_claim(self) -> bool {
        false
    }

    const fn label(self) -> &'static str {
        "cpu-scalar-reference"
    }
}

#[derive(Debug, Serialize)]
struct Quad4CapabilityProof {
    element_family: &'static str,
    basis_orders: &'static str,
    precision: &'static str,
    execution: Quad4Execution,
    accelerated: bool,
    avx_claim: bool,
}

#[derive(Debug, Serialize)]
struct PhysicalValidation {
    status: &'static str,
    budget: &'static str,
    reason: &'static str,
}

#[derive(Debug, Serialize)]
struct PhysicalPacket {
    status: &'static str,
    execution: Quad4Execution,
    basis: &'static str,
    precision: &'static str,
    geometry: &'static str,
    matrix_action_residual: f64,
    matrix_action_budget: f64,
    field_checked_legacy_residual: f64,
    field_residual_budget: f64,
    field_pressure_re: f64,
    field_pressure_im: f64,
    field_pressure_magnitude: f64,
    wavenumber_step: f64,
    wavenumber_derivative_residual: f64,
    wavenumber_derivative_budget: f64,
    wavenumber_derivative_action_re: f64,
    wavenumber_derivative_action_im: f64,
}

#[derive(Debug, Serialize)]
struct Quad4EvidenceArtifact {
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
    capability: Quad4CapabilityProof,
    phase_regimes: &'static str,
    phase_regimes_not_run: &'static str,
    tail_lengths: &'static str,
    singular_pair_families: &'static str,
    norm: &'static str,
    max_abs_entry: f64,
    pair_calls: usize,
    cases_passed: usize,
    cases_skipped: usize,
    cases_failed: usize,
    physical_validation: PhysicalValidation,
    physical_packet: PhysicalPacket,
    observables: Vec<f64>,
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

fn quad(connectivity: Vec<usize>, dof_addresses: Vec<usize>, center: [f64; 3]) -> Element {
    Element {
        connectivity,
        element_type: ElementType::Quad4,
        property: ElementProperty::Surface,
        normal: [0.0, 0.0, 1.0],
        node_normals: vec![[0.0, 0.0, 1.0]; 4],
        center,
        area: 1.0,
        boundary_condition: BoundaryCondition::Velocity(vec![Complex64::default()]),
        group: 0,
        dof_addresses,
    }
}

struct PairFixture {
    source: Element,
    field: Element,
    nodes: Vec<[f64; 3]>,
}

fn pair_fixtures() -> Vec<PairFixture> {
    let separated_nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let touching_nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 0.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
    ];
    let vertex_nodes = vec![
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [2.0, 1.0, 0.0],
        [2.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
    ];

    let separated_source = quad(vec![0, 1, 2, 3], (0..9).collect(), [0.5, 0.5, 0.0]);
    let separated_field = quad(vec![4, 5, 6, 7], (9..18).collect(), [0.5, 0.5, 1.0]);
    let touching_source = quad(vec![0, 1, 2, 3], (0..9).collect(), [0.5, 0.5, 0.0]);
    let touching_field = quad(vec![1, 4, 5, 2], (9..18).collect(), [1.5, 0.5, 0.0]);
    let vertex_source = quad(vec![0, 1, 2, 3], (0..9).collect(), [0.5, 0.5, 0.0]);
    let vertex_field = quad(vec![2, 4, 5, 6], (9..18).collect(), [1.5, 1.5, 0.0]);

    vec![
        PairFixture {
            source: separated_source.clone(),
            field: separated_field,
            nodes: separated_nodes,
        },
        PairFixture {
            source: touching_source,
            field: touching_field,
            nodes: touching_nodes,
        },
        PairFixture {
            source: vertex_source,
            field: vertex_field,
            nodes: vertex_nodes,
        },
        PairFixture {
            source: separated_source.clone(),
            field: separated_source,
            nodes: vec![
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
        },
    ]
}

fn append_matrix(
    observables: &mut Vec<f64>,
    matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
) -> f64 {
    let mut maximum = 0.0_f64;
    for value in matrix.as_slice() {
        assert!(
            value.re.is_finite() && value.im.is_finite(),
            "Quad4 matrix entry is non-finite"
        );
        maximum = maximum.max(value.norm());
        observables.extend([value.re, value.im]);
    }
    maximum
}

fn metadata_value(name: &str) -> String {
    std::env::var(name).unwrap_or_else(|_| "unspecified".into())
}

fn with_wavenumber(medium: &PhysicsParams, wave_number: f64) -> PhysicsParams {
    let mut perturbed = medium.clone();
    perturbed.speed_of_sound = perturbed.omega / wave_number;
    perturbed.wave_number = wave_number;
    perturbed.wave_length = perturbed.speed_of_sound / perturbed.frequency;
    perturbed.pressure_factor = perturbed.density * perturbed.omega * perturbed.harmonic_factor;
    perturbed
}

fn matrix_action(
    matrix: &sonium_solvers::dense::DenseMatrix<Complex64>,
    vector: &[Complex64],
) -> Vec<Complex64> {
    (0..matrix.nrows())
        .map(|row| {
            (0..matrix.ncols())
                .map(|column| matrix[[row, column]] * vector[column])
                .sum()
        })
        .collect()
}

fn relative_residual(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    let numerator = actual
        .iter()
        .zip(reference)
        .map(|(value, expected)| (*value - *expected).norm())
        .fold(0.0_f64, f64::max);
    let denominator = reference
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max)
        .max(1.0);
    numerator / denominator
}

fn physical_packet(fixture: &PairFixture) -> PhysicalPacket {
    let physics =
        PhysicsParams::try_new(420.0, 343.0, 1.21, false).expect("valid physical Quad4 medium");
    let mut source = fixture.source.clone();
    let mut field = fixture.field.clone();
    source.dof_addresses = (0..4).collect();
    field.dof_addresses = (4..8).collect();
    let gauss_order = 4;
    let pair = assemble_galerkin_single_layer_quad4_pair(
        &source,
        &field,
        &fixture.nodes,
        &physics,
        BasisOrder::Linear,
        gauss_order,
    )
    .expect("physical Quad4 single-layer pair");
    let full = assemble_galerkin_single_layer_with_basis(
        &[source.clone(), field.clone()],
        &fixture.nodes,
        &physics,
        BasisOrder::Linear,
        gauss_order,
    )
    .expect("physical Quad4 panel-local matrix");
    let action_input = [
        Complex64::new(0.25, -0.5),
        Complex64::new(-0.75, 0.125),
        Complex64::new(0.5, 0.875),
        Complex64::new(-0.125, -0.25),
    ];
    let pair_action = matrix_action(&pair, &action_input);
    let full_action: Vec<_> = (0..4)
        .map(|row| {
            (0..4)
                .map(|column| full[[row, 4 + column]] * action_input[column])
                .sum()
        })
        .collect();
    let matrix_action_residual = relative_residual(&pair_action, &full_action);

    let eval_points = [[0.25, 0.35, 1.75]];
    let surface_pressure = [
        Complex64::new(1.0, -0.25),
        Complex64::new(-0.5, 0.75),
        Complex64::new(0.25, 0.5),
        Complex64::new(0.125, -0.625),
    ];
    let checked = compute_scattered_field_checked(
        &eval_points,
        std::slice::from_ref(&source),
        &fixture.nodes,
        &surface_pressure,
        None,
        &physics,
        BasisOrder::Linear,
    )
    .expect("checked Quad4 scattered field");
    let legacy = compute_scattered_field(
        &eval_points,
        std::slice::from_ref(&source),
        &fixture.nodes,
        &surface_pressure,
        None,
        &physics,
        BasisOrder::Linear,
    );
    let field_checked_legacy_residual = relative_residual(&checked, &legacy);
    let field_pressure = checked[0];

    let interior = PhysicsParams::try_new(420.0, 1_480.0, 1_000.0, true)
        .expect("valid physical Quad4 interior medium");
    let derivative = assemble_galerkin_muller_transmission_quad4_pair_wavenumber_derivative(
        &source,
        &field,
        &fixture.nodes,
        &physics,
        &interior,
        BasisOrder::Linear,
        2,
        TransmissionSide::Exterior,
    )
    .expect("Quad4 analytic wavenumber derivative");
    let step = (physics.wave_number.abs() * 1.0e-5).max(1.0e-7);
    let plus = assemble_galerkin_muller_transmission_quad4_pair(
        &source,
        &field,
        &fixture.nodes,
        &with_wavenumber(&physics, physics.wave_number + step),
        &interior,
        BasisOrder::Linear,
        2,
    )
    .expect("Quad4 plus wavenumber perturbation");
    let minus = assemble_galerkin_muller_transmission_quad4_pair(
        &source,
        &field,
        &fixture.nodes,
        &with_wavenumber(&physics, physics.wave_number - step),
        &interior,
        BasisOrder::Linear,
        2,
    )
    .expect("Quad4 minus wavenumber perturbation");
    let centered: Vec<_> = plus
        .as_slice()
        .iter()
        .zip(minus.as_slice())
        .map(|(plus, minus)| (*plus - *minus) / (2.0 * step))
        .collect();
    let wavenumber_derivative_residual = relative_residual(derivative.as_slice(), &centered);
    let transmission_action_input = [
        Complex64::new(0.25, -0.5),
        Complex64::new(-0.75, 0.125),
        Complex64::new(0.5, 0.875),
        Complex64::new(-0.125, -0.25),
        Complex64::new(0.625, 0.25),
        Complex64::new(-0.375, -0.75),
        Complex64::new(0.875, -0.125),
        Complex64::new(-0.25, 0.5),
    ];
    let derivative_action = matrix_action(&derivative, &transmission_action_input);

    assert!(matrix_action_residual <= 1.0e-12);
    assert!(field_checked_legacy_residual <= 1.0e-12);
    assert!(field_pressure.norm().is_finite() && field_pressure.norm() > 0.0);
    assert!(wavenumber_derivative_residual <= 5.0e-4);

    PhysicalPacket {
        status: "pass",
        execution: Quad4Execution::ScalarReferenceOnly,
        basis: "linear",
        precision: "Complex64",
        geometry: "separated",
        matrix_action_residual,
        matrix_action_budget: 1.0e-12,
        field_checked_legacy_residual,
        field_residual_budget: 1.0e-12,
        field_pressure_re: field_pressure.re,
        field_pressure_im: field_pressure.im,
        field_pressure_magnitude: field_pressure.norm(),
        wavenumber_step: step,
        wavenumber_derivative_residual,
        wavenumber_derivative_budget: 5.0e-4,
        wavenumber_derivative_action_re: derivative_action.iter().map(|value| value.re).sum(),
        wavenumber_derivative_action_im: derivative_action.iter().map(|value| value.im).sum(),
    }
}

#[test]
fn native_x86_b4_quad4_scalar_reference_v1() {
    // Opt-in evidence producer: skip under the general suite, run only via
    // the dedicated native-x86 script that sets the requirement flag.
    if std::env::var("SONIUM_B4_QUAD4_REQUIRE_NATIVE_X86").as_deref() != Ok("1") {
        eprintln!(
            "skipping native B4 Quad4 evidence: set SONIUM_B4_QUAD4_REQUIRE_NATIVE_X86=1 to produce it"
        );
        return;
    }
    assert_eq!(
        std::env::consts::OS,
        "linux",
        "B4 Quad4 evidence requires Linux"
    );

    let execution = Quad4Execution::ScalarReferenceOnly;
    let mut observables = Vec::new();
    let fixtures = pair_fixtures();
    let mut pair_calls = 0;
    let mut max_abs_entry = 0.0_f64;

    for &(phase, frequency) in &PHASES {
        let physics = PhysicsParams::try_new(frequency, 343.0, 1.21, false)
            .expect("deterministic Quad4 physics is valid");
        for basis in [BasisOrder::Linear, BasisOrder::Quadratic] {
            for (tail_index, &tail_length) in TAIL_LENGTHS.iter().enumerate() {
                for pair_index in 0..tail_length {
                    let fixture = &fixtures[(tail_index + pair_index) % fixtures.len()];
                    let local_basis_count = match basis {
                        BasisOrder::Linear => 4,
                        BasisOrder::Quadratic => 9,
                        BasisOrder::Constant => {
                            unreachable!("Quad4 evidence excludes constant basis")
                        }
                    };
                    let mut source = fixture.source.clone();
                    let mut field = fixture.field.clone();
                    source.dof_addresses = (0..local_basis_count).collect();
                    field.dof_addresses = (0..local_basis_count).collect();
                    let matrix = assemble_galerkin_single_layer_quad4_pair(
                        &source,
                        &field,
                        &fixture.nodes,
                        &physics,
                        basis,
                        4,
                    )
                    .unwrap_or_else(|error| {
                        panic!(
                            "Quad4 scalar reference failed for phase={phase}, basis={basis:?}, tail={tail_length}, pair={pair_index}: {error}"
                        )
                    });
                    max_abs_entry = max_abs_entry.max(append_matrix(&mut observables, &matrix));
                    pair_calls += 1;
                }
            }
        }
    }

    assert_eq!(pair_calls, EXPECTED_PAIR_CALLS);
    assert!(!execution.avx_claim());
    assert!(!observables.is_empty());
    let physical_packet = physical_packet(&fixtures[0]);

    let artifact = Quad4EvidenceArtifact {
        schema: EVIDENCE_SCHEMA,
        status: "pass",
        target_arch: "x86_64",
        os: "linux-x86_64",
        cpu_model: cpuinfo_value("model name"),
        microcode: cpuinfo_value("microcode"),
        compiler: compiler_version(),
        features: format!(
            "avx2={},fma={},quad4_acceleration=false",
            std::is_x86_feature_detected!("avx2"),
            std::is_x86_feature_detected!("fma")
        ),
        selected_kernel: execution.label(),
        runtime_dispatch: "selected_kernel=cpu-scalar-reference,avx_claim=false",
        thread_policy: metadata_value("SONIUM_QA_THREAD_POLICY"),
        affinity_policy: metadata_value("SONIUM_QA_AFFINITY_POLICY"),
        numa_policy: metadata_value("SONIUM_QA_NUMA_POLICY"),
        deterministic_mode: metadata_value("SONIUM_QA_DETERMINISTIC_MODE"),
        capability: Quad4CapabilityProof {
            element_family: "Quad4",
            basis_orders: "linear,quadratic",
            precision: "Complex64",
            execution,
            accelerated: false,
            avx_claim: execution.avx_claim(),
        },
        phase_regimes: "low,acoustic,high",
        phase_regimes_not_run: "static (PhysicsParams requires positive frequency)",
        tail_lengths: "0,1,2,3,4,5,7,9",
        singular_pair_families: "separated,common-edge,common-vertex,self",
        norm: "maximum absolute matrix-entry magnitude",
        max_abs_entry,
        pair_calls,
        cases_passed: CASE_COUNT,
        cases_skipped: 0,
        cases_failed: 0,
        physical_validation: PhysicalValidation {
            status: "pass",
            budget: "strict packet gates",
            reason: "Quad4 remains scalar-reference-only; the packet validates a production matrix action, checked field observable, and analytic wavenumber derivative against centered reassembly.",
        },
        physical_packet,
        observables,
    };

    let bytes = serde_json::to_vec_pretty(&artifact).expect("Quad4 evidence serializes");
    eprintln!(
        "sonium_bem_native_x86_b4_quad={{\"schema\":\"{}\",\"status\":\"{}\",\"selected_kernel\":\"{}\",\"capability\":\"scalar-reference-only\",\"cases_passed\":{},\"pair_calls\":{},\"observables\":{}}}",
        artifact.schema,
        artifact.status,
        artifact.selected_kernel,
        artifact.cases_passed,
        artifact.pair_calls,
        artifact.observables.len(),
    );
    if let Some(output_dir) = std::env::var_os("SONIUM_OUTPUT_DIR") {
        let output_dir = std::path::PathBuf::from(output_dir);
        std::fs::create_dir_all(&output_dir).expect("create Quad4 evidence directory");
        std::fs::write(output_dir.join("sonium_bem_native_x86_b4_quad.json"), bytes)
            .expect("write Quad4 evidence artifact");
    }
}
