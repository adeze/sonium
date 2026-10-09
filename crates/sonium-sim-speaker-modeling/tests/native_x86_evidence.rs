#![cfg(target_arch = "x86_64")]
#![recursion_limit = "256"]

use num_complex::Complex64;
use serde_json::json;
use sha2::{Digest, Sha256};
use sonium_fem::mesh::Mesh as FemMesh;
use sonium_sim_speaker_modeling::fem_bem_coupling::{
    CouplingDrive, CylinderFixtureDiscretization, MonolithicFemBemCoupling,
    MonolithicSealedProductCoupling, build_product_coupling_mesh,
    cylindrical_box_with_circular_aperture,
};
use sonium_sim_speaker_modeling::{
    ArtifactBackend, AxisymmetricLinearConfig, AxisymmetricLinearModel, CabinetDesign,
    CabinetMaterial, CrossoverBranch, CrossoverNetwork, DriverModel, DriverModelReferenceArtifact,
    DriverModelReferenceTolerance, ProductDesign, ProductDriver, ProductInteriorMeshConfig,
    ProductInteriorPatchKind, Sector3dConfig, Sector3dModel, SpeakerDriverStudy,
    StructuralAcousticFeedback, StructuralAcousticModalBlock, StructuralAcousticMode,
};
use std::process::Command;
use std::{fs, path::PathBuf};

// This is the portable f64 DriverModel golden envelope. Its canonical bytes
// intentionally do not include host architecture, SIMD, or compiler details,
// so native x86 and aarch64 replays can assert byte identity against it.
const DRIVER_MODEL_CPU_GOLDEN_SHA256_V1: &str =
    "0411bd27041feef181df9468a5713993121dc904144dbdadb601584073d8f20f";

fn max_complex_difference(left: &[Complex64], right: &[Complex64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(a, b)| (*a - *b).norm())
        .fold(0.0, f64::max)
}

fn max_vector_difference(left: &[[Complex64; 2]], right: &[[Complex64; 2]]) -> f64 {
    left.iter()
        .zip(right)
        .flat_map(|(a, b)| a.iter().zip(b).map(|(x, y)| (*x - *y).norm()))
        .fold(0.0, f64::max)
}

fn boundary_face_area(mesh: &FemMesh, face_index: usize) -> f64 {
    let nodes = &mesh.boundaries[face_index].nodes;
    let [a, b, c] = [nodes[0], nodes[1], nodes[2]];
    let pa = mesh.nodes[a];
    let pb = mesh.nodes[b];
    let pc = mesh.nodes[c];
    let ab = [pb.x - pa.x, pb.y - pa.y, pb.z - pa.z];
    let ac = [pc.x - pa.x, pc.y - pa.y, pc.z - pa.z];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    0.5 * cross.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn integrate_fem_pressure_force(
    mesh: &FemMesh,
    pressure: &[Complex64],
    faces: &[usize],
) -> Complex64 {
    faces
        .iter()
        .map(|&face_index| {
            let nodes = &mesh.boundaries[face_index].nodes;
            let [a, b, c] = [nodes[0], nodes[1], nodes[2]];
            let area = boundary_face_area(mesh, face_index);
            (pressure[a] + pressure[b] + pressure[c]) * (area / 3.0)
        })
        .sum()
}

fn native_x86_product_design() -> ProductDesign {
    ProductDesign {
        cabinet: CabinetDesign {
            box_width_m: 0.40,
            box_height_m: 0.30,
            box_depth_m: 0.30,
            wall_material: CabinetMaterial::Mdf18mm,
            port: None,
            baffle_width_m: 0.44,
            baffle_height_m: 0.34,
            braces: Vec::new(),
        },
        drivers: vec![ProductDriver {
            id: "sm3-observable-driver".to_owned(),
            radius_m: 0.08,
            position_m: [0.0, 0.0, 0.0],
            normal: [0.0, 0.0, 1.0],
            force_factor_tm: 5.0,
            coil_resistance_ohm: 6.0,
            coil_reactance_ohm: 1.0,
            mechanical_resistance_n_s_per_m: 1.0,
            mechanical_reactance_n_s_per_m: 2.0,
            crossover_branch: 0,
        }],
        crossover: CrossoverNetwork {
            branches: vec![CrossoverBranch::default()],
        },
        waveguide: None,
        structural_acoustic: Some(StructuralAcousticFeedback {
            area_m2: 0.025,
            modes: vec![StructuralAcousticMode {
                natural_frequency_hz: 400.0,
                effective_mass_kg: 0.02,
                damping_ratio: 0.03,
                pressure_participation: 1.0,
                stress_per_displacement_pa_per_m: 30_000.0,
            }],
            max_iterations: 8,
            velocity_tolerance: 1.0e-7,
            under_relaxation: 0.8,
        }),
    }
}

fn assert_relative_close(label: &str, left: f64, right: f64, relative_tolerance: f64) {
    let scale = left.abs().max(right.abs()).max(f64::MIN_POSITIVE);
    let relative_error = (left - right).abs() / scale;
    assert!(
        relative_error <= relative_tolerance,
        "{label} replay drift {relative_error:.3e} exceeds {relative_tolerance:.3e}: {left:.17e} vs {right:.17e}"
    );
}

fn assert_residual_close(label: &str, left: f64, right: f64) {
    let scale = left.abs().max(right.abs()).max(f64::MIN_POSITIVE);
    let absolute_error = (left - right).abs();
    let relative_error = absolute_error / scale;
    assert!(
        absolute_error <= 1.0e-15 || relative_error <= 1.0e-12,
        "{label} replay drift absolute={absolute_error:.3e}, relative={relative_error:.3e}: {left:.17e} vs {right:.17e}"
    );
}

fn assert_complex_relative_close(
    label: &str,
    left: Complex64,
    right: Complex64,
    relative_tolerance: f64,
) {
    let scale = left.norm().max(right.norm()).max(f64::MIN_POSITIVE);
    let relative_error = (left - right).norm() / scale;
    assert!(
        relative_error <= relative_tolerance,
        "{label} replay drift {relative_error:.3e} exceeds {relative_tolerance:.3e}: {left:?} vs {right:?}"
    );
}

fn assert_complex_vectors_close(label: &str, left: &[Complex64], right: &[Complex64]) {
    assert_eq!(left.len(), right.len(), "{label} replay length changed");
    let maximum_difference = max_complex_difference(left, right);
    let maximum_scale = left
        .iter()
        .chain(right)
        .map(|value| value.norm())
        .fold(0.0, f64::max)
        .max(f64::MIN_POSITIVE);
    let relative_error = maximum_difference / maximum_scale;
    assert!(
        maximum_difference <= 1.0e-15 || relative_error <= 1.0e-12,
        "{label} replay drift absolute={maximum_difference:.3e}, relative={relative_error:.3e}"
    );
}

fn finite_difference_voltage_transfer(model: &DriverModel, frequency_hz: f64) -> Complex64 {
    let voltage = model.reference_voltage_rms_v;
    let step = 1.0e-6;
    let mut plus = model.clone();
    let mut minus = model.clone();
    plus.reference_voltage_rms_v = voltage * (1.0 + step);
    minus.reference_voltage_rms_v = voltage * (1.0 - step);
    let radiation = Complex64::new(0.4, 0.3);
    let plus_velocity = plus
        .response_with_radiation_impedance(frequency_hz, radiation)
        .unwrap()
        .0;
    let minus_velocity = minus
        .response_with_radiation_impedance(frequency_hz, radiation)
        .unwrap()
        .0;
    (plus_velocity - minus_velocity) / (2.0 * voltage * step)
}

fn cpu_vendor() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                line.contains("vendor_id")
                    .then(|| line.split(':').nth(1))
                    .flatten()
                    .map(|vendor| vendor.trim().to_string())
            })
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|contents| {
            contents.lines().find_map(|line| {
                line.contains("model name")
                    .then(|| line.split(':').nth(1))
                    .flatten()
                    .map(|model| model.trim().to_string())
            })
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn compiler_version() -> String {
    Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|version| version.trim().to_string())
        .unwrap_or_else(|| "rustc unknown".to_string())
}

#[test]
fn native_x86_driver_artifact_smoke_evidence_v1() {
    let driver =
        DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver()).unwrap();
    let artifact = DriverModelReferenceArtifact::from_model(
        &driver,
        &[100.0, 1_000.0],
        &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
    )
    .unwrap();
    artifact
        .validate_against_model(&driver, DriverModelReferenceTolerance::default())
        .unwrap();
    let matrix = sonium_sim_speaker_modeling::SpeakerCapabilityMatrix::cpu();
    assert_eq!(matrix.actual_backend, ArtifactBackend::Cpu);
    assert!(matrix
        .subsystems
        .iter()
        .all(|entry| entry.state == sonium_sim_speaker_modeling::SpeakerCapabilityState::Native));
    let bytes = artifact.canonical_json().unwrap();
    // Regression guard for the portable envelope contract: canonical bytes
    // must carry the normalized toolchain marker, never the producing
    // compiler string, or toolchain updates silently re-break the golden.
    let text = String::from_utf8(bytes.clone()).expect("canonical JSON is UTF-8");
    assert!(
        text.contains("\"toolchain\":\"canonical\""),
        "canonical bytes must normalize the producing toolchain"
    );
    assert!(
        !text.contains("rustc"),
        "canonical bytes must not bind the compiler version"
    );
    let checksum = Sha256::digest(bytes);
    assert_eq!(
        format!("{checksum:x}"),
        DRIVER_MODEL_CPU_GOLDEN_SHA256_V1,
        "native x86 replay drifted from the portable CPU/f64 golden envelope"
    );
    println!(
        "NATIVE_X86_DRIVER_ARTIFACT_EVIDENCE {{\"architecture\":\"{}\",\"cpu_vendor\":\"{}\",\"cpu_model\":\"{}\",\"compiler\":\"{}\",\"dispatch_path\":\"native-x86_64-cpu/scalar-modeling\",\"avx2_available\":{},\"fma_available\":{},\"artifact_sha256\":\"{checksum:x}\"}}",
        std::env::consts::ARCH,
        cpu_vendor(),
        cpu_model(),
        compiler_version(),
        std::is_x86_feature_detected!("avx2"),
        std::is_x86_feature_detected!("fma"),
    );
}

#[test]
fn native_x86_axisymmetric_coupled_replay_evidence_v1() {
    let mut axis_config = AxisymmetricLinearConfig::default();
    axis_config.magnetic_radial_cells = 4;
    axis_config.magnetic_axial_cells = 4;
    axis_config.structural_radial_cells = 5;
    axis_config.structural_axial_cells = 1;
    axis_config.acoustic_axial_cells = 2;
    let axis = AxisymmetricLinearModel::build(axis_config).unwrap();
    let first = axis
        .solve_frequency(1_000.0, Complex64::new(1.0, 0.0))
        .unwrap();
    let replay = axis
        .solve_frequency(1_000.0, Complex64::new(1.0, 0.0))
        .unwrap();

    assert_eq!(first.frequency_hz, replay.frequency_hz);
    assert_eq!(first.current_rms_a, replay.current_rms_a);
    assert_eq!(first.impedance_ohm, replay.impedance_ohm);
    assert_eq!(first.observation_spl_db, replay.observation_spl_db);
    assert_eq!(first.power, replay.power);
    assert_eq!(
        max_complex_difference(&first.pressure_rms_pa, &replay.pressure_rms_pa),
        0.0
    );
    assert_eq!(
        max_vector_difference(&first.displacement_rz_m, &replay.displacement_rz_m),
        0.0
    );
    assert!(first.solver.relative_residual.is_finite());
    assert!(first.power.residual_w.is_finite());

    let driver =
        DriverModel::from_study(&SpeakerDriverStudy::detailed_frequency_domain_driver()).unwrap();
    let artifact = DriverModelReferenceArtifact::from_model(
        &driver,
        &[100.0, 1_000.0],
        &[Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)],
    )
    .unwrap();
    artifact
        .validate_against_model(&driver, DriverModelReferenceTolerance::default())
        .unwrap();
    let bytes = artifact.canonical_json().unwrap();
    let text = String::from_utf8(bytes.clone()).expect("canonical JSON is UTF-8");
    assert!(
        text.contains("\"toolchain\":\"canonical\""),
        "canonical bytes must normalize the producing toolchain"
    );
    assert!(
        !text.contains("rustc"),
        "canonical bytes must not bind the compiler version"
    );
    assert_eq!(
        Sha256::digest(bytes)
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>(),
        DRIVER_MODEL_CPU_GOLDEN_SHA256_V1
    );
    println!(
        "NATIVE_X86_AXISYMMETRIC_EVIDENCE {{\"architecture\":\"{}\",\"cpu_vendor\":\"{}\",\"cpu_model\":\"{}\",\"compiler\":\"{}\",\"pressure_dofs\":{},\"current_rms_a\":[{:.17e},{:.17e}],\"impedance_ohm\":[{:.17e},{:.17e}],\"input_power_w\":{:.17e},\"power_residual_w\":{:.17e},\"relative_residual\":{:.17e},\"simd_features\":{{\"avx2\":{},\"fma\":{}}}}}",
        std::env::consts::ARCH,
        cpu_vendor(),
        cpu_model(),
        compiler_version(),
        first.pressure_rms_pa.len(),
        first.current_rms_a.re,
        first.current_rms_a.im,
        first.impedance_ohm.re,
        first.impedance_ohm.im,
        first.power.input_w,
        first.power.residual_w,
        first.solver.relative_residual,
        std::is_x86_feature_detected!("avx2"),
        std::is_x86_feature_detected!("fma"),
    );
}

#[test]
#[ignore = "expensive explicit native-x86 sector FEM evidence; run in scheduled release validation"]
fn native_x86_sector_mode_replay_evidence_v1() {
    let model = Sector3dModel::build(Sector3dConfig {
        // Keep this opt-in replay at the smallest valid sector mesh. The
        // periodic 3D solve is intentionally outside the routine x86 gate
        // because its sparse direct solve is still expensive on CI hosts.
        sector_angle_degrees: 90.0,
        conductor_conductivity_s_per_m: 1.0e4,
        quadrature_order: 1,
        ..Sector3dConfig::default()
    })
    .unwrap();
    let first = model.solve_frequency(100.0).unwrap();
    let replay = model.solve_frequency(100.0).unwrap();

    assert_eq!(first.frequency_hz, replay.frequency_hz);
    assert_complex_relative_close(
        "sector magnetic impedance",
        first.magnetic_impedance_ohm,
        replay.magnetic_impedance_ohm,
        1.0e-12,
    );
    assert_relative_close(
        "sector full-revolution eddy loss",
        first.full_revolution_eddy_loss_w,
        replay.full_revolution_eddy_loss_w,
        1.0e-12,
    );
    assert_relative_close(
        "sector full-revolution magnetic energy",
        first.full_revolution_magnetic_energy_j,
        replay.full_revolution_magnetic_energy_j,
        1.0e-12,
    );
    assert_residual_close(
        "sector periodic residual",
        first.periodic_residual_max,
        replay.periodic_residual_max,
    );
    assert_residual_close(
        "sector gauge residual",
        first.gauge_residual_max,
        replay.gauge_residual_max,
    );
    assert_residual_close(
        "sector charge-continuity residual",
        first.charge_continuity_relative_residual,
        replay.charge_continuity_relative_residual,
    );
    assert_complex_vectors_close(
        "sector vector-potential edge DOFs",
        &first.vector_potential_edge_dofs,
        &replay.vector_potential_edge_dofs,
    );
    assert_complex_vectors_close(
        "sector electric-potential node DOFs",
        &first.electric_potential_node_dofs,
        &replay.electric_potential_node_dofs,
    );
    assert!(first.periodic_residual_max.is_finite());
    assert!(first.gauge_residual_max.is_finite());
    assert!(first.charge_continuity_relative_residual.is_finite());
    println!(
        "NATIVE_X86_SECTOR_EVIDENCE {{\"architecture\":\"{}\",\"cpu_vendor\":\"{}\",\"cpu_model\":\"{}\",\"field_dofs\":{},\"mode_field_dofs\":{},\"impedance_ohm\":[{:.17e},{:.17e}],\"eddy_loss_w\":{:.17e},\"periodic_residual\":{:.17e}}}",
        std::env::consts::ARCH,
        cpu_vendor(),
        cpu_model(),
        first.vector_potential_edge_dofs.len(),
        first.electric_potential_node_dofs.len(),
        first.magnetic_impedance_ohm.re,
        first.magnetic_impedance_ohm.im,
        first.full_revolution_eddy_loss_w,
        first.periodic_residual_max,
    );
}

#[test]
#[ignore = "full axisymmetric/sector/FEM-BEM replay is an explicit evidence run"]
fn native_x86_modeling_replay_evidence_v1() {
    let study = SpeakerDriverStudy::detailed_frequency_domain_driver();
    let driver = DriverModel::from_study(&study).unwrap();
    let frequencies = [100.0, 1_000.0];
    let radiation = [Complex64::new(0.2, 0.1), Complex64::new(0.4, 0.3)];

    let artifact =
        DriverModelReferenceArtifact::from_model(&driver, &frequencies, &radiation).unwrap();
    let replay =
        DriverModelReferenceArtifact::from_model(&driver, &frequencies, &radiation).unwrap();
    artifact
        .validate_against_model(&driver, DriverModelReferenceTolerance::default())
        .unwrap();
    assert_eq!(
        artifact.canonical_json().unwrap(),
        replay.canonical_json().unwrap()
    );
    let derivative = finite_difference_voltage_transfer(&driver, 1_000.0);

    eprintln!("native-x86 evidence: axisymmetric build");
    let mut axis_config = AxisymmetricLinearConfig::default();
    axis_config.magnetic_radial_cells = 4;
    axis_config.magnetic_axial_cells = 4;
    axis_config.structural_radial_cells = 5;
    axis_config.structural_axial_cells = 1;
    axis_config.acoustic_axial_cells = 2;
    let axis = AxisymmetricLinearModel::build(axis_config).unwrap();
    let axis_first = axis
        .solve_frequency(1_000.0, Complex64::new(1.0, 0.0))
        .unwrap();
    let axis_replay = axis
        .solve_frequency(1_000.0, Complex64::new(1.0, 0.0))
        .unwrap();
    assert_eq!(axis_first.frequency_hz, axis_replay.frequency_hz);
    assert_eq!(axis_first.current_rms_a, axis_replay.current_rms_a);
    assert_eq!(axis_first.impedance_ohm, axis_replay.impedance_ohm);
    assert_eq!(
        axis_first.observation_spl_db,
        axis_replay.observation_spl_db
    );
    assert_eq!(axis_first.power, axis_replay.power);
    assert!(
        max_complex_difference(&axis_first.pressure_rms_pa, &axis_replay.pressure_rms_pa) == 0.0
    );
    assert!(
        max_vector_difference(
            &axis_first.displacement_rz_m,
            &axis_replay.displacement_rz_m
        ) == 0.0
    );
    assert!(axis_first.solver.relative_residual.is_finite());
    assert!(axis_first.power.residual_w.is_finite());

    eprintln!("native-x86 evidence: sector build/solve");
    let sector = Sector3dModel::build(Sector3dConfig {
        // Keep the released replay bounded on native CI while still solving
        // the genuine periodic H(curl)-H1 sector path.
        sector_angle_degrees: 90.0,
        conductor_conductivity_s_per_m: 1.0e4,
        quadrature_order: 1,
        ..Sector3dConfig::default()
    })
    .unwrap();
    let sector_first = sector.solve_frequency(1_000.0).unwrap();
    let sector_replay = sector.solve_frequency(1_000.0).unwrap();
    assert_complex_relative_close(
        "full modeling sector magnetic impedance",
        sector_first.magnetic_impedance_ohm,
        sector_replay.magnetic_impedance_ohm,
        1.0e-12,
    );
    assert_relative_close(
        "full modeling sector eddy loss",
        sector_first.full_revolution_eddy_loss_w,
        sector_replay.full_revolution_eddy_loss_w,
        1.0e-12,
    );
    assert_relative_close(
        "full modeling sector magnetic energy",
        sector_first.full_revolution_magnetic_energy_j,
        sector_replay.full_revolution_magnetic_energy_j,
        1.0e-12,
    );
    assert_residual_close(
        "full modeling sector periodic residual",
        sector_first.periodic_residual_max,
        sector_replay.periodic_residual_max,
    );
    assert_residual_close(
        "full modeling sector gauge residual",
        sector_first.gauge_residual_max,
        sector_replay.gauge_residual_max,
    );
    assert_residual_close(
        "full modeling sector charge-continuity residual",
        sector_first.charge_continuity_relative_residual,
        sector_replay.charge_continuity_relative_residual,
    );
    assert_complex_vectors_close(
        "full modeling sector vector-potential edge DOFs",
        &sector_first.vector_potential_edge_dofs,
        &sector_replay.vector_potential_edge_dofs,
    );

    eprintln!("native-x86 evidence: monolithic build/solve");
    let discretization = CylinderFixtureDiscretization {
        n_circumference: 8,
        n_box_depth: 2,
        n_cap_radial: 1,
        n_aperture_radial: 1,
        n_port_depth: 1,
    };
    let fixture = cylindrical_box_with_circular_aperture(0.08, 0.06, 0.02, discretization).unwrap();
    let monolithic = MonolithicFemBemCoupling::from_fixture(&fixture, 343.0, 1.2).unwrap();
    let drive = CouplingDrive::DriverVelocity(Complex64::new(1.0e-3, 0.0));
    let monolithic_first = monolithic.solve_frequency(120.0, drive.clone()).unwrap();
    let monolithic_replay = monolithic.solve_frequency(120.0, drive).unwrap();
    assert_eq!(
        monolithic_first.relative_residual,
        monolithic_replay.relative_residual
    );
    assert_eq!(
        monolithic_first.interior_relative_residual,
        monolithic_replay.interior_relative_residual
    );
    assert_eq!(
        monolithic_first.interface_relative_residual,
        monolithic_replay.interface_relative_residual
    );
    assert_eq!(
        max_complex_difference(
            &monolithic_first.interior_pressure,
            &monolithic_replay.interior_pressure,
        ),
        0.0
    );
    assert_eq!(
        max_complex_difference(
            &monolithic_first.aperture_velocity,
            &monolithic_replay.aperture_velocity,
        ),
        0.0
    );

    eprintln!("native-x86 evidence: product coupled force/mode solve");
    let product_design = native_x86_product_design();
    product_design.validate().unwrap();
    let product_mesh = build_product_coupling_mesh(
        &product_design,
        ProductInteriorMeshConfig {
            max_element_size_m: 0.06,
            min_cells_per_axis: 2,
            max_cells_per_axis: 16,
        },
    )
    .unwrap();
    let product_mesh_reference = product_mesh.clone();
    let product = MonolithicSealedProductCoupling::from_coupling_mesh(
        &product_design,
        product_mesh,
        343.0,
        1.204,
    )
    .unwrap();
    let product_first = product
        .solve_frequency(&product_design, 120.0, Complex64::new(2.0, 0.0))
        .unwrap();
    let product_replay = product
        .solve_frequency(&product_design, 120.0, Complex64::new(2.0, 0.0))
        .unwrap();

    assert_eq!(
        product_first.interior_pressure, product_replay.interior_pressure,
        "product interior pressure replay drift"
    );
    assert_eq!(
        product_first.exterior_pressure, product_replay.exterior_pressure,
        "product exterior pressure replay drift"
    );
    assert_eq!(
        product_first.driver_rear_pressure_force_n,
        product_replay.driver_rear_pressure_force_n
    );
    assert_eq!(
        product_first.driver_front_pressure_force_n,
        product_replay.driver_front_pressure_force_n
    );
    assert_eq!(
        product_first.structural_generalized_displacement_peak,
        product_replay.structural_generalized_displacement_peak
    );
    assert_eq!(
        product_first.structural_velocity_peak,
        product_replay.structural_velocity_peak
    );
    assert!(product_first.relative_residual.is_finite());
    assert!(product_first.interior_relative_residual.is_finite());
    assert!(product_first.exterior_relative_residual.is_finite());
    assert!(product_first.structural_relative_residual.is_finite());
    assert!(product_first.relative_residual < 1.0e-10);

    let driver_patch = product_mesh_reference
        .interior
        .patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
        .unwrap();
    let rear_reference = integrate_fem_pressure_force(
        &product_mesh_reference.interior.mesh,
        &product_first.interior_pressure,
        &driver_patch.boundary_faces,
    );
    let driver_trace = product_mesh_reference
        .traces
        .get(&ProductInteriorPatchKind::DriverRear { driver_index: 0 })
        .unwrap();
    let front_reference: Complex64 = driver_trace
        .bem_elements
        .iter()
        .map(|&element| {
            product_first.exterior_pressure[element]
                * product_mesh_reference.exterior.mesh.elements[element].area
        })
        .sum();
    assert_complex_relative_close(
        "product rear pressure force reference",
        product_first.driver_rear_pressure_force_n[0],
        rear_reference,
        1.0e-12,
    );
    assert_complex_relative_close(
        "product front pressure force reference",
        product_first.driver_front_pressure_force_n[0],
        front_reference,
        1.0e-12,
    );
    let product_net_force = product_first.driver_rear_pressure_force_n[0]
        - product_first.driver_front_pressure_force_n[0];
    let reference_net_force = rear_reference - front_reference;
    assert_complex_relative_close(
        "product net coupled pressure force reference",
        product_net_force,
        reference_net_force,
        1.0e-12,
    );

    let feedback = product_design.structural_acoustic.as_ref().unwrap();
    let block = StructuralAcousticModalBlock::from_feedback(feedback, 120.0).unwrap();
    let rigid_faces: Vec<usize> = product_mesh_reference
        .interior
        .patches
        .iter()
        .filter(|patch| matches!(patch.kind, ProductInteriorPatchKind::RigidWall(_)))
        .flat_map(|patch| patch.boundary_faces.iter().copied())
        .collect();
    let rigid_area: f64 = rigid_faces
        .iter()
        .map(|&face| boundary_face_area(&product_mesh_reference.interior.mesh, face))
        .sum();
    let pressure_projection_peak = rigid_faces
        .iter()
        .map(|&face| {
            let nodes = &product_mesh_reference.interior.mesh.boundaries[face].nodes;
            let [a, b, c] = [nodes[0], nodes[1], nodes[2]];
            let average = (product_first.interior_pressure[a]
                + product_first.interior_pressure[b]
                + product_first.interior_pressure[c])
                / 3.0;
            average * boundary_face_area(&product_mesh_reference.interior.mesh, face)
        })
        .sum::<Complex64>()
        / rigid_area;
    let generalized_force_reference =
        pressure_projection_peak * block.pressure_to_generalized_force()[0];
    let generalized_displacement_reference =
        generalized_force_reference / block.dynamic_stiffness()[0].conj();
    let surface_velocity_reference =
        block.generalized_to_surface_velocity()[0].conj() * generalized_displacement_reference;
    assert_eq!(
        product_first.structural_generalized_displacement_peak.len(),
        1
    );
    assert_eq!(product_first.structural_velocity_peak.len(), 1);
    assert_complex_relative_close(
        "product modal displacement reference",
        product_first.structural_generalized_displacement_peak[0],
        generalized_displacement_reference,
        1.0e-10,
    );
    assert_complex_relative_close(
        "product modal surface velocity reference",
        product_first.structural_velocity_peak[0],
        surface_velocity_reference,
        1.0e-10,
    );
    let independent_response = feedback
        .response_at(
            120.0,
            pressure_projection_peak.conj() / std::f64::consts::SQRT_2,
        )
        .unwrap();
    assert_complex_relative_close(
        "product modal displacement independent response",
        product_first.structural_generalized_displacement_peak[0],
        independent_response.displacement_peak(),
        1.0e-10,
    );
    assert_complex_relative_close(
        "product modal velocity independent response",
        product_first.structural_velocity_peak[0],
        independent_response.velocity_bem_peak(),
        1.0e-10,
    );

    eprintln!("native-x86 evidence: emitting record");
    let artifact_checksum = Sha256::digest(artifact.canonical_json().unwrap());
    let simd = json!({
        "sse2": std::is_x86_feature_detected!("sse2"),
        "avx2": std::is_x86_feature_detected!("avx2"),
        "fma": std::is_x86_feature_detected!("fma"),
    });
    let evidence = json!({
        "schema_version": 1,
        "architecture": std::env::consts::ARCH,
        "cpu_vendor": cpu_vendor(),
        "cpu_model": cpu_model(),
        "compiler": compiler_version(),
        "backend": ArtifactBackend::Cpu,
        "dispatch_path": "native-x86_64-cpu/scalar-modeling-and-native-fem-solvers",
        "simd_features_available": simd,
        "driver_artifact_sha256": format!("{artifact_checksum:x}"),
        "driver_derivative": [derivative.re, derivative.im],
        "axisymmetric": {
            "field_dofs": axis_first.pressure_rms_pa.len(),
            "integrated_current": [axis_first.current_rms_a.re, axis_first.current_rms_a.im],
            "impedance": [axis_first.impedance_ohm.re, axis_first.impedance_ohm.im],
            "input_power_w": axis_first.power.input_w,
            "power_residual_w": axis_first.power.residual_w,
            "observation_spl_db": axis_first.observation_spl_db,
        },
        "sector": {
            "field_dofs": sector_first.vector_potential_edge_dofs.len(),
            "mode_field_dofs": sector_first.electric_potential_node_dofs.len(),
            "impedance": [sector_first.magnetic_impedance_ohm.re, sector_first.magnetic_impedance_ohm.im],
            "eddy_loss_w": sector_first.full_revolution_eddy_loss_w,
            "periodic_residual": sector_first.periodic_residual_max,
        },
        "monolithic": {
            "interior_field_dofs": monolithic_first.interior_pressure.len(),
            "aperture_field_dofs": monolithic_first.aperture_velocity.len(),
            "relative_residual": monolithic_first.relative_residual,
            "interface_residual": monolithic_first.interface_relative_residual,
        },
        "product_coupled": {
            "interior_field_dofs": product_first.interior_pressure.len(),
            "exterior_field_dofs": product_first.exterior_pressure.len(),
            "relative_residual": product_first.relative_residual,
            "interior_residual": product_first.interior_relative_residual,
            "exterior_residual": product_first.exterior_relative_residual,
            "force_identity": "P1 face-area integration of FEM rear and BEM front pressure; net = rear - front",
            "rear_pressure_force_n": [product_first.driver_rear_pressure_force_n[0].re, product_first.driver_rear_pressure_force_n[0].im],
            "front_pressure_force_n": [product_first.driver_front_pressure_force_n[0].re, product_first.driver_front_pressure_force_n[0].im],
            "net_pressure_force_n": [product_net_force.re, product_net_force.im],
            "rear_reference_force_n": [rear_reference.re, rear_reference.im],
            "front_reference_force_n": [front_reference.re, front_reference.im],
            "net_reference_force_n": [reference_net_force.re, reference_net_force.im],
            "reference_relative_error": (product_net_force - reference_net_force).norm() / product_net_force.norm().max(reference_net_force.norm()).max(f64::MIN_POSITIVE),
            "modal": {
                "identity": "rigid-wall pressure projection -> generalized force -> modal displacement -> surface velocity",
                "mode_count": block.mode_count(),
                "frequency_hz": block.frequency_hz(),
                "natural_frequency_hz": feedback.modes[0].natural_frequency_hz,
                "pressure_projection_peak_pa": [pressure_projection_peak.re, pressure_projection_peak.im],
                "generalized_force_n": [generalized_force_reference.re, generalized_force_reference.im],
                "generalized_displacement_peak_m": [product_first.structural_generalized_displacement_peak[0].re, product_first.structural_generalized_displacement_peak[0].im],
                "generalized_displacement_reference_peak_m": [generalized_displacement_reference.re, generalized_displacement_reference.im],
                "surface_velocity_peak_m_per_s": [product_first.structural_velocity_peak[0].re, product_first.structural_velocity_peak[0].im],
                "surface_velocity_reference_peak_m_per_s": [surface_velocity_reference.re, surface_velocity_reference.im],
                "reference_relative_error": (product_first.structural_generalized_displacement_peak[0] - generalized_displacement_reference).norm() / product_first.structural_generalized_displacement_peak[0].norm().max(generalized_displacement_reference.norm()).max(f64::MIN_POSITIVE),
                "surface_velocity_reference_relative_error": (product_first.structural_velocity_peak[0] - surface_velocity_reference).norm() / product_first.structural_velocity_peak[0].norm().max(surface_velocity_reference.norm()).max(f64::MIN_POSITIVE),
                "independent_response_displacement_peak_m": [independent_response.displacement_peak().re, independent_response.displacement_peak().im],
                "independent_response_velocity_peak_m_per_s": [independent_response.velocity_bem_peak().re, independent_response.velocity_bem_peak().im],
                "independent_response_displacement_relative_error": (product_first.structural_generalized_displacement_peak[0] - independent_response.displacement_peak()).norm() / product_first.structural_generalized_displacement_peak[0].norm().max(independent_response.displacement_peak().norm()).max(f64::MIN_POSITIVE),
                "independent_response_velocity_relative_error": (product_first.structural_velocity_peak[0] - independent_response.velocity_bem_peak()).norm() / product_first.structural_velocity_peak[0].norm().max(independent_response.velocity_bem_peak().norm()).max(f64::MIN_POSITIVE),
                "structural_residual": product_first.structural_relative_residual,
                "independent_response_checked": true
            }
        },
    });

    // Keep the producer's raw measurements together with the execution
    // contract.  The stable runner validates this file and writes its
    // checksum sidecar; doing the serialization here means the artifact is
    // emitted only after every replay assertion above has passed.
    if let Some(output_dir) = std::env::var_os("SONIUM_OUTPUT_DIR") {
        let output_dir = PathBuf::from(output_dir);
        fs::create_dir_all(&output_dir).unwrap();
        let artifact = json!({
            "schema": "sonium.speaker-modeling.native-x86-sm3.v1",
            "status": "pass",
            "scope": "native Linux x86_64 CPU/f64; Arm64 and Windows waived",
            "producer": evidence.clone(),
            "replay": {
                "status": "pass",
                "relative_tolerance": 1.0e-12,
            "residual_absolute_floor": 1.0e-15,
            "scalar_potential_gauge": "lowest admissible periodic master node",
            "meaningful_drift_rejected": true,
            "coupled_force_and_modal_observables": "repeated product solve plus independent FEM/BEM and modal-equation references",
            "product_state_replay_exact": true
        },
            "cross_architecture": {
                "status": "waived",
                "targets": ["aarch64", "windows"],
                "reason": "user-authorized waiver; no cross-architecture budget is claimed"
            }
        });
        let bytes = serde_json::to_vec_pretty(&artifact).unwrap();
        fs::write(output_dir.join("native-x86-sm3-v1.json"), bytes).unwrap();
    }
    println!("NATIVE_X86_MODELING_EVIDENCE {}", evidence);
}
