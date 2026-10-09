//! Release-facing admission checks for the body-fitted sealed product mesh.
//!
//! This is deliberately a bounded PLC gate: it proves deterministic lowering
//! of one valid sealed product, boundary-face manifoldness/orientation, patch
//! conservation, volume conservation, and typed rejection of malformed
//! product/configuration inputs. It does not promote the full product ladder,
//! port/brace lowering, or the coupled FEM--BEM solve.

use std::collections::BTreeMap;
use std::time::Instant;

use sonium_fem::mesh::Point;
use sonium_qa::QaResult;
use sonium_sim_speaker_modeling::{
    CabinetDesign, CabinetMaterial, CrossoverBranch, CrossoverNetwork, ProductDesign,
    ProductDriver, ProductInteriorMesh, ProductInteriorMeshConfig, ProductInteriorPatchKind,
    build_body_fitted_sealed_product_interior_mesh,
};

const CASE_ID: &str = "mesher.product-plc-admission.v1";

#[derive(Debug)]
struct Metrics {
    node_count: usize,
    tetrahedron_count: usize,
    boundary_face_count: usize,
    patch_count: usize,
    orientation_sign_consistent: usize,
    manifold_faces: usize,
    minimum_tetra_quality: f64,
    fluid_volume_relative_error: f64,
    patch_area_relative_error: f64,
    driver_area_relative_error: f64,
    deterministic_replays: usize,
    malformed_rejections: usize,
    config_rejections: usize,
    elapsed_ms: u64,
}

fn fixture() -> ProductDesign {
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
            id: "plc-driver".to_string(),
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
        structural_acoustic: None,
    }
}

fn config() -> ProductInteriorMeshConfig {
    ProductInteriorMeshConfig {
        max_element_size_m: 0.025,
        min_cells_per_axis: 2,
        max_cells_per_axis: 256,
    }
}

fn signed_volume_six(nodes: &[Point], tetrahedron: &[usize]) -> Result<f64, String> {
    if tetrahedron.len() != 4 {
        return Err("non-tetrahedral element in body-fitted product mesh".into());
    }
    let a = nodes
        .get(tetrahedron[0])
        .ok_or_else(|| "tetrahedron references a missing node".to_string())?;
    let b = nodes
        .get(tetrahedron[1])
        .ok_or_else(|| "tetrahedron references a missing node".to_string())?;
    let c = nodes
        .get(tetrahedron[2])
        .ok_or_else(|| "tetrahedron references a missing node".to_string())?;
    let d = nodes
        .get(tetrahedron[3])
        .ok_or_else(|| "tetrahedron references a missing node".to_string())?;
    let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
    let ac = [c.x - a.x, c.y - a.y, c.z - a.z];
    let ad = [d.x - a.x, d.y - a.y, d.z - a.z];
    let cross = [
        ac[1] * ad[2] - ac[2] * ad[1],
        ac[2] * ad[0] - ac[0] * ad[2],
        ac[0] * ad[1] - ac[1] * ad[0],
    ];
    Ok(ab[0] * cross[0] + ab[1] * cross[1] + ab[2] * cross[2])
}

fn sorted_face(nodes: &[usize]) -> Result<Vec<usize>, String> {
    if nodes.len() != 3 {
        return Err("boundary face is not triangular".into());
    }
    let mut key = nodes.to_vec();
    key.sort_unstable();
    Ok(key)
}

fn topology_and_conservation(
    product: &ProductInteriorMesh,
    design: &ProductDesign,
) -> Result<(usize, usize, f64, f64, f64, f64), String> {
    let mesh = &product.mesh;
    if mesh.elements.is_empty() || mesh.nodes.is_empty() || mesh.boundaries.is_empty() {
        return Err("body-fitted PLC returned an empty mesh or boundary".into());
    }

    let mut face_counts: BTreeMap<Vec<usize>, usize> = BTreeMap::new();
    let mut orientation_sign = 0.0;
    let mut minimum_tetra_quality = f64::INFINITY;
    for element in &mesh.elements {
        let signed = signed_volume_six(&mesh.nodes, &element.nodes)?;
        if !signed.is_finite() || signed.abs() <= 1.0e-14 {
            return Err("body-fitted PLC contains a degenerate tetrahedron".into());
        }
        if orientation_sign == 0.0 {
            orientation_sign = signed.signum();
        } else if signed.signum() != orientation_sign {
            return Err("body-fitted PLC contains inconsistent tetrahedron orientation".into());
        }
        let points = element
            .nodes
            .iter()
            .map(|&index| {
                mesh.nodes
                    .get(index)
                    .ok_or_else(|| "tetrahedron references a missing node".to_string())
            })
            .collect::<Result<Vec<_>, _>>()?;
        let mut max_edge = 0.0_f64;
        for (left, right) in [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)] {
            let dx = points[left].x - points[right].x;
            let dy = points[left].y - points[right].y;
            let dz = points[left].z - points[right].z;
            max_edge = max_edge.max((dx * dx + dy * dy + dz * dz).sqrt());
        }
        let quality = signed.abs() / max_edge.powi(3);
        if !quality.is_finite() || quality <= 0.0 {
            return Err("body-fitted PLC contains a non-positive quality metric".into());
        }
        minimum_tetra_quality = minimum_tetra_quality.min(quality);
        let n = &element.nodes;
        for face in [
            [n[0], n[1], n[2]],
            [n[0], n[1], n[3]],
            [n[0], n[2], n[3]],
            [n[1], n[2], n[3]],
        ] {
            *face_counts.entry(sorted_face(&face)?).or_default() += 1;
        }
    }

    let mut boundary_counts = BTreeMap::new();
    for face in &mesh.boundaries {
        let key = sorted_face(&face.nodes)?;
        if boundary_counts.insert(key.clone(), 1).is_some() {
            return Err("body-fitted PLC boundary contains a duplicate face".into());
        }
        if face_counts.get(&key) != Some(&1) {
            return Err("body-fitted PLC boundary face is not an exterior one-face".into());
        }
    }
    let nonmanifold_faces = face_counts.values().filter(|&&count| count > 2).count();
    if nonmanifold_faces != 0 {
        return Err(format!(
            "body-fitted PLC has {nonmanifold_faces} non-manifold faces"
        ));
    }
    let exterior_faces = face_counts.values().filter(|&&count| count == 1).count();
    if exterior_faces != mesh.boundaries.len() {
        return Err("body-fitted PLC boundary does not cover every exterior face".into());
    }

    let expected_kinds = [
        ProductInteriorPatchKind::RigidWall(sonium_sim_speaker_modeling::InteriorWall::Left),
        ProductInteriorPatchKind::RigidWall(sonium_sim_speaker_modeling::InteriorWall::Right),
        ProductInteriorPatchKind::RigidWall(sonium_sim_speaker_modeling::InteriorWall::Bottom),
        ProductInteriorPatchKind::RigidWall(sonium_sim_speaker_modeling::InteriorWall::Top),
        ProductInteriorPatchKind::RigidWall(sonium_sim_speaker_modeling::InteriorWall::Back),
        ProductInteriorPatchKind::RigidWall(sonium_sim_speaker_modeling::InteriorWall::Front),
        ProductInteriorPatchKind::DriverRear { driver_index: 0 },
    ];
    if product.patches.len() != expected_kinds.len()
        || expected_kinds
            .iter()
            .any(|kind| product.patch(*kind).is_none())
    {
        return Err("body-fitted PLC patch registry is incomplete".into());
    }
    let mut patch_faces = BTreeMap::new();
    let mut patch_area_sum = 0.0;
    for patch in &product.patches {
        if patch.area_m2 <= 0.0 || !patch.area_m2.is_finite() || patch.boundary_faces.is_empty() {
            return Err("body-fitted PLC contains an empty or non-finite patch".into());
        }
        patch_area_sum += patch.area_m2;
        for &face in &patch.boundary_faces {
            if patch_faces.insert(face, patch.marker).is_some() {
                return Err("body-fitted PLC patches overlap on a boundary face".into());
            }
            if mesh.boundaries.get(face).map(|boundary| boundary.marker) != Some(patch.marker) {
                return Err("body-fitted PLC patch marker disagrees with its boundary face".into());
            }
        }
    }
    if patch_faces.len() != mesh.boundaries.len() {
        return Err("body-fitted PLC patches do not partition the boundary".into());
    }

    let expected_volume = design.cabinet.volume_m3();
    let volume_error =
        (product.diagnostics.fluid_volume_m3 - expected_volume).abs() / expected_volume;
    let surface_area = 2.0
        * (design.cabinet.box_width_m * design.cabinet.box_height_m
            + design.cabinet.box_width_m * design.cabinet.box_depth_m
            + design.cabinet.box_height_m * design.cabinet.box_depth_m);
    let patch_area_error = (patch_area_sum - surface_area).abs() / surface_area;
    let driver_area = product
        .patch(ProductInteriorPatchKind::DriverRear { driver_index: 0 })
        .ok_or_else(|| "driver patch disappeared".to_string())?
        .area_m2;
    let expected_driver_area = std::f64::consts::PI * design.drivers[0].radius_m.powi(2);
    let driver_area_error = (driver_area - expected_driver_area).abs() / expected_driver_area;
    if volume_error > 1.0e-10 || patch_area_error > 1.0e-10 || driver_area_error > 0.02 {
        return Err(format!(
            "PLC conservation failed: volume={volume_error:.3e}, patches={patch_area_error:.3e}, driver={driver_area_error:.3e}"
        ));
    }
    Ok((
        exterior_faces,
        if nonmanifold_faces == 0 { 1 } else { 0 },
        volume_error,
        patch_area_error,
        driver_area_error,
        minimum_tetra_quality,
    ))
}

fn compare_replay(left: &ProductInteriorMesh, right: &ProductInteriorMesh) -> bool {
    if left.geometry_fingerprint != right.geometry_fingerprint
        || left.mesh.nodes.len() != right.mesh.nodes.len()
        || left.mesh.elements.len() != right.mesh.elements.len()
        || left.mesh.boundaries.len() != right.mesh.boundaries.len()
        || left.patches.len() != right.patches.len()
    {
        return false;
    }
    for (a, b) in left.mesh.nodes.iter().zip(&right.mesh.nodes) {
        if a.x.to_bits() != b.x.to_bits()
            || a.y.to_bits() != b.y.to_bits()
            || a.z.to_bits() != b.z.to_bits()
        {
            return false;
        }
    }
    for (a, b) in left.mesh.elements.iter().zip(&right.mesh.elements) {
        if a.nodes != b.nodes {
            return false;
        }
    }
    for (a, b) in left.mesh.boundaries.iter().zip(&right.mesh.boundaries) {
        if a.nodes != b.nodes
            || a.marker != b.marker
            || a.element_idx != b.element_idx
            || a.local_idx != b.local_idx
        {
            return false;
        }
    }
    left.patches.iter().zip(&right.patches).all(|(a, b)| {
        a.kind == b.kind
            && a.marker == b.marker
            && a.boundary_faces == b.boundary_faces
            && a.area_m2.to_bits() == b.area_m2.to_bits()
    })
}

fn run() -> Result<Metrics, String> {
    let started = Instant::now();
    let design = fixture();
    design
        .validate()
        .map_err(|error| format!("valid product fixture rejected: {error}"))?;
    let first = build_body_fitted_sealed_product_interior_mesh(&design, config())
        .map_err(|error| format!("body-fitted product PLC failed: {error}"))?;
    let second = build_body_fitted_sealed_product_interior_mesh(&design, config())
        .map_err(|error| format!("body-fitted product PLC replay failed: {error}"))?;
    let (
        boundary_face_count,
        manifold_faces,
        volume_error,
        patch_area_error,
        driver_area_error,
        minimum_tetra_quality,
    ) = topology_and_conservation(&first, &design)?;
    if !compare_replay(&first, &second) {
        return Err("body-fitted product PLC is not deterministic across replay".into());
    }

    let mut malformed = design.clone();
    malformed.drivers[0].normal = [1.0, 1.0, 0.0];
    let malformed_rejections =
        usize::from(build_body_fitted_sealed_product_interior_mesh(&malformed, config()).is_err());
    let mut invalid_config = config();
    invalid_config.max_element_size_m = 0.0;
    let config_rejections = usize::from(
        build_body_fitted_sealed_product_interior_mesh(&design, invalid_config).is_err(),
    );
    if malformed_rejections != 1 || config_rejections != 1 {
        return Err("malformed product/configuration was accepted by PLC admission".into());
    }

    Ok(Metrics {
        node_count: first.mesh.nodes.len(),
        tetrahedron_count: first.mesh.elements.len(),
        boundary_face_count,
        patch_count: first.patches.len(),
        orientation_sign_consistent: 1,
        manifold_faces,
        minimum_tetra_quality,
        fluid_volume_relative_error: volume_error,
        patch_area_relative_error: patch_area_error,
        driver_area_relative_error: driver_area_error,
        deterministic_replays: 1,
        malformed_rejections,
        config_rejections,
        elapsed_ms: started.elapsed().as_millis() as u64,
    })
}

fn main() {
    match std::panic::catch_unwind(run) {
        Ok(Ok(metrics)) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("product-plc-admission-v1")
                .metric("node_count", metrics.node_count as f64)
                .metric("tetrahedron_count", metrics.tetrahedron_count as f64)
                .metric("boundary_face_count", metrics.boundary_face_count as f64)
                .metric("patch_count", metrics.patch_count as f64)
                .metric(
                    "orientation_sign_consistent",
                    metrics.orientation_sign_consistent as f64,
                )
                .metric("manifold_faces", metrics.manifold_faces as f64)
                .metric("minimum_tetra_quality", metrics.minimum_tetra_quality)
                .metric(
                    "fluid_volume_relative_error",
                    metrics.fluid_volume_relative_error,
                )
                .metric("patch_area_relative_error", metrics.patch_area_relative_error)
                .metric("driver_area_relative_error", metrics.driver_area_relative_error)
                .metric("deterministic_replays", metrics.deterministic_replays as f64)
                .metric("malformed_rejections", metrics.malformed_rejections as f64)
                .metric("config_rejections", metrics.config_rejections as f64)
                .minimum("node_count", 1.0)
                .minimum("tetrahedron_count", 1.0)
                .minimum("boundary_face_count", 1.0)
                .minimum("patch_count", 7.0)
                .minimum("orientation_sign_consistent", 1.0)
                .minimum("manifold_faces", 1.0)
                .minimum("minimum_tetra_quality", 1.0e-6)
                .maximum("fluid_volume_relative_error", 1.0e-10)
                .maximum("patch_area_relative_error", 1.0e-10)
                .maximum("driver_area_relative_error", 0.02)
                .minimum("deterministic_replays", 1.0)
                .minimum("malformed_rejections", 1.0)
                .minimum("config_rejections", 1.0)
                .input_checksum("analytic/deterministic-fixture:product-plc-v1")
                .reference("box volume/surface conservation, circular driver aperture, boundary incidence and typed rejection invariants")
                .reference_checksum("analytic/deterministic-fixture:product-plc-v1")
                .dimension("3D sealed rectangular cavity; one axis-aligned driver; P1 tetrahedra")
                .formulation("bounded body-fitted sealed product PLC admission and deterministic replay")
                .tolerance("volume and patch area errors <= 1e-10; driver area error <= 2e-2; all required patches/manifold/replay/rejections present")
                .timing_ms("product_plc_admission", metrics.elapsed_ms)
                .elapsed_ms(metrics.elapsed_ms);
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("product PLC result must serialize")
            );
            println!("SUCCESS");
        }
        Ok(Err(error)) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("product PLC failure result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
        Err(_) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, "product PLC QA panicked")
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("product PLC panic result must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
