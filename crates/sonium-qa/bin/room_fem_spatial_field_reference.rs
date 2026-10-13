//! Same-domain complex spatial accuracy and refinement of physical RoomSim FEM.

// Rust guideline compliant 2026-02-21

use num_complex::Complex64;
use serde::Deserialize;
use sha2::{Digest, Sha256};
use sonium_qa::{QaResult, QaStatus};
use sonium_sim_room::fem_runner::{
    FemMeshConfig, FemMeshConvergenceSpec, FemSourceModel, run_fem_mesh_convergence_ladder,
};
use sonium_xem::{
    BoundaryConfig, Point3D, RectangularRoom, RoomGeometry, RoomSimulation, Source, SurfaceConfig,
};

const CASE_ID: &str = "roomsim.fem-spatial-field-reference.v1";
const FIXTURE: &str = include_str!("../src/fixtures/room/robin_box_spatial_reference.json");
const FIXTURE_SHA256: &str = "721f114d05567e9a5f83ddfde41940ec54b09352aa2866327a33af8b48c584f5";
// Reference uncertainty stays below 0.01%, far below native accuracy budgets.
const REFERENCE_TOLERANCE: f64 = 1e-4;
// Native complex pressure budgets resolve both phase and amplitude, with a
// material decrease at both refinement intervals rather than a finite-only gate.
const RMS_TOLERANCE: f64 = 0.02;
const POINT_TOLERANCE: f64 = 0.03;
const CONTRACTION_TOLERANCE: f64 = 0.65;
const TRUE_RESIDUAL_TOLERANCE: f64 = 1e-9;

#[derive(Deserialize)]
struct ReferenceLevel {
    order: usize,
    source_count: usize,
    exterior_offset_m: f64,
    linear_relative_residual: f64,
    off_grid_relative_robin_l2: f64,
    pressure_pa: [[f64; 2]; 3],
    previous_level_max_relative_complex_change: Option<f64>,
}

#[derive(Deserialize)]
struct Reference {
    dimensions_m: [f64; 3],
    source_m: [f64; 3],
    probes_m: [[f64; 3]; 3],
    frequency_hz: f64,
    sound_speed_m_s: f64,
    density_kg_m3: f64,
    source_volume_velocity_m3_s: f64,
    normalized_wall_admittance: [f64; 2],
    levels: Vec<ReferenceLevel>,
    placement_cross_check: Option<Box<Reference>>,
}

fn pressures(level: &ReferenceLevel) -> [Complex64; 3] {
    level.pressure_pa.map(|[re, im]| Complex64::new(re, im))
}

fn complex_errors(actual: &[Complex64], reference: &[Complex64]) -> Result<(f64, f64), String> {
    if actual.len() != reference.len()
        || actual.is_empty()
        || actual
            .iter()
            .chain(reference)
            .any(|value| !value.re.is_finite() || !value.im.is_finite())
        || reference.iter().any(|value| value.norm() < 1e-12)
    {
        return Err(
            "complex comparison requires finite matching samples and nonzero references".into(),
        );
    }
    let error_squared = actual
        .iter()
        .zip(reference)
        .map(|(a, r)| (*a - *r).norm_sqr())
        .sum::<f64>();
    let reference_squared = reference.iter().map(|value| value.norm_sqr()).sum::<f64>();
    let point = actual
        .iter()
        .zip(reference)
        .map(|(a, r)| (*a - *r).norm() / r.norm())
        .fold(0.0, f64::max);
    Ok(((error_squared / reference_squared).sqrt(), point))
}

fn validate_reference(reference: &Reference) -> Result<(), String> {
    if reference.dimensions_m != [1.6, 1.5, 1.4]
        || reference.source_m != [0.4, 0.35, 0.35]
        || reference.probes_m != [[1.2, 1.1, 1.0], [1.0, 0.95, 0.9], [1.25, 0.6, 0.85]]
        || reference.frequency_hz != 400.0
        || reference.sound_speed_m_s != 343.0
        || reference.density_kg_m3 != 1.204
        || reference.source_volume_velocity_m3_s != 5.88e-4
        || reference.normalized_wall_admittance != [1.0, 0.0]
        || reference.levels.len() != 3
    {
        return Err(
            "independent reference does not describe the original native impedance-box fixture"
                .into(),
        );
    }
    for (index, level) in reference.levels.iter().enumerate() {
        if level.order != [16, 20, 24][index]
            || level.source_count != 6 * level.order * level.order
            || !level.exterior_offset_m.is_finite()
            || level.exterior_offset_m <= 0.0
            || !level.linear_relative_residual.is_finite()
            || !(0.0..=1e-9).contains(&level.linear_relative_residual)
            || !level.off_grid_relative_robin_l2.is_finite()
            || level.off_grid_relative_robin_l2 < 0.0
        {
            return Err("invalid reference resolution or independent equation residual".into());
        }
        complex_errors(&pressures(level), &pressures(level))?;
        if index > 0 {
            let measured =
                complex_errors(&pressures(&reference.levels[index - 1]), &pressures(level))?.1;
            let declared = level
                .previous_level_max_relative_complex_change
                .ok_or("missing reference refinement evidence")?;
            if !declared.is_finite() || (measured - declared).abs() > 1e-12 {
                return Err(
                    "reference refinement metric disagrees with stored complex pressures".into(),
                );
            }
            if level.off_grid_relative_robin_l2
                > 0.5 * reference.levels[index - 1].off_grid_relative_robin_l2
            {
                return Err("independent off-grid Robin residual does not contract".into());
            }
        }
    }
    let previous = reference.levels[1]
        .previous_level_max_relative_complex_change
        .unwrap();
    let fine = &reference.levels[2];
    let final_change = fine.previous_level_max_relative_complex_change.unwrap();
    if final_change > REFERENCE_TOLERANCE
        || fine.off_grid_relative_robin_l2 > REFERENCE_TOLERANCE
        || final_change > 0.3 * previous
    {
        return Err(
            "independent reference has not reached its refinement and boundary budgets".into(),
        );
    }
    Ok(())
}

fn run() -> Result<QaResult, String> {
    let hash = format!("{:x}", Sha256::digest(FIXTURE.as_bytes()));
    if hash != FIXTURE_SHA256 {
        return Err("reference fixture checksum mismatch".into());
    }
    let reference: Reference = serde_json::from_str(FIXTURE).map_err(|error| error.to_string())?;
    validate_reference(&reference)?;
    let cross = reference
        .placement_cross_check
        .as_ref()
        .ok_or("missing independent source-placement check")?;
    validate_reference(cross)?;
    if reference.levels[2].exterior_offset_m == cross.levels[2].exterior_offset_m {
        return Err("reference source-placement check did not change the exterior basis".into());
    }
    let expected = pressures(&cross.levels[2]);
    let placement_error = complex_errors(&pressures(&reference.levels[2]), &expected)?.1;
    if placement_error > REFERENCE_TOLERANCE {
        return Err("independent source placements disagree".into());
    }
    let probes = reference.probes_m.map(|[x, y, z]| Point3D::new(x, y, z));
    let mut simulation = RoomSimulation::new(
        RoomGeometry::Rectangular(RectangularRoom::new(1.6, 1.5, 1.4)),
        vec![Source::omnidirectional(Point3D::new(0.4, 0.35, 0.35), 1.0)],
        probes.to_vec(),
    );
    let matched = SurfaceConfig::Admittance {
        real: 1.0,
        imag: 0.0,
    };
    simulation.boundaries = BoundaryConfig {
        floor: matched.clone(),
        ceiling: matched.clone(),
        walls: matched,
        front_wall: None,
        back_wall: None,
        left_wall: None,
        right_wall: None,
    };
    simulation.frequencies = vec![400.0];
    let specs =
        [("coarse", 2.0), ("medium", 4.0 / 3.0), ("fine", 1.0)].map(|(label, grading_factor)| {
            FemMeshConvergenceSpec::new(
                label,
                FemMeshConfig {
                    elements_per_wavelength: 72.0,
                    grading_factor,
                    fine_radius: 0.0,
                    modal_damping_ratio: 0.0,
                    // Solve below the independently checked equation-residual budget.
                    gmres_tolerance: 1e-10,
                    source_model: FemSourceModel::PointLoad,
                    ..FemMeshConfig::default()
                },
            )
        });
    let evidence =
        run_fem_mesh_convergence_ladder(&simulation, &specs, 1e-12, 1e-9, 1e-12, 0.03, 0.05)
            .map_err(|error| error.to_string())?;
    let mut record = QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("reference_placement_error", placement_error)
        .maximum("reference_placement_error", REFERENCE_TOLERANCE)
        .metric(
            "reference_robin_residual",
            cross.levels[2].off_grid_relative_robin_l2,
        )
        .maximum("reference_robin_residual", REFERENCE_TOLERANCE)
        .metric(
            "reference_final_complex_change",
            cross.levels[2]
                .previous_level_max_relative_complex_change
                .unwrap(),
        )
        .maximum("reference_final_complex_change", REFERENCE_TOLERANCE);
    let mut rms_errors = Vec::new();
    let mut point_errors = Vec::new();
    for (label, level) in [
        ("coarse", &evidence.coarse),
        ("medium", &evidence.medium),
        ("fine", &evidence.fine),
    ] {
        level.field.validate()?;
        level.diagnostics.require_converged()?;
        let actual = level.field.sample_p1(0, 0, &probes)?;
        let (rms, point) = complex_errors(&actual, &expected)?;
        println!(
            "{label}: complex RMS={rms:.6e}, maximum={point:.6e}, true residual={:.6e}",
            level.diagnostics.max_residual
        );
        record = record
            .metric(format!("{label}_complex_rms_error"), rms)
            .metric(format!("{label}_complex_max_error"), point)
            .metric(
                format!("{label}_true_residual"),
                level.diagnostics.max_residual,
            )
            .maximum(format!("{label}_true_residual"), TRUE_RESIDUAL_TOLERANCE);
        rms_errors.push(rms);
        point_errors.push(point);
    }
    let rms_contraction = rms_errors
        .windows(2)
        .map(|pair| pair[1] / pair[0].max(1e-12))
        .fold(0.0, f64::max);
    let point_contraction = point_errors
        .windows(2)
        .map(|pair| pair[1] / pair[0].max(1e-12))
        .fold(0.0, f64::max);
    record = record.maximum("fine_complex_rms_error", RMS_TOLERANCE).maximum("fine_complex_max_error", POINT_TOLERANCE)
        .metric("max_rms_refinement_ratio", rms_contraction).maximum("max_rms_refinement_ratio", CONTRACTION_TOLERANCE)
        .metric("max_point_refinement_ratio", point_contraction).maximum("max_point_refinement_ratio", CONTRACTION_TOLERANCE)
        .input_checksum("analytic/deterministic-fixture:roomsim-robin-box-point-source-v2")
        .reference("independent exterior fundamental solutions for the same finite Robin box; off-grid boundary and basis-placement refinement checks")
        .reference_checksum(format!("sha256:{FIXTURE_SHA256}"))
        .dimension("1.6x1.5x1.4 m room, normalized wall admittance 1, point source, three off-node probes, 400 Hz; 36/54/72 EPW")
        .formulation("CPU physical FEM P1 complex nodal pressure, independent finite-domain Robin Helmholtz reference")
        .tolerance("fine complex RMS <=2%, maximum <=3%; both refinement ratios <=0.65; all true residuals <=1e-9; reference uncertainty <=1e-4")
        .seed("roomsim-fem-spatial-field:cpu:v2");
    if let Err(error) = record.release_eligible() {
        record.status = QaStatus::Failed;
        record.message = Some(format!(
            "complex spatial accuracy or refinement budget exceeded: {error}"
        ));
    }
    Ok(record)
}

fn main() {
    let result = run().unwrap_or_else(|error| {
        QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu")
    });
    println!(
        "QA_RESULT: {}",
        result
            .to_json_line()
            .expect("spatial field result must serialize")
    );
    if result.status == QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn phase_error_cannot_hide_behind_exact_magnitudes() {
        let reference = [Complex64::new(1.0, 0.0); 3];
        let actual = [Complex64::new(0.0, 1.0); 3];
        let (rms, point) = complex_errors(&actual, &reference).unwrap();
        assert!(rms > RMS_TOLERANCE && point > POINT_TOLERANCE);
    }
    #[test]
    fn independent_reference_certifies_the_original_domain_and_refinement() {
        let mut reference: Reference = serde_json::from_str(FIXTURE).unwrap();
        validate_reference(&reference).unwrap();
        validate_reference(reference.placement_cross_check.as_ref().unwrap()).unwrap();
        reference.normalized_wall_admittance = [0.0, 0.0];
        assert!(validate_reference(&reference).is_err());
    }
}
