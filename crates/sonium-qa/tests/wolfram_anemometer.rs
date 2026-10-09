//! Literal steady-state port of Wolfram's MEMS thermal anemometer example.
//!
//! Source:
//! <https://reference.wolfram.com/language/PDEModels/tutorial/HeatTransfer/ModelCollection/Anemometer.html?view=all>

use serde::Deserialize;
use sonium_fem::{
    assembly::{assemble_bilinear_per_element, assemble_linear_per_element},
    basis::PolynomialDegree,
    boundary::generic::{FemSystem, try_apply_homogeneous_dirichlet_dofs},
    forms::{BilinearForm, LinearForm, LocalMatrix, LocalVector},
    mesh::{
        ElementType, Mesh, Point, interpolate_scalar_at_point, triangulate::triangulate_points_2d,
    },
    physics::coupled::conjugate_heat_transfer::{ChtAdvectionFieldForm, ChtConductionForm},
    spaces::{BasisTabulation, H1Space},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::SparseLuFactorization;
use std::{collections::BTreeSet, env, path::PathBuf, process::Command};

// The published Wolfram notebook multiplies its numeric coordinates by one
// millimetre.  Preserve that literal scale here even though the device name
// suggests micrometre dimensions.
const LENGTH_M: f64 = 1.2;
const DEVICE_HEIGHT_M: f64 = 80.0e-3;
const FLOW_HEIGHT_M: f64 = 300.0e-3;
const BASE_LENGTH_M: f64 = 240.0e-3;
const MEMBRANE_THICKNESS_M: f64 = 20.0e-3;
const MEMBRANE_HEIGHT_M: f64 = DEVICE_HEIGHT_M - MEMBRANE_THICKNESS_M;
const SLOPE_RUN_M: f64 = MEMBRANE_HEIGHT_M / std::f64::consts::SQRT_2;
const HEATER_X_MIN_M: f64 = 590.0e-3;
const HEATER_X_MAX_M: f64 = 610.0e-3;
const HEATER_Y_MIN_M: f64 = 70.0e-3;
const SILICON_CONDUCTIVITY_W_MK: f64 = 148.0;
const AIR_CONDUCTIVITY_W_MK: f64 = 0.0239;
const AIR_VOLUMETRIC_HEAT_CAPACITY_J_M3K: f64 = 1.040 * 1012.0;
const MAX_FLOW_SPEED_M_S: f64 = 1.0e-3;
const HEATER_SOURCE_W_M3: f64 = 1.0e7;
// Full-precision live Wolfram reference values. The live-reference leg gates
// them at 1e-6/1e-7 absolute, so these must keep full double precision;
// a rounded max-temperature constant can never satisfy its own gate.
const WOLFRAM_MAX_TEMPERATURE_K: f64 = 119.5191848160281;
const WOLFRAM_SENSOR_DIFFERENCE_K: f64 = 0.0678515;

#[derive(Clone, Copy)]
struct SteadyChtForm {
    conduction: ChtConductionForm,
    advection: ChtAdvectionFieldForm<fn([f64; 3]) -> [f64; 3]>,
}

impl BilinearForm<f64> for SteadyChtForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        trial: &BasisTabulation,
        weight: f64,
        local_matrix: &mut LocalMatrix<f64>,
    ) {
        self.conduction
            .integrate_point(test, trial, weight, local_matrix);
        self.advection
            .integrate_point(test, trial, weight, local_matrix);
    }
}

#[derive(Clone, Copy)]
struct ConstantSourceForm {
    source: f64,
}

impl LinearForm<f64> for ConstantSourceForm {
    fn integrate_point(
        &self,
        test: &BasisTabulation,
        weight: f64,
        local_vector: &mut LocalVector<f64>,
    ) {
        let scaled_weight = weight * test.det_j * self.source;
        for index in 0..test.n_dofs {
            local_vector.add(index, test.value(index) * scaled_weight);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Region {
    Silicon,
    Heater,
    StagnantAir,
    FlowingAir,
}

fn membrane_underside(x: f64) -> f64 {
    let left_end = BASE_LENGTH_M + SLOPE_RUN_M;
    let right_start = LENGTH_M - BASE_LENGTH_M - SLOPE_RUN_M;
    if x <= BASE_LENGTH_M || x >= LENGTH_M - BASE_LENGTH_M {
        0.0
    } else if x < left_end {
        MEMBRANE_HEIGHT_M * (x - BASE_LENGTH_M) / SLOPE_RUN_M
    } else if x <= right_start {
        MEMBRANE_HEIGHT_M
    } else {
        MEMBRANE_HEIGHT_M * (LENGTH_M - BASE_LENGTH_M - x) / SLOPE_RUN_M
    }
}

fn region_at(x: f64, y: f64) -> Region {
    if y >= DEVICE_HEIGHT_M {
        return Region::FlowingAir;
    }
    if (HEATER_X_MIN_M..=HEATER_X_MAX_M).contains(&x) && y >= HEATER_Y_MIN_M {
        return Region::Heater;
    }
    if y >= membrane_underside(x) {
        Region::Silicon
    } else {
        Region::StagnantAir
    }
}

fn flow_speed(y: f64) -> f64 {
    let half_height = FLOW_HEIGHT_M / 2.0;
    let centered = (y - (DEVICE_HEIGHT_M + half_height)) / half_height;
    MAX_FLOW_SPEED_M_S * (1.0 - centered * centered).max(0.0)
}

fn flow_velocity(point: [f64; 3]) -> [f64; 3] {
    [flow_speed(point[1]), 0.0, 0.0]
}

fn element_forms(mesh: &Mesh) -> (Vec<SteadyChtForm>, Vec<ConstantSourceForm>) {
    mesh.elements
        .iter()
        .enumerate()
        .map(|(index, _)| {
            let centroid = mesh.element_centroid(index);
            let region = region_at(centroid.x, centroid.y);
            let (conductivity, capacity, source) = match region {
                Region::Silicon => (SILICON_CONDUCTIVITY_W_MK, 0.0, 0.0),
                Region::Heater => (SILICON_CONDUCTIVITY_W_MK, 0.0, HEATER_SOURCE_W_M3),
                Region::StagnantAir => (AIR_CONDUCTIVITY_W_MK, 0.0, 0.0),
                Region::FlowingAir => (
                    AIR_CONDUCTIVITY_W_MK,
                    AIR_VOLUMETRIC_HEAT_CAPACITY_J_M3K,
                    0.0,
                ),
            };
            (
                SteadyChtForm {
                    conduction: ChtConductionForm {
                        thermal_conductivity: conductivity,
                    },
                    advection: ChtAdvectionFieldForm {
                        volumetric_heat_capacity: capacity,
                        velocity: flow_velocity,
                    },
                },
                ConstantSourceForm { source },
            )
        })
        .unzip()
}

struct Solution {
    mesh: Mesh,
    temperature: Vec<f64>,
    maximum: f64,
    sensor_difference: f64,
    source_power_per_depth: f64,
    relative_residual: f64,
}

fn locally_refined_mesh(fine_spacing: f64) -> Mesh {
    const QUANTIZATION: f64 = 1.0e9;
    let mut quantized = BTreeSet::new();
    let mut insert_grid = |x_min: f64, x_max: f64, y_min: f64, y_max: f64, spacing: f64| {
        let nx = ((x_max - x_min) / spacing).round() as usize;
        let ny = ((y_max - y_min) / spacing).round() as usize;
        for row in 0..=ny {
            for column in 0..=nx {
                let x = x_min + column as f64 * spacing;
                let y = y_min + row as f64 * spacing;
                quantized.insert((
                    (x * QUANTIZATION).round() as i64,
                    (y * QUANTIZATION).round() as i64,
                ));
            }
        }
    };
    // Wolfram permits large elements in the bulk but limits the heater to
    // 0.1 mm². A Delaunay point cloud provides the same local-isotropic
    // refinement without propagating skinny elements across both sensors.
    insert_grid(0.0, LENGTH_M, 0.0, DEVICE_HEIGHT_M + FLOW_HEIGHT_M, 0.01);
    insert_grid(0.58, 0.62, 0.065, 0.085, fine_spacing);
    let points: Vec<_> = quantized
        .into_iter()
        .map(|(x, y)| (x as f64 / QUANTIZATION, y as f64 / QUANTIZATION))
        .collect();
    let triangles = triangulate_points_2d(&points);
    let mut mesh = Mesh::new(2);
    for &(x, y) in &points {
        mesh.add_node(Point::new_2d(x, y));
    }
    for triangle in triangles {
        mesh.add_element(ElementType::Triangle, triangle.to_vec());
    }
    mesh.detect_boundaries();
    mesh
}

fn solve_mesh(mesh: Mesh) -> Solution {
    let space = H1Space::new(PolynomialDegree::P1);
    let (bilinear_forms, source_forms) = element_forms(&mesh);
    let matrix = assemble_bilinear_per_element(&bilinear_forms, &space, &space, &mesh, 2);
    let rhs = assemble_linear_per_element(&source_forms, &space, &mesh, 2);
    let original_matrix = matrix.clone();
    let original_rhs = rhs.clone();
    let source_power_per_depth = original_rhs.iter().sum();
    let mut system = FemSystem::new(matrix, rhs);

    let constrained: Vec<_> = mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, point)| (point.y.abs() < 1.0e-14).then_some(index))
        .collect();
    assert!(
        constrained.len() >= 2,
        "literal Wolfram marker-1 boundary must constrain the full base"
    );
    try_apply_homogeneous_dirichlet_dofs(&mut system, &constrained)
        .expect("apply anemometer point temperatures");

    let csr = system.matrix.to_csr();
    let factor = SparseLuFactorization::factorize(&csr).expect("factor anemometer system");
    let temperature = factor.solve(&system.rhs).expect("solve anemometer system");
    let residual = original_matrix.to_csr().matvec(&temperature);
    let residual_norm = residual
        .iter()
        .zip(&original_rhs)
        .enumerate()
        .filter(|(index, _)| !constrained.contains(index))
        .map(|(_, (lhs, rhs))| (lhs - rhs).powi(2))
        .sum::<f64>()
        .sqrt();
    let rhs_norm = original_rhs
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();

    let left = interpolate_scalar_at_point(&mesh, &temperature, [400.0e-3, DEVICE_HEIGHT_M, 0.0])
        .expect("left sensor interpolation");
    let right = interpolate_scalar_at_point(&mesh, &temperature, [800.0e-3, DEVICE_HEIGHT_M, 0.0])
        .expect("right sensor interpolation");
    let maximum = temperature
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);

    Solution {
        mesh,
        temperature,
        maximum,
        sensor_difference: right - left,
        source_power_per_depth,
        relative_residual: residual_norm / rhs_norm,
    }
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    maximum_temperature_k: f64,
    sensor_difference_k: f64,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/anemometer.wls");
    let output = Command::new(executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
        .expect("run explicit WOLFRAMSCRIPT anemometer reference");
    assert!(
        output.status.success(),
        "Wolfram anemometer reference failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("Wolfram UTF-8 output");
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .expect("Wolfram anemometer JSON output");
    Some(serde_json::from_str(json).expect("parse Wolfram anemometer JSON"))
}

#[test]
fn anemometer_matches_peak_temperature_and_sensor_difference() {
    let coarse = solve_mesh(locally_refined_mesh(1.0e-3));
    let fine = solve_mesh(locally_refined_mesh(5.0e-4));
    eprintln!(
        "Wolfram anemometer diagnostic: coarse/fine nodes={}/{}, Tmax={:.8}/{:.8} K, sensor delta={:.9}/{:.9} K, source={:.8e} W/m, residual={:.3e}",
        coarse.mesh.nodes.len(),
        fine.mesh.nodes.len(),
        coarse.maximum,
        fine.maximum,
        coarse.sensor_difference,
        fine.sensor_difference,
        fine.source_power_per_depth,
        fine.relative_residual,
    );

    assert!(fine.temperature.iter().all(|value| value.is_finite()));
    assert!(fine.temperature.iter().all(|value| *value >= -1.0e-8));
    assert!(
        fine.sensor_difference > 0.0,
        "flow must heat the downstream sensor more"
    );
    assert!(fine.relative_residual < 1.0e-8);
    assert!((fine.source_power_per_depth - 2.0e3).abs() < 1.0e-9);
    assert!((fine.maximum - coarse.maximum).abs() / fine.maximum < 0.08);
    assert!(
        (fine.sensor_difference - coarse.sensor_difference).abs() / fine.sensor_difference.abs()
            < 0.20
    );
    let maximum_gap = (fine.maximum - WOLFRAM_MAX_TEMPERATURE_K).abs() / WOLFRAM_MAX_TEMPERATURE_K;
    let sensor_gap =
        (fine.sensor_difference - WOLFRAM_SENSOR_DIFFERENCE_K).abs() / WOLFRAM_SENSOR_DIFFERENCE_K;
    assert!(
        maximum_gap < 0.01,
        "maximum-temperature parity regressed: gap={maximum_gap:.3e}"
    );
    assert!(
        sensor_gap < 0.10,
        "sensor-difference parity regressed: gap={sensor_gap:.3e}"
    );

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert!((reference.maximum_temperature_k - WOLFRAM_MAX_TEMPERATURE_K).abs() < 1.0e-6);
        assert!((reference.sensor_difference_k - WOLFRAM_SENSOR_DIFFERENCE_K).abs() < 1.0e-7);
    }
}
