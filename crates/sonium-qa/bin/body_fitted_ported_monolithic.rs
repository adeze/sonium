//! Bounded W2-02 body-fitted ported FEM--BEM integration evidence.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_sim_speaker_modeling::fem_bem_coupling::MonolithicSealedProductCoupling;
use sonium_sim_speaker_modeling::{
    CabinetDesign, CabinetMaterial, CrossoverBranch, CrossoverKind, CrossoverNetwork,
    ProductDesign, ProductDriver, ProductInteriorMeshConfig,
    build_body_fitted_sealed_product_interior_mesh,
};
use std::time::Instant;

const CASE_ID: &str = "coupling.body-fitted-ported-monolithic.v1";

fn fixture() -> ProductDesign {
    let mut port = CabinetDesign::default()
        .port
        .expect("default cabinet port fixture");
    port.position_m = [0.0, 0.14];

    ProductDesign {
        cabinet: CabinetDesign {
            box_width_m: 0.40,
            box_height_m: 0.40,
            box_depth_m: 0.30,
            wall_material: CabinetMaterial::Mdf18mm,
            port: Some(port),
            baffle_width_m: 0.44,
            baffle_height_m: 0.44,
            braces: Vec::new(),
        },
        drivers: vec![
            ProductDriver {
                id: "w2-left-woofer".into(),
                radius_m: 0.03,
                position_m: [-0.12, -0.08, 0.0],
                normal: [0.0, 0.0, 1.0],
                force_factor_tm: 5.0,
                coil_resistance_ohm: 6.0,
                coil_reactance_ohm: 1.0,
                mechanical_resistance_n_s_per_m: 1.0,
                mechanical_reactance_n_s_per_m: 2.0,
                crossover_branch: 0,
            },
            ProductDriver {
                id: "w2-centre-tweeter".into(),
                radius_m: 0.025,
                position_m: [0.0, 0.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                force_factor_tm: 5.0,
                coil_resistance_ohm: 6.0,
                coil_reactance_ohm: 1.0,
                mechanical_resistance_n_s_per_m: 1.0,
                mechanical_reactance_n_s_per_m: 2.0,
                crossover_branch: 1,
            },
            ProductDriver {
                id: "w2-right-woofer".into(),
                radius_m: 0.03,
                position_m: [0.12, -0.08, 0.0],
                normal: [0.0, 0.0, 1.0],
                force_factor_tm: 5.0,
                coil_resistance_ohm: 6.0,
                coil_reactance_ohm: 1.0,
                mechanical_resistance_n_s_per_m: 1.0,
                mechanical_reactance_n_s_per_m: 2.0,
                crossover_branch: 2,
            },
        ],
        crossover: CrossoverNetwork {
            branches: vec![
                CrossoverBranch {
                    kind: CrossoverKind::LowPass,
                    corner_frequency_hz: 900.0,
                    gain: 1.0,
                    delay_s: 0.0,
                },
                CrossoverBranch {
                    kind: CrossoverKind::HighPass,
                    corner_frequency_hz: 900.0,
                    gain: 0.8,
                    delay_s: 0.0001,
                },
                CrossoverBranch {
                    kind: CrossoverKind::FullRange,
                    corner_frequency_hz: 900.0,
                    gain: 0.5,
                    delay_s: 0.0002,
                },
            ],
        },
        waveguide: None,
        structural_acoustic: None,
    }
}

fn run() -> Result<
    (
        usize,
        usize,
        usize,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        f64,
        u64,
    ),
    String,
> {
    let started = Instant::now();
    let design = fixture();
    let interior = build_body_fitted_sealed_product_interior_mesh(
        &design,
        ProductInteriorMeshConfig::default(),
    )
    .map_err(|error| format!("body-fitted ported mesh failed: {error}"))?;
    let node_count = interior.diagnostics.node_count;
    let tetrahedron_count = interior.diagnostics.tetrahedron_count;
    let coupling = MonolithicSealedProductCoupling::new(&design, interior, 343.0, 1.21)
        .map_err(|error| format!("body-fitted ported coupling failed: {error}"))?;
    let solution = coupling
        .solve_frequency(&design, 90.0, Complex64::new(2.0, 0.0))
        .map_err(|error| format!("body-fitted ported solve failed: {error}"))?;
    for frequency_hz in [60.0, 120.0] {
        let grid_solution = coupling
            .solve_frequency(&design, frequency_hz, Complex64::new(2.0, 0.0))
            .map_err(|error| {
                format!(
                    "body-fitted ported frequency-grid solve failed at {frequency_hz} Hz: {error}"
                )
            })?;
        let grid_residuals = [
            grid_solution.relative_residual,
            grid_solution.interior_relative_residual,
            grid_solution.exterior_relative_residual,
            grid_solution.mechanical_relative_residual,
            grid_solution.electrical_relative_residual,
        ];
        if grid_residuals
            .iter()
            .any(|value| !value.is_finite() || *value > 1.0e-7)
            || grid_solution
                .driver_mechanical_relative_residual
                .iter()
                .chain(&grid_solution.driver_electrical_relative_residual)
                .any(|value| !value.is_finite() || *value > 1.0e-7)
            || grid_solution
                .port_pressure_jump_pa
                .iter()
                .any(|value| !value.is_finite() || value.norm() > 1.0e-7)
            || !grid_solution.port_volume_velocity_m3_per_s.is_finite()
        {
            return Err(format!(
                "body-fitted ported frequency-grid residual failed at {frequency_hz} Hz"
            ));
        }
    }

    let max_port_pressure_jump = solution
        .port_pressure_jump_pa
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    let port_face_count = solution.port_velocity_m_per_s.len();
    let port_volume_velocity = solution.port_volume_velocity_m3_per_s.norm();
    let driver_count = solution.driver_velocity_m_per_s.len();
    let per_driver_residuals = solution
        .driver_mechanical_relative_residual
        .iter()
        .chain(&solution.driver_electrical_relative_residual)
        .copied()
        .collect::<Vec<_>>();
    let maximum_driver_residual = per_driver_residuals.iter().copied().fold(0.0, f64::max);
    let maximum_power_residual = solution
        .driver_electrical_power_residual_w
        .iter()
        .chain(&solution.driver_mechanical_power_residual_w)
        .copied()
        .chain([
            solution.electrical_power_residual_w,
            solution.mechanical_power_residual_w,
        ])
        .map(f64::abs)
        .fold(0.0, f64::max);
    let residuals = [
        solution.relative_residual,
        solution.interior_relative_residual,
        solution.exterior_relative_residual,
        solution.mechanical_relative_residual,
        solution.electrical_relative_residual,
        max_port_pressure_jump,
    ];
    if residuals.iter().any(|value| !value.is_finite())
        || port_face_count == 0
        || driver_count != 3
        || solution
            .driver_current_a
            .iter()
            .any(|value| value.norm() == 0.0)
        || solution
            .driver_velocity_m_per_s
            .iter()
            .any(|value| value.norm() == 0.0)
        || solution
            .driver_terminal_voltage_v
            .windows(2)
            .all(|pair| (pair[0] - pair[1]).norm() <= 1.0e-7)
        || per_driver_residuals
            .iter()
            .any(|value| !value.is_finite() || *value > 1.0e-7)
        || !maximum_power_residual.is_finite()
        || maximum_power_residual > 1.0e-7
        || port_volume_velocity <= 0.0
        || solution
            .driver_velocity_m_per_s
            .iter()
            .all(|value| value.norm() == 0.0)
        || solution
            .driver_current_a
            .iter()
            .all(|value| value.norm() == 0.0)
        || solution.relative_residual > 1.0e-7
        || solution.interior_relative_residual > 1.0e-7
        || solution.exterior_relative_residual > 1.0e-7
        || solution.mechanical_relative_residual > 1.0e-7
        || solution.electrical_relative_residual > 1.0e-7
        || max_port_pressure_jump > 1.0e-7
    {
        return Err(
            "body-fitted ported solve did not close its residual, port, or drive blocks".into(),
        );
    }

    Ok((
        node_count,
        tetrahedron_count,
        port_face_count,
        solution.relative_residual,
        solution.interior_relative_residual,
        solution.exterior_relative_residual,
        solution.mechanical_relative_residual,
        solution.electrical_relative_residual,
        maximum_driver_residual,
        maximum_power_residual,
        max_port_pressure_jump,
        started.elapsed().as_millis() as u64,
    ))
}

fn main() {
    match run() {
        Ok((
            node_count,
            tetrahedron_count,
            port_face_count,
            residual,
            interior_residual,
            exterior_residual,
            mechanical_residual,
            electrical_residual,
            maximum_driver_residual,
            maximum_power_residual,
            max_port_pressure_jump,
            elapsed_ms,
        )) => {
            let result = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("node_count", node_count as f64)
                .metric("tetrahedron_count", tetrahedron_count as f64)
                .metric("port_face_count", port_face_count as f64)
                .metric("frequency_count", 3.0)
                .metric("relative_residual", residual)
                .metric("interior_relative_residual", interior_residual)
                .metric("exterior_relative_residual", exterior_residual)
                .metric("mechanical_relative_residual", mechanical_residual)
                .metric("electrical_relative_residual", electrical_residual)
                .metric("maximum_driver_residual", maximum_driver_residual)
                .metric("maximum_power_residual", maximum_power_residual)
                .metric("max_port_pressure_jump", max_port_pressure_jump)
                .minimum("node_count", 1.0)
                .minimum("tetrahedron_count", 1.0)
                .minimum("port_face_count", 1.0)
                .maximum("relative_residual", 1.0e-7)
                .maximum("interior_relative_residual", 1.0e-7)
                .maximum("exterior_relative_residual", 1.0e-7)
                .maximum("mechanical_relative_residual", 1.0e-7)
                .maximum("electrical_relative_residual", 1.0e-7)
                .maximum("maximum_driver_residual", 1.0e-7)
                .maximum("maximum_power_residual", 1.0e-7)
                .maximum("max_port_pressure_jump", 1.0e-7)
                .input_checksum("analytic/deterministic-fixture:body-fitted-ported-monolithic-v1")
                .reference("production body-fitted FEM/BEM monolithic port continuity and driver blocks")
                .reference_checksum("analytic/deterministic-fixture:body-fitted-ported-monolithic-v1")
                .dimension("3D ported rectangular cavity; three front drivers; 60/90/120 Hz")
                .formulation("body-fitted ported product interior FEM and exterior BEM monolithic solve")
                .tolerance("all residual blocks and port pressure jump <= 1e-7 across 60/90/120 Hz; finite non-empty port flow")
                .seed("body-fitted-ported-monolithic-v1")
                .timing_ms("body_fitted_ported_monolithic", elapsed_ms)
                .elapsed_ms(elapsed_ms);
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("ported monolithic result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .requested_backend("cpu")
                    .actual_backend("cpu")
                    .to_json_line()
                    .expect("ported monolithic failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
