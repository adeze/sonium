//! Production PC-fan CFD field, conservation, and fan-curve QA.

use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    PcFanCfdConfig, PcFanConfig, PcFanFlowModel, PcFanMeshResolution, PcFanOperatingPoint,
    evaluate_pc_fan, evaluate_pc_fan_from_cfd_solution, solve_pc_fan_cfd_mean_flow,
};
use std::time::Instant;

const CASE_ID: &str = "pcfan.cfd-invariants.v1";
const MASS_BALANCE_RELATIVE_TOLERANCE: f64 = 5.0e-2;

struct Metrics {
    airflow_cfm: f64,
    inlet_airflow_cfm: f64,
    mass_balance_error_kg_s: f64,
    mass_balance_relative: f64,
    pressure_drop_pa: f64,
    max_velocity_m_s: f64,
    reynolds_number: f64,
    residual: f64,
    objective: f64,
    conservation_relative_momentum: f64,
    conservation_relative_pressure: f64,
    conservation_relative_energy: f64,
    conservation_pressure_power_w: f64,
    conservation_shaft_power_w: f64,
    fan_curve_airflow_cfm: [f64; 3],
    fan_curve_pressure_pa: [f64; 3],
}

fn run() -> Result<Metrics, String> {
    // This intentionally small mesh is the same deterministic production
    // smoke configuration used by the PC-fan crate's CFD integration tests.
    // Four Picard steps exercise the actual immersed-solid/actuator path;
    // this is a field and conservation gate, not a convergence claim.
    let config = PcFanConfig {
        mesh: PcFanMeshResolution {
            radial_segments: 2,
            chord_segments: 2,
            circumferential_segments: 8,
            harmonic_count: 2,
        },
        cfd: PcFanCfdConfig {
            flow_model: PcFanFlowModel::IncompressibleCfd,
            cells_x: 2,
            cells_y: 2,
            cells_z: 2,
            max_picard_iterations: 4,
            nonlinear_tolerance: 1.0e-3,
            linear_solver_max_iterations: 300,
            linear_solver_tolerance: 1.0e-4,
            ..PcFanCfdConfig::default()
        },
        ..PcFanConfig::default_120mm()
    };
    let design = config.default_design_variables();
    let cfd = solve_pc_fan_cfd_mean_flow(&config, &design);
    let evaluation = evaluate_pc_fan_from_cfd_solution(&config, &design, &cfd);

    if cfd.nodes.len() != cfd.velocity_m_s.len()
        || cfd.nodes.len() != cfd.pressure_pa.len()
        || cfd.fem_node_count != cfd.nodes.len()
        || cfd.fem_element_count == 0
        || cfd.boundary_dof_count == 0
    {
        return Err("production CFD returned inconsistent mesh/field dimensions".into());
    }

    let finite = cfd
        .nodes
        .iter()
        .all(|point| point.x.is_finite() && point.y.is_finite() && point.z.is_finite())
        && cfd
            .velocity_m_s
            .iter()
            .flatten()
            .all(|value| value.is_finite())
        && cfd.pressure_pa.iter().all(|value| value.is_finite())
        && [
            cfd.airflow_cfm,
            cfd.inlet_airflow_cfm,
            cfd.mass_balance_error_kg_s,
            cfd.pressure_drop_pa,
            cfd.max_velocity_m_s,
            cfd.reynolds_number,
            cfd.residual,
            evaluation.objective,
            cfd.conservation.relative_mass_imbalance,
            cfd.conservation.relative_momentum_imbalance,
            cfd.conservation.relative_pressure_closure,
            cfd.conservation.relative_energy_imbalance,
            cfd.conservation.pressure_power_w,
            cfd.conservation.shaft_power_to_fluid_w,
        ]
        .iter()
        .all(|value| value.is_finite());
    if !finite {
        return Err(
            "production PC-fan CFD/evaluation returned a non-finite field or metric".into(),
        );
    }
    if cfd.airflow_cfm <= 0.0 || cfd.inlet_airflow_cfm <= 0.0 || cfd.max_velocity_m_s <= 0.0 {
        return Err("production PC-fan CFD returned non-positive through-flow".into());
    }

    // Convert the mean inlet/outlet flow to a characteristic mass flux.  A
    // relative balance budget is stable across harmless changes to the tiny
    // smoke mesh and catches boundary-sign or missing-flux regressions.
    let characteristic_mass_flux =
        config.operating_point.air_density_kg_m3 * 0.5 * (cfd.airflow_cfm + cfd.inlet_airflow_cfm)
            / 2_118.880_003_289_3;
    let mass_balance_relative =
        cfd.mass_balance_error_kg_s.abs() / characteristic_mass_flux.max(f64::EPSILON);
    if mass_balance_relative > MASS_BALANCE_RELATIVE_TOLERANCE {
        return Err(format!(
            "PC-fan CFD mass balance exceeded {:.3e}: {:.3e}",
            MASS_BALANCE_RELATIVE_TOLERANCE, mass_balance_relative
        ));
    }

    // The reduced-order production model provides the deterministic fan-curve
    // reference while the coarse CFD case above exercises the real field path.
    let mut curve_config = config.clone();
    curve_config.cfd.flow_model = PcFanFlowModel::ReducedOrder;
    let mut curve_airflow = [0.0; 3];
    let mut curve_pressure = [0.0; 3];
    for (index, rpm) in [600.0, 1_200.0, 1_800.0].into_iter().enumerate() {
        curve_config.operating_point = PcFanOperatingPoint::bench(rpm);
        let curve = evaluate_pc_fan(&curve_config, &design);
        curve_airflow[index] = curve.airflow_cfm;
        curve_pressure[index] = curve.flow.pressure_rise_pa;
        if !curve_airflow[index].is_finite()
            || !curve_pressure[index].is_finite()
            || curve_airflow[index] <= 0.0
            || curve_pressure[index] < 0.0
        {
            return Err("reduced-order fan curve returned an invalid point".into());
        }
    }
    if !(curve_airflow[0] < curve_airflow[1] && curve_airflow[1] < curve_airflow[2])
        || !(curve_pressure[0] < curve_pressure[1] && curve_pressure[1] < curve_pressure[2])
    {
        return Err(format!(
            "fan curve is not strictly increasing: airflow={curve_airflow:?}, pressure={curve_pressure:?}"
        ));
    }

    Ok(Metrics {
        airflow_cfm: cfd.airflow_cfm,
        inlet_airflow_cfm: cfd.inlet_airflow_cfm,
        mass_balance_error_kg_s: cfd.mass_balance_error_kg_s,
        mass_balance_relative,
        pressure_drop_pa: cfd.pressure_drop_pa,
        max_velocity_m_s: cfd.max_velocity_m_s,
        reynolds_number: cfd.reynolds_number,
        residual: cfd.residual,
        objective: evaluation.objective,
        conservation_relative_momentum: cfd.conservation.relative_momentum_imbalance,
        conservation_relative_pressure: cfd.conservation.relative_pressure_closure,
        conservation_relative_energy: cfd.conservation.relative_energy_imbalance,
        conservation_pressure_power_w: cfd.conservation.pressure_power_w,
        conservation_shaft_power_w: cfd.conservation.shaft_power_to_fluid_w,
        fan_curve_airflow_cfm: curve_airflow,
        fan_curve_pressure_pa: curve_pressure,
    })
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("airflow_cfm", metrics.airflow_cfm)
                .metric("inlet_airflow_cfm", metrics.inlet_airflow_cfm)
                .metric("mass_balance_error_kg_s", metrics.mass_balance_error_kg_s)
                .metric("mass_balance_relative", metrics.mass_balance_relative)
                .metric("pressure_drop_pa", metrics.pressure_drop_pa)
                .metric("max_velocity_m_s", metrics.max_velocity_m_s)
                .metric("reynolds_number", metrics.reynolds_number)
                .metric("picard_residual", metrics.residual)
                .metric("objective", metrics.objective)
                .metric(
                    "conservation_relative_momentum_imbalance",
                    metrics.conservation_relative_momentum,
                )
                .metric(
                    "conservation_relative_pressure_closure",
                    metrics.conservation_relative_pressure,
                )
                .metric(
                    "conservation_relative_energy_imbalance",
                    metrics.conservation_relative_energy,
                )
                .metric(
                    "conservation_pressure_power_w",
                    metrics.conservation_pressure_power_w,
                )
                .metric(
                    "conservation_shaft_power_to_fluid_w",
                    metrics.conservation_shaft_power_w,
                )
                .metric(
                    "fan_curve_airflow_600_cfm",
                    metrics.fan_curve_airflow_cfm[0],
                )
                .metric(
                    "fan_curve_airflow_1200_cfm",
                    metrics.fan_curve_airflow_cfm[1],
                )
                .metric(
                    "fan_curve_airflow_1800_cfm",
                    metrics.fan_curve_airflow_cfm[2],
                )
                .metric(
                    "fan_curve_pressure_600_pa",
                    metrics.fan_curve_pressure_pa[0],
                )
                .metric(
                    "fan_curve_pressure_1200_pa",
                    metrics.fan_curve_pressure_pa[1],
                )
                .metric(
                    "fan_curve_pressure_1800_pa",
                    metrics.fan_curve_pressure_pa[2],
                )
                .maximum("mass_balance_relative", MASS_BALANCE_RELATIVE_TOLERANCE)
                .minimum("airflow_cfm", 0.0)
                .minimum("inlet_airflow_cfm", 0.0)
                .minimum("max_velocity_m_s", 0.0)
                .minimum("reynolds_number", 0.0)
                .seed("pcfan-120mm:cfd=2x2x2:picard=4:cpu:curve=600,1200,1800rpm")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            println!("SUCCESS");
        }
        Err(error) => {
            if let Ok(json) = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX))
                .to_json_line()
            {
                println!("QA_RESULT: {json}");
            }
            eprintln!("PC-FAN CFD INVARIANTS FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
