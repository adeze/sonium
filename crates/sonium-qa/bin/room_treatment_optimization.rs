//! Release-backed multi-frequency RoomSim treatment-optimization contract.

use num_complex::Complex64;
use sonium_fem::assembly::{HelmholtzAssembler, MassMatrix, StiffnessMatrix};
use sonium_fem::solver::{SolverConfig, SolverType};
use sonium_qa::QaResult;
use sonium_sim_room::treatment_optimization::{
    ListenerFunctional, MultiFrequencyRoomTreatmentConfig, MultiFrequencyRoomTreatmentProblem,
    TreatmentBoundary, TreatmentFrequencySetup, TreatmentObjectiveConfig, TreatmentObjectiveKind,
};
use std::collections::HashMap;

const CASE_ID: &str = "roomsim.treatment-optimization.v1";

fn diagonal_mass(size: usize, entries: &[(usize, f64)]) -> MassMatrix {
    let mut matrix = MassMatrix::new(size);
    for &(row, value) in entries {
        matrix.add(row, row, value);
    }
    matrix
}

fn fixture() -> Result<MultiFrequencyRoomTreatmentProblem, String> {
    let mut stiffness = StiffnessMatrix::new(3);
    for (row, column, value) in [
        (0, 0, 4.0),
        (0, 1, -1.0),
        (1, 0, -1.0),
        (1, 1, 3.0),
        (1, 2, -0.5),
        (2, 1, -0.5),
        (2, 2, 2.0),
    ] {
        stiffness.add(row, column, value);
    }
    let mass = diagonal_mass(3, &[(0, 0.2), (1, 0.2), (2, 0.2)]);
    let boundaries = vec![
        (1, diagonal_mass(3, &[(0, 1.0), (1, 0.2)])),
        (2, diagonal_mass(3, &[(1, 0.1), (2, 1.0)])),
    ];
    let assembler = HelmholtzAssembler::from_matrices(&stiffness, &mass, &boundaries);
    let treatments = vec![
        TreatmentBoundary {
            marker: 1,
            coefficient_direction: Complex64::new(0.0, -1.5),
            area_m2: Some(2.0),
            cost_per_m2: Some(10.0),
        },
        TreatmentBoundary {
            marker: 2,
            coefficient_direction: Complex64::new(0.0, -0.8),
            area_m2: Some(1.0),
            cost_per_m2: Some(15.0),
        },
    ];
    let setups = [
        (63.0, 0.8, 1.0, 0.0),
        (125.0, 1.0, 2.0, 0.1),
        (250.0, 1.25, 1.5, -0.1),
    ]
    .into_iter()
    .map(
        |(frequency_hz, wavenumber, weight, rhs_imag)| TreatmentFrequencySetup {
            frequency_hz,
            wavenumber: Complex64::new(wavenumber, 0.0),
            weight,
            base_boundary_coefficients: HashMap::new(),
            rhs: vec![
                Complex64::new(1.0, 0.2),
                Complex64::new(0.1, rhs_imag),
                Complex64::new(0.0, 0.0),
            ],
        },
    )
    .collect::<Vec<_>>();

    MultiFrequencyRoomTreatmentProblem::from_helmholtz(
        &assembler,
        &setups,
        &treatments,
        vec![
            ListenerFunctional::at_node(0),
            ListenerFunctional::at_node(1),
            ListenerFunctional::at_node(2),
        ],
        SolverConfig {
            solver_type: SolverType::Direct,
            ..SolverConfig::default()
        },
    )
    .map_err(|error| error.to_string())
}

fn run() -> Result<QaResult, String> {
    let problem = fixture()?;
    problem
        .check_passivity()
        .map_err(|error| error.to_string())?;
    let objective = TreatmentObjectiveConfig {
        kind: TreatmentObjectiveKind::Production,
        target_weight: 1.0,
        variance_weight: 0.3,
        min_level_weight: 0.2,
        min_level_db: 40.0,
        target_levels_db: vec![-20.0, -10.0, 0.0],
        reference_pressure: 1.0,
        cost_weight: 0.1,
    };
    let initial_treatments = vec![0.2, 0.2];
    let serial = problem
        .evaluate(&initial_treatments, &objective, Some(1))
        .map_err(|error| error.to_string())?;
    let parallel = problem
        .evaluate(&initial_treatments, &objective, Some(2))
        .map_err(|error| error.to_string())?;
    let lane_objective_delta = (serial.objective - parallel.objective).abs();
    if lane_objective_delta > 1.0e-12 || serial.gradient != parallel.gradient {
        return Err("frequency-lane evaluation is not deterministic".into());
    }

    let result = problem
        .optimize(&MultiFrequencyRoomTreatmentConfig {
            max_iterations: 10,
            tolerance: 1.0e-3,
            treatment_fraction: 0.6,
            initial_treatment: 0.2,
            objective: objective.clone(),
            max_frequency_lanes: Some(2),
            decay_constraint: None,
            spatial_constraint: None,
            density_filter: None,
        })
        .map_err(|error| error.to_string())?;
    let final_evaluation = problem
        .evaluate(&result.treatments, &objective, Some(2))
        .map_err(|error| error.to_string())?;
    let weighted_fraction = (result.treatments[0] * 2.0 + result.treatments[1]) / 3.0;
    let objective_reduction = serial.objective - final_evaluation.objective;
    if !objective_reduction.is_finite() || objective_reduction <= 0.0 {
        return Err(format!(
            "treatment optimization did not improve objective: initial={}, final={}",
            serial.objective, final_evaluation.objective
        ));
    }
    if !result
        .treatments
        .iter()
        .all(|value| value.is_finite() && (0.0..=1.0).contains(value))
    {
        return Err("optimized treatments are outside [0, 1]".into());
    }

    Ok(QaResult::passed(CASE_ID)
        .requested_backend("cpu")
        .actual_backend("cpu")
        .metric("frequency_count", 3.0)
        .metric("listener_count", 3.0)
        .metric("treatment_count", 2.0)
        .metric("initial_objective", serial.objective)
        .metric("final_objective", final_evaluation.objective)
        .metric("objective_reduction", objective_reduction)
        .metric("iterations", result.iterations as f64)
        .metric("weighted_treatment_fraction", weighted_fraction)
        .metric("lane_objective_delta", lane_objective_delta)
        .minimum("frequency_count", 3.0)
        .minimum("listener_count", 2.0)
        .minimum("treatment_count", 2.0)
        .minimum("objective_reduction", 0.0)
        .maximum("weighted_treatment_fraction", 0.600000001)
        .maximum("lane_objective_delta", 1.0e-12)
        .input_checksum("analytic/deterministic-fixture:roomsim-treatment-optimization-v1")
        .reference_checksum("production/fixed-mesh-adjoint-mma:multi-frequency-v1")
        .dimension("3 frequencies, 3 listeners, 2 passive treatment boundaries")
        .formulation("fixed-mesh multi-frequency adjoint gradients with lane-deterministic MMA")
        .tolerance("objective decreases; treatment fraction <= 0.6; lane delta <= 1e-12")
        .seed("roomsim-treatment-optimization:cpu:v1"))
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result
                    .to_json_line()
                    .expect("RoomSim treatment result must serialize")
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
                    .expect("RoomSim treatment failure must serialize")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
