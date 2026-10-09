//! Deterministic transported k--omega transient objective-adjoint QA contract.
//!
//! This target deliberately uses a synthetic one-step retained trajectory. It
//! exercises the public full `[flow, k, omega]` residual/design admission and
//! adjoint solve without launching a production CFD workload.

use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::box_mesh_tetrahedra_symmetric;
use sonium_fem::physics::fluid::turbulence::KOmegaTransportLinearization;
use sonium_fem::spaces::{FunctionSpace, H1Space, VectorH1Space};
use sonium_qa::QaResult;
use sonium_sim_pcfan::{
    DESIGN_VARIABLE_COUNT, PC_FAN_KOMEGA_COUPLED_LINEARIZATION_PROVENANCE, PcFanCfdBackend,
    PcFanCfdBackendProvenance, PcFanCfdConservation, PcFanCfdExecutionEnvironment,
    PcFanCfdKOmegaBdfState, PcFanCfdMeshPattern, PcFanCfdTransientBdfState,
    PcFanCfdTransientObservation, PcFanCfdTransientTrajectory, PcFanCfdTransientTrajectoryStep,
    PcFanConfig, PcFanFlowModel, PcFanKOmegaCoupledLinearization, PcFanResolvedAdjointConfig,
    PcFanResolvedFlowSparseMatrix, PcFanResolvedTransientObjective, PcFanTurbulenceModel,
    solve_pc_fan_cfd_resolved_transient_objective_adjoint,
};
use sonium_solvers::CsrMatrix;
use std::time::Instant;

const CASE_ID: &str = "pcfan.turbulent-transient-adjoint-reference.v1";
const FLOW_PROVENANCE: &str = "fem:navier-stokes:bdf-newton-transported-monolithic:v1";
const SCALAR_PROVENANCE: &str =
    sonium_fem::physics::fluid::turbulence::K_OMEGA_TRANSPORT_LINEARIZATION_PROVENANCE;

fn csr_identity(dimension: usize) -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(
        dimension,
        dimension,
        (0..dimension).map(|index| (index, index, 1.0)).collect(),
    )
}

fn csr_zero(rows: usize, columns: usize) -> CsrMatrix<f64> {
    CsrMatrix::from_triplets(rows, columns, Vec::new())
}

fn resolved_identity(dimension: usize) -> PcFanResolvedFlowSparseMatrix {
    PcFanResolvedFlowSparseMatrix::new(
        dimension,
        (0..=dimension).collect(),
        (0..dimension).collect(),
        vec![1.0; dimension],
    )
    .expect("identity state matrix")
}

fn zero_conservation() -> PcFanCfdConservation {
    PcFanCfdConservation {
        inlet_mass_flow_kg_s: 0.0,
        outlet_mass_flow_kg_s: 0.0,
        relative_mass_imbalance: 0.0,
        inlet_axial_momentum_flux_n: 0.0,
        outlet_axial_momentum_flux_n: 0.0,
        pressure_force_n: 0.0,
        body_force_n: 0.0,
        axial_momentum_residual_n: 0.0,
        relative_momentum_imbalance: 0.0,
        pressure_force_from_loading_n: 0.0,
        relative_pressure_closure: 0.0,
        pressure_power_w: 0.0,
        shaft_power_to_fluid_w: 0.0,
        relative_energy_imbalance: 0.0,
    }
}

fn fixture() -> (
    PcFanConfig,
    sonium_sim_pcfan::PcFanDesignVariables,
    PcFanCfdTransientTrajectory,
) {
    let mut config = PcFanConfig::default_120mm();
    config.cfd.flow_model = PcFanFlowModel::IncompressibleCfdFull;
    config.cfd.backend = PcFanCfdBackend::Cpu;
    config.cfd.turbulence = PcFanTurbulenceModel::KOmegaSstTransported;
    config.cfd.mesh_pattern = PcFanCfdMeshPattern::Symmetric;
    config.cfd.cells_x = 1;
    config.cfd.cells_y = 1;
    config.cfd.cells_z = 1;

    let design = config.default_design_variables();
    let geometry = config.build_geometry(&design);
    let domain = geometry.build_cfd_domain(&config);
    let mesh = box_mesh_tetrahedra_symmetric(
        domain.min_corner.x,
        domain.max_corner.x,
        domain.min_corner.y,
        domain.max_corner.y,
        domain.min_corner.z,
        domain.max_corner.z,
        config.cfd.cells_x.max(1),
        config.cfd.cells_y.max(1),
        config.cfd.cells_z.max(1),
    );
    let velocity_space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let pressure_space = H1Space::new(PolynomialDegree::P1);
    let velocity_dimension = velocity_space.num_global_dofs(&mesh);
    let pressure_dimension = pressure_space.num_global_dofs(&mesh);
    let flow_dimension = velocity_dimension + pressure_dimension;
    let scalar_dimension = mesh.num_nodes();
    let scalar_state = PcFanCfdKOmegaBdfState {
        k: vec![1.0; scalar_dimension],
        k_prev: vec![1.0; scalar_dimension],
        omega: vec![2.0; scalar_dimension],
        omega_prev: vec![2.0; scalar_dimension],
        time: 1.0,
        step: 1,
    };
    let scalar_linearization = KOmegaTransportLinearization {
        provenance: SCALAR_PROVENANCE,
        k_state_jacobian: csr_identity(scalar_dimension),
        omega_state_jacobian: csr_identity(scalar_dimension),
        k_history_jacobian: csr_zero(scalar_dimension, scalar_dimension),
        k_previous_history_jacobian: csr_zero(scalar_dimension, scalar_dimension),
        omega_history_jacobian: csr_zero(scalar_dimension, scalar_dimension),
        omega_previous_history_jacobian: csr_zero(scalar_dimension, scalar_dimension),
    };
    let coupled_linearization = PcFanKOmegaCoupledLinearization {
        provenance: PC_FAN_KOMEGA_COUPLED_LINEARIZATION_PROVENANCE,
        momentum_wrt_k: csr_zero(velocity_dimension, scalar_dimension),
        momentum_wrt_omega: csr_zero(velocity_dimension, scalar_dimension),
        k_wrt_velocity: csr_zero(scalar_dimension, velocity_dimension),
        omega_wrt_velocity: csr_zero(scalar_dimension, velocity_dimension),
        k_wrt_k: csr_identity(scalar_dimension),
        k_wrt_omega: csr_zero(scalar_dimension, scalar_dimension),
        omega_wrt_k: csr_zero(scalar_dimension, scalar_dimension),
        omega_wrt_omega: csr_identity(scalar_dimension),
    };
    let observation = PcFanCfdTransientObservation {
        final_time_s: 1.0,
        airflow_cfm: 1.0,
        pressure_drop_pa: 1.0,
        axial_loading_n: 1.0,
        residual: 0.0,
        conservation: zero_conservation(),
        converged: true,
    };
    let step = PcFanCfdTransientTrajectoryStep {
        step: 1,
        time_s: 1.0,
        state: vec![1.0; flow_dimension],
        state_jacobian: resolved_identity(flow_dimension),
        state_jacobian_provenance: FLOW_PROVENANCE.into(),
        actual_backend: PcFanCfdBackend::Cpu,
        backend_provenance: PcFanCfdBackendProvenance::from_dispatch(
            PcFanCfdBackend::Cpu,
            PcFanCfdBackend::Cpu,
        ),
        execution_environment: PcFanCfdExecutionEnvironment::capture(
            &config.cfd,
            PcFanCfdBackendProvenance::from_dispatch(PcFanCfdBackend::Cpu, PcFanCfdBackend::Cpu),
        ),
        rhs: vec![0.0; flow_dimension],
        state_residual: vec![0.0; flow_dimension],
        residual_norm: 0.0,
        conservation: zero_conservation(),
        history_jacobians: Vec::new(),
        transported_turbulence: Some(scalar_state),
        transported_eddy_viscosity_per_element: Some(vec![0.1; mesh.num_elements()]),
        transported_turbulence_linearization: Some(scalar_linearization),
        transported_turbulence_coupled_linearization: Some(coupled_linearization),
        transported_turbulence_residual: Some((
            vec![0.0; scalar_dimension],
            vec![0.0; scalar_dimension],
        )),
        transported_turbulence_residual_design_jacobian: Some(vec![
            vec![0.0; DESIGN_VARIABLE_COUNT];
            2 * scalar_dimension
        ]),
        bdf_state: PcFanCfdTransientBdfState {
            velocity: vec![0.0; velocity_dimension],
            velocity_prev: vec![0.0; velocity_dimension],
            pressure: vec![0.0; pressure_dimension],
            time: 1.0,
            step: 1,
        },
        topology_transfer_adjoint_admissible: true,
        residual_design_jacobian: vec![vec![0.0; DESIGN_VARIABLE_COUNT]; flow_dimension],
    };
    (
        config,
        design,
        PcFanCfdTransientTrajectory {
            observation,
            steps: vec![step],
            stage_initial_history_jacobians: Vec::new(),
        },
    )
}

fn run() -> Result<(usize, f64, f64, f64), String> {
    let (config, design, trajectory) = fixture();
    let scalar_dimension = trajectory.steps[0]
        .transported_turbulence
        .as_ref()
        .map(|state| state.k.len())
        .ok_or("transported turbulence state is missing")?;
    let report = solve_pc_fan_cfd_resolved_transient_objective_adjoint(
        &PcFanResolvedAdjointConfig {
            tolerance: 1.0e-10,
            max_state_residual: 1.0e-10,
            ..PcFanResolvedAdjointConfig::default()
        },
        &config,
        &design,
        &trajectory,
        PcFanResolvedTransientObjective::Airflow,
    )
    .map_err(|error| error.to_string())?;
    if !report.converged
        || !report.adjoint_residual_norm.is_finite()
        || report.adjoints.len() != trajectory.steps.len()
        || report.adjoints[0].len() != trajectory.steps[0].state.len() + 2 * scalar_dimension
        || report
            .adjoints
            .iter()
            .flatten()
            .chain(report.design_gradient.iter())
            .any(|value| !value.is_finite())
        || report.design_gradient.len() != DESIGN_VARIABLE_COUNT
    {
        return Err("transported objective adjoint contract did not converge finitely".into());
    }
    let design_norm = report
        .design_gradient
        .iter()
        .map(|value| value * value)
        .sum::<f64>()
        .sqrt();
    Ok((
        report.adjoints.len(),
        report.adjoint_residual_norm,
        design_norm,
        report.adjoints[0].len() as f64,
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((steps, residual, design_norm, full_state_dimension)) => {
            let mut record = QaResult::passed(CASE_ID);
            record.message = Some(format!(
                "synthetic transported k-omega objective-adjoint contract passed {steps} step(s)"
            ));
            record = record
                .metric("steps", steps as f64)
                .metric("adjoint_residual_norm", residual)
                .metric("design_gradient_l2", design_norm)
                .metric("full_state_dimension", full_state_dimension)
                .reference_checksum(
                    "analytic/deterministic-fixture:pcfan-transported-transient-adjoint-v1",
                )
                .input_checksum("synthetic:pcfan-full-flow-k-omega-design-block:v1")
                .seed("pcfan-turbulent-transient-adjoint-reference:synthetic:v1")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .to_json_line()
                .expect("QA failure must serialize");
            println!("QA_RESULT: {record}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::run;

    #[test]
    fn synthetic_transported_objective_adjoint_contract() {
        let (steps, residual, design_norm, full_state_dimension) =
            run().expect("synthetic transported objective-adjoint contract");
        assert_eq!(steps, 1);
        assert!(residual.is_finite());
        assert!(design_norm.is_finite());
        assert!(full_state_dimension > 0.0);
    }
}
