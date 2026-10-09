//! Compile-only QA contract for the bounded structural/acoustic coupling path.

use std::time::Instant;

use num_complex::Complex64;
use sonium_fem::physics::acoustics::electroacoustics::{
    CoupledIterativeConfig, CoupledPreconditioner,
};
use sonium_qa::QaResult;
use sonium_sim_speaker_modeling::{
    AxisymmetricLinearConfig, AxisymmetricLinearModel, ResolvedAcousticConfig,
    ResolvedAcousticModel, ResolvedCouplingConfig, ResolvedLoadCoupler,
};

const CASE_ID: &str = "structural.resolved-coupling-reference.v1";

fn coarse_linear_model() -> Result<AxisymmetricLinearModel, String> {
    let mut model = AxisymmetricLinearModel::build(AxisymmetricLinearConfig {
        magnetic_radial_cells: 8,
        magnetic_axial_cells: 8,
        structural_radial_cells: 6,
        structural_axial_cells: 1,
        acoustic_axial_cells: 4,
        iterative_solver: CoupledIterativeConfig {
            preconditioner: CoupledPreconditioner::Monolithic,
            ..CoupledIterativeConfig::default()
        },
        ..AxisymmetricLinearConfig::default()
    })
    .map_err(|error| error.to_string())?;
    model
        .install_interface_load()
        .map_err(|error| error.to_string())?;
    Ok(model)
}

fn run() -> Result<(usize, f64, f64), String> {
    let linear = coarse_linear_model()?;
    let acoustic = ResolvedAcousticModel::build(ResolvedAcousticConfig {
        cells_per_physical_interval: 1,
        pml_cells: 1,
        pml_sigma_max: 2.0,
        ..ResolvedAcousticConfig::default()
    })
    .map_err(|error| error.to_string())?;
    let mut coupler =
        ResolvedLoadCoupler::new(&linear, &acoustic, ResolvedCouplingConfig::default())
            .map_err(|error| error.to_string())?;
    let point = coupler
        .solve_frequency(100.0, Complex64::new(2.83, 0.0))
        .map_err(|error| error.to_string())?;
    let load_defect = (point.total_acoustic_load_n_s_per_m() - point.z_resolved_n_s_per_m).norm()
        / point.z_resolved_n_s_per_m.norm().max(f64::MIN_POSITIVE);
    let config = ResolvedCouplingConfig::default();
    if !point.converged
        || point.iterations > config.max_iterations
        || point.maximum_contraction_ratio >= 1.0
        || point.monolithic_agreement_relative_error > config.monolithic_agreement_tolerance
        || point.acoustic_load_absorbed_power_w() < 0.0
        || !load_defect.is_finite()
        || load_defect >= 1.0e-6
        || point
            .velocity_change_history
            .iter()
            .any(|value| !value.is_finite())
    {
        return Err("resolved structural coupling gate failed".to_string());
    }
    Ok((
        point.iterations,
        point.monolithic_agreement_relative_error,
        point.acoustic_load_absorbed_power_w(),
    ))
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok((iterations, monolithic_error, absorbed_power_w)) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("iterations", iterations as f64)
                .metric("monolithic_agreement_relative_error", monolithic_error)
                .metric("absorbed_power_w", absorbed_power_w)
                .minimum("iterations", 1.0)
                .reference("bounded resolved FEM/acoustic load fixed point with independent condensed reference")
                .reference_checksum("analytic/deterministic-fixture:structural-resolved-coupling-reference-v1")
                .dimension("axisymmetric structural FEM plus resolved acoustic coarse reference at 100 Hz")
                .formulation("CPU partitioned structural/acoustic coupling; no product-scale or external-solver claim")
                .tolerance("converged fixed point, contraction < 1, non-negative absorbed power, monolithic agreement, load defect < 1e-6")
                .seed("structural-resolved-coupling:bounded:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error.clone())
                .requested_backend("cpu")
                .actual_backend("cpu")
                .seed("structural-resolved-coupling:bounded:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL RESOLVED COUPLING QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
