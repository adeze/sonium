//! Bounded structural nonlinear, fatigue, transient, and ROM reference QA.
//!
//! This packet deliberately covers reduced and narrow production families. It
//! does not claim general 3-D plasticity, multiaxial fatigue, or full transient
//! structural-model validation.

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{
    ElastoplasticBarStudy, FemFatigue2DLoadBlock, FemFatigue2DStudy, StructuralModelAssumption,
    StructuralRomStudy, TransientSdoFStudy, run_elastoplastic_bar, run_fem_fatigue_2d,
    run_structural_rom, run_transient_sdof,
};

const CASE_ID: &str = "structural.bounded-nonlinear-fatigue-references.v1";
const RESIDUAL_TOLERANCE: f64 = 1.0e-7;
const ENERGY_TOLERANCE: f64 = 1.0e-10;

#[derive(Debug, Clone, Copy)]
struct Metrics {
    cases: usize,
    yielded_elements: usize,
    elastoplastic_equilibrium_residual_n: f64,
    fatigue_damage: f64,
    fatigue_solver_residual: f64,
    fatigue_equilibrium_residual_n: f64,
    transient_energy_balance_residual_j: f64,
    rom_relative_residual: f64,
    rom_tip_error: f64,
}

fn run() -> Result<Metrics, String> {
    let bar_study = ElastoplasticBarStudy::introductory();
    let bar = run_elastoplastic_bar(&bar_study).map_err(|error| error.to_string())?;
    if bar.model_assumption != StructuralModelAssumption::ElastoplasticBar
        || bar.steps.len() != bar_study.load_steps
        || bar.steps.last().map(|step| step.yielded_elements) != Some(bar_study.elements)
        || !bar.steps.iter().all(|step| step.converged)
        || !bar.total_plastic_dissipation_j.is_finite()
        || bar.total_plastic_dissipation_j <= 0.0
        || bar.max_equilibrium_residual_n > 1.0e-9
    {
        return Err("elastoplastic bar convergence/yield/energy gate failed".into());
    }

    let mut fatigue_study = FemFatigue2DStudy::introductory();
    fatigue_study.geometry.nx = 4;
    fatigue_study.geometry.ny = 1;
    fatigue_study.load_blocks = vec![FemFatigue2DLoadBlock {
        minimum_force_n: [0.0, -20.0],
        maximum_force_n: [0.0, -80.0],
        applied_cycles: 10_000.0,
    }];
    let fatigue = run_fem_fatigue_2d(&fatigue_study).map_err(|error| error.to_string())?;
    let fatigue_solver_residual = fatigue
        .blocks
        .iter()
        .map(|block| block.solver_residual)
        .fold(0.0, f64::max);
    let fatigue_equilibrium_residual_n = fatigue
        .blocks
        .iter()
        .map(|block| block.equilibrium_residual_n)
        .fold(0.0, f64::max);
    if fatigue.model_assumption != StructuralModelAssumption::FatiguePlaneStress2D
        || fatigue.blocks.len() != 1
        || !fatigue.converged
        || fatigue.failed
        || !fatigue.total_damage.is_finite()
        || fatigue.total_damage <= 0.0
        || fatigue.total_damage >= fatigue_study.damage_limit
        || fatigue_solver_residual > RESIDUAL_TOLERANCE
        || fatigue_equilibrium_residual_n > RESIDUAL_TOLERANCE
    {
        return Err("2-D FEM fatigue convergence/damage gate failed".into());
    }

    let transient_study = TransientSdoFStudy {
        mass_kg: 2.0,
        stiffness_n_per_m: 50.0,
        damping_ns_per_m: 0.0,
        time_step_s: 0.01,
        force_history_n: vec![0.0; 1_001],
        initial_displacement_m: 0.1,
        initial_velocity_m_per_s: 0.0,
    };
    let transient = run_transient_sdof(&transient_study).map_err(|error| error.to_string())?;
    if transient.model_assumption != StructuralModelAssumption::TransientSingleDof
        || transient.max_equilibrium_residual_n > 1.0e-9
        || transient.max_energy_balance_residual_j > ENERGY_TOLERANCE
        || transient.total_damping_dissipation_j.abs() > 1.0e-12
    {
        return Err("transient SDOF equilibrium/energy gate failed".into());
    }

    let mut rom_study = StructuralRomStudy::introductory();
    rom_study.base.geometry.elements = 4;
    rom_study.reduced_modes = 1;
    rom_study.base.modal_modes = 1;
    let rom = run_structural_rom(&rom_study).map_err(|error| error.to_string())?;
    if rom.model_assumption != StructuralModelAssumption::ReducedOrderBeam
        || !rom.recycled_basis
        || !rom.enriched_with_full_order_snapshot
        || !rom.converged
        || rom.reduced_relative_residual > rom_study.residual_tolerance
        || rom.full_order_relative_residual > 1.0e-8
        || rom.relative_tip_error > 1.0e-7
    {
        return Err("structural ROM residual/enrichment gate failed".into());
    }

    Ok(Metrics {
        cases: 4,
        yielded_elements: bar
            .steps
            .last()
            .map(|step| step.yielded_elements)
            .unwrap_or(0),
        elastoplastic_equilibrium_residual_n: bar.max_equilibrium_residual_n,
        fatigue_damage: fatigue.total_damage,
        fatigue_solver_residual,
        fatigue_equilibrium_residual_n,
        transient_energy_balance_residual_j: transient.max_energy_balance_residual_j,
        rom_relative_residual: rom.reduced_relative_residual,
        rom_tip_error: rom.relative_tip_error,
    })
}

fn main() {
    let started = std::time::Instant::now();
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("cases", metrics.cases as f64)
                .metric("yielded_elements", metrics.yielded_elements as f64)
                .metric(
                    "elastoplastic_equilibrium_residual_n",
                    metrics.elastoplastic_equilibrium_residual_n,
                )
                .metric("fatigue_damage", metrics.fatigue_damage)
                .metric("fatigue_solver_residual", metrics.fatigue_solver_residual)
                .metric(
                    "fatigue_equilibrium_residual_n",
                    metrics.fatigue_equilibrium_residual_n,
                )
                .metric(
                    "transient_energy_balance_residual_j",
                    metrics.transient_energy_balance_residual_j,
                )
                .metric("rom_relative_residual", metrics.rom_relative_residual)
                .metric("rom_tip_error", metrics.rom_tip_error)
                .minimum("cases", 4.0)
                .minimum("yielded_elements", 1.0)
                .maximum("elastoplastic_equilibrium_residual_n", 1.0e-9)
                .maximum("fatigue_solver_residual", RESIDUAL_TOLERANCE)
                .maximum("fatigue_equilibrium_residual_n", RESIDUAL_TOLERANCE)
                .maximum("transient_energy_balance_residual_j", ENERGY_TOLERANCE)
                .maximum("rom_relative_residual", 1.0e-8)
                .maximum("rom_tip_error", 1.0e-7)
                .reference("bounded production elastoplastic, P1 FEM fatigue, transient SDOF, and beam ROM references")
                .reference_checksum(
                    "analytic/deterministic-fixture:structural-bounded-nonlinear-fatigue-references-v1",
                )
                .dimension("1-D elastoplastic bar; 2-D P1 fatigue; 1-DOF transient; reduced beam ROM")
                .formulation("return mapping, S-N/Miner, Newmark average acceleration, residual-certified Galerkin enrichment")
                .tolerance("equilibrium/residual <= 1e-7; transient energy <= 1e-10; bounded-family gates")
                .seed("structural-bounded-nonlinear-fatigue-references:cpu")
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
                .seed("structural-bounded-nonlinear-fatigue-references:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("serialize QA result")
            );
            eprintln!("STRUCTURAL BOUNDED REFERENCES QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
