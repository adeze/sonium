//! Compile-only QA contract for the public structural family reference APIs.

use std::time::Instant;

use sonium_qa::QaResult;
use sonium_sim_structural_mechanics::{
    AxisymmetricCylinderStudy, BracketShapeOptimizationStudy, BracketStudy, CompositeLaminateStudy,
    PhaseFieldDamageStudy, PhaseFieldForceControlledStudy, Plate2DStudy, ShellStaticStudy,
    Solid3DStudy, StructuralModelAssumption, ThermoelasticStudy, Viscoelastic2DStudy,
    run_axisymmetric_cylinder, run_bracket_shape_optimization, run_composite_laminate_static,
    run_phase_field_damage, run_phase_field_damage_force_controlled, run_plate2d_static,
    run_shell_static, run_solid3d_static, run_thermoelastic_static, run_viscoelastic_2d,
};

const CASE_ID: &str = "structural.public-family-references.v1";

fn run() -> Result<usize, String> {
    let mut axisymmetric = AxisymmetricCylinderStudy::introductory();
    axisymmetric.geometry.radial_elements = 8;
    let result = run_axisymmetric_cylinder(&axisymmetric).map_err(|error| error.to_string())?;
    if result.model_assumption != StructuralModelAssumption::AxisymmetricElasticity
        || !result.converged
        || result.free_equilibrium_residual_n > 1.0e-6
        || result.energy_balance_relative_error > 1.0e-10
    {
        return Err("axisymmetric reference gate failed".to_string());
    }

    let mut composite = CompositeLaminateStudy::introductory();
    composite.geometry.nx = 4;
    composite.geometry.ny = 2;
    let result = run_composite_laminate_static(&composite).map_err(|error| error.to_string())?;
    if result.model_assumption != StructuralModelAssumption::CompositeLaminate
        || !result.converged
        || result.free_equilibrium_residual_n > 1.0e-6
        || result.energy_balance_relative_error > 1.0e-10
    {
        return Err("composite reference gate failed".to_string());
    }

    let mut thermoelastic = ThermoelasticStudy::introductory();
    thermoelastic.geometry.nx = 4;
    thermoelastic.geometry.ny = 2;
    let result = run_thermoelastic_static(&thermoelastic).map_err(|error| error.to_string())?;
    if result.model_assumption != StructuralModelAssumption::PlaneStress
        || !result.converged
        || result.max_affine_displacement_error_m > 1.0e-10
        || result.energy_balance_relative_error > 1.0e-10
    {
        return Err("thermoelastic reference gate failed".to_string());
    }

    let mut viscoelastic = Viscoelastic2DStudy::introductory();
    viscoelastic.geometry.nx = 4;
    viscoelastic.geometry.ny = 2;
    viscoelastic.material.loss_factor = 0.35;
    let result = run_viscoelastic_2d(&viscoelastic).map_err(|error| error.to_string())?;
    if result.model_assumption != StructuralModelAssumption::ViscoelasticPlaneStress
        || !result.converged
        || result.compliance_scaling_relative_error > 1.0e-8
        || result.phase_error_rad.abs() > 1.0e-8
    {
        return Err("viscoelastic reference gate failed".to_string());
    }

    let mut plate = Plate2DStudy::introductory();
    plate.geometry.nx = 20;
    plate.geometry.ny = 20;
    let result = run_plate2d_static(&plate).map_err(|error| error.to_string())?;
    if result.model_assumption != StructuralModelAssumption::MindlinPlate
        || !result.converged
        || result.center_deflection_m <= 0.0
        || result.energy_balance_relative_error > 1.0e-10
    {
        return Err("plate reference gate failed".to_string());
    }

    let mut phase_field = PhaseFieldDamageStudy::introductory();
    phase_field.nx = 4;
    phase_field.ny = 2;
    phase_field.load_steps = 2;
    let prescribed = run_phase_field_damage(&phase_field).map_err(|error| error.to_string())?;
    let force_controlled = PhaseFieldForceControlledStudy {
        total_force_n: phase_field.youngs_modulus_pa
            * phase_field.height_m
            * phase_field.prescribed_extension_m
            / phase_field.width_m,
        base: phase_field,
    };
    let force = run_phase_field_damage_force_controlled(&force_controlled)
        .map_err(|error| error.to_string())?;
    if !prescribed.converged
        || !prescribed.history_monotone
        || !prescribed.damage_monotone
        || !force.converged
        || !force.history_monotone
        || !force.damage_monotone
    {
        return Err("phase-field reference gate failed".to_string());
    }

    let mut shell = ShellStaticStudy::introductory();
    shell.geometry.nx = 4;
    shell.geometry.ny = 2;
    let result = run_shell_static(&shell).map_err(|error| error.to_string())?;
    if result.model_assumption != StructuralModelAssumption::Shell
        || !result.converged
        || result.force_balance_residual_n > 1.0e-7
        || result.energy_balance_relative_error > 1.0e-9
    {
        return Err("shell reference gate failed".to_string());
    }

    let mut solid = Solid3DStudy::introductory();
    solid.geometry.nx = 4;
    solid.geometry.ny = 1;
    solid.geometry.nz = 1;
    solid.modal_modes = 1;
    let result = run_solid3d_static(&solid).map_err(|error| error.to_string())?;
    if result.model_assumption != StructuralModelAssumption::Solid3D
        || !result.converged
        || result.force_balance_residual_n > 1.0e-6
        || result.energy_balance_relative_error > 1.0e-10
    {
        return Err("solid reference gate failed".to_string());
    }

    let mut base = BracketStudy::introductory_bracket();
    base.geometry.nx = 4;
    base.geometry.ny = 1;
    let mut shape = BracketShapeOptimizationStudy::introductory(base);
    shape.iterations = 1;
    shape.step_size = 0.10;
    shape.derivative_step_fraction = 2.0e-4;
    let result = run_bracket_shape_optimization(&shape).map_err(|error| error.to_string())?;
    if result.model_assumption != StructuralModelAssumption::BracketShapeOptimization
        || !result.body_fitted_reanalysis
        || !result.final_static_result.converged
        || result.gradient_taylor_relative_error > 1.0e-6
    {
        return Err("bracket shape reference gate failed".to_string());
    }

    Ok(9)
}

fn main() {
    let started = Instant::now();
    match run() {
        Ok(families) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("families", families as f64)
                .minimum("families", 9.0)
                .reference("public bounded structural family APIs with model, finite, residual, energy, and reanalysis gates")
                .reference_checksum("analytic/deterministic-fixture:structural-public-family-references-v1")
                .dimension("axisymmetric, composite, thermoelastic, viscoelastic, plate, phase-field, shell, solid, shape")
                .formulation("focused CPU reference workflows; no external solver or product-scale claim")
                .tolerance("family-specific residual, energy, analytical, phase, monotonicity, and reanalysis gates")
                .seed("structural-public-family-references:bounded:cpu")
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
                .seed("structural-public-family-references:bounded:cpu")
                .elapsed_ms(started.elapsed().as_millis().try_into().unwrap_or(u64::MAX));
            if let Ok(json) = record.to_json_line() {
                println!("QA_RESULT: {json}");
            }
            eprintln!("STRUCTURAL PUBLIC FAMILY QA FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
