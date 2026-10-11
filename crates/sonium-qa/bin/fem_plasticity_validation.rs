//! Plasticity validation tests
//!
//! Tests for J2 (von Mises) elastoplasticity material models.
//!
//! - Material property calculations
//! - Perfect plasticity verification
//! - Hardening law behavior
//! - Elastic return mapping cases
//! - Yielding load/unload scenario: plastic return mapping, yield
//!   consistency, hardening, plastic work, and unload/reload behavior

use sonium_fem::physics::solid::plasticity::{
    HardeningLaw, J2Material, PlasticState, return_mapping_3d, update_plastic_state,
};
use sonium_qa::{QaResult, QaStatus};

const CASE_ID: &str = "fem.plasticity-validation.v1";

// Material constants for the load/unload scenario.
const YOUNGS_MODULUS: f64 = 200e9;
const POISSON_RATIO: f64 = 0.3;
const YIELD_STRESS: f64 = 250e6;
const ISOTROPIC_HARDENING_MODULUS: f64 = 1e9;

// Shear strain increment per loading step (engineering shear).
const SHEAR_STEP: f64 = 1e-3;
const LOAD_STEPS: usize = 10;

// Acceptance tolerances.
const TOL_CONSISTENCY: f64 = 1e-9;
const TOL_PLASTIC_WORK: f64 = 1e-9;
const TOL_UNLOAD_SLOPE: f64 = 1e-9;
const TOL_HARDENING_SLOPE: f64 = 1e-9;
const TOL_PROPERTY: f64 = 1e-6;

/// Von Mises equivalent stress for 6-component Voigt stress.
///
/// Ordering `[s_xx, s_yy, s_zz, s_yz, s_xz, s_xy]` with tensor shear
/// components, matching the return mapping state layout.
fn von_mises(stress: &[f64]) -> f64 {
    debug_assert_eq!(stress.len(), 6);
    let d0 = stress[0] - stress[1];
    let d1 = stress[1] - stress[2];
    let d2 = stress[2] - stress[0];
    let shear = stress[3] * stress[3] + stress[4] * stress[4] + stress[5] * stress[5];
    (0.5 * (d0 * d0 + d1 * d1 + d2 * d2) + 3.0 * shear).sqrt()
}

fn check(condition: bool, message: String) -> Result<(), String> {
    if condition { Ok(()) } else { Err(message) }
}

#[derive(Debug, Clone, Copy)]
struct Metrics {
    property_checks: usize,
    elastic_checks: usize,
    first_yield_step: usize,
    yield_consistency_error: f64,
    plastic_work_error: f64,
    hardening_consistency_error: f64,
    hardening_slope_error: f64,
    unload_slope_error: f64,
    reverse_consistency_error: f64,
    total_equivalent_plastic_strain: f64,
    plastic_work: f64,
}

fn run() -> Result<Metrics, String> {
    let mut property_checks = 0usize;
    let mut elastic_checks = 0usize;

    // Material creation and elastic properties.
    let material = J2Material::new(
        YOUNGS_MODULUS,
        POISSON_RATIO,
        YIELD_STRESS,
        HardeningLaw::Perfect,
    );
    check(
        (material.youngs_modulus - YOUNGS_MODULUS).abs() < 1e3,
        format!("Young modulus mismatch: {}", material.youngs_modulus),
    )?;
    property_checks += 1;
    check(
        (material.poisson_ratio - POISSON_RATIO).abs() < 1e-6,
        format!("Poisson ratio mismatch: {}", material.poisson_ratio),
    )?;
    property_checks += 1;
    check(
        (material.yield_stress - YIELD_STRESS).abs() < 1e3,
        format!("yield stress mismatch: {}", material.yield_stress),
    )?;
    property_checks += 1;

    let shear_modulus = material.shear_modulus();
    let expected_shear = YOUNGS_MODULUS / (2.0 * (1.0 + POISSON_RATIO));
    check(
        (shear_modulus - expected_shear).abs() / expected_shear < TOL_PROPERTY,
        format!("shear modulus mismatch: {shear_modulus} vs {expected_shear}"),
    )?;
    property_checks += 1;

    let bulk_modulus = material.bulk_modulus();
    let expected_bulk = YOUNGS_MODULUS / (3.0 * (1.0 - 2.0 * POISSON_RATIO));
    check(
        (bulk_modulus - expected_bulk).abs() / expected_bulk < TOL_PROPERTY,
        format!("bulk modulus mismatch: {bulk_modulus} vs {expected_bulk}"),
    )?;
    property_checks += 1;

    // Perfect plasticity keeps the yield stress constant.
    check(
        (material.current_yield_stress(0.0) - YIELD_STRESS).abs() < 1e3,
        "perfect plasticity yield at zero strain".to_string(),
    )?;
    property_checks += 1;
    check(
        (material.current_yield_stress(0.1) - YIELD_STRESS).abs() < 1e3,
        "perfect plasticity yield at finite strain".to_string(),
    )?;
    property_checks += 1;

    // Linear isotropic hardening grows the yield surface.
    let hardening_material = J2Material::new(
        YOUNGS_MODULUS,
        POISSON_RATIO,
        YIELD_STRESS,
        HardeningLaw::LinearIsotropic {
            modulus: ISOTROPIC_HARDENING_MODULUS,
        },
    );
    let expected_hardened = YIELD_STRESS + ISOTROPIC_HARDENING_MODULUS * 0.01;
    check(
        (hardening_material.current_yield_stress(0.01) - expected_hardened).abs()
            / expected_hardened
            < TOL_PROPERTY,
        "linear isotropic hardening value".to_string(),
    )?;
    property_checks += 1;

    // Elastic return mapping cases stay elastic.
    let state = PlasticState::new(6);
    let axial_inc = [0.001, 0.0, 0.0, 0.0, 0.0, 0.0];
    let axial = return_mapping_3d(&material, &state, &axial_inc);
    check(
        !axial.yielding && axial.delta_equiv_plastic_strain < 1e-10,
        "small axial strain should stay elastic".to_string(),
    )?;
    elastic_checks += 1;

    let shear_inc = [0.0, 0.0, 0.0, 0.0, 0.0, 0.001];
    let shear = return_mapping_3d(&material, &state, &shear_inc);
    check(
        !shear.yielding && shear.delta_equiv_plastic_strain < 1e-10,
        "small shear strain should stay elastic".to_string(),
    )?;
    elastic_checks += 1;

    // Yielding load/unload scenario under pure shear with perfect plasticity.
    // Trial equivalent stress per step is sqrt(3) * G * SHEAR_STEP, so the
    // first step is elastic and the second step yields.
    let mut load_state = PlasticState::new(6);
    let mut first_yield_step = 0usize;
    let mut yield_consistency_error = 0.0f64;
    let mut plastic_work = 0.0f64;
    for step in 1..=LOAD_STEPS {
        let update = return_mapping_3d(&material, &load_state, &shear_inc);
        if update.yielding {
            if first_yield_step == 0 {
                first_yield_step = step;
            }
            // Yield consistency: the updated stress must lie on the yield
            // surface for perfect plasticity.
            let equivalent = von_mises(&update.stress);
            let consistency = (equivalent - YIELD_STRESS).abs() / YIELD_STRESS;
            yield_consistency_error = yield_consistency_error.max(consistency);
            check(
                update.delta_equiv_plastic_strain > 0.0,
                format!("plastic step {step} has no plastic strain increment"),
            )?;
            // J2 plastic work rate equals equivalent stress times the
            // equivalent plastic strain rate.
            plastic_work += equivalent * update.delta_equiv_plastic_strain;
        } else {
            check(
                update.delta_equiv_plastic_strain == 0.0,
                format!("elastic step {step} accumulated plastic strain"),
            )?;
        }
        update_plastic_state(&mut load_state, &update, &material);
    }
    check(
        first_yield_step == 2,
        format!("first yield at step {first_yield_step}, expected step 2"),
    )?;
    check(
        yield_consistency_error <= TOL_CONSISTENCY,
        format!(
            "yield consistency error {yield_consistency_error:.3e} exceeds {TOL_CONSISTENCY:.1e}"
        ),
    )?;
    let total_equivalent_plastic_strain = load_state.equiv_plastic_strain;
    check(
        total_equivalent_plastic_strain > 0.0,
        "loading produced no accumulated plastic strain".to_string(),
    )?;
    // Perfect plasticity dissipates exactly yield stress times accumulated
    // equivalent plastic strain.
    let expected_work = YIELD_STRESS * total_equivalent_plastic_strain;
    let plastic_work_error = (plastic_work - expected_work).abs() / expected_work;
    check(
        plastic_work_error <= TOL_PLASTIC_WORK,
        format!("plastic work error {plastic_work_error:.3e} exceeds {TOL_PLASTIC_WORK:.1e}"),
    )?;

    // Hardening scenario: the updated stress must track the expanded yield
    // surface, and the surface must grow with the hardening modulus.
    let mut hardening_state = PlasticState::new(6);
    let mut hardening_consistency_error = 0.0f64;
    let mut previous_yield = YIELD_STRESS;
    let mut previous_plastic = 0.0f64;
    let mut hardening_slope_error = 0.0f64;
    for _ in 0..LOAD_STEPS {
        let update = return_mapping_3d(&hardening_material, &hardening_state, &shear_inc);
        update_plastic_state(&mut hardening_state, &update, &hardening_material);
        if update.yielding {
            let current_yield =
                hardening_material.current_yield_stress(hardening_state.equiv_plastic_strain);
            let equivalent = von_mises(&hardening_state.stress);
            let consistency = (equivalent - current_yield).abs() / current_yield;
            hardening_consistency_error = hardening_consistency_error.max(consistency);
            let delta_plastic = hardening_state.equiv_plastic_strain - previous_plastic;
            if delta_plastic > 0.0 && previous_plastic > 0.0 {
                let slope = (current_yield - previous_yield) / delta_plastic;
                let slope_error =
                    (slope - ISOTROPIC_HARDENING_MODULUS).abs() / ISOTROPIC_HARDENING_MODULUS;
                hardening_slope_error = hardening_slope_error.max(slope_error);
            }
            previous_yield = current_yield;
            previous_plastic = hardening_state.equiv_plastic_strain;
        }
    }
    check(
        hardening_consistency_error <= TOL_CONSISTENCY,
        format!(
            "hardening consistency error {hardening_consistency_error:.3e} exceeds {TOL_CONSISTENCY:.1e}"
        ),
    )?;
    check(
        hardening_slope_error <= TOL_HARDENING_SLOPE,
        format!(
            "hardening slope error {hardening_slope_error:.3e} exceeds {TOL_HARDENING_SLOPE:.1e}"
        ),
    )?;

    // Unload/reload: a small reversed increment unloads elastically with the
    // elastic shear stiffness, then continued reversal yields again.
    let stress_before_unload = load_state.stress[5];
    let plastic_before_unload = load_state.equiv_plastic_strain;
    let reverse_inc = [0.0, 0.0, 0.0, 0.0, 0.0, -SHEAR_STEP];
    let unload = return_mapping_3d(&material, &load_state, &reverse_inc);
    check(
        !unload.yielding,
        "small reversed increment should unload elastically".to_string(),
    )?;
    let unload_stress_change = unload.stress[5] - stress_before_unload;
    let expected_unload_change = -shear_modulus * SHEAR_STEP;
    let unload_slope_error =
        (unload_stress_change - expected_unload_change).abs() / expected_unload_change.abs();
    check(
        unload_slope_error <= TOL_UNLOAD_SLOPE,
        format!("unload slope error {unload_slope_error:.3e} exceeds {TOL_UNLOAD_SLOPE:.1e}"),
    )?;
    update_plastic_state(&mut load_state, &unload, &material);
    check(
        (load_state.equiv_plastic_strain - plastic_before_unload).abs() == 0.0,
        "elastic unload accumulated plastic strain".to_string(),
    )?;

    let mut reverse_consistency_error = f64::INFINITY;
    let mut reverse_yielded = false;
    for _ in 0..6 {
        let update = return_mapping_3d(&material, &load_state, &reverse_inc);
        update_plastic_state(&mut load_state, &update, &material);
        if update.yielding {
            reverse_yielded = true;
            let equivalent = von_mises(&load_state.stress);
            reverse_consistency_error = (equivalent - YIELD_STRESS).abs() / YIELD_STRESS;
        }
    }
    check(
        reverse_yielded,
        "continued reversal never yielded".to_string(),
    )?;
    check(
        reverse_consistency_error <= TOL_CONSISTENCY,
        format!(
            "reverse consistency error {reverse_consistency_error:.3e} exceeds {TOL_CONSISTENCY:.1e}"
        ),
    )?;

    Ok(Metrics {
        property_checks,
        elastic_checks,
        first_yield_step,
        yield_consistency_error,
        plastic_work_error,
        hardening_consistency_error,
        hardening_slope_error,
        unload_slope_error,
        reverse_consistency_error,
        total_equivalent_plastic_strain,
        plastic_work,
    })
}

fn main() {
    let result = match run() {
        Ok(metrics) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("property_checks", metrics.property_checks as f64)
            .minimum("property_checks", 7.0)
            .metric("elastic_checks", metrics.elastic_checks as f64)
            .minimum("elastic_checks", 2.0)
            .metric("first_yield_step", metrics.first_yield_step as f64)
            .minimum("first_yield_step", 2.0)
            .maximum("first_yield_step", 2.0)
            .metric(
                "total_equivalent_plastic_strain",
                metrics.total_equivalent_plastic_strain,
            )
            .minimum("total_equivalent_plastic_strain", 1e-6)
            .metric("plastic_work", metrics.plastic_work)
            .minimum("plastic_work", 1.0)
            .metric("yield_consistency_error", metrics.yield_consistency_error)
            .maximum("yield_consistency_error", TOL_CONSISTENCY)
            .metric("plastic_work_error", metrics.plastic_work_error)
            .maximum("plastic_work_error", TOL_PLASTIC_WORK)
            .metric(
                "hardening_consistency_error",
                metrics.hardening_consistency_error,
            )
            .maximum("hardening_consistency_error", TOL_CONSISTENCY)
            .metric("hardening_slope_error", metrics.hardening_slope_error)
            .maximum("hardening_slope_error", TOL_HARDENING_SLOPE)
            .metric("unload_slope_error", metrics.unload_slope_error)
            .maximum("unload_slope_error", TOL_UNLOAD_SLOPE)
            .metric(
                "reverse_consistency_error",
                metrics.reverse_consistency_error,
            )
            .maximum("reverse_consistency_error", TOL_CONSISTENCY)
            .reference(
                "J2 radial return: yield consistency, plastic work, and elastic unload slope",
            )
            .reference_checksum("analytic/deterministic-fixture:fem-plasticity-validation-v1")
            .dimension("3-D Voigt material point under pure shear load/unload")
            .formulation("radial return mapping with perfect and linear isotropic hardening")
            .tolerance("yield consistency 1e-9; plastic work 1e-9; unload slope 1e-9")
            .input_checksum("material:E200GPa-nu0.3-sigmay250MPa-H1GPa:v1"),
        Err(error) => QaResult::failed(CASE_ID, error)
            .requested_backend("cpu")
            .actual_backend("cpu"),
    };

    match result.to_json_line() {
        Ok(json) => println!("QA_RESULT: {json}"),
        Err(error) => {
            eprintln!("plasticity QA serialization failed: {error}");
            std::process::exit(1);
        }
    }

    if matches!(result.status, QaStatus::Passed) {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
