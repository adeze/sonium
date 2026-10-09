//! Material-point reduction of Wolfram's linear-hardening Plasticity example.
//!
//! The dimensions, material, and 20-step displacement multipliers are literal.
//! This checks the constitutive loading path, not the page's plate-with-hole FEM.
//!
//! Source:
//! <https://reference.wolfram.com/language/PDEModels/tutorial/StructuralMechanics/Plasticity.html?view=all>

use serde::Deserialize;
use sonium_fem::physics::solid::plasticity::{
    HardeningLaw, J2Material, PlasticState, StressUpdate, return_mapping_3d, update_plastic_state,
};
use sonium_qa::subprocess::WolframCommandExt;
use std::{env, path::PathBuf, process::Command};

const YOUNGS_MODULUS_PA: f64 = 70.0e9;
const POISSON_RATIO: f64 = 0.3;
const YIELD_STRESS_PA: f64 = 2.5e9;
const HARDENING_MODULUS_PA: f64 = 15.0e9;
const LENGTH_M: f64 = 0.04;
const MAX_DISPLACEMENT_M: f64 = 3.9e-3;

fn wolfram_multipliers() -> Vec<f64> {
    (1..=20)
        .map(|step| 1.0 - (step as f64 / 10.0 - 1.0).abs())
        .collect()
}

fn uniaxial_update(
    material: &J2Material,
    state: &PlasticState,
    axial_strain_increment: f64,
) -> StressUpdate {
    let mut lateral_increment = -material.poisson_ratio * axial_strain_increment;
    let stress_scale = material.yield_stress.max(1.0);
    let mut update = return_mapping_3d(
        material,
        state,
        &[
            axial_strain_increment,
            lateral_increment,
            lateral_increment,
            0.0,
            0.0,
            0.0,
        ],
    );
    for _ in 0..20 {
        let residual = update.stress[1];
        if residual.abs() < 1.0e-11 * stress_scale {
            break;
        }
        let derivative = update.tangent[7] + update.tangent[8];
        assert!(derivative.is_finite() && derivative.abs() > 1.0);
        lateral_increment -= residual / derivative;
        update = return_mapping_3d(
            material,
            state,
            &[
                axial_strain_increment,
                lateral_increment,
                lateral_increment,
                0.0,
                0.0,
                0.0,
            ],
        );
    }
    assert!(update.stress[1].abs() < 1.0e-8 * stress_scale);
    assert!(update.stress[2].abs() < 1.0e-8 * stress_scale);
    update
}

struct AnalyticalState {
    stress: f64,
    signed_plastic_strain: f64,
    accumulated_plastic_strain: f64,
}

impl AnalyticalState {
    fn new() -> Self {
        Self {
            stress: 0.0,
            signed_plastic_strain: 0.0,
            accumulated_plastic_strain: 0.0,
        }
    }

    fn update(&mut self, strain_increment: f64) {
        let trial = self.stress + YOUNGS_MODULUS_PA * strain_increment;
        let current_yield =
            YIELD_STRESS_PA + HARDENING_MODULUS_PA * self.accumulated_plastic_strain;
        let excess = trial.abs() - current_yield;
        if excess <= 0.0 {
            self.stress = trial;
            return;
        }
        let plastic_increment = excess / (YOUNGS_MODULUS_PA + HARDENING_MODULUS_PA);
        let direction = trial.signum();
        self.stress = trial - YOUNGS_MODULUS_PA * plastic_increment * direction;
        self.signed_plastic_strain += plastic_increment * direction;
        self.accumulated_plastic_strain += plastic_increment;
    }
}

struct LoadingPath {
    total_strain: Vec<f64>,
    stress_pa: Vec<f64>,
    accumulated_plastic_strain: Vec<f64>,
    analytical_stress_pa: Vec<f64>,
    analytical_accumulated_plastic_strain: Vec<f64>,
}

fn solve_loading_path() -> LoadingPath {
    let material = J2Material::new(
        YOUNGS_MODULUS_PA,
        POISSON_RATIO,
        YIELD_STRESS_PA,
        HardeningLaw::LinearIsotropic {
            modulus: HARDENING_MODULUS_PA,
        },
    );
    let mut state = PlasticState::new_3d();
    let mut analytical = AnalyticalState::new();
    let mut previous_strain = 0.0;
    let mut path = LoadingPath {
        total_strain: Vec::new(),
        stress_pa: Vec::new(),
        accumulated_plastic_strain: Vec::new(),
        analytical_stress_pa: Vec::new(),
        analytical_accumulated_plastic_strain: Vec::new(),
    };

    for multiplier in wolfram_multipliers() {
        let total_strain = multiplier * MAX_DISPLACEMENT_M / LENGTH_M;
        let increment = total_strain - previous_strain;
        let update = uniaxial_update(&material, &state, increment);
        update_plastic_state(&mut state, &update, &material);
        analytical.update(increment);

        path.total_strain.push(total_strain);
        path.stress_pa.push(state.stress[0]);
        path.accumulated_plastic_strain
            .push(state.equiv_plastic_strain);
        path.analytical_stress_pa.push(analytical.stress);
        path.analytical_accumulated_plastic_strain
            .push(analytical.accumulated_plastic_strain);
        previous_strain = total_strain;
    }
    path
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    multipliers: Vec<f64>,
    total_strain: Vec<f64>,
    stress_pa: Vec<f64>,
    accumulated_plastic_strain: Vec<f64>,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("wolfram/plasticity_loading_path_reduction.wls");
    let output = Command::new(executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
        .expect("run explicit WOLFRAMSCRIPT plasticity reference");
    assert!(
        output.status.success(),
        "Wolfram plasticity reference failed:\nstdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("Wolfram UTF-8 output");
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .expect("Wolfram plasticity JSON output");
    Some(serde_json::from_str(json).expect("parse Wolfram plasticity JSON"))
}

fn relative_l2(actual: &[f64], reference: &[f64]) -> f64 {
    let numerator = actual
        .iter()
        .zip(reference)
        .map(|(actual, reference)| (actual - reference).powi(2))
        .sum::<f64>();
    let denominator = reference.iter().map(|value| value * value).sum::<f64>();
    (numerator / denominator).sqrt()
}

#[test]
fn sonium_matches_wolfram_linear_hardening_loading_path_reduction() {
    let path = solve_loading_path();
    let stress_error = relative_l2(&path.stress_pa, &path.analytical_stress_pa);
    let plastic_strain_error = relative_l2(
        &path.accumulated_plastic_strain,
        &path.analytical_accumulated_plastic_strain,
    );
    eprintln!(
        "Wolfram plasticity loading-path reduction: peak stress={:.8e} Pa, final stress={:.8e} Pa, final accumulated plastic strain={:.8e}, stress/plastic errors={:.3e}/{:.3e}",
        path.stress_pa[9],
        path.stress_pa[19],
        path.accumulated_plastic_strain[19],
        stress_error,
        plastic_strain_error,
    );

    assert_eq!(path.total_strain.len(), 20);
    assert!((path.total_strain[9] - 0.0975).abs() < 1.0e-14);
    assert!(path.total_strain[19].abs() < 1.0e-14);
    assert!(path.stress_pa[9] > YIELD_STRESS_PA);
    assert!(
        path.stress_pa[19] < 0.0,
        "unloaded hardened point retains compression"
    );
    assert!(path.accumulated_plastic_strain[19] > 0.0);
    assert!(stress_error < 1.0e-10);
    assert!(plastic_strain_error < 1.0e-10);

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert!(
            reference
                .multipliers
                .iter()
                .zip(wolfram_multipliers())
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-14)
        );
        assert!(
            reference
                .total_strain
                .iter()
                .zip(&path.total_strain)
                .all(|(actual, expected)| (actual - expected).abs() < 1.0e-14)
        );
        assert!(relative_l2(&path.stress_pa, &reference.stress_pa) < 1.0e-10);
        assert!(
            relative_l2(
                &path.accumulated_plastic_strain,
                &reference.accumulated_plastic_strain,
            ) < 1.0e-10
        );
    }
}
