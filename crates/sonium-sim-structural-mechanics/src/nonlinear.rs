//! Small, validated nonlinear structural execution families.
//!
//! These models are deliberately reduced so their constitutive and contact
//! gates can be checked against closed-form solutions. They are useful
//! reference families for orchestration and QA; they are not a substitute for
//! a three-dimensional plasticity or surface-contact FEM implementation.

use super::{StructuralExecutionProvenance, StructuralMechanicsError, StructuralModelAssumption};
use serde::{Deserialize, Serialize};

/// Monotone small-strain axial bar with one-dimensional J2-equivalent
/// elastoplastic return mapping and linear isotropic hardening.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElastoplasticBarStudy {
    pub length_m: f64,
    pub area_m2: f64,
    pub elements: usize,
    pub youngs_modulus_pa: f64,
    pub yield_stress_pa: f64,
    pub isotropic_hardening_pa: f64,
    pub max_force_n: f64,
    pub load_steps: usize,
    pub max_newton_iterations: usize,
    pub residual_tolerance_n: f64,
}

impl ElastoplasticBarStudy {
    /// A compact bar case that crosses yield while remaining well-conditioned.
    pub fn introductory() -> Self {
        Self {
            length_m: 1.0,
            area_m2: 1.0e-4,
            elements: 4,
            youngs_modulus_pa: 200.0e9,
            yield_stress_pa: 100.0e6,
            isotropic_hardening_pa: 10.0e9,
            max_force_n: 15_000.0,
            load_steps: 20,
            max_newton_iterations: 20,
            residual_tolerance_n: 1.0e-9,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        let positive = [
            ("elastoplastic_bar.length_m", self.length_m),
            ("elastoplastic_bar.area_m2", self.area_m2),
            (
                "elastoplastic_bar.youngs_modulus_pa",
                self.youngs_modulus_pa,
            ),
            ("elastoplastic_bar.yield_stress_pa", self.yield_stress_pa),
            (
                "elastoplastic_bar.isotropic_hardening_pa",
                self.isotropic_hardening_pa,
            ),
            ("elastoplastic_bar.max_force_n", self.max_force_n),
            (
                "elastoplastic_bar.residual_tolerance_n",
                self.residual_tolerance_n,
            ),
        ];
        for (field, value) in positive {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if self.elements == 0 || self.load_steps == 0 || self.max_newton_iterations == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "elastoplastic_bar.discretization",
                value: format!(
                    "elements={}, load_steps={}, max_newton_iterations={}",
                    self.elements, self.load_steps, self.max_newton_iterations
                ),
            });
        }
        Ok(())
    }
}

/// One converged load increment of the elastoplastic bar.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElastoplasticBarStepResult {
    pub load_factor: f64,
    pub applied_force_n: f64,
    pub right_displacement_m: f64,
    pub max_stress_pa: f64,
    pub yielded_elements: usize,
    pub newton_iterations: usize,
    pub residual_n: f64,
    pub reaction_force_n: f64,
    pub equilibrium_residual_n: f64,
    pub internal_energy_j: f64,
    pub plastic_dissipation_j: f64,
    pub converged: bool,
}

/// Full load path and final state for an elastoplastic bar execution.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ElastoplasticBarResult {
    pub study: ElastoplasticBarStudy,
    pub steps: Vec<ElastoplasticBarStepResult>,
    pub final_displacement_m: f64,
    pub final_stress_pa: Vec<f64>,
    pub final_plastic_strain: Vec<f64>,
    pub total_plastic_dissipation_j: f64,
    pub max_equilibrium_residual_n: f64,
    pub model_assumption: StructuralModelAssumption,
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct PlasticState {
    plastic_strain: f64,
    hardening_variable: f64,
}

#[derive(Debug, Clone, Copy)]
struct ElementResponse {
    stress_pa: f64,
    tangent_pa: f64,
    state: PlasticState,
}

struct BarAssembly {
    internal_force_n: Vec<f64>,
    tangent: Vec<Vec<f64>>,
    elements: Vec<ElementResponse>,
}

/// Solve the monotone elastoplastic bar with Newton iterations at every load
/// increment. The committed constitutive state is changed only after a step
/// passes its residual gate.
pub fn run_elastoplastic_bar(
    study: &ElastoplasticBarStudy,
) -> Result<ElastoplasticBarResult, StructuralMechanicsError> {
    study.validate()?;
    let node_count = study.elements + 1;
    let element_length_m = study.length_m / study.elements as f64;
    let mut displacement = vec![0.0; node_count];
    let mut committed = vec![
        PlasticState {
            plastic_strain: 0.0,
            hardening_variable: 0.0,
        };
        study.elements
    ];
    let mut steps = Vec::with_capacity(study.load_steps);
    let mut total_plastic_dissipation_j = 0.0;
    let mut max_equilibrium_residual_n: f64 = 0.0;

    for step_number in 1..=study.load_steps {
        let load_factor = step_number as f64 / study.load_steps as f64;
        let applied_force_n = study.max_force_n * load_factor;
        let previous_displacement = displacement.clone();
        let previous_states = committed.clone();
        let mut trial_displacement = displacement.clone();
        let mut converged_response = None;
        let mut residual_n = f64::INFINITY;
        let mut iterations = 0;

        for iteration in 1..=study.max_newton_iterations {
            let assembled = assemble_bar(
                &trial_displacement,
                &previous_states,
                element_length_m,
                study,
            );
            let mut residual = assembled.internal_force_n.clone();
            residual[study.elements] -= applied_force_n;
            residual_n = residual[1..]
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
                .sqrt();
            iterations = iteration;
            let tolerance = study
                .residual_tolerance_n
                .max(1.0e-12 * applied_force_n.max(1.0));
            if !residual_n.is_finite() {
                break;
            }
            if residual_n <= tolerance {
                converged_response = Some(assembled);
                break;
            }

            let free_tangent = (1..node_count)
                .map(|row| {
                    (1..node_count)
                        .map(|column| assembled.tangent[row][column])
                        .collect::<Vec<_>>()
                })
                .collect::<Vec<_>>();
            let correction_rhs = residual[1..].iter().map(|value| -value).collect::<Vec<_>>();
            let correction = solve_dense(&free_tangent, &correction_rhs).ok_or_else(|| {
                StructuralMechanicsError::SolverFailure {
                    stage: "elastoplastic bar Newton tangent",
                    message: "consistent tangent is singular".to_string(),
                }
            })?;
            for (node, increment) in correction.into_iter().enumerate() {
                trial_displacement[node + 1] += increment;
            }
        }

        let assembled =
            converged_response.ok_or(StructuralMechanicsError::NonlinearNonConvergence {
                model: StructuralModelAssumption::ElastoplasticBar,
                load_step: step_number,
                iterations,
                residual: residual_n,
            })?;
        displacement = trial_displacement;
        let mut yielded_elements = 0;
        let mut plastic_dissipation_step_j = 0.0;
        for (element, response) in assembled.elements.iter().enumerate() {
            if response.state.hardening_variable
                > previous_states[element].hardening_variable + 1.0e-14
            {
                yielded_elements += 1;
            }
            let delta_plastic_strain =
                response.state.plastic_strain - previous_states[element].plastic_strain;
            plastic_dissipation_step_j +=
                response.stress_pa * delta_plastic_strain * study.area_m2 * element_length_m;
        }
        plastic_dissipation_step_j = plastic_dissipation_step_j.max(0.0);
        total_plastic_dissipation_j += plastic_dissipation_step_j;
        committed = assembled.elements.iter().map(|item| item.state).collect();

        let reaction_force_n = -assembled.internal_force_n[0];
        let equilibrium_residual_n = (reaction_force_n - applied_force_n).abs();
        max_equilibrium_residual_n = max_equilibrium_residual_n.max(equilibrium_residual_n);
        let internal_energy_j = assembled
            .elements
            .iter()
            .enumerate()
            .map(|(element, response)| {
                let strain = (displacement[element + 1] - displacement[element]) / element_length_m;
                let elastic_strain = strain - response.state.plastic_strain;
                (0.5 * study.youngs_modulus_pa * elastic_strain * elastic_strain
                    + 0.5
                        * study.isotropic_hardening_pa
                        * response.state.hardening_variable
                        * response.state.hardening_variable)
                    * study.area_m2
                    * element_length_m
            })
            .sum();
        steps.push(ElastoplasticBarStepResult {
            load_factor,
            applied_force_n,
            right_displacement_m: displacement[study.elements],
            max_stress_pa: assembled
                .elements
                .iter()
                .map(|response| response.stress_pa.abs())
                .fold(0.0, f64::max),
            yielded_elements,
            newton_iterations: iterations,
            residual_n,
            reaction_force_n,
            equilibrium_residual_n,
            internal_energy_j,
            plastic_dissipation_j: plastic_dissipation_step_j,
            converged: true,
        });

        if equilibrium_residual_n > 1.0e-8 * applied_force_n.max(1.0) {
            return Err(StructuralMechanicsError::ValidationFailure {
                check: "elastoplastic_equilibrium",
                value: equilibrium_residual_n,
            });
        }
        let _ = previous_displacement;
    }

    let final_stress_pa = assemble_bar(&displacement, &committed, element_length_m, study)
        .elements
        .iter()
        .map(|response| response.stress_pa)
        .collect::<Vec<_>>();
    Ok(ElastoplasticBarResult {
        study: study.clone(),
        steps,
        final_displacement_m: displacement[study.elements],
        final_stress_pa,
        final_plastic_strain: committed.iter().map(|state| state.plastic_strain).collect(),
        total_plastic_dissipation_j,
        max_equilibrium_residual_n,
        model_assumption: StructuralModelAssumption::ElastoplasticBar,
    })
}

fn assemble_bar(
    displacement: &[f64],
    states: &[PlasticState],
    element_length_m: f64,
    study: &ElastoplasticBarStudy,
) -> BarAssembly {
    let mut internal_force_n = vec![0.0; study.elements + 1];
    let mut tangent = vec![vec![0.0; study.elements + 1]; study.elements + 1];
    let mut elements = Vec::with_capacity(study.elements);
    for element in 0..study.elements {
        let strain = (displacement[element + 1] - displacement[element]) / element_length_m;
        let response = return_mapping(strain, states[element], study);
        let force_n = response.stress_pa * study.area_m2;
        let stiffness_n_per_m = response.tangent_pa * study.area_m2 / element_length_m;
        internal_force_n[element] -= force_n;
        internal_force_n[element + 1] += force_n;
        tangent[element][element] += stiffness_n_per_m;
        tangent[element][element + 1] -= stiffness_n_per_m;
        tangent[element + 1][element] -= stiffness_n_per_m;
        tangent[element + 1][element + 1] += stiffness_n_per_m;
        elements.push(response);
    }
    BarAssembly {
        internal_force_n,
        tangent,
        elements,
    }
}

fn return_mapping(
    strain: f64,
    previous: PlasticState,
    study: &ElastoplasticBarStudy,
) -> ElementResponse {
    let trial_stress_pa = study.youngs_modulus_pa * (strain - previous.plastic_strain);
    let yield_limit_pa =
        study.yield_stress_pa + study.isotropic_hardening_pa * previous.hardening_variable;
    if trial_stress_pa.abs() <= yield_limit_pa {
        return ElementResponse {
            stress_pa: trial_stress_pa,
            tangent_pa: study.youngs_modulus_pa,
            state: previous,
        };
    }
    let sign = trial_stress_pa.signum();
    let plastic_increment = (trial_stress_pa.abs() - yield_limit_pa)
        / (study.youngs_modulus_pa + study.isotropic_hardening_pa);
    ElementResponse {
        stress_pa: trial_stress_pa - sign * study.youngs_modulus_pa * plastic_increment,
        tangent_pa: study.youngs_modulus_pa * study.isotropic_hardening_pa
            / (study.youngs_modulus_pa + study.isotropic_hardening_pa),
        state: PlasticState {
            plastic_strain: previous.plastic_strain + sign * plastic_increment,
            hardening_variable: previous.hardening_variable + plastic_increment,
        },
    }
}

/// One-dimensional frictionless contact against a rigid stop. The contact
/// force uses a penalty stiffness, so the result exposes penetration and does
/// not claim exact complementarity.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrictionlessContactStudy {
    pub support_stiffness_n_per_m: f64,
    pub contact_penalty_stiffness_n_per_m: f64,
    pub gap_m: f64,
    pub max_force_n: f64,
    pub load_steps: usize,
    pub max_newton_iterations: usize,
    pub residual_tolerance_n: f64,
}

impl FrictionlessContactStudy {
    pub fn introductory() -> Self {
        Self {
            support_stiffness_n_per_m: 1_000.0,
            contact_penalty_stiffness_n_per_m: 1.0e6,
            gap_m: 0.01,
            max_force_n: 20.0,
            load_steps: 10,
            max_newton_iterations: 20,
            residual_tolerance_n: 1.0e-10,
        }
    }

    fn validate(&self) -> Result<(), StructuralMechanicsError> {
        let positive = [
            (
                "frictionless_contact.support_stiffness_n_per_m",
                self.support_stiffness_n_per_m,
            ),
            (
                "frictionless_contact.contact_penalty_stiffness_n_per_m",
                self.contact_penalty_stiffness_n_per_m,
            ),
            ("frictionless_contact.max_force_n", self.max_force_n),
            (
                "frictionless_contact.residual_tolerance_n",
                self.residual_tolerance_n,
            ),
        ];
        for (field, value) in positive {
            if !value.is_finite() || value <= 0.0 {
                return Err(StructuralMechanicsError::InvalidConfiguration {
                    field,
                    value: value.to_string(),
                });
            }
        }
        if !self.gap_m.is_finite() || self.gap_m < 0.0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "frictionless_contact.gap_m",
                value: self.gap_m.to_string(),
            });
        }
        if self.load_steps == 0 || self.max_newton_iterations == 0 {
            return Err(StructuralMechanicsError::InvalidConfiguration {
                field: "frictionless_contact.discretization",
                value: format!(
                    "load_steps={}, max_newton_iterations={}",
                    self.load_steps, self.max_newton_iterations
                ),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ContactStepResult {
    pub load_factor: f64,
    pub applied_force_n: f64,
    pub displacement_m: f64,
    pub contact_active: bool,
    pub penetration_m: f64,
    pub contact_force_n: f64,
    pub reaction_force_n: f64,
    pub equilibrium_residual_n: f64,
    pub newton_iterations: usize,
    pub converged: bool,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FrictionlessContactResult {
    pub study: FrictionlessContactStudy,
    pub steps: Vec<ContactStepResult>,
    pub contact_engaged_step: Option<usize>,
    pub final_displacement_m: f64,
    pub final_penetration_m: f64,
    pub max_equilibrium_residual_n: f64,
    pub model_assumption: StructuralModelAssumption,
    pub execution: StructuralExecutionProvenance,
}

/// Solve the penalty-contact path with an active-set Newton tangent.
pub fn run_frictionless_contact(
    study: &FrictionlessContactStudy,
) -> Result<FrictionlessContactResult, StructuralMechanicsError> {
    study.validate()?;
    let mut displacement_m = 0.0;
    let mut steps = Vec::with_capacity(study.load_steps);
    let mut contact_engaged_step = None;
    let mut max_equilibrium_residual_n: f64 = 0.0;

    for step_number in 1..=study.load_steps {
        let load_factor = step_number as f64 / study.load_steps as f64;
        let applied_force_n = study.max_force_n * load_factor;
        let mut converged = false;
        let mut residual_n = f64::INFINITY;
        let mut iterations = 0;
        for iteration in 1..=study.max_newton_iterations {
            let penetration_m = (displacement_m - study.gap_m).max(0.0);
            let contact_force_n = study.contact_penalty_stiffness_n_per_m * penetration_m;
            residual_n = study.support_stiffness_n_per_m * displacement_m + contact_force_n
                - applied_force_n;
            iterations = iteration;
            let tolerance = study
                .residual_tolerance_n
                .max(1.0e-12 * applied_force_n.max(1.0));
            if residual_n.abs() <= tolerance {
                converged = true;
                break;
            }
            let tangent = study.support_stiffness_n_per_m
                + if displacement_m > study.gap_m {
                    study.contact_penalty_stiffness_n_per_m
                } else {
                    0.0
                };
            displacement_m -= residual_n / tangent;
            if !displacement_m.is_finite() {
                break;
            }
        }
        if !converged {
            return Err(StructuralMechanicsError::NonlinearNonConvergence {
                model: StructuralModelAssumption::FrictionlessContact,
                load_step: step_number,
                iterations,
                residual: residual_n.abs(),
            });
        }
        let penetration_m = (displacement_m - study.gap_m).max(0.0);
        let contact_force_n = study.contact_penalty_stiffness_n_per_m * penetration_m;
        let reaction_force_n = study.support_stiffness_n_per_m * displacement_m + contact_force_n;
        let equilibrium_residual_n = (reaction_force_n - applied_force_n).abs();
        let contact_active = penetration_m > 1.0e-14 * study.gap_m.max(1.0);
        if contact_active && contact_engaged_step.is_none() {
            contact_engaged_step = Some(step_number);
        }
        max_equilibrium_residual_n = max_equilibrium_residual_n.max(equilibrium_residual_n);
        if equilibrium_residual_n > 1.0e-8 * applied_force_n.max(1.0) {
            return Err(StructuralMechanicsError::ValidationFailure {
                check: "frictionless_contact_equilibrium",
                value: equilibrium_residual_n,
            });
        }
        steps.push(ContactStepResult {
            load_factor,
            applied_force_n,
            displacement_m,
            contact_active,
            penetration_m,
            contact_force_n,
            reaction_force_n,
            equilibrium_residual_n,
            newton_iterations: iterations,
            converged,
        });
    }

    Ok(FrictionlessContactResult {
        study: study.clone(),
        steps,
        contact_engaged_step,
        final_displacement_m: displacement_m,
        final_penetration_m: (displacement_m - study.gap_m).max(0.0),
        max_equilibrium_residual_n,
        model_assumption: StructuralModelAssumption::FrictionlessContact,
        execution: StructuralExecutionProvenance::cpu_contact_reference(),
    })
}

fn solve_dense(matrix: &[Vec<f64>], rhs: &[f64]) -> Option<Vec<f64>> {
    let n = rhs.len();
    if matrix.len() != n || matrix.iter().any(|row| row.len() != n) {
        return None;
    }
    let mut augmented = matrix
        .iter()
        .zip(rhs)
        .map(|(row, value)| {
            let mut row = row.clone();
            row.push(*value);
            row
        })
        .collect::<Vec<_>>();
    for pivot in 0..n {
        let pivot_row = (pivot..n).max_by(|&lhs, &rhs| {
            augmented[lhs][pivot]
                .abs()
                .total_cmp(&augmented[rhs][pivot].abs())
        })?;
        if augmented[pivot_row][pivot].abs() <= f64::EPSILON {
            return None;
        }
        augmented.swap(pivot, pivot_row);
        let diagonal = augmented[pivot][pivot];
        for value in augmented[pivot].iter_mut().skip(pivot).take(n + 1 - pivot) {
            *value /= diagonal;
        }
        let normalized_pivot = augmented[pivot][pivot..=n].to_vec();
        for (row, row_values) in augmented.iter_mut().enumerate().take(n) {
            if row == pivot {
                continue;
            }
            let scale = row_values[pivot];
            for (offset, value) in row_values
                .iter_mut()
                .skip(pivot)
                .take(n + 1 - pivot)
                .enumerate()
            {
                *value -= scale * normalized_pivot[offset];
            }
        }
    }
    Some((0..n).map(|row| augmented[row][n]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elastoplastic_bar_matches_elastic_plastic_closed_form_and_equilibrium() {
        let study = ElastoplasticBarStudy::introductory();
        let result = run_elastoplastic_bar(&study).unwrap();
        let stress_pa = study.max_force_n / study.area_m2;
        let expected_extension_m = study.length_m
            * (stress_pa / study.youngs_modulus_pa
                + (stress_pa - study.yield_stress_pa) / study.isotropic_hardening_pa);

        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::ElastoplasticBar
        );
        assert_eq!(result.steps.len(), study.load_steps);
        assert_eq!(
            result.steps.last().unwrap().yielded_elements,
            study.elements
        );
        assert!((result.final_displacement_m - expected_extension_m).abs() < 1.0e-11);
        assert!(result.total_plastic_dissipation_j > 0.0);
        assert!(result.max_equilibrium_residual_n < 1.0e-9);
        assert!(result.steps.iter().all(|step| step.converged));
        assert!(result.steps.first().unwrap().yielded_elements == 0);
    }

    #[test]
    fn frictionless_contact_matches_penalty_solution_and_reports_activation() {
        let study = FrictionlessContactStudy::introductory();
        let result = run_frictionless_contact(&study).unwrap();
        let expected_displacement_m = (study.max_force_n
            + study.contact_penalty_stiffness_n_per_m * study.gap_m)
            / (study.support_stiffness_n_per_m + study.contact_penalty_stiffness_n_per_m);

        assert_eq!(
            result.model_assumption,
            StructuralModelAssumption::FrictionlessContact
        );
        assert_eq!(
            result.execution.capability_id,
            "sonium-structural.cpu-reference.f64.contact-evidence.v1"
        );
        assert_eq!(result.contact_engaged_step, Some(6));
        assert!((result.final_displacement_m - expected_displacement_m).abs() < 1.0e-12);
        assert!(result.final_penetration_m > 0.0);
        assert!(result.steps[4].contact_active == false);
        assert!(result.steps[5].contact_active);
        assert!(result.max_equilibrium_residual_n < 1.0e-9);
    }

    #[test]
    fn nonlinear_families_reject_unphysical_inputs() {
        let mut bar = ElastoplasticBarStudy::introductory();
        bar.isotropic_hardening_pa = 0.0;
        assert!(matches!(
            run_elastoplastic_bar(&bar),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "elastoplastic_bar.isotropic_hardening_pa",
                ..
            })
        ));

        let mut contact = FrictionlessContactStudy::introductory();
        contact.gap_m = -1.0;
        assert!(matches!(
            run_frictionless_contact(&contact),
            Err(StructuralMechanicsError::InvalidConfiguration {
                field: "frictionless_contact.gap_m",
                ..
            })
        ));
    }
}
