//! Promotion evidence for reduced versus higher-fidelity PC-fan evaluations.
//!
//! This module does not create high-fidelity observations. It provides the
//! fail-closed contract a caller must satisfy before a reduced-order result
//! can be promoted: candidate topology, objective error, rank preservation,
//! and feasibility parity are checked together.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Fidelity label attached to one candidate observation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PcFanFidelityLevel {
    ReducedOrder,
    MeanFlowCfd,
    RotatingWallCfd,
    ResolvedTransient,
}

/// One candidate result at one fidelity level.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanFidelityObservation {
    pub candidate_id: String,
    pub fidelity: PcFanFidelityLevel,
    pub objective: f64,
    pub feasible: bool,
    pub airflow_cfm: f64,
    pub pressure_rise_pa: f64,
    pub acoustic_level_db: f64,
    /// Optional objective gradient in the shared design-vector basis.
    #[serde(default)]
    pub gradient: Option<Vec<f64>>,
}

/// Provenance-labelled candidate set used for promotion evidence.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanFidelityDataset {
    pub provenance: String,
    pub observations: Vec<PcFanFidelityObservation>,
}

/// Acceptance budgets for reduced-order promotion.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PcFanFidelityPromotionConfig {
    pub max_objective_relative_error: f64,
    pub min_rank_correlation: f64,
    pub min_candidates: usize,
    pub require_feasibility_parity: bool,
    /// Optional minimum cosine similarity between screening and validation
    /// objective gradients for every candidate.
    #[serde(default)]
    pub min_gradient_cosine: Option<f64>,
}

impl Default for PcFanFidelityPromotionConfig {
    fn default() -> Self {
        Self {
            max_objective_relative_error: 0.10,
            min_rank_correlation: 0.90,
            min_candidates: 3,
            require_feasibility_parity: true,
            min_gradient_cosine: None,
        }
    }
}

/// Input or topology failure found before promotion metrics are published.
#[derive(Debug, Clone, PartialEq)]
pub enum PcFanFidelityPromotionError {
    InvalidBudget(&'static str, f64),
    InvalidMinimumCandidates,
    MissingProvenance(&'static str),
    TooFewCandidates {
        dataset: &'static str,
        actual: usize,
        required: usize,
    },
    NonFiniteObservation {
        dataset: &'static str,
        index: usize,
    },
    DuplicateCandidate {
        dataset: &'static str,
        candidate_id: String,
    },
    CandidateTopologyMismatch {
        candidate_id: String,
    },
    FidelityMismatch {
        dataset: &'static str,
        index: usize,
        expected: PcFanFidelityLevel,
        actual: PcFanFidelityLevel,
    },
    AmbiguousObjectiveRanking {
        dataset: &'static str,
        candidate_id: String,
    },
    InvalidGradientBudget(f64),
    MissingGradient {
        dataset: &'static str,
        candidate_id: String,
    },
    GradientDimensionMismatch {
        candidate_id: String,
        screening: usize,
        validation: usize,
    },
    NonFiniteGradient {
        dataset: &'static str,
        candidate_id: String,
    },
    ZeroGradientNorm {
        candidate_id: String,
    },
}

impl fmt::Display for PcFanFidelityPromotionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidBudget(name, value) => {
                write!(f, "PC-fan fidelity budget {name} is invalid: {value}")
            }
            Self::InvalidMinimumCandidates => {
                write!(f, "PC-fan fidelity min_candidates must be at least two")
            }
            Self::MissingProvenance(dataset) => {
                write!(f, "PC-fan fidelity {dataset} provenance must not be empty")
            }
            Self::TooFewCandidates {
                dataset,
                actual,
                required,
            } => write!(
                f,
                "PC-fan fidelity {dataset} dataset has {actual} candidates; requires {required}"
            ),
            Self::NonFiniteObservation { dataset, index } => write!(
                f,
                "PC-fan fidelity {dataset} candidate {index} contains a non-finite metric"
            ),
            Self::DuplicateCandidate {
                dataset,
                candidate_id,
            } => write!(
                f,
                "PC-fan fidelity {dataset} repeats candidate {candidate_id}"
            ),
            Self::CandidateTopologyMismatch { candidate_id } => write!(
                f,
                "PC-fan fidelity candidate topology mismatch at {candidate_id}"
            ),
            Self::FidelityMismatch {
                dataset,
                index,
                expected,
                actual,
            } => write!(
                f,
                "PC-fan fidelity {dataset} candidate {index} has {actual:?}; expected {expected:?}"
            ),
            Self::AmbiguousObjectiveRanking {
                dataset,
                candidate_id,
            } => write!(
                f,
                "PC-fan fidelity {dataset} has tied objective near candidate {candidate_id}"
            ),
            Self::InvalidGradientBudget(value) => write!(
                f,
                "PC-fan fidelity minimum gradient cosine is invalid: {value}"
            ),
            Self::MissingGradient {
                dataset,
                candidate_id,
            } => write!(
                f,
                "PC-fan fidelity {dataset} is missing gradient for candidate {candidate_id}"
            ),
            Self::GradientDimensionMismatch {
                candidate_id,
                screening,
                validation,
            } => write!(
                f,
                "PC-fan fidelity gradient dimension mismatch for {candidate_id}: screening {screening}, validation {validation}"
            ),
            Self::NonFiniteGradient {
                dataset,
                candidate_id,
            } => write!(
                f,
                "PC-fan fidelity {dataset} gradient for candidate {candidate_id} is non-finite"
            ),
            Self::ZeroGradientNorm { candidate_id } => write!(
                f,
                "PC-fan fidelity gradient for candidate {candidate_id} has zero norm"
            ),
        }
    }
}

impl std::error::Error for PcFanFidelityPromotionError {}

/// Promotion metrics comparing one screening set with one validation set.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PcFanFidelityPromotionReport {
    pub screening_provenance: String,
    pub validation_provenance: String,
    pub candidate_count: usize,
    pub max_objective_relative_error: f64,
    pub rank_correlation: f64,
    /// Minimum screening/validation gradient cosine, when requested.
    pub min_gradient_cosine: Option<f64>,
    pub feasibility_mismatch_count: usize,
    pub false_promotion_count: usize,
    pub screening_best_candidate: String,
    pub validation_best_candidate: String,
    pub best_candidate_matches: bool,
    pub passed: bool,
}

impl PcFanFidelityPromotionReport {
    pub fn is_valid(&self) -> bool {
        self.candidate_count >= 2
            && self.max_objective_relative_error.is_finite()
            && self.rank_correlation.is_finite()
            && self
                .min_gradient_cosine
                .is_none_or(|value| value.is_finite() && (-1.0..=1.0).contains(&value))
            && self.screening_provenance.trim() != ""
            && self.validation_provenance.trim() != ""
            && !self.screening_best_candidate.is_empty()
            && !self.validation_best_candidate.is_empty()
            && self.passed
    }
}

/// Compare reduced-order candidates with a higher-fidelity candidate set.
pub fn evaluate_pc_fan_fidelity_promotion(
    config: &PcFanFidelityPromotionConfig,
    screening: &PcFanFidelityDataset,
    validation: &PcFanFidelityDataset,
) -> Result<PcFanFidelityPromotionReport, PcFanFidelityPromotionError> {
    validate_config(config)?;
    validate_dataset("screening", screening, config.min_candidates)?;
    validate_dataset("validation", validation, config.min_candidates)?;
    validate_topology(screening, validation)?;

    let screening_level = screening.observations[0].fidelity;
    let validation_level = validation.observations[0].fidelity;
    for (index, observation) in screening.observations.iter().enumerate() {
        if observation.fidelity != screening_level {
            return Err(PcFanFidelityPromotionError::FidelityMismatch {
                dataset: "screening",
                index,
                expected: screening_level,
                actual: observation.fidelity,
            });
        }
    }
    for (index, observation) in validation.observations.iter().enumerate() {
        if observation.fidelity != validation_level {
            return Err(PcFanFidelityPromotionError::FidelityMismatch {
                dataset: "validation",
                index,
                expected: validation_level,
                actual: observation.fidelity,
            });
        }
    }
    if screening_level == validation_level {
        return Err(PcFanFidelityPromotionError::FidelityMismatch {
            dataset: "validation",
            index: 0,
            expected: screening_level,
            actual: validation_level,
        });
    }

    let mut screening_sorted = screening.observations.clone();
    let mut validation_sorted = validation.observations.clone();
    screening_sorted.sort_by(|left, right| left.objective.total_cmp(&right.objective));
    validation_sorted.sort_by(|left, right| left.objective.total_cmp(&right.objective));

    let max_objective_relative_error = screening_sorted
        .iter()
        .map(|screening| {
            let validation = validation
                .observations
                .iter()
                .find(|candidate| candidate.candidate_id == screening.candidate_id)
                .expect("candidate topology validated");
            (screening.objective - validation.objective).abs()
                / screening
                    .objective
                    .abs()
                    .max(validation.objective.abs())
                    .max(f64::MIN_POSITIVE)
        })
        .fold(0.0, f64::max);
    let rank_correlation = spearman_rank_correlation(screening, validation);
    let feasibility_mismatch_count = screening
        .observations
        .iter()
        .filter(|screening| {
            let validation = validation
                .observations
                .iter()
                .find(|candidate| candidate.candidate_id == screening.candidate_id)
                .expect("candidate topology validated");
            screening.feasible != validation.feasible
        })
        .count();
    let min_gradient_cosine = if config.min_gradient_cosine.is_some() {
        Some(gradient_cosine_for_candidates(screening, validation)?)
    } else {
        None
    };
    let false_promotion_count = screening
        .observations
        .iter()
        .filter(|screening| {
            let validation = validation
                .observations
                .iter()
                .find(|candidate| candidate.candidate_id == screening.candidate_id)
                .expect("candidate topology validated");
            screening.feasible && !validation.feasible
        })
        .count();
    let screening_best_candidate = screening_sorted[0].candidate_id.clone();
    let validation_best_candidate = validation_sorted[0].candidate_id.clone();
    let best_candidate_matches = screening_best_candidate == validation_best_candidate;
    let passed = max_objective_relative_error <= config.max_objective_relative_error
        && rank_correlation >= config.min_rank_correlation
        && config
            .min_gradient_cosine
            .is_none_or(|minimum| min_gradient_cosine.is_some_and(|value| value >= minimum))
        && best_candidate_matches
        && (!config.require_feasibility_parity || feasibility_mismatch_count == 0);

    Ok(PcFanFidelityPromotionReport {
        screening_provenance: screening.provenance.clone(),
        validation_provenance: validation.provenance.clone(),
        candidate_count: screening_sorted.len(),
        max_objective_relative_error,
        rank_correlation,
        min_gradient_cosine,
        feasibility_mismatch_count,
        false_promotion_count,
        screening_best_candidate,
        validation_best_candidate,
        best_candidate_matches,
        passed,
    })
}

fn validate_config(
    config: &PcFanFidelityPromotionConfig,
) -> Result<(), PcFanFidelityPromotionError> {
    if !config.max_objective_relative_error.is_finite() || config.max_objective_relative_error < 0.0
    {
        return Err(PcFanFidelityPromotionError::InvalidBudget(
            "max_objective_relative_error",
            config.max_objective_relative_error,
        ));
    }
    if !config.min_rank_correlation.is_finite()
        || !(0.0..=1.0).contains(&config.min_rank_correlation)
    {
        return Err(PcFanFidelityPromotionError::InvalidBudget(
            "min_rank_correlation",
            config.min_rank_correlation,
        ));
    }
    if config.min_candidates < 2 {
        return Err(PcFanFidelityPromotionError::InvalidMinimumCandidates);
    }
    if let Some(value) = config.min_gradient_cosine
        && (!value.is_finite() || !(-1.0..=1.0).contains(&value))
    {
        return Err(PcFanFidelityPromotionError::InvalidGradientBudget(value));
    }
    Ok(())
}

fn validate_dataset(
    name: &'static str,
    dataset: &PcFanFidelityDataset,
    min_candidates: usize,
) -> Result<(), PcFanFidelityPromotionError> {
    if dataset.provenance.trim().is_empty() {
        return Err(PcFanFidelityPromotionError::MissingProvenance(name));
    }
    if dataset.observations.len() < min_candidates {
        return Err(PcFanFidelityPromotionError::TooFewCandidates {
            dataset: name,
            actual: dataset.observations.len(),
            required: min_candidates,
        });
    }
    for (index, observation) in dataset.observations.iter().enumerate() {
        if observation.candidate_id.trim().is_empty()
            || [
                observation.objective,
                observation.airflow_cfm,
                observation.pressure_rise_pa,
                observation.acoustic_level_db,
            ]
            .iter()
            .any(|value| !value.is_finite())
        {
            return Err(PcFanFidelityPromotionError::NonFiniteObservation {
                dataset: name,
                index,
            });
        }
        if let Some(gradient) = &observation.gradient
            && (gradient.is_empty() || gradient.iter().any(|value| !value.is_finite()))
        {
            return Err(PcFanFidelityPromotionError::NonFiniteGradient {
                dataset: name,
                candidate_id: observation.candidate_id.clone(),
            });
        }
    }
    for (index, observation) in dataset.observations.iter().enumerate() {
        if dataset.observations[..index]
            .iter()
            .any(|previous| previous.candidate_id == observation.candidate_id)
        {
            return Err(PcFanFidelityPromotionError::DuplicateCandidate {
                dataset: name,
                candidate_id: observation.candidate_id.clone(),
            });
        }
    }
    let mut objectives = dataset.observations.clone();
    objectives.sort_by(|left, right| left.objective.total_cmp(&right.objective));
    for pair in objectives.windows(2) {
        if pair[0].objective == pair[1].objective {
            return Err(PcFanFidelityPromotionError::AmbiguousObjectiveRanking {
                dataset: name,
                candidate_id: pair[1].candidate_id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_topology(
    screening: &PcFanFidelityDataset,
    validation: &PcFanFidelityDataset,
) -> Result<(), PcFanFidelityPromotionError> {
    if screening.observations.len() != validation.observations.len() {
        return Err(PcFanFidelityPromotionError::CandidateTopologyMismatch {
            candidate_id: format!(
                "count:{}:{}",
                screening.observations.len(),
                validation.observations.len()
            ),
        });
    }
    for observation in &screening.observations {
        if !validation
            .observations
            .iter()
            .any(|candidate| candidate.candidate_id == observation.candidate_id)
        {
            return Err(PcFanFidelityPromotionError::CandidateTopologyMismatch {
                candidate_id: observation.candidate_id.clone(),
            });
        }
    }
    Ok(())
}

fn spearman_rank_correlation(
    screening: &PcFanFidelityDataset,
    validation: &PcFanFidelityDataset,
) -> f64 {
    let mut screening_ranked = screening.observations.clone();
    screening_ranked.sort_by(|left, right| left.objective.total_cmp(&right.objective));
    let mut validation_ranked = validation.observations.clone();
    validation_ranked.sort_by(|left, right| left.objective.total_cmp(&right.objective));
    let mut squared_rank_difference = 0.0;
    for (screening_rank, screening_observation) in screening_ranked.iter().enumerate() {
        let validation_rank = validation_ranked
            .iter()
            .position(|observation| observation.candidate_id == screening_observation.candidate_id)
            .expect("candidate topology validated");
        squared_rank_difference += (screening_rank as f64 - validation_rank as f64).powi(2);
    }
    let n = screening_ranked.len() as f64;
    1.0 - 6.0 * squared_rank_difference / (n * (n * n - 1.0))
}

fn gradient_cosine_for_candidates(
    screening: &PcFanFidelityDataset,
    validation: &PcFanFidelityDataset,
) -> Result<f64, PcFanFidelityPromotionError> {
    let mut minimum = f64::INFINITY;
    for screening_observation in &screening.observations {
        let validation_observation = validation
            .observations
            .iter()
            .find(|observation| observation.candidate_id == screening_observation.candidate_id)
            .expect("candidate topology validated");
        let screening_gradient = screening_observation.gradient.as_ref().ok_or_else(|| {
            PcFanFidelityPromotionError::MissingGradient {
                dataset: "screening",
                candidate_id: screening_observation.candidate_id.clone(),
            }
        })?;
        let validation_gradient = validation_observation.gradient.as_ref().ok_or_else(|| {
            PcFanFidelityPromotionError::MissingGradient {
                dataset: "validation",
                candidate_id: validation_observation.candidate_id.clone(),
            }
        })?;
        if screening_gradient.len() != validation_gradient.len() {
            return Err(PcFanFidelityPromotionError::GradientDimensionMismatch {
                candidate_id: screening_observation.candidate_id.clone(),
                screening: screening_gradient.len(),
                validation: validation_gradient.len(),
            });
        }
        let dot = screening_gradient
            .iter()
            .zip(validation_gradient)
            .map(|(left, right)| left * right)
            .sum::<f64>();
        let screening_norm = screening_gradient
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let validation_norm = validation_gradient
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        if screening_norm == 0.0 || validation_norm == 0.0 {
            return Err(PcFanFidelityPromotionError::ZeroGradientNorm {
                candidate_id: screening_observation.candidate_id.clone(),
            });
        }
        let cosine = dot / (screening_norm * validation_norm);
        if !cosine.is_finite() {
            return Err(PcFanFidelityPromotionError::NonFiniteGradient {
                dataset: "gradient-cosine",
                candidate_id: screening_observation.candidate_id.clone(),
            });
        }
        minimum = minimum.min(cosine);
    }
    Ok(minimum)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dataset(
        provenance: &str,
        fidelity: PcFanFidelityLevel,
        objectives: [f64; 4],
        feasible: [bool; 4],
    ) -> PcFanFidelityDataset {
        PcFanFidelityDataset {
            provenance: provenance.into(),
            observations: objectives
                .into_iter()
                .enumerate()
                .map(|(index, objective)| PcFanFidelityObservation {
                    candidate_id: format!("candidate-{index}"),
                    fidelity,
                    objective,
                    feasible: feasible[index],
                    airflow_cfm: 20.0 + index as f64,
                    pressure_rise_pa: 100.0 + index as f64,
                    acoustic_level_db: 40.0 + index as f64,
                    gradient: None,
                })
                .collect(),
        }
    }

    fn with_gradients(
        mut dataset: PcFanFidelityDataset,
        gradients: [[f64; 2]; 4],
    ) -> PcFanFidelityDataset {
        for (observation, gradient) in dataset.observations.iter_mut().zip(gradients) {
            observation.gradient = Some(gradient.to_vec());
        }
        dataset
    }

    #[test]
    fn promotion_accepts_rank_preserving_fidelity_pair() {
        let screening = dataset(
            "solver:reduced:v1",
            PcFanFidelityLevel::ReducedOrder,
            [1.0, 2.0, 3.0, 4.0],
            [true, true, true, true],
        );
        let validation = dataset(
            "solver:wall-cfd:v1",
            PcFanFidelityLevel::RotatingWallCfd,
            [1.02, 2.04, 3.06, 4.08],
            [true, true, true, true],
        );
        let report = evaluate_pc_fan_fidelity_promotion(
            &PcFanFidelityPromotionConfig::default(),
            &screening,
            &validation,
        )
        .expect("valid fidelity pair");
        assert!(report.is_valid());
        assert_eq!(report.candidate_count, 4);
        assert!(report.best_candidate_matches);
        assert_eq!(report.feasibility_mismatch_count, 0);
        assert!(report.rank_correlation > 0.99);
    }

    #[test]
    fn promotion_accepts_aligned_gradients() {
        let screening = with_gradients(
            dataset(
                "solver:reduced:v1",
                PcFanFidelityLevel::ReducedOrder,
                [1.0, 2.0, 3.0, 4.0],
                [true, true, true, true],
            ),
            [[1.0, 0.0]; 4],
        );
        let validation = with_gradients(
            dataset(
                "solver:wall-cfd:v1",
                PcFanFidelityLevel::RotatingWallCfd,
                [1.02, 2.04, 3.06, 4.08],
                [true, true, true, true],
            ),
            [[2.0, 0.0]; 4],
        );
        let mut config = PcFanFidelityPromotionConfig::default();
        config.min_gradient_cosine = Some(0.99);

        let report = evaluate_pc_fan_fidelity_promotion(&config, &screening, &validation)
            .expect("aligned gradients should be accepted");

        assert_eq!(report.min_gradient_cosine, Some(1.0));
        assert!(report.passed);
        assert!(report.is_valid());
    }

    #[test]
    fn promotion_rejects_opposite_gradients() {
        let screening = with_gradients(
            dataset(
                "solver:reduced:v1",
                PcFanFidelityLevel::ReducedOrder,
                [1.0, 2.0, 3.0, 4.0],
                [true, true, true, true],
            ),
            [[1.0, 0.0]; 4],
        );
        let validation = with_gradients(
            dataset(
                "solver:wall-cfd:v1",
                PcFanFidelityLevel::RotatingWallCfd,
                [1.02, 2.04, 3.06, 4.08],
                [true, true, true, true],
            ),
            [[-1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        );
        let mut config = PcFanFidelityPromotionConfig::default();
        config.min_gradient_cosine = Some(0.0);

        let report = evaluate_pc_fan_fidelity_promotion(&config, &screening, &validation)
            .expect("finite, dimensionally compatible gradients should be evaluated");

        assert_eq!(report.min_gradient_cosine, Some(-1.0));
        assert!(!report.passed);
    }

    #[test]
    fn promotion_rejects_orthogonal_gradients() {
        let screening = with_gradients(
            dataset(
                "solver:reduced:v1",
                PcFanFidelityLevel::ReducedOrder,
                [1.0, 2.0, 3.0, 4.0],
                [true, true, true, true],
            ),
            [[1.0, 0.0]; 4],
        );
        let validation = with_gradients(
            dataset(
                "solver:wall-cfd:v1",
                PcFanFidelityLevel::RotatingWallCfd,
                [1.02, 2.04, 3.06, 4.08],
                [true, true, true, true],
            ),
            [[0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        );
        let mut config = PcFanFidelityPromotionConfig::default();
        config.min_gradient_cosine = Some(0.5);

        let report = evaluate_pc_fan_fidelity_promotion(&config, &screening, &validation)
            .expect("finite, dimensionally compatible gradients should be evaluated");

        assert_eq!(report.min_gradient_cosine, Some(0.0));
        assert!(!report.passed);
    }

    #[test]
    fn promotion_rejects_missing_gradient_when_gate_is_enabled() {
        let screening = with_gradients(
            dataset(
                "solver:reduced:v1",
                PcFanFidelityLevel::ReducedOrder,
                [1.0, 2.0, 3.0, 4.0],
                [true, true, true, true],
            ),
            [[1.0, 0.0]; 4],
        );
        let validation = dataset(
            "solver:wall-cfd:v1",
            PcFanFidelityLevel::RotatingWallCfd,
            [1.02, 2.04, 3.06, 4.08],
            [true, true, true, true],
        );
        let mut config = PcFanFidelityPromotionConfig::default();
        config.min_gradient_cosine = Some(0.5);

        assert!(matches!(
            evaluate_pc_fan_fidelity_promotion(&config, &screening, &validation),
            Err(PcFanFidelityPromotionError::MissingGradient {
                dataset: "validation",
                candidate_id,
            }) if candidate_id == "candidate-0"
        ));

        let screening = dataset(
            "solver:reduced:v1",
            PcFanFidelityLevel::ReducedOrder,
            [1.0, 2.0, 3.0, 4.0],
            [true, true, true, true],
        );
        let validation = dataset(
            "solver:wall-cfd:v1",
            PcFanFidelityLevel::RotatingWallCfd,
            [1.02, 2.04, 3.06, 4.08],
            [true, true, true, true],
        );

        assert!(matches!(
            evaluate_pc_fan_fidelity_promotion(&config, &screening, &validation),
            Err(PcFanFidelityPromotionError::MissingGradient {
                dataset: "screening",
                candidate_id,
            }) if candidate_id == "candidate-0"
        ));
    }

    #[test]
    fn promotion_rejects_gradient_dimension_mismatch() {
        let screening = with_gradients(
            dataset(
                "solver:reduced:v1",
                PcFanFidelityLevel::ReducedOrder,
                [1.0, 2.0, 3.0, 4.0],
                [true, true, true, true],
            ),
            [[1.0, 0.0]; 4],
        );
        let mut validation = with_gradients(
            dataset(
                "solver:wall-cfd:v1",
                PcFanFidelityLevel::RotatingWallCfd,
                [1.02, 2.04, 3.06, 4.08],
                [true, true, true, true],
            ),
            [[1.0, 0.0]; 4],
        );
        validation.observations[0].gradient = Some(vec![1.0]);
        let mut config = PcFanFidelityPromotionConfig::default();
        config.min_gradient_cosine = Some(0.5);

        assert!(matches!(
            evaluate_pc_fan_fidelity_promotion(&config, &screening, &validation),
            Err(PcFanFidelityPromotionError::GradientDimensionMismatch {
                candidate_id,
                screening: 2,
                validation: 1,
            }) if candidate_id == "candidate-0"
        ));
    }

    #[test]
    fn promotion_rejects_zero_and_non_finite_gradients() {
        let screening = with_gradients(
            dataset(
                "solver:reduced:v1",
                PcFanFidelityLevel::ReducedOrder,
                [1.0, 2.0, 3.0, 4.0],
                [true, true, true, true],
            ),
            [[0.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
        );
        let validation = with_gradients(
            dataset(
                "solver:wall-cfd:v1",
                PcFanFidelityLevel::RotatingWallCfd,
                [1.02, 2.04, 3.06, 4.08],
                [true, true, true, true],
            ),
            [[1.0, 0.0]; 4],
        );
        let mut config = PcFanFidelityPromotionConfig::default();
        config.min_gradient_cosine = Some(0.5);

        assert!(matches!(
            evaluate_pc_fan_fidelity_promotion(&config, &screening, &validation),
            Err(PcFanFidelityPromotionError::ZeroGradientNorm {
                candidate_id,
            }) if candidate_id == "candidate-0"
        ));

        let screening = with_gradients(
            dataset(
                "solver:reduced:v1",
                PcFanFidelityLevel::ReducedOrder,
                [1.0, 2.0, 3.0, 4.0],
                [true, true, true, true],
            ),
            [[1.0, 0.0]; 4],
        );
        let mut validation = with_gradients(
            dataset(
                "solver:wall-cfd:v1",
                PcFanFidelityLevel::RotatingWallCfd,
                [1.02, 2.04, 3.06, 4.08],
                [true, true, true, true],
            ),
            [[1.0, 0.0]; 4],
        );
        validation.observations[0].gradient = Some(vec![f64::NAN, 0.0]);

        assert!(matches!(
            evaluate_pc_fan_fidelity_promotion(&config, &screening, &validation),
            Err(PcFanFidelityPromotionError::NonFiniteGradient {
                dataset: "validation",
                candidate_id,
            }) if candidate_id == "candidate-0"
        ));
    }

    #[test]
    fn promotion_rejects_invalid_gradient_cosine_budget() {
        let screening = dataset(
            "solver:reduced:v1",
            PcFanFidelityLevel::ReducedOrder,
            [1.0, 2.0, 3.0, 4.0],
            [true, true, true, true],
        );
        let validation = dataset(
            "solver:wall-cfd:v1",
            PcFanFidelityLevel::RotatingWallCfd,
            [1.02, 2.04, 3.06, 4.08],
            [true, true, true, true],
        );
        let mut config = PcFanFidelityPromotionConfig::default();
        config.min_gradient_cosine = Some(1.1);

        assert!(matches!(
            evaluate_pc_fan_fidelity_promotion(&config, &screening, &validation),
            Err(PcFanFidelityPromotionError::InvalidGradientBudget(value))
                if value == 1.1
        ));
    }

    #[test]
    fn promotion_rejects_changed_winner_even_when_rank_budget_passes() {
        let screening = dataset(
            "solver:reduced:v1",
            PcFanFidelityLevel::ReducedOrder,
            [1.0, 2.0, 3.0, 4.0],
            [true, true, true, true],
        );
        let validation = dataset(
            "solver:wall-cfd:v1",
            PcFanFidelityLevel::RotatingWallCfd,
            [2.0, 1.0, 3.0, 4.0],
            [true, true, true, true],
        );
        let mut config = PcFanFidelityPromotionConfig::default();
        config.max_objective_relative_error = 0.5;
        config.min_rank_correlation = 0.75;

        let report = evaluate_pc_fan_fidelity_promotion(&config, &screening, &validation)
            .expect("well-formed but changed-winner promotion");
        assert!(report.rank_correlation >= config.min_rank_correlation);
        assert!(report.max_objective_relative_error <= config.max_objective_relative_error);
        assert!(!report.best_candidate_matches);
        assert!(!report.passed);
        assert!(!report.is_valid());
    }

    #[test]
    fn promotion_rejects_false_feasibility_and_rank_movement() {
        let screening = dataset(
            "solver:reduced:v1",
            PcFanFidelityLevel::ReducedOrder,
            [1.0, 2.0, 3.0, 4.0],
            [true, true, true, true],
        );
        let validation = dataset(
            "solver:resolved:v1",
            PcFanFidelityLevel::ResolvedTransient,
            [4.0, 1.0, 3.0, 2.0],
            [false, true, true, true],
        );
        let report = evaluate_pc_fan_fidelity_promotion(
            &PcFanFidelityPromotionConfig::default(),
            &screening,
            &validation,
        )
        .expect("well-formed but failed promotion");
        assert!(!report.passed);
        assert_eq!(report.false_promotion_count, 1);
        assert!(!report.best_candidate_matches);
    }

    #[test]
    fn promotion_rejects_malformed_inputs_and_ties() {
        let mut config = PcFanFidelityPromotionConfig::default();
        config.min_rank_correlation = f64::NAN;
        let screening = dataset(
            "solver:reduced:v1",
            PcFanFidelityLevel::ReducedOrder,
            [1.0, 2.0, 3.0, 4.0],
            [true, true, true, true],
        );
        let validation = dataset(
            "solver:wall-cfd:v1",
            PcFanFidelityLevel::RotatingWallCfd,
            [1.0, 2.0, 3.0, 4.0],
            [true, true, true, true],
        );
        assert!(matches!(
            evaluate_pc_fan_fidelity_promotion(&config, &screening, &validation),
            Err(PcFanFidelityPromotionError::InvalidBudget(
                "min_rank_correlation",
                value
            )) if value.is_nan()
        ));

        let mut tied = screening.clone();
        tied.observations[1].objective = tied.observations[0].objective;
        assert!(matches!(
            evaluate_pc_fan_fidelity_promotion(
                &PcFanFidelityPromotionConfig::default(),
                &tied,
                &validation,
            ),
            Err(PcFanFidelityPromotionError::AmbiguousObjectiveRanking {
                dataset: "screening",
                ..
            })
        ));
    }

    #[test]
    fn promotion_rejects_topology_and_fidelity_mismatch() {
        let screening = dataset(
            "solver:reduced:v1",
            PcFanFidelityLevel::ReducedOrder,
            [1.0, 2.0, 3.0, 4.0],
            [true, true, true, true],
        );
        let mut validation = dataset(
            "solver:wall-cfd:v1",
            PcFanFidelityLevel::RotatingWallCfd,
            [1.0, 2.0, 3.0, 4.0],
            [true, true, true, true],
        );
        validation.observations.pop();
        assert!(matches!(
            evaluate_pc_fan_fidelity_promotion(
                &PcFanFidelityPromotionConfig::default(),
                &screening,
                &validation,
            ),
            Err(PcFanFidelityPromotionError::TooFewCandidates {
                dataset: "validation",
                ..
            }) | Err(PcFanFidelityPromotionError::CandidateTopologyMismatch { .. })
        ));

        let mut same_level = dataset(
            "solver:other-reduced:v1",
            PcFanFidelityLevel::ReducedOrder,
            [1.0, 2.0, 3.0, 4.0],
            [true, true, true, true],
        );
        same_level.observations[0].fidelity = PcFanFidelityLevel::RotatingWallCfd;
        assert!(matches!(
            evaluate_pc_fan_fidelity_promotion(
                &PcFanFidelityPromotionConfig::default(),
                &screening,
                &same_level,
            ),
            Err(PcFanFidelityPromotionError::FidelityMismatch { .. })
        ));
    }
}
