//! Bounded low/high-fidelity rank-preservation evidence.
//!
//! This is a promotion gate over caller-supplied objective values. It does
//! not execute either fidelity, estimate a physical error field, or establish
//! that the high-fidelity values came from CFD or measured data.

use std::fmt;

/// Objective values for one candidate at two declared fidelity levels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FidelityPromotionSample {
    /// Stable candidate identifier used in result diagnostics.
    pub candidate_id: u64,
    /// Low-fidelity objective value.
    pub low_fidelity_score: f64,
    /// High-fidelity objective value.
    pub high_fidelity_score: f64,
}

/// Admission limits for [`evaluate_fidelity_promotion`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FidelityPromotionOptions {
    /// Maximum number of candidates in the ranking workset.
    pub max_candidates: usize,
    /// Whether smaller objective values are better.
    pub lower_is_better: bool,
    /// Minimum fraction of candidate pairs with equal ordering.
    pub minimum_pairwise_agreement: f64,
    /// Minimum Kendall-style rank correlation.
    pub minimum_rank_correlation: f64,
}

impl Default for FidelityPromotionOptions {
    fn default() -> Self {
        Self {
            max_candidates: 256,
            lower_is_better: true,
            minimum_pairwise_agreement: 0.95,
            minimum_rank_correlation: 0.9,
        }
    }
}

/// Candidate rank and score evidence.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FidelityPromotionPoint {
    /// Stable candidate identifier.
    pub candidate_id: u64,
    /// Original low-fidelity score.
    pub low_fidelity_score: f64,
    /// Original high-fidelity score.
    pub high_fidelity_score: f64,
    /// One-based low-fidelity rank.
    pub low_fidelity_rank: usize,
    /// One-based high-fidelity rank.
    pub high_fidelity_rank: usize,
    /// Whether this candidate occupies the same rank at both fidelities.
    pub rank_preserved: bool,
}

/// Checked fidelity-promotion rank result.
#[derive(Debug, Clone, PartialEq)]
pub struct FidelityPromotionResult {
    /// Per-candidate rank evidence in caller order.
    pub points: Vec<FidelityPromotionPoint>,
    /// Number of unordered candidate pairs compared.
    pub pair_count: usize,
    /// Number of pairs with identical low/high ordering.
    pub preserved_pair_count: usize,
    /// `preserved_pair_count / pair_count`.
    pub pairwise_agreement: f64,
    /// `2 * agreement - 1`, a Kendall-style ordering correlation.
    pub rank_correlation: f64,
    /// Whether configured promotion thresholds passed.
    pub passed: bool,
}

impl FidelityPromotionResult {
    /// Returns whether result fields are finite and structurally valid.
    pub fn is_finite(&self) -> bool {
        !self.points.is_empty()
            && self.pair_count > 0
            && self.preserved_pair_count <= self.pair_count
            && self.pairwise_agreement.is_finite()
            && (0.0..=1.0).contains(&self.pairwise_agreement)
            && self.rank_correlation.is_finite()
            && (-1.0..=1.0).contains(&self.rank_correlation)
            && self.points.iter().all(|point| {
                point.low_fidelity_score.is_finite()
                    && point.high_fidelity_score.is_finite()
                    && point.low_fidelity_rank > 0
                    && point.high_fidelity_rank > 0
            })
    }
}

/// Failure-closed errors from the promotion contract.
#[derive(Debug, Clone, PartialEq)]
pub enum FidelityPromotionError {
    /// Promotion options are malformed.
    InvalidOptions {
        /// Invalid option field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// No candidates were supplied.
    EmptyCandidates,
    /// Fewer than two candidates cannot establish a pairwise ranking.
    InsufficientCandidates { count: usize, minimum: usize },
    /// Candidate count exceeds the declared budget.
    TooManyCandidates { count: usize, maximum: usize },
    /// A candidate has invalid input.
    InvalidCandidate {
        /// Candidate index in caller order.
        index: usize,
        /// Invalid field.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// Candidate identifiers must be unique.
    DuplicateCandidateId { candidate_id: u64 },
    /// Strict rank evidence cannot be formed when a score is tied.
    TiedScore {
        /// Fidelity level with a tie.
        fidelity: &'static str,
        /// Tied objective score.
        score: f64,
    },
    /// Internal arithmetic produced a non-finite result.
    NonFiniteResult,
}

impl fmt::Display for FidelityPromotionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidOptions { field, requirement } => {
                write!(
                    formatter,
                    "invalid fidelity promotion option {field}: {requirement}"
                )
            }
            Self::EmptyCandidates => {
                formatter.write_str("fidelity promotion candidates must not be empty")
            }
            Self::InsufficientCandidates { count, minimum } => {
                write!(
                    formatter,
                    "fidelity promotion has {count} candidates, needs at least {minimum}"
                )
            }
            Self::TooManyCandidates { count, maximum } => {
                write!(
                    formatter,
                    "fidelity promotion candidate count {count} exceeds budget {maximum}"
                )
            }
            Self::InvalidCandidate {
                index,
                field,
                requirement,
            } => write!(
                formatter,
                "invalid fidelity promotion candidate {index} field {field}: {requirement}"
            ),
            Self::DuplicateCandidateId { candidate_id } => {
                write!(
                    formatter,
                    "fidelity promotion candidate id {candidate_id} is duplicated"
                )
            }
            Self::TiedScore { fidelity, score } => {
                write!(formatter, "{fidelity} fidelity score {score} is tied")
            }
            Self::NonFiniteResult => formatter.write_str("fidelity promotion result is non-finite"),
        }
    }
}

impl std::error::Error for FidelityPromotionError {}

/// Evaluate strict rank preservation between two caller-supplied fidelities.
pub fn evaluate_fidelity_promotion(
    candidates: &[FidelityPromotionSample],
    options: FidelityPromotionOptions,
) -> Result<FidelityPromotionResult, FidelityPromotionError> {
    validate_options(options)?;
    if candidates.is_empty() {
        return Err(FidelityPromotionError::EmptyCandidates);
    }
    if candidates.len() < 2 {
        return Err(FidelityPromotionError::InsufficientCandidates {
            count: candidates.len(),
            minimum: 2,
        });
    }
    if candidates.len() > options.max_candidates {
        return Err(FidelityPromotionError::TooManyCandidates {
            count: candidates.len(),
            maximum: options.max_candidates,
        });
    }
    for (index, candidate) in candidates.iter().enumerate() {
        validate_candidate(index, *candidate)?;
        if candidates[..index]
            .iter()
            .any(|previous| previous.candidate_id == candidate.candidate_id)
        {
            return Err(FidelityPromotionError::DuplicateCandidateId {
                candidate_id: candidate.candidate_id,
            });
        }
    }

    let low_scores = candidates
        .iter()
        .map(|candidate| candidate.low_fidelity_score)
        .collect::<Vec<_>>();
    let high_scores = candidates
        .iter()
        .map(|candidate| candidate.high_fidelity_score)
        .collect::<Vec<_>>();
    reject_ties(&low_scores, "low")?;
    reject_ties(&high_scores, "high")?;
    let low_ranks = ranks(&low_scores, options.lower_is_better);
    let high_ranks = ranks(&high_scores, options.lower_is_better);

    let mut preserved_pair_count = 0usize;
    let mut pair_count = 0usize;
    for first in 0..candidates.len() {
        for second in (first + 1)..candidates.len() {
            let low_order = low_ranks[first] < low_ranks[second];
            let high_order = high_ranks[first] < high_ranks[second];
            pair_count += 1;
            if low_order == high_order {
                preserved_pair_count += 1;
            }
        }
    }
    let pairwise_agreement = preserved_pair_count as f64 / pair_count as f64;
    let rank_correlation = 2.0 * pairwise_agreement - 1.0;
    if !pairwise_agreement.is_finite() || !rank_correlation.is_finite() {
        return Err(FidelityPromotionError::NonFiniteResult);
    }
    let points = candidates
        .iter()
        .enumerate()
        .map(|(index, candidate)| FidelityPromotionPoint {
            candidate_id: candidate.candidate_id,
            low_fidelity_score: candidate.low_fidelity_score,
            high_fidelity_score: candidate.high_fidelity_score,
            low_fidelity_rank: low_ranks[index],
            high_fidelity_rank: high_ranks[index],
            rank_preserved: low_ranks[index] == high_ranks[index],
        })
        .collect::<Vec<_>>();
    let result = FidelityPromotionResult {
        points,
        pair_count,
        preserved_pair_count,
        pairwise_agreement,
        rank_correlation,
        passed: pairwise_agreement >= options.minimum_pairwise_agreement
            && rank_correlation >= options.minimum_rank_correlation,
    };
    if !result.is_finite() {
        return Err(FidelityPromotionError::NonFiniteResult);
    }
    Ok(result)
}

fn validate_options(options: FidelityPromotionOptions) -> Result<(), FidelityPromotionError> {
    if options.max_candidates < 2 {
        return Err(FidelityPromotionError::InvalidOptions {
            field: "max_candidates",
            requirement: "at least 2",
        });
    }
    if !options.minimum_pairwise_agreement.is_finite()
        || !(0.0..=1.0).contains(&options.minimum_pairwise_agreement)
    {
        return Err(FidelityPromotionError::InvalidOptions {
            field: "minimum_pairwise_agreement",
            requirement: "finite and within [0, 1]",
        });
    }
    if !options.minimum_rank_correlation.is_finite()
        || !(-1.0..=1.0).contains(&options.minimum_rank_correlation)
    {
        return Err(FidelityPromotionError::InvalidOptions {
            field: "minimum_rank_correlation",
            requirement: "finite and within [-1, 1]",
        });
    }
    Ok(())
}

fn validate_candidate(
    index: usize,
    candidate: FidelityPromotionSample,
) -> Result<(), FidelityPromotionError> {
    if !candidate.low_fidelity_score.is_finite() {
        return Err(FidelityPromotionError::InvalidCandidate {
            index,
            field: "low_fidelity_score",
            requirement: "finite",
        });
    }
    if !candidate.high_fidelity_score.is_finite() {
        return Err(FidelityPromotionError::InvalidCandidate {
            index,
            field: "high_fidelity_score",
            requirement: "finite",
        });
    }
    Ok(())
}

fn reject_ties(scores: &[f64], fidelity: &'static str) -> Result<(), FidelityPromotionError> {
    for (index, score) in scores.iter().enumerate() {
        if scores[..index]
            .iter()
            .any(|previous| previous.total_cmp(score).is_eq())
        {
            return Err(FidelityPromotionError::TiedScore {
                fidelity,
                score: *score,
            });
        }
    }
    Ok(())
}

fn ranks(scores: &[f64], lower_is_better: bool) -> Vec<usize> {
    let mut order = (0..scores.len()).collect::<Vec<_>>();
    order.sort_by(|&left, &right| {
        let left_score = if lower_is_better {
            scores[left]
        } else {
            -scores[left]
        };
        let right_score = if lower_is_better {
            scores[right]
        } else {
            -scores[right]
        };
        left_score.total_cmp(&right_score)
    });
    let mut result = vec![0; scores.len()];
    for (rank, index) in order.into_iter().enumerate() {
        result[index] = rank + 1;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn preserved() -> [FidelityPromotionSample; 4] {
        [
            FidelityPromotionSample {
                candidate_id: 1,
                low_fidelity_score: 1.0,
                high_fidelity_score: 1.1,
            },
            FidelityPromotionSample {
                candidate_id: 2,
                low_fidelity_score: 2.0,
                high_fidelity_score: 2.2,
            },
            FidelityPromotionSample {
                candidate_id: 3,
                low_fidelity_score: 3.0,
                high_fidelity_score: 3.1,
            },
            FidelityPromotionSample {
                candidate_id: 4,
                low_fidelity_score: 4.0,
                high_fidelity_score: 4.4,
            },
        ]
    }

    #[test]
    fn exact_order_passes_promotion_gate() {
        let result = evaluate_fidelity_promotion(&preserved(), FidelityPromotionOptions::default())
            .expect("preserved order should be accepted");
        assert!(result.is_finite());
        assert_eq!(result.pair_count, 6);
        assert_eq!(result.preserved_pair_count, 6);
        assert_eq!(result.pairwise_agreement, 1.0);
        assert_eq!(result.rank_correlation, 1.0);
        assert!(result.points.iter().all(|point| point.rank_preserved));
        assert!(result.passed);
    }

    #[test]
    fn reversed_candidate_is_reported_and_rejected() {
        let mut candidates = preserved();
        candidates[0].high_fidelity_score = 5.0;
        let result = evaluate_fidelity_promotion(
            &candidates,
            FidelityPromotionOptions {
                minimum_pairwise_agreement: 1.0,
                minimum_rank_correlation: 1.0,
                ..FidelityPromotionOptions::default()
            },
        )
        .expect("finite rank inversion should return evidence");
        assert!(result.is_finite());
        assert!(result.preserved_pair_count < result.pair_count);
        assert!(!result.passed);
    }

    #[test]
    fn higher_is_better_order_is_supported() {
        let mut candidates = preserved();
        for candidate in &mut candidates {
            candidate.low_fidelity_score = -candidate.low_fidelity_score;
            candidate.high_fidelity_score = -candidate.high_fidelity_score;
        }
        let result = evaluate_fidelity_promotion(
            &candidates,
            FidelityPromotionOptions {
                lower_is_better: false,
                ..FidelityPromotionOptions::default()
            },
        )
        .expect("higher-is-better order should be accepted");
        assert!(result.passed);
        assert_eq!(result.rank_correlation, 1.0);
    }

    #[test]
    fn malformed_tied_and_budgeted_inputs_fail_closed() {
        assert!(matches!(
            evaluate_fidelity_promotion(&[], FidelityPromotionOptions::default()),
            Err(FidelityPromotionError::EmptyCandidates)
        ));
        assert!(matches!(
            evaluate_fidelity_promotion(&[preserved()[0]], FidelityPromotionOptions::default()),
            Err(FidelityPromotionError::InsufficientCandidates { .. })
        ));
        let mut duplicate = preserved();
        duplicate[1].candidate_id = duplicate[0].candidate_id;
        assert!(matches!(
            evaluate_fidelity_promotion(&duplicate, FidelityPromotionOptions::default()),
            Err(FidelityPromotionError::DuplicateCandidateId { .. })
        ));
        let mut tied = preserved();
        tied[1].low_fidelity_score = tied[0].low_fidelity_score;
        assert!(matches!(
            evaluate_fidelity_promotion(&tied, FidelityPromotionOptions::default()),
            Err(FidelityPromotionError::TiedScore {
                fidelity: "low",
                ..
            })
        ));
        let mut invalid = preserved();
        invalid[0].high_fidelity_score = f64::NAN;
        assert!(matches!(
            evaluate_fidelity_promotion(&invalid, FidelityPromotionOptions::default()),
            Err(FidelityPromotionError::InvalidCandidate {
                field: "high_fidelity_score",
                ..
            })
        ));
    }

    #[test]
    fn invalid_threshold_and_candidate_budget_fail_closed() {
        assert!(matches!(
            evaluate_fidelity_promotion(
                &preserved(),
                FidelityPromotionOptions {
                    minimum_pairwise_agreement: f64::NAN,
                    ..FidelityPromotionOptions::default()
                }
            ),
            Err(FidelityPromotionError::InvalidOptions {
                field: "minimum_pairwise_agreement",
                ..
            })
        ));
        assert!(matches!(
            evaluate_fidelity_promotion(
                &preserved(),
                FidelityPromotionOptions {
                    max_candidates: 2,
                    ..FidelityPromotionOptions::default()
                }
            ),
            Err(FidelityPromotionError::TooManyCandidates { .. })
        ));
    }
}
