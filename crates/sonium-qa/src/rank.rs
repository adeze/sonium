//! Rank-correlation statistics for multi-fidelity validation.
//!
//! These helpers back the rank-preservation benchmarks (review item B6,
//! methodology §11.3): a cheap simulation policy is judged by whether it
//! preserves the *ranking* of candidate designs produced by the reference
//! policy, not by pointwise objective error.
//!
//! - [`spearman_rho`] — Pearson correlation of average ranks (ties handled).
//! - [`kendall_tau_b`] — Kendall's tau-b with explicit tie correction.
//! - [`top_k_set`] — indices of the `k` smallest values (minimization), used
//!   for false-promotion / false-rejection counts.

/// Average ranks (1..=n) of `values`, ascending. Tied values share the mean
/// of the ranks they span.
fn average_ranks(values: &[f64]) -> Vec<f64> {
    let n = values.len();
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&i, &j| {
        values[i]
            .partial_cmp(&values[j])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut ranks = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i + 1;
        while j < n && values[order[j]] == values[order[i]] {
            j += 1;
        }
        // Ranks i+1 ..= j (1-based); ties take the average.
        let mean_rank = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[order[k]] = mean_rank;
        }
        i = j;
    }
    ranks
}

/// Spearman's rank correlation coefficient ρ between `a` and `b`.
///
/// Computed as the Pearson correlation of the average ranks, which reduces
/// to the textbook `1 - 6Σd²/(n(n²-1))` when there are no ties. Returns
/// `NaN` for empty/mismatched inputs, fewer than two samples, or a constant
/// input (zero rank variance), or non-finite objective value.
pub fn spearman_rho(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 || a.iter().chain(b).any(|value| !value.is_finite()) {
        return f64::NAN;
    }
    let ra = average_ranks(a);
    let rb = average_ranks(b);
    let n = a.len() as f64;
    let mean_a = ra.iter().sum::<f64>() / n;
    let mean_b = rb.iter().sum::<f64>() / n;
    let mut cov = 0.0;
    let mut var_a = 0.0;
    let mut var_b = 0.0;
    for i in 0..a.len() {
        let da = ra[i] - mean_a;
        let db = rb[i] - mean_b;
        cov += da * db;
        var_a += da * da;
        var_b += db * db;
    }
    if var_a <= 0.0 || var_b <= 0.0 {
        return f64::NAN;
    }
    cov / (var_a * var_b).sqrt()
}

/// Kendall's rank correlation coefficient τ-b between `a` and `b`.
///
/// τ-b corrects for ties in both inputs:
/// `τ = (C - D) / sqrt((C + D + T_a) * (C + D + T_b))` where `T_a`/`T_b` are
/// the numbers of pairs tied only in `a`/`b`. Returns `NaN` for
/// empty/mismatched inputs, fewer than two samples, a fully tied input, or a
/// non-finite objective value.
pub fn kendall_tau_b(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() || a.len() < 2 || a.iter().chain(b).any(|value| !value.is_finite()) {
        return f64::NAN;
    }
    let n = a.len();
    let mut concordant = 0i64;
    let mut discordant = 0i64;
    let mut tied_a_only = 0i64;
    let mut tied_b_only = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let da = a[i].partial_cmp(&a[j]).unwrap_or(std::cmp::Ordering::Equal);
            let db = b[i].partial_cmp(&b[j]).unwrap_or(std::cmp::Ordering::Equal);
            use std::cmp::Ordering::*;
            match (da, db) {
                (Equal, Equal) => {} // tied in both: excluded from every count
                (Equal, _) => tied_a_only += 1,
                (_, Equal) => tied_b_only += 1,
                (Less, Less) | (Greater, Greater) => concordant += 1,
                _ => discordant += 1,
            }
        }
    }
    let pairs = concordant + discordant;
    let denom = ((pairs + tied_a_only) * (pairs + tied_b_only)) as f64;
    if denom <= 0.0 {
        return f64::NAN;
    }
    (concordant - discordant) as f64 / denom.sqrt()
}

/// Indices of the `k` smallest values (ascending objective = better), with
/// ties at the cut broken by lower index for determinism.
///
/// For release gates prefer [`checked_top_k_set`], which rejects non-finite
/// objectives instead of relying on `partial_cmp`'s unordered-value fallback.
pub fn top_k_set(values: &[f64], k: usize) -> Vec<usize> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&i, &j| {
        values[i]
            .partial_cmp(&values[j])
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(i.cmp(&j))
    });
    order.truncate(k.min(values.len()));
    order.sort_unstable();
    order
}

/// Fail-closed [`top_k_set`] for release-facing objective rankings.
///
/// Returns `None` if an objective is NaN or infinite; an invalid candidate
/// must not be silently promoted or rejected through a deterministic tie.
pub fn checked_top_k_set(values: &[f64], k: usize) -> Option<Vec<usize>> {
    values
        .iter()
        .all(|value| value.is_finite())
        .then(|| top_k_set(values, k))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spearman_identical_is_one() {
        let a = [3.0, 1.0, 2.0, 5.0, 4.0];
        assert!((spearman_rho(&a, &a) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn spearman_reversed_is_minus_one() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0];
        let b = [5.0, 4.0, 3.0, 2.0, 1.0];
        assert!((spearman_rho(&a, &b) + 1.0).abs() < 1e-15);
    }

    #[test]
    fn spearman_wikipedia_example_with_ties() {
        // Classic example: ranks contain ties (86→8, 97→5, 103→7, 106→9 in x;
        // several tied pairs in y), expected ρ ≈ −0.1757575...
        let x = [
            106.0, 86.0, 100.0, 101.0, 99.0, 103.0, 97.0, 113.0, 112.0, 110.0,
        ];
        let y = [7.0, 0.0, 27.0, 50.0, 28.0, 29.0, 20.0, 12.0, 6.0, 17.0];
        let rho = spearman_rho(&x, &y);
        assert!(
            (rho - (-0.175_757_575_757_575_75)).abs() < 1e-12,
            "rho = {rho}"
        );
    }

    #[test]
    fn spearman_constant_input_is_nan() {
        let a = [2.0, 2.0, 2.0];
        let b = [1.0, 2.0, 3.0];
        assert!(spearman_rho(&a, &b).is_nan());
    }

    #[test]
    fn rank_correlations_reject_nonfinite_objectives() {
        let finite = [1.0, 2.0, 3.0];
        for nonfinite in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert!(spearman_rho(&[1.0, nonfinite, 3.0], &finite).is_nan());
            assert!(spearman_rho(&finite, &[1.0, nonfinite, 3.0]).is_nan());
            assert!(kendall_tau_b(&[1.0, nonfinite, 3.0], &finite).is_nan());
            assert!(kendall_tau_b(&finite, &[1.0, nonfinite, 3.0]).is_nan());
        }
    }

    #[test]
    fn kendall_identical_is_one() {
        let a = [3.0, 1.0, 2.0, 5.0, 4.0];
        assert!((kendall_tau_b(&a, &a) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn kendall_reversed_is_minus_one() {
        let a = [1.0, 2.0, 3.0, 4.0];
        let b = [4.0, 3.0, 2.0, 1.0];
        assert!((kendall_tau_b(&a, &b) + 1.0).abs() < 1e-15);
    }

    #[test]
    fn kendall_wikipedia_tau_b_example_with_ties() {
        // x = [12, 2, 1, 12, 2], y = [1, 4, 7, 1, 0] → τ-b ≈ −0.47140452079103173
        let x = [12.0, 2.0, 1.0, 12.0, 2.0];
        let y = [1.0, 4.0, 7.0, 1.0, 0.0];
        let tau = kendall_tau_b(&x, &y);
        assert!(
            (tau - (-0.471_404_520_791_031_73)).abs() < 1e-12,
            "tau = {tau}"
        );
    }

    #[test]
    fn kendall_perfect_agreement_with_ties_is_one() {
        // Ties in the same positions must still give τ-b = 1.
        let a = [1.0, 1.0, 2.0, 3.0, 3.0];
        let b = [5.0, 5.0, 6.0, 7.0, 7.0];
        assert!((kendall_tau_b(&a, &b) - 1.0).abs() < 1e-15);
    }

    #[test]
    fn top_k_selects_smallest() {
        let values = [0.5, 0.1, 0.9, 0.2, 0.7];
        assert_eq!(top_k_set(&values, 2), vec![1, 3]);
        assert_eq!(top_k_set(&values, 0), Vec::<usize>::new());
        assert_eq!(top_k_set(&values, 10).len(), 5);
    }

    #[test]
    fn checked_top_k_rejects_nonfinite_objectives() {
        assert_eq!(checked_top_k_set(&[0.5, 0.1, 0.9], 2), Some(vec![0, 1]));
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(checked_top_k_set(&[0.5, value], 1), None);
        }
    }
}
