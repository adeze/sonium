//! Validation helpers for integration tests

use num_traits::Float;

/// Estimate the mean convergence order for a sequence of errors.
///
/// This convenience wrapper assumes each refinement halves the characteristic
/// element size. Use [`convergence_rate_with_sizes`] when actual mesh sizes or
/// time steps are available.
pub fn convergence_rate(errors: &[f64]) -> Option<f64> {
    if errors.len() < 2 {
        return None;
    }
    let element_sizes: Vec<f64> = (0..errors.len()).map(|i| 0.5_f64.powi(i as i32)).collect();
    convergence_rate_with_sizes(errors, &element_sizes)
}

/// Estimate the mean convergence order from errors and characteristic sizes.
///
/// For two successive refinements this computes
/// `ln(e_i / e_{i+1}) / ln(h_i / h_{i+1})`, then averages all valid adjacent
/// rates. Invalid, non-finite, or non-monotone sequences are rejected rather
/// than partially skipped because dropping a bad refinement can hide
/// divergence.
pub fn convergence_rate_with_sizes(errors: &[f64], element_sizes: &[f64]) -> Option<f64> {
    if errors.len() < 2 {
        return None;
    }
    if errors.len() != element_sizes.len() {
        return None;
    }

    if errors
        .iter()
        .any(|value| !value.is_finite() || *value <= 0.0)
        || element_sizes
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.0)
        || errors.windows(2).any(|pair| pair[1] >= pair[0])
        || element_sizes.windows(2).any(|pair| pair[1] >= pair[0])
    {
        return None;
    }

    let n = errors.len();
    let mut rates = Vec::with_capacity(n - 1);
    for i in 0..n - 1 {
        let rate =
            (errors[i] / errors[i + 1]).ln() / (element_sizes[i] / element_sizes[i + 1]).ln();
        if !rate.is_finite() || rate <= 0.0 {
            return None;
        }
        rates.push(rate);
    }

    Some(rates.iter().sum::<f64>() / rates.len() as f64)
}

pub fn max_relative_error<T: Float>(a: &[T], b: &[T]) -> T {
    if a.len() != b.len() || a.is_empty() {
        return T::infinity();
    }
    let mut max_err = T::zero();
    for i in 0..a.len() {
        if !a[i].is_finite() || !b[i].is_finite() {
            return T::infinity();
        }
        let diff = (a[i] - b[i]).abs();
        let rel = if b[i].abs() > T::epsilon() {
            diff / b[i].abs()
        } else if diff <= T::epsilon() {
            T::zero()
        } else {
            T::infinity()
        };
        if !rel.is_finite() {
            return T::infinity();
        }
        if rel > max_err {
            max_err = rel;
        }
    }
    max_err
}

pub fn mean_relative_error<T: Float>(a: &[T], b: &[T]) -> T {
    if a.len() != b.len() || a.is_empty() {
        return T::infinity();
    }
    let n = a.len();
    let mut sum = T::zero();
    for i in 0..n {
        if !a[i].is_finite() || !b[i].is_finite() {
            return T::infinity();
        }
        let diff = (a[i] - b[i]).abs();
        let rel = if b[i].abs() > T::epsilon() {
            diff / b[i].abs()
        } else if diff <= T::epsilon() {
            T::zero()
        } else {
            T::infinity()
        };
        if !rel.is_finite() {
            return T::infinity();
        }
        sum = sum + rel;
        if !sum.is_finite() {
            return T::infinity();
        }
    }
    sum / T::from(n).unwrap()
}

pub fn energy_error<T: Float>(u: &[T], v: &[T]) -> T {
    if u.len() != v.len() || u.is_empty() {
        return T::infinity();
    }
    let n = u.len();
    let mut num = T::zero();
    let mut den = T::zero();
    for i in 0..n {
        if !u[i].is_finite() || !v[i].is_finite() {
            return T::infinity();
        }
        let diff = u[i] - v[i];
        num = num + diff * diff;
        den = den + v[i] * v[i];
        if !num.is_finite() || !den.is_finite() {
            return T::infinity();
        }
    }
    if den > T::epsilon() {
        (num / den).sqrt()
    } else if num <= T::epsilon() {
        T::zero()
    } else {
        T::infinity()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn convergence_rate_uses_halving_by_default() {
        let errors = [1.0, 0.25, 0.0625, 0.015625];
        let rate = convergence_rate(&errors).expect("rate");
        assert!((rate - 2.0).abs() < 1e-12);
    }

    #[test]
    fn convergence_rate_uses_actual_element_sizes() {
        let h: Vec<f64> = vec![0.4, 0.2, 0.1, 0.05];
        let errors: Vec<f64> = h.iter().map(|&hi| 3.0 * hi.powi(3)).collect();

        let rate = convergence_rate_with_sizes(&errors, &h).expect("rate");
        assert!((rate - 3.0).abs() < 1e-12);
    }

    #[test]
    fn convergence_rate_rejects_mismatched_sizes() {
        assert_eq!(convergence_rate_with_sizes(&[1.0, 0.5], &[1.0]), None);
    }

    #[test]
    fn convergence_rate_skips_invalid_pairs() {
        let rate = convergence_rate_with_sizes(&[1.0, 0.0, 0.25], &[1.0, 0.5, 0.25]);
        assert_eq!(rate, None);
    }

    #[test]
    fn convergence_rate_rejects_nonfinite_sequences() {
        assert_eq!(
            convergence_rate_with_sizes(&[1.0, f64::NAN], &[1.0, 0.5]),
            None
        );
        assert_eq!(
            convergence_rate_with_sizes(&[1.0, 0.5], &[1.0, f64::INFINITY]),
            None
        );
    }

    #[test]
    fn convergence_rate_rejects_nonmonotone_refinement() {
        assert_eq!(convergence_rate_with_sizes(&[1.0, 1.1], &[1.0, 0.5]), None);
        assert_eq!(convergence_rate_with_sizes(&[1.0, 0.5], &[1.0, 2.0]), None);
    }

    #[test]
    fn pointwise_relative_errors_fail_closed() {
        assert!(max_relative_error(&[1.0], &[0.0]).is_infinite());
        assert!(mean_relative_error(&[1.0], &[0.0]).is_infinite());
        assert!(max_relative_error(&[1.0], &[1.0, 2.0]).is_infinite());
        assert!(mean_relative_error(&[1.0], &[1.0, 2.0]).is_infinite());
        assert!(max_relative_error(&[f64::NAN], &[1.0]).is_infinite());
        assert!(mean_relative_error(&[1.0], &[f64::NAN]).is_infinite());
    }

    #[test]
    fn energy_error_rejects_invalid_inputs() {
        assert!(energy_error(&[1.0], &[1.0, 2.0]).is_infinite());
        assert!(energy_error(&[f64::NAN], &[1.0]).is_infinite());
    }
}
