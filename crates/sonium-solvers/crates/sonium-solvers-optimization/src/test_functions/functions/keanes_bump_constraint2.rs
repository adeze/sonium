//! Keanes Bump Constraint2 test function

/// Second constraint for Keane's bump function: sum(x_i) <= 7.5*n
/// Returns violation amount (0 if satisfied, positive if violated)
pub fn keanes_bump_constraint2(x: &[f64]) -> f64 {
    let sum: f64 = x.iter().sum();
    let limit = 7.5 * x.len() as f64;
    sum - limit // Constraint: sum <= limit, so violation is sum - limit
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reports_negative_or_zero_when_sum_is_within_limit() {
        let x = vec![1.0, 2.0, 3.0, 4.0];
        assert!(keanes_bump_constraint2(&x) <= 0.0);
    }

    #[test]
    fn reports_positive_violation_when_sum_exceeds_limit() {
        let x = vec![10.0, 10.0];
        assert!(keanes_bump_constraint2(&x) > 0.0);
    }
}
