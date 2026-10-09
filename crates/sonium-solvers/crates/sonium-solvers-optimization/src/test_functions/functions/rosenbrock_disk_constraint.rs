//! Rosenbrock Disk Constraint test function

/// Rosenbrock disk constraint: x^2 + y^2 <= 2
pub fn rosenbrock_disk_constraint(x: &[f64]) -> f64 {
    assert!(
        x.len() == 2,
        "rosenbrock_disk_constraint requires 2 dimensions, got {}",
        x.len()
    );
    x[0].powi(2) + x[1].powi(2) - 2.0
}
