/// Create bounds matrix for optimization (2 x n matrix)
/// bounds[[0, i]] = lower bound, bounds[[1, i]] = upper bound
pub fn create_bounds(n: usize, lower: f64, upper: f64) -> Vec<(f64, f64)> {
    vec![(lower, upper); n]
}
