use rand::Rng;
use std::f64::consts::PI;

/// Wrapper Local Search (WLS) strategy for local refinement
/// Uses Cauchy distribution to perturb selected dimensions
#[allow(dead_code)]
pub(crate) fn apply_wls<R: Rng + ?Sized>(
    x: &[f64],
    lower: &[f64],
    upper: &[f64],
    scale: f64,
    rng: &mut R,
) -> Vec<f64> {
    let mut result = x.to_vec();
    apply_wls_in_place(&mut result, lower, upper, scale, rng);
    result
}

/// In-place variant of [`apply_wls`].
pub(crate) fn apply_wls_in_place<R: Rng + ?Sized>(
    x: &mut [f64],
    lower: &[f64],
    upper: &[f64],
    scale: f64,
    rng: &mut R,
) {
    let n_dims = x.len();

    // Generate random wrapper mask - selects which dimensions to perturb
    let n_selected = rng.random_range(1..=n_dims.max(1));
    let mut dimensions: Vec<usize> = (0..n_dims).collect();
    use rand::seq::SliceRandom;
    dimensions.shuffle(rng);
    let selected_dims = &dimensions[0..n_selected];

    // Apply Cauchy perturbations to selected dimensions. Heavy tails give WLS
    // occasional longer local jumps while clipping keeps candidates feasible.
    for &dim in selected_dims {
        let u = rng.random::<f64>().clamp(f64::EPSILON, 1.0 - f64::EPSILON);
        let perturbation = (PI * (u - 0.5)).tan() * scale;
        let new_val = x[dim] + perturbation;
        // Clip to bounds
        x[dim] = new_val.max(lower[dim]).min(upper[dim]);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    #[test]
    fn wls_preserves_shape_and_bounds() {
        let x = vec![0.0, 0.5, -0.5];
        let lower = vec![-1.0, -1.0, -1.0];
        let upper = vec![1.0, 1.0, 1.0];
        let mut rng = StdRng::seed_from_u64(7);

        let y = apply_wls(&x, &lower, &upper, 10.0, &mut rng);

        assert_eq!(y.len(), x.len());
        for i in 0..y.len() {
            assert!(
                y[i] >= lower[i] && y[i] <= upper[i],
                "dimension {i} escaped bounds: {} not in [{}, {}]",
                y[i],
                lower[i],
                upper[i]
            );
        }
    }

    #[test]
    fn zero_scale_wls_is_identity() {
        let x = vec![0.0, 0.5, -0.5];
        let lower = vec![-1.0, -1.0, -1.0];
        let upper = vec![1.0, 1.0, 1.0];
        let mut rng = StdRng::seed_from_u64(11);

        let y = apply_wls(&x, &lower, &upper, 0.0, &mut rng);

        assert_eq!(y, x);
    }
}
