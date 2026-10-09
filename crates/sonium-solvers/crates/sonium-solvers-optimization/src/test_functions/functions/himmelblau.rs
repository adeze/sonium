//! Himmelblau test function

/// Himmelblau function - 2D multimodal
/// Global minima: f(x) = 0 at x = (3, 2), (-2.805118, 3.131312), (-3.779310, -3.283186), (3.584428, -1.848126)
/// Bounds: x_i in [-5, 5]
pub fn himmelblau(x: &[f64]) -> f64 {
    assert!(
        x.len() == 2,
        "himmelblau requires 2 dimensions, got {}",
        x.len()
    );
    let x1 = x[0];
    let x2 = x[1];
    (x1.powi(2) + x2 - 11.0).powi(2) + (x1 + x2.powi(2) - 7.0).powi(2)
}
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_himmelblau_known_properties() {
        use crate::test_functions::{FunctionMetadata, get_function_metadata};

        // Get metadata for this function
        let metadata = get_function_metadata();
        let meta = metadata
            .get("himmelblau")
            .expect("Function himmelblau should have metadata");

        // Test 1: Verify global minima are within bounds
        for (minimum_coords, expected_value) in &meta.global_minima {
            assert!(
                minimum_coords.len() >= meta.bounds.len() || meta.bounds.len() == 1,
                "Global minimum coordinates should match bounds dimensions"
            );

            for (i, &coord) in minimum_coords.iter().enumerate() {
                if i < meta.bounds.len() {
                    let (lower, upper) = meta.bounds[i];
                    assert!(
                        coord >= lower && coord <= upper,
                        "Global minimum coordinate {} = {} should be within bounds [{} {}]",
                        i,
                        coord,
                        lower,
                        upper
                    );
                }
            }
        }

        // Test 2: Verify function evaluates to expected values at global minima
        let tolerance = 1e-6; // Reasonable tolerance for numerical precision
        for (minimum_coords, expected_value) in &meta.global_minima {
            let x = minimum_coords.clone();
            let actual_value = himmelblau(&x);

            let error = (actual_value - expected_value).abs();
            assert!(
                error <= tolerance,
                "Function value at global minimum {:?} should be {}, got {}, error: {}",
                minimum_coords,
                expected_value,
                actual_value,
                error
            );
        }

        // Test 3: Basic function properties
        if !meta.global_minima.is_empty() {
            let (first_minimum, _) = &meta.global_minima[0];
            let x = first_minimum.clone();
            let result = himmelblau(&x);

            assert!(
                result.is_finite(),
                "Function should return finite values at global minimum"
            );
            assert!(
                !result.is_nan(),
                "Function should not return NaN at global minimum"
            );
        }
    }
}
