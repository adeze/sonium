//! Absorbing-slab, Chandrasekhar-H, and Marshak-wave references.

/// Formal solution for a homogeneous slab along one ordinate.
pub fn homogeneous_slab_intensity(
    optical_depth: f64,
    direction_cosine: f64,
    incident_intensity: f64,
    source_function: f64,
) -> f64 {
    assert!(optical_depth >= 0.0 && direction_cosine != 0.0);
    let attenuation = (-optical_depth / direction_cosine.abs()).exp();
    incident_intensity * attenuation + source_function * (1.0 - attenuation)
}

/// Chandrasekhar H-function for isotropic scattering, solved by fixed-point
/// iteration of its nonlinear integral equation using midpoint quadrature.
pub fn chandrasekhar_h(
    direction_cosine: f64,
    single_scattering_albedo: f64,
    angular_intervals: usize,
    iterations: usize,
) -> f64 {
    assert!((0.0..=1.0).contains(&direction_cosine));
    assert!((0.0..=1.0).contains(&single_scattering_albedo));
    if direction_cosine == 0.0 || single_scattering_albedo == 0.0 {
        return 1.0;
    }
    let n = angular_intervals.max(32);
    let step = 1.0 / n as f64;
    let ordinates: Vec<_> = (0..n).map(|index| (index as f64 + 0.5) * step).collect();
    let mut values = vec![1.0; n];
    for _ in 0..iterations.max(1) {
        let previous = values.clone();
        for (index, mu) in ordinates.iter().enumerate() {
            let integral = ordinates
                .iter()
                .zip(&previous)
                .map(|(mu_prime, value)| value / (mu + mu_prime))
                .sum::<f64>()
                * step;
            values[index] = 1.0 / (1.0 - 0.5 * single_scattering_albedo * mu * integral);
        }
    }
    let integral = ordinates
        .iter()
        .zip(&values)
        .map(|(mu_prime, value)| value / (direction_cosine + mu_prime))
        .sum::<f64>()
        * step;
    1.0 / (1.0 - 0.5 * single_scattering_albedo * direction_cosine * integral)
}

fn complementary_error_function(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26, maximum error about 1.5e-7.
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * x);
    let erf = 1.0
        - (((((1.061_405_429 * t - 1.453_152_027) * t) + 1.421_413_741) * t - 0.284_496_736) * t
            + 0.254_829_592)
            * t
            * (-x * x).exp();
    1.0 - sign * erf
}

/// Equilibrium-diffusion limit of the semi-infinite Marshak wave.
pub fn marshak_diffusion_energy(
    position: f64,
    time: f64,
    diffusion_coefficient: f64,
    boundary_energy: f64,
) -> f64 {
    assert!(position >= 0.0 && time >= 0.0 && diffusion_coefficient > 0.0);
    if position == 0.0 {
        return boundary_energy;
    }
    if time == 0.0 {
        return 0.0;
    }
    boundary_energy
        * complementary_error_function(position / (2.0 * (diffusion_coefficient * time).sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn absorbing_slab_matches_exponential_and_source_limits() {
        assert_eq!(homogeneous_slab_intensity(0.0, 1.0, 3.0, 5.0), 3.0);
        let thick = homogeneous_slab_intensity(100.0, 1.0, 3.0, 5.0);
        assert!((thick - 5.0).abs() < 1e-14);
        let thin = homogeneous_slab_intensity(0.2, 0.5, 3.0, 0.0);
        assert!((thin - 3.0 * (-0.4_f64).exp()).abs() < 1e-14);
    }

    #[test]
    fn conservative_chandrasekhar_h_matches_tabulated_endpoint() {
        assert_eq!(chandrasekhar_h(0.0, 1.0, 256, 100), 1.0);
        let endpoint = chandrasekhar_h(1.0, 1.0, 256, 5000);
        assert!((endpoint - 2.907_81).abs() < 2e-3, "H(1)={endpoint}");
    }

    #[test]
    fn marshak_diffusion_wave_preserves_boundary_and_similarity_scaling() {
        assert_eq!(marshak_diffusion_energy(0.0, 1.0, 2.0, 4.0), 4.0);
        let first = marshak_diffusion_energy(1.0, 1.0, 1.0, 4.0);
        let similar = marshak_diffusion_energy(2.0, 4.0, 1.0, 4.0);
        assert!((first - similar).abs() < 1e-14);
        assert!(first > 0.0 && first < 4.0);
    }
}
