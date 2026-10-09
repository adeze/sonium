//! Wolfram Beam–Spring–Mass linear mechanical-stage verification.
//!
//! Wolfram reduces a 0.5 x 0.1 x 0.1 m nearly incompressible solid cantilever
//! to a spring. Its published linear stiffness is 256.322 N/m and its mass is
//! 5.5 kg. Sonium uses the corresponding Euler–Bernoulli beam element; the
//! small model-form difference is retained explicitly in the acceptance gate.

use sonium_fem::physics::solid::beam::{BeamElement2D, BeamSection};
use std::f64::consts::PI;

const LENGTH_M: f64 = 0.5;
const WIDTH_M: f64 = 0.1;
const HEIGHT_M: f64 = 0.1;
const SHEAR_MODULUS_PA: f64 = 441_000.0;
const POISSON_RATIO: f64 = 0.49;
const DENSITY_KG_M3: f64 = 1100.0;
const WOLFRAM_STIFFNESS_N_M: f64 = 256.322;
const WOLFRAM_MASS_KG: f64 = 5.5;

fn solve_3x3(mut matrix: [[f64; 3]; 3], mut rhs: [f64; 3]) -> [f64; 3] {
    for pivot in 0..3 {
        let best = (pivot..3)
            .max_by(|&left, &right| {
                matrix[left][pivot]
                    .abs()
                    .total_cmp(&matrix[right][pivot].abs())
            })
            .expect("pivot row");
        matrix.swap(pivot, best);
        rhs.swap(pivot, best);
        assert!(matrix[pivot][pivot].abs() > 1.0e-14);
        let pivot_row = matrix[pivot];
        for row in pivot + 1..3 {
            let factor = matrix[row][pivot] / matrix[pivot][pivot];
            for (entry, pivot_entry) in matrix[row][pivot..].iter_mut().zip(&pivot_row[pivot..]) {
                *entry -= factor * pivot_entry;
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    let mut solution = [0.0; 3];
    for row in (0..3).rev() {
        let known = ((row + 1)..3)
            .map(|column| matrix[row][column] * solution[column])
            .sum::<f64>();
        solution[row] = (rhs[row] - known) / matrix[row][row];
    }
    solution
}

#[test]
fn sonium_beam_reproduces_wolfram_linear_spring_and_mass() {
    let youngs_modulus = 2.0 * SHEAR_MODULUS_PA * (1.0 + POISSON_RATIO);
    let section = BeamSection::rectangular(youngs_modulus, WIDTH_M, HEIGHT_M, DENSITY_KG_M3);
    let element = BeamElement2D::new([0, 1], 0.0, 0.0, LENGTH_M, 0.0, section);
    let stiffness = element.stiffness_global();
    let free_matrix = std::array::from_fn(|row| {
        std::array::from_fn(|column| stiffness[(row + 3) * 6 + column + 3])
    });

    let forces = [10.0, 25.0, 50.0, 75.0, 100.0];
    let inferred_stiffness = forces.map(|force| {
        let displacement = solve_3x3(free_matrix, [0.0, -force, 0.0]);
        force / displacement[1].abs()
    });
    let mean_stiffness = inferred_stiffness.iter().sum::<f64>() / forces.len() as f64;
    assert!(
        inferred_stiffness
            .iter()
            .all(|value| (value - mean_stiffness).abs() < 1.0e-9)
    );
    assert!(
        (mean_stiffness - WOLFRAM_STIFFNESS_N_M).abs() / WOLFRAM_STIFFNESS_N_M < 0.03,
        "Sonium beam stiffness {mean_stiffness:.6} N/m, Wolfram {WOLFRAM_STIFFNESS_N_M:.6} N/m"
    );

    let mass = section.mass_per_length() * LENGTH_M;
    assert!((mass - WOLFRAM_MASS_KG).abs() < 1.0e-12);
    let wolfram_natural_frequency = (WOLFRAM_STIFFNESS_N_M / WOLFRAM_MASS_KG).sqrt() / (2.0 * PI);
    let sonium_natural_frequency = (mean_stiffness / mass).sqrt() / (2.0 * PI);
    assert!(
        (sonium_natural_frequency - wolfram_natural_frequency) / wolfram_natural_frequency < 0.02
    );
}
