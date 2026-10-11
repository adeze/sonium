//! Wolfram-derived goldens for the SPH core (lane 5B W-tier).
//!
//! `fixtures/sph_kernel_consistency.wls` (50-digit precision) independently
//! derives cubic B-spline kernel values, the lattice partition-of-unity sum,
//! and the difference-form gradient of a linear field; this test replays the
//! checked-in `sph_kernel_consistency_golden.json` through the production
//! `sph` kernel path. `fixtures/sph_waves.wls` derives one explicit step of
//! an 8-particle compression-wave line (deck viscosity `qa = 2`, `qb = 1`,
//! linear polynomial EOS); the second test replays `sph_waves_golden.json`
//! through the production operator path. Regenerate with
//! `wolframscript -script fixtures/sph_kernel_consistency.wls` and
//! `wolframscript -script fixtures/sph_waves.wls` (checked-in JSONs were
//! cross-computed with mpmath; see each golden's `computed_with`).

use serde::Deserialize;
use sonium_fem::physics::fluid::sph::{
    NeighborSearch, Particle, PolynomialEos, SUPPORT_RADIUS_FACTOR, continuity_rate, energy_rate,
    kernel_gradient, kernel_gradient_factor, kernel_value, momentum_acceleration,
    summation_density,
};

const KERNEL_GOLDEN: &str = include_str!("../fixtures/sph_kernel_consistency_golden.json");
const WAVES_GOLDEN: &str = include_str!("../fixtures/sph_waves_golden.json");

#[derive(Debug, Deserialize)]
struct KernelGolden {
    schema_version: u32,
    case_id: String,
    smoothing_length: f64,
    dim: usize,
    kernel_values: Vec<KernelRow>,
    partition_of_unity: Partition,
    gradient_accuracy: GradientCase,
    tolerances: KernelTolerances,
}

#[derive(Debug, Deserialize)]
struct KernelRow {
    q: f64,
    #[serde(rename = "W")]
    w: f64,
    grad_factor: f64,
}

#[derive(Debug, Deserialize)]
struct Partition {
    lattice_spacing: f64,
    lattice_half_width: i32,
    smoothing_length: f64,
    sum: f64,
}

#[derive(Debug, Deserialize)]
struct GradientCase {
    sph_gradient: [f64; 3],
    exact_gradient: [f64; 3],
}

#[derive(Debug, Deserialize)]
struct KernelTolerances {
    #[serde(rename = "abs")]
    abs: f64,
    partition_vs_one: f64,
    gradient_vs_exact: f64,
}

#[derive(Debug, Deserialize)]
struct WavesGolden {
    schema_version: u32,
    case_id: String,
    dim: usize,
    smoothing_length: f64,
    spacing: f64,
    mass: f64,
    dt: f64,
    qa: f64,
    qb: f64,
    eos: EosSpec,
    initial_velocity_left: f64,
    count: usize,
    neighbors: Vec<Vec<usize>>,
    density: Vec<f64>,
    pressure: Vec<f64>,
    sound_speed: Vec<f64>,
    acceleration: Vec<[f64; 3]>,
    energy_rate: Vec<f64>,
    density_rate: Vec<f64>,
    velocity_after_step: Vec<[f64; 3]>,
    position_after_step: Vec<[f64; 3]>,
    energy_after_step: Vec<f64>,
    tolerances: WavesTolerances,
}

#[derive(Debug, Deserialize)]
struct EosSpec {
    c1: f64,
    c2: f64,
    c3: f64,
    rho0: f64,
}

#[derive(Debug, Deserialize)]
struct WavesTolerances {
    #[serde(rename = "abs")]
    abs: f64,
}

fn lattice_points(half_width: i32, spacing: f64) -> Vec<[f64; 3]> {
    let mut points = Vec::new();
    for i in -half_width..=half_width {
        for j in -half_width..=half_width {
            for k in -half_width..=half_width {
                points.push([i as f64 * spacing, j as f64 * spacing, k as f64 * spacing]);
            }
        }
    }
    points
}

#[test]
fn wolfram_sph_kernel_consistency_replays() {
    let golden: KernelGolden = serde_json::from_str(KERNEL_GOLDEN).expect("parse golden");
    assert_eq!(golden.schema_version, 1);
    assert_eq!(golden.case_id, "fem.sph-kernel-consistency.v1");

    // Case A: pointwise kernel values and gradient factors.
    for row in &golden.kernel_values {
        let radius = row.q * golden.smoothing_length;
        let got_w = kernel_value(radius, golden.smoothing_length, golden.dim).unwrap();
        let got_g = kernel_gradient_factor(radius, golden.smoothing_length, golden.dim).unwrap();
        assert!(
            (got_w - row.w).abs() < golden.tolerances.abs,
            "q={}: W got {got_w}, want {}",
            row.q,
            row.w
        );
        assert!(
            (got_g - row.grad_factor).abs() < golden.tolerances.abs,
            "q={}: grad factor got {got_g}, want {}",
            row.q,
            row.grad_factor
        );
    }

    // Case B: partition of unity on the cubic lattice (unit volumes).
    let points = lattice_points(
        golden.partition_of_unity.lattice_half_width,
        golden.partition_of_unity.lattice_spacing,
    );
    let center = points.len() / 2;
    assert_eq!(points[center], [0.0, 0.0, 0.0]);
    let support = SUPPORT_RADIUS_FACTOR * golden.partition_of_unity.smoothing_length;
    let mut sum = 0.0;
    for (index, point) in points.iter().enumerate() {
        let radius = point[0].hypot(point[1]).hypot(point[2]);
        if index == center || radius < support {
            sum += kernel_value(
                radius,
                golden.partition_of_unity.smoothing_length,
                golden.dim,
            )
            .unwrap();
        }
    }
    assert!(
        (sum - golden.partition_of_unity.sum).abs() < golden.tolerances.abs,
        "PoU sum got {sum}, want {}",
        golden.partition_of_unity.sum
    );
    assert!(
        (sum - 1.0).abs() < golden.tolerances.partition_vs_one,
        "PoU sum {sum} too far from one"
    );
    // The grid search must find exactly the cutoff neighbor set.
    let search = NeighborSearch::build(&points, support).unwrap();
    let cutoff: Vec<usize> = points
        .iter()
        .enumerate()
        .filter(|(index, point)| {
            *index != center && point[0].hypot(point[1]).hypot(point[2]) < support
        })
        .map(|(index, _)| index)
        .collect();
    assert_eq!(search.neighbors(center), cutoff);

    // Case C: difference-form gradient of 2x - 3y + z + 5 at the center.
    let field = |point: &[f64; 3]| 2.0 * point[0] - 3.0 * point[1] + point[2] + 5.0;
    let center_value = field(&points[center]);
    let mut gradient = [0.0; 3];
    for (index, point) in points.iter().enumerate() {
        if index == center {
            continue;
        }
        let radius = point[0].hypot(point[1]).hypot(point[2]);
        if radius >= support {
            continue;
        }
        let offset = [-point[0], -point[1], -point[2]];
        let grad = kernel_gradient(
            offset,
            golden.partition_of_unity.smoothing_length,
            golden.dim,
        )
        .unwrap();
        let weight = field(point) - center_value;
        gradient[0] += weight * grad[0];
        gradient[1] += weight * grad[1];
        gradient[2] += weight * grad[2];
    }
    let grad_pairs = gradient
        .iter()
        .zip(golden.gradient_accuracy.sph_gradient.iter())
        .zip(golden.gradient_accuracy.exact_gradient.iter());
    for (axis, ((&got, &want), &exact)) in grad_pairs.enumerate() {
        assert!(
            (got - want).abs() < golden.tolerances.abs,
            "axis {axis}: got {got}, want {want}"
        );
        assert!(
            (got - exact).abs() < golden.tolerances.gradient_vs_exact,
            "axis {axis}: {got} too far from exact {exact}"
        );
    }
}

#[test]
fn wolfram_sph_waves_replays() {
    let golden: WavesGolden = serde_json::from_str(WAVES_GOLDEN).expect("parse golden");
    assert_eq!(golden.schema_version, 1);
    assert_eq!(golden.case_id, "fem.sph-waves.v1");
    assert_eq!(golden.count, 8);

    let mut particles = Vec::new();
    for i in 0..golden.count {
        let speed = if i < golden.count / 2 {
            golden.initial_velocity_left
        } else {
            0.0
        };
        particles.push(
            Particle::new(
                [i as f64 * golden.spacing, 0.0, 0.0],
                [speed, 0.0, 0.0],
                golden.mass,
            )
            .unwrap(),
        );
    }
    // Neighbor lists must match the golden's explicit cutoff lists.
    let positions: Vec<[f64; 3]> = particles.iter().map(|p| p.position).collect();
    let search =
        NeighborSearch::build(&positions, SUPPORT_RADIUS_FACTOR * golden.smoothing_length).unwrap();
    for (i, expected) in golden.neighbors.iter().enumerate() {
        assert_eq!(search.neighbors(i), *expected, "particle {i}");
    }

    let eos =
        PolynomialEos::new(golden.eos.c1, golden.eos.c2, golden.eos.c3, golden.eos.rho0).unwrap();
    let densities: Vec<f64> = (0..golden.count)
        .map(|i| {
            summation_density(
                i,
                &particles,
                &golden.neighbors[i],
                golden.smoothing_length,
                golden.dim,
            )
            .unwrap()
        })
        .collect();
    for (i, particle) in particles.iter_mut().enumerate() {
        let rho = densities[i];
        assert!(
            (rho - golden.density[i]).abs() < golden.tolerances.abs,
            "particle {i}: density got {rho}, want {}",
            golden.density[i]
        );
        particle.density = rho;
        particle.pressure = eos.pressure(rho);
        particle.sound_speed = eos.sound_speed(rho);
        assert!(
            (particle.pressure - golden.pressure[i]).abs() < golden.tolerances.abs,
            "particle {i}: pressure got {}, want {}",
            particle.pressure,
            golden.pressure[i]
        );
        assert!(
            (particle.sound_speed - golden.sound_speed[i]).abs() < golden.tolerances.abs,
            "particle {i}: sound speed got {}, want {}",
            particle.sound_speed,
            golden.sound_speed[i]
        );
    }

    for (i, particle) in particles.iter().enumerate() {
        let accel = momentum_acceleration(
            i,
            &particles,
            &golden.neighbors[i],
            golden.smoothing_length,
            golden.dim,
            golden.qa,
            golden.qb,
        )
        .unwrap();
        for (axis, (&got, &want)) in accel.iter().zip(golden.acceleration[i].iter()).enumerate() {
            assert!(
                (got - want).abs() < golden.tolerances.abs,
                "particle {i} axis {axis}: accel got {got}, want {want}"
            );
        }
        let heat = energy_rate(
            i,
            &particles,
            &golden.neighbors[i],
            golden.smoothing_length,
            golden.dim,
            golden.qa,
            golden.qb,
        )
        .unwrap();
        assert!(
            (heat - golden.energy_rate[i]).abs() < golden.tolerances.abs,
            "particle {i}: energy rate got {heat}, want {}",
            golden.energy_rate[i]
        );
        let rate = continuity_rate(
            i,
            &particles,
            &golden.neighbors[i],
            golden.smoothing_length,
            golden.dim,
        )
        .unwrap();
        assert!(
            (rate - golden.density_rate[i]).abs() < golden.tolerances.abs,
            "particle {i}: density rate got {rate}, want {}",
            golden.density_rate[i]
        );
        // Symplectic Euler step matches the golden's updated state.
        let stepped_velocity: [f64; 3] =
            std::array::from_fn(|axis| particle.velocity[axis] + accel[axis] * golden.dt);
        let stepped_position: [f64; 3] = std::array::from_fn(|axis| {
            particle.position[axis] + stepped_velocity[axis] * golden.dt
        });
        let stepped = stepped_velocity
            .iter()
            .zip(stepped_position.iter())
            .zip(golden.velocity_after_step[i].iter())
            .zip(golden.position_after_step[i].iter());
        for (axis, (((&velocity, &position), &want_v), &want_p)) in stepped.enumerate() {
            assert!(
                (velocity - want_v).abs() < golden.tolerances.abs,
                "particle {i} axis {axis}: velocity got {velocity}"
            );
            assert!(
                (position - want_p).abs() < golden.tolerances.abs,
                "particle {i} axis {axis}: position got {position}"
            );
        }
        let energy = particle.energy + heat * golden.dt;
        assert!(
            (energy - golden.energy_after_step[i]).abs() < golden.tolerances.abs,
            "particle {i}: energy got {energy}"
        );
    }
}
