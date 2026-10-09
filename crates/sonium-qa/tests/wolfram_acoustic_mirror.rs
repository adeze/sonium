//! Wolfram Acoustic Mirror transient focusing verification.
//!
//! The literal model uses two opposing 5 m wide, 0.2 m thick parabolic
//! mirrors with focal length 1 m and vertex separation 10 m. A regularized
//! derivative-Gaussian monopole pulse starts at the left focus `(1, 0)` and is
//! measured at the right focus `(9, 0)`. Wolfram publishes an amplification
//! ratio of 2.96217 relative to propagation without the mirrors.

use sonium_fem::{
    assembly::{
        PrebakedAssembler, assemble_bilinear, assemble_boundary_mass_generic, assemble_linear,
    },
    basis::PolynomialDegree,
    forms::{LaplacianForm, MassForm, SourceForm},
    mesh::{BoundaryType, ElementType, Mesh, Point},
    spaces::H1Space,
};
use sonium_solvers::{CsrMatrix, SparseLuFactorization};
use std::f64::consts::PI;

const SOUND_SPEED_M_S: f64 = 343.2;
const AIR_DENSITY_KG_M3: f64 = 1.204;
const FOCAL_LENGTH_M: f64 = 1.0;
const MIRROR_THICKNESS_M: f64 = 0.2;
const MIRROR_WIDTH_M: f64 = 5.0;
const MIRROR_SEPARATION_M: f64 = 10.0;
const PULSE_HALF_WIDTH_S: f64 = 0.004;
const MAX_FREQUENCY_HZ: f64 = 600.0;
const END_TIME_S: f64 = 0.06;
const TIME_STEP_S: f64 = 0.0002;
const ABSORBING_MARKER: i32 = 1;
const HARD_WALL_MARKER: i32 = 2;
const WOLFRAM_AMPLIFICATION: f64 = 2.96217;

fn mirror_mask_mesh(with_mirrors: bool) -> Mesh {
    let x_min = -0.1 * MIRROR_SEPARATION_M;
    let x_max = 1.1 * MIRROR_SEPARATION_M;
    let y_min = -0.75 * MIRROR_WIDTH_M;
    let y_max = 0.75 * MIRROR_WIDTH_M;
    // 0.125 m grid spacing gives triangle edges below Wolfram's
    // c/(3 f_max) = 0.1907 m mesh-size target.
    let nx = 96;
    let ny = 60;
    let dx = (x_max - x_min) / nx as f64;
    let dy = (y_max - y_min) / ny as f64;
    let background_node = |i: usize, j: usize| j * (nx + 1) + i;

    let mut active = vec![vec![true; nx]; ny];
    if with_mirrors {
        for (j, row) in active.iter_mut().enumerate() {
            let y = y_min + (j as f64 + 0.5) * dy;
            for (i, cell) in row.iter_mut().enumerate() {
                let x = x_min + (i as f64 + 0.5) * dx;
                let left_coordinate = x - y * y / (4.0 * FOCAL_LENGTH_M);
                let right_coordinate = x - MIRROR_SEPARATION_M + y * y / (4.0 * FOCAL_LENGTH_M);
                let within_aperture = y.abs() <= MIRROR_WIDTH_M / 2.0;
                let in_left =
                    (-MIRROR_THICKNESS_M..=0.0).contains(&left_coordinate) && within_aperture;
                let in_right =
                    (0.0..=MIRROR_THICKNESS_M).contains(&right_coordinate) && within_aperture;
                *cell = !(in_left || in_right);
            }
        }
    }

    let mut used = vec![false; (nx + 1) * (ny + 1)];
    let mut cells = Vec::new();
    for (j, row) in active.iter().enumerate() {
        for (i, &is_active) in row.iter().enumerate() {
            if !is_active {
                continue;
            }
            let nodes = [
                background_node(i, j),
                background_node(i + 1, j),
                background_node(i, j + 1),
                background_node(i + 1, j + 1),
            ];
            for node in nodes {
                used[node] = true;
            }
            cells.push((i, j, nodes));
        }
    }

    let mut mesh = Mesh::new(2);
    let mut compact = vec![usize::MAX; used.len()];
    for (background, is_used) in used.into_iter().enumerate() {
        if is_used {
            let i = background % (nx + 1);
            let j = background / (nx + 1);
            compact[background] =
                mesh.add_node(Point::new_2d(x_min + i as f64 * dx, y_min + j as f64 * dy));
        }
    }

    // Mirror the diagonal choice across both model symmetry planes so the
    // discretization does not create a preferred focusing direction.
    for (i, j, [n00, n10, n01, n11]) in cells {
        let [n00, n10, n01, n11] = [compact[n00], compact[n10], compact[n01], compact[n11]];
        let forward_diagonal = (i < nx / 2) == (j < ny / 2);
        if forward_diagonal {
            mesh.add_element(ElementType::Triangle, vec![n00, n10, n11]);
            mesh.add_element(ElementType::Triangle, vec![n00, n11, n01]);
        } else {
            mesh.add_element(ElementType::Triangle, vec![n00, n10, n01]);
            mesh.add_element(ElementType::Triangle, vec![n10, n11, n01]);
        }
    }
    mesh.detect_boundaries();

    let tolerance = 0.1 * dx.min(dy);
    for boundary in &mut mesh.boundaries {
        let is_outer = boundary.nodes.iter().all(|&node| {
            let point = &mesh.nodes[node];
            (point.x - x_min).abs() < tolerance
                || (point.x - x_max).abs() < tolerance
                || (point.y - y_min).abs() < tolerance
                || (point.y - y_max).abs() < tolerance
        });
        if is_outer {
            boundary.marker = ABSORBING_MARKER;
            boundary.boundary_type = BoundaryType::Robin;
        } else {
            boundary.marker = HARD_WALL_MARKER;
            boundary.boundary_type = BoundaryType::Neumann;
        }
    }
    mesh
}

fn regularized_delta(gamma: f64, x: f64, y: f64) -> f64 {
    let dx = x - FOCAL_LENGTH_M;
    if dx.abs() > 2.0 * gamma || y.abs() > 2.0 * gamma {
        return 0.0;
    }
    let kernel = |offset: f64| (1.0 + (PI * offset / (2.0 * gamma)).cos()) / (4.0 * gamma);
    kernel(dx) * kernel(y)
}

fn derivative_gaussian_pulse(time: f64) -> f64 {
    let coordinate = PI * 2.0_f64.sqrt() * (time - PULSE_HALF_WIDTH_S) / PULSE_HALF_WIDTH_S;
    // Overall normalization and sign cancel in the mirror/free-field ratio.
    1000.0 * coordinate * (-coordinate * coordinate).exp()
}

fn csr_matvec(matrix: &CsrMatrix<f64>, vector: &[f64]) -> Vec<f64> {
    (0..matrix.num_rows)
        .map(|row| {
            (matrix.row_ptrs[row]..matrix.row_ptrs[row + 1])
                .map(|index| matrix.values[index] * vector[matrix.col_indices[index]])
                .sum()
        })
        .collect()
}

fn simulate_receiver(with_mirrors: bool) -> Vec<(f64, f64)> {
    let mesh = mirror_mask_mesh(with_mirrors);
    let space = H1Space::new(PolynomialDegree::P1);
    let stiffness_triplet = assemble_bilinear(&LaplacianForm, &space, &space, &mesh, 2);
    let mass_triplet = assemble_bilinear(&MassForm, &space, &space, &mesh, 2);
    let boundary_triplet = assemble_boundary_mass_generic(&mesh, ABSORBING_MARKER);
    let assembler = PrebakedAssembler::<3>::from_triplets(
        [stiffness_triplet, mass_triplet, boundary_triplet],
        &[],
    );
    let stiffness = assembler.assemble_real(&[1.0, 0.0, 0.0]);
    let damping = assembler.assemble_real(&[0.0, 0.0, 1.0 / SOUND_SPEED_M_S]);

    let beta = 0.25;
    let gamma = 0.5;
    let effective = assembler.assemble_real(&[
        beta * TIME_STEP_S * TIME_STEP_S,
        1.0 / SOUND_SPEED_M_S.powi(2),
        gamma * TIME_STEP_S / SOUND_SPEED_M_S,
    ]);
    let factorization =
        SparseLuFactorization::factorize(&effective).expect("mirror Newmark factorization");

    let source_width = SOUND_SPEED_M_S / (3.0 * MAX_FREQUENCY_HZ);
    let source_shape = assemble_linear(
        &SourceForm::new(move |x, y, _| regularized_delta(source_width, x, y)),
        &space,
        &mesh,
        4,
    );
    let receiver = mesh
        .nodes
        .iter()
        .enumerate()
        .min_by(|(_, left), (_, right)| {
            let left_distance = (left.x - (MIRROR_SEPARATION_M - FOCAL_LENGTH_M)).hypot(left.y);
            let right_distance = (right.x - (MIRROR_SEPARATION_M - FOCAL_LENGTH_M)).hypot(right.y);
            left_distance.total_cmp(&right_distance)
        })
        .map(|(index, _)| index)
        .expect("receiver node");
    let receiver_point = &mesh.nodes[receiver];
    assert!((receiver_point.x - 9.0).abs() < 1.0e-12 && receiver_point.y.abs() < 1.0e-12);

    let ndof = mesh.num_nodes();
    let mut pressure = vec![0.0; ndof];
    let mut velocity = vec![0.0; ndof];
    let mut acceleration = vec![0.0; ndof];
    let steps = (END_TIME_S / TIME_STEP_S).round() as usize;
    let mut receiver_signal = Vec::with_capacity(steps + 1);
    receiver_signal.push((0.0, 0.0));

    for step in 0..steps {
        let time = (step + 1) as f64 * TIME_STEP_S;
        let mut pressure_predictor = vec![0.0; ndof];
        let mut velocity_predictor = vec![0.0; ndof];
        for index in 0..ndof {
            pressure_predictor[index] = pressure[index]
                + TIME_STEP_S * velocity[index]
                + (0.5 - beta) * TIME_STEP_S.powi(2) * acceleration[index];
            velocity_predictor[index] =
                velocity[index] + (1.0 - gamma) * TIME_STEP_S * acceleration[index];
        }

        let stiffness_pressure = csr_matvec(&stiffness, &pressure_predictor);
        let damping_velocity = csr_matvec(&damping, &velocity_predictor);
        let pulse = derivative_gaussian_pulse(time);
        let rhs = (0..ndof)
            .map(|index| {
                pulse * source_shape[index] - stiffness_pressure[index] - damping_velocity[index]
            })
            .collect::<Vec<_>>();
        let acceleration_new = factorization.solve(&rhs).expect("mirror Newmark step");
        for index in 0..ndof {
            pressure[index] =
                pressure_predictor[index] + beta * TIME_STEP_S.powi(2) * acceleration_new[index];
            velocity[index] =
                velocity_predictor[index] + gamma * TIME_STEP_S * acceleration_new[index];
        }
        acceleration = acceleration_new;
        receiver_signal.push((time, pressure[receiver]));
    }
    receiver_signal
}

fn peak_in_window(signal: &[(f64, f64)], start: f64, end: f64) -> (f64, f64) {
    signal
        .iter()
        .copied()
        .filter(|(time, _)| (start..=end).contains(time))
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .expect("nonempty peak window")
}

#[test]
fn sonium_reproduces_wolfram_acoustic_mirror_amplification() {
    let mirror_signal = simulate_receiver(true);
    let refocus_time = (MIRROR_SEPARATION_M + 2.0 * FOCAL_LENGTH_M) / SOUND_SPEED_M_S;
    let (mirror_peak_time, mirror_peak) = peak_in_window(
        &mirror_signal,
        refocus_time,
        refocus_time + 2.0 * PULSE_HALF_WIDTH_S,
    );
    // Wolfram divides by its published 2D free-space Green-function
    // expression, rho*p_in/(2*pi*sqrt(r)), not by a second FEM solve.
    // The derivative-Gaussian wavelet normalization cancels in the ratio.
    let input_peak = 1000.0 / (2.0 * std::f64::consts::E).sqrt();
    let propagation_distance = MIRROR_SEPARATION_M - 2.0 * FOCAL_LENGTH_M;
    let free_peak = AIR_DENSITY_KG_M3 * input_peak / (2.0 * PI * propagation_distance.sqrt());
    let amplification = mirror_peak / free_peak;

    assert!(
        (mirror_peak_time - (refocus_time + PULSE_HALF_WIDTH_S)).abs() < 0.002,
        "mirror arrival time: {mirror_peak_time:.6} s"
    );
    assert!(mirror_peak > 0.0 && free_peak > 0.0);
    assert!(
        (amplification - WOLFRAM_AMPLIFICATION).abs() < 0.6,
        "amplification {amplification:.5}, Wolfram {WOLFRAM_AMPLIFICATION:.5}; mirror peak {mirror_peak:.6} at {mirror_peak_time:.6} s, analytical free peak {free_peak:.6}"
    );
}
