//! Computed shoebox-room solver modal benchmark.
//!
//! Two complementary legs on a rigid rectangular room. The operator-consistency
//! leg evaluates Rayleigh quotients of analytical Neumann mode shapes against
//! the assembled FEM stiffness and mass matrices. The eigensystem leg solves
//! the generalized eigenproblem `K u = lambda M u` with LOBPCG and gates the
//! solved frequencies against the closed-form shoebox spectrum, checks
//! independent eigen-residuals, and writes the solved modal fields. The solved
//! leg is the accuracy evidence; the Rayleigh leg alone cannot detect a wrong
//! eigensolver because it never solves for a mode.

use sonium_fem::assembly::{assemble_mass, assemble_stiffness};
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::io::vtk::VtuWriter;
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::physics::solid::eigenmodes::{
    AcousticEigenmodeConfig, build_acoustic_eigenmode_problem,
};
use sonium_qa::QaResult;
use sonium_solvers::LobpcgConfig;
use sonium_xem::io::binary_output_dir;
use std::f64::consts::PI;

// Named acceptance tolerances for the Rayleigh operator-consistency leg.
const RAYLEIGH_TOL_AXIAL_X: f64 = 0.08;
const RAYLEIGH_TOL_AXIAL_Y: f64 = 0.08;
const RAYLEIGH_TOL_AXIAL_Z: f64 = 0.12;
const RAYLEIGH_TOL_TANGENTIAL_XY: f64 = 0.10;

// Solved-eigensystem leg: six lowest pairs cover the constant zero mode plus
// the (1,0,0), (0,1,0), (1,1,0), (0,0,1), and (2,0,0) room modes, where the
// last two are analytically degenerate at 68.6 Hz for this room.
const SOLVED_MODE_COUNT: usize = 6;
const LOBPCG_TOLERANCE: f64 = 1.0e-8;
const LOBPCG_MAX_ITERATIONS: usize = 2000;
const ZERO_MODE_FREQUENCY_CAP_HZ: f64 = 1.0;
// The constant mode has a zero eigenvalue, so its residual is roundoff over
// roundoff in any relative norm; gate it with absolute residual and shape.
const ZERO_MODE_ABSOLUTE_RESIDUAL_TOL: f64 = 1.0e-9;
const ZERO_MODE_CONSTANCY_TOL: f64 = 1.0e-6;
const SOLVED_FREQUENCY_MAX_REL_ERROR: f64 = 0.10;
const EIGEN_RESIDUAL_TOLERANCE: f64 = 1.0e-6;

#[derive(Clone, Copy, Debug)]
struct ShoeboxRoom {
    width: f64,
    depth: f64,
    height: f64,
    sound_speed: f64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct ModeIndex {
    nx: u32,
    ny: u32,
    nz: u32,
}

impl ShoeboxRoom {
    fn analytical_frequency(self, mode: ModeIndex) -> f64 {
        let kx = mode.nx as f64 / self.width;
        let ky = mode.ny as f64 / self.depth;
        let kz = mode.nz as f64 / self.height;
        0.5 * self.sound_speed * (kx * kx + ky * ky + kz * kz).sqrt()
    }

    fn mode_value(self, mode: ModeIndex, p: &Point) -> f64 {
        let x = (mode.nx as f64 * PI * p.x / self.width).cos();
        let y = (mode.ny as f64 * PI * p.y / self.depth).cos();
        let z = (mode.nz as f64 * PI * p.z / self.height).cos();
        x * y * z
    }
}

fn shoebox_mesh(room: ShoeboxRoom, nx: usize, ny: usize, nz: usize) -> Mesh {
    let mut mesh = Mesh::new(3);
    let dx = room.width / nx as f64;
    let dy = room.depth / ny as f64;
    let dz = room.height / nz as f64;

    for k in 0..=nz {
        for j in 0..=ny {
            for i in 0..=nx {
                mesh.add_node(Point::new_3d(i as f64 * dx, j as f64 * dy, k as f64 * dz));
            }
        }
    }

    let node =
        |i: usize, j: usize, k: usize| -> usize { k * (ny + 1) * (nx + 1) + j * (nx + 1) + i };

    for k in 0..nz {
        for j in 0..ny {
            for i in 0..nx {
                let v0 = node(i, j, k);
                let v1 = node(i + 1, j, k);
                let v2 = node(i, j + 1, k);
                let v3 = node(i + 1, j + 1, k);
                let v4 = node(i, j, k + 1);
                let v5 = node(i + 1, j, k + 1);
                let v6 = node(i, j + 1, k + 1);
                let v7 = node(i + 1, j + 1, k + 1);

                mesh.add_element(ElementType::Tetrahedron, vec![v0, v1, v3, v7]);
                mesh.add_element(ElementType::Tetrahedron, vec![v0, v1, v7, v5]);
                mesh.add_element(ElementType::Tetrahedron, vec![v0, v5, v7, v4]);
                mesh.add_element(ElementType::Tetrahedron, vec![v0, v3, v2, v7]);
                mesh.add_element(ElementType::Tetrahedron, vec![v0, v2, v6, v7]);
                mesh.add_element(ElementType::Tetrahedron, vec![v0, v6, v4, v7]);
            }
        }
    }

    mesh
}

fn quadratic_form(rows: &[usize], cols: &[usize], values: &[f64], u: &[f64]) -> f64 {
    rows.iter()
        .zip(cols)
        .zip(values)
        .map(|((&row, &col), &value)| u[row] * value * u[col])
        .sum()
}

fn computed_frequency(room: ShoeboxRoom, mesh: &Mesh, mode: ModeIndex) -> Result<f64, String> {
    let stiffness = assemble_stiffness(mesh, PolynomialDegree::P1);
    let mass = assemble_mass(mesh, PolynomialDegree::P1);
    let u: Vec<f64> = mesh
        .nodes
        .iter()
        .map(|p| room.mode_value(mode, p))
        .collect();

    let ku = quadratic_form(&stiffness.rows, &stiffness.cols, &stiffness.values, &u);
    let mu = quadratic_form(&mass.rows, &mass.cols, &mass.values, &u);
    if mu <= 0.0 {
        return Err(format!(
            "mode {mode:?} produced non-positive mass norm {mu}"
        ));
    }

    let omega_h = (ku / mu).sqrt() * room.sound_speed;
    Ok(omega_h / (2.0 * PI))
}

fn validate_mode(
    room: ShoeboxRoom,
    mesh: &Mesh,
    mode: ModeIndex,
    max_relative_error: f64,
) -> Result<(), String> {
    let computed = computed_frequency(room, mesh, mode)?;
    let analytical = room.analytical_frequency(mode);
    let rel = (computed - analytical).abs() / analytical;

    println!(
        "  mode ({}, {}, {}): FEM {:.4} Hz vs analytical {:.4} Hz, rel error {:.2}%",
        mode.nx,
        mode.ny,
        mode.nz,
        computed,
        analytical,
        rel * 100.0
    );

    if rel > max_relative_error {
        return Err(format!(
            "mode {mode:?} relative error {:.2}% exceeds {:.2}%",
            rel * 100.0,
            max_relative_error * 100.0
        ));
    }
    Ok(())
}

fn vector_norm(values: &[f64]) -> f64 {
    values.iter().map(|v| v * v).sum::<f64>().sqrt()
}

/// Solved-eigensystem leg: LOBPCG eigenpairs with residual and spectrum gates.
fn validate_solved_modes(room: ShoeboxRoom, mesh: &Mesh) -> Result<SolvedModeMetrics, String> {
    // A rigid room has Neumann conditions everywhere, so no pressure-release
    // markers are eliminated and every mesh node stays a free unknown.
    let problem = build_acoustic_eigenmode_problem(
        mesh,
        &AcousticEigenmodeConfig {
            degree: PolynomialDegree::P1,
            sound_speed: room.sound_speed,
            pressure_release_boundary_markers: Vec::new(),
        },
    );
    if problem.stiffness_like.num_rows != mesh.num_nodes()
        || problem.mass_like.num_rows != mesh.num_nodes()
    {
        return Err(format!(
            "rigid-room eigensystem should keep all {} nodes free, got K={} M={}",
            mesh.num_nodes(),
            problem.stiffness_like.num_rows,
            problem.mass_like.num_rows
        ));
    }

    let config = LobpcgConfig {
        num_eigenvalues: SOLVED_MODE_COUNT,
        max_iterations: LOBPCG_MAX_ITERATIONS,
        tolerance: LOBPCG_TOLERANCE,
        ..Default::default()
    };
    let result = problem
        .solve(&config)
        .map_err(|error| format!("LOBPCG eigensolve failed: {error:?}"))?;
    if result.eigenvalues.len() != SOLVED_MODE_COUNT
        || result.residual_norms.len() != SOLVED_MODE_COUNT
    {
        return Err(format!(
            "expected {SOLVED_MODE_COUNT} eigenpairs, got {} values and {} residuals",
            result.eigenvalues.len(),
            result.residual_norms.len()
        ));
    }
    // Every nonzero pair must meet the solver tolerance. Pair 0 is the
    // constant null-space mode: its relative residual is roundoff over
    // roundoff by construction, so it is gated separately below with an
    // absolute residual, a frequency cap, and a constancy check.
    for j in 1..SOLVED_MODE_COUNT {
        if result.residual_norms[j] > LOBPCG_TOLERANCE {
            return Err(format!(
                "LOBPCG pair {j} residual {:.3e} exceeds {LOBPCG_TOLERANCE:.1e} after {} iterations",
                result.residual_norms[j], result.iterations
            ));
        }
    }

    let frequencies = problem.eigenvalues_to_frequencies_hz(&result.eigenvalues);
    println!(
        "  Solved rigid-room spectrum: {} pairs in {} iterations (solver converged flag: {})",
        frequencies.len(),
        result.iterations,
        result.converged
    );

    // The constant pressure mode must sit at zero frequency with a constant
    // shape and a small absolute residual.
    if !frequencies[0].is_finite() || frequencies[0] > ZERO_MODE_FREQUENCY_CAP_HZ {
        return Err(format!(
            "zero mode frequency {:.4} Hz exceeds the {ZERO_MODE_FREQUENCY_CAP_HZ:.1} Hz cap",
            frequencies[0]
        ));
    }
    let zero_mode = result.eigenvectors.column_to_vec(0);
    let zero_mean = zero_mode.iter().sum::<f64>() / zero_mode.len() as f64;
    let zero_constancy = vector_norm(
        &zero_mode
            .iter()
            .map(|v| v - zero_mean)
            .collect::<Vec<f64>>(),
    ) / vector_norm(&zero_mode);
    println!("  Zero mode constancy deviation: {zero_constancy:.3e}");
    if !zero_constancy.is_finite() || zero_constancy > ZERO_MODE_CONSTANCY_TOL {
        return Err(format!(
            "zero mode constancy deviation {zero_constancy:.3e} exceeds {ZERO_MODE_CONSTANCY_TOL:.1e}"
        ));
    }

    // Match the sorted solved spectrum against the sorted analytical set. The
    // (0,0,1)/(2,0,0) pair is analytically degenerate, so positional matching
    // after sorting is the unambiguous comparison.
    let mut expected = [
        ModeIndex {
            nx: 1,
            ny: 0,
            nz: 0,
        },
        ModeIndex {
            nx: 0,
            ny: 1,
            nz: 0,
        },
        ModeIndex {
            nx: 1,
            ny: 1,
            nz: 0,
        },
        ModeIndex {
            nx: 0,
            ny: 0,
            nz: 1,
        },
        ModeIndex {
            nx: 2,
            ny: 0,
            nz: 0,
        },
    ]
    .map(|mode| room.analytical_frequency(mode));
    expected.sort_by(f64::total_cmp);
    let mut solved = frequencies[1..].to_vec();
    solved.sort_by(f64::total_cmp);

    let mut max_rel_error = 0.0f64;
    for (rank, (&computed, &reference)) in solved.iter().zip(expected.iter()).enumerate() {
        let rel = (computed - reference).abs() / reference;
        max_rel_error = max_rel_error.max(rel);
        println!(
            "  solved mode {}: FEM {computed:.4} Hz vs analytical {reference:.4} Hz, rel error {rel:.2}%",
            rank + 1,
            rel = rel * 100.0
        );
        if rel > SOLVED_FREQUENCY_MAX_REL_ERROR {
            return Err(format!(
                "solved mode {} relative error {rel:.2}% exceeds {tol:.2}%",
                rank + 1,
                rel = rel * 100.0,
                tol = SOLVED_FREQUENCY_MAX_REL_ERROR * 100.0
            ));
        }
    }

    // Independent eigen-residuals recomputed from the assembled operators, not
    // the solver's self-reported residual norms. Pair 0 uses an absolute
    // residual because its relative norm is undefined at lambda = 0.
    let mut max_residual = 0.0f64;
    let mut zero_absolute_residual = 0.0f64;
    for (j, eigenvalue) in result.eigenvalues.iter().enumerate() {
        let mode = result.eigenvectors.column_to_vec(j);
        let stiffness_action = problem.stiffness_like.matvec(&mode);
        let mass_action = problem.mass_like.matvec(&mode);
        let mass_norm = vector_norm(&mass_action);
        if !mass_norm.is_finite() || mass_norm == 0.0 {
            return Err(format!("solved mode {j} has a non-positive mass norm"));
        }
        let residual: Vec<f64> = stiffness_action
            .iter()
            .zip(mass_action.iter())
            .map(|(&kx, &mx)| kx - eigenvalue * mx)
            .collect();
        let residual_norm = vector_norm(&residual);
        if j == 0 {
            zero_absolute_residual = residual_norm;
            println!("  Zero mode absolute eigen-residual: {zero_absolute_residual:.3e}");
            if !residual_norm.is_finite() || residual_norm > ZERO_MODE_ABSOLUTE_RESIDUAL_TOL {
                return Err(format!(
                    "zero mode absolute residual {residual_norm:.3e} exceeds {ZERO_MODE_ABSOLUTE_RESIDUAL_TOL:.1e}"
                ));
            }
            continue;
        }
        let scale = vector_norm(&stiffness_action) + eigenvalue.abs() * mass_norm;
        let rel = residual_norm / scale;
        max_residual = max_residual.max(rel);
        if rel > EIGEN_RESIDUAL_TOLERANCE {
            return Err(format!(
                "solved mode {j} eigen-residual {rel:.3e} exceeds {EIGEN_RESIDUAL_TOLERANCE:.1e}"
            ));
        }
    }
    println!("  Maximum independent eigen-residual: {max_residual:.3e}");

    // Solved modal fields for inspection: the first three nonzero modes.
    let dir = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"));
    let mut writer = VtuWriter::new(mesh);
    for j in 1..=3 {
        let mut mode = result.eigenvectors.column_to_vec(j);
        let peak = mode.iter().map(|v| v.abs()).fold(0.0, f64::max);
        if peak > 0.0 {
            for value in &mut mode {
                *value /= peak;
            }
        }
        writer.add_point_scalar(&format!("solved_mode_{j}"), &mode);
    }
    let _ = writer.write(dir.join("shoebox_solved_modes.vtu"));

    Ok(SolvedModeMetrics {
        zero_mode_frequency_hz: frequencies[0],
        zero_mode_absolute_residual: zero_absolute_residual,
        max_solved_frequency_rel_error: max_rel_error,
        max_eigen_residual: max_residual,
        lobpcg_iterations: result.iterations as f64,
    })
}

struct SolvedModeMetrics {
    zero_mode_frequency_hz: f64,
    zero_mode_absolute_residual: f64,
    max_solved_frequency_rel_error: f64,
    max_eigen_residual: f64,
    lobpcg_iterations: f64,
}

fn run() -> Result<SolvedModeMetrics, String> {
    let room = ShoeboxRoom {
        width: 5.0,
        depth: 4.0,
        height: 2.5,
        sound_speed: 343.0,
    };
    let mesh = shoebox_mesh(room, 8, 6, 4);

    println!(
        "Computed shoebox FEM modal benchmark: {} nodes, {} tetrahedra",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    let dir = binary_output_dir(env!("CARGO_PKG_NAME"), env!("CARGO_BIN_NAME"));
    let _ = VtuWriter::new(&mesh).write(dir.join("shoebox_mesh.vtu"));
    println!("Rayleigh operator-consistency leg:");
    validate_mode(
        room,
        &mesh,
        ModeIndex {
            nx: 1,
            ny: 0,
            nz: 0,
        },
        RAYLEIGH_TOL_AXIAL_X,
    )?;
    validate_mode(
        room,
        &mesh,
        ModeIndex {
            nx: 0,
            ny: 1,
            nz: 0,
        },
        RAYLEIGH_TOL_AXIAL_Y,
    )?;
    validate_mode(
        room,
        &mesh,
        ModeIndex {
            nx: 0,
            ny: 0,
            nz: 1,
        },
        RAYLEIGH_TOL_AXIAL_Z,
    )?;
    validate_mode(
        room,
        &mesh,
        ModeIndex {
            nx: 1,
            ny: 1,
            nz: 0,
        },
        RAYLEIGH_TOL_TANGENTIAL_XY,
    )?;
    println!("Solved-eigensystem leg:");
    validate_solved_modes(room, &mesh)
}

fn main() {
    match run() {
        Err(error) => {
            if let Ok(record) =
                QaResult::failed("room.shoebox-fem-benchmark.v1", error.as_str()).to_json_line()
            {
                println!("QA_RESULT: {record}");
            }
            eprintln!("FAILED: {error}");
            println!("FAILED");
            std::process::exit(1);
        }
        Ok(metrics) => {
            let record = QaResult::passed("room.shoebox-fem-benchmark.v1")
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("zero_mode_frequency_hz", metrics.zero_mode_frequency_hz)
                .metric(
                    "zero_mode_absolute_residual",
                    metrics.zero_mode_absolute_residual,
                )
                .metric(
                    "max_solved_frequency_rel_error",
                    metrics.max_solved_frequency_rel_error,
                )
                .metric("max_eigen_residual", metrics.max_eigen_residual)
                .metric("lobpcg_iterations", metrics.lobpcg_iterations)
                .maximum("zero_mode_frequency_hz", ZERO_MODE_FREQUENCY_CAP_HZ)
                .maximum(
                    "zero_mode_absolute_residual",
                    ZERO_MODE_ABSOLUTE_RESIDUAL_TOL,
                )
                .maximum(
                    "max_solved_frequency_rel_error",
                    SOLVED_FREQUENCY_MAX_REL_ERROR,
                )
                .maximum("max_eigen_residual", EIGEN_RESIDUAL_TOLERANCE)
                .seed("shoebox-5x4x2.5:tet-8x6x4:p1:lobpcg-6");
            match record.to_json_line() {
                Ok(json) => println!("QA_RESULT: {json}"),
                Err(error) => {
                    eprintln!("FAILED: {error}");
                    println!("FAILED");
                    std::process::exit(1);
                }
            }
            println!("SUCCESS");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computed_shoebox_modes_track_analytic_frequencies() {
        run().expect("shoebox solver modes should stay near analytic rigid-room modes");
    }
}
