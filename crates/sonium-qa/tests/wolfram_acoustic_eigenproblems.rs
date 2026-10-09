//! Sonium counterparts of Wolfram's published acoustic-tube eigensystems and
//! empty rectangular-room eigenfrequency example.
//!
//! Sources:
//! - <https://reference.wolfram.com/language/PDEModels/tutorial/Acoustics/AcousticsFrequencyDomain.html?view=all>
//! - <https://reference.wolfram.com/language/PDEModels/tutorial/Acoustics/ModelCollection/RoomEigenfrequencies.html?view=all>

use num_complex::Complex64;
use serde::Deserialize;
use sonium_fem::{
    assembly::HelmholtzProblem,
    basis::PolynomialDegree,
    boundary::{DirichletBC, RobinBC, apply_dirichlet, apply_robin},
    mesh::{box_mesh_tetrahedra, rectangular_mesh_triangles},
    physics::solid::eigenmodes::{AcousticEigenmodeConfig, build_acoustic_eigenmode_problem},
    solver::{SolverConfig, SolverType, solve},
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::LobpcgConfig;
use std::{env, path::PathBuf, process::Command};

const SOUND_SPEED_M_S: f64 = 343.0;
const TUBE_LENGTH_M: f64 = 5.0;
const TUBE_HEIGHT_M: f64 = 0.2;
const ROOM_DIMENSIONS_M: [f64; 3] = [7.5, 5.0, 3.0];
const ROOM_MODE_INDICES: [[usize; 3]; 5] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [2, 0, 0]];
const DRIVEN_TUBE_LENGTH_M: f64 = 2.0;
const DRIVEN_TUBE_WAVENUMBER_RAD_M: f64 = 1.0;
const DRIVEN_TUBE_SAMPLE_X_M: [f64; 5] = [0.0, 0.5, 1.0, 1.5, 2.0];

enum TubeTermination {
    Hard,
    Soft,
    Impedance { normalized_impedance: f64 },
}

fn solve_frequencies(
    mesh: &sonium_fem::mesh::Mesh,
    pressure_release_boundary_markers: Vec<i32>,
    num_eigenvalues: usize,
) -> Vec<f64> {
    let problem = build_acoustic_eigenmode_problem(
        mesh,
        &AcousticEigenmodeConfig {
            degree: PolynomialDegree::P1,
            sound_speed: SOUND_SPEED_M_S,
            pressure_release_boundary_markers,
        },
    );
    let result = problem
        .solve(&LobpcgConfig {
            num_eigenvalues,
            max_iterations: 1_000,
            tolerance: 1.0e-9,
            block_padding: 6,
        })
        .expect("Sonium LOBPCG should solve the acoustic eigenproblem");
    let mut frequencies = problem.eigenvalues_to_frequencies_hz(&result.eigenvalues);
    frequencies.sort_by(f64::total_cmp);
    frequencies
}

fn sonium_closed_tube_frequencies() -> Vec<f64> {
    // The transverse walls are sound-hard.  The compared modes are below the
    // first transverse cutoff c/(2 h) = 857.5 Hz and are therefore the axial
    // plane-wave family from Wolfram's 1D example.
    let mesh = rectangular_mesh_triangles(0.0, TUBE_LENGTH_M, 0.0, TUBE_HEIGHT_M, 120, 4);
    solve_frequencies(&mesh, vec![], 6)
}

fn sonium_open_tube_frequencies() -> Vec<f64> {
    let mesh = rectangular_mesh_triangles(0.0, TUBE_LENGTH_M, 0.0, TUBE_HEIGHT_M, 120, 4);
    // The rectangular mesh generator marks x=0 and x=L as 1 and 2.
    solve_frequencies(&mesh, vec![1, 2], 5)
}

fn sonium_room_frequencies() -> Vec<f64> {
    let [length, width, height] = ROOM_DIMENSIONS_M;
    let mesh = box_mesh_tetrahedra(0.0, length, 0.0, width, 0.0, height, 15, 10, 6);
    // No pressure-release markers means natural Neumann conditions: a fully
    // rigid room, including its constant zero-frequency pressure mode.
    solve_frequencies(&mesh, vec![], ROOM_MODE_INDICES.len())
}

fn exact_tube_closed_frequencies() -> Vec<f64> {
    (0..=5)
        .map(|n| n as f64 * SOUND_SPEED_M_S / (2.0 * TUBE_LENGTH_M))
        .collect()
}

fn exact_tube_open_frequencies() -> Vec<f64> {
    (1..=5)
        .map(|n| n as f64 * SOUND_SPEED_M_S / (2.0 * TUBE_LENGTH_M))
        .collect()
}

fn exact_room_frequencies() -> Vec<f64> {
    ROOM_MODE_INDICES
        .iter()
        .map(|index| {
            let squared_wavenumber = index
                .iter()
                .zip(ROOM_DIMENSIONS_M)
                .map(|(&n, length)| (n as f64 / length).powi(2))
                .sum::<f64>();
            SOUND_SPEED_M_S / 2.0 * squared_wavenumber.sqrt()
        })
        .collect()
}

fn solve_driven_tube(termination: TubeTermination) -> Vec<Complex64> {
    const NX: usize = 80;
    const NY: usize = 4;
    let mesh = rectangular_mesh_triangles(0.0, DRIVEN_TUBE_LENGTH_M, 0.0, TUBE_HEIGHT_M, NX, NY);
    let mut problem = HelmholtzProblem::assemble(
        &mesh,
        PolynomialDegree::P1,
        Complex64::new(DRIVEN_TUBE_WAVENUMBER_RAD_M, 0.0),
        |_, _, _| Complex64::new(0.0, 0.0),
    );
    let inlet = DirichletBC::new(1, |_, _, _| Complex64::new(1.0, 0.0));
    apply_dirichlet(&mut problem, &mesh, &[inlet]);
    match termination {
        TubeTermination::Hard => {}
        TubeTermination::Soft => {
            let outlet = DirichletBC::new(2, |_, _, _| Complex64::new(0.0, 0.0));
            apply_dirichlet(&mut problem, &mesh, &[outlet]);
        }
        TubeTermination::Impedance {
            normalized_impedance,
        } => {
            let alpha = Complex64::new(0.0, DRIVEN_TUBE_WAVENUMBER_RAD_M / normalized_impedance);
            let outlet = RobinBC::new(2, move |_, _, _| alpha, |_, _, _| Complex64::new(0.0, 0.0));
            apply_robin(&mut problem, &mesh, &[outlet], 2);
        }
    }
    let solution = solve(
        &problem,
        &SolverConfig {
            solver_type: SolverType::Direct,
            ..Default::default()
        },
    )
    .expect("Sonium should solve the driven impedance tube");
    assert!(solution.converged);

    DRIVEN_TUBE_SAMPLE_X_M
        .iter()
        .map(|&x| {
            let axial_index = (x / DRIVEN_TUBE_LENGTH_M * NX as f64).round() as usize;
            solution.values[axial_index]
        })
        .collect()
}

fn exact_driven_tube(reflection: f64) -> Vec<Complex64> {
    DRIVEN_TUBE_SAMPLE_X_M
        .iter()
        .map(|&x| {
            let incident = Complex64::new(0.0, -DRIVEN_TUBE_WAVENUMBER_RAD_M * x).exp();
            let reflected = Complex64::new(
                0.0,
                DRIVEN_TUBE_WAVENUMBER_RAD_M * (x - 2.0 * DRIVEN_TUBE_LENGTH_M),
            )
            .exp();
            let denominator = 1.0
                + reflection
                    * Complex64::new(
                        0.0,
                        -2.0 * DRIVEN_TUBE_WAVENUMBER_RAD_M * DRIVEN_TUBE_LENGTH_M,
                    )
                    .exp();
            (incident + reflection * reflected) / denominator
        })
        .collect()
}

#[derive(Deserialize)]
struct WolframAcousticReference {
    schema_version: u32,
    sound_speed_m_s: f64,
    tube_closed_frequencies_hz: Vec<f64>,
    tube_open_frequencies_hz: Vec<f64>,
    tube_closed_exact_hz: Vec<f64>,
    tube_open_exact_hz: Vec<f64>,
    room_mode_indices: Vec<Vec<usize>>,
    room_exact_hz: Vec<f64>,
    driven_tube_sample_x_m: Vec<f64>,
    driven_tube_hard_pressure: Vec<[f64; 2]>,
    driven_tube_soft_pressure: Vec<[f64; 2]>,
    driven_tube_impedance_pressure: Vec<[f64; 2]>,
}

fn wolfram_reference() -> Option<WolframAcousticReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/acoustic_eigenproblems.wls");
    let mut command = Command::new(&executable);
    command.arg("-file").arg(script);
    #[cfg(target_os = "macos")]
    if env::var_os("WolframKernel").is_none() {
        let kernel = PathBuf::from(
            "/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/MacOS/WolframKernel",
        );
        if kernel.is_file() {
            command.env("WolframKernel", kernel);
        }
    }
    let output = match command.wolfram_output() {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram Engine not available; using analytical acoustic references only");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram acoustic examples failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram acoustic output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn assert_spectrum_close(name: &str, actual: &[f64], reference: &[f64], relative_tolerance: f64) {
    assert_eq!(actual.len(), reference.len(), "{name} spectrum length");
    for (mode, (&got, &want)) in actual.iter().zip(reference).enumerate() {
        // The rigid zero mode is exactly 0 Hz, but each eigensolver returns
        // its own rounding-level residue (Sonium ~1e-6, Wolfram ~1e-4 Hz),
        // so near-zero references use the absolute budget: relative error
        // between two numerical zeros is meaningless. The 0.5 Hz floor only
        // catches true zero modes (the smallest nonzero reference here is
        // the 22.9 Hz room axial mode).
        if want.abs() < 0.5 {
            assert!(
                got.abs() < 0.5,
                "{name} zero mode: Sonium={got:.9} Hz, reference={want:.9} Hz"
            );
            continue;
        }
        let relative_error = (got - want).abs() / want;
        assert!(
            relative_error < relative_tolerance,
            "{name} mode {}: Sonium={got:.9} Hz, reference={want:.9} Hz, error={:.3}%",
            mode + 1,
            100.0 * relative_error,
        );
    }
}

fn complex_pairs(values: &[[f64; 2]]) -> Vec<Complex64> {
    values
        .iter()
        .map(|pair| Complex64::new(pair[0], pair[1]))
        .collect()
}

fn assert_complex_field_close(
    name: &str,
    actual: &[Complex64],
    reference: &[Complex64],
    tolerance: f64,
) {
    assert_eq!(actual.len(), reference.len());
    let error_energy = actual
        .iter()
        .zip(reference)
        .map(|(got, want)| (*got - *want).norm_sqr())
        .sum::<f64>();
    let reference_energy = reference.iter().map(Complex64::norm_sqr).sum::<f64>();
    let error = (error_energy / reference_energy).sqrt();
    assert!(
        error < tolerance,
        "{name} complex relative L2 error {:.3}% exceeds {:.3}%",
        100.0 * error,
        100.0 * tolerance,
    );
}

#[test]
fn sonium_reproduces_wolfram_acoustic_tube_and_room_eigenfrequencies() {
    let closed_tube = sonium_closed_tube_frequencies();
    let open_tube = sonium_open_tube_frequencies();
    let room = sonium_room_frequencies();
    let exact_closed_tube = exact_tube_closed_frequencies();
    let exact_open_tube = exact_tube_open_frequencies();
    let exact_room = exact_room_frequencies();

    assert_spectrum_close("closed tube", &closed_tube, &exact_closed_tube, 0.01);
    assert_spectrum_close("open tube", &open_tube, &exact_open_tube, 0.01);
    assert_spectrum_close("rigid room", &room, &exact_room, 0.04);

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.sound_speed_m_s, SOUND_SPEED_M_S);
        assert_eq!(reference.room_mode_indices, ROOM_MODE_INDICES);
        assert_spectrum_close(
            "Wolfram closed-tube discretization",
            &reference.tube_closed_frequencies_hz,
            &reference.tube_closed_exact_hz,
            0.002,
        );
        assert_spectrum_close(
            "Wolfram open-tube discretization",
            &reference.tube_open_frequencies_hz,
            &reference.tube_open_exact_hz,
            0.002,
        );
        assert_spectrum_close(
            "Sonium/Wolfram closed tube",
            &closed_tube,
            &reference.tube_closed_frequencies_hz,
            0.01,
        );
        assert_spectrum_close(
            "Sonium/Wolfram open tube",
            &open_tube,
            &reference.tube_open_frequencies_hz,
            0.01,
        );
        assert_spectrum_close(
            "Sonium/Wolfram rigid room",
            &room,
            &reference.room_exact_hz,
            0.04,
        );
    }
}

#[test]
fn sonium_reproduces_wolfram_driven_tube_terminations() {
    let hard = solve_driven_tube(TubeTermination::Hard);
    let soft = solve_driven_tube(TubeTermination::Soft);
    let impedance = solve_driven_tube(TubeTermination::Impedance {
        normalized_impedance: 2.0,
    });
    let exact_hard = exact_driven_tube(1.0);
    let exact_soft = exact_driven_tube(-1.0);
    let exact_impedance = exact_driven_tube(1.0 / 3.0);

    assert_complex_field_close("hard-ended tube", &hard, &exact_hard, 0.002);
    assert_complex_field_close("soft-ended tube", &soft, &exact_soft, 0.002);
    assert_complex_field_close("impedance-ended tube", &impedance, &exact_impedance, 0.002);

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.driven_tube_sample_x_m, DRIVEN_TUBE_SAMPLE_X_M);
        assert_complex_field_close(
            "Wolfram/Sonium hard-ended tube",
            &hard,
            &complex_pairs(&reference.driven_tube_hard_pressure),
            0.002,
        );
        assert_complex_field_close(
            "Wolfram/Sonium soft-ended tube",
            &soft,
            &complex_pairs(&reference.driven_tube_soft_pressure),
            0.002,
        );
        assert_complex_field_close(
            "Wolfram/Sonium impedance-ended tube",
            &impedance,
            &complex_pairs(&reference.driven_tube_impedance_pressure),
            0.002,
        );
    }
}
