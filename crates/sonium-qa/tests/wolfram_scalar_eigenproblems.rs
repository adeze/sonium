//! Sonium counterparts of published Wolfram NDEigensystem scalar examples.
//!
//! Source: <https://reference.wolfram.com/language/ref/NDEigensystem.html>

use num_complex::Complex64;
use serde::Deserialize;
use sonium_fem::{
    boundary::periodic::{
        PeriodicTransform, expand_node_periodic_constraints, pair_marked_boundary_nodes,
    },
    mesh::{
        generators::{rectangular_mesh_triangles, two_disk_union_mesh_triangles},
        triangulate::polygon_mesh_triangles,
    },
    physics::solid::eigenmodes::{
        IntervalLaplacianConfig, RadialSchrodingerConfig, ScalarEigenproblemConfig,
        ScalarRobinTerm, build_coupled_reaction_diffusion_eigenproblem,
        build_interval_laplacian_problem, build_radial_schrodinger_problem,
        build_scalar_eigenproblem, evolve_heat_mode_coefficients, evolve_wave_mode_coefficients,
        expand_homogeneous_dirichlet_modes, homogeneous_dirichlet_free_dofs,
        linearize_quadratic_eigenproblem, project_onto_real_modes, reconstruct_from_real_modes,
        track_real_modes_across_meshes, try_build_complex_scalar_eigenproblem,
    },
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{ComplexArnoldiConfig, CsrMatrix, LobpcgConfig, ShiftInvertConfig};
use std::{env, path::PathBuf, process::Command};

#[derive(Deserialize)]
struct WolframScalarReference {
    schema_version: u32,
    schrodinger: Vec<f64>,
    schrodinger_exact: Vec<f64>,
    sturm_liouville: Vec<f64>,
    robin: Vec<f64>,
    periodic: Vec<f64>,
    bloch: Vec<f64>,
    quantum_bands: Vec<Vec<f64>>,
    interval: Vec<f64>,
    radial_hydrogen: Vec<f64>,
    damped_wave: Vec<[f64; 2]>,
    coupled_system: [f64; 2],
    l_shape: Vec<f64>,
    avoided_crossing: Vec<Vec<f64>>,
    image_mask: Vec<f64>,
    neumann: Vec<f64>,
    wave_heat_samples: [f64; 2],
    interval_1d: Vec<f64>,
    partially_constrained: Vec<f64>,
}

fn solve_interval_1d() -> Vec<f64> {
    let mut eigenvalues = build_interval_laplacian_problem(&IntervalLaplacianConfig::default())
        .solve(&LobpcgConfig {
            num_eigenvalues: 4,
            max_iterations: 400,
            tolerance: 1.0e-10,
            block_padding: 4,
        })
        .expect("Sonium should solve the 1D interval spectrum")
        .eigenvalues;
    eigenvalues.sort_by(f64::total_cmp);
    eigenvalues
}

fn solve_neumann() -> Vec<f64> {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.8, 28, 22);
    let mut eigenvalues = build_scalar_eigenproblem(
        &mesh,
        |_, _, _| 1.0,
        |_, _, _| 0.0,
        |_, _, _| 1.0,
        &ScalarEigenproblemConfig::default(),
    )
    .solve(&LobpcgConfig {
        num_eigenvalues: 4,
        max_iterations: 600,
        tolerance: 1.0e-8,
        block_padding: 4,
    })
    .expect("Sonium should solve the homogeneous Neumann spectrum")
    .eigenvalues;
    eigenvalues.sort_by(f64::total_cmp);
    if eigenvalues[0].abs() < 1.0e-8 {
        eigenvalues[0] = 0.0;
    }
    eigenvalues
}

fn solve_wave_heat_modal_samples() -> [f64; 2] {
    let nx = 32;
    let ny = 24;
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.8, nx, ny);
    let markers = [1, 2, 3, 4];
    let problem = build_scalar_eigenproblem(
        &mesh,
        |_, _, _| 1.0,
        |_, _, _| 0.0,
        |_, _, _| 1.0,
        &ScalarEigenproblemConfig {
            dirichlet_boundary_markers: markers.to_vec(),
            ..Default::default()
        },
    );
    let result = problem
        .solve(&LobpcgConfig {
            num_eigenvalues: 4,
            max_iterations: 600,
            tolerance: 1.0e-9,
            block_padding: 4,
        })
        .expect("Sonium should solve the modal expansion basis");
    let free_dofs = homogeneous_dirichlet_free_dofs(&mesh, &markers);
    let initial: Vec<_> = free_dofs
        .iter()
        .map(|&node| {
            let point = &mesh.nodes[node];
            (std::f64::consts::PI * point.x).sin() * (std::f64::consts::PI * point.y / 0.8).sin()
                + 0.25
                    * (2.0 * std::f64::consts::PI * point.x).sin()
                    * (std::f64::consts::PI * point.y / 0.8).sin()
        })
        .collect();
    let initial_coefficients =
        project_onto_real_modes(&problem.mass_like, &result.eigenvectors, &initial)
            .expect("initial field should project onto the modal basis");
    let zero_velocity = vec![0.0; initial_coefficients.len()];
    let wave_coefficients = evolve_wave_mode_coefficients(
        &result.eigenvalues,
        &initial_coefficients,
        &zero_velocity,
        0.07,
    )
    .expect("wave coefficients should evolve");
    let heat_coefficients =
        evolve_heat_mode_coefficients(&result.eigenvalues, &initial_coefficients, 0.2, 0.03)
            .expect("heat coefficients should evolve");
    let wave = reconstruct_from_real_modes(&result.eigenvectors, &wave_coefficients)
        .expect("wave field should reconstruct");
    let heat = reconstruct_from_real_modes(&result.eigenvectors, &heat_coefficients)
        .expect("heat field should reconstruct");
    let sample_node = 9 * (nx + 1) + 12;
    let sample_row = free_dofs
        .iter()
        .position(|&node| node == sample_node)
        .expect("sample point should be an interior free DOF");
    [wave[sample_row], heat[sample_row]]
}

fn solve_image_mask() -> Vec<f64> {
    let mask: Vec<Vec<bool>> = (0..24)
        .map(|row| {
            (0..48)
                .map(|column| {
                    (8..40).contains(&column) && (6..16).contains(&row)
                        || (16..32).contains(&column) && (16..21).contains(&row)
                })
                .collect()
        })
        .collect();
    let mesh =
        sonium_fem::mesh::generators::binary_mask_mesh_triangles(&mask, 0.0, 2.0, 0.0, 1.0, 1);
    let mut eigenvalues = build_scalar_eigenproblem(
        &mesh,
        |_, _, _| 1.0,
        |_, _, _| 0.0,
        |_, _, _| 1.0,
        &ScalarEigenproblemConfig {
            dirichlet_boundary_markers: vec![1],
            ..Default::default()
        },
    )
    .solve(&LobpcgConfig {
        num_eigenvalues: 4,
        max_iterations: 600,
        tolerance: 1.0e-8,
        block_padding: 4,
    })
    .expect("Sonium should solve the image-mask Dirichlet spectrum")
    .eigenvalues;
    eigenvalues.sort_by(f64::total_cmp);
    eigenvalues
}

fn solve_avoided_crossing() -> Vec<Vec<f64>> {
    let mut previous = None;
    let mut spectra = Vec::new();
    for separation in [0.65, 0.8, 0.95] {
        let mesh = two_disk_union_mesh_triangles(
            (-separation / 2.0, 0.0),
            (separation / 2.0, 0.0),
            0.72,
            0.58,
            64,
            48,
            1,
        );
        let problem = build_scalar_eigenproblem(
            &mesh,
            |_, _, _| 1.0,
            |_, _, _| 0.0,
            |_, _, _| 1.0,
            &ScalarEigenproblemConfig {
                dirichlet_boundary_markers: vec![1],
                ..Default::default()
            },
        );
        let result = problem
            .solve(&LobpcgConfig {
                num_eigenvalues: 4,
                max_iterations: 600,
                tolerance: 1.0e-8,
                block_padding: 4,
            })
            .expect("Sonium should solve each moving two-disk region");
        let full_modes = expand_homogeneous_dirichlet_modes(&mesh, &[1], &result.eigenvectors)
            .expect("Dirichlet eigenvectors should expand to nodal fields");
        if let Some((previous_mesh, previous_modes)) = &previous {
            let assignment = track_real_modes_across_meshes(
                previous_mesh,
                previous_modes,
                &mesh,
                &full_modes,
                41,
            )
            .expect("moving-domain modes should have a common sampling region");
            assert_eq!(assignment.current_indices.len(), 4);
            assert!(assignment.common_samples > 500);
            assert!(assignment.overlaps.iter().all(|overlap| *overlap > 0.25));
        }
        let mut eigenvalues = result.eigenvalues;
        eigenvalues.sort_by(f64::total_cmp);
        spectra.push(eigenvalues);
        previous = Some((mesh, full_modes));
    }
    spectra
}

fn solve_l_shape() -> Vec<f64> {
    let mesh = polygon_mesh_triangles(
        &[
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        ],
        4,
        1,
    )
    .expect("Sonium should triangulate the concave L-shaped region");
    let mut eigenvalues = build_scalar_eigenproblem(
        &mesh,
        |_, _, _| 1.0,
        |_, _, _| 0.0,
        |_, _, _| 1.0,
        &ScalarEigenproblemConfig {
            dirichlet_boundary_markers: vec![1],
            ..Default::default()
        },
    )
    .solve(&LobpcgConfig {
        num_eigenvalues: 4,
        max_iterations: 500,
        tolerance: 1.0e-8,
        block_padding: 4,
    })
    .expect("Sonium should solve the L-shaped Dirichlet spectrum")
    .eigenvalues;
    eigenvalues.sort_by(f64::total_cmp);
    eigenvalues
}

fn solve_coupled_system() -> Complex64 {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.8, 20, 16);
    let spatial = build_scalar_eigenproblem(
        &mesh,
        |_, _, _| 1.0,
        |_, _, _| 0.0,
        |_, _, _| 1.0,
        &ScalarEigenproblemConfig {
            dirichlet_boundary_markers: vec![1, 2, 3, 4],
            ..Default::default()
        },
    );
    build_coupled_reaction_diffusion_eigenproblem(&spatial, 2.0, 20.0, -20.0)
        .expect("Sonium should assemble the coupled nonsymmetric PDE pencil")
        .solve_near(&ComplexArnoldiConfig {
            target: Complex64::new(38.0, 15.5),
            num_eigenvalues: 1,
            krylov_dimension: 24,
            tolerance: 1.0e-6,
            max_qr_iterations: 8_000,
            breakdown_tolerance: 1.0e-14,
        })
        .expect("Sonium should solve the coupled nonsymmetric PDE pencil")
        .eigenvalues[0]
}

fn solve_damped_wave() -> Vec<Complex64> {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.8, 20, 16);
    let spatial = build_scalar_eigenproblem(
        &mesh,
        |_, _, _| 1.0,
        |_, _, _| 0.0,
        |_, _, _| 1.0,
        &ScalarEigenproblemConfig {
            dirichlet_boundary_markers: vec![1, 2, 3, 4],
            ..Default::default()
        },
    );
    let damping = CsrMatrix::from_triplets(
        spatial.mass_like.num_rows,
        spatial.mass_like.num_cols,
        (0..spatial.mass_like.num_rows)
            .flat_map(|row| {
                spatial
                    .mass_like
                    .row_entries(row)
                    .map(move |(column, value)| (row, column, 0.2 * value))
            })
            .collect(),
    );
    let quadratic =
        linearize_quadratic_eigenproblem(&spatial.stiffness_like, &damping, &spatial.mass_like)
            .expect("Sonium should linearize the damped-wave pencil");
    [5.5, -5.5]
        .into_iter()
        .map(|imaginary_target| {
            quadratic
                .solve_near(&ComplexArnoldiConfig {
                    target: Complex64::new(-0.1, imaginary_target),
                    num_eigenvalues: 1,
                    krylov_dimension: 24,
                    tolerance: 1.0e-6,
                    max_qr_iterations: 8_000,
                    breakdown_tolerance: 1.0e-14,
                })
                .expect("Sonium should solve the general damped-wave block pencil")
                .eigenvalues[0]
        })
        .collect()
}

fn solve_radial_hydrogen() -> Vec<f64> {
    let problem = build_radial_schrodinger_problem(&RadialSchrodingerConfig::default());
    let mut energies = problem
        .solve_interval(
            -0.55,
            -0.02,
            &ShiftInvertConfig {
                target: -0.2,
                num_eigenvalues: 10,
                krylov_dimension: 48,
                tolerance: 1.0e-8,
                breakdown_tolerance: 1.0e-14,
            },
        )
        .expect("Sonium should find the radial hydrogen bound states")
        .eigenvalues;
    energies.sort_by(f64::total_cmp);
    energies
}

fn solve_interval() -> Vec<f64> {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.8, 24, 20);
    let problem = build_scalar_eigenproblem(
        &mesh,
        |_, _, _| 1.0,
        |_, _, _| 5.0,
        |_, _, _| 1.0,
        &ScalarEigenproblemConfig {
            dirichlet_boundary_markers: vec![1, 2, 3, 4],
            ..Default::default()
        },
    );
    let mut result = problem
        .solve_interval(
            70.0,
            112.0,
            &ShiftInvertConfig {
                target: 91.0,
                num_eigenvalues: 8,
                krylov_dimension: 36,
                tolerance: 1.0e-8,
                breakdown_tolerance: 1.0e-14,
            },
        )
        .expect("Sonium should solve the requested spectral interval")
        .eigenvalues;
    result.sort_by(f64::total_cmp);
    result
}

fn solve_periodic() -> Vec<f64> {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.8, 24, 20);
    let pairs = pair_marked_boundary_nodes(
        &mesh,
        1,
        2,
        PeriodicTransform {
            translation: [1.0, 0.0, 0.0],
            ..PeriodicTransform::IDENTITY
        },
        1.0e-12,
    )
    .expect("periodic rectangle boundaries should pair");
    let problem = build_scalar_eigenproblem(
        &mesh,
        |_, _, _| 1.0,
        |_, _, _| 5.0,
        |_, _, _| 1.0,
        &ScalarEigenproblemConfig {
            periodic_constraints: expand_node_periodic_constraints(
                &pairs,
                1,
                Complex64::new(1.0, 0.0),
            ),
            ..Default::default()
        },
    );
    problem
        .solve(&LobpcgConfig {
            num_eigenvalues: 3,
            max_iterations: 400,
            tolerance: 1.0e-8,
            block_padding: 2,
        })
        .expect("Sonium should solve the periodic scalar eigenproblem")
        .eigenvalues
}

fn solve_bloch() -> Vec<f64> {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.8, 24, 20);
    let pairs = pair_marked_boundary_nodes(
        &mesh,
        1,
        2,
        PeriodicTransform {
            translation: [1.0, 0.0, 0.0],
            ..PeriodicTransform::IDENTITY
        },
        1.0e-12,
    )
    .expect("Bloch rectangle boundaries should pair");
    let theta = std::f64::consts::PI / 3.0;
    let problem = try_build_complex_scalar_eigenproblem(
        &mesh,
        |_, _, _| 1.0,
        |_, _, _| 5.0,
        |_, _, _| 1.0,
        &ScalarEigenproblemConfig {
            periodic_constraints: expand_node_periodic_constraints(
                &pairs,
                1,
                Complex64::from_polar(1.0, theta),
            ),
            ..Default::default()
        },
    )
    .expect("Bloch constraints should produce a Hermitian pencil");
    problem
        .solve_eigenvalues(&LobpcgConfig {
            num_eigenvalues: 3,
            max_iterations: 600,
            tolerance: 1.0e-8,
            block_padding: 3,
        })
        .expect("Sonium should solve the complex Bloch eigenproblem")
        .eigenvalues
}

fn solve_quantum_bands() -> Vec<Vec<f64>> {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.8, 24, 20);
    let pairs = pair_marked_boundary_nodes(
        &mesh,
        1,
        2,
        PeriodicTransform {
            translation: [1.0, 0.0, 0.0],
            ..PeriodicTransform::IDENTITY
        },
        1.0e-12,
    )
    .expect("quantum-cell boundaries should pair");
    let mut previous = None;
    let mut bands = Vec::new();
    for theta in [0.0, std::f64::consts::FRAC_PI_2, std::f64::consts::PI] {
        let problem = try_build_complex_scalar_eigenproblem(
            &mesh,
            |_, _, _| 1.0,
            |x, _, _| 10.0 * (1.0 + (2.0 * std::f64::consts::PI * x).cos()),
            |_, _, _| 1.0,
            &ScalarEigenproblemConfig {
                periodic_constraints: expand_node_periodic_constraints(
                    &pairs,
                    1,
                    Complex64::from_polar(1.0, theta),
                ),
                ..Default::default()
            },
        )
        .expect("quantum Bloch constraints should produce a Hermitian pencil");
        let mut result = problem
            .solve_eigenvalues(&LobpcgConfig {
                num_eigenvalues: 3,
                max_iterations: 600,
                tolerance: 1.0e-8,
                block_padding: 3,
            })
            .expect("Sonium should solve each quantum band point");
        if let Some(prior) = &previous {
            result.track_from(prior, &problem.mass_like);
        }
        bands.push(result.eigenvalues.clone());
        previous = Some(result);
    }
    bands
}

fn solve_scalar(
    diffusion: impl Fn(f64, f64, f64) -> f64 + Send + Sync,
    potential: impl Fn(f64, f64, f64) -> f64 + Send + Sync,
    config: ScalarEigenproblemConfig,
) -> Vec<f64> {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.8, 24, 20);
    let problem = build_scalar_eigenproblem(&mesh, diffusion, potential, |_, _, _| 1.0, &config);
    let mut result = problem
        .solve(&LobpcgConfig {
            num_eigenvalues: 4,
            max_iterations: 500,
            tolerance: 1.0e-8,
            block_padding: 4,
        })
        .expect("Sonium should solve the scalar generalized eigenproblem");
    assert!(result.converged, "LOBPCG did not converge: {result:?}");
    result.eigenvalues.sort_by(f64::total_cmp);
    result.eigenvalues
}

fn wolfram_reference() -> Option<WolframScalarReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/scalar_eigenproblems.wls");
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
            eprintln!("Wolfram Engine not available; skipping live scalar comparisons");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram scalar examples failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram scalar output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn assert_spectrum_close(name: &str, sonium: &[f64], wolfram: &[f64], tolerance: f64) {
    assert_eq!(sonium.len(), wolfram.len());
    for (mode, (&actual, &reference)) in sonium.iter().zip(wolfram).enumerate() {
        let relative_error = (actual - reference).abs() / reference.abs().max(1.0);
        assert!(
            relative_error < tolerance,
            "{name} mode {}: Sonium={actual:.9}, Wolfram={reference:.9}, error={:.3}%",
            mode + 1,
            100.0 * relative_error
        );
    }
}

fn assert_complex_spectrum_close(
    name: &str,
    sonium: &[Complex64],
    wolfram: &[[f64; 2]],
    tolerance: f64,
) {
    assert_eq!(sonium.len(), wolfram.len());
    for (mode, (&actual, reference)) in sonium.iter().zip(wolfram).enumerate() {
        let reference = Complex64::new(reference[0], reference[1]);
        let relative_error = (actual - reference).norm() / reference.norm().max(1.0);
        assert!(
            relative_error < tolerance,
            "{name} root {}: Sonium={actual}, Wolfram={reference}, error={:.3}%",
            mode + 1,
            100.0 * relative_error
        );
    }
}

#[test]
fn sonium_reproduces_wolfram_scalar_eigenproblem_families() {
    let dirichlet = ScalarEigenproblemConfig {
        dirichlet_boundary_markers: vec![1, 2, 3, 4],
        ..Default::default()
    };
    let schrodinger = solve_scalar(|_, _, _| 1.0, |_, _, _| 5.0, dirichlet.clone());
    let sturm_liouville = solve_scalar(|x, _, _| 1.0 + x, |_, _, _| 0.0, dirichlet);
    let robin = solve_scalar(
        |_, _, _| 1.0,
        |_, _, _| 0.0,
        ScalarEigenproblemConfig {
            robin_stiffness_terms: (1..=4)
                .map(|marker| ScalarRobinTerm {
                    marker,
                    coefficient: 2.0,
                })
                .collect(),
            ..Default::default()
        },
    );
    let periodic = solve_periodic();
    let bloch = solve_bloch();
    let quantum_bands = solve_quantum_bands();
    let interval = solve_interval();
    let radial_hydrogen = solve_radial_hydrogen();
    let damped_wave = solve_damped_wave();
    let coupled_system = solve_coupled_system();
    let l_shape = solve_l_shape();
    let avoided_crossing = solve_avoided_crossing();
    let image_mask = solve_image_mask();
    let neumann = solve_neumann();
    let wave_heat_samples = solve_wave_heat_modal_samples();
    let interval_1d = solve_interval_1d();
    let partially_constrained = solve_scalar(
        |_, _, _| 1.0,
        |_, _, _| 0.0,
        ScalarEigenproblemConfig {
            dirichlet_boundary_markers: vec![1, 2],
            ..Default::default()
        },
    );

    let exact = [
        (1.0 + 1.0 / 0.8_f64.powi(2)) * std::f64::consts::PI.powi(2) + 5.0,
        (4.0 + 1.0 / 0.8_f64.powi(2)) * std::f64::consts::PI.powi(2) + 5.0,
        (1.0 + 4.0 / 0.8_f64.powi(2)) * std::f64::consts::PI.powi(2) + 5.0,
        (4.0 + 4.0 / 0.8_f64.powi(2)) * std::f64::consts::PI.powi(2) + 5.0,
    ];
    assert_spectrum_close("Schrodinger exact", &schrodinger, &exact, 0.02);
    let periodic_exact = [
        5.0,
        (std::f64::consts::PI / 0.8).powi(2) + 5.0,
        (2.0 * std::f64::consts::PI).powi(2) + 5.0,
    ];
    assert_spectrum_close("Periodic exact", &periodic, &periodic_exact, 0.01);
    let theta = std::f64::consts::PI / 3.0;
    let bloch_exact = [
        5.0 + theta.powi(2),
        5.0 + theta.powi(2) + (std::f64::consts::PI / 0.8).powi(2),
        5.0 + (theta - 2.0 * std::f64::consts::PI).powi(2),
    ];
    assert_spectrum_close("Bloch exact", &bloch, &bloch_exact, 0.015);
    assert_eq!(quantum_bands.len(), 3);
    let interval_exact = [
        (1.0 + 4.0 / 0.8_f64.powi(2)) * std::f64::consts::PI.powi(2) + 5.0,
        (4.0 + 4.0 / 0.8_f64.powi(2)) * std::f64::consts::PI.powi(2) + 5.0,
        (9.0 + 1.0 / 0.8_f64.powi(2)) * std::f64::consts::PI.powi(2) + 5.0,
    ];
    assert_spectrum_close("Interval exact", &interval, &interval_exact, 0.025);
    let hydrogen_exact = [-0.5, -0.125, -1.0 / 18.0, -0.03125];
    assert_spectrum_close(
        "Radial hydrogen exact",
        &radial_hydrogen,
        &hydrogen_exact,
        0.01,
    );
    let neumann_exact = [
        0.0,
        std::f64::consts::PI.powi(2),
        (std::f64::consts::PI / 0.8).powi(2),
        std::f64::consts::PI.powi(2) + (std::f64::consts::PI / 0.8).powi(2),
    ];
    assert_spectrum_close("Neumann exact", &neumann, &neumann_exact, 0.015);
    let interval_1d_exact = [1.0, 4.0, 9.0, 16.0].map(|mode| mode * std::f64::consts::PI.powi(2));
    assert_spectrum_close("1D interval exact", &interval_1d, &interval_1d_exact, 0.003);
    let partial_exact = [
        std::f64::consts::PI.powi(2),
        std::f64::consts::PI.powi(2) + (std::f64::consts::PI / 0.8).powi(2),
        (2.0 * std::f64::consts::PI).powi(2),
        (2.0 * std::f64::consts::PI).powi(2) + (std::f64::consts::PI / 0.8).powi(2),
    ];
    assert_spectrum_close(
        "Partially constrained exact",
        &partially_constrained,
        &partial_exact,
        0.02,
    );
    let x = 0.375;
    let y = 0.3;
    let first_shape = (std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y / 0.8).sin();
    let second_shape =
        (2.0 * std::f64::consts::PI * x).sin() * (std::f64::consts::PI * y / 0.8).sin();
    let first_eigenvalue = std::f64::consts::PI.powi(2) + (std::f64::consts::PI / 0.8).powi(2);
    let second_eigenvalue =
        (2.0 * std::f64::consts::PI).powi(2) + (std::f64::consts::PI / 0.8).powi(2);
    let modal_exact = [
        first_shape * (first_eigenvalue.sqrt() * 0.07).cos()
            + 0.25 * second_shape * (second_eigenvalue.sqrt() * 0.07).cos(),
        first_shape * (-0.2 * first_eigenvalue * 0.03).exp()
            + 0.25 * second_shape * (-0.2 * second_eigenvalue * 0.03).exp(),
    ];
    for (&actual, &expected) in wave_heat_samples.iter().zip(&modal_exact) {
        assert!((actual - expected).abs() < 0.015);
    }
    assert!(
        quantum_bands
            .iter()
            .flatten()
            .all(|value| value.is_finite() && *value > 0.0)
    );

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 11);
        assert_spectrum_close(
            "Wolfram Schrodinger exact",
            &reference.schrodinger,
            &reference.schrodinger_exact,
            0.01,
        );
        assert_spectrum_close("Schrodinger", &schrodinger, &reference.schrodinger, 0.02);
        assert_spectrum_close(
            "Sturm-Liouville",
            &sturm_liouville,
            &reference.sturm_liouville,
            0.025,
        );
        assert_spectrum_close("Robin", &robin, &reference.robin, 0.03);
        assert_spectrum_close("Periodic", &periodic, &reference.periodic, 0.015);
        assert_spectrum_close("Bloch", &bloch, &reference.bloch, 0.02);
        assert_spectrum_close("Interval", &interval, &reference.interval, 0.025);
        assert_spectrum_close(
            "Radial hydrogen",
            &radial_hydrogen,
            &reference.radial_hydrogen,
            0.01,
        );
        assert_complex_spectrum_close("Damped wave", &damped_wave, &reference.damped_wave, 0.025);
        assert_complex_spectrum_close(
            "Coupled nonsymmetric system",
            &[coupled_system],
            &[reference.coupled_system],
            0.025,
        );
        assert_spectrum_close("L-shaped region", &l_shape, &reference.l_shape, 0.04);
        assert_spectrum_close(
            "Image-mask region",
            &image_mask,
            &reference.image_mask,
            0.08,
        );
        assert_spectrum_close("Neumann rectangle", &neumann, &reference.neumann, 0.02);
        assert_spectrum_close("1D interval", &interval_1d, &reference.interval_1d, 0.004);
        assert_spectrum_close(
            "Partially constrained rectangle",
            &partially_constrained,
            &reference.partially_constrained,
            0.025,
        );
        for (name, actual, expected) in [
            (
                "Wave modal sample",
                wave_heat_samples[0],
                reference.wave_heat_samples[0],
            ),
            (
                "Heat modal sample",
                wave_heat_samples[1],
                reference.wave_heat_samples[1],
            ),
        ] {
            let error = (actual - expected).abs() / expected.abs().max(1.0);
            assert!(
                error < 0.025,
                "{name}: Sonium={actual:.9}, Wolfram={expected:.9}, error={:.3}%",
                100.0 * error
            );
        }
        assert_eq!(avoided_crossing.len(), reference.avoided_crossing.len());
        for (index, (sonium, wolfram)) in avoided_crossing
            .iter()
            .zip(&reference.avoided_crossing)
            .enumerate()
        {
            assert_spectrum_close(
                &format!("Moving two-disk region {index}"),
                sonium,
                wolfram,
                0.06,
            );
        }
        assert_eq!(reference.quantum_bands.len(), quantum_bands.len());
        for (index, (sonium, wolfram)) in quantum_bands
            .iter()
            .zip(&reference.quantum_bands)
            .enumerate()
        {
            let mut sonium_sorted = sonium.clone();
            sonium_sorted.sort_by(f64::total_cmp);
            let mut wolfram_sorted = wolfram.clone();
            wolfram_sorted.sort_by(f64::total_cmp);
            assert_spectrum_close(
                &format!("Quantum bands phase {index}"),
                &sonium_sorted,
                &wolfram_sorted,
                0.025,
            );
        }
    }
}
