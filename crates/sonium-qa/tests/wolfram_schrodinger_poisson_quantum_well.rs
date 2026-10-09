//! Self-consistent reduction of Wolfram's 1D Schrödinger--Poisson quantum well.
//!
//! The Schrödinger stage uses Sonium's public scalar generalized-eigenproblem
//! builder on a one-cell-thick 2D strip.  Homogeneous Neumann conditions on the
//! thin faces make its lowest transverse mode exactly the page's 1D problem.
//! The missing Schrödinger--Poisson driver is represented by a test-local 1D P1
//! Poisson solve; this is therefore an exact reduction, not validation of a
//! public coupled multiphysics API.
//!
//! Source: <https://reference.wolfram.com/language/PDEModels/tutorial/Physics/ModelCollection/SchrodingerPoisson1DQuantumWell.html?view=all>

use serde::Deserialize;
use sonium_fem::{
    mesh::generators::rectangular_mesh_triangles,
    physics::solid::eigenmodes::{
        ScalarEigenproblemConfig, build_scalar_eigenproblem, expand_homogeneous_dirichlet_modes,
    },
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::LobpcgConfig;
use std::{env, path::PathBuf, process::Command, sync::Arc};

const LENGTH_ANGSTROM: f64 = 500.0;
const WELL_LEFT_ANGSTROM: f64 = 200.0;
const WELL_RIGHT_ANGSTROM: f64 = 300.0;
const BARRIER_MEV: f64 = 0.66 * 1247.0 * 0.2;
const EFFECTIVE_MASS_RATIO: f64 = 0.066;
const RELATIVE_PERMITTIVITY: f64 = 13.18;
const DOPING_M3: f64 = 2.0e24;
const ELEMENTARY_CHARGE_C: f64 = 1.602_176_634e-19;
const VACUUM_PERMITTIVITY_F_M: f64 = 8.854_187_812_8e-12;
// hbar^2/(2 m_e) in meV Angstrom^2.
const FREE_ELECTRON_DIFFUSION_MEV_A2: f64 = 3_809.982_116_154_86;
const PUBLISHED_ENERGIES_MEV: [f64; 6] = [29.6252, 32.0612, 31.8061, 31.8236, 31.8224, 31.8225];

#[derive(Debug)]
struct CoupledResult {
    energies_mev: Vec<f64>,
    integrated_charge_c_m2: f64,
    wavefunction_norm: f64,
}

fn confinement_potential(x: f64) -> f64 {
    if (WELL_LEFT_ANGSTROM..=WELL_RIGHT_ANGSTROM).contains(&x) {
        0.0
    } else {
        BARRIER_MEV
    }
}

fn interpolate_nodal(values: &[f64], spacing: f64, x: f64) -> f64 {
    let coordinate = (x / spacing).clamp(0.0, (values.len() - 1) as f64);
    let left = coordinate.floor() as usize;
    if left + 1 == values.len() {
        return values[left];
    }
    let fraction = coordinate - left as f64;
    values[left] * (1.0 - fraction) + values[left + 1] * fraction
}

fn p1_squared_norm(values: &[f64], spacing: f64) -> f64 {
    values
        .windows(2)
        .map(|pair| spacing * (pair[0] * pair[0] + pair[0] * pair[1] + pair[1] * pair[1]) / 3.0)
        .sum()
}

fn solve_ground_state(nx: usize, electrostatic_energy_mev: &[f64]) -> (f64, Vec<f64>) {
    let spacing = LENGTH_ANGSTROM / nx as f64;
    let potential = Arc::new(electrostatic_energy_mev.to_vec());
    let mesh = rectangular_mesh_triangles(0.0, LENGTH_ANGSTROM, 0.0, 1.0, nx, 1);
    let problem = build_scalar_eigenproblem(
        &mesh,
        |_, _, _| FREE_ELECTRON_DIFFUSION_MEV_A2 / EFFECTIVE_MASS_RATIO,
        {
            let potential = Arc::clone(&potential);
            move |x, _, _| confinement_potential(x) + interpolate_nodal(&potential, spacing, x)
        },
        |_, _, _| 1.0,
        &ScalarEigenproblemConfig {
            dirichlet_boundary_markers: vec![1, 2],
            ..Default::default()
        },
    );
    let result = problem
        .solve(&LobpcgConfig {
            num_eigenvalues: 1,
            max_iterations: 1_000,
            tolerance: 1.0e-10,
            block_padding: 4,
        })
        .expect("Sonium should solve the quantum-well ground state");
    assert!(result.converged, "ground-state LOBPCG did not converge");

    let full_modes = expand_homogeneous_dirichlet_modes(&mesh, &[1, 2], &result.eigenvectors)
        .expect("Sonium should reconstruct the constrained wavefunction");
    assert_eq!(full_modes.nrows(), 2 * (nx + 1));
    let mut wavefunction = vec![0.0; nx + 1];
    for i in 0..=nx {
        wavefunction[i] = 0.5 * (full_modes[(i, 0)] + full_modes[((nx + 1) + i, 0)]);
    }
    let norm = p1_squared_norm(&wavefunction, spacing).sqrt();
    for value in &mut wavefunction {
        *value /= norm;
    }
    (result.eigenvalues[0], wavefunction)
}

fn charge_density_c_m3(x: f64, psi_per_sqrt_angstrom: f64) -> f64 {
    let well_width_m = (WELL_RIGHT_ANGSTROM - WELL_LEFT_ANGSTROM) * 1.0e-10;
    let sheet_density_m2 = DOPING_M3 * well_width_m;
    let electron_density_m3 = sheet_density_m2 * psi_per_sqrt_angstrom.powi(2) * 1.0e10;
    let donor_density_m3 = if (WELL_LEFT_ANGSTROM..=WELL_RIGHT_ANGSTROM).contains(&x) {
        DOPING_M3
    } else {
        0.0
    };
    -ELEMENTARY_CHARGE_C * (electron_density_m3 - donor_density_m3)
}

fn solve_poisson(nx: usize, wavefunction: &[f64]) -> (Vec<f64>, f64) {
    let spacing = LENGTH_ANGSTROM / nx as f64;
    let epsilon = VACUUM_PERMITTIVITY_F_M * RELATIVE_PERMITTIVITY;
    let unknowns = nx - 1;
    let lower = vec![-epsilon / spacing; unknowns.saturating_sub(1)];
    let mut diagonal = vec![2.0 * epsilon / spacing; unknowns];
    let upper = vec![-epsilon / spacing; unknowns.saturating_sub(1)];
    let mut rhs = vec![0.0; unknowns];
    let mut integrated_charge_c_m2 = 0.0;

    // Three-point Gauss integration is exact for N_i times the quadratic
    // charge density induced by a P1 wavefunction on each aligned element.
    let gauss = [
        (-f64::sqrt(3.0 / 5.0), 5.0 / 9.0),
        (0.0, 8.0 / 9.0),
        (f64::sqrt(3.0 / 5.0), 5.0 / 9.0),
    ];
    for element in 0..nx {
        let x0 = element as f64 * spacing;
        for (xi, weight) in gauss {
            let local = 0.5 * (xi + 1.0);
            let x = x0 + local * spacing;
            let psi = wavefunction[element] * (1.0 - local) + wavefunction[element + 1] * local;
            let rho = charge_density_c_m3(x, psi);
            let contribution = 0.5 * spacing * weight * rho;
            integrated_charge_c_m2 += contribution * 1.0e-10;
            if element > 0 {
                rhs[element - 1] += contribution * (1.0 - local) * 1.0e-20;
            }
            if element + 1 < nx {
                rhs[element] += contribution * local * 1.0e-20;
            }
        }
    }

    // Thomas solve for the test-local 1D P1 Poisson system, with phi=0 at
    // both endpoints.  Coordinates are Angstrom, hence the RHS factor 1e-20.
    for i in 1..unknowns {
        let factor = lower[i - 1] / diagonal[i - 1];
        diagonal[i] -= factor * upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }
    let mut phi_volts = vec![0.0; nx + 1];
    if unknowns > 0 {
        phi_volts[nx - 1] = rhs[unknowns - 1] / diagonal[unknowns - 1];
        for i in (0..unknowns - 1).rev() {
            phi_volts[i + 1] = (rhs[i] - upper[i] * phi_volts[i + 2]) / diagonal[i];
        }
    }
    (phi_volts, integrated_charge_c_m2)
}

fn solve_coupled(nx: usize) -> CoupledResult {
    let mut electrostatic_energy_mev = vec![0.0; nx + 1];
    let mut energies_mev = Vec::new();
    let mut integrated_charge_c_m2 = f64::NAN;
    let mut wavefunction_norm = f64::NAN;

    for _ in 0..20 {
        let (energy, wavefunction) = solve_ground_state(nx, &electrostatic_energy_mev);
        energies_mev.push(energy);
        wavefunction_norm = p1_squared_norm(&wavefunction, LENGTH_ANGSTROM / nx as f64);
        if energies_mev.len() >= 2
            && (energies_mev[energies_mev.len() - 1] - energies_mev[energies_mev.len() - 2]).abs()
                <= 1.0e-5
        {
            break;
        }
        let (phi_volts, charge) = solve_poisson(nx, &wavefunction);
        integrated_charge_c_m2 = charge;
        electrostatic_energy_mev = phi_volts.into_iter().map(|phi| -1_000.0 * phi).collect();
    }
    CoupledResult {
        energies_mev,
        integrated_charge_c_m2,
        wavefunction_norm,
    }
}

#[derive(Deserialize)]
struct WolframReference {
    schema_version: u32,
    energies_mev: Vec<f64>,
}

fn wolfram_reference() -> Option<WolframReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("wolfram/schrodinger_poisson_quantum_well.wls");
    let output = Command::new(executable)
        .arg("-file")
        .arg(script)
        .wolfram_output()
        .expect("run explicitly configured Wolfram quantum-well reference");
    assert!(
        output.status.success(),
        "WolframScript failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("Wolfram output should be UTF-8");
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .expect("Wolfram script should emit a JSON object");
    Some(serde_json::from_str(json).expect("Wolfram JSON should match the reference schema"))
}

#[test]
fn wolfram_schrodinger_poisson_quantum_well_exact_reduction() {
    let coarse = solve_coupled(125);
    let fine = solve_coupled(250);

    eprintln!(
        "Schrodinger--Poisson: coarse={:?}, fine={:?}, charge={:.3e} C/m^2, norm={:.12}",
        coarse.energies_mev, fine.energies_mev, fine.integrated_charge_c_m2, fine.wavefunction_norm
    );

    assert!(
        (PUBLISHED_ENERGIES_MEV.len()..=PUBLISHED_ENERGIES_MEV.len() + 1)
            .contains(&fine.energies_mev.len()),
        "unexpected convergence history: {:?}",
        fine.energies_mev
    );
    let max_published_error = fine
        .energies_mev
        .iter()
        .zip(PUBLISHED_ENERGIES_MEV)
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0_f64, f64::max);
    assert!(
        max_published_error < 0.12,
        "energy iteration differs from Wolfram: actual={:?}, max error={max_published_error:.6} meV",
        fine.energies_mev
    );
    assert!(
        (coarse.energies_mev.last().unwrap() - fine.energies_mev.last().unwrap()).abs() < 0.08,
        "coupled ground-state energy did not converge under mesh refinement"
    );
    assert!((fine.wavefunction_norm - 1.0).abs() < 1.0e-12);
    assert!(
        fine.integrated_charge_c_m2.abs() < 2.0e-16,
        "electron and donor sheet charges should cancel: {} C/m^2",
        fine.integrated_charge_c_m2
    );

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.energies_mev.len(), PUBLISHED_ENERGIES_MEV.len());
        for (actual, expected) in fine
            .energies_mev
            .iter()
            .take(reference.energies_mev.len())
            .zip(reference.energies_mev)
        {
            assert!((actual - expected).abs() < 0.12);
        }
    }
}
