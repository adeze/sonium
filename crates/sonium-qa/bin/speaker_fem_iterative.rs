//! Production iterative-sweep QA for the distributed speaker FEM model.

use anyhow::{Context, ensure};
use num_complex::Complex64;
use sonium_sim_speaker_modeling::{AxisymmetricLinearConfig, AxisymmetricLinearModel};

fn run() -> anyhow::Result<()> {
    let model = AxisymmetricLinearModel::build(AxisymmetricLinearConfig {
        magnetic_radial_cells: 8,
        magnetic_axial_cells: 8,
        structural_radial_cells: 6,
        structural_axial_cells: 1,
        acoustic_axial_cells: 4,
        ..AxisymmetricLinearConfig::default()
    })
    .context("build distributed axisymmetric speaker model")?;
    let frequencies_hz = [100.0, 110.0, 200.0, 500.0, 1_000.0];
    let solutions = model
        .solve_sweep(&frequencies_hz, Complex64::new(2.83, 0.0))
        .context("solve iterative electroacoustic frequency sweep")?;
    let qa = model
        .audit_sweep(&solutions)
        .context("audit frequency sweep")?;

    ensure!(
        qa.passes(1.0e-8, 1.0e-6),
        "speaker FEM iterative QA failed: {qa:?}"
    );
    ensure!(
        solutions
            .iter()
            .skip(1)
            .any(|point| point.solver.initial_guess_reused),
        "frequency continuation never reused a solution"
    );
    ensure!(
        solutions
            .iter()
            .any(|point| !point.solver.preconditioner_rebuilt),
        "nearby sweep points never reused block factors"
    );

    println!("Speaker FEM iterative QA");
    println!("  points: {}", solutions.len());
    println!(
        "  maximum linear residual: {:.3e}",
        qa.max_relative_linear_residual
    );
    println!(
        "  maximum power residual: {:.3e}",
        qa.max_relative_power_residual
    );
    println!(
        "  maximum reciprocity defect: {:.3e}",
        qa.max_reciprocity_defect
    );
    println!("  minimum Re(Z): {:.6} ohm", qa.minimum_impedance_real_ohm);
    println!("SUCCESS");
    Ok(())
}

fn main() {
    if let Err(error) = run() {
        eprintln!("FAILED: {error:#}");
        std::process::exit(1);
    }
}
