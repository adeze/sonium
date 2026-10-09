//! Axisymmetric duct transmission loss — two-microphone method
//!
//! Validates the axisymmetric Helmholtz solver against the analytical
//! expansion chamber TL formula using the **transfer-matrix method**:
//! decompose the pressure field into incident + reflected plane waves
//! in the inlet, and transmitted wave in the outlet.
//!
//! This avoids the absorbing BC reflection contamination that plagues
//! the simpler power-ratio approach.
//!
//! Run with: `cargo run -p sonium-fem --release --example duct_transmission_loss`

use num_complex::Complex64;
use sonium_fem::assembly::assemble_axi_boundary_mass;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::{Mesh, duct_profile_mesh};
use sonium_fem::physics::acoustics::pressure::AcousticsAssemblerExt;
use sonium_fem::physics::acoustics::pressure::axisymmetric::build_axi_acoustics_assembler_with_boundaries;
use sonium_solvers::{FgmresConfig, IluPreconditioner, fgmres_preconditioned};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Average pressure over nodes near a given axial station z_target.
/// Uses nodes in the inlet/outlet tube (r < r_tube * 1.01) and within
/// dz tolerance of z_target.
fn average_pressure_at_z(
    mesh: &Mesh,
    solution: &[Complex64],
    z_target: f64,
    r_tube: f64,
    dz_tol: f64,
) -> Complex64 {
    let mut sum = Complex64::new(0.0, 0.0);
    let mut count = 0;
    for (i, node) in mesh.nodes.iter().enumerate() {
        let r = node.x;
        let z = node.y;
        if (z - z_target).abs() < dz_tol && r < r_tube * 1.01 {
            sum += solution[i];
            count += 1;
        }
    }
    if count > 0 {
        sum / count as f64
    } else {
        Complex64::new(0.0, 0.0)
    }
}

/// Two-microphone decomposition: given pressure at z1 and z2 in a straight
/// duct with wavenumber k, decompose into incident (A) and reflected (B):
///   p(z) = A * exp(-ikz) + B * exp(+ikz)
///
/// Returns (|A|², |B|²) — incident and reflected power (proportional).
fn two_mic_decompose(p1: Complex64, p2: Complex64, z1: f64, z2: f64, k: f64) -> (f64, f64) {
    let e1p = Complex64::new(0.0, k * z1).exp(); // exp(+ikz1)
    let e1m = Complex64::new(0.0, -k * z1).exp(); // exp(-ikz1)
    let e2p = Complex64::new(0.0, k * z2).exp();
    let e2m = Complex64::new(0.0, -k * z2).exp();

    // [e1m  e1p] [A]   [p1]
    // [e2m  e2p] [B] = [p2]
    let det = e1m * e2p - e1p * e2m;
    if det.norm() < 1e-30 {
        return (0.0, 0.0);
    }
    let a = (e2p * p1 - e1p * p2) / det;
    let b = (e1m * p2 - e2m * p1) / det;

    (a.norm_sqr(), b.norm_sqr())
}

fn main() {
    println!("=== Axisymmetric Duct TL (Two-Microphone Method) ===\n");

    // Geometry
    let r1 = 0.025;
    let r2 = 0.050;
    let l_taper = 0.03;
    let l_chamber = 0.15;
    let l_inlet = 0.15;
    let l_outlet = 0.15;
    let l_total = l_inlet + l_taper + l_chamber + l_taper + l_outlet;

    let m = (r2 * r2) / (r1 * r1);
    let c = 343.0;
    let rho = 1.204;
    let f_cutoff = 1.841 * c / (2.0 * PI * r2);

    println!(
        "r1 = {:.0} mm, r2 = {:.0} mm, m = {:.1}",
        r1 * 1e3,
        r2 * 1e3,
        m
    );
    println!(
        "L_chamber = {:.0} mm, L_taper = {:.0} mm",
        l_chamber * 1e3,
        l_taper * 1e3
    );
    println!("f_cutoff = {:.0} Hz\n", f_cutoff);

    let z_exp_start = l_inlet;
    let z_exp_end = l_inlet + l_taper;
    let z_con_start = l_inlet + l_taper + l_chamber;
    let z_con_end = l_inlet + 2.0 * l_taper + l_chamber;

    let wall_r = move |z: f64| -> f64 {
        if z <= z_exp_start {
            r1
        } else if z <= z_exp_end {
            let t = (z - z_exp_start) / l_taper;
            r1 + 0.5 * (r2 - r1) * (1.0 - (PI * t).cos())
        } else if z <= z_con_start {
            r2
        } else if z <= z_con_end {
            let t = (z - z_con_start) / l_taper;
            r2 - 0.5 * (r2 - r1) * (1.0 - (PI * t).cos())
        } else {
            r1
        }
    };

    // Two-mic positions: well inside the straight inlet/outlet sections
    // Spacing s = 0.03m — valid for f > c/(2*pi*s) ~ 1820 Hz; we use wider
    // spacing for low frequencies.
    let s = 0.04; // mic spacing
    let z_mic1 = l_inlet * 0.3; // inlet mic 1
    let z_mic2 = l_inlet * 0.3 + s; // inlet mic 2
    let z_mic3 = l_total - l_outlet * 0.3 - s; // outlet mic 1
    let z_mic4 = l_total - l_outlet * 0.3; // outlet mic 2

    let nz = 200;
    let nr = 16;
    let mesh = duct_profile_mesh(0.0, l_total, |_| 0.0, wall_r, nz, nr);
    let dz = l_total / nz as f64;
    println!(
        "Mesh: {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );
    println!(
        "Mic positions: z1={:.1}mm z2={:.1}mm | z3={:.1}mm z4={:.1}mm\n",
        z_mic1 * 1e3,
        z_mic2 * 1e3,
        z_mic3 * 1e3,
        z_mic4 * 1e3
    );

    let assembler =
        build_axi_acoustics_assembler_with_boundaries(&mesh, PolynomialDegree::P1, &[1, 2]);
    let b_mass_inlet = assemble_axi_boundary_mass(&mesh, PolynomialDegree::P1, 1);

    let config = FgmresConfig {
        max_iterations: 500,
        restart: 60,
        tolerance: 1e-8,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
    };

    let f_max = f_cutoff * 0.5;
    let n_freq = 30;
    let frequencies: Vec<f64> = (1..=n_freq)
        .map(|i| 100.0 + (i - 1) as f64 * (f_max - 100.0) / (n_freq - 1) as f64)
        .collect();

    let mut max_error = 0.0_f64;
    let mut n_under_3db = 0;
    let mut n_converged = 0;

    println!(
        "{:>8} {:>10} {:>10} {:>10}",
        "f (Hz)", "TL_ana", "TL_fem", "err (dB)"
    );
    println!("{:-<42}", "");

    for &freq in &frequencies {
        let omega = 2.0 * PI * freq;
        let k_val = omega / c;
        let k = Complex64::new(k_val, 0.0);

        let mut bc = HashMap::new();
        bc.insert(2, Complex64::new(0.0, k_val));

        let system = assembler.assemble_helmholtz(k, &bc);
        let precond_mat = assembler.assemble_shifted_laplacian(1.0, -0.5 * k_val * k_val);
        let precond = IluPreconditioner::from_csr(&precond_mat);

        let flux = Complex64::new(0.0, omega * rho);
        let n = mesh.num_nodes();
        let mut rhs = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..b_mass_inlet.nnz() {
            rhs[b_mass_inlet.rows[i]] += flux * b_mass_inlet.values[i];
        }

        let result = fgmres_preconditioned(&system, &precond, &rhs, None, &config);
        if !result.converged {
            eprintln!(
                "  f={freq:.0} Hz: no convergence (res={:.1e})",
                result.residual
            );
            continue;
        }
        n_converged += 1;

        // Two-mic decomposition in inlet and outlet
        let p1 = average_pressure_at_z(&mesh, &result.x, z_mic1, r1, dz * 0.6);
        let p2 = average_pressure_at_z(&mesh, &result.x, z_mic2, r1, dz * 0.6);
        let p3 = average_pressure_at_z(&mesh, &result.x, z_mic3, r1, dz * 0.6);
        let p4 = average_pressure_at_z(&mesh, &result.x, z_mic4, r1, dz * 0.6);

        let (a_sq_in, _b_sq_in) = two_mic_decompose(p1, p2, z_mic1, z_mic2, k_val);
        let (a_sq_out, _b_sq_out) = two_mic_decompose(p3, p4, z_mic3, z_mic4, k_val);

        // TL = 10*log10(|A_in|² / |A_out|²) — incident power ratio
        let tl_fem = if a_sq_out > 1e-30 {
            10.0 * (a_sq_in / a_sq_out).log10()
        } else {
            f64::INFINITY
        };

        let tl_ana =
            10.0 * (1.0 + 0.25 * (m - 1.0 / m).powi(2) * (k_val * l_chamber).sin().powi(2)).log10();

        let error = (tl_fem - tl_ana).abs();
        max_error = max_error.max(error);
        if error < 3.0 {
            n_under_3db += 1;
        }

        println!(
            "{:>8.0} {:>10.2} {:>10.2} {:>10.2}",
            freq, tl_ana, tl_fem, error
        );
    }

    println!("\n{n_converged}/{n_freq} converged, {n_under_3db}/{n_converged} within 3 dB");
    println!("Max error: {:.2} dB", max_error);
    let pass_rate = n_under_3db as f64 / n_converged.max(1) as f64;
    if pass_rate > 0.8 {
        println!("PASS: {:.0}% within 3 dB", pass_rate * 100.0);
    } else {
        println!("Pass rate: {:.0}%", pass_rate * 100.0);
    }
}
