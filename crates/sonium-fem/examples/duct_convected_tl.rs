//! Convected duct transmission loss with mean flow
//!
//! Demonstrates the Phase 2 aeroacoustics pipeline:
//! 1. Define a duct geometry (expansion chamber)
//! 2. Compute quasi-1D mean flow through the duct
//! 3. Assemble the axisymmetric convected Helmholtz system
//! 4. Solve at BPF harmonics of a drone propeller
//! 5. Compare TL with and without mean flow
//!
//! Run with: `cargo run -p sonium-fem --example duct_convected_tl`

use num_complex::Complex64;
use sonium_fem::assembly::MassMatrix;
use sonium_fem::assembly::assemble_axi_boundary_mass;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::duct_profile_mesh;
use sonium_fem::physics::acoustics::aeroacoustics::duct_flow::DuctFlowField;
use sonium_fem::physics::acoustics::aeroacoustics::{
    AxiLpfAssemblerExt, MeanFlow, PropellerSource, build_axi_lpf_assembler_with_boundaries,
};
use sonium_fem::physics::acoustics::pressure::AcousticsAssemblerExt;
use sonium_fem::physics::acoustics::pressure::axisymmetric::build_axi_acoustics_assembler_with_boundaries;
use sonium_solvers::{FgmresConfig, IluPreconditioner, fgmres_preconditioned};
use std::collections::HashMap;
use std::f64::consts::PI;

fn boundary_mean_p2(b_mass: &MassMatrix, solution: &[Complex64]) -> f64 {
    let mut integral = 0.0;
    let mut area = 0.0;
    for k in 0..b_mass.nnz() {
        let i = b_mass.rows[k];
        let j = b_mass.cols[k];
        integral += (solution[i].conj() * b_mass.values[k] * solution[j]).re;
        if i == j {
            area += b_mass.values[k];
        }
    }
    if area > 0.0 { integral / area } else { 0.0 }
}

fn main() {
    println!("=== Convected Duct Transmission Loss (Phase 2) ===\n");

    // Propeller definition
    let prop = PropellerSource::drone_5inch(8000.0);
    let bpf = prop.blade_passing_frequency();
    println!("Propeller: {} blades, {} RPM", prop.n_blades, prop.rpm);
    println!("BPF: {:.0} Hz, tip Mach: {:.3}", bpf, prop.tip_mach());

    let r1 = 0.07; // inlet/outlet radius (propeller + clearance)
    let amps = (1..=6)
        .map(|h| prop.inlet_pressure_amplitude_for_area(h, PI * r1 * r1))
        .collect::<Vec<_>>();
    println!("Equivalent inlet peak-pressure amplitudes [Pa]:");
    for (i, a) in amps.iter().enumerate() {
        println!(
            "  H{}: {:.4e} Pa ({:.1} dB re 20µPa)",
            i + 1,
            a,
            20.0 * (a / 20e-6).log10()
        );
    }

    // Geometry
    let r2 = 0.10; // chamber radius
    let l_inlet = 0.05;
    let l_chamber = 0.08;
    let l_outlet = 0.05;
    let l_total = l_inlet + l_chamber + l_outlet;

    let wall_r = move |z: f64| -> f64 {
        if (l_inlet - 1e-10..=l_inlet + l_chamber + 1e-10).contains(&z) {
            r2
        } else {
            r1
        }
    };

    let nz = 60;
    let nr = 10;
    let mesh = duct_profile_mesh(0.0, l_total, |_| 0.0, wall_r, nz, nr);
    println!(
        "\nMesh: {} nodes, {} elements",
        mesh.num_nodes(),
        mesh.num_elements()
    );

    // Mean flow: M_inlet = 0.05 (light throughflow from propeller)
    let mach_inlet = 0.05;
    let reference = MeanFlow::air_standard([0.0, 0.0, 0.0]);
    let duct_flow = DuctFlowField::from_area_profile(
        reference,
        mach_inlet,
        |z| PI * wall_r(z) * wall_r(z),
        0.0,
        l_total,
        200,
    );
    println!("Mean flow: M_inlet = {mach_inlet}");
    println!("  M at inlet:   {:.4}", duct_flow.mach_at(0.0));
    println!(
        "  M at chamber: {:.4}",
        duct_flow.mach_at(l_inlet + l_chamber / 2.0)
    );
    println!("  M at outlet:  {:.4}", duct_flow.mach_at(l_total));

    // Build assemblers
    let flow_uniform = MeanFlow::air_standard([0.0, mach_inlet * 343.0, 0.0]); // uniform approx

    let assembler_no_flow =
        build_axi_acoustics_assembler_with_boundaries(&mesh, PolynomialDegree::P1, &[1, 2]);
    let assembler_with_flow = build_axi_lpf_assembler_with_boundaries(
        &mesh,
        PolynomialDegree::P1,
        &flow_uniform,
        &[1, 2],
    );

    let b_mass_inlet = assemble_axi_boundary_mass(&mesh, PolynomialDegree::P1, 1);
    let b_mass_outlet = assemble_axi_boundary_mass(&mesh, PolynomialDegree::P1, 2);

    let c = 343.0;
    let _rho = 1.204;

    let config = FgmresConfig {
        max_iterations: 500,
        restart: 50,
        tolerance: 1e-8,
        print_interval: 0,
        convergence_check_interval: 1,
        exact_residual_check_interval: 0,
    };

    // Solve at BPF harmonics
    println!(
        "\n{:>6} {:>8} {:>10} {:>10} {:>10}",
        "Harm.", "f (Hz)", "TL_noflow", "TL_flow", "Δ TL"
    );
    println!("{:-<50}", "");

    for h in 1..=6 {
        let freq = prop.harmonic_frequency(h);
        let omega = 2.0 * PI * freq;
        let k_val = omega / c;
        let k = Complex64::new(k_val, 0.0);

        let robin = Complex64::new(0.0, k_val);
        let mut bc = HashMap::new();
        bc.insert(2, robin);

        // Convert the incident peak-pressure phasor to the equivalent
        // Helmholtz Neumann datum: i*omega*rho*(p/(rho*c)) = i*k*p.
        let flux = Complex64::new(0.0, k_val) * amps[h - 1];
        let n = mesh.num_nodes();

        // --- No-flow solve ---
        let system_nf = assembler_no_flow.assemble_helmholtz(k, &bc);
        let precond_nf = IluPreconditioner::from_csr(&system_nf);

        let mut rhs = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..b_mass_inlet.nnz() {
            rhs[b_mass_inlet.rows[i]] += flux * b_mass_inlet.values[i];
        }
        let result_nf = fgmres_preconditioned(&system_nf, &precond_nf, &rhs, None, &config);

        let tl_nf = if result_nf.converged {
            let p2_in = boundary_mean_p2(&b_mass_inlet, &result_nf.x);
            let p2_out = boundary_mean_p2(&b_mass_outlet, &result_nf.x);
            if p2_out > 1e-30 {
                10.0 * (p2_in / p2_out).log10()
            } else {
                f64::INFINITY
            }
        } else {
            f64::NAN
        };

        // --- With-flow solve ---
        let system_wf =
            assembler_with_flow.assemble_axi_convected_helmholtz(&flow_uniform, omega, &bc);
        let precond_wf = IluPreconditioner::from_csr(&system_wf);

        let mut rhs = vec![Complex64::new(0.0, 0.0); n];
        for i in 0..b_mass_inlet.nnz() {
            rhs[b_mass_inlet.rows[i]] += flux * b_mass_inlet.values[i];
        }
        let result_wf = fgmres_preconditioned(&system_wf, &precond_wf, &rhs, None, &config);

        let tl_wf = if result_wf.converged {
            let p2_in = boundary_mean_p2(&b_mass_inlet, &result_wf.x);
            let p2_out = boundary_mean_p2(&b_mass_outlet, &result_wf.x);
            if p2_out > 1e-30 {
                10.0 * (p2_in / p2_out).log10()
            } else {
                f64::INFINITY
            }
        } else {
            f64::NAN
        };

        let delta = tl_wf - tl_nf;
        let conv_nf = if result_nf.converged { "✓" } else { "✗" };
        let conv_wf = if result_wf.converged { "✓" } else { "✗" };

        println!(
            "  H{h}   {:>7.0} {:>8.1} {conv_nf} {:>8.1} {conv_wf} {:>+8.1}",
            freq, tl_nf, tl_wf, delta
        );
    }

    println!("\nPhase 2 complete: axisymmetric convected Helmholtz with propeller source.");
}
