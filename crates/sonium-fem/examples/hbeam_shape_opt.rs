//! H-beam cross-section shape optimisation via adjoint Lagrangian
//!
//! Finds the optimal H-section parameters (b, h, t) that minimise tip deflection
//! of a cantilever beam under a point load, subject to a material-budget constraint.
//!
//! Usage:
//!   cargo run -p sonium-fem --features cli --example hbeam_shape_opt
//!   cargo run -p sonium-fem --features cli --example hbeam_shape_opt -- --help
//!   cargo run -p sonium-fem --features cli --example hbeam_shape_opt -- --tip-load 5000 --area-max 0.006
//!   cargo run -p sonium-fem --features cli --example hbeam_shape_opt -- --b0 0.03 --h0 0.08 --t0 0.004

use clap::Parser;
use sonium_fem::physics::solid::beam::HSection;
use sonium_fem::physics::solid::beam::shape_opt::HBeamShapeOpt;

#[derive(Parser)]
#[command(name = "hbeam-shape-opt")]
#[command(about = "Optimise H-beam cross-section to minimise cantilever tip deflection")]
struct Cli {
    /// Beam span length (m)
    #[arg(long, default_value_t = 1.0)]
    length: f64,

    /// Number of beam elements
    #[arg(long, default_value_t = 20)]
    n_elements: usize,

    /// Young's modulus (Pa)
    #[arg(long, default_value_t = 200e9)]
    youngs_modulus: f64,

    /// Density (kg/m³)
    #[arg(long, default_value_t = 7800.0)]
    density: f64,

    /// Tip load, positive downward (N)
    #[arg(long, default_value_t = 1000.0)]
    tip_load: f64,

    /// Maximum cross-sectional area budget (m²)
    #[arg(long, default_value_t = 0.004)]
    area_max: f64,

    // --- Design variable bounds ---
    /// Minimum flange width (m)
    #[arg(long, default_value_t = 0.01)]
    b_min: f64,
    /// Maximum flange width (m)
    #[arg(long, default_value_t = 0.10)]
    b_max: f64,

    /// Minimum web height (m)
    #[arg(long, default_value_t = 0.02)]
    h_min: f64,
    /// Maximum web height (m)
    #[arg(long, default_value_t = 0.20)]
    h_max: f64,

    /// Minimum thickness (m)
    #[arg(long, default_value_t = 0.002)]
    t_min: f64,
    /// Maximum thickness (m)
    #[arg(long, default_value_t = 0.015)]
    t_max: f64,

    // --- Initial guess ---
    /// Initial flange width (m)
    #[arg(long, default_value_t = 0.04)]
    b0: f64,
    /// Initial web height (m)
    #[arg(long, default_value_t = 0.06)]
    h0: f64,
    /// Initial thickness (m)
    #[arg(long, default_value_t = 0.008)]
    t0: f64,
}

fn main() {
    let cli = Cli::parse();

    println!("=== H-Beam Shape Optimisation ===");
    println!();
    println!(
        "Cantilever beam:  L = {:.3} m,  {} elements",
        cli.length, cli.n_elements
    );
    println!(
        "Material:         E = {:.1e} Pa,  rho = {:.0} kg/m³",
        cli.youngs_modulus, cli.density
    );
    println!("Tip load:         P = {:.1} N (downward)", cli.tip_load);
    println!("Area budget:      A_max = {:.6} m²", cli.area_max);
    println!();
    println!("Design bounds:");
    println!("  b (flange width): [{:.4}, {:.4}] m", cli.b_min, cli.b_max);
    println!("  h (web height):   [{:.4}, {:.4}] m", cli.h_min, cli.h_max);
    println!("  t (thickness):    [{:.4}, {:.4}] m", cli.t_min, cli.t_max);
    println!();

    let initial = HSection::new(cli.b0, cli.h0, cli.t0);
    println!(
        "Initial section:  b = {:.4}, h = {:.4}, t = {:.4}",
        initial.b, initial.h, initial.t
    );
    println!(
        "  A = {:.6} m²,  I = {:.6e} m⁴",
        initial.area(),
        initial.moment_of_inertia()
    );

    let problem = HBeamShapeOpt {
        length: cli.length,
        n_elements: cli.n_elements,
        youngs_modulus: cli.youngs_modulus,
        density: cli.density,
        tip_load: cli.tip_load,
        area_max: cli.area_max,
        bounds: [
            (cli.b_min, cli.b_max),
            (cli.h_min, cli.h_max),
            (cli.t_min, cli.t_max),
        ],
    };

    // Evaluate initial design
    let (initial_defl, initial_grad) = problem.evaluate(&[initial.b, initial.h, initial.t]);
    println!("  tip deflection = {:.6e} m", initial_defl);
    println!(
        "  grad = [{:.4e}, {:.4e}, {:.4e}]",
        initial_grad[0], initial_grad[1], initial_grad[2]
    );
    println!();

    // Run optimisation
    println!("--- Running adjoint-based optimisation ---");
    let result = problem.optimise(&initial);

    println!();
    println!("=== Result ===");
    println!("Converged:        {}", result.optim.converged);
    println!("Iterations:       {}", result.optim.iterations);
    println!("Fun evals:        {}", result.optim.fun_evals);
    println!("Grad evals:       {}", result.optim.grad_evals);
    println!();
    println!(
        "Optimal section:  b = {:.6}, h = {:.6}, t = {:.6}",
        result.section.b, result.section.h, result.section.t
    );
    println!("  A = {:.6} m²  (budget: {:.6})", result.area, cli.area_max);
    println!("  I = {:.6e} m⁴", result.section.moment_of_inertia());
    println!("  tip deflection = {:.6e} m", result.tip_deflection);
    println!();

    let improvement = (1.0 - result.tip_deflection / initial_defl) * 100.0;
    println!("Deflection reduction: {:.1}% vs initial", improvement);

    // Analytical reference: PL³ / (3EI)
    let analytical = cli.tip_load * cli.length.powi(3)
        / (3.0 * cli.youngs_modulus * result.section.moment_of_inertia());
    println!(
        "Euler-Bernoulli check: {:.6e} m  (FEM: {:.6e})",
        analytical, result.tip_deflection
    );

    // Draw the cross-section
    println!();
    print_cross_section(&result.section);
}

fn print_cross_section(sec: &HSection) {
    let total_h = sec.h + 2.0 * sec.t;
    let scale = 40.0 / sec.b; // normalise flange width to ~40 chars

    let flange_chars = (sec.b * scale).round() as usize;
    let web_chars = (sec.t * scale).round() as usize;
    let flange_rows = (sec.t / total_h * 10.0).round().max(1.0) as usize;
    let web_rows = (sec.h / total_h * 10.0).round().max(1.0) as usize;

    let pad = (flange_chars.saturating_sub(web_chars)) / 2;

    println!(
        "Cross-section (b={:.4}, h={:.4}, t={:.4}):",
        sec.b, sec.h, sec.t
    );
    for _ in 0..flange_rows {
        println!("  {}", "=".repeat(flange_chars));
    }
    for _ in 0..web_rows {
        println!("  {}{}", " ".repeat(pad), "|".repeat(web_chars.max(1)));
    }
    for _ in 0..flange_rows {
        println!("  {}", "=".repeat(flange_chars));
    }
}
