//! Native rigid-sphere BEM accuracy against analytical Mie pressure.
//!
//! The canonical command is `qa-bem-rigid-sphere-mie-accuracy`. The historical
//! `qa-bem-mesh2hrtf-parity` command remains a compatibility alias. Both run
//! native Burton--Miller solves and compare complex pressure with Mie; neither
//! executes Mesh2HRTF nor imports external solver output.
//!
//! Cases use `ka = k*radius` of 1, 2 and 5, 1280 triangular panels and nine
//! exterior evaluation points at radius 2. The existing 1%/1.5% complex L2
//! tolerances are retained as native analytical-accuracy bounds. This fixed
//! resolution gate makes no external parity or mesh-convergence claim.

// Rust guideline compliant 2026-02-21
use num_complex::Complex64;
use sonium_bem::analytical::sphere_scattering_3d;
use sonium_bem::core::bem_solver::BoundaryConditionType;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_bem::core::{BemProblem, BemSolver, SolverMethod};
use sonium_bem::testing::ErrorMetrics;
use sonium_qa::QaResult;
use std::f64::consts::PI;
use std::time::Instant;

const SPEED_OF_SOUND: f64 = 343.0;
const DENSITY: f64 = 1.21;

fn freq_for_ka(ka: f64, radius: f64) -> f64 {
    ka * SPEED_OF_SOUND / (2.0 * PI * radius)
}

fn theta_grid() -> Vec<f64> {
    (0..=8).map(|i| i as f64 * PI / 8.0).collect()
}

struct SphereCase {
    label: &'static str,
    /// `ka = k · radius` with the unit-radius convention.
    ka: f64,
    /// Icosphere subdivision level for the fixed native mesh.
    subdivisions: usize,
    /// Native complex-pressure L2 bound against the analytical Mie solution.
    threshold_l2: f64,
}

fn run_case(case: &SphereCase) -> Result<QaResult, String> {
    let radius = 1.0;
    let frequency = freq_for_ka(case.ka, radius);
    let k = 2.0 * PI * frequency / SPEED_OF_SOUND;

    let mesh = generate_icosphere_mesh(radius, case.subdivisions);
    let n_elem = mesh.elements.len();
    let physics = PhysicsParams::new(frequency, SPEED_OF_SOUND, DENSITY, false);

    println!(
        "\n=== {} : ka = {:.1}, subdiv = {}, DOFs = {} ===",
        case.label, case.ka, case.subdivisions, n_elem
    );
    let wavelength = 2.0 * PI / k;
    let h = (4.0 * PI * radius * radius / n_elem as f64).sqrt();
    println!(
        "  λ = {wavelength:.3}, h ≈ {h:.3}, elements/λ ≈ {:.1}",
        wavelength / h
    );

    let problem = BemProblem {
        mesh,
        physics,
        incident_field: IncidentField::plane_wave_z(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    };

    let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);
    let t0 = Instant::now();
    let solution = solver
        .solve(&problem)
        .map_err(|e| format!("{}: solve failed: {e:?}", case.label))?;
    let dt = t0.elapsed().as_secs_f64();
    println!("  solve: {dt:.2} s");

    let eval_r = 2.0 * radius;
    let theta = theta_grid();
    let bem_p: Vec<Complex64> = theta
        .iter()
        .map(|&t| solution.evaluate_pressure(&[eval_r * t.sin(), 0.0, eval_r * t.cos()]))
        .collect();
    let n_terms = ((case.ka + 30.0) as usize).max(15);
    let analytical = sphere_scattering_3d(k, radius, n_terms, vec![eval_r], theta.clone());
    let metrics = ErrorMetrics::compute(&analytical.pressure, &bem_p);

    println!(
        "  L² rel err  : {:.4} ({:.2}%)  [analytical accuracy bound ≤ {:.2}%]",
        metrics.l2_relative,
        metrics.l2_relative * 100.0,
        case.threshold_l2 * 100.0
    );
    println!(
        "  max rel err : {:.4} ({:.2}%)",
        metrics.max_relative,
        metrics.max_relative * 100.0,
    );

    if !metrics.l2_relative.is_finite() {
        return Err(format!("{}: L² error is non-finite", case.label));
    }
    if metrics.l2_relative > case.threshold_l2 {
        return Err(format!(
            "{}: L² {:.4} exceeds analytical accuracy bound {:.4}",
            case.label, metrics.l2_relative, case.threshold_l2
        ));
    }
    Ok(QaResult::passed(format!(
        "bem.rigid-sphere-mie-accuracy-ka{}.v1",
        case.ka as usize
    ))
    .requested_backend("cpu")
    .actual_backend("cpu")
    .seed(format!(
        "rigid-sphere:a1:ka{}:icosphere3:theta9:r2",
        case.ka
    ))
    .reference("analytical rigid-sphere Mie pressure; no Mesh2HRTF output")
    .reference_checksum("analytic:rigid-sphere-mie-exterior:v1")
    .formulation("native triangular Burton-Miller BEM; complex exterior pressure accuracy")
    .metric("relative_complex_l2_error", metrics.l2_relative)
    .maximum("relative_complex_l2_error", case.threshold_l2)
    .metric("numerical_solve_executed", 1.0)
    .minimum("numerical_solve_executed", 1.0)
    .metric("external_reference_executed", 0.0)
    .maximum("external_reference_executed", 0.0))
}

/// Run native rigid-sphere accuracy against analytical Mie fields.
pub(crate) fn main() {
    println!("=== Rigid-sphere BEM analytical Mie accuracy ===");
    println!(
        "Native Sonium solve compared with analytical Mie pressure; external_reference_executed=0."
    );

    // Preserve the existing native resolution and accuracy thresholds.
    let cases: &[SphereCase] = &[
        SphereCase {
            label: "ka=1",
            ka: 1.0,
            subdivisions: 3, // ≈ 1280 elements, ~63 elements/λ
            threshold_l2: 0.010,
        },
        SphereCase {
            label: "ka=2",
            ka: 2.0,
            subdivisions: 3, // ≈ 1280 elements, ~32 elements/λ
            threshold_l2: 0.015,
        },
        SphereCase {
            label: "ka=5",
            ka: 5.0,
            subdivisions: 3, // ≈ 1280 elements, ~13 elements/λ
            threshold_l2: 0.015,
        },
    ];

    let mut failures: Vec<String> = Vec::new();
    for c in cases {
        if let Err(e) = run_case(c)
            .and_then(|record| record.to_json_line())
            .map(|line| println!("QA_RESULT: {line}"))
        {
            failures.push(e);
        }
    }

    if failures.is_empty() {
        println!("\nSUCCESS");
        return;
    }
    for f in &failures {
        eprintln!("FAIL: {f}");
    }
    eprintln!("FAILED");
    std::process::exit(1);
}
