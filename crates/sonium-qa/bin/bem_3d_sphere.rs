//! 3D Sphere Scattering BEM Validation (Mie Theory)
//!
//! Runs actual BEM solves for rigid-sphere scattering and compares numerical
//! far-field pressures against the analytical Mie series. This replaces the
//! previous placeholder that compared the analytical solution against itself.

use num_complex::Complex64;
use sonium_bem::analytical::sphere_scattering_3d;
use sonium_bem::core::{BemProblem, BemSolver, SolverMethod};
use sonium_qa::{QaResult, relative_error, rigid_sphere_surface_relative_l2_error};
use std::f64::consts::PI;
use std::time::Instant;

fn freq_for_ka(ka: f64, radius: f64, speed_of_sound: f64) -> f64 {
    ka * speed_of_sound / (2.0 * PI * radius)
}

fn spherical_to_cartesian(r: f64, theta: f64) -> [f64; 3] {
    [r * theta.sin(), 0.0, r * theta.cos()]
}

/// Integrate scattered acoustic power on a far-field sphere.
///
/// The BEM and Mie values are sampled at the same midpoint rule in
/// `mu = cos(theta)`, so the comparison exercises the production field
/// evaluator and the independent reference with one shared physical
/// observable. The `r^2/(2 rho c)` factor is the time-averaged outgoing-wave
/// power for pressure amplitude under the repository's RMS convention.
fn scattered_power_relative_error(
    solution: &sonium_bem::core::BemSolution,
    k: f64,
    radius: f64,
    density: f64,
    speed_of_sound: f64,
) -> Result<(f64, f64), String> {
    let eval_radius = (12.0 * radius).max(radius + 1.0);
    let samples = 64usize;
    let mut theta = Vec::with_capacity(samples);
    let mut points = Vec::with_capacity(samples);
    for sample in 0..samples {
        let mu = -1.0 + 2.0 * (sample as f64 + 0.5) / samples as f64;
        let sin_theta = (1.0 - mu * mu).max(0.0).sqrt();
        theta.push(mu.acos());
        points.push([eval_radius * sin_theta, 0.0, eval_radius * mu]);
    }

    let bem_total = solution.evaluate_pressure_field(&points);
    let incident = solution
        .incident_field
        .evaluate_pressure(&points, &solution.physics);
    let bem_scattered: Vec<Complex64> = bem_total
        .iter()
        .zip(incident.iter())
        .map(|(total, incident)| total.p_total - incident)
        .collect();
    if bem_scattered
        .iter()
        .any(|value| !value.re.is_finite() || !value.im.is_finite())
    {
        return Err("far-field BEM scattered pressure is non-finite".into());
    }

    let mie = sphere_scattering_3d(k, radius, 40, vec![eval_radius], theta);
    let incident_reference: Vec<Complex64> = points
        .iter()
        .map(|point| {
            let phase = k * point[2];
            Complex64::new(phase.cos(), phase.sin())
        })
        .collect();
    let mie_scattered: Vec<Complex64> = mie
        .pressure
        .iter()
        .zip(incident_reference.iter())
        .map(|(total, incident)| *total - *incident)
        .collect();

    let angular_weight = 2.0 / samples as f64;
    let bem_integral: f64 = bem_scattered
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        * angular_weight
        * 2.0
        * PI;
    let mie_integral: f64 = mie_scattered
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        * angular_weight
        * 2.0
        * PI;
    let factor = eval_radius * eval_radius / (2.0 * density * speed_of_sound);
    let bem_power = factor * bem_integral;
    let mie_power = factor * mie_integral;
    if !bem_power.is_finite() || !mie_power.is_finite() || mie_power <= f64::MIN_POSITIVE {
        return Err("far-field scattered power is non-finite or zero".into());
    }
    Ok(((bem_power - mie_power).abs() / mie_power, bem_power))
}

#[derive(Debug, Clone, Copy)]
struct CaseMetrics {
    relative_l2: f64,
    surface_relative_l2: f64,
    radiated_power_relative_error: f64,
    radiated_power_w: f64,
    elapsed_ms: u64,
    dofs: usize,
}

fn test_rigid_sphere_scattering_ka_low() -> Result<CaseMetrics, String> {
    let radius = 1.0;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka = 0.5;
    let frequency = freq_for_ka(ka, radius, speed_of_sound);
    let k = 2.0 * PI * frequency / speed_of_sound;

    println!("\n=== 3D Sphere BEM vs Mie (ka={:.2}) ===", ka);

    let problem = BemProblem::rigid_sphere_scattering(radius, frequency, speed_of_sound, density);
    let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);

    let t_start = Instant::now();
    let solution = solver
        .solve(&problem)
        .map_err(|error| format!("ka=0.5 BEM solve failed: {error:?}"))?;
    let elapsed_ms = t_start.elapsed().as_millis() as u64;
    println!(
        "  DOFs: {}, solve time: {} ms",
        solution.num_dofs(),
        elapsed_ms
    );

    let eval_r = 2.0;
    let theta_points: Vec<f64> = vec![0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI];

    let bem_pressures: Vec<Complex64> = theta_points
        .iter()
        .map(|&theta| {
            let pt = spherical_to_cartesian(eval_r, theta);
            solution.evaluate_pressure(&pt)
        })
        .collect();

    let analytical = sphere_scattering_3d(k, radius, 30, vec![eval_r], theta_points.clone());

    let error = relative_error(&bem_pressures, &analytical.pressure);
    println!("  Relative L2 error: {:.4} ({:.2}%)", error, error * 100.0);

    let surface_error = rigid_sphere_surface_relative_l2_error(
        &solution.surface_pressure,
        &solution.elements,
        k,
        radius,
        30,
    );
    println!(
        "  Area-weighted complex surface L2 error: {:.4}",
        surface_error
    );
    if !error.is_finite() || !surface_error.is_finite() {
        return Err("ka=0.5 produced a non-finite Mie error".into());
    }
    let (radiated_power_relative_error, radiated_power_w) =
        scattered_power_relative_error(&solution, k, radius, density, speed_of_sound)?;
    println!(
        "  Far-field scattered power: {:.6e} W (relative Mie error {:.3e})",
        radiated_power_w, radiated_power_relative_error
    );
    if error >= 0.01 {
        return Err(format!(
            "ka=0.5 relative L2 error {:.2}% exceeds 1% threshold",
            error * 100.0
        ));
    }
    if radiated_power_relative_error >= 0.06 {
        return Err(format!(
            "ka=0.5 radiated-power relative error {:.2}% exceeds 6% threshold",
            radiated_power_relative_error * 100.0
        ));
    }
    Ok(CaseMetrics {
        relative_l2: error,
        surface_relative_l2: surface_error,
        radiated_power_relative_error,
        radiated_power_w,
        elapsed_ms,
        dofs: solution.num_dofs(),
    })
}

fn test_rigid_sphere_scattering_ka_1() -> Result<CaseMetrics, String> {
    let radius = 1.0;
    let speed_of_sound = 343.0;
    let density = 1.21;
    let ka = 1.0;
    let frequency = freq_for_ka(ka, radius, speed_of_sound);
    let k = 2.0 * PI * frequency / speed_of_sound;

    println!("\n=== 3D Sphere BEM vs Mie (ka={:.2}) ===", ka);

    let problem = BemProblem::rigid_sphere_scattering(radius, frequency, speed_of_sound, density);
    let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);

    let t_start = Instant::now();
    let solution = solver
        .solve(&problem)
        .map_err(|error| format!("ka=1.0 BEM solve failed: {error:?}"))?;
    let elapsed_ms = t_start.elapsed().as_millis() as u64;
    println!(
        "  DOFs: {}, solve time: {} ms",
        solution.num_dofs(),
        elapsed_ms
    );

    let eval_r = 2.0;
    let theta_points: Vec<f64> = vec![0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI];

    let bem_pressures: Vec<Complex64> = theta_points
        .iter()
        .map(|&theta| {
            let pt = spherical_to_cartesian(eval_r, theta);
            solution.evaluate_pressure(&pt)
        })
        .collect();

    let analytical = sphere_scattering_3d(k, radius, 30, vec![eval_r], theta_points.clone());

    let error = relative_error(&bem_pressures, &analytical.pressure);
    println!("  Relative L2 error: {:.4} ({:.2}%)", error, error * 100.0);

    let surface_error = rigid_sphere_surface_relative_l2_error(
        &solution.surface_pressure,
        &solution.elements,
        k,
        radius,
        30,
    );
    println!(
        "  Area-weighted complex surface L2 error: {:.4}",
        surface_error
    );
    if !error.is_finite() || !surface_error.is_finite() {
        return Err("ka=1.0 produced a non-finite Mie error".into());
    }
    let (radiated_power_relative_error, radiated_power_w) =
        scattered_power_relative_error(&solution, k, radius, density, speed_of_sound)?;
    println!(
        "  Far-field scattered power: {:.6e} W (relative Mie error {:.3e})",
        radiated_power_w, radiated_power_relative_error
    );
    if error >= 0.01 {
        return Err(format!(
            "ka=1.0 relative L2 error {:.2}% exceeds 1% threshold",
            error * 100.0
        ));
    }
    if radiated_power_relative_error >= 0.06 {
        return Err(format!(
            "ka=1.0 radiated-power relative error {:.2}% exceeds 6% threshold",
            radiated_power_relative_error * 100.0
        ));
    }
    Ok(CaseMetrics {
        relative_l2: error,
        surface_relative_l2: surface_error,
        radiated_power_relative_error,
        radiated_power_w,
        elapsed_ms,
        dofs: solution.num_dofs(),
    })
}

fn main() {
    let started = Instant::now();
    let low = test_rigid_sphere_scattering_ka_low();
    let high = test_rigid_sphere_scattering_ka_1();
    let mut record = match (&low, &high) {
        (Ok(_), Ok(_)) => QaResult::passed("bem.sphere-mie-3d.v1"),
        _ => QaResult::failed(
            "bem.sphere-mie-3d.v1",
            [low.as_ref().err(), high.as_ref().err()]
                .into_iter()
                .flatten()
                .cloned()
                .collect::<Vec<_>>()
                .join("; "),
        ),
    }
    .requested_backend("cpu")
    .actual_backend("cpu")
    .elapsed_ms(started.elapsed().as_millis() as u64);

    if let Ok(metrics) = low {
        record = record
            .metric("ka_0_5_relative_l2", metrics.relative_l2)
            .metric("ka_0_5_surface_relative_l2", metrics.surface_relative_l2)
            .metric(
                "ka_0_5_radiated_power_relative_error",
                metrics.radiated_power_relative_error,
            )
            .metric("ka_0_5_radiated_power_w", metrics.radiated_power_w)
            .metric("ka_0_5_dofs", metrics.dofs as f64)
            .metric("ka_0_5_elapsed_ms", metrics.elapsed_ms as f64)
            .maximum("ka_0_5_relative_l2", 0.01)
            .maximum("ka_0_5_surface_relative_l2", 0.02)
            .maximum("ka_0_5_radiated_power_relative_error", 0.06);
    }
    if let Ok(metrics) = high {
        record = record
            .metric("ka_1_relative_l2", metrics.relative_l2)
            .metric("ka_1_surface_relative_l2", metrics.surface_relative_l2)
            .metric(
                "ka_1_radiated_power_relative_error",
                metrics.radiated_power_relative_error,
            )
            .metric("ka_1_radiated_power_w", metrics.radiated_power_w)
            .metric("ka_1_dofs", metrics.dofs as f64)
            .metric("ka_1_elapsed_ms", metrics.elapsed_ms as f64)
            .maximum("ka_1_relative_l2", 0.01)
            .maximum("ka_1_surface_relative_l2", 0.02)
            .maximum("ka_1_radiated_power_relative_error", 0.06);
    }
    if let Ok(json) = record.to_json_line() {
        println!("QA_RESULT: {json}");
    }
    if record.status == sonium_qa::QaStatus::Passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
