//! QA for Lighthill, stationary permeable FW-H, and DRP linearized Euler.
//!
//! Each methodology has at least three independent numerical/physical checks.
//! The final comparison uses propagation velocity, the common observable across
//! the 3-D frequency-domain radiation models and the 1-D time-domain LEE model.

use num_complex::Complex64;
use sonium_fem::physics::acoustics::aeroacoustics::{
    AcousticBranch, AzimuthalMode, DrpEuler1d, DrpEuler1dConfig, DrpWaveState, FwhContourSegment,
    FwhSurfacePatch, LighthillQuadrupole, LighthillQuadrupole2d, RevolvedQuadrature,
    TAM_WEBB_DRP_7, lighthill_far_field_pressure, lighthill_far_field_pressure_2d,
    lighthill_pressure, lighthill_ring_pressure, revolved_patch_from_modal_data,
    ring_source_from_modal_stress, stationary_fwh_pressure, stationary_fwh_pressure_2d,
    stationary_fwh_ring_pressure,
};
use std::f64::consts::PI;

#[derive(Debug)]
struct QaCheck {
    method: &'static str,
    name: &'static str,
    metric: f64,
    maximum: f64,
}

impl QaCheck {
    fn passed(&self) -> bool {
        self.metric.is_finite() && self.metric <= self.maximum
    }
}

fn check(method: &'static str, name: &'static str, metric: f64, maximum: f64) -> QaCheck {
    QaCheck {
        method,
        name,
        metric,
        maximum,
    }
}

fn axial_quadrupole() -> LighthillQuadrupole {
    let mut stress = [[Complex64::new(0.0, 0.0); 3]; 3];
    stress[0][0] = Complex64::new(1.0, 0.25);
    LighthillQuadrupole::new([0.0; 3], stress).unwrap()
}

fn lighthill_checks() -> Vec<QaCheck> {
    let source = axial_quadrupole();
    let (omega, c) = (200.0, 20.0);

    let p_near = lighthill_far_field_pressure(&source, [10.0, 0.0, 0.0], omega, c).unwrap();
    let p_far = lighthill_far_field_pressure(&source, [25.0, 0.0, 0.0], omega, c).unwrap();
    let spreading_error = (p_near.norm() / p_far.norm() / 2.5 - 1.0).abs();

    let axial = p_near.norm();
    let transverse = lighthill_far_field_pressure(&source, [0.0, 10.0, 0.0], omega, c)
        .unwrap()
        .norm();
    let null_ratio = transverse / axial;

    let exact = lighthill_pressure(&source, [100.0, 0.0, 0.0], omega, c).unwrap();
    let radiating = lighthill_far_field_pressure(&source, [100.0, 0.0, 0.0], omega, c).unwrap();
    let far_field_error = relative_complex_error(exact, radiating);

    vec![
        check(
            "Lighthill",
            "inverse-distance far-field spreading",
            spreading_error,
            1e-12,
        ),
        check(
            "Lighthill",
            "axial-quadrupole transverse null",
            null_ratio,
            1e-14,
        ),
        check(
            "Lighthill",
            "exact Hessian converges to radiating term",
            far_field_error,
            2.1e-3,
        ),
    ]
}

fn fwh_checks() -> Vec<QaCheck> {
    let density = 1.2;
    let sound_speed = 3.0;
    let radius = 0.5;

    let omega_low = 1.5;
    let surface_low = monopole_surface(radius, 48, 96, omega_low, density, sound_speed);
    let observer_low = [2.0, 0.2, -0.1];
    let actual_low =
        stationary_fwh_pressure(&surface_low, observer_low, omega_low, density, sound_speed)
            .unwrap()
            .total;
    let expected_low = monopole_pressure(observer_low, omega_low / sound_speed);
    let low_ka_error = relative_complex_error(actual_low, expected_low);

    let omega_mid = 6.0;
    let surface_mid = monopole_surface(radius, 64, 128, omega_mid, density, sound_speed);
    let observer_mid = [2.5, -0.3, 0.4];
    let actual_mid =
        stationary_fwh_pressure(&surface_mid, observer_mid, omega_mid, density, sound_speed)
            .unwrap()
            .total;
    let expected_mid = monopole_pressure(observer_mid, omega_mid / sound_speed);
    let finite_ka_error = relative_complex_error(actual_mid, expected_mid);

    let observer_1 = [2.0, 0.0, 0.0];
    let observer_2 = [3.0, 0.0, 0.0];
    let result_1 =
        stationary_fwh_pressure(&surface_mid, observer_1, omega_mid, density, sound_speed).unwrap();
    let result_2 =
        stationary_fwh_pressure(&surface_mid, observer_2, omega_mid, density, sound_speed).unwrap();
    let spreading_error = (result_1.total.norm() / result_2.total.norm() / 1.5 - 1.0).abs();

    vec![
        check(
            "FW-H",
            "closed-surface monopole at ka=0.25",
            low_ka_error,
            2e-4,
        ),
        check(
            "FW-H",
            "closed-surface monopole at ka=1",
            finite_ka_error,
            2e-4,
        ),
        check(
            "FW-H",
            "reconstructed inverse-distance spreading",
            spreading_error,
            2e-4,
        ),
    ]
}

fn drp_euler_checks() -> Vec<QaCheck> {
    let config = DrpEuler1dConfig::with_cfl(256, 2.0 * PI, 1.2, 2.0, 0.3, 0.06).unwrap();

    let wavenumber = 20.0;
    let delta = 1e-5;
    let dispersion = config.dispersion(wavenumber, AcousticBranch::Downstream);
    let omega_plus = config
        .dispersion(wavenumber + delta, AcousticBranch::Downstream)
        .angular_frequency;
    let omega_minus = config
        .dispersion(wavenumber - delta, AcousticBranch::Downstream)
        .angular_frequency;
    let finite_difference_group_velocity = (omega_plus - omega_minus) / (2.0 * delta);
    let group_velocity_error = ((dispersion.group_velocity - finite_difference_group_velocity)
        / finite_difference_group_velocity)
        .abs();

    let theta = 1.2;
    let drp_ratio = TAM_WEBB_DRP_7.modified_wavenumber(theta, 1.0) / theta;
    let centered_ratio = theta.sin() / theta;
    let phase_error_ratio = (drp_ratio - 1.0).abs() / (centered_ratio - 1.0).abs();

    let mode = 9.0;
    let mut state = DrpWaveState::zeros(config.n_points);
    for i in 0..config.n_points {
        let x = i as f64 * config.dx();
        state.pressure[i] = (mode * x).cos();
        state.velocity[i] = state.pressure[i] / (config.density * config.speed_of_sound);
    }
    let mut solver = DrpEuler1d::new(config, state).unwrap();
    solver.advance(120);
    let numerical_omega = config
        .dispersion(mode, AcousticBranch::Downstream)
        .angular_frequency;
    let rk4_error = solver
        .state
        .pressure
        .iter()
        .enumerate()
        .map(|(i, actual)| {
            let x = i as f64 * config.dx();
            let expected = (mode * x - numerical_omega * solver.time).cos();
            (actual - expected).powi(2)
        })
        .sum::<f64>()
        .sqrt()
        / (config.n_points as f64 / 2.0).sqrt();

    vec![
        check(
            "DRP Euler",
            "analytic group velocity vs d(omega)/dk",
            group_velocity_error,
            1e-8,
        ),
        check(
            "DRP Euler",
            "midband phase error / centered-2 error",
            phase_error_ratio,
            0.05,
        ),
        check(
            "DRP Euler",
            "RK4 Fourier mode vs discrete dispersion",
            rk4_error,
            1e-7,
        ),
    ]
}

fn all_method_comparison() -> (QaCheck, [f64; 3]) {
    let sound_speed = 343.0;
    let density = 1.204;
    let omega = 2.0 * PI * 500.0;
    let distance_1 = 5.0;
    let distance_2 = 5.1;
    let source = axial_quadrupole();

    let lighthill_1 =
        lighthill_far_field_pressure(&source, [distance_1, 0.0, 0.0], omega, sound_speed).unwrap();
    let lighthill_2 =
        lighthill_far_field_pressure(&source, [distance_2, 0.0, 0.0], omega, sound_speed).unwrap();
    let lighthill_speed = phase_speed(lighthill_1, lighthill_2, 0.1, omega);

    let surface = monopole_surface(0.2, 64, 128, omega, density, sound_speed);
    let fwh_1 = stationary_fwh_pressure(
        &surface,
        [distance_1, 0.0, 0.0],
        omega,
        density,
        sound_speed,
    )
    .unwrap()
    .total;
    let fwh_2 = stationary_fwh_pressure(
        &surface,
        [distance_2, 0.0, 0.0],
        omega,
        density,
        sound_speed,
    )
    .unwrap()
    .total;
    let fwh_speed = phase_speed(fwh_1, fwh_2, 0.1, omega);

    let euler_config =
        DrpEuler1dConfig::with_cfl(512, 51.2, density, sound_speed, 0.0, 0.2).unwrap();
    let euler_wavenumber = 0.3;
    let euler_speed = euler_config
        .dispersion(euler_wavenumber, AcousticBranch::Downstream)
        .group_velocity;

    let speeds = [lighthill_speed, fwh_speed, euler_speed];
    let maximum_deviation = speeds
        .iter()
        .map(|speed| (speed / sound_speed - 1.0).abs())
        .fold(0.0_f64, f64::max);
    (
        check(
            "All three",
            "common propagation-speed agreement",
            maximum_deviation,
            2e-4,
        ),
        speeds,
    )
}

fn planar_axi_checks() -> Vec<QaCheck> {
    let (omega, c) = (2.0, 1.0);

    // 2-D quadrupole: cylindrical far-field spreading (4x distance halves).
    let mut planar = [[Complex64::new(0.0, 0.0); 2]; 2];
    planar[0][0] = Complex64::new(1.0, 0.25);
    let quad_2d = LighthillQuadrupole2d::new([0.0, 0.0], planar).unwrap();
    let near_2d = lighthill_far_field_pressure_2d(&quad_2d, [10.0, 0.0], omega, c).unwrap();
    let far_2d = lighthill_far_field_pressure_2d(&quad_2d, [40.0, 0.0], omega, c).unwrap();
    let cylindrical_error = (near_2d.norm() / far_2d.norm() / 2.0 - 1.0).abs();

    // Spinning ring (m = 3): exact null on the axis.
    let mut stress = [[Complex64::new(0.0, 0.0); 3]; 3];
    stress[2][2] = Complex64::new(1.0, 0.0);
    let ring = ring_source_from_modal_stress(0.5, 0.0, AzimuthalMode::new(3), stress, 1.0).unwrap();
    let ring_axis = lighthill_ring_pressure(&ring, [0.0, 0.0, 5.0], omega, c, 1e-10).unwrap();
    let ring_side = lighthill_ring_pressure(&ring, [5.0, 0.0, 0.0], omega, c, 1e-10).unwrap();
    let ring_null = ring_axis.norm() / ring_side.norm().max(1e-30);

    // Compact closed 2-D contour (ka ~ 0.3), uniform normal-velocity
    // pulsation: cylindrical spreading of the thickness field.
    let mut contour = Vec::new();
    let half_side = 0.1;
    let sides: [([f64; 2], [f64; 2]); 4] = [
        ([0.0, -half_side], [0.0, -1.0]),
        ([0.0, half_side], [0.0, 1.0]),
        ([-half_side, 0.0], [-1.0, 0.0]),
        ([half_side, 0.0], [1.0, 0.0]),
    ];
    for (tangent_point, normal) in sides {
        for index in 0..4 {
            let offset = -half_side + 2.0 * half_side * (index as f64 + 0.5) / 4.0;
            let position = if tangent_point[1].abs() > 0.5 * half_side {
                [offset, tangent_point[1]]
            } else {
                [tangent_point[0], offset]
            };
            contour.push(
                FwhContourSegment::new(
                    position,
                    normal,
                    half_side / 2.0,
                    Complex64::new(0.0, 0.0),
                    Complex64::new(1.0, 0.0),
                )
                .unwrap(),
            );
        }
    }
    let contour_near = stationary_fwh_pressure_2d(&contour, [20.0, 0.0], omega, 1.2, c)
        .unwrap()
        .total;
    let contour_far = stationary_fwh_pressure_2d(&contour, [80.0, 0.0], omega, 1.2, c)
        .unwrap()
        .total;
    let contour_spreading = (contour_near.norm() / contour_far.norm() / 2.0 - 1.0).abs();

    // Revolved strip (m = 2): exact null on the axis.
    let strip = revolved_patch_from_modal_data(
        [0.4, 0.0],
        [0.6, 0.0],
        [0.0, 1.0],
        AzimuthalMode::new(2),
        Complex64::new(1.0, 0.0),
        Complex64::new(0.0, 0.5),
    )
    .unwrap();
    let quadrature = RevolvedQuadrature::new(1e-10, 8).unwrap();
    let strip_axis = stationary_fwh_ring_pressure(
        std::slice::from_ref(&strip),
        [0.0, 0.0, 5.0],
        omega,
        1.2,
        c,
        quadrature,
    )
    .unwrap()
    .total;
    let strip_side = stationary_fwh_ring_pressure(
        std::slice::from_ref(&strip),
        [5.0, 0.0, 0.0],
        omega,
        1.2,
        c,
        quadrature,
    )
    .unwrap()
    .total;
    let strip_null = strip_axis.norm() / strip_side.norm().max(1e-30);

    vec![
        check(
            "Lighthill 2D",
            "cylindrical far-field spreading",
            cylindrical_error,
            2e-2,
        ),
        check(
            "Lighthill ring",
            "spinning m=3 on-axis null",
            ring_null,
            1e-9,
        ),
        check(
            "FW-H 2D",
            "closed-contour cylindrical spreading",
            contour_spreading,
            5e-2,
        ),
        check("FW-H ring", "spinning m=2 on-axis null", strip_null, 1e-9),
    ]
}

fn monopole_surface(
    radius: f64,
    n_mu: usize,
    n_phi: usize,
    omega: f64,
    density: f64,
    sound_speed: f64,
) -> Vec<FwhSurfacePatch> {
    let wavenumber = omega / sound_speed;
    let area = 4.0 * PI * radius * radius / (n_mu * n_phi) as f64;
    let mut patches = Vec::with_capacity(n_mu * n_phi);
    for mu_index in 0..n_mu {
        let mu = -1.0 + 2.0 * (mu_index as f64 + 0.5) / n_mu as f64;
        let radial = (1.0 - mu * mu).sqrt();
        for phi_index in 0..n_phi {
            let phi = 2.0 * PI * (phi_index as f64 + 0.5) / n_phi as f64;
            let normal = [radial * phi.cos(), radial * phi.sin(), mu];
            let position = normal.map(|component| radius * component);
            let pressure = monopole_pressure(position, wavenumber);
            let normal_gradient = pressure * Complex64::new(-1.0 / radius, -wavenumber);
            let normal_velocity = Complex64::new(0.0, 1.0) * normal_gradient / (omega * density);
            patches.push(
                FwhSurfacePatch::new(position, normal, area, pressure, normal_velocity).unwrap(),
            );
        }
    }
    patches
}

fn monopole_pressure(position: [f64; 3], wavenumber: f64) -> Complex64 {
    let radius = norm(position);
    Complex64::from_polar(1.0 / radius, -wavenumber * radius)
}

fn phase_speed(first: Complex64, second: Complex64, distance: f64, omega: f64) -> f64 {
    let phase_difference = (second / first).arg();
    -omega * distance / phase_difference
}

fn relative_complex_error(actual: Complex64, expected: Complex64) -> f64 {
    (actual - expected).norm() / expected.norm().max(1e-30)
}

fn norm(vector: [f64; 3]) -> f64 {
    vector.iter().map(|value| value * value).sum::<f64>().sqrt()
}

fn main() {
    let mut checks = Vec::new();
    checks.extend(lighthill_checks());
    checks.extend(fwh_checks());
    checks.extend(drp_euler_checks());
    checks.extend(planar_axi_checks());
    let (comparison, speeds) = all_method_comparison();
    checks.push(comparison);

    println!("Aeroacoustics analogy/DRP validation");
    println!(
        "{:<12} {:<48} {:>12} {:>12}  result",
        "method", "check", "metric", "maximum"
    );
    for result in &checks {
        println!(
            "{:<12} {:<48} {:>12.4e} {:>12.4e}  {}",
            result.method,
            result.name,
            result.metric,
            result.maximum,
            if result.passed() { "pass" } else { "fail" }
        );
    }
    println!();
    println!("Common-observable comparison (propagation velocity):");
    println!("  Lighthill phase velocity : {:10.5} m/s", speeds[0]);
    println!("  FW-H phase velocity      : {:10.5} m/s", speeds[1]);
    println!("  DRP Euler group velocity : {:10.5} m/s", speeds[2]);

    let failures: Vec<_> = checks.iter().filter(|result| !result.passed()).collect();
    if failures.is_empty() {
        println!("SUCCESS");
    } else {
        eprintln!("FAILED: {} of {} QA checks", failures.len(), checks.len());
        std::process::exit(1);
    }
}
