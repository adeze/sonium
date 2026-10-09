//! Independent physical reference for the production 3-D multi-region BEM.
//!
//! A lossless penetrable fluid sphere is solved on two icosphere levels with
//! the dense constant-panel Müller operator. Exterior fields are compared to
//! the independent fluid-sphere Mie series from `sonium-wave`; the gate also
//! checks the exact coupled residual, axial symmetry, and closed-interface
//! active-power balance.

use num_complex::Complex64;
use sonium_bem::build_transmission_tbem_system_from_xem;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generate_icosphere_mesh;
use sonium_bem::core::solver::solve_incident_transmission;
use sonium_wave::analytical::FluidSphereTransmission3d;
use sonium_xem::AcousticMedium;
use std::f64::consts::PI;

const RADIUS_M: f64 = 0.2;
const EXTERIOR_SOUND_SPEED_M_S: f64 = 343.0;
const EXTERIOR_DENSITY_KG_M3: f64 = 1.21;
const INTERIOR_SOUND_SPEED_M_S: f64 = 240.0;
const INTERIOR_DENSITY_KG_M3: f64 = 2.4;
const EXTERIOR_KA: f64 = 1.0;
const EVALUATION_RADIUS_M: f64 = 1.0;
const MIE_TERMS: usize = 32;
const RESIDUAL_TOLERANCE: f64 = 1.0e-9;

#[derive(Debug, Clone, Copy)]
pub struct PenetrableSphereMetrics {
    pub coarse_l2_error: f64,
    pub coarse_pointwise_error: f64,
    pub fine_l2_error: f64,
    pub fine_pointwise_error: f64,
    pub fine_residual: f64,
    pub coarse_mie_trace_residual: f64,
    pub fine_mie_trace_residual: f64,
    pub axisymmetry_error: f64,
    pub active_power_balance: f64,
    pub coarse_panels: usize,
    pub fine_panels: usize,
}

#[derive(Debug)]
struct LevelResult {
    total_pressure: Vec<Complex64>,
    residual: f64,
    axisymmetry_error: f64,
    active_power_balance: f64,
    mie_trace_residual: f64,
    panels: usize,
}

fn frequency_hz() -> f64 {
    EXTERIOR_KA * EXTERIOR_SOUND_SPEED_M_S / (2.0 * PI * RADIUS_M)
}

fn polar_angles() -> Vec<f64> {
    vec![0.2, 0.55, 0.9, 1.25, 1.6, 2.0, 2.4, 2.8]
}

fn point(radius: f64, theta: f64, azimuth: f64) -> [f64; 3] {
    [
        radius * theta.sin() * azimuth.cos(),
        radius * theta.sin() * azimuth.sin(),
        radius * theta.cos(),
    ]
}

fn relative_error(actual: Complex64, reference: Complex64) -> f64 {
    (actual - reference).norm() / actual.norm().max(reference.norm()).max(f64::MIN_POSITIVE)
}

fn solve_level(subdivisions: usize, mie: &FluidSphereTransmission3d) -> LevelResult {
    let mesh = generate_icosphere_mesh(RADIUS_M, subdivisions);
    let exterior = AcousticMedium::try_new(
        frequency_hz(),
        EXTERIOR_SOUND_SPEED_M_S,
        EXTERIOR_DENSITY_KG_M3,
    )
    .expect("checked exterior fluid");
    let interior = AcousticMedium::try_new(
        frequency_hz(),
        INTERIOR_SOUND_SPEED_M_S,
        INTERIOR_DENSITY_KG_M3,
    )
    .expect("checked interior fluid");
    let system =
        build_transmission_tbem_system_from_xem(&mesh.elements, &mesh.nodes, exterior, interior)
            .expect("production penetrable-sphere Müller assembly");
    let incident = IncidentField::plane_wave_z();
    let solution = solve_incident_transmission(&system, &incident, RESIDUAL_TOLERANCE)
        .expect("production penetrable-sphere solve");

    let mut exact_trace = Vec::with_capacity(system.total_dofs());
    exact_trace.extend(mesh.elements.iter().map(|element| {
        let radius =
            (element.center[0].powi(2) + element.center[1].powi(2) + element.center[2].powi(2))
                .sqrt();
        mie.interface_pressure((element.center[2] / radius).clamp(-1.0, 1.0).acos())
            .expect("Mie interface pressure")
    }));
    exact_trace.extend(mesh.elements.iter().map(|element| {
        let radius =
            (element.center[0].powi(2) + element.center[1].powi(2) + element.center[2].powi(2))
                .sqrt();
        mie.interface_normal_velocity(
            (element.center[2] / radius).clamp(-1.0, 1.0).acos(),
            2.0 * PI * frequency_hz(),
        )
        .expect("Mie interface velocity")
    }));
    let incident_rhs = system
        .incident_rhs(&incident)
        .expect("finite incident transmission RHS");
    let exact_applied: Vec<Complex64> = (0..system.total_dofs())
        .map(|row| {
            (0..system.total_dofs())
                .map(|column| system.matrix()[[row, column]] * exact_trace[column])
                .sum()
        })
        .collect();
    let mie_trace_residual = exact_applied
        .iter()
        .zip(&incident_rhs)
        .map(|(&actual, &expected)| (actual - expected).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / incident_rhs
            .iter()
            .map(|value| value.norm_sqr())
            .sum::<f64>()
            .sqrt()
            .max(f64::MIN_POSITIVE);

    let evaluation_points: Vec<_> = polar_angles()
        .into_iter()
        .map(|theta| point(EVALUATION_RADIUS_M, theta, 0.0))
        .collect();
    let total_pressure = solution
        .evaluate_exterior_field_checked(&system, &evaluation_points, &incident)
        .expect("checked penetrable-sphere exterior field")
        .into_iter()
        .map(|field| field.p_total)
        .collect();

    let symmetry_theta = 0.82;
    let symmetry_points = [0.0, 0.5 * PI, PI, 1.5 * PI]
        .map(|azimuth| point(EVALUATION_RADIUS_M, symmetry_theta, azimuth));
    let symmetry_fields = solution
        .evaluate_exterior_field_checked(&system, &symmetry_points, &incident)
        .expect("checked axisymmetric penetrable-sphere field");
    let axisymmetry_error = symmetry_fields
        .iter()
        .skip(1)
        .map(|field| relative_error(field.p_total, symmetry_fields[0].p_total))
        .fold(0.0, f64::max);

    let signed_power = mesh
        .elements
        .iter()
        .zip(solution.first_trace())
        .zip(solution.second_trace())
        .map(|((element, &pressure), &velocity)| {
            0.5 * element.area * (pressure * velocity.conj()).re
        })
        .sum::<f64>();
    let absolute_power = mesh
        .elements
        .iter()
        .zip(solution.first_trace())
        .zip(solution.second_trace())
        .map(|((element, &pressure), &velocity)| {
            0.5 * element.area * (pressure * velocity.conj()).re.abs()
        })
        .sum::<f64>();
    let active_power_balance = signed_power.abs() / absolute_power.max(f64::MIN_POSITIVE);

    LevelResult {
        total_pressure,
        residual: solution.relative_residual,
        axisymmetry_error,
        active_power_balance,
        mie_trace_residual,
        panels: mesh.elements.len(),
    }
}

fn field_errors(actual: &[Complex64], reference: &[Complex64]) -> (f64, f64) {
    let reference_norm = reference
        .iter()
        .map(|value| value.norm_sqr())
        .sum::<f64>()
        .sqrt();
    let l2 = actual
        .iter()
        .zip(reference)
        .map(|(&actual, &reference)| (actual - reference).norm_sqr())
        .sum::<f64>()
        .sqrt()
        / reference_norm.max(f64::MIN_POSITIVE);
    let pointwise = actual
        .iter()
        .zip(reference)
        .map(|(&actual, &reference)| relative_error(actual, reference))
        .fold(0.0, f64::max);
    (l2, pointwise)
}

pub fn run_penetrable_sphere_mie_convergence() -> PenetrableSphereMetrics {
    let exterior_wave_number = EXTERIOR_KA / RADIUS_M;
    let interior_wave_number = 2.0 * PI * frequency_hz() / INTERIOR_SOUND_SPEED_M_S;
    let mie = FluidSphereTransmission3d::try_new(
        RADIUS_M,
        exterior_wave_number,
        interior_wave_number,
        EXTERIOR_DENSITY_KG_M3,
        INTERIOR_DENSITY_KG_M3,
        MIE_TERMS,
    )
    .expect("independent penetrable-sphere Mie reference");
    assert!(mie.pressure_continuity_error(33) < 1.0e-11);
    assert!(mie.velocity_continuity_error(33) < 1.0e-11);
    let reference: Vec<_> = polar_angles()
        .into_iter()
        .map(|theta| {
            mie.exterior_pressure(EVALUATION_RADIUS_M, theta)
                .expect("Mie exterior field")
        })
        .collect();

    let coarse = solve_level(1, &mie);
    let fine = solve_level(2, &mie);
    let (coarse_l2_error, coarse_pointwise_error) =
        field_errors(&coarse.total_pressure, &reference);
    let (fine_l2_error, fine_pointwise_error) = field_errors(&fine.total_pressure, &reference);

    assert!(fine.panels > coarse.panels);
    assert!(fine.residual <= RESIDUAL_TOLERANCE);
    assert!(
        fine.mie_trace_residual / coarse.mie_trace_residual <= 0.50,
        "penetrable-sphere exact-trace operator residual did not contract enough: coarse={:.3e}, fine={:.3e}",
        coarse.mie_trace_residual,
        fine.mie_trace_residual
    );
    assert!(
        fine.mie_trace_residual <= 0.03,
        "penetrable-sphere fine exact-trace residual {:.3e} exceeds 3%",
        fine.mie_trace_residual
    );
    assert!(
        fine_l2_error <= 0.005,
        "penetrable-sphere fine L2 error {fine_l2_error:.3e} exceeds 0.5%"
    );
    assert!(
        fine_pointwise_error <= 0.01,
        "penetrable-sphere fine pointwise error {fine_pointwise_error:.3e} exceeds 1%"
    );
    assert!(
        fine.axisymmetry_error <= 0.08,
        "penetrable-sphere axial-symmetry error {:.3e} exceeds 8%",
        fine.axisymmetry_error
    );
    assert!(
        fine.active_power_balance <= 0.10,
        "lossless penetrable-sphere active-power imbalance {:.3e} exceeds 10%",
        fine.active_power_balance
    );

    PenetrableSphereMetrics {
        coarse_l2_error,
        coarse_pointwise_error,
        fine_l2_error,
        fine_pointwise_error,
        fine_residual: fine.residual,
        coarse_mie_trace_residual: coarse.mie_trace_residual,
        fine_mie_trace_residual: fine.mie_trace_residual,
        axisymmetry_error: fine.axisymmetry_error,
        active_power_balance: fine.active_power_balance,
        coarse_panels: coarse.panels,
        fine_panels: fine.panels,
    }
}

#[test]
fn penetrable_sphere_converges_to_independent_mie_reference() {
    let metrics = run_penetrable_sphere_mie_convergence();
    println!(
        "BEM_EVIDENCE: {{\"case_id\":\"bem.penetrable-sphere-mie-convergence.v1\",\"status\":\"passed\",\"coarse_l2_error\":{:.6e},\"coarse_pointwise_error\":{:.6e},\"fine_l2_error\":{:.6e},\"fine_pointwise_error\":{:.6e},\"fine_residual\":{:.6e},\"coarse_mie_trace_residual\":{:.6e},\"fine_mie_trace_residual\":{:.6e},\"axisymmetry_error\":{:.6e},\"active_power_balance\":{:.6e},\"coarse_panels\":{},\"fine_panels\":{}}}",
        metrics.coarse_l2_error,
        metrics.coarse_pointwise_error,
        metrics.fine_l2_error,
        metrics.fine_pointwise_error,
        metrics.fine_residual,
        metrics.coarse_mie_trace_residual,
        metrics.fine_mie_trace_residual,
        metrics.axisymmetry_error,
        metrics.active_power_balance,
        metrics.coarse_panels,
        metrics.fine_panels,
    );
}
