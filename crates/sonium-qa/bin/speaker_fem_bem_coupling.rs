//! QA: 3D interior-FEM/exterior-BEM coupling for cabinet and ports (review item C2).
//!
//! Gates (`reviews/acoustics-20250730.md` Phase C item 2, §2.3, §10):
//!
//! 1. **Radiation (DtN exterior chain).** Cylindrical box with a circular
//!    aperture, uniform prescribed aperture velocity: far-field on-axis
//!    pressure vs the exact baffled-piston on-axis field
//!    (`electroacoustics_piston_baffle_on_axis`, conjugated into the shared
//!    `exp(+ikr)` convention) at low ka. Also asserts DtN passivity,
//!    `Re(v*·D·v) ≥ 0`.
//! 2. **Helmholtz resonator (coupled solve).** Cylindrical box with an
//!    external port tube driven by a back-face piston: aperture volume
//!    velocity resonance vs the lumped prediction
//!    `f = c/2π·√(S/(V·L_eff))` with standard end corrections.
//! 3. **Sealed cavity (rigid limit).** Rectangular sealed box driven by a
//!    face piston: first three cavity mode peaks vs analytical box modes —
//!    the `D⁻¹ → 0` limit where the exterior influence vanishes.

use num_complex::Complex64;
use sonium_fem::mesh::tetrahedralize_box;
use sonium_fem::physics::acoustics::pressure::resolved::{
    helmholtz_resonance_frequency_hz, rigid_rectangular_cavity_mode_hz,
};
use sonium_qa::analytical::electroacoustics_piston_baffle_on_axis;
use sonium_sim_speaker_modeling::fem_bem_coupling::{
    CouplingDrive, CylinderFixtureDiscretization, FemBemCoupling,
    cylindrical_box_with_circular_aperture, cylindrical_box_with_external_port,
};
use sonium_solvers::{LinearSolveConfig, LinearSolverType};

const C0: f64 = 343.0;
const RHO: f64 = 1.204;

fn solver_config() -> LinearSolveConfig {
    LinearSolveConfig {
        solver: LinearSolverType::GmresIlu,
        max_iterations: 2000,
        tolerance: 1e-9,
        restart: 100,
        exact_residual_check_interval: 0,
        verbosity: 0,
    }
}

fn qa_discretization() -> CylinderFixtureDiscretization {
    CylinderFixtureDiscretization {
        n_circumference: 16,
        n_box_depth: 5,
        n_cap_radial: 3,
        n_aperture_radial: 2,
        n_port_depth: 4,
    }
}

/// Gate 1: far-field on-axis pressure of the uniformly driven aperture vs
/// the exact baffled-piston field, plus DtN passivity.
fn gate_radiation() -> Result<(), String> {
    let fixture = cylindrical_box_with_circular_aperture(0.15, 0.30, 0.05, qa_discretization())
        .map_err(|e| format!("fixture: {e}"))?;
    let model =
        FemBemCoupling::from_fixture(&fixture, C0, RHO).map_err(|e| format!("model: {e}"))?;
    let exterior = model.exterior().expect("coupled model has an exterior");

    // Low ka: ka = 0.27 at 300 Hz for a = 0.05 m.
    let frequency = 300.0;
    let factors = exterior
        .factor(frequency)
        .map_err(|e| format!("factor: {e}"))?;
    let dtn = factors.build_dtn().map_err(|e| format!("dtn: {e}"))?;

    let m = dtn.n;
    let v = vec![Complex64::new(1.0e-2, 0.0); m];
    let areas = model.aperture_element_areas();
    let power = dtn.radiated_power(&v, &areas);
    if power <= 0.0 {
        return Err(format!("DtN passivity violated: radiated power {power}"));
    }

    // Far field from the prescribed-velocity BEM solve.
    let (surface_pressure, surface_velocity) = factors
        .solve_prescribed_velocity(&v)
        .map_err(|e| format!("prescribed solve: {e}"))?;
    let eval_point = [0.0, 0.0, 1.0];
    let p_bem = sonium_bem::core::postprocess::pressure::compute_scattered_field(
        &[eval_point],
        factors.elements(),
        factors.nodes(),
        &surface_pressure,
        Some(&surface_velocity),
        factors.physics(),
        sonium_bem::core::types::BasisOrder::Constant,
    )[0];

    // Reference: exact on-axis piston field with the faceted aperture's
    // effective radius, conjugated into the shared exp(+ikr) convention.
    let k = 2.0 * std::f64::consts::PI * frequency / C0;
    let aperture_area: f64 = areas.iter().sum();
    let a_eff = (aperture_area / std::f64::consts::PI).sqrt();
    let p_ref = electroacoustics_piston_baffle_on_axis(RHO, C0, k, a_eff, 1.0e-2, 1.0).conj();

    // The on-axis near field of a piston is a Fresnel near-null
    // (|e^{ikr} − e^{ik·r_edge}| with kΔr ≈ 7e-3 here): the two terms cancel
    // to <1 %, so the *complex* value is dominated by finite-cabinet
    // diffraction (absent from the infinite-baffle reference) and is not a
    // well-conditioned metric at low ka. The magnitude is well conditioned;
    // gate on it. The complex DtN map itself is validated to 0.7 % against
    // the pulsating-sphere analytical solution in the crate test suite.
    let rel = (p_bem.norm() - p_ref.norm()).abs() / p_ref.norm();
    println!(
        "radiation gate: |p_bem| = {:.6e}, |p_ref| = {:.6e}, mag rel = {:.3e}, power = {power:.3e} W",
        p_bem.norm(),
        p_ref.norm(),
        rel
    );
    // Low-ka monopole agreement; constant-panel BEM + faceted aperture error
    // dominate the residual (calibrated: ~4 %).
    if rel > 0.10 {
        return Err(format!("far-field on-axis magnitude off by {rel:.3e}"));
    }
    Ok(())
}

/// Gate 2: port resonance of the coupled box+port model vs lumped Helmholtz.
fn gate_helmholtz_resonator() -> Result<(), String> {
    let box_radius = 0.15;
    let box_depth = 0.25;
    let port_radius = 0.03;
    let port_length = 0.10;
    let fixture = cylindrical_box_with_external_port(
        box_radius,
        box_depth,
        port_radius,
        port_length,
        qa_discretization(),
    )
    .map_err(|e| format!("fixture: {e}"))?;
    let model =
        FemBemCoupling::from_fixture(&fixture, C0, RHO).map_err(|e| format!("model: {e}"))?;

    // Lumped reference: L_eff = L + 0.85a (inner end, flanged by the front
    // face) + 0.61a (unflanged protruding mouth).
    let volume = fixture.cavity_volume_m3();
    let port_area = std::f64::consts::PI * port_radius * port_radius;
    let l_eff = port_length + (0.85 + 0.61) * port_radius;
    let f_ref = helmholtz_resonance_frequency_hz(C0, volume, port_area, l_eff);

    let config = solver_config();
    let areas = model.aperture_element_areas();
    let sweep: Vec<f64> = (0..=20).map(|i| 40.0 + 2.0 * i as f64).collect();
    let mut best = (0.0, 0.0);
    let mut curve = Vec::new();
    for &f in &sweep {
        let solution = model
            .solve_frequency(
                f,
                CouplingDrive::DriverVelocity(Complex64::new(1.0e-2, 0.0)),
                &config,
            )
            .map_err(|e| format!("coupled solve at {f} Hz: {e}"))?;
        let u = solution.aperture_volume_velocity(&areas).norm();
        curve.push((f, u));
        if u > best.1 {
            best = (f, u);
        }
    }
    let f_peak = parabolic_peak(&curve).unwrap_or(best.0);
    let rel = (f_peak - f_ref).abs() / f_ref;
    println!(
        "helmholtz gate: f_peak = {f_peak:.2} Hz, f_lumped = {f_ref:.2} Hz, rel = {:.3e}",
        rel
    );
    // P1 mesh + discrete sweep: calibrated error ~2%, assert with headroom.
    if rel > 0.08 {
        return Err(format!(
            "port resonance {f_peak:.2} Hz vs lumped {f_ref:.2} Hz ({rel:.3e})"
        ));
    }
    Ok(())
}

/// Gate 3: sealed rectangular box modes vs analytical cavity modes.
fn gate_sealed_cavity_modes() -> Result<(), String> {
    let (lx, ly, lz) = (0.34, 0.22, 0.18);
    let mesh = tetrahedralize_box(0.0, lx, 0.0, ly, 0.0, lz, 12);
    // tetrahedralize_box marks the six faces 1..=6; drive the z_max face (6).
    let model = FemBemCoupling::sealed(mesh, -1, Some(6), C0, RHO);
    let config = solver_config();

    let modes = [
        (
            [1, 0, 0],
            rigid_rectangular_cavity_mode_hz(C0, [lx, ly, lz], [1, 0, 0]),
        ),
        (
            [0, 1, 0],
            rigid_rectangular_cavity_mode_hz(C0, [lx, ly, lz], [0, 1, 0]),
        ),
        (
            [0, 0, 1],
            rigid_rectangular_cavity_mode_hz(C0, [lx, ly, lz], [0, 0, 1]),
        ),
    ];
    for (mode, f_mode) in modes {
        let drive = CouplingDrive::DriverPartialPiston {
            value: Complex64::new(1.0e-2, 0.0),
            x_split: 0.5 * lx,
            y_split: 0.5 * ly,
        };
        // Two-stage peak location: lossless resonances are narrow poles, so
        // locate the maximum on a coarse grid, then refine locally.
        let response = |f: f64| -> Result<f64, String> {
            let solution = model
                .solve_frequency(f, drive.clone(), &config)
                .map_err(|e| format!("sealed solve at {f:.1} Hz: {e}"))?;
            Ok(solution.mean_abs_pressure())
        };
        let coarse: Vec<f64> = (0..=16)
            .map(|i| f_mode * (0.92 + 0.01 * i as f64))
            .collect();
        let mut best = (0.0, 0.0);
        for &f in &coarse {
            let r = response(f)?;
            if r > best.1 {
                best = (f, r);
            }
        }
        let mut f_peak = best.0;
        let mut best_r = best.1;
        for i in -4..=4 {
            let f = best.0 * (1.0 + 0.003 * i as f64);
            let r = response(f)?;
            if r > best_r {
                best_r = r;
                f_peak = f;
            }
        }
        let rel = (f_peak - f_mode).abs() / f_mode;
        println!(
            "sealed gate: mode {mode:?} analytical {f_mode:.1} Hz, peak {f_peak:.1} Hz, rel = {:.3e}",
            rel
        );
        // P1 tet discretization of a rigid cavity: calibrated ~1-3%.
        if rel > 0.03 {
            return Err(format!(
                "cavity mode {mode:?}: peak {f_peak:.1} Hz vs analytical {f_mode:.1} Hz"
            ));
        }
    }
    Ok(())
}

/// Vertex of the parabola through the three largest consecutive samples.
fn parabolic_peak(curve: &[(f64, f64)]) -> Option<f64> {
    let (idx, _) = curve
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.1.partial_cmp(&b.1.1).unwrap())?;
    if idx == 0 || idx + 1 >= curve.len() {
        return Some(curve[idx].0);
    }
    let (x0, y0) = curve[idx - 1];
    let (x1, y1) = curve[idx];
    let (x2, y2) = curve[idx + 1];
    let denom = (x0 - x1) * (x0 - x2) * (x1 - x2);
    if denom.abs() < 1e-30 {
        return Some(x1);
    }
    let a = (x2 * (y1 - y0) + x1 * (y0 - y2) + x0 * (y2 - y1)) / denom;
    let b = (x2 * x2 * (y0 - y1) + x1 * x1 * (y2 - y0) + x0 * x0 * (y1 - y2)) / denom;
    if a.abs() < 1e-30 {
        return Some(x1);
    }
    Some(-b / (2.0 * a))
}

fn main() {
    let result = std::panic::catch_unwind(|| -> Result<(), String> {
        gate_radiation()?;
        gate_helmholtz_resonator()?;
        gate_sealed_cavity_modes()?;
        Ok(())
    });
    match result {
        Ok(Ok(())) => println!("SUCCESS"),
        Ok(Err(e)) => {
            println!("FAILED: {e}");
            std::process::exit(1);
        }
        Err(_) => {
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
