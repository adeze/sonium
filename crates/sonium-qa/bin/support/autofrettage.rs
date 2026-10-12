//! Spatial pressure-cycle verification using native J2 constitutive updates.
//!
//! Axisymmetric long-cylinder plane strain, with radial P1 displacements and
//! two material points per element. B-bar projects the volumetric strain to
//! its radius-weighted element mean to avoid near-incompressible locking.
//! All radial unknowns are free: bore pressure is a natural traction and the
//! outer wall is traction-free. Newton trials never commit material history.

// Rust guideline compliant 2026-02-21

use sonium_fem::physics::solid::plasticity::{
    HardeningLaw, J2Material, PlasticState, StressUpdateModel,
};
use sonium_qa::QaResult;

const INNER: f64 = 0.05;
const OUTER: f64 = 0.10;
const YIELD: f64 = 250e6;
const PRESSURE: f64 = 150e6;
// The independent closed form assumes incompressible plane strain.
// This finite bulk modulus approaches that limit without a singular material.
const POISSON: f64 = 0.49999;
const GAUSS: [f64; 2] = [-0.5773502691896258, 0.5773502691896258];

struct Assembly {
    residual: Vec<f64>,
    diagonal: Vec<f64>,
    upper: Vec<f64>,
    states: Vec<PlasticState>,
}

struct Cycle {
    residual_error: f64,
    loaded_error: f64,
    equilibrium: f64,
    unload_plastic_increment: f64,
    final_stress: Vec<[f64; 2]>,
    minimum_hoop: f64,
    peak_plastic: f64,
    plastic_points: usize,
    steps: usize,
}

fn basis(element: usize, count: usize, xi: f64) -> (f64, f64, [[f64; 3]; 2]) {
    let h = (OUTER - INNER) / count as f64;
    let left = INNER + element as f64 * h;
    let shape = [(1.0 - xi) / 2.0, (1.0 + xi) / 2.0];
    let r = left + h * shape[1];
    let midpoint = left + h / 2.0;
    let derivative = [-1.0 / h, 1.0 / h];
    let mean_divergence = [-left / (midpoint * h), (left + h) / (midpoint * h)];
    let b = std::array::from_fn(|i| {
        let correction = (mean_divergence[i] - derivative[i] - shape[i] / r) / 3.0;
        [
            derivative[i] + correction,
            shape[i] / r + correction,
            correction,
        ]
    });
    (r, r * h / 2.0, b)
}

fn assemble(
    count: usize,
    u: &[f64],
    previous: &[f64],
    committed: &[PlasticState],
    material: &J2Material,
    pressure: f64,
) -> Assembly {
    let mut out = Assembly {
        residual: vec![0.0; count + 1],
        diagonal: vec![0.0; count + 1],
        upper: vec![0.0; count],
        states: Vec::with_capacity(count * 2),
    };
    for e in 0..count {
        for (q, xi) in GAUSS.into_iter().enumerate() {
            let (_, weight, b) = basis(e, count, xi);
            let mut increment = [0.0; 6];
            for i in 0..2 {
                for (k, value) in increment[..3].iter_mut().enumerate() {
                    *value += b[i][k] * (u[e + i] - previous[e + i]);
                }
            }
            let update = material.update(&committed[e * 2 + q], &increment);
            let mut state = committed[e * 2 + q].clone();
            material.commit(&mut state, &update);
            out.states.push(state);
            for i in 0..2 {
                out.residual[e + i] +=
                    weight * (0..3).map(|k| b[i][k] * update.stress[k]).sum::<f64>();
                for j in i..2 {
                    let stiffness = weight
                        * (0..3)
                            .map(|k| {
                                (0..3)
                                    .map(|l| b[i][k] * update.tangent[k * 6 + l] * b[j][l])
                                    .sum::<f64>()
                            })
                            .sum::<f64>();
                    if i == j {
                        out.diagonal[e + i] += stiffness;
                    } else {
                        out.upper[e] += stiffness;
                    }
                }
            }
        }
    }
    out.residual[0] -= pressure * INNER;
    out
}

fn norm(residual: &[f64]) -> f64 {
    if residual.iter().any(|value| !value.is_finite()) {
        return f64::INFINITY;
    }
    residual.iter().map(|v| v.abs()).fold(0.0, f64::max) / (PRESSURE * INNER)
}

fn correction(out: &Assembly) -> Result<Vec<f64>, String> {
    let mut diagonal = out.diagonal.clone();
    let mut rhs: Vec<_> = out.residual.iter().map(|v| -v).collect();
    for i in 1..diagonal.len() {
        if !diagonal[i - 1].is_finite() || diagonal[i - 1] <= 0.0 {
            return Err("nonpositive cylinder tangent pivot".into());
        }
        let factor = out.upper[i - 1] / diagonal[i - 1];
        diagonal[i] -= factor * out.upper[i - 1];
        rhs[i] -= factor * rhs[i - 1];
    }
    for i in (0..rhs.len()).rev() {
        if !diagonal[i].is_finite() || diagonal[i] <= 0.0 {
            return Err("nonpositive cylinder tangent pivot".into());
        }
        if i + 1 < rhs.len() {
            rhs[i] -= out.upper[i] * rhs[i + 1];
        }
        rhs[i] /= diagonal[i];
    }
    Ok(rhs)
}

fn reference(pressure: f64, r: f64) -> (f64, f64) {
    let yield_pressure = YIELD * (1.0 - (INNER / OUTER).powi(2)) / 3.0_f64.sqrt();
    if pressure <= yield_pressure {
        return super::lame(INNER, OUTER, pressure, r);
    }
    let c = super::interface_radius(INNER, OUTER, YIELD, pressure);
    super::loaded_stress(INNER, OUTER, c, YIELD, pressure, r)
}

fn field_error(count: usize, states: &[PlasticState], pressure: f64, unloading: bool) -> f64 {
    let mut error = 0.0;
    let mut volume = 0.0;
    for e in 0..count {
        for (q, xi) in GAUSS.into_iter().enumerate() {
            let (r, weight, _) = basis(e, count, xi);
            let (mut sr, mut st) = reference(if unloading { PRESSURE } else { pressure }, r);
            if unloading {
                let (ur, ut) = super::lame(INNER, OUTER, PRESSURE - pressure, r);
                sr -= ur;
                st -= ut;
            }
            let stress = &states[e * 2 + q].stress;
            error += weight * ((stress[0] - sr).powi(2) + (stress[1] - st).powi(2));
            volume += weight;
        }
    }
    (error / volume).sqrt() / YIELD
}

fn cycle(count: usize, increments: usize, elastic: bool) -> Result<Cycle, String> {
    let material = J2Material::new(
        200e9,
        POISSON,
        if elastic { 1e20 } else { YIELD },
        HardeningLaw::Perfect,
    );
    let mut states = vec![PlasticState::new_3d(); count * 2];
    let mut u = vec![0.0; count + 1];
    let mut equilibrium: f64 = 0.0;
    let mut loaded_error: f64 = 0.0;
    let mut peak_ep = vec![0.0; count * 2];
    let mut peak_plastic: f64 = 0.0;
    let mut unload_plastic_increment: f64 = 0.0;
    for step in 1..=2 * increments {
        let unloading = step > increments;
        let pressure = PRESSURE
            * if unloading {
                (2 * increments - step) as f64
            } else {
                step as f64
            }
            / increments as f64;
        let previous = u.clone();
        let mut accepted = None;
        for _ in 0..80 {
            let out = assemble(count, &u, &previous, &states, &material, pressure);
            let residual = norm(&out.residual);
            if !residual.is_finite() {
                return Err("nonfinite cylinder equilibrium residual".into());
            }
            if residual.is_finite() && residual < 1e-9 {
                equilibrium = equilibrium.max(residual);
                accepted = Some(out.states);
                break;
            }
            let delta = correction(&out)?;
            // Yield-surface activation changes the one-sided tangent. A full
            // Newton step permits the active set to change before reassembly;
            // an L-infinity residual line search can reject that transition.
            for (value, change) in u.iter_mut().zip(delta) {
                if !change.is_finite() {
                    return Err("nonfinite cylinder Newton correction".into());
                }
                *value += change;
            }
        }
        states = accepted
            .ok_or_else(|| format!("cylinder Newton iteration cap n={count}, step={step}"))?;
        loaded_error = loaded_error.max(field_error(count, &states, pressure, unloading));
        if step == increments {
            peak_ep = states.iter().map(|s| s.equiv_plastic_strain).collect();
            peak_plastic = peak_ep.iter().copied().fold(0.0, f64::max);
        }
        if unloading {
            for (state, peak) in states.iter().zip(&peak_ep) {
                unload_plastic_increment =
                    unload_plastic_increment.max(state.equiv_plastic_strain - peak);
            }
        }
    }
    Ok(Cycle {
        residual_error: field_error(count, &states, 0.0, true),
        loaded_error,
        equilibrium,
        unload_plastic_increment,
        minimum_hoop: states
            .iter()
            .map(|s| s.stress[1])
            .fold(f64::INFINITY, f64::min),
        final_stress: states.iter().map(|s| [s.stress[0], s.stress[1]]).collect(),
        peak_plastic,
        plastic_points: peak_ep.iter().filter(|ep| **ep > 1e-8).count(),
        steps: 2 * increments,
    })
}

pub(super) fn run() -> Result<QaResult, String> {
    let mut runs = Vec::new();
    for n in [16, 32, 64] {
        let run = cycle(n, 64, false)?;
        println!(
            "native autofrettage n={n}: residual field error={:.6e}, all-history error={:.6e}, equilibrium={:.3e}, ep={:.3e}, plastic points={}",
            run.residual_error,
            run.loaded_error,
            run.equilibrium,
            run.peak_plastic,
            run.plastic_points
        );
        runs.push(run);
    }
    let fine = &runs[2];
    let rate = runs
        .windows(2)
        .map(|pair| (pair[0].residual_error / pair[1].residual_error).ln() / 2.0_f64.ln())
        .fold(f64::INFINITY, f64::min);
    let refined = cycle(64, 128, false)?;
    let step_error = fine
        .final_stress
        .iter()
        .zip(&refined.final_stress)
        .map(|(a, b)| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2))
        .sum::<f64>()
        .sqrt()
        / ((fine.final_stress.len() as f64).sqrt() * YIELD);
    let control = cycle(64, 64, true)?;
    let control_error = control
        .final_stress
        .iter()
        .zip(&fine.final_stress)
        .map(|(a, b)| (a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2))
        .sum::<f64>()
        .sqrt()
        / ((fine.final_stress.len() as f64).sqrt() * YIELD);
    let mut result = QaResult::passed("fem.autofrettage-spatial.v1")
        .metric("numerical_solve_executed", 1.0)
        .minimum("numerical_solve_executed", 1.0)
        .metric("minimum_free_radial_unknowns", 17.0)
        .minimum("minimum_free_radial_unknowns", 17.0)
        .formulation("native J2 perfect plasticity; near-incompressible plane strain radial P1 B-bar FEM; pressure loading/unloading, all radial unknowns free")
        .dimension("axisymmetric long cylinder: spatial radial reduction, two material points per element")
        .reference("independent incompressible elastic/plastic interface and Lame elastic-unloading closed forms")
        .reference_checksum("analytic:plane-strain-perfect-plastic-cylinder:v1");
    for (name, value, bound) in [
        ("fine_residual_stress_error", fine.residual_error, 0.01),
        ("fine_history_stress_error", fine.loaded_error, 0.02),
        ("pressure_increment_refinement_error", step_error, 0.001),
        (
            "maximum_equilibrium_residual",
            runs.iter()
                .map(|r| r.equilibrium)
                .fold(refined.equilibrium, f64::max),
            1e-9,
        ),
        (
            "unloading_plastic_increment",
            fine.unload_plastic_increment,
            1e-9,
        ),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}={value} exceeds {bound}"));
        }
        result = result.metric(name, value).maximum(name, bound);
    }
    for (name, value, bound) in [
        ("residual_stress_convergence_rate", rate, 0.7),
        ("peak_plastic_strain", fine.peak_plastic, 1e-5),
        ("compressive_hoop_residual", -fine.minimum_hoop / YIELD, 0.2),
        ("elastic_control_error", control_error, 0.05),
        (
            "executed_pressure_steps",
            runs.iter().map(|r| r.steps).sum::<usize>() as f64 + refined.steps as f64,
            640.0,
        ),
    ] {
        if !value.is_finite() || value < bound {
            return Err(format!("{name}={value} below {bound}"));
        }
        result = result.metric(name, value).minimum(name, bound);
    }
    Ok(result)
}
