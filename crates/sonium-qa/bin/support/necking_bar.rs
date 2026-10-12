//! Spatial finite-strain bar evolution with native J2 integration.
//!
//! A slender-bar reduction: axial P1 displacement, two Gauss material points
//! per element, traction-free transverse stresses and evolving area. The
//! smooth area imperfection resolves diffuse necking before the first local
//! turning point. No post-localization mesh-objectivity or fracture claim.
// Rust guideline compliant 2026-02-21

use sonium_fem::physics::solid::plasticity::{
    FiniteStrainJ2Material, FiniteStrainState, HardeningLaw, StressUpdateModel,
};
use sonium_qa::QaResult;

const E: f64 = 200e9;
const NU: f64 = 0.3;
const YIELD: f64 = 250e6;
const Q: f64 = 300e6;
const B: f64 = 5.0;
const AREA: f64 = 1e-4;
const GAUSS: [f64; 2] = [-0.5773502691896258, 0.5773502691896258];

#[derive(Clone)]
struct PointState {
    material: FiniteStrainState,
    axial: f64,
    lateral: f64,
}

impl Default for PointState {
    fn default() -> Self {
        Self {
            material: FiniteStrainState::new(),
            axial: 1.0,
            lateral: 1.0,
        }
    }
}

struct PointTrial {
    state: PointState,
    force: f64,
    tangent: f64,
    transverse_error: f64,
}

fn material(plastic: bool) -> FiniteStrainJ2Material {
    FiniteStrainJ2Material::new(
        E,
        NU,
        if plastic { YIELD } else { 1e15 },
        HardeningLaw::VoceSaturation { q: Q, b: B },
    )
}

fn initial_area(x: f64, imperfection: f64) -> f64 {
    AREA * (1.0 - imperfection * (-0.5 * ((x - 0.5) / 0.1).powi(2)).exp())
}

fn point_trial(
    old: &PointState,
    axial: f64,
    a0: f64,
    mat: &FiniteStrainJ2Material,
) -> Result<PointTrial, String> {
    if axial <= 0.0 || !axial.is_finite() {
        return Err("bar inverted during Newton trial".into());
    }
    let dx = (axial / old.axial).ln();
    let mut dt = -0.5 * dx;
    for _ in 0..30 {
        let increment = [dx.exp(), 0.0, 0.0, 0.0, dt.exp(), 0.0, 0.0, 0.0, dt.exp()];
        let update = mat.update(&old.material, &increment);
        let residual = 0.5 * (update.stress[1] + update.stress[2]);
        let c = &update.tangent;
        let transverse = c[7] + c[8];
        if !transverse.is_finite() || transverse <= 0.0 {
            return Err("invalid transverse tangent".into());
        }
        if residual.abs() / YIELD < 1e-12 {
            let lateral = old.lateral * dt.exp();
            let slope = -c[6] / transverse;
            let condensed = c[0] + (c[1] + c[2]) * slope;
            let area = a0 * lateral * lateral;
            let mut state = old.material.clone();
            mat.commit(&mut state, &update);
            return Ok(PointTrial {
                state: PointState {
                    material: state,
                    axial,
                    lateral,
                },
                force: area * update.stress[0],
                tangent: area / axial * (condensed + 2.0 * update.stress[0] * slope),
                transverse_error: update.stress[1].abs().max(update.stress[2].abs()) / YIELD,
            });
        }
        dt -= residual / transverse;
    }
    Err("bar traction-free transverse solve failed".into())
}

struct Assembly {
    residual: Vec<f64>,
    diagonal: Vec<f64>,
    upper: Vec<f64>,
    states: Vec<PointState>,
    transverse_error: f64,
}

fn assemble(
    u: &[f64],
    old: &[PointState],
    imperfection: f64,
    mat: &FiniteStrainJ2Material,
) -> Result<Assembly, String> {
    let n = u.len() - 1;
    let h = 1.0 / n as f64;
    let mut out = Assembly {
        residual: vec![0.0; n + 1],
        diagonal: vec![0.0; n + 1],
        upper: vec![0.0; n],
        states: Vec::with_capacity(2 * n),
        transverse_error: 0.0,
    };
    for e in 0..n {
        let axial = 1.0 + (u[e + 1] - u[e]) / h;
        for (q, xi) in GAUSS.into_iter().enumerate() {
            let x = (e as f64 + 0.5 * (1.0 + xi)) * h;
            let p = point_trial(&old[2 * e + q], axial, initial_area(x, imperfection), mat)?;
            // Reference-coordinate virtual work: integral P A0 du/dX dX.
            // P A0 = sigma A, with A = A0 lambda_transverse^2.
            let force = 0.5 * p.force;
            let stiffness = 0.5 * p.tangent / h;
            out.residual[e] -= force;
            out.residual[e + 1] += force;
            out.diagonal[e] += stiffness;
            out.diagonal[e + 1] += stiffness;
            out.upper[e] -= stiffness;
            out.transverse_error = out.transverse_error.max(p.transverse_error);
            out.states.push(p.state);
        }
    }
    Ok(out)
}

fn solve_tridiagonal(diagonal: &[f64], upper: &[f64], rhs: &[f64]) -> Result<Vec<f64>, String> {
    let mut d = diagonal.to_vec();
    let mut r = rhs.to_vec();
    for i in 1..d.len() {
        if d[i - 1].abs() < 1e-20 || !d[i - 1].is_finite() {
            return Err("singular bar tangent".into());
        }
        let factor = upper[i - 1] / d[i - 1];
        d[i] -= factor * upper[i - 1];
        r[i] -= factor * r[i - 1];
    }
    let mut x = vec![0.0; d.len()];
    for i in (0..d.len()).rev() {
        x[i] = (r[i]
            - if i + 1 < d.len() {
                upper[i] * x[i + 1]
            } else {
                0.0
            })
            / d[i];
    }
    if x.iter().any(|v| !v.is_finite()) {
        return Err("nonfinite bar correction".into());
    }
    Ok(x)
}

struct History {
    u: Vec<f64>,
    states: Vec<PointState>,
    peak_force: f64,
    peak_extension: f64,
    final_force: f64,
    equilibrium: f64,
    transverse: f64,
    minimum_area_ratio: f64,
    strain_amplification: f64,
    steps: usize,
    stress_history_error: f64,
    area_history_error: f64,
}

fn evolve(
    n: usize,
    steps: usize,
    imperfection: f64,
    end_extension: f64,
    plastic: bool,
) -> Result<History, String> {
    let mat = material(plastic);
    let mut u = vec![0.0; n + 1];
    let mut states = vec![PointState::default(); 2 * n];
    let mut history = History {
        u: Vec::new(),
        states: Vec::new(),
        peak_force: 0.0,
        peak_extension: 0.0,
        final_force: 0.0,
        equilibrium: 0.0,
        transverse: 0.0,
        minimum_area_ratio: 1.0,
        strain_amplification: 1.0,
        steps,
        stress_history_error: 0.0,
        area_history_error: 0.0,
    };
    for step in 1..=steps {
        let extension = end_extension * step as f64 / steps as f64;
        let increment = extension - u[n];
        for (i, value) in u.iter_mut().enumerate() {
            *value += increment * i as f64 / n as f64;
        }
        u[0] = 0.0;
        u[n] = extension;
        let mut converged = None;
        for _ in 0..200 {
            let trial = assemble(&u, &states, imperfection, &mat)?;
            let norm = trial.residual[1..n]
                .iter()
                .map(|v| v * v)
                .sum::<f64>()
                .sqrt()
                / (AREA * YIELD);
            if norm < 1e-9 {
                history.equilibrium = history.equilibrium.max(norm);
                converged = Some(trial);
                break;
            }
            let rhs: Vec<_> = trial.residual[1..n].iter().map(|v| -v).collect();
            let correction =
                solve_tridiagonal(&trial.diagonal[1..n], &trial.upper[1..n - 1], &rhs)?;
            let mut accepted = false;
            let mut alpha = 1.0;
            for _ in 0..16 {
                let mut candidate = u.clone();
                for (v, du) in candidate[1..n].iter_mut().zip(&correction) {
                    *v += alpha * du;
                }
                if let Ok(next) = assemble(&candidate, &states, imperfection, &mat) {
                    let next_norm = next.residual[1..n]
                        .iter()
                        .map(|v| v * v)
                        .sum::<f64>()
                        .sqrt()
                        / (AREA * YIELD);
                    if next_norm < norm {
                        u = candidate;
                        accepted = true;
                        break;
                    }
                }
                alpha *= 0.5;
            }
            if !accepted {
                return Err(format!(
                    "bar Newton line search stalled: n={n},step={step},residual={norm}"
                ));
            }
        }
        let trial = converged.ok_or_else(|| format!("bar Newton failed at step {step}"))?;
        history.transverse = history.transverse.max(trial.transverse_error);
        history.final_force = trial.residual[n];
        if history.final_force > history.peak_force {
            history.peak_force = history.final_force;
            history.peak_extension = extension;
        }
        for point in &trial.states {
            let (sigma, lateral) = reference_uniaxial(point.axial, plastic);
            history.stress_history_error = history
                .stress_history_error
                .max((point.material.stress[0] - sigma).abs() / YIELD);
            history.area_history_error = history
                .area_history_error
                .max((point.lateral.powi(2) - lateral.powi(2)).abs() / lateral.powi(2));
        }
        states = trial.states;
    }
    let center = &states[(n / 2) * 2];
    let end = &states[0];
    history.minimum_area_ratio = center.lateral.powi(2) / end.lateral.powi(2);
    history.strain_amplification = center.axial.ln() / end.axial.ln();
    history.u = u;
    history.states = states;
    Ok(history)
}

// Independent monotone uniaxial solution. For native logarithmic J2 under
// transverse traction zero: epsilon = ep + sigma/E and ln(A/A0) =
// -ep - 2 nu sigma/E. The Voce force derivative supplies the turning point.
fn stress(ep: f64) -> f64 {
    YIELD + Q * (1.0 - (-B * ep).exp())
}
fn force_density(ep: f64) -> f64 {
    stress(ep) * (-ep - 2.0 * NU * stress(ep) / E).exp()
}
fn critical_ep() -> f64 {
    let mut lo = 0.0;
    let mut hi = 1.0;
    for _ in 0..80 {
        let ep = 0.5 * (lo + hi);
        let slope = Q * B * (-B * ep).exp();
        if slope * (1.0 - 2.0 * NU * stress(ep) / E) > stress(ep) {
            lo = ep;
        } else {
            hi = ep;
        }
    }
    0.5 * (lo + hi)
}

fn reference_uniaxial(axial: f64, plastic: bool) -> (f64, f64) {
    let epsilon = axial.ln();
    let mut ep = 0.0;
    if plastic && epsilon > YIELD / E {
        let mut lo = 0.0;
        let mut hi = epsilon;
        for _ in 0..70 {
            let midpoint = 0.5 * (lo + hi);
            if midpoint + stress(midpoint) / E < epsilon {
                lo = midpoint;
            } else {
                hi = midpoint;
            }
        }
        ep = 0.5 * (lo + hi);
    }
    let sigma = E * (epsilon - ep);
    (sigma, (-0.5 * ep - NU * sigma / E).exp())
}

fn reference_stretch(force: f64, area: f64) -> f64 {
    let nominal = force / area;
    let yield_nominal = YIELD * (-2.0 * NU * YIELD / E).exp();
    if nominal < yield_nominal {
        let mut lo = 0.0;
        let mut hi = YIELD / E;
        for _ in 0..70 {
            let eps = 0.5 * (lo + hi);
            if E * eps * (-2.0 * NU * eps).exp() < nominal {
                lo = eps;
            } else {
                hi = eps;
            }
        }
        return (0.5 * (lo + hi)).exp();
    }
    let mut lo = 0.0;
    let mut hi = critical_ep();
    for _ in 0..70 {
        let ep = 0.5 * (lo + hi);
        if force_density(ep) < nominal {
            lo = ep;
        } else {
            hi = ep;
        }
    }
    let ep = 0.5 * (lo + hi);
    (ep + stress(ep) / E).exp()
}

fn reference(imperfection: f64, force: f64) -> (f64, Vec<f64>) {
    // Fine independent integration, not the native FE grid or quadrature.
    const N: usize = 20000;
    let mut displacement = vec![0.0; N + 1];
    for i in 0..N {
        let x = (i as f64 + 0.5) / N as f64;
        displacement[i + 1] = displacement[i]
            + (reference_stretch(force, initial_area(x, imperfection)) - 1.0) / N as f64;
    }
    (displacement[N], displacement)
}

pub(super) fn run() -> Result<QaResult, String> {
    let ep = critical_ep();
    let exact_peak_extension = (ep + stress(ep) / E).exp() - 1.0;
    let homogeneous = evolve(16, 400, 0.0, 0.4, true)?;
    let target_force = 0.995 * initial_area(0.5, 0.01) * force_density(ep);
    let (end_extension, reference_u) = reference(0.01, target_force);
    let mut runs = Vec::new();
    let mut errors = Vec::new();
    for n in [16, 32, 64] {
        let run = evolve(n, 200, 0.01, end_extension, true)?;
        let error = run
            .u
            .iter()
            .enumerate()
            .map(|(i, u)| {
                let position = i as f64 * 20000.0 / n as f64;
                let index = position.floor() as usize;
                let fraction = position - index as f64;
                let reference = if index == 20000 {
                    reference_u[index]
                } else {
                    reference_u[index] * (1.0 - fraction) + reference_u[index + 1] * fraction
                };
                (u - reference).powi(2)
            })
            .sum::<f64>()
            .sqrt()
            / run.u.iter().map(|u| u * u).sum::<f64>().sqrt();
        println!(
            "native bar n={n}: force={:.9e}, displacement error={error:.3e}, strain amplification={:.6}, area ratio={:.6}",
            run.final_force, run.strain_amplification, run.minimum_area_ratio
        );
        errors.push(error);
        runs.push(run);
    }
    let fine = &runs[2];
    let elastic = evolve(64, 200, 0.01, end_extension, false)?;
    let temporal = evolve(64, 400, 0.01, end_extension, true)?;
    println!(
        "homogeneous: peak extension={:.9e}, exact={exact_peak_extension:.9e}, peak force={:.9e}",
        homogeneous.peak_extension, homogeneous.peak_force
    );
    let mut result=QaResult::passed("fem.necking-bar-spatial.v1")
        .formulation("axial P1 slender bar, native finite-strain J2, plane-stress condensation, evolving area, Newton equilibrium; endpoints prescribed, interior displacements free")
        .dimension("1D spatial bar with 3D material points; 16/32/64 elements, two Gauss points; diffuse necking onset only")
        .reference("independent Voce uniaxial turning point and nonuniform equilibrium quadrature; no post-localization or 3D Bridgman claim")
        .reference_checksum("analytic:voce-logarithmic-j2-slender-bar:v1")
        .metric("numerical_solve_executed",1.0).minimum("numerical_solve_executed",1.0)
        .metric("native_steps",(homogeneous.steps+runs.iter().map(|r| r.steps).sum::<usize>()+elastic.steps+temporal.steps) as f64).minimum("native_steps",1600.0);
    for (name, value, bound) in [
        (
            "stress_history_error",
            runs.iter()
                .map(|r| r.stress_history_error)
                .chain([
                    homogeneous.stress_history_error,
                    elastic.stress_history_error,
                    temporal.stress_history_error,
                ])
                .fold(0.0, f64::max),
            1e-7,
        ),
        (
            "area_history_error",
            runs.iter()
                .map(|r| r.area_history_error)
                .chain([
                    homogeneous.area_history_error,
                    elastic.area_history_error,
                    temporal.area_history_error,
                ])
                .fold(0.0, f64::max),
            1e-9,
        ),
        (
            "peak_extension_error",
            (homogeneous.peak_extension - exact_peak_extension).abs() / exact_peak_extension,
            0.01,
        ),
        (
            "peak_force_error",
            (homogeneous.peak_force - AREA * force_density(ep)).abs() / (AREA * force_density(ep)),
            0.001,
        ),
        ("nonuniform_displacement_error", errors[2], 0.001),
        (
            "reaction_error",
            (fine.final_force - target_force).abs() / target_force,
            0.001,
        ),
        (
            "free_force_residual",
            runs.iter()
                .map(|r| r.equilibrium)
                .chain([
                    homogeneous.equilibrium,
                    elastic.equilibrium,
                    temporal.equilibrium,
                ])
                .fold(0.0, f64::max),
            1e-9,
        ),
        (
            "transverse_stress_error",
            runs.iter()
                .map(|r| r.transverse)
                .chain([
                    homogeneous.transverse,
                    elastic.transverse,
                    temporal.transverse,
                ])
                .fold(0.0, f64::max),
            1e-9,
        ),
        (
            "increment_reaction_error",
            (fine.final_force - temporal.final_force).abs() / fine.final_force,
            1e-5,
        ),
    ] {
        if !value.is_finite() || value > bound {
            return Err(format!("{name}={value} exceeds {bound}"));
        }
        result = result.metric(name, value).maximum(name, bound);
    }
    for (name, value, bound) in [
        ("spatial_rate", (errors[1] / errors[2]).log2(), 1.5),
        ("strain_amplification", fine.strain_amplification, 1.1),
        (
            "relative_neck_area_loss",
            1.0 - fine.minimum_area_ratio,
            0.01,
        ),
        (
            "plastic_strain",
            fine.states
                .iter()
                .map(|s| s.material.equiv_plastic_strain)
                .fold(0.0, f64::max),
            0.1,
        ),
        (
            "elastic_counterfactual_force_gap",
            (elastic.final_force - fine.final_force).abs() / fine.final_force,
            10.0,
        ),
        (
            "post_peak_force_drop",
            (homogeneous.peak_force - homogeneous.final_force) / homogeneous.peak_force,
            0.01,
        ),
    ] {
        if !value.is_finite() || value < bound {
            return Err(format!("{name}={value} below {bound}"));
        }
        result = result.metric(name, value).minimum(name, bound);
    }
    Ok(result)
}
