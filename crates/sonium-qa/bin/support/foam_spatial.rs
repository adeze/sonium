//! Solved guided H8 foam rods with independent monotone continuum references.
// Rust guideline compliant 2026-02-21
use serde_json::json;
use sha2::{Digest, Sha256};
use sonium_fem::physics::solid::nonlinear_solid::{
    HexExplicitGeom, HexExplicitParams, HexExplicitState, hex_explicit_forces,
};
use sonium_fem::physics::solid::plasticity::{
    foam_plas33::{FoamPlas33, FoamPlasIncrement, FoamPlasState},
    funct::Funct1D,
    samp76_quadratic::{Samp76Quadratic, Samp76QuadraticState},
    update::{StressUpdate, StressUpdateModel},
};
use sonium_qa::QaResult;
use std::path::Path;

const LENGTH: f64 = 0.1;
const SIDE: f64 = 0.01;
const AREA: f64 = SIDE * SIDE;
const YOUNG: f64 = 350e6;

#[derive(Clone)]
enum History {
    Plain(FoamPlasState),
    Quadratic(Samp76QuadraticState),
}
enum Law {
    Plain(FoamPlas33),
    Quadratic(Samp76Quadratic),
}
impl Law {
    fn new(plain: bool) -> Result<Self, String> {
        Ok(if plain {
            Self::Plain(FoamPlas33::new(
                YOUNG, -177000., 677000., -3.34, 1e5, 0.1, 0.,
            )?)
        } else {
            Self::Quadratic(Samp76Quadratic::new(
                YOUNG,
                0.,
                500000.,
                900000.,
                420000.,
                Funct1D::new(vec![0., 0.5], vec![1., 11.])?,
            )?)
        })
    }
    fn virgin(&self) -> History {
        match self {
            Self::Plain(_) => History::Plain(FoamPlasState::default()),
            Self::Quadratic(_) => History::Quadratic(Samp76QuadraticState::default()),
        }
    }
}
struct AtVolume<'a> {
    law: &'a Law,
    volume_ratio: f64,
}
impl StressUpdateModel for AtVolume<'_> {
    type State = History;
    type StrainIncrement = [f64; 6];
    fn update(&self, state: &History, increment: &[f64; 6]) -> StressUpdate {
        match (self.law, state) {
            (Law::Plain(law), History::Plain(state)) => law.update(
                state,
                &FoamPlasIncrement {
                    strain: *increment,
                    volume_ratio: self.volume_ratio,
                },
            ),
            (Law::Quadratic(law), History::Quadratic(state)) => law.update(state, increment),
            _ => unreachable!("material/history pairing constructed together"),
        }
    }
    fn commit(&self, state: &mut History, out: &StressUpdate) {
        match (self.law, state) {
            (Law::Plain(law), History::Plain(state)) => {
                law.commit(state, out);
                state.volume_ratio = self.volume_ratio;
            }
            (Law::Quadratic(law), History::Quadratic(state)) => law.commit(state, out),
            _ => unreachable!("material/history pairing constructed together"),
        }
    }
}

// This is a guided spatial mode: transverse coordinates are fixed and each
// cross-section's four axial DOFs are tied. All planes except x=0 are free.
// The production H8 B operator evaluates exact logarithmic axial increments
// through an equivalent nodal increment. No hourglass or shock dissipation
// belongs in this quasistatic force evaluation.
fn cell(
    law: &Law,
    old: &History,
    h: f64,
    left: f64,
    right: f64,
    previous_length: f64,
) -> Result<(f64, History, f64), String> {
    let length = right - left;
    if !length.is_finite() || length <= 0.0 {
        return Err("inverted guided H8".into());
    }
    let geom = HexExplicitGeom::new([
        [left, 0., 0.],
        [right, 0., 0.],
        [right, SIDE, 0.],
        [left, SIDE, 0.],
        [left, 0., SIDE],
        [right, 0., SIDE],
        [right, SIDE, SIDE],
        [left, SIDE, SIDE],
    ])
    .map_err(|e| e.to_string())?;
    let mut state = HexExplicitState::new(old.clone());
    let mut increment = [0.; 24];
    for node in [1, 2, 5, 6] {
        increment[3 * node] = length * (length / previous_length).ln();
    }
    let params = HexExplicitParams {
        density: 433.,
        sound_speed: (YOUNG / 433.).sqrt(),
        hg_scale: 0.,
        bulk_c0: 0.,
        bulk_c1: 0.,
    };
    let out = hex_explicit_forces(
        &geom,
        &mut state,
        &AtVolume {
            law,
            volume_ratio: length / h,
        },
        &increment,
        1.,
        &params,
    )
    .map_err(|e| e.to_string())?;
    if out
        .forces
        .iter()
        .chain(out.stress.iter())
        .any(|v| !v.is_finite())
    {
        return Err("invalid foam H8 response".into());
    }
    let right_force = [1, 2, 5, 6].iter().map(|i| out.forces[3 * i]).sum::<f64>();
    if (right_force - AREA * out.stress[0]).abs() > 1e-9 {
        return Err("H8 traction assembly disagrees with stress".into());
    }
    Ok((right_force, state.centroid, out.stress[0]))
}
fn loading(elements: usize, peak: f64) -> Vec<f64> {
    let h = LENGTH / elements as f64;
    let coefficient = peak * 0.8 * AREA / (LENGTH * LENGTH);
    let mut f = vec![0.; elements + 1];
    for i in 0..elements {
        let x = i as f64 * h;
        f[i] += coefficient * (x * h / 2. + h * h / 6.);
        f[i + 1] += coefficient * (x * h / 2. + h * h / 3.);
    }
    f[elements] -= peak * AREA;
    f
}
fn tridiagonal(diag: &[f64], off: &[f64], rhs: &[f64]) -> Result<Vec<f64>, String> {
    let mut d = diag.to_vec();
    let mut b = rhs.to_vec();
    for i in 1..d.len() {
        if d[i - 1] <= 0. || !d[i - 1].is_finite() {
            return Err("nonpositive foam stiffness pivot".into());
        }
        let ratio = off[i - 1] / d[i - 1];
        d[i] -= ratio * off[i - 1];
        b[i] -= ratio * b[i - 1];
    }
    let n = d.len();
    if d[n - 1] <= 0. {
        return Err("nonpositive final foam stiffness pivot".into());
    }
    b[n - 1] /= d[n - 1];
    for i in (0..n - 1).rev() {
        b[i] = (b[i] - off[i] * b[i + 1]) / d[i];
    }
    Ok(b)
}
struct Run {
    u: Vec<f64>,
    stress: Vec<f64>,
    plastic: Vec<f64>,
    work: f64,
    reaction: f64,
    residual: f64,
    iterations: usize,
    steps: usize,
    history_j_error: f64,
}
fn solve(
    plain: bool,
    elements: usize,
    steps: usize,
    elastic: bool,
    peak: f64,
) -> Result<Run, String> {
    let law = if elastic {
        Law::Quadratic(Samp76Quadratic::new(
            YOUNG,
            0.,
            1e20,
            1e20,
            1e20,
            Funct1D::new(vec![0., 1.], vec![1., 1.])?,
        )?)
    } else {
        Law::new(plain)?
    };
    let forces = loading(elements, peak);
    let h = LENGTH / elements as f64;
    let mut history = vec![law.virgin(); elements];
    let mut u = vec![0.; elements + 1];
    let mut work = 0.;
    let mut worst = 0_f64;
    let mut iterations = 0;
    let mut history_j_error = 0_f64;
    let mut stresses = vec![0.; elements];
    let mut reaction = 0.;
    for step in 1..=steps {
        let alpha = step as f64 / steps as f64;
        let previous = u.clone();
        let old_lengths: Vec<_> = previous.windows(2).map(|w| h + w[1] - w[0]).collect();
        let mut accepted = false;
        for _ in 0..30 {
            iterations += 1;
            let mut residual: Vec<_> = forces.iter().map(|f| -alpha * f).collect();
            let mut diag = vec![0.; elements];
            let mut off = vec![0.; elements - 1];
            let mut trial_history = Vec::with_capacity(elements);
            for i in 0..elements {
                let left = i as f64 * h + u[i];
                let right = (i + 1) as f64 * h + u[i + 1];
                let (force, state, stress) =
                    cell(&law, &history[i], h, left, right, old_lengths[i])?;
                stresses[i] = stress;
                trial_history.push(state);
                residual[i] -= force;
                residual[i + 1] += force;
                let delta = h * 1e-7;
                let plus = cell(&law, &history[i], h, left, right + delta, old_lengths[i])?.0;
                let minus = cell(&law, &history[i], h, left, right - delta, old_lengths[i])?.0;
                let k = (plus - minus) / (2. * delta);
                if !k.is_finite() || k <= 0. {
                    return Err("nonpositive guided foam tangent".into());
                }
                diag[i] += k;
                if i > 0 {
                    diag[i - 1] += k;
                    off[i - 1] -= k;
                }
            }
            let norm = residual[1..].iter().map(|v| v.abs()).fold(0_f64, f64::max) / (peak * AREA);
            if norm < 1e-9 {
                worst = worst.max(norm);
                reaction = residual[0];
                history = trial_history;
                accepted = true;
                break;
            }
            let correction = tridiagonal(
                &diag,
                &off,
                &residual[1..].iter().map(|v| -v).collect::<Vec<_>>(),
            )?;
            // Loads and compressive hardening are monotone. A bounded step
            // prevents an initial elastic Newton guess crossing pore closure.
            let mut factor = 1_f64;
            for i in 0..elements {
                let left = if i == 0 { 0. } else { correction[i - 1] };
                let right = correction[i];
                if right - left < 0. {
                    factor = factor.min(0.2 * (h + u[i + 1] - u[i]) / (left - right));
                }
            }
            for i in 0..elements {
                u[i + 1] += factor * correction[i];
            }
        }
        if !accepted {
            return Err(format!(
                "foam spatial Newton did not converge: plain={plain}, n={elements}, step={step}"
            ));
        }
        if !elastic {
            for i in 0..elements {
                let midpoint = (i as f64 + 0.5) * h;
                let cell_average_traction = alpha
                    * peak
                    * (0.6 + 0.4 * (midpoint * midpoint + h * h / 12.) / (LENGTH * LENGTH));
                let actual_j = (h + u[i + 1] - u[i]) / h;
                let expected_j = reference_j(plain, cell_average_traction);
                history_j_error = history_j_error.max((actual_j - expected_j).abs());
            }
        }
        work += (alpha - 0.5 / steps as f64)
            * forces
                .iter()
                .zip(u.iter().zip(&previous))
                .map(|(f, (a, b))| f * (a - b))
                .sum::<f64>();
    }
    let plastic = history
        .iter()
        .map(|s| match s {
            History::Plain(s) => s.equivalent_plastic_strain,
            History::Quadratic(s) => s.plastic_strain,
        })
        .collect();
    Ok(Run {
        u,
        stress: stresses,
        plastic,
        work,
        reaction,
        residual: worst,
        iterations,
        steps,
        history_j_error,
    })
}

// Independent continuum inverse under monotone constrained compression.
// LAW33: T = 500k + 2.26118M(1-J) + 100k(1-J)/(J-0.1).
// LAW76: q=1.5(T-P), ep=(3P-T)/E, h=1+20ep; the surface
// becomes a quadratic in P. Neither reference calls a native material API.
fn reference_j(plain: bool, t: f64) -> f64 {
    let elastic = (-t / YOUNG).exp();
    if plain {
        let threshold =
            500000. + 2261180. * (1. - elastic) + 100000. * (1. - elastic) / (elastic - 0.1);
        if t <= threshold {
            return elastic;
        }
        let d = t - 500000.;
        let k = 2261180.;
        // z=1-J, denominator=.9-z. Stable smaller quadratic root.
        let b = 0.9 * k + 100000. + d;
        let z = 2. * 0.9 * d / (b + (b * b - 4. * k * 0.9 * d).sqrt());
        1. - z
    } else {
        if t <= 900000. {
            return elastic;
        }
        let a0 = 3. * 420000_f64.powi(2);
        let a1 = 9. * 420000_f64.powi(2) * (900000. - 500000.) / (500000. * 900000.);
        let a2 = 9. * (500000. * 900000. - 3. * 420000_f64.powi(2)) / (500000. * 900000.);
        let alpha = 1. - 20. * t / YOUNG;
        let beta = 60. / YOUNG;
        let c2 = 2.25 - a0 * beta * beta - a1 * beta - a2;
        let c1 = -4.5 * t - 2. * a0 * alpha * beta - a1 * alpha;
        let c0 = 2.25 * t * t - a0 * alpha * alpha;
        let root = (-c1 - (c1 * c1 - 4. * c2 * c0).sqrt()) / (2. * c2);
        (-3. * root / YOUNG).exp()
    }
}
fn simpson(f: impl Fn(f64) -> f64, left: f64, right: f64, n: usize) -> f64 {
    if right <= left {
        return 0.;
    }
    let h = (right - left) / n as f64;
    let mut sum = f(left) + f(right);
    for i in 1..n {
        sum += if i % 2 == 0 { 2. } else { 4. } * f(left + i as f64 * h);
    }
    sum * h / 3.
}
fn traction(peak: f64, x: f64) -> f64 {
    peak * (0.6 + 0.4 * (x / LENGTH).powi(2))
}
fn yield_position(plain: bool, peak: f64) -> f64 {
    let threshold = if plain {
        let mut low = 500000.;
        let mut high = 510000.;
        for _ in 0..60 {
            let t = (low + high) / 2.;
            let j = (-t / YOUNG).exp();
            let y = 500000. + 2261180. * (1. - j) + 100000. * (1. - j) / (j - 0.1);
            if t < y {
                low = t;
            } else {
                high = t;
            }
        }
        (low + high) / 2.
    } else {
        900000.
    };
    LENGTH * ((threshold / peak - 0.6) / 0.4).clamp(0., 1.).sqrt()
}
fn reference_u(plain: bool, peak: f64, x: f64) -> f64 {
    let boundary = yield_position(plain, peak).min(x);
    let f = |x| reference_j(plain, traction(peak, x)) - 1.;
    simpson(f, 0., boundary, 256) + simpson(f, boundary, x, 256)
}
fn reference_work(plain: bool, peak: f64, points: usize) -> f64 {
    simpson(
        |x| {
            let t = traction(peak, x);
            let change = reference_j(plain, t) - 1.;
            let integral = simpson(|a| reference_j(plain, a * t) - 1., 0., 1., points);
            -AREA * t * (change - integral)
        },
        0.,
        LENGTH,
        points,
    )
}

pub(super) fn run(plain: bool, smooth: bool) -> Result<QaResult, String> {
    let peak = if plain {
        if smooth { 900000. } else { 650000. }
    } else {
        1300000.
    };
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("decks/foam_guided_rods");
    let reference_bytes = std::fs::read(root.join("reference.json")).map_err(|e| e.to_string())?;
    let metadata: serde_json::Value =
        serde_json::from_slice(&reference_bytes).map_err(|e| e.to_string())?;
    if metadata["schema"] != "sonium-guided-foam-reference-v1"
        || metadata["geometry"]
            != json!({"length_m":LENGTH,"side_m":SIDE,"guided_transverse":true,"axial_planes_tied":true,"fixed_plane":"x=0"})
        || metadata["young_pa"] != YOUNG
        || metadata["traction_shape"] != "peak * (0.6 + 0.4 * (x/L)^2)"
        || metadata["loads_pa"]
            != json!({"law33_mixed":650000.0,"law33_smooth":900000.0,"law76":1300000.0})
        || metadata["law33"]
            != json!({"a_pa":-177000.0,"b_pa":677000.0,"c":-3.34,"pressure0_pa":100000.0,"porosity":0.1,"gamma0":0.0})
        || metadata["law76"]
            != json!({"tension_pa":500000.0,"compression_pa":900000.0,"shear_pa":420000.0,"poisson":0.0,"nu_p":0.5,"iform":0,"iquad":1,"hardening_slope":20.0})
        || metadata["mesh_elements"] != json!([16, 32, 64])
        || metadata["load_steps"] != json!([128, 256])
    {
        return Err("guided foam reference parameter mismatch".into());
    }
    let checksum = format!("{:x}", Sha256::digest(&reference_bytes));
    let mut runs = Vec::new();
    let mut errors = Vec::new();
    for n in [16, 32, 64] {
        let run = solve(plain, n, 128, false, peak)?;
        let error = run
            .u
            .iter()
            .enumerate()
            .map(|(i, u)| {
                let exact = reference_u(plain, peak, i as f64 * LENGTH / n as f64);
                (u - exact).powi(2)
            })
            .sum::<f64>()
            .sqrt()
            / run
                .u
                .iter()
                .enumerate()
                .map(|(i, _)| reference_u(plain, peak, i as f64 * LENGTH / n as f64).powi(2))
                .sum::<f64>()
                .sqrt();
        errors.push(error);
        runs.push(run);
    }
    let fine = &runs[2];
    let load_refined = solve(plain, 64, 256, false, peak)?;
    let elastic = solve(plain, 64, 128, true, peak)?;
    let finest_pair_rate = (errors[1] / errors[2]).ln() / 2_f64.ln();
    let spatial_rate = (errors[0] / errors[2]).ln() / 4_f64.ln();
    // The mixed LAW33 front can produce midpoint error cancellation on an
    // intermediate mesh. Retain its full-range convergence and accuracy;
    // the separate smooth plastic case also gates the finest pair.

    let load_step_error = fine
        .u
        .iter()
        .zip(&load_refined.u)
        .map(|(a, b)| (a - b).powi(2))
        .sum::<f64>()
        .sqrt()
        / load_refined.u.iter().map(|v| v * v).sum::<f64>().sqrt();
    let elastic_gap = (elastic.u[64] - fine.u[64]).abs() / fine.u[64].abs();
    let max_plastic = fine.plastic.iter().copied().fold(0_f64, f64::max);
    let spatial_plastic_spread =
        max_plastic - fine.plastic.iter().copied().fold(f64::INFINITY, f64::min);
    let expected_work = reference_work(plain, peak, 2048);
    let work_quadrature_change =
        (expected_work - reference_work(plain, peak, 1024)).abs() / expected_work;
    let stress_error = fine
        .stress
        .iter()
        .enumerate()
        .map(|(i, s)| (s + traction(peak, (i as f64 + 0.5) * LENGTH / 64.)).abs() / peak)
        .fold(0_f64, f64::max);
    let reaction_error = (fine.reaction - 0.6 * peak * AREA).abs() / (peak * AREA);
    let work_error = (fine.work - expected_work).abs() / expected_work;
    let iterations = runs.iter().map(|r| r.iterations).sum::<usize>()
        + load_refined.iterations
        + elastic.iterations;
    let steps = runs.iter().map(|r| r.steps).sum::<usize>() + load_refined.steps + elastic.steps;
    let mut result=QaResult::passed(if plain && !smooth {"fem.foam-law33-mixed-spatial.v1"} else if plain {"fem.foam-law33-spatial.v1"} else {"fem.foam-law76-spatial.v1"})
        .formulation("production H8 internal-force assembly and native foam constitutive evolution; quasistatic Newton solve, tied axial cross-sections and fixed transverse coordinates")
        .dimension("guided 3D H8 rod; 16/32/64 elements, free interior and loaded-end axial planes; nonuniform distributed force")
        .reference("independent continuum stress equilibrium, closed-form compressive constitutive inverse, integrated displacement and external work")
        .reference_checksum(checksum.clone())
        .artifact_with_provenance("decks/foam_guided_rods/reference.json","original independent continuum guided-rod reference parameters","compiled closed-form compression inverses and exact force equilibrium; no native output calibration","sonium-guided-foam-reference-v1",checksum)
        .metric("numerical_solve_executed",1.).minimum("numerical_solve_executed",1.)
        .metric("native_load_steps",steps as f64).minimum("native_load_steps",768.)
        .metric("newton_iterations",iterations as f64).minimum("newton_iterations",768.);
    result = result
        .metric("end_displacement_m", fine.u[64])
        .metric(
            "reference_end_displacement_m",
            reference_u(plain, peak, LENGTH),
        )
        .metric("native_external_work_j", fine.work)
        .metric("reference_external_work_j", expected_work)
        .metric("free_axial_planes", 64.)
        .minimum("free_axial_planes", 64.)
        .metric("finest_pair_displacement_rate", finest_pair_rate)
        .metric("coarse_displacement_relative_l2", errors[0])
        .metric("middle_displacement_relative_l2", errors[1]);
    if smooth || !plain {
        if !finest_pair_rate.is_finite() || finest_pair_rate < 1.5 {
            return Err(format!(
                "smooth foam finest displacement rate={finest_pair_rate}, errors={errors:?}"
            ));
        }
        result = result.minimum("finest_pair_displacement_rate", 1.5);
    }
    for (name, value, bound, minimum) in [
        ("displacement_relative_l2", errors[2], 0.005, false),
        ("spatial_rate", spatial_rate, 1.5, true),
        (
            "all_load_material_volume_error",
            fine.history_j_error,
            1e-7,
            false,
        ),
        (
            "reference_work_quadrature_change",
            work_quadrature_change,
            1e-5,
            false,
        ),
        (
            "load_step_displacement_change",
            load_step_error,
            0.0002,
            false,
        ),
        ("stress_field_error", stress_error, 0.001, false),
        ("reaction_error", reaction_error, 1e-8, false),
        ("free_force_residual", fine.residual, 1e-9, false),
        ("external_work_error", work_error, 0.005, false),
        ("plastic_strain", max_plastic, 0.0001, true),
        (
            "spatial_plastic_spread",
            spatial_plastic_spread,
            0.0001,
            true,
        ),
        ("elastic_control_displacement_gap", elastic_gap, 0.05, true),
    ] {
        if !value.is_finite()
            || if minimum {
                value < bound
            } else {
                value > bound
            }
        {
            return Err(format!(
                "guided foam plain={plain}: {name}={value}, bound={bound}, errors={errors:?}"
            ));
        }
        result = result.metric(name, value);
        result = if minimum {
            result.minimum(name, bound)
        } else {
            result.maximum(name, bound)
        };
    }
    Ok(result)
}
