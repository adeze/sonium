//! Follow actual steady fan roots through a parameter fold without changing the target.
//!
//! Intermediate viscosity is explicit. Native state derivatives retain every
//! equation; parameter differences are diagnostic only, never adjoint evidence.
//! The exact original viscosity, conservation and mesh gates remain required.

// Rust guideline compliant 2026-02-21

use sonium_sim_pcfan::{
    PcFanConfig, PcFanDesignVariables, evaluate_pc_fan_cfd_newton_state_linearization,
};
use sonium_solvers::{CsrMatrix, direct::SparseLuFactorization, traits::LinearOperator};

type Error = Box<dyn std::error::Error>;
const RAW_TOLERANCE: f64 = 1e-8;
const PHASE_TOLERANCE: f64 = 1e-8;

#[derive(Clone)]
struct Root {
    state: Vec<f64>,
    log_viscosity: f64,
}

struct Evaluation {
    residual: Vec<f64>,
    jacobian: CsrMatrix<f64>,
    parameter_derivative: Vec<f64>,
}

struct Phase {
    predictor: Root,
    gradient: Vec<f64>,
    parameter_gradient: f64,
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

impl Phase {
    fn value(&self, root: &Root) -> f64 {
        self.gradient
            .iter()
            .zip(&root.state)
            .zip(&self.predictor.state)
            .map(|((g, a), b)| g * (a - b))
            .sum::<f64>()
            + self.parameter_gradient * (root.log_viscosity - self.predictor.log_viscosity)
    }
}

fn predictor(a: &Root, b: &Root, length: f64) -> Result<Phase, Error> {
    if a.state.len() != b.state.len() || a.state.is_empty() || !length.is_finite() || length <= 0.0
    {
        return Err("invalid secant roots or arclength".into());
    }
    // Each coordinate uses the same scale at both roots. RMS state distance
    // prevents a larger mesh from changing the relative parameter metric.
    let scales = a
        .state
        .iter()
        .zip(&b.state)
        .map(|(a, b)| a.abs().max(b.abs()).max(0.05))
        .collect::<Vec<_>>();
    let mut tangent = a
        .state
        .iter()
        .zip(&b.state)
        .zip(&scales)
        .map(|((a, b), d)| (b - a) / d)
        .collect::<Vec<_>>();
    let mut parameter = b.log_viscosity - a.log_viscosity;
    let n = tangent.len() as f64;
    let size = (norm(&tangent).powi(2) / n + parameter.powi(2)).sqrt();
    if !size.is_finite() || size == 0.0 {
        return Err("invalid or duplicate continuation roots".into());
    }
    for t in &mut tangent {
        *t /= size;
    }
    parameter /= size;
    let predicted = Root {
        state: b
            .state
            .iter()
            .zip(&scales)
            .zip(&tangent)
            .map(|((u, d), t)| u + length * d * t)
            .collect(),
        log_viscosity: b.log_viscosity + length * parameter,
    };
    Ok(Phase {
        predictor: predicted,
        gradient: tangent
            .iter()
            .zip(scales)
            .map(|(t, d)| t / (n * d))
            .collect(),
        parameter_gradient: parameter,
    })
}

fn augmented(e: &Evaluation, phase: &Phase) -> Result<CsrMatrix<f64>, Error> {
    let n = e.residual.len();
    if e.jacobian.num_rows != n
        || e.jacobian.num_cols != n
        || e.parameter_derivative.len() != n
        || phase.gradient.len() != n
    {
        return Err("augmented continuation dimensions differ".into());
    }
    let mut rows = vec![0];
    let mut columns = Vec::new();
    let mut values = Vec::new();
    for row in 0..n {
        for i in e.jacobian.row_range(row) {
            columns.push(e.jacobian.col_indices[i]);
            values.push(e.jacobian.values[i]);
        }
        if e.parameter_derivative[row] != 0.0 {
            columns.push(n);
            values.push(e.parameter_derivative[row]);
        }
        rows.push(values.len());
    }
    for (column, &value) in phase.gradient.iter().enumerate() {
        if value != 0.0 {
            columns.push(column);
            values.push(value);
        }
    }
    if phase.parameter_gradient != 0.0 {
        columns.push(n);
        values.push(phase.parameter_gradient);
    }
    rows.push(values.len());
    Ok(CsrMatrix::try_from_raw_parts(
        n + 1,
        n + 1,
        rows,
        columns,
        values,
    )?)
}

fn correct(
    phase: &Phase,
    evaluate: &impl Fn(&Root) -> Result<Evaluation, Error>,
) -> Result<(Root, usize), Error> {
    let mut root = phase.predictor.clone();
    for iteration in 0..=20 {
        if !root.log_viscosity.is_finite() || !(0.0..=65536_f64.ln()).contains(&root.log_viscosity)
        {
            return Err("continuation parameter outside declared positive viscosity range".into());
        }
        let e = evaluate(&root)?;
        let raw = norm(&e.residual);
        let constraint = phase.value(&root);
        println!(
            "Arc corrector {iteration}: viscosity={:.12e},raw={raw:.12e},phase={constraint:.12e}",
            root.log_viscosity.exp()
        );
        if raw.is_finite() && raw <= RAW_TOLERANCE && constraint.abs() <= PHASE_TOLERANCE {
            return Ok((root, iteration));
        }
        if !raw.is_finite() || !constraint.is_finite() || iteration == 20 {
            break;
        }
        let matrix = augmented(&e, phase)?;
        let weights = (0..matrix.num_rows)
            .map(|row| {
                let size = norm(&matrix.values[matrix.row_range(row)]);
                if size == 0.0 { 1.0 } else { size.recip() }
            })
            .collect::<Vec<_>>();
        if weights.iter().any(|w| !w.is_finite() || *w <= 0.0) {
            return Err("invalid augmented equation weights".into());
        }
        let mut residual = e.residual.clone();
        residual.push(constraint);
        let rhs = residual.iter().map(|v| -v).collect::<Vec<_>>();
        let delta = SparseLuFactorization::factorize_supernodal(&matrix)?.solve(&rhs)?;
        let error = matrix
            .apply(&delta)
            .iter()
            .zip(&rhs)
            .map(|(a, b)| a - b)
            .collect::<Vec<_>>();
        let relative = norm(&error) / norm(&rhs);
        println!("Arc actual augmented linear residual={relative:.12e}");
        if !relative.is_finite() || relative > 1e-8 {
            return Err("augmented linear equation did not converge".into());
        }
        let merit = |r: &[f64]| {
            norm(
                &r.iter()
                    .zip(&weights)
                    .map(|(r, w)| r * w)
                    .collect::<Vec<_>>(),
            )
        };
        let current = merit(&residual);
        let mut accepted = None;
        let mut step = 1.0;
        for _ in 0..24 {
            let trial = Root {
                state: root
                    .state
                    .iter()
                    .zip(&delta)
                    .map(|(v, d)| v + step * d)
                    .collect(),
                log_viscosity: root.log_viscosity + step * delta[root.state.len()],
            };
            if (0.0..=65536_f64.ln()).contains(&trial.log_viscosity) {
                let actual = evaluate(&trial)?;
                let mut r = actual.residual;
                r.push(phase.value(&trial));
                let actual_merit = merit(&r);
                if actual_merit.is_finite() && actual_merit <= (1.0 - 1e-4 * step) * current {
                    accepted = Some(trial);
                    break;
                }
            }
            step *= 0.5;
        }
        root = accepted.ok_or("augmented corrector line search exhausted")?;
    }
    Err("augmented corrector budget exhausted".into())
}

fn read_root(path: &str) -> Result<(serde_json::Value, Root), Error> {
    let saved: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
    let multiplier = saved["viscosity_multiplier"]
        .as_f64()
        .ok_or("root viscosity missing")?;
    if !multiplier.is_finite()
        || !(1.0..=65536.0).contains(&multiplier)
        || saved["actual_stage_residual"]
            .as_f64()
            .is_none_or(|r| !r.is_finite() || r < 0.0 || r > RAW_TOLERANCE)
    {
        return Err("input must be an actual converged declared-viscosity root".into());
    }
    let root = Root {
        state: serde_json::from_value(saved["state"].clone())?,
        log_viscosity: multiplier.ln(),
    };
    if root.state.is_empty() || root.state.iter().any(|v| !v.is_finite()) {
        return Err("invalid root state".into());
    }
    Ok((saved, root))
}

fn main() -> Result<(), Error> {
    let (a, mut previous) = read_root(&std::env::var("P2_PCFAN_ARC_ROOT_A")?)?;
    let (b, mut current) = read_root(&std::env::var("P2_PCFAN_ARC_ROOT_B")?)?;
    if a["configuration"] != b["configuration"] || a["design"] != b["design"] {
        return Err("continuation roots differ in original physical inputs".into());
    }
    let original: PcFanConfig = serde_json::from_value(a["configuration"].clone())?;
    let design: PcFanDesignVariables = serde_json::from_value(a["design"].clone())?;
    for (saved, root) in [(&a, &previous), (&b, &current)] {
        let mut config = original.clone();
        config.operating_point.dynamic_viscosity_pa_s *= root.log_viscosity.exp();
        // The logarithm round trip is checked numerically; every other input
        // must still match its declared effective configuration exactly.
        let mut expected = saved["effective_configuration"].clone();
        let declared = expected["operating_point"]["dynamic_viscosity_pa_s"]
            .as_f64()
            .ok_or("root effective viscosity missing")?;
        let actual = config.operating_point.dynamic_viscosity_pa_s;
        if !declared.is_finite() || (declared - actual).abs() > 32.0 * f64::EPSILON * actual.abs() {
            return Err("root declared effective viscosity differs".into());
        }
        expected["operating_point"]["dynamic_viscosity_pa_s"] =
            serde_json::to_value(config.operating_point.dynamic_viscosity_pa_s)?;
        if expected != serde_json::to_value(&config)? {
            return Err("root effective physics differs".into());
        }
    }
    let raw_evaluate = |root: &Root| -> Result<(Vec<f64>, CsrMatrix<f64>), Error> {
        let mut config = original.clone();
        config.operating_point.dynamic_viscosity_pa_s *= root.log_viscosity.exp();
        let observed =
            evaluate_pc_fan_cfd_newton_state_linearization(&config, &design, &root.state)?;
        observed.validate()?;
        if observed.state != root.state {
            return Err("assembly changed the supplied continuation state".into());
        }
        let j = observed.state_jacobian;
        Ok((
            observed.state_residual,
            CsrMatrix::try_from_raw_parts(
                j.dimension,
                j.dimension,
                j.row_ptrs,
                j.column_indices,
                j.values,
            )?,
        ))
    };
    let evaluate = |root: &Root| -> Result<Evaluation, Error> {
        let (residual, jacobian) = raw_evaluate(root)?;
        // Only the diagnostic continuation parameter is differenced. The
        // production state tangent remains native and analytic.
        let h = 1e-5;
        let mut plus = root.clone();
        plus.log_viscosity += h;
        let p = raw_evaluate(&plus)?.0;
        let parameter_derivative = if root.log_viscosity >= h {
            let mut minus = root.clone();
            minus.log_viscosity -= h;
            let m = raw_evaluate(&minus)?.0;
            p.iter().zip(m).map(|(p, m)| (p - m) / (2.0 * h)).collect()
        } else {
            p.iter().zip(&residual).map(|(p, r)| (p - r) / h).collect()
        };
        Ok(Evaluation {
            residual,
            jacobian,
            parameter_derivative,
        })
    };
    for root in [&previous, &current] {
        if norm(&raw_evaluate(root)?.0) > RAW_TOLERANCE {
            return Err("saved root failed independent reassembly".into());
        }
    }
    let budget: usize = std::env::args().nth(1).map_or(Ok(100), |v| v.parse())?;
    let mut length: f64 = std::env::args().nth(2).map_or(Ok(0.1), |v| v.parse())?;
    if budget == 0 || !length.is_finite() || !(1e-4..=0.5).contains(&length) {
        return Err("positive bounded branch budget and length required".into());
    }
    let path = std::path::PathBuf::from(std::env::var("P2_PCFAN_CHECKPOINT")?);
    let mut iterations = 0;
    for branch_step in 0..budget {
        let mut accepted = None;
        for _ in 0..8 {
            let mut phase = predictor(&previous, &current, length)?;
            let original_target = phase.predictor.log_viscosity <= 0.0;
            if original_target {
                // End at the exact original viscosity, rather than admitting
                // an interpolated or slightly modified final parameter.
                phase.predictor.log_viscosity = 0.0;
                phase.gradient.fill(0.0);
                phase.parameter_gradient = 1.0;
            }
            let corrected = correct(&phase, &evaluate).and_then(|(mut root, count)| {
                if original_target {
                    root.log_viscosity = 0.0;
                    let exact_residual = norm(&raw_evaluate(&root)?.0);
                    if !exact_residual.is_finite() || exact_residual > RAW_TOLERANCE {
                        return Err("exact original-viscosity root did not converge".into());
                    }
                }
                Ok((root, count))
            });
            match corrected {
                Ok(result) => {
                    accepted = Some(result);
                    break;
                }
                Err(error) => {
                    println!("Arc step rejected: {error}; length={length:.6e}");
                    length *= 0.5;
                }
            }
            if length < 1e-4 {
                break;
            }
        }
        let (next, count) = accepted.ok_or("branch step could not converge")?;
        let residual = norm(&raw_evaluate(&next)?.0);
        if !residual.is_finite() || residual > RAW_TOLERANCE {
            return Err("corrected root failed independent reassembly".into());
        }
        iterations += count;
        let multiplier = next.log_viscosity.exp();
        let mut effective = original.clone();
        effective.operating_point.dynamic_viscosity_pa_s *= multiplier;
        let payload = serde_json::json!({"configuration":original,"effective_configuration":effective,
            "design":design,"state":next.state,"viscosity_multiplier":multiplier,
            "actual_stage_residual":residual,"accepted_iterations":iterations,"branch_steps":branch_step+1,
            "diagnostic":"Actual continuation root only; original viscosity/conservation/mesh QA remains unproven"});
        let temporary = path.with_extension("pending.json");
        std::fs::write(&temporary, serde_json::to_vec_pretty(&payload)?)?;
        std::fs::rename(temporary, &path)?;
        println!(
            "Arc root {}: viscosity={multiplier:.12e},raw={residual:.12e}",
            branch_step + 1
        );
        if next.log_viscosity == 0.0 {
            return Ok(());
        }
        previous = current;
        current = next;
        length = if count <= 4 {
            (length * 1.2).min(0.5)
        } else if count > 10 {
            (length * 0.5).max(1e-4)
        } else {
            length
        };
    }
    Err("branch budget ended before exact original viscosity".into())
}

#[cfg(test)]
mod tests {
    use super::*;
    fn fold(root: &Root) -> Result<Evaluation, Error> {
        let u = root.state[0];
        Ok(Evaluation {
            residual: vec![u * u + root.log_viscosity - 1.0],
            jacobian: CsrMatrix::try_from_raw_parts(1, 1, vec![0, 1], vec![0], vec![2.0 * u])?,
            parameter_derivative: vec![1.0],
        })
    }
    #[test]
    fn augmented_corrector_crosses_a_fold_and_reverses_parameter_direction() {
        let mut a = Root {
            state: vec![0.6],
            log_viscosity: 0.64,
        };
        let mut b = Root {
            state: vec![0.4],
            log_viscosity: 0.84,
        };
        let mut maximum = b.log_viscosity;
        for _ in 0..60 {
            let phase = predictor(&a, &b, 0.12).unwrap();
            let (next, _) = correct(&phase, &fold).unwrap();
            assert!(norm(&fold(&next).unwrap().residual) <= RAW_TOLERANCE);
            maximum = maximum.max(next.log_viscosity);
            a = b;
            b = next;
            if b.state[0] < -0.1 {
                break;
            }
        }
        assert!(b.state[0] < 0.0);
        assert!(b.log_viscosity < maximum);
    }
    #[test]
    fn zero_state_jacobian_at_fold_does_not_make_augmented_system_singular() {
        let root = Root {
            state: vec![0.0],
            log_viscosity: 1.0,
        };
        let phase = Phase {
            predictor: root.clone(),
            gradient: vec![1.0],
            parameter_gradient: 0.0,
        };
        let matrix = augmented(&fold(&root).unwrap(), &phase).unwrap();
        let result = SparseLuFactorization::factorize_supernodal(&matrix)
            .unwrap()
            .solve(&[2.0, 3.0])
            .unwrap();
        assert!((result[0] - 3.0).abs() < 1e-12 && (result[1] - 2.0).abs() < 1e-12);
    }
}
