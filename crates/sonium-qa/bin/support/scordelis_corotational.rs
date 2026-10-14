//! Measure the original complete nonlinear roof history with native shell forces.
//!
//! Uses the existing corotational CST/DKT kernel, physical lumped translational
//! and rotary mass, and velocity Verlet without damping. Original gravity,
//! diaphragm supports,5s ramp/3s hold and8s stop are retained. Diagnostic only;
//! numerical parity and published linear benchmark accuracy remain unproven.

// Rust guideline compliant 2026-02-21

use sonium_fem::forms::NonlinearElementForm;
use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::shell::explicit::ShellExplicitModel;
use sonium_fem::physics::solid::shell::{CorotationalShellForm, ShellElement, ShellSection};
use sonium_qa::rad::{scordelis_lo_shell_mesh, shell_scordelis_lo_case};

#[derive(serde::Serialize)]
pub(crate) struct History {
    pub(crate) times_s: Vec<f64>,
    pub(crate) observations: Vec<Vec<f64>>,
    pub(crate) final_displacement: Vec<f64>,
    pub(crate) dt_s: f64,
    pub(crate) steps: usize,
    pub(crate) final_kinetic_energy_j: f64,
    pub(crate) internal_force_work_j: f64,
    pub(crate) external_force_work_j: f64,
    pub(crate) final_energy_balance_relative_error: f64,
    pub(crate) maximum_late_energy_balance_relative_error: f64,
}

pub(crate) struct Integration<'a> {
    pub(crate) masses: &'a [f64],
    pub(crate) fixed: &'a [bool],
    pub(crate) observed: &'a [usize],
    pub(crate) stop: f64,
    pub(crate) requested_dt: f64,
    pub(crate) period: f64,
}

pub(crate) fn integrate(
    config: &Integration<'_>,
    mut internal: impl FnMut(&[f64]) -> Vec<f64>,
    mut external: impl FnMut(f64) -> Vec<f64>,
    initial_u: Vec<f64>,
    initial_v: Vec<f64>,
) -> Result<History, Box<dyn std::error::Error>> {
    let n = config.masses.len();
    if n == 0
        || config.fixed.len() != n
        || initial_u.len() != n
        || initial_v.len() != n
        || config.observed.iter().any(|&dof| dof >= n)
        || config.masses.iter().any(|m| !m.is_finite() || *m <= 0.0)
        || initial_u.iter().chain(&initial_v).any(|v| !v.is_finite())
        || !config.stop.is_finite()
        || config.stop <= 0.0
        || !config.requested_dt.is_finite()
        || config.requested_dt <= 0.0
        || !config.period.is_finite()
        || config.period <= 0.0
    {
        return Err("invalid complete native transient inputs".into());
    }
    let steps = (config.stop / config.requested_dt).ceil();
    if !(1.0..=20_000_000.0).contains(&steps) {
        return Err("native transient timestep implies invalid step count".into());
    }
    let steps = steps as usize;
    let dt = config.stop / steps as f64;
    let mut u = initial_u;
    let mut v = initial_v;
    if config
        .fixed
        .iter()
        .enumerate()
        .any(|(i, &fixed)| fixed && (u[i] != 0.0 || v[i] != 0.0))
    {
        return Err("homogeneous constraints need zero initial displacement and velocity".into());
    }
    let validate_force = |force: &[f64]| -> Result<(), Box<dyn std::error::Error>> {
        if force.len() != n || force.iter().any(|v| !v.is_finite()) {
            return Err("nonfinite or ragged native force".into());
        }
        Ok(())
    };
    let mut fi = internal(&u);
    let mut fe = external(0.0);
    validate_force(&fi)?;
    validate_force(&fe)?;
    let kinetic = |velocity: &[f64]| -> f64 {
        velocity
            .iter()
            .zip(config.masses)
            .map(|(&v, &m)| 0.5 * m * v * v)
            .sum()
    };
    let initial_energy = kinetic(&v);
    let mut wi = 0.0;
    let mut we = 0.0;
    let mut late_drift: f64 = 0.0;
    let mut times = vec![0.0];
    let mut values = vec![config.observed.iter().map(|&i| u[i]).collect::<Vec<_>>()];
    let mut next_output = config.period;
    for step in 1..=steps {
        let time = step as f64 * dt;
        let previous = u.clone();
        for i in 0..n {
            if !config.fixed[i] {
                v[i] += 0.5 * dt * (fe[i] - fi[i]) / config.masses[i];
                u[i] += dt * v[i];
            }
        }
        let next_fi = internal(&u);
        let next_fe = external(time);
        validate_force(&next_fi)?;
        validate_force(&next_fe)?;
        for i in 0..n {
            if !config.fixed[i] {
                v[i] += 0.5 * dt * (next_fe[i] - next_fi[i]) / config.masses[i];
                wi += 0.5 * (fi[i] + next_fi[i]) * (u[i] - previous[i]);
                we += 0.5 * (fe[i] + next_fe[i]) * (u[i] - previous[i]);
            }
        }
        if u.iter().chain(&v).any(|v| !v.is_finite()) {
            return Err("native transient state became nonfinite".into());
        }
        fi = next_fi;
        fe = next_fe;
        let ke = kinetic(&v);
        // Report integrated internal-force work separately from stored elastic
        // energy. This diagnostic does not evaluate the potential at each step.
        // Early ramp energy vanishes; assess relative balance after10% of the
        // original duration, retaining the full initial history separately.
        if time >= 0.1 * config.stop {
            let drift = (ke - initial_energy + wi - we).abs()
                / (ke.abs() + initial_energy.abs() + wi.abs() + we.abs()).max(1e-30);
            late_drift = late_drift.max(drift);
        }
        while next_output <= time + 32.0 * f64::EPSILON * config.stop && next_output < config.stop {
            let fraction = ((next_output - (time - dt)) / dt).clamp(0.0, 1.0);
            values.push(
                config
                    .observed
                    .iter()
                    .map(|&i| previous[i] + fraction * (u[i] - previous[i]))
                    .collect(),
            );
            times.push(next_output);
            next_output = config.period * times.len() as f64;
        }
        if step.is_multiple_of((steps / 20).max(1)) {
            println!(
                "native complete roof: step={step}/{steps},time={time:.6},KE={ke:.6e},late balance={late_drift:.6e}"
            );
        }
    }
    times.push(config.stop);
    values.push(config.observed.iter().map(|&i| u[i]).collect());
    let ke = kinetic(&v);
    Ok(History {
        times_s: times,
        observations: values,
        final_displacement: u,
        dt_s: dt,
        steps,
        final_kinetic_energy_j: ke,
        internal_force_work_j: wi,
        external_force_work_j: we,
        final_energy_balance_relative_error: (ke - initial_energy + wi - we).abs()
            / (ke.abs() + initial_energy.abs() + wi.abs() + we.abs()).max(1e-30),
        maximum_late_energy_balance_relative_error: late_drift,
    })
}

/// Measure the complete original nonlinear roof history with native shell forces.
///
/// Uses physical translational/rotary mass, the original diaphragms and full
/// gravity ramp/hold. No damping, mass scaling or shortened history is used.
/// The returned observations do not themselves establish reference parity.
///
/// # Errors
/// Rejects invalid mesh/timestep options, invalid native mass/forces, nonfinite
/// states and incomplete original history observations.
pub fn probe(cells: usize, scale: f64) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    if !(4..=64).contains(&cells)
        || !cells.is_multiple_of(2)
        || !scale.is_finite()
        || !(0.0 < scale && scale <= 1.0)
    {
        return Err("even cells in[4,64] and timestep scale in(0,1] required".into());
    }
    let original = shell_scordelis_lo_case();
    let mesh = scordelis_lo_shell_mesh(cells, cells);
    let m = original.base.material;
    let section = ShellSection::new(
        m.youngs_modulus_pa,
        m.poisson_ratio,
        original
            .shell_prop
            .as_ref()
            .ok_or("original thickness missing")?
            .thickness_m,
        m.density_kg_per_m3,
    );
    let points = mesh
        .nodes
        .iter()
        .map(|p| Point::new_3d(p[0], p[1], p[2]))
        .collect::<Vec<_>>();
    let mut elements = Vec::new();
    for quad in &mesh.quads {
        for local in [[0, 1, 2], [0, 2, 3]] {
            let ids = local.map(|i| quad[i] - 1);
            elements.push(ShellElement::from_points(
                ids,
                &points[ids[0]],
                &points[ids[1]],
                &points[ids[2]],
                section,
            ));
        }
    }
    let dofs = 6 * points.len();
    let mut elastic = ShellExplicitModel::elastic(elements.clone(), dofs)?;
    let mut fixed = vec![false; dofs];
    for &node in mesh.end_lo.iter().chain(&mesh.end_hi) {
        for component in [0, 1] {
            let dof = 6 * (node - 1) + component;
            fixed[dof] = true;
            elastic.constrain_dof(dof);
        }
    }
    // Conservative safety margin below both physical wave and initial elastic
    // modal bounds. Both default and half-scale runs retain all physical mass.
    let critical = elastic.elastic_critical_dt(200)?;
    let requested_dt = 0.4 * scale * critical.min(elastic.wave_dt());
    let masses = elastic.masses().to_vec();
    let form = CorotationalShellForm::new(elements, dofs);
    let observed = [
        6 * (mesh.probe_node - 1),
        6 * (mesh.probe_node - 1) + 1,
        6 * (mesh.probe_node - 1) + 2,
    ];
    let gravity = original
        .gravity
        .as_ref()
        .ok_or("original gravity missing")?;
    let curve =
        &original.base.functions[gravity.function.ok_or("original gravity ramp missing")?].points;
    let value = |t: f64| -> f64 {
        for pair in curve.windows(2) {
            if t <= pair[1].0 {
                return pair[0].1
                    + (pair[1].1 - pair[0].1)
                        * ((t - pair[0].0) / (pair[1].0 - pair[0].0)).clamp(0.0, 1.0);
            }
        }
        curve.last().expect("validated original gravity curve").1
    };
    println!(
        "Native nonlinear roof{cells}: physical DOFs={dofs},full stop8s,requested dt={requested_dt:.12e},initial elastic critical dt={critical:.12e}"
    );
    let history = integrate(
        &Integration {
            masses: &masses,
            fixed: &fixed,
            observed: &observed,
            stop: original.base.stop_time_s,
            requested_dt,
            period: original.base.th_period_s,
        },
        |u| {
            let mut force = vec![0.0; dofs];
            for element in 0..form.num_elements() {
                let indices = form.local_to_global(element);
                for (&dof, f) in indices.iter().zip(form.element_residual(element, u)) {
                    force[dof] += f;
                }
            }
            force
        },
        |t| {
            let mut force = vec![0.0; dofs];
            for node in 0..points.len() {
                force[6 * node + 1] = masses[6 * node + 1] * gravity.value_m_per_s2 * value(t);
            }
            force
        },
        vec![0.0; dofs],
        vec![0.0; dofs],
    )?;
    let hold = history
        .times_s
        .iter()
        .zip(&history.observations)
        .filter(|(t, _)| **t >= 5.5)
        .map(|(_, v)| v[1])
        .collect::<Vec<_>>();
    if hold.is_empty() {
        return Err("missing original roof hold window".into());
    }
    let mean = hold.iter().sum::<f64>() / hold.len() as f64;
    println!(
        "Native complete nonlinear roof{cells}: hold meanDY={mean:.12e},final balance={:.6e}",
        history.final_energy_balance_relative_error
    );
    Ok(serde_json::json!({
            "status":"complete native nonlinear diagnostic; original QA native/external parity unproven",
            "force_model":"elastic virtual work; full corotated-displacement derivative transpose",
            "surface_cells":cells,"triangles":form.num_elements(),"dofs":dofs,"timestep_scale":scale,
            "initial_elastic_critical_dt_s":critical,"hold_mean_edge_dy_m":mean,"history":history,
            "scope":"Original roof geometry/E/nu/rho/t,physical lumped translational and rotary mass,original XY diaphragms with all rotations and axial translation free; full gravity ramp5s/hold to8s; native corotational CST/DKT velocity Verlet with no damping or mass scaling. Internal-force work is not claimed conservative stored strain energy. Diagnostic only, no original gate success or published linear accuracy claim."
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn oscillator_matches_analytic_solution_and_second_order_refinement() {
        let run = |dt| {
            integrate(
                &Integration {
                    masses: &[1.0],
                    fixed: &[false],
                    observed: &[0],
                    stop: 1.0,
                    requested_dt: dt,
                    period: 0.01,
                },
                |u| vec![4.0 * u[0]],
                |_| vec![0.0],
                vec![1.0],
                vec![0.0],
            )
            .unwrap()
        };
        let coarse = run(0.002);
        let fine = run(0.001);
        let exact = 2.0_f64.cos();
        let a = (coarse.final_displacement[0] - exact).abs();
        let b = (fine.final_displacement[0] - exact).abs();
        assert!(a < 2e-6 && b < 5e-7 && b / a < 0.26);
        assert!(fine.maximum_late_energy_balance_relative_error < 3e-6);
        assert_eq!(fine.times_s[0], 0.0);
        assert_eq!(*fine.times_s.last().unwrap(), 1.0);
        assert!(fine.times_s.windows(2).all(|t| t[1] > t[0]));
    }

    #[test]
    fn physical_free_translation_and_homogeneous_constraint_match_exact_motion() {
        let run = integrate(
            &Integration {
                masses: &[2.0, 3.0],
                fixed: &[false, true],
                observed: &[0, 1],
                stop: 1.0,
                requested_dt: 0.003,
                period: 0.05,
            },
            |_| vec![0.0; 2],
            |_| vec![-19.62, -29.43],
            vec![0.0; 2],
            vec![0.0; 2],
        )
        .unwrap();
        assert!((run.final_displacement[0] + 4.905).abs() < 1e-12);
        assert_eq!(run.final_displacement[1], 0.0);
        assert!(run.final_energy_balance_relative_error < 1e-12);
    }
}
