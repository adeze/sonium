//! Full original nonlinear roof histories with native corotational MITC4 quads.

// Rust guideline compliant 2026-02-21

use super::corotational::{Integration, integrate};
use sonium_fem::forms::NonlinearElementForm;
use sonium_fem::mesh::Point;
use sonium_fem::physics::solid::shell::{
    CorotationalQuadShellForm, MembraneFormulation, QuadShellElement, ShellSection,
};

/// Measure the original complete roof using energy-consistent nonlinear quads.
///
/// Retains all physical mass, rotational DOFs, original gravity and diaphragms.
/// Uses a conservative initial Gershgorin stability bound without mass scaling.
/// Outputs provide numerical evidence only; no original QA parity is assumed.
///
/// # Errors
/// Rejects invalid options, geometry/mass/stability bounds and incomplete or
/// nonfinite physical histories.
pub fn probe(cells: usize, scale: f64) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    if !(4..=128).contains(&cells)
        || !cells.is_multiple_of(2)
        || !scale.is_finite()
        || !(0.0 < scale && scale <= 1.0)
    {
        return Err("even quad roof cells in[4,128],timestep scale in(0,1] required".into());
    }
    let case = super::refined_case(cells)?;
    // Roof facets are planar. Compare the existing compatible membrane
    // with fictitious drilling against Allman/Hughes-Brezzi enrichment;
    // this changes numerical formulation, not the physical roof or load.
    let membrane = match std::env::var("P2_SCORDELIS_NATIVE_MEMBRANE").as_deref() {
        Ok("legacy") => MembraneFormulation::Legacy,
        Ok("warp-aware") | Err(_) => MembraneFormulation::WarpAware,
        Ok(_) => return Err("P2_SCORDELIS_NATIVE_MEMBRANE must be legacy or warp-aware".into()),
    };
    let material = case.base.material;
    let section = ShellSection::new(
        material.youngs_modulus_pa,
        material.poisson_ratio,
        case.shell_prop
            .as_ref()
            .ok_or("original roof section missing")?
            .thickness_m,
        material.density_kg_per_m3,
    );
    let points = case
        .base
        .mesh
        .nodes
        .iter()
        .map(|p| Point::new_3d(p[0], p[1], p[2]))
        .collect::<Vec<_>>();
    let elements = case
        .shells
        .quads
        .iter()
        .map(|nodes| {
            let ids = nodes.map(|id| id - 1);
            let mut element = QuadShellElement::from_points_with_section(
                ids,
                &points[ids[0]],
                &points[ids[1]],
                &points[ids[2]],
                &points[ids[3]],
                section,
            );
            element.membrane = membrane;
            element
        })
        .collect::<Vec<_>>();
    let dofs = 6 * points.len();
    let mut masses = vec![0.0; dofs];
    for element in &elements {
        let mass = element.mass_global();
        for (i, &dof) in element.global_dofs().iter().enumerate() {
            masses[dof] += mass[24 * i + i];
        }
    }
    if masses.iter().any(|m| !m.is_finite() || *m <= 0.0) {
        return Err("invalid physical quad mass".into());
    }
    let mut fixed = vec![false; dofs];
    for boundary in &case.base.bcs {
        for &id in &case.base.node_sets[boundary.set].nodes {
            for (component, &lock) in boundary.lock.iter().enumerate() {
                if lock {
                    fixed[6 * (id - 1) + component] = true;
                }
            }
        }
    }
    // For symmetric M^-1/2 K M^-1/2, the maximum absolute row sum bounds
    // lambda_max. Summing elementwise absolute values is also conservative.
    let mut row_sums = vec![0.0; dofs];
    for element in &elements {
        let stiffness = element.stiffness_global();
        let ids = element.global_dofs();
        for (i, &row) in ids.iter().enumerate() {
            if !fixed[row] {
                for (j, &column) in ids.iter().enumerate() {
                    if !fixed[column] {
                        row_sums[row] +=
                            stiffness[24 * i + j].abs() / (masses[row] * masses[column]).sqrt();
                    }
                }
            }
        }
    }
    let bound = row_sums.iter().copied().fold(0.0, f64::max);
    if !bound.is_finite() || bound <= 0.0 {
        return Err("invalid physical quad stability bound".into());
    }
    let critical = 2.0 / bound.sqrt();
    // Ample margin for nonlinear stiffness changes; halve again for temporal
    // validation. No damping, stiffness modification or artificial mass.
    let requested_dt = 0.4 * scale * critical;
    let form = CorotationalQuadShellForm::new(elements, dofs);
    let node = case.base.th_nodes[0] - 1;
    let full_profile = match std::env::var("P2_SCORDELIS_NATIVE_FULL_PROFILE").as_deref() {
        Ok("1") => true,
        Ok("0") | Err(_) => false,
        Ok(_) => return Err("P2_SCORDELIS_NATIVE_FULL_PROFILE must be0 or1".into()),
    };
    // Observation indices never affect force evaluation,physical steps or mass.
    // The integrator already retains the full preceding state for force work.
    let observed = if full_profile {
        (0..dofs).collect::<Vec<_>>()
    } else {
        vec![6 * node, 6 * node + 1, 6 * node + 2]
    };
    let probe_index = if full_profile { 6 * node + 1 } else { 1 };
    let gravity = case.gravity.as_ref().ok_or("original gravity missing")?;
    let curve =
        &case.base.functions[gravity.function.ok_or("original gravity ramp missing")?].points;
    let value = |t: f64| {
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
        "Native nonlinear MITC4 roof{cells}:physical DOFs={dofs},full stop8s,requested dt={requested_dt:.12e}"
    );
    let history = integrate(
        &Integration {
            masses: &masses,
            fixed: &fixed,
            observed: &observed,
            stop: case.base.stop_time_s,
            requested_dt,
            period: case.base.th_period_s,
        },
        |u| {
            let mut force = vec![0.0; dofs];
            for element in 0..form.num_elements() {
                for (&dof, f) in form
                    .local_to_global(element)
                    .iter()
                    .zip(form.element_residual(element, u))
                {
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
        .map(|(_, v)| v[probe_index])
        .collect::<Vec<_>>();
    if hold.is_empty() {
        return Err("missing original full-load roof hold".into());
    }
    let mean = hold.iter().sum::<f64>() / hold.len() as f64;
    let profile = if full_profile {
        let samples = history
            .times_s
            .iter()
            .zip(&history.observations)
            .filter(|(t, _)| **t >= 5.5)
            .map(|(_, v)| v)
            .collect::<Vec<_>>();
        let displacements = (0..points.len())
            .map(|node| {
                std::array::from_fn::<_, 3, _>(|axis| {
                    samples.iter().map(|v| v[6 * node + axis]).sum::<f64>() / samples.len() as f64
                })
            })
            .collect::<Vec<_>>();
        serde_json::json!({"node_ids":(1..=points.len()).collect::<Vec<_>>(),
            "mean_hold_displacements_m":displacements,
            "scope":"Actual full native nodal observations averaged over original5.5s..8s;no reference/full-field parity admission."})
    } else {
        serde_json::Value::Null
    };
    let potential = (0..form.num_elements())
        .map(|index| form.element_strain_energy(index, &history.final_displacement))
        .sum::<f64>();
    let balance = (history.final_kinetic_energy_j + potential - history.external_force_work_j)
        .abs()
        / (history.final_kinetic_energy_j.abs()
            + potential.abs()
            + history.external_force_work_j.abs())
        .max(1e-30);
    if !potential.is_finite() || !balance.is_finite() {
        return Err("nonfinite actual quad elastic energy".into());
    }
    println!(
        "Native complete nonlinear MITC4 roof{cells}:hold DY={mean:.12e},physical energy balance={balance:.6e}"
    );
    Ok(
        serde_json::json!({"status":"complete original nonlinear diagnostic;QA parity unproven",
        "surface_cells":cells,"quads":form.num_elements(),"dofs":dofs,"timestep_scale":scale,
        "force_model":"native MITC4;cached reference matrix;full elastic virtual-work frame derivative",
        "membrane_formulation":format!("{membrane:?}"),
        "initial_elastic_stability_upper_bound":bound,"initial_elastic_critical_dt_lower_bound_s":critical,
        "hold_mean_edge_dy_m":mean,"final_elastic_potential_j":potential,
        "full_profile_enabled":full_profile,"full_profile":profile,
        "history_observed_dofs":observed,
        "final_physical_energy_balance_relative_error":balance,"history":history,
        "scope":"Original physical roof geometry,E/nu/rho/t,XY diaphragms with all rotations and axial translations free,physical translational/rotary mass,full gravity5s ramp/hold to8s;native nonlinear MITC4 with no damping or mass scaling. Diagnostic only;no native/reference parity or published linear accuracy admission."}),
    )
}
