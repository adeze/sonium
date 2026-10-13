//! Native transient continuum solves of the original OpenCourant bending and torsion bars.

// Rust guideline compliant 2026-02-21

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::standard::VectorMassForm;
use sonium_fem::mesh::{Mesh, box_mesh_tetrahedra};
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial};
use sonium_fem::solver::implicit_dynamics::{
    ImplicitDynamicsConfig, ImplicitDynamicsIntegrator, ImplicitDynamicsSystem,
};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_qa::rad::OpenCourantCase;
#[cfg(test)]
use sonium_qa::rad::cantilever_case;
use sonium_qa::{QaResult, QaStatus};
use sonium_solvers::{CsrMatrix, LinearOperator};

fn constrain_zero(matrix: &mut CsrMatrix<f64>, constrained: &[bool]) {
    for row in 0..matrix.num_rows {
        for entry in matrix.row_ptrs[row]..matrix.row_ptrs[row + 1] {
            let column = matrix.col_indices[entry];
            if constrained[row] {
                matrix.values[entry] = if row == column { 1.0 } else { 0.0 };
            } else if constrained[column] {
                matrix.values[entry] = 0.0;
            }
        }
    }
}

fn curve(points: &[(f64, f64)], time: f64) -> f64 {
    for pair in points.windows(2) {
        if time <= pair[1].0 {
            let fraction = ((time - pair[0].0) / (pair[1].0 - pair[0].0)).clamp(0.0, 1.0);
            return pair[0].1 + fraction * (pair[1].1 - pair[0].1);
        }
    }
    points.last().unwrap().1
}

struct NativeRun {
    tip: f64,
    max_residual: f64,
}

fn probe_dof(channel: &str) -> Result<(usize, usize), Box<dyn std::error::Error>> {
    let (component, node) = channel.split_once('/').ok_or("missing probe node")?;
    let axis = match component {
        "DX" => 0,
        "DY" => 1,
        "DZ" => 2,
        _ => return Err("native cantilever requires a translational displacement probe".into()),
    };
    Ok((node.parse()?, axis))
}

fn geometry(case: &OpenCourantCase) -> Result<(f64, usize), Box<dyn std::error::Error>> {
    match case.name.as_str() {
        "cantilever" | "beam_3d_cantilever" | "bending_3pt" => Ok((1.0, 10)),
        // Twelve axial cells include the original midspan probe and both tip loads.
        "beam_torsion" => Ok((2.0, 12)),
        "beam_curved_arch" => Ok((std::f64::consts::FRAC_PI_2, 10)),
        _ => Err("unsupported native transient bar case".into()),
    }
}

fn mesh_ladder(case: &OpenCourantCase) -> [usize; 3] {
    if case.name == "beam_torsion" {
        // The actual 12x2x2 -> 24x4x4 study changed the probe by 3.48%, above 2%.
        [2, 3, 4]
    } else {
        [1, 2, 3]
    }
}

fn constrained_component(case: &OpenCourantCase, point: [f64; 3], axis: usize) -> bool {
    if case.name == "beam_curved_arch" {
        return point[1].abs() < 1e-12;
    }
    if case.name != "bending_3pt" {
        return point[0].abs() < 1e-12;
    }
    let left = point[0].abs() < 1e-12;
    let right = (point[0] - 1.0).abs() < 1e-12;
    let neutral = point[2].abs() < 1e-12;
    match axis {
        0 => left && neutral,
        1 => left && neutral && point[1].abs() < 1e-12,
        2 => (left || right) && neutral,
        _ => false,
    }
}

fn native_mesh(case: &OpenCourantCase, factor: usize) -> Result<Mesh, Box<dyn std::error::Error>> {
    let (length, axial_cells) = geometry(case)?;
    let mut mesh = box_mesh_tetrahedra(
        0.0,
        length,
        -0.025,
        0.025,
        -0.025,
        0.025,
        axial_cells * factor,
        2 * factor,
        2 * factor,
    );
    if case.name == "beam_curved_arch" {
        // Sweep the same structured topology around the original radius-one quarter circle.
        // Piecewise planar geometry converges to the curved solid under arc refinement.
        for point in &mut mesh.nodes {
            let radius = 1.0 + point.y;
            let (sin, cos) = point.x.sin_cos();
            point.x = radius * cos;
            point.y = radius * sin;
        }
        // (theta, radial, z) has negative orientation in global (x, y, z).
        for element in &mut mesh.elements {
            element.nodes.swap(1, 2);
        }
        mesh.invalidate_caches();
        mesh.detect_boundaries();
    }
    Ok(mesh)
}

fn solve(
    case: &OpenCourantCase,
    factor: usize,
    requested_dt: f64,
) -> Result<NativeRun, Box<dyn std::error::Error>> {
    let mesh = native_mesh(case, factor)?;
    let space = VectorH1Space::new(PolynomialDegree::P2, 3);
    let dimension = space.num_global_dofs(&mesh);
    let coordinates = space
        .dof_coordinates(&mesh)
        .ok_or("missing nodal coordinates")?;
    let material =
        IsotropicMaterial::new(case.material.youngs_modulus_pa, case.material.poisson_ratio);
    let mut stiffness =
        assemble_bilinear(&ElasticityForm::new_3d(material), &space, &space, &mesh, 4).to_csr();
    let mut mass = assemble_bilinear(
        &VectorMassForm {
            coefficient: case.material.density_kg_per_m3,
        },
        &space,
        &space,
        &mesh,
        4,
    )
    .to_csr();
    let constrained = coordinates
        .iter()
        .enumerate()
        .filter_map(|(dof, point)| constrained_component(case, *point, dof % 3).then_some(dof))
        .collect::<Vec<_>>();
    let zero = vec![0.0; dimension];
    let mut is_constrained = vec![false; dimension];
    for dof in constrained {
        is_constrained[dof] = true;
    }
    constrain_zero(&mut stiffness, &is_constrained);
    constrain_zero(&mut mass, &is_constrained);
    let native_dof = |point: [f64; 3], axis: usize| {
        coordinates
            .iter()
            .enumerate()
            .find_map(|(dof, position)| {
                (dof % 3 == axis
                    && position
                        .iter()
                        .zip(point)
                        .all(|(a, b)| (*a - b).abs() < 1e-12))
                .then_some(dof)
            })
            .ok_or("original deck node absent from native nodal space")
    };
    let mut loads = Vec::new();
    for load in &case.cloads {
        for &node in &case.node_sets[load.set].nodes {
            loads.push((
                native_dof(case.mesh.nodes[node - 1], load.direction - 1)?,
                load.force_scale_n,
                load.function,
            ));
        }
    }
    let (probe_node, probe_axis) = probe_dof(&case.probes[0].channel)?;
    let probe = native_dof(case.mesh.nodes[probe_node - 1], probe_axis)?;
    let steps = (case.stop_time_s / requested_dt).ceil() as usize;
    let dt = case.stop_time_s / steps as f64;
    println!(
        "native P2 transient: {dimension} DOFs, {} tetrahedra, {steps} steps, dt={dt:.6e}",
        mesh.num_elements()
    );
    let mut integrator = ImplicitDynamicsIntegrator::new(
        ImplicitDynamicsSystem {
            mass: mass.clone(),
            stiffness: stiffness.clone(),
            damping: None,
        },
        ImplicitDynamicsConfig::default(),
        dt,
    )?;
    integrator.initialize(&zero, &zero, &zero)?;
    let mut max_true_residual = 0.0_f64;
    for step in 1..=steps {
        let time = step as f64 * dt;
        let mut force = zero.clone();
        for &(dof, scale, function) in &loads {
            force[dof] += scale * curve(&case.functions[function].points, time);
        }
        let force_norm = force.iter().map(|value| value * value).sum::<f64>().sqrt();
        let state = integrator.step_checked(&force, 0.5e-8 * force_norm.max(1e-12))?;
        let inertia = mass.apply(&state.acceleration);
        let elastic = stiffness.apply(&state.displacement);
        let norm = force.iter().map(|value| value * value).sum::<f64>().sqrt();
        let residual = inertia
            .iter()
            .zip(&elastic)
            .zip(&force)
            .map(|((m, k), f)| (m + k - f).powi(2))
            .sum::<f64>()
            .sqrt()
            / norm.max(1e-12);
        if !residual.is_finite() {
            return Err("native equation residual is non-finite".into());
        }
        max_true_residual = max_true_residual.max(residual);
    }
    let actual = integrator.state().displacement[probe];
    let expected = case.probes[0].expected;
    println!(
        "final probe {}={actual:.12e}, analytical={expected:.12e}, relative error={:.6e}, max independent dynamic residual={max_true_residual:.6e}",
        case.probes[0].channel,
        (actual - expected).abs() / expected.abs()
    );
    if max_true_residual > 1e-8 {
        return Err("native true dynamic residual exceeds 1e-8".into());
    }
    Ok(NativeRun {
        tip: actual,
        max_residual: max_true_residual,
    })
}

pub(super) fn validate_case(case: &OpenCourantCase) -> Result<(), Box<dyn std::error::Error>> {
    if case.name == "beam_curved_arch" {
        return validate_curved_arch(case);
    }
    let (length, _) = geometry(case)?;
    // Refuse changed geometry or clamp rather than silently solving a different bar.
    if case.probes.len() != 1
        || !case.walls.is_empty()
        || case.stop_time_s != 1.0
        || (case.name != "bending_3pt"
            && (case.bcs.len() != 1
                || case.bcs[0].lock[..3] != [true, true, true]
                || case.node_sets[case.bcs[0].set]
                    .nodes
                    .iter()
                    .any(|&node| case.mesh.nodes[node - 1][0].abs() > 1e-12)))
    {
        return Err("native cantilever case identity or root clamp differs".into());
    }
    let bounds = [(0.0, length), (-0.025, 0.025), (-0.025, 0.025)];
    for (axis, &(low, high)) in bounds.iter().enumerate() {
        let min = case
            .mesh
            .nodes
            .iter()
            .map(|point| point[axis])
            .fold(f64::INFINITY, f64::min);
        let max = case
            .mesh
            .nodes
            .iter()
            .map(|point| point[axis])
            .fold(f64::NEG_INFINITY, f64::max);
        if (min - low).abs() > 1e-12
            || (max - high).abs() > 1e-12
            || case.mesh.nodes.iter().any(|point| !point[axis].is_finite())
        {
            return Err("native cantilever dimensions differ from original deck".into());
        }
    }
    if case.name == "bending_3pt" {
        return validate_three_point_supports(case);
    }
    let mut deck_clamp = case.node_sets[case.bcs[0].set].nodes.clone();
    deck_clamp.sort_unstable();
    let root_nodes = case
        .mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, point)| (point[0].abs() < 1e-12).then_some(index + 1))
        .collect::<Vec<_>>();
    if deck_clamp != root_nodes {
        return Err("native full root clamp differs from original deck".into());
    }
    Ok(())
}

fn validate_curved_arch(case: &OpenCourantCase) -> Result<(), Box<dyn std::error::Error>> {
    if case.probes.len() != 1
        || !case.walls.is_empty()
        || case.stop_time_s != 1.0
        || case.bcs.len() != 1
        || case.bcs[0].lock[..3] != [true, true, true]
    {
        return Err("original curved-arch case identity differs".into());
    }
    let mut min_radius = f64::INFINITY;
    let mut max_radius = f64::NEG_INFINITY;
    for point in &case.mesh.nodes {
        let radius = point[0].hypot(point[1]);
        if point.iter().any(|x| !x.is_finite())
            || point[0] < -1e-12
            || point[1] < -1e-12
            || point[2].abs() > 0.025 + 1e-12
            || !(0.975 - 1e-12..=1.025 + 1e-12).contains(&radius)
        {
            return Err("original quarter-circle solid geometry differs".into());
        }
        min_radius = min_radius.min(radius);
        max_radius = max_radius.max(radius);
    }
    if (min_radius - 0.975).abs() > 1e-12 || (max_radius - 1.025).abs() > 1e-12 {
        return Err("original arch section width differs".into());
    }
    let expected = case
        .mesh
        .nodes
        .iter()
        .enumerate()
        .filter_map(|(index, p)| (p[1].abs() < 1e-12).then_some(index + 1))
        .collect::<Vec<_>>();
    let mut actual = case.node_sets[case.bcs[0].set].nodes.clone();
    actual.sort_unstable();
    if actual != expected {
        return Err("original arch root clamp differs".into());
    }
    let (node, axis) = probe_dof(&case.probes[0].channel)?;
    let p = case.mesh.nodes[node - 1];
    if axis != 0 || p[0].abs() > 1e-12 || (p[1] - 1.0).abs() > 1e-12 || p[2].abs() > 1e-12 {
        return Err("original tangential tip displacement probe differs".into());
    }
    Ok(())
}

fn validate_three_point_supports(case: &OpenCourantCase) -> Result<(), Box<dyn std::error::Error>> {
    if case.bcs.len() != 3
        || case.bcs[0].lock != [true, false, true, false, false, false]
        || case.bcs[1].lock != [false, false, true, false, false, false]
        || case.bcs[2].lock != [false, true, false, false, false, false]
    {
        return Err("original pin/roller component constraints differ".into());
    }
    for (bc, x) in [(0, 0.0), (1, 1.0)] {
        let expected = case
            .mesh
            .nodes
            .iter()
            .enumerate()
            .filter_map(|(index, point)| {
                ((point[0] - x).abs() < 1e-12 && point[2].abs() < 1e-12).then_some(index + 1)
            })
            .collect::<Vec<_>>();
        let mut actual = case.node_sets[case.bcs[bc].set].nodes.clone();
        actual.sort_unstable();
        if actual != expected {
            return Err("original neutral-axis support line differs".into());
        }
    }
    let y_pin = &case.node_sets[case.bcs[2].set].nodes;
    if y_pin.len() != 1
        || case.mesh.nodes[y_pin[0] - 1]
            .iter()
            .any(|x| x.abs() > 1e-12)
    {
        return Err("original transverse rigid-motion pin differs".into());
    }
    let (node, axis) = probe_dof(&case.probes[0].channel)?;
    if axis != 2 || case.mesh.nodes[node - 1] != [0.5, 0.0, 0.0] {
        return Err("original center displacement probe differs".into());
    }
    Ok(())
}

/// Compare the original transient continuum problem with the external history.
pub(super) fn parity_result(
    case: &OpenCourantCase,
    case_id: &str,
    engine: &str,
    external_tip: f64,
    reference_passed: bool,
) -> Result<QaResult, Box<dyn std::error::Error>> {
    validate_case(case)?;
    // Resolve spatial and temporal errors independently. Each solve retains
    // the original one-second history, nodal loads, material and clamp.
    let [a, b, c] = mesh_ladder(case);
    let coarse = solve(case, a, 0.00025)?;
    let medium = solve(case, b, 0.00025)?;
    let fine = solve(case, c, 0.00025)?;
    let time_coarse = solve(case, c, 0.0005)?;
    let (length, axial_cells) = geometry(case)?;
    let dimension = if case.name == "beam_curved_arch" {
        format!(
            "Original radius-one quarter-circle 3D arch, 0.05x0.05 m section and 1 s load history; {}x{}x{}, {}x{}x{}, {}x{}x{} arc/radial/axial quadratic tetrahedral meshes",
            axial_cells * a,
            2 * a,
            2 * a,
            axial_cells * b,
            2 * b,
            2 * b,
            axial_cells * c,
            2 * c,
            2 * c
        )
    } else {
        format!(
            "Original {length}x0.05x0.05 m 3D bar; 1 s complete load history; {}x{}x{}, {}x{}x{} and {}x{}x{} quadratic tetrahedral meshes; original displacement probe",
            axial_cells * a,
            2 * a,
            2 * a,
            axial_cells * b,
            2 * b,
            2 * b,
            axial_cells * c,
            2 * c,
            2 * c
        )
    };
    Ok(build_result(
        case,
        case_id,
        engine,
        external_tip,
        reference_passed,
        [&coarse, &medium, &fine, &time_coarse],
    )
    .dimension(dimension))
}

fn build_result(
    case: &OpenCourantCase,
    case_id: &str,
    engine: &str,
    external_tip: f64,
    reference_passed: bool,
    runs: [&NativeRun; 4],
) -> QaResult {
    let expected = case.probes[0].expected;
    // Euler-Bernoulli is the original approximate 3-point solid-beam reference.
    let analytical_budget = if matches!(case.name.as_str(), "bending_3pt" | "beam_curved_arch") {
        case.probes[0].rel_tol
    } else {
        0.01
    };
    let [coarse, medium, fine, time_coarse] = runs;
    let first_change = (medium.tip - coarse.tip).abs() / medium.tip.abs();
    let final_change = (fine.tip - medium.tip).abs() / fine.tip.abs();
    let contraction = final_change / first_change.max(1e-15);
    let time_change = (fine.tip - time_coarse.tip).abs() / fine.tip.abs();
    let parity = (fine.tip - external_tip).abs() / external_tip.abs();
    let analytical = (fine.tip - expected).abs() / expected.abs();
    let mut result = QaResult::passed(case_id)
        .requested_backend("cpu+opencourant")
        .actual_backend("cpu+opencourant")
        .metric("external_reference_executed", 1.0)
        .metric("external_reference_health_passed", if reference_passed { 1.0 } else { 0.0 })
        .minimum("external_reference_health_passed", 1.0)
        .metric("native_parity_executed", 1.0)
        .minimum("native_parity_executed", 1.0)
        .metric("native_external_relative_error", parity)
        .maximum("native_external_relative_error", 0.03)
        .metric("native_analytical_relative_error", analytical)
        .maximum("native_analytical_relative_error", analytical_budget)
        .metric("spatial_first_relative_change", first_change)
        .maximum("spatial_first_relative_change", 0.02)
        .metric("spatial_final_relative_change", final_change)
        .maximum("spatial_final_relative_change", 0.005)
        .metric("spatial_contraction", contraction)
        .maximum("spatial_contraction", 0.6)
        .metric("temporal_relative_change", time_change)
        .maximum("temporal_relative_change", 0.003)
        .metric("native_tip_displacement_m", fine.tip)
        .metric("external_tip_displacement_m", external_tip)
        .reference(engine)
        .dimension("Original 1x0.05x0.05 m 3D cantilever; 1 s complete load history; 10x2x2, 20x4x4 and 30x6x6 quadratic tetrahedral meshes")
        .formulation("Native 3D linear elasticity with consistent mass and undamped Newmark; original nodal CLOADs and component constraints; independently recomputed dynamic residual at every step")
        .tolerance(format!("Native/external displacement <=3%, native/analytical <={analytical_budget}; spatial changes <=2% then <=0.5% with contraction <=0.6; temporal change <=0.3%; every-step true residual <=1e-8"));
    for (name, run) in [
        ("coarse", &coarse),
        ("medium", &medium),
        ("fine", &fine),
        ("time_coarse", &time_coarse),
    ] {
        result = result
            .metric(format!("{name}_true_dynamic_residual"), run.max_residual)
            .maximum(format!("{name}_true_dynamic_residual"), 1e-8);
    }
    if let Err(error) = result.release_eligible() {
        result.status = QaStatus::Failed;
        result.message = Some(format!(
            "native transient parity or refinement failed: {error}"
        ));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn runs(residual: f64) -> [NativeRun; 4] {
        [
            NativeRun {
                tip: 0.000314,
                max_residual: 1e-10,
            },
            NativeRun {
                tip: 0.000318,
                max_residual: 1e-10,
            },
            NativeRun {
                tip: 0.000319,
                max_residual: residual,
            },
            NativeRun {
                tip: 0.0003191,
                max_residual: 1e-10,
            },
        ]
    }

    #[test]
    fn matching_tip_cannot_hide_failed_dynamic_equation() {
        let cases = runs(1e-5);
        let result = build_result(
            &cantilever_case(),
            "test",
            "reference",
            0.000319,
            true,
            cases.each_ref(),
        );
        assert_eq!(result.status, QaStatus::Failed);
        assert!(result.release_eligible().is_err());
        assert_eq!(result.metrics["fine_true_dynamic_residual"], 1e-5);
    }

    #[test]
    fn matching_tip_cannot_hide_noncontracting_mesh_study() {
        let mut cases = runs(1e-10);
        cases[0].tip = 0.0003179;
        let result = build_result(
            &cantilever_case(),
            "test",
            "reference",
            0.000319,
            true,
            cases.each_ref(),
        );
        assert_eq!(result.status, QaStatus::Failed);
        assert!(result.metrics["spatial_contraction"] > 0.6);
        assert_eq!(result.metrics["native_external_relative_error"], 0.0);
    }
}

#[cfg(test)]
#[test]
fn original_probe_component_is_preserved() {
    assert_eq!(probe_dof("DY/95").unwrap(), (95, 1));
    assert_eq!(probe_dof("DZ/95").unwrap(), (95, 2));
    assert!(probe_dof("RX/95").is_err());
}

#[cfg(test)]
#[test]
fn original_dimensions_and_clamp_cannot_be_substituted() {
    let mut case = cantilever_case();
    validate_case(&case).unwrap();
    case.mesh.nodes.last_mut().unwrap()[0] = 1.1;
    assert!(validate_case(&case).is_err());
    case = cantilever_case();
    case.node_sets[case.bcs[0].set].nodes.pop();
    assert!(validate_case(&case).is_err());
}

#[cfg(test)]
#[test]
fn three_point_supports_allow_end_rotation_and_keep_original_bar_clamp() {
    let mut case = cantilever_case();
    assert!(constrained_component(&case, [0.0, 0.02, 0.02], 0));
    case.name = "bending_3pt".into();
    assert!(!constrained_component(&case, [0.0, 0.02, 0.02], 0));
    assert!(constrained_component(&case, [0.0, 0.02, 0.0], 0));
    assert!(!constrained_component(&case, [1.0, 0.02, 0.0], 0));
    assert!(constrained_component(&case, [1.0, 0.02, 0.0], 2));
    assert!(!constrained_component(&case, [0.0, 0.02, 0.0], 1));
    assert!(constrained_component(&case, [0.0, 0.0, 0.0], 1));
}

#[cfg(test)]
#[test]
fn torsion_refines_the_failed_coarse_ladder_without_changing_bending_meshes() {
    let mut case = cantilever_case();
    assert_eq!(mesh_ladder(&case), [1, 2, 3]);
    case.name = "beam_torsion".into();
    assert_eq!(mesh_ladder(&case), [2, 3, 4]);
}

#[cfg(test)]
#[test]
fn native_arch_has_positive_elements_and_converging_quarter_ring_volume() {
    let mut case = cantilever_case();
    case.name = "beam_curved_arch".into();
    let exact = std::f64::consts::FRAC_PI_2 * 0.05 * 0.05;
    let mut previous = f64::INFINITY;
    for factor in [1, 2, 3] {
        let mesh = native_mesh(&case, factor).unwrap();
        let mut volume = 0.0;
        for element in &mesh.elements {
            let p = element
                .nodes
                .iter()
                .map(|&i| mesh.nodes[i])
                .collect::<Vec<_>>();
            let a = [p[1].x - p[0].x, p[1].y - p[0].y, p[1].z - p[0].z];
            let b = [p[2].x - p[0].x, p[2].y - p[0].y, p[2].z - p[0].z];
            let c = [p[3].x - p[0].x, p[3].y - p[0].y, p[3].z - p[0].z];
            let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                + a[2] * (b[0] * c[1] - b[1] * c[0]);
            assert!(det > 0.0);
            volume += det / 6.0;
        }
        let error = (volume - exact).abs() / exact;
        assert!(error < 0.005 && error < previous);
        previous = error;
    }
    assert!(previous < 0.0005);
    assert!(constrained_component(&case, [1.0, 0.0, 0.02], 2));
    assert!(!constrained_component(&case, [0.0, 1.0, 0.02], 2));
}
