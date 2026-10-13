//! Native continuum warping and twist of the original OpenCourant I-section bar.

// Rust guideline compliant 2026-02-21

use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::forms::standard::VectorMassForm;
use sonium_fem::mesh::{ElementType, Mesh, Point};
use sonium_fem::physics::solid::elasticity::{ElasticityForm, IsotropicMaterial};
use sonium_fem::solver::implicit_dynamics::{
    ImplicitDynamicsConfig, ImplicitDynamicsIntegrator, ImplicitDynamicsSystem,
};
use sonium_fem::spaces::{FunctionSpace, VectorH1Space};
use sonium_qa::rad::OpenCourantCase;
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
    probes: Vec<f64>,
    axial_warping: f64,
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

fn native_mesh(factor: usize) -> Mesh {
    // Refine all three axes while retaining the original I-section and probe coordinates.
    let (bricks, _, _, _, _) = super::i_beam_mesh(
        10 * factor,
        2.0,
        0.1,
        0.11,
        0.01,
        0.01,
        2 * factor,
        2 * factor,
        factor,
        4 * factor,
    );
    let mut mesh = Mesh::new(3);
    for point in &bricks.nodes {
        mesh.add_node(Point::new_3d(point[0], point[1], point[2]));
    }
    // The same body diagonal on every brick gives matching shared-face diagonals.
    for brick in &bricks.bricks {
        for local in [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ] {
            mesh.add_element(
                ElementType::Tetrahedron,
                local.iter().map(|&slot| brick[slot] - 1).collect(),
            );
        }
    }
    mesh.detect_boundaries();
    mesh
}

fn solve(
    case: &OpenCourantCase,
    factor: usize,
    requested_dt: f64,
    cholesky: bool,
) -> Result<NativeRun, Box<dyn std::error::Error>> {
    let mesh = native_mesh(factor);
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
        .filter_map(|(dof, point)| (point[0].abs() < 1e-12).then_some(dof))
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
    let mut probes = Vec::new();
    for probe in &case.probes {
        let (plus, minus) = probe
            .channel
            .split_once('-')
            .ok_or("missing paired twist probe")?;
        let (plus_node, plus_axis) = probe_dof(plus)?;
        let (minus_node, minus_axis) = probe_dof(minus)?;
        probes.push((
            native_dof(case.mesh.nodes[plus_node - 1], plus_axis)?,
            native_dof(case.mesh.nodes[minus_node - 1], minus_axis)?,
        ));
    }
    let axial_plus = native_dof([1.0, 0.05, 0.11], 0)?;
    let axial_minus = native_dof([1.0, -0.05, 0.11], 0)?;
    let steps = (case.stop_time_s / requested_dt).ceil() as usize;
    let dt = case.stop_time_s / steps as f64;
    println!(
        "native P2 transient: {dimension} DOFs, {} tetrahedra, {steps} steps, dt={dt:.6e}",
        mesh.num_elements()
    );
    let constructor = if cholesky {
        ImplicitDynamicsIntegrator::new_spd
    } else {
        ImplicitDynamicsIntegrator::new
    };
    let mut integrator = constructor(
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
    let state = integrator.state();
    let actual = probes
        .iter()
        .map(|&(plus, minus)| state.displacement[plus] - state.displacement[minus])
        .collect::<Vec<_>>();
    let axial_warping = (state.displacement[axial_plus] - state.displacement[axial_minus]).abs();
    for (probe, &value) in case.probes.iter().zip(&actual) {
        println!(
            "final probe {}={value:.12e}, analytical={:.12e}, relative error={:.6e}, max independent dynamic residual={max_true_residual:.6e}",
            probe.channel,
            probe.expected,
            (value - probe.expected).abs() / probe.expected.abs()
        );
    }
    println!("midspan axial flange displacement difference={axial_warping:.12e}");
    if max_true_residual > 1e-8 {
        return Err("native true dynamic residual exceeds 1e-8".into());
    }
    Ok(NativeRun {
        probes: actual,
        axial_warping,
        max_residual: max_true_residual,
    })
}

fn validate_case(case: &OpenCourantCase) -> Result<(), Box<dyn std::error::Error>> {
    let original = super::beam_warping_case();
    // Compare the complete original geometry, material, clamp and load-history cards.
    if sonium_qa::rad::write_starter(case) != sonium_qa::rad::write_starter(&original)
        || sonium_qa::rad::write_engine(case) != sonium_qa::rad::write_engine(&original)
        || case.probes.len() != original.probes.len()
        || case.probes.iter().zip(&original.probes).any(|(a, b)| {
            a.channel != b.channel
                || a.expected != b.expected
                || a.abs_tol != b.abs_tol
                || a.rel_tol != b.rel_tol
        })
    {
        return Err("original I-section warping case identity differs".into());
    }
    Ok(())
}

/// Compare full 3D restrained I-section warping with the original external history.
pub(super) fn parity_result(
    case: &OpenCourantCase,
    case_id: &str,
    engine: &str,
    external: &[f64],
    reference_passed: bool,
) -> Result<QaResult, Box<dyn std::error::Error>> {
    validate_case(case)?;
    if external.len() != 2 {
        return Err("original paired external probes unavailable".into());
    }
    // The actual factor-1/2 study changed both probes by about 10%, above 2%.
    // Cholesky retains the same complete transient and independent residual checks.
    let coarse = solve(case, 2, 0.00025, true)?;
    let medium = solve(case, 3, 0.00025, true)?;
    let fine = solve(case, 4, 0.00025, true)?;
    let time_coarse = solve(case, 4, 0.0005, true)?;
    Ok(build_result(
        case,
        case_id,
        engine,
        external,
        reference_passed,
        [&coarse, &medium, &fine, &time_coarse],
    ))
}

/// Compare both direct factorizations on the original complete I-section transient.
///
/// # Errors
///
/// Returns an error for a zero refinement factor, failed transient solve,
/// or LU/Cholesky probe or axial-warping differences above `1e-6`.
pub fn compare_factorizations(factor: usize) -> Result<(), Box<dyn std::error::Error>> {
    let case = super::beam_warping_case();
    validate_case(&case)?;
    if factor == 0 {
        return Err("positive refinement factor required".into());
    }
    let start = std::time::Instant::now();
    let lu = solve(&case, factor, 0.00025, false)?;
    let lu_seconds = start.elapsed().as_secs_f64();
    let start = std::time::Instant::now();
    let spd = solve(&case, factor, 0.00025, true)?;
    let spd_seconds = start.elapsed().as_secs_f64();
    let error = lu
        .probes
        .iter()
        .zip(&spd.probes)
        .map(|(a, b)| (a - b).abs() / a.abs())
        .fold(0.0_f64, f64::max);
    let axial_error = (lu.axial_warping - spd.axial_warping).abs() / lu.axial_warping;
    println!(
        "factorization comparison: factor={factor}, LU seconds={lu_seconds:.6}, Cholesky seconds={spd_seconds:.6}, paired-displacement relative difference={error:.6e}, axial-warping relative difference={axial_error:.6e}, LU max residual={:.6e}, Cholesky max residual={:.6e}",
        lu.max_residual, spd.max_residual
    );
    if !error.is_finite() || !axial_error.is_finite() || error > 1e-6 || axial_error > 1e-6 {
        return Err("full transient LU/Cholesky parity exceeds 1e-6".into());
    }
    Ok(())
}

/// Measure one complete original native warping history at a chosen resolution.
///
/// This diagnostic retains all original physical parameters and every-step
/// force-balance checks. It does not claim external parity or QA success.
///
/// # Errors
///
/// Returns an error for zero resolution, a non-positive or non-finite timestep,
/// failed assembly/factorization, or a true dynamic residual above `1e-8`.
pub fn probe_transient(
    factor: usize,
    timestep: f64,
) -> Result<serde_json::Value, Box<dyn std::error::Error>> {
    if factor == 0 || !timestep.is_finite() || timestep <= 0.0 {
        return Err("positive native refinement and finite positive timestep required".into());
    }
    let case = super::beam_warping_case();
    validate_case(&case)?;
    let run = solve(&case, factor, timestep, true)?;
    Ok(serde_json::json!({
        "scope":"Original complete 0.5 s restrained I-section warping transient; diagnostic only, not external parity or QA success",
        "refinement_factor":factor,"requested_timestep_s":timestep,
        "midspan_displacement_pair_m":run.probes[0],
        "tip_displacement_pair_m":run.probes[1],
        "axial_warping_displacement_m":run.axial_warping,
        "max_true_dynamic_residual":run.max_residual
    }))
}

fn analytical_axial_warping(case: &OpenCourantCase) -> f64 {
    // Thin-wall sectorial coordinate on the top flange is omega = +/- y*h_w/2.
    // Therefore the opposite flange-node axial difference is b*h_w*theta'(L/2)/2.
    let width: f64 = 0.1;
    let wall: f64 = 0.01;
    let web_height: f64 = 0.2;
    let flange_separation = web_height + wall;
    let torsion = (2.0 * width * wall.powi(3) + web_height * wall.powi(3)) / 3.0;
    let iy = (2.0 * wall * width.powi(3) + web_height * wall.powi(3)) / 12.0;
    let warping = iy * flange_separation.powi(2) / 4.0;
    let young = case.material.youngs_modulus_pa;
    let shear = young / (2.0 * (1.0 + case.material.poisson_ratio));
    let mu = (shear * torsion / (young * warping)).sqrt();
    let torque = case.cloads[0].force_scale_n * width;
    let slope = torque / (shear * torsion) * (1.0 - mu.cosh() + (2.0 * mu).tanh() * mu.sinh());
    (width * flange_separation * slope / 2.0).abs()
}

fn build_result(
    case: &OpenCourantCase,
    case_id: &str,
    engine: &str,
    external: &[f64],
    reference_passed: bool,
    runs: [&NativeRun; 4],
) -> QaResult {
    let [coarse, medium, fine, time_coarse] = runs;
    let mut result = QaResult::passed(case_id)
        .requested_backend("cpu+opencourant").actual_backend("cpu+opencourant")
        .metric("external_reference_executed", 1.0)
        .metric("external_reference_health_passed", if reference_passed { 1.0 } else { 0.0 })
        .minimum("external_reference_health_passed", 1.0)
        .metric("native_parity_executed", 1.0).minimum("native_parity_executed", 1.0)
        .metric("native_axial_warping_displacement_m", fine.axial_warping)
        .metric("analytical_axial_warping_displacement_m", analytical_axial_warping(case))
        .metric("native_axial_warping_relative_error",
            (fine.axial_warping - analytical_axial_warping(case)).abs()
                / analytical_axial_warping(case))
        // The original tip Vlasov budget also covers the thin-wall axial approximation.
        .maximum("native_axial_warping_relative_error", 0.20)
        .reference(engine)
        .dimension("Original 2 m I-section: 0.1 m flange width, 0.22 m height, 0.01 m flange/web thickness; full 0.5 s history; three quadratic tetrahedral meshes with 20/30/40 axial cells and proportional cross-section refinement")
        .formulation("Full 3D elasticity and consistent mass with undamped Newmark and checked nested-dissection supernodal Cholesky; root translations restrain twist and axial warping; original 1000 Nm tip nodal force couple; original midspan and tip DZ pairs")
        .tolerance("Native/external paired displacements <=3%; original Vlasov probe budgets 15% midspan and 20% tip; axial Vlasov warping <=20%; spatial changes <=2% then <=0.5%, contraction <=0.6; temporal change <=0.3%; every-step dynamic residual <=1e-8");
    for (index, probe) in case.probes.iter().enumerate() {
        let first =
            (medium.probes[index] - coarse.probes[index]).abs() / medium.probes[index].abs();
        let last = (fine.probes[index] - medium.probes[index]).abs() / fine.probes[index].abs();
        for (name, value, budget) in [
            (
                "native_external_relative_error",
                (fine.probes[index] - external[index]).abs() / external[index].abs(),
                0.03,
            ),
            (
                "native_analytical_relative_error",
                (fine.probes[index] - probe.expected).abs() / probe.expected.abs(),
                probe.rel_tol,
            ),
            ("spatial_first_relative_change", first, 0.02),
            ("spatial_final_relative_change", last, 0.005),
            ("spatial_contraction", last / first.max(1e-15), 0.6),
            (
                "temporal_relative_change",
                (fine.probes[index] - time_coarse.probes[index]).abs() / fine.probes[index].abs(),
                0.003,
            ),
        ] {
            let name = format!("probe_{index}_{name}");
            result = result.metric(&name, value).maximum(name, budget);
        }
        result = result
            .metric(
                format!("probe_{index}_native_displacement_m"),
                fine.probes[index],
            )
            .metric(
                format!("probe_{index}_external_displacement_m"),
                external[index],
            );
    }
    for (name, run) in [
        ("coarse", coarse),
        ("medium", medium),
        ("fine", fine),
        ("time_coarse", time_coarse),
    ] {
        result = result
            .metric(format!("{name}_true_dynamic_residual"), run.max_residual)
            .maximum(format!("{name}_true_dynamic_residual"), 1e-8);
    }
    if let Err(error) = result.release_eligible() {
        result.status = QaStatus::Failed;
        result.message = Some(format!(
            "native warping parity or refinement failed: {error}"
        ));
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn i_section_mesh_preserves_volume_and_shared_faces() {
        for factor in [1, 2] {
            let mesh = native_mesh(factor);
            let mut volume = 0.0;
            let mut faces = std::collections::HashMap::new();
            for element in &mesh.elements {
                let n = &element.nodes;
                let p = n.iter().map(|&i| mesh.nodes[i]).collect::<Vec<_>>();
                let a = [p[1].x - p[0].x, p[1].y - p[0].y, p[1].z - p[0].z];
                let b = [p[2].x - p[0].x, p[2].y - p[0].y, p[2].z - p[0].z];
                let c = [p[3].x - p[0].x, p[3].y - p[0].y, p[3].z - p[0].z];
                let det = a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0])
                    + a[2] * (b[0] * c[1] - b[1] * c[0]);
                assert!(det > 0.0);
                volume += det / 6.0;
                for local in [[0, 1, 2], [0, 1, 3], [0, 2, 3], [1, 2, 3]] {
                    let mut face = local.map(|j| n[j]);
                    face.sort_unstable();
                    *faces.entry(face).or_insert(0usize) += 1;
                }
            }
            assert!((volume - 2.0 * (2.0 * 0.1 * 0.01 + 0.2 * 0.01)).abs() < 1e-12);
            for (face, count) in faces {
                assert!((1..=2).contains(&count));
                if count == 2 {
                    continue;
                }
                let p = face.map(|i| mesh.nodes[i]);
                let plane = |axis: usize, value: f64| {
                    p.iter().all(|q| {
                        let coords = [q.x, q.y, q.z];
                        (coords[axis] - value).abs() < 1e-12
                    })
                };
                let boundary = plane(0, 0.0)
                    || plane(0, 2.0)
                    || plane(1, -0.05)
                    || plane(1, 0.05)
                    || plane(2, -0.11)
                    || plane(2, 0.11)
                    || ((plane(2, -0.1) || plane(2, 0.1))
                        && p.iter().all(|q| q.y.abs() >= 0.005 - 1e-12))
                    || ((plane(1, -0.005) || plane(1, 0.005))
                        && p.iter().all(|q| q.z.abs() <= 0.1 + 1e-12));
                assert!(boundary, "unpaired interior face {p:?}");
            }
        }
    }

    #[test]
    fn altered_original_section_or_root_clamp_is_rejected() {
        let mut case = super::super::beam_warping_case();
        validate_case(&case).unwrap();
        case.mesh.nodes.last_mut().unwrap()[0] = 2.1;
        assert!(validate_case(&case).is_err());
        case = super::super::beam_warping_case();
        case.node_sets[case.bcs[0].set].nodes.pop();
        assert!(validate_case(&case).is_err());
    }

    #[test]
    fn matching_twist_cannot_hide_bad_second_probe_or_dynamic_equation() {
        let case = super::super::beam_warping_case();
        let good = case.probes.iter().map(|p| p.expected).collect::<Vec<_>>();
        let make = |scale: f64| NativeRun {
            probes: good.iter().map(|p| p * scale).collect(),
            axial_warping: analytical_axial_warping(&case),
            max_residual: 1e-10,
        };
        let mut runs = [make(0.995), make(0.999), make(1.0), make(1.0001)];
        assert!(
            build_result(&case, "test", "reference", &good, true, runs.each_ref())
                .release_eligible()
                .is_ok()
        );
        runs[2].axial_warping = 0.0;
        assert!(
            build_result(&case, "test", "reference", &good, true, runs.each_ref())
                .release_eligible()
                .is_err()
        );
        runs[2].axial_warping = analytical_axial_warping(&case);
        runs[2].max_residual = 1e-5;
        assert!(
            build_result(&case, "test", "reference", &good, true, runs.each_ref())
                .release_eligible()
                .is_err()
        );
        runs[2].max_residual = 1e-10;
        runs[2].probes[1] *= 1.1;
        assert!(
            build_result(&case, "test", "reference", &good, true, runs.each_ref())
                .release_eligible()
                .is_err()
        );
    }
}
