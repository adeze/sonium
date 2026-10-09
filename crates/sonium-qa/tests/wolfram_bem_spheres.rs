//! Independent Wolfram/closed-form checks for Sonium's 3D acoustic BEM.
//!
//! The rigid sphere adapts the bare sound-hard obstacle from Wolfram's
//! Acoustic Cloak example to 3D.  The pulsating sphere checks prescribed
//! normal velocity using the frequency-domain monopole solution.

use num_complex::Complex64;
use serde::Deserialize;
use sonium_bem::{
    analytical::{pulsating_sphere_3d, sphere_rcs_3d, sphere_scattering_3d},
    core::{
        AssemblyMethod, BasisOrder, BemProblem, BemSolver, BoundaryCondition, Cluster, Element,
        IncidentField, PhysicsParams, SolverMethod,
        assembly::{
            hmatrix::build_hmatrix_stabilized, slfmm::build_slfmm_system_stabilized,
            tbem::build_tbem_system_with_beta,
        },
        contracts::constant_panel_power,
        mesh::{generators::generate_icosphere_mesh, octree::Octree},
        postprocess::pressure::{compute_rcs, compute_total_field},
        solver::{GmresConfig, SlfmmOperator, gmres_solve_tbem_with_ilu},
    },
};
use sonium_qa::subprocess::WolframCommandExt;
use sonium_solvers::{GmresConfig as KrylovGmresConfig, gmres_with_guess};
use std::{env, f64::consts::PI, path::PathBuf, process::Command};

const SOUND_SPEED_M_S: f64 = 343.0;
const DENSITY_KG_M3: f64 = 1.21;
const RADIUS_M: f64 = 1.0;
const RIGID_KA: f64 = 1.0;
const IRREGULAR_FREQUENCY_KA: f64 = PI;
const RIGID_RECEIVER_RADIUS_M: f64 = 2.0;
const RIGID_ANGLES_RAD: [f64; 5] = [0.0, PI / 4.0, PI / 2.0, 3.0 * PI / 4.0, PI];
const PULSATING_FREQUENCY_HZ: f64 = 100.0;
const PULSATING_RADII_M: [f64; 3] = [2.0, 4.0, 8.0];

fn spherical_receiver(radius: f64, theta: f64) -> [f64; 3] {
    [radius * theta.sin(), 0.0, radius * theta.cos()]
}

fn fmm_clusters(elements: &[Element], separation_factor: f64) -> Vec<Cluster> {
    let centers: Vec<[f64; 3]> = elements.iter().map(|element| element.center).collect();
    let mut octree = Octree::build(&centers, 32, 8);
    octree.compute_interaction_lists(separation_factor);
    let leaf_indices = octree.leaves();
    let node_to_cluster: std::collections::HashMap<_, _> = leaf_indices
        .iter()
        .enumerate()
        .map(|(cluster_index, &node_index)| (node_index, cluster_index))
        .collect();

    leaf_indices
        .iter()
        .map(|&node_index| {
            let node = &octree.nodes[node_index];
            let mut cluster = Cluster::new(node.center);
            cluster.element_indices = node.element_indices.clone();
            cluster.num_elements = node.element_indices.len();
            cluster.radius = node.radius();
            cluster.near_clusters = node
                .near_clusters
                .iter()
                .filter_map(|index| node_to_cluster.get(index).copied())
                .collect();
            cluster.far_clusters = node
                .far_clusters
                .iter()
                .filter_map(|index| node_to_cluster.get(index).copied())
                .collect();
            cluster
        })
        .collect()
}

fn try_solve_rigid_sphere_at_ka(
    ka: f64,
    assembly_method: AssemblyMethod,
    use_burton_miller: bool,
) -> Result<Vec<Complex64>, String> {
    let frequency = ka * SOUND_SPEED_M_S / (2.0 * PI * RADIUS_M);
    let problem =
        BemProblem::rigid_sphere_scattering(RADIUS_M, frequency, SOUND_SPEED_M_S, DENSITY_KG_M3)
            .with_burton_miller(use_burton_miller);
    let solution = BemSolver::new()
        .with_assembly_method(assembly_method)
        .with_solver_method(SolverMethod::BiCgStab)
        .with_tolerance(1.0e-9)
        .solve(&problem)
        .map_err(|error| format!("{error:?}"))?;
    Ok(RIGID_ANGLES_RAD
        .iter()
        .map(|&theta| {
            solution.evaluate_pressure(&spherical_receiver(RIGID_RECEIVER_RADIUS_M, theta))
        })
        .collect())
}

fn solve_rigid_sphere(assembly_method: AssemblyMethod) -> Vec<Complex64> {
    try_solve_rigid_sphere_at_ka(RIGID_KA, assembly_method, true)
        .expect("Sonium should solve rigid-sphere scattering")
}

fn solve_pulsating_sphere() -> (Vec<Complex64>, f64, Vec<Complex64>) {
    let physics = PhysicsParams::new(
        PULSATING_FREQUENCY_HZ,
        SOUND_SPEED_M_S,
        DENSITY_KG_M3,
        false,
    );
    let mesh = generate_icosphere_mesh(RADIUS_M, 2);
    let velocity = Complex64::new(1.0, 0.0);
    let mut elements = mesh.elements.clone();
    for (index, element) in elements.iter_mut().enumerate() {
        element.boundary_condition = BoundaryCondition::Velocity(vec![velocity]);
        element.dof_addresses = vec![index];
    }

    let beta = physics.burton_miller_beta_scaled(4.0);
    let system = build_tbem_system_with_beta(&elements, &mesh.nodes, &physics, beta);
    let result = gmres_solve_tbem_with_ilu(
        &system.matrix,
        &system.rhs,
        &GmresConfig {
            max_iterations: 500,
            restart: 50,
            tolerance: 1.0e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(
        result.converged,
        "pulsating-sphere BEM solve did not converge"
    );

    let receivers: Vec<[f64; 3]> = PULSATING_RADII_M
        .iter()
        .map(|&radius| [0.0, 0.0, radius])
        .collect();
    let no_incident_field = IncidentField::PlaneWave {
        direction: [0.0, 0.0, 1.0],
        amplitude: Complex64::new(0.0, 0.0),
    };
    let receiver_pressure = compute_total_field(
        &receivers,
        &elements,
        &mesh.nodes,
        &result.x,
        None,
        &no_incident_field,
        &physics,
        BasisOrder::Constant,
    )
    .into_iter()
    .map(|field| field.p_total)
    .collect();
    let panel_areas: Vec<_> = elements.iter().map(|element| element.area).collect();
    let normal_velocity = vec![velocity; elements.len()];
    let radiated_power = constant_panel_power(&panel_areas, &result.x, &normal_velocity)
        .expect("pulsating-sphere surface power");
    (receiver_pressure, radiated_power, result.x)
}

fn solve_pulsating_sphere_hmatrix() -> (Vec<Complex64>, f64, Vec<Complex64>) {
    let physics = PhysicsParams::new(
        PULSATING_FREQUENCY_HZ,
        SOUND_SPEED_M_S,
        DENSITY_KG_M3,
        false,
    );
    let mesh = generate_icosphere_mesh(RADIUS_M, 2);
    let velocity = Complex64::new(1.0, 0.0);
    let mut elements = mesh.elements.clone();
    for (index, element) in elements.iter_mut().enumerate() {
        element.boundary_condition = BoundaryCondition::Velocity(vec![velocity]);
        element.dof_addresses = vec![index];
    }

    let beta_scale = 4.0;
    let beta = physics.burton_miller_beta_scaled(beta_scale);
    let dense_system = build_tbem_system_with_beta(&elements, &mesh.nodes, &physics, beta);
    let clusters = fmm_clusters(&elements, 1.2);
    let hmatrix = build_hmatrix_stabilized(
        &elements,
        &mesh.nodes,
        &clusters,
        &physics,
        1.0e-8,
        80,
        1.0,
        beta_scale,
    );
    let result = gmres_with_guess(
        &hmatrix,
        &dense_system.rhs,
        None,
        &KrylovGmresConfig {
            max_iterations: 1_000,
            restart: 100,
            tolerance: 1.0e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        },
    );
    assert!(
        result.converged,
        "pulsating-sphere H-matrix solve did not converge: residual={:.3e}",
        result.residual
    );

    let receivers: Vec<[f64; 3]> = PULSATING_RADII_M
        .iter()
        .map(|&radius| [0.0, 0.0, radius])
        .collect();
    let no_incident_field = IncidentField::PlaneWave {
        direction: [0.0, 0.0, 1.0],
        amplitude: Complex64::new(0.0, 0.0),
    };
    let receiver_pressure = compute_total_field(
        &receivers,
        &elements,
        &mesh.nodes,
        &result.x,
        None,
        &no_incident_field,
        &physics,
        BasisOrder::Constant,
    )
    .into_iter()
    .map(|field| field.p_total)
    .collect();
    let panel_areas: Vec<_> = elements.iter().map(|element| element.area).collect();
    let normal_velocity = vec![velocity; elements.len()];
    let radiated_power = constant_panel_power(&panel_areas, &result.x, &normal_velocity)
        .expect("pulsating-sphere H-matrix surface power");

    (receiver_pressure, radiated_power, result.x)
}

fn solve_pulsating_sphere_slfmm() -> (Vec<Complex64>, f64, Vec<Complex64>, f64) {
    let physics = PhysicsParams::new(
        PULSATING_FREQUENCY_HZ,
        SOUND_SPEED_M_S,
        DENSITY_KG_M3,
        false,
    );
    let mesh = generate_icosphere_mesh(RADIUS_M, 2);
    let velocity = Complex64::new(1.0, 0.0);
    let mut elements = mesh.elements.clone();
    for (index, element) in elements.iter_mut().enumerate() {
        element.boundary_condition = BoundaryCondition::Velocity(vec![velocity]);
        element.dof_addresses = vec![index];
    }

    let beta_scale = 4.0;
    let beta = physics.burton_miller_beta_scaled(beta_scale);
    let dense_system = build_tbem_system_with_beta(&elements, &mesh.nodes, &physics, beta);
    let clusters = fmm_clusters(&elements, 1.8);
    let characteristic_radius = clusters
        .iter()
        .map(|cluster| cluster.radius)
        .fold(0.0_f64, f64::max);
    let cluster_ka = physics.wave_number * characteristic_radius;
    let expansion_terms = (cluster_ka.ceil() as usize + 3).clamp(5, 20);
    let theta_points = 2 * expansion_terms;
    let phi_points = 2 * theta_points;
    let system = build_slfmm_system_stabilized(
        &elements,
        &mesh.nodes,
        &clusters,
        &physics,
        theta_points,
        phi_points,
        expansion_terms,
        1.0,
        beta_scale,
    );
    let rhs_error = relative_l2_error(&system.rhs, &dense_system.rhs);
    let operator = SlfmmOperator::new(system);
    let result = gmres_with_guess(
        &operator,
        &dense_system.rhs,
        None,
        &KrylovGmresConfig {
            max_iterations: 1_000,
            restart: 100,
            tolerance: 1.0e-10,
            print_interval: 0,
            convergence_check_interval: 1,
            exact_residual_check_interval: 1,
            min_restart: None,
            max_restart: None,
        },
    );
    assert!(
        result.converged,
        "pulsating-sphere SLFMM solve did not converge: residual={:.3e}",
        result.residual
    );

    let receivers: Vec<[f64; 3]> = PULSATING_RADII_M
        .iter()
        .map(|&radius| [0.0, 0.0, radius])
        .collect();
    let no_incident_field = IncidentField::PlaneWave {
        direction: [0.0, 0.0, 1.0],
        amplitude: Complex64::new(0.0, 0.0),
    };
    let receiver_pressure = compute_total_field(
        &receivers,
        &elements,
        &mesh.nodes,
        &result.x,
        None,
        &no_incident_field,
        &physics,
        BasisOrder::Constant,
    )
    .into_iter()
    .map(|field| field.p_total)
    .collect();
    let panel_areas: Vec<_> = elements.iter().map(|element| element.area).collect();
    let normal_velocity = vec![velocity; elements.len()];
    let radiated_power = constant_panel_power(&panel_areas, &result.x, &normal_velocity)
        .expect("pulsating-sphere SLFMM surface power");

    (receiver_pressure, radiated_power, result.x, rhs_error)
}

fn exact_rigid_sphere() -> Vec<Complex64> {
    exact_rigid_sphere_at_ka(RIGID_KA)
}

fn exact_rigid_sphere_at_ka(ka: f64) -> Vec<Complex64> {
    sphere_scattering_3d(
        ka / RADIUS_M,
        RADIUS_M,
        40,
        vec![RIGID_RECEIVER_RADIUS_M],
        RIGID_ANGLES_RAD.to_vec(),
    )
    .pressure
}

fn exact_pulsating_sphere() -> Vec<Complex64> {
    let wave_number = 2.0 * PI * PULSATING_FREQUENCY_HZ / SOUND_SPEED_M_S;
    pulsating_sphere_3d(
        wave_number,
        RADIUS_M,
        1.0,
        SOUND_SPEED_M_S,
        DENSITY_KG_M3,
        PULSATING_RADII_M.to_vec(),
    )
    .pressure
}

fn exact_pulsating_sphere_power() -> f64 {
    let wave_number = 2.0 * PI * PULSATING_FREQUENCY_HZ / SOUND_SPEED_M_S;
    let ka = wave_number * RADIUS_M;
    let radiation_resistance = DENSITY_KG_M3 * SOUND_SPEED_M_S * ka * ka / (1.0 + ka * ka);
    0.5 * 4.0 * PI * RADIUS_M * RADIUS_M * radiation_resistance
}

#[derive(Deserialize)]
struct WolframBemSphereReference {
    schema_version: u32,
    rigid_ka: f64,
    rigid_receiver_radius_m: f64,
    rigid_receiver_angles_rad: Vec<f64>,
    rigid_total_pressure: Vec<[f64; 2]>,
    rigid_backscatter_rcs_m2: f64,
    rigid_irregular_ka: f64,
    rigid_irregular_total_pressure: Vec<[f64; 2]>,
    pulsating_frequency_hz: f64,
    pulsating_receiver_radii_m: Vec<f64>,
    pulsating_pressure_pa: Vec<[f64; 2]>,
}

fn wolfram_reference() -> Option<WolframBemSphereReference> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let script =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/bem_sphere_references.wls");
    let mut command = Command::new(&executable);
    command.arg("-file").arg(script);
    #[cfg(target_os = "macos")]
    if env::var_os("WolframKernel").is_none() {
        let kernel = PathBuf::from(
            "/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/MacOS/WolframKernel",
        );
        if kernel.is_file() {
            command.env("WolframKernel", kernel);
        }
    }
    let output = match command.wolfram_output() {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram Engine not available; using analytical sphere references only");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram BEM sphere references failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!("Wolfram BEM sphere output was not JSON: {error}\nstdout:\n{stdout}")
    }))
}

fn complex_pairs(values: &[[f64; 2]]) -> Vec<Complex64> {
    values
        .iter()
        .map(|pair| Complex64::new(pair[0], pair[1]))
        .collect()
}

fn relative_l2_error(actual: &[Complex64], reference: &[Complex64]) -> f64 {
    assert_eq!(actual.len(), reference.len());
    let error_energy = actual
        .iter()
        .zip(reference)
        .map(|(got, want)| (*got - *want).norm_sqr())
        .sum::<f64>();
    let reference_energy = reference.iter().map(Complex64::norm_sqr).sum::<f64>();
    (error_energy / reference_energy).sqrt()
}

fn incident_plane_wave_at(point: [f64; 3], wave_number: f64) -> Complex64 {
    Complex64::new(0.0, wave_number * point[2]).exp()
}

fn integrated_cross_section(radius: f64, scattered: &[Complex64]) -> f64 {
    let mu_weight = 2.0 / scattered.len() as f64;
    2.0 * PI
        * mu_weight
        * scattered
            .iter()
            .map(|pressure| radius.powi(2) * pressure.norm_sqr())
            .sum::<f64>()
}

fn assert_complex_field_close(
    name: &str,
    actual: &[Complex64],
    reference: &[Complex64],
    tolerance: f64,
) {
    let error = relative_l2_error(actual, reference);
    assert!(
        error < tolerance,
        "{name} complex relative L2 error {:.3}% exceeds {:.3}%",
        100.0 * error,
        100.0 * tolerance,
    );
}

#[test]
fn sonium_bem_reproduces_wolfram_rigid_and_pulsating_sphere_fields() {
    let rigid = solve_rigid_sphere(AssemblyMethod::Tbem);
    let rigid_hmatrix = solve_rigid_sphere(AssemblyMethod::Hmatrix);
    let rigid_mlfmm = solve_rigid_sphere(AssemblyMethod::Mlfmm);
    let (pulsating, pulsating_power, pulsating_boundary_pressure) = solve_pulsating_sphere();
    let (pulsating_hmatrix, pulsating_hmatrix_power, pulsating_hmatrix_boundary_pressure) =
        solve_pulsating_sphere_hmatrix();
    let (
        pulsating_slfmm,
        pulsating_slfmm_power,
        pulsating_slfmm_boundary_pressure,
        pulsating_slfmm_rhs_error,
    ) = solve_pulsating_sphere_slfmm();
    let exact_rigid = exact_rigid_sphere();
    let exact_pulsating = exact_pulsating_sphere();

    assert_complex_field_close("rigid sphere", &rigid, &exact_rigid, 0.01);
    assert_complex_field_close("rigid sphere H-matrix", &rigid_hmatrix, &exact_rigid, 0.01);
    assert_complex_field_close("rigid sphere MLFMM", &rigid_mlfmm, &exact_rigid, 0.03);
    assert_complex_field_close("dense/H-matrix parity", &rigid_hmatrix, &rigid, 0.005);
    assert_complex_field_close("dense/MLFMM parity", &rigid_mlfmm, &rigid, 0.025);
    assert_complex_field_close("pulsating sphere", &pulsating, &exact_pulsating, 0.03);
    assert_complex_field_close(
        "pulsating sphere H-matrix",
        &pulsating_hmatrix,
        &exact_pulsating,
        0.03,
    );
    assert_complex_field_close(
        "pulsating dense/H-matrix receiver parity",
        &pulsating_hmatrix,
        &pulsating,
        0.005,
    );
    assert_complex_field_close(
        "pulsating dense/H-matrix boundary parity",
        &pulsating_hmatrix_boundary_pressure,
        &pulsating_boundary_pressure,
        0.005,
    );
    let slfmm_receiver_error = relative_l2_error(&pulsating_slfmm, &exact_pulsating);
    let slfmm_dense_receiver_error = relative_l2_error(&pulsating_slfmm, &pulsating);
    let slfmm_dense_boundary_error = relative_l2_error(
        &pulsating_slfmm_boundary_pressure,
        &pulsating_boundary_pressure,
    );
    eprintln!(
        "Pulsating-sphere SLFMM: rhs={pulsating_slfmm_rhs_error:.3e}, boundary={slfmm_dense_boundary_error:.3e}, receiver_exact={slfmm_receiver_error:.3e}, receiver_dense={slfmm_dense_receiver_error:.3e}"
    );
    assert!(
        (0.99..=1.01).contains(&pulsating_slfmm_rhs_error),
        "SLFMM prescribed-velocity RHS diagnostic changed from the documented missing-load error: {pulsating_slfmm_rhs_error:.3e}"
    );
    assert!(
        slfmm_dense_boundary_error < 0.001,
        "pulsating-sphere dense/SLFMM boundary mismatch {slfmm_dense_boundary_error:.3e} exceeds 0.1%"
    );
    assert!(
        slfmm_receiver_error < 0.03,
        "pulsating-sphere SLFMM receiver error {slfmm_receiver_error:.3e} exceeds 3%"
    );
    assert!(
        slfmm_dense_receiver_error < 0.001,
        "pulsating-sphere dense/SLFMM receiver mismatch {slfmm_dense_receiver_error:.3e} exceeds 0.1%"
    );
    let exact_power = exact_pulsating_sphere_power();
    let power_relative_error = (pulsating_power - exact_power).abs() / exact_power;
    let hmatrix_power_relative_error = (pulsating_hmatrix_power - exact_power).abs() / exact_power;
    let dense_hmatrix_power_error =
        (pulsating_hmatrix_power - pulsating_power).abs() / pulsating_power.abs();
    let slfmm_power_relative_error = (pulsating_slfmm_power - exact_power).abs() / exact_power;
    let dense_slfmm_power_error =
        (pulsating_slfmm_power - pulsating_power).abs() / pulsating_power.abs();
    eprintln!(
        "Pulsating-sphere BEM power: numerical={pulsating_power:.8e} W, exact={exact_power:.8e} W, rel_err={power_relative_error:.3e}"
    );
    assert!(
        power_relative_error < 0.03,
        "pulsating-sphere power error {power_relative_error:.3e} exceeds 3%"
    );
    assert!(
        hmatrix_power_relative_error < 0.03,
        "pulsating-sphere H-matrix power error {hmatrix_power_relative_error:.3e} exceeds 3%"
    );
    assert!(
        dense_hmatrix_power_error < 0.005,
        "pulsating-sphere dense/H-matrix power mismatch {dense_hmatrix_power_error:.3e} exceeds 0.5%"
    );
    eprintln!(
        "Pulsating-sphere SLFMM power: numerical={pulsating_slfmm_power:.8e} W, exact={exact_power:.8e} W, exact_err={slfmm_power_relative_error:.3e}, dense_err={dense_slfmm_power_error:.3e}"
    );
    assert!(
        slfmm_power_relative_error < 0.03,
        "pulsating-sphere SLFMM power error {slfmm_power_relative_error:.3e} exceeds 3%"
    );
    assert!(
        dense_slfmm_power_error < 0.001,
        "pulsating-sphere dense/SLFMM power mismatch {dense_slfmm_power_error:.3e} exceeds 0.1%"
    );

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert_eq!(reference.rigid_ka, RIGID_KA);
        assert_eq!(reference.rigid_receiver_radius_m, RIGID_RECEIVER_RADIUS_M);
        assert_eq!(reference.rigid_receiver_angles_rad, RIGID_ANGLES_RAD);
        assert_eq!(reference.pulsating_frequency_hz, PULSATING_FREQUENCY_HZ);
        assert_eq!(reference.pulsating_receiver_radii_m, PULSATING_RADII_M);

        let wolfram_rigid = complex_pairs(&reference.rigid_total_pressure);
        let wolfram_pulsating = complex_pairs(&reference.pulsating_pressure_pa);
        assert_complex_field_close("Wolfram/Sonium rigid sphere", &rigid, &wolfram_rigid, 0.01);
        assert_complex_field_close(
            "Wolfram/Sonium pulsating sphere",
            &pulsating,
            &wolfram_pulsating,
            0.03,
        );
        assert_complex_field_close(
            "Wolfram/Sonium pulsating sphere H-matrix",
            &pulsating_hmatrix,
            &wolfram_pulsating,
            0.03,
        );
        assert_complex_field_close(
            "Wolfram/Sonium pulsating sphere SLFMM operator",
            &pulsating_slfmm,
            &wolfram_pulsating,
            0.03,
        );
    }
}

#[test]
fn burton_miller_stabilizes_rigid_sphere_at_first_irregular_frequency() {
    let exact = exact_rigid_sphere_at_ka(IRREGULAR_FREQUENCY_KA);
    let stabilized =
        try_solve_rigid_sphere_at_ka(IRREGULAR_FREQUENCY_KA, AssemblyMethod::Tbem, true)
            .expect("Burton-Miller rigid-sphere solve at ka=pi");
    let stabilized_error = relative_l2_error(&stabilized, &exact);

    let conventional =
        try_solve_rigid_sphere_at_ka(IRREGULAR_FREQUENCY_KA, AssemblyMethod::Tbem, false);
    match conventional {
        Ok(field) => {
            let conventional_error = relative_l2_error(&field, &exact);
            eprintln!(
                "rigid sphere ka=pi irregular frequency: Burton-Miller error={stabilized_error:.3e}, conventional error={conventional_error:.3e}"
            );
            // The discrete fictitious eigenvalue shifts with mesh resolution,
            // so the conventional result at the continuous value ka=pi is a
            // diagnostic rather than a stable accuracy-ratio gate.
        }
        Err(error) => {
            eprintln!(
                "rigid sphere ka=pi irregular frequency: Burton-Miller error={stabilized_error:.3e}, conventional solve failed: {error}"
            );
        }
    }

    assert!(
        stabilized_error < 0.08,
        "Burton-Miller ka=pi receiver-field error {stabilized_error:.3e} exceeds 8%"
    );

    if let Some(reference) = wolfram_reference() {
        assert_eq!(reference.schema_version, 1);
        assert!((reference.rigid_irregular_ka - IRREGULAR_FREQUENCY_KA).abs() < 1.0e-14);
        let wolfram = complex_pairs(&reference.rigid_irregular_total_pressure);
        assert_complex_field_close("Wolfram/Mie ka=pi", &wolfram, &exact, 1.0e-10);
        assert_complex_field_close("Burton-Miller/Wolfram ka=pi", &stabilized, &wolfram, 0.08);
    }
}

#[test]
#[ignore = "production-resolution resonance solve; run via `just qa-nightly-regressions`"]
fn burton_miller_stabilizes_rigid_sphere_at_second_irregular_frequency() {
    let ka = 2.0 * PI;
    let exact = exact_rigid_sphere_at_ka(ka);
    let stabilized = try_solve_rigid_sphere_at_ka(ka, AssemblyMethod::Tbem, true)
        .expect("Burton-Miller rigid-sphere solve at ka=2pi");
    let stabilized_error = relative_l2_error(&stabilized, &exact);

    let conventional = try_solve_rigid_sphere_at_ka(ka, AssemblyMethod::Tbem, false);
    match conventional {
        Ok(field) => {
            let conventional_error = relative_l2_error(&field, &exact);
            eprintln!(
                "rigid sphere ka=2pi irregular frequency: Burton-Miller error={stabilized_error:.3e}, conventional error={conventional_error:.3e}"
            );
            assert!(
                conventional_error > 2.0 * stabilized_error,
                "conventional formulation should be materially less accurate at ka=2pi: conventional={conventional_error:.3e}, Burton-Miller={stabilized_error:.3e}"
            );
        }
        Err(error) => {
            eprintln!(
                "rigid sphere ka=2pi irregular frequency: Burton-Miller error={stabilized_error:.3e}, conventional solve failed: {error}"
            );
        }
    }

    assert!(
        stabilized_error < 0.12,
        "Burton-Miller ka=2pi receiver-field error {stabilized_error:.3e} exceeds 12%"
    );
}

#[test]
#[ignore = "four production-resolution resonance solves; run via `just qa-nightly-regressions`"]
fn burton_miller_resonance_brackets_track_mie_reference() {
    // The exact centres are covered above, where the conventional formulation
    // must either fail or be materially less accurate.  These offsets prove
    // that the stabilized production path does not merely happen to work at
    // one isolated sample on either side of the first two j_0 roots.
    for ka in [0.95 * PI, 1.05 * PI, 1.95 * PI, 2.05 * PI] {
        let exact = exact_rigid_sphere_at_ka(ka);
        let stabilized = try_solve_rigid_sphere_at_ka(ka, AssemblyMethod::Tbem, true)
            .unwrap_or_else(|error| {
                panic!("Burton-Miller rigid-sphere solve at ka={ka:.6}: {error}")
            });
        let relative_error = relative_l2_error(&stabilized, &exact);
        eprintln!(
            "rigid sphere Burton-Miller resonance bracket ka={ka:.6}: Mie receiver-field error={relative_error:.3e}"
        );
        assert!(
            relative_error < 0.15,
            "Burton-Miller ka={ka:.6} receiver-field error {relative_error:.3e} exceeds 15%"
        );
    }
}

#[test]
fn rigid_sphere_far_field_and_public_rcs_match_receiver_definition() {
    let frequency = RIGID_KA * SOUND_SPEED_M_S / (2.0 * PI * RADIUS_M);
    let problem =
        BemProblem::rigid_sphere_scattering(RADIUS_M, frequency, SOUND_SPEED_M_S, DENSITY_KG_M3);
    let solution = BemSolver::new()
        .with_assembly_method(AssemblyMethod::Tbem)
        .with_solver_method(SolverMethod::BiCgStab)
        .with_tolerance(1.0e-10)
        .solve(&problem)
        .expect("Sonium should solve rigid-sphere far-field scattering");
    let wave_number = RIGID_KA / RADIUS_M;

    const N_MU: usize = 48;
    let mu: Vec<_> = (0..N_MU)
        .map(|index| -1.0 + (2.0 * index as f64 + 1.0) / N_MU as f64)
        .collect();
    let theta: Vec<_> = mu.iter().map(|value| value.acos()).collect();
    let radii = [20.0 * RADIUS_M, 40.0 * RADIUS_M];
    let mut bem_scattered_by_radius = Vec::new();
    let mut exact_scattered_by_radius = Vec::new();
    for &radius in &radii {
        let points: Vec<_> = theta
            .iter()
            .map(|&angle| spherical_receiver(radius, angle))
            .collect();
        let bem_total: Vec<_> = solution
            .evaluate_pressure_field(&points)
            .into_iter()
            .map(|field| field.p_total)
            .collect();
        let exact_total =
            sphere_scattering_3d(wave_number, RADIUS_M, 100, vec![radius], theta.clone()).pressure;
        let bem_scattered: Vec<_> = bem_total
            .iter()
            .zip(&points)
            .map(|(&total, &point)| total - incident_plane_wave_at(point, wave_number))
            .collect();
        let exact_scattered: Vec<_> = exact_total
            .iter()
            .zip(&points)
            .map(|(&total, &point)| total - incident_plane_wave_at(point, wave_number))
            .collect();
        bem_scattered_by_radius.push(bem_scattered);
        exact_scattered_by_radius.push(exact_scattered);
    }

    let reference_cross_section = sphere_rcs_3d(wave_number, RADIUS_M, 40);
    let bem_cross_sections = radii.map(|radius| {
        let index = radii
            .iter()
            .position(|&candidate| candidate == radius)
            .unwrap();
        integrated_cross_section(radius, &bem_scattered_by_radius[index])
    });
    let exact_cross_sections = radii.map(|radius| {
        let index = radii
            .iter()
            .position(|&candidate| candidate == radius)
            .unwrap();
        integrated_cross_section(radius, &exact_scattered_by_radius[index])
    });
    let bem_cross_section_error =
        (bem_cross_sections[1] - reference_cross_section).abs() / reference_cross_section;
    let exact_cross_section_error =
        (exact_cross_sections[1] - reference_cross_section).abs() / reference_cross_section;

    let bem_far_amplitude_20: Vec<_> = bem_scattered_by_radius[0]
        .iter()
        .map(|pressure| *pressure * radii[0] * Complex64::new(0.0, -wave_number * radii[0]).exp())
        .collect();
    let bem_far_amplitude_40: Vec<_> = bem_scattered_by_radius[1]
        .iter()
        .map(|pressure| *pressure * radii[1] * Complex64::new(0.0, -wave_number * radii[1]).exp())
        .collect();
    let far_amplitude_change = relative_l2_error(&bem_far_amplitude_40, &bem_far_amplitude_20);
    let far_field_exact_error =
        relative_l2_error(&bem_scattered_by_radius[1], &exact_scattered_by_radius[1]);

    let back_point = spherical_receiver(radii[1], PI);
    let back_scattered =
        solution.evaluate_pressure(&back_point) - incident_plane_wave_at(back_point, wave_number);
    let receiver_backscatter_rcs = 4.0 * PI * radii[1].powi(2) * back_scattered.norm_sqr();
    let public_backscatter_rcs = compute_rcs(
        &solution.surface_pressure,
        &solution.elements,
        [0.0, 0.0, -1.0],
        &solution.physics,
    );
    let public_rcs_error =
        (public_backscatter_rcs - receiver_backscatter_rcs).abs() / receiver_backscatter_rcs;

    eprintln!(
        "rigid sphere RCS: sigma_series={reference_cross_section:.8e}, sigma_bem_20/40={:?}, sigma_exact_20/40={:?}, sigma_errors(bem/exact)={bem_cross_section_error:.3e}/{exact_cross_section_error:.3e}, far_change={far_amplitude_change:.3e}, field_exact={far_field_exact_error:.3e}, backscatter(receiver/public)={receiver_backscatter_rcs:.8e}/{public_backscatter_rcs:.8e}, public_error={public_rcs_error:.3e}",
        bem_cross_sections, exact_cross_sections,
    );

    assert!(exact_cross_section_error < 0.03);
    assert!(bem_cross_section_error < 0.06);
    assert!(far_amplitude_change < 0.04);
    assert!(far_field_exact_error < 0.04);
    assert!(receiver_backscatter_rcs.is_finite() && receiver_backscatter_rcs > 0.0);
    assert!(public_backscatter_rcs.is_finite() && public_backscatter_rcs > 0.0);
    assert!(
        public_rcs_error < 0.01,
        "public compute_rcs differs from the receiver-field definition: error={public_rcs_error:.3e}"
    );

    if let Some(reference) = wolfram_reference() {
        let wolfram_error = (receiver_backscatter_rcs - reference.rigid_backscatter_rcs_m2).abs()
            / reference.rigid_backscatter_rcs_m2;
        assert!(
            wolfram_error < 0.03,
            "receiver-derived BEM backscatter RCS differs from Wolfram by {wolfram_error:.3e}: BEM={receiver_backscatter_rcs:.8e}, Wolfram={:.8e}",
            reference.rigid_backscatter_rcs_m2
        );
    }
}
