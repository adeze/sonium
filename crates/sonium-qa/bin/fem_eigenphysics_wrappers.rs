use sonium_fem::assembly::TripletMatrix;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::electromagnetics::maxwell::EmMaterial;
use sonium_fem::physics::solid::eigenmodes::{
    AcousticEigenmodeConfig, ElectromagneticEigenmodeConfig, StructuralModalConfig,
    ThermalStabilityConfig, build_acoustic_eigenmode_problem,
    build_electromagnetic_eigenmode_problem, build_linear_buckling_problem,
    build_structural_modal_problem, build_thermal_stability_problem, critical_loads,
    euler_column_buckling_load, thermal_decay_rates,
};
use sonium_fem::physics::solid::elasticity::IsotropicMaterial;
use sonium_fem::physics::thermal::heat_conduction::ThermalMaterial;
use sonium_solvers::LobpcgConfig;

fn lobpcg_one() -> LobpcgConfig {
    LobpcgConfig {
        num_eigenvalues: 1,
        max_iterations: 80,
        tolerance: 1.0e-7,
        ..Default::default()
    }
}

fn test_structural_modal_wrapper() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.2, 4, 1);
    let problem = build_structural_modal_problem(
        &mesh,
        IsotropicMaterial::new(1000.0, 0.25),
        &StructuralModalConfig::plane_stress(2.0, vec![1]),
    );
    let result = problem.solve(&lobpcg_one()).unwrap();
    let frequencies = problem.eigenvalues_to_frequencies_hz(&result.eigenvalues);

    println!(
        "Structural modal wrapper: lambda={:.4e}, f={:.4e} Hz, converged={}",
        result.eigenvalues[0], frequencies[0], result.converged
    );
    assert!(problem.stiffness_like.num_rows > 0);
    assert!(result.eigenvalues[0] > 0.0);
    assert!(frequencies[0] > 0.0);
}

fn test_linear_buckling_wrapper() {
    let mut elastic = TripletMatrix::new(2, 2);
    elastic.push(0, 0, 4.0);
    elastic.push(1, 1, 9.0);
    let mut geometric = TripletMatrix::new(2, 2);
    geometric.push(0, 0, 1.0);
    geometric.push(1, 1, 1.0);
    let problem = build_linear_buckling_problem(&elastic, &geometric, &[]);
    let result = problem.solve(&lobpcg_one()).unwrap();
    let loads = critical_loads(&result.eigenvalues);
    let euler = euler_column_buckling_load(200.0e9, 1.0e-8, 2.0, 1.0);

    println!(
        "Buckling wrapper: lambda={:.4e}, Euler load={:.4e}",
        loads[0], euler
    );
    assert!((loads[0] - 4.0).abs() < 1.0e-8);
    assert!(euler > 0.0);
}

fn test_thermal_stability_wrapper() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.2, 8, 1);
    let material = ThermalMaterial::new(2.0, 4.0, 5.0);
    let problem = build_thermal_stability_problem(
        &mesh,
        material,
        &ThermalStabilityConfig {
            degree: PolynomialDegree::P1,
            reaction_coefficient: 0.0,
            fixed_temperature_boundary_markers: vec![1, 2],
        },
    );
    let result = problem.solve(&lobpcg_one()).unwrap();
    let rates = thermal_decay_rates(&result.eigenvalues);
    let expected = material.diffusivity() * std::f64::consts::PI.powi(2);

    println!(
        "Thermal stability wrapper: rate={:.4e}, expected~{:.4e}",
        rates[0], expected
    );
    assert!(rates[0] > 0.0);
    assert!((rates[0] - expected).abs() / expected < 0.02);
}

fn test_acoustic_eigenmode_wrapper() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.2, 8, 1);
    let c = 343.0;
    let problem = build_acoustic_eigenmode_problem(
        &mesh,
        &AcousticEigenmodeConfig {
            degree: PolynomialDegree::P1,
            sound_speed: c,
            pressure_release_boundary_markers: vec![1, 2],
        },
    );
    let result = problem.solve(&lobpcg_one()).unwrap();
    let frequencies = problem.eigenvalues_to_frequencies_hz(&result.eigenvalues);
    let expected = c / 2.0;

    println!(
        "Acoustic eigenmode wrapper: f={:.4e} Hz, expected~{:.4e} Hz",
        frequencies[0], expected
    );
    assert!(frequencies[0] > 0.0);
    assert!((frequencies[0] - expected).abs() / expected < 0.01);
}

fn test_em_eigenmode_wrapper_builds() {
    let mesh = rectangular_mesh_triangles(0.0, 1.0, 0.0, 0.5, 2, 1);
    let problem = build_electromagnetic_eigenmode_problem(
        &mesh,
        EmMaterial::vacuum(),
        &ElectromagneticEigenmodeConfig::default(),
    );

    println!(
        "EM eigenmode wrapper: dofs={}, nnz K={}, nnz M={}",
        problem.stiffness_like.num_rows,
        problem.stiffness_like.nnz(),
        problem.mass_like.nnz()
    );
    assert!(problem.stiffness_like.num_rows > 0);
    assert_eq!(problem.stiffness_like.num_rows, problem.mass_like.num_rows);
    assert!(problem.stiffness_like.nnz() > 0);
    assert!(problem.mass_like.nnz() > 0);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: FEM eigenphysics wrappers ===");
        test_structural_modal_wrapper();
        test_linear_buckling_wrapper();
        test_thermal_stability_wrapper();
        test_acoustic_eigenmode_wrapper();
        test_em_eigenmode_wrapper_builds();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
