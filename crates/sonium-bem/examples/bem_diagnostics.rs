//! BEM Diagnostics - Step-by-step investigation of high-frequency issues

fn main() {
    use sonium_bem::core::mesh::generators::generate_sphere_mesh;

    use sonium_bem::core::types::PhysicsParams;

    use std::f64::consts::PI;

    println!("=== BEM Diagnostics: Step-by-Step Investigation ===\n");

    let radius = 0.1;
    let speed_of_sound = 343.0;
    let density = 1.21;

    for ka_target in [0.2, 0.5, 1.0, 2.0] {
        let frequency = ka_target * speed_of_sound / (2.0 * PI * radius);
        let k = 2.0 * PI * frequency / speed_of_sound;
        let ka = k * radius;

        println!("\n============================================================");
        println!(
            "=== ka = {:.2} (f = {:.1} Hz, k = {:.4}) ===",
            ka, frequency, k
        );
        println!("============================================================");

        let n_theta = 8;
        let n_phi = 16;
        let mesh = generate_sphere_mesh(radius, n_theta, n_phi);
        let physics = PhysicsParams::new(frequency, speed_of_sound, density, false);

        println!("\n--- Step 1: Incident Field RHS ---");
        investigate_incident_field(&mesh, &physics, k, radius);

        println!("\n--- Step 2: System Matrix Properties ---");
        investigate_system_matrix(&mesh, &physics);

        println!("\n--- Step 3: Solution Comparison ---");
        investigate_solution(&mesh, &physics, k, radius);
    }
}

fn investigate_incident_field(
    mesh: &sonium_bem::core::types::Mesh,
    physics: &sonium_bem::core::types::PhysicsParams,
    k: f64,
    radius: f64,
) {
    use sonium_bem::core::incident::IncidentField;

    let incident = IncidentField::plane_wave_z();

    let centers: Vec<[f64; 3]> = mesh.elements.iter().map(|e| e.center).collect();
    let normals: Vec<[f64; 3]> = mesh.elements.iter().map(|e| e.normal).collect();

    let p_inc = incident.evaluate_pressure(&centers, physics);
    let dpdn_inc = incident.evaluate_normal_derivative(&centers, &normals, physics);

    let rhs_cbie = incident.compute_rhs(&centers, &normals, physics, false);
    let rhs_bm = incident.compute_rhs(&centers, &normals, physics, true);

    let p_inc_max = p_inc.iter().map(|x| x.norm()).fold(0.0f64, f64::max);
    let p_inc_min = p_inc.iter().map(|x| x.norm()).fold(f64::INFINITY, f64::min);
    let dpdn_max = dpdn_inc.iter().map(|x| x.norm()).fold(0.0f64, f64::max);
    let dpdn_min = dpdn_inc
        .iter()
        .map(|x| x.norm())
        .fold(f64::INFINITY, f64::min);
    let rhs_cbie_max = rhs_cbie.iter().map(|x| x.norm()).fold(0.0f64, f64::max);
    let rhs_bm_max = rhs_bm.iter().map(|x| x.norm()).fold(0.0f64, f64::max);

    println!("  |p_inc|: min={:.4}, max={:.4}", p_inc_min, p_inc_max);
    println!("  |dpdn|: min={:.4}, max={:.4}", dpdn_min, dpdn_max);
    println!("  |RHS_CBIE|_max = {:.4}", rhs_cbie_max);
    println!("  |RHS_BM|_max   = {:.4}", rhs_bm_max);

    let expected_p = (k * radius).cos();
    println!(
        "\n  Expected p_inc ≈ exp(ik*r) = cos({:.4}) = {:.6}",
        k * radius,
        expected_p
    );
}

fn investigate_system_matrix(
    mesh: &sonium_bem::core::types::Mesh,
    physics: &sonium_bem::core::types::PhysicsParams,
) {
    use num_complex::Complex64;
    use sonium_bem::core::assembly::tbem::build_tbem_system;
    use sonium_bem::core::integration::singular_integration;
    use sonium_bem::core::types::BoundaryCondition;
    use sonium_solvers::dense::DenseMatrix;

    let mut elements = mesh.elements.clone();
    for (i, elem) in elements.iter_mut().enumerate() {
        elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        elem.dof_addresses = vec![i];
    }

    let system = build_tbem_system(&elements, &mesh.nodes, physics);

    let n = system.num_dofs;
    let mut diag_magnitudes = Vec::with_capacity(n);
    let mut off_diag_magnitudes = Vec::new();

    for i in 0..n {
        diag_magnitudes.push(system.matrix[[i, i]].norm());
        for j in 0..n {
            if i != j {
                off_diag_magnitudes.push(system.matrix[[i, j]].norm());
            }
        }
    }

    let diag_max = diag_magnitudes.iter().cloned().fold(0.0f64, f64::max);
    let diag_min = diag_magnitudes
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    let diag_avg: f64 = diag_magnitudes.iter().sum::<f64>() / n as f64;

    let off_diag_max = off_diag_magnitudes.iter().cloned().fold(0.0f64, f64::max);
    let off_diag_avg: f64 =
        off_diag_magnitudes.iter().sum::<f64>() / off_diag_magnitudes.len() as f64;

    println!("  Matrix size: {} x {}", n, n);
    println!(
        "  Diagonal |A[i,i]|: min={:.6}, avg={:.6}, max={:.6}",
        diag_min, diag_avg, diag_max
    );
    println!(
        "  Off-diagonal: avg={:.6}, max={:.6}",
        off_diag_avg, off_diag_max
    );
    println!("  Diagonal dominance ratio: {:.3}", diag_avg / off_diag_avg);

    let beta = physics.burton_miller_beta();
    let gamma = physics.gamma();
    let tau = physics.tau;
    println!("\n  Burton-Miller params:");
    println!("    gamma = {:.6}", gamma);
    println!(
        "    beta = {:.6} + {:.6}i (i.e., i/k where k={:.4})",
        beta.re, beta.im, physics.wave_number
    );
    println!("    tau = {:.6}", tau);

    let elem = &elements[0];
    let num_nodes = elem.connectivity.len();
    let mut elem_coords = DenseMatrix::<f64>::zeros(num_nodes, 3);
    for (i, &node_idx) in elem.connectivity.iter().enumerate() {
        for j in 0..3 {
            elem_coords[[i, j]] = mesh.nodes[node_idx][j];
        }
    }

    let result = singular_integration(
        &elem.center,
        &elem.normal,
        &elem_coords,
        elem.element_type,
        physics,
        None,
        0,
        false,
    );

    println!(
        "\n    G integral   = {:.6} + {:.6}i",
        result.g_integral.re, result.g_integral.im
    );
    println!(
        "    H integral   = {:.6} + {:.6}i",
        result.dg_dn_integral.re, result.dg_dn_integral.im
    );
    println!(
        "    E integral   = {:.6} + {:.6}i",
        result.d2g_dnxdny_integral.re, result.d2g_dnxdny_integral.im
    );

    let rhs_max = system.rhs.iter().map(|x| x.norm()).fold(0.0f64, f64::max);
    println!("\n  System RHS (from assembly): |rhs|_max = {:.6}", rhs_max);
}

fn investigate_solution(
    mesh: &sonium_bem::core::types::Mesh,
    physics: &sonium_bem::core::types::PhysicsParams,
    k: f64,
    radius: f64,
) {
    use num_complex::Complex64;
    use sonium_bem::analytical::sphere_scattering_3d;
    use sonium_bem::core::assembly::tbem::{build_tbem_system, build_tbem_system_with_beta};
    use sonium_bem::core::incident::IncidentField;
    use sonium_bem::core::solver::direct::lu_solve;
    use sonium_bem::core::types::BoundaryCondition;

    let mut elements = mesh.elements.clone();
    for (i, elem) in elements.iter_mut().enumerate() {
        elem.boundary_condition = BoundaryCondition::Velocity(vec![Complex64::new(0.0, 0.0)]);
        elem.dof_addresses = vec![i];
    }

    let edge_e_magnitude = 70.0;
    let min_beta_e = 10.0;
    let beta_floored = physics.burton_miller_beta_floored(edge_e_magnitude, min_beta_e);

    let system_traditional = build_tbem_system(&elements, &mesh.nodes, physics);
    let system_floored = build_tbem_system_with_beta(&elements, &mesh.nodes, physics, beta_floored);

    let incident = IncidentField::plane_wave_z();
    let n = elements.len();
    let centers: Vec<[f64; 3]> = elements.iter().map(|e| e.center).collect();
    let normals: Vec<[f64; 3]> = elements.iter().map(|e| e.normal).collect();

    let incident_rhs_trad = incident.compute_rhs(&centers, &normals, physics, true);
    let total_rhs_trad: Vec<Complex64> = system_traditional
        .rhs
        .iter()
        .zip(incident_rhs_trad.iter())
        .map(|(a, b)| a + b)
        .collect();

    let incident_rhs_floored =
        incident.compute_rhs_with_beta(&centers, &normals, physics, beta_floored);
    let total_rhs_floored: Vec<Complex64> = system_floored
        .rhs
        .iter()
        .zip(incident_rhs_floored.iter())
        .map(|(a, b)| a + b)
        .collect();

    let beta_zero = Complex64::new(0.0, 0.0);
    let system_cbie = build_tbem_system_with_beta(&elements, &mesh.nodes, physics, beta_zero);
    let incident_rhs_cbie = incident.compute_rhs_with_beta(&centers, &normals, physics, beta_zero);
    let total_rhs_cbie: Vec<Complex64> = system_cbie
        .rhs
        .iter()
        .zip(incident_rhs_cbie.iter())
        .map(|(a, b)| a + b)
        .collect();

    let solution_trad =
        lu_solve(&system_traditional.matrix, &total_rhs_trad).expect("Solver failed");
    let solution_floored =
        lu_solve(&system_floored.matrix, &total_rhs_floored).expect("Solver failed");
    let solution_cbie = lu_solve(&system_cbie.matrix, &total_rhs_cbie).expect("Solver failed");

    let p_avg = |sol: &[Complex64]| sol.iter().map(|x| x.norm()).sum::<f64>() / n as f64;
    let p_max = |sol: &[Complex64]| sol.iter().map(|x| x.norm()).fold(0.0f64, f64::max);

    println!("  CBIE-only (β=0):");
    println!(
        "    Surface |p|: avg={:.6}, max={:.6}",
        p_avg(&solution_cbie),
        p_max(&solution_cbie)
    );

    println!("\n  Traditional Burton-Miller β=i/k:");
    println!(
        "    Surface |p|: avg={:.6}, max={:.6}",
        p_avg(&solution_trad),
        p_max(&solution_trad)
    );

    println!(
        "\n  Floored β = max(i/k, i*{:.4}):",
        min_beta_e / edge_e_magnitude
    );
    println!(
        "    Surface |p|: avg={:.6}, max={:.6}",
        p_avg(&solution_floored),
        p_max(&solution_floored)
    );

    let analytical_surface = sphere_scattering_3d(k, radius, 50, vec![radius], vec![0.0]);
    let analytical_p_surface = analytical_surface.pressure[0].norm();
    println!(
        "\n  Analytical surface |p| at forward point: {:.6}",
        analytical_p_surface
    );
}
