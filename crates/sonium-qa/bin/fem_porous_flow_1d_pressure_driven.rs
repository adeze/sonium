use sonium_fem::assembly::assemble_bilinear;
use sonium_fem::basis::PolynomialDegree;
use sonium_fem::boundary::generic::{FemSystem, apply_dirichlet_values};
use sonium_fem::mesh::generators::rectangular_mesh_triangles;
use sonium_fem::physics::fluid::porous_flow::{
    BiotMaterial, DarcyMaterial, DarcyPressureForm, RichardsMaterial, VanGenuchtenMualem,
    terzaghi_consolidation_coefficient, terzaghi_excess_pressure_ratio,
    theis_cooper_jacob_drawdown,
};
use sonium_fem::spaces::{FunctionSpace, H1Space};
use sonium_solvers::{GmresConfig, gmres};

fn solve_darcy_1d(nx: usize, ny: usize, perm: f64, mu: f64) -> (f64, f64) {
    let l = 1.0;
    let h = 0.1;
    let p0 = 1.0;
    let p_l = 0.0;

    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);

    let material = DarcyMaterial::isotropic(perm, mu, 0.0);
    let form = DarcyPressureForm::new(&material);
    let matrix = assemble_bilinear(&form, &space, &space, &mesh, 2);
    let rhs = vec![0.0; n_dofs];

    let mut system = FemSystem::new(matrix, rhs);
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| p0, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| p_l, None);

    let csr = system.matrix.to_csr();
    let result = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 2000,
            restart: 200,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "GMRES did not converge");

    let mut max_p_err = 0.0;
    let exact_grad_x = -(p0 - p_l) / l;
    let exact_u = -(perm / mu) * exact_grad_x;
    let mut max_u_err = 0.0;

    for element in &mesh.elements {
        let n_local = element.nodes.len();
        if n_local == 3 {
            let n0 = element.nodes[0];
            let n1 = element.nodes[1];
            let n2 = element.nodes[2];
            let x0 = mesh.nodes[n0].x;
            let y0 = mesh.nodes[n0].y;
            let x1 = mesh.nodes[n1].x;
            let y1 = mesh.nodes[n1].y;
            let x2 = mesh.nodes[n2].x;
            let y2 = mesh.nodes[n2].y;

            let area2 = ((x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)).abs();
            if area2 < 1e-14 {
                continue;
            }
            let inv_area2 = 1.0 / area2;

            let dp_dx = inv_area2
                * ((result.x[n1] - result.x[n0]) * (y2 - y0)
                    - (result.x[n2] - result.x[n0]) * (y1 - y0));
            let dp_dy = -(inv_area2
                * ((result.x[n1] - result.x[n0]) * (x2 - x0)
                    - (result.x[n2] - result.x[n0]) * (x1 - x0)));

            let u_x = -(perm / mu) * dp_dx;
            let u_y = -(perm / mu) * dp_dy;

            let err_u = ((u_x - exact_u).powi(2) + u_y.powi(2)).sqrt();
            if err_u > max_u_err {
                max_u_err = err_u;
            }
        }

        for &n in &element.nodes {
            let x = mesh.nodes[n].x;
            let exact_p = p0 + (p_l - p0) * (x / l);
            let err = (result.x[n] - exact_p).abs();
            if err > max_p_err {
                max_p_err = err;
            }
        }
    }

    (max_p_err, max_u_err)
}

fn test_darcy_1d_pressure_driven() {
    let perm = 1.0;
    let mu = 1.0;
    let (max_p_err, max_u_err) = solve_darcy_1d(20, 2, perm, mu);
    println!(
        "Porous flow (nx=20): max_p_err={:.2e}, max_u_err={:.2e}",
        max_p_err, max_u_err
    );
    assert!(
        max_p_err < 1e-10,
        "Pressure error too large: {:.2e}",
        max_p_err
    );
    assert!(
        max_u_err < 1e-10,
        "Velocity error too large: {:.2e}",
        max_u_err
    );
}

fn test_darcy_1d_refinement() {
    let perm = 1.0;
    let mu = 1.0;

    for nx in [10, 20, 40, 80] {
        let ny = 2.max(nx / 10);
        let (max_p_err, max_u_err) = solve_darcy_1d(nx, ny, perm, mu);
        println!(
            "  nx={:3} max_p_err={:.2e} max_u_err={:.2e}",
            nx, max_p_err, max_u_err
        );
        assert!(
            max_p_err < 1e-9,
            "Pressure error too large at nx={}: {:.2e}",
            nx,
            max_p_err
        );
        assert!(
            max_u_err < 1e-9,
            "Velocity error too large at nx={}: {:.2e}",
            nx,
            max_u_err
        );
    }
}

fn test_darcy_1d_anisotropic() {
    let l = 1.0;
    let h = 0.1;
    let p0 = 1.0;
    let p_l = 0.0;
    let nx = 20;
    let ny = 2;

    let mesh = rectangular_mesh_triangles(0.0, l, 0.0, h, nx, ny);
    let space = H1Space::new(PolynomialDegree::P1);
    let n_dofs = space.num_global_dofs(&mesh);

    let mut material = DarcyMaterial::isotropic(1.0, 1.0, 0.0);
    material.permeability = [[1.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]];

    let form = DarcyPressureForm::new(&material);
    let matrix = assemble_bilinear(&form, &space, &space, &mesh, 2);
    let rhs = vec![0.0; n_dofs];

    let mut system = FemSystem::new(matrix, rhs);
    apply_dirichlet_values(&mut system, &mesh, &space, 1, &|_, _, _| p0, None);
    apply_dirichlet_values(&mut system, &mesh, &space, 2, &|_, _, _| p_l, None);

    let csr = system.matrix.to_csr();
    let result = gmres(
        &csr,
        &system.rhs,
        &GmresConfig {
            max_iterations: 2000,
            restart: 200,
            tolerance: 1e-12,
            print_interval: 0,
            convergence_check_interval: 1,
            min_restart: None,
            max_restart: None,
            exact_residual_check_interval: 0,
        },
    );
    assert!(result.converged, "GMRES did not converge (anisotropic)");

    let mut max_p_err = 0.0;
    for element in &mesh.elements {
        for &n in &element.nodes {
            let x = mesh.nodes[n].x;
            let exact_p = p0 + (p_l - p0) * (x / l);
            let err = (result.x[n] - exact_p).abs();
            if err > max_p_err {
                max_p_err = err;
            }
        }
    }
    println!("Porous flow anisotropic: max_p_err={:.2e}", max_p_err);
    assert!(
        max_p_err < 1e-10,
        "Anisotropic pressure error too large: {:.2e}",
        max_p_err
    );
}

fn test_richards_van_genuchten_closure() {
    let retention = VanGenuchtenMualem {
        alpha: 0.8,
        n: 2.0,
        pore_connectivity: 0.5,
        residual_saturation: 0.1,
        saturated_saturation: 1.0,
    };
    let material = RichardsMaterial::new(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        1.0,
        0.1,
        3.0,
        0.0,
    )
    .with_retention(retention);

    let dry_sat = retention.saturation_from_pressure(-10.0);
    let wet_sat = retention.saturation_from_pressure(-0.1);
    let dry_mobility = material.hydraulic_mobility(dry_sat)[0][0];
    let wet_mobility = material.hydraulic_mobility(wet_sat)[0][0];
    let capacity = retention.moisture_capacity(-1.0, 0.35);

    println!(
        "Richards closure: dry_sat={:.4}, wet_sat={:.4}, kr_ratio={:.4e}, capacity={:.4e}",
        dry_sat,
        wet_sat,
        dry_mobility / wet_mobility,
        capacity
    );
    assert!(dry_sat > 0.1 && dry_sat < wet_sat);
    assert!(wet_sat < 1.0);
    assert!(dry_mobility < wet_mobility);
    assert!(capacity > 0.0);
}

fn test_terzaghi_theis_and_biot_coefficients() {
    let darcy = DarcyMaterial::isotropic(2.0e-12, 1.0e-3, 1.0e-7);
    let cv = terzaghi_consolidation_coefficient(&darcy);
    let pressure_ratio = terzaghi_excess_pressure_ratio(0.5, 0.25);
    let drawdown = theis_cooper_jacob_drawdown(0.01, 1.0e-3, 1.0e-4, 10.0, 1.0e7);
    let biot = BiotMaterial {
        biot_alpha: 0.8,
        storage: 2.0e-10,
        drained_bulk_modulus: 5.0e9,
    };
    let b = biot.skempton_coefficient();
    let cp = biot.fast_p_wave_speed(1800.0, 2.0e9);

    println!(
        "Porous coefficients: cv={:.4e}, Terzaghi u/u0={:.4e}, Theis s={:.4e}, B={:.4e}, cp={:.2}",
        cv, pressure_ratio, drawdown, b, cp
    );
    assert!((cv - 0.02).abs() < 1.0e-12);
    assert!(pressure_ratio > 0.0 && pressure_ratio < 1.0);
    assert!(drawdown > 0.0);
    assert!(b > 0.0 && b < 1.0);
    assert!(cp > 1000.0);
}

fn main() {
    let result = std::panic::catch_unwind(|| {
        println!("=== QA: porous flow 1D pressure driven ===");
        test_darcy_1d_pressure_driven();
        test_darcy_1d_refinement();
        test_darcy_1d_anisotropic();
        test_richards_van_genuchten_closure();
        test_terzaghi_theis_and_biot_coefficients();
        println!("SUCCESS");
    });
    if result.is_err() {
        eprintln!("FAILED");
        std::process::exit(1);
    }
}
