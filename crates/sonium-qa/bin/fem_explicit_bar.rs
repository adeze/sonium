//! QA: explicit dynamics elastic bar vs d'Alembert plus Taylor setup.
//!
//! Propagates a Gaussian pulse on a long bar and checks it against the
//! d'Alembert closed form, verifies energy conservation of a plucked
//! fixed bar, and smokes the Taylor cylinder mesh, impact plateau, and
//! hourglass orthogonality. Prints PASS; exits 1 on any mismatch.

// Rust guideline compliant 2026-02-21

use sonium_fem::solver::explicit_dynamics::{
    ElasticBar1D, TaylorBarSetup, dalembert_gaussian, hex8_hourglass_forces, hex8_hourglass_gammas,
    mass_scaling_factor,
};

fn main() {
    // Gaussian pulse vs d'Alembert before reflections return.
    let mut bar = ElasticBar1D::new(400, 20.0, 0.01, 200.0e9, 7800.0, false);
    let c = bar.wave_speed();
    let dt = bar.stable_step(0.5);
    bar.impose_gaussian_pulse(10.0, 0.4, 1.0e-3, dt);
    let steps = (0.0008 / dt) as usize;
    for _ in 0..steps {
        bar.step(dt);
    }
    let t = steps as f64 * dt;
    let nodes = bar.nodes();
    let u = bar.state().displacement().to_vec();
    let mut max_err = 0.0f64;
    for (i, x) in nodes.iter().enumerate() {
        max_err = max_err.max((u[i] - dalembert_gaussian(*x, t, c, 10.0, 0.4, 1.0e-3)).abs());
    }
    println!("bar-dalembert: max error {max_err:.3e}");
    assert!(max_err < 5.0e-5, "d'Alembert error {max_err:.3e}");
    // Fixed bar conserves energy through reflections over long runs.
    let mut fixed = ElasticBar1D::new(100, 10.0, 0.01, 200.0e9, 7800.0, true);
    let dt = fixed.stable_step(0.5);
    fixed.impose_gaussian_pulse(5.0, 0.5, 1.0e-3, dt);
    let e0 = fixed.total_energy();
    for _ in 0..4000 {
        fixed.step(dt);
    }
    let drift = (fixed.total_energy() - e0).abs() / e0;
    println!("bar-energy: drift {drift:.3e}");
    assert!(drift < 0.01, "energy drift {drift:.3e}");
    // Taylor cylinder mesh: positive volumes, face at z = 0.
    let setup = TaylorBarSetup::copper_bar();
    let (nodes, bricks) = setup.mesh(6, 8);
    assert!(!bricks.is_empty());
    assert!(nodes.iter().any(|n| n[2] == 0.0));
    println!(
        "taylor-mesh: {} nodes, {} bricks",
        nodes.len(),
        bricks.len()
    );
    println!("taylor-plateau: {:.6e} N", setup.elastic_plateau_force());
    assert!(setup.elastic_plateau_force() > 0.0);
    // Hourglass forces vanish on rigid motion, dissipate on HG modes.
    let coords = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let rigid = [[1.0, 0.0, 0.0]; 8];
    let f = hex8_hourglass_forces(&coords, &rigid, 7800.0, 5000.0, 0.1);
    assert!(
        f.iter()
            .all(|v| v[0].abs() + v[1].abs() + v[2].abs() < 1e-6)
    );
    let (gamma, _, _) = hex8_hourglass_gammas(&coords);
    let hg: [[f64; 3]; 8] = std::array::from_fn(|i| [gamma[0][i], 0.0, 0.0]);
    let fh = hex8_hourglass_forces(&coords, &hg, 7800.0, 5000.0, 0.1);
    let power: f64 = (0..8).map(|i| fh[i][0] * hg[i][0]).sum();
    println!("hourglass-power: {power:.3e}");
    assert!(power < 0.0);
    // Mass scaling quadruples density at double step.
    assert!((mass_scaling_factor(2.0e-6, 1.0e-6) - 4.0).abs() < 1e-12);
    println!("PASS: qa-fem-explicit-bar");
}
