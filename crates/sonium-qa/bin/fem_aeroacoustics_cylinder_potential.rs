//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::aero_cylinder_potential_velocity;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let radius = 0.5;
        let u_inf = 20.0;
        let (u_r_surface, _) = aero_cylinder_potential_velocity(radius, radius, 0.0, u_inf);
        let (_, u_theta_side) =
            aero_cylinder_potential_velocity(radius, radius, std::f64::consts::FRAC_PI_2, u_inf);
        assert!(u_r_surface.abs() < 1e-12);
        assert!((u_theta_side + 2.0 * u_inf).abs() < 1e-12);
        println!(
            "Cylinder surface tangential velocity = {:.6e}",
            u_theta_side
        );
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
