use sonium_fem::physics::electromagnetics::electrostatic::{
    EPSILON_0, grounded_plane_image_potential,
};

fn main() {
    let result = std::panic::catch_unwind(|| {
        let q = 1e-9;
        let phi_plane =
            grounded_plane_image_potential([0.2, 0.0, 0.0], [0.0, 0.0, 0.5], q, EPSILON_0);
        let phi_above =
            grounded_plane_image_potential([0.2, 0.0, 0.25], [0.0, 0.0, 0.5], q, EPSILON_0);
        assert!(phi_plane.abs() < 1e-9);
        assert!(phi_above > 0.0);
        println!("Electrostatics image charge: phi_above = {:.6e}", phi_above);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
