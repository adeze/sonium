//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::eddy_slab_filament_liftoff;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let near = eddy_slab_filament_liftoff(0.001, 0.01);
        let far = eddy_slab_filament_liftoff(0.01, 0.01);
        assert!(near > far);
        println!(
            "Eddy Dodd-Deeds lift-off envelope: near={:.6e}, far={:.6e}",
            near, far
        );
    });
    reference_result::emit("reference.fem-eddy-current-slab-filament.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
