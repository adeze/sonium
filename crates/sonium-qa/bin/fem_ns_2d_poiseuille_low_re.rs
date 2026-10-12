//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::ns_plane_poiseuille_low_re;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let h = 1.0;
        let umax = 0.2;
        assert!(ns_plane_poiseuille_low_re(0.0, h, umax).abs() < 1e-12);
        assert!(ns_plane_poiseuille_low_re(h, h, umax).abs() < 1e-12);
        assert!((ns_plane_poiseuille_low_re(0.5 * h, h, umax) - umax).abs() < 1e-12);
        println!("Low-Re Poiseuille center velocity = {:.6e}", umax);
    });
    reference_result::emit("reference.fem-ns-2d-poiseuille-low-re.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
