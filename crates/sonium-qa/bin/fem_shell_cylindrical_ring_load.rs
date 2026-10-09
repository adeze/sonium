//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::shell_cylindrical_ring_load_decay_length;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let ell = shell_cylindrical_ring_load_decay_length(0.2, 0.002);
        assert!((ell - 0.02).abs() < 1e-14);
        println!("Shell cylindrical ring-load decay length = {:.6e}", ell);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
