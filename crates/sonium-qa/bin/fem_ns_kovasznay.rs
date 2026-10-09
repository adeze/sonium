//! Analytical reference check.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

use sonium_qa::analytical::ns_kovasznay_solution;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let (u0, v0, _) = ns_kovasznay_solution(0.0, 0.0, 40.0);
        let (u1, _, _) = ns_kovasznay_solution(1.0, 0.0, 40.0);
        assert!(u0.abs() < 1e-12);
        assert!(v0.abs() < 1e-12);
        assert!(u1 > u0);
        println!("Kovasznay downstream velocity = {:.6e}", u1);
    });
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
