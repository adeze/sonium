//! Analytical reference-only check; no numerical FEM accuracy claim.
//!
//! This binary validates a closed-form benchmark or helper without assembling a `sonium_fem` finite-element system.
//! Keep FEM solver and assembly validations in binaries that import `sonium_fem`.

#[path = "support/reference_result.rs"]
mod reference_result;

use sonium_qa::analytical::maxwell_rectangular_waveguide_te_cutoff;

fn main() {
    let result = std::panic::catch_unwind(|| {
        let fc = maxwell_rectangular_waveguide_te_cutoff(1, 0, 0.02286, 0.01016);
        assert!((fc - 6.557e9).abs() / 6.557e9 < 1e-3);
        println!("Maxwell rectangular TE10 cutoff: {:.6e} Hz", fc);
    });
    reference_result::emit("reference.fem-maxwell-2d-waveguide-te10.v1", result.is_ok());
    if result.is_ok() {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
