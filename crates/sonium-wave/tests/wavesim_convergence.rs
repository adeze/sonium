//! WaveSim convergence test: homogeneous-medium 1D Helmholtz solve against
//! the discrete-kernel analytical reference (`analytical_1d`).
//!
//! Scaled-down port of `examples/wavesim_helmholtz_1d_analytical.rs` so the
//! Richardson iteration is covered by `cargo test`, not just by examples.

use num_complex::Complex64;
use sonium_wave::array::{Array3, crop3};
use sonium_wave::wavesim::{IterationOptions, Source, analytical_1d, point_source, simulate};

#[test]
fn richardson_1d_point_source_matches_analytical_reference() {
    let wavelength = 0.5_f64;
    let pixel_size = wavelength / 10.0;
    let sim_size = 16.0_f64;
    let nx = (sim_size / pixel_size) as usize;

    let permittivity = Array3::<Complex64>::from_elem((nx, 1, 1), Complex64::new(1.0, 0.0));
    let source = point_source(
        [sim_size / 2.0, 0.0, 0.0],
        pixel_size,
        Complex64::new(1.0, 0.0),
    );
    let sources: Vec<Source> = vec![source.clone()];
    let source_index = source.position[0];

    let result = simulate(
        &permittivity,
        &sources,
        wavelength,
        pixel_size,
        2.0,
        [false, true, true],
        IterationOptions::default(),
    );
    assert!(
        result.residual_norm < 1e-6,
        "Richardson must converge, got residual {}",
        result.residual_norm
    );

    let lo = result.boundary_pad[0].0;
    let cropped3 = crop3(&result.u, (lo, 0, 0), (nx, 1, 1));
    let cropped: Vec<Complex64> = cropped3.as_slice().to_vec();
    let x_um: Vec<f64> = (0..nx)
        .map(|i| (i as f64 - source_index as f64) * pixel_size)
        .collect();
    let u_ref = analytical_1d::analytical_solution(&x_um, wavelength);
    let re = analytical_1d::relative_error(&cropped, &u_ref);
    assert!(
        re < 1e-2,
        "relative error vs 1D analytical reference {re:.2e} exceeds 1e-2"
    );
    assert!(analytical_1d::all_close(&cropped, &u_ref, 0.15));
}
