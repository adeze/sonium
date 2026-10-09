//! WaveSim 1D analytical test — port of `wavesim_py/examples/helmholtz_1d_analytical.py`.

use std::time::Instant;

use num_complex::Complex64;
use sonium_wave::array::{Array3, crop3};
use sonium_wave::wavesim::{IterationOptions, Source, analytical_1d, point_source, simulate};

fn main() {
    let wavelength = 0.5_f64;
    let pixel_size = wavelength / 10.0;
    let sim_size = 128.0_f64;
    let nx = (sim_size / pixel_size) as usize; // 2560

    let permittivity = Array3::<Complex64>::from_elem((nx, 1, 1), Complex64::new(1.0, 0.0));

    let source = point_source(
        [sim_size / 2.0, 0.0, 0.0],
        pixel_size,
        Complex64::new(1.0, 0.0),
    );
    let sources: Vec<Source> = vec![source.clone()];
    let source_index = source.position[0];

    println!("Running simulation: nx={nx}, λ={wavelength} µm, Δx={pixel_size} µm");

    let start = Instant::now();
    let result = simulate(
        &permittivity,
        &sources,
        wavelength,
        pixel_size,
        5.0,
        [false, true, true],
        IterationOptions::default(),
    );
    let elapsed = start.elapsed();

    println!(
        "Time {:.2} s; iterations {}; per-iter {:.4} s",
        elapsed.as_secs_f64(),
        result.iterations,
        elapsed.as_secs_f64() / result.iterations.max(1) as f64
    );
    println!("Residual norm {:.2e}", result.residual_norm);

    let lo = result.boundary_pad[0].0;
    // Crop padding then collapse the singleton (ny=1, nz=1) axes to a 1D field.
    let cropped3 = crop3(&result.u, (lo, 0, 0), (nx, 1, 1));
    let cropped: Vec<Complex64> = cropped3.as_slice().to_vec();

    let x_um: Vec<f64> = (0..nx)
        .map(|i| (i as f64 - source_index as f64) * pixel_size)
        .collect();
    let u_ref = analytical_1d::analytical_solution(&x_um, wavelength);
    let re = analytical_1d::relative_error(&cropped, &u_ref);
    println!("Relative error vs analytical: {re:.2e}");

    println!("Sample comparison (real, imag):");
    for &i in &[
        source_index.saturating_sub(200),
        source_index.saturating_sub(50),
        source_index,
        source_index + 50,
        source_index + 200,
    ] {
        if i < nx {
            println!(
                "  x_idx={i:5}  computed=({:+.4e}, {:+.4e})  ref=({:+.4e}, {:+.4e})",
                cropped[i].re, cropped[i].im, u_ref[i].re, u_ref[i].im
            );
        }
    }

    let threshold = 1.0e-3;
    assert!(
        re < threshold,
        "Relative error {re:.2e} exceeds threshold {threshold:.2e}"
    );
    assert!(
        analytical_1d::all_close(&cropped, &u_ref, 4e-2),
        "all_close(rtol=4e-2) failed"
    );
    println!("PASS: relative error below {threshold:.2e}");
}
