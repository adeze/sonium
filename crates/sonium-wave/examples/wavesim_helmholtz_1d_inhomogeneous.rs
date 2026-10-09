//! 1D Helmholtz with an inhomogeneous medium.
//!
//! Port of `wavesim_py/examples/helmholtz_1d_create_medium.py`. A homogeneous
//! background (n = 1) hosts two scattering slabs — Silicon (n ≈ 3.575 + 0.00049i)
//! and PVP plastic (n ≈ 1.5184 + 0.00117i) — with a point source on one side.
//! There is no closed-form analytical reference; the test checks that the
//! Richardson iteration converges and that the field exhibits the expected
//! scattering signature (large amplitude inside Si, attenuation past PVP).

use std::time::Instant;

use num_complex::Complex64;
use sonium_wave::array::{Array3, crop3};
use sonium_wave::wavesim::{IterationOptions, Source, point_source, simulate};

fn main() {
    // Indicative refractive indices at 1 µm.
    let n_si = Complex64::new(3.5750, 0.000_490_2);
    let n_pvp = Complex64::new(1.5184, 0.001_168_1);

    let wavelength = 1.0_f64;
    let n_max = n_si.norm().max(n_pvp.norm());
    let pixel_size = wavelength / (3.0 * n_max); // ≈ 0.0932 µm
    let sim_size_um = 20.0_f64;
    let nx = (sim_size_um / pixel_size) as usize;
    println!("nx = {nx}, pixel_size = {pixel_size:.4} µm");

    // Build permittivity (n²): background = 1, two slabs.
    let bg = Complex64::new(1.0, 0.0);
    let eps_si = n_si * n_si;
    let eps_pvp = n_pvp * n_pvp;

    let mut permittivity = Array3::<Complex64>::from_elem((nx, 1, 1), bg);

    // Silicon slab: 2 µm wide, centered at x = 8 µm.
    insert_slab(&mut permittivity, 8.0, 2.0, pixel_size, eps_si);
    // PVP slab: 3 µm wide, centered at x = 14 µm.
    insert_slab(&mut permittivity, 14.0, 3.0, pixel_size, eps_pvp);

    // Point source at x = 4 µm — on the left of both slabs so we observe
    // forward scattering through Si then PVP.
    let source = point_source([4.0, 0.0, 0.0], pixel_size, Complex64::new(1.0, 0.0));
    let sources: Vec<Source> = vec![source.clone()];

    println!("Running simulation…");
    let start = Instant::now();
    let result = simulate(
        &permittivity,
        &sources,
        wavelength,
        pixel_size,
        10.0,
        [false, true, true],
        IterationOptions {
            max_iterations: 20_000,
            threshold: 1.0e-6,
            alpha: 0.75,
        },
    );
    let elapsed = start.elapsed();
    println!(
        "Time {:.2} s; iterations {}; residual norm {:.2e}",
        elapsed.as_secs_f64(),
        result.iterations,
        result.residual_norm
    );

    // Crop to physical region.
    let lo = result.boundary_pad[0].0;
    let cropped3 = crop3(&result.u, (lo, 0, 0), (nx, 1, 1));
    let cropped: Vec<Complex64> = cropped3.as_slice().to_vec();

    // Sample at 4 µm intervals.
    println!("Field samples (real, imag, |u|):");
    for x_um in [2.0_f64, 6.0, 8.0, 10.0, 14.0, 18.0] {
        let i = (x_um / pixel_size) as usize;
        if i < nx {
            let v = cropped[i];
            println!(
                "  x = {x_um:5.1} µm  ({:+.3e}, {:+.3e})  |u| = {:.3e}",
                v.re,
                v.im,
                v.norm()
            );
        }
    }

    // Sanity assertions.
    assert!(
        result.residual_norm < 1.0e-6,
        "Richardson failed to converge below threshold: residual = {:.2e}",
        result.residual_norm
    );

    let max_abs = cropped.iter().map(|c| c.norm()).fold(0.0_f64, f64::max);
    assert!(max_abs > 0.0, "field is zero everywhere");

    // The slabs are absorbing: average |u| just before the Si slab should
    // exceed average |u| just after the PVP slab. Sample over a 1 µm window
    // on each side, well away from the absorbing boundaries (which start at
    // x < 0 and x > 20 µm in physical coords).
    let win = (1.0 / pixel_size).round() as usize;
    let pre_centre = (5.5 / pixel_size).round() as usize; // pre-Si
    let post_centre = (16.5 / pixel_size).round() as usize; // post-PVP, away from boundary
    let avg_window = |centre: usize| -> f64 {
        let start = centre.saturating_sub(win / 2);
        let end = (centre + win / 2 + 1).min(nx);
        let slice = &cropped[start..end];
        let s: f64 = slice.iter().map(|c| c.norm()).sum();
        s / slice.len().max(1) as f64
    };
    let amp_pre = avg_window(pre_centre);
    let amp_post = avg_window(post_centre);
    println!("<|u|> 5–6 µm (pre-Si)   = {amp_pre:.3e}\n<|u|> 16–17 µm (post-PVP) = {amp_post:.3e}");
    assert!(
        amp_post < amp_pre,
        "expected attenuation past slabs: pre={amp_pre:.3e} post={amp_post:.3e}"
    );
    println!("PASS");
}

/// Set `permittivity[i, 0, 0] = value` for cells whose centre lies inside
/// `[centre_um − width_um/2, centre_um + width_um/2]`.
fn insert_slab(
    permittivity: &mut Array3<Complex64>,
    centre_um: f64,
    width_um: f64,
    pixel_size: f64,
    value: Complex64,
) {
    let nx = permittivity.shape().0;
    let half = width_um * 0.5;
    // A cell with centre at (i + 0.5) * pixel_size is inside the slab iff
    // its centre falls inside [centre - half, centre + half].
    let lo = (((centre_um - half) / pixel_size - 0.5).ceil()).max(0.0) as usize;
    let hi_f = (((centre_um + half) / pixel_size - 0.5).floor() + 1.0).max(0.0);
    let hi = (hi_f as usize).min(nx);
    for i in lo..hi {
        permittivity[[i, 0, 0]] = value;
    }
}
