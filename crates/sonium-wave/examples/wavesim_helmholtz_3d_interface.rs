//! 3D Helmholtz with a refractive-index interface.
//!
//! Inspired by `wavesim_py/examples/helmholtz_3d_create_source.py` (point-source
//! variant). A 3D box has refractive index `n = 1` in the upper half along the
//! y-axis and `n = 2` in the lower half. A unit point source at the centre
//! radiates spherical waves that refract/reflect at the interface.
//!
//! Validates that the 3D FFT propagator path converges and produces a non-zero
//! field with the expected qualitative asymmetry across the y-interface.

use std::time::Instant;

use num_complex::Complex64;
use sonium_wave::array::{Array3, crop3};
use sonium_wave::wavesim::{IterationOptions, Source, point_source, simulate};

fn main() {
    let wavelength = 1.0_f64;
    let pixel_size = wavelength / 5.0; // 0.2 µm — coarse but adequate for n_max=2

    let sim_um = [4.0_f64, 4.0, 4.0]; // small box keeps it fast
    let nx = (sim_um[0] / pixel_size) as usize;
    let ny = (sim_um[1] / pixel_size) as usize;
    let nz = (sim_um[2] / pixel_size) as usize;
    println!("3D grid {nx} × {ny} × {nz}, λ = {wavelength} µm, Δx = {pixel_size} µm");

    let bg = Complex64::new(1.0, 0.0);
    let lower_eps = Complex64::new(4.0, 0.0); // n = 2
    let mut permittivity = Array3::<Complex64>::from_elem((nx, ny, nz), bg);
    for i in 0..nx {
        for j in (ny / 2)..ny {
            for k in 0..nz {
                permittivity[[i, j, k]] = lower_eps;
            }
        }
    }

    // Source one pixel above the interface so it sits cleanly in the n=1 region.
    // (Putting it exactly at sim_um[1]/2 lands on j=ny/2, which is the first
    // cell of the n=2 slab — the asymmetry test then depends on an arbitrary
    // tiebreak rather than on physics.)
    let source_y_um = sim_um[1] / 2.0 - pixel_size;
    let source = point_source(
        [sim_um[0] / 2.0, source_y_um, sim_um[2] / 2.0],
        pixel_size,
        Complex64::new(1.0, 0.0),
    );
    let sources: Vec<Source> = vec![source.clone()];
    debug_assert!(
        source.position[1] < ny / 2,
        "source must sit in the n=1 half"
    );

    println!("Running simulation…");
    let start = Instant::now();
    let result = simulate(
        &permittivity,
        &sources,
        wavelength,
        pixel_size,
        2.0,
        [false, false, false],
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

    // Crop padded boundaries away.
    let (lx, ly, lz) = (
        result.boundary_pad[0].0,
        result.boundary_pad[1].0,
        result.boundary_pad[2].0,
    );
    let cropped = crop3(&result.u, (lx, ly, lz), (nx, ny, nz));

    // Sample magnitude at the source, then symmetric points in the n=1 (up)
    // and n=2 (down) media at equal pixel distance from the source.
    let cx = nx / 2;
    let cz = nz / 2;
    let source_j = source.position[1];
    let probe_dist = (ny / 2 - source_j).max(2);
    let amp_at = |j: usize| cropped[[cx, j, cz]].norm();
    let amp_src = amp_at(source_j);
    let amp_up = amp_at(source_j.saturating_sub(probe_dist));
    let amp_dn = amp_at((source_j + probe_dist).min(ny - 1));

    let probe_um = probe_dist as f64 * pixel_size;
    println!("|u| at source                                 = {amp_src:.3e}");
    println!("|u| {probe_um:.2} µm into n=1 (free) side  = {amp_up:.3e}");
    println!("|u| {probe_um:.2} µm into n=2 (dense) side = {amp_dn:.3e}");

    // Slab through the source plane: report max |u| per row of y.
    println!("Max |u| along y (x=cx, z=cz):");
    for j in 0..ny {
        let v = cropped[[cx, j, cz]].norm();
        let bar_len = ((v / amp_src.max(1e-30)) * 40.0).clamp(0.0, 40.0) as usize;
        println!(
            "  j={j:3} y={:5.2}µm |u|={v:.2e} {}",
            (j as f64 + 0.5) * pixel_size,
            "#".repeat(bar_len)
        );
    }

    assert!(
        result.residual_norm < 1.0e-6,
        "Richardson failed to converge: residual = {:.2e}",
        result.residual_norm
    );
    assert!(amp_src > 0.0, "field is zero at the source");
    assert!(
        amp_up > 0.0 && amp_dn > 0.0,
        "field decays to zero off-source"
    );

    // The two halves carry different optical impedances, so amplitudes
    // measured at the same geometric distance from the source must differ.
    let asymmetry = (amp_up - amp_dn).abs() / amp_up.max(amp_dn);
    println!("interface asymmetry = {asymmetry:.3}");
    assert!(
        asymmetry > 1e-3,
        "expected asymmetry across interface, got {asymmetry:.3e}"
    );
    println!("PASS");
}
