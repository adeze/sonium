//! Reference-only QA for the three-dimensional manufactured diffusion family.

use sonium_qa::QaResult;
use sonium_wave::analytical::ManufacturedDiffusion3d;
use std::f64::consts::PI;

const CASE_ID: &str = "wave.diffusion-manufactured-3d.v1";
const TOLERANCE: f64 = 5.0e-13;

fn main() {
    match run() {
        Ok(metrics) => {
            let record = QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .metric("max_pde_residual", metrics.max_pde_residual)
                .metric("max_unforced_source", metrics.max_unforced_source)
                .metric("max_dirichlet_boundary", metrics.max_dirichlet_boundary)
                .metric("sample_count", metrics.sample_count as f64)
                .maximum("max_pde_residual", TOLERANCE)
                .maximum("max_unforced_source", TOLERANCE)
                .maximum("max_dirichlet_boundary", TOLERANCE)
                .seed("manufactured-diffusion-3d:uniform-11x9x7:cpu");
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("manufactured 3-D diffusion QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("manufactured 3-D diffusion QA failure result must serialize");
            println!("QA_RESULT: {record}");
            println!("FAILED");
            std::process::exit(1);
        }
    }
}

struct Metrics {
    max_pde_residual: f64,
    max_unforced_source: f64,
    max_dirichlet_boundary: f64,
    sample_count: usize,
}

fn run() -> Result<Metrics, String> {
    let reference =
        ManufacturedDiffusion3d::try_new(0.7, [1.3, 0.9, 1.1], 0.4, 2.0, [-0.2, 0.4, 0.1])
            .map_err(|error| error.to_string())?;
    let sample = reference
        .try_sample([[-1.0, 1.0], [-0.5, 1.5], [-0.2, 0.8]], [11, 9, 7], 0.8)
        .map_err(|error| error.to_string())?;
    sample.validate().map_err(|error| error.to_string())?;
    let max_pde_residual = sample
        .time_derivatives
        .iter()
        .zip(&sample.laplacians)
        .zip(&sample.sources)
        .map(|((time_derivative, laplacian), source)| {
            (time_derivative - reference.diffusivity_m2_per_s() * laplacian - source).abs()
        })
        .fold(0.0_f64, f64::max);

    let lengths = [2.0, 3.0, 1.5];
    let spatial_wavenumbers = lengths.map(|length| PI / length);
    let diffusivity = 0.5;
    let unforced = ManufacturedDiffusion3d::try_new(
        diffusivity,
        spatial_wavenumbers,
        diffusivity
            * spatial_wavenumbers
                .into_iter()
                .map(|value| value.powi(2))
                .sum::<f64>(),
        1.5,
        [0.0; 3],
    )
    .map_err(|error| error.to_string())?;
    let unforced_sample = unforced
        .try_sample(
            [[0.0, lengths[0]], [0.0, lengths[1]], [0.0, lengths[2]]],
            [11, 9, 7],
            0.3,
        )
        .map_err(|error| error.to_string())?;
    let max_unforced_source = unforced_sample
        .sources
        .iter()
        .copied()
        .map(f64::abs)
        .fold(0.0_f64, f64::max);
    let nx = unforced_sample.x_positions_m.len();
    let ny = unforced_sample.y_positions_m.len();
    let nz = unforced_sample.z_positions_m.len();
    let max_dirichlet_boundary = unforced_sample
        .values
        .iter()
        .enumerate()
        .filter_map(|(index, value)| {
            let x = index % nx;
            let y = (index / nx) % ny;
            let z = index / (nx * ny);
            (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz)
                .then_some(value.abs())
        })
        .fold(0.0_f64, f64::max);

    for (name, value) in [
        ("PDE residual", max_pde_residual),
        ("unforced source", max_unforced_source),
        ("Dirichlet boundary", max_dirichlet_boundary),
    ] {
        if !value.is_finite() || value > TOLERANCE {
            return Err(format!("{name} failed: {value:.3e} > {TOLERANCE:.3e}"));
        }
    }
    Ok(Metrics {
        max_pde_residual,
        max_unforced_source,
        max_dirichlet_boundary,
        sample_count: sample.values.len() + unforced_sample.values.len(),
    })
}
