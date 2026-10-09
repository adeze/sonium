//! Reference-only QA for the three-dimensional manufactured Helmholtz family.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_wave::analytical::ManufacturedHelmholtz3d;
use std::f64::consts::PI;

const CASE_ID: &str = "wave.helmholtz-manufactured-3d.v1";
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
                .seed("manufactured-helmholtz-3d:complex-11x9x7:cpu");
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("manufactured 3-D Helmholtz QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("manufactured 3-D Helmholtz QA failure result must serialize");
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
    let reference = ManufacturedHelmholtz3d::try_new(
        3.4,
        [1.3, 0.9, 1.1],
        Complex64::new(2.0, -0.7),
        [-0.2, 0.4, 0.1],
    )
    .map_err(|error| error.to_string())?;
    let sample = reference
        .try_sample([-1.0, 1.0], 11, [-0.5, 1.5], 9, [-0.2, 0.8], 7)
        .map_err(|error| error.to_string())?;
    sample.validate().map_err(|error| error.to_string())?;
    let max_pde_residual = sample
        .laplacians_pa_per_m2
        .iter()
        .zip(&sample.pressures_pa)
        .zip(&sample.sources_pa_per_m2)
        .map(|((laplacian, pressure), source)| {
            (laplacian + pressure * reference.equation_wavenumber_per_m().powi(2) - source).norm()
        })
        .fold(0.0_f64, f64::max);

    let lengths = [2.0, 3.0, 1.5];
    let spatial_wavenumbers = lengths.map(|length| PI / length);
    let unforced = ManufacturedHelmholtz3d::try_new(
        spatial_wavenumbers
            .into_iter()
            .map(|value| value.powi(2))
            .sum::<f64>()
            .sqrt(),
        spatial_wavenumbers,
        Complex64::new(1.5, -0.4),
        [0.0; 3],
    )
    .map_err(|error| error.to_string())?;
    let unforced_sample = unforced
        .try_sample(
            [0.0, lengths[0]],
            11,
            [0.0, lengths[1]],
            9,
            [0.0, lengths[2]],
            7,
        )
        .map_err(|error| error.to_string())?;
    let max_unforced_source = unforced_sample
        .sources_pa_per_m2
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    let nx = unforced_sample.x_positions_m.len();
    let ny = unforced_sample.y_positions_m.len();
    let nz = unforced_sample.z_positions_m.len();
    let max_dirichlet_boundary = unforced_sample
        .pressures_pa
        .iter()
        .enumerate()
        .filter_map(|(index, pressure)| {
            let x = index % nx;
            let y = (index / nx) % ny;
            let z = index / (nx * ny);
            (x == 0 || x + 1 == nx || y == 0 || y + 1 == ny || z == 0 || z + 1 == nz)
                .then_some(pressure.norm())
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
        sample_count: sample.pressures_pa.len() + unforced_sample.pressures_pa.len(),
    })
}
