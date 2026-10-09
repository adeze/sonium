//! Reference-only QA for the two-dimensional manufactured Helmholtz family.

use num_complex::Complex64;
use sonium_qa::QaResult;
use sonium_wave::analytical::ManufacturedHelmholtz2d;
use std::f64::consts::PI;

const CASE_ID: &str = "wave.helmholtz-manufactured-2d.v1";
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
                .seed("manufactured-helmholtz-2d:complex-17x13:cpu");
            println!(
                "QA_RESULT: {}",
                record
                    .to_json_line()
                    .expect("manufactured 2-D Helmholtz QA result must serialize")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            let record = QaResult::failed(CASE_ID, error)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .to_json_line()
                .expect("manufactured 2-D Helmholtz QA failure result must serialize");
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
        ManufacturedHelmholtz2d::try_new(3.1, [1.3, 0.9], Complex64::new(2.0, -0.7), [-0.2, 0.4])
            .map_err(|error| error.to_string())?;
    let sample = reference
        .try_sample([-1.0, 1.0], 17, [-0.5, 1.5], 13)
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

    let (length_x, length_y) = (2.0, 3.0);
    let qx = PI / length_x;
    let qy = PI / length_y;
    let unforced = ManufacturedHelmholtz2d::try_new(
        (qx.powi(2) + qy.powi(2)).sqrt(),
        [qx, qy],
        Complex64::new(1.5, -0.4),
        [0.0, 0.0],
    )
    .map_err(|error| error.to_string())?;
    let unforced_sample = unforced
        .try_sample([0.0, length_x], 17, [0.0, length_y], 13)
        .map_err(|error| error.to_string())?;
    let max_unforced_source = unforced_sample
        .sources_pa_per_m2
        .iter()
        .map(|value| value.norm())
        .fold(0.0_f64, f64::max);
    let x_count = unforced_sample.x_positions_m.len();
    let y_count = unforced_sample.y_positions_m.len();
    let max_dirichlet_boundary = (0..y_count)
        .flat_map(|y| [y * x_count, y * x_count + x_count - 1])
        .chain(0..x_count)
        .chain((y_count - 1) * x_count..y_count * x_count)
        .map(|index| unforced_sample.pressures_pa[index].norm())
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
