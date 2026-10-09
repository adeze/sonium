//! Bounded CPU multi-RHS transmission solve contract for the 2-D fluid cylinder.

use num_complex::Complex64;
use sonium_bem::{TwoDFluidCylinder, TwoDPlaneWave};
use sonium_qa::QaResult;
use sonium_xem::{
    AcousticMedium, ConventionRegistry, DensityKgM3, FrequencyHz, LengthM, SpeedOfSoundMPerS,
};
use std::f64::consts::PI;
use std::time::Instant;

const CASE_ID: &str = "bem.2d-fluid-cylinder-batch.v1";
const RADIUS: f64 = 0.4;
const EXTERIOR_SPEED: f64 = 343.0;
const INTERIOR_SPEED: f64 = 228.666_666_666_666_66;
const FREQUENCY: f64 = 2.0 * EXTERIOR_SPEED / (2.0 * PI);

fn medium(speed: f64, density: f64) -> AcousticMedium {
    let mut conventions = ConventionRegistry::default();
    conventions.rms_amplitudes = false;
    AcousticMedium::try_new_with_conventions(
        FrequencyHz::try_new(FREQUENCY).expect("finite frequency"),
        SpeedOfSoundMPerS::try_new(speed).expect("finite speed"),
        DensityKgM3::try_new(density).expect("finite density"),
        conventions,
    )
    .expect("valid medium")
}

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        let problem = TwoDFluidCylinder::try_from_xem(
            LengthM::try_new(RADIUS).expect("finite radius"),
            medium(EXTERIOR_SPEED, 1.21),
            medium(INTERIOR_SPEED, 1.5),
            64,
        )
        .expect("valid fluid cylinder");
        let incidents = [
            TwoDPlaneWave::plane_wave_x(),
            TwoDPlaneWave::try_new([0.0, 1.0], Complex64::new(0.7, -0.2))
                .expect("valid second incident"),
        ];
        let batch = problem
            .solve_many(&incidents, 1.0e-10)
            .expect("batch solve");
        assert_eq!(batch.len(), incidents.len());
        let mut max_density_error: f64 = 0.0;
        let mut max_residual: f64 = 0.0;
        for (incident, batch_solution) in incidents.iter().zip(&batch) {
            let single = problem.solve(*incident, 1.0e-10).expect("single solve");
            max_density_error = max_density_error.max(
                batch_solution
                    .exterior_density()
                    .iter()
                    .zip(single.exterior_density())
                    .chain(
                        batch_solution
                            .interior_density()
                            .iter()
                            .zip(single.interior_density()),
                    )
                    .map(|(batch, single)| (*batch - *single).norm())
                    .fold(0.0, f64::max),
            );
            max_residual = max_residual.max(batch_solution.relative_residual);
        }
        assert!(max_density_error < 1.0e-11);
        assert!(max_residual <= 1.0e-10);
        assert!(
            problem
                .solve_many(&[], 1.0e-10)
                .expect("empty batch")
                .is_empty()
        );
        (max_density_error, max_residual)
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok((max_density_error, max_residual)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("rhs_count", 2.0)
            .metric("max_batch_density_error", max_density_error)
            .maximum("max_batch_density_error", 1.0e-11)
            .metric("max_batch_residual", max_residual)
            .maximum("max_batch_residual", 1.0e-10)
            .metric("empty_batch_contract", 1.0)
            .minimum("empty_batch_contract", 1.0)
            .input_checksum("analytic/deterministic-fixture:bem-2d-fluid-cylinder-batch-v1")
            .reference("independent single-RHS solves using the same assembled operator")
            .reference_checksum("analytic/deterministic-fixture:bem-2d-fluid-cylinder-batch-v1")
            .dimension("2-D penetrable cylinder; 64 constant panels; two incident RHS columns")
            .formulation("shared-LU multi-RHS constant-panel Müller transmission solve")
            .tolerance("batch/single coefficient error <= 1e-11; residual <= 1e-10")
            .seed("bem-2d-fluid-cylinder-batch:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => {
            QaResult::failed(CASE_ID, format!("2-D batch contract panicked: {payload:?}"))
                .elapsed_ms(elapsed_ms)
        }
    };
    println!(
        "QA_RESULT: {}",
        result.to_json_line().expect("serialize QA result")
    );
    if passed {
        println!("SUCCESS");
    } else {
        println!("FAILED");
        std::process::exit(1);
    }
}
