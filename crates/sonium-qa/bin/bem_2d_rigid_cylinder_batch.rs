//! Bounded CPU multi-RHS rigid-cylinder solve contract.

use num_complex::Complex64;
use sonium_bem::{TwoDPlaneWave, TwoDRigidCylinder};
use sonium_qa::QaResult;
use sonium_xem::{
    AcousticMedium, ConventionRegistry, DensityKgM3, FrequencyHz, LengthM, SpeedOfSoundMPerS,
};
use std::f64::consts::PI;
use std::time::Instant;

const CASE_ID: &str = "bem.2d-rigid-cylinder-batch.v1";

fn main() {
    let started = Instant::now();
    let outcome = std::panic::catch_unwind(|| {
        let mut conventions = ConventionRegistry::default();
        conventions.rms_amplitudes = false;
        let medium = AcousticMedium::try_new_with_conventions(
            FrequencyHz::try_new(2.0 * 343.0 / (2.0 * PI)).expect("frequency"),
            SpeedOfSoundMPerS::try_new(343.0).expect("speed"),
            DensityKgM3::try_new(1.21).expect("density"),
            conventions,
        )
        .expect("medium");
        let problem =
            TwoDRigidCylinder::try_from_xem(LengthM::try_new(0.4).expect("radius"), medium, 64)
                .expect("rigid cylinder");
        let incidents = [
            TwoDPlaneWave::plane_wave_x(),
            TwoDPlaneWave::try_new([0.0, 1.0], Complex64::new(0.6, -0.1)).expect("second incident"),
        ];
        let batch = problem
            .solve_many(&incidents, 1.0e-10)
            .expect("batch solve");
        let mut max_error: f64 = 0.0;
        let mut max_residual: f64 = 0.0;
        for (incident, batch_solution) in incidents.iter().zip(&batch) {
            let single = problem.solve(*incident, 1.0e-10).expect("single solve");
            max_error = max_error.max(
                batch_solution
                    .density()
                    .iter()
                    .zip(single.density())
                    .map(|(batch, single)| (*batch - *single).norm())
                    .fold(0.0, f64::max),
            );
            max_residual = max_residual.max(batch_solution.relative_residual);
        }
        assert!(max_error < 1.0e-11);
        assert!(max_residual <= 1.0e-10);
        assert!(
            problem
                .solve_many(&[], 1.0e-10)
                .expect("empty batch")
                .is_empty()
        );
        (max_error, max_residual)
    });
    let elapsed_ms = started.elapsed().as_millis() as u64;
    let passed = outcome.is_ok();
    let result = match outcome {
        Ok((max_error, max_residual)) => QaResult::passed(CASE_ID)
            .requested_backend("cpu")
            .actual_backend("cpu")
            .metric("rhs_count", 2.0)
            .metric("max_batch_density_error", max_error)
            .maximum("max_batch_density_error", 1.0e-11)
            .metric("max_batch_residual", max_residual)
            .maximum("max_batch_residual", 1.0e-10)
            .metric("empty_batch_contract", 1.0)
            .minimum("empty_batch_contract", 1.0)
            .input_checksum("analytic/deterministic-fixture:bem-2d-rigid-cylinder-batch-v1")
            .reference("independent single-RHS solves using the same assembled Neumann operator")
            .reference_checksum("analytic/deterministic-fixture:bem-2d-rigid-cylinder-batch-v1")
            .dimension("2-D rigid cylinder; 64 constant panels; two incident RHS columns")
            .formulation("shared-LU multi-RHS constant-panel Neumann solve")
            .tolerance("batch/single coefficient error <= 1e-11; residual <= 1e-10")
            .seed("bem-2d-rigid-cylinder-batch:cpu:v1")
            .elapsed_ms(elapsed_ms),
        Err(payload) => QaResult::failed(CASE_ID, format!("2-D rigid batch panicked: {payload:?}"))
            .elapsed_ms(elapsed_ms),
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
