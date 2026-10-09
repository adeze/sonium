//! Release-facing independent CPU FEM PML/open-boundary reference.

#[allow(dead_code)]
#[path = "../../sonium-fem/tests/pml_open_boundary_reference.rs"]
mod reference;

const CASE_ID: &str = "fem.pml-open-boundary-reference.v1";

fn main() {
    let started = std::time::Instant::now();
    let result = std::panic::catch_unwind(reference::run_cpu_fem_pml_outgoing_acoustic_reference);
    match result {
        Ok(()) => {
            let record = sonium_qa::QaResult::passed(CASE_ID)
                .requested_backend("cpu")
                .actual_backend("cpu")
                .reference("analytically continued outgoing plane wave with independent attenuation integral")
                .reference_checksum("analytic/deterministic-fixture:fem-pml-open-boundary-v1")
                .formulation("independent analytically continued outgoing plane wave through physical strip and x-positive PML")
                .dimension("2D triangular strip; physical domain [0,0.5] plus 0.25 m x-PML")
                .tolerance("physical error <= 2e-3; full error <= 3e-3; reflection ratio <= 3e-3; residual <= 1e-11")
                .metric("reference_checks", 1.0)
                .minimum("reference_checks", 1.0)
                .seed("fem:pml-open-boundary:outgoing-plane-wave:cpu:v1")
                .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("PML QA result serializes")
            );
            println!("SUCCESS");
        }
        Err(_) => {
            let record = sonium_qa::QaResult::failed(
                CASE_ID,
                "independent FEM PML outgoing-wave reference panicked",
            )
            .requested_backend("cpu")
            .actual_backend("cpu")
            .elapsed_ms(started.elapsed().as_millis() as u64);
            println!(
                "QA_RESULT: {}",
                record.to_json_line().expect("PML failure serializes")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
