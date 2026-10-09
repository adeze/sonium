mod reference {
    include!("../../sonium-bem/tests/transmission_operator.rs");
}

use sonium_qa::QaResult;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::time::Instant;

const CASE_ID: &str = "bem.typed-raw-transmission-parity.v1";

fn main() {
    let started = Instant::now();
    let result = catch_unwind(AssertUnwindSafe(
        reference::typed_transmission_adapter_matches_raw_operator_and_medium_reference_v1,
    ));

    if result.is_ok() {
        let record = QaResult::passed(CASE_ID)
            .reference("independent off-diagonal Green-kernel and k = 2πf/c fixture")
            .reference_checksum("analytic/deterministic-fixture:bem-typed-raw-transmission-v1")
            .dimension("3D closed icosphere; constant-panel two-medium transmission")
            .formulation("constant-panel two-medium Müller transmission operator")
            .tolerance("typed/raw relative matrix error <= 1e-14; independent wavenumber checks")
            .metric("checks_passed", 1.0)
            .metric("checks_total", 1.0)
            .maximum("checks_passed", 1.0)
            .elapsed_ms(started.elapsed().as_millis() as u64);
        println!("QA_RESULT: {}", record.to_json_line().unwrap());
        println!("SUCCESS");
    } else {
        let record = QaResult::failed(
            CASE_ID,
            "typed/raw transmission parity or independent wavenumber check failed",
        )
        .reference("independent off-diagonal Green-kernel and k = 2πf/c fixture")
        .reference_checksum("analytic/deterministic-fixture:bem-typed-raw-transmission-v1")
        .dimension("3D closed icosphere; constant-panel two-medium transmission")
        .formulation("constant-panel two-medium Müller transmission operator")
        .metric("checks_passed", 0.0)
        .metric("checks_total", 1.0)
        .elapsed_ms(started.elapsed().as_millis() as u64);
        println!("QA_RESULT: {}", record.to_json_line().unwrap());
        eprintln!("TYPED/RAW TRANSMISSION PARITY FAILED");
        println!("FAILED");
        std::process::exit(1);
    }
}
