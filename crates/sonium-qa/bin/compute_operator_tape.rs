//! Device-resident OperatorTape contract.
//!
//! This is a small production-path fixture: values, replay, and adjoint
//! gradients remain backend-owned, and the tape budget proves that no
//! host/device transfer is reported by the callbacks.

use sonium_compute::{Backend, DeviceOperatorTape, TapeBudget, TapeMetrics, TapeNodeMetadata};
use sonium_qa::QaResult;
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};
use std::time::Instant;

const CASE_ID: &str = "compute.operator-tape.v1";
const MAX_ERROR: f64 = 1.0e-12;

fn run_backend<B: Backend>(backend_name: &str, backend: &B) -> Result<QaResult, String> {
    let started = Instant::now();
    let mut input_buffer = backend
        .raw_alloc::<f64>(2)
        .map_err(|error| format!("{backend_name} input allocation: {error}"))?;
    backend
        .try_copy_h2d(&[1.5, -2.0], &mut input_buffer)
        .map_err(|error| format!("{backend_name} input upload: {error}"))?;
    let mut tape = DeviceOperatorTape::<B, f64>::with_budget(TapeBudget {
        max_flops: u64::MAX,
        max_launches: u64::MAX,
        max_synchronisations: u64::MAX,
        max_transferred_bytes: 0,
        max_allocations: 4,
    });
    let input = tape
        .record_input(backend, &input_buffer)
        .map_err(|error| format!("{backend_name} record input: {error}"))?;

    let mut panic_tape = DeviceOperatorTape::<B, f64>::new();
    let panic_input = panic_tape
        .record_input(backend, &input_buffer)
        .map_err(|error| format!("{backend_name} panic-test input: {error}"))?;
    // The callback panic is intentionally contained by OperatorTape. Silence
    // the process-wide panic hook for this one deterministic probe so a
    // successful QA record is not polluted with a misleading panic log.
    let previous_panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let panic_result = panic_tape.record_linear(
        backend,
        panic_input,
        2,
        TapeNodeMetadata::new("qa_callback_panic", TapeMetrics::default()),
        |_backend, _input, _output| panic!("injected OperatorTape callback panic"),
        |_backend, _input, _output, _dy, _dx| Ok(()),
    );
    std::panic::set_hook(previous_panic_hook);
    let panic_error = match panic_result {
        Ok(_) => return Err(format!("{backend_name} callback panic was not contained")),
        Err(error) => error,
    };
    if !matches!(
        panic_error,
        sonium_compute::TapeError::OperationFailed {
            operation: "qa_callback_panic",
            ..
        }
    ) || panic_tape.node_count() != 0
    {
        return Err(format!(
            "{backend_name} callback panic changed tape topology or error type"
        ));
    }

    let replay_calls = Arc::new(AtomicUsize::new(0));
    let replay_calls_cb = Arc::clone(&replay_calls);
    let mut replay_tape = DeviceOperatorTape::<B, f64>::new();
    let replay_input = replay_tape
        .record_input(backend, &input_buffer)
        .map_err(|error| format!("{backend_name} replay-poison input: {error}"))?;
    let replay_output = replay_tape
        .record_linear(
            backend,
            replay_input,
            2,
            TapeNodeMetadata::new("qa_replay_poison", TapeMetrics::default()),
            move |backend, source, destination| {
                if replay_calls_cb.fetch_add(1, Ordering::SeqCst) != 0 {
                    panic!("injected replay callback failure");
                }
                backend.try_copy_d2d(source, destination)
            },
            |_backend, _input, _output, _dy, _dx| Ok(()),
        )
        .map_err(|error| format!("{backend_name} replay-poison record: {error}"))?;
    let previous_panic_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let replay_failure = replay_tape
        .replay(backend)
        .expect_err("replay failure must be reported");
    std::panic::set_hook(previous_panic_hook);
    let replay_poisoned = matches!(
        replay_failure,
        sonium_compute::TapeError::OperationFailed {
            operation: "qa_replay_poison",
            ..
        }
    ) && matches!(
        replay_tape.value(replay_output),
        Err(sonium_compute::TapeError::InvalidState { operation: "value" })
    );
    replay_tape.clear();
    if !replay_poisoned || replay_tape.value_count() != 0 {
        return Err(format!(
            "{backend_name} replay failure left tape observable or uncleared"
        ));
    }
    let output = tape
        .record_linear(
            backend,
            input,
            2,
            TapeNodeMetadata::new("scale_two", TapeMetrics::new(4, 1, 0, 0)),
            |backend, x, y| {
                backend.try_copy_d2d(x, y)?;
                backend.try_scale(2.0, y)
            },
            |backend, _x, _y, dy, dx| backend.try_axpy(2.0, dy, dx),
        )
        .map_err(|error| format!("{backend_name} record action: {error}"))?;

    tape.check_budget()
        .map_err(|error| format!("{backend_name} initial budget check: {error}"))?;
    let rejected_budget = tape
        .set_budget(TapeBudget {
            max_allocations: 1,
            ..TapeBudget {
                max_flops: u64::MAX,
                max_launches: u64::MAX,
                max_synchronisations: u64::MAX,
                max_transferred_bytes: 0,
                max_allocations: 4,
            }
        })
        .expect_err("tightened budget must reject current tape allocations");
    if !matches!(
        rejected_budget,
        sonium_compute::TapeError::BudgetExceeded {
            metric: "allocations",
            limit: 1,
            actual: 2,
        }
    ) {
        return Err(format!(
            "{backend_name} unexpected budget error: {rejected_budget}"
        ));
    }
    tape.check_budget()
        .map_err(|error| format!("{backend_name} budget was not transactional: {error}"))?;

    let mut actual = [0.0; 2];
    backend
        .try_copy_d2h(
            tape.value(output).map_err(|error| error.to_string())?,
            &mut actual,
        )
        .map_err(|error| format!("{backend_name} output download: {error}"))?;
    let max_forward_error = actual
        .iter()
        .zip([3.0, -4.0])
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);

    let mut updated = backend
        .raw_alloc::<f64>(2)
        .map_err(|error| format!("{backend_name} update allocation: {error}"))?;
    backend
        .try_copy_h2d(&[2.0, 3.0], &mut updated)
        .map_err(|error| format!("{backend_name} update upload: {error}"))?;
    tape.update_value(backend, input, &updated)
        .map_err(|error| format!("{backend_name} update value: {error}"))?;
    tape.replay(backend)
        .map_err(|error| format!("{backend_name} replay: {error}"))?;
    backend
        .try_copy_d2h(
            tape.value(output).map_err(|error| error.to_string())?,
            &mut actual,
        )
        .map_err(|error| format!("{backend_name} replay download: {error}"))?;
    let replay_error = actual
        .iter()
        .zip([4.0, 6.0])
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);

    let mut seed = backend
        .raw_alloc::<f64>(2)
        .map_err(|error| format!("{backend_name} seed allocation: {error}"))?;
    backend
        .try_copy_h2d(&[1.0, -0.5], &mut seed)
        .map_err(|error| format!("{backend_name} seed upload: {error}"))?;
    let gradients = tape
        .backward(backend, output, &seed)
        .map_err(|error| format!("{backend_name} backward: {error}"))?;
    let mut gradient = [0.0; 2];
    backend
        .try_copy_d2h(
            gradients
                .gradient(input)
                .map_err(|error| error.to_string())?
                .ok_or_else(|| format!("{backend_name} missing input gradient"))?,
            &mut gradient,
        )
        .map_err(|error| format!("{backend_name} gradient download: {error}"))?;
    let max_adjoint_error = gradient
        .iter()
        .zip([2.0, -1.0])
        .map(|(actual, expected)| (actual - expected).abs())
        .fold(0.0, f64::max);
    if !max_forward_error.is_finite()
        || !replay_error.is_finite()
        || !max_adjoint_error.is_finite()
        || max_forward_error > MAX_ERROR
        || replay_error > MAX_ERROR
        || max_adjoint_error > MAX_ERROR
    {
        return Err(format!(
            "{backend_name} tape error exceeds threshold: forward={max_forward_error:e}, replay={replay_error:e}, adjoint={max_adjoint_error:e}"
        ));
    }
    let metrics = tape.metrics();
    if metrics.transferred_bytes != 0 || metrics.allocations != 4 {
        return Err(format!(
            "{backend_name} tape resource contract violated: transfers={}, allocations={}",
            metrics.transferred_bytes, metrics.allocations
        ));
    }
    Ok(QaResult::passed(CASE_ID)
        .actual_backend(backend_name)
        .backend_passed(backend_name)
        .metric("max_forward_error", max_forward_error)
        .metric("max_replay_error", replay_error)
        .metric("max_adjoint_error", max_adjoint_error)
        .metric("transferred_bytes", metrics.transferred_bytes as f64)
        .metric("total_allocations", metrics.allocations as f64)
        .metric("budget_update_rejected", 1.0)
        .metric("callback_panic_contained", 1.0)
        .metric("replay_failure_poisoned", replay_poisoned as u8 as f64)
        .metric("tape_nodes", tape.node_count() as f64)
        .minimum("tape_nodes", 1.0)
        .minimum("replay_failure_poisoned", 1.0)
        .maximum("max_forward_error", MAX_ERROR)
        .maximum("max_replay_error", MAX_ERROR)
        .maximum("max_adjoint_error", MAX_ERROR)
        .input_checksum("analytic/deterministic-fixture:operator-tape-scale-two-v1")
        .reference("backend-owned scale-two forward/Hermitian adjoint")
        .reference_checksum("analytic")
        .dimension("2 elements, one linear node, one RHS")
        .formulation("device-resident forward/replay/backward")
        .tolerance("1e-12")
        .seed("compute-operator-tape:scale-two:v1")
        .timing_ms("tape", started.elapsed().as_millis() as u64))
}

fn required_backend() -> Result<Option<String>, String> {
    match std::env::var("SONIUM_REQUIRE_BACKEND") {
        Ok(value) if matches!(value.as_str(), "cpu" | "cuda" | "metal") => Ok(Some(value)),
        Ok(value) => Err(format!(
            "SONIUM_REQUIRE_BACKEND must be cpu, cuda, or metal; got {value:?}"
        )),
        Err(std::env::VarError::NotPresent) => Ok(None),
        Err(error) => Err(error.to_string()),
    }
}

fn run() -> Result<QaResult, String> {
    match required_backend()?.as_deref().unwrap_or("cpu") {
        "cpu" => run_backend("cpu", &sonium_compute::CpuBackend::new()),
        "cuda" => {
            #[cfg(feature = "cuda")]
            {
                let backend = sonium_compute::CudaBackend::new_default()
                    .map_err(|error| format!("required CUDA backend unavailable: {error}"))?;
                run_backend("cuda", &backend)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("required CUDA backend unavailable: build with --features cuda".into())
            }
        }
        "metal" => {
            #[cfg(feature = "metal")]
            {
                let backend = sonium_compute::MetalBackend::new()
                    .map_err(|error| format!("required Metal backend unavailable: {error}"))?;
                run_backend("metal", &backend)
            }
            #[cfg(not(feature = "metal"))]
            {
                Err("required Metal backend unavailable: build with --features metal".into())
            }
        }
        _ => unreachable!("required_backend validates values"),
    }
}

fn main() {
    match run() {
        Ok(result) => {
            println!(
                "QA_RESULT: {}",
                result.to_json_line().expect("tape QA result serializes")
            );
            println!("SUCCESS");
        }
        Err(error) => {
            println!(
                "QA_RESULT: {}",
                QaResult::failed(CASE_ID, error)
                    .to_json_line()
                    .expect("tape QA failure serializes")
            );
            println!("FAILED");
            std::process::exit(1);
        }
    }
}
