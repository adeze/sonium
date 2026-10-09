use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const SMOKE_BINS: &[&str] = &[
    "qa-fft-family-contracts",
    "qa-compute-gmres-workspace",
    "qa-compute-block-krylov-workspace",
    "qa-compute-workloads",
    "qa-bem-basic",
    "qa-fem-analytical-validation",
    "qa-solvers-iterative",
    "qa-structural-beam-balance",
    "qa-structural-beam-modal",
    "qa-structural-beam-convergence",
    "qa-structural-fem-topology-2d",
    "qa-structural-solid-topology-3d",
    "qa-structural-solid-topology-3d-robust",
    "qa-structural-external-execution-contract",
    "qa-structural-nafems-reference",
    "qa-structural-fem-topology-2d-robust",
    "qa-structural-bounded-references",
    "qa-structural-family-references",
    "qa-structural-resolved-coupling-reference",
    "qa-structural-constrained-topology-reference",
    "qa-structural-backend-capabilities",
    "qa-structural-topology-gallery",
    "qa-structural-solid-topology-gallery",
    "qa-structural-solid-topology-convergence",
    "qa-structural-hyperelastic-3d-contact",
    "qa-structural-hyperelastic-3d-force",
    "qa-structural-hyperelastic-3d-force-contact",
    "qa-structural-transient-fem-reference",
];

#[test]
#[ignore = "covered by `just qa-smoke`; runs full QA executables"]
fn qa_smoke_binaries_report_success() {
    for &name in SMOKE_BINS {
        run_qa_binary(name, Duration::from_secs(180));
    }
}

fn run_qa_binary(name: &str, timeout: Duration) {
    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("sonium-qa must live below the workspace root")
        .to_owned();
    let mut child = Command::new("cargo")
        .args([
            "run",
            "--quiet",
            "-p",
            "sonium-qa",
            "--features",
            "qa-binaries",
            "--bin",
            name,
        ])
        .current_dir(workspace)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|err| panic!("failed to spawn {name}: {err}"));

    let started = Instant::now();
    loop {
        match child.try_wait() {
            Ok(Some(_)) => break,
            Ok(None) if started.elapsed() < timeout => {
                thread::sleep(Duration::from_millis(100));
            }
            Ok(None) => {
                let _ = child.kill();
                let output = child.wait_with_output().expect("collect timed-out output");
                panic!(
                    "{name} timed out after {:.0}s\n{}",
                    timeout.as_secs_f64(),
                    format_output(&output)
                );
            }
            Err(err) => panic!("failed while waiting for {name}: {err}"),
        }
    }

    let output = child.wait_with_output().expect("collect QA output");
    let text = format_output(&output);

    assert!(
        output.status.success(),
        "{name} exited with status {}\n{text}",
        output.status
    );
    assert!(
        !text.contains("FAILED"),
        "{name} reported FAILED in output\n{text}"
    );
    assert!(
        text.contains("SUCCESS"),
        "{name} did not report SUCCESS\n{text}"
    );
}

fn format_output(output: &std::process::Output) -> String {
    let mut text = String::new();
    text.push_str(&String::from_utf8_lossy(&output.stdout));
    text.push_str(&String::from_utf8_lossy(&output.stderr));
    text
}
