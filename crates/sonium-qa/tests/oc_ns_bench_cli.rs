//! CLI integration tests for `qa-oc-ns-bench` (Navier-Stokes harness).
//!
//! Fast contract tests only: flag surface and fail-closed argument
//! handling. Timed solves are covered by release-mode QA runs, not by
//! this suite.

#![cfg(feature = "qa-binaries")]

// Rust guideline compliant 2026-02-21

use std::path::PathBuf;
use std::process::Command;

fn bin() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_qa-oc-ns-bench"))
}

fn scratch(test: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!("oc_ns_bench_{test}_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

fn base(test: &str) -> (Command, PathBuf) {
    let dir = scratch(test);
    let bundle = dir.join("bundle.json");
    let mut command = Command::new(bin());
    command
        .arg("--emit-bundle")
        .arg(&bundle)
        .arg("--scratch-root")
        .arg(dir.join("scratch"))
        .env_remove("OPENCOURANT_EXE")
        .env_remove("RADISS_EXE");
    (command, bundle)
}

#[test]
fn help_lists_ns_flags() {
    let output = Command::new(bin()).arg("--help").output().unwrap();
    assert!(output.status.success());
    let text = String::from_utf8_lossy(&output.stdout);
    for flag in [
        "--deck",
        "--emit-bundle",
        "--side",
        "--threads",
        "--repeats",
        "--quick",
        "--sonium-only",
        "--engine-only",
        "--flavor",
        "--mpi-ranks",
        "--backend",
        "--backend-precond",
        "--gpu-mixed",
        "--mesh-n",
        "--tg-n",
        "--tg-dt",
        "--tg-steps",
    ] {
        assert!(text.contains(flag), "help misses {flag}");
    }
}

#[test]
fn unknown_deck_fails_closed() {
    let (mut command, _) = base("unknown_deck");
    let output = command
        .arg("--deck")
        .arg("no_such_deck")
        .arg("--quick")
        .arg("--sonium-only")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(
        stdout.contains("FAILED: qa-oc-ns-bench"),
        "stdout: {stdout}"
    );
}

#[test]
fn taylor_green_engine_arm_fails_with_guard() {
    let (mut command, _) = base("tg_guard");
    let output = command
        .arg("--deck")
        .arg("ns_taylor_green")
        .arg("--quick")
        .arg("--engine-only")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("supports ns_lid_cavity only"),
        "stderr: {stderr}"
    );
}

#[test]
fn contradictory_side_flags_fail_closed() {
    let (mut command, _) = base("sides");
    let output = command
        .arg("--quick")
        .arg("--sonium-only")
        .arg("--engine-only")
        .output()
        .unwrap();
    assert!(!output.status.success());
}

#[test]
fn quick_with_repeats_fails_closed() {
    let (mut command, _) = base("quick_repeats");
    let output = command
        .arg("--quick")
        .arg("--repeats")
        .arg("3")
        .arg("--sonium-only")
        .output()
        .unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--quick runs one repeat"),
        "stderr: {stderr}"
    );
}
