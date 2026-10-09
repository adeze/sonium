use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for key in [
        "RUSTC",
        "PROFILE",
        "OPT_LEVEL",
        "CARGO_CFG_TARGET_FEATURE",
        "CARGO_ENCODED_RUSTFLAGS",
    ] {
        println!("cargo:rerun-if-env-changed={key}");
    }
    let output = Command::new(env::var_os("RUSTC").expect("Cargo RUSTC"))
        .arg("--version")
        .output()
        .expect("query build compiler");
    assert!(output.status.success(), "compiler version query failed");
    let version = String::from_utf8(output.stdout).expect("UTF-8 compiler version");
    println!("cargo:rustc-env=SONIUM_BENCH_RUSTC={}", version.trim());
    println!(
        "cargo:rustc-env=SONIUM_BENCH_PROFILE={};opt-level={};rustflags={}",
        env::var("PROFILE").expect("Cargo PROFILE"),
        env::var("OPT_LEVEL").expect("Cargo OPT_LEVEL"),
        env::var("CARGO_ENCODED_RUSTFLAGS")
            .unwrap_or_default()
            .replace('\u{1f}', " ")
    );
    let features = env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_default();
    println!(
        "cargo:rustc-env=SONIUM_BENCH_TARGET_FEATURES={}",
        if features.is_empty() {
            "baseline"
        } else {
            &features
        }
    );
}
