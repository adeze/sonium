#![cfg(feature = "cli")]

use std::process::Command;

#[test]
fn cli_rejects_an_incompatible_xem_convention_before_running_bem() {
    let mut config = sonium_xem::create_default_config();
    config.conventions.harmonic_time = sonium_xem::HarmonicConvention::ExpPlusIwt;

    let path = std::env::temp_dir().join(format!(
        "sonium-roomsim-invalid-conventions-{}-{}.json",
        std::process::id(),
        std::thread::current().name().unwrap_or("cli-test")
    ));
    std::fs::write(
        &path,
        serde_json::to_vec(&config).expect("serialize XEM configuration"),
    )
    .expect("write temporary XEM configuration");

    let output = Command::new(env!("CARGO_BIN_EXE_room_simulator_bem"))
        .arg("--config")
        .arg(&path)
        .output()
        .expect("run room BEM executable");
    let _ = std::fs::remove_file(&path);

    assert!(!output.status.success(), "unsupported convention must fail");
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("canonical exp(-iωt)") || stderr.contains("exp(-iωt)"),
        "unexpected CLI failure: {stderr}"
    );
}
