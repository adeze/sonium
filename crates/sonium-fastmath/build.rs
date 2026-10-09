use std::process::Command;

fn main() {
    let version = Command::new("rustc")
        .arg("--version")
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=SONIUM_RUSTC_VERSION={version}");
    let target_features =
        std::env::var("CARGO_CFG_TARGET_FEATURE").unwrap_or_else(|_| "unknown".to_owned());
    println!("cargo:rustc-env=SONIUM_TARGET_FEATURES={target_features}");
}
