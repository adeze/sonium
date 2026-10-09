use std::{env, process::Command};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=RUSTC");
    let compiler = env::var_os("RUSTC").expect("Cargo must provide RUSTC");
    let output = Command::new(compiler)
        .arg("--version")
        .output()
        .expect("could not query the build compiler");
    assert!(output.status.success(), "build compiler --version failed");
    let version = String::from_utf8(output.stdout).expect("compiler version is not UTF-8");
    let version = version.trim();
    assert!(
        version.starts_with("rustc ") && !version.contains(['\n', '\r']),
        "unexpected compiler version output"
    );
    println!("cargo:rustc-env=SONIUM_BUILD_RUSTC_VERSION={version}");
}
