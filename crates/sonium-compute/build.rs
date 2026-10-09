fn main() {
    println!("cargo:rerun-if-changed=build.rs");

    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_arch == "aarch64" && matches!(target_os.as_str(), "macos" | "linux") {
        println!("cargo:rerun-if-changed=src/sme2/sme2_kernels.S");

        cc::Build::new()
            .file("src/sme2/sme2_kernels.S")
            .flag("-march=armv9-a+sme2+sme-f64f64")
            .compile("sme2_kernels");
    }
}
