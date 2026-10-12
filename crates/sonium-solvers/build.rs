use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_accelerate)");
    println!("cargo:rustc-check-cfg=cfg(has_superlu)");
    println!("cargo:rerun-if-changed=native/direct_solver_perf_superlu.c");

    if env::var_os("CARGO_FEATURE_NATIVE_DIRECT_PERF").is_none() {
        return;
    }

    if env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("macos") {
        println!("cargo:rustc-cfg=has_accelerate");
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }

    configure_superlu();
}

fn configure_superlu() {
    let pkg_config = env::var("PKG_CONFIG").unwrap_or_else(|_| {
        ["/opt/homebrew/bin/pkg-config", "/usr/local/bin/pkg-config"]
            .into_iter()
            .find(|path| std::path::Path::new(path).is_file())
            .unwrap_or("pkg-config")
            .into()
    });
    let flags = match Command::new(&pkg_config)
        .args(["--cflags", "--libs", "superlu"])
        .output()
    {
        Ok(output) if output.status.success() => {
            String::from_utf8(output.stdout).expect("pkg-config output must be UTF-8")
        }
        _ => {
            println!(
                "cargo:warning=SuperLU not found through pkg-config; perf/PERF.md will mark it unavailable"
            );
            return;
        }
    };

    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("Cargo sets OUT_DIR"));
    let object = out_dir.join("direct_solver_perf_superlu.o");
    let archive = out_dir.join("libdirect_solver_perf_superlu.a");
    let source = PathBuf::from("native/direct_solver_perf_superlu.c");
    let compiler = env::var("CC").unwrap_or_else(|_| "cc".into());

    let mut compile = Command::new(compiler);
    compile.args(["-O3", "-std=c11", "-c"]);
    for flag in flags
        .split_whitespace()
        .filter(|flag| flag.starts_with("-I"))
    {
        compile.arg(flag);
    }
    let status = compile
        .arg(&source)
        .arg("-o")
        .arg(&object)
        .status()
        .expect("failed to execute C compiler for SuperLU benchmark shim");
    if !status.success() {
        println!(
            "cargo:warning=SuperLU benchmark shim failed to compile; perf/PERF.md will mark it unavailable"
        );
        return;
    }

    let archiver = env::var("AR").unwrap_or_else(|_| "ar".into());
    let status = Command::new(archiver)
        .args(["crus"])
        .arg(&archive)
        .arg(&object)
        .status()
        .expect("failed to execute archiver for SuperLU benchmark shim");
    if !status.success() {
        panic!("failed to archive SuperLU benchmark shim");
    }

    println!("cargo:rustc-link-search=native={}", out_dir.display());
    println!("cargo:rustc-link-lib=static=direct_solver_perf_superlu");
    for flag in flags.split_whitespace() {
        if let Some(path) = flag.strip_prefix("-L") {
            println!("cargo:rustc-link-search=native={path}");
        } else if let Some(library) = flag.strip_prefix("-l") {
            println!("cargo:rustc-link-lib={library}");
        }
    }
    println!("cargo:rustc-cfg=has_superlu");

    if let Ok(output) = Command::new(pkg_config)
        .args(["--modversion", "superlu"])
        .output()
        && output.status.success()
    {
        let version = String::from_utf8_lossy(&output.stdout);
        println!("cargo:rustc-env=SONIUM_SUPERLU_VERSION={}", version.trim());
    }
}
