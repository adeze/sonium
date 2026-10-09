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
    // Best-effort source revision for benchmark artifact provenance. These
    // describe the checkout the binary was built from, not a verified
    // hermetic source digest: rebuild the binary after committing for a
    // release artifact, and treat "unknown" as missing provenance rather
    // than a clean tree.
    let commit = Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_owned())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=SONIUM_GIT_COMMIT={commit}");
    let dirty = Command::new("git")
        .args(["status", "--porcelain"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| {
            if String::from_utf8_lossy(&output.stdout).trim().is_empty() {
                "false".to_owned()
            } else {
                "true".to_owned()
            }
        })
        .unwrap_or_else(|| "unknown".to_owned());
    println!("cargo:rustc-env=SONIUM_GIT_DIRTY={dirty}");
}
