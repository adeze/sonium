use crate::results::PlatformInfo;
use std::process::Command;

/// Detect the current platform, optionally overriding the name.
pub fn detect_platform(name_override: Option<&str>) -> PlatformInfo {
    let name = match name_override {
        Some(n) => n.to_string(),
        None => {
            let host = hostname();
            format!(
                "{}-{}-{}",
                host,
                std::env::consts::OS,
                std::env::consts::ARCH
            )
        }
    };

    let total_cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    PlatformInfo {
        name,
        os: std::env::consts::OS.to_string(),
        arch: std::env::consts::ARCH.to_string(),
        cpu_model: detect_cpu_model(),
        total_cores,
    }
}

fn detect_cpu_model() -> String {
    match std::env::consts::OS {
        "macos" => {
            let output = Command::new("sysctl")
                .args(["-n", "machdep.cpu.brand_string"])
                .output();
            match output {
                Ok(o) if o.status.success() => {
                    String::from_utf8_lossy(&o.stdout).trim().to_string()
                }
                _ => "unknown".to_string(),
            }
        }
        "linux" => {
            let content = std::fs::read_to_string("/proc/cpuinfo").unwrap_or_default();
            for line in content.lines() {
                if line.starts_with("model name")
                    && let Some((_key, value)) = line.split_once(':')
                {
                    return value.trim().to_string();
                }
            }
            "unknown".to_string()
        }
        _ => "unknown".to_string(),
    }
}

/// Detect the current short git SHA, or None if not in a git repo.
pub fn detect_git_sha() -> Option<String> {
    let output = Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output()
        .ok()?;
    if output.status.success() {
        Some(String::from_utf8_lossy(&output.stdout).trim().to_string())
    } else {
        None
    }
}

fn hostname() -> String {
    let output = Command::new("hostname").output();
    match output {
        Ok(o) if o.status.success() => String::from_utf8_lossy(&o.stdout).trim().to_string(),
        _ => "unknown".to_string(),
    }
}
