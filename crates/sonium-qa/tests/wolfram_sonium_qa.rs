use serde::Deserialize;
use sonium_qa::subprocess::WolframCommandExt;
use std::{collections::BTreeMap, env, path::PathBuf, process::Command};

const MANIFEST_CASES: &[&str] = &[
    "qa-compute-properties",
    "qa-compute-backend-parity",
    "qa-compute-workloads",
    "qa-fem-validation-matrix",
    "qa-fem-element-contracts",
    "qa-fem-analytical-validation",
    "qa-solvers-iterative",
    "qa-solvers-direct",
    "qa-solvers-robustness",
    "qa-bem-3d-sphere",
    "qa-bem-sota-validation",
    "qa-wave-1d-analytical-consistency",
    "qa-wave-cylinder-2d-analytical-consistency",
];

#[derive(Debug, Deserialize)]
struct Suite {
    schema_version: u32,
    manifest_case_count: usize,
    cases: BTreeMap<String, Scenario>,
}

#[derive(Debug, Deserialize)]
struct Scenario {
    passed: bool,
    check_count: usize,
    checks: BTreeMap<String, bool>,
}

fn script_path() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("wolfram/sonium_qa.wls")
}

fn run_wolfram() -> Option<Suite> {
    let executable = env::var_os("WOLFRAMSCRIPT")?;
    let mut command = Command::new(&executable);
    command.arg("-file").arg(script_path());
    #[cfg(target_os = "macos")]
    if env::var_os("WolframKernel").is_none() {
        let kernel = PathBuf::from(
            "/Applications/Wolfram Engine.app/Contents/Resources/Wolfram Player.app/Contents/MacOS/WolframKernel",
        );
        if kernel.is_file() {
            command.env("WolframKernel", kernel);
        }
    }
    let output = match command.wolfram_output() {
        Ok(output) => output,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            eprintln!("Wolfram Engine not available; skipping optional sonium-qa reproductions");
            return None;
        }
        Err(error) => panic!("failed to launch {executable:?}: {error}"),
    };
    assert!(
        output.status.success(),
        "Wolfram sonium-qa script failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let json = stdout
        .lines()
        .rev()
        .find(|line| line.trim_start().starts_with('{'))
        .unwrap_or(&stdout);
    Some(serde_json::from_str(json).unwrap_or_else(|error| {
        panic!(
            "Wolfram sonium-qa output was not JSON: {error}\nstdout:\n{stdout}\nstderr:\n{}",
            String::from_utf8_lossy(&output.stderr)
        )
    }))
}

#[test]
fn wolfram_reproduces_canonical_sonium_qa_manifest() {
    let Some(suite) = run_wolfram() else { return };
    assert_eq!(suite.schema_version, 1);
    assert_eq!(suite.manifest_case_count, MANIFEST_CASES.len());
    assert_eq!(suite.cases.len(), MANIFEST_CASES.len());
    for expected in MANIFEST_CASES {
        let scenario = suite
            .cases
            .get(*expected)
            .unwrap_or_else(|| panic!("Wolfram suite omitted {expected}"));
        assert_eq!(scenario.check_count, scenario.checks.len(), "{expected}");
        assert!(scenario.check_count > 0, "{expected} has no checks");
        let failed = scenario
            .checks
            .iter()
            .filter_map(|(name, &passed)| (!passed).then_some(name.as_str()))
            .collect::<Vec<_>>();
        assert!(
            scenario.passed,
            "{expected} failed Wolfram checks: {failed:?}"
        );
    }
}

#[test]
fn wolfram_suite_tracks_the_manifest_commands() {
    let manifest = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../sonium-qa/validation-manifest.toml"),
    )
    .expect("read sonium-qa validation manifest");
    let script = std::fs::read_to_string(script_path()).expect("read Wolfram sonium-qa script");
    for command in MANIFEST_CASES {
        assert!(
            manifest.contains(&format!("bin = \"{command}\"")),
            "manifest omitted {command}"
        );
        assert_eq!(
            script.matches(&format!("\"{command}\" ->")).count(),
            1,
            "Wolfram script must define {command} once"
        );
    }
}
