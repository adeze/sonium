//! Phase-4 lane 4C G-leg: mesh build, metrics, starter acceptance.
//!
//! Builds every `rad_mesh` patch and extrusion, gates quality metrics
//! (minimum scaled Jacobian above 0.2, watertight extrusions), then runs
//! the vendor starter on the mesh-bearing decks and fails on starter
//! errors. Exit 0 = PASS, exit 1 = FAIL with a per-mesh report on stdout.
//!
//! Usage: `qa-rad-mesh-build [--metrics-only] [--keep-scratch]`.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    RadError, engine_exe, resolve_engine_pair, write_engine, write_extended_engine,
    write_extended_starter, write_starter,
};
use sonium_qa::rad_mesh::{
    MIN_SCALED_JACOBIAN, annulus_patch, crash_box_case, disk_patch, is_solid_watertight,
    measure_patch, measure_solid, rect_patch, shell_plate_case, square_ring_patch, taylor_bar_case,
    taylor_patch,
};
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let metrics_only = std::env::args().any(|arg| arg == "--metrics-only");
    let keep_scratch = std::env::args().any(|arg| arg == "--keep-scratch");
    match run(metrics_only, keep_scratch) {
        Ok(()) => {
            println!("PASS: qa-rad-mesh-build");
        }
        Err(message) => {
            eprintln!("FAIL: {message}");
            std::process::exit(1);
        }
    }
}

fn run(metrics_only: bool, keep_scratch: bool) -> Result<(), String> {
    // Patch-level metrics over every delivered builder.
    let patches = [
        ("rect", rect_patch(3, 2, 0.1, 0.2)),
        ("disk", disk_patch(6, 24, 1.0)),
        ("taylor-section", taylor_patch(6, 0.003_175)),
        ("square-ring", square_ring_patch(0.05, 0.046, 4, 1)),
        ("annulus", annulus_patch(0.025, 0.023, 1, 24)),
    ];
    for (name, patch) in &patches {
        patch
            .validate()
            .map_err(|error| format!("{name} patch invalid: {error}"))?;
        let metrics = measure_patch(patch);
        println!("mesh {name}: {metrics}");
        if metrics.quads + metrics.tris == 0 {
            return Err(format!("{name} patch holds no faces"));
        }
        if metrics.min_scaled_jacobian <= MIN_SCALED_JACOBIAN {
            return Err(format!(
                "{name} patch min scaled Jacobian {} below {MIN_SCALED_JACOBIAN}",
                metrics.min_scaled_jacobian
            ));
        }
    }
    // The disk fan converts to pure quad before extrusion.
    let disk_quad = patches[1].1.to_all_quad();
    disk_quad
        .validate()
        .map_err(|error| format!("disk all-quad invalid: {error}"))?;
    println!("mesh disk-all-quad: {}", measure_patch(&disk_quad));

    // Extrusion metrics: every extrusion must be a watertight solid.
    let extrusions: [(&str, sonium_qa::rad_mesh::ShellPatch); 5] = [
        ("rect-block", patches[0].1.clone()),
        ("disk-block", disk_quad),
        ("taylor-bar", patches[2].1.clone()),
        ("square-tube", patches[3].1.clone()),
        ("round-tube", patches[4].1.clone()),
    ];
    for (name, patch) in &extrusions {
        let mesh = patch
            .extrude(0.0, 0.1, 3)
            .map_err(|error| format!("{name} extrusion failed: {error}"))?;
        let metrics = measure_solid(&mesh);
        println!("mesh {name}: {metrics}");
        if metrics.min_scaled_jacobian <= MIN_SCALED_JACOBIAN {
            return Err(format!(
                "{name} min scaled Jacobian {} below {MIN_SCALED_JACOBIAN}",
                metrics.min_scaled_jacobian
            ));
        }
        if metrics.min_signed_volume <= 0.0 {
            return Err(format!("{name} holds an inverted brick"));
        }
        if !is_solid_watertight(&mesh) {
            return Err(format!(
                "{name} is not watertight (open edges {})",
                metrics.open_edges
            ));
        }
    }

    // Case-level metrics on the delivered RadCase builders.
    let taylor = taylor_bar_case();
    taylor
        .mesh
        .validate()
        .map_err(|error| format!("taylor case invalid: {error}"))?;
    println!("mesh mesh_taylor_bar: {}", measure_solid(&taylor.mesh));
    let crash = crash_box_case();
    crash
        .mesh
        .validate()
        .map_err(|error| format!("crash case invalid: {error}"))?;
    println!("mesh mesh_crash_box: {}", measure_solid(&crash.mesh));
    let plate = shell_plate_case();
    plate
        .validate()
        .map_err(|error| format!("plate case invalid: {error}"))?;
    println!(
        "mesh mesh_shell_plate: {} shells ({} quads, {} tris)",
        plate.shells.num_quads() + plate.shells.num_tris(),
        plate.shells.num_quads(),
        plate.shells.num_tris()
    );

    if metrics_only {
        println!("starter stage skipped (--metrics-only)");
        return Ok(());
    }
    let exe = resolve_engine().ok_or_else(|| {
        "no starter found: set RADISS_EXE or OPENCOURANT_EXE, or keep the vendor tree at ~/src_3rdparties/OpenCourant".to_string()
    })?;
    let scratch_root = scratch_parent();
    let decks = [
        (
            taylor.name.clone(),
            write_starter(&taylor),
            write_engine(&taylor),
        ),
        (
            crash.name.clone(),
            write_starter(&crash),
            write_engine(&crash),
        ),
        (
            plate.base.name.clone(),
            write_extended_starter(&plate),
            write_extended_engine(&plate),
        ),
    ];
    for (name, starter_deck, engine_deck) in &decks {
        let scratch = scratch_root.join(format!("qa_rad_mesh_{name}_{}", std::process::id()));
        if scratch.exists() {
            std::fs::remove_dir_all(&scratch).map_err(|error| format!("clean {name}: {error}"))?;
        }
        std::fs::create_dir_all(&scratch).map_err(|error| format!("scratch {name}: {error}"))?;
        std::fs::write(scratch.join(format!("{name}_0000.rad")), starter_deck)
            .map_err(|error| format!("write {name} starter: {error}"))?;
        std::fs::write(scratch.join(format!("{name}_0001.rad")), engine_deck)
            .map_err(|error| format!("write {name} engine: {error}"))?;
        let outcome = run_starter(&exe, &scratch, name);
        let keep = keep_scratch || outcome.is_err();
        match outcome {
            Ok(warnings) => {
                println!("starter {name}: accepted ({warnings} warning(s))");
            }
            Err(error) => {
                return Err(format!(
                    "starter {name} rejected: {error} (scratch kept at {})",
                    scratch.display()
                ));
            }
        }
        if !keep {
            std::fs::remove_dir_all(&scratch).ok();
        }
    }
    Ok(())
}

/// Resolve the starter/engine vendor path.
///
/// Honors `RADISS_EXE`/`OPENCOURANT_EXE` first, then the documented
/// vendor tree location.
fn resolve_engine() -> Option<PathBuf> {
    if let Some(exe) = engine_exe() {
        return Some(exe);
    }
    let home = std::env::var("HOME").ok()?;
    let exec = PathBuf::from(home).join("src_3rdparties/OpenCourant/exec");
    if exec.is_dir() { Some(exec) } else { None }
}

/// Scratch parent honoring the workspace `TMPDIR` rule.
fn scratch_parent() -> PathBuf {
    std::env::var("TMPDIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir())
}

/// Run the vendor starter on one deck; returns the warning count.
///
/// Fails when the starter exits nonzero, when its summary reports
/// errors, or when error markers appear without a summary (fail-closed).
fn run_starter(exe: &Path, dir: &Path, name: &str) -> Result<usize, String> {
    let (starter_exe, _) = resolve_engine_pair(exe).map_err(|error: RadError| error.to_string())?;
    let exec_dir = if exe.is_dir() {
        exe.to_path_buf()
    } else {
        exe.parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| exe.to_path_buf())
    };
    let mut command = Command::new(&starter_exe);
    command
        .arg("-i")
        .arg(dir.join(format!("{name}_0000.rad")))
        .current_dir(dir)
        .stdin(std::process::Stdio::null());
    apply_vendor_env(&mut command, &exec_dir);
    let output = command
        .output()
        .map_err(|error| format!("spawn starter: {error}"))?;
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    // The listing file repeats the summary; scan it too when present.
    for entry in std::fs::read_dir(dir)
        .map(|entries| entries.flatten())
        .into_iter()
        .flatten()
    {
        if entry.path().extension().is_some_and(|ext| ext == "out")
            && let Ok(text) = std::fs::read_to_string(entry.path())
        {
            combined.push('\n');
            combined.push_str(&text);
        }
    }
    std::fs::write(dir.join("starter_output.log"), &combined).ok();
    if !output.status.success() {
        let tail: String = combined
            .chars()
            .rev()
            .take(800)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        return Err(format!("exit {}: {tail}", output.status));
    }
    let (errors, warnings) = starter_summary(&combined);
    match errors {
        Some(0) => Ok(warnings.unwrap_or(0)),
        Some(count) => Err(format!("starter reports {count} error(s)")),
        None if combined.contains("ERROR") => Err("starter output holds ERROR markers".to_string()),
        None => Ok(warnings.unwrap_or(0)),
    }
}

/// Parse the `N ERROR(S)` / `N WARNING(S)` starter summary.
///
/// Returns the counts when the summary lines exist; missing lines yield
/// `None` so callers can fail closed on bare `ERROR` markers.
fn starter_summary(output: &str) -> (Option<usize>, Option<usize>) {
    fn count_before(output: &str, marker: &str) -> Option<usize> {
        for line in output.lines() {
            if let Some(at) = line.find(marker) {
                // Counts print space-separated (`0 ERROR(S)`) or glued
                // (`2WARNING(S)`); skip the gap before taking digits.
                let digits: String = line[..at]
                    .chars()
                    .rev()
                    .skip_while(|cell| cell.is_whitespace())
                    .take_while(char::is_ascii_digit)
                    .collect::<String>()
                    .chars()
                    .rev()
                    .collect();
                if let Ok(count) = digits.parse::<usize>() {
                    return Some(count);
                }
            }
        }
        None
    }
    (
        count_before(output, "ERROR(S)"),
        count_before(output, "WARNING(S)"),
    )
}

#[cfg(test)]
mod tests {
    use super::starter_summary;

    #[test]
    fn summary_parses_spaced_and_glued_counts() {
        // Observed vendor formats (OpenCourant starter, Oct 2026).
        let (errors, warnings) = starter_summary(
            "              0 ERROR(S)            \n              2 WARNING(S)          \n",
        );
        assert_eq!((errors, warnings), (Some(0), Some(2)));
        let (errors, warnings) = starter_summary("              2WARNING(S)          \n");
        assert_eq!((errors, warnings), (None, Some(2)));
        // Header lines without counts do not parse.
        let (errors, warnings) = starter_summary("|            ERROR(S) SUMMARY\n");
        assert_eq!((errors, warnings), (None, None));
    }
}

/// Apply the vendor environment to a stage command.
///
/// Mirrors `oc_golden`'s private setup: preset variables win, derived
/// paths apply only when they exist, one thread keeps runs small.
fn apply_vendor_env(command: &mut Command, exec_dir: &Path) {
    if std::env::var("OMP_STACKSIZE").is_err() {
        command.env("OMP_STACKSIZE", "400m");
    }
    command.env("OMP_NUM_THREADS", "1");
    if std::env::var("RAD_CFG_PATH").is_err() {
        let root = exec_dir.parent().unwrap_or(exec_dir);
        let candidate = root.join("hm_cfg_files");
        if candidate.is_dir() {
            command.env("RAD_CFG_PATH", &candidate);
        }
    }
    if std::env::var("RAD_H3D_PATH").is_err() {
        let root = exec_dir.parent().unwrap_or(exec_dir);
        let candidate = root.join("extlib").join("h3d").join("lib").join("linux64");
        if candidate.is_dir() {
            command.env("RAD_H3D_PATH", &candidate);
        }
    }
    if exec_dir.parent().is_some() {
        let root = exec_dir.parent().unwrap_or(exec_dir);
        let hm_reader = root.join("extlib").join("hm_reader").join("linux64");
        if hm_reader.is_dir() {
            let prepend = hm_reader.to_string_lossy().into_owned();
            let merged = std::env::var("LD_LIBRARY_PATH")
                .map(|current| format!("{prepend}:{current}"))
                .unwrap_or(prepend);
            command.env("LD_LIBRARY_PATH", merged);
        }
    }
}
