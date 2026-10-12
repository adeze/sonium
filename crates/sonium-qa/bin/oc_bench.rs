//! OpenCourant-vs-sonium explicit-dynamics comparison runner.
//!
//! Lane 6A implementation of the frozen `qa-oc-bench` surface from
//! `PHASE6_CONTRACTS.md`. The engine arm stages decks to scratch and
//! times vendor starter+engine runs; the sonium arm times the native
//! explicit path on the same C3 decks. Both arms check probes and emit
//! one frozen `sonium.oc-bench/v1` bundle.
//!
//! Deck families: an id matching a case under the crate `decks/`
//! dir selects a C3 deck (both arms); anything else resolves as a
//! miniqa case id through `find_deck` (engine arm only; the sonium
//! arm fails closed with the reason). Same deck is the bar: both
//! arms consume the identical starter/engine deck pair, and the
//! sonium arm documents every deliberate modeling difference in its
//! equivalence basis.
//!
//! ```sh
//! qa-oc-bench --deck taylor_bar --emit-bundle bundle.json --side both
//! qa-oc-bench --deck taylor_bar --emit-bundle ci.json --quick --sonium-only
//! ```

// Rust guideline compliant 2026-02-21

use clap::{Parser, ValueEnum};
use sonium_bench::oc_compare::{
    BUNDLE_SCHEMA, BenchBundle, BenchRow, HardwareInfo, PROTOCOL_CORRECTNESS, ProtocolInfo, Side,
    bundle_json, render_bundle_markdown,
};
use sonium_compute::explicit_dt::critical_dt;
use sonium_compute::explicit_state::ExplicitState;
use sonium_fem::physics::solid::contact::explicit::{
    RigidWall, rigid_wall_contact, rigid_wall_project, suggested_penalty,
};
use sonium_fem::physics::solid::fracture::ductile::{ErosionCriterion, ErosionSet};
use sonium_fem::physics::solid::nonlinear_solid::{
    ExplicitPlasticMaterial, HexExplicitGeom, HexExplicitParams, HexExplicitState, HexMesh,
    NeoHookean, hex_jacobian, hex_plastic_forces, hex_quadrature, hex_shape_derivatives,
    q1_element_residual,
};
use sonium_fem::physics::solid::plasticity::law_metals::{
    Law2, Law2Flow, Law2State, Law36, Law36State, MaxRateEffect, RateCurve,
};
use sonium_qa::rad::oc_golden::{
    CompareOptions, MINIQA_ENV, compare_cycle, compare_history, find_deck, golden_path,
    load_golden, parse_engine_out, parse_th_csv, resolve_scratch_root,
};
use sonium_qa::rad::{TimeHistory, assert_probe_close, engine_exe, resolve_engine_pair};
use sonium_qa::rad_parse::{CardKind, DeckCard, ElemType, LoadCard, RadDeck, TypedCard};
use sonium_solvers::time_integration::{AdaptiveDtConfig, ExplicitStepDriver};
use std::cell::RefCell;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Default deck: the smallest C3 explicit case, suited for CI.
const DEFAULT_DECK: &str = "taylor_bar";
/// Penalty factor for the plastic arm (Taylor-plastic leg value: the
/// stiff LAW1 factor rings undamped on mushrooming/eroding impact).
const PLASTIC_PENALTY_FACTOR: f64 = 0.5;
/// Overdamped contact dashpot ratio for the plastic arm (leg value);
/// settles the penalty-contact bounce mode, footprint printed.
const PLASTIC_CONTACT_DAMPING: f64 = 3.0;
/// Native explicit step budget. Covers the rail crush run (about 2.6M
/// estimated steps); the pre-run estimate rejects slower decks instantly
/// instead of burning this budget on a doomed run.
const MAX_NATIVE_STEPS: u64 = 8_000_000;
/// Gap below which a wall starts in contact (sub-nanometer).
const TOUCHING_GAP_M: f64 = 1e-9;
/// Cap on sampled wall-force points per run; stride grows to fit.
const MAX_FORCE_SAMPLES: usize = 100_000;
/// Gate on the native energy ledger, mirroring the Taylor driver.
const DRIFT_TOL: f64 = 0.03;
/// Vendor OpenMP stack default, mirroring the harvest precedent.
const OMP_STACKSIZE: &str = "400m";

/// Which side(s) to time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum SideOpt {
    /// Native sonium explicit run only.
    Sonium,
    /// Vendor engine run only.
    Engine,
    /// Both arms on the same deck.
    Both,
}

/// Vendor engine launch mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum FlavorOpt {
    /// One process, one thread, no `-nt`.
    Serial,
    /// One process, OpenMP threads via `-nt` and `OMP_NUM_THREADS`.
    Smp,
    /// MPI ranks via `mpirun` and the `_ompi` engine binary.
    Mpi,
}

#[derive(Parser, Debug)]
#[command(
    name = "qa-oc-bench",
    about = "Time sonium-vs-engine explicit dynamics and emit a comparison bundle."
)]
struct Args {
    /// Deck id: C3 case (`taylor_bar`) or miniqa path (`LOIS/...`).
    /// Defaults to the smallest C3 explicit deck.
    #[arg(long)]
    deck: Option<String>,
    /// Path for the emitted `sonium.oc-bench/v1` bundle JSON.
    #[arg(long)]
    emit_bundle: PathBuf,
    /// Which side(s) to time.
    #[arg(long, value_enum, default_value_t = SideOpt::Both)]
    side: SideOpt,
    /// Worker threads (SMP threads; MPI ranks when `--flavor mpi`
    /// runs without `--mpi-ranks`).
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Timed repeats per row (median wall time wins).
    #[arg(long)]
    repeats: Option<u32>,
    /// CI shortcut: one repeat, no warmup (cycle counts stay
    /// deck-dictated; combine with the default smallest deck).
    #[arg(long)]
    quick: bool,
    /// Run the sonium arm only (CI has no vendor engine).
    #[arg(long)]
    sonium_only: bool,
    /// Run the engine arm only.
    #[arg(long)]
    engine_only: bool,
    /// Engine launch mode; defaults to serial on one thread, else SMP.
    #[arg(long, value_enum)]
    flavor: Option<FlavorOpt>,
    /// MPI ranks for `--flavor mpi`; defaults to `--threads`.
    #[arg(long)]
    mpi_ranks: Option<usize>,
    /// Engine/starter exe or exec dir; else `RADISS_EXE`/`OPENCOURANT_EXE`.
    #[arg(long)]
    exe: Option<PathBuf>,
    /// miniqa root (`.../qa-tests/miniqa`); else `OPENCOURANT_MINIQA`.
    #[arg(long)]
    miniqa: Option<PathBuf>,
    /// Goldens root; defaults to the crate `goldens/` dir.
    #[arg(long)]
    goldens: Option<PathBuf>,
    /// Scratch root; defaults to `SONIUM_QA_SCRATCH`, else `TMPDIR`.
    #[arg(long)]
    scratch_root: Option<PathBuf>,
    /// Keep per-repeat scratch dirs after the run.
    #[arg(long)]
    keep_scratch: bool,
}

/// Resolved side selection: (run sonium, run engine).
fn resolve_sides(args: &Args) -> Result<(bool, bool), String> {
    if args.sonium_only && args.engine_only {
        return Err("--sonium-only and --engine-only contradict".to_string());
    }
    if args.sonium_only && args.side != SideOpt::Both && args.side != SideOpt::Sonium {
        return Err("--sonium-only contradicts --side engine".to_string());
    }
    if args.engine_only && args.side != SideOpt::Both && args.side != SideOpt::Engine {
        return Err("--engine-only contradicts --side sonium".to_string());
    }
    if args.sonium_only {
        return Ok((true, false));
    }
    if args.engine_only {
        return Ok((false, true));
    }
    match args.side {
        SideOpt::Sonium => Ok((true, false)),
        SideOpt::Engine => Ok((false, true)),
        SideOpt::Both => Ok((true, true)),
    }
}

/// Resolved repeat counts: (timed repeats, warmup runs).
fn resolve_repeats(args: &Args) -> Result<(u32, u32), String> {
    // Three timed repeats plus one warmup is the default protocol.
    let repeats = args.repeats.unwrap_or(3);
    if args.quick && args.repeats.is_some_and(|repeats| repeats != 1) {
        return Err("--quick runs one repeat; drop --repeats".to_string());
    }
    let repeats = if args.quick { 1 } else { repeats };
    if repeats == 0 {
        return Err("--repeats must be positive".to_string());
    }
    // One timed repeat leaves no warmup to discard; otherwise one
    // warmup run absorbs cold caches and first-touch effects.
    let warmup = u32::from(repeats > 1);
    Ok((repeats, warmup))
}

/// One engine-verified probe expectation from `probes.json`.
#[derive(Debug, Clone)]
struct Probe {
    label: String,
    channel: String,
    expected: f64,
    abs_tol: f64,
    rel_tol: f64,
}

/// Reads probe expectations from a C3 `probes.json` file.
fn read_probe_file(path: &Path) -> Result<Vec<Probe>, String> {
    let text = std::fs::read_to_string(path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let value: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| format!("cannot parse {}: {error}", path.display()))?;
    let probes = value
        .get("probes")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| format!("{} has no probes array", path.display()))?;
    if probes.is_empty() {
        return Err(format!("{} holds no probes", path.display()));
    }
    let mut out = Vec::with_capacity(probes.len());
    for probe in probes {
        let get = |key: &str| {
            probe
                .get(key)
                .ok_or_else(|| format!("probe is missing {key}"))
        };
        out.push(Probe {
            label: get("label")?
                .as_str()
                .ok_or_else(|| "probe label is not a string".to_string())?
                .to_string(),
            channel: get("channel")?
                .as_str()
                .ok_or_else(|| "probe channel is not a string".to_string())?
                .to_string(),
            expected: get("expected")?
                .as_f64()
                .ok_or_else(|| "probe expected is not a number".to_string())?,
            abs_tol: get("abs_tol")?
                .as_f64()
                .ok_or_else(|| "probe abs_tol is not a number".to_string())?,
            rel_tol: get("rel_tol")?
                .as_f64()
                .ok_or_else(|| "probe rel_tol is not a number".to_string())?,
        });
    }
    Ok(out)
}

/// One timed repeat: wall time plus the correctness verdict.
#[derive(Debug, Clone)]
struct RepeatSample {
    wall_s: f64,
    cycles: u64,
    dof: u64,
    peak_rss_bytes: u64,
    correct: bool,
}

/// Aggregates repeats: median wall/cycles, worst RSS, AND verdict.
///
/// Sorting by wall time and taking the middle repeat resists one slow
/// outlier; RSS takes the maximum so the footprint never under-reports.
fn aggregate_repeats(samples: &[RepeatSample]) -> Result<RepeatSample, String> {
    if samples.is_empty() {
        return Err("no timed repeats to aggregate".to_string());
    }
    let mut ordered = samples.to_vec();
    ordered.sort_by(|a, b| a.wall_s.total_cmp(&b.wall_s));
    let median = ordered[ordered.len() / 2].clone();
    let peak_rss_bytes = samples
        .iter()
        .map(|sample| sample.peak_rss_bytes)
        .max()
        .unwrap_or(0);
    let correct = samples.iter().all(|sample| sample.correct);
    Ok(RepeatSample {
        wall_s: median.wall_s,
        cycles: median.cycles,
        dof: median.dof,
        peak_rss_bytes,
        correct,
    })
}

/// Piecewise-linear load curve with clamped ends.
fn eval_funct(points: &[[f64; 2]], time: f64) -> Result<f64, String> {
    if points.is_empty() {
        return Err("load curve holds no points".to_string());
    }
    if time <= points[0][0] {
        return Ok(points[0][1]);
    }
    for pair in points.windows(2) {
        let (x0, y0) = (pair[0][0], pair[0][1]);
        let (x1, y1) = (pair[1][0], pair[1][1]);
        if time <= x1 {
            if x1 <= x0 {
                return Ok(y1);
            }
            let weight = (time - x0) / (x1 - x0);
            return Ok(y0 * (1.0 - weight) + y1 * weight);
        }
    }
    Ok(points[points.len() - 1][1])
}

/// Reads the vendor rootname from an engine deck `/RUN/<name>/1` line.
fn run_rootname(engine_deck_text: &str) -> Result<String, String> {
    for line in engine_deck_text.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("/RUN/") {
            let name = rest.split('/').next().unwrap_or_default().trim();
            if !name.is_empty() {
                return Ok(name.to_string());
            }
        }
    }
    Err("engine deck holds no /RUN/<name>/ line".to_string())
}

/// True when the starter deck carries an SI (kg/m/s) unit line.
fn has_si_units(starter_deck_text: &str) -> bool {
    starter_deck_text
        .lines()
        .any(|line| line.split_whitespace().collect::<Vec<_>>() == ["kg", "m", "s"])
}

/// Peak RSS of this process in bytes via `getrusage`.
fn self_peak_rss_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return 0;
    }
    maxrss_to_bytes(usage.ru_maxrss)
}

/// Converts `ru_maxrss` to bytes (kilobytes on Linux, bytes on macOS).
fn maxrss_to_bytes(maxrss: libc::c_long) -> u64 {
    let units: u64 = maxrss.max(0) as u64;
    #[cfg(target_os = "macos")]
    {
        units
    }
    #[cfg(not(target_os = "macos"))]
    {
        units.saturating_mul(1024)
    }
}

/// Copies a directory tree (vendor `data/` into scratch).
fn copy_dir_all(from: &Path, to: &Path) -> Result<(), String> {
    std::fs::create_dir_all(to)
        .map_err(|error| format!("cannot create {}: {error}", to.display()))?;
    let mut entries: Vec<std::fs::DirEntry> = std::fs::read_dir(from)
        .map_err(|error| format!("cannot list {}: {error}", from.display()))?
        .flatten()
        .collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let target = to.join(entry.file_name());
        if entry
            .file_type()
            .map_err(|error| format!("cannot stat {}: {error}", entry.path().display()))?
            .is_dir()
        {
            copy_dir_all(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)
                .map_err(|error| format!("cannot copy {}: {error}", entry.path().display()))?;
        }
    }
    Ok(())
}

/// Output of one timed child process.
struct TimedOutput {
    success: bool,
    code: Option<i32>,
    combined_tail: String,
    wall_s: f64,
    peak_rss_bytes: u64,
}

/// Runs a child, timing wall seconds and reaping peak RSS via `wait4`.
///
/// Stdout/stderr drain on reader threads while the parent blocks in
/// `wait4`, so large vendor logs never deadlock the pipes. `ru_maxrss`
/// is that child's own peak, unlike the cumulative `RUSAGE_CHILDREN`.
fn run_timed(
    exe: &Path,
    args: &[String],
    dir: &Path,
    env: &[(String, String)],
) -> Result<TimedOutput, String> {
    let mut command = Command::new(exe);
    command
        .args(args)
        .current_dir(dir)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    for (key, value) in env {
        command.env(key, value);
    }
    let mut child = command
        .spawn()
        .map_err(|error| format!("cannot spawn {}: {error}", exe.display()))?;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| "child stdout unavailable".to_string())?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| "child stderr unavailable".to_string())?;
    let stdout_thread = std::thread::spawn(move || {
        use std::io::Read;
        let mut pipe = stdout;
        let mut text = String::new();
        let _ = pipe.read_to_string(&mut text);
        text
    });
    let stderr_thread = std::thread::spawn(move || {
        use std::io::Read;
        let mut pipe = stderr;
        let mut text = String::new();
        let _ = pipe.read_to_string(&mut text);
        text
    });
    let start = std::time::Instant::now();
    let pid = child.id() as libc::pid_t;
    let mut status: libc::c_int = 0;
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    let waited = loop {
        let ret = unsafe { libc::wait4(pid, &mut status, 0, &mut usage) };
        if ret < 0 {
            let errno = std::io::Error::last_os_error();
            if errno.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            let _ = child.kill();
            return Err(format!("wait4 for {} failed ({errno})", exe.display()));
        }
        break ret;
    };
    let wall_s = start.elapsed().as_secs_f64();
    let stdout_text = stdout_thread.join().unwrap_or_default();
    let stderr_text = stderr_thread.join().unwrap_or_default();
    debug_assert!(waited > 0);
    let success = libc::WIFEXITED(status) && libc::WEXITSTATUS(status) == 0;
    let code = if libc::WIFEXITED(status) {
        Some(libc::WEXITSTATUS(status))
    } else {
        None
    };
    let mut combined = stdout_text;
    combined.push_str(&stderr_text);
    let combined_tail: String = combined
        .chars()
        .rev()
        .take(2000)
        .collect::<String>()
        .chars()
        .rev()
        .collect();
    Ok(TimedOutput {
        success,
        code,
        combined_tail,
        wall_s,
        peak_rss_bytes: maxrss_to_bytes(usage.ru_maxrss),
    })
}

/// Resolved engine launch: mode plus the thread/rank count on the row.
struct EngineLaunch {
    flavor: FlavorOpt,
    label: String,
    row_threads: usize,
    omp_threads: usize,
    mpi_ranks: usize,
}

/// Resolves the engine launch mode and row labels from CLI flags.
fn resolve_launch(args: &Args, engine_exe: &Path) -> Result<EngineLaunch, String> {
    if args.threads == 0 {
        return Err("--threads must be positive".to_string());
    }
    let flavor = args.flavor.unwrap_or(if args.threads > 1 {
        FlavorOpt::Smp
    } else {
        FlavorOpt::Serial
    });
    let mpi_ranks = args.mpi_ranks.unwrap_or(args.threads);
    if flavor == FlavorOpt::Mpi && mpi_ranks == 0 {
        return Err("--mpi-ranks must be positive".to_string());
    }
    if flavor == FlavorOpt::Serial && args.threads != 1 {
        return Err("--flavor serial runs one thread; pass --threads 1".to_string());
    }
    let mut label = match flavor {
        FlavorOpt::Serial => "serial".to_string(),
        FlavorOpt::Smp => "smp".to_string(),
        FlavorOpt::Mpi => "mpi".to_string(),
    };
    if engine_exe.to_string_lossy().to_lowercase().contains("_sp") {
        label.push_str("_sp");
    }
    // MPI rows count ranks; ranks own the parallelism, so each rank
    // runs single-threaded until hybrid mode is validated. Serial
    // always runs (and records) one thread per the guard above.
    let (row_threads, omp_threads) = match flavor {
        FlavorOpt::Mpi => (mpi_ranks, 1),
        FlavorOpt::Serial => (1, 1),
        FlavorOpt::Smp => (args.threads, args.threads),
    };
    Ok(EngineLaunch {
        flavor,
        label,
        row_threads,
        omp_threads,
        mpi_ranks,
    })
}

/// Picks the `_ompi` engine binary beside a resolved pair.
fn select_mpi_engine(exe: &Path) -> Result<PathBuf, String> {
    let dir = if exe.is_dir() {
        exe.to_path_buf()
    } else {
        exe.parent()
            .map(Path::to_path_buf)
            .ok_or_else(|| format!("--exe {} has no parent dir for MPI lookup", exe.display()))?
    };
    let mut candidates: Vec<(u32, PathBuf)> = Vec::new();
    let entries = std::fs::read_dir(&dir)
        .map_err(|error| format!("cannot list {}: {error}", dir.display()))?;
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let lower = name.to_lowercase();
        if !lower.starts_with("engine") || !(lower.contains("ompi") || lower.contains("_mpi")) {
            continue;
        }
        // Double precision first; single precision only when the user
        // points --exe at an _sp binary directly is handled by label.
        let rank = u32::from(lower.contains("_sp"));
        candidates.push((rank, entry.path()));
    }
    candidates.sort();
    candidates
        .into_iter()
        .next()
        .map(|(_, path)| path)
        .ok_or_else(|| format!("no engine_*ompi* binary beside {}", exe.display()))
}

/// Vendor environment for one stage, mirroring the harvest precedent.
fn vendor_env(exec_dir: &Path, omp_threads: usize) -> Vec<(String, String)> {
    let mut env = Vec::new();
    if std::env::var("OMP_STACKSIZE").is_err() {
        env.push(("OMP_STACKSIZE".to_string(), OMP_STACKSIZE.to_string()));
    }
    env.push(("OMP_NUM_THREADS".to_string(), omp_threads.to_string()));
    let root = exec_dir.parent().unwrap_or(exec_dir);
    let cfg = root.join("hm_cfg_files");
    if std::env::var("RAD_CFG_PATH").is_err() && cfg.is_dir() {
        env.push((
            "RAD_CFG_PATH".to_string(),
            cfg.to_string_lossy().into_owned(),
        ));
    }
    let h3d = root.join("extlib").join("h3d").join("lib").join("linux64");
    if std::env::var("RAD_H3D_PATH").is_err() && h3d.is_dir() {
        env.push((
            "RAD_H3D_PATH".to_string(),
            h3d.to_string_lossy().into_owned(),
        ));
    }
    let reader = root.join("extlib").join("hm_reader").join("linux64");
    if reader.is_dir() {
        let prepend = reader.to_string_lossy().into_owned();
        let merged = std::env::var("LD_LIBRARY_PATH")
            .map(|current| format!("{prepend}:{current}"))
            .unwrap_or(prepend);
        env.push(("LD_LIBRARY_PATH".to_string(), merged));
    }
    env
}

/// Engine argv for one deck under the resolved launch mode.
fn engine_argv(launch: &EngineLaunch, engine_exe: &Path, deck: &str) -> (PathBuf, Vec<String>) {
    match launch.flavor {
        FlavorOpt::Mpi => (
            PathBuf::from("mpirun"),
            vec![
                "-np".to_string(),
                launch.mpi_ranks.to_string(),
                engine_exe.to_string_lossy().into_owned(),
                "-i".to_string(),
                deck.to_string(),
            ],
        ),
        FlavorOpt::Smp if launch.row_threads > 1 => (
            engine_exe.to_path_buf(),
            vec![
                "-i".to_string(),
                deck.to_string(),
                "-nt".to_string(),
                launch.row_threads.to_string(),
            ],
        ),
        _ => (
            engine_exe.to_path_buf(),
            vec!["-i".to_string(), deck.to_string()],
        ),
    }
}

/// Starter argv: `-np` only carries domain counts for MPI launches and
/// miniqa decks that declare them.
fn starter_argv(deck: &str, np: Option<usize>, extra: &[String]) -> Vec<String> {
    let mut args = vec!["-i".to_string(), deck.to_string()];
    if let Some(np) = np {
        args.push("-np".to_string());
        args.push(np.to_string());
    }
    args.extend(extra.iter().cloned());
    args
}

/// Locates the `th_to_csv` converter beside the engine or on `PATH`.
fn find_th_converter(engine_exe: &Path) -> Result<PathBuf, String> {
    let mut candidates = Vec::new();
    let dir = if engine_exe.is_dir() {
        Some(engine_exe.to_path_buf())
    } else {
        engine_exe.parent().map(Path::to_path_buf)
    };
    if let Some(dir) = dir {
        let plain = dir.join("th_to_csv");
        if plain.is_file() {
            return Ok(plain);
        }
        if let Ok(entries) = std::fs::read_dir(&dir) {
            let mut suffixed: Vec<PathBuf> = entries
                .flatten()
                .map(|entry| entry.path())
                .filter(|path| {
                    path.file_name()
                        .and_then(|name| name.to_str())
                        .is_some_and(|name| name.starts_with("th_to_csv") && name != "th_to_csv")
                })
                .collect();
            suffixed.sort();
            if let Some(tool) = suffixed.into_iter().next() {
                return Ok(tool);
            }
        }
    }
    candidates.push(PathBuf::from("th_to_csv"));
    candidates
        .into_iter()
        .next()
        .ok_or_else(|| "th_to_csv converter missing".to_string())
}

/// True when a TH file already holds comma-separated text.
fn is_probably_csv(path: &Path) -> Result<bool, String> {
    let bytes =
        std::fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let head = &bytes[..bytes.len().min(4096)];
    Ok(!head.contains(&0) && head.contains(&b','))
}

/// Converts a binary T01 with `th_to_csv`, writing the sibling CSV.
fn convert_th(engine_exe: &Path, t01: &Path, dir: &Path) -> Result<(), String> {
    let tool = find_th_converter(engine_exe)?;
    // Vendor quirk: th_to_csv dies with SIGFPE on long absolute input
    // paths (seen past ~130 chars), so pass the bare file name with
    // the scratch dir as the working directory.
    let arg = t01
        .file_name()
        .ok_or_else(|| format!("{} has no file name", t01.display()))?;
    let output = Command::new(&tool)
        .arg(arg)
        .current_dir(dir)
        .stdin(std::process::Stdio::null())
        .output()
        .map_err(|error| format!("cannot spawn {}: {error}", tool.display()))?;
    if !output.status.success() {
        #[cfg(unix)]
        let signal = std::os::unix::process::ExitStatusExt::signal(&output.status);
        #[cfg(not(unix))]
        let signal: Option<i32> = None;
        return Err(format!(
            "{} failed on {} (code {:?}, signal {signal:?})",
            tool.display(),
            t01.display(),
            output.status.code()
        ));
    }
    Ok(())
}

/// Counts structural dofs from a starter deck: three translational
/// dofs per element-referenced node.
///
/// The starter `.out` carries no model node total (its `NUMBER OF
/// NODES` lines echo wall slave groups), so both arms parse the same
/// staged deck instead, which also keeps C3 dofs exactly equal.
/// Shell rotations and rigid-body condensation are not modeled; the
/// subset decks are brick models where the count is exact.
fn deck_dof(starter_deck: &Path) -> Result<u64, String> {
    let deck = RadDeck::parse_file(starter_deck).map_err(|error| error.to_string())?;
    let mut referenced = std::collections::BTreeSet::new();
    for card in &deck.cards {
        match card.typed.as_ref() {
            Some(TypedCard::Elems(block)) => {
                for elem in &block.elems {
                    referenced.extend(elem.nodes.iter().copied());
                }
            }
            Some(TypedCard::Sphcel(sph)) => {
                referenced.extend(sph.nodes.iter().copied());
            }
            _ => {}
        }
    }
    if referenced.is_empty() {
        return Err(format!(
            "{} references no structural nodes",
            starter_deck.display()
        ));
    }
    u64::try_from(referenced.len())
        .ok()
        .and_then(|nodes| nodes.checked_mul(3))
        .ok_or_else(|| "node count overflows dof".to_string())
}

/// Sums `TOTAL NUMBER OF CYCLES` over engine `.out` files.
///
/// The last printed cycle line goes stale under `/PRINT` strides (a
/// 24-cycle run with stride 1000 prints only cycle 0), while the run
/// summary carries the exact completed count, so the summary wins.
fn total_cycles(outs: &[PathBuf]) -> Result<u64, String> {
    let mut total = 0u64;
    for out in outs {
        let text = std::fs::read_to_string(out)
            .map_err(|error| format!("cannot read {}: {error}", out.display()))?;
        let mut found = None;
        for line in text.lines() {
            if line.contains("TOTAL NUMBER OF CYCLES") {
                found = line
                    .split_whitespace()
                    .next_back()
                    .and_then(|token| token.parse::<u64>().ok());
            }
        }
        let count = found
            .ok_or_else(|| format!("{} holds no TOTAL NUMBER OF CYCLES line", out.display()))?;
        total = total
            .checked_add(count)
            .ok_or_else(|| "cycle count overflows".to_string())?;
    }
    if total == 0 {
        return Err("engine completed zero cycles".to_string());
    }
    Ok(total)
}

/// First existing path among rootname-case variants.
fn first_existing(dir: &Path, names: &[String]) -> Result<PathBuf, String> {
    for name in names {
        let path = dir.join(name);
        if path.is_file() {
            return Ok(path);
        }
    }
    Err(format!("none of {names:?} exists in {}", dir.display()))
}

/// Candidate `.out` names: verbatim rootname, then uppercased vendor style.
fn out_candidates(rootname: &str, run: usize) -> Vec<String> {
    vec![
        format!("{rootname}_{run:04}.out"),
        format!("{}_{run:04}.out", rootname.to_uppercase()),
    ]
}

/// Candidate TH names: verbatim rootname, then uppercased vendor style.
fn th_candidates(rootname: &str, run: usize) -> Vec<String> {
    vec![
        format!("{rootname}T{run:02}"),
        format!("{}T{run:02}", rootname.to_uppercase()),
    ]
}

/// Shared engine-arm context for one deck.
struct EngineCtx {
    starter_exe: PathBuf,
    engine_exe: PathBuf,
    exec_dir: PathBuf,
    launch: EngineLaunch,
    scratch_root: PathBuf,
    keep_scratch: bool,
}

/// Runs one vendor stage, logging combined output beside the decks.
fn run_vendor_stage(
    ctx: &EngineCtx,
    exe: &Path,
    args: &[String],
    dir: &Path,
    stage: &str,
    timed: bool,
) -> Result<TimedOutput, String> {
    let env = vendor_env(&ctx.exec_dir, ctx.launch.omp_threads);
    let output = run_timed(exe, args, dir, &env)?;
    let log = if stage == "starter" {
        "starter_output.log"
    } else {
        "engine_output.log"
    };
    let _ = std::fs::write(dir.join(log), &output.combined_tail);
    if !output.success {
        return Err(format!(
            "{stage} failed (code {:?}): {}",
            output.code, output.combined_tail
        ));
    }
    if timed && output.wall_s <= 0.0 {
        return Err(format!("{stage} measured non-positive wall time"));
    }
    Ok(output)
}

/// Fresh per-repeat scratch dir for one arm and repeat index.
fn repeat_scratch(root: &Path, deck: &str, side: &str, index: usize) -> PathBuf {
    let safe = deck.replace(['/', '\\'], "_");
    root.join(format!(
        "oc_bench_{safe}_{side}_{}_{index}",
        std::process::id()
    ))
}

/// Cleans a scratch dir unless the run keeps forensics.
fn maybe_clean(dir: &Path, keep: bool) {
    if !keep {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// C3 deck file set for one case.
#[derive(Debug)]
struct C3Decks {
    starter: PathBuf,
    engine: PathBuf,
    probes: PathBuf,
}

/// Resolves a C3 case dir, failing closed on missing artifacts.
fn resolve_c3_decks(deck: &str) -> Result<C3Decks, String> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("decks")
        .join(deck);
    let starter = dir.join(format!("{deck}_0000.rad"));
    let engine = dir.join(format!("{deck}_0001.rad"));
    let probes = dir.join("probes.json");
    for path in [&starter, &engine, &probes] {
        if !path.is_file() {
            return Err(format!("C3 deck {deck} misses {}", path.display()));
        }
    }
    Ok(C3Decks {
        starter,
        engine,
        probes,
    })
}

/// Parses one converted TH file, converting binary input first.
fn load_th_history(engine_exe: &Path, t01: &Path, dir: &Path) -> Result<TimeHistory, String> {
    if !is_probably_csv(t01)? {
        convert_th(engine_exe, t01, dir)?;
    }
    let csv = t01.with_extension("csv");
    let history_path = if csv.is_file() {
        csv
    } else {
        t01.to_path_buf()
    };
    sonium_qa::rad::parse_t01(&history_path).map_err(|error| error.to_string())
}

/// Second-half mean of a channel history (impact-plateau rule).
///
/// Returns `None` for empty histories.
fn second_half_mean(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let window = &values[values.len() / 2..];
    Some(window.iter().sum::<f64>() / window.len() as f64)
}

/// Settled wall-force value of a TH channel.
///
/// Rigid-wall contact forces ring on penalty impact, so the plateau
/// probes average the settled second half instead of reading the
/// phase-sensitive final sample (taylor_plastic engine FNZ: final
/// -5.77e4 vs plateau mean -3.82e4); mirrors the native arm.
fn wall_plateau_mean(history: &TimeHistory, channel: &str) -> Result<f64, String> {
    let values = history
        .channel(channel)
        .map_err(|error| error.to_string())?;
    second_half_mean(values).ok_or_else(|| format!("channel {channel} has no samples"))
}

/// Checks C3 probes against TH channel values (engine-leg rule).
///
/// Each `probes.json` expectation is anchored to the estimand that
/// produced it: taylor_plastic wall force is a settled second-half
/// plateau mean, while rail_crush and taylor_bar wall forces are
/// final-anchored; the thick cylinder follows its engine leg
/// positionally (bore DX at the 50 MPa crossing, outer DX one transit
/// later, hoop residuals as trailing means). Every other channel
/// reads the final TH value.
fn check_c3_probes(deck: &str, history: &TimeHistory, probes: &[Probe]) -> Vec<String> {
    const TRAILMEAN_SAMPLES: usize = 50;
    let sample_times = (deck == "plasticity_thick_cylinder").then(thick_cylinder_sample_times);
    let mut failures = Vec::new();
    for (index, probe) in probes.iter().enumerate() {
        let transient = match sample_times {
            Some((t_elastic, t_outer)) => Some(match index {
                0 => history
                    .value_at(&probe.channel, t_elastic)
                    .map_err(|error| error.to_string()),
                1 => history
                    .value_at(&probe.channel, t_outer)
                    .map_err(|error| error.to_string()),
                _ => history
                    .channel(&probe.channel)
                    .map_err(|error| error.to_string())
                    .and_then(|values| {
                        history_trailmean(values, TRAILMEAN_SAMPLES)
                            .ok_or_else(|| format!("channel {} has no samples", probe.channel))
                    }),
            }),
            None => None,
        };
        let plateau = deck == "taylor_plastic" && probe.channel.contains("/RWALL/");
        let sampled = if let Some(transient) = transient {
            transient
        } else if plateau {
            wall_plateau_mean(history, &probe.channel)
        } else {
            history
                .final_value(&probe.channel)
                .map_err(|error| error.to_string())
        };
        match sampled {
            Ok(value) => {
                println!(
                    "probe {} [{}]: actual {value:.6e}, expected {:.6e}",
                    probe.label, probe.channel, probe.expected
                );
                if let Err(error) = assert_probe_close(
                    value,
                    probe.expected,
                    probe.abs_tol,
                    probe.rel_tol,
                    &probe.label,
                ) {
                    failures.push(error.to_string());
                }
            }
            Err(error) => failures.push(format!("probe {} unreadable: {error}", probe.label)),
        }
    }
    failures
}

/// One timed C3 engine repeat: stage, starter, timed engine, probes.
fn engine_c3_repeat(
    ctx: &EngineCtx,
    deck: &str,
    decks: &C3Decks,
    probes: &[Probe],
    index: usize,
) -> Result<RepeatSample, String> {
    let dir = repeat_scratch(&ctx.scratch_root, deck, "engine", index);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)
            .map_err(|error| format!("cannot clean {}: {error}", dir.display()))?;
    }
    std::fs::create_dir_all(&dir)
        .map_err(|error| format!("cannot create {}: {error}", dir.display()))?;
    let outcome = engine_c3_repeat_inner(ctx, deck, decks, probes, &dir);
    match &outcome {
        Ok(_) => maybe_clean(&dir, ctx.keep_scratch),
        Err(_) => println!("note: engine scratch kept at {}", dir.display()),
    }
    outcome
}

/// Inner C3 repeat so failures keep scratch for forensics.
fn engine_c3_repeat_inner(
    ctx: &EngineCtx,
    deck: &str,
    decks: &C3Decks,
    probes: &[Probe],
    dir: &Path,
) -> Result<RepeatSample, String> {
    let starter_name = format!("{deck}_0000.rad");
    let engine_name = format!("{deck}_0001.rad");
    std::fs::copy(&decks.starter, dir.join(&starter_name))
        .map_err(|error| format!("cannot stage starter deck: {error}"))?;
    std::fs::copy(&decks.engine, dir.join(&engine_name))
        .map_err(|error| format!("cannot stage engine deck: {error}"))?;
    let engine_text = std::fs::read_to_string(dir.join(&engine_name))
        .map_err(|error| format!("cannot read staged engine deck: {error}"))?;
    let rootname = run_rootname(&engine_text)?;
    let starter_np = (ctx.launch.flavor == FlavorOpt::Mpi).then_some(ctx.launch.mpi_ranks);
    run_vendor_stage(
        ctx,
        &ctx.starter_exe,
        &starter_argv(&starter_name, starter_np, &[]),
        dir,
        "starter",
        false,
    )?;
    let (exe, argv) = engine_argv(&ctx.launch, &ctx.engine_exe, &engine_name);
    let timed = run_vendor_stage(ctx, &exe, &argv, dir, "engine", true)?;
    let engine_out = first_existing(dir, &out_candidates(&rootname, 1))?;
    let (_scalars, normal_termination) =
        parse_engine_out(&engine_out).map_err(|error| error.to_string())?;
    if !normal_termination {
        return Err(format!(
            "engine run ended without NORMAL TERMINATION in {}",
            engine_out.display()
        ));
    }
    let cycles = total_cycles(std::slice::from_ref(&engine_out))?;
    let t01 = first_existing(dir, &th_candidates(&rootname, 1))?;
    let history = load_th_history(&ctx.engine_exe, &t01, dir)?;
    let failures = check_c3_probes(deck, &history, probes);
    for failure in &failures {
        println!("FAIL: {failure}");
    }
    let dof = deck_dof(&dir.join(&starter_name))?;
    println!(
        "engine {deck}: wall {:.3}s, cycles {cycles}, dof {dof}, rss {} bytes, correct {}",
        timed.wall_s,
        timed.peak_rss_bytes,
        failures.is_empty()
    );
    Ok(RepeatSample {
        wall_s: timed.wall_s,
        cycles,
        dof,
        peak_rss_bytes: timed.peak_rss_bytes,
        correct: failures.is_empty(),
    })
}

/// One timed miniqa engine repeat against the checked-in golden.
fn engine_miniqa_repeat(
    ctx: &EngineCtx,
    miniqa_root: &Path,
    goldens_root: &Path,
    case: &str,
    index: usize,
) -> Result<RepeatSample, String> {
    let dir = repeat_scratch(&ctx.scratch_root, case, "engine", index);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)
            .map_err(|error| format!("cannot clean {}: {error}", dir.display()))?;
    }
    let outcome = engine_miniqa_repeat_inner(ctx, miniqa_root, goldens_root, case, &dir);
    match &outcome {
        Ok(_) => maybe_clean(&dir, ctx.keep_scratch),
        Err(_) => println!("note: engine scratch kept at {}", dir.display()),
    }
    outcome
}

/// Inner miniqa repeat so failures keep scratch for forensics.
fn engine_miniqa_repeat_inner(
    ctx: &EngineCtx,
    miniqa_root: &Path,
    goldens_root: &Path,
    case: &str,
    dir: &Path,
) -> Result<RepeatSample, String> {
    let deck = find_deck(miniqa_root, case).map_err(|error| error.to_string())?;
    copy_dir_all(&deck.data_dir, dir)?;
    let starter_np = match ctx.launch.flavor {
        FlavorOpt::Mpi => ctx.launch.mpi_ranks,
        _ => deck.np,
    };
    run_vendor_stage(
        ctx,
        &ctx.starter_exe,
        &starter_argv(&deck.starter_deck, Some(starter_np), &deck.starter_args),
        dir,
        "starter",
        false,
    )?;
    let mut wall_s = 0.0;
    let mut peak_rss_bytes = 0;
    for engine_deck in &deck.engine_decks {
        let (exe, argv) = engine_argv(&ctx.launch, &ctx.engine_exe, engine_deck);
        let timed = run_vendor_stage(ctx, &exe, &argv, dir, "engine", true)?;
        wall_s += timed.wall_s;
        peak_rss_bytes = peak_rss_bytes.max(timed.peak_rss_bytes);
    }
    let runs = deck.engine_decks.len();
    let engine_out = first_existing(dir, &out_candidates(&deck.rootname, runs))?;
    let (scalars, normal_termination) =
        parse_engine_out(&engine_out).map_err(|error| error.to_string())?;
    if !normal_termination {
        return Err(format!(
            "engine run ended without NORMAL TERMINATION in {}",
            engine_out.display()
        ));
    }
    let mut run_outs = Vec::new();
    for run in 1..=runs.max(1) {
        run_outs.push(first_existing(dir, &out_candidates(&deck.rootname, run))?);
    }
    let cycles = total_cycles(&run_outs)?;
    // TH files are collected by existence: most multi-run decks write
    // T01..T0N, but some runs append to an earlier file instead.
    let mut history: Option<TimeHistory> = None;
    for run in 1..=runs.max(1) {
        let Ok(t01) = first_existing(dir, &th_candidates(&deck.rootname, run)) else {
            continue;
        };
        if !is_probably_csv(&t01)? {
            convert_th(&ctx.engine_exe, &t01, dir)?;
        }
        let csv = t01.with_extension("csv");
        let csv_path = if csv.is_file() { csv } else { t01.clone() };
        let run_history = parse_th_csv(&csv_path).map_err(|error| error.to_string())?;
        match &mut history {
            None => history = Some(run_history),
            Some(first) => sonium_qa::rad::oc_golden::concat_th_runs(first, &run_history)
                .map_err(|error| error.to_string())?,
        }
    }
    let history = history.ok_or_else(|| format!("engine runs for {case} produced no TH file"))?;
    let golden_file = golden_path(goldens_root, case);
    let golden = load_golden(&golden_file)
        .map_err(|error| format!("miniqa correctness needs the checked-in golden: {error}"))?;
    let report = compare_history(&golden, &history, &CompareOptions::default());
    println!("{}", report.summary());
    for failure in &report.failures {
        println!("FAIL: {failure}");
    }
    let cycle_failures = compare_cycle(&golden, &scalars);
    for failure in &cycle_failures {
        println!("FAIL: {failure}");
    }
    let dof = deck_dof(&dir.join(&deck.starter_deck))?;
    let correct = report.passed() && cycle_failures.is_empty();
    println!(
        "engine {case}: wall {wall_s:.3}s, cycles {cycles}, dof {dof}, rss {peak_rss_bytes} bytes, correct {correct}"
    );
    Ok(RepeatSample {
        wall_s,
        cycles,
        dof,
        peak_rss_bytes,
        correct,
    })
}

/// Native rigid wall parsed from one `/RWALL/PLANE` block.
#[derive(Debug, Clone)]
struct NativeWall {
    number: i64,
    point: [f64; 3],
    normal: [f64; 3],
    velocity: [f64; 3],
    slaves: Vec<usize>,
}

/// Native concentrated load parsed from one `/CLOAD` block.
#[derive(Debug, Clone)]
struct NativeCload {
    nodes: Vec<usize>,
    axis: usize,
    funct: i64,
    scale: f64,
}

/// Native explicit model parsed from a C3 deck pair.
///
/// Equivalence basis (documented, not silent): same mesh (deck nodes
/// and bricks, driver nodes excluded), same LAW1 E/nu/rho, same wall
/// planes/velocities/slave groups, same BCS pins and CLOAD curves,
/// same stop time. What differs: the engine integrates LAW1 bricks
/// with its QEPH formulation while sonium integrates Neo-Hookean Q1
/// bricks (identical at small strain; validated by the shared probes),
/// and contact is a kinematic rigid wall (lagged constraint force
/// in the dynamics plus post-step velocity adoption; engine RWALL
/// semantics, frictionless).
/// Plastic decks (LAW2/LAW36) run the same mesh/BCs through the
/// [`hex_plastic_forces`] path with deck-parsed hardening and
/// deck-driven erosion; damage tracking stays off (no damage card).
/// The plastic arm keeps penalty walls (kinematic migration pending),
/// so its contact CFL cap stays.
#[derive(Debug, Clone)]
struct NativeModel {
    coords: Vec<[f64; 3]>,
    bricks: Vec<[usize; 8]>,
    /// Deck node id to compact native index.
    index_of: BTreeMap<i64, usize>,
    /// Deck brick id to compact native index (element probe channels).
    elem_index_of: BTreeMap<i64, usize>,
    youngs: f64,
    poisson: f64,
    density: f64,
    material: NativeMaterial,
    walls: Vec<NativeWall>,
    cloads: Vec<NativeCload>,
    /// Pinned translation axes per native node.
    pinned: Vec<[bool; 3]>,
    functs: BTreeMap<i64, Vec<[f64; 2]>>,
    stop: f64,
}

/// Native brick material parsed from the single `/MAT/*` block.
#[derive(Debug, Clone)]
enum NativeMaterial {
    /// Linear elastic (LAW1): Neo-Hookean Q1 bricks.
    Law1,
    /// Johnson-Cook (LAW2) with deck initial temperature and the
    /// deck failure strain driving erosion.
    Law2 {
        material: Law2,
        t0: f64,
        fail_ep: f64,
    },
    /// Tabulated hardening (LAW36); rupture/deletion stay off (the
    /// deck carries off-scale sentinels, verified at parse).
    Law36 { material: Law36 },
}

/// Parses one f64 token with deck context.
fn parse_scalar(token: &str, what: &str) -> Result<f64, String> {
    token
        .parse::<f64>()
        .ok()
        .filter(|value| value.is_finite())
        .ok_or_else(|| format!("{what} token {token:?} is not a finite number"))
}

/// Parses one i64 token with deck context.
fn parse_int(token: &str, what: &str) -> Result<i64, String> {
    token
        .parse::<i64>()
        .ok()
        .ok_or_else(|| format!("{what} token {token:?} is not an integer"))
}

/// Flattens a `/GRNOD/NODE` block to deck node ids, skipping padding.
fn grnod_ids(card: &DeckCard) -> Result<Vec<i64>, String> {
    let mut ids = Vec::new();
    for row in &card.fields {
        for token in row {
            let id = parse_int(token, "GRNOD node")?;
            // Vendor groups pad short rows with zeros; the C3 writer
            // never emits them, but skipping is harmless either way.
            if id != 0 {
                ids.push(id);
            }
        }
    }
    Ok(ids)
}

/// Builds the native explicit model from a C3 deck pair.
fn parse_native_model(decks: &C3Decks) -> Result<NativeModel, String> {
    let starter_text = std::fs::read_to_string(&decks.starter)
        .map_err(|error| format!("cannot read {}: {error}", decks.starter.display()))?;
    if !has_si_units(&starter_text) {
        return Err(format!(
            "sonium arm needs SI (kg/m/s) C3 decks: {} carries other units",
            decks.starter.display()
        ));
    }
    let starter = RadDeck::parse_file(&decks.starter).map_err(|error| error.to_string())?;
    let engine = RadDeck::parse_file(&decks.engine).map_err(|error| error.to_string())?;
    let mut stop: Option<f64> = None;
    for card in &engine.cards {
        match card.kind {
            CardKind::Run => {
                let value = card
                    .fields
                    .first()
                    .and_then(|row| row.first())
                    .ok_or_else(|| "engine /RUN block holds no stop time".to_string())?;
                stop = Some(parse_scalar(value, "RUN stop")?);
            }
            // Engine-side output controls do not affect the native run.
            CardKind::Title
            | CardKind::Begin
            | CardKind::End
            | CardKind::Tfile
            | CardKind::ThTitle
            | CardKind::AnimDt
            | CardKind::Print => {}
            _ => {
                return Err(format!(
                    "sonium arm supports no engine card {} (deck {})",
                    card.kind.as_str(),
                    decks.engine.display()
                ));
            }
        }
    }
    let stop = stop
        .filter(|stop| *stop > 0.0)
        .ok_or_else(|| "engine deck holds no positive /RUN stop time".to_string())?;

    let mut nodes: Vec<(i64, [f64; 3])> = Vec::new();
    let mut bricks: Vec<[i64; 8]> = Vec::new();
    let mut brick_ids: Vec<i64> = Vec::new();
    let mut groups: BTreeMap<i64, Vec<i64>> = BTreeMap::new();
    let mut functs: BTreeMap<i64, Vec<[f64; 2]>> = BTreeMap::new();
    for card in &starter.cards {
        match card.kind {
            CardKind::Node => {
                let TypedCard::Nodes(entries) = card.typed.as_ref().ok_or_else(|| {
                    format!("{} /NODE block undecodable", decks.starter.display())
                })?
                else {
                    return Err(format!(
                        "{} /NODE block undecodable",
                        decks.starter.display()
                    ));
                };
                for node in entries {
                    nodes.push((node.id, [node.x, node.y, node.z]));
                }
            }
            CardKind::Brick => {
                let TypedCard::Elems(block) = card.typed.as_ref().ok_or_else(|| {
                    format!("{} /BRICK block undecodable", decks.starter.display())
                })?
                else {
                    return Err(format!(
                        "{} /BRICK block undecodable",
                        decks.starter.display()
                    ));
                };
                if block.etype != ElemType::Brick {
                    return Err(format!("sonium arm supports no {:?} elements", block.etype));
                }
                for elem in &block.elems {
                    if elem.nodes.len() != 8 {
                        return Err(format!(
                            "brick {} has {} nodes, need 8",
                            elem.id,
                            elem.nodes.len()
                        ));
                    }
                    let mut conn = [0i64; 8];
                    conn.copy_from_slice(&elem.nodes);
                    bricks.push(conn);
                    brick_ids.push(elem.id);
                }
            }
            CardKind::GrnodNode => {
                let id = card
                    .id
                    .ok_or_else(|| "/GRNOD/NODE block misses its group id".to_string())?;
                groups.insert(id, grnod_ids(card)?);
            }
            CardKind::Funct => {
                let id = card
                    .id
                    .ok_or_else(|| "/FUNCT block misses its id".to_string())?;
                let TypedCard::Funct(funct) = card.typed.as_ref().ok_or_else(|| {
                    format!("/FUNCT {id} undecodable (need strictly increasing X Y pairs)")
                })?
                else {
                    return Err(format!("/FUNCT {id} undecodable"));
                };
                functs.insert(id, funct.points.clone());
            }
            CardKind::MatLaw1
            | CardKind::MatLaw2
            | CardKind::MatLaw36
            | CardKind::PropSolid
            | CardKind::Part
            | CardKind::Unit
            | CardKind::Title
            | CardKind::Begin
            | CardKind::End
            | CardKind::ThNode
            | CardKind::ThRwall
            | CardKind::ThTitle
            | CardKind::ThBric
            | CardKind::ThBrick => {}
            CardKind::Bcs | CardKind::Cload | CardKind::RwallPlane => {}
            _ => {
                return Err(format!(
                    "sonium arm supports no starter card {} (deferred: shells, other plastics, imposed motion)",
                    card.kind.as_str()
                ));
            }
        }
    }
    if nodes.is_empty() || bricks.is_empty() {
        return Err("starter deck holds no mesh (need /NODE + /BRICK)".to_string());
    }
    let (material, youngs, poisson, density) = parse_native_material(&starter, &functs)?;
    // Compact native numbering over brick-referenced nodes only, so
    // wall driver nodes carry no mass or dofs on either side.
    let mut referenced = std::collections::BTreeSet::new();
    for brick in &bricks {
        for id in brick {
            referenced.insert(*id);
        }
    }
    let coords_of: BTreeMap<i64, [f64; 3]> = nodes.into_iter().collect();
    let mut index_of: BTreeMap<i64, usize> = BTreeMap::new();
    let mut coords: Vec<[f64; 3]> = Vec::new();
    for id in &referenced {
        let position = coords_of
            .get(id)
            .ok_or_else(|| format!("brick references missing node {id}"))?;
        index_of.insert(*id, coords.len());
        coords.push(*position);
    }
    let mut native_bricks: Vec<[usize; 8]> = Vec::with_capacity(bricks.len());
    for brick in &bricks {
        let mut conn = [0usize; 8];
        for (slot, id) in brick.iter().enumerate() {
            conn[slot] = index_of[id];
        }
        native_bricks.push(conn);
    }
    normalize_brick_orientation(&coords, &mut native_bricks)?;
    let mut elem_index_of: BTreeMap<i64, usize> = BTreeMap::new();
    for (index, id) in brick_ids.iter().enumerate() {
        elem_index_of.insert(*id, index);
    }
    let pinned = parse_bcs(&starter, &groups, &index_of)?;
    let walls = parse_walls(&starter, &groups, &index_of, &coords_of)?;
    let cloads = parse_cloads(&starter, &groups, &index_of)?;
    Ok(NativeModel {
        coords,
        bricks: native_bricks,
        index_of,
        elem_index_of,
        youngs,
        poisson,
        density,
        material,
        walls,
        cloads,
        pinned,
        functs,
        stop,
    })
}

/// Reads E/nu/rho from the single `/MAT/LAW1` block.
fn parse_law1(starter: &RadDeck) -> Result<(f64, f64, f64), String> {
    let mut mats = Vec::new();
    for card in &starter.cards {
        if card.kind == CardKind::MatLaw1 {
            mats.push(card);
        }
    }
    let [mat] = mats.as_slice() else {
        return Err(format!(
            "sonium arm needs one /MAT/LAW1 block, found {}",
            mats.len()
        ));
    };
    let density = mat
        .fields
        .first()
        .and_then(|row| row.first())
        .ok_or_else(|| "/MAT/LAW1 block holds no density row".to_string())?;
    let density = parse_scalar(density, "LAW1 density")?;
    let moduli = mat
        .fields
        .get(1)
        .ok_or_else(|| "/MAT/LAW1 block holds no moduli row".to_string())?;
    if moduli.len() < 2 {
        return Err("/MAT/LAW1 moduli row needs E and nu".to_string());
    }
    let youngs = parse_scalar(&moduli[0], "LAW1 E")?;
    let poisson = parse_scalar(&moduli[1], "LAW1 nu")?;
    if youngs <= 0.0 || !(0.0..0.5).contains(&poisson) || density <= 0.0 {
        return Err(format!(
            "LAW1 parameters out of range: E={youngs}, nu={poisson}, rho={density}"
        ));
    }
    Ok((youngs, poisson, density))
}

/// Parses the single `/MAT/*` block into the native brick material.
///
/// Exactly one of LAW1/LAW2/LAW36 must be present; anything else (or
/// a mix) fails closed. Returns the material plus E/nu/rho for the
/// elastic wave-speed bound shared by all arms.
fn parse_native_material(
    starter: &RadDeck,
    functs: &BTreeMap<i64, Vec<[f64; 2]>>,
) -> Result<(NativeMaterial, f64, f64, f64), String> {
    let mut law1 = Vec::new();
    let mut law2 = Vec::new();
    let mut law36 = Vec::new();
    for card in &starter.cards {
        match card.kind {
            CardKind::MatLaw1 => law1.push(card),
            CardKind::MatLaw2 => law2.push(card),
            CardKind::MatLaw36 => law36.push(card),
            _ => {}
        }
    }
    let total = law1.len() + law2.len() + law36.len();
    if total != 1 {
        return Err(format!(
            "sonium arm needs one /MAT block (LAW1/LAW2/LAW36), found {total}"
        ));
    }
    if let [card] = law2.as_slice() {
        let (material, t0, fail_ep, youngs, poisson, density) = parse_law2_card(card)?;
        return Ok((
            NativeMaterial::Law2 {
                material,
                t0,
                fail_ep,
            },
            youngs,
            poisson,
            density,
        ));
    }
    if let [card] = law36.as_slice() {
        let (material, youngs, poisson, density) = parse_law36_card(card, functs)?;
        return Ok((NativeMaterial::Law36 { material }, youngs, poisson, density));
    }
    let (youngs, poisson, density) = parse_law1(starter)?;
    Ok((NativeMaterial::Law1, youngs, poisson, density))
}

/// Reads one data row of a `/MAT/*` card with deck context.
fn mat_row<'a>(card: &'a DeckCard, row: usize, what: &str) -> Result<&'a [String], String> {
    card.fields
        .get(row)
        .map(Vec::as_slice)
        .ok_or_else(|| format!("{} block holds no {what} row", card.kind.as_str()))
}

/// Parses a C3 `/MAT/LAW2` Johnson-Cook block (row layout mirrors
/// `write_extended_material`: classic input, total strain rate,
/// guide-default rate block, isothermal thermal line).
fn parse_law2_card(card: &DeckCard) -> Result<(Law2, f64, f64, f64, f64, f64), String> {
    let density = parse_scalar(
        mat_row(card, 0, "density")?
            .first()
            .ok_or_else(|| "/MAT/LAW2 block holds no density".to_string())?,
        "LAW2 density",
    )?;
    let moduli = mat_row(card, 1, "moduli")?;
    if moduli.len() < 5 {
        return Err("/MAT/LAW2 moduli row needs E, nu, Iflag, Icpre, cutoff".to_string());
    }
    let youngs = parse_scalar(&moduli[0], "LAW2 E")?;
    let poisson = parse_scalar(&moduli[1], "LAW2 nu")?;
    let iflag = parse_int(&moduli[2], "LAW2 Iflag")?;
    if iflag != 0 {
        return Err(format!(
            "LAW2 Iflag {iflag} is not classic Johnson-Cook (need 0)"
        ));
    }
    let flow_row = mat_row(card, 2, "flow")?;
    if flow_row.len() < 5 {
        return Err("/MAT/LAW2 flow row needs a, b, n, fail strain, max stress".to_string());
    }
    let flow_a = parse_scalar(&flow_row[0], "LAW2 a")?;
    let flow_b = parse_scalar(&flow_row[1], "LAW2 b")?;
    let flow_n = parse_scalar(&flow_row[2], "LAW2 n")?;
    let fail_ep = parse_scalar(&flow_row[3], "LAW2 fail strain")?;
    let max_stress = parse_scalar(&flow_row[4], "LAW2 max stress")?;
    let rate_row = mat_row(card, 3, "rate")?;
    if rate_row.len() < 6 {
        return Err("/MAT/LAW2 rate row needs C, ref rate, ICC, Fsmooth, fcut, Chard".to_string());
    }
    let rate_c = parse_scalar(&rate_row[0], "LAW2 C")?;
    let ref_rate = parse_scalar(&rate_row[1], "LAW2 ref rate")?;
    let icc = parse_int(&rate_row[2], "LAW2 ICC")?;
    if icc != 1 {
        return Err(format!(
            "LAW2 ICC {icc} is not the smoothed-rate treatment (need 1)"
        ));
    }
    let fcut = parse_scalar(&rate_row[4], "LAW2 fcut")?;
    let thermal = mat_row(card, 4, "thermal")?;
    if thermal.len() < 5 {
        return Err("/MAT/LAW2 thermal row needs T0, Tmelt, rhoCp, Tref, Tmax".to_string());
    }
    let t0 = parse_scalar(&thermal[3], "LAW2 T0")?;
    if youngs <= 0.0 || !(0.0..0.5).contains(&poisson) || density <= 0.0 {
        return Err(format!(
            "LAW2 elastic parameters out of range: E={youngs}, nu={poisson}, rho={density}"
        ));
    }
    if !(0.0..=1.0).contains(&flow_n) || rate_c < 0.0 || ref_rate <= 0.0 {
        return Err(format!(
            "LAW2 flow parameters out of range: n={flow_n}, C={rate_c}, ref rate={ref_rate}"
        ));
    }
    if fail_ep <= 0.0 || max_stress <= 0.0 || fcut <= 0.0 {
        return Err("LAW2 fail strain, max stress, and fcut must be positive".to_string());
    }
    // Isothermal run (writer: no thermal softening): melt/heat terms
    // off, reference temperature from the deck; damage tracking stays
    // off (Law2::new defaults damage to None; no damage card).
    let material = Law2::new(
        youngs,
        poisson,
        Law2Flow::JohnsonCook {
            a: flow_a,
            b: flow_b,
            n: flow_n,
            c: rate_c,
        },
        ref_rate,
        0.0,
        0.0,
        t0,
        0.0,
        max_stress,
        MaxRateEffect::Scaled,
        fail_ep,
        0.0,
        fcut,
    );
    Ok((material, t0, fail_ep, youngs, poisson, density))
}

/// Parses a C3 `/MAT/LAW36` tabulated-hardening block with its single
/// hardening curve resolved through the parsed `/FUNCT` table.
fn parse_law36_card(
    card: &DeckCard,
    functs: &BTreeMap<i64, Vec<[f64; 2]>>,
) -> Result<(Law36, f64, f64, f64), String> {
    let density = parse_scalar(
        mat_row(card, 0, "density")?
            .first()
            .ok_or_else(|| "/MAT/LAW36 block holds no density".to_string())?,
        "LAW36 density",
    )?;
    let moduli = mat_row(card, 1, "moduli")?;
    if moduli.len() < 5 {
        return Err("/MAT/LAW36 moduli row needs E, nu, rupture, t1, t2".to_string());
    }
    let youngs = parse_scalar(&moduli[0], "LAW36 E")?;
    let poisson = parse_scalar(&moduli[1], "LAW36 nu")?;
    let rupture = parse_scalar(&moduli[2], "LAW36 rupture")?;
    let tensile_t1 = parse_scalar(&moduli[3], "LAW36 t1")?;
    let tensile_t2 = parse_scalar(&moduli[4], "LAW36 t2")?;
    for (name, value) in [("rupture", rupture), ("t1", tensile_t1), ("t2", tensile_t2)] {
        if value < 1.0e19 {
            return Err(format!(
                "LAW36 {name} {value:.3e} is active (need off-scale sentinel)"
            ));
        }
    }
    let flags = mat_row(card, 2, "flags")?;
    if flags.len() < 6 {
        return Err(
            "/MAT/LAW36 flags row needs Nc, Ismooth, Fsmooth, Vref, deletion, Idel".to_string(),
        );
    }
    let deletion = parse_scalar(&flags[4], "LAW36 deletion")?;
    if deletion < 1.0e19 {
        return Err(format!(
            "LAW36 deletion {deletion:.3e} is active (need off-scale sentinel)"
        ));
    }
    let model_row = mat_row(card, 3, "model")?;
    if model_row.len() < 5 {
        return Err("LAW36 model row needs Iform, Ifil, Ipos, Fcut, Vrel".to_string());
    }
    let mut model_values = [0.0; 5];
    for (slot, token) in model_row.iter().take(5).enumerate() {
        model_values[slot] = parse_scalar(token, "LAW36 model")?;
    }
    if model_values != [0.0, 1.0, 0.0, 0.0, 0.0] {
        return Err("LAW36 model row differs from the C3 writer pattern".to_string());
    }
    let curve_id = parse_int(
        mat_row(card, 4, "curve")?
            .first()
            .ok_or_else(|| "/MAT/LAW36 block holds no curve id".to_string())?,
        "LAW36 curve",
    )?;
    let scale = parse_scalar(
        mat_row(card, 5, "scale")?
            .first()
            .ok_or_else(|| "/MAT/LAW36 block holds no curve scale".to_string())?,
        "LAW36 scale",
    )?;
    if youngs <= 0.0 || !(0.0..0.5).contains(&poisson) || density <= 0.0 {
        return Err(format!(
            "LAW36 elastic parameters out of range: E={youngs}, nu={poisson}, rho={density}"
        ));
    }
    let points = functs
        .get(&curve_id)
        .ok_or_else(|| format!("/MAT/LAW36 references missing /FUNCT {curve_id}"))?;
    let mut x = Vec::with_capacity(points.len());
    let mut y = Vec::with_capacity(points.len());
    for point in points {
        x.push(point[0]);
        y.push(point[1]);
    }
    // Single quasi-static curve (no rate table in the C3 subset).
    println!(
        "sonium LAW36 /FUNCT {curve_id}: {} points, y0 {:.6e}, ymax {:.6e}, scale {scale:.6e}",
        x.len(),
        y.first().copied().unwrap_or(f64::NAN),
        y.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
    );
    let curve = RateCurve::new(0.0, x, y, scale)
        .map_err(|error| format!("LAW36 curve invalid: {error}"))?;
    let material = Law36::new(
        youngs,
        poisson,
        vec![curve],
        0.0,
        rupture,
        tensile_t1,
        tensile_t2,
        deletion,
    );
    Ok((material, youngs, poisson, density))
}

/// Maps a deck group to compact native indices, skipping driver nodes.
fn group_indices(
    groups: &BTreeMap<i64, Vec<i64>>,
    group: i64,
    index_of: &BTreeMap<i64, usize>,
    what: &str,
) -> Result<Vec<usize>, String> {
    let members = groups
        .get(&group)
        .ok_or_else(|| format!("{what} references missing group {group}"))?;
    let mut indices = Vec::new();
    for id in members {
        // Driver nodes (wall masters) carry no structural mass; the
        // engine likewise excludes them from the model.
        if let Some(index) = index_of.get(id) {
            indices.push(*index);
        }
    }
    Ok(indices)
}

/// Parses `/BCS` pins into per-node locked-axis flags.
fn parse_bcs(
    starter: &RadDeck,
    groups: &BTreeMap<i64, Vec<i64>>,
    index_of: &BTreeMap<i64, usize>,
) -> Result<Vec<[bool; 3]>, String> {
    let mut pinned = vec![[false; 3]; index_of.len()];
    for card in &starter.cards {
        if card.kind != CardKind::Bcs {
            continue;
        }
        let TypedCard::Bcs(bcs) = card
            .typed
            .as_ref()
            .ok_or_else(|| "/BCS block undecodable".to_string())?
        else {
            return Err("/BCS block undecodable".to_string());
        };
        if bcs.skew.is_some_and(|skew| skew != 0) {
            return Err("/BCS in a skewed frame is not supported".to_string());
        }
        // Rotation locks are meaningless on bricks (no rotational
        // dofs); the C3 writer sets them, so ignore them openly.
        let tra: Vec<char> = bcs.tra.chars().collect();
        if tra.len() != 3 || !tra.iter().all(|flag| matches!(flag, '0' | '1')) {
            return Err(format!(
                "/BCS translation code {:?} is not three flags",
                bcs.tra
            ));
        }
        let group = bcs
            .grnod
            .ok_or_else(|| "/BCS block references no group".to_string())?;
        for index in group_indices(groups, group, index_of, "/BCS")? {
            for axis in 0..3 {
                pinned[index][axis] |= tra[axis] == '1';
            }
        }
    }
    Ok(pinned)
}

/// Parses `/RWALL/PLANE` blocks into native penalty walls.
fn parse_walls(
    starter: &RadDeck,
    groups: &BTreeMap<i64, Vec<i64>>,
    index_of: &BTreeMap<i64, usize>,
    coords_of: &BTreeMap<i64, [f64; 3]>,
) -> Result<Vec<NativeWall>, String> {
    let mut walls = Vec::new();
    for card in &starter.cards {
        if card.kind != CardKind::RwallPlane {
            continue;
        }
        let number = card
            .id
            .ok_or_else(|| "/RWALL/PLANE block misses its number".to_string())?;
        let TypedCard::Rwall(header) = card
            .typed
            .as_ref()
            .ok_or_else(|| format!("/RWALL/PLANE {number} header undecodable"))?
        else {
            return Err(format!("/RWALL/PLANE {number} header undecodable"));
        };
        if card.fields.len() < 3 {
            return Err(format!("/RWALL/PLANE {number} holds too few rows"));
        }
        let friction = card.fields[1]
            .get(1)
            .map(|token| parse_scalar(token, "RWALL friction"))
            .transpose()?
            .unwrap_or(0.0);
        if friction != 0.0 {
            return Err(format!(
                "/RWALL/PLANE {number} carries friction (unvalidated mapping)"
            ));
        }
        // Moving walls (driver node set) carry `0 VX VY VZ`; fixed
        // walls carry the plane point; both close with M1 = M + normal.
        let (point, velocity) = if header.node != 0 {
            let point = coords_of.get(&header.node).ok_or_else(|| {
                format!("/RWALL/PLANE {number} driver node {} missing", header.node)
            })?;
            if card.fields[2].len() < 4 {
                return Err(format!("/RWALL/PLANE {number} velocity row too short"));
            }
            let velocity = [
                parse_scalar(&card.fields[2][1], "RWALL VX")?,
                parse_scalar(&card.fields[2][2], "RWALL VY")?,
                parse_scalar(&card.fields[2][3], "RWALL VZ")?,
            ];
            (*point, velocity)
        } else {
            if card.fields[2].len() < 3 {
                return Err(format!("/RWALL/PLANE {number} point row too short"));
            }
            let point = [
                parse_scalar(&card.fields[2][0], "RWALL PX")?,
                parse_scalar(&card.fields[2][1], "RWALL PY")?,
                parse_scalar(&card.fields[2][2], "RWALL PZ")?,
            ];
            (point, [0.0; 3])
        };
        let orientation = card.fields.last().unwrap_or(&card.fields[2]);
        if orientation.len() < 3 {
            return Err(format!("/RWALL/PLANE {number} orientation row too short"));
        }
        let tip = [
            parse_scalar(&orientation[0], "RWALL M1X")?,
            parse_scalar(&orientation[1], "RWALL M1Y")?,
            parse_scalar(&orientation[2], "RWALL M1Z")?,
        ];
        let mut normal = [tip[0] - point[0], tip[1] - point[1], tip[2] - point[2]];
        let length = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
        if !length.is_finite() || length <= 0.0 {
            return Err(format!("/RWALL/PLANE {number} normal is degenerate"));
        }
        // The hertz deck carries a non-unit orientation offset; the
        // engine normalizes, so the native wall does the same.
        normal = [normal[0] / length, normal[1] / length, normal[2] / length];
        let mut slaves = Vec::new();
        for group in &header.grnds {
            if *group != 0 {
                slaves.extend(group_indices(groups, *group, index_of, "/RWALL")?);
            }
        }
        slaves.sort_unstable();
        slaves.dedup();
        if slaves.is_empty() {
            return Err(format!("/RWALL/PLANE {number} has no slave nodes"));
        }
        walls.push(NativeWall {
            number,
            point,
            normal,
            velocity,
            slaves,
        });
    }
    walls.sort_by_key(|wall| wall.number);
    Ok(walls)
}

/// Parses `/CLOAD` blocks into native nodal loads.
fn parse_cloads(
    starter: &RadDeck,
    groups: &BTreeMap<i64, Vec<i64>>,
    index_of: &BTreeMap<i64, usize>,
) -> Result<Vec<NativeCload>, String> {
    let mut cloads = Vec::new();
    for card in &starter.cards {
        if card.kind != CardKind::Cload {
            continue;
        }
        let TypedCard::Load(LoadCard::Cload(cload)) = card
            .typed
            .as_ref()
            .ok_or_else(|| "/CLOAD block undecodable".to_string())?
        else {
            return Err("/CLOAD block undecodable".to_string());
        };
        let axis = match cload.dir.to_uppercase().as_str() {
            "X" => 0,
            "Y" => 1,
            "Z" => 2,
            other => return Err(format!("/CLOAD direction {other} unsupported on bricks")),
        };
        // Writer row: fct Dir skew sens grnd 0 Fscale Magnitude; the
        // scale applies to EACH node of the set (rad writer precedent).
        let row = card
            .fields
            .first()
            .ok_or_else(|| "/CLOAD block holds no row".to_string())?;
        if row.len() < 8 {
            return Err("/CLOAD row too short for the writer shape".to_string());
        }
        let fscale = parse_scalar(&row[6], "CLOAD Fscale")?;
        let magnitude = parse_scalar(&row[7], "CLOAD magnitude")?;
        let nodes = group_indices(groups, cload.grnod, index_of, "/CLOAD")?;
        if nodes.is_empty() {
            return Err("/CLOAD group holds no structural nodes".to_string());
        }
        cloads.push(NativeCload {
            nodes,
            axis,
            funct: cload.funct,
            scale: fscale * magnitude,
        });
    }
    Ok(cloads)
}

/// Twelve H8 edges in `HEX_NODES` slot order.
const HEX_EDGES: [(usize, usize); 12] = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 0),
    (4, 5),
    (5, 6),
    (6, 7),
    (7, 4),
    (0, 4),
    (1, 5),
    (2, 6),
    (3, 7),
];

/// Reference volume of one element by Gauss quadrature of `det J`.
fn element_volume(mesh: &HexMesh, elem: usize) -> f64 {
    signed_brick_volume(&mesh.element_coords(elem))
}

/// Signed quadrature volume of one brick from its 8 corner positions.
fn signed_brick_volume(coords: &[[f64; 3]; 8]) -> f64 {
    let mut volume = 0.0;
    for (natural, weight) in hex_quadrature() {
        let derivatives = hex_shape_derivatives(&natural);
        let (_, det, _) = hex_jacobian(coords, &derivatives);
        volume += weight * det;
    }
    volume
}

/// Reorders mirrored-winding bricks into the canonical slot convention.
///
/// Some C3 decks (beam_curved_arch) author bricks with both faces
/// wound clockwise seen from above; the engine accepts either
/// winding, but the isoparametric map is then orientation-reversing
/// and signed quadrature weights would negate the stiffness and
/// lumped mass. Reversing both faces (`[0,3,2,1,4,7,6,5]`) keeps
/// the same 8 nodes with straight edges while restoring the
/// canonical counter-clockwise convention. Each brick is handled
/// independently (already-positive bricks are untouched) and the
/// fix is verified per brick, failing closed on degenerate
/// geometry rather than a winding convention.
fn normalize_brick_orientation(
    coords: &[[f64; 3]],
    bricks: &mut [[usize; 8]],
) -> Result<(), String> {
    /// Slot permutation that reverses both face windings.
    const FACE_REVERSAL: [usize; 8] = [0, 3, 2, 1, 4, 7, 6, 5];
    for (elem, conn) in bricks.iter_mut().enumerate() {
        let gather = |conn: &[usize; 8]| {
            let mut corners = [[0.0; 3]; 8];
            for (slot, node) in conn.iter().enumerate() {
                corners[slot] = coords[*node];
            }
            corners
        };
        if signed_brick_volume(&gather(conn)) > 0.0 {
            continue;
        }
        let mut reordered = [0usize; 8];
        for (slot, source) in FACE_REVERSAL.iter().enumerate() {
            reordered[slot] = conn[*source];
        }
        *conn = reordered;
        let fixed = signed_brick_volume(&gather(conn));
        if !fixed.is_finite() || fixed <= 0.0 {
            return Err(format!(
                "element {elem} has non-positive volume {fixed:.6e} even after orientation fix"
            ));
        }
    }
    Ok(())
}

/// Minimum edge length of one element in the reference configuration.
fn min_edge_length(mesh: &HexMesh, elem: usize) -> f64 {
    let coords = mesh.element_coords(elem);
    let mut shortest = f64::INFINITY;
    for (a, b) in HEX_EDGES {
        let dx = coords[a][0] - coords[b][0];
        let dy = coords[a][1] - coords[b][1];
        let dz = coords[a][2] - coords[b][2];
        shortest = shortest.min((dx * dx + dy * dy + dz * dz).sqrt());
    }
    shortest
}

/// Native probe target: a deck node displacement, a wall force, or a
/// brick element value (centroid stress component / plastic strain).
#[derive(Clone, Copy)]
enum ProbeTarget {
    Displacement { axis: usize, node: usize },
    WallForce { axis: usize, wall: usize },
    ElemStress { comp: usize, elem: usize },
    ElemPlas { elem: usize },
}

/// Linearly interpolates a recorded probe history at a time.
///
/// Returns `None` when the history is empty, ragged, or the time
/// falls outside the sampled grid (no extrapolation).
fn history_lerp(times: &[f64], values: &[f64], time: f64) -> Option<f64> {
    if times.len() != values.len() || times.is_empty() {
        return None;
    }
    if time < times[0] || time > times[times.len() - 1] {
        return None;
    }
    for window in times.windows(2).zip(values.windows(2)) {
        let (bounds, pair) = window;
        if time >= bounds[0] && time <= bounds[1] {
            let span = bounds[1] - bounds[0];
            if span <= 0.0 {
                return Some(pair[0]);
            }
            let weight = (time - bounds[0]) / span;
            return Some(pair[0] * (1.0 - weight) + pair[1] * weight);
        }
    }
    values.last().copied()
}

/// Averages the trailing `n` history samples (ringdown-canceling mean).
///
/// Returns `None` for empty histories; shorter histories average over
/// whatever samples exist.
fn history_trailmean(values: &[f64], n: usize) -> Option<f64> {
    if values.is_empty() || n == 0 {
        return None;
    }
    let take = values.len().min(n);
    let window = &values[values.len() - take..];
    Some(window.iter().sum::<f64>() / take as f64)
}

/// Averages history samples at or after `start` (zero-hold mean).
///
/// Returns `None` when the history is ragged or no sample reaches
/// `start`. Averaging the full free-ringdown hold cancels the breathing
/// mode over many periods where a short trailing window keeps a
/// phase-dependent residue.
fn history_holdmean(times: &[f64], values: &[f64], start: f64) -> Option<f64> {
    if times.len() != values.len() {
        return None;
    }
    let mut sum = 0.0;
    let mut count = 0usize;
    for (time, value) in times.iter().zip(values.iter()) {
        if *time >= start {
            sum += *value;
            count += 1;
        }
    }
    if count == 0 {
        None
    } else {
        Some(sum / count as f64)
    }
}

/// Transient sample times of the thick-cylinder engine leg.
///
/// Mirrors `fem_opencourant_plasticity_thick_cylinder`: the bore probe
/// reads where the raised cosine crosses 50 MPa on the loading leg,
/// the outer probe one dilatational transit later. Constants are
/// transcribed from that leg; the unit test pins both times.
fn thick_cylinder_sample_times() -> (f64, f64) {
    const A_M: f64 = 0.05;
    const B_M: f64 = 0.10;
    const E_PA: f64 = 200.0e9;
    const NU: f64 = 0.30;
    const RHO: f64 = 7800.0;
    const P_STAR_PA: f64 = 150.0e6;
    const LOAD_TIME_S: f64 = 0.001;
    const ELASTIC_PROBE_PA: f64 = 50.0e6;
    let t_elastic =
        LOAD_TIME_S * (1.0 - 2.0 * ELASTIC_PROBE_PA / P_STAR_PA).acos() / std::f64::consts::PI;
    let lame_lambda = E_PA * NU / ((1.0 + NU) * (1.0 - 2.0 * NU));
    let shear_mu = E_PA / (2.0 * (1.0 + NU));
    let c_dil = ((lame_lambda + 2.0 * shear_mu) / RHO).sqrt();
    (t_elastic, t_elastic + (B_M - A_M) / c_dil)
}

/// Resolves a C3 probe channel to a native target.
fn resolve_probe_channel(channel: &str, model: &NativeModel) -> Result<ProbeTarget, String> {
    if let Some(rest) = channel
        .strip_prefix("DX/")
        .map(|id| (0, id))
        .or_else(|| channel.strip_prefix("DY/").map(|id| (1, id)))
        .or_else(|| channel.strip_prefix("DZ/").map(|id| (2, id)))
    {
        let id: i64 = rest
            .1
            .parse()
            .ok()
            .ok_or_else(|| format!("probe channel {channel} names no node"))?;
        let node = model
            .index_of
            .get(&id)
            .copied()
            .ok_or_else(|| format!("probe channel {channel} names a non-structural node"))?;
        return Ok(ProbeTarget::Displacement { axis: rest.0, node });
    }
    for (prefix, axis) in [("FNX/RWALL/", 0), ("FNY/RWALL/", 1), ("FNZ/RWALL/", 2)] {
        if let Some(id) = channel.strip_prefix(prefix) {
            let number: i64 = id
                .parse()
                .ok()
                .ok_or_else(|| format!("probe channel {channel} names no wall"))?;
            let wall = model
                .walls
                .iter()
                .position(|wall| wall.number == number)
                .ok_or_else(|| format!("probe channel {channel} names a missing wall"))?;
            return Ok(ProbeTarget::WallForce { axis, wall });
        }
    }
    for (prefix, comp) in [
        ("SX/BRIC/", 0),
        ("SY/BRIC/", 1),
        ("SZ/BRIC/", 2),
        ("SXY/BRIC/", 3),
        ("SYZ/BRIC/", 4),
        ("SZX/BRIC/", 5),
    ] {
        if let Some(id) = channel.strip_prefix(prefix) {
            let number: i64 = id
                .parse()
                .ok()
                .ok_or_else(|| format!("probe channel {channel} names no brick"))?;
            let elem = model
                .elem_index_of
                .get(&number)
                .copied()
                .ok_or_else(|| format!("probe channel {channel} names a missing brick"))?;
            return Ok(ProbeTarget::ElemStress { comp, elem });
        }
    }
    if let Some(id) = channel.strip_prefix("PLAS/BRIC/") {
        let number: i64 = id
            .parse()
            .ok()
            .ok_or_else(|| format!("probe channel {channel} names no brick"))?;
        let elem = model
            .elem_index_of
            .get(&number)
            .copied()
            .ok_or_else(|| format!("probe channel {channel} names a missing brick"))?;
        return Ok(ProbeTarget::ElemPlas { elem });
    }
    Err(format!(
        "probe channel {channel} unsupported (need DX/DY/DZ, FN/RWALL, or SX/SY/SZ/SXY/SYZ/SZX/PLAS per BRIC)"
    ))
}

/// One timed native explicit repeat over the parsed deck model.
fn sonium_repeat(
    deck: &str,
    model: &NativeModel,
    probes: &[Probe],
) -> Result<RepeatSample, String> {
    if !matches!(model.material, NativeMaterial::Law1) {
        return sonium_repeat_plastic(deck, model, probes);
    }
    let mesh = HexMesh::new(model.coords.clone(), model.bricks.clone());
    let num_nodes = mesh.num_nodes();
    let num_elems = mesh.conn.len();
    if num_nodes == 0 || num_elems == 0 {
        return Err("native model holds no mesh".to_string());
    }
    let mut masses = vec![0.0; num_nodes];
    let mut char_lengths = Vec::with_capacity(num_elems);
    for elem in 0..num_elems {
        let volume = element_volume(&mesh, elem);
        if !volume.is_finite() || volume <= 0.0 {
            return Err(format!(
                "element {elem} has non-positive volume {volume:.6e}"
            ));
        }
        for node in mesh.conn[elem] {
            masses[node] += model.density * volume / 8.0;
        }
        char_lengths.push(min_edge_length(&mesh, elem));
    }
    let min_mass = masses.iter().cloned().fold(f64::INFINITY, f64::min);
    if !min_mass.is_finite() || min_mass <= 0.0 {
        return Err(format!("non-positive lumped mass {min_mass:.6e}"));
    }
    let material = NeoHookean::from_young(model.youngs, model.poisson, 3);
    let dilatational = ((material.lambda + 2.0 * material.mu) / model.density).sqrt();
    let sound_speeds = vec![dilatational; num_elems];
    // Kinematic walls apply a lagged contact force (a prescribed
    // load, not a spring), so no contact stiffness enters the
    // dynamics and every deck rides the bulk CFL cap: the old
    // `sqrt(m/k)` cap cost wall decks ~12x in step count for zero
    // accuracy gain (engine RWALLs are force-based too).
    let dt_max = critical_dt(&char_lengths, &sound_speeds);
    if !dt_max.is_finite() || dt_max <= 0.0 {
        return Err("native dt bound is not positive".to_string());
    }
    // Sound pre-check: steps can only exceed stop/dt_max (dt never
    // passes the cap), so an over-budget estimate rejects instantly
    // instead of burning hours on a quasi-static deck.
    let estimated = (model.stop / dt_max).ceil() as u64;
    if estimated > MAX_NATIVE_STEPS {
        return Err(format!(
            "native run needs about {estimated} steps (stop {}s, dt_max {dt_max:.3e}s): quasi-static timescale deferred (needs implicit or mass-scaling support)",
            model.stop
        ));
    }
    println!(
        "sonium mesh: {num_nodes} nodes, {num_elems} bricks, dt_max {dt_max:.3e}s, about {estimated} steps"
    );
    let config = AdaptiveDtConfig::new(0.9, 1.0e-12, dt_max, 1.1)
        .map_err(|error| format!("adaptive dt config rejected: {error}"))?;
    let mut driver = ExplicitStepDriver::new(config)
        .map_err(|error| format!("step driver rejected: {error}"))?;
    let mut state = ExplicitState::new(num_nodes);
    for (node, xyz) in model.coords.iter().enumerate() {
        state.positions[3 * node] = xyz[0];
        state.positions[3 * node + 1] = xyz[1];
        state.positions[3 * node + 2] = xyz[2];
    }
    state.masses.copy_from_slice(&masses);
    let reference: Vec<f64> = state.positions.clone();
    // Wall-force history per wall: (time, force on wall). Stride keeps
    // sampling bounded on long runs without biasing the tail mean.
    let stride = ((estimated / MAX_FORCE_SAMPLES as u64).max(1)) as usize;

    // Reused scratch: the driver calls force closures sequentially (never
    // re-entrantly), so one displacement buffer serves every evaluation and
    // the per-step allocation drops to the returned force vector alone (the
    // driver takes ownership of it, hence it cannot be recycled here).
    let displacement_buf = RefCell::new(vec![0.0; 3 * num_nodes]);
    let internal_force = |positions: &[f64], _: &[f64], _: f64| {
        let mut displacement = displacement_buf.borrow_mut();
        for (slot, (pos, pos0)) in displacement
            .iter_mut()
            .zip(positions.iter().zip(reference.iter()))
        {
            *slot = pos - pos0;
        }
        let mut forces = vec![0.0; positions.len()];
        for elem in 0..num_elems {
            let residual = q1_element_residual(&material, &mesh, elem, &displacement);
            for (slot, node) in mesh.conn[elem].iter().enumerate() {
                forces[3 * node] -= residual[3 * slot];
                forces[3 * node + 1] -= residual[3 * slot + 1];
                forces[3 * node + 2] -= residual[3 * slot + 2];
            }
        }
        // Ideal pins: zero force rows freeze pinned dofs exactly at
        // their zero initial state (the driver rejects zero masses,
        // so pinning lives in the force closures, not the mass list).
        for (node, axes) in model.pinned.iter().enumerate() {
            for axis in 0..3 {
                if axes[axis] {
                    forces[3 * node + axis] = 0.0;
                }
            }
        }
        forces
    };
    // Lagged kinematic contact force: the previous step's recovered
    // constraint force, applied as a prescribed external load. A
    // prescribed force adds no stiffness (unlike a penalty spring),
    // so the step stays CFL-limited while the force itself cancels
    // the bulk load; the post-step adoption kills residual drift.
    let contact_applied = RefCell::new(vec![0.0; 3 * num_nodes]);
    let mut contact_staging = vec![0.0; 3 * num_nodes];
    let external_force = |positions: &[f64], _: &[f64], time: f64| {
        let mut forces = vec![0.0; positions.len()];
        let applied = contact_applied.borrow();
        for (slot, lagged) in forces.iter_mut().zip(applied.iter()) {
            *slot += *lagged;
        }
        for cload in &model.cloads {
            let shape = model
                .functs
                .get(&cload.funct)
                .and_then(|points| eval_funct(points, time).ok());
            let Some(shape) = shape else { continue };
            for node in &cload.nodes {
                forces[3 * node + cload.axis] += cload.scale * shape;
            }
        }
        for (node, axes) in model.pinned.iter().enumerate() {
            for axis in 0..3 {
                if axes[axis] {
                    forces[3 * node + axis] = 0.0;
                }
            }
        }
        forces
    };

    let start = std::time::Instant::now();
    let mut steps: u64 = 0;
    // Wall histories sampled inline per step with stride (kept outside
    // the force closures so the driver sees pure evaluations).
    let mut sampled: Vec<Vec<(f64, [f64; 3])>> = vec![Vec::new(); model.walls.len()];
    // Accumulated velocity-adoption kinetic jumps; the driver's
    // ledger sees the applied contact force work (it flows through
    // the external channel) but not these jumps, so the drift gate
    // below adds them back.
    let mut wall_ke = 0.0;
    // Deck walls move at constant velocity (no acceleration card).
    let wall_acceleration = [0.0; 3];
    while state.time < model.stop {
        if steps >= MAX_NATIVE_STEPS {
            return Err(format!(
                "stalled: {MAX_NATIVE_STEPS} steps reached at t = {:.6e}s, short of stop {:.6e}s",
                state.time, model.stop
            ));
        }
        if let Err(error) = driver.step(
            &mut state,
            internal_force,
            external_force,
            &char_lengths,
            &sound_speeds,
        ) {
            return Err(format!(
                "unstable at t = {:.6e}s after {steps} steps: {error}",
                state.time
            ));
        }
        // Kinematic walls AFTER the free step: adopt wall velocities
        // and recover constraint forces from the driver's freshly
        // evaluated channels (no re-evaluation); the recovery stages
        // the next step's applied contact force.
        let (free_internal, free_external) = driver.last_forces().ok_or_else(|| {
            format!("step {steps} completed without evaluated forces for the wall projection")
        })?;
        let sample = ((steps + 1) as usize).is_multiple_of(stride);
        contact_staging.fill(0.0);
        let applied = contact_applied.borrow();
        for (index, wall) in model.walls.iter().enumerate() {
            let point = [
                wall.point[0] + wall.velocity[0] * state.time,
                wall.point[1] + wall.velocity[1] * state.time,
                wall.point[2] + wall.velocity[2] * state.time,
            ];
            let mut rigid = RigidWall::new(point, wall.normal, 0.0);
            rigid.velocity = wall.velocity;
            let contact = rigid_wall_project(
                &rigid,
                wall_acceleration,
                &wall.slaves,
                &state.masses,
                &state.positions,
                &mut state.velocities,
                free_internal,
                free_external,
                &applied,
                state.dt,
            );
            wall_ke += contact.kinetic_jump;
            for pair in &contact.pairs {
                contact_staging[3 * pair.slave] += pair.force[0];
                contact_staging[3 * pair.slave + 1] += pair.force[1];
                contact_staging[3 * pair.slave + 2] += pair.force[2];
            }
            // TEMP-DIAG: remove before finishing.
            if steps % 200 == 0 && std::env::var("SONIUM_OC_DIAG").is_ok() {
                let inst: f64 = contact.pairs.iter().map(|p| p.force[0]).sum();
                // Native indices of deck nodes 95 (crusher face) and 5 (anvil face).
                let i95 = model.index_of.get(&95).copied().unwrap_or(0);
                let i5 = model.index_of.get(&5).copied().unwrap_or(0);
                eprintln!(
                    "diag step {steps} t={:.6e} w{} pairs={} instFx={inst:.3e} x95={:.6e} v95={:.3e} x5={:.6e} v5={:.3e}",
                    state.time,
                    wall.number,
                    contact.pairs.len(),
                    state.positions[3 * i95],
                    state.velocities[3 * i95],
                    state.positions[3 * i5],
                    state.velocities[3 * i5],
                );
            }
            if sample {
                let mut on_wall = [0.0; 3];
                for pair in &contact.pairs {
                    on_wall[0] -= pair.force[0];
                    on_wall[1] -= pair.force[1];
                    on_wall[2] -= pair.force[2];
                }
                sampled[index].push((state.time, on_wall));
            }
        }
        drop(applied);
        contact_applied
            .borrow_mut()
            .copy_from_slice(&contact_staging);
        steps += 1;
    }
    let wall_s = start.elapsed().as_secs_f64();
    println!(
        "sonium run: reached t = {:.6e}s in {steps} steps",
        state.time
    );

    let mut failures = Vec::new();
    for probe in probes {
        let target = match resolve_probe_channel(&probe.channel, model) {
            Ok(target) => target,
            Err(error) => {
                failures.push(error);
                continue;
            }
        };
        let actual = match target {
            ProbeTarget::Displacement { axis, node } => {
                state.positions[3 * node + axis] - reference[3 * node + axis]
            }
            ProbeTarget::WallForce { axis, wall } => {
                let history = &sampled[wall];
                if history.is_empty() {
                    failures.push(format!("probe {} has no wall-force samples", probe.label));
                    continue;
                }
                // Impact walls (initial gap) settle past the initial
                // transient, so the probe reads the second-half plateau
                // mean; touching walls load quasi-statically, so the
                // probe reads the final load.
                let wall = &model.walls[wall];
                let mut gap = f64::INFINITY;
                for slave in &wall.slaves {
                    let position = model.coords[*slave];
                    let distance = (position[0] - wall.point[0]) * wall.normal[0]
                        + (position[1] - wall.point[1]) * wall.normal[1]
                        + (position[2] - wall.point[2]) * wall.normal[2];
                    gap = gap.min(distance);
                }
                if gap <= TOUCHING_GAP_M {
                    history[history.len() - 1].1[axis]
                } else {
                    let half = history.len() / 2;
                    let window = &history[half..];
                    window.iter().map(|(_, force)| force[axis]).sum::<f64>() / window.len() as f64
                }
            }
            ProbeTarget::ElemStress { .. } | ProbeTarget::ElemPlas { .. } => {
                failures.push(format!(
                    "probe {} needs element state from a plastic run",
                    probe.label
                ));
                continue;
            }
        };
        println!(
            "probe {} [{}]: actual {actual:.6e}, expected {:.6e}",
            probe.label, probe.channel, probe.expected
        );
        if let Err(error) = assert_probe_close(
            actual,
            probe.expected,
            probe.abs_tol,
            probe.rel_tol,
            &probe.label,
        ) {
            failures.push(error.to_string());
        }
    }
    for failure in &failures {
        println!("FAIL: {failure}");
    }
    let balance = driver.balance().clone();
    let energy_scale = balance.kinetic + balance.internal.abs();
    // True drift adds the adoption kinetic jumps the driver's ledger
    // cannot see (the applied contact force work flows through Wext
    // already): drift = (KE - KE0) + IE - Wext - Wadopt.
    let drift = balance.drift - wall_ke;
    let drift_frac = if energy_scale > 0.0 {
        drift.abs() / energy_scale
    } else {
        f64::INFINITY
    };
    println!(
        "energy: KE {:.6e}J, IE {:.6e}J, Wext {:.6e}J, Wadopt {:.6e}J, drift {:.3e} rel",
        balance.kinetic,
        balance.internal,
        balance.external_work,
        wall_ke,
        drift_frac
    );
    if !drift_frac.is_finite() || drift_frac > DRIFT_TOL {
        failures.push(format!(
            "energy drift {drift_frac:.3e} exceeds {DRIFT_TOL:.3e}"
        ));
    }
    let dof = u64::try_from(num_nodes)
        .ok()
        .and_then(|nodes| nodes.checked_mul(3))
        .ok_or_else(|| "node count overflows dof".to_string())?;
    println!(
        "sonium: wall {wall_s:.3}s, cycles {steps}, dof {dof}, correct {}",
        failures.is_empty()
    );
    Ok(RepeatSample {
        wall_s,
        cycles: steps,
        dof,
        peak_rss_bytes: self_peak_rss_bytes(),
        correct: failures.is_empty(),
    })
}

/// Minimum edge length of one brick at the given positions.
fn min_edge_length_at(positions: &[f64], conn: &[usize; 8]) -> f64 {
    let mut shortest = f64::INFINITY;
    for (a, b) in HEX_EDGES {
        let dx = positions[3 * conn[a]] - positions[3 * conn[b]];
        let dy = positions[3 * conn[a] + 1] - positions[3 * conn[b] + 1];
        let dz = positions[3 * conn[a] + 2] - positions[3 * conn[b] + 2];
        shortest = shortest.min((dx * dx + dy * dy + dz * dz).sqrt());
    }
    shortest
}

/// One timed native explicit repeat for LAW2/LAW36 brick decks.
///
/// Dispatches on the parsed material into the generic plastic core
/// below; LAW1 never reaches here (guarded by the caller).
fn sonium_repeat_plastic(
    deck: &str,
    model: &NativeModel,
    probes: &[Probe],
) -> Result<RepeatSample, String> {
    match &model.material {
        NativeMaterial::Law2 {
            material,
            t0,
            fail_ep,
        } => {
            let states: Vec<HexExplicitState<Law2State>> = model
                .bricks
                .iter()
                .map(|_| HexExplicitState::new(Law2State::new(*t0)))
                .collect();
            let criterion = ErosionCriterion {
                max_equiv_plastic_strain: *fail_ep,
                max_damage: 1.0,
                max_porosity: f64::INFINITY,
                max_tabular_damage: f64::INFINITY,
            };
            repeat_plastic_core(deck, model, probes, material, states, criterion)
        }
        NativeMaterial::Law36 { material } => {
            let states: Vec<HexExplicitState<Law36State>> = model
                .bricks
                .iter()
                .map(|_| HexExplicitState::new(Law36State::new()))
                .collect();
            // The C3 LAW36 subset carries no failure data (verified
            // off-scale sentinels at parse), so erosion stays off and
            // the sweep below only deletes inverted elements.
            let criterion = ErosionCriterion {
                max_equiv_plastic_strain: f64::INFINITY,
                max_damage: 1.0,
                max_porosity: f64::INFINITY,
                max_tabular_damage: f64::INFINITY,
            };
            repeat_plastic_core(deck, model, probes, material, states, criterion)
        }
        NativeMaterial::Law1 => Err("plastic repeat called on a LAW1 model".to_string()),
    }
}

/// Generic plastic repeat: fixed-step CFL loop over
/// [`hex_plastic_forces`] with an erosion sweep, mirroring the
/// validated Taylor-plastic leg driver (fixed stepping threads the
/// true step rate into the flow stress; per-step edge lengths keep
/// the CFL bound honest under large deformation).
fn repeat_plastic_core<M: ExplicitPlasticMaterial>(
    deck: &str,
    model: &NativeModel,
    probes: &[Probe],
    material: &M,
    initial: Vec<HexExplicitState<M::State>>,
    criterion: ErosionCriterion,
) -> Result<RepeatSample, String> {
    let mesh = HexMesh::new(model.coords.clone(), model.bricks.clone());
    let num_nodes = mesh.num_nodes();
    let num_elems = mesh.conn.len();
    if num_nodes == 0 || num_elems == 0 {
        return Err("native model holds no mesh".to_string());
    }
    let mut masses = vec![0.0; num_nodes];
    let mut char_lengths = Vec::with_capacity(num_elems);
    for elem in 0..num_elems {
        let volume = element_volume(&mesh, elem);
        if !volume.is_finite() || volume <= 0.0 {
            return Err(format!(
                "element {elem} has non-positive volume {volume:.6e}"
            ));
        }
        for node in mesh.conn[elem] {
            masses[node] += model.density * volume / 8.0;
        }
        char_lengths.push(min_edge_length(&mesh, elem));
    }
    let min_mass = masses.iter().cloned().fold(f64::INFINITY, f64::min);
    if !min_mass.is_finite() || min_mass <= 0.0 {
        return Err(format!("non-positive lumped mass {min_mass:.6e}"));
    }
    let min_char = char_lengths.iter().cloned().fold(f64::INFINITY, f64::min);
    let elastic = NeoHookean::from_young(model.youngs, model.poisson, 3);
    let dilatational = ((elastic.lambda + 2.0 * elastic.mu) / model.density).sqrt();
    let sound_speeds = vec![dilatational; num_elems];
    let dt_crit = critical_dt(&char_lengths, &sound_speeds);
    let dt_max = if model.walls.is_empty() {
        dt_crit
    } else {
        // Contact RESOLUTION cap (leg precedent), not just stability:
        // dt <= 2*sqrt(m/k) is stable but chatters (force zero most
        // steps, rare spikes, low plateau mean); omega dt = 0.2 holds
        // persistent contact and the pilot mean matches the engine.
        let penalty = suggested_penalty(model.youngs, min_char, PLASTIC_PENALTY_FACTOR);
        0.1 * 2.0 * (min_mass / penalty).sqrt()
    };
    if !dt_max.is_finite() || dt_max <= 0.0 {
        return Err("native dt bound is not positive".to_string());
    }
    let estimated = (model.stop / dt_max).ceil() as u64;
    if estimated > MAX_NATIVE_STEPS {
        return Err(format!(
            "native run needs about {estimated} steps (stop {}s, dt_max {dt_max:.3e}s): quasi-static timescale deferred (needs implicit or mass-scaling support)",
            model.stop
        ));
    }
    println!(
        "sonium mesh: {num_nodes} nodes, {num_elems} bricks, dt_max {dt_max:.3e}s, about {estimated} steps"
    );
    match &model.material {
        NativeMaterial::Law1 => {}
        NativeMaterial::Law2 { t0, fail_ep, .. } => {
            println!("sonium material: LAW2 T0 {t0:.1}K, fail ep {fail_ep:.3e}");
        }
        NativeMaterial::Law36 { material } => {
            let curves: Vec<String> = material
                .curves
                .iter()
                .map(|curve| format!("rate {:.3e} scale {:.3e}", curve.rate, curve.scale))
                .collect();
            println!("sonium material: LAW36 [{}]", curves.join(", "));
        }
    }
    let mut params = HexExplicitParams::new(model.density, dilatational);
    // The linear bulk term has a kink at zero volumetric rate (it fires
    // in compression only), which chatters under velocity-Verlet on
    // quasi-static ramps: the thick cylinder grew 55%/step from roundoff
    // noise into a 13 mm plastic limit cycle with the default c1 = 0.05
    // and matches the engine to 1.5% with c1 = 0. The quadratic term is
    // C1-smooth at zero rate (no kink) and stays on for shock capturing;
    // Taylor impact probes pass identically either way.
    params.bulk_c1 = 0.0;
    let states = RefCell::new(initial);
    let erosion = RefCell::new(ErosionSet::new(num_elems));
    let step_dt = RefCell::new(dt_max);
    let max_ep = RefCell::new(0.0f64);
    let mut driver = ExplicitStepDriver::new(
        AdaptiveDtConfig::new(0.9, 1.0e-12, dt_max, 1.1)
            .map_err(|error| format!("adaptive dt config rejected: {error}"))?,
    )
    .map_err(|error| format!("step driver rejected: {error}"))?;
    let mut state = ExplicitState::new(num_nodes);
    for (node, xyz) in model.coords.iter().enumerate() {
        state.positions[3 * node] = xyz[0];
        state.positions[3 * node + 1] = xyz[1];
        state.positions[3 * node + 2] = xyz[2];
    }
    state.masses.copy_from_slice(&masses);
    let reference: Vec<f64> = state.positions.clone();
    let penalty = suggested_penalty(model.youngs, min_char, PLASTIC_PENALTY_FACTOR);
    let stride = ((estimated / MAX_FORCE_SAMPLES as u64).max(1)) as usize;
    // Dashpot energy dissipated over the run (stabilization footprint;
    // wall-force probes read the penalty spring alone, leg precedent).
    let damp_dissipated = RefCell::new(0.0f64);

    let internal_force = |positions: &[f64], velocities: &[f64], _: f64| {
        let dt = *step_dt.borrow();
        let mut forces = vec![0.0; positions.len()];
        let mut states = states.borrow_mut();
        let mut erosion = erosion.borrow_mut();
        let mut features = Vec::with_capacity(num_elems);
        for (elem, conn) in mesh.conn.iter().enumerate() {
            if !erosion.is_active(elem) {
                features.push((0.0, 0.0, 0.0));
                continue;
            }
            let mut elem_coords = [[0.0; 3]; 8];
            for (slot, node) in conn.iter().enumerate() {
                elem_coords[slot] = [
                    positions[3 * node],
                    positions[3 * node + 1],
                    positions[3 * node + 2],
                ];
            }
            let geom = match HexExplicitGeom::new(elem_coords) {
                Ok(geom) => geom,
                Err(_) => {
                    features.push((f64::INFINITY, 0.0, 0.0));
                    continue;
                }
            };
            let mut vel = [0.0; 24];
            for (slot, node) in conn.iter().enumerate() {
                vel[slot * 3] = velocities[3 * node];
                vel[slot * 3 + 1] = velocities[3 * node + 1];
                vel[slot * 3 + 2] = velocities[3 * node + 2];
            }
            match hex_plastic_forces(&geom, &mut states[elem], material, &vel, dt, &params) {
                Ok(out) => {
                    for (slot, node) in conn.iter().enumerate() {
                        forces[3 * node] -= out.base.forces[slot * 3];
                        forces[3 * node + 1] -= out.base.forces[slot * 3 + 1];
                        forces[3 * node + 2] -= out.base.forces[slot * 3 + 2];
                    }
                    let mut peak = max_ep.borrow_mut();
                    *peak = peak.max(out.features.0);
                    features.push(out.features);
                }
                Err(_) => {
                    features.push((f64::INFINITY, 0.0, 0.0));
                }
            }
        }
        erosion.sweep(&criterion, &features);
        for (node, axes) in model.pinned.iter().enumerate() {
            for axis in 0..3 {
                if axes[axis] {
                    forces[3 * node + axis] = 0.0;
                }
            }
        }
        forces
    };
    let external_force = |positions: &[f64], velocities: &[f64], time: f64| {
        let mut forces = vec![0.0; positions.len()];
        for wall in &model.walls {
            let point = [
                wall.point[0] + wall.velocity[0] * time,
                wall.point[1] + wall.velocity[1] * time,
                wall.point[2] + wall.velocity[2] * time,
            ];
            let mut rigid = RigidWall::new(point, wall.normal, penalty);
            rigid.velocity = wall.velocity;
            let pairs =
                rigid_wall_contact(&rigid, &wall.slaves, positions, velocities, &mut forces);
            // Overdamped dashpot on each penetrating slave (leg
            // precedent): settles the bounce mode; its reaction joins
            // the dynamics but stays out of the spring-force probe.
            let dt = *step_dt.borrow();
            let mut dissipated = damp_dissipated.borrow_mut();
            for pair in &pairs {
                let node = pair.slave;
                let c = 2.0 * PLASTIC_CONTACT_DAMPING * (penalty * masses[node]).sqrt();
                let vn_rel = (velocities[3 * node] - wall.velocity[0]) * wall.normal[0]
                    + (velocities[3 * node + 1] - wall.velocity[1]) * wall.normal[1]
                    + (velocities[3 * node + 2] - wall.velocity[2]) * wall.normal[2];
                let damp = -c * vn_rel;
                forces[3 * node] += damp * wall.normal[0];
                forces[3 * node + 1] += damp * wall.normal[1];
                forces[3 * node + 2] += damp * wall.normal[2];
                *dissipated += c * vn_rel * vn_rel * dt;
            }
        }
        for cload in &model.cloads {
            let shape = model
                .functs
                .get(&cload.funct)
                .and_then(|points| eval_funct(points, time).ok());
            let Some(shape) = shape else { continue };
            for node in &cload.nodes {
                forces[3 * node + cload.axis] += cload.scale * shape;
            }
        }
        for (node, axes) in model.pinned.iter().enumerate() {
            for axis in 0..3 {
                if axes[axis] {
                    forces[3 * node + axis] = 0.0;
                }
            }
        }
        forces
    };

    // The thick-cylinder engine leg samples its displacement probes at
    // transient times (50 MPa crossing, outer transit) and its stress
    // probes as trailing means, so the native arm records per-step
    // histories for those targets; every other deck keeps final-state
    // sampling and skips the recording.
    let record_histories = deck == "plasticity_thick_cylinder";
    let targets: Vec<Option<ProbeTarget>> = if record_histories {
        probes
            .iter()
            .map(|probe| resolve_probe_channel(&probe.channel, model).ok())
            .collect()
    } else {
        Vec::new()
    };
    let mut probe_times: Vec<f64> = Vec::new();
    let mut probe_values: Vec<Vec<f64>> = vec![Vec::new(); targets.len()];
    let start = std::time::Instant::now();
    let mut steps: u64 = 0;
    let mut dt_last = dt_max;
    let mut sampled: Vec<Vec<(f64, [f64; 3])>> = vec![Vec::new(); model.walls.len()];
    while state.time < model.stop {
        if steps >= MAX_NATIVE_STEPS {
            return Err(format!(
                "stalled: {MAX_NATIVE_STEPS} steps reached at t = {:.6e}s, short of stop {:.6e}s",
                state.time, model.stop
            ));
        }
        if (steps as usize).is_multiple_of(stride) {
            let mut scratch = vec![0.0; state.positions.len()];
            for (index, wall) in model.walls.iter().enumerate() {
                let point = [
                    wall.point[0] + wall.velocity[0] * state.time,
                    wall.point[1] + wall.velocity[1] * state.time,
                    wall.point[2] + wall.velocity[2] * state.time,
                ];
                let mut rigid = RigidWall::new(point, wall.normal, penalty);
                rigid.velocity = wall.velocity;
                let pairs = rigid_wall_contact(
                    &rigid,
                    &wall.slaves,
                    &state.positions,
                    &state.velocities,
                    &mut scratch,
                );
                let mut on_wall = [0.0; 3];
                for pair in &pairs {
                    on_wall[0] -= pair.force[0];
                    on_wall[1] -= pair.force[1];
                    on_wall[2] -= pair.force[2];
                }
                sampled[index].push((state.time, on_wall));
            }
        }
        let char_now: Vec<f64> = mesh
            .conn
            .iter()
            .map(|conn| min_edge_length_at(&state.positions, conn))
            .collect();
        let dt_crit = critical_dt(&char_now, &sound_speeds);
        let mut dt = (0.9 * dt_crit).min(dt_max).min(1.1 * dt_last);
        dt = dt.min(model.stop - state.time).max(1.0e-12);
        if !dt.is_finite() || dt <= 0.0 {
            return Err(format!(
                "non-positive step {dt:.6e} at t = {:.6e}s",
                state.time
            ));
        }
        *step_dt.borrow_mut() = dt;
        if let Err(error) = driver.step_fixed(&mut state, internal_force, external_force, dt) {
            return Err(format!(
                "unstable at t = {:.6e}s after {steps} steps: {error}",
                state.time
            ));
        }
        dt_last = dt;
        steps += 1;
        if record_histories {
            probe_times.push(state.time);
            let snapshots = states.borrow();
            for (slot, target) in targets.iter().copied().enumerate() {
                let value = match target {
                    Some(ProbeTarget::Displacement { axis, node }) => {
                        state.positions[3 * node + axis] - reference[3 * node + axis]
                    }
                    Some(ProbeTarget::ElemStress { comp, elem }) => {
                        M::centroid_stress(&snapshots[elem].centroid)[comp]
                    }
                    _ => f64::NAN,
                };
                probe_values[slot].push(value);
            }
        }
    }
    let wall_s = start.elapsed().as_secs_f64();
    println!(
        "sonium run: reached t = {:.6e}s in {steps} steps",
        state.time
    );

    // Engine-leg sampling mirror for the thick cylinder: probe order is
    // positional there (bore DX at the 50 MPa crossing, outer DX one
    // transit later, hoop residuals as trailing means), so the native
    // arm applies the same schedule to its recorded histories, except
    // the residuals average the full zero hold: the native run is
    // undamped (c1 = 0 kills the only damper), so its ringdown never
    // decays and a 50-sample window keeps a phase-dependent residue,
    // while the hold mean cancels the breathing mode over ~19 periods.
    // Any unrecordable target falls back to final-state sampling below.
    // Hold start transcribed from the leg deck (raised cosine unloads
    // to zero at twice the load time, then holds to stop).
    const HOLD_START_S: f64 = 0.002;
    let sample_times = record_histories.then(thick_cylinder_sample_times);
    let mut failures = Vec::new();
    for (index, probe) in probes.iter().enumerate() {
        let target = match resolve_probe_channel(&probe.channel, model) {
            Ok(target) => target,
            Err(error) => {
                failures.push(error);
                continue;
            }
        };
        let mut actual = match target {
            ProbeTarget::Displacement { axis, node } => {
                state.positions[3 * node + axis] - reference[3 * node + axis]
            }
            ProbeTarget::WallForce { axis, wall } => {
                let history = &sampled[wall];
                if history.is_empty() {
                    failures.push(format!("probe {} has no wall-force samples", probe.label));
                    continue;
                }
                let wall = &model.walls[wall];
                let mut gap = f64::INFINITY;
                for slave in &wall.slaves {
                    let position = model.coords[*slave];
                    let distance = (position[0] - wall.point[0]) * wall.normal[0]
                        + (position[1] - wall.point[1]) * wall.normal[1]
                        + (position[2] - wall.point[2]) * wall.normal[2];
                    gap = gap.min(distance);
                }
                if gap <= TOUCHING_GAP_M {
                    history[history.len() - 1].1[axis]
                } else {
                    let half = history.len() / 2;
                    let window = &history[half..];
                    window.iter().map(|(_, force)| force[axis]).sum::<f64>() / window.len() as f64
                }
            }
            ProbeTarget::ElemStress { comp, elem } => {
                let states = states.borrow();
                M::centroid_stress(&states[elem].centroid)[comp]
            }
            ProbeTarget::ElemPlas { elem } => {
                let states = states.borrow();
                material.erosion_features(&states[elem].centroid).0
            }
        };
        if let (Some((t_elastic, t_outer)), Some(values)) = (sample_times, probe_values.get(index))
        {
            let sampled = match index {
                0 => history_lerp(&probe_times, values, t_elastic),
                1 => history_lerp(&probe_times, values, t_outer),
                _ => history_holdmean(&probe_times, values, HOLD_START_S),
            };
            if let Some(value) = sampled.filter(|value| value.is_finite()) {
                actual = value;
            }
        }
        println!(
            "probe {} [{}]: actual {actual:.6e}, expected {:.6e}",
            probe.label, probe.channel, probe.expected
        );
        if let Err(error) = assert_probe_close(
            actual,
            probe.expected,
            probe.abs_tol,
            probe.rel_tol,
            &probe.label,
        ) {
            failures.push(error.to_string());
        }
    }
    for failure in &failures {
        println!("FAIL: {failure}");
    }
    let balance = driver.balance().clone();
    let energy_scale = balance.kinetic + balance.internal.abs();
    let drift_frac = if energy_scale > 0.0 {
        balance.drift.abs() / energy_scale
    } else {
        f64::INFINITY
    };
    println!(
        "energy: KE {:.6e}J, IE {:.6e}J, Wext {:.6e}J, drift {:.3e} rel",
        balance.kinetic, balance.internal, balance.external_work, drift_frac
    );
    if !drift_frac.is_finite() || drift_frac > DRIFT_TOL {
        failures.push(format!(
            "energy drift {drift_frac:.3e} exceeds {DRIFT_TOL:.3e}"
        ));
    }
    let eroded = num_elems - erosion.borrow().active_count();
    println!(
        "plastic: max ep {:.6e}, eroded {eroded}/{num_elems}",
        *max_ep.borrow(),
    );
    println!(
        "contact: dashpot dissipated {:.6e} J over the run",
        *damp_dissipated.borrow(),
    );
    let dof = u64::try_from(num_nodes)
        .ok()
        .and_then(|nodes| nodes.checked_mul(3))
        .ok_or_else(|| "node count overflows dof".to_string())?;
    println!(
        "sonium: wall {wall_s:.3}s, cycles {steps}, dof {dof}, correct {}",
        failures.is_empty()
    );
    Ok(RepeatSample {
        wall_s,
        cycles: steps,
        dof,
        peak_rss_bytes: self_peak_rss_bytes(),
        correct: failures.is_empty(),
    })
}

/// Resolved deck family: C3 cases live under the crate `decks/`
/// dir; anything else resolves as a miniqa case id.
#[derive(Debug)]
enum DeckFamily {
    C3(C3Decks),
    Miniqa(PathBuf),
}

/// Resolves a deck id to its family, failing closed on unknown ids.
fn resolve_deck_family(deck: &str, miniqa_root: Option<PathBuf>) -> Result<DeckFamily, String> {
    if let Ok(c3) = resolve_c3_decks(deck) {
        return Ok(DeckFamily::C3(c3));
    }
    let Some(root) = miniqa_root else {
        return Err(format!(
            "deck {deck} is not a C3 case and no --miniqa/{MINIQA_ENV} resolves it as miniqa"
        ));
    };
    find_deck(&root, deck)
        .map_err(|error| format!("deck {deck} resolves neither as C3 nor miniqa: {error}"))?;
    Ok(DeckFamily::Miniqa(root))
}

/// Builds the engine-arm context, resolving exes and launch mode.
fn engine_context(args: &Args) -> Result<EngineCtx, String> {
    let exe = args
        .exe
        .clone()
        .or_else(engine_exe)
        .ok_or_else(|| "engine arm needs --exe, RADISS_EXE, or OPENCOURANT_EXE".to_string())?;
    let (starter_exe, mut engine_exe) =
        resolve_engine_pair(&exe).map_err(|error| error.to_string())?;
    let provisional = resolve_launch(args, &engine_exe)?;
    if provisional.flavor == FlavorOpt::Mpi
        && !engine_exe.to_string_lossy().to_lowercase().contains("ompi")
        && !engine_exe.to_string_lossy().to_lowercase().contains("_mpi")
    {
        engine_exe = select_mpi_engine(&exe)?;
    }
    let launch = resolve_launch(args, &engine_exe)?;
    let exec_dir = if exe.is_dir() {
        exe.clone()
    } else {
        exe.parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| exe.clone())
    };
    Ok(EngineCtx {
        starter_exe,
        engine_exe,
        exec_dir,
        launch,
        scratch_root: resolve_scratch_root(args.scratch_root.clone()),
        keep_scratch: args.keep_scratch,
    })
}

/// Runs the engine arm over warmup plus timed repeats.
fn run_engine_arm(args: &Args, deck: &str, repeats: u32, warmup: u32) -> Result<BenchRow, String> {
    let ctx = engine_context(args)?;
    let miniqa = args.miniqa.clone().or_else(|| {
        std::env::var(MINIQA_ENV)
            .ok()
            .filter(|root| !root.trim().is_empty())
            .map(PathBuf::from)
    });
    let goldens_root = args
        .goldens
        .clone()
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens"));
    let family = resolve_deck_family(deck, miniqa)?;
    let c3_probes = match &family {
        DeckFamily::C3(decks) => Some(read_probe_file(&decks.probes)?),
        DeckFamily::Miniqa(_) => None,
    };
    let one = |index: usize| match (&family, &c3_probes) {
        (DeckFamily::C3(decks), Some(probes)) => engine_c3_repeat(&ctx, deck, decks, probes, index),
        (DeckFamily::Miniqa(root), _) => {
            engine_miniqa_repeat(&ctx, root, &goldens_root, deck, index)
        }
        (DeckFamily::C3(_), None) => Err("C3 probes unreadable".to_string()),
    };
    for warmup_index in 0..warmup {
        println!("engine {deck}: warmup {}/{}", warmup_index + 1, warmup);
        one(warmup_index as usize)?;
    }
    let mut samples = Vec::new();
    for repeat in 0..repeats {
        println!("engine {deck}: repeat {}/{}", repeat + 1, repeats);
        samples.push(one((warmup + repeat) as usize)?);
    }
    let total = aggregate_repeats(&samples)?;
    BenchRow::measured(
        deck,
        Side::Engine,
        &ctx.launch.label,
        u32::try_from(ctx.launch.row_threads)
            .map_err(|_| "thread count overflows u32".to_string())?,
        total.wall_s,
        total.cycles,
        total.dof,
        total.peak_rss_bytes,
        total.correct,
    )
    .map_err(|error| error.to_string())
}

/// Runs the sonium arm over warmup plus timed repeats (C3 only).
fn run_sonium_arm(args: &Args, deck: &str, repeats: u32, warmup: u32) -> Result<BenchRow, String> {
    if resolve_c3_decks(deck).is_err() && deck.contains('/') {
        return Err(format!(
            "sonium arm supports C3 decks only; miniqa case {deck} deferred (no native model for vendor decks)"
        ));
    }
    // The native explicit path is serial today; the setting is
    // recorded on the row so the 6B matrix stays explicit.
    // Safety: no other thread reads the environment concurrently at
    // this point (single-threaded startup before any spawn).
    unsafe {
        std::env::set_var("RAYON_NUM_THREADS", args.threads.max(1).to_string());
    }
    let decks = resolve_c3_decks(deck)?;
    let probes = read_probe_file(&decks.probes)?;
    let model = parse_native_model(&decks)?;
    for warmup_index in 0..warmup {
        println!("sonium {deck}: warmup {}/{}", warmup_index + 1, warmup);
        let _ = sonium_repeat(deck, &model, &probes)?;
    }
    let mut samples = Vec::new();
    for repeat in 0..repeats {
        println!("sonium {deck}: repeat {}/{}", repeat + 1, repeats);
        samples.push(sonium_repeat(deck, &model, &probes)?);
    }
    let total = aggregate_repeats(&samples)?;
    BenchRow::measured(
        deck,
        Side::Sonium,
        "native",
        u32::try_from(args.threads.max(1)).map_err(|_| "thread count overflows u32".to_string())?,
        total.wall_s,
        total.cycles,
        total.dof,
        total.peak_rss_bytes,
        total.correct,
    )
    .map_err(|error| error.to_string())
}

/// First CPU model string from `/proc/cpuinfo`, else `unknown`.
fn cpu_model() -> String {
    std::fs::read_to_string("/proc/cpuinfo")
        .ok()
        .and_then(|text| {
            text.lines().find_map(|line| {
                line.strip_prefix("model name")
                    .and_then(|rest| rest.split(':').nth(1))
                    .map(|model| model.trim().to_string())
                    .filter(|model| !model.is_empty())
            })
        })
        .unwrap_or_else(|| "unknown".to_string())
}

fn run() -> Result<(), String> {
    let args = Args::parse();
    let deck = args
        .deck
        .clone()
        .unwrap_or_else(|| DEFAULT_DECK.to_string());
    if deck.trim().is_empty() {
        return Err("--deck must not be empty".to_string());
    }
    let (run_sonium, run_engine) = resolve_sides(&args)?;
    let (repeats, warmup) = resolve_repeats(&args)?;
    if args.threads == 0 {
        return Err("--threads must be positive".to_string());
    }
    let mut rows = Vec::new();
    if run_sonium {
        rows.push(run_sonium_arm(&args, &deck, repeats, warmup)?);
    }
    if run_engine {
        rows.push(run_engine_arm(&args, &deck, repeats, warmup)?);
    }
    let bundle = BenchBundle {
        schema: BUNDLE_SCHEMA.to_string(),
        hardware: HardwareInfo {
            arch: std::env::consts::ARCH.to_string(),
            model: cpu_model(),
            threads: args.threads.max(1).to_string(),
        },
        protocol: ProtocolInfo {
            repeats,
            warmup,
            correctness: PROTOCOL_CORRECTNESS.to_string(),
        },
        rows,
    };
    let text = bundle_json(&bundle).map_err(|error| error.to_string())?;
    if let Some(parent) = args
        .emit_bundle
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        std::fs::create_dir_all(parent)
            .map_err(|error| format!("cannot create {}: {error}", parent.display()))?;
    }
    std::fs::write(&args.emit_bundle, &text)
        .map_err(|error| format!("cannot write {}: {error}", args.emit_bundle.display()))?;
    println!("wrote {}", args.emit_bundle.display());
    let report = render_bundle_markdown(&bundle).map_err(|error| error.to_string())?;
    println!("{report}");
    if bundle.rows.iter().all(|row| row.correctness_passed) {
        Ok(())
    } else {
        Err("one or more rows failed their correctness probes".to_string())
    }
}

fn main() {
    match run() {
        Ok(()) => println!("SUCCESS: qa-oc-bench"),
        Err(error) => {
            eprintln!("error: {error}");
            println!("FAILED: qa-oc-bench");
            std::process::exit(1);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args_with(side: SideOpt, sonium_only: bool, engine_only: bool) -> Args {
        Args {
            deck: None,
            emit_bundle: PathBuf::from("bundle.json"),
            side,
            threads: 1,
            repeats: None,
            quick: false,
            sonium_only,
            engine_only,
            flavor: None,
            mpi_ranks: None,
            exe: None,
            miniqa: None,
            goldens: None,
            scratch_root: None,
            keep_scratch: false,
        }
    }

    #[test]
    fn sides_default_to_both() {
        assert_eq!(
            resolve_sides(&args_with(SideOpt::Both, false, false)).unwrap(),
            (true, true)
        );
    }

    #[test]
    fn only_flags_select_single_arms() {
        assert_eq!(
            resolve_sides(&args_with(SideOpt::Both, true, false)).unwrap(),
            (true, false)
        );
        assert_eq!(
            resolve_sides(&args_with(SideOpt::Both, false, true)).unwrap(),
            (false, true)
        );
    }

    #[test]
    fn contradictory_sides_fail() {
        assert!(resolve_sides(&args_with(SideOpt::Both, true, true)).is_err());
        assert!(resolve_sides(&args_with(SideOpt::Engine, true, false)).is_err());
        assert!(resolve_sides(&args_with(SideOpt::Sonium, false, true)).is_err());
    }

    #[test]
    fn repeats_default_with_warmup() {
        let args = args_with(SideOpt::Both, false, false);
        assert_eq!(resolve_repeats(&args).unwrap(), (3, 1));
    }

    #[test]
    fn quick_forces_single_repeat() {
        let mut args = args_with(SideOpt::Both, false, false);
        args.quick = true;
        assert_eq!(resolve_repeats(&args).unwrap(), (1, 0));
        args.repeats = Some(3);
        assert!(resolve_repeats(&args).is_err());
    }

    #[test]
    fn funct_interpolates_and_clamps() {
        let points = [[0.0, 0.0], [1.0, 2.0]];
        assert_eq!(eval_funct(&points, -1.0).unwrap(), 0.0);
        assert_eq!(eval_funct(&points, 0.5).unwrap(), 1.0);
        assert_eq!(eval_funct(&points, 2.0).unwrap(), 2.0);
        assert!(eval_funct(&[], 0.0).is_err());
    }

    #[test]
    fn rootname_parses_run_line() {
        let text = "#RADIOSS ENGINE\n/RUN/taylor_bar/1\n   4.000000000000E-6\n";
        assert_eq!(run_rootname(text).unwrap(), "taylor_bar");
        assert!(run_rootname("no run here\n").is_err());
    }

    #[test]
    fn si_units_detected() {
        assert!(has_si_units("/BEGIN\n  kg   m   s\n"));
        assert!(!has_si_units("/BEGIN\n  g   mm   ms\n"));
    }

    #[test]
    fn deck_family_prefers_c3() {
        let family = resolve_deck_family("taylor_bar", None).unwrap();
        assert!(matches!(family, DeckFamily::C3(_)));
        let error = resolve_deck_family("no_such_deck", None).unwrap_err();
        assert!(error.contains("not a C3 case"));
    }

    fn sample(wall_s: f64, correct: bool) -> RepeatSample {
        RepeatSample {
            wall_s,
            cycles: 100,
            dof: 600,
            peak_rss_bytes: 1000,
            correct,
        }
    }

    #[test]
    fn aggregation_takes_median_and_and() {
        let total =
            aggregate_repeats(&[sample(3.0, true), sample(1.0, true), sample(2.0, true)]).unwrap();
        assert_eq!(total.wall_s, 2.0);
        assert!(total.correct);
        let mixed = aggregate_repeats(&[sample(1.0, true), sample(2.0, false)]).unwrap();
        assert!(!mixed.correct);
        assert!(aggregate_repeats(&[]).is_err());
    }

    #[test]
    fn launch_labels_precision_suffix() {
        let mut args = args_with(SideOpt::Both, false, false);
        args.threads = 4;
        let launch = resolve_launch(&args, Path::new("engine_linux64_gf")).unwrap();
        assert_eq!(launch.flavor, FlavorOpt::Smp);
        assert_eq!(launch.label, "smp");
        assert_eq!(launch.row_threads, 4);
        let launch = resolve_launch(&args, Path::new("engine_linux64_gf_sp")).unwrap();
        assert_eq!(launch.label, "smp_sp");
        args.flavor = Some(FlavorOpt::Mpi);
        let launch = resolve_launch(&args, Path::new("engine_linux64_gf_ompi")).unwrap();
        assert_eq!(launch.row_threads, 4);
        assert_eq!(launch.omp_threads, 1);
    }

    #[test]
    fn engine_argv_shapes() {
        let smp = EngineLaunch {
            flavor: FlavorOpt::Smp,
            label: "smp".to_string(),
            row_threads: 4,
            omp_threads: 4,
            mpi_ranks: 4,
        };
        let (exe, argv) = engine_argv(&smp, Path::new("engine"), "case_0001.rad");
        assert_eq!(exe, PathBuf::from("engine"));
        assert_eq!(argv, vec!["-i", "case_0001.rad", "-nt", "4"]);
        let mpi = EngineLaunch {
            flavor: FlavorOpt::Mpi,
            label: "mpi".to_string(),
            row_threads: 2,
            omp_threads: 1,
            mpi_ranks: 2,
        };
        let (exe, argv) = engine_argv(&mpi, Path::new("/exec/engine_ompi"), "case_0001.rad");
        assert_eq!(exe, PathBuf::from("mpirun"));
        assert!(argv.contains(&"-np".to_string()));
    }

    #[test]
    fn total_cycles_sums_run_outs() {
        let dir = std::env::temp_dir().join(format!("oc_bench_cycles_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let first = dir.join("a_0001.out");
        let second = dir.join("a_0002.out");
        std::fs::write(&first, "     TOTAL NUMBER OF CYCLES  :      24\n").unwrap();
        std::fs::write(&second, "     TOTAL NUMBER OF CYCLES  :      10\n").unwrap();
        assert_eq!(total_cycles(&[first, second]).unwrap(), 34);
        assert!(total_cycles(&[]).is_err());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn starter_argv_carries_np() {
        assert_eq!(starter_argv("a.rad", None, &[]), vec!["-i", "a.rad"]);
        assert_eq!(
            starter_argv("a.rad", Some(2), &["-x".to_string()]),
            vec!["-i", "a.rad", "-np", "2", "-x"]
        );
    }

    #[test]
    fn history_lerp_interpolates_and_rejects_extrapolation() {
        let times = [0.0, 1.0, 2.0];
        let values = [0.0, 10.0, 20.0];
        assert_eq!(history_lerp(&times, &values, 0.5).unwrap(), 5.0);
        assert_eq!(history_lerp(&times, &values, 2.0).unwrap(), 20.0);
        assert!(history_lerp(&times, &values, -0.1).is_none());
        assert!(history_lerp(&times, &values, 2.1).is_none());
        assert!(history_lerp(&[], &[], 0.0).is_none());
        assert!(history_lerp(&[0.0], &[0.0, 1.0], 0.0).is_none());
    }

    #[test]
    fn second_half_mean_skips_transient() {
        assert_eq!(second_half_mean(&[1.0, 2.0, 3.0, 4.0]).unwrap(), 3.5);
        assert_eq!(second_half_mean(&[7.0]).unwrap(), 7.0);
        assert!(second_half_mean(&[]).is_none());
    }

    #[test]
    fn history_trailmean_averages_last_n() {
        let values = [1.0, 2.0, 3.0, 4.0];
        assert_eq!(history_trailmean(&values, 2).unwrap(), 3.5);
        assert_eq!(history_trailmean(&values, 99).unwrap(), 2.5);
        assert!(history_trailmean(&[], 50).is_none());
        assert!(history_trailmean(&values, 0).is_none());
    }

    #[test]
    fn history_holdmean_averages_from_start() {
        let times = [0.0, 1.0, 2.0, 3.0];
        let values = [10.0, 20.0, 30.0, 40.0];
        assert_eq!(history_holdmean(&times, &values, 2.0).unwrap(), 35.0);
        assert!(history_holdmean(&times, &values, 9.0).is_none());
        assert!(history_holdmean(&[], &[], 0.0).is_none());
        assert!(history_holdmean(&[0.0], &[0.0, 1.0], 0.0).is_none());
    }

    #[test]
    fn thick_cylinder_times_match_engine_leg() {
        // Transcribed from fem_opencourant_plasticity_thick_cylinder:
        // 50 MPa crossing of the raised cosine plus one dilatational
        // transit (8.5 us) for the outer probe.
        let (t_elastic, t_outer) = thick_cylinder_sample_times();
        assert!((t_elastic - 3.918_266e-4).abs() < 1.0e-10);
        assert!((t_outer - t_elastic - 8.5e-6).abs() < 0.1e-6);
    }
}
