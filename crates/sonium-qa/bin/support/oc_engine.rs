//! Vendor engine launch and timing machinery for OpenCourant benchmarks.
//!
//! Houses the process plumbing shared by the Navier-Stokes comparison
//! runner: side/flavor selection, starter+engine argv construction, timed
//! child execution with peak-RSS reaping, per-repeat scratch management,
//! engine `.out`/TH parsing helpers, and probe-file loading. The logic
//! mirrors `oc_bench.rs`, which predates this module and keeps its own
//! private copies; new benchmark binaries should build on this module
//! instead of duplicating the launch code a third time.

// Rust guideline compliant 2026-02-21

use clap::ValueEnum;
use sonium_qa::rad::TimeHistory;
use sonium_qa::rad::{engine_exe, resolve_engine_pair};
use sonium_qa::rad::oc_golden::resolve_scratch_root;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Vendor OpenMP stack default, mirroring the harvest precedent.
const OMP_STACKSIZE: &str = "400m";

/// Tail length kept from combined child output for failure forensics.
const OUTPUT_TAIL_CHARS: usize = 2000;

/// Which side(s) to time.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum SideOpt {
    /// Native sonium run only.
    Sonium,
    /// Vendor engine run only.
    Engine,
    /// Both arms on the same deck.
    Both,
}

/// Vendor engine launch mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub enum FlavorOpt {
    /// One process, one thread, no `-nt`.
    Serial,
    /// One process, OpenMP threads via `-nt` and `OMP_NUM_THREADS`.
    Smp,
    /// MPI ranks via `mpirun` and the `_ompi` engine binary.
    Mpi,
}

/// Resolves side selection from CLI flags.
///
/// Returns `(run sonium, run engine)`, failing closed on contradictory
/// flag combinations.
pub fn resolve_sides(
    sonium_only: bool,
    engine_only: bool,
    side: SideOpt,
) -> Result<(bool, bool), String> {
    if sonium_only && engine_only {
        return Err("--sonium-only and --engine-only contradict".to_string());
    }
    if sonium_only && side != SideOpt::Both && side != SideOpt::Sonium {
        return Err("--sonium-only contradicts --side engine".to_string());
    }
    if engine_only && side != SideOpt::Both && side != SideOpt::Engine {
        return Err("--engine-only contradicts --side sonium".to_string());
    }
    if sonium_only {
        return Ok((true, false));
    }
    if engine_only {
        return Ok((false, true));
    }
    match side {
        SideOpt::Sonium => Ok((true, false)),
        SideOpt::Engine => Ok((false, true)),
        SideOpt::Both => Ok((true, true)),
    }
}

/// Resolves repeat counts from CLI flags.
///
/// Returns `(timed repeats, warmup runs)`: three timed repeats plus one
/// warmup by default, or a single unwarmed repeat under `--quick`.
pub fn resolve_repeats(repeats: Option<u32>, quick: bool) -> Result<(u32, u32), String> {
    if quick && repeats.is_some_and(|repeats| repeats != 1) {
        return Err("--quick runs one repeat; drop --repeats".to_string());
    }
    let repeats = if quick { 1 } else { repeats.unwrap_or(3) };
    if repeats == 0 {
        return Err("--repeats must be positive".to_string());
    }
    // One timed repeat leaves no warmup to discard; otherwise one
    // warmup run absorbs cold caches and first-touch effects.
    let warmup = u32::from(repeats > 1);
    Ok((repeats, warmup))
}

/// TH sampling rule for one probe expectation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SampleRule {
    /// Read the final TH sample (settled steady flow).
    Final,
    /// Average the settled second half (ringing transient).
    Plateau,
}

/// One engine-verified probe expectation from `probes.json`.
#[derive(Debug, Clone)]
pub struct Probe {
    /// Human-readable probe label for failure messages.
    pub label: String,
    /// TH channel name, e.g. `/TH/NODE/...`.
    pub channel: String,
    /// Expected sampled value.
    pub expected: f64,
    /// Absolute tolerance around the expectation.
    pub abs_tol: f64,
    /// Relative tolerance around the expectation.
    pub rel_tol: f64,
    /// How the channel history reduces to one sample.
    pub rule: SampleRule,
}

/// Reads probe expectations from a `probes.json` file.
///
/// The optional per-probe `"rule"` selects the sampling rule
/// (`"final"` by default, `"plateau"` for the second-half mean);
/// unknown rules fail closed so a typo never silently weakens a gate.
pub fn read_probe_file(path: &Path) -> Result<Vec<Probe>, String> {
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
        let rule = match probe.get("rule").and_then(serde_json::Value::as_str) {
            None | Some("final") => SampleRule::Final,
            Some("plateau") => SampleRule::Plateau,
            Some(other) => return Err(format!("probe has unknown rule {other:?}")),
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
            rule,
        });
    }
    Ok(out)
}

/// One timed repeat: wall time plus the correctness verdict.
#[derive(Debug, Clone)]
pub struct RepeatSample {
    /// Timed wall seconds for this repeat.
    pub wall_s: f64,
    /// Completed solver steps (engine cycles, Picard iterations, BDF steps).
    pub cycles: u64,
    /// Degrees of freedom in the timed model.
    pub dof: u64,
    /// Peak resident bytes observed for this repeat.
    pub peak_rss_bytes: u64,
    /// True when every probe passed.
    pub correct: bool,
}

/// Aggregates repeats: median wall/cycles, worst RSS, AND verdict.
///
/// Sorting by wall time and taking the middle repeat resists one slow
/// outlier; RSS takes the maximum so the footprint never under-reports.
pub fn aggregate_repeats(samples: &[RepeatSample]) -> Result<RepeatSample, String> {
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

/// Output of one timed child process.
#[derive(Debug)]
pub struct TimedOutput {
    /// True when the child exited with status zero.
    pub success: bool,
    /// Exit code when the child exited normally.
    pub code: Option<i32>,
    /// Tail of combined stdout/stderr for forensics.
    pub combined_tail: String,
    /// Measured wall seconds.
    pub wall_s: f64,
    /// Peak resident bytes reaped via `wait4`.
    pub peak_rss_bytes: u64,
}

/// Runs a child, timing wall seconds and reaping peak RSS via `wait4`.
///
/// Stdout/stderr drain on reader threads while the parent blocks in
/// `wait4`, so large vendor logs never deadlock the pipes. `ru_maxrss`
/// is that child's own peak, unlike the cumulative `RUSAGE_CHILDREN`.
pub fn run_timed(
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
        .take(OUTPUT_TAIL_CHARS)
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

/// Peak RSS of this process in bytes via `getrusage`.
pub fn self_peak_rss_bytes() -> u64 {
    let mut usage: libc::rusage = unsafe { std::mem::zeroed() };
    if unsafe { libc::getrusage(libc::RUSAGE_SELF, &mut usage) } != 0 {
        return 0;
    }
    maxrss_to_bytes(usage.ru_maxrss)
}

/// Converts `ru_maxrss` to bytes (kilobytes on Linux, bytes on macOS).
pub fn maxrss_to_bytes(maxrss: libc::c_long) -> u64 {
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

/// Resolved engine launch: mode plus the thread/rank count on the row.
#[derive(Debug, Clone)]
pub struct EngineLaunch {
    /// Selected launch mode.
    pub flavor: FlavorOpt,
    /// Row flavor label (`serial`/`smp`/`mpi`, optional `_sp` suffix).
    pub label: String,
    /// Worker count recorded on the bundle row.
    pub row_threads: usize,
    /// OpenMP threads exported to the vendor stage.
    pub omp_threads: usize,
    /// MPI ranks for the engine stage.
    pub mpi_ranks: usize,
}

/// Resolves the engine launch mode and row labels from CLI flags.
pub fn resolve_launch(
    threads: usize,
    flavor: Option<FlavorOpt>,
    mpi_ranks: Option<usize>,
    engine_exe: &Path,
) -> Result<EngineLaunch, String> {
    if threads == 0 {
        return Err("--threads must be positive".to_string());
    }
    let flavor = flavor.unwrap_or(if threads > 1 {
        FlavorOpt::Smp
    } else {
        FlavorOpt::Serial
    });
    let mpi_ranks = mpi_ranks.unwrap_or(threads);
    if flavor == FlavorOpt::Mpi && mpi_ranks == 0 {
        return Err("--mpi-ranks must be positive".to_string());
    }
    if flavor == FlavorOpt::Serial && threads != 1 {
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
        FlavorOpt::Smp => (threads, threads),
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
pub fn select_mpi_engine(exe: &Path) -> Result<PathBuf, String> {
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
pub fn vendor_env(exec_dir: &Path, omp_threads: usize) -> Vec<(String, String)> {
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
pub fn engine_argv(
    launch: &EngineLaunch,
    engine_exe: &Path,
    deck: &str,
) -> (PathBuf, Vec<String>) {
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

/// Starter argv: `-np` only carries domain counts for MPI launches.
pub fn starter_argv(deck: &str, np: Option<usize>) -> Vec<String> {
    let mut args = vec!["-i".to_string(), deck.to_string()];
    if let Some(np) = np {
        args.push("-np".to_string());
        args.push(np.to_string());
    }
    args
}

/// Shared engine-arm context for one deck.
#[derive(Debug)]
pub struct EngineCtx {
    /// Resolved starter executable.
    pub starter_exe: PathBuf,
    /// Resolved engine executable.
    pub engine_exe: PathBuf,
    /// Directory holding the vendor executables.
    pub exec_dir: PathBuf,
    /// Resolved launch mode and row labels.
    pub launch: EngineLaunch,
    /// Root for per-repeat scratch dirs.
    pub scratch_root: PathBuf,
    /// Keep per-repeat scratch dirs after the run.
    pub keep_scratch: bool,
}

/// Builds the engine-arm context, resolving exes and launch mode.
pub fn engine_context(
    exe: Option<PathBuf>,
    threads: usize,
    flavor: Option<FlavorOpt>,
    mpi_ranks: Option<usize>,
    scratch_root: Option<PathBuf>,
    keep_scratch: bool,
) -> Result<EngineCtx, String> {
    let exe = exe
        .or_else(engine_exe)
        .ok_or_else(|| "engine arm needs --exe, RADISS_EXE, or OPENCOURANT_EXE".to_string())?;
    let (starter_exe, mut engine_exe) =
        resolve_engine_pair(&exe).map_err(|error| error.to_string())?;
    let provisional = resolve_launch(threads, flavor, mpi_ranks, &engine_exe)?;
    if provisional.flavor == FlavorOpt::Mpi
        && !engine_exe.to_string_lossy().to_lowercase().contains("ompi")
        && !engine_exe.to_string_lossy().to_lowercase().contains("_mpi")
    {
        engine_exe = select_mpi_engine(&exe)?;
    }
    let launch = resolve_launch(threads, flavor, mpi_ranks, &engine_exe)?;
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
        scratch_root: resolve_scratch_root(scratch_root),
        keep_scratch,
    })
}

/// Runs one vendor stage, logging combined output beside the decks.
pub fn run_vendor_stage(
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

/// Fresh per-repeat scratch dir for one deck, arm, and repeat index.
pub fn repeat_scratch(root: &Path, deck: &str, side: &str, index: usize) -> PathBuf {
    let safe = deck.replace(['/', '\\'], "_");
    root.join(format!(
        "oc_ns_bench_{safe}_{side}_{}_{index}",
        std::process::id()
    ))
}

/// Cleans a scratch dir unless the run keeps forensics.
pub fn maybe_clean(dir: &Path, keep: bool) {
    if !keep {
        let _ = std::fs::remove_dir_all(dir);
    }
}

/// Deck file set for one Navier-Stokes case.
#[derive(Debug)]
pub struct NsDecks {
    /// Starter input deck (`<deck>_0000.rad`).
    pub starter: PathBuf,
    /// Engine input deck (`<deck>_0001.rad`).
    pub engine: PathBuf,
    /// Probe expectations (`probes.json`).
    pub probes: PathBuf,
}

/// Resolves an NS case dir, failing closed on missing artifacts.
pub fn resolve_ns_decks(deck: &str) -> Result<NsDecks, String> {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("decks")
        .join(deck);
    let starter = dir.join(format!("{deck}_0000.rad"));
    let engine = dir.join(format!("{deck}_0001.rad"));
    let probes = dir.join("probes.json");
    for path in [&starter, &engine, &probes] {
        if !path.is_file() {
            return Err(format!("NS deck {deck} misses {}", path.display()));
        }
    }
    Ok(NsDecks {
        starter,
        engine,
        probes,
    })
}

/// Reads the vendor rootname from an engine deck `/RUN/<name>/1` line.
pub fn run_rootname(engine_deck_text: &str) -> Result<String, String> {
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

/// First existing path among rootname-case variants.
pub fn first_existing(dir: &Path, names: &[String]) -> Result<PathBuf, String> {
    for name in names {
        let path = dir.join(name);
        if path.is_file() {
            return Ok(path);
        }
    }
    Err(format!("none of {names:?} exists in {}", dir.display()))
}

/// Candidate `.out` names: verbatim rootname, then uppercased vendor style.
pub fn out_candidates(rootname: &str, run: usize) -> Vec<String> {
    vec![
        format!("{rootname}_{run:04}.out"),
        format!("{}_{run:04}.out", rootname.to_uppercase()),
    ]
}

/// Candidate TH names: verbatim rootname, then uppercased vendor style.
pub fn th_candidates(rootname: &str, run: usize) -> Vec<String> {
    vec![
        format!("{rootname}T{run:02}"),
        format!("{}T{run:02}", rootname.to_uppercase()),
    ]
}

/// Sums `TOTAL NUMBER OF CYCLES` over engine `.out` files.
///
/// The last printed cycle line goes stale under `/PRINT` strides (a
/// 24-cycle run with stride 1000 prints only cycle 0), while the run
/// summary carries the exact completed count, so the summary wins.
pub fn total_cycles(outs: &[PathBuf]) -> Result<u64, String> {
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

/// Counts velocity dofs from a deck: three per element-referenced node.
///
/// The starter `.out` carries no model node total, so both arms parse
/// the same staged deck instead. For ALE fluid decks this counts the
/// Eulerian mesh velocity unknowns; pressure dofs are excluded on both
/// arms' engine side, so the column stays comparable across decks.
pub fn deck_dof(starter_deck: &Path) -> Result<u64, String> {
    use sonium_qa::rad_parse::{RadDeck, TypedCard};
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
            "{} references no element nodes",
            starter_deck.display()
        ));
    }
    u64::try_from(referenced.len())
        .ok()
        .and_then(|nodes| nodes.checked_mul(3))
        .ok_or_else(|| "node count overflows dof".to_string())
}

/// Locates the `th_to_csv` converter beside the engine or on `PATH`.
pub fn find_th_converter(engine_exe: &Path) -> Result<PathBuf, String> {
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
    Ok(PathBuf::from("th_to_csv"))
}

/// True when a TH file already holds comma-separated text.
pub fn is_probably_csv(path: &Path) -> Result<bool, String> {
    let bytes =
        std::fs::read(path).map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    let head = &bytes[..bytes.len().min(4096)];
    Ok(!head.contains(&0) && head.contains(&b','))
}

/// Converts a binary T01 with `th_to_csv`, writing the sibling CSV.
pub fn convert_th(engine_exe: &Path, t01: &Path, dir: &Path) -> Result<(), String> {
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

/// Parses one converted TH file, converting binary input first.
pub fn load_th_history(engine_exe: &Path, t01: &Path, dir: &Path) -> Result<TimeHistory, String> {
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

/// Second-half mean of a channel history (settled-plateau rule).
///
/// Errors on empty histories so a missing channel never reads as zero.
pub fn plateau_mean(history: &TimeHistory, channel: &str) -> Result<f64, String> {
    let values = history
        .channel(channel)
        .map_err(|error| error.to_string())?;
    if values.is_empty() {
        return Err(format!("channel {channel} has no samples"));
    }
    let window = &values[values.len() / 2..];
    Ok(window.iter().sum::<f64>() / window.len() as f64)
}

/// First CPU model string from `/proc/cpuinfo`, else `unknown`.
pub fn cpu_model() -> String {
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
