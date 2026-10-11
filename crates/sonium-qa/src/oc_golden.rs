//! OpenCourant miniqa golden harvest + offline replay (Lane 0B).
//!
//! G-tier validation rests on checked-in JSON goldens, one per miniqa deck
//! at `goldens/miniqa/<case>.json` (see `goldens/SCHEMA.md`). This module
//! owns the schema, the harvester (copy a miniqa `data/` dir to `/tmp`
//! scratch, run vendor starter+engine, convert each `T0N` with `th_to_csv`,
//! curate channels, stamp provenance), and the offline replay comparator.
//!
//! Vendor CLI/env mirrors `qa-tests/qa_tools`: starter runs as
//! `<starter> -i <deck> -np <np>`, engine as `<engine> -i <deck>` (SMP),
//! with `OMP_STACKSIZE`, `OMP_NUM_THREADS`, `RAD_CFG_PATH`, and
//! `LD_LIBRARY_PATH` set the way `qa_system.set_or_variables` sets them.
//! Only numeric facts harvested through `th_to_csv` enter the repo; no
//! vendor decks, binaries, or sources are copied in.
//!
//! Regen convention (mirrors `REGEN_DECKS=1`): with `REGEN_OC_GOLDENS=1`
//! the replay binary re-harvests from the engine; without it, missing or
//! deck-hash-stale goldens are hard errors that name the regen command.

use super::{RadError, TimeHistory, best_sibling, convert_with_th_to_csv, resolve_engine_pair};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::BTreeMap;
use std::fmt;
use std::path::{Path, PathBuf};
use std::process::Command;

/// Golden JSON schema tag. Bump on any breaking shape change.
pub const GOLDEN_SCHEMA: &str = "opencourant-golden-v1";

/// Env var that arms golden (re)generation (mirrors `REGEN_DECKS=1`).
pub const REGEN_ENV: &str = "REGEN_OC_GOLDENS";

/// Canonical `--max-samples` for the checked-in corpus (uniform for
/// every deck). Decks under the budget keep full series; the 6
/// monster decks (2.4k–12.5k samples) keep stride ≤ 7 plus
/// per-channel extrema. Corpus total ~41MB (guard: 50MB, see
/// `goldens/SCHEMA.md`). The harvester default stays 0 (full);
/// this policy lives in [`regen_command`] and the schema doc.
pub const CANONICAL_MAX_SAMPLES: usize = 2000;

/// Env var override for harvest scratch (below `--scratch-root`,
/// above the platform temp dir, which honors TMPDIR on Unix).
pub const SCRATCH_ENV: &str = "SONIUM_QA_SCRATCH";

/// Resolve the harvest scratch root: explicit flag, then
/// [`SCRATCH_ENV`], then the platform temp dir.
pub fn resolve_scratch_root(flag: Option<PathBuf>) -> PathBuf {
    let env = std::env::var(SCRATCH_ENV).ok();
    resolve_scratch_root_impl(flag, env)
}

/// Precedence core for [`resolve_scratch_root`], split out so tests
/// need not mutate the process environment.
fn resolve_scratch_root_impl(flag: Option<PathBuf>, env: Option<String>) -> PathBuf {
    if let Some(root) = flag.filter(|root| !root.as_os_str().is_empty()) {
        return root;
    }
    match env {
        Some(root) if !root.trim().is_empty() => PathBuf::from(root),
        _ => std::env::temp_dir(),
    }
}

/// Env var carrying the miniqa root (`.../qa-tests/miniqa`) when the
/// harvest/replay CLIs get no explicit `--miniqa` flag.
pub const MINIQA_ENV: &str = "OPENCOURANT_MINIQA";

/// Default per-channel relative tolerance: the vendor `Diff_tolerance`
/// from `qa-tests/qa_tools/constants.json`.
pub const DEFAULT_REL_TOL: f64 = 6.6e-4;

/// Near-zero skip band for replay, mirroring vendor `Zero_tolerance`.
pub const ZERO_TOL: f64 = 5e-14;

/// Scale for the per-channel absolute tolerance floor:
/// `abs_tol = ABS_TOL_SCALE * max(1, peak)`.
pub const ABS_TOL_SCALE: f64 = 1e-12;

/// Provenance block of a golden. Extends the `probes.json` shape with
/// everything needed to reproduce or invalidate the harvest.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenMeta {
    /// Human stamp, e.g. `OpenRadioss linux64_gf CommitID 33e68517`.
    pub engine: String,
    /// Full 40-hex engine CommitID parsed from the `.out` banner.
    pub engine_commit: String,
    /// Starter/engine executable file names that ran.
    pub starter_exe: String,
    /// Engine executable file name that ran.
    pub engine_exe: String,
    /// sha256 (hex) of the harvested deck files.
    pub deck_sha256: DeckHashes,
    /// Starter deck file name inside the miniqa `data/` dir.
    pub starter_deck: String,
    /// Engine deck file names in run order.
    pub engine_decks: Vec<String>,
    /// Exact starter argv (after the exe) used for the harvest.
    pub starter_options: String,
    /// Exact engine argv per run, joined with ` ; `.
    pub engine_options: String,
    /// Vendor MPI mode (`smp`), domain count, thread count, precision.
    pub mpi: String,
    /// MPI domain count passed to the starter (`-np`).
    pub np: usize,
    /// `OMP_NUM_THREADS` used for the harvest.
    pub threads: usize,
    /// Engine precision (`dp`).
    pub precision: String,
    /// Harvest timestamp (RFC 3339, UTC).
    pub harvest_date: String,
    /// Deck unit system parsed from `/BEGIN` (mass, length, time).
    pub deck_units: Option<[String; 3]>,
    /// Deck rootname (`BESHV44` for `BESHV44_0000.rad`).
    pub rootname: String,
    /// Whether the engine `.out` reports `NORMAL TERMINATION`.
    pub normal_termination: bool,
    /// `th_to_csv` binary file name used for the T01 conversion.
    pub th_tool: String,
    /// sha256 of the binaries that produced the golden.
    pub exe_sha256: ExeHashes,
    /// Peak-preserving decimation applied at harvest (`None` = full).
    pub decimation: Option<Decimation>,
    /// Why this golden carries no TH grid. `None` means a full TH
    /// golden; `Some` marks a cycle-only golden (the converter could
    /// not produce a CSV) and explains why. Skipped when `None`, so
    /// full goldens keep their exact v1 shape.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub th_note: Option<String>,
    /// Local vendor-tree deviations this golden was harvested from
    /// (`None` = deck ran as shipped). Set explicitly via
    /// `--local-patches`; see `goldens/miniqa/PATCHES.md`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub local_patches: Option<String>,
}

/// sha256 (hex) of the harvest binaries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExeHashes {
    /// Starter binary digest.
    pub starter: String,
    /// Engine binary digest.
    pub engine: String,
    /// `th_to_csv` binary digest.
    pub th_to_csv: String,
}

/// Peak-preserving decimation record: uniform stride plus the kept
/// per-channel extrema (see [`decimate_history`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Decimation {
    /// Uniform stride over the harvest grid.
    pub stride: usize,
    /// Samples kept in the golden.
    pub kept: usize,
    /// Harvest-grid samples before decimation.
    pub of: usize,
}

/// Deck hashes. `engine` is `None` only for starter-only decks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeckHashes {
    /// sha256 hex of the `_0000.rad` starter deck.
    pub starter: String,
    /// sha256 hex of the concatenated `_000N.rad` engine decks, in run
    /// order (hash of the byte concatenation, not of each file).
    pub engine: Option<String>,
}

/// One curated channel: full series plus key scalars and tolerances.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldenChannel {
    /// Samples in golden time order, one per `Golden::time` stamp.
    pub values: Vec<f64>,
    /// Last sample.
    #[serde(rename = "final")]
    pub final_value: f64,
    /// Minimum sample.
    pub min: f64,
    /// Maximum sample.
    pub max: f64,
    /// Absolute tolerance for replay.
    pub abs_tol: f64,
    /// Relative tolerance for replay.
    pub rel_tol: f64,
}

/// A checked-in golden: schema tag, provenance, shared time grid,
/// curated channels, and the engine last-cycle scalars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Golden {
    /// Must equal [`GOLDEN_SCHEMA`].
    pub schema: String,
    /// miniqa-relative case dir, e.g. `LOIS/LOI2/BESH/input_44`.
    pub case: String,
    /// Harvest provenance.
    pub meta: GoldenMeta,
    /// Shared time grid in seconds (deck time unit).
    pub time: Vec<f64>,
    /// Curated channels keyed by whitespace-collapsed `th_to_csv`
    /// header (see `SCHEMA.md` for the dedup rule).
    pub channels: BTreeMap<String, GoldenChannel>,
    /// Last explicit-cycle scalars from the engine `.out`
    /// (`CYCLE`, `TIME`, `TSTEP`, `IENERGY`, ...).
    pub cycle: BTreeMap<String, f64>,
}

/// Regen command named by every stale/missing-golden error. Carries
/// the canonical corpus policy ([`CANONICAL_MAX_SAMPLES`]) so a
/// single-deck regen matches the checked-in corpus shape.
pub fn regen_command(case: &str) -> String {
    format!(
        "{REGEN_ENV}=1 cargo run -p sonium-qa --features qa-binaries --bin qa-oc-harvest -- --deck {case} --max-samples {CANONICAL_MAX_SAMPLES}"
    )
}

/// Golden path for a case under a goldens root:
/// `<root>/miniqa/<case>.json`.
pub fn golden_path(goldens_root: &Path, case: &str) -> PathBuf {
    goldens_root.join("miniqa").join(format!("{case}.json"))
}

/// True when [`REGEN_ENV`] is set to exactly `1`.
pub fn regen_armed() -> bool {
    std::env::var(REGEN_ENV).as_deref() == Ok("1")
}

/// sha256 hex digest of a file.
///
/// # Errors
///
/// Returns [`RadError::Io`] when the file cannot be read.
pub fn sha256_hex_file(path: &Path) -> Result<String, RadError> {
    let bytes = std::fs::read(path)?;
    let digest = Sha256::digest(&bytes);
    Ok(format!("{digest:x}"))
}

/// Collapse a raw `th_to_csv` header to golden key form: strip quotes,
/// collapse every whitespace run to one space, trim.
pub fn collapse_header(raw: &str) -> String {
    raw.trim()
        .trim_matches('"')
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

/// Strip the engine-local `var NN` suffix from a collapsed header.
///
/// `th_to_csv` numbers TH-group variables file-wide (`... var 37`),
/// and the numbering shifts between engine versions, so the suffix
/// carries no cross-version identity and must not enter golden keys.
/// Returns the title with the suffix removed when it matches
/// `var <digits>` as the final two tokens.
pub fn strip_var_suffix(collapsed: &str) -> String {
    let mut tokens: Vec<&str> = collapsed.split_whitespace().collect();
    if tokens.len() >= 3 {
        let last = tokens[tokens.len() - 1];
        let prev = tokens[tokens.len() - 2];
        if prev.eq_ignore_ascii_case("var")
            && !last.is_empty()
            && last.chars().all(|c| c.is_ascii_digit())
        {
            tokens.truncate(tokens.len() - 2);
            return tokens.join(" ");
        }
    }
    collapsed.to_string()
}

/// Build golden channel keys from collapsed headers: strip the
/// `var NN` suffix, then disambiguate repeated titles with a `~k`
/// ordinal (1-based file order within the title group). Unique
/// titles keep their bare name; a title whose arity changes between
/// harvests renames, which is a deliberate fail-closed regen event.
pub fn golden_channel_names(headers: &[String]) -> Vec<String> {
    let titles: Vec<String> = headers
        .iter()
        .map(|header| strip_var_suffix(header))
        .collect();
    let mut counts: BTreeMap<&str, usize> = BTreeMap::new();
    for title in &titles {
        *counts.entry(title.as_str()).or_insert(0) += 1;
    }
    let mut seen: BTreeMap<&str, usize> = BTreeMap::new();
    titles
        .iter()
        .map(|title| {
            let count = seen.entry(title.as_str()).or_insert(0);
            *count += 1;
            if counts[title.as_str()] > 1 {
                format!("{title} ~{}", *count)
            } else {
                title.clone()
            }
        })
        .collect()
}

/// Split one CSV line, honoring double-quoted cells (which may contain
/// commas and `""` escapes). Vendor headers are always quoted; data
/// cells are bare numbers.
pub fn split_csv_line(line: &str) -> Vec<String> {
    let mut cells = Vec::new();
    let mut cell = String::new();
    let mut chars = line.chars().peekable();
    let mut quoted = false;
    while let Some(char) = chars.next() {
        if quoted {
            if char == '"' {
                if chars.peek() == Some(&'"') {
                    chars.next();
                    cell.push('"');
                } else {
                    quoted = false;
                }
            } else {
                cell.push(char);
            }
        } else if char == '"' {
            quoted = true;
        } else if char == ',' {
            cells.push(std::mem::take(&mut cell));
        } else {
            cell.push(char);
        }
    }
    cells.push(cell);
    cells
}

/// Drop trailing cells that are empty after trimming.
fn strip_trailing_empty(cells: &mut Vec<String>) {
    while cells.len() > 1 && cells[cells.len() - 1].trim().is_empty() {
        cells.pop();
    }
}

/// Parse one `th_to_csv` CSV file into a [`TimeHistory`].
///
/// Unlike [`super::parse_t01`] (shaped for our own writer's decks),
/// channel keys follow [`golden_channel_names`]: collapsed vendor
/// titles with the version-fragile `var NN` suffix stripped and a
/// `~k` ordinal for repeated titles. Binary input is rejected.
/// Columns titled `(null)` (unresolvable TH requests, always empty)
/// are dropped from both header and rows.
///
/// # Errors
///
/// Returns [`RadError::Parse`] for binary input, ragged rows outside
/// dropped `(null)` columns, non-numeric cells, or a missing time
/// column.
pub fn parse_th_csv(path: &Path) -> Result<TimeHistory, RadError> {
    let bytes = std::fs::read(path)?;
    if bytes.contains(&0) {
        return Err(RadError::Parse(format!(
            "binary TH file at {} needs th_to_csv conversion first",
            path.display()
        )));
    }
    let text = String::from_utf8(bytes)
        .map_err(|_| RadError::Parse(format!("{} is not valid UTF-8", path.display())))?;
    let mut lines = text.lines().filter(|line| !line.trim().is_empty());
    let header = lines
        .next()
        .ok_or_else(|| RadError::Parse(format!("{} is empty", path.display())))?;
    // th_to_csv ends some header rows with a bare comma; trailing empty
    // cells carry no channel and are dropped (same for data rows).
    let mut raw: Vec<String> = split_csv_line(header);
    strip_trailing_empty(&mut raw);
    if raw.len() < 2 || !collapse_header(&raw[0]).to_lowercase().contains("time") {
        return Err(RadError::Parse(format!(
            "{} header must start with a time column",
            path.display()
        )));
    }
    let collapsed: Vec<String> = raw.iter().map(|cell| collapse_header(cell)).collect();
    // `th_to_csv` emits `(null) var NN` columns for TH requests it cannot
    // resolve; their data cells are empty and carry no signal, so drop
    // them (and only them) from both header and rows. Index 0 is time.
    let null_column: Vec<bool> = collapsed
        .iter()
        .enumerate()
        .map(|(index, title)| index > 0 && strip_var_suffix(title).eq_ignore_ascii_case("(null)"))
        .collect();
    let keep: Vec<usize> = (0..raw.len()).filter(|&i| !null_column[i]).collect();
    let kept_headers: Vec<String> = keep
        .iter()
        .skip(1)
        .map(|&index| collapsed[index].clone())
        .collect();
    let names = golden_channel_names(&kept_headers);
    let mut times = Vec::new();
    let mut columns: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    for (row, line) in lines.enumerate() {
        let mut cells = split_csv_line(line);
        strip_trailing_empty(&mut cells);
        if cells.len() < raw.len() {
            // A short row is only acceptable over dropped `(null)`
            // columns; any other raggedness stays a hard fail.
            while cells.len() < raw.len() && null_column[cells.len()] {
                cells.push(String::new());
            }
        }
        if cells.len() != raw.len() {
            return Err(RadError::Parse(format!(
                "{} row {} has {} cells, expected {}",
                path.display(),
                row + 2,
                cells.len(),
                raw.len()
            )));
        }
        let cells: Vec<String> = keep.iter().map(|&index| cells[index].clone()).collect();
        let time: f64 = cells[0].trim().parse().map_err(|_| {
            RadError::Parse(format!(
                "{} row {} has a bad time stamp",
                path.display(),
                row + 2
            ))
        })?;
        if !time.is_finite() {
            return Err(RadError::Parse(format!(
                "{} row {} time stamp is non-finite",
                path.display(),
                row + 2
            )));
        }
        times.push(time);
        for (column, cell) in cells.iter().skip(1).enumerate() {
            let value: f64 = cell.trim().parse().map_err(|_| {
                RadError::Parse(format!(
                    "{} row {} column {} is not a number",
                    path.display(),
                    row + 2,
                    names[column]
                ))
            })?;
            if !value.is_finite() {
                return Err(RadError::Parse(format!(
                    "{} row {} column {} is non-finite",
                    path.display(),
                    row + 2,
                    names[column]
                )));
            }
            columns[column].push(value);
        }
    }
    let mut channels = BTreeMap::new();
    for (index, name) in names.iter().enumerate() {
        channels.insert(name.clone(), std::mem::take(&mut columns[index]));
    }
    Ok(TimeHistory { times, channels })
}

/// Append run `next` to run `history`, dropping `next` rows at or
/// before the last stamp (multi-run restarts re-emit the boundary).
///
/// # Errors
///
/// Returns [`RadError::Parse`] on channel-set drift between runs or
/// an empty leading history.
pub fn concat_th_runs(history: &mut TimeHistory, next: &TimeHistory) -> Result<(), RadError> {
    if history.times.is_empty() {
        return Err(RadError::Parse(
            "cannot extend an empty time history".to_string(),
        ));
    }
    if history.channels.keys().ne(next.channels.keys()) {
        return Err(RadError::Parse(
            "TH channel set drifted between engine runs".to_string(),
        ));
    }
    let last = history.times[history.times.len() - 1];
    let mut start = next.times.len();
    for (index, time) in next.times.iter().enumerate() {
        if *time > last {
            start = index;
            break;
        }
    }
    history.times.extend_from_slice(&next.times[start..]);
    for (name, values) in &mut history.channels {
        if let Some(more) = next.channels.get(name) {
            values.extend_from_slice(&more[start..]);
        }
    }
    Ok(())
}

/// Final, minimum, and maximum of a series (zeros when empty).
pub fn channel_scalars(values: &[f64]) -> (f64, f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0, 0.0);
    }
    let mut min = values[0];
    let mut max = values[0];
    for value in &values[1..] {
        min = min.min(*value);
        max = max.max(*value);
    }
    (values[values.len() - 1], min, max)
}

/// Absolute tolerance floor from a series peak:
/// [`ABS_TOL_SCALE`] times `max(1, peak)`.
pub fn default_abs_tol(values: &[f64]) -> f64 {
    let mut peak = 0.0f64;
    for value in values {
        peak = peak.max(value.abs());
    }
    ABS_TOL_SCALE * peak.max(1.0)
}

/// Peak-preserving decimation of a history to about `max_samples`
/// stamps: uniform stride over the grid plus every channel's argmin
/// and argmax plus the endpoints, so no channel's extrema are lost.
/// `max_samples < 3` (or a grid already that short) keeps the full
/// series. Returns the decimated history and the [`Decimation`]
/// record (`None` when nothing was dropped).
pub fn decimate_history(
    history: &TimeHistory,
    max_samples: usize,
) -> (TimeHistory, Option<Decimation>) {
    let n = history.times.len();
    if max_samples < 3 || n <= max_samples {
        return (history.clone(), None);
    }
    let stride = (n - 1).div_ceil(max_samples - 1).max(1);
    let mut keep = vec![false; n];
    keep[0] = true;
    keep[n - 1] = true;
    let mut index = 0;
    while index < n {
        keep[index] = true;
        index += stride;
    }
    for values in history.channels.values() {
        if values.len() != n {
            continue;
        }
        let mut argmin = 0;
        let mut argmax = 0;
        for (i, value) in values.iter().enumerate() {
            if *value < values[argmin] {
                argmin = i;
            }
            if *value > values[argmax] {
                argmax = i;
            }
        }
        keep[argmin] = true;
        keep[argmax] = true;
    }
    let mut decimated = TimeHistory {
        times: Vec::new(),
        channels: BTreeMap::new(),
    };
    for name in history.channels.keys() {
        decimated.channels.insert(name.clone(), Vec::new());
    }
    for (i, time) in history.times.iter().enumerate() {
        if !keep[i] {
            continue;
        }
        decimated.times.push(*time);
        for (name, values) in &history.channels {
            if let Some(slot) = decimated.channels.get_mut(name)
                && let Some(value) = values.get(i)
            {
                slot.push(*value);
            }
        }
    }
    let kept = decimated.times.len();
    let record = if kept >= n {
        None
    } else {
        Some(Decimation {
            stride,
            kept,
            of: n,
        })
    };
    (decimated, record)
}

/// Linearly interpolate `values` (on non-decreasing `grid`) at `t`.
/// Returns `None` when `t` falls outside the grid span.
/// Shared crate-wide so the regen drift check interpolates identically.
pub(crate) fn interp_on_grid(grid: &[f64], values: &[f64], t: f64) -> Option<f64> {
    if grid.len() != values.len() || grid.is_empty() {
        return None;
    }
    if t < grid[0] || t > grid[grid.len() - 1] {
        return None;
    }
    let upper = grid
        .partition_point(|stamp| *stamp <= t)
        .min(grid.len() - 1);
    if upper == 0 {
        return Some(values[0]);
    }
    let (t0, t1) = (grid[upper - 1], grid[upper]);
    if t1 <= t0 {
        return Some(values[upper]);
    }
    let weight = (t - t0) / (t1 - t0);
    Some(values[upper - 1] * (1.0 - weight) + values[upper] * weight)
}

/// True when a grid is non-decreasing (interpolation-safe).
/// Shared crate-wide so the regen drift check validates identically.
pub(crate) fn grid_monotone(grid: &[f64]) -> bool {
    grid.windows(2).all(|pair| pair[1] >= pair[0])
}

/// Last explicit-cycle scalars plus the termination flag of an
/// engine `.out` file. Ports `verify_results.extract_engine_results`:
/// an explicit-cycle line has 13 whitespace fields with a `%` on the
/// 6th, and termination is the `NORMAL TERMINATION` line.
///
/// # Errors
///
/// Returns [`RadError::Parse`] when no cycle line is found.
pub fn parse_engine_out(path: &Path) -> Result<(BTreeMap<String, f64>, bool), RadError> {
    let text = std::fs::read_to_string(path)?;
    let mut last: Option<Vec<String>> = None;
    let mut normal_termination = false;
    for line in text.lines() {
        let fields: Vec<String> = line.split_whitespace().map(str::to_string).collect();
        if fields.len() == 13 && fields[5].ends_with('%') {
            last = Some(fields.clone());
        }
        if fields.len() == 2 && fields[0] == "NORMAL" && fields[1] == "TERMINATION" {
            normal_termination = true;
        }
    }
    let fields = last.ok_or_else(|| {
        RadError::Parse(format!("{} holds no explicit-cycle line", path.display()))
    })?;
    let mut scalars = BTreeMap::new();
    // Vendor keys; ELEMENT (field 3) is a type word, skipped like the
    // vendor script skips it.
    for (key, index) in [
        ("CYCLE", 0),
        ("TIME", 1),
        ("TSTEP", 2),
        ("ELTID", 4),
        ("IENERGY", 6),
        ("KENERGYT", 7),
        ("KENERGYR", 8),
        ("EXTWORK", 9),
        ("MASERR", 10),
        ("TOTALMASS", 11),
        ("ADDEDMASS", 12),
    ] {
        let value: f64 = fields[index].parse().map_err(|_| {
            RadError::Parse(format!(
                "{} cycle line field {key} is not a number",
                path.display()
            ))
        })?;
        scalars.insert(key.to_string(), value);
    }
    let error: f64 = fields[5].trim_end_matches('%').parse().map_err(|_| {
        RadError::Parse(format!(
            "{} cycle line field ERROR is not a number",
            path.display()
        ))
    })?;
    scalars.insert("ERROR".to_string(), error);
    Ok((scalars, normal_termination))
}

/// Engine identity parsed from a starter `.out` banner.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineId {
    /// Product word from the banner (`OpenRadioss`).
    pub product: String,
    /// Full CommitID hex from the banner.
    pub commit: String,
}

/// Parse the product name and CommitID from a starter `.out` banner.
///
/// # Errors
///
/// Returns [`RadError::Parse`] when the banner carries no CommitID.
pub fn parse_engine_id(starter_out: &Path) -> Result<EngineId, RadError> {
    let text = std::fs::read_to_string(starter_out)?;
    let mut product = String::new();
    let mut commit = String::new();
    for line in text.lines() {
        if line.contains("CommitID:")
            && let Some(hex) = line.split("CommitID:").nth(1)
        {
            let token = hex.split_whitespace().next().unwrap_or_default();
            if !token.is_empty() {
                commit = token.to_string();
            }
        }
        if product.is_empty() && line.contains("Starter") && line.contains("**") {
            let words: Vec<&str> = line.split_whitespace().collect();
            if words.len() >= 3 {
                product = words[1].to_string();
            }
        }
    }
    if commit.is_empty() {
        return Err(RadError::Parse(format!(
            "{} banner holds no CommitID",
            starter_out.display()
        )));
    }
    if product.is_empty() {
        product = "OpenRadioss".to_string();
    }
    Ok(EngineId { product, commit })
}

/// Build the `meta.engine` stamp: `<product> <arch> CommitID <short>`,
/// where `<arch>` is the engine exe stem after the `engine_` prefix.
pub fn engine_stamp(id: &EngineId, engine_exe: &Path) -> String {
    let stem = engine_exe
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or_default();
    let arch = stem.strip_prefix("engine_").unwrap_or(stem);
    let short = id.commit.chars().take(8).collect::<String>();
    format!("{} {arch} CommitID {short}", id.product)
}

/// Parse the deck unit system from the `/BEGIN` card of a starter deck:
/// title, invers, then the input-unit line (mass, length, time).
/// Returns `None` when the card layout is unrecognized.
pub fn parse_deck_units(starter_deck: &Path) -> Option<[String; 3]> {
    let text = std::fs::read_to_string(starter_deck).ok()?;
    let lines: Vec<&str> = text.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        if line.trim() == "/BEGIN" && index + 3 < lines.len() {
            let words: Vec<&str> = lines[index + 3].split_whitespace().collect();
            if words.len() == 3 {
                return Some([
                    words[0].to_string(),
                    words[1].to_string(),
                    words[2].to_string(),
                ]);
            }
        }
    }
    None
}

/// A miniqa deck resolved for harvest: case-relative paths plus the
/// vendor `test_files.json` overrides when present.
#[derive(Debug, Clone)]
pub struct MiniqaDeck {
    /// miniqa-relative case dir.
    pub case: String,
    /// Absolute `data/` dir (never written to; copied to scratch).
    pub data_dir: PathBuf,
    /// Deck rootname (`BESHV44`).
    pub rootname: String,
    /// Starter deck file name.
    pub starter_deck: String,
    /// Engine deck file names in run order.
    pub engine_decks: Vec<String>,
    /// Extra starter argv from `starter_argument` (whitespace-split).
    pub starter_args: Vec<String>,
    /// Starter `-np` value (`np` key, default 1).
    pub np: usize,
}

/// Resolve a miniqa deck for harvest.
///
/// Honors `data/test_files.json` (`starter`, `engines`, `enabled`,
/// `run_starter`, `run_engine`, `starter_argument`, `np`) the way
/// `or_execute.py` does; without it, `*_0000.rad` is the starter and
/// `*_0001.rad` (or the starter itself, vendor one-file fallback) is
/// the engine deck.
///
/// # Errors
///
/// Returns [`RadError::Parse`] for disabled decks, skipped stages
/// (v0 requires starter+engine), missing decks, or undeclared
/// multi-run decks.
pub fn find_deck(miniqa_root: &Path, case: &str) -> Result<MiniqaDeck, RadError> {
    let data_dir = miniqa_root.join(case).join("data");
    if !data_dir.is_dir() {
        return Err(RadError::Parse(format!(
            "miniqa case {case} has no data dir at {}",
            data_dir.display()
        )));
    }
    let json_path = data_dir.join("test_files.json");
    if json_path.is_file() {
        return find_deck_from_json(case, &data_dir, &json_path);
    }
    let mut starter: Option<String> = None;
    let mut engine: Option<String> = None;
    let mut extra_runs = false;
    let mut entries: Vec<String> = std::fs::read_dir(&data_dir)?
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .collect();
    entries.sort();
    for name in &entries {
        if name.ends_with("_0000.rad") {
            starter = Some(name.clone());
        } else if name.ends_with("_0001.rad") {
            engine = Some(name.clone());
        } else if name.len() > 9 && name.ends_with(".rad") {
            let suffix = &name[name.len() - 9..name.len() - 4];
            if suffix.starts_with('_') && suffix[1..].chars().all(|c| c.is_ascii_digit()) {
                extra_runs = true;
            }
        }
    }
    let starter_deck = starter.ok_or_else(|| {
        RadError::Parse(format!(
            "{} holds no *_0000.rad starter deck",
            data_dir.display()
        ))
    })?;
    if extra_runs {
        return Err(RadError::Parse(format!(
            "{case} has multi-run decks without test_files.json; declare the run order there"
        )));
    }
    // Vendor one-file fallback: the starter deck doubles as the engine
    // deck when no _0001.rad exists.
    let engine_deck = engine.unwrap_or_else(|| starter_deck.clone());
    let rootname = rootname_of(&starter_deck).ok_or_else(|| {
        RadError::Parse(format!("starter deck {starter_deck} has no _NNNN.rad stem"))
    })?;
    Ok(MiniqaDeck {
        case: case.to_string(),
        data_dir,
        rootname,
        starter_deck,
        engine_decks: vec![engine_deck],
        starter_args: Vec::new(),
        np: 1,
    })
}

/// Strip the `_NNNN.rad` run suffix from a deck file name.
fn rootname_of(deck: &str) -> Option<String> {
    let stem = deck.strip_suffix(".rad")?;
    if stem.len() < 6 {
        return None;
    }
    let (root, suffix) = stem.split_at(stem.len() - 5);
    if !suffix.starts_with('_') || !suffix[1..].chars().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(root.to_string())
}

fn find_deck_from_json(
    case: &str,
    data_dir: &Path,
    json_path: &Path,
) -> Result<MiniqaDeck, RadError> {
    let text = std::fs::read_to_string(json_path)?;
    let json: serde_json::Value = serde_json::from_str(&text).map_err(|error| {
        RadError::Parse(format!(
            "{} is not valid JSON: {error}",
            json_path.display()
        ))
    })?;
    if json.get("enabled").and_then(serde_json::Value::as_i64) == Some(0) {
        return Err(RadError::Parse(format!("miniqa case {case} is disabled")));
    }
    for key in ["run_starter", "run_engine"] {
        if json.get(key).and_then(serde_json::Value::as_i64) == Some(0) {
            return Err(RadError::Parse(format!(
                "miniqa case {case} sets {key}=0; the v0 harvester requires starter+engine stages"
            )));
        }
    }
    let starter_deck = json
        .get("starter")
        .and_then(serde_json::Value::as_str)
        .ok_or_else(|| RadError::Parse(format!("{} has no starter entry", json_path.display())))?
        .to_string();
    let engines = json
        .get("engines")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| RadError::Parse(format!("{} has no engines entry", json_path.display())))?;
    let mut engine_decks = Vec::new();
    for deck in engines {
        let name = deck.as_str().ok_or_else(|| {
            RadError::Parse(format!(
                "{} has a non-string engine entry",
                json_path.display()
            ))
        })?;
        engine_decks.push(name.to_string());
    }
    if engine_decks.is_empty() {
        return Err(RadError::Parse(format!(
            "{} declares no engine decks",
            json_path.display()
        )));
    }
    for name in std::iter::once(&starter_deck).chain(&engine_decks) {
        if !data_dir.join(name).is_file() {
            return Err(RadError::Parse(format!(
                "{} lists missing deck {name}",
                json_path.display()
            )));
        }
    }
    let starter_args: Vec<String> = json
        .get("starter_argument")
        .and_then(serde_json::Value::as_str)
        .map(|args| args.split_whitespace().map(str::to_string).collect())
        .unwrap_or_default();
    let np = json
        .get("np")
        .and_then(serde_json::Value::as_i64)
        .map(|np| np.max(1) as usize)
        .unwrap_or(1);
    let rootname = rootname_of(&starter_deck).ok_or_else(|| {
        RadError::Parse(format!(
            "{} starter {starter_deck} has no _NNNN.rad stem",
            json_path.display()
        ))
    })?;
    Ok(MiniqaDeck {
        case: case.to_string(),
        data_dir: data_dir.to_path_buf(),
        rootname,
        starter_deck,
        engine_decks,
        starter_args,
        np,
    })
}

/// Provenance inputs for [`golden_from_history`].
pub struct GoldenInputs {
    /// miniqa-relative case dir.
    pub case: String,
    /// Engine stamp from [`engine_stamp`].
    pub engine: String,
    /// Full CommitID hex.
    pub engine_commit: String,
    /// Starter/engine exe file names.
    pub starter_exe: String,
    /// Engine exe file name.
    pub engine_exe: String,
    /// Deck hashes.
    pub deck_sha256: DeckHashes,
    /// Starter deck file name.
    pub starter_deck: String,
    /// Engine deck file names in run order.
    pub engine_decks: Vec<String>,
    /// Exact starter/engine argv strings.
    pub starter_options: String,
    /// Engine argv per run joined with ` ; `.
    pub engine_options: String,
    /// Starter `-np` value.
    pub np: usize,
    /// `OMP_NUM_THREADS` used.
    pub threads: usize,
    /// Deck unit system, if parsed.
    pub deck_units: Option<[String; 3]>,
    /// Deck rootname.
    pub rootname: String,
    /// Termination flag from the engine `.out`.
    pub normal_termination: bool,
    /// `th_to_csv` binary file name.
    pub th_tool: String,
    /// sha256 of the harvest binaries.
    pub exe_sha256: ExeHashes,
    /// Decimation applied to the history (`None` = full series).
    pub decimation: Option<Decimation>,
    /// TH-absence note for cycle-only goldens (`None` = full TH).
    pub th_note: Option<String>,
    /// Local vendor-tree deviations (`None` = ran as shipped).
    pub local_patches: Option<String>,
    /// Last-cycle scalars from the engine `.out`.
    pub cycle: BTreeMap<String, f64>,
}

/// Build a [`Golden`] from a harvested history plus provenance.
///
/// An empty history is only accepted with a [`GoldenInputs::th_note`],
/// producing a cycle-only golden (no TH grid, cycle scalars only).
///
/// # Errors
///
/// Returns [`RadError::Parse`] when the history is empty without a
/// note, or when channels disagree with the grid.
pub fn golden_from_history(
    history: &TimeHistory,
    inputs: GoldenInputs,
) -> Result<Golden, RadError> {
    let empty_grid = history.times.is_empty() || history.channels.is_empty();
    if empty_grid && inputs.th_note.is_none() {
        return Err(RadError::Parse(
            "cannot build a golden from an empty history".to_string(),
        ));
    }
    for (name, values) in &history.channels {
        if values.len() != history.times.len() {
            return Err(RadError::Parse(format!(
                "channel {name} has {} samples for {} stamps",
                values.len(),
                history.times.len()
            )));
        }
    }
    let mut channels = BTreeMap::new();
    for (name, values) in &history.channels {
        let (final_value, min, max) = channel_scalars(values);
        channels.insert(
            name.clone(),
            GoldenChannel {
                values: values.clone(),
                final_value,
                min,
                max,
                abs_tol: default_abs_tol(values),
                rel_tol: DEFAULT_REL_TOL,
            },
        );
    }
    let harvest_date = chrono::Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
    Ok(Golden {
        schema: GOLDEN_SCHEMA.to_string(),
        case: inputs.case,
        meta: GoldenMeta {
            engine: inputs.engine,
            engine_commit: inputs.engine_commit,
            starter_exe: inputs.starter_exe,
            engine_exe: inputs.engine_exe,
            deck_sha256: inputs.deck_sha256,
            starter_deck: inputs.starter_deck,
            engine_decks: inputs.engine_decks,
            starter_options: inputs.starter_options,
            engine_options: inputs.engine_options,
            mpi: "smp".to_string(),
            np: inputs.np,
            threads: inputs.threads,
            precision: "dp".to_string(),
            harvest_date,
            deck_units: inputs.deck_units,
            rootname: inputs.rootname,
            normal_termination: inputs.normal_termination,
            th_tool: inputs.th_tool,
            exe_sha256: inputs.exe_sha256,
            decimation: inputs.decimation,
            th_note: inputs.th_note,
            local_patches: inputs.local_patches,
        },
        time: history.times.clone(),
        channels,
        cycle: inputs.cycle,
    })
}

/// Serialize a golden to pretty JSON text.
pub fn golden_json(golden: &Golden) -> Result<String, RadError> {
    serde_json::to_string_pretty(golden)
        .map_err(|error| RadError::Parse(format!("golden serialization failed: {error}")))
}

/// Write a golden to `<out_dir>/miniqa/<case>.json`, creating parents.
///
/// # Errors
///
/// Returns [`RadError`] when serialization or the write fails.
pub fn write_golden(out_dir: &Path, golden: &Golden) -> Result<PathBuf, RadError> {
    let path = golden_path(out_dir, &golden.case);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let mut text = golden_json(golden)?;
    text.push('\n');
    std::fs::write(&path, text)?;
    Ok(path)
}

/// Load and validate a golden: schema tag, non-empty grid (unless a
/// cycle-only `th_note` is present), and channel/grid length agreement.
///
/// # Errors
///
/// Returns [`RadError::Parse`] for schema or shape violations, or
/// [`RadError::Io`] when the file cannot be read.
pub fn load_golden(path: &Path) -> Result<Golden, RadError> {
    let text = std::fs::read_to_string(path).map_err(|error| {
        if error.kind() == std::io::ErrorKind::NotFound {
            let case = case_from_golden_path(path);
            RadError::Parse(format!(
                "golden {} is missing; regenerate with: {}",
                path.display(),
                regen_command(&case)
            ))
        } else {
            RadError::Io(error)
        }
    })?;
    parse_golden_str(&text, &path.display().to_string())
}

/// Recover the case id from a golden path by cutting at `miniqa/` and
/// stripping `.json`; falls back to the file stem.
fn case_from_golden_path(path: &Path) -> String {
    let text = path.to_string_lossy().replace('\\', "/");
    if let Some((_, rest)) = text.split_once("miniqa/") {
        return rest.strip_suffix(".json").unwrap_or(rest).to_string();
    }
    path.file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("CASE")
        .to_string()
}

/// Options for [`compare_history`].
#[derive(Debug, Clone)]
pub struct CompareOptions {
    /// Near-zero skip band (default [`ZERO_TOL`).
    pub zero_tol: f64,
}

impl Default for CompareOptions {
    fn default() -> Self {
        Self { zero_tol: ZERO_TOL }
    }
}

/// Outcome of an offline replay: every golden channel checked, every
/// violation collected (never fail-fast, so one run reports all
/// drift). Extra provided channels are ignored: native runs may emit
/// subsets ordered differently.
#[derive(Debug, Clone)]
pub struct ReplayReport {
    /// Golden case id.
    pub case: String,
    /// Golden channels checked.
    pub channels_checked: usize,
    /// Samples compared (time grid + channels).
    pub samples_checked: u64,
    /// One entry per failing channel (or the time grid).
    pub failures: Vec<String>,
}

impl ReplayReport {
    /// True when every check passed.
    pub fn passed(&self) -> bool {
        self.failures.is_empty()
    }

    /// One-line human summary.
    pub fn summary(&self) -> String {
        if self.passed() {
            format!(
                "golden {} replay PASS: {} channels, {} samples",
                self.case, self.channels_checked, self.samples_checked
            )
        } else {
            format!(
                "golden {} replay FAIL: {} of {} channels failing ({} samples checked)",
                self.case,
                self.failures.len(),
                self.channels_checked,
                self.samples_checked
            )
        }
    }
}

/// Compare provided series against a golden with per-channel
/// tolerances.
///
/// The provided series need not share the golden grid (decimated
/// goldens, native runs with their own dt): each provided channel is
/// linearly interpolated onto the golden stamps, which the provided
/// grid must cover. A sample passes when
/// `|actual - expected| <= abs + rel * |expected|`, with the vendor
/// near-zero skip: samples where both sides sit below `zero_tol`, or
/// differ by less than twice it, are noise, not drift.
pub fn compare_history(
    golden: &Golden,
    provided: &TimeHistory,
    options: &CompareOptions,
) -> ReplayReport {
    let mut report = ReplayReport {
        case: golden.case.clone(),
        channels_checked: golden.channels.len(),
        samples_checked: 0,
        failures: Vec::new(),
    };
    if provided.times.is_empty() {
        report
            .failures
            .push("provided series has an empty time grid".to_string());
        return report;
    }
    if !grid_monotone(&provided.times) {
        report
            .failures
            .push("provided time grid is not monotone".to_string());
        return report;
    }
    let span = (provided.times[0], provided.times[provided.times.len() - 1]);
    for (name, channel) in &golden.channels {
        let Some(values) = provided.channels.get(name) else {
            report.failures.push(format!(
                "channel {name:?} missing from provided series ({} provided channels)",
                provided.channels.len()
            ));
            continue;
        };
        if values.len() != provided.times.len() {
            report.failures.push(format!(
                "channel {name:?} has {} samples for {} provided stamps",
                values.len(),
                provided.times.len()
            ));
            continue;
        }
        let mut bad = 0usize;
        let mut uncovered = 0usize;
        let mut worst_ratio = 0.0f64;
        let mut worst_time = 0.0f64;
        let mut worst_actual = 0.0f64;
        let mut worst_expected = 0.0f64;
        for (stamp, expected) in golden.time.iter().zip(&channel.values) {
            let Some(actual) = interp_on_grid(&provided.times, values, *stamp) else {
                uncovered += 1;
                continue;
            };
            report.samples_checked += 1;
            if !actual.is_finite() {
                bad += 1;
                worst_ratio = f64::INFINITY;
                worst_time = *stamp;
                worst_actual = actual;
                worst_expected = *expected;
                continue;
            }
            let diff = (actual - expected).abs();
            // Vendor Zero_tolerance skip.
            if (expected.abs() < options.zero_tol && actual.abs() < options.zero_tol)
                || diff < 2.0 * options.zero_tol
            {
                continue;
            }
            let bound = channel.abs_tol + channel.rel_tol * expected.abs();
            if diff > bound {
                bad += 1;
                let ratio = diff / bound.max(f64::MIN_POSITIVE);
                if ratio > worst_ratio {
                    worst_ratio = ratio;
                    worst_time = *stamp;
                    worst_actual = actual;
                    worst_expected = *expected;
                }
            }
        }
        if uncovered > 0 {
            report.failures.push(format!(
                "channel {name:?}: provided grid [{:.6e}..{:.6e}] misses {uncovered}/{} golden stamps",
                span.0,
                span.1,
                golden.time.len(),
            ));
        } else if bad > 0 {
            report.failures.push(format!(
                "channel {name:?}: {bad}/{} samples out of tolerance (worst t={worst_time:.6e}: actual {worst_actual:.6e} vs golden {worst_expected:.6e}, {worst_ratio:.2}x bound)",
                golden.time.len(),
            ));
        }
    }
    report
}

/// Compare provided last-cycle scalars against the golden `cycle` map
/// with [`DEFAULT_REL_TOL`] (absolute floor 0). Returns failure lines.
pub fn compare_cycle(golden: &Golden, provided: &BTreeMap<String, f64>) -> Vec<String> {
    let mut failures = Vec::new();
    for (key, expected) in &golden.cycle {
        match provided.get(key) {
            None => failures.push(format!("cycle scalar {key} missing from provided run")),
            Some(actual) => {
                let diff = (actual - expected).abs();
                if (expected.abs() < ZERO_TOL && actual.abs() < ZERO_TOL) || diff < 2.0 * ZERO_TOL {
                    continue;
                }
                if diff > DEFAULT_REL_TOL * expected.abs() {
                    failures.push(format!(
                        "cycle scalar {key}: actual {actual:.6e} vs golden {expected:.6e}"
                    ));
                }
            }
        }
    }
    failures
}

/// Recompute the miniqa deck hashes and compare with the golden stamp.
/// A mismatch names the regen command (fail closed on stale goldens).
///
/// # Errors
///
/// Returns [`RadError::Parse`] when a deck is unreadable or any hash
/// differs.
pub fn verify_deck_hashes(golden: &Golden, miniqa_root: &Path) -> Result<(), RadError> {
    let data_dir = miniqa_root.join(&golden.case).join("data");
    let starter_bytes = std::fs::read(data_dir.join(&golden.meta.starter_deck)).map_err(|_| {
        RadError::Parse(format!(
            "golden {} references unreadable deck {}; regenerate with: {}",
            golden.case,
            golden.meta.starter_deck,
            regen_command(&golden.case)
        ))
    })?;
    let starter_hash = format!("{:x}", Sha256::digest(&starter_bytes));
    if starter_hash != golden.meta.deck_sha256.starter {
        return Err(RadError::Parse(format!(
            "golden {} is stale: starter deck {} hash {} != golden {}; regenerate with: {}",
            golden.case,
            golden.meta.starter_deck,
            &starter_hash[..16],
            &golden.meta.deck_sha256.starter[..16.min(golden.meta.deck_sha256.starter.len())],
            regen_command(&golden.case)
        )));
    }
    let mut engine_bytes = Vec::new();
    for deck in &golden.meta.engine_decks {
        let bytes = std::fs::read(data_dir.join(deck)).map_err(|_| {
            RadError::Parse(format!(
                "golden {} references unreadable deck {deck}; regenerate with: {}",
                golden.case,
                regen_command(&golden.case)
            ))
        })?;
        engine_bytes.extend_from_slice(&bytes);
    }
    let engine_hash = if golden.meta.engine_decks.is_empty() {
        None
    } else {
        Some(format!("{:x}", Sha256::digest(&engine_bytes)))
    };
    if engine_hash != golden.meta.deck_sha256.engine {
        return Err(RadError::Parse(format!(
            "golden {} is stale: engine deck hash drifted; regenerate with: {}",
            golden.case,
            regen_command(&golden.case)
        )));
    }
    Ok(())
}

/// Recompute deck hashes for a resolved deck without running the engine.
///
/// Reads the starter deck plus the concatenated engine decks from the
/// miniqa `data/` dir and hashes them exactly the way [`harvest`]
/// stamps [`GoldenMeta::deck_sha256`], so regen and replay tooling can
/// pre-check staleness offline.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::oc_golden::{current_deck_hashes, find_deck};
///
/// let root = tempfile::tempdir().unwrap();
/// let data = root.path().join("C").join("data");
/// std::fs::create_dir_all(&data).unwrap();
/// std::fs::write(data.join("A_0000.rad"), "starter").unwrap();
/// std::fs::write(data.join("A_0001.rad"), "engine").unwrap();
/// let deck = find_deck(root.path(), "C").unwrap();
/// let hashes = current_deck_hashes(&deck).unwrap();
/// assert_eq!(hashes.starter.len(), 64);
/// assert!(hashes.engine.is_some());
/// ```
///
/// # Errors
///
/// Returns [`RadError::Io`] when a deck file cannot be read.
pub fn current_deck_hashes(deck: &MiniqaDeck) -> Result<DeckHashes, RadError> {
    let starter = sha256_hex_file(&deck.data_dir.join(&deck.starter_deck))?;
    let mut engine_bytes = Vec::new();
    for engine_deck in &deck.engine_decks {
        engine_bytes.extend_from_slice(&std::fs::read(deck.data_dir.join(engine_deck))?);
    }
    Ok(DeckHashes {
        starter,
        engine: Some(format!("{:x}", Sha256::digest(&engine_bytes))),
    })
}

/// Fail-closed stamp presence check for a golden.
///
/// Returns one line per missing engine-stamp or deck-hash meta field
/// (`engine`, `engine_commit`, exe names, `th_tool`, starter deck
/// name, deck hashes, binary hashes); empty means fully stamped. The
/// engine hash is required exactly when `engine_decks` is non-empty.
/// Regen drift checks call this so unstamped goldens fail loudly.
///
/// # Examples
///
/// ```rust
/// use sonium_qa::rad::oc_golden::{parse_golden_str, stamp_failures};
///
/// let text = r#"{
///   "schema": "opencourant-golden-v1",
///   "case": "DEMO/case",
///   "meta": {
///     "engine": "OpenRadioss linux64_gf CommitID deadbeef",
///     "engine_commit": "deadbeefdeadbeefdeadbeefdeadbeefdeadbeef",
///     "starter_exe": "starter_linux64_gf",
///     "engine_exe": "engine_linux64_gf",
///     "deck_sha256": {"starter": "aa", "engine": "bb"},
///     "starter_deck": "CASE_0000.rad",
///     "engine_decks": ["CASE_0001.rad"],
///     "starter_options": "-i CASE_0000.rad -np 1",
///     "engine_options": "-i CASE_0001.rad",
///     "mpi": "smp", "np": 1, "threads": 1, "precision": "dp",
///     "harvest_date": "2026-01-01T00:00:00Z",
///     "deck_units": ["kg", "m", "s"],
///     "rootname": "CASE",
///     "normal_termination": true,
///     "th_tool": "th_to_csv_linux64_gf",
///     "exe_sha256": {"starter": "s", "engine": "e", "th_to_csv": "t"},
///     "decimation": null
///   },
///   "time": [0.0, 1.0],
///   "channels": {
///     "FORCE": {"values": [1.0, 2.0], "final": 2.0, "min": 1.0,
///               "max": 2.0, "abs_tol": 2e-12, "rel_tol": 6.6e-4}
///   },
///   "cycle": {"CYCLE": 2.0}
/// }"#;
/// let golden = parse_golden_str(text, "demo").unwrap();
/// assert!(stamp_failures(&golden).is_empty());
/// ```
pub fn stamp_failures(golden: &Golden) -> Vec<String> {
    let meta = &golden.meta;
    let mut failures = Vec::new();
    for (field, value) in [
        ("meta.engine", meta.engine.as_str()),
        ("meta.engine_commit", meta.engine_commit.as_str()),
        ("meta.starter_exe", meta.starter_exe.as_str()),
        ("meta.engine_exe", meta.engine_exe.as_str()),
        ("meta.th_tool", meta.th_tool.as_str()),
        ("meta.starter_deck", meta.starter_deck.as_str()),
        (
            "meta.deck_sha256.starter",
            meta.deck_sha256.starter.as_str(),
        ),
        ("meta.exe_sha256.starter", meta.exe_sha256.starter.as_str()),
        ("meta.exe_sha256.engine", meta.exe_sha256.engine.as_str()),
        (
            "meta.exe_sha256.th_to_csv",
            meta.exe_sha256.th_to_csv.as_str(),
        ),
    ] {
        if value.is_empty() {
            failures.push(format!("golden {} {field} stamp is missing", golden.case));
        }
    }
    let engine_hash_missing = meta
        .deck_sha256
        .engine
        .as_deref()
        .is_none_or(|hash| hash.is_empty());
    if !meta.engine_decks.is_empty() && engine_hash_missing {
        failures.push(format!(
            "golden {} meta.deck_sha256.engine hash is missing",
            golden.case
        ));
    }
    failures
}

/// Harvest inputs.
#[derive(Debug, Clone)]
pub struct HarvestOptions {
    /// miniqa root (`.../qa-tests/miniqa`).
    pub miniqa_root: PathBuf,
    /// miniqa-relative case dir.
    pub case: String,
    /// Engine exe, starter exe, or the `exec/` dir itself (the
    /// starter/engine pair resolves the way [`resolve_engine_pair`]
    /// resolves it).
    pub exe: PathBuf,
    /// Scratch parent (a per-case dir is created inside; `/tmp`).
    pub scratch_root: PathBuf,
    /// `OMP_NUM_THREADS` for the run (default 1).
    pub threads: usize,
    /// Keep the scratch dir after a successful harvest.
    pub keep_scratch: bool,
    /// Peak-preserving decimation budget: about this many samples are
    /// kept per channel (0 = full series).
    pub max_samples: usize,
    /// Local vendor-tree deviations the deck carries (`None` = the
    /// deck ran as shipped). Recorded verbatim in the golden; see
    /// `goldens/miniqa/PATCHES.md`.
    pub local_patches: Option<String>,
}

impl HarvestOptions {
    /// Scratch dir for this case (freshly recreated per harvest).
    pub fn scratch_dir(&self) -> PathBuf {
        let safe = self.case.replace(['/', '\\'], "_");
        self.scratch_root
            .join(format!("oc_harvest_{safe}_{}", std::process::id()))
    }
}

/// Copy a directory tree (the vendor `data/` dir into scratch).
fn copy_dir_all(from: &Path, to: &Path) -> Result<(), RadError> {
    std::fs::create_dir_all(to)?;
    let mut entries: Vec<std::fs::DirEntry> = std::fs::read_dir(from)?.flatten().collect();
    entries.sort_by_key(std::fs::DirEntry::file_name);
    for entry in entries {
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir_all(&entry.path(), &target)?;
        } else {
            std::fs::copy(entry.path(), &target)?;
        }
    }
    Ok(())
}

/// Apply the vendor environment to a stage command, mirroring
/// `qa_system.set_or_variables`: preset variables win, derived paths
/// apply only when they exist.
fn apply_vendor_env(command: &mut Command, exec_dir: &Path, threads: usize) {
    if std::env::var("OMP_STACKSIZE").is_err() {
        command.env("OMP_STACKSIZE", "400m");
    }
    command.env("OMP_NUM_THREADS", threads.to_string());
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

/// Resolve the `th_to_csv` tool path, mirroring
/// [`convert_with_th_to_csv`]'s candidate order (beside the engine,
/// then `PATH`).
fn resolve_th_tool(exe: &Path) -> Option<PathBuf> {
    let dir = if exe.is_dir() {
        Some(exe.to_path_buf())
    } else {
        exe.parent().map(Path::to_path_buf)
    };
    if let Some(dir) = dir {
        let plain = dir.join("th_to_csv");
        if plain.is_file() {
            return Some(plain);
        }
        if let Some(suffixed) = best_sibling(&dir, "th_to_csv")
            && suffixed
                .file_name()
                .map(|name| name != "th_to_csv")
                .unwrap_or(false)
        {
            return Some(suffixed);
        }
    }
    std::env::var("PATH").ok().and_then(|paths| {
        std::env::split_paths(&paths)
            .map(|dir| dir.join("th_to_csv"))
            .find(|tool| tool.is_file())
    })
}

/// Resolve the `th_to_csv` tool display name beside the engine.
fn th_tool_name(exe: &Path) -> String {
    let dir = if exe.is_dir() {
        Some(exe.to_path_buf())
    } else {
        exe.parent().map(Path::to_path_buf)
    };
    dir.and_then(|dir| {
        if dir.join("th_to_csv").is_file() {
            Some("th_to_csv".to_string())
        } else {
            best_sibling(&dir, "th_to_csv").and_then(|tool| {
                tool.file_name()
                    .and_then(|name| name.to_str())
                    .map(str::to_string)
            })
        }
    })
    .unwrap_or_else(|| "th_to_csv".to_string())
}

/// Harvest a golden for one miniqa deck: copy `data/` to fresh
/// scratch, run starter then each engine deck, require `NORMAL
/// TERMINATION`, convert every `T0N` with `th_to_csv`, and curate the
/// golden. Scratch survives failures for forensics; on success it is
/// removed unless `keep_scratch` is set.
///
/// # Errors
///
/// Returns [`RadError`] when the deck is unusable, a stage fails, the
/// run does not terminate normally, or conversion/parsing fails.
pub fn harvest(options: &HarvestOptions) -> Result<Golden, RadError> {
    harvest_impl(options, &|_| {})
}

/// One vendor-pipeline progress event during [`harvest_with_hook`].
#[derive(Debug, Clone)]
pub struct HarvestEvent {
    /// miniqa-relative case dir under harvest.
    pub case: String,
    /// Pipeline stage just reached.
    pub stage: HarvestStage,
    /// Stage detail (deck name, TH file count, channel/sample counts).
    pub detail: String,
}

/// Pipeline stages reported through [`harvest_with_hook`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HarvestStage {
    /// Harvest started (scratch copy and starter launch follow).
    Start,
    /// Starter run finished.
    StarterDone,
    /// One engine deck run finished (detail names the deck).
    EngineRunDone,
    /// Every `T0N` converted and concatenated.
    Converted,
    /// Golden curated (scratch already cleaned unless kept).
    Done,
}

impl fmt::Display for HarvestStage {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Start => write!(f, "start"),
            Self::StarterDone => write!(f, "starter-done"),
            Self::EngineRunDone => write!(f, "engine-run-done"),
            Self::Converted => write!(f, "converted"),
            Self::Done => write!(f, "done"),
        }
    }
}

/// Harvest a golden exactly like [`harvest`], reporting progress.
///
/// The hook runs on the calling thread after each pipeline stage; it
/// must not block. Corpus regen drives this so one command can report
/// per-deck progress across dozens of engine runs.
///
/// # Errors
///
/// Returns [`RadError`] under the same conditions as [`harvest`].
pub fn harvest_with_hook(
    options: &HarvestOptions,
    hook: impl Fn(&HarvestEvent),
) -> Result<Golden, RadError> {
    harvest_impl(options, &hook)
}

fn harvest_impl(
    options: &HarvestOptions,
    hook: &dyn Fn(&HarvestEvent),
) -> Result<Golden, RadError> {
    hook(&HarvestEvent {
        case: options.case.clone(),
        stage: HarvestStage::Start,
        detail: String::new(),
    });
    let deck = find_deck(&options.miniqa_root, &options.case)?;
    let (starter_exe, engine_exe) = resolve_engine_pair(&options.exe)?;
    let exec_dir = if options.exe.is_dir() {
        options.exe.clone()
    } else {
        options
            .exe
            .parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(|| options.exe.clone())
    };
    let scratch = options.scratch_dir();
    if scratch.exists() {
        std::fs::remove_dir_all(&scratch)?;
    }
    copy_dir_all(&deck.data_dir, &scratch)?;
    let threads = options.threads.max(1);

    let mut starter_args = vec![
        "-i".to_string(),
        deck.starter_deck.clone(),
        "-np".to_string(),
        deck.np.to_string(),
    ];
    starter_args.extend(deck.starter_args.clone());
    run_vendor_stage(
        &starter_exe,
        &starter_args,
        &scratch,
        &exec_dir,
        threads,
        "starter_output.log",
        "starter",
    )?;
    hook(&HarvestEvent {
        case: deck.case.clone(),
        stage: HarvestStage::StarterDone,
        detail: deck.starter_deck.clone(),
    });

    for engine_deck in &deck.engine_decks {
        let args = vec!["-i".to_string(), engine_deck.clone()];
        run_vendor_stage(
            &engine_exe,
            &args,
            &scratch,
            &exec_dir,
            threads,
            "engine_output.log",
            "engine",
        )?;
        hook(&HarvestEvent {
            case: deck.case.clone(),
            stage: HarvestStage::EngineRunDone,
            detail: engine_deck.clone(),
        });
    }

    let outcome = harvest_from_scratch(options, &deck, &starter_exe, &engine_exe, &scratch, hook);
    match outcome {
        Ok(golden) => {
            if !options.keep_scratch {
                let _ = std::fs::remove_dir_all(&scratch);
            }
            hook(&HarvestEvent {
                case: deck.case.clone(),
                stage: HarvestStage::Done,
                detail: format!(
                    "{} channels x {} samples",
                    golden.channels.len(),
                    golden.time.len()
                ),
            });
            Ok(golden)
        }
        Err(error) => Err(RadError::Parse(format!(
            "{error} (scratch kept at {})",
            scratch.display()
        ))),
    }
}

#[allow(clippy::too_many_arguments)]
fn run_vendor_stage(
    exe: &Path,
    args: &[String],
    dir: &Path,
    exec_dir: &Path,
    threads: usize,
    log_name: &str,
    stage: &'static str,
) -> Result<(), RadError> {
    let mut command = Command::new(exe);
    command
        .args(args)
        .current_dir(dir)
        .stdin(std::process::Stdio::null());
    apply_vendor_env(&mut command, exec_dir, threads);
    let output = command.output()?;
    let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
    combined.push_str(&String::from_utf8_lossy(&output.stderr));
    let _ = std::fs::write(dir.join(log_name), &combined);
    if !output.status.success() {
        let tail: String = combined
            .chars()
            .rev()
            .take(2000)
            .collect::<String>()
            .chars()
            .rev()
            .collect();
        return Err(RadError::EngineFailed {
            stage,
            code: output.status.code(),
            output: tail,
        });
    }
    Ok(())
}

fn harvest_from_scratch(
    options: &HarvestOptions,
    deck: &MiniqaDeck,
    starter_exe: &Path,
    engine_exe: &Path,
    scratch: &Path,
    hook: &dyn Fn(&HarvestEvent),
) -> Result<Golden, RadError> {
    let runs = deck.engine_decks.len();
    let engine_out = scratch.join(format!("{}_{runs:04}.out", deck.rootname));
    let (cycle, normal_termination) = parse_engine_out(&engine_out)?;
    if !normal_termination {
        return Err(RadError::Parse(format!(
            "engine run for {} ended without NORMAL TERMINATION ({})",
            deck.case,
            engine_out.display()
        )));
    }
    // T files are collected by existence, not run count: most
    // multi-run decks write T01..T0N, but some runs append to an
    // earlier file instead.
    let mut th_files = Vec::new();
    for run in 1..=runs.max(1) {
        let th = scratch.join(format!("{}T{run:02}", deck.rootname));
        if th.is_file() {
            th_files.push(th);
        }
    }
    if th_files.is_empty() {
        return Err(RadError::Parse(format!(
            "engine runs for {} produced no {}Txx file",
            deck.case, deck.rootname
        )));
    }
    let mut history: Option<TimeHistory> = None;
    let mut th_note: Option<String> = None;
    for th in &th_files {
        if let Err(error) = convert_with_th_to_csv(engine_exe, th, scratch) {
            // The converter crashed or refused this T01 (seen on
            // degenerate TH files): keep a cycle-only golden rather
            // than no golden. Any partial TH is dropped with it, so a
            // golden never mixes converted and missing runs.
            th_note = Some(format!(
                "th_to_csv failed on {} ({error}); cycle scalars only",
                th.file_name().unwrap_or_default().to_string_lossy()
            ));
            break;
        }
        let csv = PathBuf::from(format!("{}.csv", th.display()));
        let csv_usable = csv.is_file()
            && std::fs::metadata(&csv)
                .map(|meta| meta.len() > 0)
                .unwrap_or(false);
        if !csv_usable {
            th_note = Some(format!(
                "th_to_csv produced no usable CSV for {}; cycle scalars only",
                deck.case
            ));
            break;
        }
        let run_history = parse_th_csv(&csv)?;
        match &mut history {
            None => history = Some(run_history),
            Some(first) => concat_th_runs(first, &run_history)?,
        }
    }
    hook(&HarvestEvent {
        case: deck.case.clone(),
        stage: HarvestStage::Converted,
        detail: format!("{} TH files", th_files.len()),
    });
    let mut history = if th_note.is_some() {
        TimeHistory::default()
    } else {
        history.ok_or_else(|| RadError::Parse("no engine runs harvested".to_string()))?
    };
    if th_note.is_none() && !history.times.is_empty() && history.channels.is_empty() {
        th_note = Some("TH converted but every column was (null); cycle scalars only".to_string());
        history = TimeHistory::default();
    }
    let (history, decimation) = decimate_history(&history, options.max_samples);
    let starter_out = scratch.join(format!("{}_0000.out", deck.rootname));
    let id = parse_engine_id(&starter_out)?;
    let th_tool = resolve_th_tool(&options.exe).ok_or_else(|| {
        RadError::Parse("th_to_csv resolved for conversion but not for hashing".to_string())
    })?;
    let exe_sha256 = ExeHashes {
        starter: sha256_hex_file(starter_exe)?,
        engine: sha256_hex_file(engine_exe)?,
        th_to_csv: sha256_hex_file(&th_tool)?,
    };
    let starter_hash = sha256_hex_file(&deck.data_dir.join(&deck.starter_deck))?;
    let mut engine_bytes = Vec::new();
    for engine_deck in &deck.engine_decks {
        engine_bytes.extend_from_slice(&std::fs::read(deck.data_dir.join(engine_deck))?);
    }
    let engine_options = deck
        .engine_decks
        .iter()
        .map(|deck| format!("-i {deck}"))
        .collect::<Vec<_>>()
        .join(" ; ");
    let mut starter_argv = vec![
        "-i".to_string(),
        deck.starter_deck.clone(),
        "-np".to_string(),
        deck.np.to_string(),
    ];
    starter_argv.extend(deck.starter_args.clone());
    golden_from_history(
        &history,
        GoldenInputs {
            case: deck.case.clone(),
            engine: engine_stamp(&id, engine_exe),
            engine_commit: id.commit,
            starter_exe: exe_file_name(starter_exe).to_string(),
            engine_exe: exe_file_name(engine_exe).to_string(),
            deck_sha256: DeckHashes {
                starter: starter_hash,
                engine: Some(format!("{:x}", Sha256::digest(&engine_bytes))),
            },
            starter_deck: deck.starter_deck.clone(),
            engine_decks: deck.engine_decks.clone(),
            starter_options: starter_argv.join(" "),
            engine_options,
            np: deck.np,
            threads: options.threads.max(1),
            deck_units: parse_deck_units(&deck.data_dir.join(&deck.starter_deck)),
            rootname: deck.rootname.clone(),
            normal_termination,
            th_tool: th_tool_name(&options.exe),
            exe_sha256,
            decimation,
            th_note,
            local_patches: options.local_patches.clone(),
            cycle,
        },
    )
}

fn exe_file_name(exe: &Path) -> &str {
    exe.file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
}

/// Parse and validate golden JSON text (shared by [`load_golden`] and
/// the `include_str!` replay tests).
///
/// # Errors
///
/// Returns [`RadError::Parse`] for invalid JSON or schema/shape
/// violations.
pub fn parse_golden_str(text: &str, origin: &str) -> Result<Golden, RadError> {
    let golden: Golden = serde_json::from_str(text)
        .map_err(|error| RadError::Parse(format!("golden {origin} is not valid JSON: {error}")))?;
    if golden.schema != GOLDEN_SCHEMA {
        return Err(RadError::Parse(format!(
            "golden {origin} has schema {}, expected {GOLDEN_SCHEMA}",
            golden.schema
        )));
    }
    if (golden.time.is_empty() || golden.channels.is_empty()) && golden.meta.th_note.is_none() {
        return Err(RadError::Parse(format!(
            "golden {origin} has an empty time grid or no channels"
        )));
    }
    for (name, channel) in &golden.channels {
        if channel.values.len() != golden.time.len() {
            return Err(RadError::Parse(format!(
                "golden {origin} channel {name} has {} samples for {} stamps",
                channel.values.len(),
                golden.time.len()
            )));
        }
    }
    Ok(golden)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_history() -> TimeHistory {
        TimeHistory {
            times: vec![0.0, 0.5, 1.0],
            channels: BTreeMap::from([
                ("ENERGY".to_string(), vec![0.0, 1.0, 4.0]),
                ("FORCE".to_string(), vec![10.0, 9.0, 8.0]),
            ]),
        }
    }

    fn sample_inputs() -> GoldenInputs {
        GoldenInputs {
            case: "TEST/case".to_string(),
            engine: "OpenRadioss linux64_gf CommitID deadbeef".to_string(),
            engine_commit: "deadbeef".to_string(),
            starter_exe: "starter_linux64_gf".to_string(),
            engine_exe: "engine_linux64_gf".to_string(),
            deck_sha256: DeckHashes {
                starter: "aa".to_string(),
                engine: Some("bb".to_string()),
            },
            starter_deck: "CASE_0000.rad".to_string(),
            engine_decks: vec!["CASE_0001.rad".to_string()],
            starter_options: "-i CASE_0000.rad -np 1".to_string(),
            engine_options: "-i CASE_0001.rad".to_string(),
            np: 1,
            threads: 1,
            deck_units: Some(["kg".to_string(), "m".to_string(), "s".to_string()]),
            rootname: "CASE".to_string(),
            normal_termination: true,
            th_tool: "th_to_csv_linux64_gf".to_string(),
            exe_sha256: ExeHashes {
                starter: "s".to_string(),
                engine: "e".to_string(),
                th_to_csv: "t".to_string(),
            },
            decimation: None,
            th_note: None,
            local_patches: None,
            cycle: BTreeMap::from([("CYCLE".to_string(), 3.0)]),
        }
    }

    #[test]
    fn csv_split_handles_quotes() {
        let cells = split_csv_line("\"time\",\"A, B\",\"C\"\"D\"\"\",1.5");
        assert_eq!(cells, vec!["time", "A, B", "C\"D\"", "1.5"]);
    }

    #[test]
    fn header_collapse_and_var_strip() {
        assert_eq!(collapse_header("\"INTERNAL  ENERGY \""), "INTERNAL ENERGY");
        assert_eq!(
            strip_var_suffix("BESHV31D00_th_nodes 4 var 51"),
            "BESHV31D00_th_nodes 4"
        );
        assert_eq!(strip_var_suffix("INTERNAL ENERGY"), "INTERNAL ENERGY");
        assert_eq!(strip_var_suffix("A VAR"), "A VAR");
        let names = golden_channel_names(&[
            "INTERNAL ENERGY".to_string(),
            "th_nodes 4 var 49".to_string(),
            "th_nodes 4 var 50".to_string(),
            "MASS".to_string(),
        ]);
        assert_eq!(
            names,
            vec!["INTERNAL ENERGY", "th_nodes 4 ~1", "th_nodes 4 ~2", "MASS"]
        );
    }

    #[test]
    fn parse_th_csv_reads_vendor_shape() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("X01.csv");
        std::fs::write(
            &path,
            "\"time\",\"INTERNAL  ENERGY\",\"INTERNAL ENERGY\"\n0,1,2\n1,3,4\n",
        )
        .unwrap();
        let history = parse_th_csv(&path).unwrap();
        assert_eq!(history.times, vec![0.0, 1.0]);
        assert_eq!(history.channels["INTERNAL ENERGY ~1"], vec![1.0, 3.0]);
        assert_eq!(history.channels["INTERNAL ENERGY ~2"], vec![2.0, 4.0]);
    }

    #[test]
    fn parse_th_csv_drops_trailing_comma() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("X01.csv");
        std::fs::write(&path, "\"time\",\"A\",\n0,1\n1,2\n").unwrap();
        let history = parse_th_csv(&path).unwrap();
        assert_eq!(history.times, vec![0.0, 1.0]);
        assert_eq!(history.channels["A"], vec![1.0, 2.0]);
    }

    #[test]
    fn parse_th_csv_rejects_binary() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("X01");
        std::fs::write(&path, [0u8, 1, 2, 3]).unwrap();
        assert!(parse_th_csv(&path).is_err());
    }

    #[test]
    fn parse_th_csv_drops_null_columns() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("X01.csv");
        std::fs::write(
            &path,
            "\"time\",\"FORCE var 41\",\"(null) var 42\"\n0,1,\n1,2,\n",
        )
        .unwrap();
        let history = parse_th_csv(&path).unwrap();
        assert_eq!(history.times, vec![0.0, 1.0]);
        assert_eq!(history.channels.len(), 1);
        assert_eq!(history.channels["FORCE"], vec![1.0, 2.0]);
    }

    #[test]
    fn cycle_only_golden_needs_note() {
        let empty = TimeHistory::default();
        assert!(golden_from_history(&empty, sample_inputs()).is_err());
        let mut noted = sample_inputs();
        noted.th_note = Some("converter failed; cycle scalars only".to_string());
        let golden = golden_from_history(&empty, noted).unwrap();
        assert!(golden.time.is_empty() && golden.channels.is_empty());
        let text = golden_json(&golden).unwrap();
        assert!(text.contains("th_note"));
        let back = parse_golden_str(&text, "test").unwrap();
        assert_eq!(back.case, "TEST/case");
    }

    #[test]
    fn local_patches_roundtrip() {
        let mut noted = sample_inputs();
        noted.local_patches = Some("qadiags.inc: commented header".to_string());
        let golden = golden_from_history(&sample_history(), noted).unwrap();
        assert_eq!(
            golden.meta.local_patches.as_deref(),
            Some("qadiags.inc: commented header")
        );
        let text = golden_json(&golden).unwrap();
        assert!(text.contains("local_patches"));
        let back = parse_golden_str(&text, "test").unwrap();
        assert_eq!(
            back.meta.local_patches.as_deref(),
            Some("qadiags.inc: commented header")
        );
        // Full goldens omit the key entirely (v1 shape unchanged).
        let plain = golden_from_history(&sample_history(), sample_inputs()).unwrap();
        assert!(!golden_json(&plain).unwrap().contains("local_patches"));
    }

    #[test]
    fn scratch_root_precedence() {
        let flag = PathBuf::from("/flag/scratch");
        assert_eq!(
            resolve_scratch_root_impl(Some(flag.clone()), Some("/env/scratch".to_string())),
            flag
        );
        assert_eq!(
            resolve_scratch_root_impl(None, Some("/env/scratch".to_string())),
            PathBuf::from("/env/scratch")
        );
        assert_eq!(
            resolve_scratch_root_impl(None, Some("  ".to_string())),
            std::env::temp_dir()
        );
        assert_eq!(resolve_scratch_root_impl(None, None), std::env::temp_dir());
    }

    #[test]
    fn concat_th_runs_dedups_boundary() {
        let mut first = TimeHistory {
            times: vec![0.0, 1.0],
            channels: BTreeMap::from([("A".to_string(), vec![0.0, 1.0])]),
        };
        let second = TimeHistory {
            times: vec![1.0, 2.0],
            channels: BTreeMap::from([("A".to_string(), vec![1.0, 4.0])]),
        };
        concat_th_runs(&mut first, &second).unwrap();
        assert_eq!(first.times, vec![0.0, 1.0, 2.0]);
        assert_eq!(first.channels["A"], vec![0.0, 1.0, 4.0]);
    }

    #[test]
    fn engine_out_parses_cycle_and_termination() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("CASE_0001.out");
        std::fs::write(
            &path,
            "     900  0.9999E-03  0.1111E-05 SHELL          1  -0.0%  0.1738E+06   4801.       0.000      0.1786E+06   0.000      0.1920E-03   0.000    \n     NORMAL TERMINATION      \n",
        )
        .unwrap();
        let (scalars, terminated) = parse_engine_out(&path).unwrap();
        assert!(terminated);
        assert_eq!(scalars["CYCLE"], 900.0);
        assert!((scalars["IENERGY"] - 0.1738e6).abs() < 1.0);
        assert_eq!(scalars["ERROR"], -0.0);
    }

    #[test]
    fn engine_id_parses_banner() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("CASE_0000.out");
        std::fs::write(
            &path,
            "**                        OpenRadioss Starter                         **\n** CommitID: 33e685176cccf0c539a3ce07aa2096985a284e2a                 **\n",
        )
        .unwrap();
        let id = parse_engine_id(&path).unwrap();
        assert_eq!(id.product, "OpenRadioss");
        assert_eq!(id.commit, "33e685176cccf0c539a3ce07aa2096985a284e2a");
        assert_eq!(
            engine_stamp(&id, Path::new("engine_linux64_gf")),
            "OpenRadioss linux64_gf CommitID 33e68517"
        );
    }

    #[test]
    fn deck_units_parse_begin_card() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("CASE_0000.rad");
        std::fs::write(&path, "/BEGIN\nCASE\n      2019         0\n                  Mg                  mm                   s\n").unwrap();
        assert_eq!(
            parse_deck_units(&path),
            Some(["Mg".to_string(), "mm".to_string(), "s".to_string()])
        );
    }

    #[test]
    fn rootname_strips_run_suffix() {
        assert_eq!(rootname_of("BESHV44_0000.rad").as_deref(), Some("BESHV44"));
        assert!(rootname_of("nope.rad").is_none());
        assert!(rootname_of("short").is_none());
    }

    #[test]
    fn find_deck_plain_and_onefile() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("C").join("data");
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(data.join("A_0000.rad"), "starter").unwrap();
        let deck = find_deck(dir.path(), "C").unwrap();
        assert_eq!(deck.rootname, "A");
        assert_eq!(deck.engine_decks, vec!["A_0000.rad".to_string()]);
        std::fs::write(data.join("A_0001.rad"), "engine").unwrap();
        let deck = find_deck(dir.path(), "C").unwrap();
        assert_eq!(deck.engine_decks, vec!["A_0001.rad".to_string()]);
    }

    #[test]
    fn find_deck_honors_json() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("C").join("data");
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(data.join("A_0000.rad"), "s").unwrap();
        std::fs::write(data.join("A_0001.rad"), "e1").unwrap();
        std::fs::write(data.join("A_0002.rad"), "e2").unwrap();
        std::fs::write(
            data.join("test_files.json"),
            r#"{"enabled": 1, "starter": "A_0000.rad", "engines": ["A_0001.rad", "A_0002.rad"], "np": 2}"#,
        )
        .unwrap();
        let deck = find_deck(dir.path(), "C").unwrap();
        assert_eq!(deck.engine_decks.len(), 2);
        assert_eq!(deck.np, 2);
    }

    #[test]
    fn find_deck_rejects_disabled_and_skipped() {
        let dir = tempfile::tempdir().unwrap();
        for (name, json) in [("D", r#"{"enabled": 0}"#), ("S", r#"{"run_starter": 0}"#)] {
            let data = dir.path().join(name).join("data");
            std::fs::create_dir_all(&data).unwrap();
            std::fs::write(data.join("test_files.json"), json).unwrap();
            assert!(find_deck(dir.path(), name).is_err());
        }
    }

    #[test]
    fn golden_replay_self_passes() {
        let golden = golden_from_history(&sample_history(), sample_inputs()).unwrap();
        let report = compare_history(&golden, &sample_history(), &CompareOptions::default());
        assert!(report.passed(), "self replay failed: {:?}", report.failures);
        assert_eq!(report.channels_checked, 2);
        assert!(compare_cycle(&golden, &golden.cycle).is_empty());
    }

    #[test]
    fn golden_replay_perturbed_fails() {
        let golden = golden_from_history(&sample_history(), sample_inputs()).unwrap();
        let mut perturbed = sample_history();
        perturbed.channels.get_mut("FORCE").unwrap()[2] *= 1.5;
        let report = compare_history(&golden, &perturbed, &CompareOptions::default());
        assert!(!report.passed());
        assert_eq!(report.failures.len(), 1);
        assert!(report.failures[0].contains("FORCE"));
    }

    #[test]
    fn golden_replay_missing_channel_and_uncovered_grid_fail() {
        let golden = golden_from_history(&sample_history(), sample_inputs()).unwrap();
        let mut missing = sample_history();
        missing.channels.remove("ENERGY");
        let report = compare_history(&golden, &missing, &CompareOptions::default());
        assert!(!report.passed());
        // Provided grid covers only the first half: golden stamps past
        // it are uncovered, which fails closed.
        let short = TimeHistory {
            times: vec![0.0, 0.25],
            channels: BTreeMap::from([
                ("ENERGY".to_string(), vec![0.0, 0.5]),
                ("FORCE".to_string(), vec![10.0, 9.5]),
            ]),
        };
        let report = compare_history(&golden, &short, &CompareOptions::default());
        assert!(!report.passed());
        assert!(report.failures[0].contains("misses"));
    }

    #[test]
    fn golden_replay_interpolates_covering_grids() {
        let golden = golden_from_history(&sample_history(), sample_inputs()).unwrap();
        // Denser grid over the same span, coinciding with the golden
        // stamps: interpolation is exact, so the replay passes; then a
        // corrupted coincident node must fail.
        let dense = TimeHistory {
            times: vec![0.0, 0.25, 0.5, 0.75, 1.0],
            channels: BTreeMap::from([
                ("ENERGY".to_string(), vec![0.0, 0.25, 1.0, 2.25, 4.0]),
                ("FORCE".to_string(), vec![10.0, 9.5, 9.0, 8.5, 8.0]),
            ]),
        };
        let report = compare_history(&golden, &dense, &CompareOptions::default());
        assert!(report.passed(), "{:?}", report.failures);
        let mut coarse = dense;
        coarse.channels.get_mut("ENERGY").unwrap()[2] = 99.0;
        let report = compare_history(&golden, &coarse, &CompareOptions::default());
        assert!(!report.passed());
    }

    #[test]
    fn decimation_keeps_extrema_and_endpoints() {
        let history = TimeHistory {
            times: (0..100).map(|i| i as f64).collect(),
            channels: BTreeMap::from([(
                "A".to_string(),
                (0..100)
                    .map(|i| if i == 37 { 999.0 } else { i as f64 })
                    .collect(),
            )]),
        };
        let (decimated, record) = decimate_history(&history, 10);
        let record = record.expect("must decimate");
        assert!(decimated.times.len() <= 10 + 2 + 1);
        assert_eq!(decimated.times[0], 0.0);
        assert_eq!(decimated.times[decimated.times.len() - 1], 99.0);
        assert!(decimated.times.contains(&37.0), "peak dropped");
        assert_eq!(record.of, 100);
        let (full, record) = decimate_history(&history, 0);
        assert!(record.is_none());
        assert_eq!(full.times.len(), 100);
    }

    #[test]
    fn golden_json_roundtrip_and_schema_gate() {
        let golden = golden_from_history(&sample_history(), sample_inputs()).unwrap();
        let text = golden_json(&golden).unwrap();
        let back = parse_golden_str(&text, "test").unwrap();
        assert_eq!(back.case, "TEST/case");
        let bad = text.replace(GOLDEN_SCHEMA, "other");
        assert!(parse_golden_str(&bad, "test").is_err());
    }

    #[test]
    fn golden_hash_check_catches_stale_decks() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("TEST").join("case").join("data");
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(data.join("CASE_0000.rad"), "starter-v1").unwrap();
        std::fs::write(data.join("CASE_0001.rad"), "engine-v1").unwrap();
        let mut golden = golden_from_history(&sample_history(), sample_inputs()).unwrap();
        golden.meta.deck_sha256.starter = sha256_hex_file(&data.join("CASE_0000.rad")).unwrap();
        let engine_bytes = std::fs::read(data.join("CASE_0001.rad")).unwrap();
        golden.meta.deck_sha256.engine = Some(format!("{:x}", Sha256::digest(&engine_bytes)));
        verify_deck_hashes(&golden, dir.path()).unwrap();
        std::fs::write(data.join("CASE_0000.rad"), "starter-v2").unwrap();
        let error = verify_deck_hashes(&golden, dir.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("stale"), "{error}");
        assert!(error.contains(REGEN_ENV), "{error}");
    }

    /// v0 offline self-test over the checked-in sample golden: the
    /// golden replays against itself, and a perturbed copy fails.
    #[test]
    fn golden_replay_sample_offline() {
        let text = include_str!("../goldens/miniqa/LOIS/LOI2/BESH/input_44.json");
        let golden = parse_golden_str(text, "sample").unwrap();
        let mut provided = TimeHistory {
            times: golden.time.clone(),
            channels: BTreeMap::new(),
        };
        for (name, channel) in &golden.channels {
            provided
                .channels
                .insert(name.clone(), channel.values.clone());
        }
        let report = compare_history(&golden, &provided, &CompareOptions::default());
        assert!(
            report.passed(),
            "sample self replay failed: {:?}",
            report.failures
        );
        let first = golden.channels.keys().next().unwrap().clone();
        let values = provided.channels.get_mut(&first).unwrap();
        let peak = values
            .iter()
            .fold(0.0f64, |peak, value| peak.max(value.abs()));
        let index = values
            .iter()
            .position(|value| value.abs() > 0.5 * peak)
            .unwrap_or(values.len() - 1);
        values[index] += peak.max(1.0);
        let report = compare_history(&golden, &provided, &CompareOptions::default());
        assert!(!report.passed(), "perturbed sample replay passed");
    }

    /// E-tier live harvest of the sample deck; skip-gated on the
    /// engine + miniqa env (mirrors the C3 binary skip convention).
    #[test]
    fn golden_harvest_sample_gated() {
        let Some(exe) = super::super::engine_exe() else {
            eprintln!("skipped: set RADISS_EXE or OPENCOURANT_EXE");
            return;
        };
        let Ok(miniqa) = std::env::var(MINIQA_ENV) else {
            eprintln!("skipped: set {MINIQA_ENV}");
            return;
        };
        let options = HarvestOptions {
            miniqa_root: PathBuf::from(miniqa),
            case: "LOIS/LOI2/BESH/input_44".to_string(),
            exe,
            scratch_root: resolve_scratch_root(None),
            threads: 1,
            keep_scratch: false,
            max_samples: 0,
            local_patches: None,
        };
        let golden = harvest(&options).expect("sample harvest failed");
        assert!(golden.meta.normal_termination);
        assert!(!golden.channels.is_empty());
        verify_deck_hashes(&golden, &options.miniqa_root).unwrap();
        let out = tempfile::tempdir().unwrap();
        let path = write_golden(out.path(), &golden).unwrap();
        let back = load_golden(&path).unwrap();
        assert_eq!(back.case, golden.case);
    }

    #[test]
    fn deck_hashes_match_harvest_stamps() {
        let dir = tempfile::tempdir().unwrap();
        let data = dir.path().join("C").join("data");
        std::fs::create_dir_all(&data).unwrap();
        std::fs::write(data.join("A_0000.rad"), "starter-bytes").unwrap();
        std::fs::write(data.join("A_0001.rad"), "engine-bytes").unwrap();
        let deck = find_deck(dir.path(), "C").unwrap();
        let hashes = current_deck_hashes(&deck).unwrap();
        assert_eq!(
            hashes.starter,
            sha256_hex_file(&data.join("A_0000.rad")).unwrap()
        );
        let engine_bytes = std::fs::read(data.join("A_0001.rad")).unwrap();
        let expected = format!("{:x}", Sha256::digest(&engine_bytes));
        assert_eq!(hashes.engine.as_deref(), Some(expected.as_str()));
        // One-file fallback: the starter doubles as the engine deck.
        std::fs::remove_file(data.join("A_0001.rad")).unwrap();
        let deck = find_deck(dir.path(), "C").unwrap();
        let hashes = current_deck_hashes(&deck).unwrap();
        let starter_bytes = std::fs::read(data.join("A_0000.rad")).unwrap();
        let expected = format!("{:x}", Sha256::digest(&starter_bytes));
        assert_eq!(hashes.engine.as_deref(), Some(expected.as_str()));
    }

    #[test]
    fn stamp_check_flags_missing_fields() {
        let golden = golden_from_history(&sample_history(), sample_inputs()).unwrap();
        assert!(stamp_failures(&golden).is_empty());
        let mut bare = golden.clone();
        bare.meta.engine.clear();
        bare.meta.engine_commit.clear();
        bare.meta.deck_sha256.engine = None;
        bare.meta.exe_sha256.th_to_csv.clear();
        let failures = stamp_failures(&bare);
        assert_eq!(failures.len(), 4);
        assert!(failures.iter().any(|line| line.contains("meta.engine ")));
        assert!(
            failures
                .iter()
                .any(|line| line.contains("deck_sha256.engine"))
        );
        // Starter-only decks need no engine hash.
        let mut solo = golden;
        solo.meta.engine_decks.clear();
        solo.meta.deck_sha256.engine = None;
        assert!(stamp_failures(&solo).is_empty());
    }

    #[test]
    fn harvest_stage_names() {
        assert_eq!(HarvestStage::Start.to_string(), "start");
        assert_eq!(HarvestStage::StarterDone.to_string(), "starter-done");
        assert_eq!(HarvestStage::EngineRunDone.to_string(), "engine-run-done");
        assert_eq!(HarvestStage::Converted.to_string(), "converted");
        assert_eq!(HarvestStage::Done.to_string(), "done");
    }
}
