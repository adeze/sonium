//! One-command miniqa golden regen plus offline drift check.
//!
//! Live regen re-harvests every selected deck through the existing
//! harvester (engine stamp + deck hashes on every golden), writes
//! `<out>/miniqa/<case>.json`, and diffs each fresh golden against
//! the checked-in baseline with per-channel max-abs/rel deltas; the
//! exit is nonzero when any drift exceeds the baseline's own
//! tolerances. `--drift-only` skips harvests entirely: it diffs
//! staged goldens (or self-checks the baselines) and verifies engine
//! stamps plus miniqa deck hashes, fully offline.
//!
//! ```sh
//! qa-oc-regen --all --drift-only
//! qa-oc-regen --cases LOIS/LOI2/BESH/input_44 --out /tmp/staged
//! ```

// Rust guideline compliant 2026-02-21

use clap::Parser;
use sonium_qa::rad::engine_exe;
use sonium_qa::rad::oc_golden::{CANONICAL_MAX_SAMPLES, MINIQA_ENV, resolve_scratch_root};
use sonium_qa::rad::oc_regen::{CaseSelect, RegenOptions, run_regen};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "qa-oc-regen",
    about = "Regen miniqa goldens with drift-diff vs checked-in baselines."
)]
struct Args {
    /// Comma-separated case ids (repeatable), e.g. --cases a,b.
    #[arg(long, value_delimiter = ',', num_args = 1..)]
    cases: Vec<String>,
    /// Regen every checked-in golden under <goldens>/miniqa/.
    #[arg(long)]
    all: bool,
    /// Offline only: no harvests; diff staged goldens (or self-check)
    /// and verify stamps plus deck hashes.
    #[arg(long)]
    drift_only: bool,
    /// Goldens root; defaults to the crate `goldens/` dir.
    #[arg(long)]
    goldens: Option<PathBuf>,
    /// Fresh-golden target dir; defaults to --goldens (in place).
    #[arg(long)]
    out: Option<PathBuf>,
    /// miniqa root; defaults to OPENCOURANT_MINIQA.
    #[arg(long)]
    miniqa: Option<PathBuf>,
    /// Engine/starter exe or exec dir; defaults to
    /// RADISS_EXE/OPENCOURANT_EXE (live regen only).
    #[arg(long)]
    exe: Option<PathBuf>,
    /// OMP threads per run.
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Peak-preserving decimation budget (0 = full series).
    #[arg(long, default_value_t = CANONICAL_MAX_SAMPLES)]
    max_samples: usize,
    /// Scratch root (default: SONIUM_QA_SCRATCH, else TMPDIR).
    #[arg(long)]
    scratch_root: Option<PathBuf>,
    /// Keep per-case scratch dirs after success.
    #[arg(long)]
    keep_scratch: bool,
    /// Local vendor-tree deviations recorded verbatim in fresh goldens.
    #[arg(long)]
    local_patches: Option<String>,
}

fn main() {
    let args = Args::parse();
    if args.all && !args.cases.is_empty() {
        eprintln!("error: pass either --cases a,b or --all, not both");
        std::process::exit(1);
    }
    if !args.all && args.cases.is_empty() {
        eprintln!("error: pass --cases a,b or --all");
        std::process::exit(1);
    }
    let goldens_root = args
        .goldens
        .clone()
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens"));
    let miniqa = args.miniqa.clone().or_else(|| {
        std::env::var(MINIQA_ENV)
            .ok()
            .filter(|root| !root.trim().is_empty())
            .map(PathBuf::from)
    });
    let exe = args.exe.clone().or_else(engine_exe);
    if !args.drift_only {
        if miniqa.is_none() {
            eprintln!("error: live regen needs --miniqa or {MINIQA_ENV}");
            std::process::exit(1);
        }
        if exe.is_none() {
            eprintln!("error: live regen needs --exe, RADISS_EXE, or OPENCOURANT_EXE");
            std::process::exit(1);
        }
    }
    let options = RegenOptions {
        goldens_root,
        out_root: args.out.clone(),
        miniqa_root: miniqa,
        exe,
        cases: if args.all {
            CaseSelect::All
        } else {
            CaseSelect::Only(args.cases.clone())
        },
        scratch_root: resolve_scratch_root(args.scratch_root.clone()),
        threads: args.threads,
        max_samples: args.max_samples,
        keep_scratch: args.keep_scratch,
        local_patches: args.local_patches.clone(),
        drift_only: args.drift_only,
    };
    let summary = run_regen(&options, |event| {
        println!(
            "[{}/{}] {} {}",
            event.index + 1,
            event.total,
            event.case,
            event.stage
        );
    })
    .unwrap_or_else(|error| {
        eprintln!("error: {error}");
        std::process::exit(1);
    });
    for case in &summary.cases {
        println!("{}", case.summary());
        if let Some(drift) = &case.drift {
            println!("  {}", drift.summary());
            for channel in &drift.channels {
                println!("  {}", channel.summary());
            }
            for scalar in &drift.cycle {
                println!("  {}", scalar.summary());
            }
            if let Some(error) = &drift.grid_error {
                println!("  grid: FAIL ({error})");
            }
        }
        for note in &case.notes {
            println!("  note: {note}");
        }
        for failure in &case.failures {
            println!("  failure: {failure}");
        }
    }
    println!("{}", summary.summary());
    if !summary.passed() {
        std::process::exit(1);
    }
}
