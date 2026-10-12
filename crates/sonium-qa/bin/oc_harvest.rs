//! E-tier harvester: run a miniqa deck through vendor starter+engine
//! and write the checked-in JSON golden.
//!
//! Reads (never writes) the miniqa tree, runs everything in `/tmp`
//! scratch, and writes `goldens/miniqa/<case>.json`. Requires an engine
//! (`--exe`, `RADISS_EXE`, or `OPENCOURANT_EXE`) and the miniqa root
//! (`--miniqa` or `OPENCOURANT_MINIQA`).
//!
//! ```sh
//! OPENCOURANT_MINIQA=.../qa-tests/miniqa OPENCOURANT_EXE=.../exec \
//!   qa-oc-harvest --deck LOIS/LOI2/BESH/input_44
//! ```

// Rust guideline compliant 2026-10-10

use clap::Parser;
use sonium_qa::rad::engine_exe;
use sonium_qa::rad::oc_golden::{
    HarvestOptions, MINIQA_ENV, find_deck, harvest, resolve_scratch_root, write_golden,
};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(
    name = "qa-oc-harvest",
    about = "Harvest one miniqa deck into a checked-in JSON golden."
)]
struct Args {
    /// miniqa-relative case dir, e.g. LOIS/LOI2/BESH/input_44.
    #[arg(long)]
    deck: Option<String>,
    /// List harvestable case ids and exit.
    #[arg(long)]
    list: bool,
    /// miniqa root (`.../qa-tests/miniqa`); defaults to OPENCOURANT_MINIQA.
    #[arg(long)]
    miniqa: Option<PathBuf>,
    /// Engine/starter exe or exec dir; defaults to RADISS_EXE/OPENCOURANT_EXE.
    #[arg(long)]
    exe: Option<PathBuf>,
    /// Goldens root (goldens land in `<out>/miniqa/`); defaults to the
    /// crate `goldens/` dir.
    #[arg(long)]
    out: Option<PathBuf>,
    /// OMP threads for the run.
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Keep the /tmp scratch dir after a successful harvest.
    #[arg(long)]
    keep_scratch: bool,
    /// Peak-preserving decimation budget (samples per channel, 0 = full).
    #[arg(long, default_value_t = 0)]
    max_samples: usize,
    /// Scratch root for the run (default: SONIUM_QA_SCRATCH, else TMPDIR).
    #[arg(long)]
    scratch_root: Option<PathBuf>,
    /// Local vendor-tree deviations the deck carries, recorded
    /// verbatim in the golden (None = ran as shipped).
    #[arg(long)]
    local_patches: Option<String>,
}

fn miniqa_root(args: &Args) -> PathBuf {
    if let Some(root) = &args.miniqa {
        return root.clone();
    }
    match std::env::var(MINIQA_ENV) {
        Ok(root) if !root.trim().is_empty() => PathBuf::from(root),
        _ => {
            eprintln!("error: set --miniqa or {MINIQA_ENV} to the miniqa root");
            std::process::exit(1);
        }
    }
}

fn collect_cases(root: &PathBuf, dir: &PathBuf, cases: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    let mut entries: Vec<_> = entries.flatten().collect();
    entries.sort_by_key(|entry| entry.file_name());
    for entry in entries {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }
        if path.join("data").is_dir()
            && let Ok(case) = path.strip_prefix(root)
        {
            cases.push(case.to_string_lossy().replace('\\', "/"));
        }
        // Always recurse: cases nest (e.g. INTERF/INT_7/igsti/small_boule_igsti
        // contains dtmin_02 with its own data/ + reference/).
        collect_cases(root, &path, cases);
    }
}

fn main() {
    let args = Args::parse();
    let miniqa = miniqa_root(&args);
    if args.list {
        let mut cases = Vec::new();
        collect_cases(&miniqa, &miniqa, &mut cases);
        for case in cases {
            println!("{case}");
        }
        return;
    }
    let Some(deck) = args.deck.clone() else {
        eprintln!("error: pass --deck <case> or --list");
        std::process::exit(1);
    };
    let exe = args.exe.clone().or_else(engine_exe).unwrap_or_else(|| {
        eprintln!("error: set --exe, RADISS_EXE, or OPENCOURANT_EXE");
        std::process::exit(1);
    });
    let resolved = find_deck(&miniqa, &deck).unwrap_or_else(|error| {
        eprintln!("error: {error}");
        std::process::exit(1);
    });
    println!(
        "harvesting {deck}: starter {} + {} engine deck(s)",
        resolved.starter_deck,
        resolved.engine_decks.len()
    );
    let options = HarvestOptions {
        miniqa_root: miniqa,
        case: deck,
        exe,
        scratch_root: resolve_scratch_root(args.scratch_root.clone()),
        threads: args.threads,
        keep_scratch: args.keep_scratch,
        max_samples: args.max_samples,
        local_patches: args.local_patches.clone(),
    };
    let golden = harvest(&options).unwrap_or_else(|error| {
        eprintln!("harvest failed: {error}");
        println!("FAILED: qa-oc-harvest");
        std::process::exit(1);
    });
    let out = args
        .out
        .clone()
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens"));
    let path = write_golden(&out, &golden).unwrap_or_else(|error| {
        eprintln!("error: {error}");
        println!("FAILED: qa-oc-harvest");
        std::process::exit(1);
    });
    println!(
        "wrote {} ({} channels x {} samples, engine {})",
        path.display(),
        golden.channels.len(),
        golden.time.len(),
        golden.meta.engine
    );
    if let Some(note) = &golden.meta.th_note {
        println!("note: cycle-only golden ({note})");
    }
    println!("SUCCESS: qa-oc-harvest");
}
