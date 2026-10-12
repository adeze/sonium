//! Offline golden replay runner v0 (G-tier, no engine on the path).
//!
//! Loads a checked-in golden, verifies the miniqa deck hashes when a
//! miniqa root is available, and compares provided series (or the
//! golden against itself) with per-channel tolerances. Missing or
//! stale goldens fail closed and name the regen command; with
//! `REGEN_OC_GOLDENS=1` (and an engine) the golden is re-harvested
//! instead, mirroring `REGEN_DECKS=1`.
//!
//! Full integration with native sonium explicit runs lands in Phase 2;
//! v0 compares TH-CSV-shaped series (`--series`) or self-tests.
//!
//! ```sh
//! qa-oc-replay --deck LOIS/LOI2/BESH/input_44
//! qa-oc-replay --deck LOIS/LOI2/BESH/input_44 --series runT01.csv
//! REGEN_OC_GOLDENS=1 qa-oc-replay --deck LOIS/LOI2/BESH/input_44
//! ```

// Rust guideline compliant 2026-10-10

use clap::Parser;
use sonium_qa::rad::engine_exe;
use sonium_qa::rad::oc_golden::{
    CompareOptions, HarvestOptions, MINIQA_ENV, compare_history, golden_path, harvest, load_golden,
    parse_th_csv, regen_armed, resolve_scratch_root, verify_deck_hashes, write_golden,
};
use std::collections::BTreeMap;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "qa-oc-replay", about = "Replay a checked-in golden offline.")]
struct Args {
    /// miniqa-relative case dir, resolved under the goldens root.
    #[arg(long)]
    deck: Option<String>,
    /// Explicit golden JSON path (overrides --deck).
    #[arg(long)]
    golden: Option<PathBuf>,
    /// Goldens root; defaults to the crate `goldens/` dir.
    #[arg(long)]
    goldens: Option<PathBuf>,
    /// miniqa root for the deck-hash check; defaults to OPENCOURANT_MINIQA.
    #[arg(long)]
    miniqa: Option<PathBuf>,
    /// Provided time-history CSV to compare (default: golden self-test).
    #[arg(long)]
    series: Option<PathBuf>,
}

fn goldens_root(args: &Args) -> PathBuf {
    args.goldens
        .clone()
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("goldens"))
}

fn main() {
    let args = Args::parse();
    let root = goldens_root(&args);
    let golden_path = match (&args.golden, &args.deck) {
        (Some(path), _) => path.clone(),
        (None, Some(deck)) => golden_path(&root, deck),
        (None, None) => {
            eprintln!("error: pass --deck <case> or --golden <path>");
            std::process::exit(1);
        }
    };
    let miniqa = args.miniqa.clone().or_else(|| {
        std::env::var(MINIQA_ENV)
            .ok()
            .filter(|root| !root.trim().is_empty())
            .map(PathBuf::from)
    });

    if regen_armed() {
        let Some(case) = args.deck.clone() else {
            eprintln!("error: regen needs --deck <case>");
            std::process::exit(1);
        };
        let exe = engine_exe().unwrap_or_else(|| {
            eprintln!("error: regen needs --exe, RADISS_EXE, or OPENCOURANT_EXE");
            std::process::exit(1);
        });
        let Some(miniqa) = miniqa else {
            eprintln!("error: regen needs --miniqa or {MINIQA_ENV}");
            std::process::exit(1);
        };
        // A regen keeps the previous golden's patch record, so a
        // reharvested patched deck never silently drops its deviation.
        let inherit = load_golden(&golden_path)
            .ok()
            .and_then(|golden| golden.meta.local_patches);
        let golden = harvest(&HarvestOptions {
            miniqa_root: miniqa,
            case,
            exe,
            scratch_root: resolve_scratch_root(None),
            threads: 1,
            keep_scratch: false,
            max_samples: 0,
            local_patches: inherit,
        })
        .unwrap_or_else(|error| {
            eprintln!("regen failed: {error}");
            std::process::exit(1);
        });
        let path = write_golden(&root, &golden).unwrap_or_else(|error| {
            eprintln!("error: {error}");
            std::process::exit(1);
        });
        println!(
            "regenerated {} ({} channels x {} samples)",
            path.display(),
            golden.channels.len(),
            golden.time.len()
        );
        return;
    }

    let golden = load_golden(&golden_path).unwrap_or_else(|error| {
        eprintln!("error: {error}");
        std::process::exit(1);
    });
    match &miniqa {
        Some(miniqa) => verify_deck_hashes(&golden, miniqa).unwrap_or_else(|error| {
            eprintln!("error: {error}");
            std::process::exit(1);
        }),
        None => println!("note: deck-hash check skipped (set --miniqa or {MINIQA_ENV})"),
    }

    if golden.time.is_empty() {
        // load_golden guarantees a th_note here (fail-closed above).
        if args.series.is_some() {
            eprintln!(
                "error: golden {} is cycle-only ({}); --series replay needs a TH grid",
                golden.case,
                golden.meta.th_note.as_deref().unwrap_or("no TH grid")
            );
            std::process::exit(1);
        }
        println!(
            "golden {} replay PASS: cycle-only golden ({}), {} cycle scalars recorded",
            golden.case,
            golden.meta.th_note.as_deref().unwrap_or("no TH grid"),
            golden.cycle.len()
        );
        return;
    }
    let provided = match &args.series {
        Some(csv) => parse_th_csv(csv).unwrap_or_else(|error| {
            eprintln!("error: {error}");
            std::process::exit(1);
        }),
        None => sonium_qa::rad::TimeHistory {
            times: golden.time.clone(),
            channels: golden
                .channels
                .iter()
                .map(|(name, channel)| (name.clone(), channel.values.clone()))
                .collect::<BTreeMap<_, _>>(),
        },
    };
    let report = compare_history(&golden, &provided, &CompareOptions::default());
    println!("{}", report.summary());
    for failure in &report.failures {
        println!("  FAIL {failure}");
    }
    if !report.passed() {
        println!("FAILED: qa-oc-replay");
        std::process::exit(1);
    }
    println!("SUCCESS: qa-oc-replay");
}
