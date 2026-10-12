//! Phase-4 lane 4A G-leg (lane-owned): corpus parse.
//!
//! Parses every miniqa `.rad` deck with the universal deck reader,
//! checks the canonical round trip per deck, and prints the typed,
//! family-decoded, and raw coverage table per card family.
//! Exit 0 = PASS, exit 1 = FAIL with per-deck report on stdout.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad_cards::{CardKind, INVENTORY_COUNT};
use sonium_qa::rad_parse::{RadDeck, TypedCard};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

/// Per-family coverage counters.
#[derive(Default)]
struct FamilyCover {
    /// Total blocks observed.
    total: usize,
    /// Blocks with a structural typed decoding.
    typed: usize,
    /// Blocks with a flag-family decoding (`Anim`, `Th`, `Print`).
    family: usize,
}

/// Usage text for the one-flag CLI.
fn usage(program: &str) -> String {
    format!(
        "usage: {program} [--dir MINIQA]\n\n\
         Parses all miniqa .rad decks and checks canonical round trips.\n\
         Defaults to $MINIQA_DIR, else ~/src_3rdparties/OpenCourant/OpenCourant/qa-tests/miniqa."
    )
}

/// Resolves the miniqa root from flags and environment.
fn miniqa_root(args: &[String]) -> Result<PathBuf, String> {
    let mut dir: Option<PathBuf> = None;
    let mut index = 1;
    while index < args.len() {
        match args[index].as_str() {
            "--dir" => {
                index += 1;
                dir = args.get(index).map(PathBuf::from);
                if dir.is_none() {
                    return Err("missing --dir value".to_string());
                }
            }
            "-h" | "--help" => {
                let help = usage(&args[0]);
                println!("{help}");
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument {other:?}")),
        }
        index += 1;
    }
    if let Some(dir) = dir {
        return Ok(dir);
    }
    if let Ok(env) = std::env::var("MINIQA_DIR") {
        return Ok(PathBuf::from(env));
    }
    let home = std::env::var("HOME").map_err(|_| "HOME is not set".to_string())?;
    Ok(Path::new(&home).join("src_3rdparties/OpenCourant/OpenCourant/qa-tests/miniqa"))
}

/// Collects `.rad` files below a directory in sorted order.
fn collect_decks(root: &Path, out: &mut Vec<PathBuf>) -> std::io::Result<()> {
    let mut entries: Vec<_> = std::fs::read_dir(root)?.collect::<Result<_, _>>()?;
    entries.sort_by_key(|e| e.path());
    for entry in entries {
        let path = entry.path();
        if path.is_dir() {
            collect_decks(&path, out)?;
        } else if path.extension().is_some_and(|e| e == "rad") {
            out.push(path);
        }
    }
    Ok(())
}

/// Reports whether deck text holds an `#include` directive.
fn uses_include(text: &str) -> bool {
    text.lines().any(|line| {
        let trimmed = line.trim_start();
        trimmed
            .get(..8)
            .is_some_and(|p| p.eq_ignore_ascii_case("#include"))
            && (trimmed.len() == 8 || trimmed[8..].starts_with([' ', '\t']))
    })
}

/// Classifies a card as typed, family-decoded, or raw.
fn classify(typed: &Option<TypedCard>) -> &'static str {
    match typed {
        None => "raw",
        Some(TypedCard::Anim(_) | TypedCard::Th(_) | TypedCard::Print(_)) => "family",
        Some(_) => "typed",
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let root = match miniqa_root(&args) {
        Ok(root) => root,
        Err(message) => {
            println!("FAIL: {message}");
            let help = usage(&args[0]);
            println!("{help}");
            std::process::exit(1);
        }
    };
    let mut decks = Vec::new();
    if let Err(error) = collect_decks(&root, &mut decks) {
        let display = root.display();
        println!("FAIL: cannot scan {display}: {error}");
        std::process::exit(1);
    }
    if decks.is_empty() {
        let display = root.display();
        println!("FAIL: no .rad decks under {display}");
        std::process::exit(1);
    }
    let mut cover: BTreeMap<CardKind, FamilyCover> = BTreeMap::new();
    let mut failures = 0;
    let mut with_include = 0;
    let mut total_cards = 0;
    for deck_path in &decks {
        let name = deck_path
            .strip_prefix(&root)
            .unwrap_or(deck_path)
            .display()
            .to_string();
        let deck = match RadDeck::parse_file(deck_path) {
            Ok(deck) => deck,
            Err(error) => {
                failures += 1;
                println!("FAIL {name}: {error}");
                continue;
            }
        };
        match deck.round_trip() {
            Ok(again) if again == deck => {}
            Ok(_) => {
                failures += 1;
                println!("FAIL {name}: round trip model differs");
                continue;
            }
            Err(error) => {
                failures += 1;
                println!("FAIL {name}: round trip: {error}");
                continue;
            }
        }
        if std::fs::read_to_string(deck_path).is_ok_and(|t| uses_include(&t)) {
            with_include += 1;
        }
        total_cards += deck.cards.len();
        for card in &deck.cards {
            let entry = cover.entry(card.kind).or_default();
            entry.total += 1;
            match classify(&card.typed) {
                "typed" => entry.typed += 1,
                "family" => entry.family += 1,
                _ => {}
            }
        }
        let cards = deck.cards.len();
        println!("PASS {name} ({cards} cards)");
    }
    println!();
    println!("coverage: kind total typed family raw");
    for kind in CardKind::ALL {
        let entry = cover.get(kind);
        let (total, typed, family) = match entry {
            Some(cover) => (cover.total, cover.typed, cover.family),
            None => (0, 0, 0),
        };
        if total > 0 {
            let raw = total - typed - family;
            println!("{kind} {total} {typed} {family} {raw}");
        }
    }
    let mut zero_extra = Vec::new();
    let mut zero_inventory = Vec::new();
    for kind in CardKind::ALL {
        if !cover.contains_key(kind) {
            if kind.is_extra() {
                zero_extra.push(kind.as_str());
            } else {
                zero_inventory.push(kind.as_str());
            }
        }
    }
    println!();
    let observed = INVENTORY_COUNT - zero_inventory.len();
    let extras_seen: Vec<_> = CardKind::ALL
        .iter()
        .filter(|k| k.is_extra() && cover.contains_key(k))
        .map(|k| k.as_str())
        .collect();
    let extras_count = extras_seen.len();
    let extras_list = extras_seen.join(",");
    println!(
        "inventory: {observed}/{INVENTORY_COUNT} headers observed \
         ({extras_count} extra kinds observed: {extras_list})"
    );
    if !zero_inventory.is_empty() {
        let missing = zero_inventory.join(",");
        println!("unobserved inventory: {missing}");
    }
    if !zero_extra.is_empty() {
        let missing = zero_extra.join(",");
        println!("unobserved extras: {missing}");
    }
    println!();
    let decks_count = decks.len();
    println!(
        "summary: {decks_count} decks, {total_cards} cards, \
         {with_include} decks with #include, {failures} failures"
    );
    if failures > 0 {
        println!("FAILED: qa-rad-parse-corpus");
        std::process::exit(1);
    }
    println!("SUCCESS: qa-rad-parse-corpus");
}
