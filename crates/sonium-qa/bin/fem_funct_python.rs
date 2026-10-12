//! Phase-7 lane 7D leg: trusted-deck `/FUNCT_PYTHON` evaluation.
//!
//! Default mode (no arguments) runs the embedded self-test: pure
//! math, `TIME`/`DT` scalars, sensor snapshots, and the fail-closed
//! missing-snapshot probe, each with an asserted value. `--deck`
//! evaluates one block of a deck file at an abscissa with the given
//! `TIME`/`DT` and empty sensor snapshots.
//!
//! Trust boundary: this leg executes deck Python with full
//! interpreter power and no sandbox; run trusted decks only. Sensor
//! calls with no snapshot entry fail closed (evaluation error, never
//! a silent zero).
//!
//! Without the `funct-python` feature the leg defers loudly
//! (`DEFERRED` plus the [`DEFER_UNTIL`], [`DEFER_MISSING`], and
//! [`DEFER_TARGET`] lines) and exits 0; evaluation failures exit 1.
//! Exit 0 = leg executed (self-test passed or deferral reported).

// Rust guideline compliant 2026-02-21

#[cfg(feature = "funct-python")]
use std::path::Path;
use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;
#[cfg(not(feature = "funct-python"))]
use sonium_qa::funct_python::{DEFER_MISSING, DEFER_TARGET, DEFER_UNTIL};
#[cfg(feature = "funct-python")]
use sonium_qa::funct_python::{PythonContext, classify, collect_blocks, evaluate_python};
#[cfg(feature = "funct-python")]
use sonium_qa::rad_parse::RadDeck;

#[derive(Debug, Parser)]
#[command(name = "qa-funct-python", about = "Evaluate /FUNCT_PYTHON deck blocks")]
struct Args {
    /// Deck file to evaluate (default: embedded self-test).
    #[arg(long)]
    deck: Option<PathBuf>,
    /// Block id to call in `--deck` mode.
    #[arg(long, default_value_t = 1)]
    id: i64,
    /// Abscissa passed to the block in `--deck` mode.
    #[arg(long, default_value_t = 0.0)]
    abscissa: f64,
    /// `TIME` scalar in `--deck` mode.
    #[arg(long, default_value_t = 0.0)]
    time: f64,
    /// `DT` scalar in `--deck` mode.
    #[arg(long, default_value_t = 0.0)]
    dt: f64,
}

fn main() -> ExitCode {
    let args = Args::parse();
    #[cfg(not(feature = "funct-python"))]
    {
        deferred(&args)
    }
    #[cfg(feature = "funct-python")]
    {
        run(&args)
    }
}

/// Reports the feature-off deferral; the only off-config verdict.
#[cfg(not(feature = "funct-python"))]
fn deferred(args: &Args) -> ExitCode {
    if let Some(deck) = &args.deck {
        println!(
            "CANNOT EVALUATE: {} needs the funct-python feature",
            deck.display()
        );
    }
    println!("VERDICT: DEFERRED (P5C3 open; rebuild with --features funct-python)");
    println!("  until: {DEFER_UNTIL}");
    println!("  missing: {DEFER_MISSING}");
    println!("  target: {DEFER_TARGET}");
    println!("G-LEG qa-funct-python: deferred without the funct-python feature");
    println!("DEFERRED: qa-funct-python");
    ExitCode::SUCCESS
}

/// Runs the self-test or a deck evaluation with the feature on.
#[cfg(feature = "funct-python")]
fn run(args: &Args) -> ExitCode {
    println!(
        "TRUST: qa-funct-python executes deck Python with full interpreter power \
         (no sandbox); run trusted decks only."
    );
    match &args.deck {
        None => self_test(),
        Some(path) => eval_deck(args, path),
    }
}

/// One self-test probe outcome.
#[cfg(feature = "funct-python")]
struct Probe {
    name: &'static str,
    pass: bool,
    detail: String,
}

/// Runs the embedded self-test with asserted values.
#[cfg(feature = "funct-python")]
fn self_test() -> ExitCode {
    use std::collections::BTreeMap;

    let probes = [
        Probe {
            name: "pure-math",
            ..check_value(
                "/FUNCT_PYTHON/1\ndef linear(x):\n    return 2 * x + 1\n/END\n",
                1,
                3.0,
                &PythonContext::default(),
                7.0,
            )
        },
        Probe {
            name: "scalars",
            ..check_value(
                "/FUNCT_PYTHON/1\ndef clock(x):\n    return TIME - DT\n/END\n",
                1,
                0.0,
                &PythonContext {
                    time: 2.0,
                    dt: 0.5,
                    ..Default::default()
                },
                1.5,
            )
        },
        Probe {
            name: "sensors",
            ..check_value(
                "/FUNCT_PYTHON/1\ndef sense(x):\n    return node_disp(10, 'Z') + element_var(3, 'K')\n/END\n",
                1,
                0.0,
                &PythonContext {
                    nodal_disp: BTreeMap::from([((10, 'Z'), 1.0)]),
                    element_var: BTreeMap::from([((3, "K".to_string()), 2.0)]),
                    ..Default::default()
                },
                3.0,
            )
        },
        Probe {
            name: "fail-closed",
            ..check_missing_snapshot(
                "/FUNCT_PYTHON/1\ndef sense(x):\n    return node_vel(1, 'X')\n/END\n",
                1,
                0.0,
            )
        },
    ];
    let mut failures = 0;
    for probe in &probes {
        if probe.pass {
            println!("probe {}: PASS ({})", probe.name, probe.detail);
        } else {
            println!("probe {}: FAIL ({})", probe.name, probe.detail);
            failures += 1;
        }
    }
    println!("VERDICT: IMPLEMENTED (P7 lane 7D; P5C3 replay target closed)");
    println!("  self-test: pure math, TIME/DT scalars, sensor snapshots,");
    println!("    and the fail-closed missing-snapshot probe");
    if failures > 0 {
        println!("G-LEG qa-funct-python: FAIL ({failures} probe failures)");
        println!("FAILED: qa-funct-python");
        std::process::exit(1);
    }
    println!("G-LEG qa-funct-python: self-test passes");
    println!("SUCCESS: qa-funct-python");
    ExitCode::SUCCESS
}

/// Evaluates an embedded deck and checks the value (caller sets `name`).
#[cfg(feature = "funct-python")]
fn check_value(
    deck: &str,
    id: i64,
    abscissa: f64,
    context: &PythonContext,
    expected: f64,
) -> Probe {
    let parsed = RadDeck::parse_str(deck).map_err(|err| err.to_string());
    let blocks = parsed.and_then(|deck| collect_blocks(&deck).map_err(|err| err.to_string()));
    match blocks {
        Err(err) => Probe {
            name: "",
            pass: false,
            detail: format!("collect failed: {err}"),
        },
        Ok(blocks) => match evaluate_python(&blocks, id, abscissa, context) {
            Err(err) => Probe {
                name: "",
                pass: false,
                detail: format!("eval failed: {err}"),
            },
            Ok(value) if value == expected => Probe {
                name: "",
                pass: true,
                detail: format!("value {value}"),
            },
            Ok(value) => Probe {
                name: "",
                pass: false,
                detail: format!("value {value}, expected {expected}"),
            },
        },
    }
}

/// Checks that an empty snapshot fails closed as an evaluation error.
#[cfg(feature = "funct-python")]
fn check_missing_snapshot(deck: &str, id: i64, abscissa: f64) -> Probe {
    let context = PythonContext::default();
    let parsed = RadDeck::parse_str(deck).map_err(|err| err.to_string());
    let blocks = parsed.and_then(|deck| collect_blocks(&deck).map_err(|err| err.to_string()));
    match blocks {
        Err(err) => Probe {
            name: "",
            pass: false,
            detail: format!("collect failed: {err}"),
        },
        Ok(blocks) => match evaluate_python(&blocks, id, abscissa, &context) {
            Ok(value) => Probe {
                name: "",
                pass: false,
                detail: format!("returned {value} despite empty snapshot"),
            },
            Err(err) if err.is_evaluation() => Probe {
                name: "",
                pass: true,
                detail: format!("evaluation error: {err}"),
            },
            Err(err) => Probe {
                name: "",
                pass: false,
                detail: format!("wrong error kind: {err}"),
            },
        },
    }
}

/// Evaluates one block of a deck file.
#[cfg(feature = "funct-python")]
fn eval_deck(args: &Args, path: &Path) -> ExitCode {
    let deck = match RadDeck::parse_file(path) {
        Err(err) => return fail(&format!("cannot parse {}: {err}", path.display())),
        Ok(deck) => deck,
    };
    let blocks = match collect_blocks(&deck) {
        Err(err) => return fail(&format!("cannot collect blocks: {err}")),
        Ok(blocks) => blocks,
    };
    let Some(block) = blocks.iter().find(|block| block.id == args.id) else {
        return fail(&format!("block {} absent from {}", args.id, path.display()));
    };
    let context = PythonContext {
        time: args.time,
        dt: args.dt,
        ..Default::default()
    };
    println!("deck: {}", path.display());
    println!(
        "block {}: {} ({:?}, param {})",
        block.id,
        block.name,
        classify(block, &blocks),
        block.param
    );
    match evaluate_python(&blocks, args.id, args.abscissa, &context) {
        Err(err) => {
            println!(
                "hint: this leg passes empty sensor snapshots; sensor calls \
                 fail closed by design"
            );
            fail(&format!("evaluation failed: {err}"))
        }
        Ok(value) => {
            println!("value at x={}: {value:?}", args.abscissa);
            println!("VERDICT: IMPLEMENTED (P7 lane 7D; P5C3 replay target closed)");
            println!("G-LEG qa-funct-python: deck block evaluates");
            println!("SUCCESS: qa-funct-python");
            ExitCode::SUCCESS
        }
    }
}

/// Prints the failure markers and exits non-zero.
#[cfg(feature = "funct-python")]
fn fail(message: &str) -> ExitCode {
    println!("ERROR: {message}");
    println!("G-LEG qa-funct-python: FAIL ({message})");
    println!("FAILED: qa-funct-python");
    std::process::exit(1);
}
