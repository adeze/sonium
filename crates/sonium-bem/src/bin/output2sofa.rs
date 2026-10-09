//! `output2sofa` — convert `numcalc`'s per-frequency JSON output into a SOFA
//! file (SimpleFreeFieldHRTF in the frequency domain).
//!
//! ```text
//! cargo run -p sonium-bem --features sofa --bin output2sofa -- \
//!     path/to/Project/be.out.json \
//!     --out path/to/Project/HRTF.sofa
//! ```
//!
//! For now we emit `SimpleFreeFieldHRTF`, since the BEM produces complex
//! pressure on the evaluation grid frequency-by-frequency. Time-domain
//! `SimpleFreeFieldHRIR` requires an IFFT step and a uniformly sampled
//! frequency grid; once those are in place we can add a `--ir` switch.

use std::path::PathBuf;
use std::process::ExitCode;

use clap::Parser;

use sonium_bem::numcalc::SweepOutput;
use sonium_xem::io::sofa::{CoordinateSystem, SimpleFreeFieldHrtf, write_simple_free_field_hrtf};

#[derive(Parser, Debug)]
#[command(
    name = "output2sofa",
    about = "Convert numcalc JSON output into a SOFA HRTF file"
)]
struct Args {
    /// Path to the `be.out.json` produced by `numcalc`.
    json: PathBuf,
    /// Output SOFA path.
    #[arg(long)]
    out: PathBuf,
    /// Receiver-position coordinate file (CSV `x,y,z` per line). Defaults to
    /// a single receiver at the origin.
    #[arg(long)]
    receivers_csv: Option<PathBuf>,
    /// Optional title attribute.
    #[arg(long)]
    title: Option<String>,
}

fn main() -> ExitCode {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let raw = match std::fs::read_to_string(&args.json) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("read {}: {e}", args.json.display());
            return ExitCode::FAILURE;
        }
    };
    let sweep: SweepOutput = match serde_json::from_str(&raw) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("parse {}: {e}", args.json.display());
            return ExitCode::FAILURE;
        }
    };

    if sweep.eval_pressure.is_empty() {
        eprintln!(
            "input has no evaluation-grid pressure (sweep was run on a project without an eval grid)"
        );
        return ExitCode::FAILURE;
    }

    let n_freq = sweep.frequencies.len();
    if !sweep.eval_points.len().is_multiple_of(3) {
        eprintln!(
            "eval_points length {} is not a multiple of 3",
            sweep.eval_points.len()
        );
        return ExitCode::FAILURE;
    }
    let n_eval = sweep.eval_points.len() / 3;
    if sweep.eval_pressure.len() != n_freq * n_eval {
        eprintln!(
            "eval_pressure length {} doesn't match {} freq × {} points",
            sweep.eval_pressure.len(),
            n_freq,
            n_eval
        );
        return ExitCode::FAILURE;
    }

    // SOFA layout: M = source positions = eval-grid points, R = receivers.
    // For the reciprocal HRTF formulation, each ear is a separate source on
    // the surface mesh; the eval grid points are where we measured pressure
    // (= the listener-reference grid). Single-receiver default is fine for
    // free-field BEM scattering; users override with --receivers-csv.
    let receivers = match args.receivers_csv.as_ref() {
        Some(p) => match read_csv_xyz(p) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("read {}: {e}", p.display());
                return ExitCode::FAILURE;
            }
        },
        None => vec![0.0, 0.0, 0.0],
    };
    let n_receivers = receivers.len() / 3;
    if n_receivers != 1 {
        eprintln!(
            "this CLI assumes 1 receiver per sweep; multi-ear support requires running numcalc once per ear and then merging"
        );
        return ExitCode::FAILURE;
    }

    // Build row-major (M, R, N) real / imag arrays. Eval points become the M axis.
    let mut real = vec![0.0; n_eval * n_receivers * n_freq];
    let mut imag = vec![0.0; n_eval * n_receivers * n_freq];
    for fi in 0..n_freq {
        for pi in 0..n_eval {
            let idx = fi * n_eval + pi;
            let sofa_idx = (pi * n_receivers) * n_freq + fi;
            let [re, im] = sweep.eval_pressure[idx];
            real[sofa_idx] = re;
            imag[sofa_idx] = im;
        }
    }

    let source_position = sweep.eval_points.clone();

    let hrtf = SimpleFreeFieldHrtf {
        real: &real,
        imag: &imag,
        frequencies: &sweep.frequencies,
        source_position: &source_position,
        source_coords: CoordinateSystem::Cartesian,
        receiver_position: &receivers,
        receiver_coords: CoordinateSystem::Cartesian,
        listener_position: None,
        title: args.title.as_deref(),
        measurements: n_eval,
        receivers: n_receivers,
    };

    if let Err(e) = write_simple_free_field_hrtf(&args.out, &hrtf) {
        eprintln!("write {}: {e}", args.out.display());
        return ExitCode::FAILURE;
    }
    log::info!(
        "Wrote {} ({} sources × {} receivers × {} freqs)",
        args.out.display(),
        n_eval,
        n_receivers,
        n_freq
    );
    ExitCode::SUCCESS
}

fn read_csv_xyz(path: &std::path::Path) -> std::io::Result<Vec<f64>> {
    let text = std::fs::read_to_string(path)?;
    let mut rows: Vec<[f64; 3]> = Vec::new();
    for (n, line) in text.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parts: Vec<&str> = line.split([',', ' ']).filter(|s| !s.is_empty()).collect();
        if parts.len() < 3 {
            return Err(std::io::Error::other(format!(
                "{}:{}: expected `x,y,z`, got `{line}`",
                path.display(),
                n + 1
            )));
        }
        let x = parts[0]
            .parse::<f64>()
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let y = parts[1]
            .parse::<f64>()
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        let z = parts[2]
            .parse::<f64>()
            .map_err(|e| std::io::Error::other(e.to_string()))?;
        rows.push([x, y, z]);
    }
    Ok(rows.into_iter().flatten().collect())
}
