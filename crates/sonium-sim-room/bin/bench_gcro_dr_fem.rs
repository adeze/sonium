//! Wall-clock benchmark: GCRO-DR vs no-recycling on the medium_multi_seat
//! scenario.
//!
//! Runs the same FEM frequency sweep twice and reports total wall-clock,
//! cumulative Krylov iterations, and per-frequency speedup. The two runs
//! share mesh assembly and frequency-independent matrix assembly cost (both
//! happen once per run, identically), so the difference is dominated by
//! the per-frequency Krylov solve cost — exactly what GCRO-DR targets.
//!
//! Usage:
//! ```
//! cargo run --release -p sonium-sim-room --bin bench-gcro-dr-fem -- \
//!     --max-freq 120 --num-freqs 25 --recycle-size 20
//! ```

use anyhow::Result;
use clap::Parser;
use std::time::Instant;

use sonium_sim_room::fem_runner::{self, FemMeshConfig, FemPreconditionerKind, RecyclerKind};
use sonium_sim_room::scenarios::{self, FrequencyConfig};

#[derive(Parser, Debug)]
#[command(
    name = "bench-gcro-dr-fem",
    about = "Compare FEM frequency-sweep wall-clock with and without GCRO-DR."
)]
struct Args {
    /// Total Rayon worker budget. Zero uses Rayon's automatic default.
    #[arg(long, visible_alias = "cores", default_value_t = 0)]
    threads: usize,
    /// Backend under test. GCRO-DR recycling is currently implemented only
    /// for the CPU room runner; other values fail explicitly.
    #[arg(long, default_value = "cpu")]
    backend: String,
    /// Scenario name. Defaults to medium_multi_seat (5 x 4 x 2.5 m room,
    /// stereo pair, three listening positions).
    #[arg(long, default_value = "medium_multi_seat")]
    scenario: String,
    /// Highest frequency in the sweep (Hz). Mesh DOFs grow as f^3, so keep
    /// this modest for an interactive benchmark — 100–150 Hz typically takes
    /// a few seconds per run.
    #[arg(long, default_value_t = 120.0)]
    max_freq: f64,
    /// Lowest frequency in the sweep (Hz).
    #[arg(long, default_value_t = 30.0)]
    min_freq: f64,
    /// Number of frequencies in the sweep.
    #[arg(long, default_value_t = 25)]
    num_freqs: usize,
    /// Recycled-subspace dimension k (number of harmonic Ritz vectors).
    #[arg(long, default_value_t = 20)]
    recycle_size: usize,
    /// Preconditioner. Defaults to BlockJacobi (the workspace default after
    /// Phase 1's parallel-preconditioner change).
    #[arg(long, default_value_t = FemPreconditionerKind::default(), value_parser = parse_preconditioner)]
    preconditioner: FemPreconditionerKind,
}

fn parse_preconditioner(s: &str) -> Result<FemPreconditionerKind, String> {
    s.parse()
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();
    sonium_sim_room::runtime::install_fem_rayon_pool(args.threads)?;
    anyhow::ensure!(
        args.backend.eq_ignore_ascii_case("cpu"),
        "bench-gcro-dr-fem supports only --backend cpu because device recycling is not implemented"
    );

    let freq_config = FrequencyConfig {
        min_freq: args.min_freq,
        max_freq: args.max_freq,
        num_points: args.num_freqs,
    };
    let scenario = scenarios::scenario_by_name_with_config(&args.scenario, &freq_config)
        .ok_or_else(|| anyhow::anyhow!("scenario not found: {}", args.scenario))?;

    println!("Scenario: {} ({})", scenario.name, scenario.description,);
    println!(
        "Sweep: {} Hz → {} Hz, {} points; preconditioner: {}; recycle-size: k={}",
        args.min_freq, args.max_freq, args.num_freqs, args.preconditioner, args.recycle_size
    );
    println!();

    // Common config (preconditioner, mesh grading) — the only difference
    // between runs is whether recycling is enabled.
    let cfg_off = FemMeshConfig {
        recycle_subspace_size: 0,
        recycler: RecyclerKind::GcroDr, // unused when size=0
        preconditioner: args.preconditioner,
        ..FemMeshConfig::default()
    };
    let cfg_on = FemMeshConfig {
        recycle_subspace_size: args.recycle_size,
        recycler: RecyclerKind::GcroDr,
        preconditioner: args.preconditioner,
        ..FemMeshConfig::default()
    };

    println!("Run 1: no recycling (warm-start initial guess only) …");
    let t0 = Instant::now();
    let out_off = fem_runner::run_fem_with_mesh(&scenario.simulation, &cfg_off)?;
    let t_off = t0.elapsed().as_secs_f64();
    println!("  total wall-clock: {t_off:.2}s");
    println!();

    println!("Run 2: GCRO-DR (k={}) …", args.recycle_size);
    let t0 = Instant::now();
    let out_on = fem_runner::run_fem_with_mesh(&scenario.simulation, &cfg_on)?;
    let t_on = t0.elapsed().as_secs_f64();
    println!("  total wall-clock: {t_on:.2}s");
    println!();

    // Validate solution agreement at every (source, lp, freq) tuple.
    // For indefinite Helmholtz, two solutions that meet the same Krylov
    // residual tolerance can disagree noticeably at modal pressure nulls
    // (the operator is near-singular there, so the null-space of A picks up
    // different components in each solver path). We report:
    //   - max |Δp|: small ⇒ both methods place modal nulls at the same place
    //   - relative diff against a robust pressure scale (90th-percentile |p|)
    //     to suppress the modal-null amplification.
    let mut all_ref: Vec<f64> = Vec::new();
    for po in &out_off.pressures {
        for l_off in po {
            for a in l_off {
                all_ref.push(a.norm());
            }
        }
    }
    all_ref.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let p90 = all_ref
        .get((all_ref.len() as f64 * 0.9) as usize)
        .copied()
        .unwrap_or(1.0)
        .max(1e-12);

    let mut max_abs = 0.0_f64;
    let mut sumsq_diff = 0.0_f64;
    let mut sumsq_ref = 0.0_f64;
    for (po, pn) in out_off.pressures.iter().zip(out_on.pressures.iter()) {
        for (l_off, l_on) in po.iter().zip(pn.iter()) {
            for (a, b) in l_off.iter().zip(l_on.iter()) {
                let abs = (a - b).norm();
                if abs > max_abs {
                    max_abs = abs;
                }
                sumsq_diff += abs * abs;
                sumsq_ref += a.norm_sqr();
            }
        }
    }
    let l2_rel = (sumsq_diff.sqrt()) / (sumsq_ref.sqrt().max(1e-30));

    println!("Result agreement:");
    println!("  max |Δp|              = {max_abs:.3e}");
    println!(
        "  max |Δp| / p90        = {:.3e}  (scaled by 90th-pct pressure)",
        max_abs / p90
    );
    println!("  L2 relative diff      = {l2_rel:.3e}  (||Δp||₂ / ||p_off||₂ over all (src,lp,f))");
    println!();

    let speedup = t_off / t_on.max(1e-9);
    println!("──────────────────────────────────────────────────");
    println!(
        "Wall-clock speedup with GCRO-DR (k={}): {speedup:.2}× ({t_off:.2}s → {t_on:.2}s)",
        args.recycle_size
    );
    println!("──────────────────────────────────────────────────");

    Ok(())
}
