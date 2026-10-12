//! QA: I-beam warping torsion vs OpenCourant explicit quasi-statics.
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` writer, then runs
//! the starter and engine when `OPENCOURANT_EXE` (or legacy `RADISS_EXE`) is set and
//! compares brick-mesh twist angles against the Vlasov warping closed form.
//! Without an engine it prints the documented skip message and exits 0. Case
//! contributed by the beams lane; Worker 1 owns the runner.
//!
//! The rad API ships brick meshes only (no beam cards), so the I-beam is a
//! brick twin: flanges plus web extruded along X, tip torque applied as a
//! force couple, twist recovered from flange DZ pairs. Gaps for Worker 1:
//! no /PROP/BEAM cards, no I-section mesher in rad.rs (built in-bin here),
//! no twist-angle or bimoment probe channels (twist is differenced DZ).

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    OpenCourantCase, RadBoundaryCondition, RadConcentratedLoad, RadElasticMaterial, RadFunction,
    RadMesh, RadNodeSet, RadProbe, assert_probe_close, engine_exe, probes_json, run_case,
    write_engine, write_starter,
};
use std::path::PathBuf;

/// Brick twin of an I-beam cantilever along X.
///
/// Cross-section: flanges span `flange_b` in y at |z| in [hz - tf, hz];
/// the web spans `tw` in y across |z| < hz - tf. Each region carries
/// `ny_f`/`ny_w`/`nz_f`/`nz_w` cells (St. Venant shear needs 2+ through
/// every wall; a single brick per wall locked 10x stiff). Returns the
/// mesh, the root face set, two single-node tip couple sets (flange tips
/// at +/- y on the top flange), and (mid_plus, mid_minus, tip_plus,
/// tip_minus) probe nodes on the top flange for twist differencing.
/// Node ids are 1-based.
type IBeamTwin = (
    RadMesh,
    Vec<usize>,
    Vec<usize>,
    Vec<usize>,
    (usize, usize, usize, usize),
);

fn i_beam_mesh(
    n_x: usize,
    length_m: f64,
    flange_b: f64,
    half_h: f64,
    tf: f64,
    tw: f64,
    ny_f: usize,
    ny_w: usize,
    nz_f: usize,
    nz_w: usize,
) -> IBeamTwin {
    // Cross-section grid: subdivide each region (flange halves, web band,
    // flange thicknesses, web height) separately.
    let mut y_cuts = Vec::new();
    for i in 0..=ny_f {
        y_cuts.push(-flange_b / 2.0 + (flange_b / 2.0 - tw / 2.0) * i as f64 / ny_f as f64);
    }
    for i in 1..=ny_w {
        y_cuts.push(-tw / 2.0 + tw * i as f64 / ny_w as f64);
    }
    for i in 1..=ny_f {
        y_cuts.push(tw / 2.0 + (flange_b / 2.0 - tw / 2.0) * i as f64 / ny_f as f64);
    }
    let web_h = 2.0 * (half_h - tf);
    let mut z_cuts = Vec::new();
    for i in 0..=nz_f {
        z_cuts.push(-half_h + tf * i as f64 / nz_f as f64);
    }
    for i in 1..=nz_w {
        z_cuts.push(-half_h + tf + web_h * i as f64 / nz_w as f64);
    }
    for i in 1..=nz_f {
        z_cuts.push(half_h - tf + tf * i as f64 / nz_f as f64);
    }
    let ny = y_cuts.len() - 1;
    let nz = z_cuts.len() - 1;
    // Material mask per (y, z) cell: flange z-bands (all columns) plus the
    // web (center y-band, center z-band).
    let y_web = ny_f..ny_f + ny_w;
    let z_web = nz_f..nz_f + nz_w;
    let material = |jy: usize, kz: usize| !z_web.contains(&kz) || y_web.contains(&jy);
    // Node id map: a grid node exists when any adjacent cell is material.
    let mut node_id = vec![vec![vec![0usize; z_cuts.len()]; y_cuts.len()]; n_x + 1];
    let mut nodes = Vec::new();
    for i in 0..=n_x {
        let x = length_m * i as f64 / n_x as f64;
        for (jy, &y) in y_cuts.iter().enumerate() {
            for (kz, &z) in z_cuts.iter().enumerate() {
                // A cut node exists when any adjacent cell is material.
                let mut used = false;
                for cy in [jy.wrapping_sub(1), jy] {
                    for cz in [kz.wrapping_sub(1), kz] {
                        if cy < ny && cz < nz && material(cy, cz) {
                            used = true;
                        }
                    }
                }
                if used {
                    node_id[i][jy][kz] = nodes.len() + 1;
                    nodes.push([x, y, z]);
                }
            }
        }
    }
    let mut bricks = Vec::new();
    for i in 0..n_x {
        for jy in 0..ny {
            for kz in 0..nz {
                if !material(jy, kz) {
                    continue;
                }
                bricks.push([
                    node_id[i][jy][kz],
                    node_id[i + 1][jy][kz],
                    node_id[i + 1][jy + 1][kz],
                    node_id[i][jy + 1][kz],
                    node_id[i][jy][kz + 1],
                    node_id[i + 1][jy][kz + 1],
                    node_id[i + 1][jy + 1][kz + 1],
                    node_id[i][jy + 1][kz + 1],
                ]);
            }
        }
    }
    assert!(bricks.iter().all(|b| b.iter().all(|&n| n > 0)));
    assert!(
        bricks.iter().all(|b| {
            let mut sorted = *b;
            sorted.sort_unstable();
            sorted.windows(2).all(|pair| pair[0] != pair[1])
        }),
        "degenerate brick: N8 must differ from N1..N7 (N8=N5 collapsed every hex to a wedge)"
    );
    let mut root_set = Vec::new();
    for jy in 0..y_cuts.len() {
        for kz in 0..z_cuts.len() {
            if node_id[0][jy][kz] > 0 {
                root_set.push(node_id[0][jy][kz]);
            }
        }
    }
    // Top-flange corners (top z-cut): first (-y) and last (+y) y-cuts.
    let top = z_cuts.len() - 1;
    let couple_plus = vec![node_id[n_x][y_cuts.len() - 1][top]];
    let couple_minus = vec![node_id[n_x][0][top]];
    let mid = n_x / 2;
    let probes = (
        node_id[mid][y_cuts.len() - 1][top],
        node_id[mid][0][top],
        node_id[n_x][y_cuts.len() - 1][top],
        node_id[n_x][0][top],
    );
    (
        RadMesh { nodes, bricks },
        root_set,
        couple_plus,
        couple_minus,
        probes,
    )
}

/// Single-sourced warping-torsion comparison case (SI units).
pub fn beam_warping_case() -> OpenCourantCase {
    let length = 2.0;
    let flange_b = 0.1;
    let half_h = 0.11;
    let tf = 0.01;
    let tw = 0.01;
    let torque = 1000.0;
    let (mesh, root_set, couple_plus, couple_minus, (mid_plus, mid_minus, tip_plus, tip_minus)) =
        i_beam_mesh(30, length, flange_b, half_h, tf, tw, 4, 2, 2, 8);
    mesh.validate().unwrap();
    // Thin-wall section constants replicate Section3D::i_beam with web
    // height h = 2 * half_h - 2 * tf between the flanges.
    let e = 200.0e9;
    let nu = 0.3;
    let g = e / (2.0 * (1.0 + nu));
    let web_h = 2.0 * half_h - 2.0 * tf;
    let torsion_j = (2.0 * flange_b * tf.powi(3) + web_h * tw.powi(3)) / 3.0;
    let iy = (2.0 * tf * flange_b.powi(3) + web_h * tw.powi(3)) / 12.0;
    let hw = web_h + tf;
    let warping_i = iy * hw * hw / 4.0;
    let mu = (g * torsion_j / (e * warping_i)).sqrt();
    // Vlasov restrained cantilever: tip twist and midspan twist.
    let tip_twist = torque / (g * torsion_j) * (length - (mu * length).tanh() / mu);
    let x = length / 2.0;
    let mid_twist = torque / (g * torsion_j)
        * (x - (mu * x).sinh() / mu + (mu * length).tanh() * ((mu * x).cosh() - 1.0) / mu);
    // Twist is differenced from flange DZ pairs over the flange width.
    let dz_pair = |twist: f64| twist * flange_b;
    // Torque couple: +/- F at the +/- y flange tips (arm = flange width).
    let couple_f = torque / flange_b;
    OpenCourantCase {
        name: "beam_warping".to_string(),
        title: "sonium_qa rad beam_warping quasi-static tip torque".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: e,
            poisson_ratio: nu,
            density_kg_per_m3: 7800.0,
        },
        node_sets: vec![
            RadNodeSet {
                name: "root".to_string(),
                nodes: root_set,
            },
            RadNodeSet {
                name: "couple plus".to_string(),
                nodes: couple_plus,
            },
            RadNodeSet {
                name: "couple minus".to_string(),
                nodes: couple_minus,
            },
        ],
        bcs: vec![RadBoundaryCondition {
            name: "built-in root (twist and warping fixed)".to_string(),
            set: 0,
            lock: [true, true, true, true, true, true],
        }],
        functions: vec![RadFunction {
            // Torsional T1 is 34 ms; the 0.35 s ramp tracks
            // quasi-statically and the 0.15 s hold settles it.
            name: "ramp 0 to 1 over 0.35 s then hold".to_string(),
            points: vec![(0.0, 0.0), (0.35, 1.0), (0.5, 1.0)],
        }],
        cloads: vec![
            RadConcentratedLoad {
                name: "couple +Z".to_string(),
                set: 1,
                direction: 3,
                function: 0,
                force_scale_n: couple_f,
            },
            RadConcentratedLoad {
                name: "couple -Z".to_string(),
                set: 2,
                direction: 3,
                function: 0,
                force_scale_n: -couple_f,
            },
        ],
        walls: Vec::new(),
        th_nodes: vec![mid_plus, mid_minus, tip_plus, tip_minus],
        probes: vec![
            RadProbe {
                label: "midspan DZ difference".to_string(),
                channel: format!("DZ/{mid_plus}-DZ/{mid_minus}"),
                expected: dz_pair(mid_twist),
                abs_tol: 1e-9,
                rel_tol: 0.15,
            },
            RadProbe {
                label: "tip DZ difference".to_string(),
                channel: format!("DZ/{tip_plus}-DZ/{tip_minus}"),
                expected: dz_pair(tip_twist),
                abs_tol: 1e-9,
                rel_tol: 0.20,
            },
        ],
        stop_time_s: 0.5,
        th_period_s: 0.005,
        anim_period_s: 0.05,
    }
}

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/beam_warping")
}

/// Evaluate a differenced channel `A-B` from the converted history.
fn diff_channel(history: &sonium_qa::rad::TimeHistory, channel: &str) -> Option<f64> {
    let (a, b) = channel.split_once('-')?;
    let va = history.final_value(a.trim()).ok()?;
    let vb = history.final_value(b.trim()).ok()?;
    Some(va - vb)
}

fn main() {
    let case = beam_warping_case();
    if std::env::var("REGEN_DECKS").as_deref() == Ok("1") {
        let dir = decks_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("beam_warping_0000.rad"), write_starter(&case)).unwrap();
        std::fs::write(dir.join("beam_warping_0001.rad"), write_engine(&case)).unwrap();
        std::fs::write(
            dir.join("probes.json"),
            probes_json(&case, "none (prove-out deferred)"),
        )
        .unwrap();
        println!("regenerated decks for beam_warping");
        return;
    }
    let dir = decks_dir();
    let starter_path = dir.join("beam_warping_0000.rad");
    let engine_path = dir.join("beam_warping_0001.rad");
    let probes_path = dir.join("probes.json");

    // No hand-built decks: the checked-in pair must match the writer.
    let starter = write_starter(&case);
    let engine = write_engine(&case);
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "decks/beam_warping/beam_warping_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "decks/beam_warping/beam_warping_0001.rad drifted from the rad writer"
    );
    assert!(
        probes_path.exists(),
        "decks/beam_warping/probes.json must be checked in"
    );
    let _ = probes_json(&case, "none (prove-out deferred)");

    let Some(exe) = engine_exe() else {
        println!("skipped: set OPENCOURANT_EXE (or legacy RADISS_EXE)");
        return;
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!("sonium_rad_beam_warping_{}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("beam_warping_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("beam_warping_0001.rad"), &engine).unwrap();
    let output = run_case(&work, &exe, &case).unwrap_or_else(|error| {
        eprintln!("rad beam_warping run failed: {error}");
        std::process::exit(1);
    });
    println!("engine: {}", output.engine);
    let mut failed = false;
    for probe in &case.probes {
        match diff_channel(&output.history, &probe.channel) {
            Some(actual) => {
                println!(
                    "probe {}: actual {actual:.6e}, expected {:.6e}",
                    probe.label, probe.expected
                );
                if let Err(error) = assert_probe_close(
                    actual,
                    probe.expected,
                    probe.abs_tol,
                    probe.rel_tol,
                    &probe.label,
                ) {
                    eprintln!("FAIL: {error}");
                    failed = true;
                }
            }
            None => {
                eprintln!("FAIL: probe {} unreadable", probe.label);
                failed = true;
            }
        }
    }
    std::fs::remove_dir_all(&work).ok();
    if failed {
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
