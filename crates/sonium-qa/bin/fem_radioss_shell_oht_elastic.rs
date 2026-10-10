//! QA: open-hole plate shells vs Radioss/OpenRadioss quasi-statics.
//!
//! Elastic precursor of the progressive-failure open-hole tension case:
//! the Part-2 planform (L = 128 mm, W = 32 mm, D = 6.35 mm, t = 1.0 mm)
//! as homogenized quasi-isotropic QEPH shells (CLT E = 52.70 GPa,
//! nu = 0.24 for the [45/0/-45/90]s T300/914-ish layup) pulled to a
//! 100 MPa nominal stress. Probes replay the bar compliance and the
//! Howland finite-width stress concentration at the hole minimum
//! section, the site where the in-process progressive-damage QA
//! (`fem_composites_progressive_failure.rs`) initiates and cascades.
//!
//! References: Howland, R. C. J. (1930), "On the stresses in the
//! neighbourhood of a circular hole in a strip under tension", Phil.
//! Trans. R. Soc. A 229; Green, B. G. et al. (2007), "Experimental
//! investigation of the failure of scaled open-hole tensile composite
//! laminates", Composites Part A 38.
//!
//! Seam (Worker-1 gap): `sonium_qa::rad` ships no layered composite
//! property and no ply-failure material (no /PROP/SHELL composite, no
//! /MAT/LAW25), so the engine deck can only cover the linear elastic
//! precursor. Progressive Hashin/Puck damage stays in-process in the
//! Part-2 QA, which reproduces the Green et al. notched/unnotched trend
//! (ratio 0.62 in the 0.35-0.85 band) with hole-edge initiation and a
//! 0-degree fiber cascade.
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` extended
//! writer, then runs the starter and engine when `RADISS_EXE` or
//! `OPENCOURANT_EXE` is set and compares compliance plus hole-edge
//! membrane stress against the closed forms. Without an engine it prints
//! the documented skip message and exits 0. Decks regenerate with
//! `REGEN_DECKS=1`.
//!
//! Prove-out note: `/TH/SHELL` F1/F2/F12 channels report membrane STRESS
//! (Pa), not force resultants (N/m) -- doubling the shell thickness at
//! fixed load halves the channel. Moment channels (M1, ...) do report
//! resultants (clamped-plate M1 passes as such).

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    RadBoundaryCondition, RadCase, RadConcentratedLoad, RadElasticMaterial, RadExtendedCase,
    RadFunction, RadMesh, RadNodeSet, RadProbe, RadShellMesh, RadShellProperty, RadThRequest,
    assert_probe_close, engine_exe, extended_probes_json, run_extended_case, shel_channel,
    write_extended_engine, write_extended_starter,
};
use std::path::PathBuf;

const LENGTH_M: f64 = 128.0e-3;
const WIDTH_M: f64 = 32.0e-3;
const HOLE_D_M: f64 = 6.35e-3;
const THICK_M: f64 = 1.0e-3;
/// CLT-homogenized quasi-isotropic modulus of the [45/0/-45/90]s layup
/// (E1 = 126 GPa, E2 = 11 GPa, G12 = 6.6 GPa, nu12 = 0.28): 52.70 GPa.
const YOUNG_PA: f64 = 52.7e9;
const POISSON: f64 = 0.24;
const DENSITY: f64 = 1_600.0;
/// Total pull force: 100 MPa over the gross section.
const LOAD_N: f64 = 100.0e6 * WIDTH_M * THICK_M;

/// Piecewise-linear 1D coordinates with `fine_frac` of the `n` cells
/// packed into the center `span_frac` of `[0, total]`. The fine block
/// holds an even cell count with the same parity remainder as `n`, so
/// faces land symmetrically about the center.
fn graded_coords(n: usize, total: f64, fine_frac: f64, span_frac: f64) -> Vec<f64> {
    let mut n_fine = (n as f64 * fine_frac).round() as usize;
    n_fine = n_fine.min(n);
    if (n - n_fine) % 2 == 1 {
        n_fine = n_fine.saturating_sub(1).max(2);
    }
    let n_side = (n - n_fine) / 2;
    let fine_span = total * span_frac;
    let side_span = (total - fine_span) / 2.0;
    let mut xs = Vec::with_capacity(n + 1);
    for i in 0..=n_side {
        xs.push(side_span * i as f64 / n_side as f64);
    }
    for i in 1..=n_fine {
        xs.push(side_span + fine_span * i as f64 / n_fine as f64);
    }
    for i in 1..=n_side {
        xs.push(side_span + fine_span + side_span * i as f64 / n_side as f64);
    }
    xs
}

struct OhtMesh {
    nodes: Vec<[f64; 3]>,
    quads: Vec<[usize; 4]>,
    root: Vec<usize>,
    root_center: usize,
    pulled: Vec<usize>,
    pulled_center: usize,
    /// Shell id at the hole minimum section (+y side), with centroid.
    probe_shell: usize,
    probe_centroid: [f64; 2],
}

/// Graded quad mesh of the holed plate with a staircase hole: cells
/// whose center falls inside the hole circle are dropped. Connectivity
/// runs N1 -> N2 along +x, so the shell F1 channel is the pull-direction
/// membrane stress and normals point +z.
fn oht_mesh(n_x: usize, n_y: usize) -> OhtMesh {
    assert!(n_y % 2 == 0, "centers need an even across count");
    let xs = graded_coords(n_x, LENGTH_M, 0.4, 0.25);
    let ys: Vec<f64> = graded_coords(n_y, WIDTH_M, 0.5, 0.25)
        .iter()
        .map(|y| y - WIDTH_M / 2.0)
        .collect();
    let mut nodes = Vec::with_capacity((n_x + 1) * (n_y + 1));
    for ix in 0..=n_x {
        for iy in 0..=n_y {
            nodes.push([xs[ix], ys[iy], 0.0]);
        }
    }
    let id = |ix: usize, iy: usize| ix * (n_y + 1) + iy + 1;
    let hole_r = HOLE_D_M / 2.0;
    let mut quads = Vec::with_capacity(n_x * n_y);
    let mut probe_shell = 0;
    let mut probe_centroid = [0.0, 0.0];
    let mut best = f64::INFINITY;
    for ix in 0..n_x {
        for iy in 0..n_y {
            let cx = 0.5 * (xs[ix] + xs[ix + 1]);
            let cy = 0.5 * (ys[iy] + ys[iy + 1]);
            let dx = cx - LENGTH_M / 2.0;
            if dx * dx + cy * cy < hole_r * hole_r {
                continue;
            }
            quads.push([
                id(ix, iy),
                id(ix + 1, iy),
                id(ix + 1, iy + 1),
                id(ix, iy + 1),
            ]);
            // Minimum-section candidate on the +y hole edge.
            if cy > 0.0 {
                let d = dx * dx + (cy - hole_r) * (cy - hole_r);
                if d < best {
                    best = d;
                    probe_shell = quads.len();
                    probe_centroid = [cx, cy];
                }
            }
        }
    }
    assert!(probe_shell > 0, "hole edge must leave a probe shell");
    // The probe centroid must sit within two center pitches of the edge.
    let edge_gap = (probe_centroid[1] - hole_r).abs();
    assert!(
        edge_gap < 1.0e-3,
        "probe shell too far from the hole edge: {edge_gap:.3e} m"
    );
    let root: Vec<usize> = (0..=n_y).map(|iy| id(0, iy)).collect();
    let pulled: Vec<usize> = (0..=n_y).map(|iy| id(n_x, iy)).collect();
    OhtMesh {
        nodes,
        quads,
        root,
        root_center: id(0, n_y / 2),
        pulled,
        pulled_center: id(n_x, n_y / 2),
        probe_shell,
        probe_centroid,
    }
}

/// Howland (1930) finite-width stress concentration for a hole in a
/// strip under tension, on the gross stress P / (W t).
fn howland_kt(d_over_w: f64) -> f64 {
    3.0 - 3.13 * d_over_w + 3.66 * d_over_w.powi(2) - 1.53 * d_over_w.powi(3)
}

/// Build the open-hole elastic comparison case.
///
/// The root edge is held in x (plus y at its center node) with rotations
/// locked against rigid motion; the pulled edge carries a uniform total
/// force. The 1 ms ramp spans ~45 wave transits, so the 2 ms plateau is
/// quasi-static. The 10% compliance tolerance covers grip end effects;
/// the 20% Kt tolerance covers the staircase hole edge plus the probe
/// centroid offset from the true edge.
fn shell_oht_elastic_case() -> RadExtendedCase {
    let mesh = oht_mesh(100, 40);
    let kt = howland_kt(HOLE_D_M / WIDTH_M);
    let compliance_m = LOAD_N * LENGTH_M / (YOUNG_PA * WIDTH_M * THICK_M);
    let peak_stress = kt * LOAD_N / (WIDTH_M * THICK_M);
    let per_node_n = LOAD_N / mesh.pulled.len() as f64;
    let base = RadCase {
        name: "shell_oht_elastic".to_string(),
        title: "sonium_qa rad open-hole plate quasi-static tension".to_string(),
        mesh: RadMesh {
            nodes: mesh.nodes.clone(),
            bricks: Vec::new(),
        },
        material: RadElasticMaterial {
            youngs_modulus_pa: YOUNG_PA,
            poisson_ratio: POISSON,
            density_kg_per_m3: DENSITY,
        },
        node_sets: vec![
            RadNodeSet {
                name: "root edge".to_string(),
                nodes: mesh.root,
            },
            RadNodeSet {
                name: "root center".to_string(),
                nodes: vec![mesh.root_center],
            },
            RadNodeSet {
                name: "pulled edge".to_string(),
                nodes: mesh.pulled,
            },
            RadNodeSet {
                name: "pulled center".to_string(),
                nodes: vec![mesh.pulled_center],
            },
        ],
        bcs: vec![
            RadBoundaryCondition {
                name: "hold root edge".to_string(),
                set: 0,
                lock: [true, false, true, true, true, true],
            },
            RadBoundaryCondition {
                name: "kill rigid y".to_string(),
                set: 1,
                lock: [false, true, false, false, false, false],
            },
        ],
        functions: vec![RadFunction {
            name: "ramp 1 ms then hold".to_string(),
            points: vec![(0.0, 0.0), (0.001, 1.0), (0.002, 1.0)],
        }],
        cloads: vec![RadConcentratedLoad {
            name: "uniform pull".to_string(),
            set: 2,
            direction: 1,
            function: 0,
            force_scale_n: per_node_n,
        }],
        walls: Vec::new(),
        th_nodes: vec![mesh.pulled_center],
        probes: vec![
            RadProbe {
                label: "pulled center DX".to_string(),
                channel: format!("DX/{}", mesh.pulled_center),
                expected: compliance_m,
                abs_tol: 1e-9,
                rel_tol: 0.10,
            },
            RadProbe {
                label: "hole edge F1".to_string(),
                channel: shel_channel("F1", mesh.probe_shell),
                expected: peak_stress,
                abs_tol: 1e6,
                rel_tol: 0.20,
            },
            RadProbe {
                label: "hole edge THIC".to_string(),
                channel: shel_channel("THIC", mesh.probe_shell),
                expected: THICK_M,
                abs_tol: 1e-9,
                rel_tol: 1e-6,
            },
        ],
        stop_time_s: 0.002,
        th_period_s: 2e-5,
        anim_period_s: 2e-4,
    };
    let mut case = RadExtendedCase::wrap_elastic(base);
    case.shells = RadShellMesh {
        quads: mesh.quads,
        tris: Vec::new(),
    };
    case.shell_prop = Some(RadShellProperty::qeph("oht section", THICK_M));
    case.th_shell = RadThRequest::shell_resultants(vec![mesh.probe_shell]);
    let _ = mesh.probe_centroid;
    case
}

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/shell_oht_elastic")
}

fn main() {
    let case = shell_oht_elastic_case();
    case.validate().expect("shell_oht_elastic must validate");
    if std::env::var("REGEN_DECKS").as_deref() == Ok("1") {
        let dir = decks_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(
            dir.join("shell_oht_elastic_0000.rad"),
            write_extended_starter(&case),
        )
        .unwrap();
        std::fs::write(
            dir.join("shell_oht_elastic_0001.rad"),
            write_extended_engine(&case),
        )
        .unwrap();
        std::fs::write(
            dir.join("probes.json"),
            extended_probes_json(&case, "none (prove-out deferred)"),
        )
        .unwrap();
        println!("regenerated decks for shell_oht_elastic");
        return;
    }
    let dir = decks_dir();
    let starter_path = dir.join("shell_oht_elastic_0000.rad");
    let engine_path = dir.join("shell_oht_elastic_0001.rad");
    let probes_path = dir.join("probes.json");

    // No hand-built decks: the checked-in trio must match the writer.
    let starter = write_extended_starter(&case);
    let engine = write_extended_engine(&case);
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    let checked_probes = std::fs::read_to_string(&probes_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "decks/shell_oht_elastic/shell_oht_elastic_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "decks/shell_oht_elastic/shell_oht_elastic_0001.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_probes,
        extended_probes_json(&case, "none (prove-out deferred)"),
        "decks/shell_oht_elastic/probes.json drifted from the rad writer"
    );

    let Some(exe) = engine_exe() else {
        println!("skipped: set RADISS_EXE or OPENCOURANT_EXE");
        return;
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!(
        "sonium_rad_shell_oht_elastic_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("shell_oht_elastic_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("shell_oht_elastic_0001.rad"), &engine).unwrap();
    let output = run_extended_case(&work, &exe, &case).unwrap_or_else(|error| {
        eprintln!("rad shell_oht_elastic run failed: {error}");
        std::process::exit(1);
    });
    println!("engine: {}", output.engine);
    let mut failed = false;
    for probe in &case.base.probes {
        match output.history.final_value(&probe.channel) {
            Ok(actual) => {
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
            Err(error) => {
                eprintln!("FAIL: probe {} unreadable: {error}", probe.label);
                failed = true;
            }
        }
    }
    std::fs::remove_dir_all(&work).ok();
    if failed {
        std::process::exit(1);
    }
    println!("PASS: qa-fem-radioss-shell-oht-elastic");
}
