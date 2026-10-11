//! QA: curved arch cantilever vs Radioss/OpenRadioss explicit quasi-statics.
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` writer, then runs
//! the starter and engine when `RADISS_EXE` or `OPENCOURANT_EXE` is set and
//! compares the brick-mesh tip displacement against the Castigliano curved
//! cantilever closed form. Without an engine it prints the documented skip
//! message and exits 0. Case contributed by the beams lane; Worker 1 owns
//! the runner.
//!
//! The rad API ships brick meshes only (no beam cards), so the arch is a
//! brick twin: a square section swept along a quarter circle. The tip load
//! runs along global -X (the tip tangent), fitting the single-axis /CLOAD
//! card. Gaps for Worker 1: no /PROP/BEAM cards, no curved-geometry mesher
//! in rad.rs (built in-bin here), no twist/rotation probe channels.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    RadBoundaryCondition, RadCase, RadConcentratedLoad, RadElasticMaterial, RadFunction, RadMesh,
    RadNodeSet, RadProbe, assert_probe_close, engine_exe, probes_json, run_case, write_engine,
    write_starter,
};
use std::path::PathBuf;

/// Quarter-circular arch swept with bricks: (mesh, root set, tip set, tip node).
///
/// Root face sits at angle 0 (point (R, 0)); tip face at pi/2 (point
/// (0, R)) with tangent along -X. Node ids are 1-based.
fn arch_brick_mesh(
    n_arc: usize,
    n_sec: usize,
    radius_m: f64,
    width_m: f64,
) -> (RadMesh, Vec<usize>, Vec<usize>, usize) {
    let side = n_sec + 1;
    let id = |i: usize, j: usize, k: usize| i * side * side + j * side + k + 1;
    let mut nodes = Vec::with_capacity((n_arc + 1) * side * side);
    for i in 0..=n_arc {
        let theta = std::f64::consts::FRAC_PI_2 * i as f64 / n_arc as f64;
        let (sin, cos) = (theta.sin(), theta.cos());
        for j in 0..=n_sec {
            let radial = width_m * (j as f64 / n_sec as f64 - 0.5);
            for k in 0..=n_sec {
                let axial = width_m * (k as f64 / n_sec as f64 - 0.5);
                nodes.push([(radius_m + radial) * cos, (radius_m + radial) * sin, axial]);
            }
        }
    }
    let mut bricks = Vec::with_capacity(n_arc * n_sec * n_sec);
    for i in 0..n_arc {
        for j in 0..n_sec {
            for k in 0..n_sec {
                bricks.push([
                    id(i, j, k),
                    id(i + 1, j, k),
                    id(i + 1, j + 1, k),
                    id(i, j + 1, k),
                    id(i, j, k + 1),
                    id(i + 1, j, k + 1),
                    id(i + 1, j + 1, k + 1),
                    id(i, j + 1, k + 1),
                ]);
            }
        }
    }
    let mut root_set = Vec::new();
    let mut tip_set = Vec::new();
    for j in 0..=n_sec {
        for k in 0..=n_sec {
            root_set.push(id(0, j, k));
            tip_set.push(id(n_arc, j, k));
        }
    }
    let tip_node = id(n_arc, n_sec / 2, n_sec / 2);
    (RadMesh { nodes, bricks }, root_set, tip_set, tip_node)
}

/// Single-sourced curved-arch comparison case (SI units).
pub fn beam_curved_arch_case() -> RadCase {
    let (mesh, root_set, tip_set, tip_node) = arch_brick_mesh(20, 2, 1.0, 0.05);
    mesh.validate().unwrap();
    // Castigliano quarter-cantilever tip deflection (tangential tip load):
    // bend P R^3 (3pi/4 - 2)/EI + axial P R pi/(4 EA).
    let r: f64 = 1.0;
    let p: f64 = 1000.0;
    let e: f64 = 200.0e9;
    let pi = std::f64::consts::PI;
    let inertia = 0.05 * 0.05_f64.powi(3) / 12.0;
    let area = 0.05 * 0.05;
    let expected =
        p * r.powi(3) / (e * inertia) * (3.0 * pi / 4.0 - 2.0) + p * r * pi / (4.0 * e * area);
    // /CLOAD applies the scale to EACH node of the set; the tip tangent
    // runs along -X, so the scale is negative.
    let tip_scale = -p / tip_set.len() as f64;
    RadCase {
        name: "beam_curved_arch".to_string(),
        title: "sonium_qa rad beam_curved_arch quasi-static tip load".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: e,
            poisson_ratio: 0.3,
            density_kg_per_m3: 7800.0,
        },
        node_sets: vec![
            RadNodeSet {
                name: "root".to_string(),
                nodes: root_set,
            },
            RadNodeSet {
                name: "tip".to_string(),
                nodes: tip_set,
            },
        ],
        bcs: vec![RadBoundaryCondition {
            name: "clamp root".to_string(),
            set: 0,
            lock: [true, true, true, true, true, true],
        }],
        functions: vec![RadFunction {
            name: "ramp 0 to 1 over stop time".to_string(),
            points: vec![(0.0, 0.0), (0.9, 0.9), (1.0, 1.0)],
        }],
        cloads: vec![RadConcentratedLoad {
            name: "tip tangential load".to_string(),
            set: 1,
            direction: 1,
            function: 0,
            force_scale_n: tip_scale,
        }],
        walls: Vec::new(),
        th_nodes: vec![tip_node],
        probes: vec![RadProbe {
            label: "tip DX".to_string(),
            channel: format!("DX/{tip_node}"),
            expected: -expected,
            abs_tol: 1e-9,
            rel_tol: 0.10,
        }],
        stop_time_s: 1.0,
        th_period_s: 0.01,
        anim_period_s: 0.1,
    }
}

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/beam_curved_arch")
}

fn main() {
    let case = beam_curved_arch_case();
    if std::env::var("REGEN_DECKS").as_deref() == Ok("1") {
        let dir = decks_dir();
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("beam_curved_arch_0000.rad"), write_starter(&case)).unwrap();
        std::fs::write(dir.join("beam_curved_arch_0001.rad"), write_engine(&case)).unwrap();
        std::fs::write(
            dir.join("probes.json"),
            probes_json(&case, "none (prove-out deferred)"),
        )
        .unwrap();
        println!("regenerated decks for beam_curved_arch");
        return;
    }
    let dir = decks_dir();
    let starter_path = dir.join("beam_curved_arch_0000.rad");
    let engine_path = dir.join("beam_curved_arch_0001.rad");
    let probes_path = dir.join("probes.json");

    // No hand-built decks: the checked-in pair must match the writer.
    let starter = write_starter(&case);
    let engine = write_engine(&case);
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "decks/beam_curved_arch/beam_curved_arch_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "decks/beam_curved_arch/beam_curved_arch_0001.rad drifted from the rad writer"
    );
    assert!(
        probes_path.exists(),
        "decks/beam_curved_arch/probes.json must be checked in"
    );
    let _ = probes_json(&case, "none (prove-out deferred)");

    let Some(exe) = engine_exe() else {
        println!("skipped: set RADISS_EXE or OPENCOURANT_EXE");
        return;
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!(
        "sonium_rad_beam_curved_arch_{}",
        std::process::id()
    ));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("beam_curved_arch_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("beam_curved_arch_0001.rad"), &engine).unwrap();
    let output = run_case(&work, &exe, &case).unwrap_or_else(|error| {
        eprintln!("rad beam_curved_arch run failed: {error}");
        std::process::exit(1);
    });
    println!("engine: {}", output.engine);
    let mut failed = false;
    for probe in &case.probes {
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
        println!("FAILED");
        std::process::exit(1);
    }
    println!("SUCCESS");
}
