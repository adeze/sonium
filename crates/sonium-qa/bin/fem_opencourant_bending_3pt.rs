//! External reference health and native parity gate for bending-3pt.
//!
//! Verifies the checked-in decks match the `sonium_qa::rad` writer, then runs the starter and
//! engine when `OPENCOURANT_EXE` (or legacy `RADISS_EXE`) is set and compares the center
//! deflection against Euler-Bernoulli (FL^3/48EI).
//!
//! Supports are transverse BCS pins at both ends with a ramped center concentrated load: the
//! supported-card equivalent of the 3-point rig. Indenter-contact and plastic-hinge probes need
//! the shell and plastic LAW cards tracked for Worker 1.
//!
//! Missing references emit a structured optional skip, or fail when
//! SONIUM_QA_REQUIRE_OPENCOURANT=1. External probe health is recorded separately; release
//! success also requires a matching native solve and full named phenomenon.

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    OpenCourantCase, RadBoundaryCondition, RadConcentratedLoad, RadElasticMaterial, RadFunction,
    RadMesh, RadNodeSet, RadProbe, assert_probe_close, engine_exe, probes_json, run_case,
    write_engine, write_starter,
};
use std::path::PathBuf;

#[path = "support/cantilever_native.rs"]
mod cantilever_native;

const NAME: &str = "bending_3pt";

/// Builds the 3-point bending case on the real `sonium_qa::rad` API.
fn bending_case() -> OpenCourantCase {
    let length = 1.0;
    let section = 0.05;
    let youngs = 200.0e9;
    let force = 100.0;
    let (mesh, root_set, tip_set, center_set, center_node, left_center) =
        beam_mesh(20, 2, 2, length, section);
    let inertia = section * section.powi(3) / 12.0;
    let expected = force * length.powi(3) / (48.0 * youngs * inertia);
    let center_scale = force / center_set.len() as f64;
    OpenCourantCase {
        name: NAME.to_string(),
        title: "sonium_qa rad three point bending quasi-static".to_string(),
        mesh,
        material: RadElasticMaterial {
            youngs_modulus_pa: youngs,
            poisson_ratio: 0.3,
            density_kg_per_m3: 7800.0,
        },
        node_sets: vec![
            RadNodeSet {
                name: "left support".to_string(),
                nodes: root_set,
            },
            RadNodeSet {
                name: "right support".to_string(),
                nodes: tip_set,
            },
            RadNodeSet {
                name: "load line".to_string(),
                nodes: center_set,
            },
            RadNodeSet {
                name: "left center".to_string(),
                nodes: vec![left_center],
            },
        ],
        bcs: vec![
            RadBoundaryCondition {
                name: "pin left".to_string(),
                set: 0,
                lock: [true, false, true, false, false, false],
            },
            RadBoundaryCondition {
                name: "roller right".to_string(),
                set: 1,
                lock: [false, false, true, false, false, false],
            },
            RadBoundaryCondition {
                name: "kill rigid y".to_string(),
                set: 3,
                lock: [false, true, false, false, false, false],
            },
        ],
        functions: vec![RadFunction {
            name: "ramp 0 to 1 over stop time".to_string(),
            points: vec![(0.0, 0.0), (0.9, 0.9), (1.0, 1.0)],
        }],
        cloads: vec![RadConcentratedLoad {
            name: "center load".to_string(),
            set: 2,
            direction: 3,
            function: 0,
            force_scale_n: center_scale,
        }],
        walls: Vec::new(),
        th_nodes: vec![center_node],
        probes: vec![RadProbe {
            label: "center DZ".to_string(),
            channel: format!("DZ/{center_node}"),
            expected,
            abs_tol: 1e-9,
            rel_tol: 0.10,
        }],
        stop_time_s: 1.0,
        th_period_s: 0.01,
        anim_period_s: 0.1,
    }
}

/// Bricks a beam with 1-based connectivity plus support/load sets.
///
/// Returns the mesh, the end neutral-axis support lines (true simple
/// supports: sections rotate and warp freely), the center cross-section
/// set, the center node id, and the left-center node id. Requires even
/// axial and transverse counts for center planes.
fn beam_mesh(
    nx: usize,
    ny: usize,
    nz: usize,
    length: f64,
    section: f64,
) -> (RadMesh, Vec<usize>, Vec<usize>, Vec<usize>, usize, usize) {
    assert!(
        nx.is_multiple_of(2),
        "bending mesh needs an even axial count"
    );
    let mut nodes = Vec::with_capacity((nx + 1) * (ny + 1) * (nz + 1));
    let id = |i: usize, j: usize, k: usize| i * (ny + 1) * (nz + 1) + j * (nz + 1) + k + 1;
    for i in 0..=nx {
        for j in 0..=ny {
            for k in 0..=nz {
                nodes.push([
                    length * i as f64 / nx as f64,
                    section * (j as f64 / ny as f64 - 0.5),
                    section * (k as f64 / nz as f64 - 0.5),
                ]);
            }
        }
    }
    let mut bricks = Vec::with_capacity(nx * ny * nz);
    for i in 0..nx {
        for j in 0..ny {
            for k in 0..nz {
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
    // Knife-edge supports on the end neutral-axis lines: locking full
    // end planes would clamp end rotation (measured 2.3x too stiff).
    let mut root_set = Vec::new();
    let mut tip_set = Vec::new();
    let mut center_set = Vec::new();
    for j in 0..=ny {
        root_set.push(id(0, j, nz / 2));
        tip_set.push(id(nx, j, nz / 2));
        for k in 0..=nz {
            center_set.push(id(nx / 2, j, k));
        }
    }
    let center_node = id(nx / 2, ny / 2, nz / 2);
    let left_center = id(0, ny / 2, nz / 2);
    (
        RadMesh { nodes, bricks },
        root_set,
        tip_set,
        center_set,
        center_node,
        left_center,
    )
}

fn decks_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("decks/bending_3pt")
}

const CASE_ID: &str = "fem.opencourant.bending-3pt.native-parity.v1";

fn main() {
    let case = bending_case();
    let dir = decks_dir();
    let starter_path = dir.join("bending_3pt_0000.rad");
    let engine_path = dir.join("bending_3pt_0001.rad");
    let probes_path = dir.join("probes.json");

    let starter = write_starter(&case);
    let engine = write_engine(&case);
    if std::env::var("SONIUM_RAD_WRITE_DECKS").is_ok() {
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(&starter_path, &starter).unwrap();
        std::fs::write(&engine_path, &engine).unwrap();
        std::fs::write(
            &probes_path,
            probes_json(&case, "none (prove-out deferred)"),
        )
        .unwrap();
        println!("wrote decks/bending_3pt artifacts");
        return;
    }
    // No hand-built decks: the checked-in pair must match the writer.
    let checked_starter = std::fs::read_to_string(&starter_path).unwrap_or_default();
    let checked_engine = std::fs::read_to_string(&engine_path).unwrap_or_default();
    assert_eq!(
        checked_starter, starter,
        "decks/bending_3pt/bending_3pt_0000.rad drifted from the rad writer"
    );
    assert_eq!(
        checked_engine, engine,
        "decks/bending_3pt/bending_3pt_0001.rad drifted from the rad writer"
    );
    assert!(
        probes_path.exists(),
        "decks/bending_3pt/probes.json must be checked in"
    );
    let _ = probes_json(&case, "none (prove-out deferred)");

    let Some(exe) = engine_exe() else {
        sonium_qa::rad::gate::report_missing_engine(CASE_ID);
    };

    // Run in a scratch dir so engine artifacts never pollute the repo.
    let work = std::env::temp_dir().join(format!("sonium_rad_bending_{}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    std::fs::write(work.join("bending_3pt_0000.rad"), &starter).unwrap();
    std::fs::write(work.join("bending_3pt_0001.rad"), &engine).unwrap();
    let output = run_case(&work, &exe, &case).unwrap_or_else(|error| {
        sonium_qa::rad::gate::publish_and_exit(sonium_qa::rad::gate::execution_failure_result(
            CASE_ID,
            &error.to_string(),
        ));
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
    let result = output
        .history
        .final_value(&case.probes[0].channel)
        .map_err(|error| error.to_string())
        .and_then(|probe| {
            cantilever_native::parity_result(&case, CASE_ID, &output.engine, probe, !failed)
                .map_err(|error| error.to_string())
        })
        .unwrap_or_else(|error| {
            sonium_qa::QaResult::failed(
                CASE_ID,
                format!("native three-point transient validation failed: {error}"),
            )
            .requested_backend("cpu+opencourant")
            .actual_backend("cpu+opencourant")
            .metric("external_reference_executed", 1.0)
            .metric("native_parity_executed", 0.0)
        });
    sonium_qa::rad::gate::publish_and_exit(result);
}

#[cfg(test)]
#[test]
fn original_three_point_supports_cannot_be_replaced_by_end_clamps() {
    let mut case = bending_case();
    cantilever_native::validate_case(&case).unwrap();
    case.bcs[0].lock = [true; 6];
    assert!(cantilever_native::validate_case(&case).is_err());
    case = bending_case();
    case.node_sets[case.bcs[1].set].nodes.pop();
    assert!(cantilever_native::validate_case(&case).is_err());
}
