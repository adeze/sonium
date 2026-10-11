//! Phase-4 lane 4B G-leg: writer acceptance.
//!
//! Emits one tiny combined deck exercising every new 4B card family
//! (`/INTER/TYPE2/7/11`, `/TH` completion, `/MONVOL/AIRBAG1`,
//! `/MONVOL/COMMU`, `/STATE`, `//SUBMODEL`, `/DAMP`, `/SENSOR/*`)
//! and checks vendor starter acceptance. Exit 0 = PASS, exit 1 =
//! FAIL with a per-card report on stdout.
//!
//! Starter gate: FAIL when the starter exits nonzero or its log
//! carries an `ERROR` marker; warnings are reported, never fatal.
//! The engine also runs a few cycles as a smoke check of the
//! engine-only cards (`/STATE`, engine `/DAMP`, `/TH/VERS`); it
//! fails the leg only when the engine dies before its first cycle
//! with an error marker (input defect rather than physics).
//!
//! Environment: `OPENCOURANT_EXE` (or `RADISS_EXE`) overrides the
//! default vendor `exec/` dir; `KEEP_SCRATCH=1` keeps the per-run
//! scratch dir under `$TMPDIR` (standing rule: project `tmp/`).

// Rust guideline compliant 2026-02-21

use sonium_qa::rad::{
    RadAirbagInjector, RadAirbagPorous, RadBeamElems, RadBeamProp, RadBoundaryCondition, RadCase,
    RadCommuBag, RadCommuJet, RadCommuVent, RadDamping, RadDampingFull, RadElasticMaterial,
    RadEngineDamping, RadExtendedCase, RadExtra, RadExtraContext, RadExtraPart, RadFunction,
    RadGasKind, RadGasPredef, RadGrnodSurfs, RadInject2, RadInjectGas, RadInterType2,
    RadInterType7, RadInterType11, RadInterface, RadLineSurfs, RadMesh, RadMonvol,
    RadMonvolAirbag1, RadMonvolCommu, RadNodeSet, RadSensor, RadShellMesh, RadShellProperty,
    RadSpringElems, RadSpringProp, RadStateFlag, RadStateOutput, RadSubmodel, RadSubset,
    RadSurfParts, RadThElemFamily, RadThElemRequest, RadThObjKind, RadThObjRequest, RadThRequest,
    RadTrussElems, RadTrussProp, RadWall, engine_exe, extended_starter_th_groups,
    resolve_engine_pair, write_engine_extra, write_extended_engine, write_extended_starter,
    write_starter_extra,
};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

/// Vendor `exec/` dir fallback (task-pinned install path).
fn default_exec_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/home/pierre".to_string());
    PathBuf::from(home).join("src_3rdparties/OpenCourant/exec")
}

/// Apply the vendor starter/engine environment.
///
/// Mirrors `oc_golden` harvest setup: preset variables win, derived
/// paths apply only when they exist, threads pinned to one.
fn apply_vendor_env(command: &mut Command, exec_dir: &Path) {
    if std::env::var("OMP_STACKSIZE").is_err() {
        command.env("OMP_STACKSIZE", "400m");
    }
    command.env("OMP_NUM_THREADS", "1");
    let root = exec_dir.parent().unwrap_or(exec_dir);
    if std::env::var("RAD_CFG_PATH").is_err() {
        let candidate = root.join("hm_cfg_files");
        if candidate.is_dir() {
            command.env("RAD_CFG_PATH", &candidate);
        }
    }
    if std::env::var("RAD_H3D_PATH").is_err() {
        let candidate = root.join("extlib").join("h3d").join("lib").join("linux64");
        if candidate.is_dir() {
            command.env("RAD_H3D_PATH", &candidate);
        }
    }
    let hm_reader = root.join("extlib").join("hm_reader").join("linux64");
    if hm_reader.is_dir() {
        let prepend = hm_reader.to_string_lossy().into_owned();
        let merged = std::env::var("LD_LIBRARY_PATH")
            .map(|current| format!("{prepend}:{current}"))
            .unwrap_or(prepend);
        command.env("LD_LIBRARY_PATH", merged);
    }
}

/// Build the combined acceptance case: closed shell box (part 1),
/// brick on the box nodes (part 2), one free triangle, and 1D
/// nodes for the truss/beam/spring support elements.
fn acceptance_case() -> RadExtendedCase {
    // Box corners 1..8 (1 m cube at the origin).
    let box_nodes = [
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 1.0],
        [1.0, 1.0, 1.0],
        [0.0, 1.0, 1.0],
    ];
    let mut nodes: Vec<[f64; 3]> = box_nodes.to_vec();
    // Free triangle 9..11 hovers 0.2 m off the right box face so the
    // TYPE2 tie projects it onto the skin; truss 12..13, beam
    // 14..16, spring 17..18 follow.
    nodes.extend([
        [1.2, 0.0, 0.0],
        [1.6, 0.0, 0.0],
        [1.2, 0.4, 0.0],
        [0.0, 2.0, 0.0],
        [1.0, 2.0, 0.0],
        [0.0, 3.0, 0.0],
        [1.0, 3.0, 0.0],
        [0.0, 3.0, 1.0],
        [0.0, 4.0, 0.0],
        [1.0, 4.0, 0.0],
    ]);
    let base = RadCase {
        name: "rad4b_accept".to_string(),
        title: "sonium_qa rad lane 4B writer acceptance".to_string(),
        mesh: RadMesh {
            nodes,
            bricks: vec![[1, 2, 3, 4, 5, 6, 7, 8]],
        },
        material: RadElasticMaterial {
            youngs_modulus_pa: 200.0e9,
            poisson_ratio: 0.3,
            density_kg_per_m3: 7800.0,
        },
        node_sets: vec![
            RadNodeSet {
                name: "box bottom".to_string(),
                nodes: vec![1, 2, 3, 4],
            },
            RadNodeSet {
                name: "box top".to_string(),
                nodes: vec![5, 6, 7, 8],
            },
            RadNodeSet {
                name: "tie patch".to_string(),
                nodes: vec![9, 10, 11],
            },
        ],
        bcs: vec![RadBoundaryCondition {
            name: "clamp bottom".to_string(),
            set: 0,
            lock: [true, true, true, true, true, true],
        }],
        functions: vec![
            RadFunction {
                name: "injector mass".to_string(),
                points: vec![(0.0, 0.0), (1.0, 1.0)],
            },
            RadFunction {
                name: "injector temperature".to_string(),
                points: vec![(0.0, 1.0), (1.0, 1.0)],
            },
        ],
        cloads: Vec::new(),
        walls: vec![RadWall {
            name: "floor".to_string(),
            normal: [0.0, 0.0, 1.0],
            point_m: [0.0, 0.0, -0.5],
            friction: 0.0,
            velocity_m_per_s: None,
            driver_node: None,
            // No secondary set: the wall exists only as a
            // `/TH/RWALL` target, avoiding a kinematic clash with
            // the clamped bottom set (starter warning 312).
            secondary_set: None,
            search_distance_m: 0.0,
        }],
        th_nodes: vec![7],
        probes: Vec::new(),
        stop_time_s: 1.0e-6,
        th_period_s: 1.0e-7,
        anim_period_s: 1.0e-6,
    };
    let mut case = RadExtendedCase::wrap_elastic(base);
    case.shells = RadShellMesh {
        // Closed box with outward windings (airbag needs a volume).
        quads: vec![
            [1, 4, 3, 2],
            [5, 6, 7, 8],
            [1, 2, 6, 5],
            [4, 8, 7, 3],
            [1, 5, 8, 4],
            [2, 3, 7, 6],
        ],
        tris: vec![[9, 10, 11]],
    };
    case.shell_prop = Some(RadShellProperty::qeph("box section", 0.01));
    case.th_shell = RadThRequest::shell_resultants(vec![1]);
    case.th_brick = RadThRequest::brick_stress(vec![1]);
    case
}

/// Build the extra bundle exercising every new 4B card family.
fn acceptance_extra() -> RadExtra {
    let mut extra = RadExtra::default();
    extra.surfs.push(RadSurfParts {
        id: 4,
        title: "box skin".to_string(),
        parts: vec![1],
    });
    extra.surfs.push(RadSurfParts {
        id: 5,
        title: "box skin copy".to_string(),
        parts: vec![1],
    });
    extra.grnods.push(RadGrnodSurfs {
        id: 6,
        title: "contact nodes".to_string(),
        surfs: vec![4],
    });
    extra.lines.push(RadLineSurfs {
        id: 1,
        title: "contact edges".to_string(),
        surfs: vec![4],
    });
    extra.subsets.push(RadSubset {
        id: 1,
        title: "all parts".to_string(),
    });
    // TYPE2 ties the hovering triangle (base set 3) onto the box
    // skin; the 5 m search radius covers the 0.6 m worst gap.
    let mut tie = RadInterType2::basic(1, "patch tie", 3, 4);
    tie.dsearch_m = 5.0;
    extra.interfaces.push(RadInterface::Type2(tie));
    extra
        .interfaces
        .push(RadInterface::Type7(RadInterType7::basic(
            2,
            "box contact",
            6,
            4,
        )));
    extra
        .interfaces
        .push(RadInterface::Type11(RadInterType11::basic(
            3,
            "box edges",
            1,
            1,
        )));
    extra.gases.push(RadGasPredef {
        id: 2,
        title: "bag air".to_string(),
        gas: RadGasKind::Air,
    });
    extra.injectors.push(RadInject2 {
        id: 6,
        title: "bag injector".to_string(),
        iflow: 0,
        mass_function: 1,
        temp_function: 2,
        scal_m: 1.0,
        scal_t: 0.0,
        ascal_t: 0.0,
        gases: vec![RadInjectGas {
            mat: 2,
            molar_fraction: 1.0,
            function: 0,
        }],
    });
    extra.sensors.push(RadSensor::Time {
        id: 1,
        title: "fire time".to_string(),
        tdelay_s: 0.0,
    });
    extra.sensors.push(RadSensor::Sens {
        id: 2,
        title: "follow fire".to_string(),
        tdelay_s: 0.0,
        sens_on: 1,
        sens_off: 0,
    });
    extra.sensors.push(RadSensor::Dist {
        id: 3,
        title: "box diagonal".to_string(),
        tdelay_s: 0.0,
        node_a: 1,
        node_b: 7,
        dmin_m: -1.0e30,
        dmax_m: 1.0e30,
    });
    extra.sensors.push(RadSensor::AndOr {
        id: 4,
        title: "both fired".to_string(),
        tdelay_s: 0.0,
        sens_a: 1,
        sens_b: 2,
        is_and: true,
    });
    extra.sensors.push(RadSensor::AndOr {
        id: 5,
        title: "either fired".to_string(),
        tdelay_s: 0.0,
        sens_a: 1,
        sens_b: 2,
        is_and: false,
    });
    extra.monvols.push(RadMonvol::Airbag1(RadMonvolAirbag1 {
        id: 1,
        title: "box bag".to_string(),
        surf_external: 4,
        hconv: 0.0,
        ascale: [0.0; 5],
        mat_gas: 2,
        gas_mu: 0.1,
        pext_pa: 101_325.0,
        t0_k: 0.0,
        iequi: 0,
        ittf: 0,
        injectors: vec![RadAirbagInjector {
            inject_id: 6,
            sens_id: 1,
            ijet: 0,
            nodes: [0; 3],
            jetting: None,
        }],
        vents: Vec::new(),
        porous: vec![RadAirbagPorous {
            surf: 4,
            iformps: 1,
            iblockage: 0,
            title: "fabric".to_string(),
            tstart_s: 0.0,
            tstop_s: 1.0e30,
            dpdef: 0.0,
            dtpdef_s: 0.0,
            idtpdef: 0,
            leakage: None,
            chemkin: None,
        }],
    }));
    // Cross-linked chamber pair mirroring miniqa JETV41B volume 2
    // (zeroed jet and vent, communicating bags).
    for (id, surf, peer) in [(2, 4, 3), (3, 5, 2)] {
        extra.monvols.push(RadMonvol::Commu(RadMonvolCommu {
            id,
            title: format!("chamber {id}"),
            surf_external: surf,
            ascale: [0.0; 5],
            gas_mu: 0.1,
            pext_pa: 101_325.0,
            t0_k: 0.0,
            iequi: 0,
            ittf: 0,
            gamma_i: 1.4,
            cp_i: [1005.0, 0.0, 0.0],
            jets: vec![RadCommuJet {
                gamma: 1.4,
                cp: [1005.0, 0.0, 0.0],
                fct_mass: 0,
                iflow: 0,
                fscale_mass: 0.0,
                fct_temp: 0,
                fscale_temp: 0.0,
                sens_id: 0,
                ijet: 0,
                nodes: [0; 3],
                jetting: None,
            }],
            vents: vec![RadCommuVent {
                surf: 0,
                avent_m2: 0.0,
                bvent: 0.0,
                tstop_s: 0.0,
                tvent_s: 1.0e30,
                dpdef: 0.0,
                dtpdef_s: 0.0,
                fct_v: 0,
                fscale_v: 0.0,
                idtpdef: 0,
                fcts: [0; 3],
                fscales: [0.0; 3],
                fcts_p: [0; 3],
                fscales_p: [0.0; 3],
            }],
            bags: vec![RadCommuBag {
                bag: peer,
                surf_c: 0,
                dpcdef_pa: 0.0,
                acom_m2: 1.0e-3,
                tcom_s: 1.0e30,
                dtpcdef_s: 0.0,
            }],
        }));
    }
    extra.damps.push(RadDamping {
        id: 1,
        title: "bottom damping".to_string(),
        alpha: 1.0,
        beta_s: 0.0,
        grnd: 1,
        skew: 0,
        tstart_s: 0.0,
        tstop_s: 1.0e30,
    });
    extra.full_damps.push(RadDampingFull {
        id: 2,
        title: "top damping".to_string(),
        alpha: [1.0; 6],
        beta_s: [0.0; 6],
        grnd: 2,
        skew: 0,
        tstart_s: 0.0,
        tstop_s: 1.0e30,
    });
    extra.truss_props.push(RadTrussProp {
        id: 3,
        title: "rod".to_string(),
        area_m2: 1.0e-4,
        gap_init_m: 0.0,
    });
    extra.beam_props.push(RadBeamProp {
        id: 4,
        title: "beam".to_string(),
        ismstr: 0,
        dm: 0.0,
        df: 0.0,
        area_m2: 1.0e-4,
        inertia_m4: [1.0e-9, 1.0e-9, 2.0e-9],
        ishear: 0,
    });
    extra.spring_props.push(RadSpringProp {
        id: 5,
        title: "spring".to_string(),
        mass_kg: 1.0e-3,
        inertia: 1.0e-3,
        k_trans: 10000.0,
        k_rot: 100000.0,
    });
    extra.parts.push(RadExtraPart {
        id: 3,
        title: "rod".to_string(),
        prop: 3,
        mat: 1,
    });
    extra.parts.push(RadExtraPart {
        id: 4,
        title: "beam".to_string(),
        prop: 4,
        mat: 1,
    });
    extra.parts.push(RadExtraPart {
        id: 5,
        title: "spring".to_string(),
        prop: 5,
        mat: 0,
    });
    extra.truss.push(RadTrussElems {
        part: 3,
        elems: vec![[12, 13]],
    });
    extra.beams.push(RadBeamElems {
        part: 4,
        elems: vec![[14, 15, 16]],
    });
    extra.springs.push(RadSpringElems {
        part: 5,
        elems: vec![[17, 18]],
    });
    for (family, title) in [
        (RadThElemFamily::Shell, "shells long"),
        (RadThElemFamily::Brick, "bricks long"),
        (RadThElemFamily::Sh3n, "triangles"),
        (RadThElemFamily::Truss, "rods"),
        (RadThElemFamily::Beam, "beams"),
        (RadThElemFamily::Spring, "springs"),
    ] {
        extra.th_elems.push(RadThElemRequest {
            family,
            title: title.to_string(),
            vars: vec!["DEF".to_string()],
            elems: vec![1],
        });
    }
    for (kind, title, ids) in [
        (RadThObjKind::Part, "parts", vec![1, 2, 3, 4, 5]),
        (RadThObjKind::Inter, "interfaces", vec![1, 2, 3]),
        (RadThObjKind::Monvol, "volumes", vec![1, 2, 3]),
        (RadThObjKind::Monv, "volumes short", vec![1]),
        (RadThObjKind::Subs, "subsets", vec![1]),
        (RadThObjKind::Rwall, "walls", vec![1]),
        // Sections, rigid bodies, and accelerometers belong to other
        // lanes; the starter only warns (id 257) on these ids.
        (RadThObjKind::Sectio, "sections", vec![9]),
        (RadThObjKind::Rbody, "bodies", vec![9]),
        (RadThObjKind::Accel, "accelerometers", vec![9]),
    ] {
        extra.th_objs.push(RadThObjRequest {
            kind,
            title: title.to_string(),
            vars: vec!["DEF".to_string()],
            ids,
        });
    }
    extra.submodel = Some(RadSubmodel::plain(1, "lane 4B extra"));
    extra.state = Some(RadStateOutput {
        tstart_s: 0.0,
        tfreq_s: 1.0e-6,
        all_parts: false,
        parts: vec![1, 2],
    });
    extra.state_flags.extend([
        RadStateFlag::BrickAux,
        RadStateFlag::BrickStrain,
        RadStateFlag::BrickStress,
        RadStateFlag::ShellAux,
        RadStateFlag::ShellStrain,
        RadStateFlag::ShellStress,
        RadStateFlag::NodeVel,
    ]);
    extra.engine_damps.push(RadEngineDamping {
        damp_id: 1,
        alpha: 2.0,
        beta_s: 0.0,
    });
    extra.th_vers = Some(41);
    extra
}

/// Headers (card families) emitted into a deck.
///
/// Strips the trailing numeric id (`/TH/PART/2` becomes `/TH/PART`,
/// `/DAMP/1` becomes `/DAMP`) so headers compare by family.
fn deck_headers(deck: &str) -> Vec<String> {
    deck.lines()
        .filter(|line| line.starts_with('/'))
        .map(|line| {
            let mut parts: Vec<&str> = line.split('/').collect();
            if parts.len() > 2
                && parts
                    .last()
                    .is_some_and(|last| last.parse::<usize>().is_ok())
            {
                parts.pop();
            }
            parts.join("/")
        })
        .collect::<std::collections::BTreeSet<_>>()
        .into_iter()
        .collect()
}

/// Error and warning blocks parsed from a starter log.
#[derive(Debug, Default)]
struct LogReport {
    /// True when the log carries an error marker.
    has_error: bool,
    /// Error detail lines (`ERROR ID` blocks, capped).
    errors: Vec<String>,
    /// Warning block cards (`-- BLOCK:` lines, capped).
    warnings: Vec<String>,
    /// Total warning markers seen.
    warning_count: usize,
}

/// Scan a starter/engine log for error and warning markers.
fn scan_log(log: &str) -> LogReport {
    let mut report = LogReport::default();
    let lines: Vec<&str> = log.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        let upper = line.to_uppercase();
        if upper.contains("ERROR ID") || upper.contains("ERROR TERMINATION") {
            report.has_error = true;
            let mut block = vec![line.trim().to_string()];
            for extra in lines.iter().skip(index + 1).take(8) {
                let trimmed = extra.trim();
                if trimmed.is_empty() {
                    break;
                }
                block.push(trimmed.to_string());
                if report.errors.len() + block.len() > 60 {
                    break;
                }
            }
            if report.errors.len() < 60 {
                report.errors.extend(block);
            }
        }
        // Trailer lines look like `5ERROR(S)`; `0ERROR(S)` is clean.
        if let Some(at) = upper.find("ERROR(S)") {
            let digits: String = upper[..at]
                .chars()
                .rev()
                .take_while(|c| c.is_numeric())
                .collect::<String>()
                .chars()
                .rev()
                .collect();
            if digits.parse::<usize>().unwrap_or(0) > 0 {
                report.has_error = true;
            }
        }
        if upper.contains("WARNING ID") {
            report.warning_count += 1;
            for extra in lines.iter().skip(index + 1).take(6) {
                if extra.contains("-- BLOCK:") && report.warnings.len() < 40 {
                    report.warnings.push(extra.trim().to_string());
                    break;
                }
            }
        }
    }
    report
}

/// Run one stage, capturing combined output (never inherits stdin).
fn run_stage(exe: &Path, deck: &Path, dir: &Path, exec_dir: &Path) -> (bool, String) {
    let mut command = Command::new(exe);
    command.arg("-i").arg(deck).current_dir(dir);
    command.stdin(Stdio::null());
    apply_vendor_env(&mut command, exec_dir);
    match command.output() {
        Ok(output) => {
            let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
            combined.push_str(&String::from_utf8_lossy(&output.stderr));
            (output.status.success(), combined)
        }
        Err(error) => (false, format!("failed to spawn {}: {error}", exe.display())),
    }
}

/// Run the engine with a bounded wait, killing it on timeout.
fn run_engine_bounded(
    exe: &Path,
    deck: &Path,
    dir: &Path,
    exec_dir: &Path,
    timeout_secs: u64,
) -> (bool, String) {
    let mut command = Command::new(exe);
    command.arg("-i").arg(deck).current_dir(dir);
    command.stdin(Stdio::null());
    command.stdout(Stdio::piped());
    command.stderr(Stdio::piped());
    apply_vendor_env(&mut command, exec_dir);
    let mut child = match command.spawn() {
        Ok(child) => child,
        Err(error) => return (false, format!("failed to spawn {}: {error}", exe.display())),
    };
    for _ in 0..timeout_secs {
        match child.try_wait() {
            Ok(Some(status)) => {
                let output = child
                    .wait_with_output()
                    .unwrap_or_else(|_| std::process::Output {
                        status,
                        stdout: Vec::new(),
                        stderr: Vec::new(),
                    });
                let mut combined = String::from_utf8_lossy(&output.stdout).into_owned();
                combined.push_str(&String::from_utf8_lossy(&output.stderr));
                return (output.status.success(), combined);
            }
            Ok(None) => std::thread::sleep(std::time::Duration::from_secs(1)),
            Err(error) => return (false, format!("engine wait failed: {error}")),
        }
    }
    let _ = child.kill();
    let _ = child.wait();
    (false, format!("engine timed out after {timeout_secs}s"))
}

fn main() {
    let case = acceptance_case();
    case.validate().expect("acceptance case must validate");
    let extra = acceptance_extra();
    let ctx = RadExtraContext::of_extended(&case);
    if let Err(error) = extra.validate(&ctx) {
        println!("4B-VALIDATE FAIL: {error}");
        std::process::exit(1);
    }
    let first_th = extended_starter_th_groups(&case);
    let mut starter = write_extended_starter(&case);
    assert!(starter.ends_with("/END\n"), "base deck must end with /END");
    starter.truncate(starter.len() - "/END\n".len());
    starter.push_str(&write_starter_extra(&extra, first_th));
    starter.push_str("/END\n");
    let mut engine = write_extended_engine(&case);
    engine.push_str(&write_engine_extra(&extra));

    let scratch = std::env::temp_dir().join(format!("rad4b_accept_{}", std::process::id()));
    if let Err(error) = std::fs::create_dir_all(&scratch) {
        println!("4B-SCRATCH FAIL: {error}");
        std::process::exit(1);
    }
    let keep = std::env::var("KEEP_SCRATCH").as_deref() == Ok("1");
    let cleanup = |scratch: &Path| {
        if !keep {
            std::fs::remove_dir_all(scratch).ok();
        }
    };
    let starter_deck = scratch.join("rad4b_accept_0000.rad");
    let engine_deck = scratch.join("rad4b_accept_0001.rad");
    if std::fs::write(&starter_deck, &starter).is_err()
        || std::fs::write(&engine_deck, &engine).is_err()
    {
        println!("4B-WRITE FAIL: cannot write decks to {}", scratch.display());
        cleanup(&scratch);
        std::process::exit(1);
    }
    let exe = engine_exe().unwrap_or_else(default_exec_dir);
    let exec_dir = if exe.is_dir() {
        exe.clone()
    } else {
        exe.parent()
            .map(Path::to_path_buf)
            .unwrap_or_else(default_exec_dir)
    };
    let (starter_exe, engine_exe) = match resolve_engine_pair(&exe) {
        Ok(pair) => pair,
        Err(error) => {
            println!("4B-RESOLVE FAIL: {error}");
            cleanup(&scratch);
            std::process::exit(1);
        }
    };

    println!("4B starter: {}", starter_exe.display());
    println!("4B engine:  {}", engine_exe.display());
    let (starter_ok, starter_log) = run_stage(&starter_exe, &starter_deck, &scratch, &exec_dir);
    let report = scan_log(&starter_log);
    let starter_fail = !starter_ok || report.has_error;

    // Per-card report: every emitted header with its log status.
    let headers = deck_headers(&starter);
    let mut failed_cards = 0;
    for header in &headers {
        let hit = report.errors.iter().any(|line| line.contains(header));
        if hit {
            failed_cards += 1;
            println!("4B-CARD FAIL {header}");
        }
    }
    let new_cards = [
        "/INTER/TYPE2",
        "/INTER/TYPE7",
        "/INTER/TYPE11",
        "/MONVOL/AIRBAG1",
        "/MONVOL/COMMU",
        "/DAMP",
        "/SENSOR/TIME",
        "/SENSOR/SENS",
        "/SENSOR/DIST",
        "/SENSOR/AND",
        "/SENSOR/OR",
        "//SUBMODEL",
        "/TH/PART",
        "/TH/INTER",
        "/TH/MONVOL",
        "/TH/MONV",
        "/TH/SUBS",
        "/TH/RWALL",
        "/TH/SECTIO",
        "/TH/RBODY",
        "/TH/ACCEL",
        "/TH/SHELL",
        "/TH/BRICK",
        "/TH/SH3N",
        "/TH/TRUSS",
        "/TH/BEAM",
        "/TH/SPRING",
    ];
    let mut missing = 0;
    for card in new_cards {
        if headers.iter().any(|header| header == card) {
            println!("4B-CARD PASS {card}");
        } else {
            missing += 1;
            println!("4B-CARD MISSING {card}");
        }
    }
    println!(
        "4B starter exit_ok={starter_ok} errors={} warnings={}",
        report.has_error as usize, report.warning_count
    );
    for line in report.errors.iter().take(30) {
        println!("4B-LOG-ERR {line}");
    }
    for line in report.warnings.iter().take(20) {
        println!("4B-LOG-WARN {line}");
    }

    // Engine smoke for the engine-only cards (best effort, bounded).
    let (engine_ok, engine_log) =
        run_engine_bounded(&engine_exe, &engine_deck, &scratch, &exec_dir, 180);
    let engine_report = scan_log(&engine_log);
    let cycled = engine_log.contains("CYCLE");
    println!(
        "4B engine exit_ok={engine_ok} cycled={cycled} errors={} warnings={}",
        engine_report.has_error as usize, engine_report.warning_count
    );
    for line in engine_report.errors.iter().take(20) {
        println!("4B-ENG-ERR {line}");
    }
    // The engine gates only on input-phase failure: no cycle plus
    // an error marker means the engine deck itself was rejected.
    let engine_fail = !cycled && (!engine_ok || engine_report.has_error);

    cleanup(&scratch);
    if starter_fail || failed_cards > 0 || missing > 0 {
        println!(
            "4B RESULT FAIL (starter_fail={starter_fail} failed_cards={failed_cards} missing={missing}) scratch={} keep={keep}",
            scratch.display(),
        );
        std::process::exit(1);
    }
    if engine_fail {
        println!(
            "4B RESULT FAIL (engine input rejected) scratch={} keep={keep}",
            scratch.display()
        );
        std::process::exit(1);
    }
    println!("4B RESULT PASS scratch={} keep={keep}", scratch.display());
}
