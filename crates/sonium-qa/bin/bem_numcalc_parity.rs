//! Direct NumCalc (Mesh2HRTF C++) BEM parity gate — Stage-1 follow-up.
//!
//! Generates rigid-sphere scattering fixtures, writes each as a NumCalc
//! `NC.inp` project, runs the Mesh2HRTF NumCalc binary as a subprocess,
//! parses `pEvalGrid`, then solves the same problem with `sonium-bem` and
//! compares per-evaluation-point complex pressure.
//!
//! Coverage deliberately lives in `sonium-qa` rather than quick unit tests:
//! these are solver-to-solver physics checks and can be long-running at
//! higher mesh/frequency tiers. The default tier includes:
//!
//! - the original plane-wave smoke fixture,
//! - a Mesh2HRTF tutorial-shaped rigid sphere with a point source at
//!   `(0, 0.2, 0)` and horizontal-plane evaluation points,
//! - a Mesh2HRTF `test_numcalc`-shaped 4 kHz point-source fixture.
//!
//! NumCalc location resolution (in order):
//!   1. `NUMCALC_BIN` env var
//!   2. `~/Mesh2HRTF/mesh2hrtf/NumCalc/bin/NumCalc`
//!
//! If neither is found, the binary prints `[SKIP]` and exits SUCCESS so
//! CI / non-Mesh2HRTF dev boxes don't fail. To opt in: clone Mesh2HRTF
//! (`git clone https://github.com/Any2HRTF/Mesh2HRTF.git`), build NumCalc
//! (`make` in `mesh2hrtf/NumCalc/src`), then re-run.
//!
//! Acceptance uses loose solver-to-solver tolerances. Both solvers run
//! constant-element BEM with Burton-Miller, but NumCalc uses its own
//! iterative solver stack and singular-quadrature recipe. Matching to a
//! few percent validates that the physics and formulation agree; it is not
//! intended to assert bit-exact numerical equivalence.

use num_complex::Complex64;
use sonium_bem::analytical::sphere_scattering_3d;
use sonium_bem::core::bem_solver::BoundaryConditionType;
use sonium_bem::core::incident::IncidentField;
use sonium_bem::core::mesh::generators::generate_icosphere_mesh;
use sonium_bem::core::types::PhysicsParams;
use sonium_bem::core::{BemProblem, BemSolver, SolverMethod};
use std::f64::consts::PI;
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::Command;

const DEFAULT_DENSITY: f64 = 1.1839;
const EVAL_NODE_ID_BASE: usize = 200_000;
const EVAL_GROUP: usize = 1;

#[derive(Clone, Copy)]
enum SourceSpec {
    PlaneWaveZ,
    PointSource { position: [f64; 3], strength: f64 },
}

impl SourceSpec {
    fn incident(self) -> IncidentField {
        match self {
            SourceSpec::PlaneWaveZ => IncidentField::plane_wave_z(),
            SourceSpec::PointSource { position, strength } => {
                IncidentField::point_source(position, strength)
            }
        }
    }

    fn label(self) -> String {
        match self {
            SourceSpec::PlaneWaveZ => "plane wave +z".to_string(),
            SourceSpec::PointSource { position, strength } => format!(
                "point source at ({:.3}, {:.3}, {:.3}), strength {:.3}",
                position[0], position[1], position[2], strength
            ),
        }
    }
}

struct ParityCase {
    label: &'static str,
    radius: f64,
    frequency_hz: f64,
    speed_of_sound: f64,
    density: f64,
    subdivisions: usize,
    source: SourceSpec,
    eval_points: Vec<[f64; 3]>,
    l2_tol: f64,
    max_tol: f64,
    report_mie: bool,
}

fn locate_numcalc() -> Option<PathBuf> {
    if let Ok(p) = std::env::var("NUMCALC_BIN")
        && Path::new(&p).is_file()
    {
        return Some(PathBuf::from(p));
    }
    if let Ok(home) = std::env::var("HOME") {
        let p = PathBuf::from(home).join("Mesh2HRTF/mesh2hrtf/NumCalc/bin/NumCalc");
        if p.is_file() {
            return Some(p);
        }
    }
    None
}

fn polar_sphere_points(radius: f64, n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| {
            let theta = (i as f64) * PI / (n as f64 - 1.0);
            [radius * theta.sin(), 0.0, radius * theta.cos()]
        })
        .collect()
}

fn horizontal_circle_points(radius: f64, n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| {
            let phi = 2.0 * PI * (i as f64) / n as f64;
            [radius * phi.cos(), radius * phi.sin(), 0.0]
        })
        .collect()
}

fn parity_cases() -> Vec<ParityCase> {
    let mut cases = vec![
        ParityCase {
            label: "legacy rigid sphere, plane-wave smoke fixture",
            radius: 0.05,
            frequency_hz: 1000.0,
            speed_of_sound: 343.0,
            density: DEFAULT_DENSITY,
            subdivisions: 2,
            source: SourceSpec::PlaneWaveZ,
            eval_points: polar_sphere_points(0.20, 8),
            l2_tol: 0.02,
            max_tol: 0.05,
            report_mie: true,
        },
        ParityCase {
            label: "Mesh2HRTF rigid_sphere_scattering tutorial shape",
            radius: 0.10,
            frequency_hz: 1000.0,
            speed_of_sound: 343.18,
            density: DEFAULT_DENSITY,
            subdivisions: 3,
            source: SourceSpec::PointSource {
                position: [0.0, 0.2, 0.0],
                strength: 0.1,
            },
            // Mesh2HRTF's tutorial selects PlaneHorizontal. Use a sparse
            // horizontal slice here to keep the QA gate bounded while
            // preserving the same source/evaluation geometry.
            eval_points: horizontal_circle_points(1.0, 12),
            l2_tol: 0.05,
            max_tol: 0.08,
            report_mie: false,
        },
        ParityCase {
            label: "Mesh2HRTF test_numcalc rigid_point_bem shape",
            radius: 0.10,
            frequency_hz: 4000.0,
            speed_of_sound: 343.0,
            density: DEFAULT_DENSITY,
            subdivisions: 3,
            source: SourceSpec::PointSource {
                position: [0.2, 0.0, 0.0],
                strength: 0.1,
            },
            eval_points: horizontal_circle_points(1.0, 12),
            l2_tol: 0.08,
            max_tol: 0.12,
            report_mie: false,
        },
    ];

    if std::env::var("NUMCALC_MESH2HRTF_HEAVY").as_deref() == Ok("1") {
        cases.push(ParityCase {
            label: "HEAVY Mesh2HRTF point-source tutorial shape, finer sphere",
            radius: 0.10,
            frequency_hz: 4000.0,
            speed_of_sound: 343.18,
            density: DEFAULT_DENSITY,
            subdivisions: 4,
            source: SourceSpec::PointSource {
                position: [0.0, 0.2, 0.0],
                strength: 0.1,
            },
            eval_points: horizontal_circle_points(1.0, 24),
            l2_tol: 0.08,
            max_tol: 0.12,
            report_mie: false,
        });
    }

    cases
}

fn write_object_files(dir: &Path, mesh: &sonium_bem::core::types::Mesh) -> std::io::Result<()> {
    let nodes_path = dir.join("ObjectMeshes/Reference/Nodes.txt");
    let elems_path = dir.join("ObjectMeshes/Reference/Elements.txt");
    fs::create_dir_all(nodes_path.parent().unwrap())?;

    let n_nodes = mesh.nodes.len();
    let mut nf = File::create(&nodes_path)?;
    writeln!(nf, "{n_nodes}")?;
    for i in 0..n_nodes {
        writeln!(
            nf,
            "{i} {:.6e} {:.6e} {:.6e}",
            mesh.nodes[i][0], mesh.nodes[i][1], mesh.nodes[i][2]
        )?;
    }

    let n_elem = mesh.elements.len();
    let mut ef = File::create(&elems_path)?;
    writeln!(ef, "{n_elem}")?;
    for (i, e) in mesh.elements.iter().enumerate() {
        // type=0 (boundary), material=0, group=0
        writeln!(
            ef,
            "{i} {} {} {} 0 0 0",
            e.connectivity[0], e.connectivity[1], e.connectivity[2]
        )?;
    }
    Ok(())
}

fn write_eval_grid(dir: &Path, eval_points: &[[f64; 3]]) -> std::io::Result<Vec<usize>> {
    let nodes_path = dir.join("EvaluationGrids/Far/Nodes.txt");
    let elems_path = dir.join("EvaluationGrids/Far/Elements.txt");
    fs::create_dir_all(nodes_path.parent().unwrap())?;

    let n = eval_points.len();
    let ids: Vec<usize> = (0..n).map(|i| EVAL_NODE_ID_BASE + i).collect();

    let mut nf = File::create(&nodes_path)?;
    writeln!(nf, "{n}")?;
    for (i, p) in eval_points.iter().enumerate() {
        writeln!(nf, "{} {:.6e} {:.6e} {:.6e}", ids[i], p[0], p[1], p[2])?;
    }

    // NumCalc requires evaluation elements (type=2) to attach the eval
    // nodes to an evaluation mesh. It reports pressure at the nodes, so
    // these triangles only need to be topologically valid enough to load.
    let mut ef = File::create(&elems_path)?;
    let n_tri = if n >= 3 { n - 2 } else { 1 };
    writeln!(ef, "{n_tri}")?;
    if n >= 3 {
        for k in 0..(n - 2) {
            let elem_id = EVAL_NODE_ID_BASE + k;
            writeln!(
                ef,
                "{elem_id} {} {} {} 2 0 {EVAL_GROUP}",
                ids[k],
                ids[k + 1],
                ids[k + 2]
            )?;
        }
    } else {
        let a = ids[0];
        let b = ids.get(1).copied().unwrap_or(a);
        writeln!(ef, "{} {a} {b} {b} 2 0 {EVAL_GROUP}", EVAL_NODE_ID_BASE)?;
    }
    Ok(ids)
}

fn write_nc_inp(
    dir: &Path,
    case: &ParityCase,
    n_total_elem: usize,
    n_total_node: usize,
    n_obj_elem: usize,
) -> std::io::Result<()> {
    let path = dir.join("NumCalc/source_1/NC.inp");
    fs::create_dir_all(path.parent().unwrap())?;
    let mut f = File::create(&path)?;
    let last_obj_elem = n_obj_elem - 1;
    let (n_plane_waves, n_point_sources) = match case.source {
        SourceSpec::PlaneWaveZ => (1, 0),
        SourceSpec::PointSource { .. } => (0, 1),
    };
    writeln!(
        f,
        r#"##-------------------------------------------
## sonium-bem NumCalc parity fixture
##-------------------------------------------
Mesh2HRTF 1.0.0
##
{title}
##
## Controlparameter I
0 0 0 0 7 0
##
## Controlparameter II
1 1 0.000001 0.00e+00 1 0 0
##
## Load Frequency Curve
0 2
0.000000 0.000000e+00 0.0
0.000001 {frequency_hz:.6e} 0.0
##
## 1. Main Parameters I
2 {n_total_elem} {n_total_node} 0 0 2 1 0 0
##
## 2. Main Parameters II
{n_plane_waves} {n_point_sources} 0 0.0000e+00 0 0 0
##
## 3. Main Parameters III
0 0 0 0
##
## 4. Main Parameters IV
{c} {rho} 1.0 0.0e+00 0.0e+00 0.0e+00 0.0e+00
##
NODES
../../ObjectMeshes/Reference/Nodes.txt
../../EvaluationGrids/Far/Nodes.txt
##
ELEMENTS
../../ObjectMeshes/Reference/Elements.txt
../../EvaluationGrids/Far/Elements.txt
##
# SYMMETRY
# 0 0 0
# 0.0000e+00 0.0000e+00 0.0000e+00
##
BOUNDARY
ELEM 0 TO {last_obj_elem} VELO 0.0 -1 0.0 -1
RETU
##"#,
        title = case.label,
        frequency_hz = case.frequency_hz,
        c = case.speed_of_sound,
        rho = case.density,
    )?;

    match case.source {
        SourceSpec::PlaneWaveZ => {
            writeln!(
                f,
                r#"
PLANE WAVES
1 0.0000e+00 0.0000e+00 1.0000e+00 1.0000e+00 -1 0.0000e+00 -1
##
# POINT SOURCES
# 0 0.2 0.0 0.0 0.1 -1 0.0 -1"#
            )?;
        }
        SourceSpec::PointSource { position, strength } => {
            writeln!(
                f,
                r#"
# PLANE WAVES
# 1 0.0000e+00 0.0000e+00 1.0000e+00 1.0000e+00 -1 0.0000e+00 -1
##
POINT SOURCES
0 {:.6e} {:.6e} {:.6e} {:.6e} -1 0.0000e+00 -1"#,
                position[0], position[1], position[2], strength
            )?;
        }
    }

    writeln!(
        f,
        r#"
##
# CURVES
# Frequency Factor 0.0
##
POST PROCESS
##
END"#
    )?;
    Ok(())
}

fn run_numcalc(numcalc: &Path, project_dir: &Path) -> Result<(), String> {
    let working = project_dir.join("NumCalc/source_1");
    let status = Command::new(numcalc)
        .current_dir(&working)
        .arg("-istart")
        .arg("1")
        .arg("-iend")
        .arg("1")
        .status()
        .map_err(|e| format!("NumCalc spawn failed: {e}"))?;
    if !status.success() {
        return Err(format!("NumCalc exited with status {status:?}"));
    }
    Ok(())
}

fn parse_p_eval_grid(path: &Path) -> Result<Vec<(usize, Complex64)>, String> {
    let content = fs::read_to_string(path).map_err(|e| format!("read {path:?}: {e}"))?;
    let mut out = Vec::new();
    let mut header_lines = 0usize;
    for line in content.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        // First three non-empty lines are header (Mesh2HRTF version, ?, range).
        if header_lines < 3 {
            header_lines += 1;
            continue;
        }
        let parts: Vec<&str> = trimmed.split_whitespace().collect();
        if parts.len() < 3 {
            continue;
        }
        let id: usize = parts[0].parse().map_err(|e| format!("bad id: {e}"))?;
        let re: f64 = parts[1].parse().map_err(|e| format!("bad re: {e}"))?;
        let im: f64 = parts[2].parse().map_err(|e| format!("bad im: {e}"))?;
        out.push((id, Complex64::new(re, im)));
    }
    if out.is_empty() {
        return Err(format!("no eval-grid pressures parsed from {path:?}"));
    }
    Ok(out)
}

fn solve_with_sonium_bem(mesh: sonium_bem::core::types::Mesh, case: &ParityCase) -> Vec<Complex64> {
    let physics = PhysicsParams::new(case.frequency_hz, case.speed_of_sound, case.density, false);
    let problem = BemProblem {
        mesh,
        physics,
        incident_field: case.source.incident(),
        bc_type: BoundaryConditionType::Rigid,
        use_burton_miller: true,
    };
    let solver = BemSolver::new().with_solver_method(SolverMethod::BiCgStab);
    let solution = solver.solve(&problem).expect("sonium-bem solve");
    case.eval_points
        .iter()
        .map(|p| solution.evaluate_pressure(p))
        .collect()
}

fn report_mie_reference(case: &ParityCase, nc_pressures: &[(usize, Complex64)]) {
    if !case.report_mie {
        return;
    }
    let k = 2.0 * PI * case.frequency_hz / case.speed_of_sound;
    let ka = k * case.radius;
    let eval_r = case.eval_points[0]
        .iter()
        .map(|v| v * v)
        .sum::<f64>()
        .sqrt();
    let theta_grid: Vec<f64> = case
        .eval_points
        .iter()
        .map(|p| (p[2] / eval_r).acos())
        .collect();
    let mie = sphere_scattering_3d(
        k,
        case.radius,
        ((ka + 30.0) as usize).max(15),
        vec![eval_r],
        theta_grid,
    );
    let mie_diffs: Vec<f64> = mie
        .pressure
        .iter()
        .zip(nc_pressures.iter().map(|(_, p)| *p))
        .map(|(a, b)| (a - b).norm() / a.norm().max(1e-30))
        .collect();
    let mie_l2 = {
        let mut diff = 0.0;
        let mut refn = 0.0;
        for (a, b) in mie
            .pressure
            .iter()
            .zip(nc_pressures.iter().map(|(_, p)| *p))
        {
            diff += (a - b).norm_sqr();
            refn += a.norm_sqr();
        }
        (diff / refn.max(1e-30)).sqrt()
    };
    println!(
        "\nFor reference, NumCalc-vs-Mie max per-point: {:.3e}, L2 rel: {mie_l2:.3e}",
        mie_diffs.iter().cloned().fold(0.0_f64, f64::max)
    );
}

fn run_case(numcalc: &Path, case: &ParityCase) -> Result<(), String> {
    let mesh = generate_icosphere_mesh(case.radius, case.subdivisions);
    let n_obj_elem = mesh.elements.len();
    let n_obj_node = mesh.nodes.len();
    let k = 2.0 * PI * case.frequency_hz / case.speed_of_sound;
    let ka = k * case.radius;

    println!("\n=== {} ===", case.label);
    println!(
        "  R={:.3}, f={:.0} Hz, ka={ka:.3}, c={:.2}, rho={:.4}, subdiv={} \
         ({} elements, {} nodes)",
        case.radius,
        case.frequency_hz,
        case.speed_of_sound,
        case.density,
        case.subdivisions,
        n_obj_elem,
        n_obj_node,
    );
    println!(
        "  source: {}; eval points: {}",
        case.source.label(),
        case.eval_points.len()
    );

    let tmp = tempfile::tempdir().expect("tempdir");
    write_object_files(tmp.path(), &mesh).expect("write object files");
    let _eval_ids = write_eval_grid(tmp.path(), &case.eval_points).expect("write eval grid");

    let n_total_node = n_obj_node + case.eval_points.len();
    let n_eval_elem = if case.eval_points.len() >= 3 {
        case.eval_points.len() - 2
    } else {
        1
    };
    let n_total_elem = n_obj_elem + n_eval_elem;

    write_nc_inp(tmp.path(), case, n_total_elem, n_total_node, n_obj_elem).expect("write NC.inp");

    println!("  Running NumCalc...");
    let t_nc = std::time::Instant::now();
    run_numcalc(numcalc, tmp.path())?;
    println!("  NumCalc wall: {:.2} s", t_nc.elapsed().as_secs_f64());

    let p_eval_grid = tmp.path().join("NumCalc/source_1/be.out/be.1/pEvalGrid");
    let nc_pressures = parse_p_eval_grid(&p_eval_grid).inspect_err(|_| {
        if let Ok(entries) = fs::read_dir(tmp.path().join("NumCalc/source_1")) {
            eprintln!("source_1/ contents:");
            for e in entries.flatten() {
                eprintln!("  {:?}", e.path());
            }
        }
    })?;
    if nc_pressures.len() != case.eval_points.len() {
        return Err(format!(
            "{}: NumCalc returned {} eval pressures for {} eval nodes",
            case.label,
            nc_pressures.len(),
            case.eval_points.len()
        ));
    }

    println!("  Running sonium-bem...");
    let t_bem = std::time::Instant::now();
    let bem_pressures = solve_with_sonium_bem(mesh, case);
    println!("  sonium-bem wall: {:.2} s", t_bem.elapsed().as_secs_f64());

    println!(
        "\n  {:>10}  {:>22}  {:>22}  {:>10}",
        "node", "NumCalc p", "sonium-bem p", "rel err"
    );
    println!("  {}", "-".repeat(70));

    let mut max_rel = 0.0_f64;
    let mut sum_diff_sq = 0.0_f64;
    let mut sum_ref_sq = 0.0_f64;

    for (i, (id, p_nc)) in nc_pressures.iter().enumerate() {
        let p_bem = bem_pressures[i];
        let diff = (p_nc - p_bem).norm();
        let denom = p_nc.norm().max(1e-30);
        let rel = diff / denom;
        max_rel = max_rel.max(rel);
        sum_diff_sq += diff * diff;
        sum_ref_sq += p_nc.norm_sqr();
        println!(
            "  {:>10}  {:>10.4e} {:+10.4e}i  {:>10.4e} {:+10.4e}i  {:>10.3e}",
            id, p_nc.re, p_nc.im, p_bem.re, p_bem.im, rel
        );
    }
    let l2_rel = (sum_diff_sq / sum_ref_sq.max(1e-30)).sqrt();
    println!("\n  L2 relative error: {l2_rel:.3e}, max per-point: {max_rel:.3e}");
    report_mie_reference(case, &nc_pressures);

    if !l2_rel.is_finite() || !max_rel.is_finite() {
        return Err(format!("{}: non-finite parity metric", case.label));
    }
    if l2_rel > case.l2_tol {
        return Err(format!(
            "{}: L2 relative error {l2_rel:.3e} exceeds {:.3e}",
            case.label, case.l2_tol
        ));
    }
    if max_rel > case.max_tol {
        return Err(format!(
            "{}: max per-point relative error {max_rel:.3e} exceeds {:.3e}",
            case.label, case.max_tol
        ));
    }

    drop(_eval_ids);
    drop(tmp);
    Ok(())
}

fn main() {
    println!("=== NumCalc / sonium-bem direct BEM parity ===");

    let numcalc = match locate_numcalc() {
        Some(p) => {
            println!("Using NumCalc at: {p:?}");
            p
        }
        None => {
            println!(
                "[SKIP] NumCalc not found. Set NUMCALC_BIN or build it at \
                 ~/Mesh2HRTF/mesh2hrtf/NumCalc/bin/NumCalc to run this gate."
            );
            println!("\nSUCCESS");
            return;
        }
    };

    let mut failures = Vec::new();
    for case in parity_cases() {
        if let Err(e) = run_case(&numcalc, &case) {
            failures.push(e);
        }
    }

    if failures.is_empty() {
        println!("\nSUCCESS");
        return;
    }
    for failure in &failures {
        eprintln!("FAIL: {failure}");
    }
    eprintln!("FAILED");
    std::process::exit(1);
}
