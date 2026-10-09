//! Wolfram golden replay: MeshLang CSG volumes meshed end to end by Gmsh.
//!
//! Requires the `meshlang` cargo feature (MeshLang compiler + Gmsh `.msh`
//! import) and a `gmsh` binary on `PATH` (or `GMSH_PATH`/`MESHLANG_GMSH`).
//! When no Gmsh binary is found the tests print a skip notice and pass, so
//! hermetic CI without Gmsh stays green; the checked-in fixture
//! (`fixtures/wolfram-golden-v1.json`) needs no live Wolfram engine.
//!
//! The oracle chain is three-deep: Wolfram's exact CSG volumes (2*3*5,
//! 1^3-(1/2)^3) are replayed against `mesh_from_mlg_str` output whose
//! volume is summed per-tetrahedron (`|det|/6`) inline here. The box case
//! pins the Box-primitive scaling path; the notched-cube case pins the
//! OpenCASCADE boolean-difference path (`cube - corner`).

use serde::Deserialize;
use sonium_mesher::mesh::{ElementType, Mesh, Point, mesh_from_mlg_str};
use std::collections::HashMap;
use std::process::Command;

const FIXTURE: &str = include_str!("../fixtures/wolfram-golden-v1.json");

#[derive(Debug, Deserialize)]
struct Entry {
    value: String,
    exact: String,
}

#[derive(Debug, Deserialize)]
struct Meta {
    engine: String,
    protocol: String,
}

#[derive(Debug, Deserialize)]
struct Fixture {
    meta: Meta,
    values: HashMap<String, Entry>,
}

fn parse_fixture() -> Fixture {
    let fixture: Fixture = serde_json::from_str(FIXTURE).expect("valid Wolfram mesher fixture");
    assert!(
        fixture.meta.engine.contains("Wolfram Engine"),
        "fixture must record its Wolfram provenance"
    );
    assert_eq!(fixture.meta.protocol, "audit-oracle-hermetic-v1");
    fixture
}

fn golden(fixture: &Fixture, name: &str) -> f64 {
    fixture.values[name]
        .value
        .parse()
        .expect("fixture value is a decimal")
}

fn tet_volume(a: &Point, b: &Point, c: &Point, d: &Point) -> f64 {
    let ab = (b.x - a.x, b.y - a.y, b.z - a.z);
    let ac = (c.x - a.x, c.y - a.y, c.z - a.z);
    let ad = (d.x - a.x, d.y - a.y, d.z - a.z);
    (ab.0 * (ac.1 * ad.2 - ac.2 * ad.1) - ab.1 * (ac.0 * ad.2 - ac.2 * ad.0)
        + ab.2 * (ac.0 * ad.1 - ac.1 * ad.0))
        / 6.0
}

fn mesh_volume(mesh: &Mesh) -> (f64, usize) {
    let mut volume = 0.0;
    let mut count = 0;
    for element in &mesh.elements {
        if element.element_type != ElementType::Tetrahedron {
            continue;
        }
        let p = &element.nodes;
        volume += tet_volume(
            &mesh.nodes[p[0]],
            &mesh.nodes[p[1]],
            &mesh.nodes[p[2]],
            &mesh.nodes[p[3]],
        )
        .abs();
        count += 1;
    }
    (volume, count)
}

/// True when a Gmsh binary answers `--version`; otherwise the caller skips.
fn gmsh_available() -> bool {
    let binary = std::env::var("GMSH_PATH")
        .or_else(|_| std::env::var("MESHLANG_GMSH"))
        .unwrap_or_else(|_| "gmsh".to_string());
    Command::new(binary)
        .arg("--version")
        .output()
        .is_ok_and(|output| output.status.success())
}

fn check_csg_volume(source: &str, name: &str, exact_symbolic: &str, tol: f64) {
    if !gmsh_available() {
        eprintln!("skipping {name}: no gmsh binary on PATH");
        return;
    }
    let fixture = parse_fixture();
    assert_eq!(fixture.values[name].exact.as_str(), exact_symbolic);
    let mesh = mesh_from_mlg_str(source, "wolfram_golden.mlg")
        .unwrap_or_else(|error| panic!("{name}: mlg->gmsh pipeline failed: {error}"));
    let (volume, count) = mesh_volume(&mesh);
    assert!(count > 0, "{name}: gmsh mesh must contain tetrahedra");
    let expected = golden(&fixture, name);
    let rel = ((volume - expected) / expected).abs();
    assert!(
        rel < tol,
        "{name}: CSG mesh volume {volume} vs Wolfram {expected} (rel {rel:.2e})"
    );
}

const BOX_MLG: &str = r#"
geometry box_vol {
    let b = Box(2.0, 3.0, 5.0)
}

mesh for box_vol {
    default_size: 1.0
}
"#;

const NOTCH_MLG: &str = r#"
geometry notch_vol {
    let cube = Box(1.0, 1.0, 1.0)
    let corner = Box(0.5, 0.5, 0.5)
    let part = cube - corner
}

mesh for notch_vol {
    default_size: 0.15
}
"#;

const SPHERE_MLG: &str = r#"
geometry sphere_vol {
    let s = Sphere(r: 1.0)
}

mesh for sphere_vol {
    default_size: 0.3
}
"#;

// Partial-overlap difference with a translated inner box: the affine
// `translate` path is load-bearing here. An identity-translate bug would
// nest the boxes fully and report 7.0 instead of 7.5.
const SHIFTED_NOTCH_MLG: &str = r#"
geometry shifted_notch_vol {
    let outer = Box(2.0, 2.0, 2.0)
    let inner = Box(1.0, 1.0, 1.0) |> translate(1.5, 0.0, 0.0)
    let part = outer - inner
}

mesh for shifted_notch_vol {
    default_size: 0.2
}
"#;

#[test]
fn meshlang_box_volume_matches_wolfram() {
    // Gmsh tiles the box with exact planar facets: tight band.
    check_csg_volume(BOX_MLG, "box_2x3x5_volume", "30", 1e-9);
}

#[test]
fn meshlang_notched_cube_volume_matches_wolfram() {
    // Boolean difference keeps planar faces; the band only absorbs Gmsh's
    // decimal coordinate round-trip, not shape error.
    check_csg_volume(NOTCH_MLG, "notched_cube_volume", "7/8", 1e-6);
}

#[test]
fn meshlang_translated_difference_matches_wolfram() {
    // The 7.5 golden is unreachable unless `translate` moves the inner box
    // to a partial overlap: identity translation would nest fully (7.0),
    // and a dropped difference would leave the full box (8.0).
    check_csg_volume(SHIFTED_NOTCH_MLG, "shifted_notch_volume", "15/2", 1e-6);
}

#[test]
fn meshlang_sphere_volume_converges_to_wolfram_with_sagitta_bound() {
    // Faceted-sphere deficit bound, derived not fitted: Gmsh places surface
    // nodes on the true sphere, so every boundary facet of longest edge h
    // cuts off at most its area times the exact spherical sagitta
    // s(h) = R - sqrt(R^2 - h^2/4). Summed over the surface,
    // deficit <= 4 pi R^2 s(h_max) with h_max measured from the imported
    // boundary triangles. The Wolfram golden is the exact 4 pi/3.
    if !gmsh_available() {
        eprintln!("skipping unit_sphere_volume: no gmsh binary on PATH");
        return;
    }
    let fixture = parse_fixture();
    assert_eq!(
        fixture.values["unit_sphere_volume"].exact.as_str(),
        "(4*Pi)/3"
    );
    let mesh = mesh_from_mlg_str(SPHERE_MLG, "wolfram_golden.mlg")
        .unwrap_or_else(|error| panic!("sphere mlg->gmsh pipeline failed: {error}"));
    let (volume, count) = mesh_volume(&mesh);
    assert!(count > 0, "sphere mesh must contain tetrahedra");

    let mut max_edge: f64 = 0.0;
    let mut boundary_triangles = 0;
    for face in &mesh.boundaries {
        if face.nodes.len() != 3 {
            continue;
        }
        boundary_triangles += 1;
        for edge in 0..3 {
            let a = &mesh.nodes[face.nodes[edge]];
            let b = &mesh.nodes[face.nodes[(edge + 1) % 3]];
            let length = ((b.x - a.x).powi(2) + (b.y - a.y).powi(2) + (b.z - a.z).powi(2)).sqrt();
            max_edge = max_edge.max(length);
        }
    }
    assert!(
        boundary_triangles > 0,
        "sphere mesh must expose boundary faces"
    );

    let expected = golden(&fixture, "unit_sphere_volume");
    let deficit = expected - volume;
    assert!(deficit > 0.0, "inscribed facets must sit inside the sphere");
    let sagitta = 1.0 - (1.0 - max_edge * max_edge / 4.0).sqrt();
    let bound = 4.0 * std::f64::consts::PI * sagitta;
    assert!(
        deficit < bound,
        "sphere deficit {deficit} exceeds sagitta bound {bound} at h_max {max_edge}"
    );
}
