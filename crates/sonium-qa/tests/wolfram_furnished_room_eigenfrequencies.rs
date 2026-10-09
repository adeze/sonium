//! Literal, opt-in probe for Wolfram's furnished-room eigenfrequency example.
//!
//! Wolfram ships the geometry as the proprietary support asset
//! `SmoothWallModel.stl`; it cannot be vendored in this repository. Set
//! `WOLFRAM_PDEMODELS_SUPPORT_DIR` to the directory containing that file and
//! run this ignored test explicitly.
//!
//! Source:
//! <https://reference.wolfram.com/language/PDEModels/tutorial/Acoustics/ModelCollection/RoomEigenfrequencies.html?view=all>

use sonium_fem::{
    basis::PolynomialDegree,
    mesh::{
        Point, SurfaceFacet, TetrahedralizeConfig, refinement::uniform_refine_checked,
        tetrahedralize_surface_with_facets_result,
    },
    physics::solid::eigenmodes::{AcousticEigenmodeConfig, build_acoustic_eigenmode_problem},
};
use sonium_solvers::LobpcgConfig;
use std::{collections::HashMap, env, fs, path::Path, time::Instant};

const SOUND_SPEED_M_S: f64 = 343.344;
const PUBLISHED_FURNISHED_FREQUENCIES_HZ: [f64; 5] = [0.0, 23.152, 33.982, 40.876, 44.8];

fn read_f32_le(bytes: &[u8]) -> f32 {
    f32::from_le_bytes(bytes.try_into().expect("four-byte float"))
}

fn load_binary_stl(path: &Path) -> (Vec<Point>, Vec<SurfaceFacet>) {
    let bytes = fs::read(path).expect("read Wolfram SmoothWallModel.stl");
    assert!(bytes.len() >= 84, "binary STL header is truncated");
    let triangle_count =
        u32::from_le_bytes(bytes[80..84].try_into().expect("triangle count")) as usize;
    assert_eq!(bytes.len(), 84 + 50 * triangle_count);

    let mut points = Vec::new();
    let mut point_by_bits = HashMap::<[u32; 3], usize>::new();
    let mut facets = Vec::with_capacity(triangle_count);
    for triangle in 0..triangle_count {
        let record = 84 + 50 * triangle;
        let mut nodes = [0; 3];
        for (vertex, node) in nodes.iter_mut().enumerate() {
            let offset = record + 12 + 12 * vertex;
            let xyz = [
                read_f32_le(&bytes[offset..offset + 4]),
                read_f32_le(&bytes[offset + 4..offset + 8]),
                read_f32_le(&bytes[offset + 8..offset + 12]),
            ];
            let key = xyz.map(f32::to_bits);
            *node = *point_by_bits.entry(key).or_insert_with(|| {
                let index = points.len();
                points.push(Point::new_3d(
                    f64::from(xyz[0]),
                    f64::from(xyz[1]),
                    f64::from(xyz[2]),
                ));
                index
            });
        }
        facets.push(SurfaceFacet {
            nodes,
            marker: Some(1),
        });
    }
    (points, facets)
}

#[test]
#[ignore = "requires Wolfram's proprietary SmoothWallModel.stl support asset"]
fn sonium_reproduces_wolfram_furnished_room_eigenfrequencies() {
    let support_dir = env::var_os("WOLFRAM_PDEMODELS_SUPPORT_DIR")
        .expect("set WOLFRAM_PDEMODELS_SUPPORT_DIR to Wolfram PDEModels/SupportFiles");
    let (points, facets) = load_binary_stl(&Path::new(&support_dir).join("SmoothWallModel.stl"));

    // These counts and bounds identify the literal Wolfram support geometry,
    // rather than a box-furniture adaptation.
    assert_eq!(points.len(), 415);
    assert_eq!(facets.len(), 838);
    for (axis, expected_max) in [7.5, 5.0, 3.0].into_iter().enumerate() {
        let min = points
            .iter()
            .map(|point| [point.x, point.y, point.z][axis])
            .fold(f64::INFINITY, f64::min);
        let max = points
            .iter()
            .map(|point| [point.x, point.y, point.z][axis])
            .fold(f64::NEG_INFINITY, f64::max);
        assert_eq!(min, 0.0);
        assert!((max - expected_max).abs() < 1.0e-6);
    }

    let meshing_started = Instant::now();
    let mut mesh = tetrahedralize_surface_with_facets_result(
        &points,
        &facets,
        &TetrahedralizeConfig {
            max_ratio: 1.5,
            max_smooth_iterations: 0,
            add_interior_points: false,
            lloyd_iterations: 0,
            preserve_surface: true,
        },
    )
    .expect("Sonium should conformingly tetrahedralize the furnished-room surface")
    .mesh;
    eprintln!(
        "base furnished-room mesh: {} nodes, {} elements in {:.3?}",
        mesh.nodes.len(),
        mesh.elements.len(),
        meshing_started.elapsed()
    );

    // Wolfram uses h = lambda(200 Hz)/12 = 0.14306 m. Sonium's constrained
    // tetrahedralizer has no target-edge-length control, so uniform refinement
    // is currently the closest public route to a converged low-mode probe.
    uniform_refine_checked(&mut mesh).expect("first uniform tetrahedral refinement");
    eprintln!(
        "first refinement: {} nodes, {} elements",
        mesh.nodes.len(),
        mesh.elements.len()
    );
    uniform_refine_checked(&mut mesh).expect("second uniform tetrahedral refinement");
    eprintln!(
        "second refinement: {} nodes, {} elements",
        mesh.nodes.len(),
        mesh.elements.len()
    );

    let problem = build_acoustic_eigenmode_problem(
        &mesh,
        &AcousticEigenmodeConfig {
            degree: PolynomialDegree::P1,
            sound_speed: SOUND_SPEED_M_S,
            pressure_release_boundary_markers: Vec::new(),
        },
    );
    let result = problem
        .solve(&LobpcgConfig {
            num_eigenvalues: 5,
            max_iterations: 2_000,
            tolerance: 1.0e-8,
            block_padding: 8,
        })
        .expect("solve furnished-room acoustic eigenproblem");
    let mut frequencies = problem.eigenvalues_to_frequencies_hz(&result.eigenvalues);
    frequencies.sort_by(f64::total_cmp);

    for (mode, (&actual, &published)) in frequencies
        .iter()
        .zip(&PUBLISHED_FURNISHED_FREQUENCIES_HZ)
        .enumerate()
    {
        let tolerance_hz = if mode == 0 { 0.5 } else { 0.03 * published };
        assert!(
            (actual - published).abs() <= tolerance_hz,
            "mode {mode}: Sonium {actual:.6} Hz, Wolfram {published:.6} Hz, tolerance {tolerance_hz:.6} Hz"
        );
    }
}
