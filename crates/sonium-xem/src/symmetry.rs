//! Scene symmetry detection and configuration.
//!
//! For a rectangular room, symmetry planes can halve (1 plane) or quarter (2 planes)
//! the computational domain. Each source is decomposed into even + odd modes solved
//! on the reduced domain, then reconstructed at arbitrary listening positions.

use crate::{
    BoundaryConfig, CrossoverFilter, Directivity, Point3D, RoomGeometry, RoomSimulation,
    SurfaceConfig,
};

/// A symmetry plane of the scene.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SymmetryPlane {
    /// x = width / 2
    X,
    /// y = depth / 2
    Y,
}

/// Detected symmetry configuration for a simulation.
#[derive(Debug, Clone)]
pub struct SymmetryConfig {
    pub planes: Vec<SymmetryPlane>,
}

impl SymmetryConfig {
    pub fn none() -> Self {
        Self { planes: vec![] }
    }

    pub fn has_x(&self) -> bool {
        self.planes.contains(&SymmetryPlane::X)
    }

    pub fn has_y(&self) -> bool {
        self.planes.contains(&SymmetryPlane::Y)
    }

    /// Number of solves per source: 2^(number of planes).
    /// Each plane doubles the modes (even/odd).
    pub fn num_modes(&self) -> usize {
        1 << self.planes.len()
    }

    /// Domain reduction factor (1, 2, or 4).
    pub fn reduction_factor(&self) -> usize {
        self.num_modes()
    }
}

/// Parity of a mode with respect to each symmetry plane.
/// For N planes, mode index 0..2^N encodes parity as bits:
///   bit 0 = first plane parity (0=even, 1=odd)
///   bit 1 = second plane parity
#[derive(Debug, Clone, Copy)]
pub struct ModeParity {
    mode_index: usize,
    num_planes: usize,
}

impl ModeParity {
    pub fn new(mode_index: usize, num_planes: usize) -> Self {
        Self {
            mode_index,
            num_planes,
        }
    }

    /// Is this mode odd (Dirichlet) on the i-th symmetry plane?
    ///
    /// Out-of-range plane indices report even (`false`) instead of
    /// shifting garbage bits (the old `debug_assert!` was a no-op in
    /// release builds).
    pub fn is_odd(&self, plane_idx: usize) -> bool {
        if plane_idx >= self.num_planes {
            return false;
        }
        (self.mode_index >> plane_idx) & 1 == 1
    }

    /// Sign for reconstruction: +1 if point is on the "original" side of all
    /// planes, -1 for each plane where the point is on the mirrored side and
    /// this mode is odd.
    pub fn sign(&self, point_is_mirrored: &[bool]) -> f64 {
        let mut s = 1.0;
        for (i, &mirrored) in point_is_mirrored.iter().enumerate() {
            if mirrored && self.is_odd(i) {
                s = -s;
            }
        }
        s
    }
}

/// Detect exploitable symmetry planes for a simulation.
///
/// Checks geometry, boundary conditions, and source positions.
/// Only rectangular rooms are considered (L-shaped → no symmetry).
pub fn detect_symmetry(sim: &RoomSimulation) -> SymmetryConfig {
    let (width, depth, _height) = match &sim.room {
        RoomGeometry::Rectangular(r) => (r.width, r.depth, r.height),
        RoomGeometry::LShaped(_) => return SymmetryConfig::none(),
    };

    let mut planes = Vec::new();

    if check_x_symmetry(&sim.boundaries, &sim.sources, width) {
        planes.push(SymmetryPlane::X);
    }
    if check_y_symmetry(&sim.boundaries, &sim.sources, depth) {
        planes.push(SymmetryPlane::Y);
    }

    SymmetryConfig { planes }
}

/// Map a point into the reduced (half/quarter) domain and return which planes
/// were mirrored.
pub fn map_to_reduced_domain(
    point: Point3D,
    symmetry: &SymmetryConfig,
    width: f64,
    depth: f64,
) -> (Point3D, Vec<bool>) {
    let mut p = point;
    let mut mirrored = Vec::with_capacity(symmetry.planes.len());
    for &plane in &symmetry.planes {
        match plane {
            SymmetryPlane::X => {
                let half = width / 2.0;
                if p.x > half {
                    p.x = width - p.x;
                    mirrored.push(true);
                } else {
                    mirrored.push(false);
                }
            }
            SymmetryPlane::Y => {
                let half = depth / 2.0;
                if p.y > half {
                    p.y = depth - p.y;
                    mirrored.push(true);
                } else {
                    mirrored.push(false);
                }
            }
        }
    }
    (p, mirrored)
}

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

fn surface_eq(a: &SurfaceConfig, b: &SurfaceConfig) -> bool {
    match (a, b) {
        (SurfaceConfig::Rigid, SurfaceConfig::Rigid) => true,
        (
            SurfaceConfig::Absorption { coefficient: ca },
            SurfaceConfig::Absorption { coefficient: cb },
        ) => (ca - cb).abs() < 1e-12,
        (
            SurfaceConfig::Impedance { real: ra, imag: ia },
            SurfaceConfig::Impedance { real: rb, imag: ib },
        ) => (ra - rb).abs() < 1e-12 && (ia - ib).abs() < 1e-12,
        (
            SurfaceConfig::Admittance { real: ra, imag: ia },
            SurfaceConfig::Admittance { real: rb, imag: ib },
        ) => (ra - rb).abs() < 1e-12 && (ia - ib).abs() < 1e-12,
        (
            SurfaceConfig::PorousLayer {
                model: ma,
                thickness: ta,
                flow_resistivity: fa,
                air_gap: ga,
                ambient: aa,
            },
            SurfaceConfig::PorousLayer {
                model: mb,
                thickness: tb,
                flow_resistivity: fb,
                air_gap: gb,
                ambient: ab,
            },
        ) => {
            ma == mb
                && (ta - tb).abs() < 1e-12
                && (fa - fb).abs() < 1e-12
                && (ga.unwrap_or(0.0) - gb.unwrap_or(0.0)).abs() < 1e-12
                && (aa.density - ab.density).abs() < 1e-12
                && (aa.speed_of_sound - ab.speed_of_sound).abs() < 1e-12
        }
        _ => false,
    }
}

fn effective_wall<'a>(
    boundaries: &'a BoundaryConfig,
    wall: &'a Option<SurfaceConfig>,
) -> &'a SurfaceConfig {
    wall.as_ref().unwrap_or(&boundaries.walls)
}

fn crossover_eq(a: &CrossoverFilter, b: &CrossoverFilter) -> bool {
    match (a, b) {
        (CrossoverFilter::FullRange, CrossoverFilter::FullRange) => true,
        (
            CrossoverFilter::Lowpass {
                cutoff_freq: fa,
                order: oa,
            },
            CrossoverFilter::Lowpass {
                cutoff_freq: fb,
                order: ob,
            },
        ) => oa == ob && (fa - fb).abs() < 1e-6,
        (
            CrossoverFilter::Highpass {
                cutoff_freq: fa,
                order: oa,
            },
            CrossoverFilter::Highpass {
                cutoff_freq: fb,
                order: ob,
            },
        ) => oa == ob && (fa - fb).abs() < 1e-6,
        (
            CrossoverFilter::Bandpass {
                low_cutoff: la,
                high_cutoff: ha,
                order: oa,
            },
            CrossoverFilter::Bandpass {
                low_cutoff: lb,
                high_cutoff: hb,
                order: ob,
            },
        ) => oa == ob && (la - lb).abs() < 1e-6 && (ha - hb).abs() < 1e-6,
        _ => false,
    }
}

fn approx_eq_slice(a: &[f64], b: &[f64], tol: f64) -> bool {
    a.len() == b.len() && a.iter().zip(b.iter()).all(|(x, y)| (x - y).abs() < tol)
}

fn directivity_eq(a: &Directivity, b: &Directivity) -> bool {
    match (a, b) {
        (
            Directivity::Classical {
                h_angle: ha,
                v_angle: va,
            },
            Directivity::Classical {
                h_angle: hb,
                v_angle: vb,
            },
        ) => (ha - hb).abs() < 1e-6 && (va - vb).abs() < 1e-6,
        (Directivity::Grid(ga), Directivity::Grid(gb)) => {
            // Tolerance-based: exact `==` on floats silently disables
            // symmetry after a JSON round-trip or renormalization. Uses the
            // same 1e-6 tolerance as `Directivity::is_isotropic`.
            approx_eq_slice(ga.horizontal_angles(), gb.horizontal_angles(), 1e-6)
                && approx_eq_slice(ga.vertical_angles(), gb.vertical_angles(), 1e-6)
                && approx_eq_slice(ga.magnitude().as_slice(), gb.magnitude().as_slice(), 1e-6)
        }
        (Directivity::ComplexGrid(ga), Directivity::ComplexGrid(gb)) => {
            // Norm-based complex comparison, mirroring
            // `Directivity::is_mirror_symmetric`.
            approx_eq_slice(ga.horizontal_angles(), gb.horizontal_angles(), 1e-6)
                && approx_eq_slice(ga.vertical_angles(), gb.vertical_angles(), 1e-6)
                && approx_eq_slice(ga.frequencies(), gb.frequencies(), 1e-6)
                && ga.patterns().len() == gb.patterns().len()
                && ga
                    .patterns()
                    .iter()
                    .zip(gb.patterns().iter())
                    .all(|(pa, pb)| {
                        pa.as_slice().len() == pb.as_slice().len()
                            && pa
                                .as_slice()
                                .iter()
                                .zip(pb.as_slice().iter())
                                .all(|(a, b)| (a - b).norm() < 1e-6)
                    })
        }
        _ => false,
    }
}

/// Check that two sources have identical acoustic properties (ignoring position
/// and orientation, which symmetry callers handle separately).
pub fn sources_match(a: &crate::Source, b: &crate::Source) -> bool {
    (a.amplitude - b.amplitude).abs() < 1e-12
        && crossover_eq(&a.crossover, &b.crossover)
        && directivity_eq(&a.directivity, &b.directivity)
}

fn orientations_match_under_x_mirror(a: &Point3D, b: &Point3D) -> bool {
    let eps = 1e-9;
    (a.x + b.x).abs() < eps && (a.y - b.y).abs() < eps && (a.z - b.z).abs() < eps
}

fn orientations_match_under_y_mirror(a: &Point3D, b: &Point3D) -> bool {
    let eps = 1e-9;
    (a.x - b.x).abs() < eps && (a.y + b.y).abs() < eps && (a.z - b.z).abs() < eps
}

fn orientation_lies_in_plane_x(o: &Point3D) -> bool {
    o.x.abs() < 1e-9
}

fn orientation_lies_in_plane_y(o: &Point3D) -> bool {
    o.y.abs() < 1e-9
}

fn check_x_symmetry(boundaries: &BoundaryConfig, sources: &[crate::Source], width: f64) -> bool {
    // Left wall must match right wall
    let left = effective_wall(boundaries, &boundaries.left_wall);
    let right = effective_wall(boundaries, &boundaries.right_wall);
    if !surface_eq(left, right) {
        return false;
    }

    // Every source must be on the plane or have a mirror partner
    let half = width / 2.0;
    let eps = 1e-6 * width;
    let mut used = vec![false; sources.len()];

    for (i, src) in sources.iter().enumerate() {
        if used[i] {
            continue;
        }
        if (src.position.x - half).abs() < eps {
            // Source on the symmetry plane: orientation must lie in the plane
            // (no x-component) for the field itself to be x-symmetric — unless
            // the source is truly isotropic, in which case orientation has no
            // effect. A directional on-plane source must also have a
            // mirror-symmetric pattern: it contributes only even (Neumann)
            // modes, whose reconstruction is the source plus its own image.
            if !src.directivity.is_isotropic()
                && (!orientation_lies_in_plane_x(&src.orientation)
                    || !src.directivity.is_mirror_symmetric())
            {
                return false;
            }
            used[i] = true;
            continue;
        }
        // Find mirror partner (matching position, acoustic properties, and
        // x-mirrored orientation — orientation match is waived for isotropic
        // sources).
        let mirror_x = width - src.position.x;
        let mut found = false;
        for (j, other) in sources.iter().enumerate() {
            if i == j || used[j] {
                continue;
            }
            let orient_ok = src.directivity.is_isotropic()
                || orientations_match_under_x_mirror(&src.orientation, &other.orientation);
            if (other.position.x - mirror_x).abs() < eps
                && (other.position.y - src.position.y).abs() < eps
                && (other.position.z - src.position.z).abs() < eps
                && sources_match(src, other)
                && orient_ok
            {
                used[i] = true;
                used[j] = true;
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }
    true
}

fn check_y_symmetry(boundaries: &BoundaryConfig, sources: &[crate::Source], depth: f64) -> bool {
    let front = effective_wall(boundaries, &boundaries.front_wall);
    let back = effective_wall(boundaries, &boundaries.back_wall);
    if !surface_eq(front, back) {
        return false;
    }

    let half = depth / 2.0;
    let eps = 1e-6 * depth;
    let mut used = vec![false; sources.len()];

    for (i, src) in sources.iter().enumerate() {
        if used[i] {
            continue;
        }
        if (src.position.y - half).abs() < eps {
            if !src.directivity.is_isotropic()
                && (!orientation_lies_in_plane_y(&src.orientation)
                    || !src.directivity.is_mirror_symmetric())
            {
                return false;
            }
            used[i] = true;
            continue;
        }
        let mirror_y = depth - src.position.y;
        let mut found = false;
        for (j, other) in sources.iter().enumerate() {
            if i == j || used[j] {
                continue;
            }
            let orient_ok = src.directivity.is_isotropic()
                || orientations_match_under_y_mirror(&src.orientation, &other.orientation);
            if (other.position.y - mirror_y).abs() < eps
                && (other.position.x - src.position.x).abs() < eps
                && (other.position.z - src.position.z).abs() < eps
                && sources_match(src, other)
                && orient_ok
            {
                used[i] = true;
                used[j] = true;
                found = true;
                break;
            }
        }
        if !found {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CrossoverFilter, Directivity, DirectivityGrid, RectangularRoom, Source};

    /// Truly isotropic monopole (Grid-omni — orientation has no effect on the
    /// radiated field, so it is treated as such by the symmetry checker).
    fn monopole(x: f64, y: f64, z: f64) -> Source {
        Source {
            position: Point3D::new(x, y, z),
            orientation: Point3D::new(0.0, 1.0, 0.0),
            directivity: Directivity::Grid(DirectivityGrid::omnidirectional()),
            amplitude: 1.0,
            crossover: CrossoverFilter::FullRange,
            name: String::new(),
        }
    }

    fn sym_room(sources: Vec<Source>) -> RoomSimulation {
        // 4×3×2.4 room, uniform absorption, sources as given
        RoomSimulation::with_frequencies(
            RoomGeometry::Rectangular(RectangularRoom::new(4.0, 3.0, 2.4)),
            sources,
            vec![Point3D::new(2.0, 1.5, 1.2)],
            20.0,
            500.0,
            10,
        )
    }

    #[test]
    fn test_single_centered_source_has_dual_symmetry() {
        let sim = sym_room(vec![monopole(2.0, 1.5, 1.2)]);
        let sym = detect_symmetry(&sim);
        assert!(sym.has_x());
        assert!(sym.has_y());
        assert_eq!(sym.num_modes(), 4);
    }

    #[test]
    fn test_stereo_pair_has_x_symmetry() {
        // Left/right pair symmetric about x = 2.0
        let sim = sym_room(vec![monopole(0.5, 0.5, 1.2), monopole(3.5, 0.5, 1.2)]);
        let sym = detect_symmetry(&sim);
        assert!(sym.has_x());
        assert!(!sym.has_y()); // not y-symmetric (sources at y=0.5, not y=1.5)
    }

    #[test]
    fn test_asymmetric_sources_no_symmetry() {
        let sim = sym_room(vec![monopole(0.5, 0.5, 1.2), monopole(1.0, 2.0, 1.2)]);
        let sym = detect_symmetry(&sim);
        assert!(!sym.has_x());
        assert!(!sym.has_y());
    }

    #[test]
    fn test_asymmetric_bc_breaks_symmetry() {
        let mut sim = sym_room(vec![monopole(2.0, 1.5, 1.2)]);
        sim.boundaries.left_wall = Some(SurfaceConfig::Absorption { coefficient: 0.5 });
        sim.boundaries.right_wall = Some(SurfaceConfig::Absorption { coefficient: 0.1 });
        let sym = detect_symmetry(&sim);
        assert!(!sym.has_x());
        assert!(sym.has_y()); // y still symmetric
    }

    #[test]
    fn test_mismatched_crossover_breaks_symmetry() {
        let sim = sym_room(vec![
            {
                let mut s = monopole(0.5, 1.5, 1.2);
                s.crossover = CrossoverFilter::Highpass {
                    cutoff_freq: 80.0,
                    order: 4,
                };
                s
            },
            {
                let mut s = monopole(3.5, 1.5, 1.2);
                s.crossover = CrossoverFilter::FullRange;
                s
            },
        ]);
        // Same position, amplitude, directivity — but different crossover
        let sym = detect_symmetry(&sim);
        assert!(!sym.has_x(), "different crossover should break X symmetry");
    }

    #[test]
    fn test_on_plane_mirror_asymmetric_pattern_breaks_symmetry() {
        // Lobe toward local +X (phi = 0): not invariant under the mirror flip.
        let horizontal: Vec<f64> = (0..36).map(|i| i as f64 * 10.0).collect();
        let vertical: Vec<f64> = (0..19).map(|i| i as f64 * 10.0).collect();
        let magnitude =
            sonium_compute::HostMatrix::from_fn(vertical.len(), horizontal.len(), |v, h| {
                0.5 * (1.0 + vertical[v].to_radians().sin() * horizontal[h].to_radians().cos())
            });
        let grid = DirectivityGrid::try_new(horizontal, vertical, magnitude).unwrap();
        let mut src = monopole(2.0, 0.5, 1.2); // on the x = 2.0 plane
        src.directivity = Directivity::Grid(grid);
        src.orientation = Point3D::new(0.0, 1.0, 0.0); // in-plane orientation
        let sim = sym_room(vec![src]);
        let sym = detect_symmetry(&sim);
        assert!(
            !sym.has_x(),
            "on-plane source with mirror-asymmetric pattern must not claim X symmetry"
        );

        // The mirror-symmetric cardioid on the plane is fine.
        let mut card = monopole(2.0, 0.5, 1.2);
        card.directivity = Directivity::Grid(DirectivityGrid::cardioid());
        card.orientation = Point3D::new(0.0, 1.0, 0.0);
        let sym = detect_symmetry(&sym_room(vec![card]));
        assert!(sym.has_x(), "cardioid on the plane keeps X symmetry");
    }

    #[test]
    fn test_map_to_reduced_domain() {
        let sym = SymmetryConfig {
            planes: vec![SymmetryPlane::X, SymmetryPlane::Y],
        };
        let (p, mirrored) = map_to_reduced_domain(Point3D::new(3.5, 2.5, 1.0), &sym, 4.0, 3.0);
        assert!((p.x - 0.5).abs() < 1e-12); // 4.0 - 3.5 = 0.5
        assert!((p.y - 0.5).abs() < 1e-12); // 3.0 - 2.5 = 0.5
        assert!(mirrored[0]); // x was mirrored
        assert!(mirrored[1]); // y was mirrored
    }

    #[test]
    fn test_grid_eq_survives_json_round_trip_noise() {
        // Exact `==` on floats used to disable symmetry after a
        // serialize/deserialize cycle; the comparison is now 1e-6 tolerant.
        let a = Directivity::Grid(DirectivityGrid::cardioid());
        let mut b = DirectivityGrid::cardioid();
        let noisy =
            sonium_compute::HostMatrix::from_fn(19, 36, |v, h| b.magnitude()[[v, h]] + 1e-9);
        b = DirectivityGrid::try_new(
            b.horizontal_angles().to_vec(),
            b.vertical_angles().to_vec(),
            noisy,
        )
        .unwrap();
        assert!(directivity_eq(&a, &Directivity::Grid(b)));
    }

    #[test]
    fn test_complex_grid_sources_match() {
        use crate::ComplexDirectivityGrid;
        let horizontal: Vec<f64> = (0..36).map(|i| i as f64 * 10.0).collect();
        let vertical: Vec<f64> = (0..19).map(|i| i as f64 * 10.0).collect();
        let frequencies = vec![100.0, 200.0];
        let patterns: Vec<sonium_compute::HostMatrix<num_complex::Complex64>> = frequencies
            .iter()
            .map(|_| {
                sonium_compute::HostMatrix::from_elem(
                    vertical.len(),
                    horizontal.len(),
                    num_complex::Complex64::new(1.0, 0.0),
                )
            })
            .collect();
        let ga = ComplexDirectivityGrid::try_new(
            horizontal.clone(),
            vertical.clone(),
            frequencies.clone(),
            patterns.clone(),
        )
        .unwrap();
        let gb =
            ComplexDirectivityGrid::try_new(horizontal, vertical, frequencies, patterns).unwrap();
        assert!(directivity_eq(
            &Directivity::ComplexGrid(ga.clone()),
            &Directivity::ComplexGrid(gb)
        ));
        // Different phase must not match.
        let flipped: Vec<sonium_compute::HostMatrix<num_complex::Complex64>> =
            vec![sonium_compute::HostMatrix::from_elem(
                19,
                36,
                num_complex::Complex64::new(1.0, 1.0),
            )];
        let gc = ComplexDirectivityGrid::try_new(
            (0..36).map(|i| i as f64 * 10.0).collect(),
            (0..19).map(|i| i as f64 * 10.0).collect(),
            vec![150.0],
            flipped,
        )
        .unwrap();
        assert!(!directivity_eq(
            &Directivity::ComplexGrid(ga),
            &Directivity::ComplexGrid(gc)
        ));
    }

    #[test]
    fn test_mode_parity_out_of_range_is_even() {
        let m = ModeParity::new(0b111, 2);
        assert!(!m.is_odd(2));
        assert!(!m.is_odd(usize::MAX));
        assert!(m.is_odd(0));
    }

    #[test]
    fn test_mode_parity_sign() {
        // Mode 0 (even,even): always +1
        let m0 = ModeParity::new(0, 2);
        assert_eq!(m0.sign(&[true, true]), 1.0);

        // Mode 1 (odd on plane 0, even on plane 1)
        let m1 = ModeParity::new(1, 2);
        assert_eq!(m1.sign(&[true, false]), -1.0);
        assert_eq!(m1.sign(&[false, true]), 1.0);

        // Mode 3 (odd, odd): mirrored on both → +1 (two negations cancel)
        let m3 = ModeParity::new(3, 2);
        assert_eq!(m3.sign(&[true, true]), 1.0);
        assert_eq!(m3.sign(&[true, false]), -1.0);
    }
}
