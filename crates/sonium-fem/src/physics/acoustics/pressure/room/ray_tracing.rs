//! Geometrical acoustics ray tracing for room analysis
//!
//! Traces acoustic rays through a room geometry, computing specular and
//! diffuse reflections at walls. Builds an energetic impulse response at
//! each receiver for subsequent room acoustic metric computation.
//!
//! # Algorithm
//!
//! 1. Emit N rays quasi-uniformly from the source (Fibonacci spiral).
//! 2. For each ray, trace through the room:
//!    a. Find the nearest wall intersection (Möller-Trumbore).
//!    b. Apply wall absorption (energy × (1 − α)).
//!    c. Apply air absorption along the path length.
//!    d. Check receiver sphere detection.
//!    e. Reflect (specular or diffuse via scattering coefficient).
//! 3. Collect ray arrivals into time-binned histograms at each receiver.

use std::f64::consts::PI;

// ---------------------------------------------------------------------------
// 3D vector helpers (self-contained, no external dependency)
// ---------------------------------------------------------------------------

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(v: [f64; 3], s: f64) -> [f64; 3] {
    [v[0] * s, v[1] * s, v[2] * s]
}

fn length(v: [f64; 3]) -> f64 {
    dot(v, v).sqrt()
}

fn normalize(v: [f64; 3]) -> [f64; 3] {
    let len = length(v);
    if len < 1e-15 {
        return [0.0, 0.0, 0.0];
    }
    scale(v, 1.0 / len)
}

fn distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    length(sub(a, b))
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// An acoustic ray with position, direction, energy, and accumulated time.
#[derive(Debug, Clone)]
pub struct Ray {
    pub origin: [f64; 3],
    pub direction: [f64; 3],
    pub energy: f64,
    pub time: f64,
    pub order: usize,
}

/// A triangular surface in the room with material assignment.
#[derive(Debug, Clone)]
pub struct Surface {
    pub vertices: [[f64; 3]; 3],
    pub normal: [f64; 3],
    pub material_index: usize,
}

impl Surface {
    /// Build from three vertices; computes the outward normal.
    pub fn from_vertices(v0: [f64; 3], v1: [f64; 3], v2: [f64; 3], material_index: usize) -> Self {
        let e1 = sub(v1, v0);
        let e2 = sub(v2, v0);
        let normal = normalize(cross(e1, e2));
        Self {
            vertices: [v0, v1, v2],
            normal,
            material_index,
        }
    }
}

/// Frequency-dependent surface material (octave bands: 125–4000 Hz).
#[derive(Debug, Clone)]
pub struct SurfaceMaterial {
    /// Name for identification.
    pub name: String,
    /// Absorption coefficients at \[125, 250, 500, 1k, 2k, 4k\] Hz.
    pub absorption: [f64; 6],
    /// Scattering coefficients at \[125, 250, 500, 1k, 2k, 4k\] Hz.
    pub scattering: [f64; 6],
}

impl SurfaceMaterial {
    pub fn new(name: &str, absorption: [f64; 6], scattering: [f64; 6]) -> Self {
        Self {
            name: name.to_string(),
            absorption,
            scattering,
        }
    }

    /// Concrete or painted plaster (typical).
    pub fn concrete() -> Self {
        Self::new(
            "concrete",
            [0.01, 0.01, 0.02, 0.02, 0.02, 0.03],
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        )
    }

    /// Heavy curtains.
    pub fn curtain() -> Self {
        Self::new(
            "curtain",
            [0.07, 0.31, 0.49, 0.75, 0.70, 0.60],
            [0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
        )
    }

    /// Carpet on concrete.
    pub fn carpet() -> Self {
        Self::new(
            "carpet",
            [0.02, 0.06, 0.14, 0.37, 0.60, 0.65],
            [0.10, 0.10, 0.15, 0.20, 0.30, 0.40],
        )
    }

    /// Glass window.
    pub fn glass() -> Self {
        Self::new(
            "glass",
            [0.35, 0.25, 0.18, 0.12, 0.07, 0.04],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.05],
        )
    }

    /// Wooden floor.
    pub fn wood() -> Self {
        Self::new(
            "wood",
            [0.15, 0.11, 0.10, 0.07, 0.06, 0.07],
            [0.10, 0.10, 0.10, 0.10, 0.10, 0.10],
        )
    }

    /// Acoustic ceiling tiles.
    pub fn acoustic_tile() -> Self {
        Self::new(
            "acoustic_tile",
            [0.50, 0.70, 0.60, 0.70, 0.70, 0.50],
            [0.15, 0.20, 0.30, 0.40, 0.50, 0.50],
        )
    }
}

/// Octave band center frequencies \[Hz\].
pub const OCTAVE_BANDS: [f64; 6] = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0];

/// A spherical receiver that collects ray energy.
#[derive(Debug, Clone)]
pub struct Receiver {
    pub position: [f64; 3],
    pub radius: f64,
    /// Energy histogram per octave band: `histograms[band][time_bin]`.
    pub histograms: Vec<Vec<f64>>,
    /// Time resolution of the histogram \[s\].
    pub dt: f64,
}

impl Receiver {
    pub fn new(position: [f64; 3], radius: f64, max_time: f64, dt: f64) -> Self {
        let n_bins = (max_time / dt).ceil() as usize + 1;
        Self {
            position,
            radius,
            histograms: vec![vec![0.0; n_bins]; OCTAVE_BANDS.len()],
            dt,
        }
    }

    /// Record a ray arrival at a given time and energy (per octave band).
    fn record(&mut self, time: f64, energies: &[f64; 6]) {
        let bin = (time / self.dt) as usize;
        for (band, hist) in self.histograms.iter_mut().enumerate() {
            if bin < hist.len() {
                hist[bin] += energies[band];
            }
        }
    }

    /// Check if a ray segment from `origin` in `direction` for distance `dist`
    /// passes through this receiver sphere.
    fn intersects_segment(&self, origin: [f64; 3], direction: [f64; 3], dist: f64) -> bool {
        let oc = sub(origin, self.position);
        let b = dot(oc, direction);
        let c = dot(oc, oc) - self.radius * self.radius;
        let discriminant = b * b - c;
        if discriminant < 0.0 {
            return false;
        }
        let t = -b - discriminant.sqrt();
        if t > 0.0 && t < dist {
            return true;
        }
        let t2 = -b + discriminant.sqrt();
        t2 > 0.0 && t2 < dist
    }
}

// ---------------------------------------------------------------------------
// Ray-triangle intersection (Möller-Trumbore)
// ---------------------------------------------------------------------------

/// Result of a ray-surface intersection test.
pub struct HitRecord {
    pub distance: f64,
    pub surface_index: usize,
}

/// Find the nearest surface intersection for a ray.
///
/// Returns `None` if no intersection is found.
pub fn nearest_intersection(
    origin: [f64; 3],
    direction: [f64; 3],
    surfaces: &[Surface],
) -> Option<HitRecord> {
    let mut best: Option<HitRecord> = None;
    let eps = 1e-8;

    for (idx, surf) in surfaces.iter().enumerate() {
        let v0 = surf.vertices[0];
        let edge1 = sub(surf.vertices[1], v0);
        let edge2 = sub(surf.vertices[2], v0);
        let h = cross(direction, edge2);
        let a = dot(edge1, h);

        if a.abs() < eps {
            continue; // Ray parallel to triangle
        }

        let f = 1.0 / a;
        let s = sub(origin, v0);
        let u = f * dot(s, h);
        if !(0.0..=1.0).contains(&u) {
            continue;
        }

        let q = cross(s, edge1);
        let v = f * dot(direction, q);
        if v < 0.0 || u + v > 1.0 {
            continue;
        }

        let t = f * dot(edge2, q);
        if t > eps && best.as_ref().is_none_or(|b| t < b.distance) {
            best = Some(HitRecord {
                distance: t,
                surface_index: idx,
            });
        }
    }
    best
}

// ---------------------------------------------------------------------------
// Ray emission
// ---------------------------------------------------------------------------

/// Generate quasi-uniform ray directions on the unit sphere using the
/// Fibonacci spiral method.
pub fn fibonacci_sphere_directions(n: usize) -> Vec<[f64; 3]> {
    let golden_ratio = (1.0 + 5.0_f64.sqrt()) / 2.0;
    (0..n)
        .map(|i| {
            let theta = (1.0 - 2.0 * (i as f64 + 0.5) / n as f64).acos();
            let phi = 2.0 * PI * i as f64 / golden_ratio;
            [
                theta.sin() * phi.cos(),
                theta.sin() * phi.sin(),
                theta.cos(),
            ]
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Specular reflection
// ---------------------------------------------------------------------------

/// Specular reflection of direction `d` about surface normal `n`.
///
/// r = d − 2(d·n)n
pub fn reflect(direction: [f64; 3], normal: [f64; 3]) -> [f64; 3] {
    let dn = dot(direction, normal);
    sub(direction, scale(normal, 2.0 * dn))
}

// ---------------------------------------------------------------------------
// Ray tracer
// ---------------------------------------------------------------------------

/// Configuration for the ray tracing simulation.
#[derive(Debug, Clone)]
pub struct RayTracerConfig {
    /// Number of rays to emit from the source.
    pub num_rays: usize,
    /// Maximum number of reflections per ray.
    pub max_order: usize,
    /// Maximum simulation time \[s\].
    pub max_time: f64,
    /// Energy threshold (stop ray when energy drops below this).
    pub energy_threshold: f64,
    /// Speed of sound \[m/s\].
    pub speed_of_sound: f64,
    /// Air absorption coefficients at octave bands \[Np/m\].
    /// Set to zeros to disable.
    pub air_absorption: [f64; 6],
    /// Time resolution for receiver histograms \[s\].
    pub histogram_dt: f64,
}

impl RayTracerConfig {
    /// Default configuration for a typical room.
    pub fn default_room() -> Self {
        Self {
            num_rays: 10_000,
            max_order: 200,
            max_time: 2.0,
            energy_threshold: 1e-6,
            speed_of_sound: 343.0,
            air_absorption: [0.0; 6],
            histogram_dt: 0.001, // 1 ms bins
        }
    }
}

/// Trace all rays from a source through the room and collect impulse
/// responses at each receiver.
///
/// Returns the receivers with populated energy histograms.
pub fn trace_room(
    source: [f64; 3],
    surfaces: &[Surface],
    materials: &[SurfaceMaterial],
    receivers: &mut [Receiver],
    config: &RayTracerConfig,
) {
    let directions = fibonacci_sphere_directions(config.num_rays);
    let energy_per_ray = 1.0 / config.num_rays as f64;

    for dir in &directions {
        let mut ray = Ray {
            origin: source,
            direction: *dir,
            energy: energy_per_ray,
            time: 0.0,
            order: 0,
        };

        // Per-band energy tracking
        let mut band_energy = [energy_per_ray; 6];

        while ray.order < config.max_order && ray.time < config.max_time {
            // Check if any band still has significant energy
            let max_energy = band_energy.iter().copied().fold(0.0_f64, f64::max);
            if max_energy < config.energy_threshold {
                break;
            }

            // Find nearest wall
            let hit = match nearest_intersection(ray.origin, ray.direction, surfaces) {
                Some(h) => h,
                None => break, // Ray escaped (open room)
            };

            let travel_time = hit.distance / config.speed_of_sound;

            // Check receiver intersections along this segment
            for recv in receivers.iter_mut() {
                if recv.intersects_segment(ray.origin, ray.direction, hit.distance) {
                    let recv_dist = distance(ray.origin, recv.position);
                    let recv_time = ray.time + recv_dist / config.speed_of_sound;
                    // Apply air absorption over the distance to the receiver
                    let mut recv_energies = band_energy;
                    for (band, e) in recv_energies.iter_mut().enumerate() {
                        *e *= (-config.air_absorption[band] * recv_dist).exp();
                    }
                    recv.record(recv_time, &recv_energies);
                }
            }

            // Advance ray to intersection point
            let hit_point = add(ray.origin, scale(ray.direction, hit.distance));
            ray.time += travel_time;

            // Apply air absorption along path
            for (band, energy) in band_energy.iter_mut().enumerate() {
                *energy *= (-config.air_absorption[band] * hit.distance).exp();
            }

            // Apply wall absorption and scattering
            let surf = &surfaces[hit.surface_index];
            let mat = &materials[surf.material_index];
            for (band, energy) in band_energy.iter_mut().enumerate() {
                *energy *= 1.0 - mat.absorption[band];
            }

            // Reflect (specular only for now)
            ray.origin = add(hit_point, scale(surf.normal, 1e-6)); // offset to avoid self-intersection
            ray.direction = reflect(ray.direction, surf.normal);
            ray.order += 1;
        }
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fibonacci_sphere_coverage() {
        let dirs = fibonacci_sphere_directions(100);
        assert_eq!(dirs.len(), 100);
        // All directions should be unit length
        for d in &dirs {
            let len = length(*d);
            assert!((len - 1.0).abs() < 1e-10, "Non-unit direction: len = {len}");
        }
    }

    #[test]
    fn test_reflect_normal_incidence() {
        let dir = [0.0, 0.0, -1.0]; // Downward
        let normal = [0.0, 0.0, 1.0]; // Upward
        let r = reflect(dir, normal);
        assert!((r[0]).abs() < 1e-10);
        assert!((r[1]).abs() < 1e-10);
        assert!((r[2] - 1.0).abs() < 1e-10, "Should reflect upward");
    }

    #[test]
    fn test_reflect_45_degrees() {
        let dir = normalize([1.0, 0.0, -1.0]);
        let normal = [0.0, 0.0, 1.0];
        let r = reflect(dir, normal);
        // Should reflect at 45° upward
        let expected = normalize([1.0, 0.0, 1.0]);
        for i in 0..3 {
            assert!((r[i] - expected[i]).abs() < 1e-10);
        }
    }

    #[test]
    fn test_ray_triangle_intersection() {
        // Triangle on the XY plane at z=1
        let surf = Surface::from_vertices([0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [0.0, 2.0, 1.0], 0);
        // Ray from origin pointing up
        let hit = nearest_intersection([0.5, 0.5, 0.0], [0.0, 0.0, 1.0], &[surf]);
        assert!(hit.is_some());
        let h = hit.unwrap();
        assert!((h.distance - 1.0).abs() < 1e-10, "dist = {}", h.distance);
    }

    #[test]
    fn test_ray_miss() {
        let surf = Surface::from_vertices([0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], 0);
        // Ray pointing away
        let hit = nearest_intersection([0.5, 0.5, 0.0], [0.0, 0.0, -1.0], &[surf]);
        assert!(hit.is_none());
    }

    #[test]
    fn test_receiver_detection() {
        let recv = Receiver::new([5.0, 0.0, 0.0], 0.5, 1.0, 0.001);
        // Ray passing through the receiver
        assert!(recv.intersects_segment([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], 10.0));
        // Ray missing the receiver
        assert!(!recv.intersects_segment([0.0, 2.0, 0.0], [1.0, 0.0, 0.0], 10.0));
    }

    #[test]
    fn test_trace_shoebox_room() {
        // Simple 6-walled shoebox room: 4m × 3m × 2.5m
        let materials = vec![SurfaceMaterial::concrete()];
        let surfaces = shoebox_surfaces(4.0, 3.0, 2.5, 0);
        let mut receivers = vec![Receiver::new([3.0, 1.5, 1.2], 0.2, 0.5, 0.001)];

        let config = RayTracerConfig {
            num_rays: 1000,
            max_order: 50,
            max_time: 0.5,
            energy_threshold: 1e-6,
            speed_of_sound: 343.0,
            air_absorption: [0.0; 6],
            histogram_dt: 0.001,
        };

        trace_room(
            [1.0, 1.5, 1.2],
            &surfaces,
            &materials,
            &mut receivers,
            &config,
        );

        // Receiver should have collected some energy
        let total: f64 = receivers[0].histograms[2].iter().sum(); // 500 Hz band
        assert!(
            total > 0.0,
            "Receiver should collect energy: total = {total}"
        );
    }

    /// Build shoebox room surfaces (12 triangles for 6 faces).
    fn shoebox_surfaces(lx: f64, ly: f64, lz: f64, material: usize) -> Vec<Surface> {
        vec![
            // Floor (z=0): two triangles
            Surface::from_vertices([0., 0., 0.], [lx, 0., 0.], [lx, ly, 0.], material),
            Surface::from_vertices([0., 0., 0.], [lx, ly, 0.], [0., ly, 0.], material),
            // Ceiling (z=lz)
            Surface::from_vertices([0., 0., lz], [lx, ly, lz], [lx, 0., lz], material),
            Surface::from_vertices([0., 0., lz], [0., ly, lz], [lx, ly, lz], material),
            // Front wall (y=0)
            Surface::from_vertices([0., 0., 0.], [lx, 0., lz], [lx, 0., 0.], material),
            Surface::from_vertices([0., 0., 0.], [0., 0., lz], [lx, 0., lz], material),
            // Back wall (y=ly)
            Surface::from_vertices([0., ly, 0.], [lx, ly, 0.], [lx, ly, lz], material),
            Surface::from_vertices([0., ly, 0.], [lx, ly, lz], [0., ly, lz], material),
            // Left wall (x=0)
            Surface::from_vertices([0., 0., 0.], [0., ly, 0.], [0., ly, lz], material),
            Surface::from_vertices([0., 0., 0.], [0., ly, lz], [0., 0., lz], material),
            // Right wall (x=lx)
            Surface::from_vertices([lx, 0., 0.], [lx, ly, lz], [lx, ly, 0.], material),
            Surface::from_vertices([lx, 0., 0.], [lx, 0., lz], [lx, ly, lz], material),
        ]
    }
}
