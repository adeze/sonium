//! Three-dimensional round/flared port geometry and tetrahedral meshing.
//!
//! Nightly tier: 3D pipe-flow runtime/reference-convergence gates are
//! incomplete and run only under `SPEAKER_PORT_3D=1` in
//! `qa-speaker-port-nonlinear`; do not report them passed from 2D evidence.

use std::collections::HashMap;
use std::f64::consts::TAU;

use serde::{Deserialize, Serialize};
#[cfg(test)]
use sonium_fem::mesh::SurfaceFacet;
use sonium_fem::mesh::{BoundaryType, ElementType, Mesh, Point};
use sonium_mesher::mesh::refinement::uniform_refine_checked;

use super::profile::SplinePortProfile;
use super::{MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL, PortFlowError};

const MAX_SUBDIVISIONS: usize = 4096;
const MAX_MESH_NODES: usize = 500_000;
const MAX_MESH_ELEMENTS: usize = 3_000_000;
const MIN_ANGULAR_STEPS: usize = 20;

fn checked_subdivisions(
    length: f64,
    mesh_size: f64,
    dimension: &str,
) -> Result<usize, PortFlowError> {
    let ratio = length / mesh_size;
    if !ratio.is_finite() {
        return Err(PortFlowError::InvalidConfig(format!(
            "{dimension} subdivision ratio is not finite"
        )));
    }
    let count = ratio.ceil().max(1.0);
    if count > MAX_SUBDIVISIONS as f64 {
        return Err(PortFlowError::InvalidConfig(format!(
            "{dimension} requires {count:.0} subdivisions, above the resource limit {MAX_SUBDIVISIONS}"
        )));
    }
    Ok(count as usize)
}

/// Axisymmetric chamber–duct–chamber port geometry.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct RoundPortGeometry {
    pub duct_length_m: f64,
    pub duct_radius_m: f64,
    pub chamber_length_m: f64,
    pub chamber_radius_m: f64,
    /// Axial extent of each smooth chamber-to-duct transition; zero gives a
    /// sharp, planar shoulder at each duct end.
    pub flare_radius_m: f64,
    pub mesh_size_m: f64,
}

impl RoundPortGeometry {
    fn validate(self) -> Result<(), PortFlowError> {
        for (name, value) in [
            ("duct_length_m", self.duct_length_m),
            ("duct_radius_m", self.duct_radius_m),
            ("chamber_length_m", self.chamber_length_m),
            ("chamber_radius_m", self.chamber_radius_m),
            ("mesh_size_m", self.mesh_size_m),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(PortFlowError::InvalidConfig(format!(
                    "{name} must be positive and finite, got {value}"
                )));
            }
        }
        if !(self.flare_radius_m.is_finite() && self.flare_radius_m >= 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "flare_radius_m must be finite and non-negative, got {}",
                self.flare_radius_m
            )));
        }
        if self.chamber_radius_m < self.duct_radius_m {
            return Err(PortFlowError::InvalidConfig(format!(
                "chamber_radius_m ({}) must be at least duct_radius_m ({})",
                self.chamber_radius_m, self.duct_radius_m
            )));
        }
        if self.flare_radius_m > self.chamber_length_m {
            return Err(PortFlowError::InvalidConfig(format!(
                "flare_radius_m ({}) must not exceed chamber_length_m ({})",
                self.flare_radius_m, self.chamber_length_m
            )));
        }
        let h = self.mesh_size_m;
        let axial = checked_subdivisions(
            self.duct_length_m + 2.0 * self.chamber_length_m,
            h,
            "axial extent",
        )?;
        let radial = checked_subdivisions(self.chamber_radius_m, h, "radial extent")?;
        let angular = checked_subdivisions(TAU * self.chamber_radius_m, h, "circumference")?
            .max(MIN_ANGULAR_STEPS);
        let radius_change = checked_subdivisions(
            self.chamber_radius_m - self.duct_radius_m,
            h,
            "chamber-to-duct radius change",
        )?;
        let layers = axial
            .checked_add(
                radius_change
                    .checked_mul(2)
                    .ok_or_else(|| PortFlowError::InvalidConfig("layer count overflow".into()))?,
            )
            .and_then(|value| value.checked_add(8))
            .ok_or_else(|| PortFlowError::InvalidConfig("layer count overflow".into()))?;
        let nodes_per_layer = radial
            .checked_mul(angular)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| PortFlowError::InvalidConfig("node count overflow".into()))?;
        let estimated_nodes = layers
            .checked_mul(nodes_per_layer)
            .ok_or_else(|| PortFlowError::InvalidConfig("node count overflow".into()))?;
        let triangles_per_disk = radial
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .and_then(|value| value.checked_mul(angular))
            .ok_or_else(|| PortFlowError::InvalidConfig("element count overflow".into()))?;
        let estimated_elements = layers
            .saturating_sub(1)
            .checked_mul(triangles_per_disk)
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| PortFlowError::InvalidConfig("element count overflow".into()))?;
        if estimated_nodes > MAX_MESH_NODES || estimated_elements > MAX_MESH_ELEMENTS {
            return Err(PortFlowError::InvalidConfig(format!(
                "requested mesh exceeds resource limits: estimated nodes={estimated_nodes} (max {MAX_MESH_NODES}), elements={estimated_elements} (max {MAX_MESH_ELEMENTS})"
            )));
        }
        Ok(())
    }

    /// Circular cross-section radius at axial coordinate `x_m`.
    ///
    /// The domain spans both finite chambers and the duct. Smooth flares use
    /// the same cubic smoothstep profile as the mesher; a zero flare gives the
    /// sharp shoulder limit. Coordinates outside the modeled domain are
    /// rejected rather than silently clamped.
    pub fn local_radius_m(&self, x_m: f64) -> Result<f64, PortFlowError> {
        self.validate()?;
        if !x_m.is_finite() {
            return Err(PortFlowError::InvalidConfig(format!(
                "round-port axial coordinate must be finite, got {x_m}"
            )));
        }
        let x_min = -self.chamber_length_m;
        let x_max = self.duct_length_m + self.chamber_length_m;
        if x_m < x_min || x_m > x_max {
            return Err(PortFlowError::InvalidConfig(format!(
                "round-port axial coordinate {x_m} lies outside [{x_min}, {x_max}]"
            )));
        }

        let rc = self.chamber_radius_m;
        let rd = self.duct_radius_m;
        let flare = self.flare_radius_m;
        if flare == 0.0 {
            return Ok(if (0.0..=self.duct_length_m).contains(&x_m) {
                rd
            } else {
                rc
            });
        }

        let smoothstep = |t: f64| t * t * (3.0 - 2.0 * t);
        if x_m < -flare {
            Ok(rc)
        } else if x_m < 0.0 {
            let t = (x_m + flare) / flare;
            Ok(rc + smoothstep(t) * (rd - rc))
        } else if x_m <= self.duct_length_m {
            Ok(rd)
        } else if x_m < self.duct_length_m + flare {
            let t = (x_m - self.duct_length_m) / flare;
            Ok(rd + smoothstep(t) * (rc - rd))
        } else {
            Ok(rc)
        }
    }
}

fn subdivisions(length: f64, mesh_size: f64) -> usize {
    debug_assert!((length / mesh_size).is_finite());
    debug_assert!((length / mesh_size).ceil().max(1.0) <= MAX_SUBDIVISIONS as f64);
    (length / mesh_size).ceil().max(1.0) as usize
}

/// Axisymmetric chamber–duct–chamber port geometry carrying a spline profile.
///
/// The duct wall `r(x)` on `[0, Ld]` comes from [`SplinePortProfile`]; each
/// duct end joins its chamber through a cubic smoothstep over `flare_length_m`
/// (the same polynomial as [`RoundPortGeometry`], blending to the profile end
/// radii). A zero flare is only valid when the chambers already match the
/// bore, i.e. the straight-cylinder limit with a (possibly dipped) spline
/// duct; sharp chamber-to-duct shoulders are rejected on the spline path
/// because one axial layer cannot carry two radii in a single sweep.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SplineRoundPortGeometry {
    /// Duct (profile) length `Ld` in m; must match the profile length.
    pub duct_length_m: f64,
    /// Straight-bore radius `R0` in m; must match the profile bore.
    pub bore_radius_m: f64,
    /// Axial length of each end-effect chamber in m.
    pub chamber_length_m: f64,
    /// Chamber radius in m; must be at least the bore radius.
    pub chamber_radius_m: f64,
    /// Axial extent of each smooth chamber-to-duct transition in m.
    pub flare_length_m: f64,
    /// Target cell size in m.
    pub mesh_size_m: f64,
    /// Explicit circumferential sector count, or `None` for the automatic
    /// `ceil(2π·Rc/h)` (minimum 20). An explicit count decouples angular
    /// resolution from the axial/radial mesh size so the polygon-area deficit
    /// of the reference solve can be reduced without over-refining the sweep;
    /// values below the 20-sector minimum are rejected.
    pub angular_sectors: Option<usize>,
    /// Duct wall profile `r(x)` on `[0, Ld]`.
    pub profile: SplinePortProfile,
}

impl SplineRoundPortGeometry {
    fn validate(&self) -> Result<(), PortFlowError> {
        for (name, value) in [
            ("duct_length_m", self.duct_length_m),
            ("bore_radius_m", self.bore_radius_m),
            ("chamber_length_m", self.chamber_length_m),
            ("chamber_radius_m", self.chamber_radius_m),
            ("mesh_size_m", self.mesh_size_m),
        ] {
            if !(value.is_finite() && value > 0.0) {
                return Err(PortFlowError::InvalidConfig(format!(
                    "{name} must be positive and finite, got {value}"
                )));
            }
        }
        if !(self.flare_length_m.is_finite() && self.flare_length_m >= 0.0) {
            return Err(PortFlowError::InvalidConfig(format!(
                "flare_length_m must be finite and non-negative, got {}",
                self.flare_length_m
            )));
        }
        if self.chamber_radius_m < self.bore_radius_m {
            return Err(PortFlowError::InvalidConfig(format!(
                "chamber_radius_m ({}) must be at least bore_radius_m ({})",
                self.chamber_radius_m, self.bore_radius_m
            )));
        }
        if self.flare_length_m > self.chamber_length_m {
            return Err(PortFlowError::InvalidConfig(format!(
                "flare_length_m ({}) must not exceed chamber_length_m ({})",
                self.flare_length_m, self.chamber_length_m
            )));
        }
        if self.flare_length_m == 0.0 && self.chamber_radius_m != self.bore_radius_m {
            return Err(PortFlowError::InvalidConfig(
                "spline round-port geometry with a chamber wider than the bore needs a nonzero flare_length_m; sharp shoulders are unsupported on the spline path"
                    .into(),
            ));
        }
        if let Some(sectors) = self.angular_sectors
            && sectors < MIN_ANGULAR_STEPS
        {
            return Err(PortFlowError::InvalidConfig(format!(
                "angular_sectors ({sectors}) must be at least the {MIN_ANGULAR_STEPS}-sector minimum"
            )));
        }
        let length_tol = 1.0e-9 * self.duct_length_m.max(1.0e-12);
        if (self.profile.length_m() - self.duct_length_m).abs() > length_tol {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline profile length {} does not match duct_length_m {}",
                self.profile.length_m(),
                self.duct_length_m
            )));
        }
        let bore_tol = 1.0e-9 * self.bore_radius_m.max(1.0e-12);
        if (self.profile.bore_radius_m() - self.bore_radius_m).abs() > bore_tol {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline profile bore {} does not match bore_radius_m {}",
                self.profile.bore_radius_m(),
                self.bore_radius_m
            )));
        }
        // Re-audit the profile so a deserialized geometry cannot smuggle an
        // illegal wall into a solve.
        self.profile.validate()?;
        let h = self.mesh_size_m;
        let axial = checked_subdivisions(
            self.duct_length_m + 2.0 * self.chamber_length_m,
            h,
            "axial extent",
        )?;
        let radial = checked_subdivisions(self.chamber_radius_m, h, "radial extent")?;
        let angular = self.angular_sectors.unwrap_or(
            checked_subdivisions(TAU * self.chamber_radius_m, h, "circumference")?
                .max(MIN_ANGULAR_STEPS),
        );
        let radius_change = checked_subdivisions(
            self.chamber_radius_m - self.bore_radius_m,
            h,
            "chamber-to-duct radius change",
        )?;
        let layers = axial
            .checked_add(
                radius_change
                    .checked_mul(2)
                    .ok_or_else(|| PortFlowError::InvalidConfig("layer count overflow".into()))?,
            )
            .and_then(|value| value.checked_add(8))
            .ok_or_else(|| PortFlowError::InvalidConfig("layer count overflow".into()))?;
        let nodes_per_layer = radial
            .checked_mul(angular)
            .and_then(|value| value.checked_add(1))
            .ok_or_else(|| PortFlowError::InvalidConfig("node count overflow".into()))?;
        let estimated_nodes = layers
            .checked_mul(nodes_per_layer)
            .ok_or_else(|| PortFlowError::InvalidConfig("node count overflow".into()))?;
        let triangles_per_disk = radial
            .checked_mul(2)
            .and_then(|value| value.checked_sub(1))
            .and_then(|value| value.checked_mul(angular))
            .ok_or_else(|| PortFlowError::InvalidConfig("element count overflow".into()))?;
        let estimated_elements = layers
            .saturating_sub(1)
            .checked_mul(triangles_per_disk)
            .and_then(|value| value.checked_mul(3))
            .ok_or_else(|| PortFlowError::InvalidConfig("element count overflow".into()))?;
        if estimated_nodes > MAX_MESH_NODES || estimated_elements > MAX_MESH_ELEMENTS {
            return Err(PortFlowError::InvalidConfig(format!(
                "requested mesh exceeds resource limits: estimated nodes={estimated_nodes} (max {MAX_MESH_NODES}), elements={estimated_elements} (max {MAX_MESH_ELEMENTS})"
            )));
        }
        Ok(())
    }

    /// Circular cross-section radius at axial coordinate `x_m`.
    ///
    /// Chambers are exact cylinders; the duct follows the spline profile and
    /// each duct end blends to its chamber through the same cubic smoothstep
    /// [`RoundPortGeometry`] uses. Coordinates outside the modeled domain are
    /// rejected rather than silently clamped.
    pub fn local_radius_m(&self, x_m: f64) -> Result<f64, PortFlowError> {
        self.validate()?;
        if !x_m.is_finite() {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline round-port axial coordinate must be finite, got {x_m}"
            )));
        }
        let x_min = -self.chamber_length_m;
        let x_max = self.duct_length_m + self.chamber_length_m;
        if x_m < x_min || x_m > x_max {
            return Err(PortFlowError::InvalidConfig(format!(
                "spline round-port axial coordinate {x_m} lies outside [{x_min}, {x_max}]"
            )));
        }

        let rc = self.chamber_radius_m;
        let flare = self.flare_length_m;
        if flare == 0.0 {
            return Ok(if (0.0..=self.duct_length_m).contains(&x_m) {
                self.profile.local_radius_m(x_m)
            } else {
                rc
            });
        }

        let smoothstep = |t: f64| t * t * (3.0 - 2.0 * t);
        // Bit-identical to the RoundPortGeometry smoothstep when the profile
        // ends equal the bore (the straight case evaluates the spline to
        // exactly R0 at the knots).
        let r_in = self.profile.local_radius_m(0.0);
        let r_out = self.profile.local_radius_m(self.duct_length_m);
        if x_m < -flare {
            Ok(rc)
        } else if x_m < 0.0 {
            let t = (x_m + flare) / flare;
            Ok(rc + smoothstep(t) * (r_in - rc))
        } else if x_m <= self.duct_length_m {
            Ok(self.profile.local_radius_m(x_m))
        } else if x_m < self.duct_length_m + flare {
            let t = (x_m - self.duct_length_m) / flare;
            Ok(r_out + smoothstep(t) * (rc - r_out))
        } else {
            Ok(rc)
        }
    }
}

fn push_linear_profile(profile: &mut Vec<(f64, f64)>, a: (f64, f64), b: (f64, f64), h: f64) {
    let length = (b.0 - a.0).hypot(b.1 - a.1);
    if length <= f64::EPSILON {
        return;
    }
    let n = subdivisions(length, h);
    for i in 1..=n {
        let t = i as f64 / n as f64;
        profile.push((a.0 + t * (b.0 - a.0), a.1 + t * (b.1 - a.1)));
    }
}

fn push_smooth_flare(
    profile: &mut Vec<(f64, f64)>,
    geo: &RoundPortGeometry,
    x0: f64,
    x1: f64,
    h: f64,
) {
    let r0 = geo
        .local_radius_m(x0)
        .expect("validated meridian start lies in the geometry domain");
    let r1 = geo
        .local_radius_m(x1)
        .expect("validated meridian end lies in the geometry domain");
    let n = subdivisions((x1 - x0).hypot(r1 - r0), h).max(2);
    for i in 1..=n {
        let t = i as f64 / n as f64;
        let x = x0 + t * (x1 - x0);
        let radius = geo
            .local_radius_m(x)
            .expect("sampled meridian point lies in the geometry domain");
        profile.push((x, radius));
    }
}

fn meridian_profile(geo: &RoundPortGeometry, surface_size: f64) -> Vec<(f64, f64)> {
    let lc = geo.chamber_length_m;
    let ld = geo.duct_length_m;
    let rc = geo.chamber_radius_m;
    let rd = geo.duct_radius_m;
    let flare = geo.flare_radius_m;
    let h = surface_size;
    let mut profile = vec![(-lc, rc)];

    if flare == 0.0 {
        push_linear_profile(&mut profile, (-lc, rc), (0.0, rc), h);
        push_linear_profile(&mut profile, (0.0, rc), (0.0, rd), h);
    } else {
        push_linear_profile(&mut profile, (-lc, rc), (-flare, rc), h);
        push_smooth_flare(&mut profile, geo, -flare, 0.0, h);
    }
    push_linear_profile(&mut profile, (0.0, rd), (ld, rd), h);
    if flare == 0.0 {
        push_linear_profile(&mut profile, (ld, rd), (ld, rc), h);
        push_linear_profile(&mut profile, (ld, rc), (ld + lc, rc), h);
    } else {
        push_smooth_flare(&mut profile, geo, ld, ld + flare, h);
        push_linear_profile(&mut profile, (ld + flare, rc), (ld + lc, rc), h);
    }
    profile
}

/// Sample an arbitrary radius function at uniform axial stations.
///
/// Same layering rule as [`push_smooth_flare`] (arc-length subdivisions with
/// a floor of `min_points` intervals); the spline meridian uses it for both
/// the smoothstep flares and the profile duct so a straight spline reproduces
/// the flared-branch layering.
fn push_sampled_radius(
    profile: &mut Vec<(f64, f64)>,
    radius_at: &dyn Fn(f64) -> f64,
    x0: f64,
    x1: f64,
    h: f64,
    min_points: usize,
) {
    let r0 = radius_at(x0);
    let r1 = radius_at(x1);
    let n = subdivisions((x1 - x0).hypot(r1 - r0), h).max(min_points);
    for i in 1..=n {
        let t = i as f64 / n as f64;
        let x = x0 + t * (x1 - x0);
        profile.push((x, radius_at(x)));
    }
}

fn spline_meridian_profile(geo: &SplineRoundPortGeometry, surface_size: f64) -> Vec<(f64, f64)> {
    let lc = geo.chamber_length_m;
    let ld = geo.duct_length_m;
    let rc = geo.chamber_radius_m;
    let flare = geo.flare_length_m;
    let h = surface_size;
    let radius_at = |x: f64| {
        geo.local_radius_m(x)
            .expect("sampled meridian point lies in the geometry domain")
    };
    let mut profile = vec![(-lc, rc)];
    if flare == 0.0 {
        // Validated sharp-free: the chambers already match the bore.
        push_linear_profile(&mut profile, (-lc, rc), (0.0, rc), h);
    } else {
        push_linear_profile(&mut profile, (-lc, rc), (-flare, rc), h);
        push_sampled_radius(&mut profile, &radius_at, -flare, 0.0, h, 2);
    }
    push_sampled_radius(&mut profile, &radius_at, 0.0, ld, h, 1);
    if flare == 0.0 {
        push_linear_profile(&mut profile, (ld, rc), (ld + lc, rc), h);
    } else {
        push_sampled_radius(&mut profile, &radius_at, ld, ld + flare, h, 2);
        push_linear_profile(&mut profile, (ld + flare, rc), (ld + lc, rc), h);
    }
    profile
}

#[cfg(test)]
fn add_cap(
    nodes: &mut Vec<Point>,
    facets: &mut Vec<SurfaceFacet>,
    x: f64,
    radius: f64,
    radial_steps: usize,
    angular_steps: usize,
    marker: i32,
    outward_positive_x: bool,
    outer_ring: &[usize],
) {
    let center = nodes.len();
    nodes.push(Point::new_3d(x, 0.0, 0.0));
    let mut previous = vec![center; angular_steps];
    for radial in 1..=radial_steps {
        let current = if radial == radial_steps {
            outer_ring.to_vec()
        } else {
            let r = radius * radial as f64 / radial_steps as f64;
            (0..angular_steps)
                .map(|j| {
                    let theta = TAU * j as f64 / angular_steps as f64;
                    let index = nodes.len();
                    nodes.push(Point::new_3d(x, r * theta.cos(), r * theta.sin()));
                    index
                })
                .collect()
        };
        for j in 0..angular_steps {
            let next = (j + 1) % angular_steps;
            if radial == 1 {
                let tri = if outward_positive_x {
                    [center, current[j], current[next]]
                } else {
                    [center, current[next], current[j]]
                };
                facets.push(SurfaceFacet {
                    nodes: tri,
                    marker: Some(marker),
                });
            } else {
                let (a, b, c, d) = (previous[j], current[j], current[next], previous[next]);
                let triangles = if outward_positive_x {
                    [[a, b, c], [a, c, d]]
                } else {
                    [[a, c, b], [a, d, c]]
                };
                facets.extend(triangles.into_iter().map(|nodes| SurfaceFacet {
                    nodes,
                    marker: Some(marker),
                }));
            }
        }
        previous = current;
    }
}

#[cfg(test)]
fn validate_closed_surface(facets: &[SurfaceFacet]) -> Result<(), PortFlowError> {
    let mut uses: HashMap<(usize, usize), usize> = HashMap::new();
    for facet in facets {
        let [a, b, c] = facet.nodes;
        if a == b || b == c || c == a {
            return Err(PortFlowError::Mesh(
                "surface contains a degenerate facet".into(),
            ));
        }
        for (u, v) in [(a, b), (b, c), (c, a)] {
            *uses.entry((u.min(v), u.max(v))).or_default() += 1;
        }
    }
    if let Some((edge, count)) = uses.into_iter().find(|(_, count)| *count != 2) {
        return Err(PortFlowError::Mesh(format!(
            "surface is not watertight: edge {edge:?} is used {count} times"
        )));
    }
    Ok(())
}

/// Build a closed body-of-revolution surface and tetrahedralize its interior.
#[cfg(test)]
fn build_surface(
    geo: &RoundPortGeometry,
    profile_size: f64,
) -> Result<(Vec<Point>, Vec<SurfaceFacet>), PortFlowError> {
    let profile = meridian_profile(geo, profile_size);
    let angular_steps =
        subdivisions(TAU * geo.chamber_radius_m, geo.mesh_size_m).max(MIN_ANGULAR_STEPS);
    let mut nodes = Vec::with_capacity(profile.len() * angular_steps);
    let rings: Vec<Vec<usize>> = profile
        .iter()
        .map(|&(x, radius)| {
            (0..angular_steps)
                .map(|j| {
                    let theta = TAU * j as f64 / angular_steps as f64;
                    let index = nodes.len();
                    nodes.push(Point::new_3d(x, radius * theta.cos(), radius * theta.sin()));
                    index
                })
                .collect()
        })
        .collect();
    let mut facets = Vec::new();
    for pair in rings.windows(2) {
        for j in 0..angular_steps {
            let next = (j + 1) % angular_steps;
            let (a, b, c, d) = (pair[0][j], pair[1][j], pair[1][next], pair[0][next]);
            facets.push(SurfaceFacet {
                nodes: [a, d, b],
                marker: Some(MARKER_WALL),
            });
            facets.push(SurfaceFacet {
                nodes: [d, c, b],
                marker: Some(MARKER_WALL),
            });
        }
    }
    let radial_steps = subdivisions(geo.chamber_radius_m, profile_size);
    let last_ring = rings.last().ok_or_else(|| {
        PortFlowError::Mesh("round-port meridian produced no surface rings".into())
    })?;
    add_cap(
        &mut nodes,
        &mut facets,
        -geo.chamber_length_m,
        geo.chamber_radius_m,
        radial_steps,
        angular_steps,
        MARKER_BOX_CAP,
        false,
        &rings[0],
    );
    add_cap(
        &mut nodes,
        &mut facets,
        geo.duct_length_m + geo.chamber_length_m,
        geo.chamber_radius_m,
        radial_steps,
        angular_steps,
        MARKER_EXTERIOR_CAP,
        true,
        last_ring,
    );
    validate_closed_surface(&facets)?;
    Ok((nodes, facets))
}

fn disk_triangles(radial_steps: usize, angular_steps: usize) -> Vec<[usize; 3]> {
    let ring = |radial: usize, angle: usize| 1 + (radial - 1) * angular_steps + angle;
    let mut triangles = Vec::new();
    for angle in 0..angular_steps {
        let next = (angle + 1) % angular_steps;
        triangles.push([0, ring(1, angle), ring(1, next)]);
    }
    for radial in 1..radial_steps {
        for angle in 0..angular_steps {
            let next = (angle + 1) % angular_steps;
            let (a, b, c, d) = (
                ring(radial, angle),
                ring(radial + 1, angle),
                ring(radial + 1, next),
                ring(radial, next),
            );
            triangles.extend([[a, b, c], [a, c, d]]);
        }
    }
    triangles
}

fn add_volume_layer(mesh: &mut Mesh, x: f64, radii: &[f64], angular_steps: usize) -> Vec<usize> {
    let mut layer = Vec::new();
    layer.push(mesh.add_node(Point::new_3d(x, 0.0, 0.0)));
    for &radius in radii {
        for angle in 0..angular_steps {
            let theta = TAU * angle as f64 / angular_steps as f64;
            layer.push(mesh.add_node(Point::new_3d(x, radius * theta.cos(), radius * theta.sin())));
        }
    }
    layer
}

fn tet_determinant(mesh: &Mesh, nodes: [usize; 4]) -> f64 {
    let [a, b, c, d] = nodes.map(|index| mesh.nodes[index]);
    let u = [b.x - a.x, b.y - a.y, b.z - a.z];
    let v = [c.x - a.x, c.y - a.y, c.z - a.z];
    let w = [d.x - a.x, d.y - a.y, d.z - a.z];
    u[0] * (v[1] * w[2] - v[2] * w[1]) - u[1] * (v[0] * w[2] - v[2] * w[0])
        + u[2] * (v[0] * w[1] - v[1] * w[0])
}

fn add_oriented_tet(mesh: &mut Mesh, mut nodes: [usize; 4]) -> Result<(), PortFlowError> {
    let mut determinant = tet_determinant(mesh, nodes);
    if determinant < 0.0 {
        nodes.swap(0, 1);
        determinant = -determinant;
    }
    if !(determinant.is_finite() && determinant > 0.0) {
        return Err(PortFlowError::Mesh(format!(
            "structured sweep produced a degenerate tetrahedron {nodes:?}"
        )));
    }
    mesh.add_element(ElementType::Tetrahedron, nodes.to_vec());
    Ok(())
}

fn sweep_volume_layers(
    mesh: &mut Mesh,
    lower: &[usize],
    upper: &[usize],
    triangles: &[[usize; 3]],
) -> Result<(), PortFlowError> {
    for &triangle in triangles {
        let mut local = triangle;
        local.sort_unstable();
        let [a, b, c] = local;
        for tet in [
            [lower[a], lower[b], lower[c], upper[c]],
            [lower[a], lower[b], upper[b], upper[c]],
            [lower[a], upper[a], upper[b], upper[c]],
        ] {
            add_oriented_tet(mesh, tet)?;
        }
    }
    Ok(())
}

fn validate_boundary_watertight(mesh: &Mesh) -> Result<(), PortFlowError> {
    let mut edge_uses: HashMap<(usize, usize), usize> = HashMap::new();
    for boundary in &mesh.boundaries {
        if boundary.nodes.len() != 3 {
            return Err(PortFlowError::Mesh(format!(
                "3D boundary facet has {} nodes instead of 3",
                boundary.nodes.len()
            )));
        }
        let [a, b, c] = [boundary.nodes[0], boundary.nodes[1], boundary.nodes[2]];
        for (u, v) in [(a, b), (b, c), (c, a)] {
            *edge_uses.entry((u.min(v), u.max(v))).or_default() += 1;
        }
    }
    if let Some((edge, uses)) = edge_uses.into_iter().find(|(_, uses)| *uses != 2) {
        return Err(PortFlowError::Mesh(format!(
            "volume boundary is not watertight: edge {edge:?} is used {uses} times"
        )));
    }
    Ok(())
}

pub(crate) fn boundary_outward_normal(
    mesh: &Mesh,
    boundary_index: usize,
) -> Result<([f64; 3], f64), PortFlowError> {
    let boundary = mesh
        .boundaries
        .get(boundary_index)
        .ok_or_else(|| PortFlowError::Mesh(format!("missing boundary facet {boundary_index}")))?;
    let owner = mesh.elements.get(boundary.element_idx).ok_or_else(|| {
        PortFlowError::Mesh(format!(
            "boundary facet {boundary_index} references missing owner {}",
            boundary.element_idx
        ))
    })?;
    if boundary.nodes.len() != 3 || !boundary.nodes.iter().all(|node| owner.nodes.contains(node)) {
        return Err(PortFlowError::Mesh(format!(
            "boundary facet {boundary_index} is not a face of owner {}",
            boundary.element_idx
        )));
    }
    let [a, b, c] = [
        mesh.nodes[boundary.nodes[0]],
        mesh.nodes[boundary.nodes[1]],
        mesh.nodes[boundary.nodes[2]],
    ];
    let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
    let ac = [c.x - a.x, c.y - a.y, c.z - a.z];
    let mut normal = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let magnitude = (normal[0] * normal[0] + normal[1] * normal[1] + normal[2] * normal[2]).sqrt();
    if !(magnitude.is_finite() && magnitude > 0.0) {
        return Err(PortFlowError::Mesh(format!(
            "boundary facet {boundary_index} has zero area"
        )));
    }
    let face_centroid = [
        (a.x + b.x + c.x) / 3.0,
        (a.y + b.y + c.y) / 3.0,
        (a.z + b.z + c.z) / 3.0,
    ];
    let mut owner_centroid = [0.0; 3];
    for &node in &owner.nodes {
        let point = mesh.nodes[node];
        owner_centroid[0] += point.x;
        owner_centroid[1] += point.y;
        owner_centroid[2] += point.z;
    }
    for coordinate in &mut owner_centroid {
        *coordinate /= owner.nodes.len() as f64;
    }
    let outward = [
        face_centroid[0] - owner_centroid[0],
        face_centroid[1] - owner_centroid[1],
        face_centroid[2] - owner_centroid[2],
    ];
    if normal[0] * outward[0] + normal[1] * outward[1] + normal[2] * outward[2] < 0.0 {
        normal = [-normal[0], -normal[1], -normal[2]];
    }
    Ok((
        [
            normal[0] / magnitude,
            normal[1] / magnitude,
            normal[2] / magnitude,
        ],
        0.5 * magnitude,
    ))
}

fn validate_boundary_geometry(
    mesh: &Mesh,
    chamber_length_m: f64,
    duct_length_m: f64,
    mesh_size_m: f64,
) -> Result<(), PortFlowError> {
    let tolerance = mesh_size_m * 1.0e-8;
    for (index, boundary) in mesh.boundaries.iter().enumerate() {
        let (normal, _) = boundary_outward_normal(mesh, index)?;
        let points: Vec<Point> = boundary
            .nodes
            .iter()
            .map(|&node| mesh.nodes[node])
            .collect();
        let on_box_cap = points
            .iter()
            .all(|point| (point.x + chamber_length_m).abs() <= tolerance);
        let on_exterior_cap = points
            .iter()
            .all(|point| (point.x - (duct_length_m + chamber_length_m)).abs() <= tolerance);
        match boundary.marker {
            MARKER_BOX_CAP if on_box_cap && normal[0] < 0.0 => {}
            MARKER_EXTERIOR_CAP if on_exterior_cap && normal[0] > 0.0 => {}
            MARKER_WALL if !on_box_cap && !on_exterior_cap => {
                let centroid = points.iter().fold([0.0; 3], |mut sum, point| {
                    sum[0] += point.x / points.len() as f64;
                    sum[1] += point.y / points.len() as f64;
                    sum[2] += point.z / points.len() as f64;
                    sum
                });
                let on_left_shoulder = points.iter().all(|point| point.x.abs() <= tolerance);
                let on_right_shoulder = points
                    .iter()
                    .all(|point| (point.x - duct_length_m).abs() <= tolerance);
                let radial_dot = normal[1] * centroid[1] + normal[2] * centroid[2];
                let orientation_valid = if on_left_shoulder {
                    normal[0] > 0.0
                } else if on_right_shoulder {
                    normal[0] < 0.0
                } else {
                    radial_dot > 0.0
                };
                if !orientation_valid {
                    return Err(PortFlowError::Mesh(format!(
                        "wall facet {index} has an inward normal"
                    )));
                }
            }
            marker => {
                return Err(PortFlowError::Mesh(format!(
                    "boundary facet {index} has inconsistent marker/orientation {marker}"
                )));
            }
        }
    }
    Ok(())
}

fn spaced_positions(a: f64, b: f64, mesh_size: f64) -> Vec<f64> {
    let count = subdivisions(b - a, mesh_size);
    (0..=count)
        .map(|index| a + (b - a) * index as f64 / count as f64)
        .collect()
}

fn swept_region(
    mesh: &mut Mesh,
    positions: &[f64],
    radii: &[f64],
    angular_steps: usize,
    triangles: &[[usize; 3]],
    first: Option<Vec<usize>>,
) -> Result<Vec<usize>, PortFlowError> {
    let mut previous =
        first.unwrap_or_else(|| add_volume_layer(mesh, positions[0], radii, angular_steps));
    for &x in positions.iter().skip(1) {
        let current = add_volume_layer(mesh, x, radii, angular_steps);
        sweep_volume_layers(mesh, &previous, &current, triangles)?;
        previous = current;
    }
    Ok(previous)
}

/// Build a conforming structured tetrahedral mesh of the round/flared port.
pub fn build_round_port_mesh(geo: &RoundPortGeometry) -> Result<Mesh, PortFlowError> {
    geo.validate()?;
    let angular_steps =
        subdivisions(TAU * geo.chamber_radius_m, geo.mesh_size_m).max(MIN_ANGULAR_STEPS);
    let mut mesh = Mesh::new(3);

    if geo.chamber_radius_m == geo.duct_radius_m {
        let radial_steps = subdivisions(geo.duct_radius_m, geo.mesh_size_m);
        let radii: Vec<f64> = (1..=radial_steps)
            .map(|i| geo.duct_radius_m * i as f64 / radial_steps as f64)
            .collect();
        let triangles = disk_triangles(radial_steps, angular_steps);
        let mut positions = spaced_positions(-geo.chamber_length_m, 0.0, geo.mesh_size_m);
        positions.extend(
            spaced_positions(0.0, geo.duct_length_m, geo.mesh_size_m)
                .into_iter()
                .skip(1),
        );
        positions.extend(
            spaced_positions(
                geo.duct_length_m,
                geo.duct_length_m + geo.chamber_length_m,
                geo.mesh_size_m,
            )
            .into_iter()
            .skip(1),
        );
        swept_region(
            &mut mesh,
            &positions,
            &radii,
            angular_steps,
            &triangles,
            None,
        )?;
    } else if geo.flare_radius_m == 0.0 {
        // A true sharp shoulder cannot be represented by a single scaled
        // sweep. Share the inner disk between three swept subdomains and leave
        // the two chamber annuli exposed as wall facets.
        let inner_steps = subdivisions(geo.duct_radius_m, geo.mesh_size_m);
        let mut radii: Vec<f64> = (1..=inner_steps)
            .map(|i| geo.duct_radius_m * i as f64 / inner_steps as f64)
            .collect();
        let outer_span = geo.chamber_radius_m - geo.duct_radius_m;
        let outer_steps = subdivisions(outer_span, geo.mesh_size_m);
        radii.extend(
            (1..=outer_steps)
                .map(|i| geo.duct_radius_m + outer_span * i as f64 / outer_steps as f64),
        );
        let inner_radii = radii[..inner_steps].to_vec();
        let full_triangles = disk_triangles(radii.len(), angular_steps);
        let inner_triangles = disk_triangles(inner_steps, angular_steps);
        let left_end = swept_region(
            &mut mesh,
            &spaced_positions(-geo.chamber_length_m, 0.0, geo.mesh_size_m),
            &radii,
            angular_steps,
            &full_triangles,
            None,
        )?;
        let inner_len = 1 + inner_steps * angular_steps;
        let duct_end = swept_region(
            &mut mesh,
            &spaced_positions(0.0, geo.duct_length_m, geo.mesh_size_m),
            &inner_radii,
            angular_steps,
            &inner_triangles,
            Some(left_end[..inner_len].to_vec()),
        )?;
        let mut right_start = duct_end;
        for &radius in &radii[inner_steps..] {
            for angle in 0..angular_steps {
                let theta = TAU * angle as f64 / angular_steps as f64;
                right_start.push(mesh.add_node(Point::new_3d(
                    geo.duct_length_m,
                    radius * theta.cos(),
                    radius * theta.sin(),
                )));
            }
        }
        swept_region(
            &mut mesh,
            &spaced_positions(
                geo.duct_length_m,
                geo.duct_length_m + geo.chamber_length_m,
                geo.mesh_size_m,
            ),
            &radii,
            angular_steps,
            &full_triangles,
            Some(right_start),
        )?;
    } else {
        let profile = meridian_profile(geo, geo.mesh_size_m);
        let radial_steps = subdivisions(geo.chamber_radius_m, geo.mesh_size_m);
        let triangles = disk_triangles(radial_steps, angular_steps);
        let fractions: Vec<f64> = (1..=radial_steps)
            .map(|i| i as f64 / radial_steps as f64)
            .collect();
        let mut previous: Option<Vec<usize>> = None;
        for &(x, radius) in &profile {
            let radii: Vec<f64> = fractions.iter().map(|fraction| fraction * radius).collect();
            let current = add_volume_layer(&mut mesh, x, &radii, angular_steps);
            if let Some(lower) = previous.as_ref() {
                sweep_volume_layers(&mut mesh, lower, &current, &triangles)?;
            }
            previous = Some(current);
        }
    }

    finish_swept_port_mesh(
        &mut mesh,
        geo.chamber_length_m,
        geo.duct_length_m,
        geo.mesh_size_m,
    )?;
    Ok(mesh)
}

/// Classify cap/wall boundaries and audit a swept chamber–duct–chamber mesh.
///
/// Shared tail of the round- and spline-port builders: cap planes are exact
/// to `mesh_size_m * 1e-8`, the boundary must be watertight with outward
/// normals, every tetrahedron must keep a positive determinant, and all three
/// markers must survive.
fn finish_swept_port_mesh(
    mesh: &mut Mesh,
    chamber_length_m: f64,
    duct_length_m: f64,
    mesh_size_m: f64,
) -> Result<(), PortFlowError> {
    mesh.detect_boundaries();
    let tolerance = mesh_size_m * 1.0e-8;
    mesh.set_boundary_condition(BoundaryType::Neumann, MARKER_BOX_CAP, |points| {
        points
            .iter()
            .all(|point| (point.x + chamber_length_m).abs() <= tolerance)
    });
    mesh.set_boundary_condition(BoundaryType::Neumann, MARKER_EXTERIOR_CAP, |points| {
        points
            .iter()
            .all(|point| (point.x - (duct_length_m + chamber_length_m)).abs() <= tolerance)
    });
    mesh.set_boundary_condition(BoundaryType::Dirichlet, MARKER_WALL, |points| {
        !points
            .iter()
            .all(|point| (point.x + chamber_length_m).abs() <= tolerance)
            && !points
                .iter()
                .all(|point| (point.x - (duct_length_m + chamber_length_m)).abs() <= tolerance)
    });
    validate_boundary_watertight(mesh)?;
    validate_boundary_geometry(mesh, chamber_length_m, duct_length_m, mesh_size_m)?;
    if mesh.elements.is_empty()
        || mesh.elements.iter().any(|element| {
            if element.element_type != ElementType::Tetrahedron || element.nodes.len() != 4 {
                return true;
            }
            let determinant = tet_determinant(
                mesh,
                [
                    element.nodes[0],
                    element.nodes[1],
                    element.nodes[2],
                    element.nodes[3],
                ],
            );
            !(determinant.is_finite() && determinant > 0.0)
        })
    {
        return Err(PortFlowError::Mesh(
            "structured sweep produced invalid tetrahedra".into(),
        ));
    }
    for marker in [MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL] {
        if !mesh
            .boundaries
            .iter()
            .any(|boundary| boundary.marker == marker)
        {
            return Err(PortFlowError::Mesh(format!(
                "structured sweep lost required boundary marker {marker}"
            )));
        }
    }
    Ok(())
}

/// Build a conforming structured tetrahedral mesh of the spline port.
///
/// Sweeps radially scaled disks along the spline meridian (chambers, smooth
/// flares, profile duct), the same construction as the flared round-port
/// branch, so a straight spline reproduces that mesh up to spline-evaluation
/// roundoff. A straight spline with matching chambers additionally reproduces
/// the straight-cylinder layering.
pub fn build_spline_round_port_mesh(geo: &SplineRoundPortGeometry) -> Result<Mesh, PortFlowError> {
    geo.validate()?;
    let angular_steps = geo.angular_sectors.unwrap_or(
        subdivisions(TAU * geo.chamber_radius_m, geo.mesh_size_m).max(MIN_ANGULAR_STEPS),
    );
    let mut mesh = Mesh::new(3);
    let profile = spline_meridian_profile(geo, geo.mesh_size_m);
    let radial_steps = subdivisions(geo.chamber_radius_m, geo.mesh_size_m);
    let triangles = disk_triangles(radial_steps, angular_steps);
    let fractions: Vec<f64> = (1..=radial_steps)
        .map(|i| i as f64 / radial_steps as f64)
        .collect();
    let mut previous: Option<Vec<usize>> = None;
    for &(x, radius) in &profile {
        let radii: Vec<f64> = fractions.iter().map(|fraction| fraction * radius).collect();
        let current = add_volume_layer(&mut mesh, x, &radii, angular_steps);
        if let Some(lower) = previous.as_ref() {
            sweep_volume_layers(&mut mesh, lower, &current, &triangles)?;
        }
        previous = Some(current);
    }
    finish_swept_port_mesh(
        &mut mesh,
        geo.chamber_length_m,
        geo.duct_length_m,
        geo.mesh_size_m,
    )?;
    Ok(mesh)
}

/// Uniformly refine a round-port mesh and project new wall nodes back onto the
/// canonical smooth radius profile.
///
/// Generic straight-sided tetrahedral refinement places new flare-wall nodes
/// on boundary chords. This application-level helper restores the exact
/// [`RoundPortGeometry`] domain while preserving cap planes and marker sets.
/// Sharp shoulders are rejected because one axial layer contains both chamber
/// and duct radii and cannot be represented by this single-radius projection.
/// The operation is transactional: `mesh` is unchanged on every error.
pub fn refine_round_port_mesh_projected(
    mesh: &mut Mesh,
    geo: &RoundPortGeometry,
) -> Result<(), PortFlowError> {
    geo.validate()?;
    if geo.flare_radius_m == 0.0 {
        return Err(PortFlowError::InvalidConfig(
            "projected round-port refinement requires a smooth nonzero flare; sharp shoulders have multiple radii on one axial layer".into(),
        ));
    }
    let mut candidate = mesh.clone();
    uniform_refine_checked(&mut candidate)
        .map_err(|error| PortFlowError::Mesh(format!("round-port refinement failed: {error}")))?;
    let mut wall_nodes = std::collections::HashSet::new();
    for boundary in candidate
        .boundaries
        .iter()
        .filter(|boundary| boundary.marker == MARKER_WALL)
    {
        wall_nodes.extend(boundary.nodes.iter().copied());
    }
    let mut layer_wall_radius: std::collections::HashMap<u64, f64> =
        std::collections::HashMap::new();
    for &node in &wall_nodes {
        let point = candidate.nodes[node];
        let radius = point.y.hypot(point.z);
        if !(radius.is_finite() && radius > 0.0) {
            return Err(PortFlowError::Mesh(format!(
                "wall node {node} has invalid radial coordinate {radius}"
            )));
        }
        layer_wall_radius
            .entry(point.x.to_bits())
            .and_modify(|maximum| *maximum = maximum.max(radius))
            .or_insert(radius);
    }
    for point in &mut candidate.nodes {
        let Some(&layer_radius) = layer_wall_radius.get(&point.x.to_bits()) else {
            continue;
        };
        let target = geo.local_radius_m(point.x)?;
        let scale = target / layer_radius;
        point.y *= scale;
        point.z *= scale;
    }
    for node in wall_nodes {
        let point = candidate.nodes[node];
        let radius = point.y.hypot(point.z);
        let target = geo.local_radius_m(point.x)?;
        let scale = target / radius;
        candidate.nodes[node].y *= scale;
        candidate.nodes[node].z *= scale;
    }
    candidate.invalidate_caches();
    for (index, element) in candidate.elements.iter().enumerate() {
        let determinant = tet_determinant(
            &candidate,
            [
                element.nodes[0],
                element.nodes[1],
                element.nodes[2],
                element.nodes[3],
            ],
        );
        if !(determinant.is_finite() && determinant > 0.0) {
            return Err(PortFlowError::Mesh(format!(
                "projected refinement inverted tetrahedron {index}"
            )));
        }
    }
    validate_boundary_watertight(&candidate)?;
    validate_boundary_geometry(
        &candidate,
        geo.chamber_length_m,
        geo.duct_length_m,
        geo.mesh_size_m,
    )?;
    *mesh = candidate;
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    // Scale-independent regression envelopes measured over the sharp,
    // flared, and exact-cylinder fixtures below.
    const MIN_NORMALIZED_QUALITY: f64 = 0.08;
    const MAX_EDGE_ASPECT: f64 = 28.0;
    const MAX_EDGE_TO_MESH_SIZE: f64 = 2.1;

    fn geometry(flare_radius_m: f64) -> RoundPortGeometry {
        RoundPortGeometry {
            duct_length_m: 0.15,
            duct_radius_m: 0.025,
            chamber_length_m: 0.05,
            chamber_radius_m: 0.08,
            flare_radius_m,
            mesh_size_m: 0.01,
        }
    }

    fn distance(a: Point, b: Point) -> f64 {
        ((a.x - b.x).powi(2) + (a.y - b.y).powi(2) + (a.z - b.z).powi(2)).sqrt()
    }

    fn tetra_quality(mesh: &Mesh, nodes: &[usize]) -> (f64, f64, f64) {
        let points = [
            mesh.nodes[nodes[0]],
            mesh.nodes[nodes[1]],
            mesh.nodes[nodes[2]],
            mesh.nodes[nodes[3]],
        ];
        let determinant = tet_determinant(mesh, [nodes[0], nodes[1], nodes[2], nodes[3]]);
        let edges = [
            distance(points[0], points[1]),
            distance(points[0], points[2]),
            distance(points[0], points[3]),
            distance(points[1], points[2]),
            distance(points[1], points[3]),
            distance(points[2], points[3]),
        ];
        let min_edge = edges.iter().copied().fold(f64::INFINITY, f64::min);
        let max_edge = edges.iter().copied().fold(0.0, f64::max);
        let sum_squared = edges.iter().map(|edge| edge * edge).sum::<f64>();
        let volume = determinant / 6.0;
        let mean_ratio = 12.0 * (3.0 * volume).powf(2.0 / 3.0) / sum_squared;
        (mean_ratio, max_edge / min_edge, max_edge)
    }

    fn assert_valid_round_mesh(mesh: &Mesh, geo: &RoundPortGeometry) {
        println!(
            "round-port mesh: nodes={}, tetrahedra={}, boundary_facets={}",
            mesh.num_nodes(),
            mesh.elements.len(),
            mesh.boundaries.len()
        );
        assert_eq!(mesh.dimension, 3);
        assert!(mesh.num_nodes() > 1000, "only {} nodes", mesh.num_nodes());
        for marker in [MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL] {
            assert!(
                mesh.boundaries
                    .iter()
                    .any(|boundary| boundary.marker == marker)
            );
        }
        assert!(
            mesh.boundaries
                .iter()
                .all(|boundary| boundary.nodes.len() == 3)
        );
        let mut min_quality = f64::INFINITY;
        let mut max_aspect = 0.0_f64;
        let mut max_edge = 0.0_f64;
        for element in &mesh.elements {
            assert_eq!(element.element_type, ElementType::Tetrahedron);
            assert_eq!(element.nodes.len(), 4);
            let determinant = tet_determinant(
                mesh,
                [
                    element.nodes[0],
                    element.nodes[1],
                    element.nodes[2],
                    element.nodes[3],
                ],
            );
            assert!(determinant.is_finite() && determinant > 0.0);
            let (quality, aspect, edge) = tetra_quality(mesh, &element.nodes);
            min_quality = min_quality.min(quality);
            max_aspect = max_aspect.max(aspect);
            max_edge = max_edge.max(edge);
        }
        println!(
            "quality: min_mean_ratio={min_quality:.4}, max_edge_aspect={max_aspect:.4}, max_edge/h={:.4}",
            max_edge / geo.mesh_size_m
        );
        assert!(min_quality >= MIN_NORMALIZED_QUALITY);
        assert!(max_aspect <= MAX_EDGE_ASPECT);
        assert!(max_edge <= MAX_EDGE_TO_MESH_SIZE * geo.mesh_size_m);

        for index in 0..mesh.boundaries.len() {
            let boundary = &mesh.boundaries[index];
            let owner = &mesh.elements[boundary.element_idx];
            assert!(boundary.nodes.iter().all(|node| owner.nodes.contains(node)));
            let (normal, area) = boundary_outward_normal(mesh, index).unwrap();
            assert!(area.is_finite() && area > 0.0);
            let centroid = boundary.nodes.iter().fold([0.0; 3], |mut sum, &node| {
                let point = mesh.nodes[node];
                sum[0] += point.x / 3.0;
                sum[1] += point.y / 3.0;
                sum[2] += point.z / 3.0;
                sum
            });
            match boundary.marker {
                MARKER_BOX_CAP => assert!(normal[0] < 0.0),
                MARKER_EXTERIOR_CAP => assert!(normal[0] > 0.0),
                MARKER_WALL if centroid[0].abs() <= 1.0e-12 => assert!(normal[0] > 0.0),
                MARKER_WALL if (centroid[0] - geo.duct_length_m).abs() <= 1.0e-12 => {
                    assert!(normal[0] < 0.0)
                }
                MARKER_WALL => {
                    assert!(normal[1] * centroid[1] + normal[2] * centroid[2] > 0.0)
                }
                marker => panic!("unexpected boundary marker {marker}"),
            }
        }
    }

    fn boundary_area_at(mesh: &Mesh, marker: i32, x: f64) -> f64 {
        mesh.boundaries
            .iter()
            .enumerate()
            .filter(|(_, boundary)| {
                boundary.marker == marker
                    && boundary
                        .nodes
                        .iter()
                        .all(|&node| (mesh.nodes[node].x - x).abs() <= 1.0e-12)
            })
            .map(|(index, _)| boundary_outward_normal(mesh, index).unwrap().1)
            .sum()
    }

    fn assert_area_close(actual: f64, expected: f64) {
        let relative_error = (actual - expected).abs() / expected;
        assert!(
            relative_error < 0.02,
            "area {actual} differs from {expected} by {:.2}%",
            100.0 * relative_error
        );
    }

    fn assert_unique_join_nodes(mesh: &Mesh, x: f64) {
        let nodes: Vec<_> = mesh
            .nodes
            .iter()
            .filter(|point| point.x.to_bits() == x.to_bits())
            .collect();
        let unique: HashSet<_> = nodes
            .iter()
            .map(|point| (point.x.to_bits(), point.y.to_bits(), point.z.to_bits()))
            .collect();
        assert!(!nodes.is_empty(), "missing expected layer at x={x}");
        assert_eq!(nodes.len(), unique.len(), "duplicate nodes at x={x}");
    }

    #[test]
    fn canonical_radius_profile_covers_sharp_shoulders_and_domain() {
        let geo = geometry(0.0);
        let rc = geo.chamber_radius_m;
        let rd = geo.duct_radius_m;
        for (x, expected) in [
            (-geo.chamber_length_m, rc),
            (-0.5 * geo.chamber_length_m, rc),
            (-f64::EPSILON, rc),
            (0.0, rd),
            (0.5 * geo.duct_length_m, rd),
            (geo.duct_length_m, rd),
            (geo.duct_length_m + f64::EPSILON, rc),
            (geo.duct_length_m + 0.5 * geo.chamber_length_m, rc),
            (geo.duct_length_m + geo.chamber_length_m, rc),
        ] {
            assert_eq!(geo.local_radius_m(x).unwrap(), expected, "x={x}");
        }
        for x in [
            -geo.chamber_length_m - f64::EPSILON,
            geo.duct_length_m + geo.chamber_length_m + f64::EPSILON,
            f64::NAN,
        ] {
            assert!(matches!(
                geo.local_radius_m(x),
                Err(PortFlowError::InvalidConfig(_))
            ));
        }
    }

    #[test]
    fn canonical_flared_radius_is_continuous_symmetric_and_uses_smoothstep() {
        let geo = geometry(0.02);
        let flare = geo.flare_radius_m;
        let rc = geo.chamber_radius_m;
        let rd = geo.duct_radius_m;
        let midpoint = 0.5 * (rc + rd);
        for (x, expected) in [
            (-geo.chamber_length_m, rc),
            (-flare, rc),
            (-0.5 * flare, midpoint),
            (0.0, rd),
            (0.5 * geo.duct_length_m, rd),
            (geo.duct_length_m, rd),
            (geo.duct_length_m + 0.5 * flare, midpoint),
            (geo.duct_length_m + flare, rc),
            (geo.duct_length_m + geo.chamber_length_m, rc),
        ] {
            assert!(
                (geo.local_radius_m(x).unwrap() - expected).abs() < 1e-15,
                "x={x}"
            );
        }

        let epsilon = flare * 1e-9;
        for join in [-flare, 0.0, geo.duct_length_m, geo.duct_length_m + flare] {
            let left = geo.local_radius_m(join - epsilon).unwrap();
            let at = geo.local_radius_m(join).unwrap();
            let right = geo.local_radius_m(join + epsilon).unwrap();
            assert!((left - at).abs() < 1e-12, "left discontinuity at {join}");
            assert!((right - at).abs() < 1e-12, "right discontinuity at {join}");
        }
        for i in 0..=20 {
            let left_x = -flare + flare * i as f64 / 20.0;
            let right_x = geo.duct_length_m - left_x;
            assert!(
                (geo.local_radius_m(left_x).unwrap() - geo.local_radius_m(right_x).unwrap()).abs()
                    < 1e-15
            );
        }
    }

    #[test]
    fn projected_flared_refinement_preserves_exact_profile_for_two_levels() {
        let geo = RoundPortGeometry {
            duct_length_m: 0.04,
            duct_radius_m: 0.01,
            chamber_length_m: 0.015,
            chamber_radius_m: 0.02,
            flare_radius_m: 0.005,
            mesh_size_m: 0.01,
        };
        let mut mesh = build_round_port_mesh(&geo).unwrap();
        for _ in 0..2 {
            refine_round_port_mesh_projected(&mut mesh, &geo).unwrap();
            for boundary in mesh
                .boundaries
                .iter()
                .filter(|boundary| boundary.marker == MARKER_WALL)
            {
                for &node in &boundary.nodes {
                    let point = mesh.nodes[node];
                    let exact = geo.local_radius_m(point.x).unwrap();
                    assert!((point.y.hypot(point.z) - exact).abs() < 1e-12);
                }
            }
            for marker in [MARKER_BOX_CAP, MARKER_EXTERIOR_CAP, MARKER_WALL] {
                assert!(
                    mesh.boundaries
                        .iter()
                        .any(|boundary| boundary.marker == marker)
                );
            }
        }
    }

    #[test]
    fn projected_refinement_rejects_sharp_shoulders_without_mutating_mesh() {
        let geo = geometry(0.0);
        let mut mesh = build_round_port_mesh(&geo).unwrap();
        let original = format!("{mesh:?}");

        assert!(matches!(
            refine_round_port_mesh_projected(&mut mesh, &geo),
            Err(PortFlowError::InvalidConfig(_))
        ));
        assert_eq!(format!("{mesh:?}"), original);
    }

    #[test]
    fn projected_refinement_failure_is_transactional() {
        let geo = RoundPortGeometry {
            duct_length_m: 0.04,
            duct_radius_m: 0.01,
            chamber_length_m: 0.015,
            chamber_radius_m: 0.02,
            flare_radius_m: 0.005,
            mesh_size_m: 0.01,
        };
        let mut mesh = build_round_port_mesh(&geo).unwrap();
        mesh.boundaries[0].marker = i32::MAX;
        let original = format!("{mesh:?}");

        let error = refine_round_port_mesh_projected(&mut mesh, &geo).unwrap_err();
        assert!(
            matches!(
                &error,
                PortFlowError::Mesh(message)
                    if message.contains("inconsistent marker/orientation")
            ),
            "unexpected error: {error}"
        );
        assert_eq!(format!("{mesh:?}"), original);
    }

    #[test]
    fn round_port_mesh_sanity() {
        let geo = geometry(0.01);
        assert_valid_round_mesh(&build_round_port_mesh(&geo).unwrap(), &geo);
    }

    #[test]
    fn straight_round_port_preserves_caps_walls_and_tetrahedra() {
        let geo = geometry(0.0);
        let (nodes, facets) = build_surface(&geo, geo.mesh_size_m).unwrap();
        validate_closed_surface(&facets).unwrap();
        assert!(nodes.len() > 1000);
        assert!(
            facets
                .iter()
                .any(|facet| facet.marker == Some(MARKER_BOX_CAP))
        );
        assert!(
            facets
                .iter()
                .any(|facet| facet.marker == Some(MARKER_EXTERIOR_CAP))
        );
        assert!(facets.iter().any(|facet| facet.marker == Some(MARKER_WALL)));
    }

    #[test]
    fn structured_sharp_round_port_has_valid_shoulders() {
        let geo = geometry(0.0);
        let mesh = build_round_port_mesh(&geo).unwrap();
        assert_valid_round_mesh(&mesh, &geo);
        assert_unique_join_nodes(&mesh, 0.0);
        assert_unique_join_nodes(&mesh, geo.duct_length_m);
        assert_area_close(
            boundary_area_at(&mesh, MARKER_BOX_CAP, -geo.chamber_length_m),
            std::f64::consts::PI * geo.chamber_radius_m.powi(2),
        );
        assert_area_close(
            boundary_area_at(
                &mesh,
                MARKER_EXTERIOR_CAP,
                geo.duct_length_m + geo.chamber_length_m,
            ),
            std::f64::consts::PI * geo.chamber_radius_m.powi(2),
        );
        let shoulder_area =
            std::f64::consts::PI * (geo.chamber_radius_m.powi(2) - geo.duct_radius_m.powi(2));
        assert_area_close(boundary_area_at(&mesh, MARKER_WALL, 0.0), shoulder_area);
        assert_area_close(
            boundary_area_at(&mesh, MARKER_WALL, geo.duct_length_m),
            shoulder_area,
        );
    }

    #[test]
    fn equal_radii_build_one_cylinder_without_internal_shoulders_or_duplicate_joins() {
        let mut geo = geometry(0.0);
        geo.chamber_radius_m = geo.duct_radius_m;
        let mesh = build_round_port_mesh(&geo).unwrap();
        assert_valid_round_mesh(&mesh, &geo);
        for x in [0.0, geo.duct_length_m] {
            assert_unique_join_nodes(&mesh, x);
            assert!(!mesh.boundaries.iter().any(|boundary| {
                boundary
                    .nodes
                    .iter()
                    .all(|&node| mesh.nodes[node].x.to_bits() == x.to_bits())
            }));
        }
        let cap_area = std::f64::consts::PI * geo.duct_radius_m.powi(2);
        assert_area_close(
            boundary_area_at(&mesh, MARKER_BOX_CAP, -geo.chamber_length_m),
            cap_area,
        );
        assert_area_close(
            boundary_area_at(
                &mesh,
                MARKER_EXTERIOR_CAP,
                geo.duct_length_m + geo.chamber_length_m,
            ),
            cap_area,
        );
    }

    #[test]
    fn flared_round_port_profile_is_smooth_and_non_inverted() {
        let geo = geometry(0.02);
        let profile = meridian_profile(&geo, geo.mesh_size_m);
        assert!(profile.windows(2).all(|pair| pair[1].0 >= pair[0].0));
        assert!(
            profile.iter().all(|&(_, radius)| {
                radius >= geo.duct_radius_m && radius <= geo.chamber_radius_m
            })
        );
        let (nodes, facets) = build_surface(&geo, geo.mesh_size_m).unwrap();
        validate_closed_surface(&facets).unwrap();
        for facet in &facets {
            let [a, b, c] = facet.nodes.map(|index| nodes[index]);
            let ab = [b.x - a.x, b.y - a.y, b.z - a.z];
            let ac = [c.x - a.x, c.y - a.y, c.z - a.z];
            let normal = [
                ab[1] * ac[2] - ab[2] * ac[1],
                ab[2] * ac[0] - ab[0] * ac[2],
                ab[0] * ac[1] - ab[1] * ac[0],
            ];
            let centroid = [
                (a.x + b.x + c.x) / 3.0,
                (a.y + b.y + c.y) / 3.0,
                (a.z + b.z + c.z) / 3.0,
            ];
            let outward_dot = match facet.marker {
                Some(MARKER_BOX_CAP) => -normal[0],
                Some(MARKER_EXTERIOR_CAP) => normal[0],
                _ => normal[1] * centroid[1] + normal[2] * centroid[2],
            };
            assert!(outward_dot > 0.0, "inward facet {:?}", facet.nodes);
        }
    }

    #[test]
    fn round_port_rejects_unphysical_dimensions() {
        let mut geo = geometry(0.01);
        geo.chamber_radius_m = geo.duct_radius_m * 0.5;
        assert!(matches!(
            build_round_port_mesh(&geo),
            Err(PortFlowError::InvalidConfig(_))
        ));
        geo = geometry(0.06);
        assert!(matches!(
            build_round_port_mesh(&geo),
            Err(PortFlowError::InvalidConfig(_))
        ));
    }

    #[test]
    fn round_port_rejects_meshes_above_resource_limits_without_allocating() {
        let mut geo = geometry(0.01);
        geo.mesh_size_m = 1.0e-300;
        assert!(matches!(
            build_round_port_mesh(&geo),
            Err(PortFlowError::InvalidConfig(_))
        ));
    }
}
