//! 3D propeller blade geometry and panel source metrics.
//!
//! This is intentionally application-level geometry rather than FEM core code.
//! It generates panelized conventional and toroidal loop blades so that design
//! studies can compare real 3D planform effects before a full volumetric CFD or
//! acoustic solve is added.

use rayon::prelude::*;
use sonium_fem::physics::acoustics::aeroacoustics::PropellerSource;
use std::fmt;

use crate::source_model::{
    EdgeTreatment, EvaluationMode, PropellerDesign, PropellerPlanform, PropellerSpectrum,
    SerrationGeometry,
};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

impl Point3 {
    fn sub(self, other: Self) -> Self {
        Self {
            x: self.x - other.x,
            y: self.y - other.y,
            z: self.z - other.z,
        }
    }

    fn cross(self, other: Self) -> Self {
        Self {
            x: self.y * other.z - self.z * other.y,
            y: self.z * other.x - self.x * other.z,
            z: self.x * other.y - self.y * other.x,
        }
    }

    fn norm(self) -> f64 {
        (self.x * self.x + self.y * self.y + self.z * self.z).sqrt()
    }
}

#[derive(Debug, Clone)]
pub struct BladePanel {
    pub vertices: [Point3; 4],
    /// Stable zero-based blade identity within a replicated mesh.
    pub blade_index: usize,
    pub radius_fraction: f64,
    pub chord_fraction: f64,
    pub area: f64,
}

#[derive(Debug, Clone)]
pub struct BladeMesh {
    pub panels: Vec<BladePanel>,
    pub planform: PropellerPlanform,
    pub edge_treatment: EdgeTreatment,
}

impl BladeMesh {
    /// Sum of panel areas in deterministic panel order.
    ///
    /// Naive sequential summation: worst-case error ~`n·eps·max area`, i.e.
    /// sub-ppm for the panel counts used here (≤ 10⁴). No compensation is
    /// needed at this scale; revisit with Kahan summation if panel counts
    /// grow by orders of magnitude.
    pub fn total_area(&self) -> f64 {
        self.panels.iter().map(|p| p.area).sum()
    }

    pub fn tip_panel_area(&self) -> f64 {
        self.panels
            .iter()
            .filter(|p| p.radius_fraction > TIP_RADIUS_FRACTION)
            .map(|p| p.area)
            .sum()
    }

    /// Write the panelised blade mesh to a `.vtu` for ParaView. Each panel is
    /// emitted as an independent quad with its own 4 nodes, with per-cell
    /// radius_fraction, chord_fraction, and area scalars.
    pub fn write_vtu(&self, path: impl AsRef<std::path::Path>) -> std::io::Result<()> {
        struct PanelView<'a>(&'a BladeMesh);
        impl<'a> sonium_xem::io::vtu::MeshView for PanelView<'a> {
            fn num_nodes(&self) -> usize {
                self.0.panels.len() * 4
            }
            fn num_cells(&self) -> usize {
                self.0.panels.len()
            }
            fn nodes(&self) -> Box<dyn Iterator<Item = [f64; 3]> + '_> {
                Box::new(
                    self.0
                        .panels
                        .iter()
                        .flat_map(|p| p.vertices.iter().map(|v| [v.x, v.y, v.z])),
                )
            }
            fn cells(
                &self,
            ) -> Box<dyn Iterator<Item = (sonium_xem::io::vtu::CellKind, Vec<u64>)> + '_>
            {
                Box::new((0..self.0.panels.len()).map(|i| {
                    let base = (i * 4) as u64;
                    (
                        sonium_xem::io::vtu::CellKind::Quad,
                        vec![base, base + 1, base + 2, base + 3],
                    )
                }))
            }
        }

        let view = PanelView(self);
        let mut w = sonium_xem::io::vtu::VtuWriter::new(&view);
        let radius: Vec<f64> = self.panels.iter().map(|p| p.radius_fraction).collect();
        let chord: Vec<f64> = self.panels.iter().map(|p| p.chord_fraction).collect();
        let area: Vec<f64> = self.panels.iter().map(|p| p.area).collect();
        w.add_cell_scalar("radius_fraction", &radius);
        w.add_cell_scalar("chord_fraction", &chord);
        w.add_cell_scalar("area", &area);
        w.write(path)
    }
}

/// Valid range for `loop_closure_fraction`, shared by `Blade3dConfig::validate`
/// and the optimizer clamp so direct config users can build every geometry
/// the optimizer can emit and vice versa.
pub const LOOP_CLOSURE_FRACTION_RANGE: [f64; 2] = [0.35, 0.95];

/// Radius fraction above which a panel counts as a tip panel.
pub const TIP_RADIUS_FRACTION: f64 = 0.88;

#[derive(Debug, Clone, Copy)]
pub struct Blade3dConfig {
    pub radial_segments: usize,
    pub chord_segments: usize,
    pub loop_segments: usize,
    pub pitch_deg_root: f64,
    pub pitch_deg_tip: f64,
    pub chord_root: f64,
    pub chord_tip: f64,
    pub loop_closure_fraction: f64,
    pub loop_axial_lift: f64,
}

/// Invalid public input to checked blade-mesh generation.
#[derive(Debug, Clone, PartialEq)]
pub enum BladeMeshError {
    /// A mesh resolution is zero.
    ZeroSegments {
        /// Resolution field name.
        field: &'static str,
    },
    /// A geometric parameter is non-finite or outside its physical range.
    InvalidGeometry {
        /// Parameter name.
        field: &'static str,
        /// Required domain.
        requirement: &'static str,
    },
    /// A propeller source has no physical blade count.
    ZeroBlades,
    /// The requested panel count cannot be represented by `usize`.
    PanelCountOverflow,
}

impl fmt::Display for BladeMeshError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroSegments { field } => write!(f, "blade mesh {field} must be non-zero"),
            Self::InvalidGeometry { field, requirement } => {
                write!(f, "blade mesh {field} must be {requirement}")
            }
            Self::ZeroBlades => write!(f, "propeller source must contain at least one blade"),
            Self::PanelCountOverflow => write!(f, "blade mesh panel count overflows usize"),
        }
    }
}

impl std::error::Error for BladeMeshError {}

impl Default for Blade3dConfig {
    fn default() -> Self {
        Self {
            radial_segments: 96,
            chord_segments: 18,
            loop_segments: 40,
            pitch_deg_root: 28.0,
            pitch_deg_tip: 12.0,
            chord_root: 0.028,
            chord_tip: 0.010,
            loop_closure_fraction: 0.78,
            loop_axial_lift: 0.012,
        }
    }
}

impl Blade3dConfig {
    pub fn research_control_64c() -> Self {
        Self {
            radial_segments: 320,
            chord_segments: 64,
            loop_segments: 160,
            ..Self::default()
        }
    }

    /// Validate the geometric and discretization inputs before mesh creation.
    pub fn validate(&self, design: &PropellerDesign) -> Result<(), BladeMeshError> {
        for (field, value) in [
            ("radial_segments", self.radial_segments),
            ("chord_segments", self.chord_segments),
            ("loop_segments", self.loop_segments),
        ] {
            if value == 0 {
                return Err(BladeMeshError::ZeroSegments { field });
            }
        }
        for (field, value) in [
            ("pitch_deg_root", self.pitch_deg_root),
            ("pitch_deg_tip", self.pitch_deg_tip),
            ("chord_root", self.chord_root),
            ("chord_tip", self.chord_tip),
            ("loop_closure_fraction", self.loop_closure_fraction),
            ("loop_axial_lift", self.loop_axial_lift),
            ("tip_radius", design.source.tip_radius),
        ] {
            if !value.is_finite() {
                return Err(BladeMeshError::InvalidGeometry {
                    field,
                    requirement: "finite",
                });
            }
        }
        if self.chord_root <= 0.0 || self.chord_tip <= 0.0 {
            return Err(BladeMeshError::InvalidGeometry {
                field: "chord_root/chord_tip",
                requirement: "finite and positive",
            });
        }
        if design.source.tip_radius <= 0.0 {
            return Err(BladeMeshError::InvalidGeometry {
                field: "tip_radius",
                requirement: "finite and positive",
            });
        }
        if !(LOOP_CLOSURE_FRACTION_RANGE[0]..=LOOP_CLOSURE_FRACTION_RANGE[1])
            .contains(&self.loop_closure_fraction)
        {
            return Err(BladeMeshError::InvalidGeometry {
                field: "loop_closure_fraction",
                requirement: "within [0.35, 0.95]",
            });
        }
        if self.loop_axial_lift < 0.0 {
            return Err(BladeMeshError::InvalidGeometry {
                field: "loop_axial_lift",
                requirement: "finite and non-negative",
            });
        }
        if design.source.n_blades == 0 {
            return Err(BladeMeshError::ZeroBlades);
        }

        let per_blade_rows = match design.planform {
            PropellerPlanform::Conventional => self.radial_segments,
            PropellerPlanform::ToroidalLoop => self
                .radial_segments
                .checked_add(self.loop_segments)
                .ok_or(BladeMeshError::PanelCountOverflow)?,
        };
        per_blade_rows
            .checked_mul(self.chord_segments)
            .and_then(|count| count.checked_mul(design.source.n_blades))
            .ok_or(BladeMeshError::PanelCountOverflow)?;
        Ok(())
    }

    /// Build a validated panel mesh for a conventional or toroidal blade.
    pub fn try_build_mesh(&self, design: &PropellerDesign) -> Result<BladeMesh, BladeMeshError> {
        self.validate(design)?;
        let mut panels = match design.planform {
            PropellerPlanform::Conventional => self.build_conventional_blade(design),
            PropellerPlanform::ToroidalLoop => self.build_toroidal_blade(design),
        };

        if design.source.n_blades > 1 {
            panels = replicate_blades(&panels, design.source.n_blades);
        }

        Ok(BladeMesh {
            panels,
            planform: design.planform,
            edge_treatment: design.edge_treatment,
        })
    }

    /// Build a mesh for trusted legacy configuration.
    pub fn build_mesh(&self, design: &PropellerDesign) -> BladeMesh {
        self.try_build_mesh(design)
            .expect("invalid propeller blade mesh configuration")
    }

    fn build_conventional_blade(&self, design: &PropellerDesign) -> Vec<BladePanel> {
        self.build_radial_blade(design, 1.0)
    }

    fn build_radial_blade(
        &self,
        design: &PropellerDesign,
        outer_span_fraction: f64,
    ) -> Vec<BladePanel> {
        let mut panels = Vec::with_capacity(self.radial_segments * self.chord_segments);
        let source = &design.source;
        let eta_end = outer_span_fraction.clamp(0.0, 1.0);
        for ir in 0..self.radial_segments {
            let e0 = eta_end * ir as f64 / self.radial_segments as f64;
            let e1 = eta_end * (ir + 1) as f64 / self.radial_segments as f64;
            for ic in 0..self.chord_segments {
                let c0 = ic as f64 / self.chord_segments as f64;
                let c1 = (ic + 1) as f64 / self.chord_segments as f64;
                let vertices = [
                    self.blade_point(source, design, e0, c0, 0.0),
                    self.blade_point(source, design, e1, c0, 0.0),
                    self.blade_point(source, design, e1, c1, 0.0),
                    self.blade_point(source, design, e0, c1, 0.0),
                ];
                panels.push(panel(vertices, 0.5 * (e0 + e1), 0.5 * (c0 + c1)));
            }
        }
        panels
    }

    fn build_toroidal_blade(&self, design: &PropellerDesign) -> Vec<BladePanel> {
        let source = &design.source;
        let eta0 = self.loop_closure_fraction.clamp(
            LOOP_CLOSURE_FRACTION_RANGE[0],
            LOOP_CLOSURE_FRACTION_RANGE[1],
        );
        // The loop replaces the outer radial blade from eta0 to the tip.
        // Keeping a full conventional blade underneath the loop creates an
        // overlapping duplicate lifting surface and double-counts area.
        let mut panels = self.build_radial_blade(design, eta0);
        for il in 0..self.loop_segments {
            let u0 = il as f64 / self.loop_segments as f64;
            let u1 = (il + 1) as f64 / self.loop_segments as f64;
            for ic in 0..self.chord_segments {
                let c0 = ic as f64 / self.chord_segments as f64;
                let c1 = (ic + 1) as f64 / self.chord_segments as f64;
                let vertices = [
                    self.loop_point(source, design, eta0, u0, c0),
                    self.loop_point(source, design, eta0, u1, c0),
                    self.loop_point(source, design, eta0, u1, c1),
                    self.loop_point(source, design, eta0, u0, c1),
                ];
                let eta_mid = eta0
                    + (1.0 - eta0) * (0.5 - 0.5 * (std::f64::consts::TAU * 0.5 * (u0 + u1)).cos());
                panels.push(panel(vertices, eta_mid, 0.5 * (c0 + c1)));
            }
        }
        panels
    }

    fn blade_point(
        &self,
        source: &PropellerSource,
        design: &PropellerDesign,
        eta: f64,
        chord_fraction: f64,
        loop_z: f64,
    ) -> Point3 {
        let radius = source.hub_radius + eta * (source.tip_radius - source.hub_radius);
        let chord = self.chord_root + eta * (self.chord_tip - self.chord_root);
        let pitch =
            (self.pitch_deg_root + eta * (self.pitch_deg_tip - self.pitch_deg_root)).to_radians();
        let serration = serration_offset(
            design.edge_treatment,
            design.serrations,
            eta,
            chord_fraction,
            chord,
            source.hub_radius,
            source.tip_radius,
        );
        let x_chord = (chord_fraction - 0.5) * chord + serration;
        let theta = 0.35 * eta.powf(1.25);
        let swept_radius = radius + x_chord * pitch.sin();
        let axial = x_chord * pitch.cos() + loop_z;
        Point3 {
            x: swept_radius * theta.cos(),
            y: swept_radius * theta.sin(),
            z: axial,
        }
    }

    fn loop_point(
        &self,
        source: &PropellerSource,
        design: &PropellerDesign,
        eta0: f64,
        u: f64,
        chord_fraction: f64,
    ) -> Point3 {
        let loop_phase = std::f64::consts::TAU * u;
        let eta = eta0 + (1.0 - eta0) * (0.5 - 0.5 * loop_phase.cos());
        let loop_z = self.loop_axial_lift * loop_phase.sin();
        self.blade_point(source, design, eta, chord_fraction, loop_z)
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelAcousticSummary {
    pub total_area: f64,
    pub tip_area_fraction: f64,
    pub vortex_source_index: f64,
    pub edge_coherence_index: f64,
    pub structural_rigidity_index: f64,
    pub sensitive_band_reduction_db: f64,
}

pub fn compare_panel_acoustics(
    baseline: &BladeMesh,
    candidate: &BladeMesh,
    baseline_spectrum: &PropellerSpectrum,
    candidate_spectrum: &PropellerSpectrum,
) -> PanelAcousticSummary {
    let total_area = candidate.total_area();
    let baseline_tip_fraction = baseline.tip_panel_area() / baseline.total_area().max(f64::EPSILON);
    let tip_area_fraction = candidate.tip_panel_area() / total_area.max(f64::EPSILON);
    let vortex_source_index = if candidate.planform == PropellerPlanform::ToroidalLoop {
        let closed_tip_leakage = 0.18;
        closed_tip_leakage * tip_area_fraction / baseline_tip_fraction.max(f64::EPSILON)
    } else {
        tip_area_fraction / baseline_tip_fraction.max(f64::EPSILON)
    };
    let edge_coherence_index = match candidate.edge_treatment {
        EdgeTreatment::Smooth => 1.0,
        EdgeTreatment::LeadingEdgeSerrations | EdgeTreatment::TrailingEdgeSerrations => 0.72,
        EdgeTreatment::LeadingAndTrailingSerrations => 0.54,
    };
    let structural_rigidity_index = if candidate.planform == PropellerPlanform::ToroidalLoop {
        1.45
    } else {
        1.0
    };

    PanelAcousticSummary {
        total_area,
        tip_area_fraction,
        vortex_source_index,
        edge_coherence_index,
        structural_rigidity_index,
        sensitive_band_reduction_db: candidate_spectrum.band_reduction_db_against(
            baseline_spectrum,
            1_000.0,
            5_000.0,
        ),
    }
}

pub fn design_spectrum_for_mesh(design: &PropellerDesign, n_harmonics: usize) -> PropellerSpectrum {
    design.spectrum(n_harmonics, EvaluationMode::ResearchControl)
}

fn panel(vertices: [Point3; 4], radius_fraction: f64, chord_fraction: f64) -> BladePanel {
    let a = triangle_area(vertices[0], vertices[1], vertices[2])
        + triangle_area(vertices[0], vertices[2], vertices[3]);
    BladePanel {
        vertices,
        blade_index: 0,
        radius_fraction,
        chord_fraction,
        area: a,
    }
}

fn triangle_area(a: Point3, b: Point3, c: Point3) -> f64 {
    0.5 * b.sub(a).cross(c.sub(a)).norm()
}

fn replicate_blades(base: &[BladePanel], n_blades: usize) -> Vec<BladePanel> {
    // Rayon flat_map + collect preserves blade order, so the serial and
    // parallel paths produce identical panel sequences.
    let replicate_blade = |blade: usize| {
        let angle = std::f64::consts::TAU * blade as f64 / n_blades as f64;
        base.iter()
            .map(|panel| BladePanel {
                vertices: panel.vertices.map(|p| rotate_z(p, angle)),
                blade_index: blade,
                radius_fraction: panel.radius_fraction,
                chord_fraction: panel.chord_fraction,
                area: panel.area,
            })
            .collect::<Vec<_>>()
    };
    if crate::parallelism::use_propeller_parallel(
        n_blades.saturating_mul(base.len()),
        crate::parallelism::REPLICATED_PANELS_PER_THREAD,
    ) {
        (0..n_blades)
            .into_par_iter()
            .flat_map(replicate_blade)
            .collect()
    } else {
        (0..n_blades).flat_map(replicate_blade).collect()
    }
}

fn rotate_z(p: Point3, angle: f64) -> Point3 {
    let (s, c) = angle.sin_cos();
    Point3 {
        x: c * p.x - s * p.y,
        y: s * p.x + c * p.y,
        z: p.z,
    }
}

fn serration_offset(
    treatment: EdgeTreatment,
    serrations: SerrationGeometry,
    eta: f64,
    chord_fraction: f64,
    chord: f64,
    hub_radius: f64,
    tip_radius: f64,
) -> f64 {
    if treatment == EdgeTreatment::Smooth {
        return 0.0;
    }

    let span_start = 1.0 - serrations.span_fraction.clamp(0.0, 1.0);
    if eta < span_start {
        return 0.0;
    }

    let leading = matches!(
        treatment,
        EdgeTreatment::LeadingEdgeSerrations | EdgeTreatment::LeadingAndTrailingSerrations
    ) && chord_fraction < 0.18;
    let trailing = matches!(
        treatment,
        EdgeTreatment::TrailingEdgeSerrations | EdgeTreatment::LeadingAndTrailingSerrations
    ) && chord_fraction > 0.82;

    if !leading && !trailing {
        return 0.0;
    }

    // Wavelength is specified as a fraction of local chord, so the phase
    // must use physical span distance divided by a physical wavelength.
    // The previous eta/lambda ratio silently treated lambda as a fraction
    // of total span and changed tooth count with aspect ratio.
    let treated_span_m = (eta - span_start).max(0.0) * (tip_radius - hub_radius).max(0.0);
    let wavelength_m = serrations.wavelength_chord_fraction.max(0.005) * chord.max(1.0e-9);
    let tooth_phase = treated_span_m / wavelength_m;
    let tooth = (std::f64::consts::TAU * tooth_phase).sin();
    let sign = if leading { -1.0 } else { 1.0 };
    sign * serrations.amplitude_chord_fraction * chord * tooth
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toroidal_mesh_has_more_area_and_lower_vortex_index() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let baseline_design = PropellerDesign::conventional(source.clone());
        let candidate_design = PropellerDesign::toroidal_serrated(source);
        let cfg = Blade3dConfig::default();
        let baseline_mesh = cfg.build_mesh(&baseline_design);
        let candidate_mesh = cfg.build_mesh(&candidate_design);
        let baseline_spectrum = baseline_design.spectrum(12, EvaluationMode::ResearchControl);
        let candidate_spectrum = candidate_design.spectrum(12, EvaluationMode::ResearchControl);
        let summary = compare_panel_acoustics(
            &baseline_mesh,
            &candidate_mesh,
            &baseline_spectrum,
            &candidate_spectrum,
        );

        assert!(candidate_mesh.total_area() > baseline_mesh.total_area());
        assert!(summary.vortex_source_index < 0.6);
        assert!(summary.sensitive_band_reduction_db > 3.0);
    }

    #[test]
    fn toroidal_loop_replaces_instead_of_overlapping_outer_blade() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let design = PropellerDesign::toroidal_serrated(source);
        let cfg = Blade3dConfig {
            radial_segments: 20,
            chord_segments: 4,
            loop_segments: 10,
            loop_closure_fraction: 0.75,
            ..Blade3dConfig::default()
        };
        let mesh = cfg.build_mesh(&design);
        let panels_per_blade =
            cfg.radial_segments * cfg.chord_segments + cfg.loop_segments * cfg.chord_segments;
        assert_eq!(mesh.panels.len(), design.source.n_blades * panels_per_blade);

        let radial_tip = mesh
            .panels
            .iter()
            .filter(|panel| panel.radius_fraction <= cfg.loop_closure_fraction)
            .map(|panel| panel.radius_fraction)
            .fold(0.0_f64, f64::max);
        assert!(radial_tip <= cfg.loop_closure_fraction);
    }

    #[test]
    fn toroidal_loop_is_geometrically_closed_and_lifted_out_of_the_blade_plane() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let design = PropellerDesign::toroidal_serrated(source.clone());
        let cfg = Blade3dConfig {
            radial_segments: 4,
            chord_segments: 2,
            loop_segments: 12,
            loop_closure_fraction: 0.75,
            loop_axial_lift: 0.015,
            ..Blade3dConfig::default()
        };
        for chord_fraction in [0.0, 0.5, 1.0] {
            let first = cfg.loop_point(
                &source,
                &design,
                cfg.loop_closure_fraction,
                0.0,
                chord_fraction,
            );
            let closing = cfg.loop_point(
                &source,
                &design,
                cfg.loop_closure_fraction,
                1.0,
                chord_fraction,
            );
            assert!((first.x - closing.x).abs() < 1.0e-14);
            assert!((first.y - closing.y).abs() < 1.0e-14);
            assert!((first.z - closing.z).abs() < 1.0e-14);
        }
        let quarter = cfg.loop_point(&source, &design, cfg.loop_closure_fraction, 0.25, 0.5);
        let three_quarter = cfg.loop_point(&source, &design, cfg.loop_closure_fraction, 0.75, 0.5);
        assert!(quarter.z > 0.0);
        assert!(three_quarter.z < 0.0);
    }

    #[test]
    fn checked_mesh_builder_preserves_valid_legacy_geometry() {
        let design = PropellerDesign::toroidal_serrated(PropellerSource::drone_5inch(18_000.0));
        let config = Blade3dConfig {
            radial_segments: 8,
            chord_segments: 3,
            loop_segments: 6,
            ..Blade3dConfig::default()
        };
        let checked = config.try_build_mesh(&design).unwrap();
        let legacy = config.build_mesh(&design);

        assert_eq!(checked.panels.len(), legacy.panels.len());
        assert!((checked.total_area() - legacy.total_area()).abs() < 1.0e-15);
        assert!(
            checked
                .panels
                .iter()
                .all(|panel| panel.area.is_finite() && panel.area > 0.0)
        );
    }

    #[test]
    fn checked_mesh_builder_rejects_invalid_geometry_before_generation() {
        let source = PropellerSource::drone_5inch(18_000.0);
        let design = PropellerDesign::toroidal_serrated(source.clone());
        let zero_segments = Blade3dConfig {
            radial_segments: 0,
            ..Blade3dConfig::default()
        };
        assert!(matches!(
            zero_segments.try_build_mesh(&design),
            Err(BladeMeshError::ZeroSegments {
                field: "radial_segments"
            })
        ));

        let invalid_loop = Blade3dConfig {
            loop_closure_fraction: 0.2,
            ..Blade3dConfig::default()
        };
        assert!(matches!(
            invalid_loop.try_build_mesh(&design),
            Err(BladeMeshError::InvalidGeometry {
                field: "loop_closure_fraction",
                ..
            })
        ));

        let mut no_blades = source;
        no_blades.n_blades = 0;
        let no_blade_design = PropellerDesign::conventional(no_blades);
        assert!(matches!(
            Blade3dConfig::default().try_build_mesh(&no_blade_design),
            Err(BladeMeshError::ZeroBlades)
        ));
    }
}
