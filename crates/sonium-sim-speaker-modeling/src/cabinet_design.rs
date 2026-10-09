//! Parametric cabinet design variables (review `acoustics-20250730.md`, C4).
//!
//! [`CabinetDesign`] is the single source of truth for box, port, baffle, and
//! bracing geometry. It lowers to every fidelity level the crate supports:
//!
//! - [`CabinetDesign::to_lumped`] — lumped [`EnclosureSpec`] / [`PortSpec`]
//!   (volume- and port-exact).
//! - [`CabinetDesign::to_resolved_config`] — axisymmetric resolved FEM
//!   [`ResolvedAcousticConfig`] (equivalent cylinder).
//! - [`CabinetDesign::to_coupling_fixture`] — 3D FEM-BEM [`CouplingFixture`]
//!   (same equivalent cylinder; structured tet mesh).
//! - [`CabinetDesign::to_cabinet`] — panel-radiation [`Cabinet`] with braces
//!   resolved into sub-panels.
//!
//! Optimizers consume [`CabinetDesign::design_variables`] /
//! [`CabinetDesign::from_design_variables`].
//!
//! # Equivalence choices
//!
//! - **Canonical box form**: internal dimensions `w × h × d`. Volume is
//!   exactly `w·h·d` at every fidelity level.
//! - **Equivalent cylinder** (resolved + coupling fixture): cylinder depth =
//!   box depth `d`; radius `R = √(w·h/π)` matches the front-panel area, so
//!   `πR²d = w·h·d` exactly. Rectangular corner modes are not representable;
//!   the cross-section is treated by area.
//! - **Port**: circular, on the front panel. The resolved and coupling
//!   models are axisymmetric, so the port is mapped to the centerline
//!   regardless of any real-world offset — placement remains
//!   [`PortPlacement::FrontPanel`] and the offset is not modeled there. The lumped
//!   end correction is `1.46·r` (one flanged, one free termination). The
//!   resolved model represents the port as the annular slit through its
//!   baffle: we set `baffle_half_thickness = L/2`, carry the same end
//!   correction, and size the annulus to the same area, so
//!   `compact_helmholtz_frequency_hz` matches the lumped tuning to solver
//!   precision.
//! - **Baffle**: `baffle_w × baffle_h ≥ w × h` (the exterior baffle extends
//!   the front panel). The resolved model receives the equal-area circular
//!   baffle `R_b = √(baffle_w·baffle_h/π)`.
//! - **Bracing**: a brace splits one panel into two simply-supported
//!   sub-panels at `position_fraction` along the chosen axis (mode
//!   frequencies scale ≈ 1/L²). All braces on one panel must share an axis;
//!   multiple braces subdivide sequentially at their sorted fractions. The
//!   front panel is not braceable (it carries the driver and is excluded
//!   from the [`Cabinet`] panel list, matching `Cabinet::sealed_box`).
//!
//! # Documented limitations
//!
//! - The C2 coupling fixture (`CouplingFixture`) has no baffle
//!   representation: its exterior BEM radiates from the cylinder walls
//!   directly. Baffle dimensions are **not** carried into
//!   [`CabinetDesign::to_coupling_fixture`]; use the resolved model for
//!   finite-baffle effects.
//! - The resolved axisymmetric model has no sealed topology: its port slit
//!   is structural. [`CabinetDesign::to_resolved_config`] rejects sealed
//!   designs ([`CabinetDesignError::SealedNotResolvable`]).
//! - The resolved port is a coaxial annular slit placed just outside the
//!   diaphragm radius; designs whose port area does not fit the annulus
//!   inside the enclosure radius are rejected.

use serde::{Deserialize, Serialize};
use sonium_fem::physics::acoustics::electroacoustics::cabinet_radiation::{Cabinet, PanelMaterial};

use super::fem_bem_coupling::fixtures::{
    CouplingFixture, CylinderFixtureDiscretization, FixtureError,
    cylindrical_box_with_circular_aperture, cylindrical_box_with_external_port,
};
use super::{EnclosureSpec, PortSpec, ResolvedAcousticConfig};

/// Total lumped port end correction as a multiple of the port radius:
/// 0.85·r (flanged, baffle side) + 0.61·r (free, cabinet side) ≈ 1.46·r.
pub const PORT_END_CORRECTION_RADIUS_FACTOR: f64 = 1.46;

/// Minimum separation between two brace positions on one panel, and the
/// minimum distance of a brace from the panel edges, as a fraction of the
/// panel dimension. Closer positions would create degenerate sub-panels.
pub const MIN_SUBPANEL_FRACTION: f64 = 1.0e-6;

/// Errors raised by [`CabinetDesign`] validation and lowering.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum CabinetDesignError {
    #[error("{field} must be finite and greater than zero")]
    InvalidDimension { field: &'static str },
    #[error("port diameter must not exceed the smallest front-panel dimension")]
    PortDoesNotFitPanel,
    #[error("port length must not exceed cabinet box depth")]
    PortLengthExceedsDepth,
    #[error("baffle dimensions must cover the front panel (baffle >= box width/height)")]
    BaffleSmallerThanFront,
    #[error("brace position_fraction must lie strictly inside (0, 1)")]
    BraceFractionOutOfRange,
    #[error("all braces on panel {panel:?} must split along the same axis")]
    BraceAxesConflict { panel: PanelId },
    #[error("braces on panel {panel:?} are too close together or to an edge")]
    DegenerateSubPanels { panel: PanelId },
    #[error("expected {expected} design variables, got {got}")]
    DesignVariableCountMismatch { expected: usize, got: usize },
    #[error("design variable {name} = {value} lies outside [{lower}, {upper}]")]
    DesignVariableOutOfBounds {
        name: String,
        value: f64,
        lower: f64,
        upper: f64,
    },
    #[error("the resolved axisymmetric model requires a port; sealed designs are not resolvable")]
    SealedNotResolvable,
    #[error("the equivalent annular port does not fit inside the enclosure radius")]
    EquivalentPortTooLarge,
    #[error("diaphragm radius must be finite, positive, and smaller than the enclosure radius")]
    InvalidDiaphragmRadius,
}

fn require_positive(field: &'static str, value: f64) -> Result<(), CabinetDesignError> {
    if value.is_finite() && value > 0.0 {
        Ok(())
    } else {
        Err(CabinetDesignError::InvalidDimension { field })
    }
}

/// Wall material choice for the cabinet panels.
///
/// This is a categorical variable: continuous material interpolation is not
/// physically meaningful for sheet goods, so it stays out of the
/// design-variable vector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum CabinetMaterial {
    /// Standard 18 mm MDF.
    #[default]
    Mdf18mm,
    /// Standard 18 mm birch plywood.
    Plywood18mm,
}

impl CabinetMaterial {
    /// The sonium-fem panel material used by the panel-radiation model.
    pub fn panel_material(self) -> PanelMaterial {
        match self {
            Self::Mdf18mm => PanelMaterial::mdf_18mm(),
            Self::Plywood18mm => PanelMaterial::plywood_18mm(),
        }
    }
}

/// Port placement. Only the front panel is supported; the axisymmetric and
/// cylindrical lowerings map the port onto the centerline, while the 3-D
/// product path uses [`PortDesign::position_m`] (see the module notes).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PortPlacement {
    /// Centered on the front panel (the axisymmetric-equivalent location).
    #[default]
    FrontPanel,
}

/// Circular bass-reflex port on the front panel.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PortDesign {
    /// Inner diameter (m).
    pub diameter_m: f64,
    /// Tube length (m).
    pub length_m: f64,
    /// Placement (currently always [`PortPlacement::FrontPanel`]).
    pub placement: PortPlacement,
    /// `[x, y]` offset from the baffle centre (m).
    ///
    /// The resolved axisymmetric lowerings intentionally ignore this offset;
    /// the 3-D product path preserves it.
    #[serde(default)]
    pub position_m: [f64; 2],
    /// Radial tube-wall thickness (m) for a resolved product duct. `None`
    /// preserves legacy lumped/axisymmetric artifacts but is insufficient for
    /// a body-fitted interior solid model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub wall_thickness_m: Option<f64>,
}

impl PortDesign {
    /// Lumped end correction for one flanged and one free termination.
    pub fn end_correction_m(&self) -> f64 {
        PORT_END_CORRECTION_RADIUS_FACTOR * self.diameter_m / 2.0
    }

    /// Circular opening area (m²).
    pub fn area_m2(&self) -> f64 {
        std::f64::consts::PI * (self.diameter_m / 2.0).powi(2)
    }

    /// Outer tube diameter when resolved wall geometry is available.
    pub fn outer_diameter_m(&self) -> Option<f64> {
        self.wall_thickness_m
            .map(|thickness| self.diameter_m + 2.0 * thickness)
    }
}

/// A cabinet panel that can carry braces. The front panel is excluded: it
/// carries the driver cutout and is not part of the [`Cabinet`] panel list.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum PanelId {
    Top,
    Bottom,
    Left,
    Right,
    Back,
}

/// Panel axis along which a brace position is measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum PanelAxis {
    /// Along the panel width (the first dimension of the panel).
    Width,
    /// Along the panel height (the second dimension of the panel).
    Height,
}

/// Physical cross-section of a brace that intrudes into the interior air
/// volume. Width runs across the supported panel; depth is normal to that
/// panel and points into the cavity.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BraceCrossSection {
    pub width_m: f64,
    pub depth_m: f64,
}

/// One brace: splits `panel` into two sub-panels at `position_fraction`
/// along `split_along`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct BraceDesign {
    pub panel: PanelId,
    pub split_along: PanelAxis,
    /// Split position as a fraction of the panel dimension along
    /// `split_along`, strictly inside (0, 1).
    pub position_fraction: f64,
    /// Required by a resolved acoustic product mesh. `None` keeps legacy
    /// panel-modal bracing semantics without claiming an acoustic solid.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cross_section: Option<BraceCrossSection>,
}

/// One continuous design variable with its current value and bounds.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DesignVariable {
    pub name: String,
    pub value: f64,
    pub lower: f64,
    pub upper: f64,
}

/// Parametric cabinet design: box, port, baffle, and bracing.
///
/// All box dimensions are **internal**. See the module docs for the
/// equivalence choices used by each lowering.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CabinetDesign {
    /// Internal box width (m) — front panel horizontal.
    pub box_width_m: f64,
    /// Internal box height (m) — front panel vertical.
    pub box_height_m: f64,
    /// Internal box depth (m) — front-to-back.
    pub box_depth_m: f64,
    /// Panel material (categorical; not a continuous design variable).
    pub wall_material: CabinetMaterial,
    /// Bass-reflex port; `None` for a sealed cabinet (categorical).
    pub port: Option<PortDesign>,
    /// Exterior baffle width (m), ≥ `box_width_m`.
    pub baffle_width_m: f64,
    /// Exterior baffle height (m), ≥ `box_height_m`.
    pub baffle_height_m: f64,
    /// Panel braces. The brace **count** is categorical (structural); the
    /// brace **positions** are continuous design variables.
    pub braces: Vec<BraceDesign>,
}

impl Default for CabinetDesign {
    /// A 20 litre vented bookshelf cabinet (0.25 × 0.40 × 0.20 m internal).
    fn default() -> Self {
        Self {
            box_width_m: 0.25,
            box_height_m: 0.40,
            box_depth_m: 0.20,
            wall_material: CabinetMaterial::Mdf18mm,
            port: Some(PortDesign {
                diameter_m: 0.05,
                length_m: 0.15,
                placement: PortPlacement::FrontPanel,
                position_m: [0.0, 0.0],
                wall_thickness_m: Some(0.018),
            }),
            baffle_width_m: 0.286,
            baffle_height_m: 0.436,
            braces: Vec::new(),
        }
    }
}

impl CabinetDesign {
    /// Validate all dimensions, port fit, baffle coverage, and brace layout.
    pub fn validate(&self) -> Result<(), CabinetDesignError> {
        require_positive("cabinet.box_width_m", self.box_width_m)?;
        require_positive("cabinet.box_height_m", self.box_height_m)?;
        require_positive("cabinet.box_depth_m", self.box_depth_m)?;
        require_positive("cabinet.baffle_width_m", self.baffle_width_m)?;
        require_positive("cabinet.baffle_height_m", self.baffle_height_m)?;
        if self.baffle_width_m < self.box_width_m || self.baffle_height_m < self.box_height_m {
            return Err(CabinetDesignError::BaffleSmallerThanFront);
        }
        if let Some(port) = &self.port {
            require_positive("cabinet.port.diameter_m", port.diameter_m)?;
            require_positive("cabinet.port.length_m", port.length_m)?;
            if port.diameter_m > self.box_width_m.min(self.box_height_m) {
                return Err(CabinetDesignError::PortDoesNotFitPanel);
            }
            if port.position_m.iter().any(|value| !value.is_finite())
                || port.position_m[0].abs() + port.diameter_m / 2.0 > self.baffle_width_m / 2.0
                || port.position_m[1].abs() + port.diameter_m / 2.0 > self.baffle_height_m / 2.0
            {
                return Err(CabinetDesignError::PortDoesNotFitPanel);
            }
            if port.length_m > self.box_depth_m {
                return Err(CabinetDesignError::PortLengthExceedsDepth);
            }
            if let Some(thickness) = port.wall_thickness_m {
                require_positive("cabinet.port.wall_thickness_m", thickness)?;
                let outer_radius = (port.diameter_m + 2.0 * thickness) * 0.5;
                if port.position_m[0].abs() + outer_radius > self.box_width_m * 0.5
                    || port.position_m[1].abs() + outer_radius > self.box_height_m * 0.5
                {
                    return Err(CabinetDesignError::PortDoesNotFitPanel);
                }
            }
        }
        let mut sorted: Vec<&BraceDesign> = self.braces.iter().collect();
        sorted.sort_by(|a, b| {
            (a.panel, a.split_along)
                .cmp(&(b.panel, b.split_along))
                .then(a.position_fraction.total_cmp(&b.position_fraction))
        });
        for brace in &self.braces {
            if !brace.position_fraction.is_finite()
                || !(0.0..1.0).contains(&brace.position_fraction)
            {
                return Err(CabinetDesignError::BraceFractionOutOfRange);
            }
            if brace.position_fraction < MIN_SUBPANEL_FRACTION
                || brace.position_fraction > 1.0 - MIN_SUBPANEL_FRACTION
            {
                return Err(CabinetDesignError::DegenerateSubPanels { panel: brace.panel });
            }
            if let Some(cross_section) = brace.cross_section {
                require_positive("cabinet.brace.width_m", cross_section.width_m)?;
                require_positive("cabinet.brace.depth_m", cross_section.depth_m)?;
            }
        }
        for window in sorted.windows(2) {
            let (a, b) = (window[0], window[1]);
            if a.panel == b.panel {
                if a.split_along != b.split_along {
                    return Err(CabinetDesignError::BraceAxesConflict { panel: a.panel });
                }
                if b.position_fraction - a.position_fraction < MIN_SUBPANEL_FRACTION {
                    return Err(CabinetDesignError::DegenerateSubPanels { panel: a.panel });
                }
            }
        }
        Ok(())
    }

    /// Internal box volume `w·h·d` (m³). Exact at every fidelity level.
    pub fn volume_m3(&self) -> f64 {
        self.box_width_m * self.box_height_m * self.box_depth_m
    }

    /// Equivalent cylinder radius `√(w·h/π)` (front-panel area match).
    pub fn equivalent_radius_m(&self) -> f64 {
        (self.box_width_m * self.box_height_m / std::f64::consts::PI).sqrt()
    }

    /// Equal-area circular baffle radius `√(baffle_w·baffle_h/π)`.
    pub fn equivalent_baffle_radius_m(&self) -> f64 {
        (self.baffle_width_m * self.baffle_height_m / std::f64::consts::PI).sqrt()
    }

    /// Base panel dimensions `(width, height)` in the [`CabinetPanel`] sense
    /// (internal dims), matching `Cabinet::sealed_box` layout.
    fn panel_dimensions(&self) -> [(PanelId, f64, f64); 5] {
        let (w, h, d) = (self.box_width_m, self.box_height_m, self.box_depth_m);
        [
            (PanelId::Top, w, d),
            (PanelId::Bottom, w, d),
            (PanelId::Left, d, h),
            (PanelId::Right, d, h),
            (PanelId::Back, w, h),
        ]
    }

    /// Lower to the lumped enclosure model. Volume- and port-exact.
    pub fn to_lumped(&self) -> Result<EnclosureSpec, CabinetDesignError> {
        self.validate()?;
        let volume_m3 = self.volume_m3();
        Ok(match &self.port {
            None => EnclosureSpec::Sealed {
                volume_m3,
                leakage_resistance_pa_s_per_m3: 0.0,
                absorption_resistance_pa_s_per_m3: 0.0,
            },
            Some(port) => EnclosureSpec::Vented {
                volume_m3,
                port: PortSpec {
                    radius_m: port.diameter_m / 2.0,
                    length_m: port.length_m,
                    end_correction_m: port.end_correction_m(),
                    resistance_pa_s_per_m3: 0.0,
                },
                leakage_resistance_pa_s_per_m3: 0.0,
                absorption_resistance_pa_s_per_m3: 0.0,
            },
        })
    }

    /// Lower to the axisymmetric resolved FEM configuration.
    ///
    /// Equivalent cylinder (see module docs): depth = box depth, radius =
    /// [`Self::equivalent_radius_m`], baffle = equal-area circle. The port is
    /// the resolved annular slit: `baffle_half_thickness = port_length / 2`
    /// and the annulus area equals the circular port area, so the compact
    /// Helmholtz frequency matches the lumped tuning exactly (same medium).
    ///
    /// `diaphragm_radius_m` is the driver diaphragm radius (not a cabinet
    /// variable); the annular port slit is placed just outside it.
    ///
    /// Sealed designs are rejected: the resolved model's port slit is
    /// structural. PML bounds are extended when the cabinet exceeds the
    /// default exterior extents.
    pub fn to_resolved_config(
        &self,
        diaphragm_radius_m: f64,
    ) -> Result<ResolvedAcousticConfig, CabinetDesignError> {
        self.validate()?;
        let port = self
            .port
            .as_ref()
            .ok_or(CabinetDesignError::SealedNotResolvable)?;
        let enclosure_radius = self.equivalent_radius_m();
        if !diaphragm_radius_m.is_finite()
            || diaphragm_radius_m <= 0.0
            || diaphragm_radius_m >= enclosure_radius
        {
            return Err(CabinetDesignError::InvalidDiaphragmRadius);
        }
        let port_area = std::f64::consts::PI * (port.diameter_m / 2.0).powi(2);
        // Annular slit just outside the diaphragm, sized to the port area.
        let port_inner = diaphragm_radius_m + 5.0e-3;
        let port_outer = (port_inner.powi(2) + port_area / std::f64::consts::PI).sqrt();
        if port_outer >= enclosure_radius {
            return Err(CabinetDesignError::EquivalentPortTooLarge);
        }
        let baffle_radius = self.equivalent_baffle_radius_m();
        let baffle_half_thickness = port.length_m / 2.0;

        let mut config = ResolvedAcousticConfig {
            diaphragm_radius_m,
            enclosure_radius_m: enclosure_radius,
            cavity_depth_m: self.box_depth_m,
            baffle_radius_m: baffle_radius,
            baffle_half_thickness_m: baffle_half_thickness,
            port_inner_radius_m: port_inner,
            port_outer_radius_m: port_outer,
            port_end_correction_m: port.end_correction_m(),
            // A cabinet design describes a real box: the enclosure shell is
            // solid, not an open pocket of exterior air.
            seal_enclosure_walls: true,
            ..ResolvedAcousticConfig::default()
        };
        // Extend the PML/exterior when the cabinet exceeds default extents so
        // validation's "exterior surrounds the enclosure" invariant holds.
        config.radial_pml_start_m = config
            .radial_pml_start_m
            .max(baffle_radius + config.radial_pml_thickness_m);
        config.upper_pml_start_m = config
            .upper_pml_start_m
            .max(baffle_half_thickness + config.upper_pml_thickness_m);
        config.lower_pml_start_m = config
            .lower_pml_start_m
            .min(-(self.box_depth_m + baffle_half_thickness + config.lower_pml_thickness_m));
        Ok(config)
    }

    /// Lower to the 3D FEM-BEM coupling fixture (same equivalent cylinder).
    ///
    /// Vented designs map to `cylindrical_box_with_external_port` (port tube
    /// radius/length exact); sealed designs map to
    /// `cylindrical_box_with_circular_aperture` with `driver_aperture_radius_m`
    /// as the driver cutout (ignored for vented designs, where the aperture
    /// is the port mouth). `discretization` passes the mesh-density controls
    /// through unchanged.
    ///
    /// **Limitation**: the fixture has no baffle representation (exterior BEM
    /// radiates from the cylinder walls); baffle dimensions are not carried.
    pub fn to_coupling_fixture(
        &self,
        driver_aperture_radius_m: f64,
        discretization: CylinderFixtureDiscretization,
    ) -> Result<CouplingFixture, FixtureError> {
        // Dimension errors surface as FixtureError::InvalidDimension via the
        // builders; run our validation first for the cabinet-specific rules.
        if self.validate().is_err() {
            return Err(FixtureError::InvalidDimension {
                field: "cabinet_design",
            });
        }
        let radius = self.equivalent_radius_m();
        match &self.port {
            Some(port) => cylindrical_box_with_external_port(
                radius,
                self.box_depth_m,
                port.diameter_m / 2.0,
                port.length_m,
                discretization,
            ),
            None => cylindrical_box_with_circular_aperture(
                radius,
                self.box_depth_m,
                driver_aperture_radius_m,
                discretization,
            ),
        }
    }

    /// Lower to the panel-radiation cabinet model.
    ///
    /// Unbraced: identical panel layout to `Cabinet::sealed_box` with the
    /// external dims set so the internal dims match this design exactly.
    /// Braced: braced panels are subdivided into simply-supported sub-panels
    /// (sorted fractions along the shared split axis); the sub-panel list
    /// replaces the parent panel in place.
    pub fn to_cabinet(&self) -> Result<Cabinet, CabinetDesignError> {
        self.validate()?;
        let material = self.wall_material.panel_material();
        if self.braces.is_empty() {
            let t = material.thickness;
            return Ok(Cabinet::sealed_box(
                self.box_width_m + 2.0 * t,
                self.box_height_m + 2.0 * t,
                self.box_depth_m + 2.0 * t,
                material,
            ));
        }
        let mut panels = Vec::new();
        for (panel_id, width, height) in self.panel_dimensions() {
            let braces: Vec<&BraceDesign> = self
                .braces
                .iter()
                .filter(|brace| brace.panel == panel_id)
                .collect();
            if braces.is_empty() {
                panels.push(
                    sonium_fem::physics::acoustics::electroacoustics::cabinet_radiation::CabinetPanel {
                        width,
                        height,
                        material,
                    },
                );
                continue;
            }
            // Validation guarantees a shared axis per panel.
            let axis = braces[0].split_along;
            let mut fractions: Vec<f64> =
                braces.iter().map(|brace| brace.position_fraction).collect();
            fractions.sort_by(f64::total_cmp);
            let mut edges = vec![0.0_f64];
            edges.extend(fractions);
            edges.push(1.0);
            for segment in edges.windows(2) {
                let fraction = segment[1] - segment[0];
                let (w, h) = match axis {
                    PanelAxis::Width => (width * fraction, height),
                    PanelAxis::Height => (width, height * fraction),
                };
                panels.push(
                    sonium_fem::physics::acoustics::electroacoustics::cabinet_radiation::CabinetPanel {
                        width: w,
                        height: h,
                        material,
                    },
                );
            }
        }
        Ok(Cabinet { panels })
    }

    /// Continuous design variables with bounds derived from the current
    /// design.
    ///
    /// Order: box width/height/depth, baffle width/height, port
    /// diameter/length/x/y (vented only), then one variable per brace
    /// position (in stored brace order).
    ///
    /// The port **presence**, the brace **count/panel/axis**, and the wall
    /// material are categorical structural choices and stay out of this
    /// vector: continuous relaxations of discrete structure give optimizers
    /// non-physical intermediate states (a half-existent brace) and stall
    /// gradient-based methods. Structural search belongs in an outer
    /// categorical loop that rebuilds the [`CabinetDesign`] topology.
    pub fn design_variables(&self) -> Vec<DesignVariable> {
        let mut variables = Vec::new();
        let mut push = |name: &str, value: f64, lower: f64, upper: f64| {
            variables.push(DesignVariable {
                name: name.to_string(),
                value,
                lower,
                upper,
            });
        };
        // Box dims: ±50 % around the current design.
        push(
            "box.width_m",
            self.box_width_m,
            0.5 * self.box_width_m,
            1.5 * self.box_width_m,
        );
        push(
            "box.height_m",
            self.box_height_m,
            0.5 * self.box_height_m,
            1.5 * self.box_height_m,
        );
        push(
            "box.depth_m",
            self.box_depth_m,
            0.5 * self.box_depth_m,
            1.5 * self.box_depth_m,
        );
        // Baffle: at least the front panel, at most 2× the current baffle.
        push(
            "baffle.width_m",
            self.baffle_width_m,
            self.box_width_m,
            2.0 * self.baffle_width_m,
        );
        push(
            "baffle.height_m",
            self.baffle_height_m,
            self.box_height_m,
            2.0 * self.baffle_height_m,
        );
        if let Some(port) = &self.port {
            // Physical port ranges, clipped so the port fits the front panel.
            let max_diameter = 0.9 * self.box_width_m.min(self.box_height_m);
            push(
                "port.diameter_m",
                port.diameter_m,
                0.01_f64.min(port.diameter_m),
                max_diameter.max(port.diameter_m),
            );
            push(
                "port.length_m",
                port.length_m,
                0.02_f64.min(port.length_m),
                self.box_depth_m.max(0.02),
            );
            push(
                "port.position_x_m",
                port.position_m[0],
                -self.baffle_width_m / 2.0 + port.diameter_m / 2.0,
                self.baffle_width_m / 2.0 - port.diameter_m / 2.0,
            );
            push(
                "port.position_y_m",
                port.position_m[1],
                -self.baffle_height_m / 2.0 + port.diameter_m / 2.0,
                self.baffle_height_m / 2.0 - port.diameter_m / 2.0,
            );
        }
        for (index, brace) in self.braces.iter().enumerate() {
            push(
                &format!("brace[{index}].position_fraction"),
                brace.position_fraction,
                0.05,
                0.95,
            );
        }
        variables
    }

    /// Rebuild a design from a continuous variable vector, keeping this
    /// design's structure (port presence, brace layout, material).
    ///
    /// `values` must match [`Self::design_variables`] in length and order and
    /// respect its bounds; the result is re-validated.
    pub fn from_design_variables(
        &self,
        values: &[f64],
    ) -> Result<CabinetDesign, CabinetDesignError> {
        let variables = self.design_variables();
        if values.len() != variables.len() {
            return Err(CabinetDesignError::DesignVariableCountMismatch {
                expected: variables.len(),
                got: values.len(),
            });
        }
        for (variable, &value) in variables.iter().zip(values) {
            if !value.is_finite() || value < variable.lower || value > variable.upper {
                return Err(CabinetDesignError::DesignVariableOutOfBounds {
                    name: variable.name.clone(),
                    value,
                    lower: variable.lower,
                    upper: variable.upper,
                });
            }
        }
        let mut design = self.clone();
        // The count check above guarantees `values` yields exactly one
        // entry per assignment below; each `take` still returns the
        // count-mismatch error (instead of panicking) so a future layout
        // edit that desynchronises the two fails closed.
        let mut cursor = 0;
        let mut take = |values: &[f64]| -> Result<f64, CabinetDesignError> {
            let value = values.get(cursor).copied().ok_or(
                CabinetDesignError::DesignVariableCountMismatch {
                    expected: variables.len(),
                    got: values.len(),
                },
            )?;
            cursor += 1;
            Ok(value)
        };
        design.box_width_m = take(values)?;
        design.box_height_m = take(values)?;
        design.box_depth_m = take(values)?;
        design.baffle_width_m = take(values)?;
        design.baffle_height_m = take(values)?;
        if let Some(port) = &mut design.port {
            port.diameter_m = take(values)?;
            port.length_m = take(values)?;
            port.position_m[0] = take(values)?;
            port.position_m[1] = take(values)?;
        }
        for brace in &mut design.braces {
            brace.position_fraction = take(values)?;
        }
        design.validate()?;
        Ok(design)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::AcousticMedium;
    use sonium_fem::mesh::ElementType;

    fn sealed_design() -> CabinetDesign {
        CabinetDesign {
            port: None,
            ..CabinetDesign::default()
        }
    }

    /// Sum of tet volumes of a coupling-fixture mesh.
    fn mesh_volume(fixture: &CouplingFixture) -> f64 {
        fixture
            .fem_mesh
            .elements
            .iter()
            .filter(|e| e.element_type == ElementType::Tetrahedron)
            .map(|tet| {
                let [a, b, c, d] = [
                    fixture.fem_mesh.nodes[tet.nodes[0]],
                    fixture.fem_mesh.nodes[tet.nodes[1]],
                    fixture.fem_mesh.nodes[tet.nodes[2]],
                    fixture.fem_mesh.nodes[tet.nodes[3]],
                ];
                let v1 = [b.x - a.x, b.y - a.y, b.z - a.z];
                let v2 = [c.x - a.x, c.y - a.y, c.z - a.z];
                let v3 = [d.x - a.x, d.y - a.y, d.z - a.z];
                let det = v1[0] * (v2[1] * v3[2] - v2[2] * v3[1])
                    - v1[1] * (v2[0] * v3[2] - v2[2] * v3[0])
                    + v1[2] * (v2[0] * v3[1] - v2[1] * v3[0]);
                det.abs() / 6.0
            })
            .sum()
    }

    fn fine_discretization() -> CylinderFixtureDiscretization {
        CylinderFixtureDiscretization {
            n_circumference: 48,
            n_box_depth: 6,
            n_cap_radial: 4,
            n_aperture_radial: 3,
            n_port_depth: 4,
        }
    }

    // --- Validation ------------------------------------------------------

    #[test]
    fn default_design_validates() {
        CabinetDesign::default().validate().unwrap();
        sealed_design().validate().unwrap();
    }

    #[test]
    fn validation_rejects_bad_dimensions() {
        let design = CabinetDesign {
            box_width_m: -0.1,
            ..CabinetDesign::default()
        };
        assert_eq!(
            design.validate(),
            Err(CabinetDesignError::InvalidDimension {
                field: "cabinet.box_width_m"
            })
        );
        let design = CabinetDesign {
            box_height_m: f64::NAN,
            ..CabinetDesign::default()
        };
        assert!(matches!(
            design.validate(),
            Err(CabinetDesignError::InvalidDimension { .. })
        ));
    }

    #[test]
    fn validation_rejects_oversize_port_and_small_baffle() {
        let mut design = CabinetDesign::default();
        design.port.as_mut().unwrap().diameter_m = 0.30; // wider than the 0.25 m front
        assert_eq!(
            design.validate(),
            Err(CabinetDesignError::PortDoesNotFitPanel)
        );

        let mut design = CabinetDesign::default();
        design.port.as_mut().unwrap().length_m = design.box_depth_m + 1.0e-3;
        assert_eq!(
            design.validate(),
            Err(CabinetDesignError::PortLengthExceedsDepth)
        );

        let design = CabinetDesign {
            baffle_height_m: 0.30, // below the 0.40 m box height
            ..CabinetDesign::default()
        };
        assert_eq!(
            design.validate(),
            Err(CabinetDesignError::BaffleSmallerThanFront)
        );
    }

    #[test]
    fn validation_rejects_bad_braces() {
        // Fraction outside (0, 1).
        let mut design = CabinetDesign::default();
        design.braces.push(BraceDesign {
            panel: PanelId::Back,
            split_along: PanelAxis::Width,
            position_fraction: 1.0,
            cross_section: None,
        });
        assert_eq!(
            design.validate(),
            Err(CabinetDesignError::BraceFractionOutOfRange)
        );

        // Mixed axes on one panel.
        let mut design = CabinetDesign::default();
        design.braces.push(BraceDesign {
            panel: PanelId::Back,
            split_along: PanelAxis::Width,
            position_fraction: 0.4,
            cross_section: None,
        });
        design.braces.push(BraceDesign {
            panel: PanelId::Back,
            split_along: PanelAxis::Height,
            position_fraction: 0.6,
            cross_section: None,
        });
        assert_eq!(
            design.validate(),
            Err(CabinetDesignError::BraceAxesConflict {
                panel: PanelId::Back
            })
        );

        // Duplicate fractions (degenerate sub-panel).
        let mut design = CabinetDesign::default();
        design.braces.push(BraceDesign {
            panel: PanelId::Top,
            split_along: PanelAxis::Width,
            position_fraction: 0.5,
            cross_section: None,
        });
        design.braces.push(BraceDesign {
            panel: PanelId::Top,
            split_along: PanelAxis::Width,
            position_fraction: 0.5,
            cross_section: None,
        });
        assert_eq!(
            design.validate(),
            Err(CabinetDesignError::DegenerateSubPanels {
                panel: PanelId::Top
            })
        );
    }

    #[test]
    fn validation_checks_optional_resolved_port_and_brace_solids() {
        let mut design = CabinetDesign::default();
        design.port.as_mut().unwrap().wall_thickness_m = Some(0.0);
        assert_eq!(
            design.validate(),
            Err(CabinetDesignError::InvalidDimension {
                field: "cabinet.port.wall_thickness_m"
            })
        );

        let mut design = CabinetDesign::default();
        design.braces.push(BraceDesign {
            panel: PanelId::Back,
            split_along: PanelAxis::Width,
            position_fraction: 0.5,
            cross_section: Some(BraceCrossSection {
                width_m: 0.0,
                depth_m: 0.02,
            }),
        });
        assert_eq!(
            design.validate(),
            Err(CabinetDesignError::InvalidDimension {
                field: "cabinet.brace.width_m"
            })
        );

        let mut design = CabinetDesign::default();
        let port = design.port.as_mut().unwrap();
        port.position_m = [0.1, 0.0];
        assert_eq!(
            design.validate(),
            Err(CabinetDesignError::PortDoesNotFitPanel)
        );
    }

    // --- Lumped lowering ---------------------------------------------------

    #[test]
    fn lumped_lowering_is_volume_and_port_exact() {
        let design = CabinetDesign::default();
        let EnclosureSpec::Vented {
            volume_m3, port, ..
        } = design.to_lumped().unwrap()
        else {
            panic!("default design is vented");
        };
        assert_eq!(volume_m3, design.volume_m3());
        assert_eq!(port.radius_m, 0.025);
        assert_eq!(port.length_m, 0.15);
        assert_eq!(port.end_correction_m, 1.46 * 0.025);

        let sealed = sealed_design().to_lumped().unwrap();
        assert_eq!(
            sealed,
            EnclosureSpec::Sealed {
                volume_m3: sealed_design().volume_m3(),
                leakage_resistance_pa_s_per_m3: 0.0,
                absorption_resistance_pa_s_per_m3: 0.0,
            }
        );
    }

    // --- Cross-fidelity volume consistency ---------------------------------

    #[test]
    fn volume_is_consistent_across_all_lowerings() {
        let design = CabinetDesign::default();
        let volume = design.volume_m3();

        // Lumped.
        let EnclosureSpec::Vented { volume_m3, .. } = design.to_lumped().unwrap() else {
            panic!("vented");
        };
        assert_eq!(volume_m3, volume);

        // Resolved: πR²d == w·h·d exactly.
        let resolved = design.to_resolved_config(0.04).unwrap();
        assert!(
            (resolved.cavity_volume_m3() - volume).abs() / volume < 1e-12,
            "resolved {} vs {volume}",
            resolved.cavity_volume_m3()
        );

        // Coupling fixture: geometric volume exact, tet-mesh volume within
        // the polygonal cross-section discretization error (48 sectors).
        let fixture = design
            .to_coupling_fixture(0.04, fine_discretization())
            .unwrap();
        assert!(
            (fixture.cavity_volume_m3() - volume).abs() / volume < 1e-12,
            "fixture {} vs {volume}",
            fixture.cavity_volume_m3()
        );
        let port = design.port.unwrap();
        let port_volume = std::f64::consts::PI * (port.diameter_m / 2.0).powi(2) * port.length_m;
        let expected_mesh = volume + port_volume;
        let rel = (mesh_volume(&fixture) - expected_mesh).abs() / expected_mesh;
        assert!(
            rel < 0.01,
            "mesh volume {} vs {expected_mesh} (rel {rel})",
            mesh_volume(&fixture)
        );
    }

    #[test]
    fn sealed_fixture_volume_matches_mesh_volume() {
        let design = sealed_design();
        let fixture = design
            .to_coupling_fixture(0.04, fine_discretization())
            .unwrap();
        assert!(fixture.port_length_m.is_none());
        let rel = (mesh_volume(&fixture) - design.volume_m3()).abs() / design.volume_m3();
        assert!(rel < 0.01, "sealed mesh volume rel error {rel}");
    }

    // --- Port tuning consistency ------------------------------------------

    #[test]
    fn lumped_tuning_matches_resolved_compact_helmholtz() {
        let design = CabinetDesign::default();
        let lumped = design
            .to_lumped()
            .unwrap()
            .tuning_frequency_hz(AcousticMedium::standard_air())
            .unwrap()
            .expect("vented design has a tuning frequency");
        let resolved = design
            .to_resolved_config(0.04)
            .unwrap()
            .compact_helmholtz_frequency_hz();
        // Same volume, same port area, same effective length, same medium
        // constants (air at 20 °C: 343 m/s, 1.204 kg/m³ on both sides).
        let rel = (lumped - resolved).abs() / lumped;
        assert!(rel < 1e-9, "lumped {lumped} Hz vs resolved {resolved} Hz");
        // Sanity: a 20 L box with a 50 mm × 150 mm port tunes around 40 Hz.
        assert!((30.0..=55.0).contains(&lumped), "tuning {lumped} Hz");
    }

    // --- Resolved lowering -------------------------------------------------

    #[test]
    fn resolved_lowering_maps_geometry_and_baffle() {
        let design = CabinetDesign::default();
        let resolved = design.to_resolved_config(0.04).unwrap();
        resolved.validate().unwrap();
        assert_eq!(resolved.cavity_depth_m, design.box_depth_m);
        assert_eq!(resolved.enclosure_radius_m, design.equivalent_radius_m());
        // Baffle: equal-area circle, and it must cover the enclosure radius.
        assert_eq!(
            resolved.baffle_radius_m,
            design.equivalent_baffle_radius_m()
        );
        assert!(resolved.baffle_radius_m >= resolved.enclosure_radius_m);
        // The solid shell flag from C1 is set for a real cabinet.
        assert!(resolved.seal_enclosure_walls);
        // Annular port area equals the circular port area.
        let port = design.port.unwrap();
        let expected_area = std::f64::consts::PI * (port.diameter_m / 2.0).powi(2);
        assert!(
            (resolved.port_area_m2() - expected_area).abs() / expected_area < 1e-9,
            "annulus {} vs {expected_area}",
            resolved.port_area_m2()
        );
        // Effective length = 2·(L/2) + end correction = L + end correction.
        assert!(
            (resolved.effective_port_length_m() - (port.length_m + port.end_correction_m())).abs()
                < 1e-12
        );
    }

    #[test]
    fn resolved_lowering_rejects_sealed_and_bad_diaphragm() {
        assert!(matches!(
            sealed_design().to_resolved_config(0.04),
            Err(CabinetDesignError::SealedNotResolvable)
        ));
        let design = CabinetDesign::default();
        assert!(matches!(
            design.to_resolved_config(design.equivalent_radius_m()),
            Err(CabinetDesignError::InvalidDiaphragmRadius)
        ));
        assert!(matches!(
            design.to_resolved_config(0.0),
            Err(CabinetDesignError::InvalidDiaphragmRadius)
        ));
    }

    // --- Cabinet (panel) lowering -----------------------------------------

    #[test]
    fn unbraced_cabinet_matches_sealed_box_layout() {
        let design = CabinetDesign::default();
        let cabinet = design.to_cabinet().unwrap();
        assert_eq!(cabinet.panels.len(), 5);
        // Internal dims are used directly (Cabinet::sealed_box subtracts 2t
        // from the external dims we pass in).
        let (w, h, d) = (design.box_width_m, design.box_height_m, design.box_depth_m);
        let mut areas: Vec<f64> = cabinet.panels.iter().map(|p| p.width * p.height).collect();
        areas.sort_by(f64::total_cmp);
        let mut expected = vec![w * d, w * d, d * h, d * h, w * h];
        expected.sort_by(f64::total_cmp);
        for (a, e) in areas.iter().zip(expected) {
            assert!((a - e).abs() / e < 1e-12, "panel area {a} vs {e}");
        }
    }

    #[test]
    fn braced_subpanels_partition_area_and_raise_modes() {
        let mut design = CabinetDesign::default();
        design.braces.push(BraceDesign {
            panel: PanelId::Back,
            split_along: PanelAxis::Height,
            position_fraction: 0.45,
            cross_section: None,
        });
        design.braces.push(BraceDesign {
            panel: PanelId::Top,
            split_along: PanelAxis::Width,
            position_fraction: 0.5,
            cross_section: None,
        });
        design.braces.push(BraceDesign {
            panel: PanelId::Top,
            split_along: PanelAxis::Width,
            position_fraction: 0.8,
            cross_section: None,
        });
        let braced = design.to_cabinet().unwrap();
        // 5 base panels: back → 2 sub-panels, top → 3 sub-panels.
        assert_eq!(braced.panels.len(), 5 - 2 + 2 + 3);

        // Sub-panel areas sum to their parent panel areas.
        let (w, h, d) = (design.box_width_m, design.box_height_m, design.box_depth_m);
        // Layout order: Top (3 sub-panels, idx 0..3), Bottom (3), Left (4),
        // Right (5), Back (2 sub-panels, idx 6..8).
        let back_area: f64 = braced.panels[6..8].iter().map(|p| p.width * p.height).sum();
        assert!(
            (back_area - w * h).abs() / (w * h) < 1e-12,
            "back sub-panel area {back_area} vs {}",
            w * h
        );
        let top_area: f64 = braced.panels[0..3].iter().map(|p| p.width * p.height).sum();
        assert!(
            (top_area - w * d).abs() / (w * d) < 1e-12,
            "top sub-panel area {top_area} vs {}",
            w * d
        );

        // Bracing raises the lowest mode of the braced panel.
        let unbraced = CabinetDesign {
            braces: Vec::new(),
            ..design.clone()
        }
        .to_cabinet()
        .unwrap();
        let back_unbraced = &unbraced.panels[4];
        let f_unbraced = back_unbraced.mode_frequency(1, 1);
        let f_braced = braced.panels[6..8]
            .iter()
            .map(|p| p.mode_frequency(1, 1))
            .fold(f64::INFINITY, f64::min);
        assert!(
            f_braced > f_unbraced,
            "braced {f_braced} Hz should exceed unbraced {f_unbraced} Hz"
        );
    }

    // --- Design variables ---------------------------------------------------

    #[test]
    fn design_variable_round_trip_is_identical() {
        let mut design = CabinetDesign::default();
        design.braces.push(BraceDesign {
            panel: PanelId::Back,
            split_along: PanelAxis::Width,
            position_fraction: 0.5,
            cross_section: None,
        });
        let variables = design.design_variables();
        // 3 box + 2 baffle + 4 port + 1 brace.
        assert_eq!(variables.len(), 10);
        let values: Vec<f64> = variables.iter().map(|v| v.value).collect();
        let rebuilt = design.from_design_variables(&values).unwrap();
        assert_eq!(rebuilt, design);

        // Sealed design: two fewer variables.
        let sealed = sealed_design();
        assert_eq!(sealed.design_variables().len(), 5);
        let values: Vec<f64> = sealed.design_variables().iter().map(|v| v.value).collect();
        assert_eq!(sealed.from_design_variables(&values).unwrap(), sealed);
    }

    #[test]
    fn design_variables_respect_bounds_and_reject_bad_vectors() {
        let design = CabinetDesign::default();
        let variables = design.design_variables();
        for variable in &variables {
            assert!(variable.lower <= variable.value && variable.value <= variable.upper);
            assert!(variable.lower < variable.upper);
        }

        // Wrong length.
        assert!(matches!(
            design.from_design_variables(&[0.1, 0.2]),
            Err(CabinetDesignError::DesignVariableCountMismatch { .. })
        ));

        // Above the upper bound.
        let mut values: Vec<f64> = variables.iter().map(|v| v.value).collect();
        values[0] = variables[0].upper + 1e-6;
        assert!(matches!(
            design.from_design_variables(&values),
            Err(CabinetDesignError::DesignVariableOutOfBounds { .. })
        ));

        // In-bounds but physically invalid: baffle below the front panel
        // passes the bound check only if lower < box height... the baffle
        // lower bound equals the box height, so pick a NaN instead.
        values
            .iter_mut()
            .zip(&variables)
            .for_each(|(v, dv)| *v = dv.value);
        values[4] = f64::NAN;
        assert!(matches!(
            design.from_design_variables(&values),
            Err(CabinetDesignError::DesignVariableOutOfBounds { .. })
        ));

        // A valid perturbation rebuilds and validates.
        values
            .iter_mut()
            .zip(&variables)
            .for_each(|(v, dv)| *v = dv.value);
        values[2] = 0.25; // deeper box within ±50 %
        let perturbed = design.from_design_variables(&values).unwrap();
        assert_eq!(perturbed.box_depth_m, 0.25);
        perturbed.validate().unwrap();
    }
}
